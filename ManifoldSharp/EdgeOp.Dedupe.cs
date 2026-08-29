// Copyright 2026 Lars Brubaker
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// EdgeOp.Dedupe.cs — SplitPinchedVerts, DedupeEdge and DedupeEdges: the
// even-manifold-to-2-manifold half of edge_op.rs. The module header and the
// file-split map live in EdgeOp.cs.
//
// This file is where the "orbit walks that look alike and are not" note in the
// module header bites hardest: there are EIGHT vertex orbits below and no two
// of them have the same guards. Worse, they do not even all take the same step,
// and the two spellings are one transposition apart:
//
//   step A  current = NextHalfedge(Halfedge[current].PairedHalfedge)
//           "next of paired" — the ForVert step. For current v→w it lands on a
//           halfedge that also STARTS at v, so it orbits the start vertex.
//   step B  current = Halfedge[NextHalfedge(current)].PairedHalfedge
//           "paired of next" — the transposition. For current v→w it lands on a
//           halfedge that ENDS at w, so it orbits the end vertex instead.
//
// The eight, in file order:
//
//   1. SplitPinchedVerts, pinched branch  — step A; relabels StartVert/EndVert
//      as it goes, bounded by halfedgeProcessed.Length, ends on `current == i`
//   2. SplitPinchedVerts, first-seen branch — step A; marks the cycle processed
//      only, same bounds, no relabelling
//   3. DedupeEdge, the `while (current != edge)` scan — step B; may exit by
//      finding StartVert == startVert, by an unpaired -1, or by reaching `edge`,
//      and which of the three happened is read back afterwards as `current == edge`
//   4. DedupeEdge, the "separate topological unit" relabel — step A; writes
//      BEFORE it steps and ends on returning to its own start, not to `edge`
//   5. DedupeEdge, the "is endVert pinched" scan — step B; returns early (not
//      breaks) when it finds endVert, because the caller distinguishes the two
//   6. DedupeEdge, the "split the new pinched vert" relabel — step A; same
//      write-then-step shape as 4
//   7. DedupeEdges, the endVert-collecting scan — step A, plus an explicit
//      orbitSteps counter
//   8. DedupeEdges, the duplicate-flagging rescan — step A, same counter
//
// Walks 3 and 5 are the two step-B orbits, and they are the ones that look most
// like a typo for step A. They are not: rewriting either into ForVert shape
// silently changes which of the edge's two endpoints is being walked.
// The counters on 7 and 8 exist because the meshes reaching DedupeEdges are not
// yet 2-manifold and can orbit forever. None of the eight is
// ManifoldImpl.ForVert, which has no unpaired-halfedge guard at all.
//
// DedupeEdge appends to mesh.Halfedge while walking it — the reason the module
// writes halfedge fields through the read-modify-store helpers rather than
// through a CollectionsMarshal span.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <content>
	/// The pinched-vert and duplicate-edge repairs of edge_op.rs.
	/// </content>
	public static partial class EdgeOp
	{
		// -----------------------------------------------------------------------
		// SplitPinchedVerts
		// -----------------------------------------------------------------------

		/// <summary>
		/// Finds vertices where multiple halfedge cycles meet (pinched verts) and splits them
		/// into separate vertices — one per cycle.
		/// </summary>
		/// <param name="mesh">The mesh to edit.</param>
		public static void SplitPinchedVerts(ManifoldImpl mesh)
		{
			int nEdges = mesh.Halfedge.Count;
			bool[] vertProcessed = new bool[mesh.VertPos.Count];
			bool[] halfedgeProcessed = new bool[nEdges];

			int i = 0;
			while (i < nEdges)
			{
				if (halfedgeProcessed[i])
				{
					i += 1;
					continue;
				}

				int vert = mesh.Halfedge[i].StartVert;
				if (vert < 0)
				{
					i += 1;
					continue;
				}

				// Rust reads this as `vert_processed.get(vert) == Some(&true)`, which is
				// false past the end rather than a panic; the C# bound check is explicit.
				if (vert < vertProcessed.Length && vertProcessed[vert])
				{
					// Pinched: create a new vertex for this cycle.
					// ForVert(i, ...) visits all edges in the orbit using the step:
					//   current = next_halfedge(halfedge[current].paired_halfedge)
					Vec3 newPos = mesh.VertPos[vert];
					mesh.VertPos.Add(newPos);
					int newVert = mesh.VertPos.Count - 1;
					int current = i;
					while (true)
					{
						int paired = mesh.Halfedge[current].PairedHalfedge;
						if (paired < 0)
						{
							break;
						}

						current = Types.NextHalfedge(paired);
						if (current >= halfedgeProcessed.Length)
						{
							break;
						}

						halfedgeProcessed[current] = true;
						SetStartVert(mesh.Halfedge, current, newVert);
						int currPaired = mesh.Halfedge[current].PairedHalfedge;
						if (currPaired >= 0)
						{
							SetEndVert(mesh.Halfedge, currPaired, newVert);
						}

						if (current == i)
						{
							break;
						}
					}
				}
				else
				{
					// First time seeing this vert: mark cycle as processed.
					if (vert < vertProcessed.Length)
					{
						vertProcessed[vert] = true;
					}

					int current = i;
					while (true)
					{
						int paired = mesh.Halfedge[current].PairedHalfedge;
						if (paired < 0)
						{
							break;
						}

						current = Types.NextHalfedge(paired);
						if (current >= halfedgeProcessed.Length)
						{
							break;
						}

						halfedgeProcessed[current] = true;
						if (current == i)
						{
							break;
						}
					}
				}

				i += 1;
			}
		}

		// -----------------------------------------------------------------------
		// DedupeEdges / DedupeEdge
		// -----------------------------------------------------------------------

		/// <summary>
		/// Fixes 4-manifold edges (where two pairs of triangles share the same edge) by
		/// duplicating one endpoint vertex and splitting the topology.
		/// </summary>
		/// <param name="mesh">The mesh to edit.</param>
		/// <param name="edge">The duplicated halfedge to repair.</param>
		public static void DedupeEdge(ManifoldImpl mesh, int edge)
		{
			int startVert = mesh.Halfedge[edge].StartVert;
			int endVert = mesh.Halfedge[edge].EndVert;
			int endProp = mesh.Halfedge[Types.NextHalfedge(edge)].PropVert;

			int paired = mesh.Halfedge[edge].PairedHalfedge;
			if (paired < 0)
			{
				return;
			}

			int nextPaired = mesh.Halfedge[Types.NextHalfedge(edge)].PairedHalfedge;
			if (nextPaired < 0)
			{
				return;
			}

			int current = nextPaired;

			while (current != edge)
			{
				int vert = mesh.Halfedge[current].StartVert;
				if (vert == startVert)
				{
					// Single topological unit: need to add 2 triangles
					int newVert = mesh.VertPos.Count;
					mesh.VertPos.Add(mesh.VertPos[endVert]);

					// C++ advances current BEFORE building triangles:
					// current = halfedge_[NextHalfedge(current)].pairedHalfedge;
					int nextP = mesh.Halfedge[Types.NextHalfedge(current)].PairedHalfedge;
					if (nextP < 0)
					{
						break;
					}

					current = nextP;
					int oppP = mesh.Halfedge[Types.NextHalfedge(edge)].PairedHalfedge;
					if (oppP < 0)
					{
						break;
					}

					int opposite = oppP;

					UpdateVert(mesh, newVert, current, opposite);

					int newHe = mesh.Halfedge.Count;
					int oldFace = current / 3;
					int outsideVert = mesh.Halfedge[current].StartVert;
					mesh.Halfedge.Add(new Halfedge(endVert, newVert, -1, endProp));
					mesh.Halfedge.Add(new Halfedge(newVert, outsideVert, -1, endProp));
					int currPropVert = mesh.Halfedge[current].PropVert;
					int currPaired = mesh.Halfedge[current].PairedHalfedge;
					mesh.Halfedge.Add(new Halfedge(outsideVert, endVert, -1, currPropVert));
					PairUp(mesh.Halfedge, newHe + 2, currPaired);
					PairUp(mesh.Halfedge, newHe + 1, current);
					if (mesh.MeshRelation.TriRef.Count != 0)
					{
						TriRef triRefCopy = mesh.MeshRelation.TriRef[oldFace];
						mesh.MeshRelation.TriRef.Add(triRefCopy);
					}

					if (mesh.FaceNormal.Count != 0)
					{
						Vec3 fnCopy = mesh.FaceNormal[oldFace];
						mesh.FaceNormal.Add(fnCopy);
					}

					int newHe2 = newHe + 3;
					int oldFace2 = opposite / 3;
					int outsideVert2 = mesh.Halfedge[opposite].StartVert;
					mesh.Halfedge.Add(new Halfedge(newVert, endVert, -1, endProp));
					mesh.Halfedge.Add(new Halfedge(endVert, outsideVert2, -1, endProp));
					int oppPropVert = mesh.Halfedge[opposite].PropVert;
					int oppPaired = mesh.Halfedge[opposite].PairedHalfedge;
					mesh.Halfedge.Add(new Halfedge(outsideVert2, newVert, -1, oppPropVert));
					PairUp(mesh.Halfedge, newHe2 + 2, oppPaired);
					PairUp(mesh.Halfedge, newHe2 + 1, opposite);
					PairUp(mesh.Halfedge, newHe2, newHe);
					if (mesh.MeshRelation.TriRef.Count != 0)
					{
						TriRef triRefCopy = mesh.MeshRelation.TriRef[oldFace2];
						mesh.MeshRelation.TriRef.Add(triRefCopy);
					}

					if (mesh.FaceNormal.Count != 0)
					{
						Vec3 fnCopy = mesh.FaceNormal[oldFace2];
						mesh.FaceNormal.Add(fnCopy);
					}

					break;
				}

				int orbitP = mesh.Halfedge[Types.NextHalfedge(current)].PairedHalfedge;
				if (orbitP < 0)
				{
					break;
				}

				current = orbitP;
			}

			if (current == edge)
			{
				// Separate topological unit: no new faces needed, just duplicate the vert
				int newVert = mesh.VertPos.Count;
				mesh.VertPos.Add(mesh.VertPos[endVert]);

				// ForVert(NextHalfedge(current), ...) to update all orbit halfedges to new_vert
				int start = Types.NextHalfedge(current);
				int cur = start;
				while (true)
				{
					SetStartVert(mesh.Halfedge, cur, newVert);
					int pairedCur = mesh.Halfedge[cur].PairedHalfedge;
					if (pairedCur < 0)
					{
						break;
					}

					SetEndVert(mesh.Halfedge, pairedCur, newVert);

					// ForVert step: next_halfedge(halfedge[cur].paired_halfedge)
					cur = Types.NextHalfedge(pairedCur);
					if (cur == start)
					{
						break;
					}
				}
			}

			// Orbit startVert - check if endVert is pinched
			int pair = mesh.Halfedge[edge].PairedHalfedge;
			if (pair < 0)
			{
				return;
			}

			int cur2 = mesh.Halfedge[Types.NextHalfedge(pair)].PairedHalfedge;
			if (cur2 < 0)
			{
				return;
			}

			while (cur2 != pair)
			{
				int v = mesh.Halfedge[cur2].StartVert;
				if (v == endVert)
				{
					return; // Connected: not a pinched vert
				}

				int p = mesh.Halfedge[Types.NextHalfedge(cur2)].PairedHalfedge;
				if (p < 0)
				{
					break;
				}

				cur2 = p;
			}

			if (cur2 == pair)
			{
				// Split the new pinched vert using ForVert
				int newVert2 = mesh.VertPos.Count;
				mesh.VertPos.Add(mesh.VertPos[endVert]);
				int s2 = Types.NextHalfedge(cur2);
				int c2 = s2;
				while (true)
				{
					SetStartVert(mesh.Halfedge, c2, newVert2);
					int pairedC2 = mesh.Halfedge[c2].PairedHalfedge;
					if (pairedC2 < 0)
					{
						break;
					}

					SetEndVert(mesh.Halfedge, pairedC2, newVert2);
					c2 = Types.NextHalfedge(pairedC2);
					if (c2 == s2)
					{
						break;
					}
				}
			}
		}

		/// <summary>Finds and fixes all duplicate edges (4-manifold conditions).</summary>
		/// <param name="mesh">The mesh to edit.</param>
		public static void DedupeEdges(ManifoldImpl mesh)
		{
			int maxIterations = mesh.Halfedge.Count; // safety bound
			for (int iteration = 0; iteration < maxIterations; iteration++)
			{
				int nEdges = mesh.Halfedge.Count;
				bool[] processed = new bool[nEdges];
				List<int> duplicates = new List<int>();

				for (int i = 0; i < nEdges; i++)
				{
					if (processed[i])
					{
						continue;
					}

					int sv = mesh.Halfedge[i].StartVert;
					int ev = mesh.Halfedge[i].EndVert;
					if (sv < 0 || ev < 0)
					{
						continue;
					}

					// Track all endVerts seen in this vertex's orbit, keeping smallest edge idx.
					// Uses ForVert traversal: current = next_halfedge(halfedge[current].paired_halfedge)
					List<EndVertEntry> endVerts = new List<EndVertEntry>(); // (endVert, min_edge_idx)

					// Process i itself first
					processed[i] = true;
					int cEv0 = mesh.Halfedge[i].EndVert;
					if (cEv0 >= 0)
					{
						endVerts.Add(new EndVertEntry(cEv0, i));
					}

					// Then orbit (with safety bound to prevent infinite loops)
					int current = i;
					int orbitSteps = 0;
					while (true)
					{
						int pair = mesh.Halfedge[current].PairedHalfedge;
						if (pair < 0)
						{
							break;
						}

						current = Types.NextHalfedge(pair);
						if (current == i)
						{
							break;
						}

						orbitSteps += 1;
						if (orbitSteps > nEdges)
						{
							break;
						} // safety

						processed[current] = true;
						int cSv = mesh.Halfedge[current].StartVert;
						int cEv = mesh.Halfedge[current].EndVert;
						if (cSv >= 0 && cEv >= 0)
						{
							// Rust `iter_mut().find(...)` then mutates through the &mut;
							// EndVertEntry is a struct in a List, so the C# equivalent is a
							// scan for the index and a store back (list[j].Field = x is
							// CS1612).
							int found = -1;
							for (int j = 0; j < endVerts.Count; j++)
							{
								if (endVerts[j].EndVert == cEv)
								{
									found = j;
									break;
								}
							}

							if (found >= 0)
							{
								if (current < endVerts[found].MinEdge)
								{
									endVerts[found] = new EndVertEntry(endVerts[found].EndVert, current);
								}
							}
							else
							{
								endVerts.Add(new EndVertEntry(cEv, current));
							}
						}
					}

					// Second pass: find edges that aren't the minimum for their endVert
					cEv0 = mesh.Halfedge[i].EndVert;
					if (cEv0 >= 0)
					{
						int found = FindEndVert(endVerts, cEv0);
						if (found >= 0 && endVerts[found].MinEdge != i)
						{
							duplicates.Add(i);
						}
					}

					current = i;
					orbitSteps = 0;
					while (true)
					{
						int pair = mesh.Halfedge[current].PairedHalfedge;
						if (pair < 0)
						{
							break;
						}

						current = Types.NextHalfedge(pair);
						if (current == i)
						{
							break;
						}

						orbitSteps += 1;
						if (orbitSteps > nEdges)
						{
							break;
						} // safety

						int cEv = mesh.Halfedge[current].EndVert;
						if (cEv >= 0)
						{
							int found = FindEndVert(endVerts, cEv);
							if (found >= 0 && endVerts[found].MinEdge != current)
							{
								duplicates.Add(current);
							}
						}
					}
				}

				if (duplicates.Count == 0)
				{
					break;
				}

				foreach (int dup in duplicates)
				{
					DedupeEdge(mesh, dup);
				}
			}
		}

		/// <summary>
		/// Rust <c>end_verts.iter().find(|(v, _)| *v == endVert)</c> — the index of the first
		/// entry for <paramref name="endVert"/>, or -1.
		/// </summary>
		private static int FindEndVert(List<EndVertEntry> endVerts, int endVert)
		{
			for (int j = 0; j < endVerts.Count; j++)
			{
				if (endVerts[j].EndVert == endVert)
				{
					return j;
				}
			}

			return -1;
		}

		/// <summary>
		/// The Rust's <c>(i32, usize)</c> tuple in <c>DedupeEdges</c>: an end vertex and the
		/// smallest halfedge index in the orbit that reaches it.
		/// </summary>
		private readonly struct EndVertEntry
		{
			/// <summary>The end vertex this entry tracks.</summary>
			public readonly int EndVert;

			/// <summary>The smallest halfedge index seen ending at <see cref="EndVert"/>.</summary>
			public readonly int MinEdge;

			/// <summary>Creates an entry.</summary>
			public EndVertEntry(int endVert, int minEdge)
			{
				this.EndVert = endVert;
				this.MinEdge = minEdge;
			}
		}
	}
}
