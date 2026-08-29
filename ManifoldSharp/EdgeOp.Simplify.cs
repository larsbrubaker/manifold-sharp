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

// EdgeOp.Simplify.cs — the driver layer of edge_op.rs: CleanupTopology,
// SimplifyTopology, RemoveDegenerates and the three passes they sequence
// (CollapseShortEdges, CollapseColinearEdges, SwapDegenerates). The module
// header and the file-split map live in EdgeOp.cs.
//
// Every pass here is two-phase — scan the whole arena into a `flagged` list,
// then act on the flags — and that is not an optimisation. Acting during the
// scan would let a collapse reshape the orbit the scan is mid-walk on, and the
// set of edges C++ acts upon is exactly the set flagged against the *pre-pass*
// mesh. The flag lists are therefore built completely before the first edit.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <content>
	/// The topology-cleanup and simplification drivers of edge_op.rs.
	/// </content>
	public static partial class EdgeOp
	{
		// -----------------------------------------------------------------------
		// CleanupTopology / SimplifyTopology / RemoveDegenerates
		// -----------------------------------------------------------------------

		/// <summary>
		/// Coerces an even-manifold into a proper 2-manifold by splitting non-manifold verts
		/// and deduplicating edges.
		/// </summary>
		/// <param name="mesh">The mesh to edit.</param>
		public static void CleanupTopology(ManifoldImpl mesh)
		{
			if (mesh.Halfedge.Count == 0)
			{
				return;
			}

			SplitPinchedVerts(mesh);
			DedupeEdges(mesh);
		}

		/// <summary>
		/// Collapses short edges and colinear edges, and swaps degenerate edge diagonals.
		/// <paramref name="firstNewVert"/> constrains which verts can be collapsed.
		/// </summary>
		/// <param name="mesh">The mesh to edit.</param>
		/// <param name="firstNewVert">The first vertex index considered new.</param>
		public static void SimplifyTopology(ManifoldImpl mesh, int firstNewVert)
		{
			if (mesh.Halfedge.Count == 0)
			{
				return;
			}

			CleanupTopology(mesh);
			CollapseShortEdges(mesh, firstNewVert);
			CollapseColinearEdges(mesh, firstNewVert);
			SwapDegenerates(mesh, firstNewVert);
			FaceOp.CalculateVertNormals(mesh);
		}

		/// <summary>Like <see cref="SimplifyTopology"/> but without colinear-edge collapse.</summary>
		/// <param name="mesh">The mesh to edit.</param>
		/// <param name="firstNewVert">The first vertex index considered new.</param>
		public static void RemoveDegenerates(ManifoldImpl mesh, int firstNewVert)
		{
			if (mesh.Halfedge.Count == 0)
			{
				return;
			}

			CleanupTopology(mesh);
			CollapseShortEdges(mesh, firstNewVert);
			SwapDegenerates(mesh, firstNewVert);
			FaceOp.CalculateVertNormals(mesh);
		}

		/// <summary>Collapses edges shorter than epsilon_ when at least one endpoint is new.</summary>
		/// <param name="mesh">The mesh to edit.</param>
		/// <param name="firstNewVert">The first vertex index considered new.</param>
		public static void CollapseShortEdges(ManifoldImpl mesh, int firstNewVert)
		{
			List<int> scratch = new List<int>(10);
			int n = mesh.Halfedge.Count;
			List<int> flagged = new List<int>();

			// Per #1671: outside a Boolean (first_new_vert == 0) collapse only up to
			// epsilon to avoid error stacking; in a Boolean we only touch new verts, so
			// we may collapse up to tolerance. The per-edge max length below still
			// restricts new->old edges (old verts only move by epsilon).
			double tol = firstNewVert == 0 ? mesh.Epsilon : mesh.Tolerance;

			for (int i = 0; i < n; i++)
			{
				Halfedge h = mesh.Halfedge[i];
				if (h.PairedHalfedge < 0)
				{
					continue;
				}

				if (h.StartVert < firstNewVert && h.EndVert < firstNewVert)
				{
					continue;
				}

				if (h.StartVert < 0 || h.EndVert < 0)
				{
					continue;
				}

				Vec3 delta = mesh.VertPos[h.EndVert] - mesh.VertPos[h.StartVert];
				double lenSq = Dot(delta, delta);
				double maxLen = h.EndVert < firstNewVert
					? tol * tol
					: mesh.Epsilon * mesh.Epsilon;
				if (lenSq < maxLen)
				{
					flagged.Add(i);
				}
			}

			foreach (int i in flagged)
			{
				scratch.Clear();
				CollapseEdge(mesh, i, scratch, tol, firstNewVert);
			}
		}

		/// <summary>
		/// Collapses redundant colinear edges (edges where the start vertex touches only two
		/// distinct face groups).
		/// </summary>
		/// <param name="mesh">The mesh to edit.</param>
		/// <param name="firstNewVert">The first vertex index considered new.</param>
		public static void CollapseColinearEdges(ManifoldImpl mesh, int firstNewVert)
		{
			List<int> scratch = new List<int>(10);
			while (true)
			{
				int n = mesh.Halfedge.Count;
				List<int> flagged = new List<int>();

				for (int i = 0; i < n; i++)
				{
					Halfedge h = mesh.Halfedge[i];
					if (h.PairedHalfedge < 0 || h.StartVert < firstNewVert)
					{
						continue;
					}

					if (h.StartVert < 0)
					{
						continue;
					}

					if (mesh.MeshRelation.TriRef.Count == 0)
					{
						continue;
					}

					if (i / 3 >= mesh.MeshRelation.TriRef.Count)
					{
						continue;
					}

					TriRef ref0 = mesh.MeshRelation.TriRef[i / 3];
					int current = Types.NextHalfedge(mesh.Halfedge[i].PairedHalfedge);
					if (current >= mesh.Halfedge.Count)
					{
						continue;
					}

					if (current / 3 >= mesh.MeshRelation.TriRef.Count)
					{
						continue;
					}

					TriRef ref1 = mesh.MeshRelation.TriRef[current / 3];
					bool ref1Updated = !ref0.SameFace(ref1);

					bool isRedundant = true;
					int maxOrbit = mesh.Halfedge.Count; // orbit can't exceed total halfedges
					int orbitCount = 0;
					while (current != i)
					{
						orbitCount += 1;
						if (orbitCount > maxOrbit)
						{
							isRedundant = false;
							break;
						}

						// FlagEdge traversal: current = NextHalfedge(halfedge[current].pairedHalfedge)
						int pair = mesh.Halfedge[current].PairedHalfedge;
						if (pair < 0)
						{
							isRedundant = false;
							break;
						}

						current = Types.NextHalfedge(pair);
						if (current >= mesh.Halfedge.Count)
						{
							isRedundant = false;
							break;
						}

						int tri = current / 3;
						if (tri >= mesh.MeshRelation.TriRef.Count)
						{
							isRedundant = false;
							break;
						}

						TriRef refCur = mesh.MeshRelation.TriRef[tri];
						if (!refCur.SameFace(ref0) && !refCur.SameFace(ref1))
						{
							if (!ref1Updated)
							{
								ref1 = refCur;
								ref1Updated = true;
							}
							else
							{
								isRedundant = false;
								break;
							}
						}
					}

					if (isRedundant)
					{
						flagged.Add(i);
					}
				}

				if (flagged.Count == 0)
				{
					break;
				}

				int numCollapsed = 0;
				foreach (int i in flagged)
				{
					scratch.Clear();
					if (CollapseEdge(mesh, i, scratch, -1.0, 0))
					{
						numCollapsed += 1;
					}
				}

				if (numCollapsed == 0)
				{
					break;
				}
			}
		}

		/// <summary>Swaps long edges of degenerate triangles.</summary>
		/// <param name="mesh">The mesh to edit.</param>
		/// <param name="firstNewVert">The first vertex index considered new.</param>
		public static void SwapDegenerates(ManifoldImpl mesh, int firstNewVert)
		{
			if (mesh.FaceNormal.Count == 0)
			{
				return;
			}

			int n = mesh.Halfedge.Count;
			List<int> flagged = new List<int>();

			for (int i = 0; i < n; i++)
			{
				Halfedge h = mesh.Halfedge[i];
				if (h.PairedHalfedge < 0)
				{
					continue;
				}

				// Skip edges where all 4 involved verts are old
				int[] tri0edge = TriOf(i);
				int pair = h.PairedHalfedge;
				int[] tri1edge = TriOf(pair);
				if (mesh.Halfedge[i].StartVert < firstNewVert
					&& mesh.Halfedge[i].EndVert < firstNewVert
					&& mesh.Halfedge[Types.NextHalfedge(i)].EndVert < firstNewVert
					&& mesh.Halfedge[Types.NextHalfedge(pair)].EndVert < firstNewVert)
				{
					continue;
				}

				int tri = i / 3;
				if (tri >= mesh.FaceNormal.Count)
				{
					continue;
				}

				Proj2x3 proj = FaceOp.GetAxisAlignedProjection(mesh.FaceNormal[tri]);
				Vec2[] v = new Vec2[3];
				for (int j = 0; j < 3; j++)
				{
					v[j] = new Vec2(0.0, 0.0);
				}

				for (int j = 0; j < 3; j++)
				{
					int sv = mesh.Halfedge[tri0edge[j]].StartVert;
					if (sv >= 0 && sv < mesh.VertPos.Count)
					{
						v[j] = proj.Apply(mesh.VertPos[sv]);
					}
				}

				if (Polygon.Ccw(v[0], v[1], v[2], mesh.Tolerance) > 0
					|| !Is01Longest(v[0], v[1], v[2]))
				{
					continue;
				}

				// Switch to the neighbor's projection — C++ projects the PAIR
				// triangle's verts (pairTriEdge), not tri0's.
				//
				// RecursiveEdgeSwap does the corresponding step on TRI0's verts. The two
				// sites genuinely differ; do not "fix" either one to match the other.
				int triP = pair / 3;
				if (triP >= mesh.FaceNormal.Count)
				{
					continue;
				}

				Proj2x3 projP = FaceOp.GetAxisAlignedProjection(mesh.FaceNormal[triP]);
				for (int j = 0; j < 3; j++)
				{
					int sv = mesh.Halfedge[tri1edge[j]].StartVert;
					if (sv >= 0 && sv < mesh.VertPos.Count)
					{
						v[j] = projP.Apply(mesh.VertPos[sv]);
					}
				}

				if (Polygon.Ccw(v[0], v[1], v[2], mesh.Tolerance) > 0
					|| Is01Longest(v[0], v[1], v[2]))
				{
					flagged.Add(i);
				}
			}

			List<int> visited = new List<int>(n);
			for (int k = 0; k < n; k++)
			{
				visited.Add(-1);
			}

			List<int> edgeSwapStack = new List<int>();
			List<int> scratch = new List<int>();
			int tag = 0;

			foreach (int i in flagged)
			{
				tag += 1;
				RecursiveEdgeSwap(mesh, i, ref tag, visited, edgeSwapStack, scratch);

				// The stack drain must see the tag increments RecursiveEdgeSwap makes —
				// hence `ref int tag`, not a by-value copy.
				while (edgeSwapStack.Count > 0)
				{
					int e = edgeSwapStack[edgeSwapStack.Count - 1];
					edgeSwapStack.RemoveAt(edgeSwapStack.Count - 1);
					RecursiveEdgeSwap(mesh, e, ref tag, visited, edgeSwapStack, scratch);
				}
			}
		}
	}
}
