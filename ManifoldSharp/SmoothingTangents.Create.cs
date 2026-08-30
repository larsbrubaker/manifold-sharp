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

// SmoothingTangents.Create.cs — the three tangent entry points of
// smoothing_tangents.rs, split out of SmoothingTangents.cs for the 800-line
// cap. That file carries the module notes and the orbit-walk audit.
//
// ── The tangent buffer is built beside the mesh, not on it ───────────────────
// Both creators fill a local `tangent` array and only assign it to
// HalfedgeTangent at the end. That is not tidiness: IsInsideQuad and
// IsMarkedInsideQuad both consult HalfedgeTangent when it is non-empty, so
// publishing the half-built array early would change the answers the rest of
// the loop gets. The array is a Vec4[] rather than a List<Vec4> because
// CreateTangentsFromNormals writes `tangent[i].W` on its own — a field write
// through a List<T> indexer is CS1612, and the array is the shape that permits
// exactly the Rust's statement.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <content>
	/// <c>CreateTangentsFromNormals</c>, <c>ValidTangents</c> and <c>CreateTangents</c>.
	/// </content>
	public sealed partial class ManifoldImpl
	{
		/// <summary>
		/// Builds halfedge tangents from per-property-vertex normals, splitting the
		/// surface wherever two halfedges around a vertex disagree about the normal.
		/// </summary>
		/// <param name="normalIdx">The first of three property slots holding the normal.</param>
		public void CreateTangentsFromNormals(int normalIdx)
		{
			if (this.IsEmpty())
			{
				return;
			}

			// special flags for tangent.w (matches C++ kInsideQuad/kMissingNormal)
			const double KInsideQuad = -1.0;
			const double KMissingNormal = -3.0;

			int numVert = this.NumVert();
			int numHalfedge = this.Halfedge.Count;
			Vec4[] tangent = new Vec4[numHalfedge];
			for (int i = 0; i < numHalfedge; i++)
			{
				tangent[i] = new Vec4(0.0, 0.0, 0.0, 0.0);
			}

			bool[] fixedHalfedge = new bool[numHalfedge];
			List<int> vertHalfedge = this.VertHalfedge();

			for (int v = 0; v < numVert; v++)
			{
				int e = vertHalfedge[v];
				if (e < 0)
				{
					continue;
				}

				List<int> cycle = Smoothing.CollectVertexCycle(this, e);
				int[] faceEdges = new int[] { -1, -1 };
				int startHalfedge = -1;
				Vec3 lastNormal = new Vec3(0.0, 0.0, 0.0);

				for (int idx = 0; idx < cycle.Count; idx++)
				{
					int halfedge = cycle[idx];
					int nextHe = cycle[(idx + 1) % cycle.Count];
					Vec3 hereNormal = this.GetNormal(halfedge, normalIdx);
					Vec3 nextNormal = this.GetNormal(nextHe, normalIdx);

					// Per #1671: a halfedge is flat when its normal matches the face
					// normal within the EqualNormals tolerance (was kPrecision-squared).
					bool hereIsFlat = Smoothing.EqualNormals(hereNormal, this.FaceNormal[halfedge / 3]);
					bool nextIsFlat = Smoothing.EqualNormals(nextNormal, this.FaceNormal[nextHe / 3]);

					// Start with flag clear
					tangent[halfedge].W = 1.0;

					if (hereIsFlat != nextIsFlat)
					{
						// Record halfedges bordering a single flat face
						if (faceEdges[0] == -1)
						{
							faceEdges[0] = halfedge;
						}
						else if (faceEdges[1] == -1)
						{
							faceEdges[1] = halfedge;
						}
						else
						{
							faceEdges[0] = -2;
						}
					}

					bool hereZero = hereNormal == new Vec3(0.0, 0.0, 0.0);
					bool nextZero = nextNormal == new Vec3(0.0, 0.0, 0.0);

					if (hereZero || nextZero)
					{
						if (!hereZero)
						{
							// next missing — record the last good normal
							lastNormal = hereNormal;
						}
						else if (!nextZero)
						{
							// here missing, next present — record start of missing segment
							if (startHalfedge < 0)
							{
								startHalfedge = halfedge;
							}
						}
						else
						{
							// both missing
							if (startHalfedge < 0)
							{
								startHalfedge = -2;
							}
						}

						tangent[halfedge] = new Vec4(lastNormal.X, lastNormal.Y, lastNormal.Z, KMissingNormal);
					}

					if (this.IsInsideQuad(halfedge))
					{
						tangent[halfedge] = new Vec4(lastNormal.X, lastNormal.Y, lastNormal.Z, KInsideQuad);
					}

					if (tangent[halfedge].W < 0.0)
					{
						continue;
					}

					bool differentNormals = !Smoothing.EqualNormals(nextNormal, hereNormal);
					if (differentNormals)
					{
						fixedHalfedge[halfedge] = true;
						faceEdges[0] = -2; // override flat face logic when multiple normals present
					}

					if (differentNormals)
					{
						Vec3 edgeVec = this.VertPos[this.Halfedge[halfedge].EndVert]
							- this.VertPos[this.Halfedge[halfedge].StartVert];
						Vec3 dir = Cross(hereNormal, nextNormal);
						Vec3 signedDir = Dot(dir, edgeVec) < 0.0 ? -dir : dir;
						tangent[halfedge] = Smoothing.CircularTangent(signedDir, edgeVec);
					}
					else
					{
						tangent[halfedge] = this.TangentFromNormal(hereNormal, halfedge);
					}
				}

				// All normals missing: use vertex pseudonormal
				bool lastZero = lastNormal == new Vec3(0.0, 0.0, 0.0);
				if (startHalfedge != -1 && lastZero)
				{
					int vert = this.Halfedge[e].StartVert;
					Vec3 normal = this.VertNormal[vert];
					foreach (int halfedge in cycle)
					{
						if (tangent[halfedge].W != KInsideQuad)
						{
							tangent[halfedge] = this.TangentFromNormal(normal, halfedge);
						}
					}

					continue;
				}

				// Some normals missing: orbit backwards from start_halfedge to fill in
				if (startHalfedge >= 0)
				{
					int start = startHalfedge;

					// prevNormal = GetNormal(NextHalfedge(paired(start)), normalIdx)
					int pairedStart = this.Halfedge[start].PairedHalfedge;
					int nextOfPaired = Types.NextHalfedge(pairedStart);
					Vec3 prevNorm = this.GetNormal(nextOfPaired, normalIdx);

					// ORBIT WALK (backward) — see SmoothingTangents.cs's header.
					int current = start;
					while (true)
					{
						if (tangent[current].W == KMissingNormal)
						{
							Vec3 stored = new Vec3(tangent[current].X, tangent[current].Y, tangent[current].Z);
							Vec3 nextNorm = stored == new Vec3(0.0, 0.0, 0.0) ? lastNormal : stored;

							if (Smoothing.EqualNormals(prevNorm, nextNorm))
							{
								tangent[current] = this.TangentFromNormal(prevNorm, current);
							}
							else
							{
								Vec3 dir = Cross(prevNorm, nextNorm);
								Vec3 edgeVec = this.VertPos[this.Halfedge[current].EndVert]
									- this.VertPos[this.Halfedge[current].StartVert];
								Vec3 signedDir = Dot(dir, edgeVec) < 0.0 ? -dir : dir;
								tangent[current] = Smoothing.CircularTangent(signedDir, edgeVec);
							}
						}

						Vec3 currentNormal = this.GetNormal(current, normalIdx);
						if (currentNormal != new Vec3(0.0, 0.0, 0.0))
						{
							prevNorm = currentNormal;
						}

						// advance backward: paired(PrevHalfedge(current))
						int prevHe = Types.PrevHalfedge(current);
						current = this.Halfedge[prevHe].PairedHalfedge;
						if (current == start)
						{
							break;
						}
					}
				}

				if (faceEdges[0] >= 0 && faceEdges[1] >= 0)
				{
					int f0 = faceEdges[0];
					int f1 = faceEdges[1];
					Vec3 edge0 = this.VertPos[this.Halfedge[f0].EndVert] - this.VertPos[this.Halfedge[f0].StartVert];
					Vec3 edge1 = this.VertPos[this.Halfedge[f1].EndVert] - this.VertPos[this.Halfedge[f1].StartVert];
					Vec3 newTangent = Normalize(edge0) - Normalize(edge1);
					tangent[f0] = Smoothing.CircularTangent(newTangent, edge0);
					tangent[f1] = Smoothing.CircularTangent(-newTangent, edge1);

					// Fix these tangents to keep them aligned to the edges
					fixedHalfedge[f0] = true;
					fixedHalfedge[f1] = true;
				}
			}

			this.HalfedgeTangent = new List<Vec4>(tangent);
			this.DistributeTangents(fixedHalfedge);
		}

		/// <summary>
		/// Returns true if halfedge tangents form a valid quad/triangle arrangement.
		/// Checks that kInsideQuad (-1.0) markers are consistent: paired halfedges
		/// must agree, and marked halfedges cannot be adjacent within a triangle.
		/// </summary>
		/// <returns>True when the quad markers are consistent.</returns>
		public bool ValidTangents()
		{
			if (this.HalfedgeTangent.Count != this.Halfedge.Count)
			{
				return true; // no tangents means nothing to validate
			}

			int numHalfedge = this.Halfedge.Count;
			for (int edgeIdx = 0; edgeIdx < numHalfedge; edgeIdx++)
			{
				bool inQuad = this.IsMarkedInsideQuad(edgeIdx);
				int pair = this.Halfedge[edgeIdx].PairedHalfedge;
				if (inQuad != this.IsMarkedInsideQuad(pair))
				{
					return false;
				}

				if (!inQuad)
				{
					continue;
				}

				// A kInsideQuad halfedge cannot have adjacent kInsideQuad halfedges
				int nextE = Types.NextHalfedge(edgeIdx);
				int prevE = Types.PrevHalfedge(edgeIdx);
				int pairNext = Types.NextHalfedge(pair);
				int pairPrev = Types.PrevHalfedge(pair);
				if (this.IsMarkedInsideQuad(nextE)
					|| this.IsMarkedInsideQuad(prevE)
					|| this.IsMarkedInsideQuad(pairNext)
					|| this.IsMarkedInsideQuad(pairPrev))
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// Builds halfedge tangents from vertex normals, sharpening the edges named in
		/// <paramref name="sharpenedEdges"/> and every edge of a flat face.
		/// </summary>
		/// <param name="sharpenedEdges">
		/// The requested per-halfedge smoothness values. The Rust takes this by value and
		/// pushes the flat-face edges onto it; the copy taken below reproduces that without
		/// mutating the caller's list.
		/// </param>
		public void CreateTangents(IReadOnlyList<Smoothness> sharpenedEdges)
		{
			if (this.IsEmpty())
			{
				return;
			}

			List<Smoothness> sharpened = new List<Smoothness>(sharpenedEdges);

			int numHalfedge = this.Halfedge.Count;
			List<int> vertHalfedge = this.VertHalfedge();
			List<bool> triIsFlatFace = this.FlatFaces();
			List<int> vertFlatFace = this.VertFlatFace(triIsFlatFace);
			List<Vec3> vertNormal = new List<Vec3>(this.VertNormal);
			for (int v = 0; v < this.NumVert(); v++)
			{
				if (vertFlatFace[v] >= 0)
				{
					vertNormal[v] = this.FaceNormal[vertFlatFace[v]];
				}
			}

			Vec4[] tangent = new Vec4[numHalfedge];
			bool[] fixedHalfedge = new bool[numHalfedge];
			for (int edgeIdx = 0; edgeIdx < numHalfedge; edgeIdx++)
			{
				tangent[edgeIdx] = this.IsInsideQuad(edgeIdx)
					? new Vec4(0.0, 0.0, 0.0, -1.0)
					: this.TangentFromNormal(vertNormal[this.Halfedge[edgeIdx].StartVert], edgeIdx);
			}

			this.HalfedgeTangent = new List<Vec4>(tangent);

			for (int tri = 0; tri < this.NumTri(); tri++)
			{
				if (!triIsFlatFace[tri])
				{
					continue;
				}

				for (int j = 0; j < 3; j++)
				{
					int tri2 = this.Halfedge[(3 * tri) + j].PairedHalfedge / 3;
					if (!triIsFlatFace[tri2]
						|| !this.MeshRelation.TriRef[tri].SameFace(this.MeshRelation.TriRef[tri2]))
					{
						sharpened.Add(new Smoothness((3 * tri) + j, 0.0));
					}
				}
			}

			// The Rust's BTreeMap: iteration below is in ascending halfedge order, and that
			// order reaches the output through the per-vertex lists it fills, so a plain
			// Dictionary would not do.
			SortedDictionary<int, (Smoothness Forward, Smoothness Reverse)> edges =
				new SortedDictionary<int, (Smoothness, Smoothness)>();
			foreach (Smoothness edge in sharpened)
			{
				if (edge.SmoothnessValue >= 1.0)
				{
					continue;
				}

				bool forward = this.Halfedge[edge.Halfedge].IsForward();
				int pair = this.Halfedge[edge.Halfedge].PairedHalfedge;
				int idx = forward ? edge.Halfedge : pair;
				if (edges.TryGetValue(idx, out (Smoothness Forward, Smoothness Reverse) existing))
				{
					if (forward)
					{
						existing.Forward.SmoothnessValue =
							MinF64(existing.Forward.SmoothnessValue, edge.SmoothnessValue);
					}
					else
					{
						existing.Reverse.SmoothnessValue =
							MinF64(existing.Reverse.SmoothnessValue, edge.SmoothnessValue);
					}

					edges[idx] = existing;
				}
				else
				{
					(Smoothness, Smoothness) pairEntry = (edge, new Smoothness(pair, 1.0));
					if (!forward)
					{
						pairEntry = (pairEntry.Item2, pairEntry.Item1);
					}

					edges[idx] = pairEntry;
				}
			}

			SortedDictionary<int, List<(Smoothness Forward, Smoothness Reverse)>> vertTangents =
				new SortedDictionary<int, List<(Smoothness, Smoothness)>>();
			foreach ((Smoothness Forward, Smoothness Reverse) edge in edges.Values)
			{
				AddVertTangent(vertTangents, this.Halfedge[edge.Forward.Halfedge].StartVert, edge);
				AddVertTangent(
					vertTangents,
					this.Halfedge[edge.Reverse.Halfedge].StartVert,
					(edge.Reverse, edge.Forward));
			}

			for (int v = 0; v < this.NumVert(); v++)
			{
				if (!vertTangents.TryGetValue(v, out List<(Smoothness Forward, Smoothness Reverse)>? vert))
				{
					if (vertHalfedge[v] >= 0)
					{
						fixedHalfedge[vertHalfedge[v]] = true;
					}

					continue;
				}

				if (vert.Count == 1)
				{
					continue;
				}

				if (vert.Count == 2)
				{
					int first = vert[0].Forward.Halfedge;
					int second = vert[1].Forward.Halfedge;
					fixedHalfedge[first] = true;
					fixedHalfedge[second] = true;
					Vec3 newTangent = Normalize(
						Smoothing.Vec3FromVec4(this.HalfedgeTangent[first])
						- Smoothing.Vec3FromVec4(this.HalfedgeTangent[second]));
					Vec3 pos = this.VertPos[this.Halfedge[first].StartVert];
					this.HalfedgeTangent[first] = Smoothing.CircularTangent(
						newTangent,
						this.VertPos[this.Halfedge[first].EndVert] - pos);
					this.HalfedgeTangent[second] = Smoothing.CircularTangent(
						-newTangent,
						this.VertPos[this.Halfedge[second].EndVert] - pos);

					double smoothness = (vert[0].Reverse.SmoothnessValue + vert[1].Forward.SmoothnessValue) / 2.0;
					foreach (int current in Smoothing.CollectVertexCycle(this, first))
					{
						if (current == second)
						{
							smoothness = (vert[1].Reverse.SmoothnessValue + vert[0].Forward.SmoothnessValue) / 2.0;
						}
						else if (current != first && !this.IsMarkedInsideQuad(current))
						{
							this.SharpenTangent(current, smoothness);
						}
					}
				}
				else
				{
					double smoothness = 0.0;
					double denom = 0.0;
					foreach ((Smoothness Forward, Smoothness Reverse) pair in vert)
					{
						smoothness += pair.Forward.SmoothnessValue + pair.Reverse.SmoothnessValue;
						denom += pair.Forward.SmoothnessValue == 0.0 ? 0.0 : 1.0;
						denom += pair.Reverse.SmoothnessValue == 0.0 ? 0.0 : 1.0;
					}

					if (denom > 0.0)
					{
						smoothness /= denom;
					}

					foreach (int current in Smoothing.CollectVertexCycle(this, vert[0].Forward.Halfedge))
					{
						if (!this.IsMarkedInsideQuad(current))
						{
							int pair = this.Halfedge[current].PairedHalfedge;
							double s = triIsFlatFace[current / 3] || triIsFlatFace[pair / 3] ? 0.0 : smoothness;
							this.SharpenTangent(current, s);
						}
					}
				}
			}

			this.LinearizeFlatTangents();
			this.DistributeTangents(fixedHalfedge);
		}

		/// <summary>
		/// The Rust's <c>entry(v).or_default().push(edge)</c> — appends to the per-vertex
		/// list, creating it on first use.
		/// </summary>
		private static void AddVertTangent(
			SortedDictionary<int, List<(Smoothness Forward, Smoothness Reverse)>> vertTangents,
			int vert,
			(Smoothness Forward, Smoothness Reverse) edge)
		{
			if (!vertTangents.TryGetValue(vert, out List<(Smoothness Forward, Smoothness Reverse)>? list))
			{
				list = new List<(Smoothness Forward, Smoothness Reverse)>();
				vertTangents[vert] = list;
			}

			list.Add(edge);
		}
	}
}
