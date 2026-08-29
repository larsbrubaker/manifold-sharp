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

// The QuickHull driver half of quickhull_algo.rs — see QuickHull.Mesh.cs for
// the arena it grows the hull in, and QuickHull.cs for the module header, the
// split rationale and the geometry helpers.
//
// This file is 867 lines — 67 over the 800-line cap — and takes the exemption CLAUDE.md
// grants quickhull_algo. Splitting further would have to cut between
// SetupInitialTetrahedron and CreateConvexHalfedgeMesh, and those two are one
// argument: the degenerate branches in the first (single point, 1D, planar) are
// what the second's loop is allowed to assume away, and neither is checkable
// without the other in view.
//
// ── Two Rust clones that are borrow-checker artifacts, not semantics ─────────
// `create_convex_halfedge_mesh` clones `visible_faces` and `new_face_indices`
// before iterating them, because the loop bodies call `&mut self` methods. C#
// has no such restriction, and neither loop body touches the list it walks
// (`disable_face` writes `mesh.faces`/`disabled_faces`; `add_point_to_face`
// writes `mesh.faces` and the pool), so both are iterated in place here. The
// third one, `std::mem::take(&mut self.disabled_face_point_vectors)`, is *not*
// dropped: the reclaim inside that loop hands lists back to the pool where
// `add_point_to_face` can immediately take them again, so the walk order and
// the clear-at-the-end both matter and are transcribed.
//
// ── Stale fields on a recycled face, and what guards them ───────────────────
// `AddFace` leaves `MostDistantPoint` (safe by value: read only once
// `PointsOnPositiveSide` is non-empty, which only `AddPointToFace` makes true)
// and `VisibilityCheckedOnIteration`, which is NOT safe by value — a face
// recycled inside iteration K carries K, so `== iter` fires on it. Ordering is
// the guard: K's fill completes before K's first `AddFace`, and K+1 uses a new
// counter. Re-entering the fill after new faces exist in K would call a
// recycled face already-visible and silently drop a horizon edge, unasserted.

using System.Diagnostics;

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;
using static ManifoldSharp.QuickHullFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// The QuickHull algorithm state: the point cloud, the working half-edge mesh, and the
	/// scratch buffers one hull iteration needs.
	/// </summary>
	internal sealed class QuickHull
	{
		private double epsilon;
		private double epsilonSquared;
		private double scale;
		private bool planar;
		private List<Vec3> planarPointCloudTemp;
		private List<Vec3> verts;
		private readonly int originalVertexCount;
		private readonly MeshBuilder mesh;
		private int[] extremeValues;
		private int failedHorizonEdges;

		// Temporary variables used during iteration
		private readonly List<int> newFaceIndices;
		private readonly List<int> newHalfedgeIndices;
		private readonly List<int> visibleFaces;
		private readonly List<int> horizonEdgesData;
		private readonly List<FaceData> possiblyVisibleFaces;
		private readonly List<List<int>> disabledFacePointVectors;
		private readonly Queue<int> faceList;
		private readonly Pool indexVectorPool;

		/// <summary>Creates the algorithm state over a copy of the input points.</summary>
		/// <param name="vertexData">The point cloud to hull.</param>
		public QuickHull(IReadOnlyList<Vec3> vertexData)
		{
			this.epsilon = 0.0;
			this.epsilonSquared = 0.0;
			this.scale = 0.0;
			this.planar = false;
			this.planarPointCloudTemp = new List<Vec3>();
			this.verts = new List<Vec3>(vertexData);
			this.originalVertexCount = vertexData.Count;
			this.mesh = new MeshBuilder();
			this.extremeValues = new int[6];
			this.failedHorizonEdges = 0;
			this.newFaceIndices = new List<int>();
			this.newHalfedgeIndices = new List<int>();
			this.visibleFaces = new List<int>();
			this.horizonEdgesData = new List<int>();
			this.possiblyVisibleFaces = new List<FaceData>();
			this.disabledFacePointVectors = new List<List<int>>();
			this.faceList = new Queue<int>();
			this.indexVectorPool = new Pool();
		}

		/// <summary>
		/// Builds the hull and exports it as a halfedge array in the 3-per-triangle layout
		/// <see cref="ManifoldImpl"/> expects, plus the used vertices.
		/// </summary>
		/// <param name="eps">The relative epsilon, scaled by the cloud's extent.</param>
		/// <returns>The halfedges and the compacted vertex positions.</returns>
		public (List<Halfedge> Halfedges, List<Vec3> Vertices) BuildMesh(double eps)
		{
			if (this.verts.Count == 0)
			{
				return (new List<Halfedge>(), new List<Vec3>());
			}

			this.extremeValues = this.GetExtremeValues();
			int[] ev = this.extremeValues;
			this.scale = this.GetScale(ev);
			this.epsilon = eps * this.scale;
			this.epsilonSquared = this.epsilon * this.epsilon;

			this.planar = false;
			this.CreateConvexHalfedgeMesh();

			if (this.planar)
			{
				// Reset the synthetic extra point back to coincide with verts[0] so
				// the final hull vertices lie on the input plane. Matches C++
				// `planarPointCloudTemp.back() = planarPointCloudTemp.front();`
				// where `originalVertexData` is a view sharing that storage.
				int last = this.verts.Count - 1;
				this.verts[last] = this.verts[0];
				if (this.planarPointCloudTemp.Count != 0)
				{
					int tlast = this.planarPointCloudTemp.Count - 1;
					this.planarPointCloudTemp[tlast] = this.planarPointCloudTemp[0];
				}
			}

			// Reorder halfedges into 3-consecutive-per-face layout
			MeshBuilder mesh = this.mesh;
			int heCount = mesh.Halfedges.Count;

			// The Rust allocates `he_count` halfedges and truncates to `j` at the end; the
			// array here is that allocation and `j` is the truncation, so every loop below
			// that the Rust runs over the truncated vector runs to `j`.
			Halfedge[] halfedges = new Halfedge[heCount];
			for (int i = 0; i < heCount; i++)
			{
				halfedges[i] = new Halfedge(0, 0, -1, 0);
			}

			int[] mapping = new int[heCount];
			int[] counts = new int[Math.Max(heCount, 1)];
			int j = 0;

			for (int i = 0; i < heCount; i++)
			{
				if (mesh.Halfedges[i].PairedHalfedge < 0)
				{
					continue;
				}

				int faceIdx = mesh.HalfedgeToFace[i];
				if (faceIdx < mesh.Faces.Count && mesh.Faces[faceIdx].IsDisabled())
				{
					continue;
				}

				if (counts[faceIdx] > 0)
				{
					continue;
				}

				counts[faceIdx] += 1;

				int currIndex = j;
				j += 3;

				// First halfedge of the face
				mapping[i] = currIndex;
				halfedges[currIndex].EndVert = mesh.Halfedges[i].EndVert;
				halfedges[currIndex].PairedHalfedge = mesh.Halfedges[i].PairedHalfedge;

				// Second
				int k1 = mesh.HalfedgeNext[i];
				mapping[k1] = currIndex + 1;
				halfedges[currIndex + 1].EndVert = mesh.Halfedges[k1].EndVert;
				halfedges[currIndex + 1].PairedHalfedge = mesh.Halfedges[k1].PairedHalfedge;

				// Third
				int k2 = mesh.HalfedgeNext[k1];
				mapping[k2] = currIndex + 2;
				halfedges[currIndex + 2].EndVert = mesh.Halfedges[k2].EndVert;
				halfedges[currIndex + 2].PairedHalfedge = mesh.Halfedges[k2].PairedHalfedge;

				// Set start_vert from the previous halfedge's end_vert
				halfedges[currIndex].StartVert = halfedges[currIndex + 2].EndVert;
				halfedges[currIndex + 1].StartVert = halfedges[currIndex].EndVert;
				halfedges[currIndex + 2].StartVert = halfedges[currIndex + 1].EndVert;
			}

			// Fix paired_halfedge IDs
			for (int i = 0; i < j; i++)
			{
				if (halfedges[i].PairedHalfedge >= 0)
				{
					halfedges[i].PairedHalfedge = mapping[halfedges[i].PairedHalfedge];
				}
			}

			// Remove unused vertices
			List<Vec3> vertData = this.verts;
			int vertCount = vertData.Count;
			int[] vertUsed = new int[vertCount + 1];
			for (int i = 0; i < j / 3; i++)
			{
				vertUsed[halfedges[3 * i].StartVert] += 1;
				vertUsed[halfedges[(3 * i) + 1].StartVert] += 1;
				vertUsed[halfedges[(3 * i) + 2].StartVert] += 1;
			}

			// Exclusive scan with saturation
			int[] prefix = new int[vertCount + 1];
			int running = 0;
			for (int i = 0; i < vertCount; i++)
			{
				prefix[i] = running;
				if (vertUsed[i] > 0)
				{
					running += 1;
				}
			}

			prefix[vertCount] = running;

			// A zeroed Vec3[] is the Rust `vec![Vec3::new(0.0, 0.0, 0.0); running]`.
			Vec3[] vertices = new Vec3[running];
			for (int i = 0; i < vertCount; i++)
			{
				if (prefix[i + 1] - prefix[i] > 0)
				{
					vertices[prefix[i]] = vertData[i];
				}
			}

			// Remap vertex indices in halfedges
			for (int i = 0; i < j; i++)
			{
				halfedges[i].StartVert = prefix[halfedges[i].StartVert];
				halfedges[i].EndVert = prefix[halfedges[i].EndVert];

				// prop_vert mirrors start_vert for hull output (no properties)
				halfedges[i].PropVert = halfedges[i].StartVert;
			}

			List<Halfedge> outHalfedges = new List<Halfedge>(j);
			for (int i = 0; i < j; i++)
			{
				outHalfedges.Add(halfedges[i]);
			}

			return (outHalfedges, new List<Vec3>(vertices));
		}

		private void CreateConvexHalfedgeMesh()
		{
			this.visibleFaces.Clear();
			this.horizonEdgesData.Clear();
			this.possiblyVisibleFaces.Clear();

			this.SetupInitialTetrahedron();
			Debug.Assert(this.mesh.Faces.Count == 4, "the initial tetrahedron must have four faces");

			// Init face stack
			this.faceList.Clear();
			for (int i = 0; i < 4; i++)
			{
				List<int>? initialPoints = this.mesh.Faces[i].PointsOnPositiveSide;
				bool hasInitialPoints = initialPoints != null && initialPoints.Count != 0;
				if (hasInitialPoints)
				{
					this.faceList.Enqueue(i);
					this.mesh.Faces[i].InFaceStack = true;
				}
			}

			ulong iter = 0;
			while (this.faceList.Count > 0)
			{
				int topFaceIndex = this.faceList.Dequeue();
				iter = unchecked(iter + 1);
				if (iter == ulong.MaxValue)
				{
					iter = 0;
				}

				this.mesh.Faces[topFaceIndex].InFaceStack = false;

				List<int>? topPoints = this.mesh.Faces[topFaceIndex].PointsOnPositiveSide;
				bool hasPoints = topPoints != null && topPoints.Count != 0;
				if (!hasPoints || this.mesh.Faces[topFaceIndex].IsDisabled())
				{
					continue;
				}

				int activePointIndex = this.mesh.Faces[topFaceIndex].MostDistantPoint;
				Vec3 activePoint = this.verts[activePointIndex];

				// Find visible faces and horizon edges
				this.horizonEdgesData.Clear();
				this.possiblyVisibleFaces.Clear();
				this.visibleFaces.Clear();
				this.possiblyVisibleFaces.Add(new FaceData
				{
					FaceIndex = topFaceIndex,
					EnteredFromHalfedge = -1,
				});

				while (this.possiblyVisibleFaces.Count > 0)
				{
					FaceData faceData = this.possiblyVisibleFaces[this.possiblyVisibleFaces.Count - 1];
					this.possiblyVisibleFaces.RemoveAt(this.possiblyVisibleFaces.Count - 1);
					int fi = faceData.FaceIndex;
					Debug.Assert(!this.mesh.Faces[fi].IsDisabled(), "flood fill reached a disabled face");

					if (this.mesh.Faces[fi].VisibilityCheckedOnIteration == iter)
					{
						if (this.mesh.Faces[fi].IsVisibleFaceOnCurrentIteration)
						{
							continue;
						}
					}
					else
					{
						Vec3 planeN = this.mesh.Faces[fi].Plane.N;
						double planeD = this.mesh.Faces[fi].Plane.D;
						this.mesh.Faces[fi].VisibilityCheckedOnIteration = iter;
						double d = Dot(planeN, activePoint) + planeD;
						if (d > 0.0)
						{
							this.mesh.Faces[fi].IsVisibleFaceOnCurrentIteration = true;
							this.mesh.Faces[fi].HorizonEdgesOnCurrentIteration = 0;
							this.visibleFaces.Add(fi);
							IVec3 heIndices = this.mesh.GetHalfedgeIndicesOfFaceByIndex(fi);
							for (int k = 0; k < 3; k++)
							{
								int heIndex = heIndices[k];
								int paired = this.mesh.Halfedges[heIndex].PairedHalfedge;
								if (paired != faceData.EnteredFromHalfedge)
								{
									int neighborFace = this.mesh.HalfedgeToFace[paired];
									this.possiblyVisibleFaces.Add(new FaceData
									{
										FaceIndex = neighborFace,
										EnteredFromHalfedge = heIndex,
									});
								}
							}

							continue;
						}

						Debug.Assert(fi != topFaceIndex, "the face the fill started at must be visible");
					}

					// Face is not visible -- the halfedge we came from is a horizon edge
					this.mesh.Faces[fi].IsVisibleFaceOnCurrentIteration = false;
					this.horizonEdgesData.Add(faceData.EnteredFromHalfedge);

					// Mark which halfedge of the source face is the horizon edge
					int sourceFaceIdx = this.mesh.HalfedgeToFace[faceData.EnteredFromHalfedge];
					IVec3 halfEdgesMesh = this.mesh.GetHalfedgeIndicesOfFaceByIndex(sourceFaceIdx);
					int ind;
					if (halfEdgesMesh[0] == faceData.EnteredFromHalfedge)
					{
						ind = 0;
					}
					else if (halfEdgesMesh[1] == faceData.EnteredFromHalfedge)
					{
						ind = 1;
					}
					else
					{
						ind = 2;
					}

					Face sourceFace = this.mesh.Faces[sourceFaceIdx];
					sourceFace.HorizonEdgesOnCurrentIteration =
						(byte)(sourceFace.HorizonEdgesOnCurrentIteration | (1 << ind));
				}

				int horizonEdgeCount = this.horizonEdgesData.Count;

				// Reorder horizon edges to form a loop
				if (!this.ReorderHorizonEdges())
				{
					this.failedHorizonEdges += 1;

					// Remove the active point from the face's point list
					List<int>? pts = this.mesh.Faces[topFaceIndex].PointsOnPositiveSide;
					if (pts != null)
					{
						int pos = pts.IndexOf(activePointIndex);
						if (pos >= 0)
						{
							pts.RemoveAt(pos);
						}

						if (pts.Count == 0)
						{
							this.mesh.Faces[topFaceIndex].PointsOnPositiveSide = null;
							this.indexVectorPool.Reclaim(pts);
						}
					}

					continue;
				}

				// Disable visible faces and reclaim their halfedges
				this.newFaceIndices.Clear();
				this.newHalfedgeIndices.Clear();
				this.disabledFacePointVectors.Clear();
				int disableCounter = 0;

				for (int vf = 0; vf < this.visibleFaces.Count; vf++)
				{
					int faceIndex = this.visibleFaces[vf];
					IVec3 halfEdgesMesh = this.mesh.GetHalfedgeIndicesOfFaceByIndex(faceIndex);
					byte horizonBits = this.mesh.Faces[faceIndex].HorizonEdgesOnCurrentIteration;
					for (int hb = 0; hb < 3; hb++)
					{
						if ((horizonBits & (1 << hb)) == 0)
						{
							if (disableCounter < horizonEdgeCount * 2)
							{
								this.newHalfedgeIndices.Add(halfEdgesMesh[hb]);
								disableCounter += 1;
							}
							else
							{
								this.mesh.DisableHalfedge(halfEdgesMesh[hb]);
							}
						}
					}

					List<int>? disabledPts = this.mesh.DisableFace(faceIndex);
					if (disabledPts != null && disabledPts.Count != 0)
					{
						this.disabledFacePointVectors.Add(disabledPts);
					}
				}

				if (disableCounter < horizonEdgeCount * 2)
				{
					int needed = (horizonEdgeCount * 2) - disableCounter;
					for (int n = 0; n < needed; n++)
					{
						int idx = this.mesh.AddHalfedge();
						this.newHalfedgeIndices.Add(idx);
					}
				}

				// Create new faces using the horizon edge loop
				for (int i = 0; i < horizonEdgeCount; i++)
				{
					int ab = this.horizonEdgesData[i];
					IVec2 abVerts = this.mesh.GetVertexIndicesOfHalfedge(ab);
					int a = abVerts[0];
					int b = abVerts[1];
					int c = activePointIndex;

					int newFaceIndex = this.mesh.AddFace();
					this.newFaceIndices.Add(newFaceIndex);

					int ca = this.newHalfedgeIndices[2 * i];
					int bc = this.newHalfedgeIndices[(2 * i) + 1];

					this.mesh.HalfedgeNext[ab] = bc;
					this.mesh.HalfedgeNext[bc] = ca;
					this.mesh.HalfedgeNext[ca] = ab;

					this.mesh.HalfedgeToFace[bc] = newFaceIndex;
					this.mesh.HalfedgeToFace[ca] = newFaceIndex;
					this.mesh.HalfedgeToFace[ab] = newFaceIndex;

					this.mesh.SetEndVert(ca, a);
					this.mesh.SetEndVert(bc, c);

					Vec3 planeNormal = TriangleNormal(this.verts[a], this.verts[b], activePoint);
					this.mesh.Faces[newFaceIndex].Plane = new Plane(planeNormal, activePoint);
					this.mesh.Faces[newFaceIndex].He = ab;

					// Set paired halfedge links for the new edges
					this.mesh.SetPairedHalfedge(
						ca,
						this.newHalfedgeIndices[i > 0 ? (i * 2) - 1 : (2 * horizonEdgeCount) - 1]);
					this.mesh.SetPairedHalfedge(
						bc,
						this.newHalfedgeIndices[((i + 1) * 2) % (horizonEdgeCount * 2)]);
				}

				// Assign points from disabled faces to new faces
				for (int dv = 0; dv < this.disabledFacePointVectors.Count; dv++)
				{
					List<int> disabledPoints = this.disabledFacePointVectors[dv];
					for (int p = 0; p < disabledPoints.Count; p++)
					{
						int point = disabledPoints[p];
						if (point == activePointIndex)
						{
							continue;
						}

						for (int nf = 0; nf < this.newFaceIndices.Count; nf++)
						{
							if (this.AddPointToFace(this.newFaceIndices[nf], point))
							{
								break;
							}
						}
					}

					// The Rust drains as it goes, so this list is back in the pool — and
					// reachable by the AddPointToFace calls of the *next* iteration of this
					// loop — before the next vector is processed.
					this.indexVectorPool.Reclaim(disabledPoints);
				}

				this.disabledFacePointVectors.Clear();

				// Add new faces to the face list if they have points
				for (int nf = 0; nf < this.newFaceIndices.Count; nf++)
				{
					int newFaceIndex = this.newFaceIndices[nf];
					List<int>? newPoints = this.mesh.Faces[newFaceIndex].PointsOnPositiveSide;
					bool newHasPoints = newPoints != null && newPoints.Count != 0;
					bool inStack = this.mesh.Faces[newFaceIndex].InFaceStack;
					if (newHasPoints && !inStack)
					{
						this.faceList.Enqueue(newFaceIndex);
						this.mesh.Faces[newFaceIndex].InFaceStack = true;
					}
				}
			}

			this.indexVectorPool.Clear();
		}

		private int[] GetExtremeValues()
		{
			List<Vec3> verts = this.verts;
			int[] outIndices = new int[6];
			double[] extremeVals =
			{
				verts[0].X, verts[0].X, verts[0].Y, verts[0].Y, verts[0].Z, verts[0].Z,
			};

			for (int i = 1; i < verts.Count; i++)
			{
				Vec3 pos = verts[i];
				if (pos.X > extremeVals[0])
				{
					extremeVals[0] = pos.X;
					outIndices[0] = i;
				}
				else if (pos.X < extremeVals[1])
				{
					extremeVals[1] = pos.X;
					outIndices[1] = i;
				}

				if (pos.Y > extremeVals[2])
				{
					extremeVals[2] = pos.Y;
					outIndices[2] = i;
				}
				else if (pos.Y < extremeVals[3])
				{
					extremeVals[3] = pos.Y;
					outIndices[3] = i;
				}

				if (pos.Z > extremeVals[4])
				{
					extremeVals[4] = pos.Z;
					outIndices[4] = i;
				}
				else if (pos.Z < extremeVals[5])
				{
					extremeVals[5] = pos.Z;
					outIndices[5] = i;
				}
			}

			return outIndices;
		}

		private double GetScale(int[] extremeValues)
		{
			List<Vec3> verts = this.verts;
			double s = 0.0;
			for (int i = 0; i < 6; i++)
			{
				Vec3 v = verts[extremeValues[i]];
				double val;
				switch (i)
				{
					case 0:
					case 1:
						val = Math.Abs(v.X);
						break;
					case 2:
					case 3:
						val = Math.Abs(v.Y);
						break;
					default:
						val = Math.Abs(v.Z);
						break;
				}

				if (val > s)
				{
					s = val;
				}
			}

			return s;
		}

		private bool ReorderHorizonEdges()
		{
			int n = this.horizonEdgesData.Count;

			// Rust `0..n.saturating_sub(1)`; on a signed int the loop guard already stops
			// at n == 0, so no saturation is needed here.
			for (int i = 0; i < n - 1; i++)
			{
				int endVertex = this.mesh.Halfedges[this.horizonEdgesData[i]].EndVert;
				bool found = false;
				for (int k = i + 1; k < n; k++)
				{
					int paired = this.mesh.Halfedges[this.horizonEdgesData[k]].PairedHalfedge;
					int beginVertex = this.mesh.Halfedges[paired].EndVert;
					if (beginVertex == endVertex)
					{
						(this.horizonEdgesData[i + 1], this.horizonEdgesData[k]) =
							(this.horizonEdgesData[k], this.horizonEdgesData[i + 1]);
						found = true;
						break;
					}
				}

				if (!found)
				{
					return false;
				}
			}

			// Verify loop closure
			if (n > 0)
			{
				int lastEnd = this.mesh.Halfedges[this.horizonEdgesData[n - 1]].EndVert;
				int firstPaired = this.mesh.Halfedges[this.horizonEdgesData[0]].PairedHalfedge;
				int firstBegin = this.mesh.Halfedges[firstPaired].EndVert;
				Debug.Assert(lastEnd == firstBegin, "horizon edges do not form a closed loop");
			}

			return true;
		}

		private void SetupInitialTetrahedron()
		{
			int vertexCount = this.verts.Count;

			if (vertexCount <= 4)
			{
				if (vertexCount < 4)
				{
					List<Vec3> padded = new List<Vec3>(this.verts);
					while (padded.Count < 4)
					{
						padded.Add(padded[padded.Count - 1]);
					}

					this.planarPointCloudTemp = new List<Vec3>(padded);
					this.verts = padded;
				}

				int[] v = { 0, 1, 2, 3 };
				Vec3 nSmall = TriangleNormal(this.verts[v[0]], this.verts[v[1]], this.verts[v[2]]);
				Plane planeSmall = new Plane(nSmall, this.verts[v[0]]);
				if (planeSmall.IsPointOnPositiveSide(this.verts[v[3]]))
				{
					(v[0], v[1]) = (v[1], v[0]);
				}

				this.mesh.Setup(v[0], v[1], v[2], v[3]);
				return;
			}

			// Find two most distant extreme points
			double maxD = this.epsilonSquared;
			int selected0 = 0;
			int selected1 = 0;
			for (int i = 0; i < 6; i++)
			{
				for (int k = i + 1; k < 6; k++)
				{
					double d = SquaredDistance(
						this.verts[this.extremeValues[i]],
						this.verts[this.extremeValues[k]]);
					if (d > maxD)
					{
						maxD = d;
						selected0 = this.extremeValues[i];
						selected1 = this.extremeValues[k];
					}
				}
			}

			if (maxD == this.epsilonSquared)
			{
				// Degenerate: single point
				this.mesh.Setup(0, 1, 2, 3);
				return;
			}

			// Find most distant point from the line
			Vec3 rayS = this.verts[selected0];
			Vec3 rayV = this.verts[selected1] - this.verts[selected0];
			double vInvLenSq = 1.0 / Dot(rayV, rayV);
			maxD = this.epsilonSquared;

			// Rust seeds this with `usize::MAX`; -1 is the int spelling of that sentinel.
			// Neither is ever read: if the scan below improves on nothing, `maxD` is still
			// `epsilonSquared` and the 1D-degenerate branch returns before `maxI` is used.
			int maxI = -1;
			for (int i = 0; i < vertexCount; i++)
			{
				double d = SquaredDistancePointRay(this.verts[i], rayS, rayV, vInvLenSq);
				if (d > maxD)
				{
					maxD = d;
					maxI = i;
				}
			}

			if (maxD == this.epsilonSquared)
			{
				// 1D degenerate
				int third = 0;
				while (third == selected0 || third == selected1)
				{
					third += 1;
				}

				int fourth = third + 1;
				while (fourth == selected0 || fourth == selected1)
				{
					fourth += 1;
				}

				this.mesh.Setup(selected0, selected1, third, fourth);
				return;
			}

			Debug.Assert(selected0 != maxI && selected1 != maxI, "the third base vertex duplicates the first two");
			int[] baseTriangle = { selected0, selected1, maxI };
			Vec3[] baseVerts =
			{
				this.verts[baseTriangle[0]],
				this.verts[baseTriangle[1]],
				this.verts[baseTriangle[2]],
			};

			// Find 4th vertex farthest from the triangle plane
			Vec3 n = TriangleNormal(baseVerts[0], baseVerts[1], baseVerts[2]);
			Plane trianglePlane = new Plane(n, baseVerts[0]);
			maxD = this.epsilon;
			maxI = 0;
			for (int i = 0; i < vertexCount; i++)
			{
				double d = Math.Abs(SignedDistanceToPlane(this.verts[i], trianglePlane));
				if (d > maxD)
				{
					maxD = d;
					maxI = i;
				}
			}

			if (maxD == this.epsilon)
			{
				// 2D planar case: add an extra point above the plane
				this.planar = true;
				Vec3 n1 = TriangleNormal(baseVerts[1], baseVerts[2], baseVerts[0]);
				List<Vec3> temp = new List<Vec3>(this.verts);
				Vec3 extraPoint = n1 + this.verts[0];
				temp.Add(extraPoint);
				maxI = temp.Count - 1;
				this.planarPointCloudTemp = new List<Vec3>(temp);
				this.verts = temp;
			}

			// Enforce CCW orientation
			Plane triPlane = new Plane(n, baseVerts[0]);
			if (triPlane.IsPointOnPositiveSide(this.verts[maxI]))
			{
				(baseTriangle[0], baseTriangle[1]) = (baseTriangle[1], baseTriangle[0]);
			}

			// Create tetrahedron and compute face planes
			this.mesh.Setup(baseTriangle[0], baseTriangle[1], baseTriangle[2], maxI);

			for (int fi = 0; fi < this.mesh.Faces.Count; fi++)
			{
				IVec3 faceVerts = this.mesh.GetVertexIndicesOfFaceByIndex(fi);
				Vec3 n1 = TriangleNormal(
					this.verts[faceVerts[0]],
					this.verts[faceVerts[1]],
					this.verts[faceVerts[2]]);
				Plane plane = new Plane(n1, this.verts[faceVerts[0]]);
				this.mesh.Faces[fi].Plane = plane;
			}

			// Assign points to initial faces
			int vCount = this.originalVertexCount; // Only original points, not the extra planar point
			for (int i = 0; i < vCount; i++)
			{
				for (int fi = 0; fi < this.mesh.Faces.Count; fi++)
				{
					if (this.AddPointToFace(fi, i))
					{
						break;
					}
				}
			}
		}

		private bool AddPointToFace(int faceIndex, int pointIndex)
		{
			Face f = this.mesh.Faces[faceIndex];
			double d = SignedDistanceToPlane(this.verts[pointIndex], f.Plane);
			if (d > 0.0 && d * d > this.epsilonSquared * f.Plane.SqrNLength)
			{
				if (f.PointsOnPositiveSide == null)
				{
					f.PointsOnPositiveSide = this.indexVectorPool.Get();
				}

				f.PointsOnPositiveSide.Add(pointIndex);
				if (d > f.MostDistantPointDist)
				{
					f.MostDistantPointDist = d;
					f.MostDistantPoint = pointIndex;
				}

				return true;
			}

			return false;
		}
	}
}
