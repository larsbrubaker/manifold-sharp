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

// Phase 15: Subdivision — ported from C++ subdivision.cpp (811 lines)
//
// Implements the full subdivision system with:
// - Partition class with cached triangulations
// - Triangle and quad subdivision
// - Barycentric interpolation for vertices and properties
// - Edge-division-based refinement
//
// ── C# port notes ────────────────────────────────────────────────────────────
// subdivision.rs pulls the Partition half in with
// `#[path = "subdivision_partition.rs"] mod subdivision_partition`; here that is
// simply SubdivisionPartition.cs in the same assembly. The module's two free
// functions (`create_tmp_edges`, `subdivide_impl`) land on the static
// <see cref="Subdivision"/> class; the `impl ManifoldImpl` block continues the
// partial class begun in ManifoldImpl.cs.
//
// The `edge_divisions: &dyn Fn(Vec3, Vec4, Vec4) -> i32` parameter becomes a
// <see cref="Func{T1, T2, T3, TResult}"/>. It is called once per edge, in edge
// order, and it must be a pure function of its three arguments — the whole
// subdivision is keyed on the counts it returns, so a callback that consults
// call order or outside state makes the result unreproducible.
//
// `IsMarkedInsideQuad`, used all through this file, is smoothing.rs's and lives
// in Smoothing.cs — the same split the Rust has.

using System.Diagnostics;

using ManifoldSharp.Linalg;

using static ManifoldSharp.Types;

namespace ManifoldSharp
{
	/// <summary>
	/// The free functions of subdivision.rs.
	/// </summary>
	/// <remarks>
	/// Public because lib.rs declares `pub mod subdivision` and `subdivide_impl` is a
	/// `pub fn`. <see cref="CreateTmpEdges"/> is the one exception and stays internal:
	/// the Rust's `create_tmp_edges` is a module-private `fn`, visible to its in-crate
	/// test module and nothing else — which is exactly what internal plus
	/// InternalsVisibleTo gives here.
	/// </remarks>
	public static class Subdivision
	{
		// ---------------------------------------------------------------------------
		// CreateTmpEdges — port of C++ inline CreateTmpEdges()
		// ---------------------------------------------------------------------------

		/// <summary>
		/// One <see cref="TmpEdge"/> per undirected edge, taken from the forward halfedge
		/// of each pair.
		/// </summary>
		/// <param name="halfedge">The mesh's halfedge array.</param>
		/// <returns>The edges, in forward-halfedge order.</returns>
		internal static List<TmpEdge> CreateTmpEdges(IReadOnlyList<Halfedge> halfedge)
		{
			List<TmpEdge> edges = new List<TmpEdge>(halfedge.Count);
			for (int idx = 0; idx < halfedge.Count; idx++)
			{
				Halfedge half = halfedge[idx];
				if (half.IsForward())
				{
					edges.Add(new TmpEdge(half.StartVert, half.EndVert, idx));
				}
			}

			Debug.Assert(
				edges.Count == halfedge.Count / 2,
				$"Not oriented! edges={edges.Count} halfedges={halfedge.Count}");
			return edges;
		}

		/// <summary>
		/// Simple midpoint subdivision: each triangle is split into 4 by inserting
		/// edge midpoints. This is a convenience wrapper that uses uniform n=2 divisions.
		/// </summary>
		/// <param name="mesh">The mesh to subdivide; left untouched.</param>
		/// <param name="levels">How many times to subdivide; 0 returns a plain copy.</param>
		/// <returns>The subdivided mesh.</returns>
		/// <remarks>
		/// Every level must run the whole of `refine`'s finishing tail, not part of it.
		/// <see cref="ManifoldImpl.Subdivide"/> appends vertices and faces, so anything less
		/// leaves a collider built over the *pre*-subdivision faces, a `VertNormal` list
		/// shorter than the vertex list, and — if the caller's mesh had them — tangents that
		/// no longer describe the halfedges. `Boolean3Kernels.Shadow01` reads `VertNormal`
		/// per vertex and would index off the end. The tail below is exactly the one
		/// <see cref="Manifold"/>'s `FinishRefine` runs for the tangent-free case, which is
		/// the same operation this function performs.
		/// (Was docs/RUST_DIVERGENCES.md entry 5, when the Rust's `subdivide_impl` ran only
		/// two of the tail's six steps; upstream fixed in manifold-rust fa18cc5 and the
		/// entry retired. The order is now identical on both sides.)
		/// </remarks>
		public static ManifoldImpl SubdivideImpl(ManifoldImpl mesh, int levels)
		{
			ArgumentNullException.ThrowIfNull(mesh);

			if (levels == 0 || mesh.IsEmpty())
			{
				return mesh.Clone();
			}

			ManifoldImpl current = mesh.Clone();
			for (int i = 0; i < levels; i++)
			{
				current.Subdivide((vec, t0, t1) => 1, false);

				// Subdivide multiplied the halfedges, so any tangents the caller's mesh
				// carried now describe nothing. They must go before SortGeometry, which
				// gathers one tangent per new halfedge out of the old array.
				current.HalfedgeTangent.Clear();
				current.CalculateBBox();
				current.SetEpsilon(-1.0, false);

				// SortGeometry is what rebuilds the face collider, and it has to reorder to
				// do it: Collider's radix tree requires ascending Morton codes. Vertex
				// normals are then recomputed rather than permuted, because the stale list
				// SortVerts saw was too short for it to carry across.
				current.SortGeometry();
				FaceOp.CalculateVertNormals(current);

				// The subdivided mesh is no longer the original it was cloned from.
				current.MeshRelation.OriginalId = -1;
			}

			return current;
		}
	}

	// ---------------------------------------------------------------------------
	// ManifoldImpl methods for subdivision
	// ---------------------------------------------------------------------------

	/// <content>
	/// The subdivision half of <see cref="ManifoldImpl"/>: the quad-detection helpers,
	/// the barycentric bookkeeping and <see cref="Subdivide"/> itself.
	/// </content>
	public sealed partial class ManifoldImpl
	{
		/// <summary>Port of C++ Manifold::Impl::GetNeighbor(int tri).</summary>
		/// <param name="tri">The triangle index.</param>
		/// <returns>The local index (0..2) of the one quad-interior halfedge, -1 if none, -2 if several.</returns>
		public int GetNeighbor(int tri)
		{
			int neighbor = -1;
			for (int i = 0; i < 3; i++)
			{
				if (this.IsMarkedInsideQuad((3 * tri) + i))
				{
					neighbor = neighbor == -1 ? i : -2;
				}
			}

			return neighbor;
		}

		/// <summary>Port of C++ Manifold::Impl::GetHalfedges(int tri).</summary>
		/// <param name="tri">The triangle index.</param>
		/// <returns>
		/// The face's boundary halfedges: three plus a -1 for a triangle, four for a quad,
		/// or all -1 when this is the upper triangle of a quad (only the lower index is processed).
		/// </returns>
		public IVec4 GetHalfedgesQuad(int tri)
		{
			IVec4 halfedges = new IVec4(-1, -1, -1, -1);
			for (int i = 0; i < 3; i++)
			{
				halfedges[i] = (3 * tri) + i;
			}

			int neighbor = this.GetNeighbor(tri);
			if (neighbor >= 0)
			{
				// quad
				int pair = this.Halfedge[(3 * tri) + neighbor].PairedHalfedge;
				if (pair / 3 < tri)
				{
					return new IVec4(-1, -1, -1, -1); // only process lower tri index
				}

				halfedges[2] = NextHalfedge(halfedges[neighbor]);
				halfedges[3] = NextHalfedge(halfedges[2]);
				halfedges[0] = NextHalfedge(pair);
				halfedges[1] = NextHalfedge(halfedges[0]);
			}

			return halfedges;
		}

		/// <summary>Port of C++ Manifold::Impl::GetIndices(int halfedge).</summary>
		private BaryIndices GetIndices(int halfedge)
		{
			int tri = halfedge / 3;
			int idx = halfedge % 3;
			int neighbor = this.GetNeighbor(tri);
			if (idx == neighbor)
			{
				return new BaryIndices { Tri = -1, Start4 = -1, End4 = -1 };
			}

			if (neighbor < 0)
			{
				// tri
				return new BaryIndices { Tri = tri, Start4 = idx, End4 = SubdivisionPartition.Next3(idx) };
			}
			else
			{
				// quad
				int pair = this.Halfedge[(3 * tri) + neighbor].PairedHalfedge;
				if (pair / 3 < tri)
				{
					tri = pair / 3;
					idx = SubdivisionPartition.Next3(neighbor) == idx ? 0 : 1;
				}
				else
				{
					idx = SubdivisionPartition.Next3(neighbor) == idx ? 2 : 3;
				}

				return new BaryIndices { Tri = tri, Start4 = idx, End4 = (idx + 1) % 4 };
			}
		}

		/// <summary>Port of C++ Manifold::Impl::FillRetainedVerts().</summary>
		private void FillRetainedVerts(List<Barycentric> vertBary)
		{
			int numTri = this.Halfedge.Count / 3;
			for (int tri = 0; tri < numTri; tri++)
			{
				for (int i = 0; i < 3; i++)
				{
					BaryIndices indices = this.GetIndices((3 * tri) + i);
					if (indices.Start4 < 0)
					{
						continue; // skip quad interiors
					}

					Vec4 uvw = Vec4.Splat(0.0);
					uvw[indices.Start4] = 1.0;
					vertBary[this.Halfedge[(3 * tri) + i].StartVert] = new Barycentric(indices.Tri, uvw);
				}
			}
		}

		/// <summary>
		/// Port of C++ Manifold::Impl::Subdivide().
		/// </summary>
		/// <param name="edgeDivisions">
		/// Takes (edge_vec, tangent0, tangent1) → number of new vertices for that edge.
		/// </param>
		/// <param name="keepInterior">
		/// When true, short edges get extra divisions so that thin interiors keep enough
		/// triangles to hold their shape.
		/// </param>
		/// <returns>
		/// The barycentric coordinate of every vertex of the subdivided mesh, relative to
		/// the face of the *original* mesh it came from.
		/// </returns>
		/// <remarks>
		/// Returns an UNFINISHED impl — the caller must run the refine tail. Subdivide
		/// appends vertices and faces and rebuilds the halfedges, which leaves the cached
		/// collider describing the pre-subdivision faces, <see cref="VertNormal"/> shorter
		/// than <see cref="VertPos"/>, and any <see cref="HalfedgeTangent"/> at a length
		/// that no longer matches the halfedges. The two callers finish it:
		/// <c>Manifold.FinishRefine</c> in Manifold.Smooth.cs and
		/// <see cref="Subdivision.SubdivideImpl"/>'s per-level tail. A third caller that
		/// skips it hands out a mesh no boolean can consume — <c>Boolean3Kernels.Shadow01</c>
		/// reads <see cref="VertNormal"/> per vertex and indexes off the end. (That was
		/// docs/RUST_DIVERGENCES.md entry 5, upstream fixed in manifold-rust fa18cc5; the
		/// requirement stands on its own.)
		/// </remarks>
		public List<Barycentric> Subdivide(Func<Vec3, Vec4, Vec4, int> edgeDivisions, bool keepInterior)
		{
			ArgumentNullException.ThrowIfNull(edgeDivisions);

			List<TmpEdge> edges = Subdivision.CreateTmpEdges(this.Halfedge);
			int numVert = this.NumVert();
			int numEdge = edges.Count;
			int numTri = this.NumTri();

			// Build half2edge mapping
			int[] half2edge = new int[2 * numEdge];
			for (int edge = 0; edge < numEdge; edge++)
			{
				int idx = edges[edge].HalfedgeIdx;
				half2edge[idx] = edge;
				half2edge[this.Halfedge[idx].PairedHalfedge] = edge;
			}

			// Get face halfedges for each triangle
			IVec4[] faceHalfedges = new IVec4[numTri];
			for (int tri = 0; tri < numTri; tri++)
			{
				faceHalfedges[tri] = this.GetHalfedgesQuad(tri);
			}

			// Compute edge divisions
			int[] edgeAdded = new int[numEdge];
			for (int i = 0; i < numEdge; i++)
			{
				TmpEdge edge = edges[i];
				int hIdx = edge.HalfedgeIdx;
				if (this.IsMarkedInsideQuad(hIdx))
				{
					edgeAdded[i] = 0;
					continue;
				}

				Vec3 vec = this.VertPos[edge.First] - this.VertPos[edge.Second];
				Vec4 tangent0 = this.HalfedgeTangent.Count == 0
					? Vec4.Splat(0.0)
					: this.HalfedgeTangent[hIdx];
				Vec4 tangent1 = this.HalfedgeTangent.Count == 0
					? Vec4.Splat(0.0)
					: this.HalfedgeTangent[this.Halfedge[hIdx].PairedHalfedge];
				edgeAdded[i] = edgeDivisions(vec, tangent0, tangent1);
			}

			// Optional: add extra divisions to short edges for interior thickness
			if (keepInterior)
			{
				int[] origEdgeAdded = (int[])edgeAdded.Clone();
				for (int i = 0; i < numEdge; i++)
				{
					TmpEdge edge = edges[i];
					int hIdx = edge.HalfedgeIdx;
					if (this.IsMarkedInsideQuad(hIdx))
					{
						continue;
					}

					int thisAdded = origEdgeAdded[i];
					int AddedFn(int h)
					{
						int longest = 0;
						int total = 0;
						for (int step = 0; step < 3; step++)
						{
							int added = origEdgeAdded[half2edge[h]];
							longest = Math.Max(longest, added);
							total += added;
							h = NextHalfedge(h);
							if (this.IsMarkedInsideQuad(h))
							{
								longest = 0;
								total = 1;
								break;
							}
						}

						int minExtra = (int)((double)longest * 0.2) + 1;
						int extra = (2 * longest) + minExtra - total;
						if (longest == 0)
						{
							return 0;
						}

						if (extra > 0)
						{
							return (extra * (longest - thisAdded)) / longest;
						}

						return 0;
					}

					int a1 = AddedFn(hIdx);
					int a2 = AddedFn(this.Halfedge[hIdx].PairedHalfedge);
					edgeAdded[i] = origEdgeAdded[i] + Math.Max(a1, a2);
				}
			}

			// Compute edge offsets (exclusive scan)
			int[] edgeOffset = new int[numEdge];
			int acc = numVert;
			for (int i = 0; i < numEdge; i++)
			{
				edgeOffset[i] = acc;
				acc += edgeAdded[i];
			}

			// Allocate vert_bary
			int totalEdgeVerts = acc - numVert;
			List<Barycentric> vertBary = new List<Barycentric>(acc);
			vertBary.Resize(acc, new Barycentric(0, Vec4.Splat(0.0)));
			this.FillRetainedVerts(vertBary);

			// Fill edge vertex barycentric coords
			for (int i = 0; i < numEdge; i++)
			{
				int n = edgeAdded[i];
				int offset = edgeOffset[i];
				BaryIndices indices = this.GetIndices(edges[i].HalfedgeIdx);
				if (indices.Tri < 0)
				{
					continue; // inside quad
				}

				double frac = 1.0 / ((double)n + 1.0);
				for (int j = 0; j < n; j++)
				{
					Vec4 uvw = Vec4.Splat(0.0);
					uvw[indices.End4] = (double)(j + 1) * frac;
					uvw[indices.Start4] = 1.0 - uvw[indices.End4];
					vertBary[offset + j] = new Barycentric(indices.Tri, uvw);
				}
			}

			// Generate partitions for each triangle
			Partition[] subTris = new Partition[numTri];
			for (int tri = 0; tri < numTri; tri++)
			{
				IVec4 halfedges = faceHalfedges[tri];
				IVec4 divisions = default(IVec4);
				for (int i = 0; i < 4; i++)
				{
					if (halfedges[i] >= 0)
					{
						divisions[i] = edgeAdded[half2edge[halfedges[i]]] + 1;
					}
				}

				subTris[tri] = Partition.GetPartition(divisions);
			}

			// Compute triangle offsets (exclusive scan)
			int[] triOffset = new int[numTri];
			{
				int triAcc = 0;
				for (int tri = 0; tri < numTri; tri++)
				{
					triOffset[tri] = triAcc;
					triAcc += subTris[tri].TriVert.Count;
				}
			}

			// Compute interior vertex offsets (exclusive scan)
			int[] interiorOffset = new int[numTri];
			{
				int vertAcc = vertBary.Count;
				for (int tri = 0; tri < numTri; tri++)
				{
					interiorOffset[tri] = vertAcc;
					vertAcc += subTris[tri].NumInterior();
				}
			}

			// Allocate output arrays
			int totalNewTris = numTri > 0
				? triOffset[numTri - 1] + subTris[numTri - 1].TriVert.Count
				: 0;
			int totalNewVerts = numTri > 0
				? interiorOffset[numTri - 1] + subTris[numTri - 1].NumInterior()
				: vertBary.Count;

			List<IVec3> triVerts = new List<IVec3>(totalNewTris);
			triVerts.Resize(totalNewTris, default(IVec3));
			vertBary.Resize(totalNewVerts, new Barycentric(0, Vec4.Splat(0.0)));
			List<TriRef> triRefOut = new List<TriRef>(totalNewTris);
			triRefOut.Resize(totalNewTris, default(TriRef));
			List<Vec3> faceNormalOut = new List<Vec3>(totalNewTris);
			faceNormalOut.Resize(totalNewTris, Vec3.Splat(0.0));

			// Build new triangles
			for (int tri = 0; tri < numTri; tri++)
			{
				IVec4 halfedges = faceHalfedges[tri];
				if (halfedges[0] < 0)
				{
					continue;
				}

				IVec4 tri3 = default(IVec4);
				IVec4 edgeOffs = default(IVec4);
				BVec4 edgeFwd = BVec4.Splat(false);
				for (int i = 0; i < 4; i++)
				{
					if (halfedges[i] < 0)
					{
						tri3[i] = -1;
						continue;
					}

					Halfedge he = this.Halfedge[halfedges[i]];
					tri3[i] = he.StartVert;
					edgeOffs[i] = edgeOffset[half2edge[halfedges[i]]];
					edgeFwd[i] = he.IsForward();
				}

				List<IVec3> newTris = subTris[tri].Reindex(tri3, edgeOffs, edgeFwd, interiorOffset[tri]);

				int start = triOffset[tri];
				for (int j = 0; j < newTris.Count; j++)
				{
					triVerts[start + j] = newTris[j];
					triRefOut[start + j] = this.MeshRelation.TriRef[tri];
					faceNormalOut[start + j] = this.FaceNormal[tri];
				}

				// Map interior barycentric coordinates
				IVec4 idx = subTris[tri].Idx;
				IVec4 vIdx = halfedges[3] >= 0 || idx[1] == SubdivisionPartition.Next3(idx[0])
					? idx
					: new IVec4(idx[2], idx[0], idx[1], idx[3]);
				IVec4 rIdx = default(IVec4);
				for (int i = 0; i < 4; i++)
				{
					rIdx[vIdx[i]] = i;
				}

				List<Vec4> subBary = subTris[tri].VertBary;
				int intOff = subTris[tri].InteriorOffset();
				for (int j = 0; intOff + j < subBary.Count; j++)
				{
					Vec4 bary = subBary[intOff + j];
					vertBary[interiorOffset[tri] + j] = new Barycentric(
						tri,
						new Vec4(
							bary[rIdx[0]],
							bary[rIdx[1]],
							bary[rIdx[2]],
							bary[rIdx[3]]));
				}
			}

			// The Rust assigns the whole vector; MeshRelationD.TriRef is get-only here (the
			// relation table owns its list), so the same content is moved in place.
			this.MeshRelation.TriRef.Clear();
			this.MeshRelation.TriRef.AddRange(triRefOut);
			this.FaceNormal = faceNormalOut;

			// Compute new vertex positions
			List<Vec3> newVertPos = new List<Vec3>(vertBary.Count);
			newVertPos.Resize(vertBary.Count, Vec3.Splat(0.0));
			for (int vert = 0; vert < vertBary.Count; vert++)
			{
				Barycentric bary = vertBary[vert];
				IVec4 halfedges = faceHalfedges[bary.Tri];
				if (halfedges[3] < 0)
				{
					// triangle
					Mat3 triPos = Mat3.FromCols(
						this.VertPos[this.Halfedge[halfedges[0]].StartVert],
						this.VertPos[this.Halfedge[halfedges[1]].StartVert],
						this.VertPos[this.Halfedge[halfedges[2]].StartVert]);
					newVertPos[vert] = triPos * bary.Uvw.Xyz();
				}
				else
				{
					// quad
					Mat3x4 quadPos = Mat3x4.FromCols(
						this.VertPos[this.Halfedge[halfedges[0]].StartVert],
						this.VertPos[this.Halfedge[halfedges[1]].StartVert],
						this.VertPos[this.Halfedge[halfedges[2]].StartVert],
						this.VertPos[this.Halfedge[halfedges[3]].StartVert]);
					newVertPos[vert] = quadPos * bary.Uvw;
				}
			}

			this.VertPos = newVertPos;

			// Handle properties
			if (this.NumProp > 0)
			{
				this.SubdivideProperties(
					vertBary,
					edges,
					faceHalfedges,
					subTris,
					half2edge,
					edgeAdded,
					edgeOffset,
					triOffset,
					interiorOffset,
					triVerts,
					numVert,
					totalEdgeVerts,
					totalNewTris);
			}
			else
			{
				this.CreateHalfedges(triVerts, Array.Empty<IVec3>());
			}

			return vertBary;
		}
	}
}
