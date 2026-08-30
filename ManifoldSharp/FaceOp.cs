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

// face_op.rs — Phase 7a: Face normals, coplanarity, vertex normals
//
// Ports src/face_op.cpp, the face-normal and coplanarity portions of
// src/impl.cpp (SetNormalsAndCoplanar, CalculateVertNormals), and the
// GetAxisAlignedProjection utility from src/shared.h.
//
// ── C# port notes ────────────────────────────────────────────────────────────
// The Rust module's free functions land on a static class named for the module
// (`FaceOp`), per the naming rule in CLAUDE.md; `Proj2x3` is a type and stays a
// namespace-level struct. `face_op.rs` pulls its Face2Tri half in with
// `#[path = "face_op_triangulate.rs"] mod face_op_triangulate` and re-exports
// it, so `Face2Tri` / `Face2TriCt` are static members of this same `FaceOp`
// class, declared in FaceOpTriangulate.cs — the C# equivalent of that
// re-export, and the reason this class is `partial`.
//
// Rust `linalg::length2` is `LinalgFunctions.LengthSquared` here, and
// `math::acos` is `DeterministicMath.Acos` — never `Math.Acos`, which is not
// bit-stable across platforms.
//
// ── File split ───────────────────────────────────────────────────────────────
// face_op.rs is 713 lines; its C# expansion does not fit the 800-line cap, so
// it lands as four files. The first three continue one `static partial class
// FaceOp`; the fourth is the Rust file's one `impl ManifoldImpl` block, so it
// continues that class instead:
//   FaceOp.cs             this file — Proj2x3, GetAxisAlignedProjection,
//                         SetNormalsAndCoplanar, CalculateVertNormals
//   FaceOp.Helpers.cs     GetBarycentric, AssembleHalfedges, ProjectPolygons,
//                         ReorderHalfedges
//   FaceOpTriangulate.cs  face_op_triangulate.rs — Face2Tri and its two
//                         triangle writers, which face_op.rs re-exports
//   FaceOp.Slice.cs       ManifoldImpl.Slice / .Project, carrying the one
//                         documented divergence in this module: the Rust seeds
//                         each slice loop from a randomly-ordered HashSet, and
//                         this port pins that seed (docs/RUST_DIVERGENCES.md
//                         entry 3)
//
// ── Signed indices, not usize ────────────────────────────────────────────────
// Several guards in the Rust read `x as usize < slice.len()`, which rejects a
// negative sentinel (-1 becomes 2^64-1) as well as an over-large index. The
// same expression on a C# `int` *accepts* -1 and then indexes out of bounds, so
// those guards are transcribed as unsigned comparisons and marked where they
// appear.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	// -----------------------------------------------------------------------
	// Proj2x3 — 2-row, 3-column projection matrix (drops one axis)
	//
	// Used to project 3D mesh positions onto a 2D plane for CCW tests and
	// triangulation. Mirrors `mat2x3` in the C++ linalg.h library.
	// -----------------------------------------------------------------------

	/// <summary>
	/// A 2x3 projection matrix: maps <see cref="Vec3"/> to <see cref="Vec2"/> via dot
	/// products with two rows.
	/// </summary>
	/// <remarks>
	/// Plain mutable fields, not a <c>readonly struct</c>: <see cref="Vec3"/> has a
	/// ref-returning indexer, and the Linalg folder header forbids holding those types in
	/// readonly storage (the write lands in a defensive copy, silently).
	/// </remarks>
	public struct Proj2x3
	{
		/// <summary>The row whose dot product with the input gives the x output.</summary>
		public Vec3 Row0;

		/// <summary>The row whose dot product with the input gives the y output.</summary>
		public Vec3 Row1;

		/// <summary>Creates a projection from its two rows.</summary>
		/// <param name="row0">The row producing the x output.</param>
		/// <param name="row1">The row producing the y output.</param>
		public Proj2x3(Vec3 row0, Vec3 row1)
		{
			this.Row0 = row0;
			this.Row1 = row1;
		}

		/// <summary>Apply the projection: <c>[dot(row0, v), dot(row1, v)]</c>.</summary>
		/// <param name="v">The 3D point to project.</param>
		/// <returns>The projected 2D point.</returns>
		public Vec2 Apply(Vec3 v)
		{
			return new Vec2(Dot(this.Row0, v), Dot(this.Row1, v));
		}
	}

	/// <summary>
	/// Face normals, coplanar-group flood fill, vertex normals and the polygon-assembly
	/// helpers — the free functions of <c>face_op.rs</c>.
	/// </summary>
	public static partial class FaceOp
	{
		// -------------------------------------------------------------------
		// GetAxisAlignedProjection
		// -------------------------------------------------------------------

		/// <summary>
		/// Returns a projection matrix that drops the largest-magnitude axis of
		/// <paramref name="normal"/>, producing a 2D view aligned with the face plane.
		/// </summary>
		/// <remarks>Mirrors <c>GetAxisAlignedProjection</c> in <c>src/shared.h</c>.</remarks>
		/// <param name="normal">The face normal to align with.</param>
		/// <returns>The 2x3 projection.</returns>
		public static Proj2x3 GetAxisAlignedProjection(Vec3 normal)
		{
			Vec3 abs = new Vec3(Math.Abs(normal.X), Math.Abs(normal.Y), Math.Abs(normal.Z));

			// mat3x2 columns (each col is a Vec3); transposed to get mat2x3 rows.
			Vec3 row0;
			Vec3 row1;
			double xyzMax;
			if (abs.Z > abs.X && abs.Z > abs.Y)
			{
				// Drop Z, keep X and Y
				row0 = new Vec3(1.0, 0.0, 0.0);
				row1 = new Vec3(0.0, 1.0, 0.0);
				xyzMax = normal.Z;
			}
			else if (abs.Y > abs.X)
			{
				// Drop Y, keep Z and X
				row0 = new Vec3(0.0, 0.0, 1.0);
				row1 = new Vec3(1.0, 0.0, 0.0);
				xyzMax = normal.Y;
			}
			else
			{
				// Drop X, keep Y and Z
				row0 = new Vec3(0.0, 1.0, 0.0);
				row1 = new Vec3(0.0, 0.0, 1.0);
				xyzMax = normal.X;
			}

			// If the dominant axis is negative, flip the first row so that the
			// projected winding order is consistent.
			if (xyzMax < 0.0)
			{
				return new Proj2x3(new Vec3(-row0.X, -row0.Y, -row0.Z), row1);
			}

			return new Proj2x3(row0, row1);
		}

		// -------------------------------------------------------------------
		// SetNormalsAndCoplanar
		// -------------------------------------------------------------------

		/// <summary>
		/// Computes face normals and flood-fills coplanar face groups, then calls
		/// <see cref="CalculateVertNormals"/> to compute per-vertex normals.
		/// </summary>
		/// <remarks>
		/// Mirrors <c>Manifold::Impl::SetNormalsAndCoplanar()</c> in <c>src/impl.cpp</c>.
		/// </remarks>
		/// <param name="mesh">The mesh whose normals and coplanar IDs are filled in.</param>
		public static void SetNormalsAndCoplanar(ManifoldImpl mesh)
		{
			int numTri = mesh.NumTri();

			// Rust `Vec::resize`: grows with (0, 0, 1) and truncates, but leaves any
			// existing entry alone — the loop below overwrites every live triangle.
			mesh.FaceNormal.Resize(numTri, new Vec3(0.0, 0.0, 1.0));

			// Compute face normals and priorities (sort largest faces first)
			List<TriPriority> triPriority = new List<TriPriority>(numTri);
			for (int tri = 0; tri < numTri; tri++)
			{
				// Mark coplanarID as unset. TriRef is a struct in a List, so this is
				// read-modify-store; `TriRef[tri].CoplanarId = -1` is CS1612.
				if (tri < mesh.MeshRelation.TriRef.Count)
				{
					TriRef triRef = mesh.MeshRelation.TriRef[tri];
					triRef.CoplanarId = -1;
					mesh.MeshRelation.TriRef[tri] = triRef;
				}

				if (mesh.Halfedge[3 * tri].StartVert < 0)
				{
					triPriority.Add(new TriPriority(0.0, tri));
					continue;
				}

				Vec3 v = mesh.VertPos[mesh.Halfedge[3 * tri].StartVert];
				Vec3 n = Cross(
					mesh.VertPos[mesh.Halfedge[3 * tri].EndVert] - v,
					mesh.VertPos[mesh.Halfedge[(3 * tri) + 1].EndVert] - v);
				Vec3 normal = Normalize(n);
				mesh.FaceNormal[tri] = double.IsNaN(normal.X)
					? new Vec3(0.0, 0.0, 1.0)
					: normal;
				triPriority.Add(new TriPriority(LengthSquared(n), tri));
			}

			// Sort by area descending (largest triangles first → better coplanar seeds)
			//
			// SORT AUDIT (face_op.rs:119): Rust `sort_by`, which is STABLE, comparing
			// `b.area2.partial_cmp(&a.area2).unwrap_or(Equal)`. Equal-area triangles are
			// the common case, not the exception — every quad split into two halves, every
			// flat face of a primitive — and the tie order decides which of them seeds its
			// coplanar group, which becomes the CoplanarId the boolean engine's face
			// merging keys on. `Polygon.PartialCmpDescending` is exactly that comparator
			// (operands swapped, NaN compares Equal rather than sorting to one end), and
			// LINQ OrderBy is the stable sort. `List<T>.Sort` is introsort and is not
			// usable here.
			List<TriPriority> sortedPriority =
				triPriority.OrderBy(tp => tp.Area2, Polygon.PartialCmpDescending).ToList();

			// Flood-fill coplanar groups from each unassigned face
			List<int> interiorHalfedges = new List<int>();
			foreach (TriPriority tp in sortedPriority)
			{
				int tri = tp.Tri;
				if (tri >= mesh.MeshRelation.TriRef.Count)
				{
					continue;
				}

				if (mesh.MeshRelation.TriRef[tri].CoplanarId >= 0)
				{
					continue;
				}

				{
					TriRef triRef = mesh.MeshRelation.TriRef[tri];
					triRef.CoplanarId = tri;
					mesh.MeshRelation.TriRef[tri] = triRef;
				}

				if (mesh.Halfedge[3 * tri].StartVert < 0)
				{
					continue;
				}

				Vec3 baseVert = mesh.VertPos[mesh.Halfedge[3 * tri].StartVert];
				Vec3 normal = mesh.FaceNormal[tri];

				interiorHalfedges.Clear();
				interiorHalfedges.Add(3 * tri);
				interiorHalfedges.Add((3 * tri) + 1);
				interiorHalfedges.Add((3 * tri) + 2);

				while (interiorHalfedges.Count > 0)
				{
					int h = interiorHalfedges[interiorHalfedges.Count - 1];
					interiorHalfedges.RemoveAt(interiorHalfedges.Count - 1);

					int paired = mesh.Halfedge[h].PairedHalfedge;
					if (paired < 0)
					{
						continue;
					}

					int h2 = Types.NextHalfedge(paired);
					int h2Tri = h2 / 3;
					if (h2Tri >= mesh.MeshRelation.TriRef.Count)
					{
						continue;
					}

					if (mesh.MeshRelation.TriRef[h2Tri].CoplanarId >= 0)
					{
						continue;
					}

					Vec3 v = mesh.VertPos[mesh.Halfedge[h2].EndVert];
					if (Math.Abs(Dot(v - baseVert, normal)) < mesh.Tolerance)
					{
						TriRef triRef = mesh.MeshRelation.TriRef[h2Tri];
						triRef.CoplanarId = tri;
						mesh.MeshRelation.TriRef[h2Tri] = triRef;
						mesh.FaceNormal[h2Tri] = normal;

						// Avoid re-pushing paired interior halfedges (cancel out).
						// The Rust compares `Option<usize>` against
						// `Some(paired_halfedge as usize)`, so an unpaired -1 widens to
						// 2^64-1 and never matches; an int -1 never matches a stack entry
						// either, since every entry is a non-negative halfedge index.
						if (interiorHalfedges.Count > 0
							&& interiorHalfedges[interiorHalfedges.Count - 1]
								== mesh.Halfedge[h2].PairedHalfedge)
						{
							interiorHalfedges.RemoveAt(interiorHalfedges.Count - 1);
						}
						else
						{
							interiorHalfedges.Add(h2);
						}

						interiorHalfedges.Add(Types.NextHalfedge(h2));
					}
				}
			}

			CalculateVertNormals(mesh);
		}

		// -------------------------------------------------------------------
		// CalculateVertNormals
		// -------------------------------------------------------------------

		/// <summary>
		/// Computes per-vertex normals as angle-weighted averages of incident face normals.
		/// </summary>
		/// <remarks>
		/// Mirrors <c>Manifold::Impl::CalculateVertNormals()</c> in <c>src/impl.cpp</c>.
		/// </remarks>
		/// <param name="mesh">The mesh whose <see cref="ManifoldImpl.VertNormal"/> is filled in.</param>
		public static void CalculateVertNormals(ManifoldImpl mesh)
		{
			int numVert = mesh.VertPos.Count;

			// Dead in effect — the whole list is replaced below — but the Rust resizes
			// first and the transcription keeps it, so a future edit that stops replacing
			// wholesale still starts from the Rust's state.
			mesh.VertNormal.Resize(numVert, new Vec3(0.0, 0.0, 0.0));

			// For each vertex, find the first halfedge that starts there
			int[] vertFirstEdge = new int[numVert];
			Array.Fill(vertFirstEdge, int.MaxValue);
			for (int i = 0; i < mesh.Halfedge.Count; i++)
			{
				int sv = mesh.Halfedge[i].StartVert;
				if (sv >= 0 && sv < numVert)
				{
					if (i < vertFirstEdge[sv])
					{
						vertFirstEdge[sv] = i;
					}
				}
			}

			// Each vertex's normal reads only shared mesh data and its own walk, so
			// the per-vertex work parallelizes with results identical to sequential
			// (the accumulation order WITHIN a vertex is the fixed ForVert walk).
			List<Halfedge> halfedge = mesh.Halfedge;
			List<Vec3> vertPos = mesh.VertPos;
			List<Vec3> faceNormal = mesh.FaceNormal;
			Vec3[] normals = Par.MaybeParMap(numVert, 10_000, vert =>
			{
				int firstEdge = vertFirstEdge[vert];
				if (firstEdge == int.MaxValue)
				{
					return new Vec3(0.0, 0.0, 0.0);
				}

				Vec3 normal = new Vec3(0.0, 0.0, 0.0);

				// ForVert equivalent: walk CW around the vertex. C++ ForVert (impl.h)
				// STEPS FIRST and calls func after, so first_edge is processed LAST.
				// The visit order fixes the float accumulation order of the
				// angle-weighted normal sum — it must match C++ bit-for-bit because
				// vertex normals feed the Boolean3 SOS tie-breaks.
				//
				// ManifoldImpl.ForVert has the same step-first shape and would be the
				// obvious call, but it is deliberately not used: it has no
				// `paired < 0` guard, so a soup impl (or any mesh with an unpaired
				// halfedge) walks off the end of the arena. This inlined copy is the
				// Rust's, guard included.
				int current = firstEdge;
				while (true)
				{
					int paired = halfedge[current].PairedHalfedge;
					if (paired < 0)
					{
						break;
					}

					current = Types.NextHalfedge(paired);
					Halfedge h = halfedge[current];
					int[] triVerts = new int[]
					{
						h.StartVert,
						h.EndVert,
						halfedge[Types.NextHalfedge(current)].EndVert,
					};

					// Avoid degenerate triangles. The Rust compares `x as usize <
					// vert_pos.len()`, which also rejects the -1 sentinel; an int `<`
					// would accept it and index out of bounds, hence the unsigned casts.
					if ((uint)triVerts[0] < (uint)vertPos.Count
						&& (uint)triVerts[1] < (uint)vertPos.Count
						&& (uint)triVerts[2] < (uint)vertPos.Count)
					{
						Vec3 currEdgeDir = vertPos[triVerts[1]] - vertPos[triVerts[0]];
						Vec3 prevEdgeDir = vertPos[triVerts[0]] - vertPos[triVerts[2]];
						double currLen = Math.Sqrt(LengthSquared(currEdgeDir));
						double prevLen = Math.Sqrt(LengthSquared(prevEdgeDir));

						if (currLen > 0.0 && prevLen > 0.0)
						{
							Vec3 currNorm = currEdgeDir / currLen;
							Vec3 prevNorm = prevEdgeDir / prevLen;
							if (double.IsFinite(currNorm.X) && double.IsFinite(prevNorm.X))
							{
								// Rust `f64::clamp`, transcribed: a pair of comparisons
								// that leaves NaN as NaN. LinalgFunctions.Clamp is
								// max-then-min and turns a NaN into `lo`, so it is not a
								// substitute (the same trap Types.Smoothstep documents).
								double d = Dot(prevNorm, currNorm);
								d = d < -1.0 ? -1.0 : (d > 1.0 ? 1.0 : d);

								// Negate because prevEdge points into vert and currEdge points away
								double phi = DeterministicMath.Acos(-d);
								if (double.IsFinite(phi) && current / 3 < faceNormal.Count)
								{
									normal = normal + (faceNormal[current / 3] * phi);
								}
							}
						}
					}

					if (current == firstEdge)
					{
						break;
					}
				}

				double len = Math.Sqrt(LengthSquared(normal));
				if (len > 0.0)
				{
					return normal / len;
				}

				return new Vec3(0.0, 0.0, 0.0);
			});

			mesh.VertNormal = new List<Vec3>(normals);
		}

		/// <summary>
		/// The Rust's function-local <c>struct TriPriority</c>: a triangle and the squared
		/// area the coplanar seeding order sorts on.
		/// </summary>
		private struct TriPriority
		{
			/// <summary>Twice the triangle's area, squared — <c>length2(cross(e0, e1))</c>.</summary>
			public double Area2;

			/// <summary>The triangle index.</summary>
			public int Tri;

			/// <summary>Creates a priority entry for one triangle.</summary>
			/// <param name="area2">The squared cross-product length.</param>
			/// <param name="tri">The triangle index.</param>
			public TriPriority(double area2, int tri)
			{
				this.Area2 = area2;
				this.Tri = tri;
			}
		}
	}
}
