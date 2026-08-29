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

// face_op_triangulate.rs — Face2Tri: triangulate polygonal faces into
// triangle halfedges.
//
// Ports the Face2Tri portion of src/face_op.cpp (WriteLocalTriangles,
// WriteGeneralTriangulation, Face2Tri). Extracted from face_op.rs, which
// re-exports face2tri/face2tri_ct so callers (boolean_result_assemble.rs)
// keep the crate::face_op::face2tri path. Relies on face_op.rs for the
// polygon-assembly helpers (assemble_halfedges, project_polygons,
// get_axis_aligned_projection) and on polygon.rs for the general
// triangulator.
//
// ── C# port notes ────────────────────────────────────────────────────────────
// That re-export is why this file continues `public static partial class
// FaceOp` (declared in FaceOp.cs) rather than opening a class of its own: the
// call path stays `FaceOp.Face2Tri`, exactly as the Rust's stays
// `crate::face_op::face2tri`.
//
// The two invariants this file exists to preserve, both of which the Rust's
// header states and both of which a "tidier" implementation breaks:
//   * `allowConvex = false` is the caller's, and Face2Tri never overrides it.
//     constructors.rs's extrude cap passes true; boolean_result_assemble.rs
//     passes false, because a boolean result's faces are not trustworthy convex.
//   * Pairing is FACE-LOCAL during triangulation, and boundary edges are paired
//     across faces through the ORIGINAL assembly's paired_halfedge — never
//     re-derived from a global (start, end) map. Two exactly-coplanar faces can
//     contribute the same vertex pair twice, and a global map cannot tell those
//     duplicates apart; it mispairs them and the result is non-manifold.
//
// `Halfedge[]` and not `List<Halfedge>` for the output arena: the writes below
// are field writes (`output[out].StartVert = ...`), which through a `List<T>`
// indexer would compile and update a temporary. Array elements are real
// storage. The finished array becomes `mesh.Halfedge` in one copy at the end.

using System.Diagnostics;

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <content>
	/// Face2Tri and its two triangle writers — the C# form of face_op.rs's
	/// <c>pub use face_op_triangulate::{face2tri, face2tri_ct}</c>.
	/// </content>
	public static partial class FaceOp
	{
		/// <summary>
		/// Triangulates the faces represented by <paramref name="faceEdge"/> (polygon
		/// boundaries in <c>mesh.Halfedge</c>) and <paramref name="halfedgeRef"/>
		/// (per-halfedge TriRef). On entry <c>mesh.Halfedge</c> holds general polygonal faces
		/// with valid cross-face pairing (from boolean result assembly); on return it holds
		/// proper triangles and <c>mesh.MeshRelation.TriRef</c> is populated.
		/// </summary>
		/// <remarks>
		/// Mirrors <c>Manifold::Impl::Face2Tri</c> in <c>src/face_op.cpp</c> (v3.5.0):
		/// pairing is face-local during triangulation, then boundary edges are paired across
		/// faces via the *original* <c>paired_halfedge</c> relationships — never re-derived
		/// from vertex pairs, which would mispair duplicate edges on degenerate
		/// (exactly-coplanar) faces and produce a non-manifold result.
		/// </remarks>
		/// <param name="mesh">The mesh whose polygonal faces are replaced by triangles.</param>
		/// <param name="faceEdge">Exclusive-scan of per-face halfedge counts; one longer than the face count.</param>
		/// <param name="halfedgeRef">Provenance per input halfedge; the first of each face is copied to its triangles.</param>
		/// <param name="allowConvex">Permit the general triangulator's convex fast path.</param>
		public static void Face2Tri(
			ManifoldImpl mesh,
			IReadOnlyList<int> faceEdge,
			IReadOnlyList<TriRef> halfedgeRef,
			bool allowConvex)
		{
			Face2TriCt(mesh, faceEdge, halfedgeRef, allowConvex, null);
		}

		/// <summary>
		/// <see cref="Face2Tri"/> with cooperative cancellation, returning false if
		/// <paramref name="token"/> fired before the triangulation completed. On false,
		/// <paramref name="mesh"/> is left in a half-built state and must be discarded — the
		/// same contract as C++ <c>Face2Tri(..., ctx)</c>, whose cancel arms <c>return</c>
		/// mid-way and rely on the caller's <c>MakeEmpty(Cancelled)</c>
		/// (face_op.cpp:190-366).
		/// </summary>
		/// <param name="mesh">The mesh whose polygonal faces are replaced by triangles.</param>
		/// <param name="faceEdge">Exclusive-scan of per-face halfedge counts; one longer than the face count.</param>
		/// <param name="halfedgeRef">Provenance per input halfedge; the first of each face is copied to its triangles.</param>
		/// <param name="allowConvex">Permit the general triangulator's convex fast path.</param>
		/// <param name="token">The cancellation token, or null for an uncancellable run.</param>
		/// <returns>True when the triangulation completed; false when cancelled.</returns>
		public static bool Face2TriCt(
			ManifoldImpl mesh,
			IReadOnlyList<int> faceEdge,
			IReadOnlyList<TriRef> halfedgeRef,
			bool allowConvex,
			CancelToken? token)
		{
			// C++ gates the whole function on entry (face_op.cpp:192).
			if (Cancel.IsCancelled(token))
			{
				return false;
			}

			// C++ passes faceHalfedge as a separate view and writes halfedge_ fresh;
			// the Rust assembly built the polygonal faces in mesh.halfedge, so take it.
			// `std::mem::take` leaves an empty vector behind, which is what the two lines
			// below do.
			List<Halfedge> faceHalfedge = mesh.Halfedge;
			mesh.Halfedge = new List<Halfedge>();

			int numFaces = faceEdge.Count - 1;
			int[] contour2tri = new int[faceHalfedge.Count];
			Array.Fill(contour2tri, -1);

			// First pass: count triangles per face; run the general triangulator for
			// faces with more than four edges. Each face triangulates independently
			// (C++ spawns a TBB task per general face), so the parallel path yields
			// identical per-face results, merged in face order.
			List<Vec3> faceNormalRef = mesh.FaceNormal;
			List<Vec3> vertPosRef = mesh.VertPos;
			double epsilon = mesh.Epsilon;
			HalfedgeTriangulation?[]? general = Par.MaybeParMapCt<HalfedgeTriangulation?>(
				numFaces,
				512,
				token,
				face =>
				{
					int numEdgeLocal = faceEdge[face + 1] - faceEdge[face];
					if (numEdgeLocal <= 4)
					{
						return null;
					}

					int firstEdgeLocal = faceEdge[face];
					int lastEdgeLocal = faceEdge[face + 1];
					Proj2x3 projection = GetAxisAlignedProjection(faceNormalRef[face]);

					// Rust `&face_halfedge[first..last]` is a borrowed slice; GetRange copies
					// the face's own halfedges (numEdge of them, the same order as the work
					// being done on them), which is the cheapest way to hand an
					// IReadOnlyList to AssembleHalfedges without changing its Rust signature.
					List<Halfedge> faceSlice =
						faceHalfedge.GetRange(firstEdgeLocal, lastEdgeLocal - firstEdgeLocal);
					List<List<int>> polysLoops = AssembleHalfedges(faceSlice, firstEdgeLocal);

					// Note the FULL arena here, not the slice: the loop indices
					// AssembleHalfedges returned are already offset by firstEdge.
					PolygonsIdx polys = ProjectPolygons(polysLoops, faceHalfedge, vertPosRef, projection);
					return Polygon.TriangulateIdxHalfedges(polys, epsilon, allowConvex);
				});
			if (general is null)
			{
				return false;
			}

			int[] triOffset = new int[faceEdge.Count];
			Dictionary<int, HalfedgeTriangulation> results = new Dictionary<int, HalfedgeTriangulation>();
			for (int face = 0; face < general.Length; face++)
			{
				HalfedgeTriangulation? triangulation = general[face];
				int numEdgeLocal = faceEdge[face + 1] - faceEdge[face];
				if (numEdgeLocal == 0)
				{
					continue;
				}

				Debug.Assert(numEdgeLocal >= 3, "face has less than three edges");
				triOffset[face] = numEdgeLocal - 2;
				if (triangulation is not null)
				{
					triOffset[face] = triangulation.NumTri();

					// Probe-only map: nothing iterates it, only `results[face]` below, so a
					// plain Dictionary is safe (the `rustc-hash` → `Dictionary` rule in
					// docs/PORTING_PLAN.md).
					results.Add(face, triangulation);
				}
			}

			// Exclusive scan of triangle counts → per-face output offsets.
			int acc = 0;
			for (int i = 0; i < triOffset.Length; i++)
			{
				int count = triOffset[i];
				triOffset[i] = acc;
				acc += count;
			}

			int numTri = acc;

			Halfedge[] newHalfedge = new Halfedge[3 * numTri];
			Array.Fill(newHalfedge, new Halfedge(-1, -1, -1, -1));
			Vec3[] triNormal = new Vec3[numTri];
			Array.Fill(triNormal, new Vec3(0.0, 0.0, 0.0));
			TriRef[] triRef = new TriRef[numTri];

			for (int face = 0; face < numFaces; face++)
			{
				int firstEdge = faceEdge[face];
				int lastEdge = faceEdge[face + 1];
				int numEdge = lastEdge - firstEdge;
				if (numEdge == 0)
				{
					continue;
				}

				Vec3 normal = mesh.FaceNormal[face];
				int firstTri = triOffset[face];
				int faceNumTri;

				if (numEdge == 3)
				{
					// Single triangle — sort edges into correct winding order.
					int[] triEdge = new int[] { firstEdge, firstEdge + 1, firstEdge + 2 };
					int[] tri = new int[]
					{
						faceHalfedge[firstEdge].StartVert,
						faceHalfedge[firstEdge + 1].StartVert,
						faceHalfedge[firstEdge + 2].StartVert,
					};
					int[] ends = new int[]
					{
						faceHalfedge[firstEdge].EndVert,
						faceHalfedge[firstEdge + 1].EndVert,
						faceHalfedge[firstEdge + 2].EndVert,
					};
					if (ends[0] == tri[2])
					{
						(triEdge[1], triEdge[2]) = (triEdge[2], triEdge[1]);
						(tri[1], tri[2]) = (tri[2], tri[1]);
						(ends[1], ends[2]) = (ends[2], ends[1]);
					}

					Debug.Assert(
						ends[0] == tri[1] && ends[1] == tri[2] && ends[2] == tri[0],
						"these 3 edges do not form a triangle!");
					WriteLocalTriangles(
						newHalfedge,
						contour2tri,
						faceHalfedge,
						firstTri,
						new int[][] { triEdge });
					faceNumTri = 1;
				}
				else if (numEdge == 4)
				{
					// Quad — split into two triangles along the better diagonal.
					Proj2x3 projection = GetAxisAlignedProjection(normal);
					bool TriCcw(int[] t)
					{
						return Polygon.Ccw(
							projection.Apply(mesh.VertPos[faceHalfedge[t[0]].StartVert]),
							projection.Apply(mesh.VertPos[faceHalfedge[t[1]].StartVert]),
							projection.Apply(mesh.VertPos[faceHalfedge[t[2]].StartVert]),
							mesh.Epsilon) >= 0;
					}

					List<List<int>> quadLoops = AssembleHalfedges(
						faceHalfedge.GetRange(firstEdge, lastEdge - firstEdge),
						firstEdge);
					List<int> quad = quadLoops[0]; // Should be exactly one loop

					int[][] tris0 = new int[][]
					{
						new int[] { quad[0], quad[1], quad[2] },
						new int[] { quad[0], quad[2], quad[3] },
					};
					int[][] tris1 = new int[][]
					{
						new int[] { quad[1], quad[2], quad[3] },
						new int[] { quad[0], quad[1], quad[3] },
					};

					int choice;
					if (!(TriCcw(tris0[0]) && TriCcw(tris0[1])))
					{
						choice = 1;
					}
					else if (TriCcw(tris1[0]) && TriCcw(tris1[1]))
					{
						Vec3 diag0 = mesh.VertPos[faceHalfedge[quad[0]].StartVert]
							- mesh.VertPos[faceHalfedge[quad[2]].StartVert];
						Vec3 diag1 = mesh.VertPos[faceHalfedge[quad[1]].StartVert]
							- mesh.VertPos[faceHalfedge[quad[3]].StartVert];
						choice = LengthSquared(diag0) > LengthSquared(diag1) ? 1 : 0;
					}
					else
					{
						choice = 0;
					}

					int[][] chosen = choice == 0 ? tris0 : tris1;
					WriteLocalTriangles(newHalfedge, contour2tri, faceHalfedge, firstTri, chosen);
					faceNumTri = 2;
				}
				else
				{
					// General triangulation. Rust `.expect("general face missing
					// triangulation result")` — the indexer's KeyNotFoundException is the
					// faithful behaviour for a state the first pass makes impossible.
					HalfedgeTriangulation triangulation = results[face];
					WriteGeneralTriangulation(
						newHalfedge,
						contour2tri,
						faceHalfedge,
						firstTri,
						triangulation);
					faceNumTri = triangulation.NumTri();
				}

				// WriteTriRefs
				TriRef refTri = halfedgeRef[firstEdge];
				for (int t = 0; t < faceNumTri; t++)
				{
					triNormal[firstTri + t] = normal;
					triRef[firstTri + t] = refTri;
				}
			}

			// Cross-face pairing: connect each face-boundary output halfedge to its
			// counterpart via the original assembly pairing.
			for (int edge = 0; edge < faceHalfedge.Count; edge++)
			{
				int triEdge = contour2tri[edge];
				if (triEdge < 0)
				{
					continue;
				}

				int pair = faceHalfedge[edge].PairedHalfedge;
				if (pair < 0)
				{
					continue;
				}

				int pairTri = contour2tri[pair];
				Debug.Assert(pairTri >= 0, "boundary edge did not triangulate with its pair");
				newHalfedge[triEdge].PairedHalfedge = pairTri;
			}

			mesh.Halfedge = new List<Halfedge>(newHalfedge);
			mesh.FaceNormal = new List<Vec3>(triNormal);
			mesh.MeshRelation.TriRef.Clear();
			mesh.MeshRelation.TriRef.AddRange(triRef);
			return true;
		}

		/// <summary>
		/// Port of <c>WriteLocalTriangles</c> (face_op.cpp): write 1–2 triangles for a
		/// tri/quad face. <paramref name="triangles"/> entries are face-halfedge indices.
		/// Interior edges (the quad diagonal) pair against each other by matching those
		/// indices; boundary edges record their output halfedge in
		/// <c>contour2tri[originalFaceHalfedgeIndex]</c> for the cross-face pairing pass.
		/// </summary>
		/// <param name="output">The output halfedge arena.</param>
		/// <param name="contour2tri">Per input halfedge, the output halfedge lying on it; -1 when none.</param>
		/// <param name="faceHalfedge">The input (polygonal-face) halfedge arena.</param>
		/// <param name="firstTri">Index of this face's first output triangle.</param>
		/// <param name="triangles">One or two triangles, as face-halfedge index triples.</param>
		private static void WriteLocalTriangles(
			Halfedge[] output,
			int[] contour2tri,
			IReadOnlyList<Halfedge> faceHalfedge,
			int firstTri,
			int[][] triangles)
		{
			Debug.Assert(triangles.Length <= 2, "local face path only handles tris/quads");
			int firstOut = 3 * firstTri;

			// (start, end, out) — start/end are face-halfedge indices, out is the
			// output halfedge index.
			int[,] localEdges = new int[6, 3];
			int numEdge = 0;
			foreach (int[] tri in triangles)
			{
				for (int i = 0; i < 3; i++)
				{
					int outIdx = firstOut + numEdge;
					int start = tri[i];
					int end = tri[(i + 1) % 3];
					localEdges[numEdge, 0] = start;
					localEdges[numEdge, 1] = end;
					localEdges[numEdge, 2] = outIdx;
					output[outIdx].StartVert = faceHalfedge[start].StartVert;

					// C++ Halfedges derives end verts from triangle adjacency; the
					// Rust Halfedge stores them, so set explicitly.
					output[outIdx].EndVert = faceHalfedge[end].StartVert;
					output[outIdx].PropVert = faceHalfedge[start].PropVert;
					output[outIdx].PairedHalfedge = -1;
					numEdge++;
				}
			}

			for (int i = 0; i < numEdge; i++)
			{
				int edgeStart = localEdges[i, 0];
				int edgeEnd = localEdges[i, 1];
				int edgeOut = localEdges[i, 2];
				int pair = -1;
				for (int j = 0; j < numEdge; j++)
				{
					if (localEdges[j, 0] == edgeEnd && localEdges[j, 1] == edgeStart)
					{
						pair = localEdges[j, 2];
						break;
					}
				}

				if (pair >= 0)
				{
					output[edgeOut].PairedHalfedge = pair;
				}
				else
				{
					contour2tri[edgeStart] = edgeOut;
				}
			}
		}

		/// <summary>
		/// Port of <c>WriteGeneralTriangulation</c> (face_op.cpp): write the triangles of a
		/// <see cref="HalfedgeTriangulation"/>. Triangle-halfedge vertex fields hold
		/// face-halfedge indices; interior pairs translate directly, while contour pairs
		/// record the triangle halfedge lying on each original edge in
		/// <paramref name="contour2tri"/>.
		/// </summary>
		/// <param name="output">The output halfedge arena.</param>
		/// <param name="contour2tri">Per input halfedge, the output halfedge lying on it; -1 when none.</param>
		/// <param name="faceHalfedge">The input (polygonal-face) halfedge arena.</param>
		/// <param name="firstTri">Index of this face's first output triangle.</param>
		/// <param name="triangulation">The face's triangulation, in face-halfedge indices.</param>
		private static void WriteGeneralTriangulation(
			Halfedge[] output,
			int[] contour2tri,
			IReadOnlyList<Halfedge> faceHalfedge,
			int firstTri,
			HalfedgeTriangulation triangulation)
		{
			int firstOut = 3 * firstTri;
			int contourEnd = triangulation.ContourEnd;
			int numTriHalfedge = 3 * triangulation.NumTri();

			for (int local = 0; local < numTriHalfedge; local++)
			{
				int outIdx = firstOut + local;
				Halfedge edge = triangulation.Halfedges[contourEnd + local];
				output[outIdx].StartVert = faceHalfedge[edge.StartVert].StartVert;
				output[outIdx].EndVert = faceHalfedge[edge.EndVert].StartVert;
				output[outIdx].PropVert = faceHalfedge[edge.StartVert].PropVert;
				if (edge.PairedHalfedge >= contourEnd)
				{
					output[outIdx].PairedHalfedge = firstOut + edge.PairedHalfedge - contourEnd;
				}
				else
				{
					output[outIdx].PairedHalfedge = -1;
				}
			}

			for (int contour = 0; contour < contourEnd; contour++)
			{
				Halfedge edge = triangulation.Halfedges[contour];
				if (edge.PairedHalfedge < 0)
				{
					continue;
				}

				Debug.Assert(
					edge.PairedHalfedge >= contourEnd,
					"contour paired to another contour");

				// Contour halfedges are the input edges reversed, so end_vert holds
				// the original face-halfedge index of the edge's start.
				int boundary = edge.EndVert;
				Debug.Assert(
					boundary >= 0 && boundary < contour2tri.Length,
					"contour edge index out of bounds");
				contour2tri[boundary] = firstOut + edge.PairedHalfedge - contourEnd;
			}
		}
	}
}
