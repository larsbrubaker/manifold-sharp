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

// Boolean result assembly — extracted from boolean_result.rs
// Contains update_reference, create_properties, and boolean_result entry point
//
// ── C# port notes ────────────────────────────────────────────────────────────
// boolean_result.rs re-exports these two entry points, so Rust callers say
// `crate::boolean_result::boolean_result_with_token`. C# has no re-export, so
// the call in Boolean3.Functions.cs names this class directly.
//
// Rust's `drop(x)` after a stage is a memory release, not a semantic step. C#
// locals are retired by the JIT's GC info at their last use, so each of those
// sites is kept as a comment recording which C++ `.clear()` it stands for
// rather than as code.
//
// ── File split ───────────────────────────────────────────────────────────────
//   BooleanResultAssemble.cs             this file — UpdateReference and the
//                                        BooleanResult entry point
//   BooleanResultAssemble.Properties.cs  CreateProperties

using ManifoldSharp.Linalg;

// The Rust opens with `use super::{abs_sum, add_new_edge_verts, ...}` — the parent
// module's free functions, by bare name. `using static` is that import. It is also
// load-bearing here and not just cosmetic: this class has a *method* named
// BooleanResult, which shadows the class of the same name for qualified lookups
// inside it (CS0119).
using static ManifoldSharp.BooleanResult;
using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// The output-mesh assembly stage: the Rust port of <c>Boolean3::Result()</c>.
	/// </summary>
	public static partial class BooleanResultAssemble
	{
		// -------------------------------------------------------------------
		// UpdateReference -- map tri refs from input meshes to output
		// -------------------------------------------------------------------

		internal static void UpdateReference(
			ManifoldImpl outR,
			ManifoldImpl inP,
			ManifoldImpl inQ,
			bool invertQ)
		{
			int offsetQ = (int)ManifoldImpl.ReserveIds((uint)inQ.MeshRelation.MeshIdTransform.Count);

			for (int i = 0; i < outR.MeshRelation.TriRef.Count; i++)
			{
				TriRef triRef = outR.MeshRelation.TriRef[i];
				int tri = triRef.FaceId;
				bool pq = triRef.MeshId == 0;
				if (pq)
				{
					if ((uint)tri < (uint)inP.MeshRelation.TriRef.Count)
					{
						outR.MeshRelation.TriRef[i] = inP.MeshRelation.TriRef[tri];
					}
				}
				else
				{
					if ((uint)tri < (uint)inQ.MeshRelation.TriRef.Count)
					{
						TriRef updated = inQ.MeshRelation.TriRef[tri];
						updated.MeshId += offsetQ;
						outR.MeshRelation.TriRef[i] = updated;
					}
				}
			}

			foreach (KeyValuePair<int, Relation> entry in inP.MeshRelation.MeshIdTransform)
			{
				outR.MeshRelation.MeshIdTransform[entry.Key] = entry.Value;
			}

			foreach (KeyValuePair<int, Relation> entry in inQ.MeshRelation.MeshIdTransform)
			{
				Relation rel = entry.Value;
				rel.BackSide ^= invertQ;
				outR.MeshRelation.MeshIdTransform[entry.Key + offsetQ] = rel;
			}
		}

		// -------------------------------------------------------------------
		// boolean_result -- the main entry point
		// -------------------------------------------------------------------

		/// <summary>
		/// Assemble the output mesh from Boolean3 intersection data.
		/// </summary>
		/// <remarks>
		/// This is the Rust port of <c>Boolean3::Result()</c> from <c>boolean_result.cpp</c>.
		/// </remarks>
		/// <param name="inP">The first operand, P.</param>
		/// <param name="inQ">The second operand, Q.</param>
		/// <param name="op">The operation being assembled for.</param>
		/// <param name="bool3">The intersection data from <see cref="Boolean3"/>.</param>
		/// <returns>The assembled result impl.</returns>
		public static ManifoldImpl BooleanResult(
			ManifoldImpl inP,
			ManifoldImpl inQ,
			OpType op,
			Boolean3 bool3)
		{
			return BooleanResultWithToken(inP, inQ, op, bool3, null);
		}

		/// <summary>
		/// <see cref="BooleanResult"/> with cooperative cancellation, returning an empty mesh
		/// with <see cref="Error.Cancelled"/> if <paramref name="token"/> fires.
		/// </summary>
		/// <remarks>
		/// Check placement mirrors the <c>phase()</c> sites of C++ <c>Boolean3::Result</c>
		/// (boolean_result.cpp:758-963): one at every boundary between the assembly
		/// stages, each of which does <c>MakeEmpty(Cancelled); return</c>. Partial output is
		/// intentionally discarded rather than published.
		/// </remarks>
		/// <param name="inP">The first operand, P.</param>
		/// <param name="inQ">The second operand, Q.</param>
		/// <param name="op">The operation being assembled for.</param>
		/// <param name="bool3">The intersection data from <see cref="Boolean3"/>.</param>
		/// <param name="token">The cancellation token, or null for an uncancellable run.</param>
		/// <returns>The assembled result impl.</returns>
		public static ManifoldImpl BooleanResultWithToken(
			ManifoldImpl inP,
			ManifoldImpl inQ,
			OpType op,
			Boolean3 bool3,
			CancelToken? token)
		{
			ArgumentNullException.ThrowIfNull(inP);
			ArgumentNullException.ThrowIfNull(inQ);
			ArgumentNullException.ThrowIfNull(bool3);

			System.Diagnostics.Debug.Assert(
				bool3.ExpandP == (op == OpType.Add),
				"Result op type not compatible with constructor op type.");

			int c1 = op == OpType.Intersect ? 0 : 1;
			int c2 = op == OpType.Add ? 1 : 0;
			int c3 = op == OpType.Intersect ? 1 : -1;

			// Early returns for empty inputs (matches C++ boolean_result.cpp lines 680-690)
			if (inP.IsEmpty())
			{
				if (!inQ.IsEmpty() && op == OpType.Add)
				{
					return inQ.Clone();
				}

				return new ManifoldImpl();
			}
			else if (inQ.IsEmpty())
			{
				if (op == OpType.Intersect)
				{
					return new ManifoldImpl();
				}

				return inP.Clone();
			}

			// Check for valid (overflow) result
			if (!bool3.Valid)
			{
				return new ManifoldImpl();
			}

			bool invertQ = op == OpType.Subtract;

			// Phase 1 (C++ boolean_result.cpp:776): the trivial early returns above run
			// first, exactly as in C++, where the IsEmpty fast-paths precede the first
			// phase() site.
			if (Cancel.IsCancelled(token))
			{
				return Boolean3Functions.CancelledImpl();
			}

			// Timing boundaries mirror the C++ MANIFOLD_TIMING stages (Assembly /
			// Triangulation / Simplification / Sorting) for side-by-side comparison.
			long? tAssembly = Timing.Start();

			// Convert winding numbers to inclusion values
			List<int> i12 = bool3.Xv12.X12.Select(v => c3 * v).ToList();
			List<int> i21 = bool3.Xv21.X12.Select(v => c3 * v).ToList();
			List<int> i03 = bool3.W03.Select(v => c1 + (c3 * v)).ToList();
			List<int> i30 = bool3.W30.Select(v => c2 + (c3 * v)).ToList();

			// Vertex remapping via exclusive scan with abs_sum
			List<int> vp2r = ExclusiveScanAbs(i03, 0);
			int numVertR = i03.Count > 0
				? AbsSum(vp2r.Count > 0 ? vp2r[vp2r.Count - 1] : 0, i03[i03.Count - 1])
				: 0;
			int nPv = numVertR;

			List<int> vq2r = ExclusiveScanAbs(i30, numVertR);
			numVertR = i30.Count > 0
				? AbsSum(vq2r.Count > 0 ? vq2r[vq2r.Count - 1] : numVertR, i30[i30.Count - 1])
				: numVertR;
			int nQv = numVertR - nPv;

			List<int> v12r = bool3.Xv12.V12.Count > 0
				? ExclusiveScanAbs(i12, numVertR)
				: new List<int>();
			numVertR = i12.Count > 0
				? AbsSum(v12r.Count > 0 ? v12r[v12r.Count - 1] : numVertR, i12[i12.Count - 1])
				: numVertR;
			int n12 = numVertR - nPv - nQv;

			List<int> v21r = bool3.Xv21.V12.Count > 0
				? ExclusiveScanAbs(i21, numVertR)
				: new List<int>();
			numVertR = i21.Count > 0
				? AbsSum(v21r.Count > 0 ? v21r[v21r.Count - 1] : numVertR, i21[i21.Count - 1])
				: numVertR;

			// The Rust binds this as `_n21`, unused; kept for the same reason it is there.
			_ = numVertR - nPv - nQv - n12;

			// Create the output Manifold
			ManifoldImpl outR = new ManifoldImpl();
			if (numVertR == 0)
			{
				return outR;
			}

			outR.Epsilon = MaxF64(inP.Epsilon, inQ.Epsilon);
			outR.Tolerance = MaxF64(inP.Tolerance, inQ.Tolerance);

			// Allocate and populate output vertices
			outR.VertPos.Resize(numVertR, Vec3.Splat(0.0));

			// DuplicateVerts: retained vertices from P
			for (int vert = 0; vert < inP.NumVert(); vert++)
			{
				int n = Math.Abs(i03[vert]);
				for (int i = 0; i < n; i++)
				{
					outR.VertPos[vp2r[vert] + i] = inP.VertPos[vert];
				}
			}

			// Retained vertices from Q
			for (int vert = 0; vert < inQ.NumVert(); vert++)
			{
				int n = Math.Abs(i30[vert]);
				for (int i = 0; i < n; i++)
				{
					outR.VertPos[vq2r[vert] + i] = inQ.VertPos[vert];
				}
			}

			// New vertices from P edges -> Q faces
			for (int vert = 0; vert < i12.Count; vert++)
			{
				int n = Math.Abs(i12[vert]);
				for (int i = 0; i < n; i++)
				{
					outR.VertPos[v12r[vert] + i] = bool3.Xv12.V12[vert];
				}
			}

			// New vertices from Q edges -> P faces
			for (int vert = 0; vert < i21.Count; vert++)
			{
				int n = Math.Abs(i21[vert]);
				for (int i = 0; i < n; i++)
				{
					outR.VertPos[v21r[vert] + i] = bool3.Xv21.V12[vert];
				}
			}

			// Phase 2 (C++ boolean_result.cpp:847): after DuplicateVerts.
			if (Cancel.IsCancelled(token))
			{
				return Boolean3Functions.CancelledImpl();
			}

			// Build edge maps
			SortedDictionary<int, List<EdgePos>> edgesP = new SortedDictionary<int, List<EdgePos>>();
			SortedDictionary<int, List<EdgePos>> edgesQ = new SortedDictionary<int, List<EdgePos>>();
			SortedDictionary<(int FaceP, int FaceQ), List<EdgePos>> edgesNew =
				new SortedDictionary<(int, int), List<EdgePos>>();

			AddNewEdgeVerts(
				edgesP,
				edgesNew,
				bool3.Xv12.P1q2,
				i12,
				v12r,
				inP.Halfedge,
				true,
				0);
			AddNewEdgeVerts(
				edgesQ,
				edgesNew,
				bool3.Xv21.P1q2,
				i21,
				v21r,
				inQ.Halfedge,
				false,
				bool3.Xv12.P1q2.Count);

			// C++ clears v12R/v21R here (after AddNewEdgeVerts); nothing below reads them.

			// Phase 3 (C++ boolean_result.cpp:869): after AddNewEdgeVerts.
			if (Cancel.IsCancelled(token))
			{
				return Boolean3Functions.CancelledImpl();
			}

			// Size output
			(List<int> faceEdge, List<int> facePq2r) = SizeOutput(
				outR,
				inP,
				inQ,
				i03,
				i30,
				i12,
				i21,
				bool3.Xv12.P1q2,
				bool3.Xv21.P1q2,
				invertQ);

			// C++ clears i12/i21 after SizeOutput.

			// Phase 4 (C++ boolean_result.cpp:880): after SizeOutput.
			if (Cancel.IsCancelled(token))
			{
				return Boolean3Functions.CancelledImpl();
			}

			// Assemble edges
			List<int> facePtrR = new List<int>(faceEdge);
			bool[] wholeHalfedgeP = new bool[inP.Halfedge.Count];
			Array.Fill(wholeHalfedgeP, true);
			bool[] wholeHalfedgeQ = new bool[inQ.Halfedge.Count];
			Array.Fill(wholeHalfedgeQ, true);
			List<TriRef> halfedgeRef = new List<TriRef>(2 * outR.NumEdge());
			halfedgeRef.Resize(2 * outR.NumEdge(), new TriRef(0, -1, -1, -1));

			// The Rust passes `&face_pq2r[..in_p.num_tri()]` and `&face_pq2r[in_p.num_tri()..]`.
			// A C# span cannot be captured by the lambda AppendPartialEdges hands to PairUp,
			// so the two halves are materialized instead; both are read-only here.
			List<int> facePq2rP = facePq2r.GetRange(0, inP.NumTri());
			List<int> facePq2rQ = facePq2r.GetRange(inP.NumTri(), facePq2r.Count - inP.NumTri());

			AppendPartialEdges(
				outR,
				wholeHalfedgeP,
				facePtrR,
				edgesP,
				halfedgeRef,
				inP,
				i03,
				vp2r,
				facePq2rP,
				true);
			AppendPartialEdges(
				outR,
				wholeHalfedgeQ,
				facePtrR,
				edgesQ,
				halfedgeRef,
				inQ,
				i30,
				vq2r,
				facePq2rQ,
				false);

			// C++ clears edgesP/edgesQ after AppendPartialEdges.

			// Phase 5 (C++ boolean_result.cpp:905): after AppendPartialEdges.
			if (Cancel.IsCancelled(token))
			{
				return Boolean3Functions.CancelledImpl();
			}

			AppendNewEdges(
				outR,
				facePtrR,
				edgesNew,
				halfedgeRef,
				facePq2r,
				inP.NumTri());

			// C++ clears edgesNew after AppendNewEdges.

			// Phase 6 (C++ boolean_result.cpp:912): after AppendNewEdges.
			if (Cancel.IsCancelled(token))
			{
				return Boolean3Functions.CancelledImpl();
			}

			AppendWholeEdges(
				outR,
				facePtrR,
				halfedgeRef,
				inP,
				wholeHalfedgeP,
				i03,
				vp2r,
				facePq2rP,
				true);
			AppendWholeEdges(
				outR,
				facePtrR,
				halfedgeRef,
				inQ,
				wholeHalfedgeQ,
				i30,
				vq2r,
				facePq2rQ,
				false);

			// C++ clears wholeHalfedgeP/Q, vP2R/vQ2R and friends after
			// AppendWholeEdges; nothing below needs these.

			Timing.Print("Assembly", tAssembly);

			// Phase 7 (C++ boolean_result.cpp:922): after AppendWholeEdges.
			if (Cancel.IsCancelled(token))
			{
				return Boolean3Functions.CancelledImpl();
			}

			// Triangulate polygonal faces (allowConvex=false per C++ boolean_result.cpp)
			long? t = Timing.Start();
			if (!FaceOp.Face2TriCt(outR, faceEdge, halfedgeRef, false, token))
			{
				return Boolean3Functions.CancelledImpl();
			}

			FaceOp.ReorderHalfedges(outR);

			// C++ clears faceEdge after Face2Tri; halfedgeRef is likewise done.
			Timing.Print("Triangulation", t);

			// Phase 8 (C++ boolean_result.cpp:941): after Face2Tri + ReorderHalfedges.
			if (Cancel.IsCancelled(token))
			{
				return Boolean3Functions.CancelledImpl();
			}

			t = Timing.Start();

			// Create properties via barycentric interpolation
			CreateProperties(outR, inP, inQ, invertQ);

			// Phase 9 (C++ boolean_result.cpp:948): after CreateProperties.
			if (Cancel.IsCancelled(token))
			{
				return Boolean3Functions.CancelledImpl();
			}

			// Update references
			UpdateReference(outR, inP, inQ, invertQ);

			// Phase 10 (C++ boolean_result.cpp:951): after UpdateReference.
			if (Cancel.IsCancelled(token))
			{
				return Boolean3Functions.CancelledImpl();
			}

			// Simplify topology
			EdgeOp.SimplifyTopology(outR, nPv + nQv);
			outR.RemoveUnreferencedVerts();
			Timing.Print("Simplification", t);

			// Finalize
			t = Timing.Start();
			outR.CalculateBBox();
			outR.SortGeometry();
			outR.IncrementMeshIds();
			Timing.Print("Sorting", t);

			// Phase 11 (C++ boolean_result.cpp:963): after SortGeometry. Without this
			// the whole trailing block above would be a hole in the contract — a cancel
			// landing in SimplifyTopology or SortGeometry would return a *complete*
			// mesh with status NoError, which is worse than a latency cost: the caller
			// would be told the operation it cancelled had succeeded. It matters most
			// on the shortest path through the kernel, a two-operand batch_boolean,
			// where `simple_boolean` runs once with no enclosing per-round re-check to
			// catch the cancel afterwards.
			//
			// C++ additionally threads ctx *into* SortGeometry; we only bracket it, so
			// the residual cost here is latency (one run of simplify + sort), not a
			// wrong status.
			if (Cancel.IsCancelled(token))
			{
				return Boolean3Functions.CancelledImpl();
			}

			return outR;
		}
	}
}
