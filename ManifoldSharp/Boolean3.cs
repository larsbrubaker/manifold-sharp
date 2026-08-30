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

// Phase 11: Boolean Operations (Core)
//
// C++ sources: src/boolean3.cpp (531 lines), src/boolean_result.cpp (889 lines)
//
// This module implements the edge-face intersection detection algorithm from
// boolean3.cpp. The result is consumed by boolean_result.rs to assemble the
// output mesh.
//
// Key notation (from the C++ source):
// - P and Q are the two input manifolds, R is the output
// - Dimensions: vert=0, edge=1, face=2, solid=3
// - X = winding-number quantity, S = "shadow" subset of X
// - p1q2 = edges of P intersecting faces of Q
// - x12 = winding contribution at each intersection
// - v12 = 3D position of each intersection vertex
//
// (The header above is boolean3.rs's, verbatim; its phase number is the Rust
// port's own, not docs/PORTING_PLAN.md's — this is that plan's Phase 5.)
//
// ── C# port notes ────────────────────────────────────────────────────────────
// The floating-point kernels (Shadow01, Kernel11/02/12) and the broadphase
// drivers (Intersect12, Winding03) live in Boolean3Kernels.cs and
// Boolean3Kernels.Broadphase.cs — the C# form of boolean3.rs's
// `#[path = "boolean3_kernels.rs"] mod boolean3_kernels`.
//
// `Boolean3` is a type in this module, so the module's free functions cannot
// take the bare module name; per CLAUDE.md's naming rule they land on
// `Boolean3Functions` (Boolean3.Functions.cs), alongside `LinalgFunctions` and
// `SvdFunctions`.
//
// ── File split ───────────────────────────────────────────────────────────────
//   Boolean3.cs            this file — Intersections and the Boolean3
//                          constructor pipeline
//   Boolean3.Functions.cs  ComposeMeshes, Boolean, the engine dispatch and
//                          RayCast

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	// -----------------------------------------------------------------------
	// Intersections — sparse intersection data between two meshes
	// -----------------------------------------------------------------------

	/// <summary>
	/// Stores the intersections of edges of one mesh with faces of the other.
	/// In forward mode: edges of P with faces of Q.
	/// In reverse mode: edges of Q with faces of P.
	/// </summary>
	public sealed class Intersections
	{
		/// <summary>
		/// Creates the empty intersection set — the Rust's <c>Intersections::default()</c>.
		/// </summary>
		public Intersections()
		{
			this.P1q2 = new List<IVec2>();
			this.X12 = new List<int>();
			this.V12 = new List<Vec3>();
		}

		/// <summary>
		/// Pairs [edge_idx, face_idx] — in forward mode [p1, q2], reverse [q1, p2].
		/// </summary>
		public List<IVec2> P1q2 { get; }

		/// <summary>Winding number contribution at each intersection.</summary>
		public List<int> X12 { get; }

		/// <summary>3D position of each intersection vertex.</summary>
		public List<Vec3> V12 { get; }
	}

	// -----------------------------------------------------------------------
	// Boolean3 — the core intersection computation
	// -----------------------------------------------------------------------

	/// <summary>
	/// Computes all edge-face intersections and winding numbers between two meshes.
	/// </summary>
	public sealed class Boolean3
	{
		private Boolean3(
			Intersections xv12,
			Intersections xv21,
			List<int> w03,
			List<int> w30,
			bool expandP,
			bool valid)
		{
			this.Xv12 = xv12;
			this.Xv21 = xv21;
			this.W03 = w03;
			this.W30 = w30;
			this.ExpandP = expandP;
			this.Valid = valid;
		}

		/// <summary>Edges of P intersecting faces of Q.</summary>
		public Intersections Xv12 { get; }

		/// <summary>Edges of Q intersecting faces of P.</summary>
		public Intersections Xv21 { get; }

		/// <summary>Winding number of each vertex of P relative to Q.</summary>
		public List<int> W03 { get; }

		/// <summary>Winding number of each vertex of Q relative to P.</summary>
		public List<int> W30 { get; }

		/// <summary>True when the op is <see cref="OpType.Add"/> (symbolic expansion of P).</summary>
		public bool ExpandP { get; }

		/// <summary>False when the intersection counts overflowed <c>int</c>.</summary>
		public bool Valid { get; }

		// -------------------------------------------------------------------
		// Boolean3 constructor
		// -------------------------------------------------------------------

		/// <summary>
		/// Compute all intersections between meshes inP and inQ for the given op.
		/// </summary>
		/// <param name="inP">The first operand, P.</param>
		/// <param name="inQ">The second operand, Q.</param>
		/// <param name="op">The boolean operation the result will be assembled for.</param>
		/// <returns>The intersection data.</returns>
		public static Boolean3 New(ManifoldImpl inP, ManifoldImpl inQ, OpType op)
		{
			Boolean3? b3 = NewWithToken(inP, inQ, op, null);
			if (b3 is not null)
			{
				return b3;
			}

			// Unreachable: `is_cancelled(None)` is always false, so none of the
			// cancellation arms below can be taken. The debug assert makes a
			// future refactor that breaks that reasoning fail loudly in tests,
			// while release stays total — degrading to an invalid
			// (empty-result) Boolean3 rather than panicking in production.
			System.Diagnostics.Debug.Assert(
				false,
				"Boolean3.NewWithToken returned null for a null token; "
				+ "only a cancelled token can produce null");
			return new Boolean3(
				new Intersections(),
				new Intersections(),
				new List<int>(),
				new List<int>(),
				op == OpType.Add,
				false);
		}

		/// <summary>
		/// <see cref="New"/> with cooperative cancellation. Null means
		/// <paramref name="token"/> was cancelled; no usable intersection data was produced.
		/// </summary>
		/// <remarks>
		/// The check placement mirrors C++ <c>Boolean3::Boolean3</c>
		/// (boolean3.cpp:497-560): one phase-boundary check before launching each
		/// of the four heavy stages, plus the intra-stage checks that
		/// <c>Intersect12</c> and <c>Winding03</c> carry.
		/// </remarks>
		/// <param name="inP">The first operand, P.</param>
		/// <param name="inQ">The second operand, Q.</param>
		/// <param name="op">The boolean operation the result will be assembled for.</param>
		/// <param name="token">The cancellation token, or null for an uncancellable run.</param>
		/// <returns>The intersection data, or null if the token was cancelled.</returns>
		public static Boolean3? NewWithToken(
			ManifoldImpl inP,
			ManifoldImpl inQ,
			OpType op,
			CancelToken? token)
		{
			ArgumentNullException.ThrowIfNull(inP);
			ArgumentNullException.ThrowIfNull(inQ);

			bool expandP = op == OpType.Add;

			if (inP.IsEmpty() || inQ.IsEmpty() || !inP.Bbox.DoesOverlapBox(inQ.Bbox))
			{
				List<int> zeroP = new List<int>(inP.NumVert());
				zeroP.Resize(inP.NumVert(), 0);
				List<int> zeroQ = new List<int>(inQ.NumVert());
				zeroQ.Resize(inQ.NumVert(), 0);
				return new Boolean3(
					new Intersections(),
					new Intersections(),
					zeroP,
					zeroQ,
					expandP,
					true);
			}

			// Level 3: find all edge-face intersections in both directions
			long? tTotal = Timing.Start();
			long? t = Timing.Start();

			// Phase-boundary fast-path: skip launching the next stage if cancel
			// fired between stages (C++ boolean3.cpp:530/536/552/558).
			if (Cancel.IsCancelled(token))
			{
				return null;
			}

			Intersections? xv12 = Boolean3Kernels.Intersect12(inP, inQ, expandP, true, token);
			if (xv12 is null)
			{
				return null;
			}

			Timing.Print("  Intersect12 P->Q", t);
			t = Timing.Start();
			if (Cancel.IsCancelled(token))
			{
				return null;
			}

			Intersections? xv21 = Boolean3Kernels.Intersect12(inP, inQ, expandP, false, token);
			if (xv21 is null)
			{
				return null;
			}

			Timing.Print("  Intersect12 Q->P", t);

			// The Rust guards `xv12.x12.len() > i32::MAX as usize`. A C# List cannot hold
			// more than int.MaxValue elements at all, so the comparison is transcribed
			// against `Count` and is unreachable rather than merely improbable — kept
			// because it is the only producer of `valid == false`, which
			// boolean_result checks.
			if (xv12.X12.Count > int.MaxValue || xv21.X12.Count > int.MaxValue)
			{
				return new Boolean3(
					new Intersections(),
					new Intersections(),
					new List<int>(),
					new List<int>(),
					expandP,
					false);
			}

			// Compute winding numbers via flood fill
			t = Timing.Start();
			if (Cancel.IsCancelled(token))
			{
				return null;
			}

			List<int>? w03 = Boolean3Kernels.Winding03(inP, inQ, xv12.P1q2, expandP, true, token);
			if (w03 is null)
			{
				return null;
			}

			Timing.Print("  Winding03 P", t);
			t = Timing.Start();
			if (Cancel.IsCancelled(token))
			{
				return null;
			}

			List<int>? w30 = Boolean3Kernels.Winding03(inP, inQ, xv21.P1q2, expandP, false, token);
			if (w30 is null)
			{
				return null;
			}

			Timing.Print("  Winding03 Q", t);
			Timing.Print("Intersections (total)", tTotal);

			return new Boolean3(xv12, xv21, w03, w30, expandP, true);
		}
	}
}
