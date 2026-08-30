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

// RobustFunctions.cs — port of robust/mod.rs, whose header reads:
//
//   Robust boolean engine for general (possibly non-manifold) closed,
//   orientable triangle meshes.
//
//   Implements Barki, Guennebaud, Foufou 2015, "Exact, robust, and efficient
//   regularized Booleans on general 3D meshes" (docs/Exact, robust, and
//   efficient booleans.pdf). This engine is a parallel alternative to the
//   ported exact pipeline in src/boolean3.rs: it requires inputs only to be
//   geometrically closed and orientable (triangle soup is fine — connectivity
//   is never trusted), at the cost of exact rational arithmetic on the hard
//   predicate/construction cases.
//
//   Selection between the two engines is via `types::BooleanEngine`
//   (Exact | Robust | Auto); the exact engine remains the default and its
//   behavior is byte-identical to before this module existed.
//
//   Submodules (pipeline order):
//     exact              — rational points, filtered predicates, constructions
//     tri_tri            — exact triangle-triangle intersection (narrow phase)
//     arrangement        — per-triangle 2D arrangement of intersection prims
//     cdt                — exact constrained Delaunay triangulation
//     intersection_graph — broad phase, prim distribution, piece emission
//                          (split helpers: graph_types — edge keys, vertex
//                          interner, Piece/IntersectionGraph; graph_geom —
//                          boxes, clips, filtered on-segment tests;
//                          graph_self_cut — same-mesh narrow phase)
//     cells              — arrangement cell complex + winding propagation
//                          (cells_extract — containment predicate + boundary
//                          extraction, re-exported through `cells`)
//     ray_shoot          — exact winding numbers (residual component seeds)
//     soup               — triangle-soup import (closed/orientable validation)
//
//   Classification follows the mesh-arrangement formulation (Zhou, Grinspun,
//   Zorin, Jacobson 2016 — what libigl's mesh_boolean uses), which subsumes
//   both the paper's local Prop 2/3 ring walk and the per-component winding
//   queries this engine used before. The arrangement's cells carry a winding
//   number per operand, propagated combinatorially from the unbounded cell, so
//   adjacent regions cannot disagree; each operand's solid is {w ≥ 1} by
//   default, so a negative winding (an inverted region of a self-intersecting
//   scan) is never material. `types::WindingRule::Nonzero` switches that
//   predicate to {w ≠ 0} per call, which keeps inside-out geometry solid; the
//   winding numbers themselves never depend on the rule. The output keeps a
//   wall exactly where the operation's predicate differs across it, wound from
//   the cell labels rather than from the input face — which is what makes the
//   result closed and consistently oriented no matter how the input was wound.
//
//   The paper's explicit regularization pass — radial ring cancellation and
//   coincident-piece binding — has no counterpart here: thin material cancels
//   arithmetically in the winding sum, so there is nothing to discard up
//   front.
//
// ── Named `RobustFunctions` ──────────────────────────────────────────────────
// The Rust module's own name is `robust`, and that name is already this port's
// NAMESPACE (`ManifoldSharp.Robust`, which every sibling file lives in and which
// callers outside spell as `Robust.Soup`, `Robust.Cells`, …). The bare name is
// therefore taken, and the porting plan's rule gives the `Functions` suffix —
// the same call CdtFunctions, ArrangementFunctions and IntersectionGraphFunctions
// each made. The Rust's module-level submodule declarations have no counterpart:
// C# files in one namespace are siblings without a manifest.

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Robust
{
	/// <summary>
	/// The free functions of <c>robust/mod.rs</c>: the robust boolean engine's entry
	/// points and the pipeline they orchestrate.
	/// </summary>
	public static class RobustFunctions
	{
		/// <summary>
		/// Robust boolean of two impls (manifold or soup). Same observable contract as
		/// <see cref="Boolean3Functions.BooleanWithToken"/>: intersect exactly, arrange +
		/// retriangulate, build the arrangement's cell complex, propagate winding numbers,
		/// and keep the walls the operation's predicate separates.
		/// </summary>
		/// <remarks>
		/// Every operation is one predicate on a cell's winding vector, so
		/// <see cref="OpType.Subtract"/> needs no operand flip — it is simply "inside P and
		/// not inside Q".
		/// </remarks>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <param name="op">The boolean operation.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The result impl.</returns>
		public static ManifoldImpl Boolean(
			ManifoldImpl a,
			ManifoldImpl b,
			OpType op,
			CancelToken? token)
		{
			return BooleanWithProgress(a, b, op, token, null);
		}

		/// <summary>
		/// <see cref="Boolean"/> with optional progress reporting (see
		/// <see cref="ProgressReporter"/>). <c>null</c> is exactly <see cref="Boolean"/> —
		/// the fast paths below are not instrumented at all, since they return before any
		/// measurable work happens.
		/// </summary>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <param name="op">The boolean operation.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>The result impl.</returns>
		public static ManifoldImpl BooleanWithProgress(
			ManifoldImpl a,
			ManifoldImpl b,
			OpType op,
			CancelToken? token,
			ProgressReporter? progress)
		{
			return BooleanWithRule(a, b, op, WindingRule.Positive, token, progress);
		}

		/// <summary>
		/// <see cref="BooleanWithProgress"/> with an explicit winding rule.
		/// </summary>
		/// <remarks>
		/// The rule only reinterprets the arrangement's cell labels
		/// (<see cref="Cells.InResult"/>); intersection, arrangement, cell complex, and
		/// winding propagation are all rule-independent, so <see cref="WindingRule.Positive"/>
		/// here is byte-for-byte the historical pipeline.
		/// <para>
		/// The bbox-disjoint fast paths do not consult the rule at all — they never classify
		/// anything, they concatenate or return an operand. They therefore run only when every
		/// operand they *keep* is provably already the boundary of its own solid
		/// (<c>NeedsNoClassification</c>): both operands for the union, operand A alone for
		/// the difference (B is discarded whatever it is wound like). Otherwise they fall
		/// through to the full pipeline, which finds no cross intersections but still
		/// classifies each operand — so an inverted body is dropped under
		/// <see cref="WindingRule.Positive"/> and rewound to positive material under
		/// <see cref="WindingRule.Nonzero"/>, exactly as it would be if the boxes overlapped.
		/// Disjoint <see cref="OpType.Intersect"/> needs no gate: nothing can be shared,
		/// whatever the winding.
		/// </para>
		/// <para>
		/// The empty-operand fast paths above still return the other operand unclassified,
		/// keeping the historical pass-through of inverted geometry — they have no
		/// two-operand pipeline to fall through to.
		/// </para>
		/// </remarks>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <param name="op">The boolean operation.</param>
		/// <param name="rule">Which winding numbers count as solid material.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>The result impl.</returns>
		public static ManifoldImpl BooleanWithRule(
			ManifoldImpl a,
			ManifoldImpl b,
			OpType op,
			WindingRule rule,
			CancelToken? token,
			ProgressReporter? progress)
		{
			ArgumentNullException.ThrowIfNull(a);
			ArgumentNullException.ThrowIfNull(b);
			if (IsCancelled(token))
			{
				return CancelledImpl();
			}

			// Fast paths mirror the exact engine's observable behavior.
			if (a.IsEmpty())
			{
				return op == OpType.Add ? b.Clone() : new ManifoldImpl();
			}

			if (b.IsEmpty())
			{
				return op == OpType.Intersect ? new ManifoldImpl() : a.Clone();
			}

			List<Vec3[]> pTris = Soup.ImplToTris(a);
			List<Vec3[]> qTris = Soup.ImplToTris(b);
			List<double> pProps = Soup.ImplToCornerProps(a);
			List<double> qProps = Soup.ImplToCornerProps(b);

			if (!a.Bbox.DoesOverlapBox(b.Bbox))
			{
				// Only the operands a fast path actually *keeps* need vetting: the
				// union keeps both, the difference keeps only A.
				bool ACleanEnough() => NeedsNoClassification(a, pTris, token);
				bool BCleanEnough() => NeedsNoClassification(b, qTris, token);
				switch (op)
				{
					case OpType.Add when ACleanEnough() && BCleanEnough():
					{
						// Disjoint union: concatenate the soups and re-import. The
						// property context tags the two halves so each keeps its own
						// interpolated properties.
						List<Vec3[]> tris = new List<Vec3[]>(pTris);
						tris.AddRange(qTris);
						VertInterner interner = new VertInterner();
						List<Piece> pieces = new List<Piece>(tris.Count);
						for (int i = 0; i < tris.Count; i++)
						{
							Vec3[] t = tris[i];
							pieces.Add(new Piece(
								i < pTris.Count ? (byte)0 : (byte)1,
								i < pTris.Count ? i : i - pTris.Count,
								new UVec3(
									interner.InternF64(t[0]),
									interner.InternF64(t[1]),
									interner.InternF64(t[2]))));
						}

						PropCtx disjointCtx = new PropCtx(
							new[] { a.NumProp, b.NumProp },
							new IReadOnlyList<Vec3[]>[] { pTris, qTris },
							new IReadOnlyList<double>[] { pProps, qProps });
						PropCtx? disjointProps = disjointCtx.OutNumProp() > 0 ? disjointCtx : null;
						return AssembleFunctions.Assemble(
							pieces,
							interner.Verts,
							interner.VertsF64,
							_ => true,
							disjointProps).IntoImpl();
					}

					case OpType.Intersect:
						return new ManifoldImpl();
					case OpType.Subtract when ACleanEnough():
						return a.Clone();
					default:
						// A kept operand needs classification: fall through to the full
						// pipeline, which handles disjoint inputs fine (it simply finds no
						// cross intersections).
						break;
				}
			}

			// Subtraction needs no operand flip: the cell predicate expresses it
			// directly as "inside P and not inside Q", so both operands keep their
			// own winding and their corner properties stay in their original order.
			PropCtx ctx = new PropCtx(
				new[] { a.NumProp, b.NumProp },
				new IReadOnlyList<Vec3[]>[] { pTris, qTris },
				new IReadOnlyList<double>[] { pProps, qProps });
			return ClassifyAndAssemble(ctx, op, rule, token, progress);
		}

		/// <summary>
		/// Rebuild one mesh into a fresh, properly paired 2-manifold enclosing the same solid
		/// region under <paramref name="rule"/> — the single-operand form of the robust
		/// boolean.
		/// </summary>
		/// <remarks>
		/// The input may be arbitrary triangle soup: self-intersecting, T-junctioned,
		/// carrying duplicated or coincident sheets, more than two faces on an edge, or
		/// riddled with interior walls. It runs the identical pipeline
		/// <see cref="BooleanWithRule"/> uses (<c>ClassifyAndAssemble</c>), with mesh 1 empty
		/// and <see cref="OpType.Add"/>, which reduces the cell predicate to "is mesh 0's
		/// winding inside under <paramref name="rule"/>" — mesh 1's winding is 0 everywhere
		/// and no rule calls 0 solid. Self-intersections are still cut (the graph builder
		/// self-cuts every mesh, mesh 1's empty half included), so folds within the one
		/// operand classify exactly as they would against a partner.
		/// <para>
		/// This cannot be expressed as a union with an empty operand:
		/// <see cref="BooleanWithRule"/>'s empty-operand fast path returns the other operand
		/// *unclassified*, deliberately, to preserve the historical pass-through.
		/// </para>
		/// <para>
		/// Corner properties survive: the operand occupies mesh slot 0, the same slot a
		/// two-operand boolean gives it, so <see cref="PropCtx"/> interpolates them onto the
		/// rebuilt triangles unchanged.
		/// </para>
		/// </remarks>
		/// <param name="a">The mesh to rebuild.</param>
		/// <param name="rule">Which winding numbers count as solid material.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>The rebuilt impl.</returns>
		public static ManifoldImpl RebuildWithRule(
			ManifoldImpl a,
			WindingRule rule,
			CancelToken? token,
			ProgressReporter? progress)
		{
			ArgumentNullException.ThrowIfNull(a);
			if (IsCancelled(token))
			{
				return CancelledImpl();
			}

			if (a.IsEmpty())
			{
				return new ManifoldImpl();
			}

			List<Vec3[]> pTris = Soup.ImplToTris(a);
			List<double> pProps = Soup.ImplToCornerProps(a);
			PropCtx ctx = new PropCtx(
				new[] { a.NumProp, 0 },
				new IReadOnlyList<Vec3[]>[] { pTris, Array.Empty<Vec3[]>() },
				new IReadOnlyList<double>[] { pProps, Array.Empty<double>() });
			return ClassifyAndAssemble(ctx, OpType.Add, rule, token, progress);
		}

		/// <summary>
		/// Import a raw triangle list as a boolean result (used by
		/// <see cref="Boolean3Functions.ComposeMeshes"/> when any input is a soup; positions
		/// only — the property-aware disjoint-union path in <see cref="BooleanWithRule"/>
		/// builds its own tagged pieces).
		/// </summary>
		/// <param name="tris">The triangles.</param>
		/// <returns>The imported impl.</returns>
		internal static ManifoldImpl AssembleAll(IReadOnlyList<Vec3[]> tris)
		{
			ArgumentNullException.ThrowIfNull(tris);
			VertInterner interner = new VertInterner();
			List<Piece> pieces = new List<Piece>(tris.Count);
			for (int i = 0; i < tris.Count; i++)
			{
				Vec3[] t = tris[i];
				pieces.Add(new Piece(
					0,
					i,
					new UVec3(
						interner.InternF64(t[0]),
						interner.InternF64(t[1]),
						interner.InternF64(t[2]))));
			}

			return AssembleFunctions.Assemble(
				pieces,
				interner.Verts,
				interner.VertsF64,
				_ => true,
				null).IntoImpl();
		}

		/// <summary>True when a token exists and has been cancelled.</summary>
		/// <param name="token">The token, or null.</param>
		/// <returns>Whether the run should abandon.</returns>
		private static bool IsCancelled(CancelToken? token)
		{
			return token != null && token.IsCancelled;
		}

		/// <summary>The observable result of an interrupted run: empty, carrying Cancelled.</summary>
		/// <returns>The cancelled impl.</returns>
		private static ManifoldImpl CancelledImpl()
		{
			ManifoldImpl outImpl = new ManifoldImpl();
			outImpl.MakeEmpty(Error.Cancelled);
			return outImpl;
		}

		/// <summary>
		/// Can this operand be handed to a bbox-disjoint fast path verbatim — that is, is its
		/// surface already exactly the boundary of the solid it denotes, for either winding
		/// rule?
		/// </summary>
		/// <remarks>
		/// Two conditions, both exact and both conservative:
		/// <list type="bullet">
		/// <item><description>no self-intersections, so no piece of the surface is interior
		/// to its own body (a doubled or crossing sheet has walls the pipeline
		/// dissolves);</description></item>
		/// <item><description>every shell wound the way its nesting demands
		/// (<see cref="Repair.ShellsWellNested"/>), so no inverted body survives that
		/// {w &gt;= 1} would drop, and no nested outward shell hides a wall with material on
		/// both sides.</description></item>
		/// </list>
		/// <para>
		/// A <c>false</c> verdict only costs the full pipeline on a disjoint pair, which
		/// produces the same answer by construction — it just also classifies.
		/// </para>
		/// </remarks>
		/// <param name="imp">The operand.</param>
		/// <param name="tris">Its triangle soup.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>Whether the operand may pass through unclassified.</returns>
		private static bool NeedsNoClassification(
			ManifoldImpl imp,
			IReadOnlyList<Vec3[]> tris,
			CancelToken? token)
		{
			// Cheapest first, and usually already cached by `Auto`'s dispatch. A
			// cancelled scan answers "self-intersecting", which routes to the pipeline
			// and so reports `Error::Cancelled` rather than a bogus pass-through.
			return !Soup.HasSelfIntersectionsWithToken(imp, token) && Repair.ShellsWellNested(tris);
		}

		/// <summary>
		/// The pipeline proper: intersect (cross-mesh *and* self), arrange, build the cell
		/// complex, propagate winding numbers, keep the walls <paramref name="op"/>/
		/// <paramref name="rule"/> separates, assemble.
		/// </summary>
		/// <remarks>
		/// Everything the operands contribute travels in <paramref name="ctx"/> — positions
		/// in <see cref="PropCtx.Tris"/>, corner properties in <see cref="PropCtx.Props"/> —
		/// so the single-operand entry (<see cref="RebuildWithRule"/>) shares this body
		/// verbatim by handing mesh 1 an empty soup. Every stage is already indexed by mesh,
		/// and an empty mesh 1 simply contributes no triangles, no boxes and no primitives;
		/// its winding is 0 in every cell, which no rule calls solid.
		/// </remarks>
		/// <param name="ctx">The operands' positions and properties.</param>
		/// <param name="op">The boolean operation.</param>
		/// <param name="rule">The winding rule.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>The result impl.</returns>
		private static ManifoldImpl ClassifyAndAssemble(
			PropCtx ctx,
			OpType op,
			WindingRule rule,
			CancelToken? token,
			ProgressReporter? progress)
		{
			IReadOnlyList<Vec3[]> pTris = ctx.Tris[0];
			IReadOnlyList<Vec3[]> qTris = ctx.Tris[1];
			IntersectionGraph? graph = IntersectionGraphFunctions.BuildGraphWithProgress(
				ToArray(pTris),
				ToArray(qTris),
				token,
				progress);
			if (graph == null)
			{
				return CancelledImpl();
			}

			long? tCells = Timing.Start();
			CellComplex? complex = Cells.BuildCellsWithProgress(graph, token, progress);
			if (complex == null)
			{
				return CancelledImpl();
			}

			Timing.Print("robust: cell complex", tCells);

			// One exact query anchors each connected component; the rest of its
			// cells follow combinatorially. Winding and assembly report as phase
			// transitions only: neither has a work total the caller could see a
			// fraction of without instrumenting the exact ray queries themselves.
			Progress.BeginPhase(progress, Phase.Winding, 0);
			long? tWinding = Timing.Start();
			Windings wind = Cells.Windings(graph, complex, new[] { pTris, qTris });
			Timing.Print("robust: winding propagation", tWinding);
			if (IsCancelled(token))
			{
				return CancelledImpl();
			}

			// Boundary of the result, wound from the cell labels.
			Progress.BeginPhase(progress, Phase.Assemble, 0);
			List<Piece> pieces = Cells.Extract(graph, complex, wind, op, rule);
			PropCtx? props = ctx.OutNumProp() > 0 ? ctx : null;
			long? tAsm = Timing.Start();
			Manifold outManifold = AssembleFunctions.Assemble(
				pieces,
				graph.Verts,
				graph.VertsF64,
				_ => true,
				props);
			Timing.Print("robust: assemble+import", tAsm);
			return outManifold.IntoImpl();
		}

		/// <summary>
		/// The graph builder takes <c>Vec3[][]</c> (the Rust's <c>&amp;[[Vec3; 3]]</c>); this
		/// is the one adaptation the C# type system asks for at that seam.
		/// </summary>
		/// <param name="tris">The triangle soup.</param>
		/// <returns>The same triangles as an array.</returns>
		private static Vec3[][] ToArray(IReadOnlyList<Vec3[]> tris)
		{
			if (tris is Vec3[][] already)
			{
				return already;
			}

			Vec3[][] result = new Vec3[tris.Count][];
			for (int i = 0; i < tris.Count; i++)
			{
				result[i] = tris[i];
			}

			return result;
		}
	}
}
