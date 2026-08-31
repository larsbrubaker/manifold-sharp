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

// Minkowski.cs — port of minkowski.rs, whose header reads:
//
//   Phase 17: Minkowski Sum/Difference — ported from C++ minkowski.cpp (175 lines)
//
//   Implements the Minkowski sum/difference using:
//   - Convex+Convex: pairwise vertex sums → Hull
//   - NonConvex+Convex: per-triangle vertex sums → Hull → BatchBoolean (in batches)
//   - NonConvex+NonConvex: per-face-pair sums with coplanarity filtering → BatchBoolean
//
// Both per-face hull loops go through Par.MaybeParMap: they are the "minkowski
// hulls" entry in the six sites CLAUDE.md blesses for parallelism,
// and the ONLY place in the port carrying a documented determinism exception.
// Verbatim from the Rust port's plan, where it is the single caveat on
// "parallel is bit-identical to sequential":
//
//   One documented caveat: mesh-ID *values* from the global atomic counter can
//   be consumed in a different order under parallel minkowski hulls — they are
//   opaque process-global handles (already test-order dependent) and do not
//   affect geometry, topology, or any assertion.
//
// Concretely: each ConvexHull below calls InitializeOriginal, which takes an ID
// from ManifoldImpl.ReserveIds. Sequentially the hulls take them in face order;
// in parallel they take the same SET of IDs in whatever order the workers get
// there. Geometry, topology, counts and every TriRef *relationship* are
// unaffected — only which opaque integer labels which hull. Nothing may be
// tightened here that turns those handles into observable values.
//
// Measured, not merely asserted, by MinkowskiGeometryIsBitIdenticalInParallel:
// positions, indices, run boundaries and face IDs come out bit-identical, and
// the one field that moves — the exported RunOriginalId — already moves between
// two *sequential* runs, because every hull mints fresh IDs from a monotonic
// process-global counter. Parallelism reorders values that were never stable.
//
// NAME COLLISION, deliberate and load-bearing: this class makes `Minkowski` an
// unqualified name inside namespace ManifoldSharp, and Clipper2Lib has a
// `Minkowski` class of its own. A same-namespace type beats a using-directive
// one, so inside this assembly the bare spelling `Minkowski` now binds HERE.
// CrossSection.Clipper.cs therefore writes `Clipper2Lib.Minkowski.Sum(...)`
// fully qualified — that qualification is what keeps it compiling against the
// 2D Clipper sum, and removing it as "redundant" would silently rebind it to
// this class (or, more likely, fail to compile in a confusing way).
//
// Rust module-level functions land on a static class named for the module, so
// `minkowski::minkowski` would be `Minkowski.Minkowski` — which C# forbids (a
// member may not share its type's name). The three entry points are spelled
// Compute / Sum / Difference; each names the Rust function it ports.
//
// PROGRESS AND CANCELLATION, added by this port (the Rust threads neither into
// its minkowski, and CancelToken.cs's deviation list said so). Both arrive the
// way the boolean pipeline takes them — a trailing `CancelToken? token = null`
// and `ProgressReporter? progress = null`, whose null paths are byte-for-byte
// the code that ran before they existed, since Progress.MaybeParMapCtProgress
// with a null reporter and a null token IS Par.MaybeParMap and
// CsgNode.Evaluate() IS EvaluateWithToken(null). Nothing about the geometry
// moves.
//
//   Cancellation follows the port's status-not-exception rule: a cancelled run
//   answers with an empty ManifoldImpl carrying Error.Cancelled
//   (Boolean3Functions.CancelledImpl), exactly as the boolean and the CSG tree
//   do. Checks sit at the entry, at every hull-map element (Par's own per-element
//   poll), between batches / faces, and — the load-bearing one — after the final
//   BatchBoolean, so the invariant CancelToken.cs names holds here too: a
//   cancelled token can never produce a NoError result. Nothing is checked per
//   vertex; the hull inner loops stay exactly as they were.
//
//   Progress reports one Phase.Minkowski unit per hull, one per batch reduction
//   and one for the closing merge, and finishes with Progress.CompletePhase so a
//   watched run ends on exactly 1.0 — the throttle alone cannot, since it fires
//   only on multiples of total/100. The phase and that completion emit are both
//   appended to the Rust's progress module and are divergence ledger entry 4
//   (docs/RUST_DIVERGENCES.md).

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// The free functions of <c>minkowski.rs</c> — Minkowski sum and difference of two
	/// meshes.
	/// </summary>
	public static class Minkowski
	{
		private const int BatchSize = 1000;
		private const int ReduceThreshold = 200;
		private const double KCoplanarTol = 1e-12;

		/// <summary>
		/// Compute the Minkowski sum or difference of two meshes.
		/// Port of C++ <c>Manifold::Impl::Minkowski()</c> (Rust <c>minkowski</c>).
		/// </summary>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <param name="inset">
		/// If true, computes the Minkowski difference (erosion); if false, computes the
		/// Minkowski sum (dilation).
		/// </param>
		/// <param name="token">
		/// The cancellation token, or null for an uncancellable run. A cancelled run answers
		/// with an empty mesh whose <see cref="ManifoldImpl.Status"/> is
		/// <see cref="Error.Cancelled"/>, as the boolean pipeline does.
		/// </param>
		/// <param name="progress">
		/// The progress reporter, or null for an uninstrumented run. Reports
		/// <see cref="Phase.Minkowski"/> only: one unit per hull, per batch reduction and for
		/// the closing merge.
		/// </param>
		/// <returns>The resulting mesh.</returns>
		public static ManifoldImpl Compute(
			ManifoldImpl a,
			ManifoldImpl b,
			bool inset,
			CancelToken? token = null,
			ProgressReporter? progress = null)
		{
			ArgumentNullException.ThrowIfNull(a);
			ArgumentNullException.ThrowIfNull(b);

			// Entry gate before any work, the shape every cancellable entry point in the port
			// opens with (CsgTree.SimpleBoolean, Boolean3Functions.BooleanWithToken): a
			// pre-cancelled token must not buy a single hull.
			if (Cancel.IsCancelled(token))
			{
				return Boolean3Functions.CancelledImpl();
			}

			ManifoldImpl aImpl = a;
			ManifoldImpl bImpl = b;

			bool aConvex = aImpl.IsConvex();
			bool bConvex = bImpl.IsConvex();

			// If the convex manifold was supplied first, swap them
			if (aConvex && !bConvex)
			{
				aImpl = b;
				bImpl = a;
				(aConvex, bConvex) = (bConvex, aConvex);
			}

			// Early-exit if either input is empty
			if (bImpl.IsEmpty())
			{
				ReportTrivialCompletion(progress);
				return aImpl.Clone();
			}

			if (aImpl.IsEmpty())
			{
				ReportTrivialCompletion(progress);
				return bImpl.Clone();
			}

			// Costed from the branch about to run, because the three strategies are orders of
			// magnitude apart: a bar driven by a branch-blind total would move at three
			// different speeds. WorkUnits repeats the branch conditions below and must stay in
			// step with them.
			Progress.BeginPhase(progress, Phase.Minkowski, WorkUnits(aImpl, bImpl, aConvex, bConvex, inset));

			List<ManifoldImpl> composedHulls = new List<ManifoldImpl>();
			composedHulls.Add(aImpl.Clone());

			// Convex-Convex Minkowski: Very Fast
			if (!inset && aConvex && bConvex)
			{
				// Rust reserves `b.len * a.len` in usize (64-bit); the same product in C#
				// `int` wraps negative past 2^31, and List's constructor would then answer a
				// 46,341-vert operand with a bare "capacity out of range". Both ports are
				// doomed on an input this size — the pairwise sum alone is >2 billion Vec3 —
				// so widen only to fail honestly about which limit was hit.
				long pairCount = (long)bImpl.VertPos.Count * aImpl.VertPos.Count;
				if (pairCount > int.MaxValue)
				{
					throw new ArgumentException(
						$"Convex Minkowski sum needs {pairCount} vertex pairs "
						+ $"({aImpl.VertPos.Count} x {bImpl.VertPos.Count}), past the "
						+ $"{int.MaxValue} a single hull input can hold.");
				}

				List<Vec3> simpleHull = new List<Vec3>((int)pairCount);
				foreach (Vec3 aVert in aImpl.VertPos)
				{
					foreach (Vec3 bVert in bImpl.VertPos)
					{
						simpleHull.Add(aVert + bVert);
					}
				}

				composedHulls.Add(QuickHullFunctions.ConvexHull(simpleHull));
				progress?.Advance(1);
			}

			// Convex + Non-Convex (or inset): Slower
			else if ((inset || !aConvex) && bConvex)
			{
				int numTri = aImpl.NumTri();

				// Process in batches. Each per-triangle hull is independent (C++ runs
				// this loop via for_each_n); results are collected in index order so
				// the batch content matches sequential. C++ pushes every hull
				// unconditionally — no empty filter here (unlike the
				// non-convex×non-convex branch); filtering would shift BatchBoolean
				// serials and change the reduction order.
				int offset = 0;
				while (offset < numTri)
				{
					// Per-batch gate, the granularity CsgTree.BatchBoolean checks at: the hull
					// map polls per element on its own, so this is what catches a cancel that
					// landed inside the previous batch's boolean.
					if (Cancel.IsCancelled(token))
					{
						return Boolean3Functions.CancelledImpl();
					}

					int numIter = Math.Min(numTri - offset, BatchSize);
					int batchOffset = offset;
					ManifoldImpl[]? newHulls = Progress.MaybeParMapCtProgress(numIter, 8, token, progress, iter =>
					{
						int tri = batchOffset + iter;
						List<Vec3> simpleHull = new List<Vec3>(3 * bImpl.VertPos.Count);
						for (int i = 0; i < 3; i++)
						{
							Vec3 aVert = aImpl.VertPos[aImpl.Halfedge[(tri * 3) + i].StartVert];
							foreach (Vec3 bVert in bImpl.VertPos)
							{
								simpleHull.Add(aVert + bVert);
							}
						}

						return QuickHullFunctions.ConvexHull(simpleHull);
					});

					// Null is Par's "a worker saw the flag"; the partial array goes with it.
					if (newHulls is null)
					{
						return Boolean3Functions.CancelledImpl();
					}

					composedHulls.Add(BatchBooleanImpls(newHulls, OpType.Add, token));
					progress?.Advance(1);
					offset += BatchSize;
				}
			}

			// Non-Convex + Non-Convex: Very Slow
			else if (!aConvex && !bConvex)
			{
				int numTriA = aImpl.NumTri();
				int numTriB = bImpl.NumTri();

				List<ManifoldImpl> accumulated = new List<ManifoldImpl>();

				for (int aFace = 0; aFace < numTriA; aFace++)
				{
					// Per-face gate: one A-face is a whole map plus a BatchBoolean, so this is
					// the coarsest granularity that still bounds cancel latency by one face.
					if (Cancel.IsCancelled(token))
					{
						return Boolean3Functions.CancelledImpl();
					}

					Vec3 a1 = aImpl.VertPos[aImpl.Halfedge[aFace * 3].StartVert];
					Vec3 a2 = aImpl.VertPos[aImpl.Halfedge[(aFace * 3) + 1].StartVert];
					Vec3 a3 = aImpl.VertPos[aImpl.Halfedge[(aFace * 3) + 2].StartVert];
					Vec3 nA = aImpl.FaceNormal[aFace];

					// Per-B-face hulls are independent (C++ parallel for_each_n over
					// bFace); collect in index order, then filter like C++'s
					// validFaceHulls pass so batch content and order match sequential.
					ManifoldImpl?[]? hulls = Progress.MaybeParMapCtProgress<ManifoldImpl?>(numTriB, 8, token, progress, bFace =>
					{
						Vec3 nB = bImpl.FaceNormal[bFace];
						double dotSame = Dot(nA, nB);
						double dotOpp = Dot(nA, new Vec3(-nB.X, -nB.Y, -nB.Z));
						bool coplanar = Math.Abs(dotSame - 1.0) < KCoplanarTol
							|| Math.Abs(dotOpp - 1.0) < KCoplanarTol;
						if (coplanar)
						{
							return null;
						}

						Vec3 b1 = bImpl.VertPos[bImpl.Halfedge[bFace * 3].StartVert];
						Vec3 b2 = bImpl.VertPos[bImpl.Halfedge[(bFace * 3) + 1].StartVert];
						Vec3 b3 = bImpl.VertPos[bImpl.Halfedge[(bFace * 3) + 2].StartVert];

						return QuickHullFunctions.ConvexHull(new Vec3[]
						{
							a1 + b1,
							a1 + b2,
							a1 + b3,
							a2 + b1,
							a2 + b2,
							a2 + b3,
							a3 + b1,
							a3 + b2,
							a3 + b3,
						});
					});

					if (hulls is null)
					{
						return Boolean3Functions.CancelledImpl();
					}

					List<ManifoldImpl> faceHulls = new List<ManifoldImpl>();
					foreach (ManifoldImpl? hull in hulls)
					{
						if (hull is not null && !hull.IsEmpty())
						{
							faceHulls.Add(hull);
						}
					}

					if (faceHulls.Count > 0)
					{
						accumulated.Add(BatchBooleanImpls(faceHulls, OpType.Add, token));
					}

					// Periodically reduce to limit memory
					if (accumulated.Count >= ReduceThreshold)
					{
						ManifoldImpl reduced = BatchBooleanImpls(accumulated, OpType.Add, token);
						accumulated.Clear();
						accumulated.Add(reduced);
					}

					// One unit per A-face, whether or not it contributed a hull: the face is
					// the work item, and a coplanar-filtered face still cost a whole map.
					progress?.Advance(1);
				}

				if (accumulated.Count > 0)
				{
					composedHulls.Add(BatchBooleanImpls(accumulated, OpType.Add, token));
				}
			}

			// Final merge; C++ finishes with AsOriginal() = InitializeOriginal +
			// SetNormalsAndCoplanar.
			OpType op = inset ? OpType.Subtract : OpType.Add;
			ManifoldImpl outR = BatchBooleanImpls(composedHulls, op, token);

			// The closing check CancelToken.cs's invariant requires — "a cancelled token can
			// never produce a NoError result". Without it a cancel that landed inside this
			// last reduction would come back as an empty Cancelled leaf, and the two calls
			// below would dress it up into an empty NoError mesh instead.
			if (Cancel.IsCancelled(token))
			{
				return Boolean3Functions.CancelledImpl();
			}

			outR.InitializeOriginal();
			outR.SetNormalsAndCoplanar();

			// CompletePhase, not Advance(1), for the closing merge's unit. The throttle only
			// emits on a step boundary, and step is total/100, so on any run of more than 100
			// units the trailing advances are swallowed and the bar would stop just short
			// (510/514 on a 512-triangle erosion). This is the one emit that is unconditional.
			// Reached only on success: the cancelled returns above all skip it, because a full
			// bar is a claim that the work was done.
			Progress.CompletePhase(progress);
			return outR;
		}

		/// <summary>
		/// Convenience wrapper: Minkowski sum (dilation). The Rust <c>minkowski_sum</c>.
		/// </summary>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>The dilated mesh.</returns>
		public static ManifoldImpl Sum(
			ManifoldImpl a,
			ManifoldImpl b,
			CancelToken? token = null,
			ProgressReporter? progress = null)
		{
			return Compute(a, b, false, token, progress);
		}

		/// <summary>
		/// Convenience wrapper: Minkowski difference (erosion). The Rust
		/// <c>minkowski_difference</c>.
		/// </summary>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>The eroded mesh.</returns>
		public static ManifoldImpl Difference(
			ManifoldImpl a,
			ManifoldImpl b,
			CancelToken? token = null,
			ProgressReporter? progress = null)
		{
			return Compute(a, b, true, token, progress);
		}

		/// <summary>
		/// How many progress units the branch that is about to run will report.
		/// </summary>
		/// <remarks>
		/// One unit per hull, one per batch reduction and one for the closing merge, counted
		/// exactly, so the fractions are scaled against the work that actually runs. The
		/// closing merge's unit is the only one not spent by an <c>Advance</c>: it is reported
		/// by <see cref="Progress.CompletePhase"/> instead, which is what makes a watched run
		/// land on 1.0 at any total rather than wherever the throttle last fired. The
		/// conditions repeat the branch chain in <see cref="Compute"/> and have to be changed
		/// with it.
		/// </remarks>
		/// <param name="aImpl">The first operand, after the convexity swap.</param>
		/// <param name="bImpl">The second operand, after the convexity swap.</param>
		/// <param name="aConvex">Whether the first operand is convex.</param>
		/// <param name="bConvex">Whether the second operand is convex.</param>
		/// <param name="inset">Whether this is an erosion.</param>
		/// <returns>The work-unit total for <see cref="ProgressReporter.BeginPhase"/>.</returns>
		private static ulong WorkUnits(
			ManifoldImpl aImpl,
			ManifoldImpl bImpl,
			bool aConvex,
			bool bConvex,
			bool inset)
		{
			// Every branch finishes with one BatchBoolean over composedHulls.
			const ulong FinalMerge = 1;

			if (!inset && aConvex && bConvex)
			{
				// One hull over the pairwise vertex sums.
				return 1 + FinalMerge;
			}

			if ((inset || !aConvex) && bConvex)
			{
				ulong numTri = (ulong)aImpl.NumTri();
				ulong batches = (numTri + BatchSize - 1) / BatchSize;
				return numTri + batches + FinalMerge;
			}

			if (!aConvex && !bConvex)
			{
				// The periodic ReduceThreshold merges are folded into the per-A-face unit
				// rather than counted separately: how many of them run depends on how many
				// faces survived the coplanarity filter, which is not knowable up front.
				ulong numTriA = (ulong)aImpl.NumTri();
				ulong numTriB = (ulong)bImpl.NumTri();
				return (numTriA * numTriB) + numTriA + FinalMerge;
			}

			// Unreachable: the swap above puts the convex operand second, so a convex `a`
			// with a non-convex `b` cannot arrive here. Costed as the merge alone so that a
			// future fourth branch reports something rather than dividing by zero.
			return FinalMerge;
		}

		/// <summary>
		/// Reports a whole bar for a call that returns without computing anything — the empty
		/// operand early-exits, which answer with the other operand's clone.
		/// </summary>
		/// <remarks>
		/// A watching caller then sees a finished operation instead of a bar that never
		/// started, which is the same reason <see cref="ProgressReporter.BeginPhase"/> emits
		/// unconditionally.
		/// </remarks>
		/// <param name="progress">The progress reporter, or null.</param>
		private static void ReportTrivialCompletion(ProgressReporter? progress)
		{
			Progress.BeginPhase(progress, Phase.Minkowski, 1);
			Progress.CompletePhase(progress);
		}

		/// <summary>Helper: BatchBoolean on ManifoldImpl directly via the CSG tree.</summary>
		/// <param name="meshes">The operands.</param>
		/// <param name="op">The operation.</param>
		/// <param name="token">
		/// The cancellation token, or null. <c>EvaluateWithToken(null)</c> is what
		/// <c>Evaluate()</c> already called, so the untokened path is unchanged.
		/// </param>
		/// <returns>The reduced mesh.</returns>
		private static ManifoldImpl BatchBooleanImpls(
			IReadOnlyList<ManifoldImpl> meshes,
			OpType op,
			CancelToken? token)
		{
			if (meshes.Count == 0)
			{
				return new ManifoldImpl();
			}

			if (meshes.Count == 1)
			{
				return meshes[0].Clone();
			}

			CsgNode[] children = new CsgNode[meshes.Count];
			for (int i = 0; i < meshes.Count; i++)
			{
				children[i] = new CsgLeaf(new CsgLeafNode(meshes[i].Clone()));
			}

			CsgNode tree = new CsgOp(op, children);
			return tree.EvaluateWithToken(token);
		}
	}
}
