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

// Port of the tests module in minkowski.rs — all 3 cases, same inputs, same
// expected values, same tolerances, same order. Nothing deferred.
//
// All three exercise the convex×convex branch (plus the empty early-exit). The
// non-convex branches — the ones with the coplanarity filter and the
// Par.MaybeParMap batches — have no inline Rust test either; upstream covers
// them in manifold_tests/advanced.rs, and those six cases are now ported, in
// ManifoldAdvancedTests.cs (CppConvexConvexMinkowski, ...Difference,
// CppNonConvexConvexMinkowskiSum, ...Difference, and both
// CppNonConvexNonConvex... cases), with their analytical volumes, surface areas
// and pinned genus values. They are the real cover for this file's subject; the
// three below are the inline smoke tests minkowski.rs carries itself.
//
// Beyond those, parity is held by the differential harness the Minkowski step ran
// against the compiled manifold-rust, which includes non-convex×convex and
// non-convex×non-convex pairs.
//
// The last six are C#-ONLY adaptation tests, counted separately as CLAUDE.md
// requires: they cover the progress/cancellation parameters this port adds to an
// entry point the Rust threads neither into (divergence ledger entry 4). They
// never stand in for a ported test — the three above are still the Rust's, with
// the Rust's inputs and expected values.
//
// ORDERING ASSERTIONS ONLY EVER RUN SEQUENTIALLY. Either the fixture keeps the
// per-face hull map below its parallel threshold of 8 (a tetrahedron's four
// faces), or the test pins ManifoldParallel.Enabled to false through RunWith. That
// is deliberate rather than incidental: ProgressReporter's own remarks allow two
// workers to cross the throttle together and report out of order — "a UI hint, not
// a ledger" — so a monotonicity or report-count assertion over a parallel map
// would be asserting something the type does not promise, and would fail only
// under MANIFOLD_PARALLEL=1. The one thing asserted in BOTH modes is the closing
// 1.0, which is emitted on the calling thread after the workers have joined.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class MinkowskiTests
	{
		[Test]
		public async Task ConvexConvexMinkowskiSum()
		{
			ManifoldImpl a = ManifoldImpl.Cube(Mat3x4.Identity());
			ManifoldImpl b = ManifoldImpl.Cube(Mat3x4.Identity());
			ManifoldImpl sum = Minkowski.Sum(a, b);
			await Assert.That(sum.NumTri())
				.IsGreaterThan(0)
				.Because("Minkowski sum should produce non-empty mesh");

			// Two unit cubes: Minkowski sum should be a 2x2x2 cube
			double vol = Math.Abs(sum.GetProperty(Property.Volume));
			await Assert.That(Math.Abs(vol - 8.0) < 0.5)
				.IsTrue()
				.Because($"Minkowski sum of two unit cubes should have volume ~8, got {vol}");
		}

		[Test]
		public async Task ConvexConvexMinkowskiDifference()
		{
			ManifoldImpl a = ManifoldImpl.Cube(
				LinalgFunctions.Mat4ToMat3x4(LinalgFunctions.ScalingMatrix(Vec3.Splat(2.0))));
			ManifoldImpl b = ManifoldImpl.Cube(LinalgFunctions.Mat4ToMat3x4(
				LinalgFunctions.TranslationMatrix(Vec3.Splat(-0.25))
				* LinalgFunctions.ScalingMatrix(Vec3.Splat(0.5))));
			ManifoldImpl diff = Minkowski.Difference(a, b);
			await Assert.That(diff.NumTri())
				.IsGreaterThan(0)
				.Because("Minkowski difference should produce non-empty mesh");
		}

		[Test]
		public async Task EmptyMinkowski()
		{
			ManifoldImpl a = ManifoldImpl.Cube(Mat3x4.Identity());
			ManifoldImpl b = new ManifoldImpl();
			ManifoldImpl sum = Minkowski.Sum(a, b);

			// If b is empty, result should be a
			await Assert.That(sum.NumTri()).IsEqualTo(a.NumTri());
		}

		// ── C#-only adaptation tests: the progress / cancellation parameters ─────────

		/// <summary>
		/// The convex×convex fast path reports the whole bar: it does one hull and one merge,
		/// so it has only 0 and 1 to say, and it has to say both.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task SumReportsProgressThatOnlyRisesAndEndsAtOne()
		{
			ManifoldImpl a = ManifoldImpl.Cube(Mat3x4.Identity());
			ManifoldImpl b = ManifoldImpl.Cube(Mat3x4.Identity());

			Sink sink = new Sink();
			ManifoldImpl sum = Minkowski.Sum(a, b, null, sink.Reporter());

			await Assert.That(sum.NumTri()).IsGreaterThan(0);
			await AssertRisesToOne(sink);
		}

		/// <summary>
		/// The per-face path — every erosion takes it, convex operands or not — reports one
		/// unit per hull, one per batch reduction and one for the closing merge.
		/// </summary>
		/// <remarks>
		/// A tetrahedron because its four faces keep the hull map under the parallel
		/// threshold of 8; see this file's header for why that matters to the ordering
		/// assertion.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task DifferenceReportsProgressThatOnlyRisesAndEndsAtOne()
		{
			ManifoldImpl a = ManifoldImpl.Tetrahedron(
				LinalgFunctions.Mat4ToMat3x4(LinalgFunctions.ScalingMatrix(Vec3.Splat(4.0))));
			ManifoldImpl b = ManifoldImpl.Cube(LinalgFunctions.Mat4ToMat3x4(
				LinalgFunctions.TranslationMatrix(Vec3.Splat(-0.25))
				* LinalgFunctions.ScalingMatrix(Vec3.Splat(0.5))));

			Sink sink = new Sink();
			ManifoldImpl diff = Minkowski.Difference(a, b, null, sink.Reporter());

			await Assert.That(diff.NumTri()).IsGreaterThan(0);
			await AssertRisesToOne(sink);

			// Four hulls, one batch reduction, one closing merge, plus BeginPhase's own
			// report: the unit model is asserted rather than only its endpoints, because it
			// is the thing a consumer's bar is scaled against.
			await Assert.That(sink.Events().Count).IsEqualTo(7);
		}

		/// <summary>
		/// A cancel that lands mid-computation stops the work almost at once, answers with the
		/// empty <see cref="Error.Cancelled"/> mesh the boolean pipeline answers with, and
		/// leaves both operands exactly as it found them.
		/// </summary>
		/// <remarks>
		/// The token is tripped from the progress callback — the earliest in-kernel moment a
		/// test can reach — so this measures cancellation latency in work units rather than in
		/// wall time, which is what makes it stable on a loaded machine.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task CancelMidComputationAbortsPromptlyAndLeavesTheInputsUntouched()
		{
			// 128 triangles, so an uncancelled erosion reports 130 units; the assertion below
			// is that a cancel after the first hull costs a small fraction of that.
			ManifoldImpl a = Manifold.Sphere(1.0, 16).AsImpl().Clone();
			ManifoldImpl b = ManifoldImpl.Cube(LinalgFunctions.Mat4ToMat3x4(
				LinalgFunctions.ScalingMatrix(Vec3.Splat(0.1))));
			await Assert.That(a.NumTri()).IsGreaterThan(100)
				.Because("the fixture has to be big enough that stopping early is visible");

			long aBefore = GeometryHash(a);
			long bBefore = GeometryHash(b);

			CancelToken token = new CancelToken();
			int reports = 0;

			// Cancel on the second callback: the first is BeginPhase's unconditional 0.0,
			// which arrives before any hull runs, so tripping on that would only re-test the
			// entry gate.
			ProgressReporter reporter = new ProgressReporter((_, _) =>
			{
				if (Interlocked.Increment(ref reports) >= 2)
				{
					token.Cancel();
				}
			});

			ManifoldImpl result = Minkowski.Difference(a, b, token, reporter);

			await Assert.That(result.Status).IsEqualTo(Error.Cancelled);
			await Assert.That(result.NumTri()).IsEqualTo(0)
				.Because("a cancelled result must be empty, as it is for a boolean");

			// Well under the 130 an uncancelled run reports. The ceiling is loose because the
			// parallel map cannot recall iterations already in flight (Par.MaybeParMapCt's
			// remarks), so a few extra hulls per core may finish after the flag is set.
			await Assert.That(Volatile.Read(ref reports)).IsLessThan(40)
				.Because($"cancel was ignored for {Volatile.Read(ref reports)} of 130 work units");

			await Assert.That(GeometryHash(a)).IsEqualTo(aBefore)
				.Because("Minkowski must not mutate the solid it was handed");
			await Assert.That(GeometryHash(b)).IsEqualTo(bBefore)
				.Because("Minkowski must not mutate the structuring element it was handed");
		}

		/// <summary>
		/// The whole point of the additive design: the defaulted parameters produce the
		/// bit-identical mesh an instrumented, tokened run produces.
		/// </summary>
		/// <remarks>
		/// The live-token half is the load-bearing one, exactly as in
		/// <c>CancelTests.AUsedThenCancelledTokenDoesNotAffectLaterOpsWithAFreshToken</c>: a
		/// token routes the hull maps through a structurally different collect inside
		/// <c>Par.MaybeParMapCt</c>, and a reporter wraps the map's closure, so "instrumenting
		/// changes nothing" has to be measured rather than assumed. Both branches are covered
		/// — the convex sum and the per-face erosion.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task DefaultParametersAreBitIdenticalToAnInstrumentedRun()
		{
			ManifoldImpl cube = ManifoldImpl.Cube(Mat3x4.Identity());
			ManifoldImpl tetra = ManifoldImpl.Tetrahedron(
				LinalgFunctions.Mat4ToMat3x4(LinalgFunctions.ScalingMatrix(Vec3.Splat(4.0))));
			ManifoldImpl tool = ManifoldImpl.Cube(LinalgFunctions.Mat4ToMat3x4(
				LinalgFunctions.ScalingMatrix(Vec3.Splat(0.5))));

			CancelToken live = new CancelToken();
			Sink sink = new Sink();

			await Assert.That(GeometryHash(Minkowski.Sum(cube, cube, live, sink.Reporter())))
				.IsEqualTo(GeometryHash(Minkowski.Sum(cube, cube)))
				.Because("instrumenting the convex sum moved a bit");
			await Assert.That(GeometryHash(Minkowski.Difference(tetra, tool, live, sink.Reporter())))
				.IsEqualTo(GeometryHash(Minkowski.Difference(tetra, tool)))
				.Because("instrumenting the per-face erosion moved a bit");

			await Assert.That(live.IsCancelled).IsFalse();
			await Assert.That(sink.Events().Count).IsGreaterThan(0)
				.Because("a run that reported nothing would prove nothing");
		}

		/// <summary>
		/// The bar has to land on 1.0 even when the work total is past the throttle's 100
		/// reports per phase — the regime where <c>step</c> exceeds 1 and every unit after the
		/// last step boundary is swallowed. Both parallel modes, because the closing emit is
		/// made on the calling thread after the workers have joined and must not depend on
		/// which loop ran.
		/// </summary>
		/// <remarks>
		/// The bug this pins: before <see cref="ProgressReporter.CompletePhase"/> existed, a
		/// 290-unit erosion stopped reporting at 288/290 and a UI that hid its bar when full
		/// never hid it. Every other test in this file has a total under 100, where
		/// <c>step</c> is 1 and the last advance happens to land on a boundary, so none of them
		/// could see it.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task ALargeRunEndsAtExactlyOneInBothParallelModes()
		{
			// 288 triangles: 288 hulls + one batch reduction + the closing merge is 290 units,
			// so the throttle's step is 2 and roughly half the units never report.
			ManifoldImpl solid = Manifold.Sphere(1.0, 24).AsImpl().Clone();
			ManifoldImpl tool = ManifoldImpl.Cube(LinalgFunctions.Mat4ToMat3x4(
				LinalgFunctions.ScalingMatrix(Vec3.Splat(0.1))));
			await Assert.That(solid.NumTri()).IsEqualTo(288)
				.Because("the unit total below is computed from this count");

			foreach (bool parallel in new[] { false, true })
			{
				Sink sink = new Sink();
				RunWith(parallel, () => Minkowski.Difference(solid, tool, null, sink.Reporter()));

				List<(string Name, double? Fraction)> events = sink.Events();
				await Assert.That(events.Count).IsLessThan(290)
					.Because($"parallel={parallel}: fewer reports than units is what makes this the throttled regime");
				await Assert.That(events[events.Count - 1].Fraction).IsEqualTo(1.0)
					.Because($"parallel={parallel}: a finished run must leave the bar full, not at 288/290");
				foreach ((string name, double? fraction) in events)
				{
					await Assert.That(name).IsEqualTo("minkowski");
					await Assert.That(fraction!.Value).IsLessThanOrEqualTo(1.0);
				}
			}
		}

		/// <summary>
		/// The non-convex × non-convex branch — a hull per face <em>pair</em> — reports the
		/// total its work model predicts, and stops promptly when cancelled.
		/// </summary>
		/// <remarks>
		/// Two overlapping cubes unioned into an L, 28 triangles each: 28 × 28 hulls, 28
		/// per-face reductions and the closing merge is 813 units, so <c>step</c> is 8 and the
		/// throttle emits 101 times (at 8, 16 … 808) plus the opening and closing reports. That
		/// exact count is the assertion — it moves if the unit model does.
		/// <para>
		/// Run sequentially, since the per-B-face map is 28 wide and would go parallel: the
		/// report count and their order are only deterministic on one thread.
		/// </para>
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task NonConvexPairReportsItsExactTotalAndCancelsPromptly()
		{
			ManifoldImpl lShape = NonConvexPair();
			await Assert.That(lShape.IsConvex()).IsFalse()
				.Because("both operands have to be non-convex to reach the per-face-pair branch");
			await Assert.That(lShape.NumTri()).IsEqualTo(28)
				.Because("the 813-unit total below is computed from this count");

			Sink sink = new Sink();
			ManifoldImpl sum = RunWith(false, () => Minkowski.Sum(lShape, lShape, null, sink.Reporter()));
			await Assert.That(sum.NumTri()).IsGreaterThan(0);

			await AssertRisesToOne(sink);
			await Assert.That(sink.Events().Count).IsEqualTo(103)
				.Because("one opening report, 101 throttled ones and the closing 1.0");

			// And a cancel lands within a face pair or two rather than at the end.
			CancelToken token = new CancelToken();
			int reports = 0;
			ProgressReporter reporter = new ProgressReporter((_, _) =>
			{
				if (Interlocked.Increment(ref reports) >= 2)
				{
					token.Cancel();
				}
			});

			ManifoldImpl cancelled = RunWith(false, () => Minkowski.Sum(lShape, lShape, token, reporter));

			await Assert.That(cancelled.Status).IsEqualTo(Error.Cancelled);
			await Assert.That(cancelled.NumTri()).IsEqualTo(0);
			await Assert.That(Volatile.Read(ref reports)).IsLessThan(10)
				.Because($"cancel was ignored for {Volatile.Read(ref reports)} of the run's 103 reports");
		}

		/// <summary>
		/// Every reported fraction is in range and no larger than the one before it, and the
		/// last is exactly 1.0.
		/// </summary>
		/// <remarks>
		/// Only sound for a run whose Advance calls all come from one thread — see this file's
		/// header on why the fixtures are sized to keep the hull maps sequential.
		/// </remarks>
		/// <param name="sink">The sink that watched the run.</param>
		/// <returns>The assertion task.</returns>
		private static async Task AssertRisesToOne(Sink sink)
		{
			List<(string Name, double? Fraction)> events = sink.Events();
			await Assert.That(events.Count).IsGreaterThan(1)
				.Because("BeginPhase alone is not progress");

			double previous = -1.0;
			foreach ((string name, double? fraction) in events)
			{
				await Assert.That(name).IsEqualTo("minkowski");
				await Assert.That(fraction).IsNotNull()
					.Because("Minkowski knows its work total, so no report is indeterminate");
				double f = fraction!.Value;
				await Assert.That(f).IsGreaterThanOrEqualTo(previous)
					.Because($"progress went backwards, {previous} then {f}");
				await Assert.That(f).IsLessThanOrEqualTo(1.0);
				previous = f;
			}

			await Assert.That(events[0].Fraction).IsEqualTo(0.0);
			await Assert.That(events[events.Count - 1].Fraction).IsEqualTo(1.0)
				.Because("the closing merge is reported by CompletePhase, so a finished run lands on 1.0");
		}

		/// <summary>
		/// Two unit cubes unioned into an L — the cheapest non-convex solid the port can build
		/// out of its own primitives.
		/// </summary>
		/// <returns>The 28-triangle non-convex mesh.</returns>
		private static ManifoldImpl NonConvexPair()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);
			return cube.Union(cube.Translate(new Vec3(0.6, 0.6, 0.0))).AsImpl().Clone();
		}

		/// <summary>
		/// Runs <paramref name="operation"/> with <see cref="ManifoldParallel.Enabled"/> forced,
		/// restoring it afterwards — the same helper shape <c>ParallelismTests</c> uses, and the
		/// reason the tests that call it carry its <c>NotInParallel</c> key.
		/// </summary>
		/// <typeparam name="T">The operation's result type.</typeparam>
		/// <param name="parallel">Whether to run with parallelism enabled.</param>
		/// <param name="operation">The operation to run.</param>
		/// <returns>The operation's result.</returns>
		private static T RunWith<T>(bool parallel, Func<T> operation)
		{
			bool restore = ManifoldParallel.Enabled;
			try
			{
				ManifoldParallel.Enabled = parallel;
				return operation();
			}
			finally
			{
				ManifoldParallel.Enabled = restore;
			}
		}

		/// <summary>
		/// FNV-1a over the raw bit patterns of every vertex coordinate and halfedge index —
		/// a bit-exact fingerprint rather than a tolerance comparison.
		/// </summary>
		/// <remarks>
		/// Deliberately does NOT read mesh IDs. Those move between two <em>sequential</em>
		/// runs, because every hull mints fresh ones from a process-global counter; see
		/// Minkowski.cs's header on the port's one determinism exception.
		/// </remarks>
		/// <param name="mesh">The mesh to fingerprint.</param>
		/// <returns>The fingerprint.</returns>
		private static long GeometryHash(ManifoldImpl mesh)
		{
			unchecked
			{
				ulong hash = 14695981039346656037UL;

				void Mix(ulong value)
				{
					for (int shift = 0; shift < 64; shift += 8)
					{
						hash ^= (value >> shift) & 0xFF;
						hash *= 1099511628211UL;
					}
				}

				foreach (Vec3 vert in mesh.VertPos)
				{
					Mix((ulong)BitConverter.DoubleToInt64Bits(vert.X));
					Mix((ulong)BitConverter.DoubleToInt64Bits(vert.Y));
					Mix((ulong)BitConverter.DoubleToInt64Bits(vert.Z));
				}

				foreach (Halfedge edge in mesh.Halfedge)
				{
					Mix((ulong)(uint)edge.StartVert);
					Mix((ulong)(uint)edge.EndVert);
					Mix((ulong)(uint)edge.PairedHalfedge);
				}

				return (long)hash;
			}
		}

		/// <summary>
		/// Collects every callback a run emits, the same shape <c>ProgressTests</c> uses (the
		/// reporter may call back from any thread).
		/// </summary>
		private sealed class Sink
		{
			private readonly object gate = new object();

			private readonly List<(string Name, double? Fraction)> events = new List<(string Name, double? Fraction)>();

			public ProgressReporter Reporter()
			{
				return new ProgressReporter((phase, fraction) =>
				{
					lock (this.gate)
					{
						this.events.Add((phase.Name(), fraction));
					}
				});
			}

			public List<(string Name, double? Fraction)> Events()
			{
				lock (this.gate)
				{
					return new List<(string Name, double? Fraction)>(this.events);
				}
			}
		}
	}
}
