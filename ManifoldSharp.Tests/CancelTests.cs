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

// Port of cancel_tests.rs: the cancellation mechanism (CancelToken.cs) and its
// threading through the boolean / CSG-tree pipeline. The C++ counterparts live
// in cpp-reference/manifold/test/context_test.cpp.
//
// ALL NINE of cancel_tests.rs are now ported: the token-mechanics test that was
// always portable, plus the eight that drive a real boolean or a real CSG tree
// through the `Manifold` façade, which Phase 6 landed. The `slow_pair()` fixture
// (two 256-segment spheres offset by 0.5) is here too, and — as the Rust intends
// — it is genuinely slow enough to make a pre-cancelled early return measurable:
// ~150 ms for one union on this machine.
//
// Nothing in this file is deferred any more. The earlier phases' DEFERRED table
// listed six of the eight as additionally needing Phase 7's subdivide-backed
// `Manifold::sphere`; that landed alongside the façade, so the whole module is
// portable and ported.
//
// ── One RELATIVE-TIMING test, and one that counts work instead ───────────────
// `PreCancelledTokenReturnsCancelledPromptly` asserts, as the Rust does, that a
// pre-cancelled run finishes in under a quarter of the uncancelled run, measuring
// both in-test: an absolute millisecond threshold would be a machine-speed
// lottery, whereas a ratio survives a loaded CI box because both numbers inflate
// together.
//
// `CancelFromAnotherThreadInterruptsABooleanInFlight` keeps the Rust's name but
// not its 2x wall-time ratio, a C#-side adaptation: that ratio compared two
// tens-of-milliseconds runs and failed on CI noise. It counts the robust engine's
// progress phases instead — the cancelled run must enter fewer than the full run —
// which a cancel ignored until the work finishes cannot pass on any machine. Its
// remarks carry the details.

using System.Diagnostics;

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class CancelTests
	{
		[Test]
		public async Task TokenStartsUncancelledAndCancelIsObservableThroughClones()
		{
			CancelToken token = new CancelToken();
			await Assert.That(token.IsCancelled).IsFalse();

			// The Rust clones the token here to show that clones share one flag,
			// matching the C++ ExecutionContext pimpl. In C# the reference *is* the
			// clone: `Arc<AtomicBool>` maps onto a class reference, so aliasing the
			// token is the port of `token.clone()`.
			CancelToken clone = token;
			await Assert.That(clone.IsCancelled).IsFalse();
			token.Cancel();
			await Assert.That(token.IsCancelled).IsTrue();
			await Assert.That(clone.IsCancelled).IsTrue();

			// Cancel is sticky: there is deliberately no way to clear it.
			token.Cancel();
			await Assert.That(token.IsCancelled).IsTrue();

			// The canonical reader, Rust's free `cancel::is_cancelled`: a null token is
			// C++'s nullptr ctx and reports uncancelled without touching a flag.
			await Assert.That(Cancel.IsCancelled(null)).IsFalse();
			await Assert.That(Cancel.IsCancelled(token)).IsTrue();
			await Assert.That(Cancel.IsCancelled(new CancelToken())).IsFalse();
		}

		/// <summary>
		/// The whole point of the additive design: passing null must produce the
		/// byte-identical result of the pre-existing entry point.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task NoTokenPathIsUnchanged()
		{
			Manifold a = Manifold.Cube(Vec3.Splat(1.0), true);
			Manifold b = Manifold.Sphere(0.6, 32);
			Manifold plain = a.Boolean(b, OpType.Subtract);
			Manifold withNone = a.BooleanWithToken(b, OpType.Subtract, null);

			await Assert.That(plain.Status()).IsEqualTo(Error.NoError);
			await Assert.That(withNone.Status()).IsEqualTo(Error.NoError);
			MeshGL m0 = plain.GetMeshGL(-1);
			MeshGL m1 = withNone.GetMeshGL(-1);
			await Assert.That(m0.VertProperties).IsEquivalentTo(m1.VertProperties, CollectionOrdering.Matching);
			await Assert.That(m0.TriVerts).IsEquivalentTo(m1.TriVerts, CollectionOrdering.Matching);
		}

		[Test]
		public async Task PreCancelledTokenReturnsCancelledPromptly()
		{
			CancelToken token = new CancelToken();
			token.Cancel();

			// A genuinely expensive operation: with the entry gate it must return
			// without doing any of the work.
			(Manifold a, Manifold b) = SlowPair();
			Stopwatch sw = Stopwatch.StartNew();
			Manifold baseline = a.BooleanWithToken(b, OpType.Add, null);
			await Assert.That(baseline.Status()).IsEqualTo(Error.NoError);
			TimeSpan uncancelled = sw.Elapsed;

			sw.Restart();
			Manifold result = a.BooleanWithToken(b, OpType.Add, token);
			TimeSpan cancelledElapsed = sw.Elapsed;

			await Assert.That(result.Status()).IsEqualTo(Error.Cancelled);
			await Assert.That(result.IsEmpty()).IsTrue().Because("a cancelled result must be empty");
			await Assert.That(cancelledElapsed * 4 < uncancelled)
				.IsTrue()
				.Because($"pre-cancelled boolean took {cancelledElapsed}, uncancelled takes {uncancelled}");
		}

		[Test]
		public async Task PreCancelledTokenShortCircuitsCsgTreeEvaluation()
		{
			CancelToken token = new CancelToken();
			token.Cancel();

			List<CsgNode> leaves = new List<CsgNode>();
			for (int i = 0; i < 8; i++)
			{
				leaves.Add(new CsgLeaf(
					Manifold.Sphere(1.0, 64).Translate(new Vec3(i * 0.3, 0.0, 0.0)).AsImpl().Clone()));
			}

			CsgNode tree = new CsgOp(OpType.Add, leaves);

			ManifoldImpl result = tree.EvaluateWithToken(token);
			await Assert.That(result.Status).IsEqualTo(Error.Cancelled);
			await Assert.That(result.NumTri()).IsEqualTo(0);
		}

		[Test]
		public async Task EmptyInputFastPathsStillHonourAPreCancelledToken()
		{
			// C++ ExecutionContextFromMeshGL.CancelBeforeEmptyInputWinsOverNoError:
			// cancel must beat the fast paths that would otherwise report NoError.
			CancelToken token = new CancelToken();
			token.Cancel();
			Manifold empty = Manifold.Empty();
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);

			await Assert.That(empty.BooleanWithToken(cube, OpType.Add, token).Status())
				.IsEqualTo(Error.Cancelled);
			await Assert.That(cube.BooleanWithToken(empty, OpType.Add, token).Status())
				.IsEqualTo(Error.Cancelled);
		}

		/// <summary>
		/// A cancel sent from another thread while a boolean is running stops the work before
		/// it finishes.
		/// </summary>
		/// <remarks>
		/// C#-ONLY ADAPTATION of the Rust's assertion, under the Rust's name. The Rust times an
		/// uncancelled and a cancelled run and asserts the cancelled one took under half as
		/// long; on a fast or loaded CI box that compared two tens-of-milliseconds numbers and
		/// failed on noise. The claim is the same, measured in work instead of time: the
		/// progress reporter's phase reports are driven by work, never by the clock, so the
		/// uncancelled run's list of phases is the same on every machine.
		/// <para>
		/// The cancelled run parks the kernel inside its first progress report until this
		/// thread has cancelled, so the cancel lands mid-flight at the same point every time.
		/// It must then report <see cref="Error.Cancelled"/>, return nothing, and enter fewer
		/// phases than the full run. A cancel ignored until the work finishes enters every one
		/// of them, however fast the machine. Parking inside the callback is safe: it holds
		/// only the reporter's own lock, which the cancelling thread never takes, and the first
		/// report comes from the calling thread before any parallel map starts.
		/// </para>
		/// <para>
		/// The robust engine is what makes the work countable: it reports eight determinate
		/// phases. The exact engine (the Rust test's, via <c>boolean_with_token</c>) reports one
		/// indeterminate <see cref="Phase.ExactBoolean"/> just before its entry gate, so a
		/// cancel it ignored would enter exactly as many phases as one it honoured, and a cancel
		/// landing at that report would only exercise the entry gate
		/// <see cref="PreCancelledTokenReturnsCancelledPromptly"/> already covers.
		/// </para>
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CancelFromAnotherThreadInterruptsABooleanInFlight()
		{
			// Small is fine: the test no longer needs a slow input, only one that runs the
			// whole robust pipeline.
			Manifold a = Manifold.Sphere(1.0, 32);
			Manifold b = Manifold.Sphere(1.0, 32).Translate(new Vec3(0.5, 0.0, 0.0));

			// The phases the kernel entered, in order, consecutive repeats folded (a
			// determinate phase reports many fractions). The reporter may call from a
			// worker thread under MANIFOLD_PARALLEL, hence the lock.
			List<Phase> Run(CancelToken? token, Action? onFirstReport, out Manifold result)
			{
				List<Phase> phases = new List<Phase>();
				ProgressReporter reporter = new ProgressReporter((phase, fraction) =>
				{
					bool first;
					lock (phases)
					{
						first = phases.Count == 0;
						if (first || phases[^1] != phase)
						{
							phases.Add(phase);
						}
					}

					if (first)
					{
						onFirstReport?.Invoke();
					}
				});

				result = a.BooleanWithEngineAndProgress(b, OpType.Add, BooleanEngine.Robust, token, reporter);
				lock (phases)
				{
					return phases.ToList();
				}
			}

			List<Phase> fullRun = Run(null, null, out Manifold baseline);
			await Assert.That(baseline.Status()).IsEqualTo(Error.NoError);
			await Assert.That(baseline.IsEmpty()).IsFalse();
			await Assert.That(fullRun.Count)
				.IsGreaterThan(1)
				.Because(
					"the kernel must pass through more than one phase, or there is no later work a "
					+ $"cancel could be shown to skip: [{string.Join(", ", fullRun)}]");

			// The Rust clones the token into the worker; a C# reference is that clone
			// (see TokenStartsUncancelledAndCancelIsObservableThroughClones).
			CancelToken token = new CancelToken();
			using ManualResetEventSlim insideTheKernel = new ManualResetEventSlim(false);
			using ManualResetEventSlim cancelSent = new ManualResetEventSlim(false);
			List<Phase> cancelledRun = new List<Phase>();
			Manifold? cancelled = null;
			Thread worker = new Thread(() =>
			{
				cancelledRun = Run(
					token,
					() =>
					{
						// Hold the kernel inside its first phase until the other thread has
						// cancelled, so the cancel lands mid-work on every machine.
						insideTheKernel.Set();
						cancelSent.Wait();
					},
					out Manifold result);
				cancelled = result;
			});
			worker.Start();

			// Not a delay: the boolean has reported its first phase and is parked there.
			insideTheKernel.Wait();
			token.Cancel();
			cancelSent.Set();
			worker.Join();

			await Assert.That(cancelled).IsNotNull();
			await Assert.That(cancelled!.Status()).IsEqualTo(Error.Cancelled);
			await Assert.That(cancelled.IsEmpty()).IsTrue().Because("a cancelled result must be empty");
			await Assert.That(cancelledRun.Count < fullRun.Count)
				.IsTrue()
				.Because(
					$"the cancelled boolean went on to report [{string.Join(", ", cancelledRun)}], "
					+ $"every phase of the full run [{string.Join(", ", fullRun)}] - the cancel is "
					+ "being ignored until the work finishes");
		}

		/// <summary>C++ ExecutionContextFreshContextEscapesCancel.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task AUsedThenCancelledTokenDoesNotAffectLaterOpsWithAFreshToken()
		{
			Manifold a = Manifold.Cube(Vec3.Splat(1.0), true);
			Manifold b = Manifold.Sphere(0.6, 32);

			CancelToken used = new CancelToken();
			Manifold first = a.BooleanWithToken(b, OpType.Subtract, used);
			await Assert.That(first.Status()).IsEqualTo(Error.NoError);

			// Cancelling after the fact must not retroactively change anything...
			used.Cancel();
			await Assert.That(first.Status()).IsEqualTo(Error.NoError);

			// ...and a fresh token evaluates normally, producing the same geometry as
			// the completed run *and* as the untokened path. That last comparison is
			// the load-bearing one: a live token routes through a structurally
			// different collect inside `maybe_par_map_ct`, so "a token that is never
			// cancelled changes nothing" has to be asserted, not assumed.
			CancelToken fresh = new CancelToken();
			Manifold second = a.BooleanWithToken(b, OpType.Subtract, fresh);
			await Assert.That(second.Status()).IsEqualTo(Error.NoError);
			await Assert.That(fresh.IsCancelled).IsFalse();
			Manifold untokened = a.BooleanWithToken(b, OpType.Subtract, null);
			MeshGL mFirst = first.GetMeshGL(-1);
			MeshGL mSecond = second.GetMeshGL(-1);
			MeshGL mNone = untokened.GetMeshGL(-1);
			await Assert.That(mFirst.TriVerts).IsEquivalentTo(mSecond.TriVerts, CollectionOrdering.Matching);
			await Assert.That(mSecond.TriVerts).IsEquivalentTo(mNone.TriVerts, CollectionOrdering.Matching);
			await Assert.That(mSecond.VertProperties)
				.IsEquivalentTo(mNone.VertProperties, CollectionOrdering.Matching);

			// The stale token is still cancelled and still bites when it is used.
			await Assert.That(a.BooleanWithToken(b, OpType.Subtract, used).Status())
				.IsEqualTo(Error.Cancelled);
		}

		[Test]
		public async Task ALiveTokenProducesIdenticalGeometryAboveTheParallelThresholds()
		{
			// `maybe_par_map_ct` does NOT reuse `maybe_par_map` when a token is
			// present: it runs an unindexed `map(Option<T>) -> collect::<Option<Vec>>`
			// where the untokened path runs an indexed `map -> collect::<Vec>`. Under
			// `--features parallel` those are different rayon pipelines, and the port's
			// cardinal rule is exact numerical match — order preservation across that
			// switch is a rayon guarantee we should be pinning, not trusting.
			//
			// Small inputs stay sequential and would never exercise it, so this uses
			// meshes comfortably past both `crate::par` thresholds used in the boolean:
			// 10_000 in intersect12 (over halfedges) and 512 in face2tri (over faces).
			Manifold a = Manifold.Sphere(1.0, 128);
			Manifold b = Manifold.Sphere(1.0, 128).Translate(new Vec3(0.5, 0.0, 0.0));
			await Assert.That(a.AsImpl().Halfedge.Count)
				.IsGreaterThan(10000)
				.Because(
					"input must cross the intersect12 parallel threshold, got "
					+ $"{a.AsImpl().Halfedge.Count} halfedges");

			CancelToken live = new CancelToken();
			Manifold tokened = a.BooleanWithToken(b, OpType.Add, live);
			Manifold untokened = a.BooleanWithToken(b, OpType.Add, null);
			await Assert.That(tokened.Status()).IsEqualTo(Error.NoError);
			await Assert.That(untokened.Status()).IsEqualTo(Error.NoError);
			await Assert.That(live.IsCancelled).IsFalse();

			MeshGL mt = tokened.GetMeshGL(-1);
			MeshGL mu = untokened.GetMeshGL(-1);
			await Assert.That(mt.TriVerts.Count).IsGreaterThan(30000).Because("result should be substantial");
			await Assert.That(mt.NumProp).IsEqualTo(mu.NumProp);
			await Assert.That(mt.VertProperties).IsEquivalentTo(mu.VertProperties, CollectionOrdering.Matching);
			await Assert.That(mt.TriVerts).IsEquivalentTo(mu.TriVerts, CollectionOrdering.Matching);
			await Assert.That(mt.RunIndex).IsEquivalentTo(mu.RunIndex, CollectionOrdering.Matching);
			await Assert.That(mt.FaceId).IsEquivalentTo(mu.FaceId, CollectionOrdering.Matching);

			// The other op types take different branches through boolean_result.
			foreach (OpType op in new[] { OpType.Subtract, OpType.Intersect })
			{
				Manifold with = a.BooleanWithToken(b, op, live);
				Manifold without = a.BooleanWithToken(b, op, null);
				await Assert.That(with.Status()).IsEqualTo(Error.NoError).Because($"{op}");
				await Assert.That(with.GetMeshGL(-1).VertProperties)
					.IsEquivalentTo(without.GetMeshGL(-1).VertProperties, CollectionOrdering.Matching)
					.Because(
						$"{op} vertex positions diverged between the tokened and "
						+ "untokened parallel paths");
				await Assert.That(with.GetMeshGL(-1).TriVerts)
					.IsEquivalentTo(without.GetMeshGL(-1).TriVerts, CollectionOrdering.Matching)
					.Because($"{op} triangle order diverged between the tokened and untokened parallel paths");
			}
		}

		[Test]
		public async Task CancelledStatusSurvivesTheCsgTreeRoot()
		{
			// A cancelled leaf must not be silently absorbed as "empty geometry" by
			// the enclosing tree: the root's status has to stay Cancelled.
			CancelToken token = new CancelToken();
			token.Cancel();
			Manifold a = Manifold.Cube(Vec3.Splat(1.0), true);
			Manifold b = Manifold.Cube(Vec3.Splat(1.0), true).Translate(new Vec3(0.5, 0.0, 0.0));
			CsgNode tree = new CsgOp(
				OpType.Subtract,
				new CsgLeaf(a.AsImpl().Clone()),
				new CsgLeaf(b.AsImpl().Clone()));
			await Assert.That(tree.EvaluateWithToken(token).Status).IsEqualTo(Error.Cancelled);

			// Same tree, no token: unaffected.
			await Assert.That(tree.Evaluate().Status).IsEqualTo(Error.NoError);
		}

		/// <summary>
		/// Two heavily overlapping spheres: the same shape C++
		/// <c>ExecutionContextCancelMidBoolean</c> uses, sized so a single boolean is slow
		/// enough to be interrupted mid-flight.
		/// </summary>
		/// <returns>The two operands.</returns>
		private static (Manifold A, Manifold B) SlowPair()
		{
			Manifold a = Manifold.Sphere(1.0, 256);
			Manifold b = Manifold.Sphere(1.0, 256).Translate(new Vec3(0.5, 0.0, 0.0));
			return (a, b);
		}
	}
}
