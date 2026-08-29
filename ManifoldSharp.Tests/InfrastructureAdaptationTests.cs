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

// NOT PORTED FROM RUST. Every test in this file covers a mechanism that exists
// only in the C# port, so none of them counts toward the test-for-test tally in
// PORTING_PLAN.md ("the C# suite ends at the same count as the Rust's 763").
// They live in their own file precisely so that tally stays auditable: the
// ported suites are CancelTests.cs, ProgressTests.cs, DisjointSetsTests.cs and
// their siblings, and each of those maps one-to-one onto a Rust test module.
//
// The rule that produced this file: the Rust suite cannot cover code the Rust
// does not have. Where the port had to invent a mechanism — because .NET has no
// relaxed atomic, no mutex poisoning, no `Option`-returning collect, and a
// cancellation model of its own — that invention is the least-reviewed code in
// the repo and needs tests most. What is covered here:
//
//   Par.MaybeParMap / MaybeParMapCt   the hand-written short circuit that
//                                     stands in for Rust's collect::<Option<_>>
//   Progress fault latch              the stand-in for Mutex poisoning
//   Progress.MaybeParMapCtProgress    the counting wrapper's index order
//   Timing                            env gating and the C++ Timer::Print format
//   CancelToken(CancellationToken)    the BCL bridge, which has no Rust analogue

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class InfrastructureAdaptationTests
	{
		// ---------------------------------------------------------------------
		// Par: the short circuit that replaces Rust's collect::<Option<Vec<T>>>
		// ---------------------------------------------------------------------

		[Test]
		public async Task MaybeParMapProducesIndexOrderedResults()
		{
			int[] observed = new int[5];
			int calls = 0;
			int[] result = Par.MaybeParMap(5, 2, i =>
			{
				observed[i] = ++calls;
				return i * 10;
			});

			await Assert.That(result.Length).IsEqualTo(5);
			for (int i = 0; i < 5; i++)
			{
				await Assert.That(result[i]).IsEqualTo(i * 10);
			}

			// Sequential today, so f runs in index order. Phase 11 may reorder the
			// *calls*, but never the results, which is why the two are asserted apart.
			await Assert.That(calls).IsEqualTo(5);
			await Assert.That(observed[4]).IsEqualTo(5);
		}

		[Test]
		public async Task MaybeParMapCtWithALiveTokenMatchesTheUntokenedResult()
		{
			CancelToken live = new CancelToken();
			int[] tokened = Par.MaybeParMapCt(6, 2, live, i => i * i)!;
			int[] untokened = Par.MaybeParMap(6, 2, i => i * i);

			await Assert.That(live.IsCancelled).IsFalse();
			await Assert.That(tokened.Length).IsEqualTo(untokened.Length);
			for (int i = 0; i < untokened.Length; i++)
			{
				await Assert.That(tokened[i]).IsEqualTo(untokened[i]);
			}
		}

		[Test]
		public async Task MaybeParMapCtOnAPreCancelledTokenNeverCallsTheMap()
		{
			CancelToken token = new CancelToken();
			token.Cancel();

			int calls = 0;
			int[]? result = Par.MaybeParMapCt(1000, 2, token, i =>
			{
				calls++;
				return i;
			});

			await Assert.That(result).IsNull();
			await Assert.That(calls).IsEqualTo(0);
		}

		[Test]
		public async Task MaybeParMapCtOnAPreCancelledTokenAllocatesNothing()
		{
			// `calls == 0` above is satisfied by the in-loop check alone, so it does not
			// pin the reason the check is *hoisted above the allocation*. This does:
			// without the hoist `new int[1_000_000]` zero-fills 4 MB before the first
			// loop iteration discards it, and the deferred
			// pre_cancelled_token_returns_cancelled_promptly test measures that cost.
			CancelToken token = new CancelToken();
			token.Cancel();

			long before = GC.GetAllocatedBytesForCurrentThread();
			int[]? result = Par.MaybeParMapCt(1_000_000, 2, token, i => i);
			long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

			await Assert.That(result).IsNull();
			await Assert.That(allocated).IsLessThan(100_000L);
		}

		[Test]
		public async Task MaybeParMapCtStopsAtTheIndexWhereCancellationLands()
		{
			CancelToken token = new CancelToken();
			int calls = 0;
			int[]? result = Par.MaybeParMapCt(100, 2, token, i =>
			{
				calls++;
				if (i == 3)
				{
					// Cancelled from inside the work itself, which is the realistic
					// shape: another thread flips the flag mid-map.
					token.Cancel();
				}

				return i;
			});

			await Assert.That(result).IsNull();

			// 0..3 ran (the cancel lands during index 3's own body), index 4 sees the
			// flag and returns. Nothing past the cancelled index is mapped.
			await Assert.That(calls).IsEqualTo(4);
		}

		[Test]
		public async Task MaybeParMapCtProgressCountsEveryItemAndKeepsIndexOrder()
		{
			List<(string Name, double? Fraction)> events = new List<(string Name, double? Fraction)>();
			object gate = new object();
			ProgressReporter reporter = new ProgressReporter((phase, fraction) =>
			{
				lock (gate)
				{
					events.Add((phase.Name(), fraction));
				}
			});

			reporter.BeginPhase(Phase.Cells, 8);
			int[]? result = Progress.MaybeParMapCtProgress(8, 2, null, reporter, i => i + 100);

			await Assert.That(result).IsNotNull();
			for (int i = 0; i < 8; i++)
			{
				await Assert.That(result![i]).IsEqualTo(i + 100);
			}

			// total 8 / 100 reports floors to 0, so step clamps to 1 and every item
			// reports: one begin plus eight advances.
			await Assert.That(events.Count).IsEqualTo(9);
			await Assert.That(events[0].Fraction).IsEqualTo(0.0);
			await Assert.That(events[8].Fraction).IsEqualTo(1.0);

			// The wrapper must not change what the unwrapped map returns.
			int[] plain = Par.MaybeParMap(8, 2, i => i + 100);
			for (int i = 0; i < 8; i++)
			{
				await Assert.That(result![i]).IsEqualTo(plain[i]);
			}
		}

		[Test]
		public async Task MaybeParMapCtProgressWithoutAReporterIsThePlainMap()
		{
			int[]? result = Progress.MaybeParMapCtProgress(4, 2, null, null, i => i * 3);

			await Assert.That(result).IsNotNull();
			for (int i = 0; i < 4; i++)
			{
				await Assert.That(result![i]).IsEqualTo(i * 3);
			}

			CancelToken cancelled = new CancelToken();
			cancelled.Cancel();
			await Assert.That(Progress.MaybeParMapCtProgress(4, 2, cancelled, null, i => i)).IsNull();
		}

		// ---------------------------------------------------------------------
		// Progress: the fault latch that stands in for Mutex poisoning
		// ---------------------------------------------------------------------

		[Test]
		public async Task AFaultedCallbackThrowsOnceThenGoesSilent()
		{
			int calls = 0;
			ProgressReporter reporter = new ProgressReporter((phase, fraction) =>
			{
				calls++;
				throw new InvalidOperationException("sink is broken");
			});

			// The first failure unwinds into the caller, exactly as a Rust panic in the
			// callback unwinds out of `emit`.
			await Assert.That(() => reporter.BeginPhase(Phase.Winding, 0))
				.Throws<InvalidOperationException>();
			await Assert.That(calls).IsEqualTo(1);

			// Afterwards the sink is poisoned: further reports are dropped rather than
			// throwing again, so a broken progress sink cannot take down a geometry
			// operation.
			reporter.BeginPhase(Phase.Cells, 4);
			reporter.Advance(1);
			reporter.Advance(3);
			await Assert.That(calls).IsEqualTo(1);
		}

		[Test]
		public async Task TheFaultLatchIsRecheckedInsideTheLock()
		{
			// The race the outer check alone does not close: a thread that reads
			// `callbackFaulted == false`, then blocks on the lock while another thread
			// is inside the callback that is about to throw. Without the re-check
			// inside the lock it calls the sink that has just been declared broken, and
			// a second exception escapes into the kernel.
			int calls = 0;
			ManualResetEventSlim insideCallback = new ManualResetEventSlim(false);
			ManualResetEventSlim releaseCallback = new ManualResetEventSlim(false);

			ProgressReporter reporter = new ProgressReporter((phase, fraction) =>
			{
				if (Interlocked.Increment(ref calls) == 1)
				{
					insideCallback.Set();
					releaseCallback.Wait();
					throw new InvalidOperationException("sink is broken");
				}
			});

			Exception? lateFailure = null;
			Thread late = new Thread(() =>
			{
				try
				{
					reporter.BeginPhase(Phase.Cells, 0);
				}
				catch (Exception exception)
				{
					lateFailure = exception;
				}
			});

			Task<Exception?> first = Task.Run(() =>
			{
				try
				{
					reporter.BeginPhase(Phase.Winding, 0);
					return (Exception?)null;
				}
				catch (Exception exception)
				{
					return exception;
				}
			});

			insideCallback.Wait();
			late.Start();

			// Spin until the late thread is genuinely parked on the callback lock, so
			// the interleaving under test is the one that happens. Bounded rather than
			// timed: if a scheduler never parks it, it falls through to the same
			// assertions (it would then take the outer fast path instead) rather than
			// hanging the suite.
			SpinWait spin = default;
			for (int i = 0; i < 200_000 && (late.ThreadState & ThreadState.WaitSleepJoin) == 0; i++)
			{
				spin.SpinOnce();
			}

			releaseCallback.Set();
			Exception? firstFailure = await first;
			late.Join();

			await Assert.That(firstFailure).IsNotNull();
			await Assert.That(lateFailure).IsNull();
			await Assert.That(calls).IsEqualTo(1);
		}

		// ---------------------------------------------------------------------
		// CancelToken: the bridge from System.Threading.CancellationToken
		// ---------------------------------------------------------------------

		[Test]
		public async Task ABridgedTokenObservesCancellationOfItsSource()
		{
			using CancellationTokenSource source = new CancellationTokenSource();
			CancelToken bridged = new CancelToken(source.Token);

			await Assert.That(bridged.IsCancelled).IsFalse();
			await Assert.That(Cancel.IsCancelled(bridged)).IsFalse();

			source.Cancel();

			await Assert.That(bridged.IsCancelled).IsTrue();
			await Assert.That(Cancel.IsCancelled(bridged)).IsTrue();
		}

		[Test]
		public async Task ABridgedTokenIsBornCancelledWhenItsSourceAlreadyIs()
		{
			using CancellationTokenSource source = new CancellationTokenSource();
			source.Cancel();

			// Register invokes synchronously for an already-cancelled source, so no
			// separate pre-check is needed in the constructor.
			CancelToken bridged = new CancelToken(source.Token);
			await Assert.That(bridged.IsCancelled).IsTrue();
		}

		[Test]
		public async Task ABridgedNonCancellableTokenNeverCancels()
		{
			// CancellationToken.None cannot be cancelled, so Register is a no-op and the
			// bridge must neither throw nor latch.
			CancelToken bridged = new CancelToken(CancellationToken.None);
			await Assert.That(bridged.IsCancelled).IsFalse();

			// Cancelling the bridged token directly still works: the bridge adds a
			// source, it does not take ownership of the flag.
			bridged.Cancel();
			await Assert.That(bridged.IsCancelled).IsTrue();
		}

		// ---------------------------------------------------------------------
		// Timing: env gating and the C++ Timer::Print format
		// ---------------------------------------------------------------------

		[Test]
		public async Task TimingEnabledFollowsRustsNonEmptyRule()
		{
			await Assert.That(Timing.IsEnabledValue(null)).IsFalse();
			await Assert.That(Timing.IsEnabledValue(string.Empty)).IsFalse();

			// Any non-empty value is on, including the ones that read like "off". Rust
			// checks emptiness, not truthiness, and "helpfully" parsing these would be a
			// silent divergence in an instrumentation switch.
			await Assert.That(Timing.IsEnabledValue("1")).IsTrue();
			await Assert.That(Timing.IsEnabledValue("0")).IsTrue();
			await Assert.That(Timing.IsEnabledValue("false")).IsTrue();
			await Assert.That(Timing.IsEnabledValue(" ")).IsTrue();
		}

		[Test]
		public async Task TimingIsOffUnlessTheEnvironmentVariableIsSet()
		{
			// The gate is read once per process, so this test states the assumption it
			// depends on rather than pretending it can change it.
			string? configured = Environment.GetEnvironmentVariable("MANIFOLD_TIMING");
			await Assert.That(string.IsNullOrEmpty(configured)).IsTrue();

			await Assert.That(Timing.IsEnabled).IsFalse();

			// The whole no-op contract hangs off this null: Start returns nothing, so
			// Print's `t0 is null` guard fires and no stage line is ever built.
			await Assert.That(Timing.Start()).IsNull();
		}

		[Test]
		[NotInParallel]
		public async Task TimingFormatsTheCppTimerLine()
		{
			// Asserted against the format seam rather than by capturing stderr: TUnit
			// owns the console writer (analyzer TUnit0055), and this checks the string
			// the production path actually builds, not a copy of it.
			await Assert.That(Timing.FormatStageLine("stage", 0.25))
				.IsEqualTo("stage: 0.25 sec");

			// With a hook registered the line grows the two MB fields at Rust's {:.1}.
			Timing.SetMemHook(static () => (3 * 1048576L, 5 * 1048576L));
			await Assert.That(Timing.FormatStageLine("stage", 0.25))
				.IsEqualTo("stage: 0.25 sec, current = 3.0 MB, stage peak = 5.0 MB");

			// Rust's `let _ = MEM_HOOK.set(hook)`: a second registration is dropped, not
			// applied and not an error.
			Timing.SetMemHook(static () => (7 * 1048576L, 9 * 1048576L));
			await Assert.That(Timing.FormatStageLine("stage", 0.25))
				.IsEqualTo("stage: 0.25 sec, current = 3.0 MB, stage peak = 5.0 MB");
		}
	}
}
