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

// Port of progress_tests.rs: the throttle's own arithmetic, and the two
// contracts a consumer relies on — phases arrive in pipeline order with
// fractions in [0, 1], and instrumenting a boolean cannot change its result.
//
// DEFERRED, and why. The four throttle-arithmetic tests are here. The other
// four drive a real boolean, which does not exist until Phase 5/6, so they are
// listed by their Rust names for whoever closes that phase:
//
//   robust_boolean_reports_monotonic_phases_with_valid_fractions   Phase 5/6
//     (also needs the `phase_id(name)` helper and the `spheres()` fixture)
//   a_reporter_does_not_change_the_result                          Phase 5/6
//   reporter_overhead                                              Phase 5/6
//     The one #[ignore]d measurement fixture in this module. When it is
//     written it keeps its ignore, as TUnit
//     [Skip("measurement fixture; run explicitly with --ignored --nocapture")].
//   the_exact_engine_reports_one_indeterminate_phase               Phase 5/6
//     (also needs the `cube(offset)` fixture)
//
// Nothing is stubbed: no empty-bodied placeholder stands in for them.
//
// RECHECKED at Phase 5 (the boolean engine landed, and with it
// Boolean3Functions.BooleanDispatchFull, which is what raises the
// `ExactBoolean` phase). None of the four became portable: all four build their
// operands through the façade, so Phase 6 blocks every one of them, and the
// three that sweep `BooleanEngine::Robust`/`Auto` additionally need Phase 10.
// What each one actually reads afterwards differs, and it matters for who ports
// them: `a_reporter_does_not_change_the_result` compares
// `get_mesh_gl(-1).tri_verts` / `.vert_properties` and `status()` — never
// `volume()` — while `the_exact_engine_reports_one_indeterminate_phase` is the
// one that reads `volume()`. The latter is the closest to portable: its only
// remaining dependency is `Manifold::cube` plus `volume()`, so it is the first
// of these to write when Phase 6 lands.

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ProgressTests
	{
		[Test]
		public async Task BeginPhaseAlwaysReportsAndNamesThePhase()
		{
			Sink sink = new Sink();
			ProgressReporter r = sink.Reporter();
			r.BeginPhase(Phase.NarrowPhase, 10);
			r.BeginPhase(Phase.Winding, 0);

			List<(string Name, double? Fraction)> events = sink.Events();
			await Assert.That(events.Count).IsEqualTo(2);
			await Assert.That(events[0].Name).IsEqualTo("narrow phase");
			await Assert.That(events[0].Fraction).IsEqualTo(0.0);
			await Assert.That(events[1].Name).IsEqualTo("winding");
			await Assert.That(events[1].Fraction).IsNull();
		}

		[Test]
		public async Task AdvanceIsThrottledAndFractionsStayInRange()
		{
			Sink sink = new Sink();
			ProgressReporter r = sink.Reporter();

			// 1000 items / 100 reports = one callback per 10 items.
			r.BeginPhase(Phase.Arrangements, 1000);
			for (int i = 0; i < 1000; i++)
			{
				r.Advance(1);
			}

			List<(string Name, double? Fraction)> events = sink.Events();

			// The begin plus one per step; nothing like 1000 callbacks.
			await Assert.That(events.Count).IsEqualTo(101);
			foreach ((string name, double? fraction) in events)
			{
				await Assert.That(name).IsEqualTo("arrangements");
				await Assert.That(fraction).IsNotNull();

				// Rust asserts `(0.0..=1.0).contains(&f)` — inclusive at both ends. Two
				// one-sided assertions rather than one boolean, so a failure names the
				// offending fraction instead of only reporting "false".
				double f = fraction!.Value;
				await Assert.That(f).IsGreaterThanOrEqualTo(0.0);
				await Assert.That(f).IsLessThanOrEqualTo(1.0);
			}

			await Assert.That(events[events.Count - 1].Fraction).IsEqualTo(1.0);
		}

		[Test]
		public async Task AnIndeterminatePhaseNeverReportsAFraction()
		{
			Sink sink = new Sink();
			ProgressReporter r = sink.Reporter();
			r.BeginPhase(Phase.Winding, 0);
			for (int i = 0; i < 10_000; i++)
			{
				r.Advance(1);
			}

			List<(string Name, double? Fraction)> events = sink.Events();
			await Assert.That(events.Count).IsEqualTo(1);
			await Assert.That(events[0].Name).IsEqualTo("winding");
			await Assert.That(events[0].Fraction).IsNull();
		}

		[Test]
		public async Task PhaseIdsRoundTrip()
		{
			foreach (Phase p in Phases.All)
			{
				await Assert.That(Phases.FromId(p.Id())).IsEqualTo(p);
			}

			await Assert.That(Phases.FromId((uint)Phases.All.Count)).IsNull();
		}

		/// <summary>
		/// Collects every callback a run emits — the port of the Rust test module's
		/// <c>Sink</c>, whose <c>Arc&lt;Mutex&lt;Vec&lt;Event&gt;&gt;&gt;</c> becomes a
		/// locked list (the reporter may call back from any thread).
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
