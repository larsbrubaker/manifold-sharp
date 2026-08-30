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
// The four throttle-arithmetic tests are here, and so is
// `the_exact_engine_reports_one_indeterminate_phase`, which Phase 6's façade
// unblocked: it builds its operands with `Manifold::cube` and reads `volume()`,
// and it was already named in the earlier DEFERRED table as the first of the
// four to become portable.
//
// DEFERRED, and why (greppable). The remaining three all sweep
// `BooleanEngine::Robust` or `Auto`, so Phase 10 blocks them — not Phase 6:
//
//   robust_boolean_reports_monotonic_phases_with_valid_fractions   Phase 10
//     (also needs the `spheres()` fixture, which IS portable now — two
//     subdivide-backed spheres — so only the engine is missing)
//   a_reporter_does_not_change_the_result                          Phase 10
//     Compares `get_mesh_gl(-1).tri_verts` / `.vert_properties` and `status()`,
//     never `volume()`, across every engine.
//   reporter_overhead                                             Phase 10
//     The one #[ignore]d measurement fixture in this module. When it is written
//     it keeps its ignore, as TUnit
//     [Skip("measurement fixture; run explicitly with --ignored --nocapture")].
//
// The `phase_id(name)` helper those three share is deferred with them: it exists
// to look a phase up by the string the robust pipeline emits, and with only the
// exact engine running there is exactly one such string.

using ManifoldSharp;
using ManifoldSharp.Linalg;

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


		[Test]
		public async Task TheExactEngineReportsOneIndeterminatePhase()
		{
			Sink sink = new Sink();
			ProgressReporter reporter = sink.Reporter();
			Manifold outManifold = Cube(0.0).BooleanWithEngineAndProgress(
				Cube(0.5),
				OpType.Add,
				BooleanEngine.Exact,
				null,
				reporter);
			await Assert.That(outManifold.Volume()).IsGreaterThan(0.0);

			List<(string Name, double? Fraction)> events = sink.Events();
			await Assert.That(events.Count).IsEqualTo(1);
			await Assert.That(events[0].Name).IsEqualTo("exact boolean");
			await Assert.That(events[0].Fraction).IsNull();
		}

		private static Manifold Cube(double offset)
		{
			return Manifold.Cube(Vec3.Splat(1.0), true).Translate(new Vec3(offset, 0.0, 0.0));
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
