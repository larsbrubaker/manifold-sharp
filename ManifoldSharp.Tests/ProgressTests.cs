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
// All eight of progress_tests.rs are here. The last three arrived with Phase 10's
// robust engine, which is what they sweep:
// RobustBooleanReportsMonotonicPhasesWithValidFractions,
// AReporterDoesNotChangeTheResult, and ReporterOverhead — the module's one
// #[ignore]d measurement fixture, which keeps its ignore as
// [Skip("measurement fixture; run explicitly with --ignored --nocapture")].
// The `phase_id(name)` helper those three share came with them: it looks a phase
// up by the string the robust pipeline emits, and until that pipeline existed
// there was exactly one such string.

using System.Diagnostics;

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
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
		public async Task RobustBooleanReportsMonotonicPhasesWithValidFractions()
		{
			(Manifold A, Manifold B) pair = Spheres();
			Sink sink = new Sink();
			ProgressReporter reporter = sink.Reporter();
			Manifold outManifold = pair.A.BooleanWithEngineAndProgress(
				pair.B,
				OpType.Add,
				BooleanEngine.Robust,
				null,
				reporter);
			await Assert.That(outManifold.Volume()).IsGreaterThan(0.0);

			List<(string Name, double? Fraction)> events = sink.Events();
			await Assert.That(events.Count).IsGreaterThan(0)
				.Because("a robust boolean must report something");
			uint last = 0;
			List<string> seen = new List<string>();
			foreach ((string Name, double? Fraction) e in events)
			{
				uint id = PhaseId(e.Name);
				await Assert.That(id).IsGreaterThanOrEqualTo(last)
					.Because($"phase \"{e.Name}\" ({id}) went backwards from {last}");
				if (id != last || seen.Count == 0)
				{
					seen.Add(e.Name);
				}

				last = id;
				if (e.Fraction != null)
				{
					double f = e.Fraction.Value;
					await Assert.That(f).IsGreaterThanOrEqualTo(0.0)
						.Because($"fraction {f} out of range");
					await Assert.That(f).IsLessThanOrEqualTo(1.0)
						.Because($"fraction {f} out of range");
				}
			}

			// Every robust phase should appear for an input that actually intersects.
			foreach (string expected in new[]
			{
				"narrow phase",
				"self intersections",
				"candidate points",
				"registries",
				"arrangements",
				"cells",
				"winding",
				"assemble",
			})
			{
				await Assert.That(seen.Contains(expected)).IsTrue()
					.Because($"phase \"{expected}\" never reported (saw {string.Join(", ", seen)})");
			}
		}

		/// <summary>The whole point of the "reporting cannot perturb the computation" rule.</summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task AReporterDoesNotChangeTheResult()
		{
			static MeshGL Mesh(Manifold m) => m.GetMeshGL(-1);
			(Manifold A, Manifold B) pair = Spheres();
			foreach (OpType op in new[] { OpType.Add, OpType.Subtract, OpType.Intersect })
			{
				foreach (BooleanEngine engine in new[]
				{
					BooleanEngine.Exact,
					BooleanEngine.Robust,
					BooleanEngine.Auto,
				})
				{
					Manifold plain = pair.A.BooleanWithEngine(pair.B, op, engine);
					Sink sink = new Sink();
					ProgressReporter reporter = sink.Reporter();
					Manifold watched = pair.A.BooleanWithEngineAndProgress(
						pair.B, op, engine, null, reporter);
					MeshGL p = Mesh(plain);
					MeshGL w = Mesh(watched);
					await Assert.That(w.TriVerts)
						.IsEquivalentTo(p.TriVerts, CollectionOrdering.Matching)
						.Because($"{op}/{engine} topology changed");
					await Assert.That(w.VertProperties)
						.IsEquivalentTo(p.VertProperties, CollectionOrdering.Matching)
						.Because($"{op}/{engine} positions changed");
					await Assert.That(watched.Status()).IsEqualTo(plain.Status());
					await Assert.That(sink.Events().Count).IsGreaterThan(0)
						.Because($"{engine} reported nothing");
				}
			}
		}

		/// <summary>
		/// Overhead fixture, not an assertion: wall time of the same robust boolean with no
		/// reporter, with a no-op reporter, and with a counting one.
		/// </summary>
		/// <remarks>
		/// Ignored because it is a measurement, not a pass/fail property — a loaded CI box
		/// would make any threshold flaky. Run it deliberately, in Release, with the
		/// <c>--treenode-filter</c> that names it.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		[Skip("measurement fixture; run explicitly with --ignored --nocapture")]
		public async Task ReporterOverhead()
		{
			Manifold a = Manifold.Sphere(1.0, 96);
			Manifold b = Manifold.Sphere(0.8, 96).Translate(new Vec3(0.4, 0.1, 0.2));
			const uint Runs = 9;

			double Once(ProgressReporter? reporter)
			{
				long start = Stopwatch.GetTimestamp();
				Manifold outManifold = a.BooleanWithEngineAndProgress(
					b, OpType.Add, BooleanEngine.Robust, null, reporter);
				if (outManifold.NumTri() <= 0)
				{
					throw new InvalidOperationException("robust boolean produced no triangles");
				}

				return Stopwatch.GetElapsedTime(start).TotalSeconds;
			}

			ProgressReporter noop = new ProgressReporter((_, _) => { });
			long calls = 0;
			ProgressReporter counting = new ProgressReporter((_, _) => Interlocked.Increment(ref calls));

			// Warm up, then interleave the three variants round-robin: machine drift
			// (turbo, other load) then hits all three equally instead of whichever
			// block ran during it, which is what made a blocked layout unusable here.
			Once(null);
			double none = 0.0;
			double withNoop = 0.0;
			double withCounting = 0.0;
			for (int i = 0; i < Runs; i++)
			{
				none += Once(null);
				withNoop += Once(noop);
				withCounting += Once(counting);
			}

			none /= Runs;
			withNoop /= Runs;
			withCounting /= Runs;

			string line = string.Format(
				System.Globalization.CultureInfo.InvariantCulture,
				"reporter overhead: none {0:F4}s, no-op reporter {1:F4}s ({2:+0.00;-0.00}%), "
				+ "counting reporter {3:F4}s ({4:+0.00;-0.00}%), {5} callbacks per run",
				none,
				withNoop,
				100.0 * (withNoop - none) / none,
				withCounting,
				100.0 * (withCounting - none) / none,
				Interlocked.Read(ref calls) / Runs);
			TestContext.Current?.OutputWriter.WriteLine(line);
			await Assert.That(line).IsNotNull();
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

		/// <summary>Phase id of a reported name, for the monotonicity check.</summary>
		/// <param name="name">The reported phase name.</param>
		/// <returns>Its id.</returns>
		/// <exception cref="InvalidOperationException">The name is not a declared phase.</exception>
		private static uint PhaseId(string name)
		{
			foreach (Phase p in Phases.All)
			{
				if (p.Name() == name)
				{
					return p.Id();
				}
			}

			throw new InvalidOperationException($"unknown phase name \"{name}\"");
		}

		/// <summary>
		/// A sphere with a doubled shell: self-intersecting, so it exercises every robust
		/// phase rather than the trivial disjoint fast paths.
		/// </summary>
		/// <returns>The two operands.</returns>
		private static (Manifold A, Manifold B) Spheres()
		{
			return (
				Manifold.Sphere(1.0, 24),
				Manifold.Sphere(0.8, 24).Translate(new Vec3(0.4, 0.1, 0.2)));
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
