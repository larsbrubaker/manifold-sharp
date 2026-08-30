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

// Program.cs — the driver selector for the benchmark console, the C# counterpart of
// manifold-rust's `cargo run --release --example <name> [arg]`. Rust gets one binary
// per example; C# gets one binary and a first argument naming the driver, so
//
//     cargo run --release --example perf_test 7
//
// is spelled
//
//     dotnet run -c Release --project ManifoldSharp.Benchmarks -- perf 7
//
// Every driver keeps the Rust example's exact output format, so the two can be
// diffed side by side the way the Rust examples diff against the C++ extras.
//
// Two things are added on top of the Rust shape, both for measurement discipline and
// neither changing what a single run does:
//
//   --reps N   run the driver N times in one process and print a best-of-N summary.
//              N=1 (the default) is exactly the Rust example's behaviour. The first
//              rep of a C# process pays JIT on the whole boolean pipeline, so a
//              best-of-3 is the honest number to put next to an AOT-compiled Rust
//              binary; the per-rep lines are printed too, so the JIT tax stays
//              visible rather than being averaged away.
//   `all`      run every driver at its default size, for one-command capture.
//
// The parallel switch is the environment, not an argument: MANIFOLD_PARALLEL=1 is
// this port's stand-in for the Rust `parallel` Cargo feature (see Par.cs), so the
// sequential and parallel columns of docs/BENCHMARKS.md are the same command run
// twice with and without it. Each run banner echoes the switch so a captured log
// says which column it belongs to.

using System.Globalization;

namespace ManifoldSharp.Benchmarks
{
	/// <summary>
	/// Entry point: parses the driver name, the driver's one optional size argument
	/// and <c>--reps</c>, then runs the driver and reports best-of-N.
	/// </summary>
	internal static class Program
	{
		/// <summary>Every driver, keyed by the name given on the command line.</summary>
		private static readonly (string Name, string Usage, Func<string[], IReadOnlyList<Sample>> Run)[] Drivers =
		{
			("perf", "perf [rounds=8]        sphere-minus-sphere at doubling tessellation (examples/perf_test.rs)", PerfTest.Run),
			("large-scene", "large-scene [n=20]     union of an n^3 grid of spheres (examples/large_scene_test.rs)", LargeSceneTest.Run),
			("mem", "mem [round=6]          one sphere difference with per-stage heap reporting (examples/mem_profile.rs)", MemProfile.Run),
			("menger", "menger [depth=4]       convex hull of a Menger sponge (Rust test cpp_hull_menger_sponge)", MengerBench.Run),
			("bracelet", "bracelet               MinGap of two stacked bracelets (Rust test cpp_properties_mingap_stretchy_bracelet)", BraceletBench.Run),
			("twins", "twins                  Generic_Twin_7081 union (Rust test cpp_generic_twin_7081)", TwinsBench.Run),
			("sdf-blobs", "sdf-blobs              metaball level set at edge length 0.05 (Rust test cpp_sdf_blobs)", SdfBlobsBench.Run),
			("robust", "robust [rx ry rz]      the same union through both engines (examples/robust_perf.rs)", RobustBench.Run),
		};

		private static int Main(string[] args)
		{
			if (args.Length == 0 || args[0] == "-h" || args[0] == "--help")
			{
				PrintUsage();
				return args.Length == 0 ? 1 : 0;
			}

			int reps = 1;
			List<string> positional = new List<string>();
			for (int i = 0; i < args.Length; i++)
			{
				if (args[i] == "--reps")
				{
					if (i + 1 >= args.Length || !int.TryParse(args[i + 1], NumberStyles.Integer, CultureInfo.InvariantCulture, out reps) || reps < 1)
					{
						Console.Error.WriteLine("--reps needs a positive integer.");
						return 1;
					}

					i++;
					continue;
				}

				positional.Add(args[i]);
			}

			if (positional.Count == 0)
			{
				Console.Error.WriteLine("No driver named.");
				PrintUsage();
				return 1;
			}

			string driverName = positional[0];
			string[] driverArgs = positional.Skip(1).ToArray();

			PrintBanner();

			if (driverName == "all")
			{
				foreach ((string name, _, Func<string[], IReadOnlyList<Sample>> run) in Drivers)
				{
					RunDriver(name, run, Array.Empty<string>(), reps);
				}

				return 0;
			}

			foreach ((string name, _, Func<string[], IReadOnlyList<Sample>> run) in Drivers)
			{
				if (name == driverName)
				{
					RunDriver(name, run, driverArgs, reps);
					return 0;
				}
			}

			Console.Error.WriteLine($"Unknown driver '{driverName}'.");
			PrintUsage();
			return 1;
		}

		/// <summary>
		/// Run one driver <paramref name="reps"/> times and print the best value seen for
		/// each sample it reports.
		/// </summary>
		/// <remarks>
		/// Best-of, not mean-of: the workload is deterministic, so every rep computes the
		/// identical result and the spread is pure machine noise (scheduling, thermal, the
		/// first rep's JIT). The minimum is the least-noisy estimator of the work itself,
		/// and it is what the Rust side is measured with too, so the ratio compares like
		/// with like. Memory samples are aggregated the same way for the same reason.
		/// </remarks>
		private static void RunDriver(string name, Func<string[], IReadOnlyList<Sample>> run, string[] driverArgs, int reps)
		{
			Console.WriteLine();
			Console.WriteLine($"=== {name} ===");

			List<string> order = new List<string>();
			Dictionary<string, Sample> best = new Dictionary<string, Sample>(StringComparer.Ordinal);

			for (int rep = 0; rep < reps; rep++)
			{
				if (reps > 1)
				{
					Console.WriteLine($"-- rep {rep + 1}/{reps} --");
				}

				foreach (Sample sample in run(driverArgs))
				{
					if (!best.TryGetValue(sample.Label, out Sample previous))
					{
						order.Add(sample.Label);
						best[sample.Label] = sample;
					}
					else if (sample.Value < previous.Value)
					{
						best[sample.Label] = sample;
					}
				}
			}

			if (reps > 1)
			{
				Console.WriteLine($"-- best of {reps} --");
				foreach (string label in order)
				{
					Sample sample = best[label];
					Console.WriteLine(
						string.Create(
							CultureInfo.InvariantCulture,
							$"{label,-40} {sample.Value,12:F4} {sample.Unit}"));
				}
			}
		}

		private static void PrintBanner()
		{
			Console.WriteLine(
				string.Create(
					CultureInfo.InvariantCulture,
					$"ManifoldSharp benchmarks | MANIFOLD_PARALLEL={(ManifoldParallel.Enabled ? "on" : "off")}"
					+ $" | processors={Environment.ProcessorCount}"
					+ $" | {System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription}"
					+ $" | server GC={System.Runtime.GCSettings.IsServerGC}"));
		}

		private static void PrintUsage()
		{
			Console.WriteLine("usage: ManifoldSharp.Benchmarks <driver> [size] [--reps N]");
			Console.WriteLine();
			Console.WriteLine("drivers:");
			foreach ((_, string usage, _) in Drivers)
			{
				Console.WriteLine("  " + usage);
			}

			Console.WriteLine("  all                    every driver above at its default size");
			Console.WriteLine();
			Console.WriteLine("environment:");
			Console.WriteLine("  MANIFOLD_PARALLEL=1    the port's stand-in for the Rust `parallel` Cargo feature");
			Console.WriteLine("  MANIFOLD_TIMING=1      per-stage pipeline timing (and, under `mem`, per-stage heap)");
		}
	}

	/// <summary>
	/// One reported number from a driver: a stable label, the value, and its unit
	/// (<c>sec</c> or <c>MB</c>). Drivers print their Rust-format line themselves and
	/// return these so <see cref="Program"/> can aggregate across reps.
	/// </summary>
	internal readonly struct Sample
	{
		/// <summary>Create a sample.</summary>
		/// <param name="label">A label stable across reps — it is the aggregation key.</param>
		/// <param name="value">The measured value.</param>
		/// <param name="unit">The unit, <c>sec</c> or <c>MB</c>.</param>
		public Sample(string label, double value, string unit)
		{
			this.Label = label;
			this.Value = value;
			this.Unit = unit;
		}

		/// <summary>The aggregation key; must not vary between reps.</summary>
		public string Label { get; }

		/// <summary>The measured value, in <see cref="Unit"/>.</summary>
		public double Value { get; }

		/// <summary>The unit of <see cref="Value"/>.</summary>
		public string Unit { get; }
	}
}
