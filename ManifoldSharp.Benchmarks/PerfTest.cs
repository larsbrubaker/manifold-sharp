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

// PerfTest.cs — port of manifold-rust examples/perf_test.rs. That file's header:
//
//   Rust port of cpp-reference/manifold/extras/perf_test.cpp.
//
//   Benchmarks a single sphere-minus-sphere boolean at doubling tessellation
//   levels, printing the same "nTri = N, time = S sec" lines as the C++ driver
//   so the two outputs can be diffed side by side. An optional CLI argument
//   caps the number of doubling rounds (C++ hard-codes 8).
//
//   Run with: cargo run --release --example perf_test [rounds]
//
// Here: `dotnet run -c Release --project ManifoldSharp.Benchmarks -- perf [rounds]`.
//
// The one translation with any thought in it is `std::hint::black_box`. Rust needs it
// because LTO can see that `diff` is unused and delete the boolean; C# cannot, because
// the JIT does not inline across the whole call graph of a boolean and — decisively —
// `diff` escapes into a live local that a later `GC.KeepAlive` reads, which the JIT is
// not allowed to reorder around. `GC.KeepAlive` is therefore the exact counterpart:
// the same "this value is still needed" barrier, expressed the way the CLR spells it.

using System.Globalization;

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Benchmarks
{
	/// <summary>Sphere-minus-sphere at doubling tessellation levels.</summary>
	internal static class PerfTest
	{
		/// <summary>Run the driver once.</summary>
		/// <param name="args">Optional single argument: the number of doubling rounds.</param>
		/// <returns>One timing sample per round, labelled by input triangle count.</returns>
		public static IReadOnlyList<Sample> Run(string[] args)
		{
			int rounds = DriverArgs.Int(args, 0, 8);

			List<Sample> samples = new List<Sample>();
			for (int i = 0; i < rounds; i++)
			{
				Manifold sphere = Manifold.Sphere(1.0, (8 << i) * 4);
				Manifold sphere2 = sphere.Translate(Vec3.Splat(0.5));
				long start = System.Diagnostics.Stopwatch.GetTimestamp();
				Manifold diff = sphere.Difference(sphere2);
				int nTriResult = diff.NumTri();
				double elapsed = DriverArgs.SecondsSince(start);

				// Keep result live so the boolean cannot be optimized away.
				GC.KeepAlive(diff);
				GC.KeepAlive(nTriResult);

				int nTri = sphere.NumTri();
				Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"nTri = {nTri}, time = {elapsed} sec"));
				samples.Add(new Sample($"perf/nTri={nTri}", elapsed, "sec"));
			}

			return samples;
		}
	}
}
