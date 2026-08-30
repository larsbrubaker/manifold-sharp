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

// MemProfile.cs — port of manifold-rust examples/mem_profile.rs. That file's header:
//
//   Heap profiling driver for the boolean pipeline: runs one sphere-minus-sphere
//   boolean at the perf_test tessellation for round `i` (default 6 -> 2M input
//   tris) under a counting global allocator. With MANIFOLD_TIMING set, each
//   pipeline stage line also reports current heap and that stage's peak, via the
//   timing::set_mem_hook bridge. Used to chase the peak-memory gap vs the C++
//   reference (see manifold-rust's PORTING_PLAN.md, "Performance vs C++").
//
//   Run with: MANIFOLD_TIMING=1 cargo run --release --example mem_profile [round]
//
// Here: `MANIFOLD_TIMING=1 dotnet run -c Release --project ManifoldSharp.Benchmarks
// -- mem [round]`.
//
// The two stage points are the Rust's, unchanged: one reading after the two input
// spheres exist, one after the difference. What each reported number maps to — and
// which of them are translations and which approximations — is set out in
// MemorySampler.cs's header; the short version is that `current` maps to Rust's
// CURRENT with GC slack on top, `tail peak` is sampled rather than exact, `working
// set` is the cross-language comparable, and `allocated` has no Rust counterpart.
//
// Like the Rust, the report goes to stderr so that a run whose stdout is being diffed
// against the Rust example's stdout is not disturbed by it.

using System.Globalization;

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Benchmarks
{
	/// <summary>One sphere difference with per-stage heap reporting.</summary>
	internal static class MemProfile
	{
		/// <summary>Run the driver once.</summary>
		/// <param name="args">Optional single argument: the perf_test round to size at.</param>
		/// <returns>The timing sample and the memory samples.</returns>
		public static IReadOnlyList<Sample> Run(string[] args)
		{
			int round = DriverArgs.Int(args, 0, 6);

			using MemorySampler sampler = new MemorySampler();
			Timing.SetMemHook(sampler.ReadAndResetHeapPeak);

			long allocatedAtStart = GC.GetTotalAllocatedBytes(true);

			Manifold sphere = Manifold.Sphere(1.0, (8 << round) * 4);
			Manifold sphere2 = sphere.Translate(Vec3.Splat(0.5));
			(long current, _) = sampler.ReadAndResetHeapPeak();
			Console.Error.WriteLine(string.Create(
				CultureInfo.InvariantCulture,
				$"inputs ready: nTri = {sphere.NumTri()} each, current = {Mb(current):F1} MB"));

			long start = System.Diagnostics.Stopwatch.GetTimestamp();
			Manifold diff = sphere.Difference(sphere2);
			double elapsed = DriverArgs.SecondsSince(start);

			(long doneCurrent, long peak) = sampler.ReadAndResetHeapPeak();
			int nTriOut = diff.NumTri();
			Console.Error.WriteLine(string.Create(
				CultureInfo.InvariantCulture,
				$"done: nTri(out) = {nTriOut}, current = {Mb(doneCurrent):F1} MB, tail peak = {Mb(peak):F1} MB"));

			// No Rust counterpart for either line: the first is cumulative allocation
			// (what a GC charges for, where malloc/free charges per call), the second is
			// process RSS (what `/usr/bin/time -l` reports for the Rust binary, and so the
			// only memory number the two languages can be compared on directly).
			long allocated = GC.GetTotalAllocatedBytes(true) - allocatedAtStart;
			Console.Error.WriteLine(string.Create(
				CultureInfo.InvariantCulture,
				$"c#: allocated = {Mb(allocated):F1} MB total, peak working set = {Mb(sampler.PeakWorkingSetBytes):F1} MB, time = {elapsed} sec"));

			GC.KeepAlive(diff);

			return new[]
			{
				new Sample($"mem/round={round} time", elapsed, "sec"),
				new Sample($"mem/round={round} peak heap", Mb(peak), "MB"),
				new Sample($"mem/round={round} peak working set", Mb(sampler.PeakWorkingSetBytes), "MB"),
				new Sample($"mem/round={round} allocated", Mb(allocated), "MB"),
			};
		}

		/// <summary>The Rust driver's <c>mb</c> helper, same divisor.</summary>
		/// <param name="bytes">A byte count.</param>
		/// <returns>The same quantity in mebibytes.</returns>
		private static double Mb(long bytes)
		{
			return bytes / 1048576.0;
		}
	}
}
