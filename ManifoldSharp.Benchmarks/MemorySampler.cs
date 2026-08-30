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

// MemorySampler.cs — the piece of examples/mem_profile.rs that has no direct C#
// counterpart, and the one place in this project where the port is an approximation
// rather than a translation. Worth stating precisely, because the peak-memory row of
// docs/BENCHMARKS.md rests on it.
//
// The Rust driver installs a `#[global_allocator]` that adds `layout.size()` to a
// CURRENT counter on every alloc and subtracts it on every dealloc, keeping a PEAK
// watermark with `fetch_max`. That is exact and event-driven: no allocation can happen
// between the counter update and the peak update, so no spike can be missed.
//
// .NET has no supported way to interpose on the GC's allocator (AllocateArray and the
// AllocateHandler ETW events are neither complete nor cheap), so the peak here is
// SAMPLED: a background thread polls at a fixed interval and keeps the maximum. What
// that buys and what it costs:
//
//   * `GC.GetTotalMemory(false)` maps to Rust's CURRENT, but is the *managed heap's*
//     current size and includes objects that are garbage but not yet collected. Rust's
//     number drops the instant a Vec is freed; this one drops at the next collection.
//     So C# CURRENT reads at or above the equivalent Rust CURRENT, never below.
//   * The peak is a max over samples, so a spike shorter than the poll interval is
//     invisible. Two mitigations: the interval is 1 ms against stages that run for
//     hundreds of ms, and every stage boundary takes a synchronous reading too, so the
//     watermark is at least as high as the endpoints.
//   * `Environment.WorkingSet` is process RSS, which is not a heap number at all — it
//     includes the runtime, the JIT's code heap and pages the GC has committed but not
//     handed out. It is here because it is the one number that IS directly comparable
//     across the two languages: `/usr/bin/time -l` reports the same quantity for the
//     Rust process. The heap numbers describe where the memory went; the working-set
//     number is what the machine actually paid.
//
// `GC.GetTotalAllocatedBytes(precise: true)` has no Rust counterpart at all — it is
// cumulative allocation over the process's life, never decreasing. It is reported
// because on a GC runtime it, not the live-set number, is what predicts collection
// cost; the Rust driver has no reason to track it because malloc/free has no such cost.

namespace ManifoldSharp.Benchmarks
{
	/// <summary>
	/// A polling stand-in for mem_profile.rs's counting global allocator: tracks the
	/// high-water mark of the managed heap and of the process working set.
	/// </summary>
	internal sealed class MemorySampler : IDisposable
	{
		/// <summary>
		/// Poll interval. 1 ms is well under the shortest boolean pipeline stage and its
		/// cost is a rounding error next to the work being measured; the value is a
		/// constant rather than a knob because changing it changes what "peak" means.
		/// </summary>
		private const int PollIntervalMilliseconds = 1;

		private readonly Thread thread;
		private readonly ManualResetEventSlim stop = new ManualResetEventSlim(false);

		private long peakHeapBytes;
		private long peakWorkingSetBytes;

		/// <summary>Start sampling immediately.</summary>
		public MemorySampler()
		{
			this.Observe();
			this.thread = new Thread(this.Poll)
			{
				IsBackground = true,
				Name = "manifold-mem-sampler",
			};
			this.thread.Start();
		}

		/// <summary>The highest process working set observed since construction.</summary>
		public long PeakWorkingSetBytes
		{
			get { return Interlocked.Read(ref this.peakWorkingSetBytes); }
		}

		/// <summary>
		/// The Rust driver's <c>mem_stats</c>: current managed heap bytes, and the peak
		/// observed since the previous call, resetting the watermark to the current value
		/// so each stage reports its own peak.
		/// </summary>
		/// <returns>Current and stage-peak managed heap bytes.</returns>
		public (long Current, long Peak) ReadAndResetHeapPeak()
		{
			this.Observe();
			long current = GC.GetTotalMemory(false);
			long peak = Interlocked.Exchange(ref this.peakHeapBytes, current);
			return (current, Math.Max(peak, current));
		}

		/// <summary>Stop the sampling thread.</summary>
		public void Dispose()
		{
			this.stop.Set();
			this.thread.Join();
			this.stop.Dispose();
		}

		private void Poll()
		{
			while (!this.stop.Wait(PollIntervalMilliseconds))
			{
				this.Observe();
			}

			this.Observe();
		}

		private void Observe()
		{
			Max(ref this.peakHeapBytes, GC.GetTotalMemory(false));
			Max(ref this.peakWorkingSetBytes, Environment.WorkingSet);
		}

		/// <summary>
		/// The counting allocator's <c>fetch_max</c>: raise a shared maximum without
		/// losing a concurrent raise. The sampler thread and the reporting thread both
		/// call this, so a plain compare-and-assign would drop updates.
		/// </summary>
		private static void Max(ref long target, long candidate)
		{
			long seen = Interlocked.Read(ref target);
			while (candidate > seen)
			{
				long actual = Interlocked.CompareExchange(ref target, candidate, seen);
				if (actual == seen)
				{
					return;
				}

				seen = actual;
			}
		}
	}
}
