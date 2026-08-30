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

// Timing.cs — port of timing.rs — lightweight, env-gated phase timing for the
// boolean pipeline, mirroring the C++ MANIFOLD_TIMING instrumentation
// (Timer::Print in boolean3.cpp / boolean_result.cpp). Enabled by setting the
// MANIFOLD_TIMING environment variable to any non-empty value; otherwise every
// call is a no-op so release performance is unaffected. Used to compare
// per-stage wall-clock against the C++ reference when hunting performance gaps
// (see CLAUDE.md's trace-diffing verification net).
//
// Rust's `Instant` becomes a raw `System.Diagnostics.Stopwatch.GetTimestamp()`
// tick count, and `Option<Instant>` becomes `long?`, so a disabled timer is a
// null with no allocation behind it.

using System.Globalization;

namespace ManifoldSharp
{
	/// <summary>
	/// Env-gated stage timing, the port of Rust's <c>timing</c> module.
	/// </summary>
	public static class Timing
	{
		/// <summary>
		/// Optional memory reporter, registered by profiling harnesses (e.g. the
		/// mem_profile driver's counting allocator). Returns (current bytes, peak bytes
		/// since the previous call) — the implementation resets its peak watermark on
		/// read so each stage line reports that stage's own peak.
		/// </summary>
		/// <returns>Current and stage-peak bytes in use.</returns>
		public delegate (long Current, long Peak) MemHook();

		// Rust reads the environment once through a `OnceLock`. A static readonly field
		// is the C# equivalent: initialized once, thread-safely, on first use.
		private static readonly bool Enabled = ReadEnabled();

		private static MemHook? memHook;

		/// <summary>
		/// Whether MANIFOLD_TIMING was set when this process started. Read once, so it
		/// cannot be flipped mid-run.
		/// </summary>
		internal static bool IsEnabled
		{
			get { return Enabled; }
		}

		/// <summary>
		/// Registers the memory reporter. Like Rust's <c>OnceLock::set</c>, only the
		/// first registration takes; later calls are ignored rather than failing.
		/// </summary>
		/// <param name="hook">The reporter to install.</param>
		public static void SetMemHook(MemHook hook)
		{
			ArgumentNullException.ThrowIfNull(hook);

			Interlocked.CompareExchange(ref memHook, hook, null);
		}

		/// <summary>
		/// Start a stage timer. Returns null (and times nothing) unless the
		/// MANIFOLD_TIMING environment variable is set.
		/// </summary>
		/// <returns>A start timestamp, or null when timing is off.</returns>
		internal static long? Start()
		{
			return Enabled ? System.Diagnostics.Stopwatch.GetTimestamp() : null;
		}

		/// <summary>
		/// Print a counter/diagnostic line, gated on the same MANIFOLD_TIMING switch as
		/// the stage timers.
		/// </summary>
		/// <param name="label">The line to print.</param>
		internal static void PrintCount(string label)
		{
			if (Enabled)
			{
				Console.Error.WriteLine(label);
			}
		}

		/// <summary>
		/// Print the elapsed time for a stage started with <see cref="Start"/>, matching
		/// the C++ Timer::Print format ("label: N sec") on stderr. If a memory hook is
		/// registered, appends current/stage-peak heap use.
		/// </summary>
		/// <remarks>
		/// The two format strings are transcribed from timing.rs specifier for specifier —
		/// <c>"{}: {} sec"</c> and
		/// <c>"{}: {} sec, current = {:.1} MB, stage peak = {:.1} MB"</c> — because
		/// trace diffing against the Rust is one of the port's three verification nets
		/// (CLAUDE.md), and a gratuitous format difference is noise in the one tool
		/// meant to isolate real ones. Rust's <c>{}</c> for f64 and C#'s default double
		/// formatting are both shortest-round-trip, and both are culture-invariant here;
		/// <c>{:.1}</c> is <c>F1</c>. One residual difference, unfixable without
		/// hand-rolling a formatter and immaterial to a wall-clock line: C# switches to
		/// exponent notation below 1e-5 (<c>1.23E-06</c>) where Rust always expands
		/// (<c>0.00000123</c>). It shows up only on stages faster than 10 microseconds.
		/// </remarks>
		/// <param name="label">The stage name.</param>
		/// <param name="t0">The timestamp <see cref="Start"/> returned, or null.</param>
		internal static void Print(string label, long? t0)
		{
			if (t0 is null)
			{
				return;
			}

			Console.Error.WriteLine(FormatStageLine(label, ElapsedSecondsSince(t0.Value)));
		}

		/// <summary>
		/// Builds the line <see cref="Print"/> writes. Split from the write so the
		/// format is testable without redirecting the process's stderr out from under
		/// the test runner.
		/// </summary>
		/// <param name="label">The stage name.</param>
		/// <param name="seconds">Elapsed seconds for the stage.</param>
		/// <returns>The formatted line, without a trailing newline.</returns>
		internal static string FormatStageLine(string label, double seconds)
		{
			MemHook? hook = Volatile.Read(ref memHook);
			if (hook is not null)
			{
				(long current, long peak) = hook();
				return string.Create(
					CultureInfo.InvariantCulture,
					$"{label}: {seconds} sec, current = {current / 1048576.0:F1} MB, stage peak = {peak / 1048576.0:F1} MB");
			}

			return string.Create(CultureInfo.InvariantCulture, $"{label}: {seconds} sec");
		}

		/// <summary>
		/// The env-var decision, split out from the read so it is testable in a process
		/// that cannot change its own startup environment.
		/// </summary>
		/// <remarks>
		/// Rust: <c>env::var("MANIFOLD_TIMING").map_or(false, |v| !v.is_empty())</c>.
		/// Unset is off and set-but-empty is off, but <em>any</em> non-empty value is on,
		/// including <c>"0"</c> and <c>"false"</c>. That last part is exactly the thing a
		/// porter "improves" into a divergence, so a test pins it.
		/// </remarks>
		/// <param name="value">The raw environment variable value, or null if unset.</param>
		/// <returns>True when timing should be on.</returns>
		internal static bool IsEnabledValue(string? value)
		{
			return !string.IsNullOrEmpty(value);
		}

		private static bool ReadEnabled()
		{
			return IsEnabledValue(Environment.GetEnvironmentVariable("MANIFOLD_TIMING"));
		}

		private static double ElapsedSecondsSince(long t0)
		{
			return (System.Diagnostics.Stopwatch.GetTimestamp() - t0)
				/ (double)System.Diagnostics.Stopwatch.Frequency;
		}

		/// <summary>
		/// Platform-safe stopwatch for ALWAYS-ON aggregate instrumentation (hot-path
		/// counters that accumulate into atomics).
		/// </summary>
		/// <remarks>
		/// The Rust type exists because <c>std::time::Instant::now()</c> PANICS on
		/// wasm32-unknown-unknown ("time not implemented on this platform"), so any
		/// unconditional timing in library code had to go through it — on wasm it
		/// measured nothing and reported zero. .NET has no such hole:
		/// <c>System.Diagnostics.Stopwatch.GetTimestamp()</c> works on browser-wasm too,
		/// so this port measures on every platform. The type is kept so that call sites
		/// port across unchanged, and because "which timing is always-on" is worth
		/// keeping visible in the code.
		/// </remarks>
		internal readonly struct Stopwatch
		{
			private readonly long t0;

			private Stopwatch(long t0)
			{
				this.t0 = t0;
			}

			/// <summary>
			/// Starts a stopwatch. Unlike <see cref="Timing.Start"/> this is not gated on
			/// MANIFOLD_TIMING — it is for instrumentation that always runs.
			/// </summary>
			/// <returns>A running stopwatch.</returns>
			public static Stopwatch Start()
			{
				return new Stopwatch(System.Diagnostics.Stopwatch.GetTimestamp());
			}

			/// <summary>
			/// Nanoseconds since <see cref="Start"/>.
			/// </summary>
			/// <remarks>
			/// Integer math throughout, mirroring Rust's <c>as_nanos()</c>, which counts
			/// nanoseconds directly. Going via <see cref="ElapsedSecs"/> would round-trip
			/// the count through a double and lose exactness above 2^53 ns (about 104
			/// days, but also at every value in between, since seconds-then-scale is two
			/// roundings). The split into whole seconds plus remainder keeps the
			/// multiply from overflowing: the remainder is below Frequency, so
			/// <c>remainder * 1e9</c> tops out around 1e18 on a 1 GHz timer.
			/// </remarks>
			/// <returns>Elapsed nanoseconds.</returns>
			public long ElapsedNs()
			{
				long ticks = System.Diagnostics.Stopwatch.GetTimestamp() - this.t0;
				long frequency = System.Diagnostics.Stopwatch.Frequency;
				return ((ticks / frequency) * 1_000_000_000L)
					+ (((ticks % frequency) * 1_000_000_000L) / frequency);
			}

			/// <summary>
			/// Seconds since <see cref="Start"/>.
			/// </summary>
			/// <returns>Elapsed seconds.</returns>
			public double ElapsedSecs()
			{
				return ElapsedSecondsSince(this.t0);
			}
		}
	}
}
