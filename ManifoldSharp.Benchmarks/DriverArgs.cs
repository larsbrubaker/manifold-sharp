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

// DriverArgs.cs — the two things every driver needs and neither the Rust examples nor
// the BCL hand over ready-made:
//
//   * the Rust examples' argument idiom, `std::env::args().nth(k).and_then(|a|
//     a.parse().ok()).unwrap_or(default)` — note that it falls back to the default on a
//     *malformed* argument as well as a missing one, and this keeps that (a typo'd size
//     silently running the default size is the Rust behaviour, and a driver whose shape
//     differs from the Rust's is a driver whose numbers cannot be compared to it);
//
//   * elapsed seconds as an f64, the counterpart of `Instant::elapsed().as_secs_f64()`.
//     Stopwatch timestamps rather than DateTime: the same monotonic clock guarantee
//     Rust's Instant carries, where DateTime.UtcNow is wall time and can step.

using System.Globalization;

namespace ManifoldSharp.Benchmarks
{
	/// <summary>Argument parsing and elapsed-time helpers shared by every driver.</summary>
	internal static class DriverArgs
	{
		/// <summary>
		/// The Rust examples' <c>args().nth(index).and_then(parse).unwrap_or(fallback)</c>.
		/// </summary>
		/// <param name="args">The driver's positional arguments.</param>
		/// <param name="index">Which one to read.</param>
		/// <param name="fallback">The value to use when it is missing or unparseable.</param>
		/// <returns>The parsed value, or <paramref name="fallback"/>.</returns>
		public static int Int(string[] args, int index, int fallback)
		{
			if (args.Length <= index)
			{
				return fallback;
			}

			return int.TryParse(args[index], NumberStyles.Integer, CultureInfo.InvariantCulture, out int value)
				? value
				: fallback;
		}

		/// <summary>
		/// Seconds elapsed since a <see cref="System.Diagnostics.Stopwatch.GetTimestamp"/>
		/// reading — Rust's <c>Instant::elapsed().as_secs_f64()</c>.
		/// </summary>
		/// <param name="startTimestamp">The timestamp taken at the start of the region.</param>
		/// <returns>Elapsed seconds.</returns>
		public static double SecondsSince(long startTimestamp)
		{
			return (System.Diagnostics.Stopwatch.GetTimestamp() - startTimestamp)
				/ (double)System.Diagnostics.Stopwatch.Frequency;
		}
	}
}
