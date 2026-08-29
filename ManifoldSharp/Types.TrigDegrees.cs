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

// Types.TrigDegrees.cs — the sind/cosd half of types.rs, split out of Types.cs
// for the 800-line cap. Continues the same `Types` static class; the module
// header for the whole types.rs port lives in Types.cs.
//
// Every transcendental here goes through DeterministicMath, never System.Math:
// these feed circle vertex positions (Quality.GetCircularSegments consumers —
// cylinder, sphere, revolve), so a platform libm difference would move real
// geometry.

namespace ManifoldSharp
{
	/// <content>
	/// The degree-argument trigonometry of <c>types.rs</c>, whose reduction is exact at
	/// multiples of 90.
	/// </content>
	public static partial class Types
	{
		/// <summary>
		/// Sine function where multiples of 90 degrees come out exact.
		/// </summary>
		/// <remarks>
		/// Matches C++ <c>sind</c> (common.h), which reduces the argument with
		/// <c>std::remquo(x, 90.0, &amp;quo)</c> — round-to-nearest (ties to even),
		/// remainder in [-45, 45]. A floor-based reduction (remainder in [0, 90)) is
		/// mathematically equal but differs by ~1 ULP for reduced arguments in (45, 90),
		/// which breaks bit-exactness with the C++ reference (e.g. cylinder circle
		/// vertices).
		/// </remarks>
		/// <param name="x">The angle in degrees.</param>
		/// <returns>The sine of <paramref name="x"/>.</returns>
		public static double Sind(double x)
		{
			if (!double.IsFinite(x))
			{
				// Rust's f64::NAN, the positive quiet NaN — never double.NaN, which
				// carries the sign bit. The constant is shared with DeterministicMath so
				// there is exactly one of it in the assembly.
				return DeterministicMath.PositiveQuietNaN;
			}

			if (x < 0.0)
			{
				return -Sind(-x);
			}

			// Reconstruct std::remquo(x, 90.0, &quo): quo = nearest integer to the
			// exact x/90 (ties to even), remainder computed exactly. Round the
			// computed quotient, then fix up the rare off-by-one where the rounded
			// double quotient disagrees with the exact one.
			//
			// MidpointRounding.ToEven is Rust's round_ties_even (and C#'s default for
			// this overload, but it is spelled out because f64::round — halves away from
			// zero — is the wrong one and is what the other Round call sites in this port
			// ask for). The bare cast is Rust's saturating `as i64` on this target: since
			// .NET 9 every floating-point to integer conversion saturates
			// platform-independently — over-range to the type's maximum, under-range to its
			// minimum, NaN to zero (dotnet/core/compatibility/jit/9.0/fp-to-integer) — so
			// an astronomically large x degrades the same way in both ports instead of
			// trapping, and no saturating helper is needed here. (Reaching that at all
			// would take |x| > 9.2e18 degrees.)
			long quo = (long)Math.Round(x / 90.0, MidpointRounding.ToEven);
			double r = x - ((double)quo * 90.0);
			if (r > 45.0)
			{
				quo += 1;
				r -= 90.0;
			}
			else if (r < -45.0)
			{
				quo -= 1;
				r += 90.0;
			}
			else if (r == 45.0 && quo % 2 != 0)
			{
				// Exact tie: remquo rounds the quotient to even.
				quo += 1;
				r = -45.0;
			}
			else if (r == -45.0 && quo % 2 != 0)
			{
				quo -= 1;
				r = 45.0;
			}

			switch (((quo % 4) + 4) % 4)
			{
				case 0:
					return DeterministicMath.Sin(Radians(r));
				case 1:
					return DeterministicMath.Cos(Radians(r));
				case 2:
					return -DeterministicMath.Sin(Radians(r));
				case 3:
					return -DeterministicMath.Cos(Radians(r));
				default:
					return 0.0;
			}
		}

		/// <summary>
		/// Cosine function where multiples of 90 degrees come out exact.
		/// </summary>
		/// <param name="x">The angle in degrees.</param>
		/// <returns>The cosine of <paramref name="x"/>.</returns>
		public static double Cosd(double x)
		{
			return Sind(x + 90.0);
		}
	}
}
