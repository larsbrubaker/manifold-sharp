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

// Rational.cs — port of robust/exact/rational.rs. Exact rational points and
// correctly rounded rational→f64 conversion for the robust boolean engine.
//
// Input mesh vertices are finite f64 and convert to BigRational exactly
// (Rat, R3.FromVec3). Constructed points — plane/segment intersections built in
// Predicates.cs — stay rational through the whole pipeline; only output
// assembly rounds them back, via RatToF64, which rounds to the *nearest* f64
// (ties to even, subnormals and overflow handled). That single-rounding
// guarantee is what lets the robust engine's output vertices agree with the
// exact engine's to the last ulp on intersection points, and bit-for-bit on
// pass-through input vertices.
//
// The exact point types R2/R3 and the hash keys live in Rational.Points.cs.

using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace ManifoldSharp.Robust.Exact
{
	/// <summary>
	/// The free functions of rational.rs: exact f64→rational conversion and the
	/// engine's one correctly rounded rational→f64 rounding path.
	/// </summary>
	public static class Rational
	{
		/// <summary>
		/// Exact conversion of a finite f64. Every finite f64 is a dyadic rational, so
		/// this never loses information.
		/// </summary>
		/// <exception cref="InvalidOperationException">
		/// The value is NaN or infinite. Mesh import rejects non-finite vertices
		/// (Error.NonFiniteVertex) long before the robust engine runs, so a non-finite
		/// value here is an internal logic error, not bad user input.
		/// </exception>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BigRational Rat(double v)
		{
			BigRational? r = Backend.RatFromF64(v);
			if (r == null)
			{
				throw new InvalidOperationException("robust engine: coordinate must be finite");
			}

			return r.Value;
		}

		/// <summary>
		/// 2^e as f64, exact for the full representable range -1074..=1023 (subnormal
		/// powers of two included). Built by bit manipulation so no intermediate
		/// rounding can occur.
		/// </summary>
		private static double Pow2(long e)
		{
			Debug.Assert(e >= -1074 && e <= 1023, "pow2 exponent out of range");
			if (e >= -1022)
			{
				return BitConverter.Int64BitsToDouble((long)((ulong)(e + 1023) << 52));
			}

			return BitConverter.Int64BitsToDouble((long)(1UL << (int)(e + 1074)));
		}

		/// <summary>
		/// Rounds a rational to the nearest f64, ties to even — the correctly rounded
		/// result, identical to rounding the exact real value once. Values beyond f64
		/// range become ±infinity; values below half the smallest subnormal become
		/// (signed) zero.
		/// </summary>
		public static double RatToF64(in BigRational r)
		{
			if (Backend.RatIsZero(r))
			{
				return 0.0;
			}

			BigInteger nMag = Backend.NumerMag(r);
			return MagRatioToF64(nMag, Backend.Denom(r), Backend.RatIsNegative(r)).Value;
		}

		/// <summary>
		/// Correctly rounded f64 of the exact value <c>numer / denom</c>, built
		/// straight from big integers — no BigRational, hence no gcd reduction. Also
		/// reports whether the conversion was EXACT (the returned f64 equals
		/// <c>numer/denom</c> with no rounding at all).
		/// </summary>
		/// <remarks>
		/// The rounding core is shared with <see cref="RatToF64"/> and is value-based,
		/// not representation-based: it derives the binary exponent from bit lengths
		/// and rounds with one exact integer division, so an unreduced fraction and its
		/// reduced form produce the identical f64. That is what makes this a drop-in
		/// replacement for "subtract exactly in BigRational, then round" — the two paths
		/// round the same exact value and are therefore bit-identical.
		/// <para><paramref name="denom"/> must be nonzero.</para>
		/// </remarks>
		public static (double Value, bool Exact) IntRatioToF64(in BigInteger numer, in BigInteger denom)
		{
			Debug.Assert(!denom.IsZero, "IntRatioToF64: zero denominator");
			if (numer.IsZero)
			{
				// Exact: the value is zero, and +0.0 is what RatToF64 returns for a zero
				// rational whatever the denominator's sign.
				return (0.0, true);
			}

			bool neg = (numer.Sign < 0) != (denom.Sign < 0);
			return MagRatioToF64(Backend.IntMag(numer), Backend.IntMag(denom), neg);
		}

		/// <summary>
		/// <c>±n/d</c> from unsigned magnitudes (<paramref name="n"/>,
		/// <paramref name="d"/> both nonzero), correctly rounded to nearest with ties to
		/// even, plus an exactness flag. Values beyond f64 range become ±infinity;
		/// values below half the smallest subnormal become (signed) zero. Both of those,
		/// and every rounding, report false.
		/// </summary>
		private static (double Value, bool Exact) MagRatioToF64(BigInteger n, BigInteger d, bool neg)
		{
			// Exact floor exponent e: 2^e <= n/d < 2^(e+1).
			long e = Backend.UintBits(n) - Backend.UintBits(d);

			// The shift widths below are bit-length differences of values that already
			// live in memory, so they cannot exceed int range on any reachable input;
			// BigInteger's own shift operator takes an int.
			bool ge = e >= 0 ? n >= (d << (int)e) : (n << (int)(-e)) >= d;
			if (!ge)
			{
				e -= 1;
			}

			if (e > 1023)
			{
				return (neg ? double.NegativeInfinity : double.PositiveInfinity, false);
			}

			// Position of the result's least significant bit. Normal numbers carry
			// 53 bits ending at e-52; subnormals are cut off at 2^-1074.
			long lsb = Math.Max(e - 52, -1074);

			// m = round_nearest_even((n/d) / 2^lsb), computed with one exact integer
			// division plus a remainder comparison.
			BigInteger num;
			BigInteger den;
			if (lsb >= 0)
			{
				num = n;
				den = d << (int)lsb;
			}
			else
			{
				num = n << (int)(-lsb);
				den = d;
			}

			BigInteger q = num / den;
			BigInteger rem = num - (q * den);

			// q is the truncation of the value to a multiple of 2^lsb, so a zero
			// remainder means the value IS such a multiple: no rounding happens below
			// (both increments require a nonzero remainder) and the result is exact.
			bool exact = rem.IsZero;
			BigInteger m = q;
			BigInteger twiceRem = rem << 1;
			int cmp = twiceRem.CompareTo(den);
			if (cmp > 0)
			{
				m += 1;
			}
			else if (cmp == 0)
			{
				if (Backend.UintBit(m, 0))
				{
					m += 1;
				}
			}

			if (m.IsZero)
			{
				// n/d are nonzero, so a zero mantissa means the value underflowed to
				// (signed) zero — never exact.
				return (neg ? -0.0 : 0.0, false);
			}

			// Rounding up may have crossed into the next binade (m = 2^53) or past the
			// largest finite value (2^1024 -> infinity).
			if (Backend.UintBits(m) - 1 + lsb > 1023)
			{
				return (neg ? double.NegativeInfinity : double.PositiveInfinity, false);
			}

			// m <= 2^53, so both the ulong and the f64 conversion are exact, and
			// m * 2^lsb is representable by construction — the multiply is exact.
			double val = (double)(ulong)m * Pow2(lsb);
			return (neg ? -val : val, exact);
		}

		/// <summary>
		/// Field-wise equality of canonical R2 values — see <see cref="R3Eq"/>.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool R2Eq(R2 a, R2 b)
		{
			return Backend.RatEq(a.X, b.X) && Backend.RatEq(a.Y, b.Y);
		}

		/// <summary>
		/// Field-wise equality of canonical R3 values — value equality without a
		/// general comparison (which would be most expensive exactly when the values
		/// ARE equal).
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool R3Eq(R3 a, R3 b)
		{
			return Backend.RatEq(a.X, b.X) && Backend.RatEq(a.Y, b.Y) && Backend.RatEq(a.Z, b.Z);
		}
	}
}
