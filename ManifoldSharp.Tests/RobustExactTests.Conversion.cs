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

// RobustExactTests.Conversion.cs — the rational→f64 rounding tests from
// robust/exact/tests.rs. The class header is on RobustExactTests.cs.

using System.Numerics;

using TUnit.Core;

using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Tests
{
	public partial class RobustExactTests
	{
		/// <summary>
		/// The magnitude at which correctly rounded conversion overflows: the midpoint
		/// between double.MaxValue = 2^1024 − 2^971 and the next value the exponent
		/// range cannot hold, 2^1024. The tie goes UP because 2^1024's mantissa is even
		/// and MaxValue's is odd, so this value and everything above it is ±infinity.
		/// </summary>
		private static readonly BigRational OverflowMagnitude =
			Backend.RatFromInt((BigInteger.One << 1024) - (BigInteger.One << 970));

		/// <summary>
		/// Verifies the correctly-rounded property of <c>RatToF64</c> directly, in
		/// exact arithmetic: the result must be the nearest f64 to the exact value,
		/// with ties broken to the even mantissa. See the class header for why this
		/// replaces the Rust's backend oracle.
		/// </summary>
		private static void CheckCorrectlyRounded(in BigRational r, string what)
		{
			double ours = Rational.RatToF64(r);
			if (r.Abs() >= OverflowMagnitude)
			{
				double want = r.IsNegative ? double.NegativeInfinity : double.PositiveInfinity;
				AssertEq(ours, want, $"{what}: past the overflow threshold");
				return;
			}

			AssertTrue(double.IsFinite(ours), $"{what}: finite value rounded to {ours:e}");
			BigRational best = (r - Rational.Rat(ours)).Abs();
			CheckNeighborIsNoNearer(r, ours, best, Math.BitIncrement(ours), what);
			CheckNeighborIsNoNearer(r, ours, best, Math.BitDecrement(ours), what);
		}

		/// <summary>
		/// One half of the correctly-rounded property: an adjacent f64 must be strictly
		/// farther from the exact value, or — in the tie case, where it is exactly as
		/// far — the chosen result must be the one with the even mantissa.
		/// </summary>
		private static void CheckNeighborIsNoNearer(
			in BigRational r,
			double ours,
			in BigRational best,
			double neighbor,
			string what)
		{
			if (!double.IsFinite(neighbor))
			{
				// The overflow branch already covers that direction.
				return;
			}

			BigRational distance = (r - Rational.Rat(neighbor)).Abs();
			int cmp = best.CompareTo(distance);
			if (cmp > 0)
			{
				Assert.Fail($"{what}: {neighbor:e} is nearer than the returned {ours:e}");
			}

			if (cmp == 0 && (BitConverter.DoubleToInt64Bits(ours) & 1) != 0)
			{
				Assert.Fail($"{what}: tie between {ours:e} and {neighbor:e} broken to the odd mantissa");
			}
		}

		/// <summary>
		/// RatToF64 is hand-rolled (Backend.cs item 5) rather than delegated, so that a
		/// backend upgrade can never silently change output vertices. This pins it to
		/// the specification: every f64 boundary category plus thousands of random
		/// rationals with multi-word numerators and denominators must be the correctly
		/// rounded result, nearest with ties to even.
		/// </summary>
		[Test]
		public void RatToF64IsCorrectlyRounded()
		{
			// Every representable power of two, subnormals included, and their
			// negations and halves (the halves of the smallest are underflow ties).
			for (int e = -1074; e <= 1023; e++)
			{
				BigRational p = e >= 0
					? Backend.RatFromInt(BigInteger.One << e)
					: Backend.RatNew(BigInteger.One, BigInteger.One << (-e));
				CheckCorrectlyRounded(p, $"2^{e}");
				CheckCorrectlyRounded(-p, $"-2^{e}");
				CheckCorrectlyRounded(p / Backend.RatFromInt(2), $"2^{e}/2");
				CheckCorrectlyRounded(p * Backend.RatNew(3, 2), $"3*2^{e}/2");
			}

			// Signed zero, exact f64 values across all classes, and their neighbors.
			double[] flats =
			{
				0.0,
				NegativeZero,
				1.0,
				-1.0,
				0.1,
				Math.PI,
				double.MaxValue,
				double.MinValue,
				MinPositive,
				MinPositive / 4.0,
				5e-324,
				-5e-324,
				1e300,
				1e-300,
			};
			foreach (double v in flats)
			{
				CheckCorrectlyRounded(Rational.Rat(v), $"exact {v:e}");
			}

			// Ties at 1, at the subnormal boundary, and at the overflow boundary.
			BigRational halfUlp = Backend.RatNew(1, BigInteger.One << 53);
			CheckCorrectlyRounded(Rational.Rat(1.0) + halfUlp, "tie at 1 (down)");
			CheckCorrectlyRounded(Rational.Rat(1.0) + (halfUlp * Backend.RatFromInt(3)), "tie at 1 (up)");
			CheckCorrectlyRounded(Rational.Rat(5e-324) / Backend.RatFromInt(2), "underflow tie (+)");
			CheckCorrectlyRounded(Rational.Rat(-5e-324) / Backend.RatFromInt(2), "underflow tie (-)");
			CheckCorrectlyRounded(Rational.Rat(5e-324) * Backend.RatNew(1, 4), "quarter subnormal (+)");
			CheckCorrectlyRounded(Rational.Rat(-5e-324) * Backend.RatNew(1, 4), "quarter subnormal (-)");

			// 2^1024 is the overflow threshold; MAX + half an ulp is the tie onto it.
			BigRational two1024 = Backend.RatFromInt(BigInteger.One << 1024);
			CheckCorrectlyRounded(two1024, "2^1024");
			CheckCorrectlyRounded(-two1024, "-2^1024");
			BigRational maxHalfUlp = Backend.RatFromInt(BigInteger.One << 970);
			CheckCorrectlyRounded(Rational.Rat(double.MaxValue) + maxHalfUlp, "MAX + ulp/2 (tie up)");
			CheckCorrectlyRounded(
				Rational.Rat(double.MaxValue) + maxHalfUlp - Backend.RatNew(1, BigInteger.One << 64),
				"just below the overflow tie");

			// Random rationals with multi-word numerators and denominators, scaled
			// across the whole exponent range so subnormal, normal and overflow
			// rounding all get exercised.
			Lcg rng = new Lcg(0xC0FFEE_1234_5678);
			for (int i = 0; i < 4000; i++)
			{
				int nw = (int)((rng.NextU64() % 4) + 1);
				int dw = (int)((rng.NextU64() % 4) + 1);
				BigInteger n = BigWords(rng, nw);
				BigInteger d = BigWords(rng, dw);
				if (d.IsZero)
				{
					d = BigInteger.One;
				}

				if ((rng.NextU64() & 1) == 0)
				{
					n = -n;
				}

				// Shift the value into a random binade, both directions.
				int shift = (int)(rng.NextU64() % 1300);
				if ((rng.NextU64() & 1) == 0)
				{
					n <<= shift;
				}
				else
				{
					d <<= shift;
				}

				CheckCorrectlyRounded(Backend.RatNew(n, d), $"random #{i}");
			}
		}

		/// <summary>A random magnitude of <paramref name="words"/> 64-bit limbs.</summary>
		private static BigInteger BigWords(Lcg rng, int words)
		{
			BigInteger v = BigInteger.Zero;
			for (int i = 0; i < words; i++)
			{
				v = (v << 64) + rng.NextU64();
			}

			return v;
		}

		/// <summary>
		/// IntRatioToF64 shares RatToF64's rounding core but skips the gcd that
		/// building a BigRational would pay. It must therefore agree with RatToF64 on
		/// the same value — including when the (numerator, denominator) pair handed to
		/// it is UNREDUCED, which is the whole point: robust/arrangement.rs feeds it
		/// raw homogeneous cross products.
		/// </summary>
		[Test]
		public void IntRatioToF64AgreesWithRatToF64()
		{
			void Check(BigInteger n, BigInteger d, string what)
			{
				(double ours, bool exact) = Rational.IntRatioToF64(n, d);
				double theirs = Rational.RatToF64(Backend.RatNew(n, d));
				AssertBits(ours, theirs, $"{what}: int path vs rational path");

				// The exactness flag must mean what it says: the f64 round-trips to the
				// original value.
				if (exact)
				{
					AssertTrue(double.IsFinite(ours), $"{what}: exact but not finite");
					AssertEq(Rational.Rat(ours), Backend.RatNew(n, d), $"{what}: flagged exact but {ours:e} != n/d");
				}
			}

			// Zero numerator with every denominator sign, and the sign rules.
			foreach (long d in new long[] { 1, -1, 7, -7 })
			{
				(double v, bool e) = Rational.IntRatioToF64(BigInteger.Zero, d);
				AssertBits(v, 0.0, $"0/{d} must be +0.0");
				AssertTrue(e, $"0/{d} is exact");
			}

			foreach ((long n, long d) in new (long, long)[] { (1, 2), (-1, 2), (1, -2), (-1, -2), (3, 4), (-7, 8) })
			{
				Check(n, d, $"{n}/{d}");
			}

			// Unreduced pairs must round exactly like their reduced form.
			foreach (long k in new long[] { 2, 3, 5, 1 << 20, 1L << 40 })
			{
				Check(3 * k, 4 * k, $"3k/4k k={k}");
				Check(-3 * k, 4 * k, $"-3k/4k k={k}");
			}

			// Inexact values: 1/3 rounds, and must not be flagged exact.
			(double third, bool thirdExact) = Rational.IntRatioToF64(1, 3);
			AssertEq(third, 1.0 / 3.0, "1/3");
			AssertTrue(!thirdExact, "1/3 is not representable");

			// 2^-1074 is exact; half of it underflows to a tie with zero.
			(double sub, bool subExact) = Rational.IntRatioToF64(BigInteger.One, BigInteger.One << 1074);
			AssertEq(sub, 5e-324, "2^-1074");
			AssertTrue(subExact, "2^-1074 is exact");
			(double under, bool underExact) = Rational.IntRatioToF64(BigInteger.One, BigInteger.One << 1075);
			AssertEq(under, 0.0, "2^-1075");
			AssertTrue(!underExact, "underflow is never exact");

			// Overflow in both signs, never exact.
			(double over, bool overExact) = Rational.IntRatioToF64(BigInteger.One << 1024, BigInteger.One);
			AssertEq(over, double.PositiveInfinity, "2^1024");
			AssertTrue(!overExact, "overflow is never exact");
			(double nover, bool _) = Rational.IntRatioToF64(-(BigInteger.One << 1024), BigInteger.One);
			AssertEq(nover, double.NegativeInfinity, "-2^1024");

			// Huge multi-word magnitudes across the exponent range, both signs, with
			// deliberately unreduced pairs (a shared random factor on both sides).
			Lcg rng = new Lcg(0x5EED_F00D_2024);
			for (int i = 0; i < 4000; i++)
			{
				int nw = (int)((rng.NextU64() % 4) + 1);
				int dw = (int)((rng.NextU64() % 4) + 1);
				BigInteger n = BigWords(rng, nw);
				BigInteger d = BigWords(rng, dw);
				if (d.IsZero)
				{
					d = BigInteger.One;
				}

				if ((rng.NextU64() & 1) == 0)
				{
					n = -n;
				}

				if ((rng.NextU64() & 1) == 0)
				{
					d = -d;
				}

				int shift = (int)(rng.NextU64() % 1300);
				if ((rng.NextU64() & 1) == 0)
				{
					n <<= shift;
				}
				else
				{
					d <<= shift;
				}

				// Common factor: same value, different representation.
				BigInteger f = rng.NextU64() | 1;
				Check(n, d, $"random #{i}");
				Check(n * f, d * f, $"random #{i} unreduced");
			}
		}

		/// <summary>
		/// The soundness claim behind the arrangement's gcd-free translated filter
		/// inputs: computing the translated coordinate from homogeneous cross products
		/// and rounding once produces the BIT-IDENTICAL f64 to subtracting exactly in
		/// BigRational and rounding that. Both round the same exact value, and
		/// correctly rounded conversion is a function of the value alone.
		/// </summary>
		[Test]
		public void HomogeneousTranslationRoundsIdenticallyToRationalSubtraction()
		{
			Lcg rng = new Lcg(0xA11CE_777);

			// A mix of plain f64 coordinates (exact dyadics, like mesh vertices) and
			// constructed fractions with large numerators and denominators, like
			// intersection points.
			BigRational Coord()
			{
				switch (rng.NextU64() % 3)
				{
					case 0:
						// Far-from-origin dyadics, the case the translation lever targets.
						return Rational.Rat(((double)(rng.NextU64() % (1UL << 40)) * 0.5) - 5.0e11);
					case 1:
						return Rational.Rat(((double)(rng.NextU64() % 2_000_000) * 1e-3) - 1000.0);
					default:
						BigInteger n = (BigInteger)rng.NextU64() * (BigInteger)(rng.NextU64() | 1);
						BigInteger d = (BigInteger)(rng.NextU64() | 1) + BigInteger.One;
						return Backend.RatNew((rng.NextU64() & 1) == 0 ? -n : n, d);
				}
			}

			// Exactly robust/arrangement.rs's `translated_coord`: the unreduced fraction
			// (pn·od − on·pd) / (pd·od), no BigRational anywhere.
			double TranslatedCoord(in BigRational pc, in BigRational oc)
			{
				BigInteger pn = Backend.Numer(pc);
				BigInteger pd = Backend.Denom(pc);
				BigInteger on = Backend.Numer(oc);
				BigInteger od = Backend.Denom(oc);
				BigInteger num = Backend.MulIntUint(pn, od) - Backend.MulIntUint(on, pd);
				BigInteger den = Backend.IntFromUint(Backend.MulUint(pd, od));
				return Rational.IntRatioToF64(num, den).Value;
			}

			for (int i = 0; i < 3000; i++)
			{
				R2 o = new R2(Coord(), Coord());
				R2 p = new R2(Coord(), Coord());
				double hx = TranslatedCoord(p.X, o.X);
				double hy = TranslatedCoord(p.Y, o.Y);

				// Also checked through the homogeneous form, which is the same value with
				// wider operands.
				Homog2 ho = Predicates.Homog2Of(o);
				Homog2 hp = Predicates.Homog2Of(p);
				BigInteger hden = hp.W * ho.W;
				double gx = Rational.IntRatioToF64((hp.X * ho.W) - (ho.X * hp.W), hden).Value;
				double gy = Rational.IntRatioToF64((hp.Y * ho.W) - (ho.Y * hp.W), hden).Value;
				AssertBits(gx, hx, $"#{i} x: homogeneous form differs");
				AssertBits(gy, hy, $"#{i} y: homogeneous form differs");

				// Reference: exact rational subtraction, then one rounding.
				R2 t = p.Sub(o);
				double rx = Rational.RatToF64(t.X);
				double ry = Rational.RatToF64(t.Y);
				AssertBits(hx, rx, $"#{i} x");
				AssertBits(hy, ry, $"#{i} y");
			}
		}
	}
}
