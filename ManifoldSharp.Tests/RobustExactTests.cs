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

// RobustExactTests.cs — port of robust/exact/tests.rs plus the inline test
// modules of approx.rs and intpred.rs: 36 tests over the exact-arithmetic tier.
// Same inputs, same seeds, same expected values, same order.
//
// The class is one partial class split across five files, mirroring the three
// Rust test modules:
//   RobustExactTests.cs               — shared helpers + rational→f64 rounding
//   RobustExactTests.Conversion.cs    — the rounding oracle and int_ratio_to_f64
//   RobustExactTests.Predicates.cs    — filtered predicates vs exact ground truth
//   RobustExactTests.Constructions.cs — the exact constructions
//   RobustExactTests.Approx.cs        — approx.rs's inline tests
//   RobustExactTests.IntPred.cs       — intpred.rs's inline tests
//
// ── Why these tests are synchronous ─────────────────────────────────────────
// Every Rust assert_eq! inside a differential loop becomes AssertEq (below),
// which throws through TUnit's synchronous Assert.Fail. Awaiting an assertion
// per iteration would dominate the runtime of loops that run thousands of
// times, and — decisively — an await can resume on a different thread, which
// would break the [ThreadStatic] tier counter the intpred tier-hit tests read.
// Strictness is unchanged: the test fails on the first mismatching iteration,
// exactly as the Rust does.
//
// ── The one test whose oracle could not be ported ───────────────────────────
// rat_to_f64_matches_the_backend_oracle pins the hand-written conversion
// against dashu's own correctly rounded RBig::to_f64. The BCL has no
// rational→double at all, so there is no second implementation to pin against;
// RatToF64IsCorrectlyRounded instead verifies the *specification* over the same
// inputs using exact BigRational arithmetic — that the result is the nearest
// f64 with ties to even. That is strictly stronger than agreeing with another
// implementation, and the value-level agreement with the Rust was verified
// separately by the differential harness this port was written against.

using System.Numerics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Tests
{
	public partial class RobustExactTests
	{
		/// <summary>The smallest positive normal f64 (Rust's f64::MIN_POSITIVE).</summary>
		private const double MinPositive = 2.2250738585072014E-308;

		/// <summary>
		/// Negative zero built from its bits. A `-0.0` literal is correct in C# today,
		/// but this port's rule for sign-bearing float constants is to spell out the
		/// bits (see the f64::NAN rule in CLAUDE.md).
		/// </summary>
		private static readonly double NegativeZero = BitConverter.Int64BitsToDouble(unchecked((long)0x8000_0000_0000_0000UL));

		/// <summary>
		/// Deterministic 64-bit LCG (Knuth MMIX constants) so tests need no rand
		/// dependency and reproduce exactly across runs and platforms — and identically
		/// to the Rust, which is what makes the differential harness meaningful.
		/// </summary>
		private sealed class Lcg
		{
			private ulong state;

			public Lcg(ulong seed)
			{
				this.state = seed;
			}

			public ulong NextU64()
			{
				// Rust's wrapping_mul/wrapping_add; C#'s default unchecked context wraps
				// the same way.
				unchecked
				{
					this.state = (this.state * 6364136223846793005UL) + 1442695040888963407UL;
				}

				return this.state;
			}

			/// <summary>Uniform f64 in [-scale, scale) with plenty of low-bit entropy.</summary>
			public double NextF64(double scale)
			{
				double u = (double)(this.NextU64() >> 11) / (double)(1UL << 53); // [0,1)
				return ((u * 2.0) - 1.0) * scale;
			}
		}

		/// <summary>
		/// Rust's <c>f64::powi</c>: repeated squaring (compiler-rt's __powidf2), NOT
		/// the correctly rounded libm <c>pow</c> that <see cref="Math.Pow"/> computes.
		/// The two agree on powers of two but not on, say, 10^-30, and these exponents
		/// choose the test inputs — so the port has to use the same one to compare
		/// against the Rust at all.
		/// </summary>
		private static double PowI(double a, int b)
		{
			bool recip = b < 0;
			double r = 1.0;
			while (true)
			{
				if ((b & 1) != 0)
				{
					r *= a;
				}

				b /= 2;
				if (b == 0)
				{
					break;
				}

				a *= a;
			}

			return recip ? 1.0 / r : r;
		}

		/// <summary>
		/// Rust's <c>assert_eq!</c>: synchronous and fail-fast. See the file header for
		/// why the ported loops do not await an assertion per iteration.
		/// </summary>
		private static void AssertEq<T>(T actual, T expected, string what)
		{
			if (!EqualityComparer<T>.Default.Equals(actual, expected))
			{
				Assert.Fail($"{what}: expected {expected}, got {actual}");
			}
		}

		/// <summary>Rust's <c>assert!</c>.</summary>
		private static void AssertTrue(bool condition, string what)
		{
			if (!condition)
			{
				Assert.Fail(what);
			}
		}

		/// <summary>
		/// Bit-for-bit f64 equality, for the ported <c>assert_eq!(a.to_bits(),
		/// b.to_bits())</c> sites. Distinguishes ±0.0, which is the point.
		/// </summary>
		private static void AssertBits(double actual, double expected, string what)
		{
			if (BitConverter.DoubleToInt64Bits(actual) != BitConverter.DoubleToInt64Bits(expected))
			{
				Assert.Fail($"{what}: expected {expected:e} ({BitConverter.DoubleToInt64Bits(expected):x}), "
					+ $"got {actual:e} ({BitConverter.DoubleToInt64Bits(actual):x})");
			}
		}

		private static R2 R2Of(double x, double y)
		{
			return R2.FromVec2(new Vec2(x, y));
		}

		private static R3 R3Of(double x, double y, double z)
		{
			return R3.FromVec3(new Vec3(x, y, z));
		}

		// ─── rat / RatToF64 ──────────────────────────────────────────────────────

		[Test]
		public void RoundTripIsBitExact()
		{
			// -0.0 is absent: rational zero is unsigned, so the sign bit of a
			// negative zero is (harmlessly) lost — asserted separately below.
			double[] values =
			{
				0.0,
				1.0,
				-1.0,
				0.1,
				-0.1,
				Math.PI,
				1e300,
				-1e300,
				1e-300,
				double.MaxValue,
				double.MinValue,
				MinPositive,       // smallest normal
				MinPositive / 4.0, // subnormal
				5e-324,            // smallest subnormal
				-5e-324,
				123456789.123456789,
			};

			foreach (double v in values)
			{
				double back = Rational.RatToF64(Rational.Rat(v));
				AssertBits(back, v, $"round trip failed for {v:e}");
			}

			AssertEq(Rational.RatToF64(Rational.Rat(NegativeZero)), 0.0, "negative zero");

			// And a pseudo-random sweep across magnitudes.
			Lcg rng = new Lcg(0x9E3779B97F4A7C15);
			for (int i = 0; i < 2000; i++)
			{
				double scale = PowI(10.0, (i % 61) - 30);
				double v = rng.NextF64(scale);
				double back = Rational.RatToF64(Rational.Rat(v));
				AssertBits(back, v, $"round trip failed for {v:e}");
			}
		}

		[Test]
		public void RoundingIsNearestTiesEven()
		{
			BigRational halfUlp = Backend.RatNew(1, BigInteger.One << 53);

			// 1 + 2^-53 is exactly halfway between 1 and 1+2^-52; even mantissa wins → 1.
			BigRational tieDown = Rational.Rat(1.0) + halfUlp;
			AssertEq(Rational.RatToF64(tieDown), 1.0, "tie at 1 rounds down");

			// 1 + 3·2^-53 is halfway between 1+2^-52 (odd mantissa) and 1+2^-51 (even) → up.
			BigRational tieUp = Rational.Rat(1.0) + (halfUlp * Backend.RatFromInt(3));
			AssertEq(Rational.RatToF64(tieUp), 1.0 + (2.0 * PowI(2.0, -52)), "tie at 1 rounds up");

			// Just above/below the midpoint round to the nearer neighbor.
			BigRational quarterUlp = halfUlp / Backend.RatFromInt(2);
			AssertEq(Rational.RatToF64(Rational.Rat(1.0) + quarterUlp), 1.0, "quarter ulp");
			AssertEq(
				Rational.RatToF64(Rational.Rat(1.0) + halfUlp + quarterUlp),
				1.0 + PowI(2.0, -52),
				"three quarter ulp");
		}

		[Test]
		public void RoundingHandlesOverflowAndSubnormals()
		{
			// 2 * MAX overflows to infinity; MAX itself survives.
			BigRational twoMax = Rational.Rat(double.MaxValue) * Backend.RatFromInt(2);
			AssertEq(Rational.RatToF64(twoMax), double.PositiveInfinity, "2*MAX");
			AssertEq(Rational.RatToF64(-twoMax), double.NegativeInfinity, "-2*MAX");

			// Half the smallest subnormal is a tie with 0 → even → 0.
			//
			// AssertBits, not AssertEq: underflow is the one place the sign can be lost
			// silently. `-0.0 == 0.0` is true in IEEE and EqualityComparer<double> agrees,
			// and the exact rational distance from +0.0 and from -0.0 is the same value,
			// so neither this assertion in its ported form nor the correctly-rounded
			// property check in RatToF64IsCorrectlyRounded can see a returned +0.0 where
			// the Rust returns -0.0. The negative mirror below is a hardening case with
			// no counterpart in the Rust suite; its expected value is read straight off
			// mag_ratio_to_f64, which returns `if neg { -0.0 } else { 0.0 }`.
			BigRational minSub = Rational.Rat(5e-324);
			BigRational halfMin = minSub / Backend.RatFromInt(2);
			AssertBits(Rational.RatToF64(halfMin), 0.0, "half the smallest subnormal");
			BigRational negHalfMin = Rational.Rat(-5e-324) / Backend.RatFromInt(2);
			AssertBits(Rational.RatToF64(negHalfMin), NegativeZero, "half the smallest negative subnormal");

			// Three quarters of the smallest subnormal rounds up to it.
			BigRational threeQ = minSub * Backend.RatNew(3, 4);
			AssertBits(Rational.RatToF64(threeQ), 5e-324, "three quarters of the smallest subnormal");

			// A value between two subnormals rounds to the nearer one.
			double sub = MinPositive / 8.0; // subnormal with headroom
			double next = BitConverter.Int64BitsToDouble(BitConverter.DoubleToInt64Bits(sub) + 1);
			BigRational midPlus = ((Rational.Rat(sub) + Rational.Rat(next)) * Backend.RatNew(1, 2))
				+ Backend.RatNew(1, BigInteger.One << 2000);
			AssertBits(Rational.RatToF64(midPlus), next, "just above a subnormal midpoint");
		}
	}
}
