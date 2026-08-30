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

// RobustExactTests.IntPred.cs — port of the inline test module in
// robust/exact/intpred.rs. The class header is on RobustExactTests.cs.
//
// The scaling helpers' Option<[iN; N]> returns became "bool plus a caller-owned
// span" in the port, so the tests wrap them back into a nullable array — arrays
// and not stackalloc spans, because these tests are only checking budgets and
// the allocation is irrelevant next to the exact rational references they
// compare against.

using System.Numerics;

using TUnit.Core;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Tests
{
	public partial class RobustExactTests
	{
		/// <summary>
		/// Deterministic LCG so failures reproduce. (intpred.rs's own Rng, which
		/// differs from tests.rs's Lcg in how it derives coordinates.)
		/// </summary>
		private sealed class IntPredRng
		{
			private ulong state;

			public IntPredRng(ulong seed)
			{
				this.state = seed;
			}

			public ulong Next()
			{
				unchecked
				{
					this.state = (this.state * 6364136223846793005UL) + 1442695040888963407UL;
				}

				return this.state;
			}

			/// <summary>
			/// f64 in roughly [-2, 2] with f32-like granularity (the common case:
			/// coordinates originating from f32 MeshGL input).
			/// </summary>
			public double Coord()
			{
				return (double)unchecked((int)this.Next()) * PowI(2.0, -30);
			}

			/// <summary>f64 with a wild exponent, exercising the BigInteger path.</summary>
			public double Wild()
			{
				double m = (double)unchecked((int)this.Next());
				int e = (int)(this.Next() % 500) - 250;
				double v = m * PowI(2.0, e);
				return double.IsFinite(v) ? v : m;
			}
		}

		private static Sign O2Ref(Vec2 a, Vec2 b, Vec2 c)
		{
			return Predicates.Orient2dR(R2.FromVec2(a), R2.FromVec2(b), R2.FromVec2(c));
		}

		private static Sign O3Ref(Vec3 a, Vec3 b, Vec3 c, Vec3 d)
		{
			return Predicates.Orient3dR(
				R3.FromVec3(a),
				R3.FromVec3(b),
				R3.FromVec3(c),
				R3.FromVec3(d));
		}

		/// <summary>The Rust's <c>scaled_i64(...) -> Option&lt;[i64; N]&gt;</c>.</summary>
		private static long[]? ScaledI64Or(double[] vs, uint budget)
		{
			long[] result = new long[vs.Length];
			return IntPred.ScaledI64(vs, budget, result) ? result : null;
		}

		/// <summary>The Rust's <c>narrow3(...) -> Option&lt;[i64; 3]&gt;</c>.</summary>
		private static long[]? Narrow3Or(Int128[] vs, uint bits)
		{
			long[] result = new long[3];
			return IntPred.Narrow3(vs, bits, result) ? result : null;
		}

		private static void AssertScaled(long[]? actual, long[]? expected, string what)
		{
			if (expected == null)
			{
				AssertTrue(actual == null, $"{what}: expected the tier to decline");
				return;
			}

			AssertTrue(actual != null, $"{what}: the tier declined unexpectedly");
			for (int i = 0; i < expected.Length; i++)
			{
				AssertEq(actual![i], expected[i], $"{what}: element {i}");
			}
		}

		[Test]
		public void Orient3dIMatchesRationalGeneric()
		{
			IntPredRng rng = new IntPredRng(0x5eed);
			for (int i = 0; i < 2000; i++)
			{
				Vec3[] p = new Vec3[4];
				for (int j = 0; j < 4; j++)
				{
					p[j] = new Vec3(rng.Coord(), rng.Coord(), rng.Coord());
				}

				AssertEq(
					IntPred.Orient3dI(p[0], p[1], p[2], p[3]),
					O3Ref(p[0], p[1], p[2], p[3]),
					$"#{i}");
			}
		}

		[Test]
		public void Orient3dIMatchesRationalWildExponents()
		{
			IntPredRng rng = new IntPredRng(0xbad_cafe);
			for (int i = 0; i < 500; i++)
			{
				Vec3[] p = new Vec3[4];
				for (int j = 0; j < 4; j++)
				{
					p[j] = new Vec3(rng.Wild(), rng.Wild(), rng.Wild());
				}

				AssertEq(
					IntPred.Orient3dI(p[0], p[1], p[2], p[3]),
					O3Ref(p[0], p[1], p[2], p[3]),
					$"#{i}");
			}
		}

		[Test]
		public void Orient3dIExactZeros()
		{
			IntPredRng rng = new IntPredRng(0xc0_11a9e);
			for (int i = 0; i < 500; i++)
			{
				// d on the lattice plane through a with directions (b-a), (c-a):
				// small dyadic combination keeps every coordinate exact.
				Vec3 a = new Vec3(rng.Coord(), rng.Coord(), rng.Coord());
				Vec3 b = new Vec3(rng.Coord(), rng.Coord(), rng.Coord());
				Vec3 c = new Vec3(rng.Coord(), rng.Coord(), rng.Coord());
				double s = 0.25;
				double t = 0.5;
				Vec3 d = new Vec3(
					a.X + (s * (b.X - a.X)) + (t * (c.X - a.X)),
					a.Y + (s * (b.Y - a.Y)) + (t * (c.Y - a.Y)),
					a.Z + (s * (b.Z - a.Z)) + (t * (c.Z - a.Z)));

				// The combination is exact only when no rounding occurred; verify
				// against the rational reference either way.
				AssertEq(IntPred.Orient3dI(a, b, c, d), O3Ref(a, b, c, d), $"#{i}");
			}
		}

		[Test]
		public void Orient3dIDegenerateInputs()
		{
			Vec3 z = new Vec3(0.0, NegativeZero, 0.0);
			Vec3 a = new Vec3(1.0, 2.0, 3.0);
			AssertEq(IntPred.Orient3dI(z, z, a, a), Sign.Zero, "repeated points");
			AssertEq(IntPred.Orient3dI(a, a, a, a), Sign.Zero, "one point");
			Vec3 sub = new Vec3(MinPositive / 4.0, 1e300, -1e-300);
			AssertEq(IntPred.Orient3dI(z, a, sub, sub), Sign.Zero, "repeated wild point");
			Vec3 one = new Vec3(1.0, 1.0, 1.0);
			AssertEq(IntPred.Orient3dI(z, a, sub, one), O3Ref(z, a, sub, one), "wild magnitudes");
		}

		[Test]
		public void Orient2dIMatchesRational()
		{
			IntPredRng rng = new IntPredRng(0x2d2d);
			for (int i = 0; i < 2000; i++)
			{
				Vec2[] p = new Vec2[3];
				for (int j = 0; j < 3; j++)
				{
					p[j] = new Vec2(rng.Coord(), rng.Coord());
				}

				AssertEq(IntPred.Orient2dI(p[0], p[1], p[2]), O2Ref(p[0], p[1], p[2]), $"coord #{i}");
			}

			for (int i = 0; i < 500; i++)
			{
				Vec2[] p = new Vec2[3];
				for (int j = 0; j < 3; j++)
				{
					p[j] = new Vec2(rng.Wild(), rng.Wild());
				}

				AssertEq(IntPred.Orient2dI(p[0], p[1], p[2]), O2Ref(p[0], p[1], p[2]), $"wild #{i}");
			}

			// Exact collinear.
			AssertEq(
				IntPred.Orient2dI(new Vec2(0.0, 0.0), new Vec2(1.0, 1.0), new Vec2(0.5, 0.5)),
				Sign.Zero,
				"exactly collinear");
		}

		// ---- i64 tier ----

		[Test]
		public void ScaledI64BudgetBoundary()
		{
			// 2^30-1 is the largest magnitude a 30-bit budget admits; 2^30 is the
			// first that must decline.
			// (1.0 anchors the scale at 2^0; without it a lone power of two would
			// rescale to its odd mantissa and always fit.)
			long max = (1L << 30) - 1;
			AssertScaled(
				ScaledI64Or(new double[] { (double)max, 1.0 }, 30),
				new long[] { max, 1 },
				"2^30-1 fits");
			AssertScaled(
				ScaledI64Or(new double[] { (double)(1L << 30), 1.0 }, 30),
				null,
				"2^30 declines");

			// The budget applies after per-axis rescaling: 1.0 and 2^-30 share the
			// scale 2^-30, so 1.0 becomes 2^30 and must decline.
			AssertScaled(ScaledI64Or(new double[] { 1.0, PowI(2.0, -30) }, 30), null, "rescaled 1.0 declines");
			AssertScaled(
				ScaledI64Or(new double[] { 1.0, PowI(2.0, -29) }, 30),
				new long[] { 1 << 29, 1 },
				"rescaled 1.0 fits");

			// Zeros never constrain the scale, and negatives keep their sign.
			AssertScaled(ScaledI64Or(new double[] { 0.0, -3.0, 1e300 }, 30), null, "1e300 declines");
			AssertScaled(
				ScaledI64Or(new double[] { 0.0, -3.0, 6.0 }, 30),
				new long[] { 0, -3, 6 },
				"zeros and negatives");

			// Subnormals decompose exactly too.
			AssertScaled(
				ScaledI64Or(new double[] { MinPositive / 2.0, 0.0 }, 30),
				new long[] { 1, 0 },
				"subnormal");
		}

		/// <summary>
		/// Both predicates at the exact worst case their budget admits: the i64 tier
		/// must accept it and still produce the right sign.
		/// </summary>
		[Test]
		public void I64TierWorstCaseMagnitudes()
		{
			// orient2d: values ±(2^30-1) ⇒ |det| = 2·(2^31-2)^2 at the extreme.
			double m = (double)((1L << 30) - 1);
			Vec2 a = new Vec2(-m, m);
			Vec2 b = new Vec2(m, -m);
			Vec2 c = new Vec2(-m, -m);
			AssertEq(IntPred.Orient2dI(a, b, c), O2Ref(a, b, c), "worst case orient2d");
			AssertTrue(IntPred.Orient2dI(a, b, c) != Sign.Zero, "worst case orient2d is nonzero");
			AssertTrue(ScaledI64Or(new double[] { a.X, b.X, c.X }, 30) != null, "worst case fits the budget");

			// Just over the 30-bit budget the i64 tier must decline and the i128
			// tier must give the identical answer. (2^30+1, not 2^30: a lone power
			// of two would rescale to 1 and stay inside the budget.)
			double m2 = (double)((1L << 30) + 1);
			Vec2 a2 = new Vec2(-m2, m2);
			Vec2 b2 = new Vec2(m2, -m2);
			Vec2 c2 = new Vec2(-m2, -m2);
			AssertScaled(ScaledI64Or(new double[] { a2.X, b2.X, c2.X }, 30), null, "over budget declines");
			AssertEq(IntPred.Orient2dI(a2, b2, c2), O2Ref(a2, b2, c2), "over budget orient2d");
			AssertEq(IntPred.Orient2dI(a2, b2, c2), IntPred.Orient2dI(a, b, c), "the tier boundary is invisible");

			// orient3d: differences of ±(2^20-1) around the origin.
			double d = (double)((1L << 20) - 1);
			Vec3 p = new Vec3(0.0, 0.0, 0.0);
			Vec3 q = new Vec3(d, -d, d);
			Vec3 r = new Vec3(-d, d, d);
			Vec3 s = new Vec3(d, d, -d);
			AssertEq(IntPred.Orient3dI(p, q, r, s), O3Ref(p, q, r, s), "worst case orient3d");
			AssertTrue(IntPred.Orient3dI(p, q, r, s) != Sign.Zero, "worst case orient3d is nonzero");
			AssertTrue(
				Narrow3Or(new Int128[] { (Int128)d, -(Int128)d, (Int128)d }, 20) != null,
				"worst case narrows");

			// One unit more: differences of 2^20+1 exceed the budget, so the tier
			// declines, i128 takes over, and the sign is unchanged (the whole
			// configuration is just scaled by a positive factor).
			double e = (double)((1L << 20) + 1);
			Vec3 q2 = new Vec3(e, -e, e);
			Vec3 r2 = new Vec3(-e, e, e);
			Vec3 s2 = new Vec3(e, e, -e);
			Int128 ei = ((Int128)1 << 20) + 1;
			AssertScaled(Narrow3Or(new Int128[] { ei, -ei, ei }, 20), null, "over budget narrow3 declines");
			AssertEq(IntPred.Orient3dI(p, q2, r2, s2), O3Ref(p, q2, r2, s2), "over budget orient3d");
			AssertEq(
				IntPred.Orient3dI(p, q2, r2, s2),
				IntPred.Orient3dI(p, q, r, s),
				"the orient3d tier boundary is invisible");
		}

		/// <summary>
		/// The tier boundary must be invisible: sweeping magnitudes across it, the
		/// integer predicates must agree with the exact rational reference at every
		/// step (this is what catches an off-by-one in the budget derivation).
		/// </summary>
		[Test]
		public void I64TierBoundarySweepMatchesRational()
		{
			IntPredRng rng = new IntPredRng(0x64_64_64);
			for (int k = 28; k <= 32; k++)
			{
				double scale = PowI(2.0, k);
				for (int n = 0; n < 400; n++)
				{
					// Integer-valued coordinates straddling the 2^30 / 2^20 marks.
					double G()
					{
						return (double)(rng.Next() % (1UL << 21)) * (scale / 2097152.0);
					}

					Vec2[] p2 = new Vec2[3];
					for (int j = 0; j < 3; j++)
					{
						p2[j] = new Vec2(G(), G());
					}

					AssertEq(
						IntPred.Orient2dI(p2[0], p2[1], p2[2]),
						O2Ref(p2[0], p2[1], p2[2]),
						$"orient2d at 2^{k}");

					Vec3[] p3 = new Vec3[4];
					for (int j = 0; j < 4; j++)
					{
						p3[j] = new Vec3(G(), G(), G());
					}

					AssertEq(
						IntPred.Orient3dI(p3[0], p3[1], p3[2], p3[3]),
						O3Ref(p3[0], p3[1], p3[2], p3[3]),
						$"orient3d at 2^{k}");
				}
			}
		}

		/// <summary>
		/// Clustered points: the orient3d i64 tier keys on edge *differences*, so large
		/// absolute coordinates with small offsets must still be exact.
		/// </summary>
		[Test]
		public void I64TierClusteredPointsMatchesRational()
		{
			TierStats.Reset();
			IntPredRng rng = new IntPredRng(0xc1_05_7e);
			for (int i = 0; i < 2000; i++)
			{
				// A cluster whose offsets live on the same 2^-25 lattice as the base, so
				// the common scale is 2^-25 and only the spread (< 2^20 lattice steps)
				// decides whether the i64 tier applies.
				double lat = PowI(2.0, -25);
				double[] baseCoord =
				{
					(double)(rng.Next() % (1UL << 26)) * lat,
					(double)(rng.Next() % (1UL << 26)) * lat,
					(double)(rng.Next() % (1UL << 26)) * lat,
				};
				double Off()
				{
					return (double)(rng.Next() % (1UL << 19)) * lat;
				}

				Vec3[] p = new Vec3[4];
				for (int j = 0; j < 4; j++)
				{
					p[j] = new Vec3(baseCoord[0] + Off(), baseCoord[1] + Off(), baseCoord[2] + Off());
				}

				AssertEq(
					IntPred.Orient3dI(p[0], p[1], p[2], p[3]),
					O3Ref(p[0], p[1], p[2], p[3]),
					$"#{i}");
			}

			AssertTrue(
				TierStats.I64Hits() > 1000,
				$"clustered inputs bypassed the i64 tier ({TierStats.I64Hits()} hits)");
		}

		/// <summary>
		/// The i64 tier must agree with the BigInteger/rational ground truth on bulk
		/// random input that lands squarely inside its budget.
		/// </summary>
		[Test]
		public void I64TierBulkMatchesRational()
		{
			TierStats.Reset();
			IntPredRng rng = new IntPredRng(0x1_64_7e_12);

			// 30-bit integer-valued coordinates: exactly the orient2d budget.
			double G30()
			{
				return (double)((long)(rng.Next() % (1UL << 30)) - (1L << 29));
			}

			for (int i = 0; i < 3000; i++)
			{
				Vec2[] p = new Vec2[3];
				for (int j = 0; j < 3; j++)
				{
					p[j] = new Vec2(G30(), G30());
				}

				AssertEq(IntPred.Orient2dI(p[0], p[1], p[2]), O2Ref(p[0], p[1], p[2]), $"orient2d #{i}");
			}

			ulong hits2d = TierStats.I64Hits();
			AssertTrue(hits2d > 2500, $"orient2d i64 tier fired only {hits2d} times");

			// 20-bit integer-valued coordinates: exactly the orient3d budget.
			double G20()
			{
				return (double)((long)(rng.Next() % (1UL << 20)) - (1L << 19));
			}

			for (int i = 0; i < 3000; i++)
			{
				Vec3[] p = new Vec3[4];
				for (int j = 0; j < 4; j++)
				{
					p[j] = new Vec3(G20(), G20(), G20());
				}

				AssertEq(
					IntPred.Orient3dI(p[0], p[1], p[2], p[3]),
					O3Ref(p[0], p[1], p[2], p[3]),
					$"orient3d #{i}");
			}

			ulong hits3d = TierStats.I64Hits() - hits2d;
			AssertTrue(hits3d > 2500, $"orient3d i64 tier fired only {hits3d} times");
		}
	}
}
