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

// RobustExactTests.Approx.cs — port of the inline test module in
// robust/exact/approx.rs. The class header is on RobustExactTests.cs.

using System.Numerics;

using TUnit.Core;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Tests
{
	public partial class RobustExactTests
	{
		private static Vec2 Approx2(R2 p)
		{
			return new Vec2(Rational.RatToF64(p.X), Rational.RatToF64(p.Y));
		}

		private static Vec3 Approx3(R3 p)
		{
			return new Vec3(Rational.RatToF64(p.X), Rational.RatToF64(p.Y), Rational.RatToF64(p.Z));
		}

		/// <summary>
		/// A rational point near (x,y) with a huge denominator — its rounding is a
		/// genuine ε-perturbation, exercising the indirect-point case.
		/// </summary>
		private static R2 Wobble2(double x, double y, long k)
		{
			BigRational tiny = Backend.RatNew(k, BigInteger.Pow(3, 40));
			return new R2(
				Backend.RatFromF64(x)!.Value + tiny,
				Backend.RatFromF64(y)!.Value - tiny);
		}

		/// <summary>
		/// Exact f64 → exact rational (the filters' precondition holds by construction:
		/// the point IS its own approximation).
		/// </summary>
		private static R2 Ex2(Vec2 p)
		{
			return new R2(Backend.RatFromF64(p.X)!.Value, Backend.RatFromF64(p.Y)!.Value);
		}

		/// <summary>
		/// Iteration count for the differential sweeps; raise via
		/// MANIFOLD_FILTER_STRESS=2000000 to run the multi-million-case version.
		/// </summary>
		private static int StressN(int fallback)
		{
			string? v = Environment.GetEnvironmentVariable("MANIFOLD_FILTER_STRESS");
			return v != null && int.TryParse(v, out int parsed) ? parsed : fallback;
		}

		/// <summary>
		/// A point at <c>base + k·ulp</c> on both axes: consecutive f64 values around a
		/// far-from-origin center, so the coordinates are exact by construction while
		/// the differences are as tiny as f64 allows. This is the clustered arrangement
		/// geometry that makes the magnitude-based filter useless.
		/// </summary>
		private static Vec2 Clustered(Vec2 baseValue, double ulp, long k0, long k1)
		{
			return new Vec2(baseValue.X + ((double)k0 * ulp), baseValue.Y + ((double)k1 * ulp));
		}

		[Test]
		public void CertifiedSignsAgreeWithExact()
		{
			Lcg rng = new Lcg(0xFEEDFACE);
			int certified = 0;
			for (int i = 0; i < 4000; i++)
			{
				R2[] pts = new R2[4];
				for (int j = 0; j < 4; j++)
				{
					pts[j] = Wobble2(rng.NextF64(50.0), rng.NextF64(50.0), (i * 4) + j);
				}

				Vec2[] ap = new Vec2[4];
				for (int j = 0; j < 4; j++)
				{
					ap[j] = Approx2(pts[j]);
				}

				Sign? s = Approx.Orient2dA(ap[0], ap[1], ap[2]);
				if (s != null)
				{
					AssertEq(s.Value, Predicates.Orient2dR(pts[0], pts[1], pts[2]), $"orient2d #{i}");
					certified++;
				}

				Sign? ic = Approx.IncircleA(ap[0], ap[1], ap[2], ap[3]);
				if (ic != null)
				{
					AssertEq(
						ic.Value,
						Predicates.IncircleR(pts[0], pts[1], pts[2], pts[3]),
						$"incircle #{i}");
				}
			}

			AssertTrue(certified > 3900, $"filter should certify generic input ({certified}/4000)");
		}

		// ─── Exact-input filters ────────────────────────────────────────────────

		[Test]
		public void ExactFiltersNeverMiscertify()
		{
			Lcg rng = new Lcg(0x5EED_1234);

			// (base, ulp, spread) regimes: origin-local, far-and-clustered
			// (coordinates ~1e6 with spacing at the 2⁻³² ulp ≈ 2.3e-10), and
			// far-and-wide.
			(Vec2 Base, double Ulp, long Spread)[] regimes =
			{
				(new Vec2(0.0, 0.0), 1.0, 1L << 20),
				(new Vec2(1.0e6, -2.0e6), PowI(2.0, -32), 1000),
				(new Vec2(1.0e6, -2.0e6), PowI(2.0, -32), 4),
				(new Vec2(12345.0, 6789.0), PowI(2.0, -20), 1L << 16),
			};
			int n = StressN(20_000);
			int tightCertified = 0;
			int looseCertified = 0;
			for (int ri = 0; ri < regimes.Length; ri++)
			{
				(Vec2 baseValue, double ulp, long spread) = regimes[ri];
				for (int iter = 0; iter < n; iter++)
				{
					long Kk()
					{
						return (long)(rng.NextF64(1.0) * (double)spread);
					}

					Vec2[] p = new Vec2[4];
					for (int j = 0; j < 4; j++)
					{
						p[j] = Clustered(baseValue, ulp, Kk(), Kk());
					}

					R2[] r = new R2[4];
					for (int j = 0; j < 4; j++)
					{
						r[j] = Ex2(p[j]);
					}

					// Coincident points make the predicates degenerate; skip.
					bool coincident = false;
					for (int i2 = 0; i2 < 4 && !coincident; i2++)
					{
						for (int j = i2 + 1; j < 4; j++)
						{
							if (p[i2] == p[j])
							{
								coincident = true;
								break;
							}
						}
					}

					if (coincident)
					{
						continue;
					}

					Sign truth = Predicates.IncircleR(r[0], r[1], r[2], r[3]);
					Sign? tight = Approx.IncircleAExact(p[0], p[1], p[2], p[3]);
					if (tight != null)
					{
						AssertEq(
							tight.Value,
							truth,
							$"incircle_a_exact regime {ri} {Show(p[0])} {Show(p[1])} {Show(p[2])} {Show(p[3])}");
						tightCertified++;
					}

					if (Approx.IncircleA(p[0], p[1], p[2], p[3]) != null)
					{
						looseCertified++;
					}

					Sign otruth = Predicates.Orient2dR(r[0], r[1], r[2]);
					Sign? oTight = Approx.Orient2dAExact(p[0], p[1], p[2]);
					if (oTight != null)
					{
						AssertEq(
							oTight.Value,
							otruth,
							$"orient2d_a_exact regime {ri} {Show(p[0])} {Show(p[1])} {Show(p[2])}");
					}
				}
			}

			// The whole point of the tight filter: it certifies the clustered
			// far-from-origin cases the magnitude filter cannot.
			AssertTrue(
				tightCertified > looseCertified * 2,
				$"tight filter should dominate ({tightCertified} vs {looseCertified})");
		}

		/// <summary>
		/// Integer points on the circle of radius 65 centered at the origin — 65 = 5·13
		/// has two distinct Pythagorean representations, giving eight lattice points per
		/// quadrant-pair. Every coordinate is a small integer, so all four inputs are
		/// exact and the true incircle sign is Zero.
		/// </summary>
		private static readonly long[][] Circle65 =
		{
			new long[] { 65, 0 },
			new long[] { 63, 16 },
			new long[] { 60, 25 },
			new long[] { 52, 39 },
			new long[] { 39, 52 },
			new long[] { 25, 60 },
			new long[] { 16, 63 },
			new long[] { 0, 65 },
		};

		private static Vec2 FromLattice(long[] v)
		{
			return new Vec2((double)v[0], (double)v[1]);
		}

		/// <summary>
		/// A filter must NEVER certify a nonzero sign for exactly cocircular points.
		/// </summary>
		[Test]
		public void ExactIncircleDeclinesOnTrueCocircular()
		{
			Vec2[] pts = new Vec2[Circle65.Length];
			for (int i = 0; i < Circle65.Length; i++)
			{
				pts[i] = FromLattice(Circle65[i]);
			}

			int seen = 0;
			for (int i = 0; i < pts.Length; i++)
			{
				for (int j = 0; j < pts.Length; j++)
				{
					for (int k = 0; k < pts.Length; k++)
					{
						for (int l = 0; l < pts.Length; l++)
						{
							if (i == j || i == k || i == l || j == k || j == l || k == l)
							{
								continue;
							}

							Vec2 a = pts[i];
							Vec2 b = pts[j];
							Vec2 c = pts[k];
							Vec2 d = pts[l];
							Sign truth = Predicates.IncircleR(Ex2(a), Ex2(b), Ex2(c), Ex2(d));
							AssertEq(truth, Sign.Zero, "test setup: not cocircular");
							AssertTrue(
								Approx.IncircleAExact(a, b, c, d) == null,
								$"certified a sign for cocircular points {Show(a)} {Show(b)} {Show(c)} {Show(d)}");
							seen++;
						}
					}
				}
			}

			AssertTrue(seen > 1000, $"expected a full permutation sweep ({seen})");
		}

		/// <summary>
		/// Threshold sweep: walk <c>d</c> away from an exactly cocircular position in
		/// steps of 2⁻⁴⁰ (exact: |coords| &lt; 2¹³, so 53 bits still cover them) and
		/// check the whole transition. Certified signs must always match the exact
		/// predicate; the exact-zero crossing must be declined; and far enough out the
		/// filter must certify, or it would be useless.
		/// </summary>
		[Test]
		public void ExactIncircleThresholdSweep()
		{
			double step = PowI(2.0, -40);
			Vec2 a = FromLattice(Circle65[0]);
			Vec2 b = FromLattice(Circle65[3]);
			Vec2 c = FromLattice(Circle65[6]);
			Vec2 baseValue = FromLattice(Circle65[5]);
			bool declinedAtZero = false;
			for (int mag = 0; mag <= 24; mag++)
			{
				foreach (long sign in new long[] { -1, 1 })
				{
					long k = sign * (1L << mag) * (mag == 0 ? 0 : 1);
					Vec2 d = new Vec2(baseValue.X, baseValue.Y + ((double)k * step));
					Sign truth = Predicates.IncircleR(Ex2(a), Ex2(b), Ex2(c), Ex2(d));
					Sign? got = Approx.IncircleAExact(a, b, c, d);
					if (got != null)
					{
						AssertEq(got.Value, truth, $"k={k} d={Show(d)}");
					}
					else if (truth == Sign.Zero)
					{
						declinedAtZero = true;
					}

					if (truth == Sign.Zero)
					{
						AssertTrue(Approx.IncircleAExact(a, b, c, d) == null, $"cocircular k={k}");
					}

					// Well past the bound (|det| ≈ 10³·|k|·2⁻⁴⁰ vs a bound of ~7e-8) the
					// filter must decide.
					if (mag >= 12)
					{
						AssertTrue(
							Approx.IncircleAExact(a, b, c, d) != null,
							$"filter failed to certify a clearly nonzero case k={k}");
					}
				}
			}

			AssertTrue(declinedAtZero, "the cocircular case must be declined");
		}

		/// <summary>
		/// Random near-degenerate quadruples built by placing <c>d</c> at ±few ulp from
		/// a genuinely cocircular position on an integer-coordinate circle.
		/// </summary>
		[Test]
		public void ExactIncircleAdversarialNearCocircular()
		{
			Lcg rng = new Lcg(0xC0FFEE_11);
			for (int iter = 0; iter < StressN(5_000); iter++)
			{
				long[] Pick()
				{
					int i = Math.Min((int)(Math.Abs(rng.NextF64(1.0)) * 8.0), 7);
					return new long[] { Circle65[i][0], Circle65[i][1] };
				}

				long[] p = Pick();
				long[] q = Pick();
				long[] r = Pick();
				long[] s = Pick();

				// Distinct picks only.
				if (SameLattice(p, q) || SameLattice(p, r) || SameLattice(p, s)
					|| SameLattice(q, r) || SameLattice(q, s) || SameLattice(r, s))
				{
					continue;
				}

				// Sign flips keep them on the same circle (radius 65).
				foreach (long[] v in new long[][] { p, q, r, s })
				{
					if (rng.NextF64(1.0) < 0.0)
					{
						v[0] = -v[0];
					}

					if (rng.NextF64(1.0) < 0.0)
					{
						v[1] = -v[1];
					}
				}

				Vec2 a = FromLattice(p);
				Vec2 b = FromLattice(q);
				Vec2 c = FromLattice(r);
				long nudge = (long)(rng.NextF64(1.0) * 4.0);
				Vec2 sv = FromLattice(s);
				Vec2 d = new Vec2(sv.X, sv.Y + ((double)nudge * 2.220446049250313E-16 * 64.0));
				R2 ra = Ex2(a);
				R2 rb = Ex2(b);
				R2 rc = Ex2(c);
				R2 rd = Ex2(d);
				if (a == b || a == c || a == d || b == c || b == d || c == d)
				{
					continue;
				}

				Sign truth = Predicates.IncircleR(ra, rb, rc, rd);
				Sign? sgn = Approx.IncircleAExact(a, b, c, d);
				if (sgn != null)
				{
					AssertEq(sgn.Value, truth, $"{Show(a)} {Show(b)} {Show(c)} {Show(d)}");
				}

				Sign? osgn = Approx.Orient2dAExact(a, b, c);
				if (osgn != null)
				{
					AssertEq(osgn.Value, Predicates.Orient2dR(ra, rb, rc), "orient2d_a_exact");
				}
			}
		}

		private static bool SameLattice(long[] a, long[] b)
		{
			return a[0] == b[0] && a[1] == b[1];
		}

		[Test]
		public void Orient3dFilterAgreesWithExact()
		{
			Lcg rng = new Lcg(0xDEADBEA7);
			for (int i = 0; i < 3000; i++)
			{
				R3[] p = new R3[4];
				for (int j = 0; j < 4; j++)
				{
					p[j] = R3.FromVec3(new Vec3(rng.NextF64(20.0), rng.NextF64(20.0), rng.NextF64(20.0)));
				}

				Vec3[] ap = new Vec3[4];
				for (int j = 0; j < 4; j++)
				{
					ap[j] = Approx3(p[j]);
				}

				Sign? s = Approx.Orient3dA(ap[0], ap[1], ap[2], ap[3]);
				if (s != null)
				{
					AssertEq(s.Value, Predicates.Orient3dR(p[0], p[1], p[2], p[3]), $"orient3d #{i}");
				}
			}
		}

		[Test]
		public void NearDegenerateDefersToExact()
		{
			// Points a hair off a line: the filter must return null (uncertain), never a
			// wrong certified sign.
			Vec2 a = new Vec2(12.0, 12.0);
			Vec2 b = new Vec2(24.0, 24.0);
			for (long i = -8; i <= 8; i++)
			{
				R2 c = Wobble2(48.0, 48.0, i);
				R2 ra = R2.FromVec2(a);
				R2 rb = R2.FromVec2(b);
				Vec2 ac = Approx2(c);
				Sign? s = Approx.Orient2dA(a, b, ac);
				if (s != null)
				{
					AssertEq(s.Value, Predicates.Orient2dR(ra, rb, c), $"i={i}");
				}

				// null: escalation is the correct answer here.
			}
		}

		[Test]
		public void NotOnSegmentPrefilterIsSound()
		{
			Lcg rng = new Lcg(0xBAD5EED);
			R3 a = R3.FromVec3(new Vec3(0.0, 0.0, 0.0));
			R3 b = R3.FromVec3(new Vec3(10.0, 4.0, 2.0));
			Vec3 aa = Approx3(a);
			Vec3 ab = Approx3(b);
			int rejected = 0;
			for (int i = 0; i < 4000; i++)
			{
				R3 p = new R3(
					Backend.RatFromF64(rng.NextF64(12.0))!.Value,
					Backend.RatFromF64(rng.NextF64(12.0))!.Value,
					Backend.RatFromF64(rng.NextF64(12.0))!.Value);
				bool exact = Predicates.PointOnSegmentR(p, a, b);
				bool? filtered = Approx.NotOnSegmentA(Approx3(p), aa, ab);
				if (filtered == false)
				{
					AssertTrue(!exact, $"prefilter wrongly rejected an on-segment point #{i}");
					rejected++;
				}
				else if (filtered == true)
				{
					Assert.Fail("the prefilter must never certify an on-segment point");
				}
			}

			AssertTrue(rejected > 3900, $"prefilter should reject generic points ({rejected}/4000)");

			// And points genuinely on the segment always defer or agree.
			for (int k = 1; k < 20; k++)
			{
				BigRational t = Backend.RatNew(k, 21);
				R3 on = a.Add(b.Sub(a).Scale(t));
				AssertTrue(Predicates.PointOnSegmentR(on, a, b), $"k={k} is on the segment");
				AssertTrue(Approx.NotOnSegmentA(Approx3(on), aa, ab) != false, $"k={k}");
			}
		}
	}
}
