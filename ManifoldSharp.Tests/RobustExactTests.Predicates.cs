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

// RobustExactTests.Predicates.cs — the "Predicates: filtered vs exact ground
// truth" section of robust/exact/tests.rs. The class header is on
// RobustExactTests.cs.

using TUnit.Core;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Tests
{
	public partial class RobustExactTests
	{
		private static string Show(Vec2 v)
		{
			return $"[{v.X}, {v.Y}]";
		}

		private static string Show(Vec3 v)
		{
			return $"[{v.X}, {v.Y}, {v.Z}]";
		}

		[Test]
		public void Orient2dMatchesExactOnRandomInput()
		{
			Lcg rng = new Lcg(42);
			for (int i = 0; i < 5000; i++)
			{
				Vec2 a = new Vec2(rng.NextF64(100.0), rng.NextF64(100.0));
				Vec2 b = new Vec2(rng.NextF64(100.0), rng.NextF64(100.0));
				Vec2 c = new Vec2(rng.NextF64(100.0), rng.NextF64(100.0));
				Sign exact = Predicates.Orient2dR(R2.FromVec2(a), R2.FromVec2(b), R2.FromVec2(c));
				AssertEq(Filtered.Orient2d(a, b, c), exact, $"a={Show(a)} b={Show(b)} c={Show(c)}");
			}
		}

		[Test]
		public void Orient2dAdversarialNearCollinear()
		{
			// Shewchuk's classic torture grid: points ulp-perturbed around the line
			// y = x, where naive float evaluation gets the sign wrong.
			double baseValue = 12.0;
			for (int i = 0; i < 32; i++)
			{
				for (int j = 0; j < 32; j++)
				{
					Vec2 a = new Vec2(
						baseValue + ((double)i * PowI(2.0, -52)),
						baseValue + ((double)j * PowI(2.0, -52)));
					Vec2 b = new Vec2(24.0, 24.0);
					Vec2 c = new Vec2(48.0, 48.0);
					Sign exact = Predicates.Orient2dR(R2.FromVec2(a), R2.FromVec2(b), R2.FromVec2(c));
					AssertEq(Filtered.Orient2d(a, b, c), exact, $"i={i} j={j}");
				}
			}

			// Exactly collinear must report Zero.
			AssertEq(
				Filtered.Orient2d(new Vec2(0.0, 0.0), new Vec2(1.0, 1.0), new Vec2(3.0, 3.0)),
				Sign.Zero,
				"exactly collinear");
		}

		[Test]
		public void Orient3dMatchesExactOnRandomInput()
		{
			Lcg rng = new Lcg(7);
			Vec3 P(Lcg r)
			{
				return new Vec3(r.NextF64(50.0), r.NextF64(50.0), r.NextF64(50.0));
			}

			for (int i = 0; i < 5000; i++)
			{
				Vec3 a = P(rng);
				Vec3 b = P(rng);
				Vec3 c = P(rng);
				Vec3 d = P(rng);
				Sign exact = Predicates.Orient3dR(
					R3.FromVec3(a),
					R3.FromVec3(b),
					R3.FromVec3(c),
					R3.FromVec3(d));
				AssertEq(Filtered.Orient3d(a, b, c, d), exact, $"#{i}");
			}
		}

		[Test]
		public void Orient3dAdversarialNearCoplanar()
		{
			// d ulp-perturbed off the plane z = x + y; every perturbation direction
			// must be classified correctly, and the on-plane point must be Zero.
			Vec3 a = new Vec3(0.0, 0.0, 0.0);
			Vec3 b = new Vec3(1.0, 0.0, 1.0);
			Vec3 c = new Vec3(0.0, 1.0, 1.0);
			for (int i = -16; i <= 16; i++)
			{
				double z = 7.0 + ((double)i * PowI(2.0, -50));
				Vec3 d = new Vec3(3.0, 4.0, z);
				Sign exact = Predicates.Orient3dR(
					R3.FromVec3(a),
					R3.FromVec3(b),
					R3.FromVec3(c),
					R3.FromVec3(d));
				AssertEq(Filtered.Orient3d(a, b, c, d), exact, $"i={i}");
				if (i == 0)
				{
					AssertEq(exact, Sign.Zero, "the on-plane point");
				}
			}
		}

		[Test]
		public void Orient3dSignConvention()
		{
			// CCW base triangle in z=0 viewed from +z; d above the plane → Pos.
			Vec3 a = new Vec3(0.0, 0.0, 0.0);
			Vec3 b = new Vec3(1.0, 0.0, 0.0);
			Vec3 c = new Vec3(0.0, 1.0, 0.0);
			AssertEq(Filtered.Orient3d(a, b, c, new Vec3(0.2, 0.2, 1.0)), Sign.Pos, "above");
			AssertEq(Filtered.Orient3d(a, b, c, new Vec3(0.2, 0.2, -1.0)), Sign.Neg, "below");
		}

		[Test]
		public void IncircleMatchesExactAndHandlesCocircular()
		{
			Lcg rng = new Lcg(1234);
			Vec2 P(Lcg r)
			{
				return new Vec2(r.NextF64(10.0), r.NextF64(10.0));
			}

			for (int i = 0; i < 3000; i++)
			{
				Vec2 a0 = P(rng);
				Vec2 b0 = P(rng);
				Vec2 c0 = P(rng);
				Vec2 d0 = P(rng);
				Sign exact = Predicates.IncircleR(
					R2.FromVec2(a0),
					R2.FromVec2(b0),
					R2.FromVec2(c0),
					R2.FromVec2(d0));
				AssertEq(Filtered.Incircle(a0, b0, c0, d0), exact, $"#{i}");
			}

			// Four exactly cocircular points (x² + y² = 25) → Zero regardless of filter.
			Vec2 a = new Vec2(3.0, 4.0);
			Vec2 b = new Vec2(5.0, 0.0);
			Vec2 c = new Vec2(-3.0, -4.0);
			Vec2 d = new Vec2(-5.0, 0.0);
			AssertEq(Filtered.Incircle(a, b, c, d), Sign.Zero, "cocircular");

			// Strictly inside / outside for the same CCW circle.
			AssertEq(Filtered.Incircle(a, b, c, new Vec2(0.0, 0.0)), Sign.Neg, "a,b,c is CW here");
			Vec2 a2 = b;
			Vec2 b2 = a;
			Vec2 c2 = c; // flip to CCW
			AssertEq(Filtered.Orient2d(a2, b2, c2), Sign.Pos, "flipped to CCW");
			AssertEq(Filtered.Incircle(a2, b2, c2, new Vec2(0.0, 0.0)), Sign.Pos, "center is inside");
			AssertEq(Filtered.Incircle(a2, b2, c2, new Vec2(100.0, 0.0)), Sign.Neg, "far point is outside");
		}

		[Test]
		public void FilterHitRateIsHighOnGenericInput()
		{
			// Measured from this test's own calls only: the filter helpers return null
			// exactly when the public predicates would escalate to the exact tier. (An
			// earlier version read process-global counters, which other tests running
			// concurrently in the same process polluted.)
			ulong fast = 0;
			ulong exact = 0;
			void Tally(Sign? filteredSign, Sign exactSign)
			{
				if (filteredSign != null)
				{
					// A resolved filter must agree with the predicate's final answer.
					AssertEq(filteredSign.Value, exactSign, "the filter disagreed with the predicate");
					fast++;
				}
				else
				{
					exact++;
				}
			}

			Lcg rng = new Lcg(99);
			Vec3 P(Lcg r)
			{
				return new Vec3(r.NextF64(50.0), r.NextF64(50.0), r.NextF64(50.0));
			}

			for (int i = 0; i < 10000; i++)
			{
				Vec3 a = P(rng);
				Vec3 b = P(rng);
				Vec3 c = P(rng);
				Vec3 d = P(rng);
				Tally(Filtered.Orient3dFilter(a, b, c, d), Filtered.Orient3d(a, b, c, d));
				Vec2 a2 = new Vec2(a.X, a.Y);
				Vec2 b2 = new Vec2(b.X, b.Y);
				Vec2 c2 = new Vec2(c.X, c.Y);
				Tally(Filtered.Orient2dFilter(a2, b2, c2), Filtered.Orient2d(a2, b2, c2));
			}

			double rate = (double)fast / (double)(fast + exact);
			AssertTrue(
				rate > 0.99,
				$"float filter resolved only {rate:F4} of generic predicates (fast={fast}, exact={exact})");
		}
	}
}
