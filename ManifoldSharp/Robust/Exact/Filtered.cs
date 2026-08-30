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

// Filtered.cs — port of robust/exact/filtered.rs. Float entry points for the
// exact predicates.
//
// Each predicate first evaluates the determinant in plain f64 together with
// a "permanent" (the same expression with every subtraction replaced by
// addition of absolute values). Shewchuk's static error-bound analysis
// ("Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
// Predicates", 1997) shows the f64 sign is certain whenever
// |det| > errboundA * permanent; otherwise Orient2d/Orient3d escalate to the
// exact integer evaluation in IntPred.cs (degenerate-heavy meshes make this
// tier hot) and Incircle to the rational evaluation in Predicates.cs. Unlike
// Shewchuk we skip the adaptive intermediate stages — exact integer evaluation
// is cheap enough that simplicity wins.
//
// The classic bounds assume no underflow/overflow, so any permanent that is
// subnormal, zero, or non-finite also escalates to the exact path.

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Robust.Exact
{
	/// <summary>
	/// The f64 entry points of the exact predicates: a static-error-bound filter
	/// with an escalation to the exact tiers when the filter cannot certify a sign.
	/// </summary>
	public static class Filtered
	{
		/// <summary>
		/// Shewchuk's machine epsilon, 2^-53. The literal is f64::EPSILON, which is
		/// the ulp of 1.0 — not C#'s double.Epsilon, which is the smallest subnormal.
		/// </summary>
		private const double EPS = 2.220446049250313E-16 * 0.5;

		private const double CCW_ERRBOUND_A = (3.0 + (16.0 * EPS)) * EPS;
		private const double O3D_ERRBOUND_A = (7.0 + (56.0 * EPS)) * EPS;
		private const double ICC_ERRBOUND_A = (10.0 + (96.0 * EPS)) * EPS;

		/// <summary>The smallest positive normal f64 (Rust's f64::MIN_POSITIVE).</summary>
		private const double MIN_POSITIVE = 2.2250738585072014E-308;

		/// <summary>
		/// True when the filter comparison itself is trustworthy: a normal, finite
		/// permanent. Subnormal permanents can hide underflowed terms whose error is
		/// not covered by the static bound; infinite ones mean the f64 det overflowed.
		/// </summary>
		private static bool PermanentOk(double permanent)
		{
			return permanent >= MIN_POSITIVE && double.IsFinite(permanent);
		}

		/// <summary>
		/// The float filter behind <see cref="Orient2d"/>: the sign when the static
		/// error bound certifies the f64 determinant's sign, null when the caller must
		/// escalate to the exact tier.
		/// </summary>
		/// <remarks>
		/// Split out from <see cref="Orient2d"/> so tests can measure the filter's hit
		/// rate from return values on their own inputs, rather than from process-global
		/// counters that concurrently running tests would pollute.
		/// </remarks>
		internal static Sign? Orient2dFilter(Vec2 a, Vec2 b, Vec2 c)
		{
			double detleft = (a.X - c.X) * (b.Y - c.Y);
			double detright = (a.Y - c.Y) * (b.X - c.X);
			double det = detleft - detright;
			double permanent = Math.Abs(detleft) + Math.Abs(detright);
			if (PermanentOk(permanent) && Math.Abs(det) > CCW_ERRBOUND_A * permanent)
			{
				return SignFunctions.OfF64(det);
			}

			return null;
		}

		/// <summary>
		/// Sign of cross(b-a, c-a); Pos ⇔ a,b,c counterclockwise. Exact.
		/// </summary>
		public static Sign Orient2d(Vec2 a, Vec2 b, Vec2 c)
		{
			Sign? sign = Orient2dFilter(a, b, c);
			if (sign != null)
			{
				return sign.Value;
			}

			return IntPred.Orient2dI(a, b, c);
		}

		/// <summary>
		/// The float filter behind <see cref="Orient3d"/>; see
		/// <see cref="Orient2dFilter"/> for why it is a separate, nullable-returning
		/// function.
		/// </summary>
		internal static Sign? Orient3dFilter(Vec3 a, Vec3 b, Vec3 c, Vec3 d)
		{
			double ux = b.X - a.X;
			double uy = b.Y - a.Y;
			double uz = b.Z - a.Z;
			double vx = c.X - a.X;
			double vy = c.Y - a.Y;
			double vz = c.Z - a.Z;
			double wx = d.X - a.X;
			double wy = d.Y - a.Y;
			double wz = d.Z - a.Z;

			double vywz = vy * wz;
			double vzwy = vz * wy;
			double vzwx = vz * wx;
			double vxwz = vx * wz;
			double vxwy = vx * wy;
			double vywx = vy * wx;

			double det = (ux * (vywz - vzwy)) + (uy * (vzwx - vxwz)) + (uz * (vxwy - vywx));
			double permanent = ((Math.Abs(vywz) + Math.Abs(vzwy)) * Math.Abs(ux))
				+ ((Math.Abs(vzwx) + Math.Abs(vxwz)) * Math.Abs(uy))
				+ ((Math.Abs(vxwy) + Math.Abs(vywx)) * Math.Abs(uz));
			if (PermanentOk(permanent) && Math.Abs(det) > O3D_ERRBOUND_A * permanent)
			{
				return SignFunctions.OfF64(det);
			}

			return null;
		}

		/// <summary>
		/// Sign of dot(cross(b-a, c-a), d-a); Pos ⇔ d on the CCW-normal side of
		/// plane(a,b,c), Zero ⇔ coplanar. Exact.
		/// </summary>
		public static Sign Orient3d(Vec3 a, Vec3 b, Vec3 c, Vec3 d)
		{
			Sign? sign = Orient3dFilter(a, b, c, d);
			if (sign != null)
			{
				return sign.Value;
			}

			return IntPred.Orient3dI(a, b, c, d);
		}

		/// <summary>
		/// Incircle test; with a,b,c CCW: Pos ⇔ d strictly inside the circumcircle of
		/// (a,b,c). Exact.
		/// </summary>
		public static Sign Incircle(Vec2 a, Vec2 b, Vec2 c, Vec2 d)
		{
			double adx = a.X - d.X;
			double ady = a.Y - d.Y;
			double bdx = b.X - d.X;
			double bdy = b.Y - d.Y;
			double cdx = c.X - d.X;
			double cdy = c.Y - d.Y;

			double bdxcdy = bdx * cdy;
			double cdxbdy = cdx * bdy;
			double alift = (adx * adx) + (ady * ady);

			double cdxady = cdx * ady;
			double adxcdy = adx * cdy;
			double blift = (bdx * bdx) + (bdy * bdy);

			double adxbdy = adx * bdy;
			double bdxady = bdx * ady;
			double clift = (cdx * cdx) + (cdy * cdy);

			double det = (alift * (bdxcdy - cdxbdy))
				+ (blift * (cdxady - adxcdy))
				+ (clift * (adxbdy - bdxady));
			double permanent = ((Math.Abs(bdxcdy) + Math.Abs(cdxbdy)) * alift)
				+ ((Math.Abs(cdxady) + Math.Abs(adxcdy)) * blift)
				+ ((Math.Abs(adxbdy) + Math.Abs(bdxady)) * clift);
			if (PermanentOk(permanent) && Math.Abs(det) > ICC_ERRBOUND_A * permanent)
			{
				return SignFunctions.OfF64(det);
			}

			return Predicates.IncircleR(
				R2.FromVec2(a),
				R2.FromVec2(b),
				R2.FromVec2(c),
				R2.FromVec2(d));
		}
	}
}
