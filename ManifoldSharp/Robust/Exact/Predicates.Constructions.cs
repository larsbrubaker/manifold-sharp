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

// Predicates.Constructions.cs — the exact constructions from the bottom of
// robust/exact/predicates.rs. The module header is on Predicates.cs.
//
// Every construction here builds each output coordinate with exactly ONE
// RatNew, so the gcd reduction is paid once per coordinate and never on an
// intermediate. That is item (3) of the backend checklist in Backend.cs; it is
// a measured hot spot, so a rewrite that introduces a second reduction per
// coordinate is a performance regression even when it is numerically identical.

using System.Diagnostics;
using System.Numerics;

namespace ManifoldSharp.Robust.Exact
{
	/// <content>The exact geometric constructions.</content>
	public static partial class Predicates
	{
		/// <summary>
		/// Intersection of the line through p,q with the plane of triangle (a,b,c).
		/// </summary>
		/// <returns>
		/// Null when the segment direction is parallel to the plane (no unique
		/// intersection point — the coplanar-overlap machinery handles that case
		/// separately). The caller decides whether the parameter t lies inside [0,1];
		/// use the orient3d signs of p and q for that, not float comparisons.
		/// </returns>
		public static R3? LinePlaneIntersect(R3 p, R3 q, R3 a, R3 b, R3 c)
		{
			// Integer-only formulation (same exact point as the rational one): the
			// plane normal enters both numerator and denominator of t, so any
			// positive common scale on it cancels — compute it as an integer cross
			// of denominator-cleared edge vectors and never normalize intermediates.
			// Each output coordinate reduces exactly once in RatNew.
			(BigInteger px, BigInteger py, BigInteger pz, BigInteger pw) = Homog3Parts(p);
			(BigInteger qx, BigInteger qy, BigInteger qz, BigInteger qw) = Homog3Parts(q);
			(BigInteger ax, BigInteger ay, BigInteger az, BigInteger aw) = Homog3Parts(a);
			(BigInteger bx, BigInteger by, BigInteger bz, BigInteger bw) = Homog3Parts(b);
			(BigInteger cx, BigInteger cy, BigInteger cz, BigInteger cw) = Homog3Parts(c);

			// (b−a)·AwBw and (c−a)·AwCw; their cross is the normal up to Aw²BwCw > 0.
			BigInteger ux = (bx * aw) - (ax * bw);
			BigInteger uy = (by * aw) - (ay * bw);
			BigInteger uz = (bz * aw) - (az * bw);
			BigInteger vx = (cx * aw) - (ax * cw);
			BigInteger vy = (cy * aw) - (ay * cw);
			BigInteger vz = (cz * aw) - (az * cw);
			BigInteger nx = (uy * vz) - (uz * vy);
			BigInteger ny = (uz * vx) - (ux * vz);
			BigInteger nz = (ux * vy) - (uy * vx);

			// dir = q−p scaled by PwQw; e = a−p scaled by PwAw.
			BigInteger dx = (qx * pw) - (px * qw);
			BigInteger dy = (qy * pw) - (py * qw);
			BigInteger dz = (qz * pw) - (pz * qw);
			BigInteger nDotD = (nx * dx) + (ny * dy) + (nz * dz);
			if (nDotD.IsZero)
			{
				return null;
			}

			BigInteger ex = (ax * pw) - (px * aw);
			BigInteger ey = (ay * pw) - (py * aw);
			BigInteger ez = (az * pw) - (pz * aw);
			BigInteger nDotE = (nx * ex) + (ny * ey) + (nz * ez);

			// t = (n·e)·Qw / (Aw·(n·d));  x_i = (P_i·Qw·T_d + D_i·T_n) / (Pw·Qw·T_d)
			BigInteger tn = nDotE * qw;
			BigInteger td = aw * nDotD;
			BigInteger den = pw * qw * td;

			BigRational Coord(in BigInteger pi, in BigInteger di)
			{
				return Backend.RatNew((pi * qw * td) + (di * tn), den);
			}

			return new R3(Coord(px, dx), Coord(py, dy), Coord(pz, dz));
		}

		/// <summary>
		/// Intersection point of the 2D lines through (a,b) and (c,d).
		/// </summary>
		/// <returns>
		/// Null when the lines are parallel (including collinear — overlap is handled
		/// by the caller's collinear branch).
		/// </returns>
		public static R2? LineLineIntersect2d(R2 a, R2 b, R2 c, R2 d)
		{
			// Integer-only (same exact point as the rational formulation): with
			// homogenized points, x = a + t·(b−a) where
			//   t = N·Bw / (Dn·Cw),
			//   N  = cross(c−a, d−c)·AwCw·CwDw,   Dn = cross(b−a, d−c)·AwBw·CwDw,
			// and each output coordinate reduces exactly once in RatNew.
			(BigInteger ax, BigInteger ay, BigInteger aw) = Homog2Parts(a);
			(BigInteger bx, BigInteger by, BigInteger bw) = Homog2Parts(b);
			(BigInteger cx, BigInteger cy, BigInteger cw) = Homog2Parts(c);
			(BigInteger dx, BigInteger dy, BigInteger dw) = Homog2Parts(d);

			BigInteger abx = (bx * aw) - (ax * bw);
			BigInteger aby = (by * aw) - (ay * bw);
			BigInteger cdx = (dx * cw) - (cx * dw);
			BigInteger cdy = (dy * cw) - (cy * dw);
			BigInteger dn = (abx * cdy) - (aby * cdx);
			if (dn.IsZero)
			{
				return null;
			}

			BigInteger cax = (cx * aw) - (ax * cw);
			BigInteger cay = (cy * aw) - (ay * cw);
			BigInteger n = (cax * cdy) - (cay * cdx);

			// x_i = (A_i·Dn·Cw + N·ab_i) / (Aw·Cw·Dn)
			BigInteger den = aw * cw * dn;
			BigInteger dnCw = dn * cw;
			BigRational x = Backend.RatNew((ax * dnCw) + (n * abx), den);
			BigRational y = Backend.RatNew((ay * dnCw) + (n * aby), den);
			return new R2(x, y);
		}

		/// <summary>
		/// Inverse of <see cref="R3.ProjectDrop"/>: rebuilds the dropped coordinate
		/// from the plane through <paramref name="a"/> with (unnormalized, rational)
		/// normal <paramref name="n"/>, whose <paramref name="axis"/> component must be
		/// nonzero. Integer-only: the reconstructed coordinate is one RatNew (a single
		/// gcd); the carried coordinates are copies of the projection's already
		/// canonical rationals.
		/// </summary>
		/// <exception cref="ArgumentOutOfRangeException">The axis is not 0, 1 or 2.</exception>
		public static R3 LiftToPlane(R2 p, int axis, R3 a, R3 n)
		{
			(BigInteger nx, BigInteger ny, BigInteger nz, BigInteger nw) = Homog3Parts(n);
			(BigInteger ax, BigInteger ay, BigInteger az, BigInteger aw) = Homog3Parts(a);
			(BigInteger px, BigInteger py, BigInteger pw) = Homog2Parts(p);

			// S = (n·a)·NwAw; dropped = (n·a − n_i·p_i − n_j·p_j) / n_k
			//   = (S·Pw − Aw·(N_i·P_i + N_j·P_j)) / (Aw·Pw·N_k).
			BigInteger s = (nx * ax) + (ny * ay) + (nz * az);

			BigRational Rebuild(in BigInteger ni, in BigInteger nj, in BigInteger nk)
			{
				return Backend.RatNew((s * pw) - (aw * ((ni * px) + (nj * py))), aw * pw * nk);
			}

			_ = nw; // cancels: both S and the subtracted terms carry 1/Nw
			switch (axis)
			{
				case 0:
					return new R3(Rebuild(ny, nz, nx), p.X, p.Y);
				case 1:
					return new R3(p.Y, Rebuild(nz, nx, ny), p.X);
				case 2:
					return new R3(p.X, p.Y, Rebuild(nx, ny, nz));
				default:
					throw new ArgumentOutOfRangeException(nameof(axis), "axis must be 0, 1, or 2");
			}
		}

		/// <summary>
		/// Parameter of point x on segment (p,q) along the dominant axis of the segment
		/// direction — exact, in [0,1] iff x lies within the segment. The caller
		/// guarantees x is on the line through p and q and p != q.
		/// </summary>
		public static BigRational SegmentParam(R3 p, R3 q, R3 x)
		{
			R3 d = q.Sub(p);
			BigRational num;
			BigRational den;
			if (!Backend.RatIsZero(d.X))
			{
				num = x.X - p.X;
				den = d.X;
			}
			else if (!Backend.RatIsZero(d.Y))
			{
				num = x.Y - p.Y;
				den = d.Y;
			}
			else
			{
				num = x.Z - p.Z;
				den = d.Z;
			}

			Debug.Assert(!Backend.RatIsZero(den), "SegmentParam requires p != q");
			return num / den;
		}
	}
}
