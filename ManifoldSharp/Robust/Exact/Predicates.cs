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

// Predicates.cs — port of robust/exact/predicates.rs. Fully exact predicates
// and geometric constructions on rational points. Ground truth for the robust
// boolean engine: Filtered.cs escalates here whenever its float filters cannot
// certify a sign, and the intersection code builds all new vertices through the
// constructions (which live in Predicates.Constructions.cs).
//
// Orientation conventions (documented once, used everywhere in ManifoldSharp.Robust):
//   Orient2d(a,b,c)   = sign of cross(b-a, c-a); Pos = a,b,c counterclockwise.
//   Orient3d(a,b,c,d) = sign of dot(cross(b-a, c-a), d-a); Pos = d on the
//                       side of plane(a,b,c) that its CCW normal points to.
//   Incircle(a,b,c,d) with a,b,c CCW: Pos = d strictly inside the circle
//                       through a,b,c. (For CW a,b,c the sign flips.)

using System.Numerics;
using System.Runtime.CompilerServices;

namespace ManifoldSharp.Robust.Exact
{
	/// <summary>
	/// Where a point lies relative to a non-degenerate triangle, in 2D.
	/// </summary>
	/// <remarks>
	/// The Rust is an enum with a payload (<c>OnEdge(u8)</c>/<c>OnVertex(u8)</c>).
	/// This is a struct with a <see cref="Kind"/> tag and the payload alongside it,
	/// per the plan's rule for hot-loop enums-with-data: an abstract base plus
	/// sealed subclasses would allocate on every predicate call.
	/// </remarks>
	public readonly struct TriLoc : IEquatable<TriLoc>
	{
		/// <summary>Which of the four cases this is.</summary>
		public readonly TriLocKind Kind;

		/// <summary>
		/// The edge or vertex index for <see cref="TriLocKind.OnEdge"/> /
		/// <see cref="TriLocKind.OnVertex"/>; 0 for the payload-free cases, so that
		/// equality can compare both fields unconditionally.
		/// </summary>
		public readonly byte Index;

		private TriLoc(TriLocKind kind, byte index)
		{
			this.Kind = kind;
			this.Index = index;
		}

		/// <summary>Strictly inside the triangle.</summary>
		public static TriLoc Inside
		{
			get { return new TriLoc(TriLocKind.Inside, 0); }
		}

		/// <summary>Outside the triangle (or the triangle is degenerate).</summary>
		public static TriLoc Outside
		{
			get { return new TriLoc(TriLocKind.Outside, 0); }
		}

		/// <summary>
		/// On the open edge <paramref name="i"/>, where edge i runs from vertex i to
		/// vertex i+1 (mod 3).
		/// </summary>
		public static TriLoc OnEdge(byte i)
		{
			return new TriLoc(TriLocKind.OnEdge, i);
		}

		/// <summary>Coincident with vertex <paramref name="i"/>.</summary>
		public static TriLoc OnVertex(byte i)
		{
			return new TriLoc(TriLocKind.OnVertex, i);
		}

		/// <summary>Case and payload equality.</summary>
		public bool Equals(TriLoc other)
		{
			return this.Kind == other.Kind && this.Index == other.Index;
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is TriLoc other && this.Equals(other);
		}

		/// <summary>Case and payload equality.</summary>
		public static bool operator ==(TriLoc a, TriLoc b)
		{
			return a.Equals(b);
		}

		/// <summary>Case or payload inequality.</summary>
		public static bool operator !=(TriLoc a, TriLoc b)
		{
			return !a.Equals(b);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine((int)this.Kind, (int)this.Index);
		}

		/// <summary>The case, with its payload where it has one.</summary>
		public override string ToString()
		{
			switch (this.Kind)
			{
				case TriLocKind.OnEdge:
					return "OnEdge(" + this.Index.ToString() + ")";
				case TriLocKind.OnVertex:
					return "OnVertex(" + this.Index.ToString() + ")";
				default:
					return this.Kind.ToString();
			}
		}
	}

	/// <summary>The four cases of <see cref="TriLoc"/>.</summary>
	public enum TriLocKind
	{
		/// <summary>Strictly inside.</summary>
		Inside,

		/// <summary>On an open edge; <see cref="TriLoc.Index"/> says which.</summary>
		OnEdge,

		/// <summary>Coincident with a vertex; <see cref="TriLoc.Index"/> says which.</summary>
		OnVertex,

		/// <summary>Outside.</summary>
		Outside,
	}

	/// <summary>
	/// Cached homogenization of a 2D point: (X, Y, W), x = X/W, W &gt; 0. Hot loops
	/// that test one point against many (the arrangement's segment sweep)
	/// homogenize each point once and reuse it across every predicate call.
	/// </summary>
	public readonly struct Homog2
	{
		/// <summary>The homogeneous x numerator.</summary>
		public readonly BigInteger X;

		/// <summary>The homogeneous y numerator.</summary>
		public readonly BigInteger Y;

		/// <summary>The common (strictly positive) denominator.</summary>
		public readonly BigInteger W;

		/// <summary>Wraps an already homogenized triple.</summary>
		public Homog2(BigInteger x, BigInteger y, BigInteger w)
		{
			this.X = x;
			this.Y = y;
			this.W = w;
		}
	}

	/// <summary>
	/// The exact predicates of predicates.rs, over rational points.
	/// </summary>
	/// <remarks>
	/// Predicate signs are computed in pure BigInteger arithmetic: each point is
	/// homogenized once ((x, y) = (X/W, Y/W) with W &gt; 0 — the canonical rationals
	/// keep denominators positive), and the determinant is scaled through by
	/// positive denominator products, which preserves its sign. This avoids the
	/// rationals' gcd normalization on every intermediate operation — the dominant
	/// cost of the original rational formulation (the CDT's incircle calls on
	/// constructed intersection points made it ~80% of robust-boolean wall time).
	/// </remarks>
	public static partial class Predicates
	{
		/// <summary>(X, Y, W): x = X/W, y = Y/W with W &gt; 0.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static (BigInteger X, BigInteger Y, BigInteger W) Homog2Parts(R2 p)
		{
			BigInteger xn = Backend.Numer(p.X);
			BigInteger xd = Backend.Denom(p.X);
			BigInteger yn = Backend.Numer(p.Y);
			BigInteger yd = Backend.Denom(p.Y);
			return (
				Backend.MulIntUint(xn, yd),
				Backend.MulIntUint(yn, xd),
				Backend.IntFromUint(Backend.MulUint(xd, yd)));
		}

		/// <summary>(X, Y, Z, W): coordinates over one positive common denominator.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static (BigInteger X, BigInteger Y, BigInteger Z, BigInteger W) Homog3Parts(R3 p)
		{
			BigInteger xn = Backend.Numer(p.X);
			BigInteger xd = Backend.Denom(p.X);
			BigInteger yn = Backend.Numer(p.Y);
			BigInteger yd = Backend.Denom(p.Y);
			BigInteger zn = Backend.Numer(p.Z);
			BigInteger zd = Backend.Denom(p.Z);
			BigInteger yz = Backend.MulUint(yd, zd);
			return (
				Backend.MulIntUint(xn, yz),
				Backend.MulIntUint(yn, Backend.MulUint(xd, zd)),
				Backend.MulIntUint(zn, Backend.MulUint(xd, yd)),
				Backend.IntFromUint(Backend.MulUint(xd, yz)));
		}

		/// <summary>Homogenizes a 2D point once, for reuse across many predicate calls.</summary>
		public static Homog2 Homog2Of(R2 p)
		{
			(BigInteger x, BigInteger y, BigInteger w) = Homog2Parts(p);
			return new Homog2(x, y, w);
		}

		/// <summary>
		/// <see cref="Orient2dR"/> over pre-homogenized points — identical sign, no
		/// repeated denominator work.
		/// </summary>
		public static Sign Orient2dH(in Homog2 a, in Homog2 b, in Homog2 c)
		{
			BigInteger ux = (b.X * a.W) - (a.X * b.W);
			BigInteger uy = (b.Y * a.W) - (a.Y * b.W);
			BigInteger vx = (c.X * a.W) - (a.X * c.W);
			BigInteger vy = (c.Y * a.W) - (a.Y * c.W);
			return SignOfInt((ux * vy) - (uy * vx));
		}

		/// <summary>
		/// <see cref="IncircleR"/> over pre-homogenized points — identical sign (same
		/// row scaling argument), computed without re-clearing any denominators.
		/// </summary>
		public static Sign IncircleH(in Homog2 a, in Homog2 b, in Homog2 c, in Homog2 d)
		{
			BigInteger dx = d.X;
			BigInteger dy = d.Y;
			BigInteger dw = d.W;

			(BigInteger X, BigInteger Y, BigInteger Lift) Row(in Homog2 p)
			{
				BigInteger nx = (p.X * dw) - (dx * p.W);
				BigInteger ny = (p.Y * dw) - (dy * p.W);
				BigInteger s = p.W * dw;
				BigInteger lift = (nx * nx) + (ny * ny);
				return (nx * s, ny * s, lift);
			}

			(BigInteger ux, BigInteger uy, BigInteger ul) = Row(a);
			(BigInteger vx, BigInteger vy, BigInteger vl) = Row(b);
			(BigInteger wx, BigInteger wy, BigInteger wl) = Row(c);
			BigInteger det = (ul * ((vx * wy) - (vy * wx)))
				+ (vl * ((wx * uy) - (wy * ux)))
				+ (wl * ((ux * vy) - (uy * vx)));
			return SignOfInt(det);
		}

		/// <summary><see cref="PointInTri2d"/> over pre-homogenized points.</summary>
		public static TriLoc PointInTri2dH(in Homog2 p, in Homog2 a, in Homog2 b, in Homog2 c)
		{
			Sign orient = Orient2dH(a, b, c);
			if (orient == Sign.Zero)
			{
				return TriLoc.Outside;
			}

			Sign Normalize(Sign s)
			{
				return orient == Sign.Pos ? s : s.Flip();
			}

			Sign s0 = Normalize(Orient2dH(a, b, p));
			Sign s1 = Normalize(Orient2dH(b, c, p));
			Sign s2 = Normalize(Orient2dH(c, a, p));
			return ClassifyTriLoc(s0, s1, s2);
		}

		/// <summary>
		/// The shared tail of the two <c>point_in_tri_2d</c> forms: three edge signs,
		/// already normalized to CCW, mapped onto the location.
		/// </summary>
		private static TriLoc ClassifyTriLoc(Sign s0, Sign s1, Sign s2)
		{
			if (s0 == Sign.Neg || s1 == Sign.Neg || s2 == Sign.Neg)
			{
				return TriLoc.Outside;
			}

			bool z0 = s0 == Sign.Zero;
			bool z1 = s1 == Sign.Zero;
			bool z2 = s2 == Sign.Zero;
			if (!z0 && !z1 && !z2)
			{
				return TriLoc.Inside;
			}

			if (z0 && !z1 && !z2)
			{
				return TriLoc.OnEdge(0);
			}

			if (!z0 && z1 && !z2)
			{
				return TriLoc.OnEdge(1);
			}

			if (!z0 && !z1 && z2)
			{
				return TriLoc.OnEdge(2);
			}

			if (z0 && !z1 && z2)
			{
				return TriLoc.OnVertex(0); // a: on edges c→a and a→b
			}

			if (z0 && z1 && !z2)
			{
				return TriLoc.OnVertex(1); // b
			}

			if (!z0 && z1 && z2)
			{
				return TriLoc.OnVertex(2); // c
			}

			return TriLoc.Outside; // all three zero: impossible for orient != 0
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static Sign SignOfInt(in BigInteger v)
		{
			if (v.Sign > 0)
			{
				return Sign.Pos;
			}
			else if (v.Sign < 0)
			{
				return Sign.Neg;
			}
			else
			{
				return Sign.Zero;
			}
		}

		/// <summary>
		/// Sign of cross(b-a, c-a). Pos ⇔ a,b,c wind counterclockwise.
		/// </summary>
		public static Sign Orient2dR(R2 a, R2 b, R2 c)
		{
			(BigInteger ax, BigInteger ay, BigInteger aw) = Homog2Parts(a);
			(BigInteger bx, BigInteger by, BigInteger bw) = Homog2Parts(b);
			(BigInteger cx, BigInteger cy, BigInteger cw) = Homog2Parts(c);

			// det(b-a, c-a) · Wa²WbWc = (BxWa−AxWb)(CyWa−AyWc) − (ByWa−AyWb)(CxWa−AxWc)
			BigInteger ux = (bx * aw) - (ax * bw);
			BigInteger uy = (by * aw) - (ay * bw);
			BigInteger vx = (cx * aw) - (ax * cw);
			BigInteger vy = (cy * aw) - (ay * cw);
			return SignOfInt((ux * vy) - (uy * vx));
		}

		/// <summary>
		/// Sign of dot(cross(b-a, c-a), d-a). Pos ⇔ d lies on the CCW-normal side of
		/// the plane through a, b, c; Zero ⇔ the four points are coplanar.
		/// </summary>
		public static Sign Orient3dR(R3 a, R3 b, R3 c, R3 d)
		{
			(BigInteger ax, BigInteger ay, BigInteger az, BigInteger aw) = Homog3Parts(a);
			(BigInteger bx, BigInteger by, BigInteger bz, BigInteger bw) = Homog3Parts(b);
			(BigInteger cx, BigInteger cy, BigInteger cz, BigInteger cw) = Homog3Parts(c);
			(BigInteger dx, BigInteger dy, BigInteger dz, BigInteger dw) = Homog3Parts(d);

			// Each difference row is scaled by the positive factor Wa·W_row; the
			// triple product then carries a positive overall scale.
			BigInteger ux = (bx * aw) - (ax * bw);
			BigInteger uy = (by * aw) - (ay * bw);
			BigInteger uz = (bz * aw) - (az * bw);
			BigInteger vx = (cx * aw) - (ax * cw);
			BigInteger vy = (cy * aw) - (ay * cw);
			BigInteger vz = (cz * aw) - (az * cw);
			BigInteger wx = (dx * aw) - (ax * dw);
			BigInteger wy = (dy * aw) - (ay * dw);
			BigInteger wz = (dz * aw) - (az * dw);
			BigInteger det = (((uy * vz) - (uz * vy)) * wx)
				+ (((uz * vx) - (ux * vz)) * wy)
				+ (((ux * vy) - (uy * vx)) * wz);
			return SignOfInt(det);
		}

		/// <summary>
		/// Incircle test. With a,b,c counterclockwise: Pos ⇔ d strictly inside the
		/// circumcircle of (a,b,c). Computed as the standard 3×3 determinant of
		/// coordinates lifted onto the paraboloid, with rows differenced against d.
		/// </summary>
		public static Sign IncircleR(R2 a, R2 b, R2 c, R2 d)
		{
			(BigInteger dx, BigInteger dy, BigInteger dw) = Homog2Parts(d);

			// Row i (i = a,b,c): (xi−xd, yi−yd) over the positive denominator WiWd.
			// Scaling row i by (WiWd)² keeps the lift column polynomial:
			//   Ui = (XiWd−XdWi)·WiWd,  Vi = (YiWd−YdWi)·WiWd,
			//   Li = (XiWd−XdWi)² + (YiWd−YdWi)².
			(BigInteger X, BigInteger Y, BigInteger Lift) Row(R2 p)
			{
				(BigInteger px, BigInteger py, BigInteger pw) = Homog2Parts(p);
				BigInteger nx = (px * dw) - (dx * pw);
				BigInteger ny = (py * dw) - (dy * pw);
				BigInteger s = pw * dw;
				BigInteger lift = (nx * nx) + (ny * ny);
				return (nx * s, ny * s, lift);
			}

			(BigInteger ux, BigInteger uy, BigInteger ul) = Row(a);
			(BigInteger vx, BigInteger vy, BigInteger vl) = Row(b);
			(BigInteger wx, BigInteger wy, BigInteger wl) = Row(c);
			BigInteger det = (ul * ((vx * wy) - (vy * wx)))
				+ (vl * ((wx * uy) - (wy * ux)))
				+ (wl * ((ux * vy) - (uy * vx)));
			return SignOfInt(det);
		}

		/// <summary>
		/// Exact: p collinear with (a,b) and within the closed segment [a,b]. Same
		/// integer-only strategy as the sign predicates above — the registry sweeps in
		/// robust/intersection_graph.rs call this in a tight loop.
		/// </summary>
		public static bool PointOnSegmentR(R3 p, R3 a, R3 b)
		{
			(BigInteger px, BigInteger py, BigInteger pz, BigInteger pw) = Homog3Parts(p);
			(BigInteger ax, BigInteger ay, BigInteger az, BigInteger aw) = Homog3Parts(a);
			(BigInteger bx, BigInteger by, BigInteger bz, BigInteger bw) = Homog3Parts(b);

			// ap scaled by the positive PwAw; d = b−a scaled by the positive AwBw.
			BigInteger apx = (px * aw) - (ax * pw);
			BigInteger apy = (py * aw) - (ay * pw);
			BigInteger apz = (pz * aw) - (az * pw);
			BigInteger dx = (bx * aw) - (ax * bw);
			BigInteger dy = (by * aw) - (ay * bw);
			BigInteger dz = (bz * aw) - (az * bw);

			// Collinearity: cross(ap, d) = 0 (positive scales cannot zero a component).
			if (!((apy * dz) - (apz * dy)).IsZero
				|| !((apz * dx) - (apx * dz)).IsZero
				|| !((apx * dy) - (apy * dx)).IsZero)
			{
				return false;
			}

			// 0 ≤ ap·d and ap·d ≤ d·d, cleared of their (positive) denominators:
			//   ap·d = S1 / (PwAw·AwBw),  d·d = S2 / (AwBw)²
			//   S1 ≥ 0   and   S1·AwBw ≤ S2·PwAw.
			BigInteger s1 = (apx * dx) + (apy * dy) + (apz * dz);
			if (s1.Sign < 0)
			{
				return false;
			}

			BigInteger s2 = (dx * dx) + (dy * dy) + (dz * dz);
			return s1 * (aw * bw) <= s2 * (pw * aw);
		}

		/// <summary>
		/// (a−o)·u as an unreduced fraction (numerator, positive denominator) —
		/// integer-only, no gcd normalization. For sign tests and cross-multiplied
		/// comparisons (e.g. the radial ring sort in robust/classify.rs) the unreduced
		/// form is exactly as good as the canonical one and much cheaper to produce.
		/// </summary>
		public static (BigInteger Numer, BigInteger Denom) DotDiffRaw(R3 a, R3 o, R3 u)
		{
			(BigInteger ax, BigInteger ay, BigInteger az, BigInteger aw) = Homog3Parts(a);
			(BigInteger ox, BigInteger oy, BigInteger oz, BigInteger ow) = Homog3Parts(o);
			(BigInteger ux, BigInteger uy, BigInteger uz, BigInteger uw) = Homog3Parts(u);

			// (a−o) scaled by the positive AwOw; dot with u adds a Uw denominator.
			BigInteger num = (((ax * ow) - (ox * aw)) * ux)
				+ (((ay * ow) - (oy * aw)) * uy)
				+ (((az * ow) - (oz * aw)) * uz);
			return (num, aw * ow * uw);
		}

		/// <summary>
		/// Triangle normal as a denominator-cleared integer vector: cross(b−a, c−a)
		/// scaled by the positive Aw²BwCw. Direction (and zero-ness) match
		/// <see cref="TriNormalR"/>; use where only the normal's direction matters.
		/// </summary>
		public static BigInteger[] TriNormalInt(R3 a, R3 b, R3 c)
		{
			(BigInteger ax, BigInteger ay, BigInteger az, BigInteger aw) = Homog3Parts(a);
			(BigInteger bx, BigInteger by, BigInteger bz, BigInteger bw) = Homog3Parts(b);
			(BigInteger cx, BigInteger cy, BigInteger cz, BigInteger cw) = Homog3Parts(c);
			BigInteger ux = (bx * aw) - (ax * bw);
			BigInteger uy = (by * aw) - (ay * bw);
			BigInteger uz = (bz * aw) - (az * bw);
			BigInteger vx = (cx * aw) - (ax * cw);
			BigInteger vy = (cy * aw) - (ay * cw);
			BigInteger vz = (cz * aw) - (az * cw);
			return new BigInteger[]
			{
				(uy * vz) - (uz * vy),
				(uz * vx) - (ux * vz),
				(ux * vy) - (uy * vx),
			};
		}

		/// <summary>
		/// d·p as an unreduced fraction (numerator, positive denominator), for an
		/// integer direction <paramref name="d"/> and rational point <paramref name="p"/>.
		/// Comparable across points by cross-multiplication — the segment-interval
		/// overlap in robust/tri_tri.rs orders plane-crossing points along the
		/// intersection line with this.
		/// </summary>
		/// <exception cref="ArgumentException">
		/// <paramref name="d"/> is not exactly three components. The Rust takes
		/// <c>&amp;[Int; 3]</c>, where the length is the type — the natural argument is
		/// a <see cref="TriNormalInt"/> result, which always has three.
		/// </exception>
		public static (BigInteger Numer, BigInteger Denom) DotPointRaw(BigInteger[] d, R3 p)
		{
			if (d.Length != 3)
			{
				throw new ArgumentException($"DotPointRaw takes a 3-component direction, got {d.Length}", nameof(d));
			}

			(BigInteger px, BigInteger py, BigInteger pz, BigInteger pw) = Homog3Parts(p);
			return ((d[0] * px) + (d[1] * py) + (d[2] * pz), pw);
		}

		/// <summary>
		/// Unnormalized CCW normal of triangle (a,b,c): cross(b-a, c-a). Zero vector ⇔
		/// the triangle is degenerate.
		/// </summary>
		public static R3 TriNormalR(R3 a, R3 b, R3 c)
		{
			return b.Sub(a).Cross(c.Sub(a));
		}

		/// <summary>
		/// Locates point p relative to triangle (a,b,c). Works for either winding; a
		/// degenerate (zero-area) triangle reports every point as Outside, which is
		/// consistent with the pipeline dropping degenerate triangles up front.
		/// </summary>
		public static TriLoc PointInTri2d(R2 p, R2 a, R2 b, R2 c)
		{
			Sign orient = Orient2dR(a, b, c);
			if (orient == Sign.Zero)
			{
				return TriLoc.Outside;
			}

			// Normalize so the triangle reads as CCW.
			Sign Normalize(Sign s)
			{
				return orient == Sign.Pos ? s : s.Flip();
			}

			Sign s0 = Normalize(Orient2dR(a, b, p)); // edge 0: a→b
			Sign s1 = Normalize(Orient2dR(b, c, p)); // edge 1: b→c
			Sign s2 = Normalize(Orient2dR(c, a, p)); // edge 2: c→a
			return ClassifyTriLoc(s0, s1, s2);
		}
	}
}
