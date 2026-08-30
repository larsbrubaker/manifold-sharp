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

// Approx.cs — port of robust/exact/approx.rs. Semi-static float filters over
// approximate coordinates of exact points ("indirect predicates" in the sense
// of Attene 2020, the engine room of interactive exact mesh booleans).
//
// Every point the robust pipeline touches is either an input point (exact f64)
// or a constructed rational whose f64 approximation we obtain by one correct
// rounding (Rational.RatToF64) — so every approximate coordinate x̃ satisfies
// x̃ = x·(1+δ) with |δ| ≤ ε = 2⁻⁵³. A predicate can therefore run entirely in
// f64 on the approximations and certify its sign with a magnitude-based error
// bound; only near-degenerate configurations escalate to the exact BigInteger
// evaluations in Predicates.cs.
//
// The bounds here are deliberately conservative (they cover both the
// float-evaluation roundoff and the input perturbation, with slack): an
// over-tight constant would be a soundness bug, an over-loose one only costs
// extra escalations exactly where the exact path was needed anyway. Unlike
// the classic Shewchuk filters in Filtered.cs, the "permanents" below use
// coordinate magnitudes rather than formed differences — input perturbation
// error scales with |a|+|c|, not |a−c|, under cancellation.
//
// Math.Max/Math.Min stand in for Rust's f64::max/f64::min. They differ only on
// NaN (Rust returns the non-NaN operand, C# propagates the NaN) and on the
// ±0.0 tie; every input here is a finite mesh coordinate or an approximation of
// one, and no comparison below can be changed by the sign of a zero.

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Robust.Exact
{
	/// <summary>
	/// Semi-static filters over approximate coordinates. Each returns a certified
	/// sign or null, never a wrong sign.
	/// </summary>
	public static class Approx
	{
		/// <summary>2⁻⁵³. See <see cref="Filtered"/> for why this is not double.Epsilon.</summary>
		private const double EPS = 2.220446049250313E-16 * 0.5;

		/// <summary>
		/// Conservative certified-sign helper: <paramref name="det"/> was computed in
		/// f64 from ε-perturbed inputs; <paramref name="bound"/> ≥ the true error.
		/// Returns null when the sign cannot be certified (including any non-finite
		/// intermediate).
		/// </summary>
		private static Sign? Certify(double det, double bound)
		{
			if (!double.IsFinite(det) || !double.IsFinite(bound))
			{
				return null;
			}

			if (det > bound)
			{
				return Sign.Pos;
			}
			else if (det < -bound)
			{
				return Sign.Neg;
			}
			else
			{
				return null;
			}
		}

		/// <summary>
		/// Filtered orient2d over approximate coordinates: sign of cross(b−a, c−a), or
		/// null when uncertain.
		/// </summary>
		/// <remarks>
		/// Error analysis (conservative): with per-coordinate relative error ≤ ε and
		/// ≤ 3 roundings per product term after differencing, the total error is
		/// &lt; 8ε·P where P = (|ax|+|bx|+|cx|)·(|ay|+|by|+|cy|) bounds every term's
		/// magnitude sum. We use 16ε·P for slack.
		/// </remarks>
		public static Sign? Orient2dA(Vec2 a, Vec2 b, Vec2 c)
		{
			double det = ((b.X - a.X) * (c.Y - a.Y)) - ((b.Y - a.Y) * (c.X - a.X));
			double px = Math.Abs(a.X) + Math.Abs(b.X) + Math.Abs(c.X);
			double py = Math.Abs(a.Y) + Math.Abs(b.Y) + Math.Abs(c.Y);
			return Certify(det, 16.0 * EPS * px * py);
		}

		/// <summary>
		/// Filtered incircle over approximate coordinates (a,b,c CCW ⇒ Pos = d strictly
		/// inside), or null when uncertain. Conservative bound 64ε·P with P the product
		/// of per-axis magnitude sums times the lift magnitude sum.
		/// </summary>
		public static Sign? IncircleA(Vec2 a, Vec2 b, Vec2 c, Vec2 d)
		{
			double adx = a.X - d.X;
			double ady = a.Y - d.Y;
			double bdx = b.X - d.X;
			double bdy = b.Y - d.Y;
			double cdx = c.X - d.X;
			double cdy = c.Y - d.Y;
			double alift = (adx * adx) + (ady * ady);
			double blift = (bdx * bdx) + (bdy * bdy);
			double clift = (cdx * cdx) + (cdy * cdy);
			double det = (alift * ((bdx * cdy) - (cdx * bdy)))
				+ (blift * ((cdx * ady) - (adx * cdy)))
				+ (clift * ((adx * bdy) - (bdx * ady)));

			// Magnitude-based permanent: differences bounded by |p|+|d| sums.
			double mx = Math.Max(Math.Max(Math.Abs(a.X), Math.Abs(b.X)), Math.Abs(c.X)) + Math.Abs(d.X);
			double my = Math.Max(Math.Max(Math.Abs(a.Y), Math.Abs(b.Y)), Math.Abs(c.Y)) + Math.Abs(d.Y);
			double lift = (mx * mx) + (my * my);
			return Certify(det, 64.0 * EPS * lift * mx * my);
		}

		// ─── Tight filters for EXACTLY representable inputs ─────────────────────────
		//
		// The filters above must cover input perturbation: an approximate coordinate
		// x̃ = x(1+δ) carries an error proportional to |x|, so a determinant of points
		// clustered far from the origin is swamped by a bound built from absolute
		// coordinate magnitudes — every such call escalates even though the float
		// arithmetic itself is nearly exact. When all inputs are exactly representable
		// f64 (mesh vertices, and constructed rationals whose rounding happened to be
		// exact) there is NO input error, so the only error is the arithmetic
		// roundoff of the predicate's own evaluation, and a Shewchuk-style permanent
		// built from the COMPUTED differences is sound and vastly tighter: for points
		// clustered within h of each other at distance M from the origin, the incircle
		// bound shrinks from O(M⁴) to O(h⁴).
		//
		// These may only be called when the exact value of every input coordinate
		// equals the f64 passed in. They are otherwise unsound.

		/// <summary>
		/// Filtered orient2d for EXACTLY representable f64 inputs.
		/// </summary>
		/// <remarks>
		/// Error derivation. Write L = fl(fl(ax−cx)·fl(by−cy)) and
		/// R = fl(fl(ay−cy)·fl(bx−cx)); det = fl(L−R). Inputs are exact, so each
		/// difference is the exact difference times (1+e), |e| ≤ ε, and each product
		/// adds one more rounding: L = L*(1+e)³ with L* the exact product, likewise R.
		/// Hence |L−L*| ≤ γ₃|L| and |R−R*| ≤ γ₃|R| with γ₃ = 3ε/(1−3ε). The final
		/// subtraction contributes ε|L−R|. Since |L*−R*| is what we want the sign of,
		/// |det − (L*−R*)| ≤ (γ₃(|L|+|R|))(1+ε) + ε(|L|+|R|) &lt; 4.1ε·(|L|+|R|)
		/// for ε = 2⁻⁵³. We use 8ε for slack (this also absorbs the rounding of the
		/// permanent's own additions).
		/// </remarks>
		public static Sign? Orient2dAExact(Vec2 a, Vec2 b, Vec2 c)
		{
			double left = (a.X - c.X) * (b.Y - c.Y);
			double right = (a.Y - c.Y) * (b.X - c.X);
			double det = left - right;
			double permanent = Math.Abs(left) + Math.Abs(right);
			return Certify(det, 8.0 * EPS * permanent);
		}

		/// <summary>
		/// Filtered incircle for EXACTLY representable f64 inputs (a,b,c CCW ⇒ Pos = d
		/// strictly inside), or null when uncertain.
		/// </summary>
		/// <remarks>
		/// Error derivation. All six differences adx…cdy are exact differences times
		/// (1+e), |e| ≤ ε. Then:
		/// each cross product (e.g. bdx·cdy) is its exact counterpart times (1+e)³,
		/// i.e. relative error ≤ γ₃ ≈ 3ε;
		/// each lift alift = fl(fl(adx²)+fl(ady²)) sums two NON-NEGATIVE terms of
		/// relative error ≤ γ₃ each, so its relative error is ≤ γ₄ ≈ 4ε (no
		/// cancellation);
		/// each 2×2 minor fl(P₁−P₂) has ABSOLUTE error ≤ γ₄·(|P₁|+|P₂|) — the γ₃
		/// relative errors of the two products plus the subtraction's own ε, all
		/// measured against S = |P₁|+|P₂| ≥ |P₁−P₂|;
		/// the term fl(alift·minor) then deviates from the exact alift*·minor* by at
		/// most alift·(γ₄|minor| + γ₄S) + ε|alift·minor| ≤ (2γ₄ + ε + O(ε²))·alift·S
		/// &lt; 9.1ε·alift·S;
		/// the two final additions add ≤ γ₂ ≈ 2ε times the sum of the three terms'
		/// magnitudes, each of which is ≤ alift·S for its own group.
		/// Total: |det − det*| &lt; 11.2ε·permanent, permanent = Σ liftᵢ·Sᵢ. We use 16ε,
		/// which also covers the (relative-ε) error of the computed permanent itself.
		/// </remarks>
		public static Sign? IncircleAExact(Vec2 a, Vec2 b, Vec2 c, Vec2 d)
		{
			double adx = a.X - d.X;
			double ady = a.Y - d.Y;
			double bdx = b.X - d.X;
			double bdy = b.Y - d.Y;
			double cdx = c.X - d.X;
			double cdy = c.Y - d.Y;

			double bdxcdy = bdx * cdy;
			double cdxbdy = cdx * bdy;
			double cdxady = cdx * ady;
			double adxcdy = adx * cdy;
			double adxbdy = adx * bdy;
			double bdxady = bdx * ady;

			double alift = (adx * adx) + (ady * ady);
			double blift = (bdx * bdx) + (bdy * bdy);
			double clift = (cdx * cdx) + (cdy * cdy);

			double det = (alift * (bdxcdy - cdxbdy))
				+ (blift * (cdxady - adxcdy))
				+ (clift * (adxbdy - bdxady));
			double permanent = ((Math.Abs(bdxcdy) + Math.Abs(cdxbdy)) * alift)
				+ ((Math.Abs(cdxady) + Math.Abs(adxcdy)) * blift)
				+ ((Math.Abs(adxbdy) + Math.Abs(bdxady)) * clift);
			return Certify(det, 16.0 * EPS * permanent);
		}

		/// <summary>
		/// Filtered orient3d over approximate coordinates: sign of
		/// dot(cross(b−a, c−a), d−a), or null when uncertain. Conservative 32ε·P.
		/// </summary>
		public static Sign? Orient3dA(Vec3 a, Vec3 b, Vec3 c, Vec3 d)
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
			double det = (((uy * vz) - (uz * vy)) * wx)
				+ (((uz * vx) - (ux * vz)) * wy)
				+ (((ux * vy) - (uy * vx)) * wz);
			double px = Math.Abs(a.X) + Math.Abs(b.X) + Math.Abs(c.X) + Math.Abs(d.X);
			double py = Math.Abs(a.Y) + Math.Abs(b.Y) + Math.Abs(c.Y) + Math.Abs(d.Y);
			double pz = Math.Abs(a.Z) + Math.Abs(b.Z) + Math.Abs(c.Z) + Math.Abs(d.Z);

			// Every det term is a product of one coordinate-difference per axis.
			double p = py * pz * px;
			return Certify(det, 32.0 * EPS * p);
		}

		/// <summary>
		/// Filtered collinear-and-within-segment prefilter for
		/// <see cref="Predicates.PointOnSegmentR"/>: false when p is certifiably NOT on
		/// segment [a,b] (the overwhelmingly common case in the registry sweeps), null
		/// when the exact test must decide. Never returns true — a certified "on" would
		/// need exact zero detection.
		/// </summary>
		public static bool? NotOnSegmentA(Vec3 p, Vec3 a, Vec3 b)
		{
			double apx = p.X - a.X;
			double apy = p.Y - a.Y;
			double apz = p.Z - a.Z;
			double dx = b.X - a.X;
			double dy = b.Y - a.Y;
			double dz = b.Z - a.Z;

			// Cross-product components with magnitude-permanent bounds per component.
			double px = Math.Abs(p.X) + Math.Abs(a.X) + Math.Abs(b.X);
			double py = Math.Abs(p.Y) + Math.Abs(a.Y) + Math.Abs(b.Y);
			double pz = Math.Abs(p.Z) + Math.Abs(a.Z) + Math.Abs(b.Z);
			double cx = (apy * dz) - (apz * dy);
			double cy = (apz * dx) - (apx * dz);
			double cz = (apx * dy) - (apy * dx);
			if (Math.Abs(cx) > 16.0 * EPS * py * pz
				|| Math.Abs(cy) > 16.0 * EPS * pz * px
				|| Math.Abs(cz) > 16.0 * EPS * px * py)
			{
				return false; // certifiably off the line
			}

			// Certifiably outside the parameter range?
			double s1 = (apx * dx) + (apy * dy) + (apz * dz);
			double s2 = (dx * dx) + (dy * dy) + (dz * dz);
			double pd = (px * px) + (py * py) + (pz * pz); // ≥ every dot magnitude here
			double dotErr = 16.0 * EPS * pd;
			if (s1 < -dotErr || s1 > s2 + (2.0 * dotErr))
			{
				return false;
			}

			return null;
		}

		/// <summary>
		/// Certified separating-axis disjointness for two triangles with EXACT f64
		/// vertices (mesh input triangles — no input perturbation, only evaluation
		/// roundoff). Returns true only when some edge-pair cross axis provably
		/// separates the triangles; the face-plane axes are the caller's sign gates.
		/// Division-free: every projection is a degree-3 polynomial with a
		/// magnitude-permanent error bound (conservative 64ε).
		/// </summary>
		/// <remarks>
		/// A false says nothing (the triangles may or may not intersect); a true
		/// certifies empty intersection, letting narrow phases skip all exact work for
		/// the both-straddle-but-miss pairs that otherwise pay full rational interval
		/// construction.
		/// </remarks>
		/// <exception cref="ArgumentException">
		/// Either span is not exactly three vertices. The Rust signature is
		/// <c>&amp;[[f64; 3]; 3]</c>, where the length is the type; a span carries its
		/// length at run time instead, and every read below indexes 0..2 blind.
		/// </exception>
		public static bool SatEdgeAxesDisjoint(ReadOnlySpan<Vec3> t1, ReadOnlySpan<Vec3> t2)
		{
			if (t1.Length != 3 || t2.Length != 3)
			{
				throw new ArgumentException(
					$"SatEdgeAxesDisjoint takes two triangles of three vertices, got {t1.Length} and {t2.Length}");
			}

			// Magnitude bound of every vertex coordinate, per axis.
			Vec3 m = new Vec3(0.0, 0.0, 0.0);
			for (int t = 0; t < 2; t++)
			{
				ReadOnlySpan<Vec3> tri = t == 0 ? t1 : t2;
				for (int i = 0; i < 3; i++)
				{
					Vec3 v = tri[i];
					for (int k = 0; k < 3; k++)
					{
						m[k] = Math.Max(m[k], Math.Abs(v[k]));
					}
				}
			}

			for (int i = 0; i < 3; i++)
			{
				Vec3 e1 = Sub(t1[(i + 1) % 3], t1[i]);
				Vec3 me1 = MagSum(t1[(i + 1) % 3], t1[i]);
				for (int j = 0; j < 3; j++)
				{
					Vec3 e2 = Sub(t2[(j + 1) % 3], t2[j]);
					Vec3 me2 = MagSum(t2[(j + 1) % 3], t2[j]);
					Vec3 axis = new Vec3(
						(e1.Y * e2.Z) - (e1.Z * e2.Y),
						(e1.Z * e2.X) - (e1.X * e2.Z),
						(e1.X * e2.Y) - (e1.Y * e2.X));

					// Component-wise magnitude bound of the (computed) axis.
					Vec3 ma = new Vec3(
						(me1.Y * me2.Z) + (me1.Z * me2.Y),
						(me1.Z * me2.X) + (me1.X * me2.Z),
						(me1.X * me2.Y) + (me1.Y * me2.X));

					// Projection error bound: dot of a degree-2 axis with degree-1
					// coordinates; 64ε·Σ|axis_k|·(2·m_k) is conservative for every
					// rounding along the way.
					double bound = 64.0 * EPS
						* ((ma.X * (2.0 * m.X)) + (ma.Y * (2.0 * m.Y)) + (ma.Z * (2.0 * m.Z)));
					if (!double.IsFinite(bound))
					{
						continue;
					}

					Project(t1, axis, out double lo1, out double hi1);
					Project(t2, axis, out double lo2, out double hi2);
					if (lo1 > hi2 + bound || lo2 > hi1 + bound)
					{
						return true;
					}
				}
			}

			return false;
		}

		private static Vec3 Sub(Vec3 a, Vec3 b)
		{
			return new Vec3(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
		}

		private static Vec3 MagSum(Vec3 a, Vec3 b)
		{
			return new Vec3(
				Math.Abs(a.X) + Math.Abs(b.X),
				Math.Abs(a.Y) + Math.Abs(b.Y),
				Math.Abs(a.Z) + Math.Abs(b.Z));
		}

		/// <summary>The triangle's projection interval onto <paramref name="axis"/>.</summary>
		private static void Project(ReadOnlySpan<Vec3> t, Vec3 axis, out double lo, out double hi)
		{
			lo = double.PositiveInfinity;
			hi = double.NegativeInfinity;
			for (int i = 0; i < 3; i++)
			{
				Vec3 v = t[i];
				double p = (axis.X * v.X) + (axis.Y * v.Y) + (axis.Z * v.Z);
				lo = Math.Min(lo, p);
				hi = Math.Max(hi, p);
			}
		}
	}
}
