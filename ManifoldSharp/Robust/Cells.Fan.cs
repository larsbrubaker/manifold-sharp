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

// Cells.Fan.cs — the radial fan of robust/cells.rs: the sort that puts one
// arrangement edge's incident half-faces in cyclic CCW order, the four filtered
// predicates it runs on, and the error-tracked f64 value those predicates use
// to avoid the exact tier. The module header is on Cells.cs; this file exists
// only so both stay inside the 800-line cap, and it cuts where the Rust's own
// prose already cuts — everything below `radial_fan` is the fan's private
// geometry and has no other caller.
//
// The Rust's private `struct Approx` keeps its name here and the exact tier's
// module is spelled `Exact.Approx` at the one call site in this file; see the
// note in Cells.cs.

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

// Rust's `type Half = (EdgeKey, usize, bool, u32)` from pairing.rs, which names
// the tuple `radial_fan` consumes. C# aliases are file-scoped, so this repeats
// the one Cells.cs declares.
using Half = ((uint A, uint B) Key, int Id, bool Forward, uint Apex);

namespace ManifoldSharp.Robust
{
	public static partial class Cells
	{
		/// <summary>Rust's <c>f64::EPSILON</c>, the unit roundoff the error bounds use.</summary>
		private const double U = 2.220446049250313E-16;

		/// <summary>
		/// Radially sort the incident half-faces of one arrangement edge and group coincident
		/// directions, returning null when fewer than two off-axis faces remain.
		/// </summary>
		/// <remarks>
		/// The cyclic CCW order is derived from filtered orient3d queries against a reference
		/// apex (the first off-axis face) instead of exact coordinates in a rational basis:
		/// classify every face into {reference ray, CCW half, opposite ray, CW half}, then
		/// sort each open half by pairwise orientation. Two faces compare Equal exactly when
		/// their radial directions coincide, so the Equal-runs are the coincident walls. The
		/// starting point of a cyclic order is immaterial to the wedge links, which lets the
		/// whole fan run on the f64 filter in the common case — the rational-basis
		/// construction this replaces dominated cell construction on self-intersecting scans.
		/// <para>
		/// The sort is Rust's <c>sort_by</c>, which is stable, so it is <c>OrderBy</c> here
		/// and not <c>List.Sort</c>: the comparator's last tie-break is the half-face id, and
		/// a degenerate piece can contribute the same id twice to one edge.
		/// </para>
		/// </remarks>
		/// <param name="k0">The edge's first (canonical) endpoint id.</param>
		/// <param name="k1">The edge's second endpoint id.</param>
		/// <param name="raw">The incident half-face records for this edge.</param>
		/// <param name="vt">The vertex tables.</param>
		/// <returns>The sorted half-faces and their equal-direction runs, or null.</returns>
		public static (List<Inc> Incs, List<(int Start, int End)> Groups)? RadialFan(
			uint k0,
			uint k1,
			IReadOnlyList<Half> raw,
			VertTables vt)
		{
			ArgumentNullException.ThrowIfNull(raw);
			List<Inc> incs = new List<Inc>();
			foreach (Half h in raw)
			{
				if (!OnAxis(vt, k0, k1, h.Apex))
				{
					incs.Add(new Inc(h.Id, h.Forward, h.Apex));
				}
			}

			if (incs.Count < 2)
			{
				return null;
			}

			// Class of each face relative to the reference apex: 0 = on the
			// reference ray, 1 = strictly CCW of it (first half turn), 2 = on the
			// opposite ray, 3 = strictly CW (second half turn).
			uint r = incs[0].Apex;

			byte Class(uint apex)
			{
				if (apex == r)
				{
					return 0;
				}

				switch (OrientEdge(vt, k0, k1, r, apex))
				{
					case Sign.Pos:
						return 1;
					case Sign.Neg:
						return 3;
					default:
						switch (SameRaySign(vt, k0, k1, r, apex))
						{
							case Sign.Pos:
								return 0;
							case Sign.Neg:
								return 2;
							default:
								throw new InvalidOperationException(
									"parallel nonzero radial rays have nonzero dot");
						}
				}
			}

			// Order invariance: probe-only. Rust's comment — "`classes` is probe-only" — is
			// the same licence this port's plain Dictionary needs.
			Dictionary<uint, byte> classes = new Dictionary<uint, byte>();
			foreach (Inc inc in incs)
			{
				classes[inc.Apex] = Class(inc.Apex);
			}

			int Compare(Inc a, Inc b)
			{
				byte ca = classes[a.Apex];
				byte cb = classes[b.Apex];
				int c = ca.CompareTo(cb);
				if (c != 0)
				{
					return c;
				}

				if ((ca == 1 || ca == 3) && a.Apex != b.Apex)
				{
					// Within an open half turn, Zero means the same ray (the
					// opposite ray would land in the other class).
					switch (OrientEdge(vt, k0, k1, a.Apex, b.Apex))
					{
						case Sign.Pos:
							return -1;
						case Sign.Neg:
							return 1;
						default:
							break;
					}
				}

				return a.Id.CompareTo(b.Id);
			}

			incs = incs.OrderBy(x => x, Comparer<Inc>.Create(Compare)).ToList();

			// Equal-direction runs become the walls.
			List<(int Start, int End)> groups = new List<(int Start, int End)>();
			int i = 0;
			while (i < incs.Count)
			{
				int j = i + 1;
				while (j < incs.Count)
				{
					byte ci = classes[incs[i].Apex];
					byte cj = classes[incs[j].Apex];
					bool same = ci == cj
						&& (ci == 0
							|| ci == 2
							|| incs[i].Apex == incs[j].Apex
							|| OrientEdge(vt, k0, k1, incs[i].Apex, incs[j].Apex) == Sign.Zero);
					if (!same)
					{
						break;
					}

					j++;
				}

				groups.Add((i, j));
				i = j;
			}

			return (incs, groups);
		}

		/// <summary>
		/// Filtered orientation of apex <paramref name="b"/> against apex <paramref name="a"/>
		/// around the directed edge <paramref name="k0"/> → <paramref name="k1"/>:
		/// <see cref="Sign.Pos"/> means <paramref name="b"/> sits CCW of <paramref name="a"/>
		/// by less than a half turn.
		/// </summary>
		/// <remarks>
		/// The f64 filter (on the cached correctly rounded approximations) certifies almost
		/// every query; only near-coplanar apex pairs escalate to the exact rational
		/// determinant.
		/// </remarks>
		/// <param name="vt">The vertex tables.</param>
		/// <param name="k0">The edge's first endpoint id.</param>
		/// <param name="k1">The edge's second endpoint id.</param>
		/// <param name="a">The reference apex id.</param>
		/// <param name="b">The apex being placed.</param>
		/// <returns>The exact orientation sign.</returns>
		private static Sign OrientEdge(VertTables vt, uint k0, uint k1, uint a, uint b)
		{
			Sign? s = Exact.Approx.Orient3dA(
				vt.VertsF64[(int)k0],
				vt.VertsF64[(int)k1],
				vt.VertsF64[(int)a],
				vt.VertsF64[(int)b]);
			if (s == Sign.Pos || s == Sign.Neg)
			{
				return s.Value;
			}

			return Predicates.Orient3dR(
				vt.Verts[(int)k0],
				vt.Verts[(int)k1],
				vt.Verts[(int)a],
				vt.Verts[(int)b]);
		}

		/// <summary>
		/// Exact radial direction of <paramref name="a"/>'s apex about edge
		/// <paramref name="k0"/> → <paramref name="k1"/>: the cross product
		/// <c>(k1−k0) × (a−k0)</c>. Zero iff the apex lies on the edge's axis.
		/// </summary>
		/// <param name="vt">The vertex tables.</param>
		/// <param name="k0">The edge's first endpoint id.</param>
		/// <param name="k1">The edge's second endpoint id.</param>
		/// <param name="a">The apex id.</param>
		/// <returns>The exact radial direction.</returns>
		private static R3 RadialCross(VertTables vt, uint k0, uint k1, uint a)
		{
			R3 w = vt.Verts[(int)k1].Sub(vt.Verts[(int)k0]);
			R3 d = vt.Verts[(int)a].Sub(vt.Verts[(int)k0]);
			return w.Cross(d);
		}

		/// <summary>The three components of <c>(k1−k0) × (a−k0)</c> with error bounds.</summary>
		/// <param name="vt">The vertex tables.</param>
		/// <param name="k0">The edge's first endpoint id.</param>
		/// <param name="k1">The edge's second endpoint id.</param>
		/// <param name="a">The apex id.</param>
		/// <returns>The three bounded components, x then y then z.</returns>
		private static Approx[] RadialCrossA(VertTables vt, uint k0, uint k1, uint a)
		{
			Approx[] p0 = InputPoint(vt, k0);
			Approx[] p1 = InputPoint(vt, k1);
			Approx[] pa = InputPoint(vt, a);
			Approx[] w = { p1[0].Sub(p0[0]), p1[1].Sub(p0[1]), p1[2].Sub(p0[2]) };
			Approx[] d = { pa[0].Sub(p0[0]), pa[1].Sub(p0[1]), pa[2].Sub(p0[2]) };
			return new Approx[]
			{
				w[1].Mul(d[2]).Sub(w[2].Mul(d[1])),
				w[2].Mul(d[0]).Sub(w[0].Mul(d[2])),
				w[0].Mul(d[1]).Sub(w[1].Mul(d[0])),
			};
		}

		/// <summary>Rust's <c>p</c> closure: one vertex's three bounded coordinates.</summary>
		/// <param name="vt">The vertex tables.</param>
		/// <param name="v">The vertex id.</param>
		/// <returns>Its coordinates as correctly rounded inputs.</returns>
		private static Approx[] InputPoint(VertTables vt, uint v)
		{
			Vec3 p = vt.VertsF64[(int)v];
			return new Approx[] { Approx.Input(p.X), Approx.Input(p.Y), Approx.Input(p.Z) };
		}

		/// <summary>
		/// Is the apex on the edge's axis (a degenerate sliver with no wedge)? A certifiably
		/// nonzero cross component proves off-axis; only near-axis apexes pay for the exact
		/// cross.
		/// </summary>
		/// <param name="vt">The vertex tables.</param>
		/// <param name="k0">The edge's first endpoint id.</param>
		/// <param name="k1">The edge's second endpoint id.</param>
		/// <param name="a">The apex id.</param>
		/// <returns>True when the apex lies on the edge's axis.</returns>
		private static bool OnAxis(VertTables vt, uint k0, uint k1, uint a)
		{
			foreach (Approx c in RadialCrossA(vt, k0, k1, a))
			{
				if (c.SignOf() is not null)
				{
					return false;
				}
			}

			return RadialCross(vt, k0, k1, a).IsZero();
		}

		/// <summary>
		/// For two apexes whose radial directions are exactly parallel
		/// (<see cref="OrientEdge"/> returned <see cref="Sign.Zero"/>), do they point the same
		/// way (<see cref="Sign.Pos"/>, a coincident stack) or opposite ways
		/// (<see cref="Sign.Neg"/>, a fold)?
		/// </summary>
		/// <remarks>
		/// Sign of the dot of the two radial crosses; exact fallback only when the
		/// error-tracked f64 dot cannot certify.
		/// </remarks>
		/// <param name="vt">The vertex tables.</param>
		/// <param name="k0">The edge's first endpoint id.</param>
		/// <param name="k1">The edge's second endpoint id.</param>
		/// <param name="a">The first apex id.</param>
		/// <param name="b">The second apex id.</param>
		/// <returns>The sign of the dot of the two radial directions.</returns>
		private static Sign SameRaySign(VertTables vt, uint k0, uint k1, uint a, uint b)
		{
			Approx[] ca = RadialCrossA(vt, k0, k1, a);
			Approx[] cb = RadialCrossA(vt, k0, k1, b);
			Approx dot = ca[0].Mul(cb[0]).Add(ca[1].Mul(cb[1])).Add(ca[2].Mul(cb[2]));
			Sign? s = dot.SignOf();
			if (s is not null)
			{
				return s.Value;
			}

			R3 ea = RadialCross(vt, k0, k1, a);
			R3 eb = RadialCross(vt, k0, k1, b);
			BigRational exact = (ea.X * eb.X) + (ea.Y * eb.Y) + (ea.Z * eb.Z);
			return SignFunctions.OfRat(exact);
		}

		/// <summary>
		/// A value with a rigorous absolute error bound, tracked through the few operations
		/// the fan filters need.
		/// </summary>
		/// <remarks>
		/// Inputs are correctly rounded f64 approximations of exact rationals
		/// (error ≤ 0.5 ulp ≤ u·|x|), and every operation adds its own rounding term — the
		/// bound is conservative by construction, so a certified sign is exact. This matters
		/// because the fan geometry routinely subtracts nearly equal coordinates, where the
		/// input rounding error dwarfs a magnitude bound taken on the *differences*.
		/// </remarks>
		private readonly struct Approx
		{
			/// <summary>The approximate value.</summary>
			public readonly double V;

			/// <summary>Its rigorous absolute error bound.</summary>
			public readonly double Err;

			private Approx(double v, double err)
			{
				this.V = v;
				this.Err = err;
			}

			/// <summary>A correctly rounded approximation of an exact value.</summary>
			/// <param name="v">The rounded value.</param>
			/// <returns>The bounded input.</returns>
			public static Approx Input(double v)
			{
				return new Approx(v, U * Math.Abs(v));
			}

			/// <summary>Difference, with the rounding term added to the bound.</summary>
			/// <param name="o">The subtrahend.</param>
			/// <returns>The bounded difference.</returns>
			public Approx Sub(Approx o)
			{
				double v = this.V - o.V;
				return new Approx(v, this.Err + o.Err + (U * Math.Abs(v)));
			}

			/// <summary>Product, with the rounding term added to the bound.</summary>
			/// <param name="o">The multiplier.</param>
			/// <returns>The bounded product.</returns>
			public Approx Mul(Approx o)
			{
				double v = this.V * o.V;
				return new Approx(
					v,
					(Math.Abs(this.V) * o.Err) + (this.Err * Math.Abs(o.V)) + (this.Err * o.Err)
						+ (U * Math.Abs(v)));
			}

			/// <summary>Sum, with the rounding term added to the bound.</summary>
			/// <param name="o">The addend.</param>
			/// <returns>The bounded sum.</returns>
			public Approx Add(Approx o)
			{
				double v = this.V + o.V;
				return new Approx(v, this.Err + o.Err + (U * Math.Abs(v)));
			}

			/// <summary>
			/// The certified sign, or null when the error bound straddles zero. Named
			/// <c>SignOf</c> rather than the Rust's <c>sign</c> because <see cref="Sign"/> is
			/// the enum it returns.
			/// </summary>
			/// <returns>The sign, or null when uncertified.</returns>
			public Sign? SignOf()
			{
				if (Math.Abs(this.V) > this.Err)
				{
					return this.V > 0.0 ? Sign.Pos : Sign.Neg;
				}

				return null;
			}
		}
	}
}
