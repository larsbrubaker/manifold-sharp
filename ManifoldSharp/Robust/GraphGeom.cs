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

// GraphGeom.cs — port of robust/graph_geom.rs, whose header reads:
//
//   Geometric helpers of the intersection-graph build: triangle boxes and
//   degeneracy, the filtered point-on-segment test used by the split
//   registries, and the exact coplanar clip/containment tests used by the
//   cross-copy step.
//
//   Split out of robust/intersection_graph.rs (its only caller besides
//   robust/graph_self_cut.rs and robust/soup.rs, which reach `tri_box` /
//   `is_degenerate` through the `intersection_graph` re-exports). The exact
//   predicates themselves live in robust/exact/{approx,predicates}.rs.
//
// ── `[f64; 3]` is a Vec3 here ────────────────────────────────────────────────
// The Rust passes the approximations around as bare `[f64; 3]`; this port's
// exact tier already settled on Vec3 for that shape (Approx.Orient3dA,
// Approx.NotOnSegmentA), so Approx3 returns a Vec3 and SegBox3's
// `[[f64; 3]; 2]` becomes a (Lo, Hi) pair of them. Vec3's indexer keeps the
// `for k in 0..3` loops transcribable.
//
// ── `&[Vec3; 3]` is a length-checked Vec3[] ──────────────────────────────────
// Same call as TriTri.cs made for the same reason: a C# array carries its length
// at run time instead of in its type, and Soup.ImplToTris hands triangles out as
// Vec3[] already, so the check is at the boundary and the callers pass through.

using System.Diagnostics;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp.Robust
{
	/// <summary>
	/// The free functions of <c>robust/graph_geom.rs</c>: triangle boxes and degeneracy,
	/// the filtered point-on-segment test, and the exact coplanar clip and containment
	/// tests.
	/// </summary>
	internal static class GraphGeom
	{
		/// <summary>Rust's <c>f64::EPSILON * 0.5</c>, the half-ulp unit of the error bounds.</summary>
		private const double EPS = 2.220446049250313E-16 * 0.5;

		/// <summary>The smallest positive normal f64 (Rust's <c>f64::MIN_POSITIVE</c>).</summary>
		private const double MinPositive = 2.2250738585072014E-308;

		/// <summary>The axis-aligned bounding box of a triangle.</summary>
		/// <param name="t">The triangle's three vertices.</param>
		/// <returns>Their bounding box.</returns>
		/// <exception cref="ArgumentException">The array is not exactly three vertices.</exception>
		internal static Box TriBox(Vec3[] t)
		{
			RequireTriangle(t);
			Box b = Box.FromPoints(t[0], t[1]);
			b.UnionPoint(t[2]);
			return b;
		}

		/// <summary>Exactly zero-area triangle test.</summary>
		/// <param name="t">The triangle's three vertices.</param>
		/// <returns>True when the triangle has exactly zero area.</returns>
		/// <exception cref="ArgumentException">The array is not exactly three vertices.</exception>
		internal static bool IsDegenerate(Vec3[] t)
		{
			RequireTriangle(t);

			// Certified-nonzero f64 cross first (magnitude-permanent bound, matching
			// exact/approx.rs conventions); only near-degenerate triangles pay for
			// the rational cross.
			Vec3 u = t[1] - t[0];
			Vec3 v = t[2] - t[0];
			Vec3 n = Cross(u, v);
			double mx = M(0);
			double my = M(1);
			double mz = M(2);
			if (Math.Abs(n.X) > 16.0 * EPS * my * mz
				|| Math.Abs(n.Y) > 16.0 * EPS * mz * mx
				|| Math.Abs(n.Z) > 16.0 * EPS * mx * my)
			{
				return false;
			}

			return Predicates.TriNormalR(
				R3.FromVec3(t[0]),
				R3.FromVec3(t[1]),
				R3.FromVec3(t[2]))
				.IsZero();

			double M(int k)
			{
				return Math.Abs(t[0][k]) + Math.Abs(t[1][k]) + Math.Abs(t[2][k]);
			}
		}

		/// <summary>Exact: p collinear with (a,b) and within the closed segment.</summary>
		/// <param name="p">The candidate point.</param>
		/// <param name="a">One segment endpoint.</param>
		/// <param name="b">The other segment endpoint.</param>
		/// <returns>True when p lies on the closed segment.</returns>
		internal static bool PointOnSegment(R3 p, R3 a, R3 b)
		{
			return Predicates.PointOnSegmentR(p, a, b);
		}

		/// <summary>
		/// Correctly rounded f64 approximation of an exact point (relative error ≤ ε per
		/// coordinate) for the semi-static prefilters in exact/approx.rs.
		/// </summary>
		/// <param name="p">The exact point.</param>
		/// <returns>Its correctly rounded approximation.</returns>
		internal static Vec3 Approx3(R3 p)
		{
			return new Vec3(
				Rational.RatToF64(p.X),
				Rational.RatToF64(p.Y),
				Rational.RatToF64(p.Z));
		}

		/// <summary>
		/// Conservative 3D box around a segment's exact endpoints from their correctly
		/// rounded approximations, inflated past rounding error — a point exactly on the
		/// segment is never rejected by testing its approximation against this box.
		/// Mirrors the 2D prefilter in robust/arrangement.rs; the registry sweeps in the
		/// build pipeline were quadratic in exact comparisons without it.
		/// </summary>
		/// <param name="a">One endpoint's approximation.</param>
		/// <param name="b">The other endpoint's approximation.</param>
		/// <returns>The inflated (lo, hi) corners.</returns>
		internal static (Vec3 Lo, Vec3 Hi) SegBox3(Vec3 a, Vec3 b)
		{
			Vec3 lo = default;
			Vec3 hi = default;
			for (int k = 0; k < 3; k++)
			{
				// The Rust is `a[k].min(b[k])` / `.max(..)`, i.e. f64::min / f64::max,
				// where a NaN operand loses and the other one is returned. That is
				// LinalgFunctions.MinF64 / MaxF64 here, never Math.Min / Math.Max, which
				// propagate NaN and would let a single NaN coordinate poison a bound the
				// Rust would have kept finite (docs/RUST_DIVERGENCES.md entry 2).
				double l = MinF64(a[k], b[k]);
				double h = MaxF64(a[k], b[k]);
				lo[k] = l - Pad(l);
				hi[k] = h + Pad(h);
			}

			return (lo, hi);

			static double Pad(double x)
			{
				return (Math.Abs(x) * 1e-15) + MinPositive;
			}
		}

		/// <summary>Closed containment of a point in a <see cref="SegBox3"/> box.</summary>
		/// <param name="b">The box.</param>
		/// <param name="p">The point.</param>
		/// <returns>True when the point is inside or on the box.</returns>
		internal static bool Box3Contains((Vec3 Lo, Vec3 Hi) b, Vec3 p)
		{
			for (int k = 0; k < 3; k++)
			{
				if (!(p[k] >= b.Lo[k] && p[k] <= b.Hi[k]))
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// Filtered point-on-segment: the approx prefilter rejects the generic case without
		/// touching the bignum tier; only near-incidences run the exact test.
		/// </summary>
		/// <param name="pApprox">The candidate point's approximation.</param>
		/// <param name="p">The candidate point.</param>
		/// <param name="aApprox">One endpoint's approximation.</param>
		/// <param name="a">One endpoint.</param>
		/// <param name="bApprox">The other endpoint's approximation.</param>
		/// <param name="b">The other endpoint.</param>
		/// <returns>True when p lies on the closed segment.</returns>
		internal static bool PointOnSegmentF(
			Vec3 pApprox,
			R3 p,
			Vec3 aApprox,
			R3 a,
			Vec3 bApprox,
			R3 b)
		{
			// Rust: `match not_on_segment_a(..) { Some(false) => false, _ => exact }`.
			// The filter never certifies a *hit* (it returns Some(true) for nothing), so
			// only the certified miss short-circuits; None falls through to the exact test.
			bool? filter = Approx.NotOnSegmentA(pApprox, aApprox, bApprox);
			if (filter == false)
			{
				return false;
			}

			return PointOnSegment(p, a, b);
		}

		/// <summary>
		/// Clip segment (a,b) to a convex coplanar polygon (2D test via projection on the
		/// polygon's own plane). Returns a positive-length sub-segment or null. Used to
		/// cross-copy primitives into coplanar overlap regions.
		/// </summary>
		/// <param name="a">The segment's start.</param>
		/// <param name="b">The segment's end.</param>
		/// <param name="poly">The convex coplanar polygon, at least three vertices.</param>
		/// <returns>The clipped sub-segment, or null when the clip is empty.</returns>
		internal static (R3 A, R3 B)? ClipSegmentToPolygon(R3 a, R3 b, IReadOnlyList<R3> poly)
		{
			Debug.Assert(poly.Count >= 3, "clip_segment_to_polygon: polygon needs three vertices");
			R3 n = Predicates.TriNormalR(poly[0], poly[1], poly[2]);
			int axis = TriTri.DominantAxis(n);
			List<R2> pts2 = new List<R2>(poly.Count);
			for (int i = 0; i < poly.Count; i++)
			{
				pts2.Add(poly[i].ProjectDrop(axis));
			}

			if (Predicates.Orient2dR(pts2[0], pts2[1], pts2[2]) == Sign.Neg)
			{
				pts2.Reverse();
			}

			R2 a2 = a.ProjectDrop(axis);
			R2 b2 = b.ProjectDrop(axis);
			R2 dir = b2.Sub(a2);

			// Parametric clip of [0,1] against each CCW edge halfplane.
			BigRational t0 = Backend.RatZero();
			BigRational t1 = Backend.RatOne();
			for (int i = 0; i < pts2.Count; i++)
			{
				R2 e0 = pts2[i];
				R2 e1 = pts2[(i + 1) % pts2.Count];
				R2 edge = e1.Sub(e0);

				// Signed distance numerators of a2 + t*dir against the edge line:
				// f(t) = cross(edge, a2 + t*dir - e0) = fa + t * fd.
				BigRational fa = edge.Cross(a2.Sub(e0));
				BigRational fd = edge.Cross(dir);
				if (Backend.RatIsZero(fd))
				{
					if (fa < Backend.RatZero())
					{
						return null; // parallel and strictly outside
					}

					continue;
				}

				BigRational tHit = -fa / fd;
				if (fd > Backend.RatZero())
				{
					// entering: f grows with t → require t >= t_hit
					if (tHit > t0)
					{
						t0 = tHit;
					}
				}
				else if (tHit < t1)
				{
					t1 = tHit;
				}

				if (t0 >= t1)
				{
					return null;
				}
			}

			if (t0 >= t1)
			{
				return null;
			}

			return (Seg(t0), Seg(t1));

			R3 Seg(in BigRational t)
			{
				return a.Add(b.Sub(a).Scale(t));
			}
		}

		/// <summary>Exact point-in-convex-polygon test for a point on the polygon's plane.</summary>
		/// <param name="p">The point, assumed on the polygon's plane.</param>
		/// <param name="poly">The convex polygon, at least three vertices.</param>
		/// <returns>True when the point is inside or on the polygon.</returns>
		internal static bool PointInPolygonCoplanar(R3 p, IReadOnlyList<R3> poly)
		{
			R3 n = Predicates.TriNormalR(poly[0], poly[1], poly[2]);
			int axis = TriTri.DominantAxis(n);
			List<R2> pts2 = new List<R2>(poly.Count);
			for (int i = 0; i < poly.Count; i++)
			{
				pts2.Add(poly[i].ProjectDrop(axis));
			}

			if (Predicates.Orient2dR(pts2[0], pts2[1], pts2[2]) == Sign.Neg)
			{
				pts2.Reverse();
			}

			R2 p2 = p.ProjectDrop(axis);
			for (int i = 0; i < pts2.Count; i++)
			{
				if (Predicates.Orient2dR(pts2[i], pts2[(i + 1) % pts2.Count], p2) == Sign.Neg)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// The length check that stands in for Rust's <c>&amp;[Vec3; 3]</c> parameter type.
		/// </summary>
		/// <param name="t">The candidate triangle.</param>
		/// <exception cref="ArgumentException">The array is not exactly three vertices.</exception>
		private static void RequireTriangle(Vec3[] t)
		{
			ArgumentNullException.ThrowIfNull(t);
			if (t.Length != 3)
			{
				throw new ArgumentException($"a triangle is three vertices, got {t.Length}", nameof(t));
			}
		}
	}
}
