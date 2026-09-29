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

// CrossSection.ClipperD.cs — the double-precision layer under
// CrossSection.Clipper.cs: the hand-rolled power-of-two ClipperD (flat and
// PolyTree), Clipper2's Area over a PathD, and the Polygons<->PathsD
// conversions.
//
// It is the second of the two files that name Clipper2Lib; the dependency is
// still confined to the CrossSection partials, as the Rust confines
// clipper2-rust to cross_section.rs and its child module cross_section_ops.rs.
// The split from CrossSection.Clipper.cs is C#-only, made when the PolyTree
// port of Decompose pushed that file past the 800-line cap: the operations stay
// there, the machinery they share lives here. See
// CrossSection.cs for the whole file split.
//
// ── PolyTree: Clipper2Lib 1.5.4 versus Clipper2 46f6391 ─────────────────────
// Decompose and Simplify read the containment tree of a union, which the Rust
// gets from clipper2-rust's boolean_op_tree_d (a PolyTree64 built by
// recursive_check_owners, then every node scaled by inv_scale in child order).
// UnionTree below builds the same thing from Clipper2Lib's Clipper64 and
// PolyTree64 at the booleans' power-of-two scale, and the walks scale each
// node's Path64 exactly as ScaleToDouble does. Decompiled against Clipper2
// 46f6391's clipper.engine.cpp, Clipper2Lib 1.5.4's BuildTree, CheckBounds,
// CheckSplitOwner, Path1InsidePath2 (C++ Path2ContainsPath1) and
// PolyPath64.AddChild (append) are the C++ step for step, with one exception:
// its RecursiveCheckOwners loop drops the C++'s
// `owner->bounds.Contains(outrec->bounds)` pre-test before the
// point-in-polygon test. The two walks accept the same owner unless the C#
// accepts one whose bounds do not contain the child's; so any tree whose every
// parent's bounds contain each child's bounds is the C++ tree (and the Rust's).
// Clipper's output rings do not cross, so a ring whose bounds another's do not
// contain lies outside that ring with no vertex inside it; the difference can
// only arise on the rounding micro-intersections the C++ Path2ContainsPath1
// comment mentions. Restoring the pre-test in a decompiled Clipper2Lib changed
// none of the twinning harness's outputs. Recorded in docs/FOLLOW_UPS.md.

using Clipper2Lib;

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	public sealed partial class CrossSection
	{
		// ─────────────────────────────────────────────────────────────────────────
		// The double-precision boolean layer, hand-rolled.
		//
		// The Rust calls clipper2-rust's union_d/intersect_d/difference_d, and the
		// obvious transcription is Clipper.Union/Intersect/Difference on the PathsD
		// overloads. That transcription is WRONG, and silently so: it returns
		// well-formed polygons with the right contour count in the right order, whose
		// coordinates are on a different grid.
		//
		// Both libraries do the same thing in outline — scale the doubles to int64,
		// run the integer engine, scale back — but they disagree on the scale:
		//
		//   Clipper2Lib 1.5.4 (and 2.0.0):   _scale = Math.Pow(10, precision)
		//                                           = 1e8         at precision 8
		//   Upstream C++ Clipper2 (issue
		//   #25, "set the scale to a power
		//   of double's radix"; ClipperD at
		//   46f6391, the commit C++ Manifold
		//   pins), and hence clipper2-rust
		//   1.0.3 / 1.1.0:                    scale = 2^(ilogb(10^precision) + 1)
		//                                           = 2^27        at precision 8
		//
		// The C# port simply has not taken that upstream change yet. Minimal repro,
		// pinned by CrossSectionTests.ClipperDScaleIsPowerOfTwo — Union of
		// [(0,0), (1,0), (0.1234567890123, 1)] against an empty clip, NonZero,
		// precision 8, gives for that third x:
		//
		//   Clipper2Lib's ClipperD    0.12345679          bits 0x3fbf9add3b84e659
		//   clipper2-rust (and this)  0.12345679104328156 bits 0x3fbf9add40000000
		//
		// So these three do the D-layer themselves against the Paths64 overloads.
		// That is not a divergence from the Rust — it is what reaches the Rust's
		// specified numbers, which is why there is no docs/RUST_DIVERGENCES.md entry
		// for it. A differential harness of 101 cases against the compiled
		// manifold-rust goes from 12 mismatches to 0 with this in place.
		//
		// Only the booleans need it. InflatePaths, Minkowski.Sum and SimplifyPaths
		// take an explicit decimal-places argument and scale by 10^decimals on both
		// sides — as upstream C++ InflatePaths and MinkowskiSum do too — so those keep
		// the stock double API above and already match.
		//
		// If Clipper2Lib ever adopts the upstream power-of-two scale, this wrapper
		// becomes redundant and the three methods can collapse back to the PathsD
		// overloads — but only after re-running the differential harness against the
		// Rust and seeing 101/101 again. "The changelog says they fixed it" is not
		// evidence; the harness is.
		// ─────────────────────────────────────────────────────────────────────────

		/// <summary>
		/// clipper2-rust's <c>MAX_COORD</c> (core.rs:89) as a double: the largest
		/// coordinate the integer engine accepts, <c>long.MaxValue &gt;&gt; 2</c>. The
		/// conversion is not exact — 2305843009213693951 rounds to ...952 — which is
		/// true of the Rust's <c>MAX_COORD as f64</c> too, so the bound is the same
		/// double on both sides.
		/// </summary>
		private static readonly double MaxCoordD = long.MaxValue >> 2;

		/// <summary>clipper2-rust's <c>MIN_COORD</c> (core.rs:91): the negation of <see cref="MaxCoordD"/>.</summary>
		private static readonly double MinCoordD = -(double)(long.MaxValue >> 2);

		/// <summary>
		/// clipper2-rust's <c>ClipperD::new</c> scale: a power of two rather than of ten,
		/// so that scaling in and back out is exact in binary floating point.
		/// </summary>
		/// <param name="precision">The decimal places, as passed to the Rust.</param>
		/// <returns>2 raised to <c>ilogb(10^precision) + 1</c>.</returns>
		private static double BooleanScale(int precision)
		{
			// The porting source is clipper2-rust, which spells this
			// `(10f64.powi(prec)).log2().floor() as i32` (engine_public.rs:483); upstream
			// C++ spells the same thing `std::ilogb(std::pow(10, precision))`. Math.ILogB
			// is the C++ spelling, and is used here because it is exact by construction —
			// it reads the binary exponent rather than computing a logarithm that then
			// has to be floored, so there is no input on which the two can disagree by a
			// rounding error at a power-of-two boundary. Verified equal to the Rust for
			// every precision the library accepts.
			return Math.Pow(2, Math.ILogB(Math.Pow(10, precision)) + 1);
		}

		/// <summary>
		/// Scales doubles up to the integer grid, the Rust's
		/// <c>scale_paths::&lt;i64, f64&gt;</c>.
		/// </summary>
		/// <remarks>
		/// <c>MidpointRounding.AwayFromZero</c> is required, not stylistic: the Rust's
		/// <c>i64::from_f64</c> is <c>val.round() as i64</c>, and Rust's
		/// <c>f64::round</c> rounds halves away from zero. C#'s <c>Math.Round(double)</c>
		/// defaults to banker's rounding (to even), which disagrees on every exact .5 —
		/// and exact halves are the common case here, not the exotic one, because the
		/// scale is a power of two.
		/// </remarks>
		private static Paths64 ScaleToInt(PathsD paths, double scale)
		{
			// Range check, from scale_paths (core.rs:1319-1360). The Rust runs it only for
			// integral output types, which is exactly this direction. Note it tests the
			// extremes of the whole set, not each point, and drops *everything* on
			// failure — a single out-of-range vertex empties the result, and the caller
			// then sees an empty boolean result rather than a wrapped coordinate.
			//
			// The Rust also ORs RANGE_ERROR_I into its error_code out-parameter. There is
			// no such side channel here and nothing in cross_section.rs reads it: the Rust
			// discards the code at every call site in that file, so the empty return IS
			// the observable behavior being ported.
			double xmin = double.MaxValue;
			double ymin = double.MaxValue;
			double xmax = double.MinValue;
			double ymax = double.MinValue;
			foreach (PathD path in paths)
			{
				foreach (PointD p in path)
				{
					if (p.x < xmin)
					{
						xmin = p.x;
					}

					if (p.x > xmax)
					{
						xmax = p.x;
					}

					if (p.y < ymin)
					{
						ymin = p.y;
					}

					if (p.y > ymax)
					{
						ymax = p.y;
					}
				}
			}

			if ((xmin * scale) < MinCoordD
				|| (xmax * scale) > MaxCoordD
				|| (ymin * scale) < MinCoordD
				|| (ymax * scale) > MaxCoordD)
			{
				return new Paths64();
			}

			Paths64 result = new Paths64(paths.Count);
			foreach (PathD path in paths)
			{
				Path64 scaled = new Path64(path.Count);
				foreach (PointD p in path)
				{
					scaled.Add(new Point64(
						(long)Math.Round(p.x * scale, MidpointRounding.AwayFromZero),
						(long)Math.Round(p.y * scale, MidpointRounding.AwayFromZero)));
				}

				result.Add(scaled);
			}

			return result;
		}

		/// <summary>
		/// Scales the integer solution back down, the Rust's <c>build_paths_d</c>.
		/// Multiplication by the reciprocal, not division by the scale — the Rust keeps
		/// <c>inv_scale = 1.0 / scale</c> and multiplies, and with a power-of-two scale
		/// both the reciprocal and the product are exact.
		/// </summary>
		private static PathsD ScaleToDouble(Paths64 paths, double invScale)
		{
			PathsD result = new PathsD(paths.Count);
			foreach (Path64 path in paths)
			{
				PathD scaled = new PathD(path.Count);
				foreach (Point64 p in path)
				{
					scaled.Add(new PointD(p.X * invScale, p.Y * invScale));
				}

				result.Add(scaled);
			}

			return result;
		}

		/// <summary>The Rust free function <c>union_d</c>.</summary>
		private static PathsD UnionD(PathsD subjects, PathsD clips, FillRule fillRule, int precision)
		{
			double scale = BooleanScale(precision);
			return ScaleToDouble(
				Clipper.Union(ScaleToInt(subjects, scale), ScaleToInt(clips, scale), fillRule),
				1.0 / scale);
		}

		/// <summary>The Rust free function <c>intersect_d</c>.</summary>
		private static PathsD IntersectD(PathsD subjects, PathsD clips, FillRule fillRule, int precision)
		{
			double scale = BooleanScale(precision);
			return ScaleToDouble(
				Clipper.Intersect(ScaleToInt(subjects, scale), ScaleToInt(clips, scale), fillRule),
				1.0 / scale);
		}

		/// <summary>The Rust free function <c>difference_d</c>.</summary>
		private static PathsD DifferenceD(PathsD subjects, PathsD clips, FillRule fillRule, int precision)
		{
			double scale = BooleanScale(precision);
			return ScaleToDouble(
				Clipper.Difference(ScaleToInt(subjects, scale), ScaleToInt(clips, scale), fillRule),
				1.0 / scale);
		}

		/// <summary>
		/// The Rust free function <c>path_area</c>: its <c>clipper2_area_by</c>, an exact
		/// port of Clipper2's <c>Area(const Path&lt;T&gt;&amp;)</c> (clipper.core.h at commit
		/// 46f6391, the version C++ Manifold pins), over a Clipper <c>PathD</c>.
		/// </summary>
		/// <remarks>
		/// Clipper2 walks the trapezoid form over edges (n-1,0), (0,1), ..., (n-2,n-1),
		/// accumulating <c>(prev.y + cur.y) * (prev.x - cur.x)</c> in that order; its
		/// two-edges-per-step unrolling does not change the order. clipper2-rust's
		/// <c>area</c> is a plain shoelace instead, which rounds differently in the last
		/// bits — over 20,000 random polygons the two disagreed on 13,544, always by an
		/// ulp — so the Rust stopped calling it. <see cref="Simplify"/> feeds this straight
		/// into a <c>&gt;</c> comparison against <c>maxSize * epsilon</c>, where one ulp
		/// decides whether a contour survives, so the summation order is load-bearing.
		/// <para>
		/// Clipper2Lib 1.5.4's <c>Clipper.Area(PathD)</c> decompiles to this same loop in
		/// this same order, so it would agree today; this stays hand-written because it is
		/// the same function as CrossSection.cs's <c>ContourArea</c> — the Rust's one
		/// helper over two point types — and the two must change together.
		/// </para>
		/// </remarks>
		/// <param name="path">The contour.</param>
		/// <returns>The signed area, positive for counter-clockwise.</returns>
		private static double PathArea(PathD path)
		{
			int cnt = path.Count;
			if (cnt < 3)
			{
				return 0.0;
			}

			double a = 0.0;
			int prev = cnt - 1;
			for (int cur = 0; cur < cnt; cur++)
			{
				a += (path[prev].y + path[cur].y) * (path[prev].x - path[cur].x);
				prev = cur;
			}

			return a * 0.5;
		}

		/// <summary>The Rust free function <c>to_paths</c>: Polygons to Clipper's PathsD.</summary>
		private static PathsD ToPaths(Polygons polygons)
		{
			PathsD paths = new PathsD(polygons.Count);
			foreach (SimplePolygon poly in polygons)
			{
				PathD path = new PathD(poly.Count);
				foreach (Vec2 p in poly)
				{
					path.Add(new PointD(p.X, p.Y));
				}

				paths.Add(path);
			}

			return paths;
		}

		/// <summary>The Rust free function <c>from_paths</c>: Clipper's PathsD to Polygons.</summary>
		private static Polygons FromPaths(PathsD paths)
		{
			Polygons polygons = new Polygons(paths.Count);
			foreach (PathD path in paths)
			{
				SimplePolygon poly = new SimplePolygon(path.Count);
				foreach (PointD p in path)
				{
					poly.Add(new Vec2(p.x, p.y));
				}

				polygons.Add(poly);
			}

			return polygons;
		}

		/// <summary>
		/// The Rust's <c>boolean_op_tree_d(ClipType::Union, fill_rule, subjects,
		/// &amp;PathsD::new(), &amp;mut tree, precision)</c>, as the integer tree plus the
		/// factor that scales its nodes back: clipper2-rust builds a PolyTree64 and
		/// converts each node with <c>scale_path(path, inv_scale)</c>, which
		/// <see cref="TreeNodePath"/> does at the node's visit instead — the same
		/// product per coordinate. See the file header for how Clipper2Lib's tree is
		/// the C++ one.
		/// </summary>
		private static (PolyTree64 Tree, double InvScale) UnionTree(PathsD subjects, FillRule fillRule, int precision)
		{
			double scale = BooleanScale(precision);
			PolyTree64 tree = new PolyTree64();
			Clipper.BooleanOp(ClipType.Union, ScaleToInt(subjects, scale), new Paths64(), tree, fillRule);
			return (tree, 1.0 / scale);
		}

		/// <summary>A PolyTree node's contour back on the double grid, as <see cref="ScaleToDouble"/> scales.</summary>
		private static PathD TreeNodePath(PolyPath64 node, double invScale)
		{
			Path64 path = node.Polygon!;
			PathD scaled = new PathD(path.Count);
			foreach (Point64 p in path)
			{
				scaled.Add(new PointD(p.X * invScale, p.Y * invScale));
			}

			return scaled;
		}
	}
}
