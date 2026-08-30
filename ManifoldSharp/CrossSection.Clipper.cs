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

// CrossSection.Clipper.cs — the Clipper2-backed half of cross_section.rs.
//
// This is the only file in the assembly that names Clipper2Lib, which is the
// whole dependency budget of the port (PORTING_PLAN.md's dependency table) made
// structural: cross_section.rs is likewise the only Rust file importing
// clipper2-rust. See CrossSection.cs for the file split.
//
// ── Why Clipper2Lib 1.5.4, and why the booleans bypass its D-layer ───────────
// clipper2-rust 1.0.3 is a port of upstream Clipper2 1.5.4 (its version.rs says
// so), and the NuGet Clipper2 package is that same upstream release by the same
// author. The version pin in the csproj is load-bearing; see the comment there.
//
// Matching versions are not sufficient. Clipper2Lib's `ClipperD` — the double
// wrapper behind Clipper.Union/Intersect/Difference(PathsD, ...) — snaps to a
// different grid than the Rust's, so the three boolean entry points here scale to
// integers themselves and call the Paths64 overloads instead. UnionD/IntersectD/
// DifferenceD below carry the full reasoning at the scaling site; everything else
// in this file uses the stock PathsD API, which agrees with the Rust exactly.
//
// ── Precision 6 ──────────────────────────────────────────────────────────────
// Every PathsD entry point takes a decimal-places argument that decides the
// fixed-point grid Clipper snaps to internally. The Rust hardcodes 6 at all ten
// call sites rather than taking it as a parameter, so the literal 6 is
// transcribed at each of them; changing one and not the others would silently
// re-grid one operation relative to the others.
//
// ── Name mapping ─────────────────────────────────────────────────────────────
// Most of the Rust free functions are the same function under a second spelling:
// inflate_paths_d -> Clipper.InflatePaths, simplify_paths ->
// Clipper.SimplifyPaths, minkowski_sum_d -> Minkowski.Sum.
// PathD/PathsD/PointD/FillRule/JoinType/EndType carry the same names and, for
// the enums, the same underlying values on both sides.
//
// Three of them are NOT a pure rename, and each has its own explanation below:
//   union_d/intersect_d/difference_d -> UnionD/IntersectD/DifferenceD, which
//       scale to Paths64 themselves rather than going through ClipperD.
//   area                             -> PathArea, a hand-written shoelace.
//       Clipper2Lib's Clipper.Area is the C++ trapezoid form and disagrees with
//       clipper2-rust's plain shoelace in the last bit.
//   inflate_paths_d's delta == 0 early return, which Clipper2Lib does not have,
//       so Offset and OffsetWithParams guard it themselves.

using Clipper2Lib;

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	public sealed partial class CrossSection
	{
		/// <summary>
		/// Creates a CrossSection from polygons, normalizing via Clipper2 Union.
		/// Mirrors C++ CrossSection(Polygons, FillRule) constructor which runs
		/// the polygons through C2::Union to merge overlapping regions.
		/// </summary>
		/// <param name="polygons">The contours to merge.</param>
		/// <returns>The normalized cross section.</returns>
		public static CrossSection FromPolygonsFill(Polygons polygons)
		{
			if (polygons.Count == 0)
			{
				return new CrossSection();
			}

			PathsD paths = ToPaths(polygons);
			PathsD empty = new PathsD();
			PathsD result = UnionD(paths, empty, FillRule.NonZero, 6);
			return new CrossSection(FromPaths(result));
		}

		/// <summary>
		/// Create CrossSection from a simple polygon with a specified fill rule.
		/// fill_rule: 0=EvenOdd, 1=NonZero, 2=Positive, 3=Negative
		/// </summary>
		/// <param name="polygon">The single contour.</param>
		/// <param name="fillRule">0=EvenOdd, 1=NonZero, 2=Positive, 3=Negative; any other value=Positive.</param>
		/// <returns>The filled cross section.</returns>
		public static CrossSection FromPolygonWithFillRule(SimplePolygon polygon, int fillRule)
		{
			FillRule fr;
			switch (fillRule)
			{
				case 0:
					fr = FillRule.EvenOdd;
					break;
				case 1:
					fr = FillRule.NonZero;
					break;
				case 2:
					fr = FillRule.Positive;
					break;
				case 3:
					fr = FillRule.Negative;
					break;
				default:
					fr = FillRule.Positive;
					break;
			}

			PathD path = new PathD(polygon.Count);
			foreach (Vec2 v in polygon)
			{
				path.Add(new PointD(v.X, v.Y));
			}

			PathsD paths = new PathsD { path };
			PathsD empty = new PathsD();
			PathsD result = UnionD(paths, empty, fr, 6);
			return new CrossSection(FromPaths(result));
		}

		/// <summary>Boolean union with another cross section.</summary>
		/// <param name="other">The other cross section.</param>
		/// <returns>The union.</returns>
		public CrossSection Union(CrossSection other)
		{
			return new CrossSection(FromPaths(UnionD(
				ToPaths(this.polygons),
				ToPaths(other.polygons),
				FillRule.NonZero,
				6)));
		}

		/// <summary>Boolean intersection with another cross section.</summary>
		/// <param name="other">The other cross section.</param>
		/// <returns>The intersection.</returns>
		public CrossSection Intersection(CrossSection other)
		{
			return new CrossSection(FromPaths(IntersectD(
				ToPaths(this.polygons),
				ToPaths(other.polygons),
				FillRule.NonZero,
				6)));
		}

		/// <summary>Boolean difference: this minus the other.</summary>
		/// <param name="other">The cross section to subtract.</param>
		/// <returns>The difference.</returns>
		public CrossSection Difference(CrossSection other)
		{
			return new CrossSection(FromPaths(DifferenceD(
				ToPaths(this.polygons),
				ToPaths(other.polygons),
				FillRule.NonZero,
				6)));
		}

		/// <summary>
		/// Simplify contours by removing near-collinear vertices.
		/// Mirrors C++ CrossSection::Simplify(epsilon=1e-6): normalizes via union,
		/// filters tiny polygons, then applies SimplifyPaths with epsilon.
		/// </summary>
		/// <param name="epsilon">The collinearity tolerance, also the sliver-filter threshold.</param>
		/// <returns>The simplified cross section.</returns>
		public CrossSection Simplify(double epsilon)
		{
			if (this.polygons.Count == 0)
			{
				return new CrossSection();
			}

			// Normalize via union (removes overlaps/inversions). Positive, not NonZero:
			// the filter below leans on the union having already dropped reversed contours.
			PathsD paths = ToPaths(this.polygons);
			PathsD unified = UnionD(paths, new PathsD(), FillRule.Positive, 6);

			// Filter out contours smaller than epsilon (area vs bounding box).
			PathsD filtered = new PathsD();
			foreach (PathD poly in unified)
			{
				// PathArea, not Clipper.Area — see the note on PathArea. The two disagree
				// by an ulp on most inputs, and the `a > maxSize * epsilon` test below can
				// turn that ulp into a kept-or-dropped contour.
				double a = Math.Abs(PathArea(poly));

				// Compute bounding box max extent
				double minX = double.MaxValue;
				double minY = double.MaxValue;
				double maxX = double.MinValue;
				double maxY = double.MinValue;
				foreach (PointD p in poly)
				{
					if (p.x < minX)
					{
						minX = p.x;
					}

					if (p.x > maxX)
					{
						maxX = p.x;
					}

					if (p.y < minY)
					{
						minY = p.y;
					}

					if (p.y > maxY)
					{
						maxY = p.y;
					}
				}

				double maxSize = Math.Max(maxX - minX, maxY - minY);
				if (a > maxSize * epsilon)
				{
					filtered.Add(poly);
				}
			}

			// The Rust passes is_closed_path = true. The C# parameter carries the same name
			// and the same polarity (isClosedPath, not isOpenPath), and its default happens
			// to be true as well — passed explicitly anyway so the two sources read alike
			// and a future default change cannot move the result silently.
			PathsD simplified = Clipper.SimplifyPaths(filtered, epsilon, true);
			return new CrossSection(FromPaths(simplified));
		}

		/// <summary>Offsets (inflates or deflates) every contour with round joins.</summary>
		/// <param name="delta">The offset distance; negative deflates. Zero returns the input unchanged.</param>
		/// <returns>The offset cross section.</returns>
		public CrossSection Offset(double delta)
		{
			if (delta == 0.0)
			{
				return this.ZeroOffsetIdentity();
			}

			return new CrossSection(FromPaths(Clipper.InflatePaths(
				ToPaths(this.polygons),
				delta,
				JoinType.Round,
				EndType.Polygon,
				2.0,
				6,
				0.0)));
		}

		/// <summary>
		/// Offset with explicit join type and segment count.
		/// join_type: 0=Square, 1=Round, 2=Miter
		/// </summary>
		/// <remarks>
		/// The summary above is the Rust's own doc comment and is incomplete in the same
		/// way: the match also accepts 3=Bevel, and every value outside {0, 2, 3} — 1
		/// included — falls through to Round. Kept verbatim so the two files diff cleanly.
		/// <para>
		/// <paramref name="circularSegments"/> only reaches Clipper as an arc tolerance,
		/// which bounds the chord error rather than fixing a segment count, so a round join
		/// is not guaranteed to land on exactly that many segments. The Rust's NegativeOffset
		/// test (manifold_tests/advanced.rs) widens its tolerance for precisely this reason;
		/// the behavior is ported as-is, not fixed here.
		/// </para>
		/// </remarks>
		/// <param name="delta">The offset distance; negative deflates. Zero returns the input unchanged.</param>
		/// <param name="joinType">0=Square, 2=Miter, 3=Bevel, anything else=Round.</param>
		/// <param name="miterLimit">The miter limit passed through to Clipper.</param>
		/// <param name="circularSegments">The desired segment count for round joins; ignored at 2 or below.</param>
		/// <returns>The offset cross section.</returns>
		public CrossSection OffsetWithParams(
			double delta,
			int joinType,
			double miterLimit,
			int circularSegments)
		{
			if (delta == 0.0)
			{
				return this.ZeroOffsetIdentity();
			}

			JoinType jt;
			switch (joinType)
			{
				case 0:
					jt = JoinType.Square;
					break;
				case 2:
					jt = JoinType.Miter;
					break;
				case 3:
					jt = JoinType.Bevel;
					break;
				default:
					jt = JoinType.Round;
					break;
			}

			// For round joins, compute arc_tolerance from circular_segments to get the
			// exact segment count. Matches C++ CrossSection::Offset:
			//   arc_tol = (cos(pi/n) - 1) * -|delta|
			double arcTol;
			if (jt == JoinType.Round && circularSegments > 2)
			{
				double n = circularSegments;
				double absDelta = Math.Abs(delta);

				// System.Math.Cos and not DeterministicMath.Cos: the Rust reaches for std's
				// f64::cos here, not crate::math::cos as it does in Circle and Rotate.
				arcTol = (Math.Cos(Math.PI / n) - 1.0) * -absDelta;
			}
			else
			{
				arcTol = 0.0;
			}

			return new CrossSection(FromPaths(Clipper.InflatePaths(
				ToPaths(this.polygons),
				delta,
				jt,
				EndType.Polygon,
				miterLimit,
				6,
				arcTol)));
		}

		/// <summary>
		/// The <c>delta == 0</c> result of both offsets: this cross section's contours,
		/// unchanged, as a deep copy.
		/// </summary>
		/// <remarks>
		/// clipper2-rust's <c>inflate_paths_d</c> opens with <c>if delta == 0.0 { return
		/// paths.clone(); }</c> (clipper.rs:256), returning the input <i>before</i> the
		/// scale-to-int64 round trip. Clipper2Lib's <c>InflatePaths</c> has no such guard:
		/// it scales, offsets by nothing, and scales back, which snaps every coordinate to
		/// the 10^-precision grid on the way through.
		/// <para>
		/// The difference is invisible on grid-aligned input, which is why the harness's
		/// square-based <c>Offset(0.0)</c> case matched before this guard existed. Take a
		/// contour through <c>(0.1234567890123, 1)</c> instead and Clipper2Lib returns
		/// <c>0.123457</c> (bits <c>0x3fbf9ae0c1765775</c>) where the Rust returns the
		/// input untouched (bits <c>0x3fbf9add3746e984</c>). Pinned by
		/// CrossSectionTests.OffsetByZeroIsIdentity.
		/// </para>
		/// <para>
		/// The Rust's guard sits inside <c>inflate_paths_d</c>, so in
		/// <see cref="OffsetWithParams"/> it is reached only after the join type and arc
		/// tolerance have been computed. Hoisting it above them here is observably
		/// identical — neither computation has an effect, and neither can be reached with
		/// a delta that compares equal to zero but behaves differently.
		/// </para>
		/// <para>
		/// A deep copy rather than <c>this</c>: the Rust clones, and the constructor takes
		/// ownership of the list it is handed, so sharing the field would hand the caller
		/// a second CrossSection aliasing this one's contours.
		/// </para>
		/// </remarks>
		private CrossSection ZeroOffsetIdentity()
		{
			return new CrossSection(ClonePolygons(this.polygons));
		}

		/// <summary>
		/// The Minkowski sum of every contour of this with every contour of the other,
		/// concatenated in that nesting order.
		/// </summary>
		/// <param name="other">The pattern to sweep.</param>
		/// <returns>The swept cross section.</returns>
		public CrossSection MinkowskiSum(CrossSection other)
		{
			PathsD result = new PathsD();
			foreach (PathD a in ToPaths(this.polygons))
			{
				// The Rust rebuilds the inner PathsD on every outer iteration; kept as-is,
				// because hoisting it is the first place a later edit could change the order
				// in which paths land in `result`.
				foreach (PathD b in ToPaths(other.polygons))
				{
					// Minkowski.Sum, not Clipper.MinkowskiSum: the latter's three-argument
					// form hardcodes 2 decimal places, and the Rust's minkowski_sum_d is
					// called with 6.
					//
					// Namespace-qualified deliberately, and it must stay that way. The 3D
					// Minkowski of minkowski.rs is a Phase 5 file that had not landed when
					// this was written; once ManifoldSharp.Minkowski exists, a
					// same-namespace type beats a using-directive one, so the unqualified
					// spelling would stop naming Clipper2Lib's class and start naming ours
					// — quietly, at the next build, with no error here.
					foreach (PathD path in Clipper2Lib.Minkowski.Sum(a, b, true, 6))
					{
						result.Add(path);
					}
				}
			}

			return new CrossSection(FromPaths(result));
		}

		/// <summary>
		/// Batch boolean operation on a slice of CrossSections.
		/// OpType::Add = union, Subtract = difference, Intersect = intersection.
		/// </summary>
		/// <param name="sections">The operands in order; the first is the left operand for Subtract.</param>
		/// <param name="op">The operation.</param>
		/// <returns>The combined cross section.</returns>
		public static CrossSection BatchBoolean(IReadOnlyList<CrossSection> sections, OpType op)
		{
			if (sections.Count == 0)
			{
				return new CrossSection();
			}

			switch (op)
			{
				case OpType.Add:
				{
					// Union is one Clipper call over every contour at once, not a fold of
					// pairwise unions — a fold would re-grid the intermediate at precision 6
					// once per section, and Subtract and Intersect below deliberately do fold.
					PathsD paths = new PathsD();
					foreach (CrossSection s in sections)
					{
						foreach (PathD p in ToPaths(s.polygons))
						{
							paths.Add(p);
						}
					}

					PathsD empty = new PathsD();
					return new CrossSection(FromPaths(UnionD(paths, empty, FillRule.NonZero, 6)));
				}

				case OpType.Subtract:
				{
					CrossSection result = sections[0].Clone();
					for (int i = 1; i < sections.Count; i++)
					{
						result = result.Difference(sections[i]);
					}

					return result;
				}

				default:
				{
					// OpType::Intersect. The Rust match is exhaustive over three variants, so
					// this arm is Intersect and nothing else; C# needs a default to satisfy
					// definite assignment.
					CrossSection result = sections[0].Clone();
					for (int i = 1; i < sections.Count; i++)
					{
						result = result.Intersection(sections[i]);
					}

					return result;
				}
			}
		}

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
		//                                           = 1e6         at precision 6
		//   Upstream C++ Clipper2 (issue
		//   #25, "set the scale to a power
		//   of double's radix"), and hence
		//   clipper2-rust 1.0.3 / 1.1.0:      scale = 2^(ilogb(10^precision) + 1)
		//                                           = 2^20        at precision 6
		//
		// The C# port simply has not taken that upstream change yet. Minimal repro,
		// pinned by CrossSectionTests.ClipperDScaleIsPowerOfTwo — Union of
		// [(0,0), (1,0), (0.1234567890123, 1)] against an empty clip, NonZero,
		// precision 6, gives for that third x:
		//
		//   Clipper2Lib's ClipperD    0.123457            bits 0x3fbf9ae0c1765775
		//   clipper2-rust (and this)  0.12345695495605469 bits 0x3fbf9ae000000000
		//
		// So these three do the D-layer themselves against the Paths64 overloads.
		// That is not a divergence from the Rust — it is what reaches the Rust's
		// specified numbers, which is why there is no docs/RUST_DIVERGENCES.md entry
		// for it. A differential harness of 101 cases against the compiled
		// manifold-rust goes from 12 mismatches to 0 with this in place.
		//
		// Only the booleans need it. InflatePaths, Minkowski.Sum and SimplifyPaths
		// take an explicit decimal-places argument and scale by 10^decimals on both
		// sides, so those keep the stock double API above and already match.
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
		/// clipper2-rust's <c>area</c> (core.rs:912), which Clipper2Lib's
		/// <c>Clipper.Area</c> is not a drop-in replacement for.
		/// </summary>
		/// <remarks>
		/// The two compute the same quantity by different summations, and therefore round
		/// differently:
		/// <list type="bullet">
		/// <item>clipper2-rust: the plain shoelace, <c>Σ(xᵢyⱼ − xⱼyᵢ) · 0.5</c>. Its
		/// comment says "Use the standard shoelace formula for now" — a deliberate
		/// departure from the C++, not an oversight.</item>
		/// <item>Clipper2Lib: the C++ trapezoid form, <c>Σ(yₚ + y)(xₚ − x) · 0.5</c>.</item>
		/// </list>
		/// Over 20,000 random polygons the two disagreed on 13,544 — always by an ulp,
		/// never more than ~2e-12 absolute. That is invisible almost everywhere, but
		/// <see cref="Simplify"/> feeds this straight into a <c>&gt;</c> comparison
		/// against <c>maxSize * epsilon</c>, where one ulp decides whether a contour
		/// survives, so the contour *count* can differ. Hence the transcription.
		/// <para>
		/// Not the same function as CrossSection.cs's SignedArea, which is the Rust's own
		/// <c>signed_area</c> helper: identical summation, but no <c>cnt &lt; 3</c> early
		/// return. Keep them separate — <see cref="Area"/> depends on a two-point contour
		/// summing to 0.0 arithmetically, and this one on it returning 0.0 by the guard.
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

			double area = 0.0;
			for (int i = 0; i < cnt; i++)
			{
				int j = (i + 1) % cnt;
				double xi = path[i].x;
				double yi = path[i].y;
				double xj = path[j].x;
				double yj = path[j].y;

				area += (xi * yj) - (xj * yi);
			}

			return area * 0.5;
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
	}
}
