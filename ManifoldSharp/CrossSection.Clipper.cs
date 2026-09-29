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

// CrossSection.Clipper.cs — the Clipper2-backed operations of cross_section.rs
// and its child module cross_section_ops.rs.
//
// This and CrossSection.ClipperD.cs (the double-precision layer these
// operations share) are the only files in the assembly that name Clipper2Lib,
// which is the whole dependency budget of the port (CLAUDE.md's dependency
// table) made structural: cross_section.rs and cross_section_ops.rs are likewise
// the only Rust files importing clipper2-rust. See CrossSection.cs for the file
// split.
//
// ── Why Clipper2Lib 1.5.4, and why the booleans bypass its D-layer ───────────
// clipper2-rust 1.0.3 is a port of upstream Clipper2 1.5.4 (its version.rs says
// so), and the NuGet Clipper2 package is that same upstream release by the same
// author. The version pin in the csproj is load-bearing; see the comment there.
//
// Matching versions are not sufficient. Clipper2Lib's `ClipperD` — the double
// wrapper behind Clipper.Union/Intersect/Difference(PathsD, ...) — snaps to a
// different grid than the Rust's, so every boolean here (flat or PolyTree) scales
// to integers itself and calls the Paths64 overloads instead. The D-layer in
// CrossSection.ClipperD.cs carries the full reasoning at the scaling site;
// everything else in this file uses the stock PathsD API, which agrees with the
// Rust exactly.
//
// ── Precision 8 ──────────────────────────────────────────────────────────────
// Every PathsD entry point takes a decimal-places argument that decides the
// fixed-point grid Clipper snaps to internally. The Rust passes one constant,
// PRECISION = 8 — C++ cross_section.cpp's `precision_` — at every call site, and
// so does this file (Precision below). It was a literal 6 at each site until the
// manifold-rust CrossSection Clipper2-alignment change; at 6 a 1.2e-7 feature
// snapped away, at 8 it lands on the booleans' 2^27 grid as C++ does.
//
// ── Fill rules ───────────────────────────────────────────────────────────────
// The booleans, BatchBoolean's union, Compose, Decompose and FromPolygonsFill
// fill with FillRule.Positive, as C++ BooleanOp/BatchBoolean/Decompose hard-code
// it and the C++ Polygons constructor defaults to it: a clockwise contour fills
// nothing. The
// integer codes of FromPolygonWithFillRule and OffsetWithParams follow the C++
// enumerator order, and an unknown code falls through to EvenOdd and Square
// respectively — the values C++ fr() and jt() start from before their switch.
//
// ── Name mapping ─────────────────────────────────────────────────────────────
// Most of the Rust free functions are the same function under a second spelling:
// inflate_paths_d -> Clipper.InflatePaths, simplify_paths ->
// Clipper.SimplifyPaths, minkowski_sum_d -> Minkowski.Sum.
// PathD/PathsD/PointD/FillRule/JoinType/EndType carry the same names and, for
// the enums, the same underlying values on both sides.
//
// Four of them are NOT a pure rename, and each has its own explanation at its
// definition:
//   union_d/intersect_d/difference_d -> UnionD/IntersectD/DifferenceD, which
//       scale to Paths64 themselves rather than going through ClipperD.
//   boolean_op_tree_d (always a union here) -> UnionTree, the same scaling
//       around Clipper2Lib's PolyTree64; ClipperD.cs's header shows the tree is
//       Clipper2 46f6391's.
//   area                             -> PathArea, the Rust's own port of the
//       C++ trapezoid Area (clipper2-rust's `area` is a shoelace, and the Rust
//       no longer calls it). Clipper2Lib 1.5.4's Clipper.Area(PathD) happens to
//       be the same loop in the same order, but PathArea stays hand-written so
//       it is visibly the same function as CrossSection.cs's ContourArea.
//   inflate_paths_d's delta == 0 early return, which Clipper2Lib does not have,
//       so OffsetWithParams guards it itself (and Offset, which delegates to it).

using Clipper2Lib;

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	public sealed partial class CrossSection
	{
		/// <summary>
		/// Decimal places Clipper2 keeps when scaling to integer coordinates; mirrors
		/// <c>precision_</c> in C++ cross_section.cpp, passed to every Clipper2 call.
		/// </summary>
		private const int Precision = 8;

		/// <summary>
		/// Creates a CrossSection from polygons, normalizing via Clipper2 Union.
		/// Mirrors C++ CrossSection(Polygons, FillRule) constructor with its
		/// default FillRule::Positive, which runs the polygons through C2::Union
		/// to merge overlapping regions.
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
			PathsD result = UnionD(paths, empty, FillRule.Positive, Precision);
			return new CrossSection(FromPaths(result));
		}

		/// <summary>
		/// Create CrossSection from a simple polygon with a specified fill rule.
		/// fill_rule: 0=EvenOdd, 1=NonZero, 2=Positive, 3=Negative (the C++
		/// <c>CrossSection::FillRule</c> enumerator order). Other codes fall through to
		/// EvenOdd, the value C++ <c>fr()</c> starts from before its switch.
		/// </summary>
		/// <param name="polygon">The single contour.</param>
		/// <param name="fillRule">0=EvenOdd, 1=NonZero, 2=Positive, 3=Negative; any other value=EvenOdd.</param>
		/// <returns>The filled cross section.</returns>
		public static CrossSection FromPolygonWithFillRule(SimplePolygon polygon, int fillRule)
		{
			FillRule fr;
			switch (fillRule)
			{
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
					fr = FillRule.EvenOdd;
					break;
			}

			PathD path = new PathD(polygon.Count);
			foreach (Vec2 v in polygon)
			{
				path.Add(new PointD(v.X, v.Y));
			}

			PathsD paths = new PathsD { path };
			PathsD empty = new PathsD();
			PathsD result = UnionD(paths, empty, fr, Precision);
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
				FillRule.Positive,
				Precision)));
		}

		/// <summary>Boolean intersection with another cross section.</summary>
		/// <param name="other">The other cross section.</param>
		/// <returns>The intersection.</returns>
		public CrossSection Intersection(CrossSection other)
		{
			return new CrossSection(FromPaths(IntersectD(
				ToPaths(this.polygons),
				ToPaths(other.polygons),
				FillRule.Positive,
				Precision)));
		}

		/// <summary>Boolean difference: this minus the other.</summary>
		/// <param name="other">The cross section to subtract.</param>
		/// <returns>The difference.</returns>
		public CrossSection Difference(CrossSection other)
		{
			return new CrossSection(FromPaths(DifferenceD(
				ToPaths(this.polygons),
				ToPaths(other.polygons),
				FillRule.Positive,
				Precision)));
		}

		/// <summary>
		/// Split into topologically disconnected components, each one outline with zero
		/// or more holes. Mirrors C++ <c>CrossSection::Decompose</c>: fewer than two
		/// contours return this section unchanged; otherwise a Positive union into a
		/// Clipper2 PolyTree, whose containment links decide which holes belong to which
		/// outline, walked as <c>decompose_outline</c> / <c>decompose_hole</c> do and
		/// emitted in reverse push order.
		/// </summary>
		/// <remarks>
		/// The count test is on the raw contour list (the Rust's
		/// <c>self.polygons.len() &lt; 2</c>), so an empty section decomposes to one empty
		/// section and a single contour comes back without being snapped to Clipper2's
		/// grid.
		/// </remarks>
		/// <returns>One CrossSection per outline, each carrying exactly its own holes.</returns>
		public List<CrossSection> Decompose()
		{
			if (this.polygons.Count < 2)
			{
				return new List<CrossSection> { this.Clone() };
			}

			(PolyTree64 tree, double invScale) = UnionTree(ToPaths(this.polygons), FillRule.Positive, Precision);
			List<PathsD> comps = new List<PathsD>();
			DecomposeOutlines(tree, invScale, comps);
			List<CrossSection> result = new List<CrossSection>(comps.Count);
			for (int i = comps.Count - 1; i >= 0; i--)
			{
				result.Add(new CrossSection(FromPaths(comps[i])));
			}

			return result;
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
			PathsD unified = UnionD(paths, new PathsD(), FillRule.Positive, Precision);

			// Filter out contours smaller than epsilon (area vs bounding box).
			PathsD filtered = new PathsD();
			foreach (PathD poly in unified)
			{
				// PathArea, the port of C++'s C2::Area — see the note on PathArea. A
				// different summation disagrees by an ulp on most inputs, and the
				// `a > maxSize * epsilon` test below can turn that ulp into a kept-or-dropped
				// contour.
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

		/// <summary>
		/// Offset with the C++ <c>CrossSection::Offset</c> defaults: Round joins,
		/// miter_limit 2.0, circularSegments 0 (segments from Quality).
		/// </summary>
		/// <remarks>
		/// Reads the process-global <see cref="Quality"/> settings through
		/// <see cref="OffsetWithParams"/>'s circularSegments &lt;= 2 branch.
		/// </remarks>
		/// <param name="delta">The offset distance; negative deflates. Zero returns the input unchanged.</param>
		/// <returns>The offset cross section.</returns>
		public CrossSection Offset(double delta)
		{
			return this.OffsetWithParams(delta, 1, 2.0, 0);
		}

		/// <summary>
		/// Offset with explicit join type and segment count.
		/// join_type: 0=Square, 1=Round, 2=Miter, 3=Bevel (the C++
		/// <c>CrossSection::JoinType</c> enumerator order). Other codes fall through to
		/// Square, the value C++ <c>jt()</c> starts from before its switch.
		/// </summary>
		/// <remarks>
		/// <para>
		/// <paramref name="circularSegments"/> only reaches Clipper as an arc tolerance,
		/// which bounds the chord error rather than fixing a segment count, so a round join
		/// is not guaranteed to land on exactly that many segments. The Rust's NegativeOffset
		/// test (manifold_tests/advanced.rs) widens its tolerance for precisely this reason;
		/// the behavior is ported as-is, not fixed here.
		/// </para>
		/// </remarks>
		/// <param name="delta">The offset distance; negative deflates. Zero returns the input unchanged.</param>
		/// <param name="joinType">1=Round, 2=Miter, 3=Bevel, anything else (0 included)=Square.</param>
		/// <param name="miterLimit">The miter limit passed through to Clipper.</param>
		/// <param name="circularSegments">
		/// The desired segment count for round joins; at 2 or below,
		/// <see cref="Quality.GetCircularSegments"/> of <paramref name="delta"/> instead.
		/// </param>
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
				case 1:
					jt = JoinType.Round;
					break;
				case 2:
					jt = JoinType.Miter;
					break;
				case 3:
					jt = JoinType.Bevel;
					break;
				default:
					jt = JoinType.Square;
					break;
			}

			// For round joins, compute arc_tolerance from circular_segments (or,
			// when it is <= 2, Quality's count for radius delta) to get the exact
			// segment count. Matches C++ CrossSection::Offset:
			//   arc_tol = (math::cos(pi/n) - 1) * -|delta|
			double arcTol;
			if (jt == JoinType.Round)
			{
				int n = circularSegments > 2
					? circularSegments
					: Quality.GetCircularSegments(delta);
				double absDelta = Math.Abs(delta);

				// DeterministicMath.Cos, the Rust's math::cos — the same spelling Circle
				// and Rotate use.
				arcTol = (DeterministicMath.Cos(Math.PI / (double)n) - 1.0) * -absDelta;
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
				Precision,
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
		/// <c>0.12345679</c> (bits <c>0x3fbf9add3b84e659</c>, the 10^-8 grid at precision
		/// 8) where the Rust returns the
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
					// called with PRECISION (8).
					//
					// Namespace-qualified deliberately, and it must stay that way. The 3D
					// Minkowski of minkowski.rs is a Phase 5 file that had not landed when
					// this was written; once ManifoldSharp.Minkowski exists, a
					// same-namespace type beats a using-directive one, so the unqualified
					// spelling would stop naming Clipper2Lib's class and start naming ours
					// — quietly, at the next build, with no error here.
					foreach (PathD path in Clipper2Lib.Minkowski.Sum(a, b, true, Precision))
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
					// pairwise unions — a fold would re-grid the intermediate at Precision
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
					return new CrossSection(FromPaths(UnionD(paths, empty, FillRule.Positive, Precision)));
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

		/// <summary>
		/// C++ <c>decompose_outline</c> / <c>decompose_hole</c> (cross_section.cpp:126-151),
		/// the Rust <c>decompose_outlines</c>: for each outline child of
		/// <paramref name="node"/>, first recurse into every hole's own outline children
		/// (islands), then push <c>[outline, holes...]</c>. The C++ recurses over sibling
		/// indices too; iterating them visits the same nodes in the same order without a
		/// stack frame per sibling.
		/// </summary>
		private static void DecomposeOutlines(PolyPath64 node, double invScale, List<PathsD> polys)
		{
			for (int i = 0; i < node.Count; i++)
			{
				PolyPath64 outline = node[i];
				PathsD poly = new PathsD(outline.Count + 1);
				poly.Add(TreeNodePath(outline, invScale));
				for (int j = 0; j < outline.Count; j++)
				{
					PolyPath64 hole = outline[j];
					DecomposeOutlines(hole, invScale, polys);
					poly.Add(TreeNodePath(hole, invScale));
				}

				polys.Add(poly);
			}
		}
	}
}
