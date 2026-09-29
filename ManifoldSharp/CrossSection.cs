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

// CrossSection.cs — port of cross_section.rs (Phase 8).
//
// The 2D half of the library: a set of polygon contours with boolean, offset,
// hull and Minkowski operations. cross_section.rs and its child module
// cross_section_ops.rs are the only Rust files that touch clipper2-rust, and
// this class is the only part of the assembly with a package dependency
// (CLAUDE.md's dependency table).
//
// ── File split ───────────────────────────────────────────────────────────────
// cross_section.rs and its child module cross_section_ops.rs land as four
// partials of one class:
//   CrossSection.cs          the type, its two constructors (raw FromRaw and the
//                            unioning public one), the primitives, the queries,
//                            the affine transforms, Warp, Compose
//   CrossSection.Clipper.cs  every operation that delegates to Clipper2
//   CrossSection.ClipperD.cs the power-of-two double layer those operations
//                            share, and the Polygons<->PathsD conversions
//   CrossSection.Hull.cs     Andrew's monotone chain and its comparator
// Only the two Clipper files have `using Clipper2Lib`, which makes the Rust's
// confinement of the dependency to the cross_section modules structural here
// rather than a convention. The split lines are C#-only; the Rust's
// cross_section.rs / cross_section_ops.rs line falls elsewhere.
//
// ── Two constructors, as in C++ ──────────────────────────────────────────────
// C++ has a private raw constructor that every Clipper2 result, transform, hull
// and primitive goes through, and a public Polygons constructor that always runs
// a Positive C2::Union. The Rust's pub(crate) from_raw and pub new are those two,
// and so are FromRaw (internal, for the tests as the Rust tests reach from_raw)
// and the public CrossSection(Polygons) here. Anything this class builds from
// contours it already trusts goes through FromRaw; `new CrossSection(polys)` is
// the union, snapping to Clipper2's grid.
//
// ── The wrapper owns path order, Clipper owns geometry ───────────────────────
// Everything Clipper hands back is passed through unchanged and in the order it
// arrived: FromPaths never sorts, never reverses, never filters. The only places
// this port post-processes Clipper output are Simplify (which flattens the union's
// PolyTree and filters contours by area *before* the SimplifyPaths call) and
// Decompose (which groups contours by the PolyTree's containment and emits the
// groups in reverse). Both are transcribed from the Rust literally, because the
// walk order there reaches the result.
//
// ── Trig ─────────────────────────────────────────────────────────────────────
// Rotate and OffsetWithParams' arc tolerance call DeterministicMath (the musl
// port), because the Rust calls crate::math at both. Circle calls Types.Cosd /
// Types.Sind, the Rust's types::cosd / sind and C++'s degree trig, exact on the
// axes; it used crate::math on radians until manifold-rust 9ae04a5. The arc
// tolerance used std's f64::cos (System.Math.Cos here) until the manifold-rust
// CrossSection Clipper2-alignment change moved it to math::cos, the C++
// Offset's `math::cos`; that call decides the vertex count of a round join.
//
// ── Area ─────────────────────────────────────────────────────────────────────
// Area and Simplify's sliver filter both go through the same port of Clipper2's
// Area (clipper.core.h at 46f6391, the commit C++ Manifold pins): the trapezoid
// sum over edges (n-1,0), (0,1), ..., in that order. ContourArea below and PathArea in CrossSection.Clipper.cs are that one
// function over the two point types — the Rust's clipper2_area_by with its two
// accessor closures — and must stay the same loop.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <summary>
	/// A 2D region as a set of polygon contours, with the boolean, offset, hull and
	/// Minkowski operations built on Clipper2.
	/// </summary>
	/// <remarks>
	/// Every operation returns a new instance; nothing mutates in place. The public
	/// <see cref="CrossSection(Polygons)"/> normalizes its input through a Positive
	/// union, as the C++ Polygons constructor does; the primitives, transforms and
	/// Clipper results wrap their contours raw.
	/// </remarks>
	public sealed partial class CrossSection
	{
		private readonly Polygons polygons;

		/// <summary>
		/// The Rust <c>Default</c>: no contours at all.
		/// </summary>
		public CrossSection()
		{
			this.polygons = new Polygons();
		}

		/// <summary>
		/// Create a CrossSection from contours. Mirrors C++
		/// <c>CrossSection(const Polygons&amp;, FillRule = Positive)</c>, which always runs
		/// <c>C2::Union</c> so overlapping contours merge, self-intersections resolve and
		/// coordinates snap to Clipper2's grid at <c>precision_</c>.
		/// </summary>
		/// <remarks>
		/// The input list is only read; the instance holds the union's fresh contours.
		/// </remarks>
		/// <param name="polygons">The contours to merge.</param>
		public CrossSection(Polygons polygons)
		{
			this.polygons = PositiveUnion(polygons);
		}

		/// <summary>The raw constructor behind <see cref="FromRaw"/>.</summary>
		/// <remarks>
		/// C# cannot overload the public constructor on the same parameter list, so the
		/// raw one takes a discarded marker. Stores the list by reference: the Rust
		/// <c>from_raw</c> <i>moves</i> its argument, and every caller here hands over a
		/// list it built for the purpose and never touches again.
		/// </remarks>
		private CrossSection(Polygons polygons, bool raw)
		{
			_ = raw;
			this.polygons = polygons;
		}

		/// <summary>
		/// Wrap already-clean contours without a union. Mirrors the C++ private
		/// <c>CrossSection(std::shared_ptr&lt;const PathImpl&gt;)</c> constructor that every
		/// Clipper2 result, transform, hull and primitive goes through; the Rust's
		/// <c>pub(crate) from_raw</c>.
		/// </summary>
		/// <remarks>
		/// The list is taken over by reference, not copied (see the raw constructor), so
		/// a caller must treat it as given away. <see cref="ToPolygons"/> hands back a
		/// deep copy for exactly this reason.
		/// </remarks>
		/// <param name="polygons">The contours, taken over by the new instance.</param>
		/// <returns>The cross section wrapping them as-is.</returns>
		internal static CrossSection FromRaw(Polygons polygons)
		{
			return new CrossSection(polygons, true);
		}

		/// <summary>
		/// A vertex transform for <see cref="Warp"/>. The Rust signature is
		/// <c>FnMut(&amp;mut Vec2)</c> — the function mutates the vertex in place rather
		/// than returning a new one, so the C# delegate takes <c>ref Vec2</c> and returns
		/// void. A <c>Func&lt;Vec2, Vec2&gt;</c> would read more naturally but would not
		/// be the same contract: a warp that writes only one component leaves the other at
		/// its original value here, and that is the behavior being ported.
		/// </summary>
		/// <param name="v">The vertex, to be modified in place.</param>
		public delegate void WarpFunc(ref Vec2 v);

		/// <summary>
		/// Create a CrossSection from a Rect's four corners, counter-clockwise from
		/// <c>Min</c>. Matches C++ <c>CrossSection(const Rect&amp;)</c>, which neither unions
		/// nor checks for an empty (inverted) Rect.
		/// </summary>
		/// <param name="rect">The rectangle.</param>
		/// <returns>A single contour of the four corners, whatever the Rect holds.</returns>
		public static CrossSection FromRect(Rect rect)
		{
			return FromRaw(new Polygons
			{
				new SimplePolygon
				{
					new Vec2(rect.Min.X, rect.Min.Y),
					new Vec2(rect.Max.X, rect.Min.Y),
					new Vec2(rect.Max.X, rect.Max.Y),
					new Vec2(rect.Min.X, rect.Max.Y),
				},
			});
		}

		/// <summary>
		/// A <paramref name="size"/> x <paramref name="size"/> square in the first
		/// quadrant touching the origin: C++ <c>Square(vec2(size), false)</c>.
		/// </summary>
		/// <param name="size">The edge length; zero or negative gives an empty result.</param>
		/// <returns>The square.</returns>
		public static CrossSection Square(double size)
		{
			return SquareVec2(new Vec2(size, size), false);
		}

		/// <summary>
		/// Create a rectangle of size (w, h), optionally centered at origin. Matches C++
		/// <c>CrossSection::Square(vec2, center)</c>: empty only when a dimension is
		/// negative or the size vector has zero length (so a zero-height rectangle is one
		/// degenerate contour); centered corners start at (+w/2, +h/2) and run
		/// counter-clockwise.
		/// </summary>
		/// <param name="size">The width and height.</param>
		/// <param name="center">Whether to center on the origin instead of the first quadrant.</param>
		/// <returns>The rectangle.</returns>
		public static CrossSection SquareVec2(Vec2 size, bool center)
		{
			if (size.X < 0.0 || size.Y < 0.0 || Math.Sqrt((size.X * size.X) + (size.Y * size.Y)) == 0.0)
			{
				return new CrossSection();
			}

			SimplePolygon p;
			if (center)
			{
				double w = size.X / 2.0;
				double h = size.Y / 2.0;
				p = new SimplePolygon
				{
					new Vec2(w, h),
					new Vec2(-w, h),
					new Vec2(-w, -h),
					new Vec2(w, -h),
				};
			}
			else
			{
				double x = size.X;
				double y = size.Y;
				p = new SimplePolygon
				{
					new Vec2(0.0, 0.0),
					new Vec2(x, 0.0),
					new Vec2(x, y),
					new Vec2(0.0, y),
				};
			}

			return FromRaw(new Polygons { p });
		}

		/// <summary>
		/// A circle of <c>n</c> vertices starting on +x. Matches C++
		/// <c>CrossSection::Circle</c>: <c>n</c> is <paramref name="segments"/> when above
		/// 2, otherwise <see cref="Quality.GetCircularSegments"/> of the radius, and vertex
		/// <c>i</c> sits at <c>radius * (cosd(360/n * i), sind(360/n * i))</c>, which is
		/// exact on the axes.
		/// </summary>
		/// <remarks>
		/// With <paramref name="segments"/> &lt;= 2 this reads the process-global Quality
		/// settings.
		/// </remarks>
		/// <param name="radius">The circumradius; zero or negative gives an empty result.</param>
		/// <param name="segments">The vertex count; 2 or below defers to Quality.</param>
		/// <returns>The circle.</returns>
		public static CrossSection Circle(double radius, int segments)
		{
			if (radius <= 0.0)
			{
				return new CrossSection();
			}

			int n = segments > 2 ? segments : Quality.GetCircularSegments(radius);
			double dPhi = 360.0 / (double)n;
			SimplePolygon poly = new SimplePolygon();
			for (int i = 0; i < n; i++)
			{
				double phi = dPhi * (double)i;
				poly.Add(new Vec2(radius * Types.Cosd(phi), radius * Types.Sind(phi)));
			}

			return FromRaw(new Polygons { poly });
		}

		/// <summary>The contours, as an independent deep copy.</summary>
		/// <returns>A copy of the contour list.</returns>
		public Polygons ToPolygons()
		{
			return ClonePolygons(this.polygons);
		}

		/// <summary>The Rust derived <c>Clone</c>: a deep copy of every contour.</summary>
		/// <returns>An independent copy.</returns>
		public CrossSection Clone()
		{
			return FromRaw(ClonePolygons(this.polygons));
		}

		/// <summary>Translates every vertex.</summary>
		/// <param name="v">The offset.</param>
		/// <returns>The translated cross section.</returns>
		public CrossSection Translate(Vec2 v)
		{
			Polygons result = new Polygons(this.polygons.Count);
			foreach (SimplePolygon poly in this.polygons)
			{
				SimplePolygon moved = new SimplePolygon(poly.Count);
				foreach (Vec2 p in poly)
				{
					moved.Add(p + v);
				}

				result.Add(moved);
			}

			return FromRaw(result);
		}

		/// <summary>
		/// Net enclosed area: the sum of signed contour areas, so CCW outers add
		/// and CW holes subtract. Mirrors C++ <c>CrossSection::Area</c>, i.e.
		/// Clipper2's <c>Area(Paths)</c>: an explicit fold from +0.0 in contour order,
		/// so an empty section yields +0.0 rather than <c>.sum()</c>'s -0.0.
		/// </summary>
		/// <remarks>
		/// The Rust needed the explicit fold because an iterator <c>.sum()</c> over no
		/// f64s is -0.0; a C# loop from <c>0.0</c> was already +0.0, and is the fold.
		/// </remarks>
		/// <returns>The summed area.</returns>
		public double Area()
		{
			double a = 0.0;
			foreach (SimplePolygon p in this.polygons)
			{
				a += ContourArea(p);
			}

			return a;
		}

		/// <summary>The axis-aligned bounds of every vertex.</summary>
		/// <returns>The bounding rect, empty (inverted) when there are no vertices.</returns>
		public Rect Bounds()
		{
			Rect rect = new Rect();
			foreach (SimplePolygon poly in this.polygons)
			{
				foreach (Vec2 p in poly)
				{
					rect.UnionPoint(p);
				}
			}

			return rect;
		}

		/// <summary>Scales each axis independently about the origin.</summary>
		/// <param name="v">The per-axis scale factors.</param>
		/// <returns>The scaled cross section.</returns>
		public CrossSection Scale(Vec2 v)
		{
			Polygons result = new Polygons(this.polygons.Count);
			foreach (SimplePolygon poly in this.polygons)
			{
				SimplePolygon scaled = new SimplePolygon(poly.Count);
				foreach (Vec2 p in poly)
				{
					scaled.Add(new Vec2(p.X * v.X, p.Y * v.Y));
				}

				result.Add(scaled);
			}

			return FromRaw(result);
		}

		/// <summary>Rotates about the origin.</summary>
		/// <param name="degrees">The angle in degrees, counter-clockwise.</param>
		/// <returns>The rotated cross section.</returns>
		public CrossSection Rotate(double degrees)
		{
			// Rust f64::to_radians is `self * (PI / 180.0)`, and the parenthesization
			// matters: multiplying by the pre-divided constant is not the same double as
			// `self * PI / 180.0`.
			double rad = degrees * (Math.PI / 180.0);
			double c = DeterministicMath.Cos(rad);
			double s = DeterministicMath.Sin(rad);
			Polygons result = new Polygons(this.polygons.Count);
			foreach (SimplePolygon poly in this.polygons)
			{
				SimplePolygon rotated = new SimplePolygon(poly.Count);
				foreach (Vec2 p in poly)
				{
					rotated.Add(new Vec2((p.X * c) - (p.Y * s), (p.X * s) + (p.Y * c)));
				}

				result.Add(rotated);
			}

			return FromRaw(result);
		}

		/// <summary>
		/// Mirror through a line perpendicular to the given axis vector.
		/// Matches C++ <c>CrossSection::Mirror(ax)</c> which uses <c>I - 2*n*n^T</c>.
		/// </summary>
		/// <param name="axis">The mirror plane's normal; a near-zero vector gives an empty result.</param>
		/// <returns>The mirrored cross section, with every contour reversed.</returns>
		public CrossSection Mirror(Vec2 axis)
		{
			double lenSq = (axis.X * axis.X) + (axis.Y * axis.Y);
			if (lenSq < 1e-20)
			{
				return new CrossSection();
			}

			// Reflection matrix: R = I - 2*n*n^T where n = normalize(axis).
			// Note the Rust divides by lenSq.sqrt() (the length), not by lenSq — the local
			// is named for the square of the length but is used as the length here.
			double nx = axis.X / Math.Sqrt(lenSq);
			double ny = axis.Y / Math.Sqrt(lenSq);
			double r00 = 1.0 - (2.0 * nx * nx);
			double r01 = -2.0 * nx * ny;
			double r10 = -2.0 * nx * ny;
			double r11 = 1.0 - (2.0 * ny * ny);
			Polygons result = new Polygons(this.polygons.Count);
			foreach (SimplePolygon poly in this.polygons)
			{
				// Mirror reverses winding, so reverse the polygon
				SimplePolygon mirrored = new SimplePolygon(poly.Count);
				for (int i = poly.Count - 1; i >= 0; i--)
				{
					Vec2 p = poly[i];
					mirrored.Add(new Vec2((r00 * p.X) + (r01 * p.Y), (r10 * p.X) + (r11 * p.Y)));
				}

				result.Add(mirrored);
			}

			return FromRaw(result);
		}

		/// <summary>True when there is no contour with at least three vertices.</summary>
		/// <returns>Whether the cross section encloses nothing.</returns>
		public bool IsEmpty()
		{
			if (this.polygons.Count == 0)
			{
				return true;
			}

			foreach (SimplePolygon p in this.polygons)
			{
				if (p.Count >= 3)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>The total vertex count across all contours, degenerate ones included.</summary>
		/// <returns>The vertex count.</returns>
		public int NumVert()
		{
			int sum = 0;
			foreach (SimplePolygon p in this.polygons)
			{
				sum += p.Count;
			}

			return sum;
		}

		/// <summary>The number of contours with at least three vertices.</summary>
		/// <returns>The contour count.</returns>
		public int NumContour()
		{
			int count = 0;
			foreach (SimplePolygon p in this.polygons)
			{
				if (p.Count >= 3)
				{
					count++;
				}
			}

			return count;
		}

		/// <summary>
		/// Move every vertex through <paramref name="f"/>, then re-union. Mirrors C++
		/// <c>CrossSection::Warp</c> / <c>WarpBatch</c>: vertices are visited in contour
		/// order, and the moved contours go through a FillRule::Positive union at
		/// <c>precision_</c>, so introduced self-intersections are resolved.
		/// </summary>
		/// <remarks>
		/// "In-place" describes the callback's view, not this object's: the vertices are
		/// copied first and the callback mutates the copies, so the receiver is unchanged.
		/// The Rust moves copies of its PathD points through <c>f</c> and runs
		/// <c>union_subjects_d(paths, Positive, PRECISION)</c>; moving copies of the
		/// Vec2s and unioning them is the same coordinates in the same order, which is
		/// <see cref="PositiveUnion"/>.
		/// </remarks>
		/// <param name="f">The per-vertex transform.</param>
		/// <returns>The warped cross section.</returns>
		public CrossSection Warp(WarpFunc f)
		{
			Polygons polys = new Polygons(this.polygons.Count);
			foreach (SimplePolygon poly in this.polygons)
			{
				SimplePolygon warped = new SimplePolygon(poly.Count);
				foreach (Vec2 v in poly)
				{
					Vec2 v2 = v;
					f(ref v2);
					warped.Add(v2);
				}

				polys.Add(warped);
			}

			return FromRaw(PositiveUnion(polys));
		}

		/// <summary>
		/// Batch union of the sections. Mirrors C++ <c>CrossSection::Compose</c>, which is
		/// <c>BatchBoolean(crossSections, OpType::Add)</c>.
		/// </summary>
		/// <param name="sections">The cross sections to merge.</param>
		/// <returns>The merged cross section.</returns>
		public static CrossSection Compose(IReadOnlyList<CrossSection> sections)
		{
			return BatchBoolean(sections, OpType.Add);
		}

		/// <summary>
		/// The Rust free function <c>contour_area</c>: its <c>clipper2_area_by</c>, an
		/// exact port of Clipper2's <c>Area(const Path&lt;T&gt;&amp;)</c> (clipper.core.h at
		/// commit 46f6391), over a contour's <see cref="Vec2"/>s. Positive for a
		/// counter-clockwise contour.
		/// </summary>
		/// <remarks>
		/// Clipper2 walks the trapezoid form over edges (n-1,0), (0,1), ..., (n-2,n-1),
		/// accumulating <c>(prev.y + cur.y) * (prev.x - cur.x)</c> in that order; its
		/// two-edges-per-step unrolling does not change the order. This differs in the
		/// last bits from a shoelace sum, so every place C++ calls <c>C2::Area</c> uses
		/// this. <see cref="PathArea"/> is the same loop over Clipper's <c>PointD</c> —
		/// change one, change both.
		/// </remarks>
		private static double ContourArea(SimplePolygon poly)
		{
			int cnt = poly.Count;
			if (cnt < 3)
			{
				return 0.0;
			}

			double a = 0.0;
			int prev = cnt - 1;
			for (int cur = 0; cur < cnt; cur++)
			{
				a += (poly[prev].Y + poly[cur].Y) * (poly[prev].X - poly[cur].X);
				prev = cur;
			}

			return a * 0.5;
		}

		/// <summary>Deep copy of a contour list — the Rust's derived <c>Clone</c>.</summary>
		private static Polygons ClonePolygons(Polygons polygons)
		{
			Polygons copy = new Polygons(polygons.Count);
			foreach (SimplePolygon poly in polygons)
			{
				copy.Add(new SimplePolygon(poly));
			}

			return copy;
		}
	}
}
