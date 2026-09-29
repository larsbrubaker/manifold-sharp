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
//   CrossSection.Hull.cs     C++ HullImpl (monotone chain) and V2Lesser
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
// ── Lazy transforms, as in C++ ───────────────────────────────────────────────
// C++ CrossSection holds `paths_` (a shared_ptr) and a pending mat2x3
// `transform_`. Translate / Rotate / Scale / Mirror do not touch a vertex: each
// returns a section sharing the same contours with `m * Mat3(transform_)` as its
// pending transform. GetPaths applies it once as `m * vec3(x, y, 1)`, reversing
// every contour when the linear part's determinant is negative, skips it when it
// compares `==` to identity (so -0.0 entries still count as zero and the stored
// vertices, signed zeros included, come back untouched), and otherwise bakes the
// result into the section and resets the transform to identity. The Rust
// (manifold-rust 0d60adf) is `Mutex<PathState { Arc<Polygons>, Mat2x3 }>` with
// `paths()` as GetPaths; here it is the `paths` / `transform` pair behind a
// per-instance `lock`, with Paths() as GetPaths. Every reader goes through
// Paths() — a direct read of the field would see untransformed contours.
//
// The contour list is shared, as the Arc and the shared_ptr share it: Clone and
// each transform hand the same Polygons to the new section, so no list reachable
// from `paths` is ever mutated. Baking stores a fresh list rather than rewriting
// the shared one, FromRaw takes over the list it is given, and ToPolygons hands
// out a deep copy. Clone copies the pending transform without applying it, as
// the C++ copy constructor and the Rust Clone do.
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
// OffsetWithParams' arc tolerance calls DeterministicMath (the musl port),
// because the Rust calls crate::math there. Circle and Rotate call Types.Cosd /
// Types.Sind, the Rust's types::cosd / sind and C++'s degree trig, exact on the
// axes; Circle used crate::math on radians until manifold-rust 9ae04a5, Rotate
// until 0d60adf. The arc
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
	/// Every operation returns a new instance, and the observable value of a section
	/// never changes. Internally a read may bake a pending transform into the stored
	/// contours (C++ <c>GetPaths</c>), under a per-instance lock, so an instance is
	/// safe to read from several threads. The public
	/// <see cref="CrossSection(Polygons)"/> normalizes its input through a Positive
	/// union, as the C++ Polygons constructor does; the primitives, transforms and
	/// Clipper results wrap their contours raw.
	/// </remarks>
	public sealed partial class CrossSection
	{
		/// <summary><c>la::identity</c> as a mat2x3: C++ <c>transform_</c>'s initial value.</summary>
		private static readonly Mat2x3 Identity = Mat2x3.FromCols(
			new Vec2(1.0, 0.0),
			new Vec2(0.0, 1.0),
			new Vec2(0.0, 0.0));

		/// <summary>
		/// Guards <see cref="paths"/> and <see cref="transform"/>: the Rust's
		/// <c>Mutex&lt;PathState&gt;</c>, C++'s <c>pathsMutex_</c>.
		/// </summary>
		private readonly object sync = new object();

		/// <summary>
		/// The contours before <see cref="transform"/> — C++ <c>paths_</c>. Shared with
		/// clones and transformed sections, so never mutated; see the file header.
		/// </summary>
		private Polygons paths;

		/// <summary>The pending transform still to be applied to <see cref="paths"/>.</summary>
		private Mat2x3 transform;

		/// <summary>
		/// The Rust <c>Default</c>: no contours at all.
		/// </summary>
		public CrossSection()
			: this(new Polygons(), Identity)
		{
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
			: this(PositiveUnion(polygons), Identity)
		{
		}

		/// <summary>
		/// The raw constructor behind <see cref="FromRaw"/>, <see cref="Clone"/> and the
		/// transforms: contours plus a pending transform, the Rust's <c>PathState</c>.
		/// </summary>
		/// <remarks>
		/// Stores the list by reference: the Rust <c>from_raw</c> <i>moves</i> its
		/// argument and <c>Clone</c> / <c>transform</c> share it through the <c>Arc</c>, and
		/// every caller here hands over a list it built for the purpose or one that is
		/// already shared and never mutated.
		/// </remarks>
		private CrossSection(Polygons paths, Mat2x3 transform)
		{
			this.paths = paths;
			this.transform = transform;
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
			return new CrossSection(polygons, Identity);
		}

		/// <summary>
		/// The contours with the pending transform applied — C++ <c>GetPaths</c>, the
		/// Rust's <c>paths()</c>. Every reader must go through here.
		/// </summary>
		/// <remarks>
		/// An identity transform (compared with <c>==</c>, so -0.0 counts as zero) returns
		/// the stored contours untouched; otherwise they are transformed into a fresh
		/// list, stored back, and the transform reset to identity, so later transforms
		/// compose from the baked contours. The returned list is shared: callers read it
		/// and never modify it.
		/// </remarks>
		/// <returns>The current contours.</returns>
		internal Polygons Paths()
		{
			lock (this.sync)
			{
				if (this.transform != Identity)
				{
					this.paths = TransformPolygons(this.paths, this.transform);
					this.transform = Identity;
				}

				return this.paths;
			}
		}

		/// <summary>
		/// C++ <c>CrossSection::Transform</c>: a new section sharing these contours with
		/// <paramref name="m"/> composed after the pending transform.
		/// </summary>
		private CrossSection Transform(Mat2x3 m)
		{
			lock (this.sync)
			{
				return new CrossSection(this.paths, Compose(m, this.transform));
			}
		}

		/// <summary>
		/// C++ <c>Mat3(mat2x3)</c> (utils.h): the affine 3x3 with a <c>(0, 0, 1)</c> bottom
		/// row, given as its three columns.
		/// </summary>
		private static (Vec3 C0, Vec3 C1, Vec3 C2) Mat3Cols(Mat2x3 a)
		{
			return (
				new Vec3(a.X.X, a.X.Y, 0.0),
				new Vec3(a.Y.X, a.Y.Y, 0.0),
				new Vec3(a.Z.X, a.Z.Y, 1.0));
		}

		/// <summary>
		/// C++ <c>m * Mat3(t)</c>: each result column is <c>m * column</c>, which
		/// <c>la::mul</c> sums over all three of m's columns, zero entries included.
		/// </summary>
		private static Mat2x3 Compose(Mat2x3 m, Mat2x3 t)
		{
			(Vec3 c0, Vec3 c1, Vec3 c2) = Mat3Cols(t);
			return Mat2x3.FromCols(m * c0, m * c1, m * c2);
		}

		/// <summary>
		/// C++ <c>transform</c> (cross_section.cpp:89-104): every vertex becomes
		/// <c>m * vec3(x, y, 1)</c>, and a negative determinant of the linear part
		/// reverses each contour so outlines stay counter-clockwise.
		/// </summary>
		private static Polygons TransformPolygons(Polygons ps, Mat2x3 m)
		{
			bool invert = (m.X.X * m.Y.Y) - (m.X.Y * m.Y.X) < 0.0;
			Polygons result = new Polygons(ps.Count);
			foreach (SimplePolygon path in ps)
			{
				int sz = path.Count;
				Vec2[] s = new Vec2[sz];
				for (int i = 0; i < sz; i++)
				{
					Vec2 p = path[i];
					int idx = invert ? sz - 1 - i : i;
					s[idx] = m * new Vec3(p.X, p.Y, 1.0);
				}

				result.Add(new SimplePolygon(s));
			}

			return result;
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

		/// <summary>
		/// The contours with any pending transform applied, as an independent deep copy.
		/// </summary>
		/// <returns>A copy of the contour list.</returns>
		public Polygons ToPolygons()
		{
			return ClonePolygons(this.Paths());
		}

		/// <summary>
		/// The Rust <c>Clone</c>, like the C++ copy constructor: the current contours and
		/// pending transform, with the contours shared and the transform not applied.
		/// </summary>
		/// <returns>An equal cross section.</returns>
		public CrossSection Clone()
		{
			lock (this.sync)
			{
				return new CrossSection(this.paths, this.transform);
			}
		}

		/// <summary>
		/// C++ <c>Translate</c>: the transform with columns <c>(1, 0)</c>, <c>(0, 1)</c>,
		/// <paramref name="v"/>.
		/// </summary>
		/// <param name="v">The offset.</param>
		/// <returns>The translated cross section.</returns>
		public CrossSection Translate(Vec2 v)
		{
			return this.Transform(Mat2x3.FromCols(
				new Vec2(1.0, 0.0),
				new Vec2(0.0, 1.0),
				new Vec2(v.X, v.Y)));
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
			foreach (SimplePolygon p in this.Paths())
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
			foreach (SimplePolygon poly in this.Paths())
			{
				foreach (Vec2 p in poly)
				{
					rect.UnionPoint(p);
				}
			}

			return rect;
		}

		/// <summary>
		/// C++ <c>Scale</c>: the transform with columns <c>(x, 0)</c>, <c>(0, y)</c>,
		/// <c>(0, 0)</c>. A negative determinant reverses the winding when applied.
		/// </summary>
		/// <param name="v">The per-axis scale factors.</param>
		/// <returns>The scaled cross section.</returns>
		public CrossSection Scale(Vec2 v)
		{
			return this.Transform(Mat2x3.FromCols(
				new Vec2(v.X, 0.0),
				new Vec2(0.0, v.Y),
				new Vec2(0.0, 0.0)));
		}

		/// <summary>
		/// C++ <c>Rotate</c>: counter-clockwise by <paramref name="degrees"/> about the
		/// origin, with <c>sind</c> / <c>cosd</c> so multiples of 90 degrees are exact.
		/// </summary>
		/// <param name="degrees">The angle in degrees, counter-clockwise.</param>
		/// <returns>The rotated cross section.</returns>
		public CrossSection Rotate(double degrees)
		{
			double s = Types.Sind(degrees);
			double c = Types.Cosd(degrees);
			return this.Transform(Mat2x3.FromCols(
				new Vec2(c, s),
				new Vec2(-s, c),
				new Vec2(0.0, 0.0)));
		}

		/// <summary>
		/// Mirror over the line through the origin whose normal is
		/// <paramref name="axis"/>. Matches C++ <c>CrossSection::Mirror</c>: empty only
		/// when <c>la::length(axis) == 0</c> (underflow included); otherwise
		/// <c>n = normalize(axis)</c> and the transform is
		/// <c>mat2(identity) - 2 * outerprod(n, n)</c>, whose negative determinant
		/// reverses the winding when applied.
		/// </summary>
		/// <param name="axis">The mirror line's normal; a zero-length vector gives an empty result.</param>
		/// <returns>The mirrored cross section.</returns>
		public CrossSection Mirror(Vec2 axis)
		{
			if (LinalgFunctions.Length(axis) == 0.0)
			{
				return new CrossSection();
			}

			Vec2 n = axis / LinalgFunctions.Length(axis);

			// outerprod(n, n) has columns n * n.x and n * n.y. The `0.0 - ...` entries are
			// the identity's zeros minus the product, as C++ subtracts matrices; they give
			// +0.0 where a bare negation would give -0.0.
			Vec2 o0 = new Vec2(n.X * n.X, n.Y * n.X);
			Vec2 o1 = new Vec2(n.X * n.Y, n.Y * n.Y);
			return this.Transform(Mat2x3.FromCols(
				new Vec2(1.0 - (2.0 * o0.X), 0.0 - (2.0 * o0.Y)),
				new Vec2(0.0 - (2.0 * o1.X), 1.0 - (2.0 * o1.Y)),
				new Vec2(0.0, 0.0)));
		}

		/// <summary>
		/// Does the section hold no contours? C++ <c>IsEmpty</c> is <c>paths_.empty()</c>,
		/// so a degenerate contour (such as the empty path C++ <c>Hull</c> returns for
		/// fewer than three points) is not empty.
		/// </summary>
		/// <returns>Whether there are no contours at all.</returns>
		public bool IsEmpty()
		{
			return this.Paths().Count == 0;
		}

		/// <summary>Total vertices over every contour, as C++ <c>NumVert</c>.</summary>
		/// <returns>The vertex count.</returns>
		public int NumVert()
		{
			int sum = 0;
			foreach (SimplePolygon p in this.Paths())
			{
				sum += p.Count;
			}

			return sum;
		}

		/// <summary>
		/// Number of contours, outer and hole, degenerate ones included: C++
		/// <c>NumContour</c> is <c>paths_.size()</c>.
		/// </summary>
		/// <returns>The contour count.</returns>
		public int NumContour()
		{
			return this.Paths().Count;
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
			Polygons paths = this.Paths();
			Polygons polys = new Polygons(paths.Count);
			foreach (SimplePolygon poly in paths)
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
