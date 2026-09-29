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

// CrossSection.Hull.cs — the 2D convex hull of cross_section_ops.rs.
//
// An exact port of C++ HullImpl (cross_section.cpp:183-206) as manifold-rust
// b43b4e3 ports it: Andrew's monotone chain over a V2Lesser sort, with
// HullBacktrack's CCW(.., 0.0) test, no near-duplicate removal, and exactly one
// contour out, left degenerate as C++ leaves it (empty for fewer than three
// points, two vertices when every point is collinear). No Clipper call is
// involved. This is a different algorithm from QuickHull.cs, which is the 3D hull
// of quickhull.rs — the Rust keeps the 2D one local to the cross_section modules
// for the same reason, and the two must not be conflated.
//
// The sort must be stable: the Rust's sort_by is, and V2Lesser leaves +0.0 / -0.0
// ties between distinct values equal, so their input order survives into the
// hull. List<T>.Sort (introsort) is not stable; LINQ OrderBy is.
//
// See CrossSection.cs for the file split.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	public sealed partial class CrossSection
	{
		/// <summary>
		/// Convex hull of every vertex of <paramref name="sections"/>, in section then
		/// contour order. Mirrors C++ <c>CrossSection::Hull(const
		/// std::vector&lt;CrossSection&gt;&amp;)</c>, which reads each section through a
		/// by-value copy, so the inputs' own pending transforms stay pending (hence
		/// <c>Clone().Paths()</c>).
		/// </summary>
		/// <param name="sections">The cross sections whose vertices are hulled.</param>
		/// <returns>The hull as a single, possibly degenerate, contour.</returns>
		public static CrossSection HullCrossSections(IReadOnlyList<CrossSection> sections)
		{
			List<Vec2> points = new List<Vec2>();
			foreach (CrossSection s in sections)
			{
				foreach (SimplePolygon path in s.Clone().Paths())
				{
					points.AddRange(path);
				}
			}

			return HullPoints(points);
		}

		/// <summary>
		/// Convex hull of a point set. Mirrors C++ <c>CrossSection::Hull(SimplePolygon)</c>
		/// (and <c>Hull(Polygons)</c>, which flattens its contours into one list): the
		/// result is always exactly one contour, left degenerate as C++ <c>HullImpl</c>
		/// leaves it — empty for fewer than three points, two vertices when every point is
		/// collinear.
		/// </summary>
		/// <param name="points">The input points; not modified.</param>
		/// <returns>The hull as a single, possibly degenerate, contour.</returns>
		public static CrossSection HullPoints(IReadOnlyList<Vec2> points)
		{
			return FromRaw(new Polygons { HullImpl(points) });
		}

		/// <summary>
		/// C++ <c>HullImpl</c>, Andrew's monotone chain: sorts the points with
		/// <c>V2Lesser</c>, builds the lower chain forwards and the upper chain
		/// backwards, drops each chain's last point and returns lower then upper. Fewer
		/// than three points give an empty path.
		/// </summary>
		/// <remarks>
		/// The Rust sorts a copy in place with the stable <c>sort_by</c>; this sorts a
		/// copy with LINQ <c>OrderBy</c>, documented stable, never <c>List.Sort</c>.
		/// </remarks>
		private static SimplePolygon HullImpl(IReadOnlyList<Vec2> points)
		{
			if (points.Count < 3)
			{
				return new SimplePolygon();
			}

			List<Vec2> pts = points.OrderBy(v => v, V2LesserComparer.Instance).ToList();
			List<Vec2> lower = new List<Vec2>();
			foreach (Vec2 pt in pts)
			{
				HullBacktrack(pt, lower);
				lower.Add(pt);
			}

			List<Vec2> upper = new List<Vec2>();
			for (int i = pts.Count - 1; i >= 0; i--)
			{
				Vec2 pt = pts[i];
				HullBacktrack(pt, upper);
				upper.Add(pt);
			}

			upper.RemoveAt(upper.Count - 1);
			lower.RemoveAt(lower.Count - 1);
			lower.AddRange(upper);
			return lower;
		}

		/// <summary>
		/// C++ <c>HullBacktrack</c>: pop while the last two stack points and
		/// <paramref name="pt"/> do not turn strictly counter-clockwise under
		/// <c>CCW(.., 0.0)</c>.
		/// </summary>
		private static void HullBacktrack(Vec2 pt, List<Vec2> stack)
		{
			int sz = stack.Count;
			while (sz >= 2 && Polygon.Ccw(stack[sz - 2], stack[sz - 1], pt, 0.0) <= 0)
			{
				stack.RemoveAt(sz - 1);
				sz = stack.Count;
			}
		}

		/// <summary>
		/// C++ <c>V2Lesser</c>: by x, then by y, under IEEE <c>==</c> / <c>&lt;</c>. As an
		/// ordering, pairs that are neither lesser are equal (only <c>+0.0</c> /
		/// <c>-0.0</c> ties between distinct values) — the Rust's <c>v2_lesser</c>.
		/// </summary>
		/// <remarks>
		/// A NaN coordinate makes the relation inconsistent. The Rust's <c>sort_by</c> may
		/// then panic or return an unspecified permutation; OrderBy returns an unspecified
		/// permutation. A NaN here is a caller bug either way.
		/// </remarks>
		private sealed class V2LesserComparer : IComparer<Vec2>
		{
			public static readonly V2LesserComparer Instance = new V2LesserComparer();

			public int Compare(Vec2 a, Vec2 b)
			{
				if (Lesser(a, b))
				{
					return -1;
				}

				if (Lesser(b, a))
				{
					return 1;
				}

				return 0;
			}

			private static bool Lesser(Vec2 a, Vec2 b)
			{
				if (a.X == b.X)
				{
					return a.Y < b.Y;
				}

				return a.X < b.X;
			}
		}
	}
}
