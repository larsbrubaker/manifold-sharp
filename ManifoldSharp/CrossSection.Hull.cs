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

// CrossSection.Hull.cs — the 2D convex hull of cross_section.rs.
//
// Andrew's monotone chain, and nothing else; no Clipper call is involved. This
// is a different algorithm from QuickHull.cs, which is the 3D hull of
// quickhull.rs — the Rust keeps the 2D one local to cross_section.rs for the
// same reason, and the two must not be conflated.
//
// See CrossSection.cs for the file split.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	public sealed partial class CrossSection
	{
		/// <summary>Compute convex hull of all vertices in a slice of CrossSections.</summary>
		/// <param name="sections">The cross sections whose vertices are hulled.</param>
		/// <returns>The hull as a single contour, or empty below three distinct points.</returns>
		public static CrossSection HullCrossSections(IReadOnlyList<CrossSection> sections)
		{
			List<Vec2> points = new List<Vec2>();
			foreach (CrossSection s in sections)
			{
				foreach (SimplePolygon p in s.polygons)
				{
					points.AddRange(p);
				}
			}

			return HullPoints(points);
		}

		/// <summary>Compute convex hull of a set of 2D points (Andrew's monotone chain).</summary>
		/// <param name="points">The input points; not modified.</param>
		/// <returns>The hull as a single contour, or empty below three distinct points.</returns>
		public static CrossSection HullPoints(IReadOnlyList<Vec2> points)
		{
			if (points.Count < 3)
			{
				return new CrossSection();
			}

			// Rust's sort_by is stable and the tie order survives into the dedup below
			// (equal x with equal y is exactly the duplicate case being collapsed), so this
			// is LINQ OrderBy — documented stable — and never List<T>.Sort, an unstable
			// introsort.
			List<Vec2> pts = new List<Vec2>(points.Count);
			foreach (Vec2 a in points.OrderBy(v => v, LexicographicComparer.Instance))
			{
				// Rust `dedup_by(|a, b| ...)`: `b` is the last *retained* element, not the
				// immediately preceding input, so a run of near-duplicates all collapses
				// against the first of the run instead of chaining along it.
				if (pts.Count > 0)
				{
					Vec2 b = pts[pts.Count - 1];
					if (Math.Abs(a.X - b.X) < 1e-10 && Math.Abs(a.Y - b.Y) < 1e-10)
					{
						continue;
					}
				}

				pts.Add(a);
			}

			int n = pts.Count;
			if (n < 3)
			{
				return new CrossSection();
			}

			List<Vec2> hull = new List<Vec2>(2 * n);

			// Lower hull
			foreach (Vec2 p in pts)
			{
				while (hull.Count >= 2 && Cross(hull[hull.Count - 2], hull[hull.Count - 1], p) <= 0.0)
				{
					hull.RemoveAt(hull.Count - 1);
				}

				hull.Add(p);
			}

			// Upper hull
			int lowerLen = hull.Count;
			for (int i = pts.Count - 1; i >= 0; i--)
			{
				Vec2 p = pts[i];
				while (hull.Count > lowerLen && Cross(hull[hull.Count - 2], hull[hull.Count - 1], p) <= 0.0)
				{
					hull.RemoveAt(hull.Count - 1);
				}

				hull.Add(p);
			}

			hull.RemoveAt(hull.Count - 1); // last point == first
			if (hull.Count < 3)
			{
				return new CrossSection();
			}

			return new CrossSection(new Polygons { hull });
		}

		/// <summary>The 2D cross product of (a - o) and (b - o), used by the hull.</summary>
		private static double Cross(Vec2 o, Vec2 a, Vec2 b)
		{
			return ((a.X - o.X) * (b.Y - o.Y)) - ((a.Y - o.Y) * (b.X - o.X));
		}

		/// <summary>
		/// The hull's sort key: x, then y, each compared with f64's <c>partial_cmp</c>
		/// (IEEE, so -0.0 and 0.0 tie).
		/// </summary>
		/// <remarks>
		/// The Rust <c>.unwrap()</c>s the partial comparison and therefore panics on a NaN
		/// coordinate; this reports unordered pairs equal instead, which leaves the
		/// permutation unspecified. A NaN here is a caller bug either way, and the two
		/// ports differ only in how loudly they say so.
		/// </remarks>
		private sealed class LexicographicComparer : IComparer<Vec2>
		{
			public static readonly LexicographicComparer Instance = new LexicographicComparer();

			public int Compare(Vec2 a, Vec2 b)
			{
				if (a.X < b.X)
				{
					return -1;
				}

				if (a.X > b.X)
				{
					return 1;
				}

				if (a.Y < b.Y)
				{
					return -1;
				}

				if (a.Y > b.Y)
				{
					return 1;
				}

				return 0;
			}
		}
	}
}
