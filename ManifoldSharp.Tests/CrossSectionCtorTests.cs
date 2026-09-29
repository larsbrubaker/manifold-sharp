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

// Port of cross_section_ctor_tests.rs — all 6 cases, same inputs, same expected
// bit patterns, same order. They pin CrossSection's constructors (CrossSection.cs)
// and Manifold.Slice / Project, which wrap their polygons in one, to the C++
// reference compiled with MSVC against Clipper2 46f6391. Nothing deferred.
//
// CircleMatchesCppBits reads the process-global Quality settings — Circle with
// segments <= 2 defers to Quality.GetCircularSegments — so it carries
// TypesTests.QualityGlobalStateKey, as every other test that reads them does.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class CrossSectionCtorTests
	{
		/// <summary>
		/// C++ <c>Circle</c> steps <c>360.0 / n</c> degrees through <c>cosd</c> /
		/// <c>sind</c>, which land exactly on the axes (with the signed zeros below), and
		/// falls back to <c>Quality::GetCircularSegments(radius)</c> when
		/// <c>circularSegments &lt;= 2</c>.
		/// </summary>
		[Test]
		[NotInParallel(TypesTests.QualityGlobalStateKey)]
		public async Task CircleMatchesCppBits()
		{
			(ulong, ulong)[][] circle18 =
			{
				new[]
				{
					(0x3ff0000000000000UL, 0x0000000000000000UL),
					(0x3fe6a09e667f3bccUL, 0x3fe6a09e667f3bccUL),
					(0x8000000000000000UL, 0x3ff0000000000000UL),
					(0xbfe6a09e667f3bccUL, 0x3fe6a09e667f3bccUL),
					(0xbff0000000000000UL, 0x8000000000000000UL),
					(0xbfe6a09e667f3bccUL, 0xbfe6a09e667f3bccUL),
					(0x0000000000000000UL, 0xbff0000000000000UL),
					(0x3fe6a09e667f3bccUL, 0xbfe6a09e667f3bccUL),
				},
			};
			(ulong, ulong)[][] circle20 =
			{
				new[]
				{
					(0x4000000000000000UL, 0x0000000000000000UL),
					(0x3ffbb67ae8584cabUL, 0x3fefffffffffffffUL),
					(0x3fefffffffffffffUL, 0x3ffbb67ae8584cabUL),
					(0x8000000000000000UL, 0x4000000000000000UL),
					(0xbfefffffffffffffUL, 0x3ffbb67ae8584cabUL),
					(0xbffbb67ae8584cabUL, 0x3fefffffffffffffUL),
					(0xc000000000000000UL, 0x8000000000000000UL),
					(0xbffbb67ae8584cabUL, 0xbfefffffffffffffUL),
					(0xbfefffffffffffffUL, 0xbffbb67ae8584cabUL),
					(0x0000000000000000UL, 0xc000000000000000UL),
					(0x3fefffffffffffffUL, 0xbffbb67ae8584cabUL),
					(0x3ffbb67ae8584cabUL, 0xbfefffffffffffffUL),
				},
			};
			await Assert.That(BitsEqual(CrossSection.Circle(1.0, 8).ToPolygons(), circle18)).IsTrue();
			await Assert.That(BitsEqual(CrossSection.Circle(2.0, 0).ToPolygons(), circle20)).IsTrue();
			await Assert.That(BitsEqual(CrossSection.Circle(2.0, 2).ToPolygons(), circle20)).IsTrue();
			await Assert.That(CrossSection.Circle(0.0, 8).IsEmpty()).IsTrue();
		}

		/// <summary>
		/// C++ <c>Square</c> returns empty only for a negative dimension or a zero-length
		/// size vector, so a zero-height rectangle is one degenerate contour; the centered
		/// square starts at (+w/2, +h/2) and runs counter-clockwise.
		/// </summary>
		[Test]
		public async Task SquareMatchesCpp()
		{
			await Assert.That(PolygonsEqual(
				CrossSection.SquareVec2(new Vec2(2.0, 4.0), true).ToPolygons(),
				Contour((1.0, 2.0), (-1.0, 2.0), (-1.0, -2.0), (1.0, -2.0)))).IsTrue();
			await Assert.That(PolygonsEqual(
				CrossSection.SquareVec2(new Vec2(5.0, 0.0), false).ToPolygons(),
				Contour((0.0, 0.0), (5.0, 0.0), (5.0, 0.0), (0.0, 0.0)))).IsTrue();
			await Assert.That(CrossSection.SquareVec2(new Vec2(0.0, 0.0), false).NumVert()).IsEqualTo(0);
			await Assert.That(CrossSection.SquareVec2(new Vec2(-1.0, 1.0), false).NumVert()).IsEqualTo(0);
			await Assert.That(CrossSection.Square(0.0).NumVert()).IsEqualTo(0);
		}

		/// <summary>
		/// C++ <c>CrossSection(const Polygons&amp;, FillRule = Positive)</c> always runs
		/// <c>C2::Union</c>, so overlapping contours merge and coordinates snap to
		/// Clipper2's grid at <c>precision_</c>.
		/// </summary>
		[Test]
		public async Task NewUnionsLikeCppPolygonsCtor()
		{
			(ulong, ulong)[][] triBits =
			{
				new[]
				{
					(0x0000000000000000UL, 0x3ff0000000000000UL),
					(0x0000000000000000UL, 0x0000000000000000UL),
					(0x3ff0000020000000UL, 0x0000000000000000UL),
				},
			};
			Polygons tri = Contour((0.0, 0.0), (1.000_000_12, 0.0), (0.0, 1.0));
			await Assert.That(BitsEqual(new CrossSection(ClonePolys(tri)).ToPolygons(), triBits)).IsTrue();
			await Assert.That(BitsEqual(CrossSection.FromPolygonsFill(tri).ToPolygons(), triBits)).IsTrue();
			static SimplePolygon Sq(double x, double y) => new SimplePolygon
			{
				new Vec2(x, y),
				new Vec2(x + 2.0, y),
				new Vec2(x + 2.0, y + 2.0),
				new Vec2(x, y + 2.0),
			};
			Polygons merged = Contour(
				(2.0, 1.0),
				(3.0, 1.0),
				(3.0, 3.0),
				(1.0, 3.0),
				(1.0, 2.0),
				(0.0, 2.0),
				(0.0, 0.0),
				(2.0, 0.0));
			await Assert.That(PolygonsEqual(
				new CrossSection(new Polygons { Sq(0.0, 0.0), Sq(1.0, 1.0) }).ToPolygons(),
				merged)).IsTrue();
		}

		/// <summary>
		/// C++ <c>CrossSection(const Rect&amp;)</c> wraps the four corners with no union and
		/// no emptiness check, so a default (inverted, infinite) Rect gives one contour of
		/// infinities.
		/// </summary>
		[Test]
		public async Task FromRectMatchesCpp()
		{
			Rect r = new Rect(new Vec2(0.0, 0.0), new Vec2(2.0, 1.0));
			await Assert.That(PolygonsEqual(
				CrossSection.FromRect(r).ToPolygons(),
				Contour((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)))).IsTrue();
			double inf = double.PositiveInfinity;
			await Assert.That(PolygonsEqual(
				CrossSection.FromRect(new Rect()).ToPolygons(),
				Contour((inf, inf), (-inf, inf), (-inf, -inf), (inf, -inf)))).IsTrue();
		}

		/// <summary>
		/// C++ <c>Slice</c> / <c>Project</c> return raw <c>Polygons</c>, and a
		/// CrossSection is made from them only through the Positive-union Polygons
		/// constructor (<c>CrossSection bottom = cube.Slice();</c> in manifold_test.cpp),
		/// so the wrapped sections must equal C++ <c>CrossSection(m.Slice())</c> /
		/// <c>CrossSection(m.Project())</c>. The raw projection already agrees bit for
		/// bit; the raw slice does not (C++ starts from <c>*unordered_set::begin()</c> and
		/// a few interpolated coordinates differ by one ULP), but the union's snap to
		/// Clipper2's grid and its normalized start vertex erase both.
		/// </summary>
		[Test]
		public async Task SliceAndProjectWrapLikeCpp()
		{
			(ulong, ulong)[][] sliceCs =
			{
				new[]
				{
					(0x3fda0e0998000000UL, 0xbfe6a09e68000000UL),
					(0x3fe6a09e68000000UL, 0xbfda0e0998000000UL),
					(0x3fec06075c000000UL, 0x0000000000000000UL),
					(0x3fe6a09e68000000UL, 0x3fda0e0998000000UL),
					(0x3fda0e0998000000UL, 0x3fe6a09e68000000UL),
					(0x0000000000000000UL, 0x3fec06075c000000UL),
					(0xbfda0e0998000000UL, 0x3fe6a09e68000000UL),
					(0xbfe6a09e68000000UL, 0x3fda0e0998000000UL),
					(0xbfec06075c000000UL, 0x0000000000000000UL),
					(0xbfe6a09e68000000UL, 0xbfda0e0998000000UL),
					(0xbfda0e0998000000UL, 0xbfe6a09e68000000UL),
					(0x0000000000000000UL, 0xbfec06075c000000UL),
				},
			};
			(ulong, ulong)[][] projRaw =
			{
				new[]
				{
					(0xbfe6a09e667f3bcdUL, 0xbfe6a09e667f3bcdUL),
					(0x3c91a62633145c07UL, 0xbff0000000000000UL),
					(0x3fe6a09e667f3bcdUL, 0xbfe6a09e667f3bccUL),
					(0x3ff0000000000000UL, 0x3c91a62633145c07UL),
					(0x3fe6a09e667f3bcdUL, 0x3fe6a09e667f3bcdUL),
					(0x3c91a62633145c07UL, 0x3ff0000000000000UL),
					(0xbfe6a09e667f3bccUL, 0x3fe6a09e667f3bcdUL),
					(0xbff0000000000000UL, 0x3c91a62633145c07UL),
				},
			};
			(ulong, ulong)[][] projCs =
			{
				new[]
				{
					(0x3fe6a09e68000000UL, 0xbfe6a09e68000000UL),
					(0x3ff0000000000000UL, 0x0000000000000000UL),
					(0x3fe6a09e68000000UL, 0x3fe6a09e68000000UL),
					(0x0000000000000000UL, 0x3ff0000000000000UL),
					(0xbfe6a09e68000000UL, 0x3fe6a09e68000000UL),
					(0xbff0000000000000UL, 0x0000000000000000UL),
					(0xbfe6a09e68000000UL, 0xbfe6a09e68000000UL),
					(0x0000000000000000UL, 0xbff0000000000000UL),
				},
			};
			Manifold s = Manifold.Sphere(1.0, 8);
			await Assert.That(BitsEqual(s.AsImpl().Project(), projRaw)).IsTrue();
			await Assert.That(BitsEqual(s.Slice(0.3).ToPolygons(), sliceCs)).IsTrue();
			await Assert.That(BitsEqual(s.Project().ToPolygons(), projCs)).IsTrue();
		}

		/// <summary>
		/// C++ <c>Impl::Slice</c> interpolates each crossing with <c>la::lerp(below, above,
		/// a)</c> = <c>below * (1 - a) + above * a</c>; the raw (un-unioned) slice pins that
		/// formula bit-for-bit.
		/// </summary>
		[Test]
		public async Task RawSliceMatchesCppLerpBits()
		{
			(ulong, ulong)[][] sliceRaw =
			{
				new[]
				{
					(0x3fda0e0999cb4467UL, 0xbfe6a09e667f3bccUL),
					(0x3fe6a09e667f3bcdUL, 0xbfda0e0999cb4466UL),
					(0x3fec06075c1a0f52UL, 0x3c91a62633145c07UL),
					(0x3fe6a09e667f3bcdUL, 0x3fda0e0999cb4467UL),
					(0x3fda0e0999cb4467UL, 0x3fe6a09e667f3bcdUL),
					(0x3c91a62633145c07UL, 0x3fec06075c1a0f52UL),
					(0xbfda0e0999cb4466UL, 0x3fe6a09e667f3bcdUL),
					(0xbfe6a09e667f3bccUL, 0x3fda0e0999cb4467UL),
					(0xbfec06075c1a0f52UL, 0x3c91a62633145c07UL),
					(0xbfe6a09e667f3bccUL, 0xbfda0e0999cb4467UL),
					(0xbfda0e0999cb4467UL, 0xbfe6a09e667f3bccUL),
					(0x3c91a62633145c07UL, 0xbfec06075c1a0f52UL),
				},
			};
			Manifold s = Manifold.Sphere(1.0, 8);
			List<List<(ulong, ulong)>> got = Bits(s.AsImpl().Slice(0.3)).Select(CanonicalCycle).ToList();
			List<List<(ulong, ulong)>> expected = sliceRaw.Select(c => CanonicalCycle(c.ToList())).ToList();
			await Assert.That(CyclesEqual(got, expected)).IsTrue();
		}

		/// <summary>
		/// The Rust <c>assert_eq!(bits(p), want(b))</c>: contour count, vertex count and
		/// every coordinate's bit pattern, in order.
		/// </summary>
		private static bool BitsEqual(Polygons p, (ulong X, ulong Y)[][] want)
		{
			if (p.Count != want.Length)
			{
				return false;
			}

			for (int i = 0; i < p.Count; i++)
			{
				if (p[i].Count != want[i].Length)
				{
					return false;
				}

				for (int j = 0; j < p[i].Count; j++)
				{
					if (BitConverter.DoubleToUInt64Bits(p[i][j].X) != want[i][j].X
						|| BitConverter.DoubleToUInt64Bits(p[i][j].Y) != want[i][j].Y)
					{
						return false;
					}
				}
			}

			return true;
		}

		/// <summary>The Rust <c>bits</c>: every coordinate's bit pattern, in order.</summary>
		private static List<List<(ulong, ulong)>> Bits(Polygons p)
		{
			return p.Select(c => c.Select(v => (BitConverter.DoubleToUInt64Bits(v.X), BitConverter.DoubleToUInt64Bits(v.Y))).ToList()).ToList();
		}

		/// <summary>
		/// The Rust <c>canonical_cycle</c>: rotate <paramref name="c"/> so it starts at its
		/// lexicographically smallest vertex (the first such, as <c>min_by_key</c> keeps).
		/// C++ <c>Impl::Slice</c> starts each contour at <c>*tris.begin()</c> of a
		/// <c>std::unordered_set&lt;int&gt;</c>, whose iteration order is
		/// implementation-defined, so only the cyclic sequence of vertices is comparable
		/// across ports.
		/// </summary>
		private static List<(ulong, ulong)> CanonicalCycle(List<(ulong, ulong)> c)
		{
			int start = 0;
			for (int i = 1; i < c.Count; i++)
			{
				if (c[i].CompareTo(c[start]) < 0)
				{
					start = i;
				}
			}

			return c.Skip(start).Concat(c.Take(start)).ToList();
		}

		/// <summary>The Rust <c>assert_eq!</c> on two bit-pattern contour lists, in order.</summary>
		private static bool CyclesEqual(List<List<(ulong, ulong)>> a, List<List<(ulong, ulong)>> b)
		{
			return a.Count == b.Count && a.Zip(b).All(p => p.First.SequenceEqual(p.Second));
		}

		/// <summary>The Rust tests' <c>p</c> closure: one contour from coordinate pairs.</summary>
		private static Polygons Contour(params (double X, double Y)[] c)
		{
			SimplePolygon poly = new SimplePolygon(c.Length);
			foreach ((double x, double y) in c)
			{
				poly.Add(new Vec2(x, y));
			}

			return new Polygons { poly };
		}

		/// <summary>The Rust <c>tri.clone()</c>: a deep copy for the first of two uses.</summary>
		private static Polygons ClonePolys(Polygons p)
		{
			Polygons copy = new Polygons(p.Count);
			foreach (SimplePolygon c in p)
			{
				copy.Add(new SimplePolygon(c));
			}

			return copy;
		}

		/// <summary>
		/// The Rust <c>assert_eq!</c> on two <c>Polygons</c>: contour order, vertex order
		/// and every coordinate under IEEE <c>==</c>.
		/// </summary>
		private static bool PolygonsEqual(Polygons a, Polygons b)
		{
			if (a.Count != b.Count)
			{
				return false;
			}

			for (int i = 0; i < a.Count; i++)
			{
				if (a[i].Count != b[i].Count)
				{
					return false;
				}

				for (int j = 0; j < a[i].Count; j++)
				{
					if (a[i][j] != b[i][j])
					{
						return false;
					}
				}
			}

			return true;
		}
	}
}
