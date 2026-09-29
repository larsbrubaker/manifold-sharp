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

// The tail of cross_section_tests.rs, split from CrossSectionTests.cs to keep
// both files under the 800-line cap: the cases that pin C++ Hull's degenerate
// output and the counts that see it. C++ Hull returns one contour even when it
// is degenerate (empty for fewer than three points, two vertices for collinear
// ones), and IsEmpty / NumContour count such contours, as C++ paths_.empty() /
// paths_.size() do. Same inputs, same expected values, same order as the Rust.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class CrossSectionTests
	{
		/// <summary>
		/// C++ <c>IsEmpty</c> is <c>paths_.empty()</c> and <c>NumContour</c> is
		/// <c>paths_.size()</c>: contours with fewer than three vertices still count. C++
		/// <c>Hull</c> produces exactly such sections — one empty contour for fewer than
		/// three points (<c>h_2pts</c>: contours=1 nvert=0 empty=0) and a two-vertex
		/// contour for collinear points (<c>h_collinear</c>: contours=1 nvert=2 empty=0) —
		/// through the private no-union constructor that <c>FromRaw</c> mirrors.
		/// </summary>
		[Test]
		public async Task IsEmptyAndNumContourCountEveryPathLikeCpp()
		{
			CrossSection oneEmpty = CrossSection.FromRaw(new Polygons { new SimplePolygon() });
			await Assert.That(oneEmpty.IsEmpty()).IsFalse();
			await Assert.That(oneEmpty.NumContour()).IsEqualTo(1);
			await Assert.That(oneEmpty.NumVert()).IsEqualTo(0);
			CrossSection degenerate = CrossSection.FromRaw(new Polygons
			{
				new SimplePolygon(),
				new SimplePolygon { new Vec2(0.0, 0.0), new Vec2(3.0, 0.0) },
			});
			await Assert.That(degenerate.IsEmpty()).IsFalse();
			await Assert.That(degenerate.NumContour()).IsEqualTo(2);
			await Assert.That(degenerate.NumVert()).IsEqualTo(2);
			CrossSection none = new CrossSection();
			await Assert.That(none.IsEmpty()).IsTrue();
			await Assert.That(none.NumContour()).IsEqualTo(0);
			await Assert.That(none.NumVert()).IsEqualTo(0);
		}

		/// <summary>
		/// C++ <c>HullImpl</c> (cross_section.cpp:183-206): no near-duplicate removal,
		/// <c>CCW(..., 0.0)</c> backtracking, and a single contour even when degenerate
		/// (empty for fewer than three points, two vertices for collinear ones). Expected
		/// values come from the C++ reference (MSVC).
		/// </summary>
		[Test]
		public async Task HullMatchesCppHullImpl()
		{
			// area 0x3ff0000000001198
			(ulong, ulong)[][] nearDup =
			{
				new[]
				{
					(0x0000000000000000UL, 0x0000000000000000UL),
					(0x3ff0000000000000UL, 0x0000000000000000UL),
					(0x3ff0000000001198UL, 0x3ff0000000001198UL),
					(0x0000000000000000UL, 0x3ff0000000000000UL),
				},
			};

			// area 0x3ff0000000000000
			(ulong, ulong)[][] underflowBits =
			{
				new[]
				{
					(0x0000000000000000UL, 0x0000000000000000UL),
					(0x3ff0000000000000UL, 0xbff0000000000000UL),
					(0x4000000000000000UL, 0x0000000000000000UL),
				},
			};

			// area 0x4017e064f81d2212
			(ulong, ulong)[][] hullCs =
			{
				new[]
				{
					(0xbfeccccccccccccdUL, 0x3fc999999999999aUL),
					(0xbfe36d6b334c0899UL, 0xbfe03a380018d566UL),
					(0x3fb999999999999aUL, 0xbfe999999999999aUL),
					(0x3fe9d3d199b26effUL, 0xbfe03a380018d566UL),
					(0x40096b31d45717eeUL, 0x3ff16daed770771dUL),
					(0x40050fc61e7afa27UL, 0x3ffed8e0abc78f0bUL),
					(0x3fb999999999999aUL, 0x3ff3333333333333UL),
					(0xbfe36d6b334c0899UL, 0x3fed0704cce5a232UL),
				},
			};

			// area 0x4027000000000000
			(ulong, ulong)[][] hullPolys =
			{
				new[]
				{
					(0x0000000000000000UL, 0x0000000000000000UL),
					(0x4008000000000000UL, 0xbff0000000000000UL),
					(0x4010000000000000UL, 0x0000000000000000UL),
					(0x4014000000000000UL, 0x4000000000000000UL),
					(0x4000000000000000UL, 0x4008000000000000UL),
				},
			};

			static Vec2 V(double x, double y) => new Vec2(x, y);

			CrossSection two = CrossSection.HullPoints(new[] { V(0.0, 0.0), V(1.0, 1.0) });
			await Assert.That(PolygonsEqual(two.ToPolygons(), new Polygons { new SimplePolygon() })).IsTrue();
			await Assert.That(two.IsEmpty()).IsFalse();

			CrossSection collinear =
				CrossSection.HullPoints(new[] { V(0.0, 0.0), V(2.0, 0.0), V(1.0, 0.0), V(3.0, 0.0) });
			await Assert.That(PolygonsEqual(
				collinear.ToPolygons(),
				new Polygons { new SimplePolygon { V(0.0, 0.0), V(3.0, 0.0) } })).IsTrue();

			CrossSection same = CrossSection.HullPoints(new[] { V(1.0, 1.0), V(1.0, 1.0), V(1.0, 1.0) });
			await Assert.That(PolygonsEqual(
				same.ToPolygons(),
				new Polygons { new SimplePolygon { V(1.0, 1.0), V(1.0, 1.0) } })).IsTrue();

			CrossSection nearDupHull = CrossSection.HullPoints(new[]
			{
				V(0.0, 0.0),
				V(1.0, 0.0),
				V(1.0, 1.0),
				V(1.0 + 1e-12, 1.0 + 1e-12),
				V(0.0, 1.0),
				V(1e-12, 1.0),
			});
			await Assert.That(HullBitsEqual(nearDupHull.ToPolygons(), nearDup)).IsTrue();
			await Assert.That(BitConverter.DoubleToUInt64Bits(nearDupHull.Area())).IsEqualTo(0x3ff0000000001198UL);

			// area * area * 4 underflows to 0, so CCW(.., 0.0) calls (1, 1e-200)
			// collinear and drops it.
			CrossSection underflow =
				CrossSection.HullPoints(new[] { V(0.0, 0.0), V(1.0, 1e-200), V(2.0, 0.0), V(1.0, -1.0) });
			await Assert.That(HullBitsEqual(underflow.ToPolygons(), underflowBits)).IsTrue();

			CrossSection none = CrossSection.HullCrossSections(Array.Empty<CrossSection>());
			await Assert.That(PolygonsEqual(none.ToPolygons(), new Polygons { new SimplePolygon() })).IsTrue();

			CrossSection secs = CrossSection.HullCrossSections(new[]
			{
				CrossSection.Circle(1.0, 8).Translate(V(0.1, 0.2)),
				CrossSection.SquareVec2(V(2.0, 1.0), false)
					.Rotate(33.0)
					.Translate(V(1.5, 0.0)),
			});
			await Assert.That(HullBitsEqual(secs.ToPolygons(), hullCs)).IsTrue();
			await Assert.That(BitConverter.DoubleToUInt64Bits(secs.Area())).IsEqualTo(0x4017e064f81d2212UL);

			// C++ Hull(Polygons) flattens the contours into one point list.
			CrossSection polys = CrossSection.HullPoints(new[]
			{
				V(0.0, 0.0),
				V(4.0, 0.0),
				V(2.0, 3.0),
				V(1.0, 1.0),
				V(5.0, 2.0),
				V(3.0, -1.0),
			});
			await Assert.That(HullBitsEqual(polys.ToPolygons(), hullPolys)).IsTrue();
		}

		/// <summary>
		/// The Rust test's local <c>assert_eq!(bits(p), want(b))</c>: contour count, vertex
		/// count and every coordinate's bit pattern, in order.
		/// </summary>
		private static bool HullBitsEqual(Polygons p, (ulong X, ulong Y)[][] want)
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
	}
}
