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

// Port of src/manifold_tests/cross_section2.rs — all 14 tests, same inputs, same
// expected values, same tolerances, in the same order. Nothing deferred.
//
// This is the file that actually covers the CrossSection surface. The inline
// tests in cross_section.rs (ported to CrossSectionTests.cs) reach four methods;
// these fourteen reach Warp, the four fill-rule constructors, Decompose/Compose,
// both Hull entry points, BatchBoolean, Mirror, Rect and the offset family, and
// most of them check the result by *extruding* it and measuring the solid, which
// is why they had to wait for the Phase 6 façade.
//
// Note on the name collision: `test_cpp_cross_section_square` exists twice in the
// Rust — once here (square_vec2, tolerance 1e-4) and once in cross_section.rs's
// own tests module (square, tolerance 1e-6). They are different constructions and
// different tolerances, so both are ported: this one below as CppCrossSectionSquare,
// the other in CrossSectionTests.cs. Same for `test_cpp_cross_section_empty`.
//
// JoinType is passed as the same bare integer the Rust uses (0=Square, 1=Round,
// 2=Miter, 3=Bevel), with the Rust's own comment at each call site — the enum is
// not part of the public surface either port exposes.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class CrossSection2Tests
	{
		/// <summary>C++ TEST(CrossSection, BevelOffset) — bevel join offset of a square.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionBevelOffset()
		{
			CrossSection a = CrossSection.Square(20.0).Translate(new Vec2(-10.0, -10.0));
			int segments = 20;

			// join_type=3 is Bevel
			CrossSection rounded = a.OffsetWithParams(5.0, 3, 2.0, segments);
			Manifold result = Manifold.Extrude(rounded.ToPolygons(), 5.0, 0, 0.0, new Vec2(1.0, 1.0));

			await Assert.That(result.Genus()).IsEqualTo(0).Because($"BevelOffset genus={result.Genus()}");

			// Volume = height * (outer_area - corner_cuts); with bevel, corners cut off
			// triangles of size 5*5
			double expectedVol = 5.0 * (((20.0 + (2.0 * 5.0)) * (20.0 + (2.0 * 5.0))) - (2.0 * 5.0 * 5.0));
			await Assert.That(Math.Abs(result.Volume() - expectedVol) < 1.0)
				.IsTrue()
				.Because($"BevelOffset vol={result.Volume()} expected~{expectedVol}");
			await Assert.That(rounded.NumVert())
				.IsEqualTo(4 + 4)
				.Because($"BevelOffset NumVert={rounded.NumVert()}");
		}

		/// <summary>C++ TEST(CrossSection, Warp) — warp function applied to square vertices.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionWarp()
		{
			CrossSection sq = CrossSection.Square(10.0);
			_ = sq.Scale(new Vec2(2.0, 3.0)).Translate(new Vec2(4.0, 5.0));
			_ = sq.Warp((ref Vec2 v) =>
			{
				v.X = (v.X * 2.0) + 4.0;
				v.Y = (v.Y * 3.0) + 5.0;
			});
			await Assert.That(sq.NumVert()).IsEqualTo(4).Because($"sq NumVert={sq.NumVert()}");
			await Assert.That(sq.NumContour()).IsEqualTo(1).Because($"sq NumContour={sq.NumContour()}");
		}

		/// <summary>
		/// C++ TEST(CrossSection, FillRule) — fill rule affects area of self-intersecting polygon.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionFillRule()
		{
			SimplePolygon polygon = new SimplePolygon
			{
				new Vec2(-7.0, 13.0),
				new Vec2(-7.0, 12.0),
				new Vec2(-5.0, 9.0),
				new Vec2(-5.0, 8.1),
				new Vec2(-4.8, 8.0),
			};

			// fill_rule: 0=EvenOdd, 1=NonZero, 2=Positive, 3=Negative
			CrossSection positive = CrossSection.FromPolygonWithFillRule(polygon, 2);
			await Assert.That(Math.Abs(positive.Area() - 0.683) < 0.001)
				.IsTrue()
				.Because($"Positive area={positive.Area()}");

			CrossSection negative = CrossSection.FromPolygonWithFillRule(polygon, 3);
			await Assert.That(Math.Abs(negative.Area() - 0.193) < 0.001)
				.IsTrue()
				.Because($"Negative area={negative.Area()}");

			CrossSection evenOdd = CrossSection.FromPolygonWithFillRule(polygon, 0);
			await Assert.That(Math.Abs(evenOdd.Area() - 0.875) < 0.001)
				.IsTrue()
				.Because($"EvenOdd area={evenOdd.Area()}");

			CrossSection nonZero = CrossSection.FromPolygonWithFillRule(polygon, 1);
			await Assert.That(Math.Abs(nonZero.Area() - 0.875) < 0.001)
				.IsTrue()
				.Because($"NonZero area={nonZero.Area()}");
		}

		/// <summary>C++ TEST(CrossSection, HullError) — rounded rectangle via hull of circles.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionHullError()
		{
			static CrossSection RoundedRectangle(double x, double y, double radius, int segments)
			{
				CrossSection circ = CrossSection.Circle(radius, segments);
				List<CrossSection> vl = new List<CrossSection>
				{
					circ.Translate(new Vec2(radius, radius)),
					circ.Translate(new Vec2(x - radius, radius)),
					circ.Translate(new Vec2(x - radius, y - radius)),
					circ.Translate(new Vec2(radius, y - radius)),
				};
				return CrossSection.HullCrossSections(vl);
			}

			CrossSection rr = RoundedRectangle(51.0, 36.0, 9.0, 36);
			await Assert.That(Math.Abs(rr.Area() - 1765.1790375) < 1.0)
				.IsTrue()
				.Because($"HullError area={rr.Area()}");
			await Assert.That(rr.NumVert()).IsEqualTo(40).Because($"HullError NumVert={rr.NumVert()}");
		}

		/// <summary>C++ TEST(CrossSection, BatchBoolean) — batch boolean ops on cross sections.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionBatchBoolean()
		{
			CrossSection square = CrossSection.Square(100.0);
			CrossSection circle1 = CrossSection.Circle(30.0, 30).Translate(new Vec2(-10.0, 30.0));
			CrossSection circle2 = CrossSection.Circle(20.0, 30).Translate(new Vec2(110.0, 20.0));
			CrossSection circle3 = CrossSection.Circle(40.0, 30).Translate(new Vec2(50.0, 110.0));

			List<CrossSection> sections = new List<CrossSection>
			{
				square.Clone(),
				circle1.Clone(),
				circle2.Clone(),
				circle3.Clone(),
			};

			CrossSection intersect = CrossSection.BatchBoolean(sections, OpType.Intersect);
			await Assert.That(Math.Abs(intersect.Area()) < 1e-4)
				.IsTrue()
				.Because($"BatchBoolean intersect area={intersect.Area()}");
			await Assert.That(intersect.NumVert())
				.IsEqualTo(0)
				.Because($"BatchBoolean intersect NumVert={intersect.NumVert()}");

			CrossSection add = CrossSection.BatchBoolean(sections, OpType.Add);
			await Assert.That(Math.Abs(add.Area() - 16278.637002) < 1.0)
				.IsTrue()
				.Because($"BatchBoolean add area={add.Area()}");
			await Assert.That(add.NumVert())
				.IsEqualTo(66)
				.Because($"BatchBoolean add NumVert={add.NumVert()}");

			CrossSection subtract = CrossSection.BatchBoolean(sections, OpType.Subtract);
			await Assert.That(Math.Abs(subtract.Area() - 7234.478452) < 1.0)
				.IsTrue()
				.Because($"BatchBoolean subtract area={subtract.Area()}");
			await Assert.That(subtract.NumVert())
				.IsEqualTo(42)
				.Because($"BatchBoolean subtract NumVert={subtract.NumVert()}");
		}

		/// <summary>C++ TEST(CrossSection, Rect) — Rect type and CrossSection(Rect) constructor.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionRect()
		{
			(double w, double h) = (10.0, 5.0);
			Rect rect = new Rect(new Vec2(0.0, 0.0), new Vec2(w, h));
			CrossSection cross = CrossSection.FromRect(rect);
			await Assert.That(Math.Abs(rect.Area() - (w * h)) < 1e-10).IsTrue().Because("Rect area");
			await Assert.That(Math.Abs(rect.Area() - cross.Area()) < 1e-6)
				.IsTrue()
				.Because("Rect vs CrossSection area");
			await Assert.That(rect.ContainsPoint(new Vec2(5.0, 5.0)))
				.IsTrue()
				.Because("Rect contains (5,5)");
			await Assert.That(rect.ContainsRect(cross.Bounds()))
				.IsTrue()
				.Because("Rect contains cross.bounds()");
			await Assert.That(rect.ContainsRect(new Rect())).IsTrue().Because("Rect contains empty rect");
			await Assert.That(rect.DoesOverlap(new Rect(new Vec2(5.0, 5.0), new Vec2(15.0, 15.0))))
				.IsTrue()
				.Because("Rect overlaps shifted rect");
			await Assert.That(new Rect().IsEmpty()).IsTrue().Because("Default Rect is empty");
		}

		/// <summary>
		/// C++ TEST(CrossSection, Square) — extrude of CrossSection::Square matches Manifold::Cube.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionSquare()
		{
			Manifold a = Manifold.Cube(new Vec3(5.0, 5.0, 5.0), false);
			Manifold b = Manifold.Extrude(
				CrossSection.SquareVec2(new Vec2(5.0, 5.0), false).ToPolygons(),
				5.0,
				0,
				0.0,
				new Vec2(1.0, 1.0));
			await Assert.That(Math.Abs(a.Difference(b).Volume()) < 1e-4)
				.IsTrue()
				.Because("Square: a-b volume should be ~0");
		}

		/// <summary>C++ TEST(CrossSection, MirrorUnion) — mirror and union of squares.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionMirrorUnion()
		{
			CrossSection a = CrossSection.SquareVec2(new Vec2(5.0, 5.0), true);
			CrossSection b = a.Translate(new Vec2(2.5, 2.5));
			CrossSection cross = a.Union(b).Union(b.Mirror(new Vec2(1.0, 1.0)));
			await Assert.That(Math.Abs((2.5 * a.Area()) - cross.Area()) < 1e-4)
				.IsTrue()
				.Because($"MirrorUnion: cross.area={cross.Area()} expected ~{2.5 * a.Area()}");
			await Assert.That(a.Mirror(new Vec2(0.0, 0.0)).IsEmpty())
				.IsTrue()
				.Because("MirrorUnion: mirror(0,0) should be empty");
		}

		/// <summary>
		/// C++ TEST(CrossSection, MirrorCheckAxis) — triangle mirrored about (1,1) and (-1,1).
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionMirrorCheckAxis()
		{
			CrossSection tri = new CrossSection(new Polygons
			{
				new SimplePolygon
				{
					new Vec2(0.0, 0.0),
					new Vec2(5.0, 5.0),
					new Vec2(0.0, 10.0),
				},
			});
			Rect a = tri.Mirror(new Vec2(1.0, 1.0)).Bounds();
			Rect aExpected = new CrossSection(new Polygons
			{
				new SimplePolygon
				{
					new Vec2(0.0, 0.0),
					new Vec2(-10.0, 0.0),
					new Vec2(-5.0, -5.0),
				},
			}).Bounds();
			await Assert.That(Math.Abs(a.Min.X - aExpected.Min.X) < 0.001)
				.IsTrue()
				.Because("MirrorCheckAxis a min.x");
			await Assert.That(Math.Abs(a.Min.Y - aExpected.Min.Y) < 0.001)
				.IsTrue()
				.Because("MirrorCheckAxis a min.y");
			await Assert.That(Math.Abs(a.Max.X - aExpected.Max.X) < 0.001)
				.IsTrue()
				.Because("MirrorCheckAxis a max.x");
			await Assert.That(Math.Abs(a.Max.Y - aExpected.Max.Y) < 0.001)
				.IsTrue()
				.Because("MirrorCheckAxis a max.y");

			Rect b = tri.Mirror(new Vec2(-1.0, 1.0)).Bounds();
			Rect bExpected = new CrossSection(new Polygons
			{
				new SimplePolygon
				{
					new Vec2(0.0, 0.0),
					new Vec2(10.0, 0.0),
					new Vec2(5.0, 5.0),
				},
			}).Bounds();
			await Assert.That(Math.Abs(b.Min.X - bExpected.Min.X) < 0.001)
				.IsTrue()
				.Because("MirrorCheckAxis b min.x");
			await Assert.That(Math.Abs(b.Min.Y - bExpected.Min.Y) < 0.001)
				.IsTrue()
				.Because("MirrorCheckAxis b min.y");
			await Assert.That(Math.Abs(b.Max.X - bExpected.Max.X) < 0.001)
				.IsTrue()
				.Because("MirrorCheckAxis b max.x");
			await Assert.That(Math.Abs(b.Max.Y - bExpected.Max.Y) < 0.001)
				.IsTrue()
				.Because("MirrorCheckAxis b max.y");
		}

		/// <summary>C++ TEST(CrossSection, RoundOffset) — round-join offset of centered square.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionRoundOffset()
		{
			CrossSection a = CrossSection.SquareVec2(new Vec2(20.0, 20.0), true);
			int segments = 20;

			// JoinType: 0=Square, 1=Round, 2=Miter, 3=Bevel — use 1 for Round
			CrossSection rounded = a.OffsetWithParams(5.0, 1, 2.0, segments);
			Manifold result = Manifold.Extrude(rounded.ToPolygons(), 5.0, 0, 0.0, new Vec2(1.0, 1.0));
			await Assert.That(result.Genus()).IsEqualTo(0).Because($"RoundOffset genus={result.Genus()}");
			await Assert.That(Math.Abs(result.Volume() - 4386.0) < 1.0)
				.IsTrue()
				.Because($"RoundOffset volume={result.Volume()} expected~4386");
			await Assert.That(rounded.NumVert())
				.IsEqualTo(segments + 4)
				.Because($"RoundOffset NumVert={rounded.NumVert()} expected={segments + 4}");
		}

		/// <summary>C++ TEST(CrossSection, Empty) — CrossSection from empty polygons is empty.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionEmpty()
		{
			CrossSection e = new CrossSection(new Polygons { new SimplePolygon(), new SimplePolygon() });
			await Assert.That(e.IsEmpty()).IsTrue().Because("Empty: should be empty");
		}

		/// <summary>
		/// C++ TEST(CrossSection, Decompose) — decompose+compose round-trip preserves geometry.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionDecompose()
		{
			CrossSection a = CrossSection.SquareVec2(new Vec2(2.0, 2.0), true)
				.Difference(CrossSection.SquareVec2(new Vec2(1.0, 1.0), true));
			CrossSection b = a.Translate(new Vec2(4.0, 4.0));
			CrossSection ab = a.Union(b);
			List<CrossSection> decomp = ab.Decompose();
			CrossSection recomp = CrossSection.Compose(decomp);

			await Assert.That(decomp.Count)
				.IsEqualTo(2)
				.Because($"Decompose: expected 2 pieces, got {decomp.Count}");
			await Assert.That(decomp[0].NumContour())
				.IsEqualTo(2)
				.Because($"Decompose[0]: expected 2 contours, got {decomp[0].NumContour()}");
			await Assert.That(decomp[1].NumContour())
				.IsEqualTo(2)
				.Because($"Decompose[1]: expected 2 contours, got {decomp[1].NumContour()}");

			// Volume of recomposed should match original
			double volAb = Manifold.Extrude(ab.ToPolygons(), 1.0, 0, 0.0, new Vec2(1.0, 1.0)).Volume();
			double volRecomp =
				Manifold.Extrude(recomp.ToPolygons(), 1.0, 0, 0.0, new Vec2(1.0, 1.0)).Volume();
			await Assert.That(Math.Abs(volAb - volRecomp) < 1e-4)
				.IsTrue()
				.Because($"Decompose: recomposed volume={volRecomp} vs original={volAb}");
		}

		/// <summary>C++ TEST(CrossSection, Hull) — hull of circle+translated circles, plus points.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionHull()
		{
			CrossSection circ = CrossSection.Circle(10.0, 360);
			List<CrossSection> circs = new List<CrossSection>
			{
				circ.Clone(),
				circ.Translate(new Vec2(0.0, 30.0)),
				circ.Translate(new Vec2(30.0, 0.0)),
			};

			// C++ uses circ_tri only for optional OBJ export; keep the call so the
			// hull path is still exercised.
			_ = CrossSection.HullCrossSections(circs);
			List<Vec2> centres = new List<Vec2>
			{
				new Vec2(0.0, 0.0),
				new Vec2(0.0, 30.0),
				new Vec2(30.0, 0.0),
				new Vec2(15.0, 5.0),
			};
			CrossSection tri = CrossSection.HullPoints(centres);

			double circArea = circ.Area();

			// hull of (circ - scaled_circ) should equal circ area
			CrossSection annulus = circ.Difference(circ.Scale(new Vec2(0.8, 0.8)));
			CrossSection annulusHull = CrossSection.HullCrossSections(new List<CrossSection> { annulus });
			await Assert.That(Math.Abs(circArea - annulusHull.Area()) < 1.0)
				.IsTrue()
				.Because("Hull: annulus hull area != circ area");

			// batch union of circs minus triangle center hull, area = circ_area * 2.5
			double batchArea = CrossSection.BatchBoolean(circs, OpType.Add)
				.Difference(tri)
				.Area();
			await Assert.That(Math.Abs(batchArea - (circArea * 2.5)) < 1.0)
				.IsTrue()
				.Because($"Hull: batch-minus-tri area={batchArea} expected~{circArea * 2.5}");
		}

		/// <summary>
		/// C++ TEST(CrossSection, NegativeOffset) — negative offset of plus sign gives
		/// circle-cornered square.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionNegativeOffset()
		{
			CrossSection plusSign = CrossSection.SquareVec2(new Vec2(30.0, 50.0), true)
				.Union(CrossSection.SquareVec2(new Vec2(50.0, 30.0), true));

			// JoinType 1 = Round
			CrossSection dilated = plusSign.OffsetWithParams(-10.0, 1, 2.0, 1024);
			double expected = (30.0 * 30.0) - (10.0 * 10.0 * Types.KPi);
			await Assert.That(Math.Abs(dilated.Area() - expected) < 0.01)
				.IsTrue()
				.Because($"NegativeOffset: area={dilated.Area()} expected~{expected}");
		}
	}
}
