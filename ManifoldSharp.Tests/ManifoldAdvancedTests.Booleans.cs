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

// src/manifold_tests/advanced.rs, lines 655–1094: TreeTransforms, CornerUnion
// and Perturb1, then the seven CrossSection cases the Rust sandwiches between
// them and the closing EdgeUnion / Precision2 / MeshID. The Rust's order is kept
// rather than regrouped by subject. Continues ManifoldAdvancedTests.cs, which
// carries the module header and the split rationale; read that first.
//
// ── NegativeOffset's wide tolerance is the Rust's, not a slackening ──────────
// CppCrossSectionNegativeOffset keeps advanced.rs:1008's own comment: the
// circular_segments argument reaches Clipper2 only as an ARC TOLERANCE, which
// bounds chord error instead of pinning a segment count, so a round join is not
// guaranteed to land on exactly 1024 segments. CrossSection.Clipper.cs's
// OffsetWithParams remarks carry the same explanation at the code. The ±1.0 band
// is transcribed unchanged; it is not there to absorb a port defect.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldAdvancedTests
	{
		/// <summary>C++ TEST(Boolean, TreeTransforms).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppTreeTransforms()
		{
			Manifold a = (Manifold.Cube(Vec3.Splat(1.0), false) + Manifold.Cube(Vec3.Splat(1.0), false))
				.Translate(new Vec3(1.0, 0.0, 0.0));
			Manifold b = Manifold.Cube(Vec3.Splat(1.0), false) + Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold result = a + b;
			await Assert.That(Math.Abs(result.Volume() - 2.0) < 1e-4)
				.IsTrue()
				.Because($"TreeTransforms volume: {result.Volume()} expected 2.0");
		}

		/// <summary>C++ TEST(Boolean, CornerUnion).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCornerUnion()
		{
			Manifold c = Manifold.Cube(Vec3.Splat(1.0), false);

			// The Rust writes `c.clone() + c.translate(...)` only because `translate`
			// consumes its receiver there; Manifold is a reference type here and Translate
			// returns a new one, so `c` itself is the left operand.
			Manifold cubes = c + c.Translate(new Vec3(1.0, 1.0, 1.0));

			// Should be two disjoint cubes (touching at a corner only)
			await Assert.That(cubes.NumVert())
				.IsEqualTo(16)
				.Because($"CornerUnion verts: {cubes.NumVert()} expected 16");
			await Assert.That(cubes.NumTri())
				.IsEqualTo(24)
				.Because($"CornerUnion tris: {cubes.NumTri()} expected 24");
		}

		/// <summary>C++ TEST(Boolean, Perturb1).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPerturb1()
		{
			// Diamond with square hole
			Manifold big = Manifold.Extrude(
				new Polygons
				{
					new SimplePolygon
					{
						new Vec2(0.0, 2.0),
						new Vec2(2.0, 0.0),
						new Vec2(4.0, 2.0),
						new Vec2(2.0, 4.0),
					},
					new SimplePolygon
					{
						new Vec2(1.0, 2.0),
						new Vec2(2.0, 3.0),
						new Vec2(3.0, 2.0),
						new Vec2(2.0, 1.0),
					},
				},
				1.0,
				0,
				0.0,
				new Vec2(1.0, 1.0));
			Manifold little = Manifold.Extrude(
				new Polygons
				{
					new SimplePolygon
					{
						new Vec2(2.0, 1.0),
						new Vec2(3.0, 2.0),
						new Vec2(2.0, 3.0),
						new Vec2(1.0, 2.0),
					},
				},
				1.0,
				0,
				0.0,
				new Vec2(1.0, 1.0))
				.Translate(new Vec3(0.0, 0.0, 1.0));
			Manifold punchHole = Manifold.Extrude(
				new Polygons
				{
					new SimplePolygon
					{
						new Vec2(1.0, 2.0),
						new Vec2(2.0, 2.0),
						new Vec2(2.0, 3.0),
					},
				},
				1.0,
				0,
				0.0,
				new Vec2(1.0, 1.0))
				.Translate(new Vec3(0.0, 0.0, 1.0));
			Manifold result = (big + little) - punchHole;
			await Assert.That(result.NumDegenerateTris())
				.IsEqualTo(0)
				.Because("Perturb1: has degenerate tris");
			await Assert.That(result.NumVert())
				.IsEqualTo(24)
				.Because($"Perturb1 verts: {result.NumVert()} expected 24");
			await Assert.That(Math.Abs(result.Volume() - 7.5) < 1e-4)
				.IsTrue()
				.Because($"Perturb1 volume: {result.Volume()} expected 7.5");
			await Assert.That(Math.Abs(result.SurfaceArea() - 38.2) < 0.1)
				.IsTrue()
				.Because($"Perturb1 SA: {result.SurfaceArea()} expected ~38.2");
		}

		/// <summary>
		/// C++ TEST(CrossSection, MirrorUnion) — CrossSection mirror and union.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionMirrorUnion()
		{
			// C++ uses CrossSection::Square({5,5}, true) which centers at origin
			CrossSection a = new CrossSection(new Polygons
			{
				new SimplePolygon
				{
					new Vec2(-2.5, -2.5),
					new Vec2(2.5, -2.5),
					new Vec2(2.5, 2.5),
					new Vec2(-2.5, 2.5),
				},
			});
			CrossSection b = a.Translate(new Vec2(2.5, 2.5));
			CrossSection cross = a.Union(b).Union(b.Mirror(new Vec2(1.0, 1.0)));

			// The Rust binds this to `_result`: extruding is the check that the union is
			// extrudable at all, and nothing is asserted about the solid.
			_ = Manifold.Extrude(cross.ToPolygons(), 5.0, 0, 0.0, new Vec2(1.0, 1.0));

			await Assert.That(Math.Abs(cross.Area() - (2.5 * a.Area())) < 1.0)
				.IsTrue()
				.Because($"MirrorUnion area: {cross.Area()} expected {2.5 * a.Area()}");
			await Assert.That(a.Mirror(new Vec2(0.0, 0.0)).IsEmpty())
				.IsTrue()
				.Because("Mirror with zero axis should be empty");
		}

		/// <summary>
		/// C++ TEST(CrossSection, RoundOffset) — CrossSection offset with round joins.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionRoundOffset()
		{
			CrossSection a = CrossSection.Square(20.0).Translate(new Vec2(-10.0, -10.0));
			CrossSection rounded = a.Offset(5.0);
			Manifold result = Manifold.Extrude(rounded.ToPolygons(), 5.0, 0, 0.0, new Vec2(1.0, 1.0));

			await Assert.That(result.Genus())
				.IsEqualTo(0)
				.Because($"RoundOffset genus: {result.Genus()} expected 0");
			await Assert.That(Math.Abs(result.Volume() - 4386.0) < 50.0)
				.IsTrue()
				.Because($"RoundOffset volume: {result.Volume()} expected ~4386");
		}

		/// <summary>
		/// C++ TEST(CrossSection, Decompose) — decompose disjoint cross sections.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionDecompose()
		{
			CrossSection a = CrossSection.Square(2.0)
				.Translate(new Vec2(-1.0, -1.0))
				.Difference(CrossSection.Square(1.0).Translate(new Vec2(-0.5, -0.5)));
			CrossSection b = a.Translate(new Vec2(4.0, 4.0));
			CrossSection ab = a.Union(b);
			List<CrossSection> decomp = ab.Decompose();

			await Assert.That(decomp.Count)
				.IsEqualTo(2)
				.Because($"Decompose should produce 2 components, got {decomp.Count}");
			await Assert.That(decomp[0].NumContour())
				.IsEqualTo(2)
				.Because($"Component 0 should have 2 contours, got {decomp[0].NumContour()}");
			await Assert.That(decomp[1].NumContour())
				.IsEqualTo(2)
				.Because($"Component 1 should have 2 contours, got {decomp[1].NumContour()}");
		}

		/// <summary>
		/// C++ TEST(CrossSection, Transform) — CrossSection transform operations.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionTransform()
		{
			CrossSection sq = CrossSection.Square(10.0);
			CrossSection a = sq
				.Rotate(45.0)
				.Scale(new Vec2(2.0, 3.0))
				.Translate(new Vec2(4.0, 5.0));

			Manifold exA = Manifold.Extrude(a.ToPolygons(), 1.0, 0, 0.0, new Vec2(1.0, 1.0));

			// Verify the result is valid and has the expected area
			// Original square area = 100, scaled by 2*3 = 600
			await Assert.That(Math.Abs(a.Area() - 600.0) < 1.0)
				.IsTrue()
				.Because($"Transform area: {a.Area()} expected ~600");
			await Assert.That(exA.IsEmpty())
				.IsFalse()
				.Because("Transform extrusion should not be empty");
		}

		/// <summary>
		/// C++ TEST(CrossSection, MirrorCheckAxis) — verify mirror along (1,1) and (-1,1).
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
				.Because($"MirrorCheckAxis a: min.x {a.Min.X} vs {aExpected.Min.X}");
			await Assert.That(Math.Abs(a.Min.Y - aExpected.Min.Y) < 0.001)
				.IsTrue()
				.Because($"MirrorCheckAxis a: min.y {a.Min.Y} vs {aExpected.Min.Y}");
			await Assert.That(Math.Abs(a.Max.X - aExpected.Max.X) < 0.001)
				.IsTrue()
				.Because($"MirrorCheckAxis a: max.x {a.Max.X} vs {aExpected.Max.X}");
			await Assert.That(Math.Abs(a.Max.Y - aExpected.Max.Y) < 0.001)
				.IsTrue()
				.Because($"MirrorCheckAxis a: max.y {a.Max.Y} vs {aExpected.Max.Y}");

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
				.Because($"MirrorCheckAxis b: min.x {b.Min.X} vs {bExpected.Min.X}");
			await Assert.That(Math.Abs(b.Min.Y - bExpected.Min.Y) < 0.001)
				.IsTrue()
				.Because($"MirrorCheckAxis b: min.y {b.Min.Y} vs {bExpected.Min.Y}");
			await Assert.That(Math.Abs(b.Max.X - bExpected.Max.X) < 0.001)
				.IsTrue()
				.Because($"MirrorCheckAxis b: max.x {b.Max.X} vs {bExpected.Max.X}");
			await Assert.That(Math.Abs(b.Max.Y - bExpected.Max.Y) < 0.001)
				.IsTrue()
				.Because($"MirrorCheckAxis b: max.y {b.Max.Y} vs {bExpected.Max.Y}");
		}

		/// <summary>C++ TEST(CrossSection, Rect) — rect area, contains, overlap.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionRect()
		{
			double w = 10.0;
			double h = 5.0;
			Rect rect = Rect.FromPoints(new Vec2(0.0, 0.0), new Vec2(w, h));

			// Build a CrossSection from the rect
			CrossSection cross = new CrossSection(new Polygons
			{
				new SimplePolygon
				{
					new Vec2(0.0, 0.0),
					new Vec2(w, 0.0),
					new Vec2(w, h),
					new Vec2(0.0, h),
				},
			});
			double area = rect.Area();

			await Assert.That(Math.Abs(area - (w * h)) < 1e-6)
				.IsTrue()
				.Because($"Rect area: {area} expected {w * h}");
			await Assert.That(Math.Abs(area - cross.Area()) < 1e-6)
				.IsTrue()
				.Because($"Rect area {area} != CrossSection area {cross.Area()}");
			await Assert.That(rect.ContainsPoint(new Vec2(5.0, 5.0)))
				.IsTrue()
				.Because("Rect should contain (5,5)");
			await Assert.That(rect.ContainsRect(cross.Bounds()))
				.IsTrue()
				.Because("Rect should contain cross bounds");
			await Assert.That(rect.ContainsRect(new Rect()))
				.IsTrue()
				.Because("Rect should contain empty rect");
			await Assert.That(rect.DoesOverlap(Rect.FromPoints(
					new Vec2(5.0, 5.0),
					new Vec2(15.0, 15.0))))
				.IsTrue()
				.Because("Rect should overlap shifted rect");
			await Assert.That(new Rect().IsEmpty()).IsTrue().Because("Default Rect should be empty");
		}

		/// <summary>
		/// C++ TEST(CrossSection, NegativeOffset) — inward offset on plus sign.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionNegativeOffset()
		{
			// CrossSection::Square({30, 50}, true) = centered 30x50 rect
			CrossSection sq1 = new CrossSection(new Polygons
			{
				new SimplePolygon
				{
					new Vec2(-15.0, -25.0),
					new Vec2(15.0, -25.0),
					new Vec2(15.0, 25.0),
					new Vec2(-15.0, 25.0),
				},
			});
			CrossSection sq2 = new CrossSection(new Polygons
			{
				new SimplePolygon
				{
					new Vec2(-25.0, -15.0),
					new Vec2(25.0, -15.0),
					new Vec2(25.0, 15.0),
					new Vec2(-25.0, 15.0),
				},
			});
			CrossSection plusSign = sq1.Union(sq2);

			// offset_with_params: join_type 1=Round, miter_limit=2.0
			CrossSection dilated = plusSign.OffsetWithParams(-10.0, 1, 2.0, 1024);
			double expected = (30.0 * 30.0) - (10.0 * 10.0 * Types.KPi);

			// Tolerance is wider because circular_segments param isn't fully wired
			// through to clipper2 arc precision yet
			await Assert.That(Math.Abs(dilated.Area() - expected) < 1.0)
				.IsTrue()
				.Because($"NegativeOffset area: {dilated.Area()} expected {expected}");
		}

		/// <summary>
		/// C++ TEST(Boolean, EdgeUnion) — two cubes sharing an edge stay disjoint.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppBooleanEdgeUnion()
		{
			Manifold cubes = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false);
			cubes = cubes.Union(cubes.Translate(new Vec3(1.0, 1.0, 0.0)));
			await Assert.That(cubes.IsEmpty()).IsFalse().Because("EdgeUnion should not be empty");

			// Two disjoint cubes: 16 verts, 24 tris total
			await Assert.That(cubes.NumVert())
				.IsEqualTo(16)
				.Because($"EdgeUnion: {cubes.NumVert()} verts expected 16");
			await Assert.That(cubes.NumTri())
				.IsEqualTo(24)
				.Because($"EdgeUnion: {cubes.NumTri()} tris expected 24");
		}

		/// <summary>
		/// C++ TEST(Boolean, Precision2) — cubes that barely overlap vs barely don't.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppBooleanPrecision2()
		{
			double scale = 1000.0;
			double kPrecision = Types.KPrecision;
			Manifold cube = Manifold.Cube(Vec3.Splat(scale), false);
			double distance = scale * (1.0 - (kPrecision / 2.0));

			Manifold cube2 = cube.Translate(Vec3.Splat(-distance));
			Manifold intersection = cube.Intersection(cube2);
			await Assert.That(intersection.IsEmpty())
				.IsTrue()
				.Because(
					"Precision2: cubes offset by scale*(1-kPrec/2) should have empty intersection");

			Manifold cube3 = cube2.Translate(Vec3.Splat(scale * kPrecision));
			Manifold intersection2 = cube.Intersection(cube3);
			await Assert.That(intersection2.IsEmpty())
				.IsFalse()
				.Because("Precision2: cubes shifted back by scale*kPrec should intersect");
		}

		/// <summary>
		/// C++ TEST(Manifold, MeshID) — two imports of same MeshGL get different IDs.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppManifoldMeshId()
		{
			Manifold cube = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false);
			MeshGL cubeGl = cube.GetMeshGL(3);
			cubeGl.RunIndex.Clear();
			cubeGl.RunOriginalId.Clear();
			cubeGl.RunTransform.Clear();
			Manifold cube1 = Manifold.FromMeshGL(cubeGl);
			Manifold cube2 = Manifold.FromMeshGL(cubeGl);
			await Assert.That(cube1.IsEmpty())
				.IsFalse()
				.Because($"cube1 should not be empty, status={cube1.Status()}");
			await Assert.That(cube2.IsEmpty())
				.IsFalse()
				.Because($"cube2 should not be empty, status={cube2.Status()}");
			MeshGL gl1 = cube1.GetMeshGL(3);
			await Assert.That(gl1.RunOriginalId.Count == 0)
				.IsFalse()
				.Because("gl1 should have run_original_id");
			uint id1 = gl1.RunOriginalId[0];
			uint id2 = cube2.GetMeshGL(3).RunOriginalId[0];
			await Assert.That(id1)
				.IsNotEqualTo(id2)
				.Because($"MeshID: two imports should get different IDs: {id1} vs {id2}");
		}
	}
}
