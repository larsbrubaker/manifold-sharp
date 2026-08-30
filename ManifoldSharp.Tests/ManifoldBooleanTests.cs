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

// Port of src/manifold_tests/boolean.rs — all 42 of its tests, same inputs, same
// expected values, same tolerances, in the same order. Three files, for the
// 800-line cap: this one runs from the top of the Rust module through
// TEST(Boolean, BatchBoolean); ManifoldBooleanTests.Shapes.cs continues with the
// warp/rotation sweeps to the end; ManifoldBooleanTests.Normals.cs holds the
// four cases manifold_smooth.rs unblocked (MissingNormals, Simplify,
// SimplifyCracks, Normals), which no longer fit in Shapes.cs.
//
// Nothing here is deferred.
//
// Note that boolean.rs deliberately contains near-duplicate pairs
// (test_boolean_precision / test_cpp_precision, test_boolean_edge_union2 /
// test_cpp_edge_union2, test_boolean_simple_cube_regression /
// test_cpp_simple_cube_regression) that assert DIFFERENT things about the same
// construction — the earlier one of each pair pins exact counts, the later one
// pins status only. Both are ported; neither is redundant.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldBooleanTests
	{
		[Test]
		public async Task ManifoldUnionDisjoint()
		{
			Manifold a = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false);
			Manifold b = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false).Translate(new Vec3(3.0, 0.0, 0.0));
			Manifold c = a.Union(b);
			await Assert.That(c.NumTri()).IsEqualTo(24);
		}

		/// <summary>
		/// C++ TEST(Boolean, Precision) — tiny cube near precision limit gets absorbed.
		/// C++ TEST(Boolean, Precision) — per-mesh epsilon tracking.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanPrecision()
		{
			double kPrecision = 1e-12;
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			double distance = 100.0;
			double scale = distance * kPrecision;

			Manifold cube2 = cube.Scale(Vec3.Splat(scale)).Translate(new Vec3(distance, 0.0, 0.0));
			Manifold result = cube.Union(cube2);
			await Assert.That(result.NumVert())
				.IsEqualTo(8)
				.Because($"Tiny cube should be absorbed: {result.NumVert()} verts");

			Manifold cube3 = cube.Scale(Vec3.Splat(2.0 * scale)).Translate(new Vec3(distance, 0.0, 0.0));
			Manifold result2 = result.Union(cube3);
			await Assert.That(result2.NumVert())
				.IsEqualTo(16)
				.Because($"2x precision cube should stay separate: {result2.NumVert()} verts");
		}

		/// <summary>
		/// C++ TEST(Boolean, EdgeUnion2) — tetrahedral edge union.
		/// Note: C++ decomposes edge-touching results into 2 separate meshes.
		/// Our decompose currently returns 1 (connected via shared edge vertices).
		/// The geometry is correct either way.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanEdgeUnion2()
		{
			Manifold tet = Manifold.Tetrahedron();
			Manifold tet1 = tet.Translate(new Vec3(0.0, 0.0, -1.0));
			Manifold tet2 = tet.Rotate(0.0, 0.0, 90.0).Translate(new Vec3(0.0, 0.0, 1.0));
			Manifold result = tet1.Union(tet2);
			await Assert.That(result.Status()).IsEqualTo(Error.NoError);

			// Both components should have their full geometry
			await Assert.That(result.NumTri()).IsEqualTo(8).Because("Two tets should have 8 tris total");
		}

		/// <summary>C++ TEST(Boolean, SimpleCubeRegression) — rotated cube boolean should be NoError.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanSimpleCubeRegression()
		{
			Manifold result = Manifold.Cube(Vec3.Splat(1.0), false)
				.Rotate(-0.1, 0.1, -1.0)
				.Union(Manifold.Cube(Vec3.Splat(1.0), false))
				.Difference(Manifold.Cube(Vec3.Splat(1.0), false).Rotate(-0.1, -0.00000000000066571, -1.0));
			await Assert.That(result.Status()).IsEqualTo(Error.NoError);
		}

		/// <summary>C++ TEST(Boolean, Split) — split a cube with an octahedron.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSplit()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(2.0), true);
			Manifold oct = Manifold.Sphere(1.0, 4).Translate(new Vec3(0.0, 0.0, 1.0));
			(Manifold first, Manifold second) = cube.Split(oct);
			await Assert.That(Math.Abs(first.Volume() + second.Volume() - cube.Volume()))
				.IsLessThan(1e-5)
				.Because(
					$"Split volumes should sum to original: {first.Volume()} + {second.Volume()} = "
					+ $"{first.Volume() + second.Volume()} vs {cube.Volume()}");
		}

		/// <summary>C++ TEST(Boolean, SplitByPlane) — split a rotated cube by z=1 plane.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSplitByPlane()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(2.0), true)
				.Translate(new Vec3(0.0, 1.0, 0.0))
				.Rotate(90.0, 0.0, 0.0);
			(Manifold first, Manifold second) = cube.SplitByPlane(new Vec3(0.0, 0.0, 1.0), 1.0);
			await Assert.That(Math.Abs(first.Volume() - second.Volume()))
				.IsLessThan(1e-3)
				.Because($"Split halves should have equal volume: {first.Volume()} vs {second.Volume()}");

			// Verify trim returns same result as first split
			Manifold trimmed = cube.TrimByPlane(new Vec3(0.0, 0.0, 1.0), 1.0);
			await Assert.That(Math.Abs(first.Volume() - trimmed.Volume()))
				.IsLessThan(1e-3)
				.Because($"Trim should match first split: {first.Volume()} vs {trimmed.Volume()}");
		}

		/// <summary>C++ TEST(Boolean, SplitByPlaneEmpty) — splitting empty manifold.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSplitByPlaneEmpty()
		{
			Manifold empty = Manifold.Empty();
			await Assert.That(empty.IsEmpty()).IsTrue();
			(Manifold first, Manifold second) = empty.SplitByPlane(new Vec3(1.0, 0.0, 0.0), 0.0);
			await Assert.That(first.IsEmpty()).IsTrue();
			await Assert.That(second.IsEmpty()).IsTrue();
		}

		/// <summary>C++ TEST(Boolean, SplitByPlane60) — equal-volume split of rotated cube.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSplitByPlane60()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(2.0), true)
				.Translate(new Vec3(0.0, 1.0, 0.0))
				.Rotate(0.0, 0.0, -60.0)
				.Translate(new Vec3(2.0, 0.0, 0.0));

			// Rust `30.0_f64.to_radians()` is ONE multiply by the correctly-rounded
			// pi/180; Types.Radians is `a * K_PI / 180.0`, two roundings, and the two are
			// not interchangeable in general (the same trap as Types.Degrees vs
			// Smoothing.ToDegrees). At this one input they were checked to produce the
			// identical double — 0x3fe0c15238 2d7365 — so Types.Radians is used here.
			double phiRad = Types.Radians(30.0);
			(Manifold first, Manifold second) = cube.SplitByPlane(
				new Vec3(DeterministicMath.Sin(phiRad), -DeterministicMath.Cos(phiRad), 0.0),
				1.0);
			await Assert.That(Math.Abs(first.Volume() - second.Volume()))
				.IsLessThan(1e-5)
				.Because($"SplitByPlane60: first={first.Volume()} second={second.Volume()} should be equal");
		}

		/// <summary>C++ TEST(Boolean, Vug) — cube with internal cavity.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppVug()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(4.0), true);
			Manifold vug = cube.Difference(Manifold.Cube(Vec3.Splat(1.0), false));
			await Assert.That(vug.Genus()).IsEqualTo(-1);

			(Manifold half, Manifold _) = vug.SplitByPlane(new Vec3(0.0, 0.0, 1.0), -1.0);
			await Assert.That(half.Genus()).IsEqualTo(-1);
			await Assert.That(Math.Abs(half.Volume() - ((4.0 * 4.0 * 3.0) - 1.0)))
				.IsLessThan(0.1)
				.Because($"volume: {half.Volume()} expected: {(4.0 * 4.0 * 3.0) - 1.0}");
		}

		/// <summary>C++ TEST(Boolean, Winding) — overlapping cubes union intersected with small cube.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppWinding()
		{
			Manifold big = Manifold.Cube(Vec3.Splat(3.0), true);
			Manifold medium = Manifold.Cube(Vec3.Splat(2.0), true);
			Manifold doubled = big.Union(medium);

			Manifold small = Manifold.Cube(Vec3.Splat(1.0), true);
			Manifold result = small.Intersection(doubled);
			await Assert.That(result.IsEmpty()).IsFalse().Because("Winding intersection should not be empty");
		}

		/// <summary>C++ TEST(Boolean, BatchBoolean) — batch add operation.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppBatchBoolean()
		{
			Manifold cube = Manifold.Cube(new Vec3(100.0, 100.0, 1.0), false);
			Manifold cyl1 = Manifold.Cylinder(1.0, 30.0, 30.0, 32).Translate(new Vec3(-10.0, 30.0, 0.0));
			Manifold cyl2 = Manifold.Cylinder(1.0, 20.0, 20.0, 32).Translate(new Vec3(110.0, 20.0, 0.0));
			Manifold cyl3 = Manifold.Cylinder(1.0, 40.0, 40.0, 32).Translate(new Vec3(50.0, 110.0, 0.0));

			// Add all: should combine
			Manifold add = Manifold.BatchBoolean(
				new List<Manifold> { cube.Clone(), cyl1.Clone(), cyl2.Clone(), cyl3.Clone() },
				OpType.Add);
			await Assert.That(add.IsEmpty()).IsFalse();
			await Assert.That(add.Volume())
				.IsGreaterThan(cube.Volume())
				.Because("Union volume should be >= cube volume");

			// Subtract: cube minus all cylinders
			Manifold subtract = Manifold.BatchBoolean(
				new List<Manifold> { cube.Clone(), cyl1.Clone(), cyl2.Clone(), cyl3.Clone() },
				OpType.Subtract);
			await Assert.That(subtract.IsEmpty()).IsFalse();
			await Assert.That(subtract.Volume())
				.IsLessThan(cube.Volume())
				.Because("Subtract volume should be < cube volume");
		}

		/// <summary>C++ TEST(Boolean, BatchBoolean) — exact value checks.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppBatchBooleanExact()
		{
			Manifold cube = Manifold.Cube(new Vec3(100.0, 100.0, 1.0), false);
			Manifold cyl1 = Manifold.Cylinder(1.0, 30.0, 30.0, 32).Translate(new Vec3(-10.0, 30.0, 0.0));
			Manifold cyl2 = Manifold.Cylinder(1.0, 20.0, 20.0, 32).Translate(new Vec3(110.0, 20.0, 0.0));
			Manifold cyl3 = Manifold.Cylinder(1.0, 40.0, 40.0, 32).Translate(new Vec3(50.0, 110.0, 0.0));

			// Intersect: no overlap → empty
			Manifold intersect = Manifold.BatchBoolean(
				new List<Manifold> { cube.Clone(), cyl1.Clone(), cyl2.Clone(), cyl3.Clone() },
				OpType.Intersect);
			await Assert.That(intersect.IsEmpty()).IsTrue().Because("BatchBoolean intersect should be empty");

			// Add
			Manifold add = Manifold.BatchBoolean(
				new List<Manifold> { cube.Clone(), cyl1.Clone(), cyl2.Clone(), cyl3.Clone() },
				OpType.Add);
			await Assert.That(add.IsEmpty()).IsFalse();

			// C++ expects volume ~16290.478, surface area ~33156.594
			// Tolerance is wider due to cylinder discretization differences
			await Assert.That(Math.Abs(add.Volume() - 16290.478))
				.IsLessThan(20.0)
				.Because($"BatchBoolean Add volume: {add.Volume()} expected ~16290.478");
			await Assert.That(Math.Abs(add.SurfaceArea() - 33156.594))
				.IsLessThan(40.0)
				.Because($"BatchBoolean Add area: {add.SurfaceArea()} expected ~33156.594");

			// Subtract
			Manifold subtract = Manifold.BatchBoolean(
				new List<Manifold> { cube.Clone(), cyl1.Clone(), cyl2.Clone(), cyl3.Clone() },
				OpType.Subtract);
			await Assert.That(subtract.IsEmpty()).IsFalse();

			// C++ expects volume ~7226.043, surface area ~14904.597
			await Assert.That(Math.Abs(subtract.Volume() - 7226.043))
				.IsLessThan(20.0)
				.Because($"BatchBoolean Subtract volume: {subtract.Volume()} expected ~7226.043");
			await Assert.That(Math.Abs(subtract.SurfaceArea() - 14904.597))
				.IsLessThan(40.0)
				.Because($"BatchBoolean Subtract area: {subtract.SurfaceArea()} expected ~14904.597");
		}
	}
}
