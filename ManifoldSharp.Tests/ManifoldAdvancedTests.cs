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

// Port of src/manifold_tests/advanced.rs — all 41 of its tests, same inputs,
// same expected values, same tolerances, in the same order. Nothing is deferred:
// advanced.rs touches no part of the robust engine (no RebuildSolid, no robust
// boolean, no non-manifold repair), so every case here runs on modules that
// already landed.
//
// Three files, for the 800-line cap, cutting only at Rust module boundaries so
// the order is still readable straight down:
//   ManifoldAdvancedTests.cs            the six Minkowski cases, the three SDF
//                                       bounds cases, the three Hull cases
//                                       (Rust lines 1–352)
//   ManifoldAdvancedTests.Properties.cs Tolerance/ToleranceSphere, the nine
//                                       MinGap cases, the five triangle-distance
//                                       cases (Rust lines 354–653)
//   ManifoldAdvancedTests.Booleans.cs   TreeTransforms through MeshID, which is
//                                       where the Rust interleaves its seven
//                                       CrossSection cases between boolean ones
//                                       (Rust lines 655–1094)
//
// ── Deliberate near-duplicates, all of them ported ───────────────────────────
// advanced.rs re-states cases that other modules also carry, with DIFFERENT
// inputs, and both sides are load-bearing:
//   MinGapCubeSphereOverlapping / MinGapSphereSphereOutOfBounds /
//   MingapAfterTransformations also exist in api.rs (ManifoldApiTests.cs) at
//   Sphere(1.0, 0) — the quality default — where these run at Sphere(1.0, 32)
//   and Sphere(1.0, 512).
//   MirrorUnion / RoundOffset / Decompose also exist in cross_section2.rs
//   (CrossSection2Tests.cs) on different constructions.
//   The five triangle-distance cases also exist in collider.rs's inline tests
//   (ColliderTests.cs), which check different triangle pairs.
// Nothing here is redundant with those; deleting either copy loses coverage.
//
// ── Trig, per site ───────────────────────────────────────────────────────────
// Same rule as ManifoldTestHelpers.Gyroid's note: a fixture follows the Rust
// PER SITE. Where advanced.rs calls Rust std (`to_radians`, `sin`, `sqrt`,
// `f64::consts::PI`) the C# calls System.Math, not DeterministicMath — swapping
// in the port's own transcendentals would make the C# test a different function
// from the one the Rust test defines. Rust `.min()` is
// LinalgFunctions.MinF64, because f64::min and Math.Min disagree on NaN.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldAdvancedTests
	{
		/// <summary>
		/// C++ TEST(Boolean, ConvexConvexMinkowski) — sphere + cube Minkowski sum.
		/// Checks analytical volume and surface area of a rounded cuboid.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppConvexConvexMinkowski()
		{
			double r = 0.1;
			double w = 2.0;
			Manifold sphere = Manifold.Sphere(r, 20);
			Manifold cube = Manifold.Cube(Vec3.Splat(w), false);
			Manifold sum = cube.MinkowskiSum(sphere);

			double pi = Types.KPi;

			// Analytical volume of rounded cuboid:
			// w³ + 6w²r + 3πwr² + (4/3)πr³
			double analyticalVolume =
				(w * w * w) + (6.0 * w * w * r) + (3.0 * pi * w * r * r) + ((4.0 / 3.0) * pi * r * r * r);

			// Analytical surface area:
			// 6w² + 6πwr + 4πr²
			double analyticalArea = (6.0 * w * w) + (6.0 * pi * w * r) + (4.0 * pi * r * r);

			// Discrete sphere approximation differs from analytical by ~1%
			await Assert.That(Math.Abs(sum.Volume() - analyticalVolume) < 0.15)
				.IsTrue()
				.Because($"ConvexConvexMinkowski volume: {sum.Volume()} expected ~{analyticalVolume}");
			await Assert.That(Math.Abs(sum.SurfaceArea() - analyticalArea) < 0.5)
				.IsTrue()
				.Because($"ConvexConvexMinkowski area: {sum.SurfaceArea()} expected ~{analyticalArea}");
			await Assert.That(sum.Genus()).IsEqualTo(0);
		}

		/// <summary>
		/// C++ TEST(Boolean, ConvexConvexMinkowskiDifference) — sphere erosion of cube.
		/// </summary>
		/// <remarks>
		/// Passes since <c>face2tri</c> was rewritten to the C++ v3.5.0 halfedge-index
		/// pairing scheme (boolean pipeline no longer diverges on coplanar faces).
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppConvexConvexMinkowskiDifference()
		{
			double r = 0.1;
			double w = 2.0;
			Manifold sphere = Manifold.Sphere(r, 20);
			Manifold cube = Manifold.Cube(Vec3.Splat(w), false);
			Manifold difference = cube.MinkowskiDifference(sphere);

			// Analytical volume of eroded cube: (w-2r)³
			double analyticalVolume = (w - (2.0 * r)) * (w - (2.0 * r)) * (w - (2.0 * r));

			// Analytical surface area: 6*(w-2r)²
			double analyticalArea = 6.0 * (w - (2.0 * r)) * (w - (2.0 * r));

			await Assert.That(Math.Abs(difference.Volume() - analyticalVolume) < 0.1)
				.IsTrue()
				.Because(
					$"ConvexConvexMinkowskiDifference volume: {difference.Volume()} "
					+ $"expected ~{analyticalVolume}");
			await Assert.That(Math.Abs(difference.SurfaceArea() - analyticalArea) < 0.1)
				.IsTrue()
				.Because(
					$"ConvexConvexMinkowskiDifference area: {difference.SurfaceArea()} "
					+ $"expected ~{analyticalArea}");
			await Assert.That(difference.Genus()).IsEqualTo(0);
		}

		/// <summary>C++ TEST(Boolean, NonConvexConvexMinkowskiSum).</summary>
		/// <remarks>
		/// Passes since the C++-exact BatchBoolean reduction order (serial-tie-broken
		/// max-heap, rounds of 4), the ForVert-order vertex normals, and the input-space
		/// <c>edgeVec</c> fix in <c>append_partial_edges</c> landed together.
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNonConvexConvexMinkowskiSum()
		{
			Manifold sphere = Manifold.Sphere(1.2, 20);
			Manifold cube = Manifold.Cube(Vec3.Splat(2.0), true);
			Manifold nonConvex = cube.Difference(sphere);
			Manifold sum = nonConvex.MinkowskiSum(Manifold.Sphere(0.1, 20));
			await Assert.That(Math.Abs(sum.Volume() - 4.841) < 1e-3)
				.IsTrue()
				.Because($"NonConvexConvexMinkowskiSum volume: {sum.Volume()} expected ~4.841");
			await Assert.That(Math.Abs(sum.SurfaceArea() - 34.06) < 1e-2)
				.IsTrue()
				.Because($"NonConvexConvexMinkowskiSum area: {sum.SurfaceArea()} expected ~34.06");
			await Assert.That(sum.Genus()).IsEqualTo(5);
		}

		/// <summary>C++ TEST(Boolean, NonConvexConvexMinkowskiDifference).</summary>
		/// <remarks>
		/// Passes since the C++-exact BatchBoolean reduction order, ForVert-order vertex
		/// normals, and the <c>append_partial_edges</c> edgeVec fix landed.
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNonConvexConvexMinkowskiDifference()
		{
			Manifold sphere = Manifold.Sphere(1.2, 20);
			Manifold cube = Manifold.Cube(Vec3.Splat(2.0), true);
			Manifold nonConvex = cube.Difference(sphere);
			Manifold difference = nonConvex.MinkowskiDifference(Manifold.Sphere(0.05, 20));
			await Assert.That(Math.Abs(difference.Volume() - 0.778) < 1e-3)
				.IsTrue()
				.Because(
					$"NonConvexConvexMinkowskiDifference volume: {difference.Volume()} expected ~0.778");
			await Assert.That(Math.Abs(difference.SurfaceArea() - 16.70) < 1e-2)
				.IsTrue()
				.Because(
					$"NonConvexConvexMinkowskiDifference area: {difference.SurfaceArea()} expected ~16.70");
			await Assert.That(difference.Genus()).IsEqualTo(5);
		}

		/// <summary>C++ TEST(Boolean, NonConvexNonConvexMinkowskiSum).</summary>
		/// <remarks>
		/// Passes since the ear-queue FIFO tie-break, the <c>cut_keyhole</c> connector fix,
		/// and the edge_op tag/neighbor-projection fixes landed (2026-07).
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNonConvexNonConvexMinkowskiSum()
		{
			Manifold tet = Manifold.Tetrahedron();
			Manifold nonConvex = tet.Difference(
				Manifold.Tetrahedron()
					.Rotate(0.0, 0.0, 90.0)
					.Translate(Vec3.Splat(1.0)));
			Manifold sum = nonConvex.MinkowskiSum(nonConvex.Scale(Vec3.Splat(0.5)));
			await Assert.That(Math.Abs(sum.Volume() - 8.65625) < 1e-5)
				.IsTrue()
				.Because($"NonConvexNonConvexMinkowskiSum volume: {sum.Volume()} expected ~8.65625");
			await Assert.That(Math.Abs(sum.SurfaceArea() - 31.17691) < 1e-5)
				.IsTrue()
				.Because($"NonConvexNonConvexMinkowskiSum area: {sum.SurfaceArea()} expected ~31.17691");
			await Assert.That(sum.Genus()).IsEqualTo(0);
		}

		/// <summary>C++ TEST(Boolean, NonConvexNonConvexMinkowskiDifference).</summary>
		/// <remarks>
		/// Passes since the matrix <c>rotate</c> + v3.5.0 <c>face2tri</c> pairing landed
		/// (exactly-coplanar boolean faces triangulate manifold, genus matches).
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNonConvexNonConvexMinkowskiDifference()
		{
			Manifold tet = Manifold.Tetrahedron();
			Manifold nonConvex = tet.Difference(
				Manifold.Tetrahedron()
					.Rotate(0.0, 0.0, 90.0)
					.Translate(Vec3.Splat(1.0)));
			Manifold difference = nonConvex.MinkowskiDifference(nonConvex.Scale(Vec3.Splat(0.1)));
			await Assert.That(Math.Abs(difference.Volume() - 0.815542) < 1e-5)
				.IsTrue()
				.Because(
					$"NonConvexNonConvexMinkowskiDifference volume: {difference.Volume()} "
					+ "expected ~0.815542");
			await Assert.That(Math.Abs(difference.SurfaceArea() - 6.95045) < 1e-5)
				.IsTrue()
				.Because(
					$"NonConvexNonConvexMinkowskiDifference area: {difference.SurfaceArea()} "
					+ "expected ~6.95045");
			await Assert.That(difference.Genus()).IsEqualTo(0);
		}

		/// <summary>C++ TEST(SDF, Bounds) — CubeVoid SDF with bounds check.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSdfBounds()
		{
			double size = 4.0;
			double edgeLength = 1.0;

			Manifold cubeVoid = Manifold.LevelSet(
				CubeVoidSdf,
				Box.FromPoints(Vec3.Splat(-size / 2.0), Vec3.Splat(size / 2.0)),
				edgeLength);

			await Assert.That(cubeVoid.IsEmpty())
				.IsFalse()
				.Because("SDF CubeVoid should not be empty");
			await Assert.That(cubeVoid.Genus())
				.IsEqualTo(-1)
				.Because($"SDF CubeVoid genus should be -1, got {cubeVoid.Genus()}");

			double epsilon = cubeVoid.GetTolerance();
			Box bounds = cubeVoid.BoundingBox();
			double outerBound = size / 2.0;
			await Assert.That(Math.Abs(bounds.Min.X - -outerBound) < epsilon + 0.1)
				.IsTrue()
				.Because($"min.x: {bounds.Min.X} expected ~{-outerBound}");
			await Assert.That(Math.Abs(bounds.Max.X - outerBound) < epsilon + 0.1)
				.IsTrue()
				.Because($"max.x: {bounds.Max.X} expected ~{outerBound}");
		}

		/// <summary>C++ TEST(SDF, Bounds3) — Sphere SDF with bounds check.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSdfSphereBounds()
		{
			double radius = 1.2;
			Manifold sphere = Manifold.LevelSet(
				pos => radius - Math.Sqrt((pos.X * pos.X) + (pos.Y * pos.Y) + (pos.Z * pos.Z)),
				Box.FromPoints(Vec3.Splat(-1.0), Vec3.Splat(1.0)),
				0.1);

			await Assert.That(sphere.IsEmpty()).IsFalse().Because("SDF sphere should not be empty");
			await Assert.That(sphere.Genus())
				.IsEqualTo(0)
				.Because($"SDF sphere genus should be 0, got {sphere.Genus()}");

			double epsilon = sphere.GetTolerance();
			Box bounds = sphere.BoundingBox();
			await Assert.That(Math.Abs(bounds.Min.X - -1.0) < epsilon + 0.1)
				.IsTrue()
				.Because($"min.x: {bounds.Min.X} expected ~-1");
			await Assert.That(Math.Abs(bounds.Max.X - 1.0) < epsilon + 0.1)
				.IsTrue()
				.Because($"max.x: {bounds.Max.X} expected ~1");
		}

		/// <summary>C++ TEST(SDF, Void) — Cube minus CubeVoid SDF.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSdfVoid()
		{
			double size = 4.0;
			double edgeLength = 0.5;

			Manifold cubeVoid = Manifold.LevelSet(
				CubeVoidSdf,
				Box.FromPoints(Vec3.Splat(-size / 2.0), Vec3.Splat(size / 2.0)),
				edgeLength);

			Manifold cube = Manifold.Cube(Vec3.Splat(size), true);
			Manifold result = cube.Difference(cubeVoid);

			await Assert.That(result.Genus())
				.IsEqualTo(0)
				.Because($"SDF Void genus: {result.Genus()} expected 0");
			await Assert.That(Math.Abs(result.Volume() - 8.0) < 0.001)
				.IsTrue()
				.Because($"SDF Void volume: {result.Volume()} expected ~8.0");
			await Assert.That(Math.Abs(result.SurfaceArea() - 24.0) < 0.001)
				.IsTrue()
				.Because($"SDF Void area: {result.SurfaceArea()} expected ~24.0");
		}

		/// <summary>
		/// C++ TEST(Hull, Hollow) — hull of hollow sphere equals sphere volume.
		/// C++ uses 360 segments but we use 24 for test speed.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppHullHollow()
		{
			Manifold sphere = Manifold.Sphere(100.0, 24);
			Manifold hollow = sphere.Difference(sphere.Scale(Vec3.Splat(0.8)));
			double sphereVol = sphere.Volume();
			double hullVol = hollow.ConvexHull().Volume();
			await Assert.That(Math.Abs(hullVol - sphereVol) / sphereVol < 0.01)
				.IsTrue()
				.Because($"Hull of hollow sphere: {hullVol} expected ~{sphereVol}");
		}

		/// <summary>C++ TEST(Hull, Cube) — hull of cube with interior points.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppHullCubeWithInterior()
		{
			List<Vec3> pts = new List<Vec3>
			{
				new Vec3(0.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
				new Vec3(0.0, 0.0, 1.0),
				new Vec3(1.0, 1.0, 0.0),
				new Vec3(0.0, 1.0, 1.0),
				new Vec3(1.0, 0.0, 1.0),
				new Vec3(1.0, 1.0, 1.0),
				new Vec3(0.5, 0.5, 0.5),
				new Vec3(0.5, 0.0, 0.0),
				new Vec3(0.5, 0.7, 0.2),
			};
			Manifold cube = Manifold.Hull(pts);
			await Assert.That(Math.Abs(cube.Volume() - 1.0) < 1e-6)
				.IsTrue()
				.Because($"Hull of cube points: {cube.Volume()} expected 1.0");
		}

		/// <summary>C++ TEST(Hull, Empty) — hull of coplanar/too-few points.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppHullEmpty()
		{
			List<Vec3> tooFew = new List<Vec3>
			{
				new Vec3(0.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
			};
			Manifold h = Manifold.Hull(tooFew);
			await Assert.That(h.IsEmpty() || Math.Abs(h.Volume()) < 1e-10)
				.IsTrue()
				.Because("Hull of 3 points should be empty/degenerate");

			List<Vec3> coplanar = new List<Vec3>
			{
				new Vec3(0.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
				new Vec3(1.0, 1.0, 0.0),
			};
			Manifold h2 = Manifold.Hull(coplanar);
			await Assert.That(h2.IsEmpty() || Math.Abs(h2.Volume()) < 1e-10)
				.IsTrue()
				.Because("Hull of coplanar points should be empty/degenerate");
		}

		/// <summary>
		/// The <c>cube_void_sdf</c> closure that CppSdfBounds and CppSdfVoid each declare
		/// inline in the Rust, hoisted to one method because C# cannot spell a local
		/// function twice.
		/// </summary>
		/// <remarks>
		/// Not the same function as ManifoldSdfTests.CubeVoidSdf, despite the name both
		/// inherit from C++: that one is the true signed distance (Euclidean outside),
		/// this one is the negated min-of-slabs form advanced.rs writes. Keep them apart.
		/// </remarks>
		/// <param name="p">The sample point.</param>
		/// <returns>The signed distance.</returns>
		private static double CubeVoidSdf(Vec3 p)
		{
			Vec3 minV = new Vec3(p.X + 1.0, p.Y + 1.0, p.Z + 1.0);
			Vec3 maxV = new Vec3(1.0 - p.X, 1.0 - p.Y, 1.0 - p.Z);
			double min3 = LinalgFunctions.MinF64(minV.X, LinalgFunctions.MinF64(minV.Y, minV.Z));
			double max3 = LinalgFunctions.MinF64(maxV.X, LinalgFunctions.MinF64(maxV.Y, maxV.Z));
			return -1.0 * LinalgFunctions.MinF64(min3, max3);
		}
	}
}
