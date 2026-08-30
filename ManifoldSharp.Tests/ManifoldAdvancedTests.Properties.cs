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

// src/manifold_tests/advanced.rs, lines 354–653: the two Properties tolerance
// cases, the nine MinGap cases, and the five triangle-distance cases. Continues
// ManifoldAdvancedTests.cs, which carries the module header and the split
// rationale; read that first.

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
		/// C++ TEST(Properties, Tolerance) — refine_to_tolerance, check tri count.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesTolerance()
		{
			double degrees = 1.0;

			// Rust `degrees.to_radians()` is ONE multiply by the correctly-rounded pi/180;
			// Types.Radians is `a * K_PI / 180.0`, two roundings, and the two are not
			// interchangeable in general. Spelled out here rather than argued equal.
			// `.sin()` is Rust std, so System.Math.Sin — see the per-site trig note in
			// ManifoldAdvancedTests.cs.
			double tol = Math.Sin(degrees * (Math.PI / 180.0));
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);
			Manifold imperfect = cube.Intersection(cube.Rotate(degrees, 0.0, 0.0)).AsOriginal();
			await Assert.That(imperfect.NumTri())
				.IsEqualTo(28)
				.Because($"Tolerance imperfect: {imperfect.NumTri()} tris expected 28");

			Manifold imperfect2 = imperfect.Simplify(tol);
			await Assert.That(imperfect2.NumTri())
				.IsEqualTo(12)
				.Because($"Tolerance simplified: {imperfect2.NumTri()} tris expected 12");

			await Assert.That(Math.Abs(imperfect.Volume() - imperfect2.Volume()) < 0.01)
				.IsTrue()
				.Because($"Tolerance volumes: {imperfect.Volume()} vs {imperfect2.Volume()}");
			await Assert.That(Math.Abs(imperfect.SurfaceArea() - imperfect2.SurfaceArea()) < 0.02)
				.IsTrue()
				.Because(
					$"Tolerance areas: {imperfect.SurfaceArea()} vs {imperfect2.SurfaceArea()}");
		}

		/// <summary>C++ TEST(Properties, ToleranceSphere) — sphere set_tolerance.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesToleranceSphere()
		{
			int n = 1000;
			Manifold sphere = Manifold.Sphere(1.0, 4 * n);
			await Assert.That(sphere.NumTri()).IsEqualTo(8 * n * n);

			Manifold sphere2 = sphere.SetTolerance(0.01);
			await Assert.That(sphere2.NumTri() < 2500)
				.IsTrue()
				.Because($"ToleranceSphere: {sphere2.NumTri()} tris expected < 2500");
			await Assert.That(sphere2.Genus()).IsEqualTo(0);
			await Assert.That(Math.Abs(sphere.Volume() - sphere2.Volume()) < 0.05).IsTrue();
			await Assert.That(Math.Abs(sphere.SurfaceArea() - sphere2.SurfaceArea()) < 0.06).IsTrue();
		}

		/// <summary>C++ TEST(Properties, MinGapCubeCube).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesMinGapCubeCube()
		{
			Manifold a = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold b = Manifold.Cube(Vec3.Splat(1.0), false).Translate(new Vec3(2.0, 2.0, 0.0));
			double distance = a.MinGap(b, 1.5);
			await Assert.That(Math.Abs(distance - Math.Sqrt(2.0)) < 1e-4)
				.IsTrue()
				.Because($"MinGapCubeCube: {distance} expected {Math.Sqrt(2.0)}");
		}

		/// <summary>C++ TEST(Properties, MinGapCubeCube2).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesMinGapCubeCube2()
		{
			Manifold a = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold b = Manifold.Cube(Vec3.Splat(1.0), false).Translate(new Vec3(3.0, 3.0, 0.0));
			double distance = a.MinGap(b, 3.0);
			await Assert.That(Math.Abs(distance - (2.0 * Math.Sqrt(2.0))) < 1e-4)
				.IsTrue()
				.Because($"MinGapCubeCube2: {distance} expected {2.0 * Math.Sqrt(2.0)}");
		}

		/// <summary>C++ TEST(Properties, MinGapClosestPointOnEdge).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesMinGapEdge()
		{
			Manifold a = Manifold.Cube(Vec3.Splat(1.0), true).Rotate(0.0, 0.0, 45.0);
			Manifold b = Manifold.Cube(Vec3.Splat(1.0), true)
				.Rotate(0.0, 45.0, 0.0)
				.Translate(new Vec3(2.0, 0.0, 0.0));
			double distance = a.MinGap(b, 0.7);
			await Assert.That(Math.Abs(distance - (2.0 - Math.Sqrt(2.0))) < 1e-4)
				.IsTrue()
				.Because($"MinGapEdge: {distance} expected {2.0 - Math.Sqrt(2.0)}");
		}

		/// <summary>C++ TEST(Properties, MinGapClosestPointOnTriangleFace).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesMinGapFace()
		{
			Manifold a = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold b = Manifold.Cube(Vec3.Splat(1.0), false)
				.Scale(new Vec3(10.0, 10.0, 10.0))
				.Translate(new Vec3(2.0, -5.0, -1.0));
			double distance = a.MinGap(b, 1.1);
			await Assert.That(Math.Abs(distance - 1.0) < 1e-4)
				.IsTrue()
				.Because($"MinGapFace: {distance} expected 1.0");
		}

		/// <summary>
		/// C++ TEST(Properties, MinGapSphereSphereOutOfBounds) /
		/// MinGapAfterTransformationsOutOfBounds.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesMinGapOutOfBounds()
		{
			Manifold a = Manifold.Sphere(1.0, 32);
			Manifold b = Manifold.Sphere(1.0, 32).Translate(new Vec3(2.0, 2.0, 0.0));
			double search = 0.8;
			double distance = a.MinGap(b, search);

			// Out of bounds: distance returned should be the search_length
			await Assert.That(Math.Abs(distance - search) < 0.01)
				.IsTrue()
				.Because($"MinGapOutOfBounds: {distance} expected {search}");
		}

		/// <summary>C++ TEST(Properties, MinGapCubeSphereOverlapping).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesMinGapOverlapping()
		{
			Manifold a = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold b = Manifold.Sphere(1.0, 32);
			double distance = a.MinGap(b, 0.1);
			await Assert.That(Math.Abs(distance) < 1e-4)
				.IsTrue()
				.Because($"MinGapOverlapping: {distance} expected 0");
		}

		/// <summary>C++ TEST(Properties, MinGapSphereSphere).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesMinGapSphereSphere()
		{
			Manifold a = Manifold.Sphere(1.0, 32);
			Manifold b = Manifold.Sphere(1.0, 32).Translate(new Vec3(2.0, 2.0, 0.0));
			double distance = a.MinGap(b, 0.85);
			double expected = (2.0 * Math.Sqrt(2.0)) - 2.0;
			await Assert.That(Math.Abs(distance - expected) < 1e-4)
				.IsTrue()
				.Because($"MinGapSphereSphere: {distance} expected {expected}");
		}

		/// <summary>C++ TEST(Properties, MingapAfterTransformations).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesMinGapTransformed()
		{
			Manifold a = Manifold.Sphere(1.0, 512).Rotate(30.0, 30.0, 30.0);
			Manifold b = Manifold.Sphere(1.0, 512)
				.Scale(new Vec3(3.0, 1.0, 1.0))
				.Rotate(0.0, 90.0, 45.0)
				.Translate(new Vec3(3.0, 0.0, 0.0));
			double distance = a.MinGap(b, 1.1);
			await Assert.That(Math.Abs(distance - 1.0) < 0.001)
				.IsTrue()
				.Because($"MinGapTransformed: {distance} expected ~1.0");
		}

		/// <summary>C++ TEST(Properties, MinGapAfterTransformationsOutOfBounds).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesMinGapTransformedOob()
		{
			Manifold a = Manifold.Sphere(1.0, 512).Rotate(30.0, 30.0, 30.0);
			Manifold b = Manifold.Sphere(1.0, 512)
				.Scale(new Vec3(3.0, 1.0, 1.0))
				.Rotate(0.0, 90.0, 45.0)
				.Translate(new Vec3(3.0, 0.0, 0.0));
			double distance = a.MinGap(b, 0.95);
			await Assert.That(Math.Abs(distance - 0.95) < 0.001)
				.IsTrue()
				.Because($"MinGapTransformedOOB: {distance} expected ~0.95");
		}

		/// <summary>C++ TEST(Properties, TriangleDistanceClosestPointsOnVertices).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppTriangleDistanceVertices()
		{
			Vec3[] p =
			{
				new Vec3(-1.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
			};
			Vec3[] q =
			{
				new Vec3(2.0, 0.0, 0.0),
				new Vec3(4.0, 0.0, 0.0),
				new Vec3(3.0, 1.0, 0.0),
			};
			double distance = ColliderFunctions.DistanceTriangleTriangleSquared(p, q);
			await Assert.That(Math.Abs(distance - 1.0) < 1e-6)
				.IsTrue()
				.Because($"TriangleDistanceVertices: {distance} expected 1.0");
		}

		/// <summary>C++ TEST(Properties, TriangleDistanceClosestPointOnEdge).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppTriangleDistanceEdge()
		{
			Vec3[] p =
			{
				new Vec3(-1.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
			};
			Vec3[] q =
			{
				new Vec3(-1.0, 2.0, 0.0),
				new Vec3(1.0, 2.0, 0.0),
				new Vec3(0.0, 3.0, 0.0),
			};
			double distance = ColliderFunctions.DistanceTriangleTriangleSquared(p, q);
			await Assert.That(Math.Abs(distance - 1.0) < 1e-6)
				.IsTrue()
				.Because($"TriangleDistanceEdge: {distance} expected 1.0");
		}

		/// <summary>C++ TEST(Properties, TriangleDistanceClosestPointOnEdge2).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppTriangleDistanceEdge2()
		{
			Vec3[] p =
			{
				new Vec3(-1.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
			};
			Vec3[] q =
			{
				new Vec3(1.0, 1.0, 0.0),
				new Vec3(3.0, 1.0, 0.0),
				new Vec3(2.0, 2.0, 0.0),
			};
			double distance = ColliderFunctions.DistanceTriangleTriangleSquared(p, q);
			await Assert.That(Math.Abs(distance - 0.5) < 1e-6)
				.IsTrue()
				.Because($"TriangleDistanceEdge2: {distance} expected 0.5");
		}

		/// <summary>C++ TEST(Properties, TriangleDistanceClosestPointOnFace).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppTriangleDistanceFace()
		{
			Vec3[] p =
			{
				new Vec3(-1.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
			};
			Vec3[] q =
			{
				new Vec3(-1.0, 2.0, -0.5),
				new Vec3(1.0, 2.0, -0.5),
				new Vec3(0.0, 2.0, 1.5),
			};
			double distance = ColliderFunctions.DistanceTriangleTriangleSquared(p, q);
			await Assert.That(Math.Abs(distance - 1.0) < 1e-6)
				.IsTrue()
				.Because($"TriangleDistanceFace: {distance} expected 1.0");
		}

		/// <summary>C++ TEST(Properties, TriangleDistanceOverlapping).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppTriangleDistanceOverlapping()
		{
			Vec3[] p =
			{
				new Vec3(-1.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
			};
			Vec3[] q =
			{
				new Vec3(-1.0, 0.0, 0.0),
				new Vec3(1.0, 0.5, 0.0),
				new Vec3(0.0, 1.0, 0.0),
			};
			double distance = ColliderFunctions.DistanceTriangleTriangleSquared(p, q);
			await Assert.That(Math.Abs(distance - 0.0) < 1e-6)
				.IsTrue()
				.Because($"TriangleDistanceOverlapping: {distance} expected 0.0");
		}
	}
}
