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

// Port of src/manifold_tests/raycast.rs — all 12 tests, nothing deferred, whose
// own header reads:
//
//   Tests ported from C++ TEST(Manifold, RayCast*) in manifold_test.cpp
//   These test the Manifold::ray_cast(origin, endpoint) → Vec<RayHit> API.
//
// The two degenerate-geometry cases at the bottom are the point of the file:
// a ray through a vertex and a ray along a silhouette edge must not return an
// ODD number of hits, because that would mean the surface leaked. Symbolic
// perturbation is what makes them 2 and {0 or 2}.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ManifoldRaycastTests
	{
		/// <summary>C++ TEST(Manifold, RayCastHitCube) — ray through center along Z.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppRayCastHitCube()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);
			List<RayHit> hits = cube.RayCast(new Vec3(0.0, 0.0, -5.0), new Vec3(0.0, 0.0, 5.0));
			await Assert.That(hits.Count).IsEqualTo(2).Because($"expected 2 hits, got {hits.Count}");

			// hits are sorted by distance; first hit is closer to origin
			await Assert.That(hits[0].Distance).IsLessThan(hits[1].Distance);
			await Assert.That(Math.Abs(hits[0].Position.Z - -0.5))
				.IsLessThan(1e-5)
				.Because($"hit[0].z = {hits[0].Position.Z}, expected -0.5");
			await Assert.That(Math.Abs(hits[1].Position.Z - 0.5))
				.IsLessThan(1e-5)
				.Because($"hit[1].z = {hits[1].Position.Z}, expected 0.5");
			await Assert.That(Math.Abs(hits[0].Normal.Z - -1.0))
				.IsLessThan(1e-5)
				.Because($"hit[0].normal.z = {hits[0].Normal.Z}, expected -1");
			await Assert.That(Math.Abs(hits[1].Normal.Z - 1.0))
				.IsLessThan(1e-5)
				.Because($"hit[1].normal.z = {hits[1].Normal.Z}, expected 1");
		}

		/// <summary>C++ TEST(Manifold, RayCastMiss) — ray that misses the cube.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppRayCastMiss()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);
			List<RayHit> hits = cube.RayCast(new Vec3(10.0, 10.0, -5.0), new Vec3(10.0, 10.0, 5.0));
			await Assert.That(hits.Count).IsEqualTo(0).Because($"expected 0 hits, got {hits.Count}");
		}

		/// <summary>C++ TEST(Manifold, RayCastDiagonal) — diagonal ray (not axis-aligned).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppRayCastDiagonal()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);
			List<RayHit> hits = cube.RayCast(new Vec3(-5.0, -5.0, -5.0), new Vec3(5.0, 5.0, 5.0));
			await Assert.That(hits.Count).IsEqualTo(2).Because($"expected 2 hits, got {hits.Count}");
			await Assert.That(Math.Abs(hits[0].Position.Z - -0.5))
				.IsLessThan(1e-4)
				.Because($"hit[0].z = {hits[0].Position.Z}, expected -0.5");
		}

		/// <summary>C++ TEST(Manifold, RayCastBehindOrigin) — ray pointing away from cube.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppRayCastBehindOrigin()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);

			// endpoint is further in the +Z direction than origin, but both are outside cube
			List<RayHit> hits = cube.RayCast(new Vec3(0.0, 0.0, 5.0), new Vec3(0.0, 0.0, 10.0));
			await Assert.That(hits.Count).IsEqualTo(0).Because($"expected 0 hits, got {hits.Count}");
		}

		/// <summary>C++ TEST(Manifold, RayCastSphere) — ray through sphere.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppRayCastSphere()
		{
			Manifold sphere = Manifold.Sphere(1.0, 128);
			List<RayHit> hits = sphere.RayCast(new Vec3(0.0, 0.0, -5.0), new Vec3(0.0, 0.0, 5.0));
			await Assert.That(hits.Count)
				.IsEqualTo(2)
				.Because($"expected 2 hits through sphere, got {hits.Count}");

			// Hit point should be approximately on the unit sphere
			Vec3 p = hits[0].Position;
			double r = Math.Sqrt((p.X * p.X) + (p.Y * p.Y) + (p.Z * p.Z));
			await Assert.That(Math.Abs(r - 1.0)).IsLessThan(1e-3).Because($"hit[0] radius = {r}, expected ≈1.0");

			// Ray that misses
			List<RayHit> miss = sphere.RayCast(new Vec3(2.0, 2.0, -5.0), new Vec3(2.0, 2.0, 5.0));
			await Assert.That(miss.Count).IsEqualTo(0).Because($"expected 0 hits (miss), got {miss.Count}");
		}

		/// <summary>C++ TEST(Manifold, RayCastTwoCubes) — ray through two separated cubes.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppRayCastTwoCubes()
		{
			Manifold c1 = Manifold.Cube(Vec3.Splat(1.0), true);
			Manifold c2 = Manifold.Cube(Vec3.Splat(1.0), true).Translate(new Vec3(0.0, 0.0, 5.0));
			Manifold both = c1 + c2;
			List<RayHit> hits = both.RayCast(new Vec3(0.0, 0.0, -5.0), new Vec3(0.0, 0.0, 10.0));
			await Assert.That(hits.Count)
				.IsEqualTo(4)
				.Because($"expected 4 hits through two cubes, got {hits.Count}");
			await Assert.That(Math.Abs(hits[0].Position.Z - -0.5))
				.IsLessThan(1e-4)
				.Because($"hits[0].z = {hits[0].Position.Z}");
			await Assert.That(Math.Abs(hits[1].Position.Z - 0.5))
				.IsLessThan(1e-4)
				.Because($"hits[1].z = {hits[1].Position.Z}");
			await Assert.That(Math.Abs(hits[2].Position.Z - 4.5))
				.IsLessThan(1e-4)
				.Because($"hits[2].z = {hits[2].Position.Z}");
			await Assert.That(Math.Abs(hits[3].Position.Z - 5.5))
				.IsLessThan(1e-4)
				.Because($"hits[3].z = {hits[3].Position.Z}");
		}

		/// <summary>C++ TEST(Manifold, RayCastEmpty) — ray against empty manifold.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppRayCastEmpty()
		{
			// The Rust reads `Manifold::default()`, whose Default impl is `Manifold::new`
			// — the same empty manifold `new Manifold()` gives here.
			Manifold empty = new Manifold();
			List<RayHit> hits = empty.RayCast(new Vec3(0.0, 0.0, -5.0), new Vec3(0.0, 0.0, 5.0));
			await Assert.That(hits.Count)
				.IsEqualTo(0)
				.Because($"expected 0 hits against empty, got {hits.Count}");
		}

		/// <summary>C++ TEST(Manifold, RayCastAlongX) — axis-aligned ray along X.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppRayCastAlongX()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);
			List<RayHit> hits = cube.RayCast(new Vec3(-5.0, 0.0, 0.0), new Vec3(5.0, 0.0, 0.0));
			await Assert.That(hits.Count).IsEqualTo(2).Because($"expected 2 hits along X, got {hits.Count}");
			await Assert.That(Math.Abs(hits[0].Position.X - -0.5))
				.IsLessThan(1e-5)
				.Because($"hits[0].x = {hits[0].Position.X}");
		}

		/// <summary>C++ TEST(Manifold, RayCastAlongY) — axis-aligned ray along Y.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppRayCastAlongY()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);
			List<RayHit> hits = cube.RayCast(new Vec3(0.0, -5.0, 0.0), new Vec3(0.0, 5.0, 0.0));
			await Assert.That(hits.Count).IsEqualTo(2).Because($"expected 2 hits along Y, got {hits.Count}");
			await Assert.That(Math.Abs(hits[0].Position.Y - -0.5))
				.IsLessThan(1e-5)
				.Because($"hits[0].y = {hits[0].Position.Y}");
		}

		/// <summary>C++ TEST(Manifold, RayCastZeroLength) — zero-length ray returns no hits.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppRayCastZeroLength()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);
			List<RayHit> hits = cube.RayCast(new Vec3(0.0, 0.0, 0.0), new Vec3(0.0, 0.0, 0.0));
			await Assert.That(hits.Count).IsEqualTo(0).Because("expected 0 hits for zero-length ray");
		}

		/// <summary>
		/// C++ TEST(Manifold, RayCastWatertightVertex) — ray exactly through a vertex.
		/// Symbolic perturbation should give exactly 2 hits.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppRayCastWatertightVertex()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);
			List<RayHit> hits = cube.RayCast(new Vec3(0.5, 0.5, -5.0), new Vec3(0.5, 0.5, 5.0));
			await Assert.That(hits.Count)
				.IsEqualTo(2)
				.Because($"expected 2 hits through vertex, got {hits.Count}");
			await Assert.That(Math.Abs(hits[0].Position.Z - -0.5))
				.IsLessThan(1e-5)
				.Because($"hits[0].z = {hits[0].Position.Z}");
		}

		/// <summary>
		/// C++ TEST(Manifold, RayCastSilhouetteEdge) — ray at silhouette edge.
		/// Must return 0 or 2 hits (never 1, preserves watertightness).
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppRayCastSilhouetteEdge()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);
			List<RayHit> hits = cube.RayCast(new Vec3(0.5, 0.0, -5.0), new Vec3(0.5, 0.0, 5.0));
			await Assert.That(hits.Count == 0 || hits.Count == 2)
				.IsTrue()
				.Because($"expected 0 or 2 hits at silhouette edge, got {hits.Count}");
		}
	}
}
