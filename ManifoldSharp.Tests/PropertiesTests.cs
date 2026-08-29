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

// Port of the properties.rs test module — all eighteen cases, same expected
// values, in the same order. Nothing is deferred.
//
// The one IsEquivalentTo below passes CollectionOrdering.Matching explicitly.
// TUnit's default is CollectionOrdering.Any — order-INSENSITIVE — which would
// silently turn the Rust `assert_eq!(m.properties, old_props)` on a Vec into a
// set comparison.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class PropertiesTests
	{
		[Test]
		public async Task TetrahedronVolume()
		{
			ManifoldImpl m = ManifoldImpl.Tetrahedron(Mat3x4.Identity());
			double vol = m.GetProperty(Property.Volume);

			// Regular tetrahedron with edge length 2*sqrt(2), vertices at distance
			// sqrt(3) from origin. Volume = 8/3.
			await Assert.That(Math.Abs(Math.Abs(vol) - (8.0 / 3.0)) < 1e-10)
				.IsTrue()
				.Because($"Expected volume ~2.6667, got {vol}");
		}

		[Test]
		public async Task CubeVolume()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			double vol = m.GetProperty(Property.Volume);
			await Assert.That(Math.Abs(Math.Abs(vol) - 1.0) < 1e-10)
				.IsTrue()
				.Because($"Expected unit cube volume = 1.0, got {vol}");
		}

		[Test]
		public async Task CubeSurfaceArea()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			double area = m.GetProperty(Property.SurfaceArea);
			await Assert.That(Math.Abs(area - 6.0) < 1e-10)
				.IsTrue()
				.Because($"Expected unit cube surface area = 6.0, got {area}");
		}

		[Test]
		public async Task OctahedronVolume()
		{
			ManifoldImpl m = ManifoldImpl.Octahedron(Mat3x4.Identity());
			double vol = m.GetProperty(Property.Volume);

			// Regular octahedron with vertices at ±1 on each axis: volume = 4/3
			await Assert.That(Math.Abs(Math.Abs(vol) - (4.0 / 3.0)) < 1e-10)
				.IsTrue()
				.Because($"Expected octahedron volume ~1.3333, got {vol}");
		}

		[Test]
		public async Task OctahedronSurfaceArea()
		{
			ManifoldImpl m = ManifoldImpl.Octahedron(Mat3x4.Identity());
			double area = m.GetProperty(Property.SurfaceArea);

			// 8 equilateral triangles with edge length sqrt(2), each area = sqrt(3)/2
			// Total = 8 * sqrt(3)/2 ≈ 6.9282
			double expected = 4.0 * Math.Sqrt(3.0);
			await Assert.That(Math.Abs(area - expected) < 1e-10)
				.IsTrue()
				.Because($"Expected octahedron surface area ~{expected}, got {area}");
		}

		[Test]
		public async Task EmptyMeshProperties()
		{
			ManifoldImpl m = new ManifoldImpl();
			await Assert.That(m.GetProperty(Property.Volume)).IsEqualTo(0.0);
			await Assert.That(m.GetProperty(Property.SurfaceArea)).IsEqualTo(0.0);
		}

		[Test]
		public async Task MatchesTriNormalsCube()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			await Assert.That(m.MatchesTriNormals())
				.IsTrue()
				.Because("Cube should have CCW triangles matching normals");
		}

		[Test]
		public async Task MatchesTriNormalsTetrahedron()
		{
			ManifoldImpl m = ManifoldImpl.Tetrahedron(Mat3x4.Identity());
			await Assert.That(m.MatchesTriNormals())
				.IsTrue()
				.Because("Tetrahedron should have CCW triangles matching normals");
		}

		[Test]
		public async Task NumDegenerateTrisCube()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			await Assert.That(m.NumDegenerateTris())
				.IsEqualTo(0)
				.Because("Cube should have no degenerate triangles");
		}

		[Test]
		public async Task NumDegenerateTrisTetrahedron()
		{
			ManifoldImpl m = ManifoldImpl.Tetrahedron(Mat3x4.Identity());
			await Assert.That(m.NumDegenerateTris())
				.IsEqualTo(0)
				.Because("Tetrahedron should have no degenerate triangles");
		}

		[Test]
		public async Task IsConvexCube()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			await Assert.That(m.IsConvex()).IsTrue().Because("Unit cube should be convex");
		}

		[Test]
		public async Task IsConvexTetrahedron()
		{
			ManifoldImpl m = ManifoldImpl.Tetrahedron(Mat3x4.Identity());
			await Assert.That(m.IsConvex()).IsTrue().Because("Tetrahedron should be convex");
		}

		[Test]
		public async Task IsConvexOctahedron()
		{
			ManifoldImpl m = ManifoldImpl.Octahedron(Mat3x4.Identity());
			await Assert.That(m.IsConvex()).IsTrue().Because("Octahedron should be convex");
		}

		[Test]
		public async Task IsIndexInBounds()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			List<IVec3> valid = new List<IVec3> { new IVec3(0, 1, 2), new IVec3(3, 4, 5) };
			await Assert.That(m.IsIndexInBounds(valid)).IsTrue();

			List<IVec3> invalid = new List<IVec3> { new IVec3(0, 1, 100) };
			await Assert.That(m.IsIndexInBounds(invalid)).IsFalse();

			List<IVec3> negative = new List<IVec3> { new IVec3(-1, 0, 1) };
			await Assert.That(m.IsIndexInBounds(negative)).IsFalse();

			await Assert.That(m.IsIndexInBounds(Array.Empty<IVec3>())).IsTrue();
		}

		[Test]
		public async Task CalculateCurvatureCube()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			m.CalculateCurvature(0, 1);
			await Assert.That(m.NumProp).IsEqualTo(2);
			await Assert.That(m.Properties.Count != 0)
				.IsTrue()
				.Because("Properties should be populated");
		}

		[Test]
		public async Task CalculateCurvatureSkipBoth()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			List<double> oldProps = new List<double>(m.Properties);
			int oldNumProp = m.NumProp;
			m.CalculateCurvature(-1, -1);
			await Assert.That(m.NumProp).IsEqualTo(oldNumProp);
			await Assert.That(m.Properties).IsEquivalentTo(oldProps, CollectionOrdering.Matching);
		}

		[Test]
		public async Task ScaledCubeVolume()
		{
			Vec3 scale = new Vec3(2.0, 3.0, 4.0);
			Mat3x4 t = LinalgFunctions.Mat4ToMat3x4(LinalgFunctions.ScalingMatrix(scale));
			ManifoldImpl m = ManifoldImpl.Cube(t);
			double vol = m.GetProperty(Property.Volume);
			await Assert.That(Math.Abs(Math.Abs(vol) - 24.0) < 1e-10)
				.IsTrue()
				.Because($"Expected 2×3×4 cube volume = 24.0, got {vol}");
		}

		[Test]
		public async Task ScaledCubeSurfaceArea()
		{
			Vec3 scale = new Vec3(2.0, 3.0, 4.0);
			Mat3x4 t = LinalgFunctions.Mat4ToMat3x4(LinalgFunctions.ScalingMatrix(scale));
			ManifoldImpl m = ManifoldImpl.Cube(t);
			double area = m.GetProperty(Property.SurfaceArea);

			// 2(2*3 + 2*4 + 3*4) = 2(6+8+12) = 52
			await Assert.That(Math.Abs(area - 52.0) < 1e-10)
				.IsTrue()
				.Because($"Expected 2×3×4 cube surface area = 52.0, got {area}");
		}
	}
}
