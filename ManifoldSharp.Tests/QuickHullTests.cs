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

// Port of the tests module in quickhull_tests.rs — all 9 cases, same expected
// values, same strictness, in the same order. Nothing is deferred.
//
// Nothing here touches the process-global Quality/BooleanConfig state, so no
// case carries [NotInParallel]: ConvexHull reads no global except the mesh-ID
// counter, which is atomic and whose value none of these assertions observe.
//
// The one divergence is in SpherePoints, and it cannot reach an assertion: the
// Rust test builds its point cloud with std `f64::sin`/`cos` (the platform
// libm), while this port's rule is that trig goes through DeterministicMath
// (the musl port that math.rs is). Those can differ in the last ulp, so the two
// clouds are not guaranteed bit-identical — but the test only asserts that the
// hull is non-empty and that every hull vertex is within 0.01 of the unit
// sphere, both of which are ulp-insensitive. Bit-exactness of the *hull* is
// verified separately, against the compiled Rust, on clouds shared as raw bits.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ManifoldSharp.QuickHullFunctions;

namespace ManifoldSharp.Tests
{
	public class QuickHullTests
	{
		[Test]
		public async Task ConvexHullTetrahedron()
		{
			List<Vec3> pts = new List<Vec3>
			{
				new Vec3(0.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
				new Vec3(0.0, 0.0, 1.0),
			};
			ManifoldImpl hull = ConvexHull(pts);
			await Assert.That(hull.NumVert()).IsEqualTo(4);
			await Assert.That(hull.NumTri()).IsEqualTo(4);
		}

		[Test]
		public async Task ConvexHullCubePoints()
		{
			List<Vec3> pts = new List<Vec3>
			{
				new Vec3(0.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
				new Vec3(0.0, 0.0, 1.0),
				new Vec3(1.0, 1.0, 0.0),
				new Vec3(1.0, 0.0, 1.0),
				new Vec3(0.0, 1.0, 1.0),
				new Vec3(1.0, 1.0, 1.0),
			};
			ManifoldImpl hull = ConvexHull(pts);
			await Assert.That(hull.NumVert()).IsEqualTo(8);
			await Assert.That(hull.NumTri()).IsEqualTo(12);
		}

		[Test]
		public async Task ConvexHullEmpty()
		{
			ManifoldImpl hull = ConvexHull(new List<Vec3>());
			await Assert.That(hull.IsEmpty()).IsTrue();
		}

		[Test]
		public async Task ConvexHullSinglePoint()
		{
			ManifoldImpl hull = ConvexHull(new List<Vec3> { new Vec3(1.0, 2.0, 3.0) });

			// Degenerate: should produce something (possibly degenerate mesh)
			// Just check it doesn't panic
			_ = hull.NumTri();
			await Task.CompletedTask;
		}

		[Test]
		public async Task ConvexHullTwoPoints()
		{
			ManifoldImpl hull = ConvexHull(new List<Vec3>
			{
				new Vec3(0.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
			});
			_ = hull.NumTri();
			await Task.CompletedTask;
		}

		[Test]
		public async Task ConvexHullCoplanarPoints()
		{
			List<Vec3> pts = new List<Vec3>
			{
				new Vec3(0.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(1.0, 1.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
				new Vec3(0.5, 0.5, 0.0),
			};
			ManifoldImpl hull = ConvexHull(pts);

			// Planar case -- should still produce a valid mesh
			await Assert.That(hull.NumTri() > 0).IsTrue();
		}

		[Test]
		public async Task ConvexHullSpherePoints()
		{
			// Generate points on a sphere
			List<Vec3> pts = new List<Vec3>();
			int n = 20;
			for (int i = 0; i < n; i++)
			{
				double phi = Math.PI * i / (n - 1.0);
				for (int j = 0; j < n; j++)
				{
					double theta = 2.0 * Math.PI * j / n;
					pts.Add(new Vec3(
						DeterministicMath.Sin(phi) * DeterministicMath.Cos(theta),
						DeterministicMath.Sin(phi) * DeterministicMath.Sin(theta),
						DeterministicMath.Cos(phi)));
				}
			}

			ManifoldImpl hull = ConvexHull(pts);
			await Assert.That(hull.NumTri() > 0).IsTrue();

			// All vertices should be at distance ~1 from origin
			foreach (Vec3 v in hull.VertPos)
			{
				double r = Math.Sqrt((v.X * v.X) + (v.Y * v.Y) + (v.Z * v.Z));
				await Assert.That(Math.Abs(r - 1.0) < 0.01)
					.IsTrue()
					.Because($"vertex not on unit sphere: r={r}");
			}
		}

		[Test]
		public async Task ConvexHullInteriorPointsExcluded()
		{
			// Cube corners + interior point
			List<Vec3> pts = new List<Vec3>
			{
				new Vec3(0.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
				new Vec3(0.0, 0.0, 1.0),
				new Vec3(1.0, 1.0, 0.0),
				new Vec3(1.0, 0.0, 1.0),
				new Vec3(0.0, 1.0, 1.0),
				new Vec3(1.0, 1.0, 1.0),
				new Vec3(0.5, 0.5, 0.5), // interior point
			};
			ManifoldImpl hull = ConvexHull(pts);
			await Assert.That(hull.NumVert()).IsEqualTo(8); // interior point should be excluded
			await Assert.That(hull.NumTri()).IsEqualTo(12);
		}

		[Test]
		public async Task ConvexHullIsConvex()
		{
			List<Vec3> pts = new List<Vec3>
			{
				new Vec3(0.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
				new Vec3(0.0, 0.0, 1.0),
				new Vec3(1.0, 1.0, 0.0),
				new Vec3(1.0, 0.0, 1.0),
				new Vec3(0.0, 1.0, 1.0),
				new Vec3(1.0, 1.0, 1.0),
			};
			ManifoldImpl hull = ConvexHull(pts);
			await Assert.That(hull.IsConvex()).IsTrue();
		}
	}
}
