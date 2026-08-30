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

// RobustRayShootTests.cs — port of the `#[cfg(test)] mod tests` of
// robust/ray_shoot.rs. Four tests, same inputs, same expected values, same order.

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Tests
{
	public class RobustRayShootTests
	{
		/// <summary>Unit-ish cube [0,2]³ as 12 outward-wound triangles.</summary>
		/// <param name="lo">The low corner coordinate.</param>
		/// <param name="hi">The high corner coordinate.</param>
		/// <returns>The cube's twelve triangles.</returns>
		public static List<Vec3[]> CubeTris(double lo, double hi)
		{
			(double[] A, double[] B, double[] C, double[] D)[] quads =
			{
				// -z (normal 0,0,-1)
				(new[] { 0.0, 0.0, 0.0 }, new[] { 0.0, 1.0, 0.0 }, new[] { 1.0, 1.0, 0.0 }, new[] { 1.0, 0.0, 0.0 }),

				// +z
				(new[] { 0.0, 0.0, 1.0 }, new[] { 1.0, 0.0, 1.0 }, new[] { 1.0, 1.0, 1.0 }, new[] { 0.0, 1.0, 1.0 }),

				// -y
				(new[] { 0.0, 0.0, 0.0 }, new[] { 1.0, 0.0, 0.0 }, new[] { 1.0, 0.0, 1.0 }, new[] { 0.0, 0.0, 1.0 }),

				// +y
				(new[] { 0.0, 1.0, 0.0 }, new[] { 0.0, 1.0, 1.0 }, new[] { 1.0, 1.0, 1.0 }, new[] { 1.0, 1.0, 0.0 }),

				// -x
				(new[] { 0.0, 0.0, 0.0 }, new[] { 0.0, 0.0, 1.0 }, new[] { 0.0, 1.0, 1.0 }, new[] { 0.0, 1.0, 0.0 }),

				// +x
				(new[] { 1.0, 0.0, 0.0 }, new[] { 1.0, 1.0, 0.0 }, new[] { 1.0, 1.0, 1.0 }, new[] { 1.0, 0.0, 1.0 }),
			};
			double s = hi - lo;
			Vec3 M(double[] q)
			{
				return new Vec3(lo + (q[0] * s), lo + (q[1] * s), lo + (q[2] * s));
			}

			List<Vec3[]> outTris = new List<Vec3[]>();
			foreach ((double[] a, double[] b, double[] c, double[] d) in quads)
			{
				outTris.Add(new[] { M(a), M(b), M(c) });
				outTris.Add(new[] { M(a), M(c), M(d) });
			}

			return outTris;
		}

		[Test]
		public async Task WindingOfCube()
		{
			List<Vec3[]> cube = CubeTris(0.0, 2.0);
			R3 inside = R3.FromVec3(V(1.0, 1.0, 1.0));
			R3 outside = R3.FromVec3(V(5.0, 0.5, 0.5));
			R3 nearOut = R3.FromVec3(V(-0.25, 1.0, 1.0));
			await Assert.That(RayShoot.WindingNumber(inside, cube)).IsEqualTo(1);
			await Assert.That(RayShoot.WindingNumber(outside, cube)).IsEqualTo(0);
			await Assert.That(RayShoot.WindingNumber(nearOut, cube)).IsEqualTo(0);
			await Assert.That(RayShoot.PointInside(inside, cube, false)).IsTrue();
			await Assert.That(RayShoot.PointInside(outside, cube, false)).IsFalse();

			// Complement semantics.
			List<Vec3[]> flipped = Flipped(cube);
			await Assert.That(RayShoot.PointInside(inside, flipped, true)).IsFalse();
			await Assert.That(RayShoot.PointInside(outside, flipped, true)).IsTrue();
			await Assert.That(RayShoot.WindingNumber(inside, flipped)).IsEqualTo(-1);
		}

		[Test]
		public async Task WindingSurvivesDegenerateAxisRays()
		{
			// Query point aligned with cube vertices/edges on every axis: the
			// first directions all graze; retry logic must still classify.
			List<Vec3[]> cube = CubeTris(0.0, 2.0);
			R3 trickyIn = R3.FromVec3(V(1.0, 1.0, 0.5)); // axis rays hit edges
			await Assert.That(RayShoot.WindingNumber(trickyIn, cube)).IsEqualTo(1);
			R3 trickyOut = R3.FromVec3(V(0.0, 0.0, 5.0)); // rays through corner
			await Assert.That(RayShoot.WindingNumber(trickyOut, cube)).IsEqualTo(0);
		}

		[Test]
		public async Task NestedVoidWinding()
		{
			// Outer cube with an inward-oriented inner cube = solid with a void.
			List<Vec3[]> solid = CubeTris(0.0, 6.0);
			List<Vec3[]> inner = Flipped(CubeTris(2.0, 4.0));
			solid.AddRange(inner);
			R3 inWall = R3.FromVec3(V(1.0, 1.0, 1.0));
			R3 inVoid = R3.FromVec3(V(3.0, 3.0, 3.0));
			R3 outside = R3.FromVec3(V(7.0, 3.0, 3.0));
			await Assert.That(RayShoot.WindingNumber(inWall, solid)).IsEqualTo(1);
			await Assert.That(RayShoot.WindingNumber(inVoid, solid)).IsEqualTo(0);
			await Assert.That(RayShoot.WindingNumber(outside, solid)).IsEqualTo(0);
		}

		[Test]
		public async Task CentroidIsExact()
		{
			R3[] p =
			{
				R3.FromVec3(V(0.0, 0.0, 0.0)),
				R3.FromVec3(V(1.0, 0.0, 0.0)),
				R3.FromVec3(V(0.0, 1.0, 0.0)),
			};
			R3 c = RayShoot.PieceCentroid(new[] { p[0], p[1], p[2] });
			await Assert.That(c.X).IsEqualTo(Backend.RatNew(1, 3));
			await Assert.That(Backend.RatIsZero(c.Z)).IsTrue();
		}

		private static Vec3 V(double x, double y, double z)
		{
			return new Vec3(x, y, z);
		}

		/// <summary>The Rust tests' <c>t.iter().map(|t| [t[0], t[2], t[1]])</c>.</summary>
		private static List<Vec3[]> Flipped(IReadOnlyList<Vec3[]> tris)
		{
			List<Vec3[]> outTris = new List<Vec3[]>(tris.Count);
			foreach (Vec3[] t in tris)
			{
				outTris.Add(new[] { t[0], t[2], t[1] });
			}

			return outTris;
		}
	}
}
