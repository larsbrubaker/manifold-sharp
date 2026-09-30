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

// QuickHullContainmentTests.cs — NOT A PORT; a C#-only adaptation test. It pins a
// defect found by the convex-dilation union tree on Thingi10K 63451 (normalized by
// StlFixtures.ImportStlLikeDemo) dilated by Sphere(0.3, 8), which came out 3.4e-6
// short of the true Minkowski sum.
//
// The tree's first loss (level 0, node 5: leaves 10 and 11) reduces to the exact
// union of the hulls of triangles 157 and 163, which misses 5.8e-5 of the second
// operand. The union is not at fault: the hull of triangle 163 is not convex — one
// of its hull vertices lies 0.58 outside one of its faces — so the boolean was given
// a folded operand. The triangle is flat (all three corners at z = 0.0234375), so its
// 54 vertex sums hold many exactly coplanar sets. manifold-rust's convex_hull builds
// the same 54-triangle mesh from the same points (same worst distance and volume).
//
// Where it went wrong (iteration 29, apex point 43): the apex lies on the line through
// hull points 7 and 16, so in the plane of both faces on that edge. The float plane
// distance put one face at 5.6e-17 (visible) and the other at 0 (hidden), so the edge
// became a horizon edge and the new face had zero area and a noise normal; every
// later test against it was noise. Divergence ledger entry 7 decides visibility with
// the exact orientation instead (QuickHull.Exact.cs).
//
// The same defect made Minkowski.Sum of Thingi10K 641145 (Sphere(0.0578..., 12))
// come out 4.4e-4 short and 287448 lose its genus: triangle 109's hull there had
// input points 0.234 outside it. Unioned in, that folded hull left a zero-thickness
// fin whose coplanar flood fill handed a sound triangle a reversed normal, and
// SwapDegenerates then swapped it and cut solid away (MinkowskiUnionRegressionTests
// keeps that operand pair, skipped).

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class QuickHullContainmentTests
	{
		// Triangle 163 of Thingi10K 63451 after the demo import, bit-exact (the
		// coordinates are f32 values, so the literals round-trip).
		private static readonly Vec3[] Triangle163 =
		{
			new Vec3(-0.373046875, 0.33203125, 0.0234375),
			new Vec3(-0.5078125, 0.466796875, 0.0234375),
			new Vec3(-0.5078125, 0.197265625, 0.0234375),
		};

		// Triangle 109 of Thingi10K 641145 after the demo import, bit-exact.
		private static readonly Vec3[] Triangle109 =
		{
			new Vec3(-0.6162518858909607, 0.6155887842178345, 0.3000994920730591),
			new Vec3(-0.7850430011749268, 0.6208153367042542, 0.3000994920730591),
			new Vec3(-0.6162518858909607, -0.6155887246131897, 0.3000994920730591),
		};

		/// <summary>
		/// A convex hull's vertices all lie on or behind every one of its faces.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task HullOfAFlatTriangleSweptBySphereIsConvex()
		{
			await AssertSweptHullIsConvex(Triangle163, Manifold.Sphere(0.3, 8));
		}

		/// <summary>
		/// The same, for the hull whose fold made Minkowski.Sum of Thingi10K 641145 lose
		/// solid (the sweep's radius, 0.02 of the part's diagonal, at 12 segments). Before
		/// the exact visibility test, input points lay 0.234 outside it.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task Thingi641145Triangle109SweptHullIsConvex()
		{
			await AssertSweptHullIsConvex(Triangle109, Manifold.Sphere(0.05781898171099809, 12));
		}

		private static async Task AssertSweptHullIsConvex(Vec3[] triangle, Manifold toolSolid)
		{
			// The points ConvexDilation.LeafUnion (and Minkowski.cs) hull for one triangle,
			// in the same order.
			ManifoldImpl tool = toolSolid.AsImpl();
			List<Vec3> points = new List<Vec3>(3 * tool.VertPos.Count);
			foreach (Vec3 corner in triangle)
			{
				foreach (Vec3 toolVert in tool.VertPos)
				{
					points.Add(corner + toolVert);
				}
			}

			ManifoldImpl hull = QuickHullFunctions.ConvexHull(points);

			double worstOutside = 0.0;
			for (int tri = 0; tri < hull.NumTri(); tri++)
			{
				Vec3 p0 = hull.VertPos[hull.Halfedge[3 * tri].StartVert];
				Vec3 p1 = hull.VertPos[hull.Halfedge[(3 * tri) + 1].StartVert];
				Vec3 p2 = hull.VertPos[hull.Halfedge[(3 * tri) + 2].StartVert];
				Vec3 normal = LinalgFunctions.Normalize(LinalgFunctions.Cross(p1 - p0, p2 - p0));
				foreach (Vec3 point in points)
				{
					worstOutside = Math.Max(worstOutside, LinalgFunctions.Dot(normal, point - p0));
				}
			}

			await Assert.That(hull.Status).IsEqualTo(Error.NoError);
			await Assert.That(worstOutside).IsLessThan(1e-12);
		}
	}
}
