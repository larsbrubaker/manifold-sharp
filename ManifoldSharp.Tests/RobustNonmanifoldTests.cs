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

// RobustNonmanifoldTests.cs — port of robust/nonmanifold_tests.rs, whose header reads:
//
//   Phase 8: end-to-end booleans on genuinely non-manifold (but closed,
//   orientable) inputs — the configurations the exact engine cannot even
//   import. Volumes are verified against hand geometry; `Manifold::volume()`
//   is divergence-theorem-based and valid on closed orientable soups
//   independent of connectivity.
//
// Same inputs, same expected values, same order as the Rust's nine tests.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class RobustNonmanifoldTests
	{
		[Test]
		public async Task EdgeSharedCubesUnionWithThirdSolid()
		{
			Manifold soup = EdgeKissingCubes();
			await Assert.That(soup.AsImpl().IsSoup).IsTrue();

			// A third box bridging both cubes across the shared edge.
			Manifold bridge = Manifold.Cube(V(2.0, 2.0, 1.0), false).Translate(V(1.0, 1.0, 0.5));
			Manifold u = soup.BooleanWithEngine(bridge, OpType.Add, BooleanEngine.Auto);

			// Bridge is 2x2x1 = 4; overlap with each cube is 1x1x1 = 1 each.
			await AssertVol(u, 8.0 + 8.0 + 4.0 - 1.0 - 1.0, "edge cubes + bridge union");
		}

		[Test]
		public async Task EdgeSharedCubesDifference()
		{
			Manifold soup = EdgeKissingCubes();
			Manifold cutter = Manifold.Cube(V(1.0, 1.0, 2.0), false).Translate(V(1.0, 1.0, 0.0));
			Manifold d = soup.BooleanWithEngine(cutter, OpType.Subtract, BooleanEngine.Auto);

			// Cutter (volume 2) sits fully in the first cube, corner at the shared edge.
			await AssertVol(d, 16.0 - 2.0, "edge cubes - corner cutter");

			// And intersecting returns just the cutter volume.
			Manifold i = soup.BooleanWithEngine(cutter, OpType.Intersect, BooleanEngine.Auto);
			await AssertVol(i, 2.0, "edge cubes ∩ corner cutter");
		}

		[Test]
		public async Task SixFaceEdgeFan()
		{
			// Three quadrant boxes around the z-axis edge: (+x,+y), (-x,+y), (-x,-y)
			// — the shared edge carries 6 faces, all balanced.
			List<Vec3[]> tris = CubeTris(new[] { 0.0, 0.0, 0.0 }, new[] { 2.0, 2.0, 2.0 });
			tris.AddRange(CubeTris(new[] { -2.0, 0.0, 0.0 }, new[] { 0.0, 2.0, 2.0 }));
			tris.AddRange(CubeTris(new[] { -2.0, -2.0, 0.0 }, new[] { 0.0, 0.0, 2.0 }));
			Manifold soup = SoupManifold(tris);
			await AssertVol(soup, 24.0, "six-face fan import");

			// Cut a hole straight through the fan edge region.
			Manifold cutter = Manifold.Cylinder(4.0, 0.5, 0.5, 8).Translate(V(0.0, 0.0, -1.0));
			Manifold d = soup.BooleanWithEngine(cutter, OpType.Subtract, BooleanEngine.Auto);
			await Assert.That(d.Status()).IsEqualTo(Error.NoError);

			// Cylinder(8 segs, r=0.5) cross-section area A sits 3/4 inside the solid
			// over height 2 → volume 24 - 2*(3/4)*A.
			// Inscribed octagon of the tessellated cylinder: 8 * (1/2) r² sin(45°)
			double octagonArea = 8.0 * 0.5 * 0.25 * DeterministicMath.Sin(Types.KPi / 4.0);
			double expect = 24.0 - (2.0 * 0.75 * octagonArea);
			double vol = d.Volume();
			await Assert.That(Math.Abs(vol - expect)).IsLessThan(1e-6)
				.Because($"fan - cylinder volume {vol}, expected ~{expect}");
		}

		[Test]
		public async Task VertexPinchedCubes()
		{
			// Two cubes sharing exactly one vertex (pinched surface point).
			List<Vec3[]> tris = CubeTris(new[] { 0.0, 0.0, 0.0 }, new[] { 2.0, 2.0, 2.0 });
			tris.AddRange(CubeTris(new[] { 2.0, 2.0, 2.0 }, new[] { 4.0, 4.0, 4.0 }));
			Manifold soup = SoupManifold(tris);
			await AssertVol(soup, 16.0, "vertex-pinched import");
			Manifold cutter = Manifold.Cube(V(1.0, 1.0, 1.0), false).Translate(V(0.5, 0.5, 0.5));
			Manifold d = soup.BooleanWithEngine(cutter, OpType.Subtract, BooleanEngine.Auto);
			await AssertVol(d, 15.0, "vertex-pinched - cutter");
		}

		[Test]
		public async Task InternalVoidCutOpen()
		{
			// Solid [0,6]³ with void [2,4]³; a slab cutter [2.5,3.5]×[2.5,3.5]×[-1,7]
			// drills a square channel through solid and void.
			List<Vec3[]> tris = CubeTris(new[] { 0.0, 0.0, 0.0 }, new[] { 6.0, 6.0, 6.0 });
			foreach (Vec3[] t in CubeTris(new[] { 2.0, 2.0, 2.0 }, new[] { 4.0, 4.0, 4.0 }))
			{
				tris.Add(new[] { t[0], t[2], t[1] });
			}

			Manifold soup = SoupManifold(tris);
			await AssertVol(soup, 216.0 - 8.0, "void import");
			Manifold cutter = Manifold.Cube(V(1.0, 1.0, 8.0), false).Translate(V(2.5, 2.5, -1.0));
			Manifold d = soup.BooleanWithEngine(cutter, OpType.Subtract, BooleanEngine.Auto);

			// Channel removes 1×1×6 of material minus the 1×1×2 already void.
			await AssertVol(d, 208.0 - (6.0 - 2.0), "void - channel");

			// Intersection keeps only the channel material that existed (4 units).
			Manifold i = soup.BooleanWithEngine(cutter, OpType.Intersect, BooleanEngine.Auto);
			await AssertVol(i, 4.0, "void ∩ channel");
		}

		[Test]
		public async Task MultiComponentSoupBoolean()
		{
			// Three disjoint cubes as one soup, cut by a slab crossing all three.
			List<Vec3[]> tris = new List<Vec3[]>();
			for (int k = 0; k < 3; k++)
			{
				double o = 3.0 * k;
				tris.AddRange(CubeTris(new[] { o, 0.0, 0.0 }, new[] { o + 2.0, 2.0, 2.0 }));
			}

			Manifold soup = SoupManifold(tris);
			await AssertVol(soup, 24.0, "multi-component import");
			Manifold slab = Manifold.Cube(V(9.0, 2.0, 1.0), false).Translate(V(-0.5, 0.0, 0.5));
			Manifold i = soup.BooleanWithEngine(slab, OpType.Intersect, BooleanEngine.Auto);
			await AssertVol(i, 12.0, "multi-component ∩ slab");
			Manifold d = soup.BooleanWithEngine(slab, OpType.Subtract, BooleanEngine.Auto);
			await AssertVol(d, 12.0, "multi-component - slab");
		}

		[Test]
		public async Task DoubledCoverOperand()
		{
			// Every facet listed twice with the same winding — a doubled cover, as
			// some Thingi10K scans ship (#92068 triples every facet). The regularized
			// boolean must emit each surface element once: exactly-coincident
			// pieces share one wall in robust/cells.rs, whose winding step is the
			// sum of the stack, and extraction emits a single representative —
			// otherwise the output surface is multiply covered and stops being
			// closed where it meets the other operand.
			List<Vec3[]> tris = CubeTris(new[] { 0.0, 0.0, 0.0 }, new[] { 2.0, 2.0, 2.0 });
			tris.AddRange(CubeTris(new[] { 0.0, 0.0, 0.0 }, new[] { 2.0, 2.0, 2.0 }));
			Manifold soup = SoupManifold(tris);
			await Assert.That(soup.AsImpl().IsSoup).IsTrue();

			// Divergence-theorem volume counts both covers before regularization.
			await AssertVol(soup, 16.0, "doubled cube import");

			Manifold other = Manifold.Cube(V(2.0, 2.0, 2.0), false).Translate(V(1.0, 1.0, 1.0));
			Manifold u = soup.BooleanWithEngine(other, OpType.Add, BooleanEngine.Auto);
			await AssertVol(u, 8.0 + 8.0 - 1.0, "doubled cube ∪ cube");
			Manifold i = soup.BooleanWithEngine(other, OpType.Intersect, BooleanEngine.Auto);
			await AssertVol(i, 1.0, "doubled cube ∩ cube");
			Manifold d = soup.BooleanWithEngine(other, OpType.Subtract, BooleanEngine.Auto);
			await AssertVol(d, 7.0, "doubled cube − cube");
		}

		/// <summary>
		/// Rotating a soup import must not panic. <c>Soupify</c> skips <c>SortGeometry</c>,
		/// which is where the face collider is built, so a soup impl carries a zero-leaf tree;
		/// a non-axis-aligned transform used to clone that empty tree and refit it against
		/// real face boxes, indexing out of bounds. The debug assertion guarding it is
		/// compiled out in release, so this only ever surfaced as a hard crash.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task SoupSurvivesNonAxisAlignedRotation()
		{
			Manifold soup = EdgeKissingCubes();
			await Assert.That(soup.AsImpl().IsSoup).IsTrue().Because("fixture must import as a soup");
			Manifold rotated = soup.Rotate(30.0, 45.0, 60.0);
			await Assert.That(rotated.Status()).IsEqualTo(Error.NoError).Because("rotated soup status");
			await AssertVol(rotated, 16.0, "rotated soup");
		}

		[Test]
		public async Task SoupOpSoup()
		{
			// Both operands non-manifold: edge-kissing pairs crossing each other.
			Manifold a = EdgeKissingCubes();
			List<Vec3[]> trisB = CubeTris(new[] { 1.0, -1.0, 0.5 }, new[] { 3.0, 1.0, 1.5 });
			trisB.AddRange(CubeTris(new[] { 3.0, 1.0, 0.5 }, new[] { 5.0, 3.0, 1.5 }));
			Manifold b = SoupManifold(trisB);
			await Assert.That(a.AsImpl().IsSoup && b.AsImpl().IsSoup).IsTrue();

			// b's volume: 2*2*1 + 2*2*1 = 8. Overlaps with a:
			//   b1 = [1,3]×[-1,1]×[0.5,1.5] ∩ cube1[0,2]³ = [1,2]×[0,1]×[0.5,1.5] = 1
			//   b2 = [3,5]×[1,3]×[0.5,1.5] ∩ cube2[2,4]×[2,4]×[0,2] = [3,4]×[2,3]×[0.5,1.5] = 1
			Manifold u = a.BooleanWithEngine(b, OpType.Add, BooleanEngine.Auto);
			await AssertVol(u, 16.0 + 8.0 - 2.0, "soup ∪ soup");
			Manifold i = a.BooleanWithEngine(b, OpType.Intersect, BooleanEngine.Auto);
			await AssertVol(i, 2.0, "soup ∩ soup");
			Manifold d = a.BooleanWithEngine(b, OpType.Subtract, BooleanEngine.Auto);
			await AssertVol(d, 14.0, "soup − soup");
		}

		/// <summary>Shorthand for the Rust's <c>v(x, y, z)</c>.</summary>
		/// <param name="x">The x coordinate.</param>
		/// <param name="y">The y coordinate.</param>
		/// <param name="z">The z coordinate.</param>
		/// <returns>The vector.</returns>
		private static Vec3 V(double x, double y, double z)
		{
			return new Vec3(x, y, z);
		}

		/// <summary>An axis-aligned box with independent per-axis extents, outward-wound.</summary>
		/// <param name="lo">The low corner.</param>
		/// <param name="hi">The high corner.</param>
		/// <returns>The box's twelve triangles.</returns>
		private static List<Vec3[]> CubeTris(double[] lo, double[] hi)
		{
			double x0 = lo[0];
			double y0 = lo[1];
			double z0 = lo[2];
			double x1 = hi[0];
			double y1 = hi[1];
			double z1 = hi[2];
			Vec3[][] quads =
			{
				new[] { V(x0, y0, z0), V(x0, y1, z0), V(x1, y1, z0), V(x1, y0, z0) },
				new[] { V(x0, y0, z1), V(x1, y0, z1), V(x1, y1, z1), V(x0, y1, z1) },
				new[] { V(x0, y0, z0), V(x1, y0, z0), V(x1, y0, z1), V(x0, y0, z1) },
				new[] { V(x0, y1, z0), V(x0, y1, z1), V(x1, y1, z1), V(x1, y1, z0) },
				new[] { V(x0, y0, z0), V(x0, y0, z1), V(x0, y1, z1), V(x0, y1, z0) },
				new[] { V(x1, y0, z0), V(x1, y1, z0), V(x1, y1, z1), V(x1, y0, z1) },
			};
			List<Vec3[]> outTris = new List<Vec3[]>();
			foreach (Vec3[] q in quads)
			{
				outTris.Add(new[] { q[0], q[1], q[2] });
				outTris.Add(new[] { q[0], q[2], q[3] });
			}

			return outTris;
		}

		/// <summary>Imports a raw f64 soup through the robust entry point.</summary>
		/// <param name="tris">The triangles.</param>
		/// <returns>The imported manifold.</returns>
		private static Manifold SoupManifold(IReadOnlyList<Vec3[]> tris)
		{
			MeshGL64 mesh = new MeshGL64();
			mesh.NumProp = 3;
			foreach (Vec3[] t in tris)
			{
				foreach (Vec3 p in t)
				{
					mesh.VertProperties.Add(p.X);
					mesh.VertProperties.Add(p.Y);
					mesh.VertProperties.Add(p.Z);
				}
			}

			for (ulong i = 0; i < 3 * (ulong)tris.Count; i++)
			{
				mesh.TriVerts.Add(i);
			}

			Manifold m = Manifold.FromMeshGL64Robust(mesh);
			if (m.Status() != Error.NoError)
			{
				throw new InvalidOperationException("soup import failed");
			}

			return m;
		}

		/// <summary>The Rust's <c>assert_vol</c>: status plus a relative volume check.</summary>
		/// <param name="m">The manifold.</param>
		/// <param name="expect">The expected volume.</param>
		/// <param name="what">What is being measured.</param>
		/// <returns>The assertion task.</returns>
		private static async Task AssertVol(Manifold m, double expect, string what)
		{
			await Assert.That(m.Status()).IsEqualTo(Error.NoError).Because($"{what}: status");
			double vol = m.Volume();
			await Assert.That(Math.Abs(vol - expect) <= 1e-9 * Math.Max(Math.Abs(expect), 1.0)).IsTrue()
				.Because($"{what}: volume {vol}, expected {expect}");
		}

		/// <summary>Two cubes sharing exactly one edge — the canonical non-manifold solid.</summary>
		/// <returns>The soup-backed manifold.</returns>
		private static Manifold EdgeKissingCubes()
		{
			List<Vec3[]> tris = CubeTris(new[] { 0.0, 0.0, 0.0 }, new[] { 2.0, 2.0, 2.0 });
			tris.AddRange(CubeTris(new[] { 2.0, 2.0, 0.0 }, new[] { 4.0, 4.0, 2.0 }));
			return SoupManifold(tris);
		}
	}
}
