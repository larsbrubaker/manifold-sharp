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

// Port of polygon_earclip_tests.rs (the tests module of polygon_earclip.rs) —
// every case, same expected values, same order and same strictness (exact
// equality on the hole ordering and the triangle hashes, the Rust's 1e-9 on the
// total area).
//
// These reach `EarClip`'s state directly, as the Rust child test module does:
// `ear_clip.holes` and `ear_clip.polygon[..]` are private in the Rust and
// `internal` here (Holes, PolygonVerts), which InternalsVisibleTo exposes to
// this assembly and nothing else.

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Tests
{
	public class PolygonEarclipTests
	{
		private static List<PolyVert> Contour((double X, double Y)[] coords, int firstIdx)
		{
			List<PolyVert> verts = new List<PolyVert>();
			for (int offset = 0; offset < coords.Length; offset++)
			{
				verts.Add(new PolyVert(new Vec2(coords[offset].X, coords[offset].Y), firstIdx + offset));
			}

			return verts;
		}

		private static PolygonsIdx MultiHolePolygons()
		{
			return new PolygonsIdx
			{
				Contour(new[] { (0.0, 0.0), (12.0, 0.0), (12.0, 10.0), (0.0, 10.0) }, 0),
				Contour(new[] { (1.0, 2.0), (1.0, 4.0), (3.0, 4.0), (3.0, 2.0) }, 4),
				Contour(new[] { (4.0, 2.0), (4.0, 4.0), (6.0, 4.0), (6.0, 2.0) }, 8),
				Contour(new[] { (8.0, 2.0), (8.0, 4.0), (10.0, 4.0), (10.0, 2.0) }, 12),
			};
		}

		[Test]
		public async Task OrdersHolesRightmostFirstForKeyholing()
		{
			PolygonsIdx polygons = MultiHolePolygons();

			EarClip earClip = new EarClip(polygons, 1.0e-10);
			List<double> holeXs = earClip.Holes
				.Select(hole => earClip.PolygonVerts[hole].Pos.X)
				.ToList();

			await Assert.That(holeXs.Count).IsEqualTo(3);
			await Assert.That(holeXs[0]).IsEqualTo(10.0);
			await Assert.That(holeXs[1]).IsEqualTo(6.0);
			await Assert.That(holeXs[2]).IsEqualTo(3.0);
		}

		/// <summary>
		/// End-to-end companion to <see cref="OrdersHolesRightmostFirstForKeyholing"/>:
		/// with holes keyholed in contour order the bridges cross, producing
		/// inverted triangles. Verifies the triangulation itself is valid.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task MultiHoleTriangulationHasNoInvertedTriangles()
		{
			PolygonsIdx polygons = MultiHolePolygons();
			List<Vec2> verts = polygons.SelectMany(poly => poly).Select(v => v.Pos).ToList();

			(List<IVec3> triangles, double _) = new EarClip(polygons, -1.0).Triangulate();

			// Outer 12x10 rectangle minus three 2x2 holes.
			double expectedArea = (12.0 * 10.0) - (3.0 * 4.0);
			double totalArea = 0.0;
			foreach (IVec3 tri in triangles)
			{
				Vec2 a = verts[tri.X];
				Vec2 b = verts[tri.Y];
				Vec2 c = verts[tri.Z];
				double area = 0.5 * Polygon.Determinant2x2(b - a, c - a);
				await Assert.That(area > 0.0)
					.IsTrue()
					.Because($"inverted or degenerate triangle {tri} (area {area})");
				totalArea += area;
			}

			await Assert.That(Math.Abs(totalArea - expectedArea) < 1e-9)
				.IsTrue()
				.Because($"triangulation area {totalArea} != expected {expectedArea}");
		}

		/// <summary>FNV-1a over a triangle list, pinning the triangles and their order.</summary>
		private static ulong Fnv(List<IVec3> tris)
		{
			ulong h = 0xcbf29ce484222325UL;
			foreach (IVec3 t in tris)
			{
				foreach (int c in new[] { t.X, t.Y, t.Z })
				{
					foreach (byte b in BitConverter.GetBytes(c))
					{
						h ^= b;
						h *= 0x100000001b3UL;
					}
				}
			}

			return h;
		}

		private static List<Vec2> Octagon(double cx, double cy, double r, bool hole)
		{
			(double X, double Y)[] corners =
			{
				(1.0, -0.5),
				(0.5, -1.0),
				(-0.5, -1.0),
				(-1.0, -0.5),
				(-1.0, 0.5),
				(-0.5, 1.0),
				(0.5, 1.0),
				(1.0, 0.5),
			};
			List<Vec2> ring = corners.Select(c => new Vec2(cx + (r * c.X), cy + (r * c.Y))).ToList();
			if (!hole)
			{
				ring.Reverse();
			}

			return ring;
		}

		/// <summary>
		/// Pins the triangles of a 24x24 grid of octagonal holes, every other row
		/// shifted so the bridges run between holes rather than to the outer edge.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task KeyholingManyHolesKeepsItsTriangles()
		{
			int n = 24;
			double size = (3.0 * n) + 3.0;
			Polygons polys = new Polygons
			{
				new SimplePolygon
				{
					new Vec2(0.0, 0.0),
					new Vec2(size, 0.0),
					new Vec2(size, size),
					new Vec2(0.0, size),
				},
			};
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
				{
					double cx = (3.0 * i) + 2.0 + (j % 2 == 1 ? 0.75 : 0.0);
					double cy = (3.0 * j) + 2.0;
					polys.Add(Octagon(cx, cy, 1.0, true));
				}
			}

			List<IVec3> tris = Polygon.Triangulate(polys, 1e-9, true);

			// Every vert is kept, so v + 2h - 2 triangles.
			int verts = 4 + (8 * n * n);
			await Assert.That(tris.Count).IsEqualTo(verts + (2 * n * n) - 2);
			await Assert.That(Fnv(tris)).IsEqualTo(0xb4449c9bcd61b83eUL)
				.Because($"hash 0x{Fnv(tris):x}");
		}
	}
}
