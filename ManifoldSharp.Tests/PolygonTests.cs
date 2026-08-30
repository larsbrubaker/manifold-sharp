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

// Port of polygon_tests.rs — all 7 cases, same expected values, same order and
// same strictness. Every assert_eq! compares exactly here too.
//
// The Rust builds the pentagon/octagon with `a.cos()` / `a.sin()` — std, i.e.
// the system libm, not manifold-rust's own math module. The C# uses
// DeterministicMath.Cos / Sin: same construction, but not a claim of identical
// bits, since these two libms may differ in the last ulp. That is immaterial
// here because both cases assert on the triangle *count* only.
//
// The rule this repo actually follows: PRODUCTION code never calls a
// System.Math transcendental — always DeterministicMath. TEST fixtures follow
// the Rust per site: `std` (`a.cos()`) becomes `Math.*`, `crate::math::cos`
// becomes `DeterministicMath.*`, so the C# fixture is the same *function* the
// Rust test defines. See the policy note in ManifoldTestHelpers.Gyroid for what
// that costs and why it is still the faithful choice. These two sites predate
// the rule and are the exception to it — harmless, since a count assertion
// cannot see an ulp, but do not copy the pattern into a new fixture.
//
// `is_convex`, `build_two_d_tree` and `query_two_d_tree` are private module
// items in the Rust that its in-crate test module can see; they are `internal`
// on Polygon here, which InternalsVisibleTo makes visible in exactly the same
// way.

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Tests
{
	public class PolygonTests
	{
		private static SimplePolygonIdx MakePoly((double X, double Y)[] coords)
		{
			SimplePolygonIdx poly = new SimplePolygonIdx();
			for (int i = 0; i < coords.Length; i++)
			{
				poly.Add(new PolyVert(new Vec2(coords[i].X, coords[i].Y), i));
			}

			return poly;
		}

		[Test]
		public async Task TestCcwBasic()
		{
			// CCW triangle
			await Assert.That(Polygon.Ccw(
				new Vec2(0.0, 0.0),
				new Vec2(1.0, 0.0),
				new Vec2(0.5, 1.0),
				1e-10)).IsEqualTo(1);

			// CW triangle
			await Assert.That(Polygon.Ccw(
				new Vec2(0.0, 0.0),
				new Vec2(0.5, 1.0),
				new Vec2(1.0, 0.0),
				1e-10)).IsEqualTo(-1);

			// Colinear
			await Assert.That(Polygon.Ccw(
				new Vec2(0.0, 0.0),
				new Vec2(1.0, 0.0),
				new Vec2(2.0, 0.0),
				1e-10)).IsEqualTo(0);
		}

		[Test]
		public async Task TestTriangleCcw()
		{
			PolygonsIdx poly = new PolygonsIdx
			{
				MakePoly(new[] { (0.0, 0.0), (1.0, 0.0), (0.5, 1.0) }),
			};
			List<IVec3> tris = Polygon.TriangulateIdx(poly, 1e-10, true);
			await Assert.That(tris.Count).IsEqualTo(1);
			await Assert.That(tris[0]).IsEqualTo(new IVec3(0, 1, 2));
		}

		[Test]
		public async Task TestSquareCcw()
		{
			// Unit square, CCW
			PolygonsIdx poly = new PolygonsIdx
			{
				MakePoly(new[] { (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0) }),
			};
			List<IVec3> tris = Polygon.TriangulateIdx(poly, 1e-10, true);
			await Assert.That(tris.Count).IsEqualTo(2);

			// Total triangles should cover the whole square
			foreach (IVec3 tri in tris)
			{
				await Assert.That(tri.X >= 0 && tri.X < 4).IsTrue();
				await Assert.That(tri.Y >= 0 && tri.Y < 4).IsTrue();
				await Assert.That(tri.Z >= 0 && tri.Z < 4).IsTrue();
			}
		}

		[Test]
		public async Task TestPentagonCcw()
		{
			int n = 5;
			(double X, double Y)[] coords = new (double, double)[n];
			for (int i = 0; i < n; i++)
			{
				double a = 2.0 * Types.KPi * i / n;
				coords[i] = (DeterministicMath.Cos(a), DeterministicMath.Sin(a));
			}

			PolygonsIdx poly = new PolygonsIdx { MakePoly(coords) };
			List<IVec3> tris = Polygon.TriangulateIdx(poly, 1e-10, false);
			await Assert.That(tris.Count).IsEqualTo(n - 2);
		}

		[Test]
		public async Task TestConvexFastPath()
		{
			int n = 8;
			(double X, double Y)[] coords = new (double, double)[n];
			for (int i = 0; i < n; i++)
			{
				double a = 2.0 * Types.KPi * i / n;
				coords[i] = (DeterministicMath.Cos(a), DeterministicMath.Sin(a));
			}

			PolygonsIdx poly = new PolygonsIdx { MakePoly(coords) };

			// is_convex should return true -> triangulate_convex used
			await Assert.That(Polygon.IsConvex(poly, 1e-10)).IsTrue();
			List<IVec3> tris = Polygon.TriangulateIdx(poly, 1e-10, true);
			await Assert.That(tris.Count).IsEqualTo(n - 2);
		}

		[Test]
		public async Task TestTriangulateUnindexed()
		{
			Polygons poly = new Polygons
			{
				new SimplePolygon
				{
					new Vec2(0.0, 0.0),
					new Vec2(1.0, 0.0),
					new Vec2(1.0, 1.0),
					new Vec2(0.0, 1.0),
				},
			};
			List<IVec3> tris = Polygon.Triangulate(poly, 1e-10, true);
			await Assert.That(tris.Count).IsEqualTo(2);
		}

		[Test]
		public async Task TestKdTreeQuery()
		{
			List<PolyVert> points = new List<PolyVert>();
			for (int i = 0; i < 20; i++)
			{
				points.Add(new PolyVert(new Vec2(i, i % 5), i));
			}

			Polygon.BuildTwoDTree(points);

			Rect queryRect = Rect.FromPoints(new Vec2(3.0, 0.0), new Vec2(8.0, 3.0));
			List<int> found = new List<int>();
			Polygon.QueryTwoDTree(points, queryRect, p => found.Add(p.Idx));

			// Points with x in [3,8] and y in [0,3]
			found = found.OrderBy(i => i).ToList();

			// i=3: x=3, y=3 yes; i=4: x=4, y=4 no; i=5: x=5, y=0 yes; i=6: x=6, y=1 yes;
			// i=7: x=7, y=2 yes; i=8: x=8, y=3 yes
			// (y = i%5: 3->3, 5->0, 6->1, 7->2, 8->3)
			await Assert.That(found.Contains(3)).IsTrue();
			await Assert.That(found.Contains(5)).IsTrue();
			await Assert.That(found.Contains(6)).IsTrue();
			await Assert.That(found.Contains(7)).IsTrue();
			await Assert.That(found.Contains(8)).IsTrue();
			await Assert.That(found.Contains(4)).IsFalse(); // y=4 outside
		}

		// ─── C#-port-only region: no Rust test corresponds to what follows ───────
		//
		// polygon.rs has no test for `triangulate_idx_halfedges`, so a test-for-test
		// port leaves it uncovered — and it is the one function here whose C# body is
		// not a transcription. Rust mutates the paired edge in place
		// (`halfedges[pair].paired_halfedge = idx`); C# cannot, because Halfedge is a
		// struct and `List<T>`'s indexer returns a copy, so AddHalfedge does a
		// read-modify-write-back dance instead. Delete the write-back line and the
		// code still compiles, still runs, and only the back-pointer is wrong.
		//
		// Nothing else would catch that: the reciprocity check inside
		// TriangulateIdxHalfedges is under `#if DEBUG`, and CI's Release lane compiles
		// it out. This test asserts the same invariants unconditionally.
		//
		// Two shapes, because they cover different halves of AddHalfedge:
		//   - a rect with a hole exercises keyholing, whose bridge edges are the
		//     densest source of pairing work in ordinary input;
		//   - two *coincident* contours sharing vertex indices are the only way to
		//     reach the LIFO branch of the multimap (below).
		[Test]
		public async Task HalfedgesOfRectWithHolePairReciprocally()
		{
			// 10x10 CCW outer (idx 0-3), 2x2 CW hole (idx 4-7).
			SimplePolygonIdx outer = MakePoly(new[] { (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0) });
			SimplePolygonIdx hole = new SimplePolygonIdx
			{
				new PolyVert(new Vec2(4.0, 4.0), 4),
				new PolyVert(new Vec2(4.0, 6.0), 5),
				new PolyVert(new Vec2(6.0, 6.0), 6),
				new PolyVert(new Vec2(6.0, 4.0), 7),
			};
			PolygonsIdx polys = new PolygonsIdx { outer, hole };

			HalfedgeTriangulation he = Polygon.TriangulateIdxHalfedges(polys, -1.0, false);

			// 8 input edges -> 8 contour halfedges. 8 verts with one hole triangulates
			// to nVerts + 2 * nHoles - 2 = 8 triangles, i.e. 24 more halfedges.
			await Assert.That(he.ContourEnd).IsEqualTo(8);
			await Assert.That(he.NumTri()).IsEqualTo(8);
			await Assert.That(he.Halfedges.Count).IsEqualTo(32);

			await AssertPairingIsReciprocal(he);

			// Measured, and worth recording because the obvious assumption is wrong:
			// keyholing does NOT produce duplicate directed vertex pairs. The bridge
			// duplicates a vert in EarClip's arena, but both copies carry the same
			// MeshIdx, so the bridge contributes (a,b) and (b,a) — a reciprocal pair,
			// not a repeat. All 32 halfedges here are distinct (start, end) pairs, so
			// this case never reaches the multimap's LIFO branch. The next test does.
			int distinctPairs = he.Halfedges
				.Select(edge => (edge.StartVert, edge.EndVert))
				.Distinct()
				.Count();
			await Assert.That(distinctPairs).IsEqualTo(32);
		}

		/// <summary>
		/// The LIFO branch of <c>AddHalfedge</c>'s multimap, which needs input where the
		/// same directed vertex pair occurs more than once.
		/// </summary>
		/// <remarks>
		/// Two coincident unit squares that *share* vertex indices 0-3 — the
		/// "degenerate/overlapping polygons" case named in
		/// <see cref="HalfedgeTriangulation"/>'s remarks, and the shape of what
		/// Face2Tri hands in for exactly-coplanar faces. Every one of the 20 halfedges
		/// has a twin with identical endpoints, so pairing is only well defined because
		/// the multimap pops the most recent unpaired opposite (Rust <c>Vec::pop</c>,
		/// <c>List&lt;int&gt;.RemoveAt(Count - 1)</c> here). A FIFO container would
		/// still pair everything, but into different partners.
		/// <para>
		/// Verified against manifold-rust on this exact input: 20 halfedges,
		/// contourEnd 8, 4 triangles, byte-identical pairing.
		/// </para>
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task HalfedgesOfCoincidentContoursPairThroughTheLifoMultimap()
		{
			SimplePolygonIdx square = MakePoly(new[] { (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0) });
			SimplePolygonIdx duplicate = MakePoly(new[] { (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0) });
			PolygonsIdx polys = new PolygonsIdx { square, duplicate };

			HalfedgeTriangulation he = Polygon.TriangulateIdxHalfedges(polys, -1.0, false);

			await Assert.That(he.ContourEnd).IsEqualTo(8);
			await Assert.That(he.NumTri()).IsEqualTo(4);
			await Assert.That(he.Halfedges.Count).IsEqualTo(20);

			// Every directed pair occurs exactly twice: 20 halfedges, 10 distinct pairs.
			// This is the assert that keeps the case on the LIFO branch; if it ever
			// reads 20, the test has stopped testing what it was written for.
			int distinctPairs = he.Halfedges
				.Select(edge => (edge.StartVert, edge.EndVert))
				.Distinct()
				.Count();
			await Assert.That(distinctPairs).IsEqualTo(10);

			await AssertPairingIsReciprocal(he);
		}

		/// <summary>
		/// The <c>#if DEBUG</c> Finalize checks inside
		/// <see cref="Polygon.TriangulateIdxHalfedges"/>, asserted unconditionally so
		/// the Release CI lane covers them too.
		/// </summary>
		/// <param name="he">The triangulation to check.</param>
		/// <returns>The assertion task.</returns>
		private static async Task AssertPairingIsReciprocal(HalfedgeTriangulation he)
		{
			for (int i = 0; i < he.Halfedges.Count; i++)
			{
				Halfedge edge = he.Halfedges[i];
				int pair = edge.PairedHalfedge;
				await Assert.That(pair >= 0 && pair < he.Halfedges.Count)
					.IsTrue()
					.Because($"halfedge {i} has out-of-range pair {pair}");

				Halfedge other = he.Halfedges[pair];
				await Assert.That(other.PairedHalfedge)
					.IsEqualTo(i)
					.Because($"halfedge {i} pairs to {pair}, which pairs back to {other.PairedHalfedge}");
				await Assert.That(edge.StartVert == other.EndVert && edge.EndVert == other.StartVert)
					.IsTrue()
					.Because($"halfedge {i} ({edge.StartVert}->{edge.EndVert}) does not reverse its pair "
						+ $"{pair} ({other.StartVert}->{other.EndVert})");
			}
		}
	}
}
