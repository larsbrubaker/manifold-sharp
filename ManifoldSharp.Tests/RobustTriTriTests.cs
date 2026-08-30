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

// RobustTriTriTests.cs — port of robust/tri_tri_tests.rs — unit tests for exact
// triangle-triangle intersection (Robust/TriTri.cs): generic crossings,
// vertex/edge contact, coplanar overlaps of every dimension, slivers, and
// argument symmetry. 15 tests, same inputs and same expected values.
//
// Assertions are synchronous AssertEq/AssertTrue helpers rather than awaited
// TUnit assertions, for the same reason RobustExactTests gives: a ported
// assert_eq! inside a loop should fail fast on the first mismatch, exactly as
// the Rust does.

using TUnit.Assertions;
using TUnit.Core;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Tests
{
	/// <summary>The ported <c>robust/tri_tri_tests.rs</c> suite.</summary>
	public class RobustTriTriTests
	{
		private static Vec3 V(double x, double y, double z)
		{
			return new Vec3(x, y, z);
		}

		private static R3 R3Of(double x, double y, double z)
		{
			return R3.FromVec3(V(x, y, z));
		}

		/// <summary>Rust's <c>assert_eq!</c>: synchronous and fail-fast.</summary>
		private static void AssertEq<T>(T actual, T expected, string what)
		{
			if (!EqualityComparer<T>.Default.Equals(actual, expected))
			{
				Assert.Fail($"{what}: expected {expected}, got {actual}");
			}
		}

		/// <summary>Rust's <c>assert_ne!</c>.</summary>
		private static void AssertNe<T>(T actual, T unexpected, string what)
		{
			if (EqualityComparer<T>.Default.Equals(actual, unexpected))
			{
				Assert.Fail($"{what}: expected something other than {unexpected}");
			}
		}

		/// <summary>Rust's <c>assert!</c>.</summary>
		private static void AssertTrue(bool condition, string what)
		{
			if (!condition)
			{
				Assert.Fail(what);
			}
		}

		/// <summary>Rust's <c>panic!</c> on an unexpected intersection kind.</summary>
		private static void Panic(string message)
		{
			Assert.Fail(message);
		}

		/// <summary>
		/// Segments and points compare as sets (segment endpoint order is arbitrary).
		/// </summary>
		private static void AssertSameIsect(TriTriIsect a, TriTriIsect b)
		{
			if (a.Kind == TriTriIsectKind.None && b.Kind == TriTriIsectKind.None)
			{
				return;
			}

			if (a.Kind == TriTriIsectKind.Point && b.Kind == TriTriIsectKind.Point)
			{
				AssertEq(a.P0!, b.P0!, "points differ");
				return;
			}

			if (a.Kind == TriTriIsectKind.Segment && b.Kind == TriTriIsectKind.Segment)
			{
				R3 p0 = a.P0!;
				R3 p1 = a.P1!;
				R3 q0 = b.P0!;
				R3 q1 = b.P1!;
				AssertTrue(
					(p0.Equals(q0) && p1.Equals(q1)) || (p0.Equals(q1) && p1.Equals(q0)),
					$"segments differ: {p0}-{p1} vs {q0}-{q1}");
				return;
			}

			if (a.Kind == TriTriIsectKind.Coplanar && b.Kind == TriTriIsectKind.Coplanar)
			{
				AssertEq(a.SameOrientation, b.SameOrientation, "same_orientation differs");
				AssertEq(a.Polygon!.Count, b.Polygon!.Count, "polygon vertex counts differ");

				// Rust sorts both Vec<R3> and compares; R3's Ord is the lexicographic
				// CompareTo, and OrderBy is the stable sort the plan mandates.
				List<R3> sa = a.Polygon!.OrderBy(p => p).ToList();
				List<R3> sb = b.Polygon!.OrderBy(p => p).ToList();
				for (int i = 0; i < sa.Count; i++)
				{
					AssertEq(sa[i], sb[i], "polygon vertex sets differ");
				}

				return;
			}

			Panic($"intersection kinds differ: {a} vs {b}");
		}

		/// <summary>Run both argument orders and require identical results.</summary>
		private static TriTriIsect IsectSym(Vec3[] t1, Vec3[] t2)
		{
			TriTriIsect ab = TriTri.TriTriIntersect(t1, t2);
			TriTriIsect ba = TriTri.TriTriIntersect(t2, t1);
			AssertSameIsect(ab, ba);
			return ab;
		}

		[Test]
		public void DisjointTriangles()
		{
			// Parallel planes.
			Vec3[] t1 = { V(0.0, 0.0, 0.0), V(1.0, 0.0, 0.0), V(0.0, 1.0, 0.0) };
			Vec3[] t2 = { V(0.0, 0.0, 1.0), V(1.0, 0.0, 1.0), V(0.0, 1.0, 1.0) };
			AssertEq(IsectSym(t1, t2), TriTriIsect.None, "parallel planes");

			// Crossing planes but separated triangles.
			Vec3[] t3 = { V(10.0, 10.0, -1.0), V(11.0, 10.0, 1.0), V(10.0, 11.0, 1.0) };
			AssertEq(IsectSym(t1, t3), TriTriIsect.None, "crossing planes, separated");

			// Same plane, disjoint.
			Vec3[] t4 = { V(5.0, 5.0, 0.0), V(6.0, 5.0, 0.0), V(5.0, 6.0, 0.0) };
			AssertEq(IsectSym(t1, t4), TriTriIsect.None, "same plane, disjoint");
		}

		[Test]
		public void GenericCrossingProducesSegment()
		{
			// t2 is vertical, punching through the horizontal t1.
			Vec3[] t1 = { V(0.0, 0.0, 0.0), V(4.0, 0.0, 0.0), V(0.0, 4.0, 0.0) };
			Vec3[] t2 = { V(1.0, -1.0, -1.0), V(1.0, 3.0, -1.0), V(1.0, 1.0, 2.0) };

			// In plane x=1, t2's edges cross z=0 at y ∈ {1/3, 5/3}... computed exactly:
			// edge (1,-1,-1)→(1,1,2): z=0 at t=1/3 → y = -1 + 2*(1/3)*... let the
			// assertion below verify incidence instead of hand-derived numbers.
			TriTriIsect got = IsectSym(t1, t2);
			if (got.Kind == TriTriIsectKind.Segment)
			{
				R3 p = got.P0!;
				R3 q = got.P1!;
				AssertNe(p, q, "endpoints coincide");

				// Both endpoints on both planes: z = 0 and x = 1.
				foreach (R3 e in new[] { p, q })
				{
					AssertEq(e.Z, Rational.Rat(0.0), "z = 0");
					AssertEq(e.X, Rational.Rat(1.0), "x = 1");
				}
			}
			else
			{
				Panic($"expected Segment, got {got}");
			}
		}

		[Test]
		public void KnownCrossingSegmentValues()
		{
			// t1 in z=0; t2 vertical wall x=1 with base straddling z=0 across y∈[0,2].
			Vec3[] t1 = { V(0.0, 0.0, 0.0), V(4.0, 0.0, 0.0), V(0.0, 4.0, 0.0) };
			Vec3[] t2 = { V(1.0, 0.0, -1.0), V(1.0, 2.0, -1.0), V(1.0, 1.0, 1.0) };

			// Edges cross z=0 at (1, 1/2, 0) and (1, 3/2, 0).
			R3 expect0 = new R3(Rational.Rat(1.0), Backend.RatNew(1, 2), Rational.Rat(0.0));
			R3 expect1 = new R3(Rational.Rat(1.0), Backend.RatNew(3, 2), Rational.Rat(0.0));
			TriTriIsect got = IsectSym(t1, t2);
			if (got.Kind == TriTriIsectKind.Segment)
			{
				R3 p = got.P0!;
				R3 q = got.P1!;
				(R3 lo, R3 hi) = p.Y < q.Y ? (p, q) : (q, p);
				AssertEq(lo, expect0, "the low endpoint");
				AssertEq(hi, expect1, "the high endpoint");
			}
			else
			{
				Panic($"expected Segment, got {got}");
			}
		}

		[Test]
		public void VertexTouchingFaceIsPoint()
		{
			Vec3[] t1 = { V(0.0, 0.0, 0.0), V(4.0, 0.0, 0.0), V(0.0, 4.0, 0.0) };

			// Apex touches t1's interior from above; the rest stays strictly above.
			Vec3[] t2 = { V(1.0, 1.0, 0.0), V(2.0, 1.0, 3.0), V(1.0, 2.0, 3.0) };
			AssertEq(IsectSym(t1, t2), TriTriIsect.Point(R3Of(1.0, 1.0, 0.0)), "the touch point");
		}

		[Test]
		public void VertexTouchingVertexIsPoint()
		{
			Vec3[] t1 = { V(0.0, 0.0, 0.0), V(4.0, 0.0, 0.0), V(0.0, 4.0, 0.0) };
			Vec3[] t2 = { V(0.0, 0.0, 0.0), V(-2.0, -1.0, 3.0), V(-1.0, -2.0, 3.0) };
			AssertEq(IsectSym(t1, t2), TriTriIsect.Point(R3Of(0.0, 0.0, 0.0)), "the shared vertex");
		}

		[Test]
		public void EdgeLyingInPlaneClippedToFace()
		{
			// t2 has one full edge inside t1's plane, crossing t1's interior.
			Vec3[] t1 = { V(0.0, 0.0, 0.0), V(4.0, 0.0, 0.0), V(0.0, 4.0, 0.0) };
			Vec3[] t2 = { V(1.0, -1.0, 0.0), V(1.0, 5.0, 0.0), V(1.0, 1.0, 3.0) };

			// The in-plane edge runs y ∈ [-1, 5] at x=1; t1 limits it to y ∈ [0, 3].
			TriTriIsect got = IsectSym(t1, t2);
			if (got.Kind == TriTriIsectKind.Segment)
			{
				R3 p = got.P0!;
				R3 q = got.P1!;
				(R3 lo, R3 hi) = p.Y < q.Y ? (p, q) : (q, p);
				AssertEq(lo, R3Of(1.0, 0.0, 0.0), "the low endpoint");
				AssertEq(hi, R3Of(1.0, 3.0, 0.0), "the high endpoint");
			}
			else
			{
				Panic($"expected Segment, got {got}");
			}
		}

		[Test]
		public void SharedEdgeBetweenMeshesIsSegment()
		{
			// Non-coplanar triangles sharing a full edge (the touching-cubes shape).
			Vec3[] t1 = { V(0.0, 0.0, 0.0), V(2.0, 0.0, 0.0), V(0.0, 0.0, 2.0) };
			Vec3[] t2 = { V(0.0, 0.0, 0.0), V(2.0, 0.0, 0.0), V(0.0, 3.0, 0.0) };
			TriTriIsect got = IsectSym(t1, t2);
			if (got.Kind == TriTriIsectKind.Segment)
			{
				R3 p = got.P0!;
				R3 q = got.P1!;
				(R3 lo, R3 hi) = p.X < q.X ? (p, q) : (q, p);
				AssertEq(lo, R3Of(0.0, 0.0, 0.0), "the low endpoint");
				AssertEq(hi, R3Of(2.0, 0.0, 0.0), "the high endpoint");
			}
			else
			{
				Panic($"expected Segment, got {got}");
			}
		}

		[Test]
		public void CoplanarIdenticalTriangles()
		{
			Vec3[] t1 = { V(0.0, 0.0, 1.0), V(4.0, 0.0, 1.0), V(0.0, 4.0, 1.0) };
			TriTriIsect got = IsectSym(t1, t1);
			if (got.Kind == TriTriIsectKind.Coplanar)
			{
				AssertTrue(got.SameOrientation, "same orientation");
				AssertEq(got.Polygon!.Count, 3, "polygon vertex count");
				List<R3> gotPts = got.Polygon!.OrderBy(p => p).ToList();
				List<R3> want = new List<R3>
				{
					R3Of(0.0, 0.0, 1.0),
					R3Of(4.0, 0.0, 1.0),
					R3Of(0.0, 4.0, 1.0),
				}
					.OrderBy(p => p)
					.ToList();
				for (int i = 0; i < want.Count; i++)
				{
					AssertEq(gotPts[i], want[i], $"polygon vertex {i}");
				}
			}
			else
			{
				Panic($"expected Coplanar, got {got}");
			}
		}

		[Test]
		public void CoplanarOppositeOrientationDetected()
		{
			Vec3[] t1 = { V(0.0, 0.0, 1.0), V(4.0, 0.0, 1.0), V(0.0, 4.0, 1.0) };
			Vec3[] t2 = { V(0.0, 0.0, 1.0), V(0.0, 4.0, 1.0), V(4.0, 0.0, 1.0) }; // flipped
			TriTriIsect got = IsectSym(t1, t2);
			if (got.Kind == TriTriIsectKind.Coplanar)
			{
				AssertTrue(!got.SameOrientation, "opposite orientation");
			}
			else
			{
				Panic($"expected Coplanar, got {got}");
			}
		}

		[Test]
		public void CoplanarPartialOverlapIsConvexPolygon()
		{
			// Two right triangles in z=0 overlapping in a square-ish quad.
			Vec3[] t1 = { V(0.0, 0.0, 0.0), V(4.0, 0.0, 0.0), V(0.0, 4.0, 0.0) };
			Vec3[] t2 = { V(1.0, 1.0, 0.0), V(5.0, 1.0, 0.0), V(1.0, 5.0, 0.0) };
			TriTriIsect got = IsectSym(t1, t2);
			if (got.Kind == TriTriIsectKind.Coplanar)
			{
				AssertTrue(got.SameOrientation, "same orientation");

				// Overlap is the triangle (1,1),(2,1)... worked out: the region
				// {x>=1, y>=1, x+y<=4} — a triangle with vertices (1,1),(3,1),(1,3).
				List<R3> gotPts = got.Polygon!.OrderBy(p => p).ToList();
				List<R3> want = new List<R3>
				{
					R3Of(1.0, 1.0, 0.0),
					R3Of(3.0, 1.0, 0.0),
					R3Of(1.0, 3.0, 0.0),
				}
					.OrderBy(p => p)
					.ToList();
				AssertEq(gotPts.Count, want.Count, "polygon vertex count");
				for (int i = 0; i < want.Count; i++)
				{
					AssertEq(gotPts[i], want[i], $"polygon vertex {i}");
				}
			}
			else
			{
				Panic($"expected Coplanar, got {got}");
			}
		}

		[Test]
		public void CoplanarHexagonalOverlap()
		{
			// Star-of-David configuration: two opposite equilateral-ish triangles
			// with a hexagonal intersection (6 vertices after canonicalization).
			Vec3[] t1 = { V(0.0, 0.0, 0.0), V(6.0, 0.0, 0.0), V(3.0, 6.0, 0.0) };
			Vec3[] t2 = { V(0.0, 4.0, 0.0), V(6.0, 4.0, 0.0), V(3.0, -2.0, 0.0) };
			TriTriIsect got = IsectSym(t1, t2);
			if (got.Kind == TriTriIsectKind.Coplanar)
			{
				AssertEq(got.Polygon!.Count, 6, "polygon vertex count");
			}
			else
			{
				Panic($"expected Coplanar, got {got}");
			}
		}

		[Test]
		public void CoplanarSharedEdgeOnlyIsSegment()
		{
			// Same plane, sharing exactly one edge, interiors disjoint.
			Vec3[] t1 = { V(0.0, 0.0, 0.0), V(4.0, 0.0, 0.0), V(0.0, 4.0, 0.0) };
			Vec3[] t2 = { V(0.0, 0.0, 0.0), V(4.0, 0.0, 0.0), V(2.0, -4.0, 0.0) };
			TriTriIsect got = IsectSym(t1, t2);
			if (got.Kind == TriTriIsectKind.Segment)
			{
				R3 p = got.P0!;
				R3 q = got.P1!;
				(R3 lo, R3 hi) = p.X < q.X ? (p, q) : (q, p);
				AssertEq(lo, R3Of(0.0, 0.0, 0.0), "the low endpoint");
				AssertEq(hi, R3Of(4.0, 0.0, 0.0), "the high endpoint");
			}
			else
			{
				Panic($"expected Segment, got {got}");
			}
		}

		[Test]
		public void CoplanarVertexTouchIsPoint()
		{
			Vec3[] t1 = { V(0.0, 0.0, 0.0), V(4.0, 0.0, 0.0), V(0.0, 4.0, 0.0) };
			Vec3[] t2 = { V(4.0, 0.0, 0.0), V(8.0, 0.0, 0.0), V(4.0, -4.0, 0.0) };

			// They share only the vertex (4,0,0): t2 lies in x>=4, y<=0.
			AssertEq(IsectSym(t1, t2), TriTriIsect.Point(R3Of(4.0, 0.0, 0.0)), "the shared vertex");
		}

		[Test]
		public void SliverCrossingStaysExact()
		{
			// A long, ulp-thin sliver crossing a big triangle: predicates must not
			// lose the crossing to rounding.
			// Rust's 2f64.powi(-40). Math.Pow and Rust's repeated-squaring powi differ
			// on some exponents (see RobustExactTests.PowI), but not on powers of two:
			// both are exactly 2^-40 here.
			double eps = Math.Pow(2.0, -40);
			Vec3[] t1 = { V(-10.0, -10.0, 0.0), V(10.0, -10.0, 0.0), V(0.0, 10.0, 0.0) };
			Vec3[] t2 =
			{
				V(-5.0, 0.0, -eps),
				V(5.0, 0.0, -eps),
				V(0.0, eps, 2.0 * eps),
			};
			TriTriIsect got = IsectSym(t1, t2);
			if (got.Kind == TriTriIsectKind.Segment)
			{
				AssertNe(got.P0!, got.P1!, "endpoints coincide");
			}
			else
			{
				Panic($"expected Segment, got {got}");
			}
		}

		[Test]
		public void DominantAxisAndLiftRoundTrip()
		{
			R3 a = R3Of(0.1, 0.2, 0.3);
			R3 b = R3Of(1.7, -0.4, 0.9);
			R3 c = R3Of(-0.6, 1.1, 2.2);
			R3 n = Predicates.TriNormalR(a, b, c);
			int axis = TriTri.DominantAxis(n);
			foreach (R3 p in new[] { a, b, c })
			{
				R3 lifted = TriTri.LiftToPlane(p.ProjectDrop(axis), axis, a, n);
				AssertEq(lifted, p, "the lifted point");
			}
		}
	}
}
