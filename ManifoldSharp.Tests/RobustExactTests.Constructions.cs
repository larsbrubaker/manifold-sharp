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

// RobustExactTests.Constructions.cs — the "Constructions" section of
// robust/exact/tests.rs. The class header is on RobustExactTests.cs.

using TUnit.Core;

using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Tests
{
	public partial class RobustExactTests
	{
		[Test]
		public void LinePlaneIntersectLandsOnPlaneAndSegment()
		{
			// Plane z = 1 (triangle in that plane), segment from below to above.
			R3 a = R3Of(0.0, 0.0, 1.0);
			R3 b = R3Of(4.0, 0.0, 1.0);
			R3 c = R3Of(0.0, 4.0, 1.0);
			R3 p = R3Of(1.0, 1.0, 0.0);
			R3 q = R3Of(1.0, 1.0, 3.0);
			R3? x = Predicates.LinePlaneIntersect(p, q, a, b, c);
			AssertTrue(x != null, "not parallel");
			AssertEq(x!, R3Of(1.0, 1.0, 1.0), "the intersection point");

			// Exactly on the plane: orient3d of the four points is Zero.
			AssertEq(Predicates.Orient3dR(a, b, c, x!), Sign.Zero, "on the plane");

			// Parameter is exactly 1/3.
			AssertEq(Predicates.SegmentParam(p, q, x!), Backend.RatNew(1, 3), "the segment parameter");

			// Parallel segment → null.
			R3 p2 = R3Of(0.0, 0.0, 2.0);
			R3 q2 = R3Of(1.0, 1.0, 2.0);
			AssertTrue(Predicates.LinePlaneIntersect(p2, q2, a, b, c) == null, "parallel is none");
		}

		[Test]
		public void LinePlaneIntersectIsExactOnAwkwardFractions()
		{
			// A skew plane and a segment whose crossing has no finite binary
			// representation; verify the exact incidence property instead of floats.
			R3 a = R3Of(0.1, 0.2, 0.3);
			R3 b = R3Of(1.7, -0.4, 0.9);
			R3 c = R3Of(-0.6, 1.1, 2.2);
			R3 p = R3Of(0.3, 0.3, -5.0);
			R3 q = R3Of(0.4, 0.5, 7.0);
			R3? x = Predicates.LinePlaneIntersect(p, q, a, b, c);
			AssertTrue(x != null, "not parallel");
			AssertEq(Predicates.Orient3dR(a, b, c, x!), Sign.Zero, "on the plane");

			// x is on the line p→q: (x-p) × (q-p) = 0.
			R3 n = x!.Sub(p).Cross(q.Sub(p));
			AssertTrue(n.IsZero(), "the point is off the line p→q");
		}

		[Test]
		public void LineLineIntersect2dCrossingDiagonals()
		{
			R2? x = Predicates.LineLineIntersect2d(
				R2Of(0.0, 0.0),
				R2Of(1.0, 1.0),
				R2Of(1.0, 0.0),
				R2Of(0.0, 1.0));
			AssertTrue(x != null, "not parallel");
			AssertEq(x!, R2Of(0.5, 0.5), "the crossing point");

			// Parallel and collinear both report null.
			AssertTrue(
				Predicates.LineLineIntersect2d(
					R2Of(0.0, 0.0),
					R2Of(1.0, 0.0),
					R2Of(0.0, 1.0),
					R2Of(1.0, 1.0)) == null,
				"parallel is none");
			AssertTrue(
				Predicates.LineLineIntersect2d(
					R2Of(0.0, 0.0),
					R2Of(1.0, 0.0),
					R2Of(2.0, 0.0),
					R2Of(3.0, 0.0)) == null,
				"collinear is none");
		}

		[Test]
		public void PointInTri2dClassifiesAllRegions()
		{
			R2 a = R2Of(0.0, 0.0);
			R2 b = R2Of(4.0, 0.0);
			R2 c = R2Of(0.0, 4.0);
			AssertEq(Predicates.PointInTri2d(R2Of(1.0, 1.0), a, b, c), TriLoc.Inside, "inside");
			AssertEq(Predicates.PointInTri2d(R2Of(2.0, 0.0), a, b, c), TriLoc.OnEdge(0), "edge 0");
			AssertEq(Predicates.PointInTri2d(R2Of(2.0, 2.0), a, b, c), TriLoc.OnEdge(1), "edge 1");
			AssertEq(Predicates.PointInTri2d(R2Of(0.0, 1.0), a, b, c), TriLoc.OnEdge(2), "edge 2");
			AssertEq(Predicates.PointInTri2d(a, a, b, c), TriLoc.OnVertex(0), "vertex 0");
			AssertEq(Predicates.PointInTri2d(b, a, b, c), TriLoc.OnVertex(1), "vertex 1");
			AssertEq(Predicates.PointInTri2d(c, a, b, c), TriLoc.OnVertex(2), "vertex 2");
			AssertEq(Predicates.PointInTri2d(R2Of(3.0, 3.0), a, b, c), TriLoc.Outside, "outside");
			AssertEq(Predicates.PointInTri2d(R2Of(-0.1, 1.0), a, b, c), TriLoc.Outside, "outside left");

			// Same answers with clockwise winding.
			AssertEq(Predicates.PointInTri2d(R2Of(1.0, 1.0), a, c, b), TriLoc.Inside, "inside, CW");
			AssertEq(Predicates.PointInTri2d(R2Of(2.0, 0.0), a, c, b), TriLoc.OnEdge(2), "edge 2, CW");

			// Degenerate triangle: everything is Outside.
			R2 d = R2Of(8.0, 0.0);
			AssertEq(Predicates.PointInTri2d(R2Of(1.0, 0.0), a, b, d), TriLoc.Outside, "degenerate");
		}

		[Test]
		public void TriNormalRMatchesOrientation()
		{
			R3 n = Predicates.TriNormalR(R3Of(0.0, 0.0, 0.0), R3Of(1.0, 0.0, 0.0), R3Of(0.0, 1.0, 0.0));
			AssertEq(n, R3Of(0.0, 0.0, 1.0), "the CCW normal");

			// Degenerate triangle → zero normal.
			R3 z = Predicates.TriNormalR(R3Of(0.0, 0.0, 0.0), R3Of(1.0, 1.0, 1.0), R3Of(2.0, 2.0, 2.0));
			AssertTrue(z.IsZero(), "a degenerate triangle has a zero normal");
		}
	}
}
