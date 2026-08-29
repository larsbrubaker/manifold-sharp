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

// Port of the tests module in collider.rs — same 6 cases, same expected values,
// in the same order.
//
// Every IsEquivalentTo below passes CollectionOrdering.Matching explicitly.
// TUnit's default is CollectionOrdering.Any — order-INSENSITIVE — which would
// silently turn a Rust `assert_eq!` on a Vec into a set comparison. The hit
// lists here are the BVH's leaf-visit order, which is exactly what the tests
// exist to pin.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ColliderTests
	{
		[Test]
		public async Task ColliderBoxOverlap()
		{
			Box[] boxes = new Box[]
			{
				Box.FromPoints(new Vec3(0.0, 0.0, 0.0), new Vec3(1.0, 1.0, 1.0)),
				Box.FromPoints(new Vec3(2.0, 2.0, 2.0), new Vec3(3.0, 3.0, 3.0)),
			};
			Collider collider = new Collider((Box[])boxes.Clone(), new uint[] { 0, 1 });
			List<(int, int)> hits = new List<(int, int)>();
			collider.CollisionsWithBoxes(boxes, true, (a, b) => hits.Add((a, b)));
			await Assert.That(hits.Count).IsEqualTo(0);

			Box[] queries = new Box[]
			{
				Box.FromPoints(new Vec3(0.5, 0.5, 0.5), new Vec3(2.5, 2.5, 2.5)),
			};
			collider.CollisionsWithBoxes(queries, false, (a, b) => hits.Add((a, b)));
			await Assert.That(hits).IsEquivalentTo(
				new List<(int, int)> { (0, 0), (0, 1) },
				CollectionOrdering.Matching);
		}

		[Test]
		public async Task RayTriangleIntersection()
		{
			Vec3[] tri = new Vec3[]
			{
				new Vec3(0.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
			};
			double? hit = ColliderFunctions.RayTriangleIntersection(
				new Vec3(0.25, 0.25, -1.0),
				new Vec3(0.0, 0.0, 1.0),
				tri);
			await Assert.That(hit.HasValue).IsTrue();
		}

		[Test]
		public async Task TriangleTriangleDistanceZeroForIntersection()
		{
			Vec3[] a = new Vec3[]
			{
				new Vec3(0.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
			};
			Vec3[] b = new Vec3[]
			{
				new Vec3(0.25, 0.25, -1.0),
				new Vec3(0.25, 0.25, 1.0),
				new Vec3(0.75, 0.25, 0.0),
			};
			await Assert.That(ColliderFunctions.DistanceTriangleTriangleSquared(a, b)).IsEqualTo(0.0);
		}

		/// <summary>
		/// A cube's triangles never reach the FaceNormal-reading branch of
		/// IsSelfIntersecting (Collider.Geometry.cs): every triangle pair whose boxes
		/// overlap either shares a vertex, so the epsilon early-out fires first, or is a
		/// positive distance apart. Worth stating because that branch is the only reader
		/// of FaceNormal on this path — it mattered acutely while face_op was unported and
		/// FaceNormal was empty, when reaching it would have thrown rather than answered
		/// wrongly. FromShape now populates face normals, so the branch is merely unvisited
		/// rather than unsafe; the expected answer is unchanged and matches the Rust.
		/// </summary>
		[Test]
		public async Task CubeNotSelfIntersecting()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			await Assert.That(m.IsSelfIntersecting()).IsFalse();
		}

		[Test]
		public async Task MinGapBetweenCubes()
		{
			ManifoldImpl a = ManifoldImpl.Cube(Mat3x4.Identity());
			ManifoldImpl b = ManifoldImpl.Cube(
				LinalgFunctions.Mat4ToMat3x4(LinalgFunctions.TranslationMatrix(new Vec3(2.0, 0.0, 0.0))));
			double gap = a.MinGap(b, 5.0);
			await Assert.That(Math.Abs(gap - 1.0) < 1e-8).IsTrue().Because($"gap = {gap}");
		}

		[Test]
		public async Task BvhManyBoxes()
		{
			// Create many non-overlapping boxes and verify BVH finds the right pairs
			int n = 100;
			List<Box> boxes = new List<Box>();
			List<uint> mortons = new List<uint>();
			for (int i = 0; i < n; i++)
			{
				double x = i * 3.0;
				boxes.Add(Box.FromPoints(new Vec3(x, 0.0, 0.0), new Vec3(x + 1.0, 1.0, 1.0)));
				mortons.Add((uint)i);
			}

			Collider collider = new Collider(boxes.ToArray(), mortons.ToArray());

			// Query that overlaps box 50
			Box[] query = new Box[]
			{
				Box.FromPoints(new Vec3(150.5, 0.5, 0.5), new Vec3(150.6, 0.6, 0.6)),
			};
			List<(int, int)> hits = new List<(int, int)>();
			collider.CollisionsWithBoxes(query, false, (a, b) => hits.Add((a, b)));
			await Assert.That(hits).IsEquivalentTo(
				new List<(int, int)> { (0, 50) },
				CollectionOrdering.Matching);
		}
	}
}
