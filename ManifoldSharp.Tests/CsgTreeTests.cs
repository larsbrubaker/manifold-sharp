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

// Port of the tests module in csg_tree.rs — all 10 cases, same inputs, same
// expected values, same order. Nothing is deferred: the first nine never touch
// the Phase 6 façade (the tenth only hashes its MeshGL64 export), and
// BatchBoolean/BatchUnion are reachable because the Rust's module-private `fn`s
// became `internal` (InternalsVisibleTo), which is the same
// visibility the Rust's in-module tests had.
//
// What these do NOT cover, and what does. Every assertion here is a triangle
// count or a bounding-box bracket, so the reduction ORDER that csg_tree.rs
// exists to pin down (BatchBoolean's max-heap pops, the 4-pairs-per-round
// structure, BatchUnion's front/back swap) is invisible to them — a tree
// evaluated in a different order still lands on 24 tris. CANCELLATION is no
// longer in that list: the Rust's csg cancel cases live in cancel_tests.rs, the
// Phase 6 façade they were blocked on has landed, and CancelTests.cs's
// PreCancelledTokenShortCircuitsCsgTreeEvaluation and
// CancelledStatusSurvivesTheCsgTreeRoot now reach EvaluateWithToken's Cancelled
// branches.
//
// The reduction order is still held by the differential harness this step ran
// against the compiled manifold-rust: 41 cases (26 CSG trees, 4
// pre-cancelled/live-token
// evaluations, 2 direct leaf realizations, 9 Minkowski pairs) x full impl dumps
// — verts, halfedges, face normals, vert normals, TriRefs, Relations, bbox,
// epsilon/tolerance/status, all doubles as raw IEEE bits — 16,212 lines, zero
// diffs. Two negative controls confirm the harness discriminates what these
// tests cannot: un-inverting the heap comparer moved 29,579 lines, and dropping
// the serial tie-break moved 22,522.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class CsgTreeTests
	{
		[Test]
		public async Task CsgTreeUnionDisjoint()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(3.0, 0.0, 0.0);
			CsgNode tree = new CsgOp(OpType.Add, new CsgLeaf(a), new CsgLeaf(b));
			ManifoldImpl result = tree.Evaluate();
			await Assert.That(result.NumTri()).IsEqualTo(24);
		}

		[Test]
		public async Task CsgTreeUnionOverlapping()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(0.5, 0.0, 0.0);
			CsgNode tree = new CsgOp(OpType.Add, new CsgLeaf(a), new CsgLeaf(b));
			ManifoldImpl result = tree.Evaluate();
			await Assert.That(result.NumTri())
				.IsGreaterThan(0)
				.Because("Overlapping union should produce non-empty mesh");
		}

		[Test]
		public async Task CsgTreeIntersection()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(0.5, 0.0, 0.0);
			CsgNode tree = new CsgOp(OpType.Intersect, new CsgLeaf(a), new CsgLeaf(b));
			ManifoldImpl result = tree.Evaluate();
			await Assert.That(result.NumTri())
				.IsGreaterThan(0)
				.Because("Overlapping intersection should produce non-empty mesh");
		}

		[Test]
		public async Task CsgTreeSubtract()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(0.5, 0.0, 0.0);
			CsgNode tree = new CsgOp(OpType.Subtract, new CsgLeaf(a), new CsgLeaf(b));
			ManifoldImpl result = tree.Evaluate();
			await Assert.That(result.NumTri())
				.IsGreaterThan(0)
				.Because("Subtraction should produce non-empty mesh");
		}

		[Test]
		public async Task BatchBooleanThreeCubes()
		{
			CsgLeafNode a = new CsgLeafNode(Cube(0.0, 0.0, 0.0));
			CsgLeafNode b = new CsgLeafNode(Cube(0.5, 0.0, 0.0));
			CsgLeafNode c = new CsgLeafNode(Cube(1.0, 0.0, 0.0));
			List<CsgLeafNode> children = new List<CsgLeafNode> { a, b, c };
			CsgLeafNode result = CsgTree.BatchBoolean(OpType.Add, children, null);
			ManifoldImpl mesh = result.GetImpl();
			await Assert.That(mesh.NumTri())
				.IsGreaterThan(0)
				.Because("BatchBoolean of 3 overlapping cubes should produce non-empty mesh");
		}

		[Test]
		public async Task BatchUnionDisjoint()
		{
			CsgLeafNode a = new CsgLeafNode(Cube(0.0, 0.0, 0.0));
			CsgLeafNode b = new CsgLeafNode(Cube(3.0, 0.0, 0.0));
			CsgLeafNode c = new CsgLeafNode(Cube(6.0, 0.0, 0.0));
			List<CsgLeafNode> children = new List<CsgLeafNode> { a, b, c };
			CsgLeafNode result = CsgTree.BatchUnion(children, null);
			ManifoldImpl mesh = result.GetImpl();

			// Three disjoint cubes should compose without boolean, giving 36 tris
			await Assert.That(mesh.NumTri())
				.IsEqualTo(36)
				.Because("BatchUnion of 3 disjoint cubes should have 36 tris");
		}

		[Test]
		public async Task CsgNAryUnion()
		{
			// N-ary union of 4 disjoint cubes
			List<CsgNode> nodes = new List<CsgNode>();
			for (int i = 0; i < 4; i++)
			{
				nodes.Add(new CsgLeaf(Cube(i * 3.0, 0.0, 0.0)));
			}

			CsgNode tree = new CsgOp(OpType.Add, nodes);
			ManifoldImpl result = tree.Evaluate();
			await Assert.That(result.NumTri())
				.IsEqualTo(48)
				.Because("N-ary union of 4 disjoint cubes should have 48 tris");
		}

		[Test]
		public async Task LazyLeafTransformAppliedOnEvaluate()
		{
			// Regression: GetImpl discarded ManifoldImpl.Transform's return value
			// (it is not in-place), so lazily-transformed leaves evaluated at the
			// origin. Two disjoint cubes — one translated via the *leaf* transform,
			// not baked into the mesh — must union to 24 tris, not collapse to 12.
			ManifoldImpl cube = ManifoldImpl.Cube(Mat3x4.Identity());
			CsgLeafNode a = new CsgLeafNode(cube.Clone());
			CsgLeafNode b = new CsgLeafNode(cube).ApplyTransform(
				LinalgFunctions.Mat4ToMat3x4(LinalgFunctions.TranslationMatrix(new Vec3(3.0, 0.0, 0.0))));
			Box bbox = b.GetImpl().Bbox;
			await Assert.That(bbox.Min.X >= 2.9 && bbox.Max.X <= 4.1)
				.IsTrue()
				.Because($"lazy transform not applied by GetImpl: bbox.x = [{bbox.Min.X}, {bbox.Max.X}]");

			CsgNode tree = new CsgOp(OpType.Add, new CsgLeaf(a), new CsgLeaf(b));
			await Assert.That(tree.Evaluate().NumTri()).IsEqualTo(24);
		}

		[Test]
		public async Task TreeTransforms()
		{
			// Test that transforms compose correctly through the tree
			ManifoldImpl a = ManifoldImpl.Cube(Mat3x4.Identity());
			CsgLeafNode leaf = new CsgLeafNode(a);
			CsgLeafNode translated = leaf.ApplyTransform(
				LinalgFunctions.Mat4ToMat3x4(LinalgFunctions.TranslationMatrix(new Vec3(5.0, 0.0, 0.0))));
			Box bbox = translated.GetBoundingBox();
			await Assert.That(bbox.Min.X)
				.IsGreaterThan(4.0)
				.Because($"Translated bbox min.x should be > 4.0, got {bbox.Min.X}");
			await Assert.That(bbox.Max.X)
				.IsLessThan(6.5)
				.Because($"Translated bbox max.x should be < 6.5, got {bbox.Max.X}");
		}

		/// <summary>
		/// Pins a union of 64 overlapping cubes to the sequential rounds' output, and with
		/// parallelism on as well. Coordinates are exact, so the hash holds on every platform.
		/// </summary>
		/// <remarks>
		/// The Rust's <c>#[cfg(feature = "parallel")]</c> leg runs rayon pools of 1 and 8
		/// threads; the C# analogue of the compile feature is the runtime switch, so that leg
		/// runs with <see cref="ManifoldParallel.Enabled"/> off and on, then restores it.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task BatchUnionRoundsKeepTheSequentialOutput()
		{
			static ulong Union()
			{
				List<CsgNode> leaves = Enumerable.Range(0, 64)
					.Select(i => (CsgNode)new CsgLeaf(Cube((i % 8) * 0.5, (i / 8) * 0.5, 0.0)))
					.ToList();
				MeshGL64 gl = Manifold.FromImpl(new CsgOp(OpType.Add, leaves).Evaluate()).GetMeshGL64(-1);
				ulong h = 0xcbf29ce484222325UL;
				void Eat(byte[] bytes)
				{
					foreach (byte b in bytes)
					{
						h = (h ^ b) * 0x00000100000001b3UL;
					}
				}

				foreach (double x in gl.VertProperties)
				{
					Eat(BitConverter.GetBytes(BitConverter.DoubleToInt64Bits(x)));
				}

				foreach (ulong x in gl.TriVerts.Concat(gl.RunIndex))
				{
					Eat(BitConverter.GetBytes(x));
				}

				return h;
			}

			bool restoreParallel = ManifoldParallel.Enabled;
			try
			{
				foreach (bool parallel in new[] { false, true })
				{
					ManifoldParallel.Enabled = parallel;
					await Assert.That(Union()).IsEqualTo(0x0045f3cce45928c2UL).Because($"parallel={parallel}");
				}
			}
			finally
			{
				ManifoldParallel.Enabled = restoreParallel;
			}

			await Assert.That(Union()).IsEqualTo(0x0045f3cce45928c2UL);
		}

		private static ManifoldImpl Cube(double x, double y, double z)
		{
			return ManifoldImpl.Cube(LinalgFunctions.Mat4ToMat3x4(
				LinalgFunctions.TranslationMatrix(new Vec3(x, y, z))));
		}
	}
}
