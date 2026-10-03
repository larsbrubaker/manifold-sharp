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

// Port of src/manifold_tests/compose.rs — all 4 tests, same inputs, same
// expected values, same order.
//
// Tests for how composing disjoint meshes keeps each input's runs — the MeshGL
// run table (RunIndex / RunOriginalId / RunTransform) that
// Boolean3Functions.ComposeMeshes builds for Manifold.Compose, the disjoint-union
// fast path in Boolean3Functions.BooleanWithToken and CsgTree.BatchUnion.
//
// C++ Compose (csg_tree.cpp:384-410) offsets node i's meshIDs by
// i * meshIDCounter before IncrementMeshIDs, so instanced copies of one mesh stay
// separate runs, each with its own transform, ordered node by node. The expected
// values below were captured (by the Rust) from the v3.5.2 C++ reference
// (cpp-reference/manifold built as a static library, Manifold::Compose /
// operator+ / GetMeshGL on the same inputs).

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ManifoldComposeTests
	{
		private static readonly float[] Identity = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };

		/// <summary>
		/// C++: Manifold::Compose({cube, cube.Translate({3, 0, 0})}) gives two runs of the
		/// cube's original ID, identity then the translation. Composing the same mesh twice
		/// must not merge the copies into one run.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ComposeKeepsEachInstancedCopyAsItsOwnRun()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold composed = Manifold.Compose(new List<Manifold> { cube.Clone(), cube.Translate(new Vec3(3.0, 0.0, 0.0)) });
			await AssertRuns(
				composed,
				cube.OriginalId(),
				12,
				new[] { Identity, Translated(3.0f, 0.0f, 0.0f) });
		}

		/// <summary>
		/// C++: Compose of three copies keeps node order — the (0, 5, 0) copy first even
		/// though it sorts last geometrically and none of the copies' mesh IDs differ.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ComposeOrdersInstancedCopiesNodeByNode()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold composed = Manifold.Compose(new List<Manifold>
			{
				cube.Translate(new Vec3(0.0, 5.0, 0.0)),
				cube.Clone(),
				cube.Translate(new Vec3(3.0, 0.0, 0.0)),
			});
			await AssertRuns(
				composed,
				cube.OriginalId(),
				12,
				new[] { Translated(0.0f, 5.0f, 0.0f), Identity, Translated(3.0f, 0.0f, 0.0f) });
		}

		/// <summary>
		/// C++: cube + cube.Translate({3, 0, 0}) — disjoint, so C++ BatchUnion Composes
		/// them; the port takes the disjoint-Add fast path in BooleanWithToken. Same two
		/// runs as Compose.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task DisjointUnionOfInstancedCopiesKeepsBothRuns()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold sum = cube.Union(cube.Translate(new Vec3(3.0, 0.0, 0.0)));
			await AssertRuns(
				sum,
				cube.OriginalId(),
				12,
				new[] { Identity, Translated(3.0f, 0.0f, 0.0f) });
		}

		/// <summary>
		/// The CSG tree's BatchUnion composes disjoint leaves that share one impl and
		/// differ only in their lazy leaf transform: each leaf keeps its own run, in leaf
		/// order.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task CsgBatchUnionKeepsEachInstancedLeafAsItsOwnRun()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			CsgNode Leaf(double x)
			{
				Mat3x4 t = Mat3x4.FromCols(
					new Vec3(1.0, 0.0, 0.0),
					new Vec3(0.0, 1.0, 0.0),
					new Vec3(0.0, 0.0, 1.0),
					new Vec3(x, 0.0, 0.0));
				return new CsgLeaf(new CsgLeafNode(cube.AsImpl().Clone(), t));
			}

			CsgNode tree = new CsgOp(OpType.Add, new List<CsgNode> { Leaf(0.0), Leaf(3.0), Leaf(6.0) });
			Manifold sum = Manifold.FromImpl(tree.Evaluate());
			await AssertRuns(
				sum,
				cube.OriginalId(),
				12,
				new[] { Identity, Translated(3.0f, 0.0f, 0.0f), Translated(6.0f, 0.0f, 0.0f) });
		}

		private static float[] Translated(float x, float y, float z)
		{
			float[] t = (float[])Identity.Clone();
			t[9] = x;
			t[10] = y;
			t[11] = z;
			return t;
		}

		/// <summary>
		/// Asserts the run table matches C++: one run per input node, in node order, each
		/// <paramref name="trisPerRun"/> triangles, all carrying
		/// <paramref name="originalId"/>, with the given per-run transforms.
		/// </summary>
		private static async Task AssertRuns(Manifold m, int originalId, uint trisPerRun, float[][] transforms)
		{
			MeshGL gl = m.GetMeshGL(-1);
			int n = transforms.Length;
			List<uint> expectedIndex = Enumerable.Range(0, n + 1).Select(r => 3 * trisPerRun * (uint)r).ToList();
			await Assert.That(gl.RunIndex).IsEquivalentTo(expectedIndex, CollectionOrdering.Matching).Because("run_index");
			await Assert.That(gl.RunOriginalId)
				.IsEquivalentTo(Enumerable.Repeat((uint)originalId, n).ToList(), CollectionOrdering.Matching)
				.Because("run_original_id");
			List<float> expectedTransform = transforms.SelectMany(t => t).ToList();
			await Assert.That(gl.RunTransform).IsEquivalentTo(expectedTransform, CollectionOrdering.Matching).Because("run_transform");
		}
	}
}
