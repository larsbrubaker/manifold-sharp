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

// Port of src/manifold_tests/compose.rs — all 5 tests, same inputs, same
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
// operator+ / GetMeshGL on the same inputs). The last test pins a full-MeshGL
// hash of a CSG-tree union whose pairs run in parallel under ManifoldParallel
// (CsgTree.BatchBoolean), so the run table cannot follow the order in which those
// pairs reserve mesh IDs.
//
// The Rust's `#[cfg(feature = "parallel")]` legs build rayon pools of 1 and 8
// threads; the C# analogue of that compile feature is the runtime switch, so
// those legs run with ManifoldParallel.Enabled off and on (restored afterwards,
// under ParallelismTests.ParallelismGlobalStateKey), and the unconditional
// assertion runs with whatever the process was started with.

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
		private const ulong RotatedInstancedSpheresHash = 0xe55d84ecb9f62e40UL;

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

		/// <summary>
		/// The full MeshGL of the union of four rotated instanced spheres is pinned, and
		/// with parallelism it is the same with the switch off and on, repeated, because
		/// ComposeMeshes ranks runs by input order rather than by the absolute mesh IDs.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task BatchUnionOfRotatedInstancedLeavesIsIndependentOfScheduling()
		{
			bool restoreParallel = ManifoldParallel.Enabled;
			try
			{
				foreach (bool parallel in new[] { false, true })
				{
					ManifoldParallel.Enabled = parallel;
					for (int rep = 0; rep < 8; rep++)
					{
						await Assert.That(await RotatedInstancedSphereUnion()).IsEqualTo(RotatedInstancedSpheresHash)
							.Because($"parallel={parallel}, repetition {rep}");
					}
				}
			}
			finally
			{
				ManifoldParallel.Enabled = restoreParallel;
			}

			await Assert.That(await RotatedInstancedSphereUnion()).IsEqualTo(RotatedInstancedSpheresHash);
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

		/// <summary>
		/// FNV-1a 64 over every MeshGL64 field, each array prefixed by its length (u64 LE):
		/// NumProp, VertProperties, TriVerts, MergeFromVert, MergeToVert, RunIndex,
		/// RunOriginalId (minus <paramref name="baseId"/>, as u32 LE, so the hash does not
		/// depend on where the process-wide ID counter stood), RunTransform, FaceId,
		/// HalfedgeTangent, RunFlags, then Tolerance. Floats hash as their IEEE bits,
		/// integers as LE bytes of their MeshGL64 width.
		/// </summary>
		private static ulong MeshGL64Hash(MeshGL64 gl, uint baseId)
		{
			ulong h = 0xcbf29ce484222325UL;
			void Eat(ReadOnlySpan<byte> bytes)
			{
				foreach (byte b in bytes)
				{
					h = (h ^ b) * 0x00000100000001b3UL;
				}
			}

			void EatU64(ulong x) => Eat(BitConverter.GetBytes(x));
			void EatF64(double x) => EatU64((ulong)BitConverter.DoubleToInt64Bits(x));

			if (!BitConverter.IsLittleEndian)
			{
				throw new PlatformNotSupportedException("the hash is defined over little-endian bytes");
			}

			EatU64(gl.NumProp);
			EatU64((ulong)gl.VertProperties.Count);
			gl.VertProperties.ForEach(EatF64);
			foreach (List<ulong> ints in new[] { gl.TriVerts, gl.MergeFromVert, gl.MergeToVert, gl.RunIndex })
			{
				EatU64((ulong)ints.Count);
				ints.ForEach(EatU64);
			}

			EatU64((ulong)gl.RunOriginalId.Count);
			foreach (uint id in gl.RunOriginalId)
			{
				Eat(BitConverter.GetBytes(unchecked(id - baseId))); // Rust wrapping_sub
			}

			EatU64((ulong)gl.RunTransform.Count);
			gl.RunTransform.ForEach(EatF64);
			EatU64((ulong)gl.FaceId.Count);
			gl.FaceId.ForEach(EatU64);
			EatU64((ulong)gl.HalfedgeTangent.Count);
			gl.HalfedgeTangent.ForEach(EatF64);
			EatU64((ulong)gl.RunFlags.Count);
			Eat(gl.RunFlags.ToArray());
			EatF64(gl.Tolerance);
			return h;
		}

		/// <summary>
		/// Four spheres sharing one original ID — two of 8 segments at y = 0, two of 256 at
		/// y = 2.2 — each rotated 45 degrees about z through its leaf's lazy transform, so
		/// the leaves' bounding boxes overlap (BatchUnion cannot compose them) while the
		/// meshes are disjoint (each pair takes the disjoint-Add fast path). Under
		/// ManifoldParallel the pairs of a BatchBoolean round reserve mesh IDs in whatever
		/// order they finish; the run table must not follow that order.
		/// </summary>
		private static async Task<ulong> RotatedInstancedSphereUnion()
		{
			uint id = Manifold.ReserveIds(1);
			Manifold Sphere(int segments)
			{
				MeshGL64 gl = Manifold.Sphere(1.0, segments).GetMeshGL64(-1);
				gl.RunOriginalId = new List<uint> { id };
				gl.RunIndex.Clear();
				gl.RunTransform.Clear();
				gl.RunFlags.Clear();
				gl.FaceId.Clear();
				return Manifold.FromMeshGL64(gl);
			}

			Manifold lo = Sphere(8);
			Manifold hi = Sphere(256);
			double r = 0.70710678118654752440; // std::f64::consts::FRAC_1_SQRT_2
			CsgNode Leaf(Manifold m, double x, double y)
			{
				Mat3x4 t = Mat3x4.FromCols(
					new Vec3(r, r, 0.0),
					new Vec3(-r, r, 0.0),
					new Vec3(0.0, 0.0, 1.0),
					new Vec3(x, y, 0.0));
				return new CsgLeaf(new CsgLeafNode(m.AsImpl().Clone(), t));
			}

			List<CsgNode> leaves = new List<CsgNode>
			{
				Leaf(lo, 0.0, 0.0),
				Leaf(lo, 2.2, 0.0),
				Leaf(hi, 0.0, 2.2),
				Leaf(hi, 2.2, 2.2),
			};
			Manifold sum = Manifold.FromImpl(new CsgOp(OpType.Add, leaves).Evaluate());
			MeshGL64 result = sum.GetMeshGL64(-1);
			await Assert.That(result.RunOriginalId)
				.IsEquivalentTo(new List<uint> { id, id, id, id }, CollectionOrdering.Matching)
				.Because("one run per sphere");
			return MeshGL64Hash(result, id);
		}
	}
}
