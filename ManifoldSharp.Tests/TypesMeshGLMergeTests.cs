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

// Port of types_meshgl_merge_tests.rs — regression tests for MeshGLP::merge
// (types_meshgl.rs; here MeshGLMerge.cs behind MeshGL.Merge / MeshGL64.Merge), the
// port of C++ MergeMeshGLP in src/sort.cpp. Same cases, same expected values, in the
// same order.
//
// The cases pin the open-edge bookkeeping that used to diverge from the C++
// (manifold-rust docs/CPP_DIVERGENCES.md, retired entry 6): C++ keeps open halfedges
// in a std::multiset, so a halfedge listed twice survives one reverse match, and
// every remaining open halfedge contributes one open-vertex entry.
//
// The Rust's `doubled_face_probe<P, I>` is generic over the two instantiations; C#
// has one builder per class, fed from the same position and triangle tables so the
// two cannot drift apart. Every merge-vector compare passes
// CollectionOrdering.Matching: the vectors are parallel and positional.

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class TypesMeshGLMergeTests
	{
		/// <summary>
		/// Unit tetrahedron (verts 0..3) with face (0,2,1) listed twice first, plus a
		/// separate open triangle (4,5,6) whose vert 4 coincides with vert 0.
		/// </summary>
		private static readonly double[] ProbePos = new double[]
		{
			0.0, 0.0, 0.0,
			1.0, 0.0, 0.0,
			0.0, 1.0, 0.0,
			0.0, 0.0, 1.0,
			0.0, 0.0, 0.0,
			-1.0, 0.0, 0.0,
			0.0, -1.0, 0.0,
		};

		private static readonly int[] ProbeTris = new int[]
		{
			0, 2, 1, // listed twice before its reverse halfedges arrive
			0, 2, 1,
			0, 1, 3,
			1, 2, 3,
			0, 3, 2,
			4, 5, 6, // open triangle
		};

		/// <summary>The Rust <c>doubled_face_probe::&lt;f32, u32&gt;()</c>.</summary>
		private static MeshGL DoubledFaceProbe()
		{
			MeshGL mesh = new MeshGL();
			mesh.NumProp = 3;
			mesh.VertProperties = ProbePos.Select(v => (float)v).ToList();
			mesh.TriVerts = ProbeTris.Select(v => (uint)v).ToList();
			return mesh;
		}

		/// <summary>The Rust <c>doubled_face_probe::&lt;f64, u64&gt;()</c>.</summary>
		private static MeshGL64 DoubledFaceProbe64()
		{
			MeshGL64 mesh = new MeshGL64();
			mesh.NumProp = 3;
			mesh.VertProperties = ProbePos.ToList();
			mesh.TriVerts = ProbeTris.Select(v => (ulong)v).ToList();
			return mesh;
		}

		/// <summary>
		/// C++ MergeMeshGLP: the doubled face leaves one copy of each of its three
		/// halfedges open, so verts 0, 1, 2 are open alongside 4, 5, 6 and the
		/// coincident vert 4 merges into 0. The old BTreeSet port deduplicated the
		/// doubled halfedges, left only 4, 5, 6 open and found no merge.
		/// </summary>
		[Test]
		public async Task MergeKeepsDuplicateOpenHalfedgesLikeCppMultiset()
		{
			MeshGL mesh = DoubledFaceProbe();
			await Assert.That(mesh.Merge()).IsTrue();
			await Assert.That(mesh.MergeFromVert).IsEquivalentTo(
				new List<uint> { 4 }, CollectionOrdering.Matching);
			await Assert.That(mesh.MergeToVert).IsEquivalentTo(
				new List<uint> { 0 }, CollectionOrdering.Matching);
		}

		/// <summary>
		/// C++ MergeMeshGLP is a template instantiated for MeshGL64 too; the same probe
		/// must merge identically at double precision / 64-bit indices.
		/// </summary>
		[Test]
		public async Task MergeMeshgl64MatchesMeshglOnDoubledFaceProbe()
		{
			MeshGL64 mesh = DoubledFaceProbe64();
			await Assert.That(mesh.Merge()).IsTrue();
			await Assert.That(mesh.MergeFromVert).IsEquivalentTo(
				new List<ulong> { 4 }, CollectionOrdering.Matching);
			await Assert.That(mesh.MergeToVert).IsEquivalentTo(
				new List<ulong> { 0 }, CollectionOrdering.Matching);

			// Second pass: the doubled face and the open triangle still leave open edges
			// (true), but no new coincident pairs, so the result is unchanged.
			await Assert.That(mesh.Merge()).IsTrue();
			await Assert.That(mesh.MergeFromVert).IsEquivalentTo(
				new List<ulong> { 4 }, CollectionOrdering.Matching);
			await Assert.That(mesh.MergeToVert).IsEquivalentTo(
				new List<ulong> { 0 }, CollectionOrdering.Matching);
		}

		/// <summary>
		/// MeshGL64 on a closed mesh: no open edges, so merge reports false and leaves the
		/// merge vectors untouched, as C++ returns before clearing them.
		/// </summary>
		[Test]
		public async Task MergeMeshgl64ClosedMeshReturnsFalse()
		{
			MeshGL64 mesh = DoubledFaceProbe64();

			// Drop the duplicate face and the open triangle: a clean tetrahedron.
			mesh.TriVerts = mesh.TriVerts.GetRange(3, 12);
			await Assert.That(mesh.Merge()).IsFalse();
			await Assert.That(mesh.MergeFromVert.Count).IsEqualTo(0);
			await Assert.That(mesh.MergeToVert.Count).IsEqualTo(0);
		}

		/// <summary>
		/// Pinched boundary, no duplicate halfedges: verts 2 and 4 each start two open
		/// halfedges, so C++ lists them twice in openVerts. The partition is the same
		/// either way ({0, 4, 5} merge; 2 and 3 are just over tolerance apart), but the
		/// duplicate leaves change the BVH shape and hence the order in which
		/// union-by-rank sees the pairs, so the representative differs: C++ picks 0, the
		/// old deduplicating port picked 4 (from=[0 5], to=[4 4]). Expected values come
		/// from compiling C++ v3.5.2 MergeMeshGLP (src/sort.cpp) against the reference
		/// headers and running it on this exact input (found by a randomized C++-vs-Rust
		/// sweep of 3000 open meshes, all of which now agree).
		/// </summary>
		[Test]
		public async Task MergePinchedBoundaryPicksCppRepresentative()
		{
			MeshGL mesh = new MeshGL();
			mesh.NumProp = 3;
			mesh.VertProperties = new List<float>
			{
				1.02016f, 1.03536f, 0.0f,
				0.0f, -0.06496f, 0.0f,
				-0.05568f, 0.01168f, 0.0f,
				0.07056f, 0.01072f, 0.0f,
				1.03808f, 0.99952f, 0.0f,
				0.95856f, 0.98976f, 0.0f,
			};
			mesh.TriVerts = new List<uint> { 0, 4, 2, 5, 2, 3, 5, 3, 4 };
			mesh.Tolerance = 0.1f;
			await Assert.That(mesh.Merge()).IsTrue();
			await Assert.That(mesh.MergeFromVert).IsEquivalentTo(
				new List<uint> { 4, 5 }, CollectionOrdering.Matching);
			await Assert.That(mesh.MergeToVert).IsEquivalentTo(
				new List<uint> { 0, 0 }, CollectionOrdering.Matching);
		}
	}
}
