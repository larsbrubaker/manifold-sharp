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

// Port of subdivision_tests.rs — the same seven cases, same expected values, in
// the same order. Nothing here needs the public façade or the smoothing
// tangents, so the whole Rust module ports in one go. After the seven comes one
// C#-only test in its own labeled region, guarding the hand-written
// Partition.Clone the Rust gets from #[derive(Clone)] for free.

using System.Reflection;

using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class SubdivisionTests
	{
		[Test]
		public async Task TestSubdivideCubeOnce()
		{
			ManifoldImpl cube = ManifoldImpl.Cube(Mat3x4.Identity());
			ManifoldImpl sub = Subdivision.SubdivideImpl(cube, 1);
			await Assert.That(sub.NumTri()).IsEqualTo(cube.NumTri() * 4);
			await Assert.That(sub.NumVert()).IsGreaterThan(cube.NumVert());
		}

		[Test]
		public async Task TestPartitionSingleTriangle()
		{
			// Simplest case: 1 division per edge = no subdivision
			Partition part = Partition.GetPartition(new IVec4(1, 1, 1, 0));
			await Assert.That(part.TriVert.Count).IsEqualTo(1);
			await Assert.That(part.VertBary.Count).IsEqualTo(3);
		}

		[Test]
		public async Task TestPartitionTwoDivisions()
		{
			// 2 divisions on each edge of a triangle
			Partition part = Partition.GetPartition(new IVec4(2, 2, 2, 0));
			await Assert.That(part.TriVert.Count).IsEqualTo(4); // 4 sub-triangles
		}

		[Test]
		public async Task TestSubdivideEdgeDivisions()
		{
			// Test that the full Subdivide method works with edge_divisions callback
			ManifoldImpl cube = ManifoldImpl.Cube(Mat3x4.Identity());
			List<Barycentric> vertBary = cube.Subdivide((vec, t0, t1) => 1, false);
			_ = vertBary;
			await Assert.That(cube.NumTri()).IsEqualTo(12 * 4); // each of 12 tris becomes 4
		}

		[Test]
		public async Task TestCreateTmpEdges()
		{
			ManifoldImpl cube = ManifoldImpl.Cube(Mat3x4.Identity());
			List<TmpEdge> edges = Subdivision.CreateTmpEdges(cube.Halfedge);

			// A cube has 12 triangles = 36 halfedges = 18 edges
			await Assert.That(edges.Count).IsEqualTo(cube.Halfedge.Count / 2);
		}

		[Test]
		public async Task TestPartitionAsymmetric()
		{
			// Asymmetric divisions: 3,2,1
			Partition part = Partition.GetPartition(new IVec4(3, 2, 1, 0));
			await Assert.That(part.TriVert.Count).IsGreaterThan(1);

			// All barycentric coordinates should sum to 1
			foreach (Vec4 bary in part.VertBary)
			{
				double sum = bary.X + bary.Y + bary.Z + bary.W;
				await Assert.That(Math.Abs(sum - 1.0)).IsLessThan(1e-10)
					.Because($"Barycentric coords should sum to 1, got {sum}");
			}
		}

		[Test]
		public async Task TestSubdividePreservesManifold()
		{
			ManifoldImpl cube = ManifoldImpl.Cube(Mat3x4.Identity());
			ManifoldImpl sub = Subdivision.SubdivideImpl(cube, 1);

			// Every halfedge should have a valid pair
			for (int i = 0; i < sub.Halfedge.Count; i++)
			{
				Halfedge he = sub.Halfedge[i];
				await Assert.That(he.PairedHalfedge).IsGreaterThanOrEqualTo(0)
					.Because($"Halfedge {i} has no pair");
				int pair = he.PairedHalfedge;
				await Assert.That(pair).IsLessThan(sub.Halfedge.Count)
					.Because($"Halfedge {i} pair {pair} out of range");
			}
		}

		#region C#-only regression tests (no Rust counterpart)

		/// <summary>
		/// Fails when a field is added to <see cref="Partition"/> without being added to
		/// <c>Partition.Clone</c>.
		/// </summary>
		/// <remarks>
		/// No counterpart in subdivision_partition.rs, and none possible: the Rust gets its
		/// copy from <c>#[derive(Clone)]</c>, which cannot fall behind the struct. The C#
		/// port hand-writes <c>Clone</c> because the two <c>List</c> fields need a deep
		/// copy, and a hand-written clone silently drops any field added later. The
		/// consequence would not be a crash — <see cref="Partition.GetPartition"/> hands
		/// out clones of process-global cache entries, so a dropped field means one call
		/// site quietly reads a default value, or worse, aliases the cache. Counting the
		/// fields is the cheapest alarm that fires at the moment of the edit.
		/// </remarks>
		[Test]
		public async Task PartitionCloneCoversEveryField()
		{
			// Auto-properties, so these are the compiler's backing fields; the names are
			// what tells a reader which member each one belongs to.
			List<string> fields = typeof(Partition)
				.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
				.Select(f => f.Name)
				.OrderBy(n => n, StringComparer.Ordinal)
				.ToList();

			List<string> expected = new List<string>
			{
				"<Idx>k__BackingField",
				"<SortedDivisions>k__BackingField",
				"<TriVert>k__BackingField",
				"<VertBary>k__BackingField",
			};

			// Ordered comparison: TUnit's IsEquivalentTo defaults to CollectionOrdering.Any,
			// which would silently make this a set comparison.
			await Assert.That(fields).IsEquivalentTo(expected, CollectionOrdering.Matching)
				.Because(
					"Partition's fields changed. Add the new one to Partition.Clone (a deep copy "
					+ "for any collection) and to the list in this test — the Rust's derive(Clone) "
					+ "covers new fields automatically and the C# hand-written Clone does not.");
		}

		#endregion
	}
}
