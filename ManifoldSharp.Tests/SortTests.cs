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

// Port of the tests module in sort.rs — same 7 cases, same expected values, in
// the same order.
//
// IsEquivalentTo is passed CollectionOrdering.Matching explicitly: TUnit's
// default is CollectionOrdering.Any, which is order-INSENSITIVE and would turn a
// Rust `assert_eq!` on a Vec into a set comparison. Halfedge order is the whole
// subject of this module.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class SortTests
	{
		[Test]
		public async Task SpreadBits3()
		{
			// SpreadBits3(0b1111111111) should interleave into alternating positions
			// Values verified against C++ constexpr evaluation
			await Assert.That(Sort.SpreadBits3(0)).IsEqualTo(0u);
			await Assert.That(Sort.SpreadBits3(1)).IsEqualTo(1u);

			// Each bit of the input lands 3 positions apart in the output
			await Assert.That(Sort.SpreadBits3(0b10)).IsEqualTo(0b1000u);
			await Assert.That(Sort.SpreadBits3(0b11)).IsEqualTo(0b1001u);
			await Assert.That(Sort.SpreadBits3(0b100)).IsEqualTo(0b1000000u);
		}

		[Test]
		public async Task MortonCodeBasic()
		{
			Box bbox = new Box(new Vec3(0.0, 0.0, 0.0), new Vec3(1.0, 1.0, 1.0));

			// Origin → all zeros → code 0
			uint codeOrigin = Sort.MortonCode(new Vec3(0.0, 0.0, 0.0), bbox);
			await Assert.That(codeOrigin).IsEqualTo(0u);

			// NaN → K_NO_CODE. The Rust writes `f64::NAN`, which is the *positive* quiet
			// NaN; C#'s double.NaN has the sign bit set. Only IsNaN is consulted here, but
			// the port's rule is to use the Rust value wherever the Rust names f64::NAN.
			uint codeNan = Sort.MortonCode(new Vec3(DeterministicMath.PositiveQuietNaN, 0.0, 0.0), bbox);
			await Assert.That(codeNan).IsEqualTo(Sort.KNoCode);

			// Center of cube should be a positive code less than K_NO_CODE
			uint codeCenter = Sort.MortonCode(new Vec3(0.5, 0.5, 0.5), bbox);
			await Assert.That(codeCenter > 0 && codeCenter < Sort.KNoCode).IsTrue();
		}

		[Test]
		public async Task MortonCodeOrdering()
		{
			// Points closer together should (generally) have closer Morton codes.
			// More specifically: points sorted by Morton code produce a Z-curve traversal.
			Box bbox = new Box(new Vec3(0.0, 0.0, 0.0), new Vec3(8.0, 8.0, 8.0));
			uint p0 = Sort.MortonCode(new Vec3(0.0, 0.0, 0.0), bbox);
			uint p1 = Sort.MortonCode(new Vec3(1.0, 0.0, 0.0), bbox);
			uint p2 = Sort.MortonCode(new Vec3(2.0, 0.0, 0.0), bbox);

			// These should be strictly increasing along x with y=z=0
			await Assert.That(p0 < p1).IsTrue();
			await Assert.That(p1 < p2).IsTrue();
		}

		[Test]
		public async Task SortGeometryTetrahedron()
		{
			ManifoldImpl m = ManifoldImpl.Tetrahedron(Mat3x4.Identity());

			// Should have 4 vertices and 12 halfedges (4 triangles) before sort
			await Assert.That(m.VertPos.Count).IsEqualTo(4);
			await Assert.That(m.Halfedge.Count).IsEqualTo(12);
			Sort.SortGeometry(m);

			// Sort shouldn't remove any valid verts/faces
			await Assert.That(m.VertPos.Count).IsEqualTo(4);
			await Assert.That(m.Halfedge.Count).IsEqualTo(12);
		}

		[Test]
		public async Task SortGeometryCube()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			int vertCount = m.VertPos.Count;
			int halfedgeCount = m.Halfedge.Count;
			Sort.SortGeometry(m);
			await Assert.That(m.VertPos.Count).IsEqualTo(vertCount);
			await Assert.That(m.Halfedge.Count).IsEqualTo(halfedgeCount);

			// After sort, paired halfedges should still be valid
			for (int i = 0; i < m.Halfedge.Count; i++)
			{
				Halfedge edge = m.Halfedge[i];
				await Assert.That(edge.PairedHalfedge >= 0)
					.IsTrue()
					.Because($"halfedge {i} has invalid paired_halfedge {edge.PairedHalfedge}");
				Halfedge paired = m.Halfedge[edge.PairedHalfedge];
				await Assert.That(paired.PairedHalfedge)
					.IsEqualTo(i)
					.Because($"halfedge {i} paired -> {edge.PairedHalfedge} but paired doesn't point back");
			}
		}

		[Test]
		public async Task ReindexVertsIdentity()
		{
			ManifoldImpl m = ManifoldImpl.Tetrahedron(Mat3x4.Identity());
			int n = m.VertPos.Count;

			// Identity permutation should not change anything
			int[] identity = Enumerable.Range(0, n).ToArray();
			List<(int, int)> before = new List<(int, int)>();
			foreach (Halfedge e in m.Halfedge)
			{
				before.Add((e.StartVert, e.EndVert));
			}

			Sort.ReindexVerts(m, identity, n);
			List<(int, int)> after = new List<(int, int)>();
			foreach (Halfedge e in m.Halfedge)
			{
				after.Add((e.StartVert, e.EndVert));
			}

			await Assert.That(after).IsEquivalentTo(before, CollectionOrdering.Matching);
		}

		[Test]
		public async Task SortFacesManifoldPreserved()
		{
			// After sort_geometry, mesh must still be 2-manifold
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			Sort.SortGeometry(m);
			await Assert.That(m.Is2Manifold()).IsTrue();
		}
	}
}
