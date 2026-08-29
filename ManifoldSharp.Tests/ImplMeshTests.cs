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

// Port of impl_mesh_tests.rs — same 10 cases, same expected values, in the same
// order.
//
// These tests were first written while face_op was unported, so FromShape
// stopped one call short of the Rust's and primitives had no face normals. That
// gap is closed: FromShape now ends with FaceOp.SetNormalsAndCoplanar, as the
// Rust does, and this file has been re-run against it. Nothing asserted below
// reads FaceNormal or CoplanarId either way, so the expectations are the Rust's
// unchanged.
//
// IsEquivalentTo is passed CollectionOrdering.Matching explicitly: TUnit's
// default is CollectionOrdering.Any, which is order-INSENSITIVE and would turn a
// Rust `assert_eq!` on a Vec into a set comparison.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ImplMeshTests
	{
		[Test]
		public async Task NextHalfedge()
		{
			await Assert.That(ManifoldImpl.NextHalfedge(0)).IsEqualTo(1);
			await Assert.That(ManifoldImpl.NextHalfedge(1)).IsEqualTo(2);
			await Assert.That(ManifoldImpl.NextHalfedge(2)).IsEqualTo(0);
			await Assert.That(ManifoldImpl.NextHalfedge(3)).IsEqualTo(4);
			await Assert.That(ManifoldImpl.NextHalfedge(5)).IsEqualTo(3);
		}

		[Test]
		public async Task Tetrahedron()
		{
			ManifoldImpl m = ManifoldImpl.Tetrahedron(Mat3x4.Identity());
			await Assert.That(m.NumVert()).IsEqualTo(4);
			await Assert.That(m.NumTri()).IsEqualTo(4);
			await Assert.That(m.NumEdge()).IsEqualTo(6);
			await Assert.That(m.IsManifold()).IsTrue();
			await Assert.That(m.Is2Manifold()).IsTrue();
		}

		[Test]
		public async Task Cube()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			await Assert.That(m.NumVert()).IsEqualTo(8);
			await Assert.That(m.NumTri()).IsEqualTo(12);
			await Assert.That(m.NumEdge()).IsEqualTo(18);
			await Assert.That(m.IsManifold()).IsTrue();
			await Assert.That(m.Is2Manifold()).IsTrue();
		}

		[Test]
		public async Task Octahedron()
		{
			ManifoldImpl m = ManifoldImpl.Octahedron(Mat3x4.Identity());
			await Assert.That(m.NumVert()).IsEqualTo(6);
			await Assert.That(m.NumTri()).IsEqualTo(8);
			await Assert.That(m.NumEdge()).IsEqualTo(12);
			await Assert.That(m.IsManifold()).IsTrue();
			await Assert.That(m.Is2Manifold()).IsTrue();
		}

		[Test]
		public async Task CubeBbox()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			await Assert.That(Math.Abs(m.Bbox.Min.X - 0.0) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(m.Bbox.Min.Y - 0.0) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(m.Bbox.Min.Z - 0.0) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(m.Bbox.Max.X - 1.0) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(m.Bbox.Max.Y - 1.0) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(m.Bbox.Max.Z - 1.0) < 1e-10).IsTrue();
		}

		[Test]
		public async Task TetrahedronSymmetric()
		{
			ManifoldImpl m = ManifoldImpl.Tetrahedron(Mat3x4.Identity());

			// Tetrahedron is centered about origin with vertices at distance sqrt(3)
			foreach (Vec3 v in m.VertPos)
			{
				double dist2 = (v.X * v.X) + (v.Y * v.Y) + (v.Z * v.Z);
				await Assert.That(Math.Abs(dist2 - 3.0) < 1e-10)
					.IsTrue()
					.Because($"Expected dist^2=3, got {dist2}");
			}
		}

		[Test]
		public async Task CubeTransform()
		{
			Mat3x4 t = LinalgFunctions.Mat4ToMat3x4(
				LinalgFunctions.TranslationMatrix(new Vec3(1.0, 2.0, 3.0)));
			ManifoldImpl m = ManifoldImpl.Cube(t);
			await Assert.That(Math.Abs(m.Bbox.Min.X - 1.0) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(m.Bbox.Min.Y - 2.0) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(m.Bbox.Min.Z - 3.0) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(m.Bbox.Max.X - 2.0) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(m.Bbox.Max.Y - 3.0) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(m.Bbox.Max.Z - 4.0) < 1e-10).IsTrue();
		}

		[Test]
		public async Task ForVert()
		{
			ManifoldImpl m = ManifoldImpl.Tetrahedron(Mat3x4.Identity());

			// Each vertex in the tetrahedron is surrounded by 3 triangles = 3 halfedges
			int count = 0;
			m.ForVert(0, _ => count++);

			// ForVert visits one halfedge per triangle around the vertex (3 for tetrahedron)
			await Assert.That(count > 0).IsTrue();
		}

		[Test]
		public async Task CreateHalfedgesSimple()
		{
			// Simple triangle
			ManifoldImpl m = new ManifoldImpl();
			m.VertPos = new List<Vec3>
			{
				new Vec3(0.0, 0.0, 0.0),
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
			};

			// A tetrahedron has 4 triangles and 12 halfedges
			// Let's do a single triangle (not manifold, just for halfedge construction)
			IVec3[] tri = new IVec3[] { new IVec3(0, 1, 2) };
			m.CreateHalfedges(tri, Array.Empty<IVec3>());
			await Assert.That(m.Halfedge.Count).IsEqualTo(3);

			// Single triangle -- no pairs, so all pairedHalfedge should be -1
			// (no paired halfedges available)
		}

		/// <summary>
		/// MeshIdTransform mirrors C++ <c>std::map&lt;int, Relation&gt;</c>: iteration must
		/// be ordered by mesh ID. IncrementMeshIds relies on that order to assign fresh IDs
		/// ascending by old ID; with an unordered map the old-&gt;new mapping permutes with
		/// hasher state, which made 15 boolean tests fail only in full parallel suite runs.
		/// </summary>
		[Test]
		public async Task IncrementMeshIdsPreservesIdOrder()
		{
			ManifoldImpl m = ManifoldImpl.Tetrahedron(Mat3x4.Identity());
			m.MeshRelation.MeshIdTransform.Clear();
			foreach ((int id, int orig) in new (int, int)[] { (9, 900), (2, 200), (5, 500) })
			{
				Relation relation = new Relation();
				relation.OriginalId = orig;
				m.MeshRelation.MeshIdTransform.Add(id, relation);
			}

			m.MeshRelation.TriRef.Clear();
			m.MeshRelation.TriRef.Add(new TriRef(5, 500, 0, 0));
			m.MeshRelation.TriRef.Add(new TriRef(2, 200, 1, 1));
			m.MeshRelation.TriRef.Add(new TriRef(9, 900, 2, 2));
			m.MeshRelation.TriRef.Add(new TriRef(2, 200, 3, 3));
			m.IncrementMeshIds();

			List<int> newIds = new List<int>(m.MeshRelation.MeshIdTransform.Keys);
			await Assert.That(newIds.Count).IsEqualTo(3);
			await Assert.That(newIds[1]).IsEqualTo(newIds[0] + 1).Because("fresh IDs must be consecutive");
			await Assert.That(newIds[2]).IsEqualTo(newIds[0] + 2).Because("fresh IDs must be consecutive");

			List<int> originals = new List<int>();
			foreach (Relation r in m.MeshRelation.MeshIdTransform.Values)
			{
				originals.Add(r.OriginalId);
			}

			// CollectionOrdering.Matching is load-bearing here above all: this test exists
			// to pin that fresh IDs are handed out in ascending old-ID order, and the
			// default order-insensitive comparison would pass for any permutation.
			await Assert.That(originals)
				.IsEquivalentTo(new List<int> { 200, 500, 900 }, CollectionOrdering.Matching)
				.Because("fresh IDs must be assigned in ascending old-ID order (C++ std::map)");
			await Assert.That(m.MeshRelation.TriRef[1].MeshId).IsEqualTo(newIds[0]);
			await Assert.That(m.MeshRelation.TriRef[3].MeshId).IsEqualTo(newIds[0]);
			await Assert.That(m.MeshRelation.TriRef[0].MeshId).IsEqualTo(newIds[1]);
			await Assert.That(m.MeshRelation.TriRef[2].MeshId).IsEqualTo(newIds[2]);
		}
	}
}
