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

// Tests 11–19 of src/manifold_tests/validation.rs, split out of
// ManifoldValidationTests.cs for the 800-line cap: the Rust's two banner
// sections "C++ Boolean tests: edge/corner/coplanar/perturb" and "C++ Manifold
// constructor & geometry tests".
//
// The Rust's two "already ported in advanced.rs" notes (TreeTransforms,
// CornerUnion) are kept as comments where they sit, so the file still reads
// against the Rust line for line — and TreeTransforms IS additionally ported at
// the bottom of validation.rs, so it lands in
// ManifoldValidationTests.Regressions.cs like the Rust has it.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldValidationTests
	{
		// TreeTransforms already ported in advanced.rs

		/// <summary>C++ TEST(Boolean, Perturb) — self-subtraction of tetrahedron.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPerturb()
		{
			MeshGL tmp = new MeshGL();
			tmp.NumProp = 3;
			tmp.VertProperties = new List<float>
			{
				0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
			};
			tmp.TriVerts = new List<uint> { 2, 0, 1, 0, 3, 1, 2, 3, 0, 3, 2, 1 };
			Manifold corner = Manifold.FromMeshGL(tmp);
			await Assert.That(corner.IsEmpty()).IsFalse().Because("corner tet should be valid");

			Manifold empty = corner.Clone() - corner;
			await Assert.That(empty.IsEmpty()).IsTrue().Because("self-subtraction should be empty");
			await Assert.That(Math.Abs(empty.Volume()) < 1e-10)
				.IsTrue()
				.Because($"volume={empty.Volume()}");
			await Assert.That(Math.Abs(empty.SurfaceArea()) < 1e-10)
				.IsTrue()
				.Because($"sa={empty.SurfaceArea()}");
		}

		/// <summary>
		/// C++ TEST(Boolean, EdgeUnion) — cubes touching at edge remain separate.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppEdgeUnion()
		{
			Manifold cubes = Manifold.Cube(Vec3.Splat(1.0), false);
			cubes = cubes + Manifold.Cube(Vec3.Splat(1.0), false)
				.Translate(new Vec3(1.0, 1.0, 0.0));

			// Two separate cubes touching at edge — should remain 2 components
			await Assert.That(cubes.NumVert()).IsEqualTo(16).Because("EdgeUnion: 2×8 verts");
			await Assert.That(cubes.NumTri()).IsEqualTo(24).Because("EdgeUnion: 2×12 tris");
			await Assert.That(Math.Abs(cubes.Volume() - 2.0) < 1e-5).IsTrue();
		}

		// CornerUnion already ported in advanced.rs

		/// <summary>
		/// C++ TEST(Boolean, AlmostCoplanar) — union of nearly-coplanar tetrahedra.
		/// </summary>
		/// <remarks>
		/// Passes since <c>Manifold::rotate</c> was ported to the C++ matrix construction
		/// (bit-exact rotation matrices via a local remquo-sind) and <c>face2tri</c> was
		/// rewritten to the C++ v3.5.0 halfedge-index pairing scheme (robust to the
		/// exactly-coplanar faces the corrected rotation produces).
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppAlmostCoplanar()
		{
			Manifold tet = Manifold.Tetrahedron();
			Manifold result = tet.Clone()
				+ tet.Clone().Rotate(0.001, -0.08472872823860228, 0.055910459615905288)
				+ tet;
			await Assert.That(result.NumVert())
				.IsEqualTo(20)
				.Because($"AlmostCoplanar: expected 20 verts, got {result.NumVert()}");
			await Assert.That(result.NumTri())
				.IsEqualTo(36)
				.Because($"AlmostCoplanar: expected 36 tris, got {result.NumTri()}");
		}

		/// <summary>
		/// C++ TEST(Boolean, Coplanar) — cylinder difference with coplanar faces.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCoplanar()
		{
			Manifold cylinder = Manifold.Cylinder(1.0, 1.0, -1.0, 0);
			Manifold cylinder2 = cylinder.Clone()
				.Scale(new Vec3(0.8, 0.8, 1.0))
				.Rotate(0.0, 0.0, 185.0);
			Manifold outManifold = cylinder - cylinder2;
			await Assert.That(outManifold.Status()).IsEqualTo(Error.NoError);
			await Assert.That(outManifold.Genus())
				.IsEqualTo(1)
				.Because($"Coplanar: genus should be 1, got {outManifold.Genus()}");
			await Assert.That(outManifold.NumDegenerateTris())
				.IsEqualTo(0)
				.Because("No degenerate tris");
		}

		/// <summary>
		/// C++ TEST(Boolean, MeshGLRoundTrip) — boolean result round-trips through MeshGL.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppBooleanMeshglRoundTrip()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(2.0), false);
			await Assert.That(cube.OriginalId() >= 0).IsTrue();
			MeshGL original = cube.GetMeshGL(0);

			Manifold result = cube.Clone() + cube.Translate(new Vec3(1.0, 1.0, 0.0));
			await Assert.That(result.OriginalId() < 0).IsTrue();
			await Assert.That(result.NumVert())
				.IsEqualTo(18)
				.Because($"BoolMeshGL: expected 18 verts, got {result.NumVert()}");
			await Assert.That(result.NumTri())
				.IsEqualTo(32)
				.Because($"BoolMeshGL: expected 32 tris, got {result.NumTri()}");
			await ManifoldTestHelpers.RelatedGl(result, new List<MeshGL> { original });

			MeshGL inGl = result.GetMeshGL(0);
			await Assert.That(inGl.RunOriginalId.Count).IsEqualTo(2);

			Manifold result2 = Manifold.FromMeshGL(inGl);
			await Assert.That(result2.OriginalId() < 0).IsTrue();
			await Assert.That(result2.NumVert())
				.IsEqualTo(18)
				.Because($"BoolMeshGL rt: expected 18 verts, got {result2.NumVert()}");
			await Assert.That(result2.NumTri())
				.IsEqualTo(32)
				.Because($"BoolMeshGL rt: expected 32 tris, got {result2.NumTri()}");
			await ManifoldTestHelpers.RelatedGl(result2, new List<MeshGL> { original });

			MeshGL outGl = result2.GetMeshGL(0);
			await Assert.That(outGl.RunOriginalId.Count).IsEqualTo(2);
		}

		// ====================================================================
		// C++ Manifold constructor & geometry tests
		// ====================================================================

		/// <summary>C++ TEST(Manifold, Sphere) — sphere triangle count with n=25.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSphereTriCountN25()
		{
			int n = 25;
			Manifold sphere = Manifold.Sphere(1.0, 4 * n);
			await Assert.That(sphere.NumTri())
				.IsEqualTo(n * n * 8)
				.Because($"Sphere tri count: expected {n * n * 8}, got {sphere.NumTri()}");
		}

		// MeshID test already ported in advanced.rs as test_cpp_manifold_mesh_id

		/// <summary>C++ TEST(Manifold, Slice) — slice cube at z=0 and z=1.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSlice()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			CrossSection bottom = cube.Slice(0.0);
			CrossSection top = cube.Slice(1.0);
			await Assert.That(Math.Abs(bottom.Area() - 1.0) < 1e-10)
				.IsTrue()
				.Because($"Slice at z=0 area={bottom.Area()}, expected 1.0");
			await Assert.That(Math.Abs(top.Area()) < 1e-10)
				.IsTrue()
				.Because($"Slice at z=1 area={top.Area()}, expected 0.0");
		}

		/// <summary>
		/// C++ TEST(Manifold, SliceEmptyObject) — slice empty manifold doesn't crash.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSliceEmptyObject()
		{
			Manifold empty = Manifold.Empty();
			await Assert.That(empty.IsEmpty()).IsTrue();
			CrossSection bottom = empty.Slice(0.0);
			_ = bottom;

			// Just verify it doesn't crash
		}

		/// <summary>C++ TEST(Manifold, Project) — project 3D mesh to 2D cross-section.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppProject()
		{
			MeshGL input = new MeshGL();
			input.NumProp = 3;
			input.VertProperties = new List<float>
			{
				0.0f, 0.0f, 0.0f, -2.0f, -0.7f, -0.1f, -2.0f, -0.7f, 0.0f, -1.9f, -0.7f, -0.1f,
				-1.9f, -0.6901f, -0.1f, -1.9f, -0.7f, 0.0f, -1.9f, -0.6901f, 0.0f, -2.0f, -1.0f,
				3.0f, -1.9f, -1.0f, 3.0f, -2.0f, -1.0f, 4.0f, -1.9f, -1.0f, 4.0f, -1.9f,
				-0.6901f, 3.0f, -1.9f, -0.6901f, 4.0f, -1.7f, -0.6901f, 3.0f, -1.7f, -0.6901f,
				3.2f, -2.0f, 0.0f, -0.1f, -2.0f, 0.0f, 0.0f, -2.0f, 0.0f, 3.0f, -2.0f, 0.0f,
				4.0f, -1.7f, 0.0f, 3.0f, -1.7f, 0.0f, 3.2f, -1.0f, -0.6901f, -0.1f, -1.0f,
				-0.6901f, 0.0f, -1.0f, -0.6901f, 3.2f, -1.0f, -0.6901f, 4.0f, -1.0f, 0.0f,
				-0.1f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 3.2f, -1.0f, 0.0f, 4.0f,
			};
			input.TriVerts = new List<uint>
			{
				1, 3, 2, 1, 4, 3, 2, 3, 5, 5, 6, 2, 3, 4, 6, 5, 3, 6, 6, 4, 21, 26, 22, 25, 21,
				25, 22, 25, 15, 26, 26, 6, 22, 21, 4, 25, 21, 22, 6, 16, 26, 15, 16, 6, 26, 4,
				15, 25, 15, 1, 16, 16, 2, 6, 4, 1, 15, 1, 2, 16, 12, 14, 23, 12, 13, 14, 12, 11,
				13, 18, 9, 12, 11, 7, 17, 7, 9, 18, 17, 7, 18, 13, 11, 19, 17, 18, 20, 19, 11,
				17, 19, 17, 20, 14, 13, 20, 18, 12, 24, 20, 13, 19, 20, 18, 27, 12, 10, 11, 24,
				12, 23, 9, 10, 12, 9, 8, 10, 8, 11, 10, 8, 7, 11, 8, 9, 7, 14, 20, 27, 24, 28,
				18, 27, 18, 28, 23, 14, 27, 24, 23, 28, 28, 23, 27,
			};
			Manifold m = Manifold.FromMeshGL(input);
			CrossSection projected = m.Project();
			await Assert.That(Math.Abs(projected.Area() - 0.72) < 0.01)
				.IsTrue()
				.Because($"Project area={projected.Area()}, expected 0.72");
		}
	}
}
