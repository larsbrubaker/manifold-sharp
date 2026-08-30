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

// Port of src/manifold_tests/api.rs — all 33 of its tests, same inputs, same
// expected values, in the same order. Two files, for the 800-line cap: this one
// carries the file-local fixtures (TetGl, CubeStl, CubeUv, CheckCube) and runs
// from the MinGap tests through the InvalidInput series;
// ManifoldApiTests.Relations.cs continues with Merge, the mesh-relation checks
// and the property tests.
//
// `TetGl` and `CubeUv` are `pub(super)` in the Rust — other modules in
// manifold_tests/ use them — so they are `internal static` here rather than
// private; ManifoldErrorPropagationTests.ErroredTet is the current caller of
// TetGl, exactly as error_propagation.rs calls `super::api::tet_gl()`.
//
// Nothing here is deferred. The four cases that once were —
// SmoothNormalTransform, SmoothFacetedNormals, MeshRelationRefine and
// MeshRelationRefinePrecision — landed with Manifold.Smooth.cs and live in
// ManifoldApiTests.Relations.cs in the Rust's order. The last two read the
// Csaszar fixture from ManifoldSmoothTests.CsaszarGl(), which is where
// api.rs reads it from too (`super::smooth::csaszar_gl()`).

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldApiTests
	{
		/// <summary>C++ TEST(Properties, MinGapCubeSphereOverlapping) — overlapping returns 0.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMinGapCubeSphereOverlapping()
		{
			Manifold a = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false);
			Manifold b = Manifold.Sphere(1.0, 0);
			double distance = a.MinGap(b, 0.1);
			await Assert.That(distance)
				.IsEqualTo(0.0)
				.Because($"MinGapCubeSphereOverlapping: {distance} expected 0");
		}

		/// <summary>C++ TEST(Properties, MinGapSphereSphereOutOfBounds) — returns search_length.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMinGapSphereSphereOutOfBounds()
		{
			Manifold a = Manifold.Sphere(1.0, 0);
			Manifold b = Manifold.Sphere(1.0, 0).Translate(new Vec3(2.0, 2.0, 0.0));
			double distance = a.MinGap(b, 0.8);
			await Assert.That(distance)
				.IsEqualTo(0.8)
				.Because($"MinGapSphereSphereOutOfBounds: {distance} expected 0.8 (search_length)");
		}

		/// <summary>C++ TEST(Properties, MingapAfterTransformations) — rotated/scaled spheres.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMinGapAfterTransformations()
		{
			Manifold a = Manifold.Sphere(1.0, 512).Rotate(30.0, 30.0, 30.0);
			Manifold b = Manifold.Sphere(1.0, 512)
				.Scale(new Vec3(3.0, 1.0, 1.0))
				.Rotate(0.0, 90.0, 45.0)
				.Translate(new Vec3(3.0, 0.0, 0.0));
			double distance = a.MinGap(b, 1.1);
			await Assert.That(Math.Abs(distance - 1.0))
				.IsLessThan(0.001)
				.Because($"MingapAfterTransformations: {distance} expected ~1.0");
		}

		/// <summary>C++ TEST(Manifold, ValidInputOneRunIndex) — empty mesh with runIndex={0}.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppValidInputOneRunIndex()
		{
			MeshGL emptyMesh = new MeshGL();
			emptyMesh.RunIndex = new List<uint> { 0 };
			Manifold empty = Manifold.FromMeshGL(emptyMesh);
			await Assert.That(empty.IsEmpty()).IsTrue().Because("ValidInputOneRunIndex: should be empty");
		}

		/// <summary>C++ TEST(Manifold, Empty) — default manifold is empty.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppManifoldEmpty()
		{
			Manifold empty = Manifold.Empty();
			await Assert.That(empty.IsEmpty()).IsTrue();
			await Assert.That(empty.NumVert()).IsEqualTo(0);
			await Assert.That(empty.NumTri()).IsEqualTo(0);
			await Assert.That(empty.Volume()).IsEqualTo(0.0);
		}

		/// <summary>C++ TEST(Manifold, Simplify) from manifold_test.cpp — simplify cube.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppManifoldSimplify()
		{
			Manifold cube = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false);
			Manifold simplified = cube.AsOriginal();
			await Assert.That(simplified.IsEmpty()).IsFalse().Because("Simplified cube should not be empty");
			await Assert.That(simplified.NumVert()).IsEqualTo(8);
			await Assert.That(simplified.NumTri()).IsEqualTo(12);
		}

		/// <summary>C++ TEST(Manifold, InvalidInput1) — NaN vertex.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidInput1()
		{
			MeshGL mesh = TetGl();
			mesh.VertProperties[(2 * 5) + 1] = float.NaN;
			Manifold tet = Manifold.FromMeshGL(mesh);
			await Assert.That(tet.IsEmpty()).IsTrue();
			await Assert.That(tet.Status()).IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, InvalidInput2) — swapped tri verts breaks manifold.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidInput2()
		{
			MeshGL mesh = TetGl();
			(mesh.TriVerts[(2 * 3) + 1], mesh.TriVerts[(2 * 3) + 2]) =
				(mesh.TriVerts[(2 * 3) + 2], mesh.TriVerts[(2 * 3) + 1]);
			Manifold tet = Manifold.FromMeshGL(mesh);
			await Assert.That(tet.IsEmpty()).IsTrue();
			await Assert.That(tet.Status()).IsEqualTo(Error.NotManifold);
		}

		/// <summary>C++ TEST(Manifold, InvalidInput3) — negative vertex index (wraps to huge).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidInput3()
		{
			MeshGL mesh = TetGl();

			// In C++, -2 as uint32_t = 0xFFFFFFFE
			for (int i = 0; i < mesh.TriVerts.Count; i++)
			{
				if (mesh.TriVerts[i] == 2)
				{
					mesh.TriVerts[i] = uint.MaxValue - 1;
				}
			}

			Manifold tet = Manifold.FromMeshGL(mesh);
			await Assert.That(tet.IsEmpty()).IsTrue();
			await Assert.That(tet.Status()).IsEqualTo(Error.VertexOutOfBounds);
		}

		/// <summary>C++ TEST(Manifold, InvalidInput4) — vertex index == numVert (out of bounds).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidInput4()
		{
			MeshGL mesh = TetGl();
			for (int i = 0; i < mesh.TriVerts.Count; i++)
			{
				if (mesh.TriVerts[i] == 2)
				{
					// 4 is out of range for TetGL's merged topology (4 unique verts)
					mesh.TriVerts[i] = 4;
				}
			}

			Manifold tet = Manifold.FromMeshGL(mesh);
			await Assert.That(tet.IsEmpty()).IsTrue();

			// C++ gets NotManifold because v=4 < numVert(7) but the merged topology breaks
			// Our Rust should also detect this
			await Assert.That(tet.Status() == Error.NotManifold || tet.Status() == Error.VertexOutOfBounds)
				.IsTrue()
				.Because($"Expected NotManifold or VertexOutOfBounds, got {tet.Status()}");
		}

		/// <summary>C++ TEST(Manifold, InvalidInput5) — merge index out of bounds.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidInput5()
		{
			MeshGL mesh = TetGl();
			mesh.MergeFromVert[mesh.MergeFromVert.Count - 1] = 7;
			Manifold tet = Manifold.FromMeshGL(mesh);
			await Assert.That(tet.IsEmpty()).IsTrue();
			await Assert.That(tet.Status()).IsEqualTo(Error.MergeIndexOutOfBounds);
		}

		/// <summary>C++ TEST(Manifold, InvalidInput6) — tri vert index out of bounds.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidInput6()
		{
			MeshGL mesh = TetGl();
			mesh.TriVerts[mesh.TriVerts.Count - 1] = 7;
			Manifold tet = Manifold.FromMeshGL(mesh);
			await Assert.That(tet.IsEmpty()).IsTrue();
			await Assert.That(tet.Status()).IsEqualTo(Error.VertexOutOfBounds);
		}

		/// <summary>C++ TEST(Manifold, InvalidInput7) — runIndex wrong length.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidInput7()
		{
			MeshGL cube = CubeUv();
			cube.RunIndex = new List<uint> { 0, 1, (uint)cube.TriVerts.Count };
			Manifold result = Manifold.FromMeshGL(cube);
			await Assert.That(result.IsEmpty()).IsTrue();
			await Assert.That(result.Status()).IsEqualTo(Error.RunIndexWrongLength);
		}

		/// <summary>C++ TEST(Manifold, ValidInput) — TetGL is valid.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppValidInput()
		{
			MeshGL mesh = TetGl();
			Manifold tet = Manifold.FromMeshGL(mesh);
			await Assert.That(tet.IsEmpty()).IsFalse().Because("TetGL should be valid");
			await Assert.That(tet.Status()).IsEqualTo(Error.NoError);
		}

		/// <summary>C++ TEST(Manifold, Invalid) — invalid constructor parameters.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidConstructors()
		{
			await Assert.That(Manifold.Sphere(0.0, 0).Status()).IsEqualTo(Error.InvalidConstruction);
			await Assert.That(Manifold.Cylinder(0.0, 5.0, -1.0, 0).Status()).IsEqualTo(Error.InvalidConstruction);
			await Assert.That(Manifold.Cylinder(2.0, -5.0, -1.0, 0).Status()).IsEqualTo(Error.InvalidConstruction);
			await Assert.That(Manifold.Cylinder(2.0, 0.0, -1.0, 0).Status()).IsEqualTo(Error.InvalidConstruction);
			await Assert.That(Manifold.Cylinder(2.0, 0.0, 0.0, 0).Status()).IsEqualTo(Error.InvalidConstruction);
			await Assert.That(Manifold.Cube(new Vec3(0.0, 0.0, 0.0), false).Status())
				.IsEqualTo(Error.InvalidConstruction);
			await Assert.That(Manifold.Cube(new Vec3(-1.0, 1.0, 1.0), false).Status())
				.IsEqualTo(Error.InvalidConstruction);
		}

		/// <summary>
		/// C++ TetGL() — tetrahedron with 5 properties per vert and merge vectors.
		/// </summary>
		/// <returns>The tetrahedron mesh.</returns>
		internal static MeshGL TetGl()
		{
			MeshGL tet = new MeshGL();
			tet.NumProp = 5;
			tet.VertProperties = new List<float>
			{
				-1.0f, -1.0f, 1.0f, 0.0f, 0.0f,
				-1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
				1.0f, -1.0f, -1.0f, 2.0f, -2.0f,
				1.0f, 1.0f, 1.0f, 3.0f, -3.0f,
				-1.0f, 1.0f, -1.0f, 4.0f, -4.0f,
				1.0f, -1.0f, -1.0f, 5.0f, -5.0f,
				1.0f, 1.0f, 1.0f, 6.0f, -6.0f,
			};
			tet.TriVerts = new List<uint> { 2, 0, 1, 0, 3, 1, 2, 3, 0, 6, 5, 4 };
			tet.MergeFromVert = new List<uint> { 4, 5, 6 };
			tet.MergeToVert = new List<uint> { 1, 2, 3 };
			return tet;
		}

		/// <summary>C++ CubeUV() — cube with UV coordinates.</summary>
		/// <returns>The UV cube mesh.</returns>
		internal static MeshGL CubeUv()
		{
			MeshGL mgl = new MeshGL();
			mgl.NumProp = 5;
			mgl.VertProperties = new List<float>
			{
				0.5f, -0.5f, 0.5f, 0.5f, 0.66f, -0.5f, -0.5f, 0.5f, 0.25f, 0.66f, 0.5f, 0.5f, 0.5f, 0.5f, 0.33f, -0.5f,
				0.5f, 0.5f, 0.25f, 0.33f, -0.5f, -0.5f, -0.5f, 1.0f, 0.66f, 0.5f, -0.5f, -0.5f, 0.75f, 0.66f, -0.5f, 0.5f,
				-0.5f, 1.0f, 0.33f, 0.5f, 0.5f, -0.5f, 0.75f, 0.33f, -0.5f, -0.5f, -0.5f, 0.0f, 0.66f, -0.5f, 0.5f, -0.5f,
				0.0f, 0.33f, -0.5f, 0.5f, -0.5f, 0.25f, 0.0f, 0.5f, 0.5f, -0.5f, 0.5f, 0.0f, -0.5f, -0.5f, -0.5f, 0.25f,
				1.0f, 0.5f, -0.5f, -0.5f, 0.5f, 1.0f,
			};
			mgl.TriVerts = new List<uint>
			{
				3, 1, 0, 3, 0, 2, 7, 5, 4, 7, 4, 6, 2, 0, 5, 2, 5, 7, 9, 8, 1, 9, 1, 3, 11, 10, 3, 11, 3,
				2, 0, 1, 12, 0, 12, 13,
			};
			mgl.MergeFromVert = new List<uint> { 8, 12, 13, 9, 10, 11 };
			mgl.MergeToVert = new List<uint> { 4, 4, 5, 6, 6, 7 };
			mgl.RunOriginalId.Add(ManifoldImpl.ReserveIds(1));
			return mgl;
		}

		/// <summary>
		/// C++ CubeSTL() — STL-style cube with face normals, no merge (requires Merge()).
		/// </summary>
		/// <remarks>
		/// api.rs declares its own `cube_stl` alongside the one in mod.rs. The two build
		/// the same geometry by different routes — this one normalizes with
		/// <c>linalg::normalize(cross(...))</c> on f64 and writes f32, mod.rs's works in
		/// raw arrays — so both are kept, as the Rust keeps both.
		/// </remarks>
		/// <returns>The STL-style cube mesh.</returns>
		private static MeshGL CubeStl()
		{
			MeshGL cubeIn = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), true).GetMeshGL(0);
			MeshGL cube = new MeshGL();
			cube.NumProp = 6;
			int numTri = cubeIn.NumTri();
			uint vertCount = 0;

			for (int tri = 0; tri < numTri; tri++)
			{
				float[][] triPos = new float[3][];
				for (int i = 0; i < 3; i++)
				{
					triPos[i] = new float[3];
					cube.TriVerts.Add(vertCount);
					vertCount += 1;
					int v = (int)cubeIn.TriVerts[(3 * tri) + i];
					for (int j = 0; j < 3; j++)
					{
						triPos[i][j] = cubeIn.VertProperties[((int)cubeIn.NumProp * v) + j];
					}
				}

				// Compute face normal
				Vec3 v0 = new Vec3(triPos[0][0], triPos[0][1], triPos[0][2]);
				Vec3 v1 = new Vec3(triPos[1][0], triPos[1][1], triPos[1][2]);
				Vec3 v2 = new Vec3(triPos[2][0], triPos[2][1], triPos[2][2]);
				Vec3 normal = LinalgFunctions.Normalize(LinalgFunctions.Cross(v1 - v0, v2 - v0));
				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						cube.VertProperties.Add(triPos[i][j]);
					}

					cube.VertProperties.Add((float)normal.X);
					cube.VertProperties.Add((float)normal.Y);
					cube.VertProperties.Add((float)normal.Z);
				}
			}

			cube.RunOriginalId.Add(ManifoldImpl.ReserveIds(1));
			return cube;
		}

		private static async Task CheckCube(MeshGL cubeStl)
		{
			Manifold raw = Manifold.FromMeshGL(cubeStl);
			await Assert.That(raw.IsEmpty())
				.IsFalse()
				.Because($"check_cube: raw is empty, status={raw.Status()}");
			Manifold cube = raw.AsOriginal();
			await Assert.That(cube.NumTri()).IsEqualTo(12).Because("check_cube: num_tri");
			await Assert.That(cube.NumVert())
				.IsEqualTo(8)
				.Because($"check_cube: num_vert (got {cube.NumVert()})");
			await Assert.That(cube.NumPropVert()).IsEqualTo(24);
			await Assert.That(Math.Abs(cube.Volume() - 1.0)).IsLessThan(1e-5).Because($"volume={cube.Volume()}");
			await Assert.That(Math.Abs(cube.SurfaceArea() - 6.0))
				.IsLessThan(1e-5)
				.Because($"sa={cube.SurfaceArea()}");
		}
	}
}
