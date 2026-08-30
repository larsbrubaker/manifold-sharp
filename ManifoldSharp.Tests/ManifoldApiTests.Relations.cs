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

// The second half of the api.rs port, split from ManifoldApiTests.cs for the
// 800-line file cap. That file carries the module header, the DEFERRED table and
// the shared fixtures; this one continues the same class and the same order,
// from `test_cpp_merge` to the end of the Rust module.

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
		/// <summary>C++ TEST(Manifold, Merge) — STL cube needs Merge() to become valid.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMerge()
		{
			MeshGL cubeMesh = CubeStl();
			await Assert.That(cubeMesh.NumTri()).IsEqualTo(12);
			await Assert.That(cubeMesh.NumVert()).IsEqualTo(36);

			// Verify all vertex properties are finite
			for (int i = 0; i < cubeMesh.VertProperties.Count; i++)
			{
				float v = cubeMesh.VertProperties[i];
				await Assert.That(float.IsFinite(v)).IsTrue().Because($"vertex property {i} is not finite: {v}");
			}

			// Without merge, the STL-style cube is not manifold
			Manifold bad = Manifold.FromMeshGL(cubeMesh);
			await Assert.That(bad.IsEmpty())
				.IsTrue()
				.Because($"STL cube without merge should be empty, status: {bad.Status()}");

			// C++ returns NotManifold; we may get NonFiniteVertex if topology corruption
			// causes NaN during subsequent processing. Both indicate the mesh is invalid.
			await Assert.That(bad.Status() == Error.NotManifold || bad.Status() == Error.NonFiniteVertex)
				.IsTrue()
				.Because($"Expected NotManifold or NonFiniteVertex, got {bad.Status()}");

			// Merge should find coincident vertices
			await Assert.That(cubeMesh.Merge()).IsTrue().Because("merge() should return true");
			await Assert.That(cubeMesh.MergeFromVert.Count).IsEqualTo(28);
			await CheckCube(cubeMesh);

			// Second merge should return false (no new merges)
			await Assert.That(cubeMesh.Merge()).IsFalse();
			await Assert.That(cubeMesh.MergeFromVert.Count).IsEqualTo(28);

			// Truncate merge vectors and re-merge
			cubeMesh.MergeFromVert.RemoveRange(14, cubeMesh.MergeFromVert.Count - 14);
			cubeMesh.MergeToVert.RemoveRange(14, cubeMesh.MergeToVert.Count - 14);
			await Assert.That(cubeMesh.Merge()).IsTrue();
			await Assert.That(cubeMesh.MergeFromVert.Count).IsEqualTo(28);
			await CheckCube(cubeMesh);
		}

		/// <summary>C++ TEST(Manifold, MergeDegenerates).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMergeDegenerates()
		{
			MeshGL cube = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), true).GetMeshGL(0);
			MeshGL squash = new MeshGL();
			squash.NumProp = cube.NumProp;
			squash.VertProperties = new List<float>(cube.VertProperties);
			squash.TriVerts = new List<uint>(cube.TriVerts);

			// Move one vert to the position of its neighbor
			int len = squash.VertProperties.Count;
			squash.VertProperties[len - 1] *= -1.0f;

			// Remove one triangle to break manifold
			int triLen = squash.TriVerts.Count;
			squash.TriVerts.RemoveRange(triLen - 3, 3);

			// Rotate degenerate triangle to middle
			int n = squash.TriVerts.Count;
			if (n > 15)
			{
				// Rust `slice::rotate_left(15)`: the first 15 elements move to the end.
				List<uint> head = squash.TriVerts.GetRange(0, 15);
				squash.TriVerts.RemoveRange(0, 15);
				squash.TriVerts.AddRange(head);
			}

			// Merge should find the duplicate vertex
			await Assert.That(squash.Merge()).IsTrue();

			// Manifold should remove degenerate triangles
			Manifold squashed = Manifold.FromMeshGL(squash);
			await Assert.That(squashed.IsEmpty()).IsFalse().Because("Squashed cube should not be empty");
			await Assert.That(squashed.Status()).IsEqualTo(Error.NoError);
		}

		/// <summary>C++ TEST(Manifold, MergeEmpty) — shape that becomes empty after merge.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMergeEmpty()
		{
			MeshGL shape = new MeshGL();
			shape.NumProp = 7;
			shape.TriVerts = new List<uint>
			{
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
				25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
			};
			shape.VertProperties = new List<float>
			{
				0.0f, 0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 0.0f,
				0.0f, -0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 1.0f,
				0.0f, -0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 1.0f,
				0.0f, 0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 0.0f,
				0.0f, 0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 0.0f,
				0.0f, -0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 1.0f,
				-0.0f, 0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 0.0f,
				-0.0f, -0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 1.0f,
				-0.0f, -0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 1.0f,
				-0.0f, 0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 0.0f,
				-0.0f, -0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 1.0f,
				-0.0f, 0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 0.0f,
				0.0f, 0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 0.0f,
				-0.0f, 0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 0.0f,
				-0.0f, 0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 0.0f,
				0.0f, 0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 0.0f,
				-0.0f, 0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 0.0f,
				0.0f, 0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 0.0f,
				-0.0f, -0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 1.0f,
				0.0f, -0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 1.0f,
				0.0f, -0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 1.0f,
				-0.0f, -0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 1.0f,
				0.0f, -0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 1.0f,
				-0.0f, -0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 1.0f,
				0.0f, -0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 1.0f,
				0.0f, 0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 0.0f,
				-0.0f, 0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 0.0f,
				0.0f, -0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 1.0f,
				-0.0f, 0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 0.0f,
				-0.0f, -0.5f, 0.434500008821487f, 0.0f, 0.0f, 0.0f, 1.0f,
				0.0f, 0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 0.0f,
				0.0f, -0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 1.0f,
				-0.0f, -0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 1.0f,
				0.0f, 0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 0.0f,
				-0.0f, -0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 0.0f,
				-0.0f, 0.5f, -0.43450000882149f, 0.0f, 0.0f, 1.0f, 0.0f,
			};
			await Assert.That(shape.Merge()).IsTrue();
			Manifold man = Manifold.FromMeshGL(shape);
			await Assert.That(man.Status()).IsEqualTo(Error.NoError);
			await Assert.That(man.IsEmpty()).IsTrue();
		}

		/// <summary>C++ TEST(Manifold, MeshRelationTransform).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMeshRelationTransform()
		{
			Manifold cube = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false);
			MeshGL cubeGl = cube.GetMeshGL(0);
			Manifold turned = cube.Rotate(45.0, 90.0, 0.0);
			await ManifoldTestHelpers.RelatedGl(turned, new List<MeshGL> { cubeGl });
		}

		/// <summary>C++ TEST(Manifold, Decompose) — disjoint shapes can be decomposed.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppDecompose()
		{
			Manifold tet = Manifold.Tetrahedron();
			Manifold cube = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false)
				.Translate(new Vec3(2.0, 0.0, 0.0))
				.AsOriginal();
			Manifold sphere = Manifold.Sphere(1.0, 4)
				.Translate(new Vec3(4.0, 0.0, 0.0))
				.AsOriginal();

			Manifold combined = Manifold.BatchBoolean(new List<Manifold> { tet, cube, sphere }, OpType.Add);
			await Assert.That(combined.IsEmpty()).IsFalse();

			List<Manifold> parts = combined.Decompose();
			await Assert.That(parts.Count)
				.IsEqualTo(3)
				.Because($"Expected 3 decomposed parts, got {parts.Count}");

			// Sort by num_vert descending (matching C++ ExpectMeshes). Rust `sort_by` is
			// stable, so this is OrderByDescending/ThenByDescending, not List.Sort.
			parts = parts
				.OrderByDescending(m => m.NumVert())
				.ThenByDescending(m => m.NumTri())
				.ToList();

			await Assert.That(parts[0].NumVert()).IsEqualTo(8);
			await Assert.That(parts[0].NumTri()).IsEqualTo(12);
			await Assert.That(parts[1].NumVert()).IsEqualTo(6);
			await Assert.That(parts[1].NumTri()).IsEqualTo(8);
			await Assert.That(parts[2].NumVert()).IsEqualTo(4);
			await Assert.That(parts[2].NumTri()).IsEqualTo(4);
		}

		/// <summary>C++ TEST(Manifold, GetMeshGL) — round-trip through MeshGL preserves geometry.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppGetMeshGl()
		{
			Manifold manifold = Manifold.Sphere(0.01, 0);
			MeshGL meshOut = manifold.GetMeshGL(0);
			Manifold manifold2 = Manifold.FromMeshGL(meshOut);
			MeshGL meshOut2 = manifold2.GetMeshGL(0);

			// Check same number of vertices (by position)
			int n1 = meshOut.VertProperties.Count / (int)meshOut.NumProp;
			int n2 = meshOut2.VertProperties.Count / (int)meshOut2.NumProp;
			await Assert.That(n1).IsEqualTo(n2).Because($"Vertex count mismatch: {n1} vs {n2}");

			// Check vertex positions match
			for (int i = 0; i < n1; i++)
			{
				(float p1X, float p1Y, float p1Z) = meshOut.GetVertPos(i);
				(float p2X, float p2Y, float p2Z) = meshOut2.GetVertPos(i);
				double dist = Math.Sqrt(
					Math.Pow(p1X - p2X, 2) + Math.Pow(p1Y - p2Y, 2) + Math.Pow(p1Z - p2Z, 2));
				await Assert.That(dist).IsLessThanOrEqualTo(0.0001).Because($"Vertex {i} distance {dist} > 0.0001");
			}

			// Check same number of triangles
			await Assert.That(meshOut.TriVerts.Count)
				.IsEqualTo(meshOut2.TriVerts.Count)
				.Because("Triangle count mismatch");

			// Check triangle indices match (after sorting)
			List<(uint A, uint B, uint C)> tris1 = new List<(uint, uint, uint)>();
			for (int i = 0; i < meshOut.TriVerts.Count / 3; i++)
			{
				tris1.Add((meshOut.TriVerts[3 * i], meshOut.TriVerts[(3 * i) + 1], meshOut.TriVerts[(3 * i) + 2]));
			}

			List<(uint A, uint B, uint C)> tris2 = new List<(uint, uint, uint)>();
			for (int i = 0; i < meshOut2.TriVerts.Count / 3; i++)
			{
				tris2.Add((meshOut2.TriVerts[3 * i], meshOut2.TriVerts[(3 * i) + 1], meshOut2.TriVerts[(3 * i) + 2]));
			}

			tris1.Sort();
			tris2.Sort();
			await Assert.That(tris1)
				.IsEquivalentTo(tris2, CollectionOrdering.Matching)
				.Because("Triangle indices differ after round-trip");
		}

		/// <summary>C++ TEST(Manifold, WarpBatch) — Warp and WarpBatch produce identical results.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppWarpBatch()
		{
			Manifold cube = Manifold.Cube(new Vec3(2.0, 3.0, 4.0), false);
			int id = cube.OriginalId();

			Manifold shape1 = cube.Warp((ref Vec3 v) => { v.X += v.Z * v.Z; });
			Manifold shape2 = cube.WarpBatch(vecs =>
			{
				for (int i = 0; i < vecs.Length; i++)
				{
					vecs[i].X += vecs[i].Z * vecs[i].Z;
				}
			});

			await Assert.That(id).IsGreaterThanOrEqualTo(0);
			await Assert.That(shape1.OriginalId()).IsEqualTo(-1);
			await Assert.That(shape2.OriginalId()).IsEqualTo(-1);

			MeshGL gl1 = shape1.GetMeshGL(0);
			MeshGL gl2 = shape2.GetMeshGL(0);
			await Assert.That(gl1.RunOriginalId.Count).IsEqualTo(1);
			await Assert.That(gl1.RunOriginalId[0]).IsEqualTo((uint)id);
			await Assert.That(gl2.RunOriginalId.Count).IsEqualTo(1);
			await Assert.That(gl2.RunOriginalId[0]).IsEqualTo((uint)id);
			await Assert.That(Math.Abs(shape1.Volume() - shape2.Volume()))
				.IsLessThan(1e-10)
				.Because($"Warp vs WarpBatch volume: {shape1.Volume()} vs {shape2.Volume()}");
			await Assert.That(Math.Abs(shape1.SurfaceArea() - shape2.SurfaceArea()))
				.IsLessThan(1e-10)
				.Because($"Warp vs WarpBatch area: {shape1.SurfaceArea()} vs {shape2.SurfaceArea()}");
		}

		/// <summary>C++ TEST(Manifold, MeshDeterminism) — exact deterministic output from boolean.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMeshDeterminism()
		{
			Manifold cube1 = Manifold.Cube(new Vec3(2.0, 2.0, 2.0), true);
			Manifold cube2 = Manifold.Cube(new Vec3(2.0, 2.0, 2.0), true)
				.Translate(new Vec3(-1.1091, 0.88509, 1.3099));

			Manifold result = cube1 - cube2;
			MeshGL outMesh = result.GetMeshGL(0);

			List<uint> expectedTriVerts = new List<uint>
			{
				0, 2, 7, 0, 10, 1, 0, 6, 10, 0, 1, 2, 1, 3, 2, 1, 5, 3, 1, 11, 5, 0, 7, 6, 6, 7, 8, 6, 8,
				13, 10, 12, 11, 1, 10, 11, 11, 13, 5, 6, 12, 10, 6, 13, 12, 13, 9, 5, 13, 8, 9, 11, 12, 13,
				4, 2, 3, 4, 3, 5, 4, 7, 2, 4, 5, 8, 4, 8, 7, 9, 8, 5,
			};

			List<float> expectedVertProps = new List<float>
			{
				-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -0.11491f, 0.3099f, -1.0f, -0.11491f, 1.0f, -0.1091f,
				-0.11491f, 0.3099f, -0.1091f, -0.11491f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 0.3099f, -0.1091f, 1.0f,
				0.3099f, -0.1091f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
			};

			// The Rust folds both comparisons into one `flag` and asserts once, with the
			// four lengths in the message. Transcribed as written: an exact,
			// tolerance-free comparison of both arrays.
			bool flag = true;
			if (outMesh.TriVerts.Count == expectedTriVerts.Count)
			{
				for (int i = 0; i < outMesh.TriVerts.Count; i++)
				{
					if (outMesh.TriVerts[i] != expectedTriVerts[i])
					{
						flag = false;
						break;
					}
				}
			}
			else
			{
				flag = false;
			}

			if (flag && outMesh.VertProperties.Count == expectedVertProps.Count)
			{
				for (int i = 0; i < outMesh.VertProperties.Count; i++)
				{
					if (outMesh.VertProperties[i] != expectedVertProps[i])
					{
						flag = false;
						break;
					}
				}
			}
			else if (flag)
			{
				flag = false;
			}

			await Assert.That(flag)
				.IsTrue()
				.Because(
					"MeshDeterminism: output does not match expected.\n"
					+ $"  tri_verts len: {outMesh.TriVerts.Count} vs {expectedTriVerts.Count}\n"
					+ $"  vert_props len: {outMesh.VertProperties.Count} vs {expectedVertProps.Count}");
		}

		/// <summary>
		/// C++ TEST(Manifold, DecomposeProps) — decompose preserves properties across
		/// components.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppDecomposeProps()
		{
			// Create three shapes with position-derived "color" properties
			Manifold tet = Manifold.Tetrahedron()
				.SetProperties(3, (Span<double> newProp, Vec3 pos, ReadOnlySpan<double> old) =>
				{
					newProp[0] = pos.X;
					newProp[1] = pos.Y;
					newProp[2] = pos.Z;
				})
				.AsOriginal();
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false)
				.Translate(new Vec3(2.0, 0.0, 0.0))
				.AsOriginal()
				.SetProperties(3, (Span<double> newProp, Vec3 pos, ReadOnlySpan<double> old) =>
				{
					newProp[0] = pos.X;
					newProp[1] = pos.Y;
					newProp[2] = pos.Z;
				});
			Manifold sphere = Manifold.Sphere(1.0, 4)
				.Translate(new Vec3(4.0, 0.0, 0.0))
				.AsOriginal()
				.SetProperties(3, (Span<double> newProp, Vec3 pos, ReadOnlySpan<double> old) =>
				{
					newProp[0] = pos.X;
					newProp[1] = pos.Y;
					newProp[2] = pos.Z;
				});

			Manifold manifolds = Manifold.BatchBoolean(new List<Manifold> { tet, cube, sphere }, OpType.Add);

			// Check expected meshes: cube(8v,12t), sphere(6v,8t), tet(4v,4t)
			List<Manifold> parts = manifolds.Decompose();
			await Assert.That(parts.Count)
				.IsEqualTo(3)
				.Because($"DecomposeProps: expected 3 parts, got {parts.Count}");

			// Each part should have 3 extra properties
			for (int i = 0; i < parts.Count; i++)
			{
				await Assert.That(parts[i].NumProp())
					.IsEqualTo(3)
					.Because($"DecomposeProps: part {i} has {parts[i].NumProp()} props, expected 3");
			}
		}

		/// <summary>C++ TEST(CrossSection, Square) — cube from extruded square equals cube.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionSquare()
		{
			Manifold a = Manifold.Cube(new Vec3(5.0, 5.0, 5.0), false);
			Manifold b = Manifold.Extrude(
				CrossSection.Square(5.0).ToPolygons(),
				5.0,
				0,
				0.0,
				new Vec2(1.0, 1.0));
			await Assert.That(Math.Abs((a - b).Volume()))
				.IsLessThan(1e-5)
				.Because("Square: cube - extrude(square) should have 0 volume");
		}

		/// <summary>C++ TEST(CrossSection, Empty) — empty polygons yield empty cross section.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCrossSectionEmpty()
		{
			Polygons polys = new Polygons { new SimplePolygon(), new SimplePolygon() };
			CrossSection e = new CrossSection(polys);
			await Assert.That(e.IsEmpty()).IsTrue().Because("Empty cross section should be empty");
		}

		/// <summary>C++ TEST(Properties, CalculateCurvature) — sphere curvature.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesCalculateCurvature()
		{
			double precision = 0.015;
			int gaussianIdx = 3;
			int meanIdx = 4;

			Manifold sphere = Manifold.Sphere(1.0, 64).CalculateCurvature(gaussianIdx - 3, meanIdx - 3);
			MeshGL gl = sphere.GetMeshGL(0);
			await Assert.That(gl.NumProp)
				.IsEqualTo(5u)
				.Because("Should have 5 properties (3 pos + gaussian + mean)");

			// Mean curvature of unit sphere = 2 (1/r1 + 1/r2 = 1+1 = 2)
			(float minMean, float maxMean) = GetMinMaxProperty(gl, meanIdx);
			await Assert.That(Math.Abs(minMean - 2.0f))
				.IsLessThan(2.0f * (float)precision)
				.Because($"min mean curvature: {minMean}");
			await Assert.That(Math.Abs(maxMean - 2.0f))
				.IsLessThan(2.0f * (float)precision)
				.Because($"max mean curvature: {maxMean}");

			// Gaussian curvature of unit sphere = 1
			(float minGauss, float maxGauss) = GetMinMaxProperty(gl, gaussianIdx);
			await Assert.That(Math.Abs(minGauss - 1.0f))
				.IsLessThan((float)precision)
				.Because($"min gaussian curvature: {minGauss}");
			await Assert.That(Math.Abs(maxGauss - 1.0f))
				.IsLessThan((float)precision)
				.Because($"max gaussian curvature: {maxGauss}");

			// Scaled sphere (radius 2): mean = 1, gaussian = 0.25
			Manifold sphere2 = sphere.Scale(Vec3.Splat(2.0)).CalculateCurvature(gaussianIdx - 3, meanIdx - 3);
			MeshGL gl2 = sphere2.GetMeshGL(0);
			await Assert.That(gl2.NumProp).IsEqualTo(5u);
			(float minMean2, float maxMean2) = GetMinMaxProperty(gl2, meanIdx);
			await Assert.That(Math.Abs(minMean2 - 1.0f))
				.IsLessThan((float)precision)
				.Because($"scaled min mean: {minMean2}");
			await Assert.That(Math.Abs(maxMean2 - 1.0f))
				.IsLessThan((float)precision)
				.Because($"scaled max mean: {maxMean2}");
			(float minGauss2, float maxGauss2) = GetMinMaxProperty(gl2, gaussianIdx);
			await Assert.That(Math.Abs(minGauss2 - 0.25f))
				.IsLessThan(0.25f * (float)precision)
				.Because($"scaled min gauss: {minGauss2}");
			await Assert.That(Math.Abs(maxGauss2 - 0.25f))
				.IsLessThan(0.25f * (float)precision)
				.Because($"scaled max gauss: {maxGauss2}");
		}

		/// <summary>C++ TEST(Smooth, NormalTransform) — smooth by normals after rotation.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothNormalTransform()
		{
			Manifold cube1 = Manifold.Cube(Vec3.Splat(1.0), false)
				.Rotate(30.0, 0.0, 0.0)
				.CalculateNormals(0, 60.0);
			Manifold cube2 = Manifold.Cube(Vec3.Splat(1.0), false)
				.CalculateNormals(0, 60.0)
				.Rotate(30.0, 0.0, 0.0)
				.Translate(new Vec3(3.0, 0.0, 0.0));

			Manifold combo = cube1 + cube2;
			Manifold out1 = combo.SmoothByNormals(0).Refine(10);
			await Assert.That(Math.Abs(out1.Volume() - 2.0))
				.IsLessThan(1e-4)
				.Because($"volume={out1.Volume()}");
			await Assert.That(Math.Abs(out1.SurfaceArea() - 12.0))
				.IsLessThan(1e-4)
				.Because($"sa={out1.SurfaceArea()}");

			cube1 = Manifold.Cube(Vec3.Splat(1.0), false)
				.Rotate(30.0, 0.0, 0.0)
				.CalculateNormals(0, 60.0);
			cube2 = Manifold.Cube(Vec3.Splat(1.0), false)
				.CalculateNormals(0, 60.0)
				.Rotate(30.0, 0.0, 0.0)
				.Translate(new Vec3(3.0, 0.0, 0.0));
			combo = cube1 + cube2;
			Manifold out2 = Manifold.FromMeshGL(combo.GetMeshGL(0))
				.SmoothByNormals(0)
				.Refine(10);
			await Assert.That(Math.Abs(out2.Volume() - 2.0))
				.IsLessThan(1e-4)
				.Because($"volume2={out2.Volume()}");
			await Assert.That(Math.Abs(out2.SurfaceArea() - 12.0))
				.IsLessThan(1e-4)
				.Because($"sa2={out2.SurfaceArea()}");
		}

		/// <summary>C++ TEST(Smooth, FacetedNormals) — faceted smooth preserves geometry.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothFacetedNormals()
		{
			Manifold cylinder = Manifold.Cylinder(10.0, 10.0, -1.0, 0);
			Manifold faceted = cylinder
				.CalculateNormals(0, 0.0)
				.SmoothByNormals(0)
				.RefineToLength(0.1);
			await Assert.That(faceted.Status()).IsEqualTo(Error.NoError);
			await Assert.That(Math.Abs(cylinder.Volume() - faceted.Volume()))
				.IsLessThan(0.01)
				.Because($"FacetedNormals: volume {cylinder.Volume()} vs {faceted.Volume()}");
			await Assert.That(Math.Abs(cylinder.SurfaceArea() - faceted.SurfaceArea()))
				.IsLessThan(0.01)
				.Because($"FacetedNormals: area {cylinder.SurfaceArea()} vs {faceted.SurfaceArea()}");
		}

		/// <summary>
		/// C++ TEST(Manifold, MeshID) — two Manifolds constructed from the same MeshGL must
		/// receive different run_original_ids (each import reserves its own ID).
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMeshId()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			MeshGL cubeGl = cube.GetMeshGL(0);
			cubeGl.RunIndex.Clear();
			cubeGl.RunOriginalId.Clear();
			Manifold cube1 = Manifold.FromMeshGL(cubeGl);
			Manifold cube2 = Manifold.FromMeshGL(cubeGl);
			uint id1 = cube1.GetMeshGL(0).RunOriginalId[0];
			uint id2 = cube2.GetMeshGL(0).RunOriginalId[0];
			await Assert.That(id1)
				.IsNotEqualTo(id2)
				.Because("MeshID: two imports of same MeshGL should have different run_original_ids");
		}

		/// <summary>
		/// C++ TEST(Manifold, MeshGLRoundTrip) — cylinder MeshGL round-trip preserves
		/// run_original_id, and RelatedGL validates that vertex positions trace to source
		/// triangles.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppManifoldMeshGlRoundTrip2()
		{
			Manifold cylinder = Manifold.Cylinder(2.0, 1.0, -1.0, 0);
			await Assert.That(cylinder.OriginalId())
				.IsGreaterThanOrEqualTo(0)
				.Because("MeshGLRoundTrip: cylinder should have original_id >= 0");
			MeshGL inGl = cylinder.GetMeshGL(0);
			Manifold cylinder2 = Manifold.FromMeshGL(inGl);
			MeshGL outGl = cylinder2.GetMeshGL(0);

			await Assert.That(inGl.RunOriginalId.Count)
				.IsEqualTo(1)
				.Because("MeshGLRoundTrip: inGL should have 1 run");
			await Assert.That(outGl.RunOriginalId.Count)
				.IsEqualTo(1)
				.Because("MeshGLRoundTrip: outGL should have 1 run");
			await Assert.That(outGl.RunOriginalId[0])
				.IsEqualTo(inGl.RunOriginalId[0])
				.Because("MeshGLRoundTrip: run_original_id should be preserved");

			await ManifoldTestHelpers.RelatedGl(cylinder2, new List<MeshGL> { inGl });
		}

		/// <summary>
		/// C++ TEST(Manifold, MeshRelationRefine) — refine-to-length preserves mesh relation.
		/// </summary>
		/// <remarks>
		/// Shares its name with a test in smooth.rs and is not a duplicate of it: this one
		/// takes its counts off <c>Decompose()</c>, and asserts the component count too. The
		/// Rust keeps both; so does this port. The Csaszar fixture belongs to smooth.rs,
		/// which is why it is called through ManifoldSmoothTests here — api.rs writes
		/// <c>super::smooth::csaszar_gl()</c> for the same reason.
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMeshRelationRefine()
		{
			Manifold csaszarSrc = Manifold.FromMeshGL(ManifoldSmoothTests.CsaszarGl());
			Manifold csaszar = ManifoldTestHelpers.WithPositionColors(csaszarSrc).AsOriginal();
			MeshGL inGl = csaszar.GetMeshGL(0);

			await ManifoldTestHelpers.RelatedGl(csaszar, new List<MeshGL> { inGl });

			Manifold refined = csaszar.RefineToLength(1.0);
			await Assert.That(refined.IsEmpty()).IsFalse().Because("MeshRelationRefine: refined not empty");
			await Assert.That(refined.MatchesTriNormals())
				.IsTrue()
				.Because("MeshRelationRefine: matches_tri_normals");
			List<Manifold> parts = refined.Decompose();
			await Assert.That(parts.Count).IsEqualTo(1).Because("MeshRelationRefine: 1 component");
			await Assert.That(parts[0].NumVert()).IsEqualTo(9019).Because("MeshRelationRefine: num_vert");
			await Assert.That(parts[0].NumTri()).IsEqualTo(18038).Because("MeshRelationRefine: num_tri");
			await Assert.That(parts[0].NumProp()).IsEqualTo(3).Because("MeshRelationRefine: num_prop");

			await ManifoldTestHelpers.RelatedGl(refined, new List<MeshGL> { inGl });
		}

		/// <summary>
		/// C++ TEST(Manifold, MeshRelationRefinePrecision) — smooth + refine-to-tolerance
		/// preserves run_original_id.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMeshRelationRefinePrecision()
		{
			MeshGL inGl = ManifoldTestHelpers
				.WithPositionColors(Manifold.FromMeshGL(ManifoldSmoothTests.CsaszarGl()))
				.GetMeshGL(0);
			uint id = inGl.RunOriginalId[0];
			Manifold csaszar = Manifold.Smooth(inGl, Array.Empty<Smoothness>());
			Manifold refined = csaszar.RefineToTolerance(0.05);
			await Assert.That(refined.IsEmpty())
				.IsFalse()
				.Because("MeshRelationRefinePrecision: not empty");
			List<Manifold> parts = refined.Decompose();
			await Assert.That(parts.Count).IsEqualTo(1).Because("MeshRelationRefinePrecision: 1 component");

			// C++ v3.5.0 expects {{2343, 4686, 3}} after the #1724/#1671 smoothing fixes.
			await Assert.That(parts[0].NumVert())
				.IsEqualTo(2343)
				.Because("MeshRelationRefinePrecision: num_vert");
			await Assert.That(parts[0].NumTri())
				.IsEqualTo(4686)
				.Because("MeshRelationRefinePrecision: num_tri");
			await Assert.That(parts[0].NumProp())
				.IsEqualTo(3)
				.Because("MeshRelationRefinePrecision: num_prop");

			List<uint> runIds = refined.GetMeshGL(0).RunOriginalId;
			await Assert.That(runIds.Count).IsEqualTo(1).Because("MeshRelationRefinePrecision: 1 run");
			await Assert.That(runIds[0])
				.IsEqualTo(id)
				.Because("MeshRelationRefinePrecision: original_id preserved");
		}

		private static (float Min, float Max) GetMinMaxProperty(MeshGL gl, int channel)
		{
			int numProp = (int)gl.NumProp;
			float minVal = float.MaxValue;
			float maxVal = float.MinValue;
			int numVert = gl.VertProperties.Count / numProp;
			for (int i = 0; i < numVert; i++)
			{
				float v = gl.VertProperties[(i * numProp) + channel];
				if (v < minVal)
				{
					minVal = v;
				}

				if (v > maxVal)
				{
					maxVal = v;
				}
			}

			return (minVal, maxVal);
		}
	}
}
