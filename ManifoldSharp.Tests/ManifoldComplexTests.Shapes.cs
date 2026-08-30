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

// Tests 9–13 of src/manifold_tests/complex.rs, split out of
// ManifoldComplexTests.cs for the 800-line cap: the ones built from primitives
// rather than from a file or a mesh literal — Cylinders, Close, LazyCollider,
// Perturb2, Perturb3.
//
// Perturb3 stays skipped with its reason string unchanged, even though it passes
// here (and in the Rust) when the [Skip] is removed — see the header of
// ManifoldComplexTests.cs for the verification and for why the ignore is kept.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldComplexTests
	{
		/// <summary>
		/// C++ TEST(BooleanComplex, Cylinders) — many cylinders with transforms.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppComplexCylinders()
		{
			Manifold rod = Manifold.Cylinder(1.0, 0.4, -1.0, 12);
			double[][] arrays1 =
			{
				new[] { 0.0, 0.0, 1.0, 3.0, -1.0, 0.0, 0.0, 3.0, 0.0, -1.0, 0.0, 6.0 },
				new[] { 0.0, 0.0, 1.0, 2.0, -1.0, 0.0, 0.0, 3.0, 0.0, -1.0, 0.0, 8.0 },
				new[] { 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 2.0, 0.0, -1.0, 0.0, 7.0 },
				new[] { 1.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 6.0 },
				new[] { 0.0, 0.0, 1.0, 3.0, -1.0, 0.0, 0.0, 3.0, 0.0, -1.0, 0.0, 7.0 },
				new[] { 0.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 3.0, 0.0, -1.0, 0.0, 7.0 },
				new[] { 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 4.0, 0.0, -1.0, 0.0, 6.0 },
				new[] { 1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 1.0, 4.0, 0.0, -1.0, 0.0, 6.0 },
			};
			double[][] arrays2 =
			{
				new[] { 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 2.0, 0.0, -1.0, 0.0, 6.0 },
				new[] { 1.0, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 1.0, 6.0 },
				new[] { 0.0, 0.0, 1.0, 2.0, -1.0, 0.0, 0.0, 2.0, 0.0, -1.0, 0.0, 7.0 },
				new[] { 1.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 1.0, 7.0 },
				new[] { 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 1.0, 7.0 },
				new[] { 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 1.0, 7.0 },
				new[] { 1.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 1.0, 7.0 },
				new[] { 1.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 1.0, 6.0 },
				new[] { 0.0, 0.0, 1.0, 3.0, -1.0, 0.0, 0.0, 4.0, 0.0, -1.0, 0.0, 6.0 },
			};

			// C++ layout: mat[i][j] = array[j * 4 + i]
			// Row 0: array[0..4], Row 1: array[4..8], Row 2: array[8..12]
			// Columns: x=(r00,r10,r20), y=(r01,r11,r21), z=(r02,r12,r22), w=(r03,r13,r23)
			static Mat3x4 MakeMat(double[] array) => Mat3x4.FromCols(
				new Vec3(array[0], array[4], array[8]),
				new Vec3(array[1], array[5], array[9]),
				new Vec3(array[2], array[6], array[10]),
				new Vec3(array[3], array[7], array[11]));

			Manifold m1 = Manifold.Empty();
			foreach (double[] array in arrays1)
			{
				Mat3x4 mat = MakeMat(array);
				m1 = m1.Union(rod.Transform(mat));
			}

			Manifold m2 = Manifold.Empty();
			foreach (double[] array in arrays2)
			{
				Mat3x4 mat = MakeMat(array);
				m2 = m2.Union(rod.Transform(mat));
			}

			m1 = m1.Union(m2);

			await Assert.That(m1.MatchesTriNormals())
				.IsTrue()
				.Because("Cylinders: should match tri normals");
			await Assert.That(m1.NumDegenerateTris() <= 12)
				.IsTrue()
				.Because($"Cylinders: {m1.NumDegenerateTris()} degenerate tris, expected <= 12");
		}

		/// <summary>
		/// C++ TEST(BooleanComplex, Close) — intersecting near-coincident spheres.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppComplexClose()
		{
			double r = 10.0;
			Manifold a = Manifold.Sphere(r, 256);
			Manifold result = a.Clone();
			for (int i = 0; i < 10; i++)
			{
				result = result.Intersection(
					a.Translate(new Vec3(a.GetTolerance() / 10.0 * i, 0.0, 0.0)));
			}

			double pi = Math.PI;
			double tol = 0.004;
			await Assert.That(Math.Abs(result.Volume() - (4.0 / 3.0 * pi * r * r * r)) < tol * r * r * r)
				.IsTrue()
				.Because(
					$"Close volume: {result.Volume()} expected ~{4.0 / 3.0 * pi * r * r * r}");
			await Assert.That(Math.Abs(result.SurfaceArea() - (4.0 * pi * r * r)) < tol * r * r)
				.IsTrue()
				.Because($"Close area: {result.SurfaceArea()} expected ~{4.0 * pi * r * r}");
		}

		/// <summary>
		/// C++ TEST(BooleanComplex, LazyCollider) — cylinder combos with mirror.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppComplexLazyCollider()
		{
			// C++ uses Cylinder(height, radius) with default segments (0 = auto)
			Manifold ele1 = Manifold.Cylinder(50.0, 50.0, -1.0, 0);
			Manifold ele2 = Manifold.Cylinder(60.0, 30.0, -1.0, 0);

			Manifold ele4 = ele1.Union(ele2).Mirror(new Vec3(0.0, 0.0, 1.0));
			await Assert.That(Math.Abs(ele4.Volume() - 418839.0) < 2.0)
				.IsTrue()
				.Because($"LazyCollider ele4 volume: {ele4.Volume()} expected ~418839");

			Manifold r1 = ele4.Difference(ele2.Translate(new Vec3(0.0, 0.0, -20.0)));
			await Assert.That(Math.Abs(r1.Volume() - 362577.0) < 2.0)
				.IsTrue()
				.Because($"LazyCollider r1 volume: {r1.Volume()} expected ~362577");

			Manifold ele3 = Manifold.Cylinder(60.0, 40.0, -1.0, 0).Mirror(new Vec3(0.0, 0.0, 1.0));
			Manifold r2 = ele4.Translate(new Vec3(0.0, 0.0, 1.0)).Difference(ele3);
			await Assert.That(Math.Abs(r2.Volume() - 145656.0) < 2.0)
				.IsTrue()
				.Because($"LazyCollider r2 volume: {r2.Volume()} expected ~145656");
		}

		/// <summary>
		/// C++ TEST(Boolean, Perturb2) — prism construction from cube triangles.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPerturb2()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(2.0), true);
			MeshGL cubeGl = cube.GetMeshGL(0);

			// Rotate so that nothing is axis-aligned
			Manifold result = cube.Rotate(5.0, 10.0, 15.0);

			int numTri = cubeGl.TriVerts.Count / 3;
			for (int tri = 0; tri < numTri; tri++)
			{
				List<float> prismVerts = new List<float>();
				List<uint> prismTris = new List<uint> { 4, 2, 0, 1, 3, 5 };

				for (int v0 = 0; v0 < 3; v0++)
				{
					int v1 = (v0 + 1) % 3;
					int vIn0 = (int)cubeGl.TriVerts[(3 * tri) + v0];
					int vIn1 = (int)cubeGl.TriVerts[(3 * tri) + v1];
					if (vIn1 > vIn0)
					{
						prismTris.AddRange(new uint[]
						{
							(uint)(2 * v0),
							(uint)(2 * v1),
							(uint)((2 * v1) + 1),
							(uint)(2 * v0),
							(uint)((2 * v1) + 1),
							(uint)((2 * v0) + 1),
						});
					}
					else
					{
						prismTris.AddRange(new uint[]
						{
							(uint)(2 * v0),
							(uint)(2 * v1),
							(uint)((2 * v0) + 1),
							(uint)(2 * v1),
							(uint)((2 * v1) + 1),
							(uint)((2 * v0) + 1),
						});
					}

					int np = (int)cubeGl.NumProp;
					for (int j = 0; j < 3; j++)
					{
						prismVerts.Add(cubeGl.VertProperties[(np * vIn0) + j]);
					}

					for (int j = 0; j < 3; j++)
					{
						prismVerts.Add(2.0f * cubeGl.VertProperties[(np * vIn0) + j]);
					}
				}

				MeshGL mesh = new MeshGL();
				mesh.NumProp = 3;
				mesh.VertProperties = prismVerts;
				mesh.TriVerts = prismTris;
				result = result + Manifold.FromMeshGL(mesh).Rotate(5.0, 10.0, 15.0);
			}

			await Assert.That(result.NumDegenerateTris()).IsEqualTo(0);
			await Assert.That(result.NumVert())
				.IsEqualTo(8)
				.Because($"Perturb2: {result.NumVert()} verts expected 8");
			await Assert.That(Math.Abs(result.Volume() - 64.0) < 1e-4)
				.IsTrue()
				.Because($"Perturb2 volume: {result.Volume()} expected 64");
			await Assert.That(Math.Abs(result.SurfaceArea() - 96.0) < 1e-4)
				.IsTrue()
				.Because($"Perturb2 area: {result.SurfaceArea()} expected 96");
		}

		/// <summary>C++ TEST(Boolean, Perturb3) — nasty gear pattern.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		[Skip("Gear pattern requires BatchBoolean precision improvements")]
		public async Task CppPerturb3()
		{
			int n = 16;
			double alpha = 90.0 / n;

			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);
			List<Manifold> outerCubes = new List<Manifold>();
			for (int i = 0; i < n; i++)
			{
				outerCubes.Add(cube.Rotate(0.0, 0.0, alpha * i));
			}

			Manifold gear = Manifold.BatchBoolean(outerCubes, OpType.Add);
			Manifold outerGear = gear.Scale(new Vec3(2.0, 2.0, 1.0));

			Manifold nastyGear = outerGear.Difference(gear);
			double expectedVolume = outerGear.Volume() - gear.Volume();

			await Assert.That(nastyGear.Status()).IsEqualTo(Error.NoError);
			await Assert.That(nastyGear.IsEmpty()).IsFalse();
			await Assert.That(nastyGear.Genus())
				.IsEqualTo(1)
				.Because($"Perturb3 genus: {nastyGear.Genus()} expected 1");
			await Assert.That(Math.Abs(nastyGear.Volume() - expectedVolume) < 1e-5)
				.IsTrue()
				.Because($"Perturb3 volume: {nastyGear.Volume()} expected {expectedVolume}");
			await Assert.That(Math.Abs(nastyGear.SurfaceArea() - 26.972) < 1e-3)
				.IsTrue()
				.Because($"Perturb3 area: {nastyGear.SurfaceArea()} expected 26.972");
		}
	}
}
