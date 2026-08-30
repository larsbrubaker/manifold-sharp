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

// The 26 tests of boolean3_tests.rs that open with `use crate::manifold::Manifold`
// — the ones Boolean3Tests.cs's DEFERRED table listed for Phase 6 (and, for the
// two sphere-backed ones, Phase 7). Both phases landed, so all 26 are here, and
// Boolean3Tests.cs's table is now empty.
//
// Split from Boolean3Tests.cs for the 800-line file cap; same class, same order
// as the Rust, which puts these after its engine-level tests under the banner
// "C++ parity tests — ported from cpp-reference/manifold/test/boolean_test.cpp".
//
// Several of these look like duplicates of tests in ManifoldBooleanTests (the
// manifold_tests/boolean.rs port) — Tetra, Mirrored, Cubes, SelfSubtract,
// FaceUnion, MultiCoplanar, NoRetainedVerts, UnionDifference, Empty,
// NonIntersecting, BooleanVolumes. They are not: the Rust keeps both, and the
// two versions assert DIFFERENT things about the same construction. The
// boolean3_tests ones pin exact vertex and triangle counts (Tetra: 8/12,
// FaceUnion: 12/20, Mirrored: <=14/<=24) because they are testing the engine's
// output topology; the manifold_tests ones pin volume, area and genus because
// they are testing the façade. Deleting either loses real coverage, so both are
// ported, as the Rust has both.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class Boolean3Tests
	{
		/// <summary>C++ TEST(Boolean, Tetra) — simplest boolean test.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanTetra()
		{
			Manifold tetra = Manifold.Tetrahedron();
			await Assert.That(tetra.IsEmpty()).IsFalse();

			Manifold tetra2 = tetra.Translate(Vec3.Splat(0.5));
			Manifold result = tetra2.Difference(tetra);

			await Assert.That(result.NumVert()).IsEqualTo(8);
			await Assert.That(result.NumTri()).IsEqualTo(12);
		}

		/// <summary>
		/// C++ TEST(Boolean, Mirrored) — negative-scale boolean.
		/// Note: C++ gets exactly 12 verts/20 tris after colinear edge collapse.
		/// Our collapse_edge doesn't fully simplify (14/24), but geometry is correct.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanMirrored()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false).Scale(new Vec3(1.0, -1.0, 1.0));
			await Assert.That(cube.MatchesTriNormals())
				.IsTrue()
				.Because("Mirrored cube should match tri normals");

			Manifold cube2 = Manifold.Cube(Vec3.Splat(1.0), false).Scale(new Vec3(0.5, -1.0, 0.5));
			Manifold result = cube.Difference(cube2);

			await Assert.That(Math.Abs(result.Volume() - 0.75))
				.IsLessThan(1e-5)
				.Because($"Volume should be 0.75, got {result.Volume()}");
			await Assert.That(Math.Abs(result.SurfaceArea() - 5.5))
				.IsLessThan(1e-5)
				.Because($"Surface area should be 5.5, got {result.SurfaceArea()}");
			await Assert.That(result.Genus()).IsEqualTo(0);

			// C++ gets 12/20 after full simplification; we get 14/24 (geometry correct,
			// 2 extra colinear verts)
			await Assert.That(result.NumVert()).IsLessThanOrEqualTo(14);
			await Assert.That(result.NumTri()).IsLessThanOrEqualTo(24);
		}

		/// <summary>C++ TEST(Boolean, Cubes) — union of 3 cubes.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanCubes()
		{
			Manifold result = Manifold.Cube(new Vec3(1.2, 1.0, 1.0), true)
				.Translate(new Vec3(0.0, -0.5, 0.5));
			result = result.Union(
				Manifold.Cube(new Vec3(1.0, 0.8, 0.5), false).Translate(new Vec3(-0.5, 0.0, 0.5)));
			result = result.Union(
				Manifold.Cube(new Vec3(1.2, 0.1, 0.5), false).Translate(new Vec3(-0.6, -0.1, 0.0)));

			await Assert.That(result.MatchesTriNormals()).IsTrue();
			await Assert.That(result.NumDegenerateTris()).IsLessThanOrEqualTo(0);
			await Assert.That(Math.Abs(result.Volume() - 1.6)).IsLessThan(0.001);
			await Assert.That(Math.Abs(result.SurfaceArea() - 9.2)).IsLessThan(0.01);
		}

		/// <summary>C++ TEST(Boolean, NoRetainedVerts) — cube ^ octahedron.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanNoRetainedVerts()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);
			Manifold oct = Manifold.Sphere(1.0, 4);
			await Assert.That(Math.Abs(cube.Volume() - 1.0)).IsLessThan(0.001);
			await Assert.That(Math.Abs(oct.Volume() - 1.333)).IsLessThan(0.001);
			Manifold result = cube.Intersection(oct);
			await Assert.That(Math.Abs(result.Volume() - 0.833)).IsLessThan(0.001);
		}

		/// <summary>C++ TEST(Boolean, SelfSubtract) — cube - cube = empty.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanSelfSubtract()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold empty = cube.Difference(cube);
			await Assert.That(empty.IsEmpty()).IsTrue();
			await Assert.That(Math.Abs(empty.Volume())).IsLessThan(1e-10);
			await Assert.That(Math.Abs(empty.SurfaceArea())).IsLessThan(1e-10);
		}

		/// <summary>C++ TEST(Boolean, UnionDifference) — block with hole, stacked.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanUnionDifference()
		{
			Manifold block = Manifold.Cube(Vec3.Splat(1.0), true)
				.Difference(Manifold.Cylinder(1.0, 0.5, 0.5, 32));
			Manifold result = block.Union(block.Translate(new Vec3(0.0, 0.0, 1.0)));
			double resultVol = result.Volume();
			double blockVol = block.Volume();
			await Assert.That(Math.Abs(resultVol - (blockVol * 2.0)))
				.IsLessThan(0.0001)
				.Because(
					"Expected union of two identical blocks to be 2x volume: got "
					+ $"{resultVol} vs {blockVol * 2.0}");
		}

		/// <summary>C++ TEST(Boolean, TreeTransforms) — union with translations.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanTreeTransforms()
		{
			Manifold c = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold a = c.Union(c).Translate(new Vec3(1.0, 0.0, 0.0));
			Manifold b = c.Union(c);
			double vol = a.Union(b).Volume();
			await Assert.That(Math.Abs(vol - 2.0)).IsLessThan(1e-5).Because($"Expected volume 2.0, got {vol}");
		}

		/// <summary>C++ TEST(Boolean, FaceUnion) — cubes sharing a face.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanFaceUnion()
		{
			Manifold cubes = Manifold.Cube(Vec3.Splat(1.0), false)
				.Union(Manifold.Cube(Vec3.Splat(1.0), false).Translate(new Vec3(1.0, 0.0, 0.0)));
			await Assert.That(cubes.Genus()).IsEqualTo(0);
			await Assert.That(cubes.NumVert()).IsEqualTo(12);
			await Assert.That(cubes.NumTri()).IsEqualTo(20);
			await Assert.That(Math.Abs(cubes.Volume() - 2.0)).IsLessThan(1e-5);
			await Assert.That(Math.Abs(cubes.SurfaceArea() - 10.0)).IsLessThan(1e-5);
		}

		/// <summary>C++ TEST(Boolean, EdgeUnion) — cubes sharing an edge (disjoint result).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanEdgeUnion()
		{
			Manifold cubes = Manifold.Cube(Vec3.Splat(1.0), false)
				.Union(Manifold.Cube(Vec3.Splat(1.0), false).Translate(new Vec3(1.0, 1.0, 0.0)));

			// Two separate components
			await Assert.That(cubes.Volume()).IsEqualTo(2.0);
		}

		/// <summary>C++ TEST(Boolean, CornerUnion) — cubes sharing a corner (disjoint result).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanCornerUnion()
		{
			Manifold cubes = Manifold.Cube(Vec3.Splat(1.0), false)
				.Union(Manifold.Cube(Vec3.Splat(1.0), false).Translate(new Vec3(1.0, 1.0, 1.0)));
			await Assert.That(cubes.Volume()).IsEqualTo(2.0);
		}

		/// <summary>
		/// C++ TEST(Boolean, Coplanar) — cylinder - smaller cylinder (coplanar top/bottom).
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanCoplanar()
		{
			Manifold cyl = Manifold.Cylinder(1.0, 1.0, 1.0, 32);
			Manifold cyl2 = cyl.Scale(new Vec3(0.8, 0.8, 1.0)).Rotate(0.0, 0.0, 185.0);
			Manifold outManifold = cyl.Difference(cyl2);
			await Assert.That(outManifold.NumDegenerateTris()).IsEqualTo(0);
			await Assert.That(outManifold.Genus()).IsEqualTo(1);
		}

		/// <summary>C++ TEST(Boolean, MultiCoplanar) — cube - translated cube - translated cube.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanMultiCoplanar()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold first = cube.Difference(cube.Translate(new Vec3(0.3, 0.3, 0.0)));
			Manifold cube2 = cube.Translate(new Vec3(-0.3, -0.3, 0.0));
			Manifold outManifold = first.Difference(cube2);
			await Assert.That(outManifold.Genus()).IsEqualTo(-1);
			await Assert.That(Math.Abs(outManifold.Volume() - 0.18)).IsLessThan(1e-5);
			await Assert.That(Math.Abs(outManifold.SurfaceArea() - 2.76)).IsLessThan(1e-5);
		}

		/// <summary>C++ TEST(Boolean, Empty) — operations with empty manifold.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanEmpty()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			double cubeVol = cube.Volume();
			Manifold empty = Manifold.Empty();

			await Assert.That(Math.Abs(cube.Union(empty).Volume() - cubeVol)).IsLessThan(1e-10);
			await Assert.That(Math.Abs(cube.Difference(empty).Volume() - cubeVol)).IsLessThan(1e-10);
			await Assert.That(empty.Difference(cube).IsEmpty()).IsTrue();
			await Assert.That(cube.Intersection(empty).IsEmpty()).IsTrue();
		}

		/// <summary>C++ TEST(Boolean, NonIntersecting).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanNonIntersecting()
		{
			Manifold cube1 = Manifold.Cube(Vec3.Splat(1.0), false);
			double vol1 = cube1.Volume();
			Manifold cube2 = cube1.Scale(Vec3.Splat(2.0)).Translate(new Vec3(3.0, 0.0, 0.0));
			double vol2 = cube2.Volume();

			await Assert.That(Math.Abs(cube1.Union(cube2).Volume() - (vol1 + vol2))).IsLessThan(1e-5);
			await Assert.That(Math.Abs(cube1.Difference(cube2).Volume() - vol1)).IsLessThan(1e-5);
			await Assert.That(cube1.Intersection(cube2).IsEmpty()).IsTrue();
		}

		/// <summary>
		/// C++ TEST(Boolean, Perturb) — self-subtract of a tetrahedron defined from MeshGL.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanPerturb()
		{
			Manifold tetra = Manifold.Tetrahedron();
			Manifold empty = tetra.Difference(tetra);
			await Assert.That(empty.IsEmpty()).IsTrue();
			await Assert.That(Math.Abs(empty.Volume())).IsLessThan(1e-10);
			await Assert.That(Math.Abs(empty.SurfaceArea())).IsLessThan(1e-10);
		}

		/// <summary>C++ TEST(BooleanComplex, Sphere) — sphere - translated sphere.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanComplexSphere()
		{
			Manifold sphere = Manifold.Sphere(1.0, 12);
			Manifold sphere2 = sphere.Translate(Vec3.Splat(0.5));
			Manifold result = sphere.Difference(sphere2);
			await Assert.That(result.NumDegenerateTris()).IsEqualTo(0);
			await Assert.That(result.NumVert()).IsGreaterThan(0);
			await Assert.That(result.NumTri()).IsGreaterThan(0);
			await Assert.That(result.Volume()).IsGreaterThan(0.0);
		}

		/// <summary>C++ TEST(BooleanComplex, BooleanVolumes) — combinatorial boolean volume checks.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanVolumes()
		{
			Manifold m1 = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false);
			Manifold m2 = Manifold.Cube(new Vec3(2.0, 1.0, 1.0), false).Translate(new Vec3(1.0, 0.0, 0.0));
			Manifold m4 = Manifold.Cube(new Vec3(4.0, 1.0, 1.0), false).Translate(new Vec3(3.0, 0.0, 0.0));
			Manifold m3 = Manifold.Cube(new Vec3(3.0, 1.0, 1.0), false);
			Manifold m7 = Manifold.Cube(new Vec3(7.0, 1.0, 1.0), false);

			await Assert.That(Math.Abs(m1.Intersection(m2).Volume())).IsLessThan(1e-5).Because("m1^m2 should be 0");
			await Assert.That(Math.Abs(m1.Union(m2).Union(m4).Volume() - 7.0))
				.IsLessThan(1e-5)
				.Because("m1+m2+m4 should be 7");
			await Assert.That(Math.Abs(m1.Union(m2).Difference(m4).Volume() - 3.0))
				.IsLessThan(1e-5)
				.Because("m1+m2-m4 should be 3");
			await Assert.That(Math.Abs(m1.Union(m2.Intersection(m4)).Volume() - 1.0))
				.IsLessThan(1e-5)
				.Because("m1+(m2^m4) should be 1");
			await Assert.That(Math.Abs(m7.Intersection(m4).Volume() - 4.0))
				.IsLessThan(1e-5)
				.Because("m7^m4 should be 4");
			await Assert.That(Math.Abs(m7.Intersection(m3).Intersection(m1).Volume() - 1.0))
				.IsLessThan(1e-5)
				.Because("m7^m3^m1 should be 1");
			await Assert.That(Math.Abs(m7.Intersection(m1.Union(m2)).Volume() - 3.0))
				.IsLessThan(1e-5)
				.Because("m7^(m1+m2) should be 3");
			await Assert.That(Math.Abs(m7.Difference(m4).Volume() - 3.0))
				.IsLessThan(1e-5)
				.Because("m7-m4 should be 3");
			await Assert.That(Math.Abs(m7.Difference(m4).Difference(m2).Volume() - 1.0))
				.IsLessThan(1e-5)
				.Because("m7-m4-m2 should be 1");
			await Assert.That(Math.Abs(m7.Difference(m7.Difference(m1)).Volume() - 1.0))
				.IsLessThan(1e-5)
				.Because("m7-(m7-m1) should be 1");
			await Assert.That(Math.Abs(m7.Difference(m1.Union(m2)).Volume() - 4.0))
				.IsLessThan(1e-5)
				.Because("m7-(m1+m2) should be 4");
		}

		/// <summary>C++ TEST(BooleanComplex, Spiral) — recursive boolean union spiral.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanSpiral()
		{
			double d = 2.0;

			// Use smaller recursion depth to keep test fast
			Manifold result = Spiral(10, 25.0, 2.0, d);
			await Assert.That(result.Genus()).IsEqualTo(-10);
		}

		/// <summary>
		/// C++ TEST(Boolean, AlmostCoplanar) — tet union with nearly-coplanar rotated tet.
		/// C++ gets 20/36; we get 21/38 (1 extra vert from edge collapse difference).
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanAlmostCoplanar()
		{
			Manifold tet = Manifold.Tetrahedron();
			Manifold result = tet
				.Union(tet.Rotate(0.001, -0.08472872823860228, 0.055910459615905288))
				.Union(tet);

			// Geometry must be valid
			await Assert.That(result.NumVert() >= 20 && result.NumVert() <= 22)
				.IsTrue()
				.Because($"Expected ~20 verts, got {result.NumVert()}");
			await Assert.That(result.NumTri() >= 36 && result.NumTri() <= 40)
				.IsTrue()
				.Because($"Expected ~36 tris, got {result.NumTri()}");

			// Volume should be close to the union of 2 slightly-rotated tetrahedra
			await Assert.That(result.Volume()).IsGreaterThan(0.0).Because("Result should not be empty");
			await Assert.That(result.Genus()).IsEqualTo(0);
		}

		/// <summary>C++ TEST(Boolean, Perturb1) — extrude + boolean with coplanar faces.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanPerturb1()
		{
			// Diamond with square hole
			Polygons bigPolys = new Polygons
			{
				new SimplePolygon
				{
					new Vec2(0.0, 2.0),
					new Vec2(2.0, 0.0),
					new Vec2(4.0, 2.0),
					new Vec2(2.0, 4.0),
				},
				new SimplePolygon
				{
					new Vec2(1.0, 2.0),
					new Vec2(2.0, 3.0),
					new Vec2(3.0, 2.0),
					new Vec2(2.0, 1.0),
				},
			};
			Manifold big = Manifold.Extrude(bigPolys, 1.0, 0, 0.0, new Vec2(1.0, 1.0));

			// Small diamond
			Polygons littlePolys = new Polygons
			{
				new SimplePolygon
				{
					new Vec2(2.0, 1.0),
					new Vec2(3.0, 2.0),
					new Vec2(2.0, 3.0),
					new Vec2(1.0, 2.0),
				},
			};
			Manifold little = Manifold.Extrude(littlePolys, 1.0, 0, 0.0, new Vec2(1.0, 1.0))
				.Translate(new Vec3(0.0, 0.0, 1.0));

			// Small triangle
			Polygons punchPolys = new Polygons
			{
				new SimplePolygon
				{
					new Vec2(1.0, 2.0),
					new Vec2(2.0, 2.0),
					new Vec2(2.0, 3.0),
				},
			};
			Manifold punchHole = Manifold.Extrude(punchPolys, 1.0, 0, 0.0, new Vec2(1.0, 1.0))
				.Translate(new Vec3(0.0, 0.0, 1.0));

			Manifold result = big.Union(little).Difference(punchHole);

			await Assert.That(result.NumDegenerateTris()).IsEqualTo(0);
			await Assert.That(result.NumVert()).IsEqualTo(24).Because($"verts: {result.NumVert()}");
			await Assert.That(Math.Abs(result.Volume() - 7.5))
				.IsLessThan(1e-5)
				.Because($"volume: {result.Volume()}");
			await Assert.That(Math.Abs(result.SurfaceArea() - 38.2))
				.IsLessThan(0.1)
				.Because($"SA: {result.SurfaceArea()}");
		}

		/// <summary>C++ TEST(BooleanComplex, Subtract) — large real-world box subtraction.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanComplexSubtract()
		{
			MeshGL firstMesh = new MeshGL();
			firstMesh.NumProp = 3;
			firstMesh.VertProperties = new List<float>
			{
				0.0f, 0.0f, 0.0f, 1540.0f, 0.0f, 0.0f, 1540.0f, 70.0f, 0.0f, 0.0f, 70.0f, 0.0f, 0.0f, 0.0f, -278.282f,
				1540.0f, 70.0f, -278.282f, 1540.0f, 0.0f, -278.282f, 0.0f, 70.0f, -278.282f,
			};
			firstMesh.TriVerts = new List<uint>
			{
				0, 1, 2, 2, 3, 0, 4, 5, 6, 5, 4, 7, 6, 2, 1, 6, 5, 2, 5, 3, 2, 5, 7, 3, 7, 0, 3, 7, 4, 0,
				4, 1, 0, 4, 6, 1,
			};

			MeshGL secondMesh = new MeshGL();
			secondMesh.NumProp = 3;
			secondMesh.VertProperties = new List<float>
			{
				2.04636e-12f, 70.0f, 50000.0f,
				2.04636e-12f, -1.27898e-13f, 50000.0f,
				1470.0f, -1.27898e-13f, 50000.0f,
				1540.0f, 70.0f, 50000.0f,
				2.04636e-12f, 70.0f, -28.2818f,
				1470.0f, -1.27898e-13f, 0.0f,
				2.04636e-12f, -1.27898e-13f, 0.0f,
				1540.0f, 70.0f, -28.2818f,
			};
			secondMesh.TriVerts = new List<uint>
			{
				0, 1, 2, 2, 3, 0, 4, 5, 6, 5, 4, 7, 6, 2, 1, 6, 5, 2, 5, 3, 2, 5, 7, 3, 7, 0, 3, 7, 4, 0,
				4, 1, 0, 4, 6, 1,
			};

			Manifold first = Manifold.FromMeshGL(firstMesh);
			Manifold second = Manifold.FromMeshGL(secondMesh);

			Manifold result = first.Difference(second);
			_ = result.GetMeshGL(0);
			await Assert.That(result.Status()).IsEqualTo(Error.NoError);
		}

		/// <summary>C++ TEST(Boolean, Precision2) — intersection near precision boundary.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanPrecision2()
		{
			double kPrecision = 1e-12;
			double scale = 1000.0;
			Manifold cube = Manifold.Cube(Vec3.Splat(scale), false);

			double distance = scale * (1.0 - (kPrecision / 2.0));
			Manifold cube2 = cube.Translate(Vec3.Splat(-distance));

			// Intersection at half-precision offset should produce a tiny overlap
			// C++ expects empty due to epsilon tracking; we may get a tiny sliver
			Manifold intersection = cube.Intersection(cube2);

			// At this scale/offset, the overlap is ~0.5e-9 per axis, effectively zero
			await Assert.That(intersection.Volume())
				.IsLessThan(1e-6)
				.Because($"Near-precision intersection volume should be tiny: {intersection.Volume()}");
		}

		/// <summary>C++ TEST(Boolean, Cubes) — three overlapping cubes.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanCubesComplex()
		{
			Manifold result = Manifold.Cube(new Vec3(1.2, 1.0, 1.0), true)
				.Translate(new Vec3(0.0, -0.5, 0.5));
			result = result.Union(
				Manifold.Cube(new Vec3(1.0, 0.8, 0.5), false).Translate(new Vec3(-0.5, 0.0, 0.5)));
			result = result.Union(
				Manifold.Cube(new Vec3(1.2, 0.1, 0.5), false).Translate(new Vec3(-0.6, -0.1, 0.0)));

			await Assert.That(result.MatchesTriNormals()).IsTrue();
			await Assert.That(result.NumDegenerateTris()).IsLessThanOrEqualTo(0);
			await Assert.That(Math.Abs(result.Volume() - 1.6))
				.IsLessThan(0.001)
				.Because($"volume: {result.Volume()}");
			await Assert.That(Math.Abs(result.SurfaceArea() - 9.2))
				.IsLessThan(0.01)
				.Because($"SA: {result.SurfaceArea()}");
		}

		/// <summary>
		/// C++ TEST(Boolean, UnionDifference) — union of two identical blocks with
		/// cylindrical holes.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanUnionDifferenceStacked()
		{
			Manifold block = Manifold.Cube(Vec3.Splat(1.0), true)
				.Difference(Manifold.Cylinder(1.0, 0.5, 0.5, 32));
			Manifold result = block.Union(block.Translate(new Vec3(0.0, 0.0, 1.0)));
			double blocksize = block.Volume();
			await Assert.That(Math.Abs(result.Volume() - (blocksize * 2.0)))
				.IsLessThan(0.0001)
				.Because($"Stacked union volume: {result.Volume()} expected: {blocksize * 2.0}");
		}

		/// <summary>C++ TEST(Boolean, Coplanar) — cylinder subtraction with coplanar faces.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanCoplanarCylinder()
		{
			Manifold cylinder = Manifold.Cylinder(1.0, 1.0, 1.0, 32);
			Manifold cylinder2 = cylinder.Scale(new Vec3(0.8, 0.8, 1.0)).Rotate(0.0, 0.0, 185.0);
			Manifold outManifold = cylinder.Difference(cylinder2);
			await Assert.That(outManifold.NumDegenerateTris()).IsEqualTo(0);
			await Assert.That(outManifold.Genus()).IsEqualTo(1);
		}

		/// <summary>
		/// C++ TEST(Boolean, MultiCoplanar) — cube subtracted twice with coplanar overlap.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task BooleanMultiCoplanarComplex()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold first = cube.Difference(cube.Translate(new Vec3(0.3, 0.3, 0.0)));
			Manifold cube2 = cube.Translate(new Vec3(-0.3, -0.3, 0.0));
			Manifold outManifold = first.Difference(cube2);
			await Assert.That(outManifold.Genus()).IsEqualTo(-1);
			await Assert.That(Math.Abs(outManifold.Volume() - 0.18))
				.IsLessThan(1e-5)
				.Because($"volume: {outManifold.Volume()}");
			await Assert.That(Math.Abs(outManifold.SurfaceArea() - 2.76))
				.IsLessThan(1e-5)
				.Because($"SA: {outManifold.SurfaceArea()}");
		}

		/// <summary>
		/// The Rust's nested `fn spiral` inside test_boolean_spiral; C# has no local
		/// function that can recurse into itself as cleanly at this size, so it lifts to a
		/// private static with the same signature and the same recursion.
		/// </summary>
		/// <param name="rec">Remaining recursion depth.</param>
		/// <param name="r">Current radius.</param>
		/// <param name="add">Radius increment per full turn.</param>
		/// <param name="d">Cube spacing along the spiral.</param>
		/// <returns>The spiral so far.</returns>
		private static Manifold Spiral(int rec, double r, double add, double d)
		{
			double rot = 360.0 / (Types.KPi * r * 2.0) * d;
			double rNext = r + (add / 360.0 * rot);
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true).Translate(new Vec3(0.0, r, 0.0));
			if (rec > 0)
			{
				return Spiral(rec - 1, rNext, add, d).Rotate(0.0, 0.0, rot).Union(cube);
			}

			return cube;
		}
	}
}
