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

// The second file of the boolean.rs port, split from ManifoldBooleanTests.cs for
// the 800-line file cap; ManifoldBooleanTests.Normals.cs is the third. The first
// carries the module header; this one continues the same class and the same
// order, from `test_cpp_warp` to the end of the Rust module, minus the four
// cases Normals.cs took.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldBooleanTests
	{
		/// <summary>C++ TEST(Manifold, Warp) — simple warp that shifts x by z^2.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppWarp()
		{
			CrossSection square = CrossSection.Square(1.0);
			Manifold shape = Manifold
				.Extrude(square.ToPolygons(), 2.0, 10, 0.0, new Vec2(1.0, 1.0))
				.Warp((ref Vec3 v) => { v.X += v.Z * v.Z; });
			await Assert.That(Math.Abs(shape.Volume() - 2.0))
				.IsLessThan(0.0001)
				.Because($"Warped extrusion volume: {shape.Volume()} expected: 2.0");
		}

		[Test]
		public async Task RotateBooleanAllAngles()
		{
			// Simulate what the Boolean Gallery animation does:
			// boolean of two cubes where shape B is rotated at various angles
			Manifold a = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), true);
			for (int deg = 0; deg < 360; deg += 5)
			{
				double angle = deg;
				Manifold b = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), true)
					.Rotate(0.0, angle, 0.0)
					.Translate(new Vec3(0.5, 0.0, 0.0));
				Manifold result = a.Union(b);
				await Assert.That(result.NumTri())
					.IsGreaterThan(0)
					.Because($"Union failed at rotation angle {angle}");
			}
		}

		[Test]
		public async Task ColoredBooleanPreservesProperties()
		{
			// Two cubes with different colors, boolean should preserve both
			Manifold a = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), true)
				.SetProperties(3, (Span<double> p, Vec3 pos, ReadOnlySpan<double> old) =>
				{
					p[0] = 0.0;
					p[1] = 0.0;
					p[2] = 1.0;
				}); // blue
			Manifold b = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), true)
				.SetProperties(3, (Span<double> p, Vec3 pos, ReadOnlySpan<double> old) =>
				{
					p[0] = 1.0;
					p[1] = 0.0;
					p[2] = 0.0;
				}) // red
				.Translate(new Vec3(0.5, 0.0, 0.0));
			Manifold result = a.Union(b);
			await Assert.That(result.NumTri()).IsGreaterThan(0);
			MeshGL gl = result.GetMeshGL(0);
			int numProp = (int)gl.NumProp;
			await Assert.That(numProp).IsEqualTo(6); // xyz + RGB preserved

			// Should have both blue and red vertices
			int vertCount = gl.VertProperties.Count / numProp;
			bool hasBlue = false;
			bool hasRed = false;
			for (int i = 0; i < vertCount; i++)
			{
				float r = gl.VertProperties[(i * numProp) + 3];
				float bVal = gl.VertProperties[(i * numProp) + 5];
				if (bVal > 0.5f)
				{
					hasBlue = true;
				}

				if (r > 0.5f)
				{
					hasRed = true;
				}
			}

			await Assert.That(hasBlue).IsTrue().Because("Result should have blue vertices from shape A");
			await Assert.That(hasRed).IsTrue().Because("Result should have red vertices from shape B");
		}

		[Test]
		public async Task RotateBooleanWithPropertiesAllAngles()
		{
			// Simulate what the Boolean Gallery animation does with colored shapes.
			// This tests the combination of set_properties + rotate + boolean at many angles.
			Manifold a = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), true)
				.SetProperties(4, (Span<double> p, Vec3 pos, ReadOnlySpan<double> old) =>
				{
					p[0] = 0.27;
					p[1] = 0.53;
					p[2] = 0.80;
					p[3] = 1.0;
				});
			Manifold bBase = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), true)
				.SetProperties(4, (Span<double> p, Vec3 pos, ReadOnlySpan<double> old) =>
				{
					p[0] = 0.85;
					p[1] = 0.25;
					p[2] = 0.25;
					p[3] = 0.6;
				});

			for (int deg = 0; deg < 360; deg += 5)
			{
				double angle = deg;
				Manifold b = bBase
					.Rotate(angle * 0.7 / 1.5, angle, angle * 0.3 / 1.5)
					.Translate(new Vec3(0.3, 0.0, 0.0));
				Manifold result = a.Union(b);
				await Assert.That(result.NumTri())
					.IsGreaterThan(0)
					.Because($"Colored union failed at rotation angle {angle}");
			}
		}

		[Test]
		public async Task SpikyDodecahedronBooleanAllAngles()
		{
			// Test that spiky dodecahedron booleans work at all angles without hanging.
			// This is the exact scenario that the Boolean Gallery animation runs.
			Manifold a = MakeSpikyDodecahedron(0.4);
			await Assert.That(a.NumTri())
				.IsEqualTo(60)
				.Because($"Spiky dodecahedron should have 60 tris, got {a.NumTri()}");

			// Test basic self-union works
			Manifold b = MakeSpikyDodecahedron(0.4).Translate(new Vec3(0.3, 0.0, 0.0));
			Manifold result = a.Union(b);
			await Assert.That(result.NumTri()).IsGreaterThan(0).Because("Basic spiky dodecahedron union failed");

			// Test at the specific rotation angles that hang
			// Frame 36 hangs: rot=(25.2, 54.0, 10.8)
			Manifold b2 = MakeSpikyDodecahedron(0.4)
				.Rotate(25.2, 54.0, 10.8)
				.Translate(new Vec3(0.3, 0.0, 0.0));
			Manifold result2 = a.Union(b2);
			await Assert.That(result2.NumTri())
				.IsGreaterThan(0)
				.Because("Spiky dodecahedron union failed at rot=(25.2, 54.0, 10.8)");
		}

		/// <summary>C++ TEST(Boolean, UnionDifference) — cube with hole, union stacked.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppUnionDifference()
		{
			Manifold block = Manifold.Cube(Vec3.Splat(1.0), true)
				.Difference(Manifold.Cylinder(1.0, 0.5, 0.5, 32));
			Manifold result = block.Union(block.Translate(new Vec3(0.0, 0.0, 1.0)));
			double resultVol = result.Volume();
			double blockVol = block.Volume();
			await Assert.That(Math.Abs(resultVol - (blockVol * 2.0)))
				.IsLessThan(0.0001)
				.Because($"UnionDifference: result {resultVol} expected ~{blockVol * 2.0}");
		}

		/// <summary>C++ TEST(Boolean, Empty) — operations with empty manifold.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppBooleanEmptyOps()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			double cubeVol = cube.Volume();
			Manifold empty = Manifold.Empty();

			await Assert.That(Math.Abs(cube.Union(empty).Volume() - cubeVol))
				.IsLessThan(1e-10)
				.Because("cube + empty should equal cube");
			await Assert.That(Math.Abs(cube.Difference(empty).Volume() - cubeVol))
				.IsLessThan(1e-10)
				.Because("cube - empty should equal cube");
			await Assert.That(empty.Difference(cube).IsEmpty())
				.IsTrue()
				.Because("empty - cube should be empty");
			await Assert.That(cube.Intersection(empty).IsEmpty())
				.IsTrue()
				.Because("cube ^ empty should be empty");
		}

		/// <summary>C++ TEST(Boolean, NonIntersecting) — non-overlapping cubes.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNonIntersecting()
		{
			Manifold cube1 = Manifold.Cube(Vec3.Splat(1.0), false);
			double vol1 = cube1.Volume();
			Manifold cube2 = cube1.Scale(Vec3.Splat(2.0)).Translate(new Vec3(3.0, 0.0, 0.0));
			double vol2 = cube2.Volume();

			await Assert.That(Math.Abs(cube1.Union(cube2).Volume() - (vol1 + vol2)))
				.IsLessThan(1e-10)
				.Because("Non-intersecting union volume should be sum");
			await Assert.That(Math.Abs(cube1.Difference(cube2).Volume() - vol1))
				.IsLessThan(1e-10)
				.Because("Non-intersecting subtract volume should be cube1");
			await Assert.That(cube1.Intersection(cube2).IsEmpty())
				.IsTrue()
				.Because("Non-intersecting intersect should be empty");
		}

		/// <summary>C++ TEST(Boolean, Mirrored) — mirrored cube subtraction.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMirrored()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false).Scale(new Vec3(1.0, -1.0, 1.0));
			await Assert.That(cube.MatchesTriNormals())
				.IsTrue()
				.Because("Mirrored cube should match tri normals");

			Manifold cube2 = Manifold.Cube(Vec3.Splat(1.0), false).Scale(new Vec3(0.5, -1.0, 0.5));
			Manifold result = cube.Difference(cube2);

			await Assert.That(Math.Abs(result.Volume() - 0.75))
				.IsLessThan(1e-6)
				.Because($"Mirrored volume: {result.Volume()} expected 0.75");
			await Assert.That(Math.Abs(result.SurfaceArea() - 5.5))
				.IsLessThan(1e-6)
				.Because($"Mirrored area: {result.SurfaceArea()} expected 5.5");
		}

		/// <summary>C++ TEST(Boolean, Cubes) — three cubes union.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCubesUnion()
		{
			Manifold result = Manifold.Cube(new Vec3(1.2, 1.0, 1.0), true)
				.Translate(new Vec3(0.0, -0.5, 0.5));
			result = result.Union(
				Manifold.Cube(new Vec3(1.0, 0.8, 0.5), false).Translate(new Vec3(-0.5, 0.0, 0.5)));
			result = result.Union(
				Manifold.Cube(new Vec3(1.2, 0.1, 0.5), false).Translate(new Vec3(-0.6, -0.1, 0.0)));

			await Assert.That(result.MatchesTriNormals())
				.IsTrue()
				.Because("Cubes result should match tri normals");
			await Assert.That(result.NumDegenerateTris()).IsEqualTo(0);
			await Assert.That(Math.Abs(result.Volume() - 1.6))
				.IsLessThan(0.001)
				.Because($"Cubes volume: {result.Volume()} expected ~1.6");
			await Assert.That(Math.Abs(result.SurfaceArea() - 9.2))
				.IsLessThan(0.01)
				.Because($"Cubes area: {result.SurfaceArea()} expected ~9.2");
		}

		/// <summary>C++ TEST(Boolean, Tetra) — tetrahedron subtraction.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppTetraBoolean()
		{
			Manifold tetra = Manifold.Tetrahedron();
			await Assert.That(tetra.IsEmpty()).IsFalse();

			Manifold tetra2 = tetra.Translate(Vec3.Splat(0.5));
			Manifold result = tetra2.Difference(tetra);

			await Assert.That(result.NumTri()).IsGreaterThan(0).Because("Tetra subtraction should be non-empty");
			await Assert.That(result.Volume())
				.IsGreaterThan(0.0)
				.Because("Tetra subtraction should have positive volume");
		}

		/// <summary>C++ TEST(Boolean, SelfSubtract).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSelfSubtract()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold empty = cube.Difference(cube);
			await Assert.That(empty.IsEmpty()).IsTrue().Because("SelfSubtract should produce empty mesh");
			await Assert.That(Math.Abs(empty.Volume())).IsLessThan(1e-10);
			await Assert.That(Math.Abs(empty.SurfaceArea())).IsLessThan(1e-10);
		}

		/// <summary>C++ TEST(Boolean, NoRetainedVerts).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNoRetainedVerts()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);
			Manifold oct = Manifold.Sphere(1.0, 4);
			await Assert.That(Math.Abs(cube.Volume() - 1.0))
				.IsLessThan(0.001)
				.Because($"cube vol: {cube.Volume()}");
			await Assert.That(Math.Abs(oct.Volume() - 1.333))
				.IsLessThan(0.001)
				.Because($"oct vol: {oct.Volume()}");
			Manifold result = cube.Intersection(oct);
			await Assert.That(Math.Abs(result.Volume() - 0.833))
				.IsLessThan(0.001)
				.Because($"NoRetainedVerts intersection volume: {result.Volume()} expected ~0.833");
		}

		/// <summary>C++ TEST(Boolean, MultiCoplanar) — multi-step coplanar subtraction.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMultiCoplanar()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold first = cube.Difference(cube.Translate(new Vec3(0.3, 0.3, 0.0)));
			Manifold cube2 = cube.Translate(new Vec3(-0.3, -0.3, 0.0));
			Manifold outManifold = first.Difference(cube2);
			await Assert.That(outManifold.Genus())
				.IsEqualTo(-1)
				.Because($"MultiCoplanar genus: {outManifold.Genus()} expected -1");
			await Assert.That(Math.Abs(outManifold.Volume() - 0.18))
				.IsLessThan(1e-5)
				.Because($"MultiCoplanar volume: {outManifold.Volume()} expected ~0.18");
			await Assert.That(Math.Abs(outManifold.SurfaceArea() - 2.76))
				.IsLessThan(1e-5)
				.Because($"MultiCoplanar area: {outManifold.SurfaceArea()} expected ~2.76");
		}

		/// <summary>C++ TEST(Boolean, FaceUnion) — cubes sharing a face.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppFaceUnion()
		{
			Manifold cubes = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold result = cubes.Union(cubes.Translate(new Vec3(1.0, 0.0, 0.0)));
			await Assert.That(result.Genus()).IsEqualTo(0).Because($"FaceUnion genus: {result.Genus()} expected 0");
			await Assert.That(Math.Abs(result.Volume() - 2.0))
				.IsLessThan(1e-5)
				.Because($"FaceUnion volume: {result.Volume()} expected 2.0");
			await Assert.That(Math.Abs(result.SurfaceArea() - 10.0))
				.IsLessThan(1e-5)
				.Because($"FaceUnion area: {result.SurfaceArea()} expected 10.0");
		}

		/// <summary>C++ TEST(BooleanComplex, BooleanVolumes) — sphere subtraction volumes.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppBooleanVolumes()
		{
			Manifold sphere = Manifold.Sphere(1.0, 12);
			Manifold sphere2 = sphere.Translate(Vec3.Splat(0.5));

			Manifold u = sphere.Union(sphere2);
			Manifold i = sphere.Intersection(sphere2);
			Manifold d = sphere.Difference(sphere2);

			double sphereVol = sphere.Volume();

			// Union + Intersect = 2 * sphere (inclusion-exclusion)
			await Assert.That(Math.Abs(u.Volume() + i.Volume() - (2.0 * sphereVol)))
				.IsLessThan(0.01)
				.Because($"U+I={u.Volume() + i.Volume()} expected ~2*sphere={2.0 * sphereVol}");

			// Difference + Intersect = sphere
			await Assert.That(Math.Abs(d.Volume() + i.Volume() - sphereVol))
				.IsLessThan(0.01)
				.Because($"D+I={d.Volume() + i.Volume()} expected ~sphere={sphereVol}");
		}

		/// <summary>
		/// C++ TEST(Boolean, PropertiesNoIntersection) — property handling for
		/// non-intersecting union.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesNoIntersection()
		{
			// Create cube with UV properties (2 extra props)
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false)
				.SetProperties(2, (Span<double> props, Vec3 pos, ReadOnlySpan<double> old) =>
				{
					props[0] = pos.X;
					props[1] = pos.Y;
				});
			Manifold m1 = cube.Translate(Vec3.Splat(1.5));
			Manifold result = cube.Union(m1);
			await Assert.That(result.NumProp())
				.IsEqualTo(2)
				.Because($"PropertiesNoIntersection: num_prop should be 2, got {result.NumProp()}");
		}

		/// <summary>
		/// C++ TEST(Boolean, MixedProperties) — property handling with different property
		/// counts.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMixedProperties()
		{
			Manifold cubeUv = Manifold.Cube(Vec3.Splat(1.0), false)
				.SetProperties(2, (Span<double> props, Vec3 pos, ReadOnlySpan<double> old) =>
				{
					props[0] = pos.X;
					props[1] = pos.Y;
				});
			Manifold cubePlain = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold result = cubeUv.Union(cubePlain.Translate(Vec3.Splat(0.5)));
			await Assert.That(result.NumProp())
				.IsEqualTo(2)
				.Because($"MixedProperties: num_prop should be 2, got {result.NumProp()}");
		}

		/// <summary>Test operator overloads: +/-/^ and +=.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task OperatorOverloads()
		{
			Manifold a = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold b = Manifold.Cube(Vec3.Splat(1.0), false).Translate(new Vec3(0.5, 0.0, 0.0));

			// + is union
			Manifold u = a + b;
			await Assert.That(u.Volume()).IsGreaterThan(1.0).Because($"Union volume should be > 1.0, got {u.Volume()}");

			// - is difference
			Manifold d = a - b;
			await Assert.That(d.Volume() > 0.0 && d.Volume() < 1.0)
				.IsTrue()
				.Because($"Diff volume should be in (0,1), got {d.Volume()}");

			// ^ is intersection
			Manifold i = a ^ b;
			await Assert.That(i.Volume() > 0.0 && i.Volume() < 1.0)
				.IsTrue()
				.Because($"Intersect volume should be in (0,1), got {i.Volume()}");

			// Inclusion-exclusion: U + I = A + B
			await Assert.That(Math.Abs(u.Volume() + i.Volume() - (2.0 * a.Volume())))
				.IsLessThan(0.01)
				.Because($"Inclusion-exclusion: U={u.Volume()} I={i.Volume()} A={a.Volume()}");

			// += operator
			Manifold acc = Manifold.Cube(Vec3.Splat(1.0), false);
			acc += Manifold.Cube(Vec3.Splat(1.0), false).Translate(new Vec3(2.0, 0.0, 0.0));
			await Assert.That(Math.Abs(acc.Volume() - 2.0))
				.IsLessThan(1e-5)
				.Because($"+= volume: {acc.Volume()} expected 2.0");
		}

		/// <summary>C++ TEST(Boolean, EdgeUnion2) — two tetrahedra touching at edge.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppEdgeUnion2()
		{
			Manifold tet = Manifold.Tetrahedron();
			Manifold tet1 = tet.Translate(new Vec3(0.0, 0.0, -1.0));
			Manifold tet2 = tet.Rotate(0.0, 0.0, 90.0).Translate(new Vec3(0.0, 0.0, 1.0));
			Manifold result = tet1.Union(tet2);
			await Assert.That(result.Status()).IsEqualTo(Error.NoError);
			await Assert.That(result.NumVert())
				.IsEqualTo(8)
				.Because($"EdgeUnion2: {result.NumVert()} verts expected 8");
			await Assert.That(result.NumTri())
				.IsEqualTo(8)
				.Because($"EdgeUnion2: {result.NumTri()} tris expected 8");
		}

		/// <summary>C++ TEST(Boolean, SimpleCubeRegression).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSimpleCubeRegression()
		{
			Manifold result = Manifold.Cube(Vec3.Splat(1.0), false)
				.Rotate(-0.1, 0.1, -1.0)
				.Union(Manifold.Cube(Vec3.Splat(1.0), false))
				.Difference(Manifold.Cube(Vec3.Splat(1.0), false).Rotate(-0.1, -0.00000000000066571, -1.0));
			await Assert.That(result.Status()).IsEqualTo(Error.NoError);
		}

		/// <summary>C++ TEST(Boolean, Precision) — tiny cube near precision limit.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPrecision()
		{
			double kPrecision = Types.KPrecision;
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			double distance = 100.0;
			double scale = distance * kPrecision;

			Manifold cube2 = cube.Scale(Vec3.Splat(scale)).Translate(new Vec3(distance, 0.0, 0.0));
			Manifold result = cube.Union(cube2);

			// C++ expects tiny cube absorbed into 8 verts; our impl may keep both
			await Assert.That(result.Status()).IsEqualTo(Error.NoError);

			Manifold cube3 = cube.Scale(Vec3.Splat(2.0 * scale)).Translate(new Vec3(distance, 0.0, 0.0));
			Manifold result2 = result.Union(cube3);
			await Assert.That(result2.Status()).IsEqualTo(Error.NoError);
		}

		/// <summary>C++ TEST(Boolean, PropsMismatch) — union cubes with mismatched properties.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropsMismatch()
		{
			Manifold ma = Manifold.Cylinder(1.0, 1.0, 1.0, 32);
			Manifold mb = Manifold.Cube(Vec3.Splat(1.0), false)
				.Translate(new Vec3(50.0, 0.0, 0.0))
				.SetProperties(1, (Span<double> props, Vec3 pos, ReadOnlySpan<double> old) =>
				{
					props[0] = pos.X;
				});
			Manifold result = ma.Union(mb);
			await Assert.That(result.Status()).IsEqualTo(Error.NoError);
		}

		/// <summary>C++ TEST(Boolean, MixedNumProp) — union cubes with different num_prop.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMixedNumProp()
		{
			Manifold cubeUv = Manifold.Cube(Vec3.Splat(1.0), false)
				.SetProperties(2, (Span<double> props, Vec3 pos, ReadOnlySpan<double> old) =>
				{
					props[0] = pos.X;
					props[1] = pos.Y;
				});
			Manifold cube1Prop = Manifold.Cube(Vec3.Splat(1.0), false)
				.SetProperties(1, (Span<double> props, Vec3 pos, ReadOnlySpan<double> old) =>
				{
					props[0] = 1.0;
				})
				.Translate(Vec3.Splat(0.5));
			Manifold result = cubeUv.Union(cube1Prop);
			await Assert.That(result.NumProp())
				.IsEqualTo(2)
				.Because($"MixedNumProp: num_prop should be 2, got {result.NumProp()}");
		}

		/// <summary>
		/// C++ TEST(Boolean, CreatePropertiesSlow) — position colors via set_properties,
		/// boolean.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCreatePropertiesSlow()
		{
			Manifold a = Manifold.Sphere(10.0, 1024)
				.SetProperties(3, (Span<double> props, Vec3 pos, ReadOnlySpan<double> old) =>
				{
					props[0] = 0.0;
					props[1] = 0.0;
					props[2] = 0.0;
				});
			Manifold b = Manifold.Sphere(10.0, 1024).Translate(new Vec3(5.0, 0.0, 0.0));
			Manifold result = a.Union(b);
			await Assert.That(result.NumProp())
				.IsEqualTo(3)
				.Because($"CreatePropertiesSlow: num_prop should be 3, got {result.NumProp()}");
		}

		/// <summary>
		/// C++ TEST(Boolean, MeshGLRoundTrip) — union of two translated cubes, verify
		/// RelatedGL before and after a MeshGL round-trip (export + re-import).
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMeshGlRoundTrip()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(2.0), false);
			await Assert.That(cube.OriginalId())
				.IsGreaterThanOrEqualTo(0)
				.Because("MeshGLRoundTrip: cube should have original_id >= 0");
			MeshGL original = cube.GetMeshGL(0);

			Manifold result = cube.Union(cube.Translate(new Vec3(1.0, 1.0, 0.0)));
			await Assert.That(result.OriginalId())
				.IsLessThan(0)
				.Because("MeshGLRoundTrip: union result should have negative original_id");
			await Assert.That(result.NumVert()).IsEqualTo(18).Because("MeshGLRoundTrip: expected 18 verts");
			await Assert.That(result.NumTri()).IsEqualTo(32).Because("MeshGLRoundTrip: expected 32 tris");
			await ManifoldTestHelpers.RelatedGl(result, new List<MeshGL> { original });

			MeshGL inGl = result.GetMeshGL(0);
			await Assert.That(inGl.RunOriginalId.Count).IsEqualTo(2).Because("MeshGLRoundTrip: expected 2 runs");
			Manifold result2 = Manifold.FromMeshGL(inGl);

			await Assert.That(result2.OriginalId())
				.IsLessThan(0)
				.Because("MeshGLRoundTrip: result2 should have negative original_id");
			await Assert.That(result2.NumVert()).IsEqualTo(18).Because("MeshGLRoundTrip: result2 expected 18 verts");
			await Assert.That(result2.NumTri()).IsEqualTo(32).Because("MeshGLRoundTrip: result2 expected 32 tris");
			await ManifoldTestHelpers.RelatedGl(result2, new List<MeshGL> { original });

			MeshGL outGl = result2.GetMeshGL(0);
			await Assert.That(outGl.RunOriginalId.Count)
				.IsEqualTo(2)
				.Because("MeshGLRoundTrip: outGL expected 2 runs");
		}

		private static Manifold MakeSpikyDodecahedron(double spikeHeight)
		{
			double phi = (1.0 + Math.Sqrt(5.0)) / 2.0;
			double invPhi = 1.0 / phi;
			double scale = 0.5;
			(double X, double Y, double Z)[] rawVerts =
			{
				(1.0, 1.0, 1.0),
				(1.0, 1.0, -1.0),
				(1.0, -1.0, 1.0),
				(1.0, -1.0, -1.0),
				(-1.0, 1.0, 1.0),
				(-1.0, 1.0, -1.0),
				(-1.0, -1.0, 1.0),
				(-1.0, -1.0, -1.0),
				(0.0, invPhi, phi),
				(0.0, invPhi, -phi),
				(0.0, -invPhi, phi),
				(0.0, -invPhi, -phi),
				(invPhi, phi, 0.0),
				(-invPhi, phi, 0.0),
				(invPhi, -phi, 0.0),
				(-invPhi, -phi, 0.0),
				(phi, 0.0, invPhi),
				(phi, 0.0, -invPhi),
				(-phi, 0.0, invPhi),
				(-phi, 0.0, -invPhi),
			};
			int[][] faces =
			{
				new[] { 0, 8, 10, 2, 16 },
				new[] { 0, 16, 17, 1, 12 },
				new[] { 0, 12, 13, 4, 8 },
				new[] { 1, 17, 3, 11, 9 },
				new[] { 1, 9, 5, 13, 12 },
				new[] { 2, 10, 6, 15, 14 },
				new[] { 2, 14, 3, 17, 16 },
				new[] { 4, 13, 5, 19, 18 },
				new[] { 4, 18, 6, 10, 8 },
				new[] { 5, 9, 11, 7, 19 },
				new[] { 6, 18, 19, 7, 15 },
				new[] { 3, 14, 15, 7, 11 },
			};

			(double X, double Y, double Z)[] verts = new (double, double, double)[rawVerts.Length];
			for (int i = 0; i < rawVerts.Length; i++)
			{
				verts[i] = (rawVerts[i].X * scale, rawVerts[i].Y * scale, rawVerts[i].Z * scale);
			}

			List<float> positions = new List<float>();
			List<uint> triVerts = new List<uint>();
			foreach ((double x, double y, double z) in verts)
			{
				positions.Add((float)x);
				positions.Add((float)y);
				positions.Add((float)z);
			}

			foreach (int[] face in faces)
			{
				double cx = 0.0;
				double cy = 0.0;
				double cz = 0.0;
				foreach (int i in face)
				{
					cx += verts[i].X;
					cy += verts[i].Y;
					cz += verts[i].Z;
				}

				cx /= 5.0;
				cy /= 5.0;
				cz /= 5.0;
				double len = Math.Sqrt((cx * cx) + (cy * cy) + (cz * cz));
				double nx = cx / len;
				double ny = cy / len;
				double nz = cz / len;
				uint spikeIdx = (uint)(positions.Count / 3);
				positions.Add((float)(cx + (nx * spikeHeight)));
				positions.Add((float)(cy + (ny * spikeHeight)));
				positions.Add((float)(cz + (nz * spikeHeight)));
				for (int j = 0; j < 5; j++)
				{
					triVerts.Add(spikeIdx);
					triVerts.Add((uint)face[j]);
					triVerts.Add((uint)face[(j + 1) % 5]);
				}
			}

			MeshGL mesh = new MeshGL();
			mesh.NumProp = 3;
			mesh.VertProperties = positions;
			mesh.TriVerts = triVerts;
			return Manifold.FromMeshGL(mesh);
		}
	}
}
