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

// Tests 20–30 of src/manifold_tests/validation.rs, split out of
// ManifoldValidationTests.cs for the 800-line cap: the Rust's "C++ OBJ-based
// boolean regression tests" banner section and everything after it.
//
// Several of these are the same C++ tests complex.rs also ports, with different
// operations or assertions (Craycloud, the two Generic_Twins, Havocglass8,
// HullMask, SelfIntersect). Both ports are kept, because the Rust keeps both —
// this suite is a test-for-test port, so a duplicate in the Rust is a duplicate
// here.
//
// Warp2's `v.z.round()` is Rust's f64::round, which breaks ties AWAY FROM ZERO.
// C#'s bare Math.Round is banker's rounding and would silently disagree on the
// half-integer z values an Extrude with nDivisions=10 produces, so the call
// below passes MidpointRounding.AwayFromZero explicitly.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldValidationTests
	{
		/// <summary>C++ TEST(BooleanComplex, CraycloudBool).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppCraycloudBool()
		{
			Manifold m1 = ManifoldTestHelpers.ReadTestObj("Cray_left.obj");
			Manifold m2 = ManifoldTestHelpers.ReadTestObj("Cray_right.obj");
			Manifold res = m1 - m2;
			await Assert.That(res.Status()).IsEqualTo(Error.NoError);
			await Assert.That(res.IsEmpty())
				.IsFalse()
				.Because("CraycloudBool result should not be empty");
			Manifold simplified = res.AsOriginal().Simplify(0.0);
			await Assert.That(simplified.IsEmpty())
				.IsTrue()
				.Because("CraycloudBool simplified should be empty");
		}

		/// <summary>C++ TEST(BooleanComplex, GenericTwinBooleanTest7081).</summary>
		[Test]
		[Skip("Slow in debug only — passes in release (~18s; C++ takes ~50s). No longer hangs "
			+ "after the RecursiveEdgeSwap FormLoop / edge<0 fix in edge_op.rs. Kept ignored "
			+ "for debug-suite speed, like sdf_blobs/hull_sphere.")]
		public void CppGenericTwin7081()
		{
			Manifold m1 = ManifoldTestHelpers.ReadTestObj("Generic_Twin_7081.1.t0_left.obj");
			Manifold m2 = ManifoldTestHelpers.ReadTestObj("Generic_Twin_7081.1.t0_right.obj");
			Manifold res = m1 + m2;
			MeshGL gl = res.GetMeshGL(0); // C++ test: just checks this doesn't crash
			_ = gl;
		}

		/// <summary>C++ TEST(BooleanComplex, GenericTwinBooleanTest7863).</summary>
		[Test]
		public void CppGenericTwin7863()
		{
			Manifold m1 = ManifoldTestHelpers.ReadTestObj("Generic_Twin_7863.1.t0_left.obj");
			Manifold m2 = ManifoldTestHelpers.ReadTestObj("Generic_Twin_7863.1.t0_right.obj");
			Manifold res = m1 + m2;
			MeshGL gl = res.GetMeshGL(0);
			_ = gl;
		}

		/// <summary>C++ TEST(BooleanComplex, Havocglass8Bool).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppHavocglass8Bool()
		{
			Manifold m1 = ManifoldTestHelpers.ReadTestObj("Havocglass8_left.obj");
			Manifold m2 = ManifoldTestHelpers.ReadTestObj("Havocglass8_right.obj");
			Manifold res = m1 - m2;
			await Assert.That(res.Status()).IsEqualTo(Error.NoError);
		}

		/// <summary>C++ TEST(BooleanComplex, HullMask).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppHullMask()
		{
			Manifold body = ManifoldTestHelpers.ReadTestObj("hull-body.obj");
			Manifold mask = ManifoldTestHelpers.ReadTestObj("hull-mask.obj");
			Manifold res = body - mask;
			await Assert.That(res.Status()).IsEqualTo(Error.NoError);
			await Assert.That(res.IsEmpty()).IsFalse();
		}

		/// <summary>
		/// C++ TEST(BooleanComplex, SelfIntersect) — tests with self-intersecting OBJ inputs.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSelfIntersect()
		{
			Manifold a = ManifoldTestHelpers.ReadTestObj("self_intersectA.obj");
			Manifold b = ManifoldTestHelpers.ReadTestObj("self_intersectB.obj");
			Manifold res = a - b;
			await Assert.That(res.Status()).IsEqualTo(Error.NoError);
		}

		/// <summary>C++ TEST(Manifold, Warp2) — extrude + warp + batch boolean.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppWarp2()
		{
			CrossSection circle = CrossSection.Circle(5.0, 20).Translate(new Vec2(10.0, 10.0));
			double pi = Math.PI;

			Manifold shape = Manifold.Extrude(circle.ToPolygons(), 2.0, 10, 0.0, new Vec2(1.0, 1.0))
				.Warp((ref Vec3 v) =>
				{
					int nSegments = 10;
					double angleStep = 2.0 / 3.0 * pi / nSegments;
					double zIndex = nSegments - 1.0 - Math.Round(v.Z, MidpointRounding.AwayFromZero);
					double angle = zIndex * angleStep;
					double newZ = v.Y;
					double newY = v.X * Math.Sin(angle);
					double newX = v.X * Math.Cos(angle);
					v = new Vec3(newX, newY, newZ);
				});

			Manifold simplified = Manifold.BatchBoolean(
				new List<Manifold> { shape.Clone() },
				OpType.Add);
			await Assert.That(Math.Abs(shape.Volume() - simplified.Volume()) < 0.0001)
				.IsTrue()
				.Because($"Warp2: volumes differ {shape.Volume()} vs {simplified.Volume()}");
			await Assert.That(Math.Abs(shape.SurfaceArea() - simplified.SurfaceArea()) < 0.0001)
				.IsTrue()
				.Because("Warp2: areas differ");
			await Assert.That(Math.Abs(shape.Volume() - 321.0) < 1.0)
				.IsTrue()
				.Because($"Warp2: volume={shape.Volume()}, expected ~321");
		}

		/// <summary>C++ TEST(Boolean, Precision2) — intersection at precision boundary.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppBooleanPrecision2()
		{
			double kPrecision = 1e-12;
			double scale = 1000.0;
			Manifold cube = Manifold.Cube(Vec3.Splat(scale), false);
			double distance = scale * (1.0 - (kPrecision / 2.0));

			// Overlap = scale - distance = scale * kPrecision / 2 = 5e-10 < epsilon (1e-9) → empty
			Manifold cube2 = cube.Translate(Vec3.Splat(-distance));
			await Assert.That(cube.Intersection(cube2).IsEmpty())
				.IsTrue()
				.Because("Precision2: intersection below epsilon should be empty");

			// Add kPrecision * scale ≈ 1e-9 more overlap → now above epsilon → not empty
			cube2 = cube2.Translate(Vec3.Splat(scale * kPrecision));
			await Assert.That(cube.Intersection(cube2).IsEmpty())
				.IsFalse()
				.Because("Precision2: intersection above epsilon should not be empty");
		}

		/// <summary>
		/// C++ TEST(Boolean, TreeTransforms) — transforms are correctly applied through union
		/// trees. Two cubes overlapping → union volume = 2.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppTreeTransforms()
		{
			Manifold a = Manifold.Cube(Vec3.Splat(1.0), false)
				.Union(Manifold.Cube(Vec3.Splat(1.0), false))
				.Translate(new Vec3(1.0, 0.0, 0.0));
			Manifold b = Manifold.Cube(Vec3.Splat(1.0), false)
				.Union(Manifold.Cube(Vec3.Splat(1.0), false));
			double vol = a.Union(b).Volume();
			await Assert.That(Math.Abs(vol - 2.0) < 1e-5)
				.IsTrue()
				.Because($"TreeTransforms: volume={vol:F6}, expected 2.0");
		}

		/// <summary>
		/// C++ TEST(Boolean, Normals) — boolean result with normals validated via RelatedGL
		/// with checkNormals.
		/// </summary>
		/// <remarks>
		/// Uses CubeSTL (6 props per vert: xyz+normal) and sphere with CalculateNormals, then
		/// checks the boolean result has valid unit normals pointing in the correct direction.
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppBooleanNormals()
		{
			MeshGL cubeGl = ManifoldTestHelpers.CubeStl();
			cubeGl.Merge();
			Manifold cube = Manifold.FromMeshGL(cubeGl);
			Manifold sphere = Manifold.Sphere(60.0, 0).CalculateNormals(0, 60.0);
			MeshGL sphereGl = sphere.GetMeshGL(0);

			Manifold result = cube.Scale(Vec3.Splat(100.0)).Difference(
				sphere.Rotate(180.0, 0.0, 0.0).Difference(
					sphere.Scale(Vec3.Splat(0.5))
						.Rotate(90.0, 0.0, 0.0)
						.Translate(new Vec3(40.0, 40.0, 40.0))));

			await ManifoldTestHelpers.RelatedGlCheckNormals(
				result,
				new List<MeshGL> { cubeGl, sphereGl });

			// Round-trip: export, clear merge verts, re-merge, re-import, check again
			MeshGL output = result.GetMeshGL(0);
			output.MergeFromVert.Clear();
			output.MergeToVert.Clear();
			output.Merge();
			Manifold roundTrip = Manifold.FromMeshGL(output);
			await ManifoldTestHelpers.RelatedGlCheckNormals(
				roundTrip,
				new List<MeshGL> { cubeGl, sphereGl });
		}

		/// <summary>
		/// C++ TEST(Boolean, EmptyOriginal) — tet minus non-intersecting cube.
		/// </summary>
		/// <remarks>
		/// Verifies run metadata: 2 runs (tet with tris, cube with 0 tris), and that the
		/// cube's run transform preserves the translation.
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppEmptyOriginal()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			Manifold tet = Manifold.Tetrahedron();
			Manifold result = tet.Difference(cube.Translate(new Vec3(3.0, 4.0, 5.0)));
			MeshGL mesh = result.GetMeshGL(0);

			await Assert.That(mesh.RunIndex.Count)
				.IsEqualTo(3)
				.Because($"EmptyOriginal: expected 3 run_index entries, got {mesh.RunIndex.Count}");
			await Assert.That(mesh.RunIndex[0])
				.IsEqualTo(0u)
				.Because("EmptyOriginal: first run starts at 0");
			await Assert.That(mesh.RunIndex[1])
				.IsEqualTo((uint)mesh.TriVerts.Count)
				.Because("EmptyOriginal: tet run ends at all tris");
			await Assert.That(mesh.RunIndex[2])
				.IsEqualTo((uint)mesh.TriVerts.Count)
				.Because("EmptyOriginal: cube run is empty");
			await Assert.That(mesh.RunOriginalId.Count)
				.IsEqualTo(2)
				.Because(
					$"EmptyOriginal: expected 2 run_original_ids, got {mesh.RunOriginalId.Count}");
			await Assert.That(mesh.RunOriginalId[0])
				.IsEqualTo((uint)tet.OriginalId())
				.Because("EmptyOriginal: first run is tet");
			await Assert.That(mesh.RunOriginalId[1])
				.IsEqualTo((uint)cube.OriginalId())
				.Because("EmptyOriginal: second run is cube");
			await Assert.That(mesh.RunTransform.Count)
				.IsEqualTo(24)
				.Because(
					$"EmptyOriginal: expected 24 transform elements, got {mesh.RunTransform.Count}");

			// Tet transform: identity — translation at column 3 (indices 9,10,11) = 0,0,0
			await Assert.That(Math.Abs(mesh.RunTransform[9] - 0.0f) < 1e-5)
				.IsTrue()
				.Because("EmptyOriginal: tet tx=0");
			await Assert.That(Math.Abs(mesh.RunTransform[10] - 0.0f) < 1e-5)
				.IsTrue()
				.Because("EmptyOriginal: tet ty=0");
			await Assert.That(Math.Abs(mesh.RunTransform[11] - 0.0f) < 1e-5)
				.IsTrue()
				.Because("EmptyOriginal: tet tz=0");

			// Cube transform: translated by (3,4,5) — indices 21,22,23
			await Assert.That(Math.Abs(mesh.RunTransform[12 + 9] - 3.0f) < 1e-4)
				.IsTrue()
				.Because("EmptyOriginal: cube tx=3");
			await Assert.That(Math.Abs(mesh.RunTransform[12 + 10] - 4.0f) < 1e-4)
				.IsTrue()
				.Because("EmptyOriginal: cube ty=4");
			await Assert.That(Math.Abs(mesh.RunTransform[12 + 11] - 5.0f) < 1e-4)
				.IsTrue()
				.Because("EmptyOriginal: cube tz=5");
		}
	}
}
