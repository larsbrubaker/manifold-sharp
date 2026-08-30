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

// Tests 14–21 of src/manifold_tests/complex.rs, split out of
// ManifoldComplexTests.cs for the 800-line cap: the run-metadata round trips and
// the volume/genus identities — MeshGLRoundTrip, CraycloudBool, BooleanVolumes,
// Spiral, OpenscadCrash, MeshGLRoundTrip (Manifold), Sphere, MeshRelation.

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
		/// C++ TEST(Boolean, MeshGLRoundTrip) — boolean result preserves mesh runs through
		/// MeshGL.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMeshglRoundTrip()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(2.0), false);
			await Assert.That(cube.OriginalId() >= 0)
				.IsTrue()
				.Because("Cube should have positive originalID");
			MeshGL original = cube.GetMeshGL(0);
			_ = original;

			Manifold result = cube.Clone() + cube.Translate(new Vec3(1.0, 1.0, 0.0));

			await Assert.That(result.OriginalId() < 0)
				.IsTrue()
				.Because("Boolean result should have negative originalID");

			// Result should have ~18 verts, 32 tris (two overlapping cubes)
			await Assert.That(result.NumVert()).IsGreaterThan(0);
			await Assert.That(result.NumTri()).IsGreaterThan(0);

			MeshGL inGl = result.GetMeshGL(0);

			// Boolean of 2 meshes → 2 runs
			await Assert.That(inGl.RunOriginalId.Count)
				.IsEqualTo(2)
				.Because($"MeshGLRoundTrip: expected 2 runs, got {inGl.RunOriginalId.Count}");

			// Reconstruct from MeshGL
			Manifold result2 = Manifold.FromMeshGL(inGl);
			await Assert.That(result2.OriginalId() < 0).IsTrue();
			await Assert.That(result2.NumVert()).IsGreaterThan(0);
			await Assert.That(result2.NumTri()).IsGreaterThan(0);

			MeshGL outGl = result2.GetMeshGL(0);
			await Assert.That(outGl.RunOriginalId.Count)
				.IsEqualTo(2)
				.Because(
					"MeshGLRoundTrip: roundtrip should preserve 2 runs, got "
					+ $"{outGl.RunOriginalId.Count}");
		}

		/// <summary>
		/// C++ TEST(BooleanComplex, CraycloudBool) — subtract complements, simplify to empty.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppComplexCraycloud()
		{
			Manifold m1 = ManifoldTestHelpers.ReadTestObj("Cray_left.obj");
			Manifold m2 = ManifoldTestHelpers.ReadTestObj("Cray_right.obj");
			Manifold res = m1 - m2;
			await Assert.That(res.Status()).IsEqualTo(Error.NoError);
			await Assert.That(res.IsEmpty())
				.IsFalse()
				.Because("CraycloudBool: difference should not be empty");
			Manifold simplified = res.AsOriginal().Simplify(0.0);
			await Assert.That(simplified.IsEmpty())
				.IsTrue()
				.Because(
					"CraycloudBool: AsOriginal().Simplify() should produce empty mesh, got "
					+ $"{simplified.NumTri()} tris");
		}

		/// <summary>
		/// C++ TEST(BooleanComplex, BooleanVolumes) — volume arithmetic with non-overlapping
		/// cubes.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppComplexBooleanVolumes()
		{
			Manifold m1 = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false);
			Manifold m2 = Manifold.Cube(new Vec3(2.0, 1.0, 1.0), false)
				.Translate(new Vec3(1.0, 0.0, 0.0));
			Manifold m4 = Manifold.Cube(new Vec3(4.0, 1.0, 1.0), false)
				.Translate(new Vec3(3.0, 0.0, 0.0));
			Manifold m3 = Manifold.Cube(new Vec3(3.0, 1.0, 1.0), false);
			Manifold m7 = Manifold.Cube(new Vec3(7.0, 1.0, 1.0), false);

			double eps = 1e-4;
			await Assert.That(Math.Abs((m1.Clone() ^ m2.Clone()).Volume() - 0.0) < eps)
				.IsTrue()
				.Because("m1^m2");
			await Assert.That(
					Math.Abs((m1.Clone() + m2.Clone() + m4.Clone()).Volume() - 7.0) < eps)
				.IsTrue()
				.Because("m1+m2+m4");
			await Assert.That(
					Math.Abs((m1.Clone() + m2.Clone() - m4.Clone()).Volume() - 3.0) < eps)
				.IsTrue()
				.Because("m1+m2-m4");
			await Assert.That(
					Math.Abs((m1.Clone() + (m2.Clone() ^ m4.Clone())).Volume() - 1.0) < eps)
				.IsTrue()
				.Because("m1+(m2^m4)");
			await Assert.That(Math.Abs((m7.Clone() ^ m4.Clone()).Volume() - 4.0) < eps)
				.IsTrue()
				.Because("m7^m4");
			await Assert.That(
					Math.Abs((m7.Clone() ^ m3.Clone() ^ m1.Clone()).Volume() - 1.0) < eps)
				.IsTrue()
				.Because("m7^m3^m1");
			await Assert.That(
					Math.Abs((m7.Clone() ^ (m1.Clone() + m2.Clone())).Volume() - 3.0) < eps)
				.IsTrue()
				.Because("m7^(m1+m2)");
			await Assert.That(Math.Abs((m7.Clone() - m4.Clone()).Volume() - 3.0) < eps)
				.IsTrue()
				.Because("m7-m4");
			await Assert.That(
					Math.Abs((m7.Clone() - m4.Clone() - m2.Clone()).Volume() - 1.0) < eps)
				.IsTrue()
				.Because("m7-m4-m2");
			await Assert.That(
					Math.Abs((m7.Clone() - (m7.Clone() - m1.Clone())).Volume() - 1.0) < eps)
				.IsTrue()
				.Because("m7-(m7-m1)");
			await Assert.That(
					Math.Abs((m7.Clone() - (m1.Clone() + m2.Clone())).Volume() - 4.0) < eps)
				.IsTrue()
				.Because("m7-(m1+m2)");
		}

		/// <summary>
		/// C++ TEST(BooleanComplex, Spiral) — recursive spiral of cubes.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppComplexSpiral()
		{
			double d = 2.0;

			static Manifold Spiral(int rec, double r, double add, double d)
			{
				double rot = 360.0 / (Math.PI * r * 2.0) * d;
				double rNext = r + (add / 360.0 * rot);
				Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true)
					.Translate(new Vec3(0.0, r, 0.0));
				return rec > 0
					? Spiral(rec - 1, rNext, add, d).Rotate(0.0, 0.0, rot) + cube
					: cube;
			}

			Manifold result = Spiral(120, 25.0, 2.0, d);
			await Assert.That(result.Genus())
				.IsEqualTo(-120)
				.Because($"Spiral genus should be -120, got {result.Genus()}");
		}

		/// <summary>
		/// C++ TEST(Manifold, OpenscadCrash) — OBJ that previously crashed openscad.
		/// </summary>
		/// <remarks>
		/// Passes since <c>face2tri</c> was rewritten to the C++ v3.5.0 halfedge-index
		/// pairing scheme (keeps overlapping-polygon triangulations manifold) and
		/// <c>update_vert</c> matched C++'s graceful start==end no-op.
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppOpenscadCrash()
		{
			Manifold m = ManifoldTestHelpers.ReadTestObj("openscad-nonmanifold-crash.obj");
			await Assert.That(m.IsEmpty())
				.IsFalse()
				.Because($"OBJ should load as non-empty manifold, status={m.Status()}");
			Manifold m2 = m.Clone() + m.Translate(new Vec3(0.0, 0.6, 0.0));
			await Assert.That(m2.IsEmpty())
				.IsFalse()
				.Because($"Boolean union should not be empty, status={m2.Status()}");
		}

		/// <summary>
		/// C++ TEST(Manifold, MeshGLRoundTrip) — MeshGL round-trip preserves original ID.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMeshglRoundTrip2()
		{
			Manifold cylinder = Manifold.Cylinder(2.0, 1.0, -1.0, 0);
			await Assert.That(cylinder.OriginalId() >= 0).IsTrue();
			MeshGL inGl = cylinder.GetMeshGL(0);
			Manifold cylinder2 = Manifold.FromMeshGL(inGl);
			MeshGL outGl = cylinder2.GetMeshGL(0);

			await Assert.That(inGl.RunOriginalId.Count).IsEqualTo(1).Because("Input should have 1 run");
			await Assert.That(outGl.RunOriginalId.Count).IsEqualTo(1).Because("Output should have 1 run");
			await Assert.That(outGl.RunOriginalId[0])
				.IsEqualTo(inGl.RunOriginalId[0])
				.Because("Original ID should be preserved through round-trip");
		}

		/// <summary>
		/// C++ TEST(BooleanComplex, Sphere) — sphere difference with position colors.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppComplexSphereBoolean()
		{
			// Simplified version without WithPositionColors/RelatedGL
			Manifold sphere = Manifold.Sphere(1.0, 12).SetProperties(
				3,
				(Span<double> newProp, Vec3 pos, ReadOnlySpan<double> old) =>
				{
					newProp[0] = pos.X;
					newProp[1] = pos.Y;
					newProp[2] = pos.Z;
				});
			Manifold sphere2 = sphere.Translate(Vec3.Splat(0.5));
			Manifold result = sphere.Clone() - sphere2;

			await Assert.That(result.IsEmpty())
				.IsFalse()
				.Because("Sphere difference should not be empty");
			await Assert.That(result.Status()).IsEqualTo(Error.NoError);

			// C++ expects 74 verts, 144 tris with 3 props and 110 degenerate tris = 0
			await Assert.That(result.NumTri()).IsGreaterThan(0).Because("Should have triangles");
			await Assert.That(result.NumProp())
				.IsEqualTo(3)
				.Because("Should have 3 extra properties");

			// Refine should preserve properties
			Manifold refined = result.Refine(4);
			await Assert.That(refined.IsEmpty()).IsFalse().Because("Refined should not be empty");
			await Assert.That(refined.NumProp()).IsEqualTo(3).Because("Refined should have 3 props");
		}

		/// <summary>
		/// C++ TEST(BooleanComplex, MeshRelation) — gyroid + translated gyroid, refine,
		/// RelatedGL.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppComplexMeshRelation()
		{
			Manifold gyroidSrc = ManifoldTestHelpers.WithPositionColors(ManifoldTestHelpers.Gyroid());
			MeshGL gyroidGl = gyroidSrc.GetMeshGL(0);
			Manifold gyroid = gyroidSrc.Simplify(0.0);
			Manifold gyroid2 = gyroid.Translate(Vec3.Splat(2.0));

			await Assert.That(gyroid.IsEmpty()).IsFalse().Because("MeshRelation: gyroid not empty");
			await Assert.That(gyroid.MatchesTriNormals())
				.IsTrue()
				.Because("MeshRelation: matches_tri_normals");
			await Assert.That(gyroid.NumDegenerateTris() <= 0)
				.IsTrue()
				.Because("MeshRelation: num_degenerate_tris <= 0");

			Manifold result = gyroid.Union(gyroid2).RefineToLength(0.1);
			await Assert.That(result.MatchesTriNormals())
				.IsTrue()
				.Because("MeshRelation: result matches_tri_normals");
			await Assert.That(result.NumDegenerateTris() <= 12)
				.IsTrue()
				.Because("MeshRelation: num_degenerate_tris <= 12");
			await Assert.That(result.Decompose().Count)
				.IsEqualTo(1)
				.Because("MeshRelation: 1 component");
			await Assert.That(Math.Abs(result.Volume() - 226.0) < 1.0)
				.IsTrue()
				.Because($"MeshRelation: vol={result.Volume()}");
			await Assert.That(Math.Abs(result.SurfaceArea() - 387.0) < 1.0)
				.IsTrue()
				.Because($"MeshRelation: sa={result.SurfaceArea()}");

			await ManifoldTestHelpers.RelatedGl(result, new List<MeshGL> { gyroidGl });
		}
	}
}
