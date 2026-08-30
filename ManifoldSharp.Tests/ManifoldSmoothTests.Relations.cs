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

// The second half of the src/manifold_tests/smooth.rs port, split off
// ManifoldSmoothTests.cs for the 800-line file cap. That file carries the module
// header and the helper notes; this one continues the same partial class in the
// same order as the Rust, from TEST(Smooth, MissingNormals) to the end of the
// module. Neither file defers anything.
//
// Two of these — MeshRelationRefine and MeshRelationRefinePrecision — share a
// name with a test in api.rs. They are NOT duplicates: the api.rs pair runs the
// counts through Decompose and this pair asserts them directly, and the Rust
// keeps both. Both are ported, in ManifoldApiTests.Relations.cs and here.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldSmoothTests
	{
		/// <summary>
		/// C++ TEST(Smooth, MissingNormals) — smooth with zero-length normals at boolean
		/// boundary.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothMissingNormals()
		{
			Manifold tetNorm = Manifold.Tetrahedron().CalculateNormals(0, 60.0);
			Manifold diff = tetNorm.Difference(
				Manifold.Tetrahedron().Translate(new Vec3(0.5, 0.5, 0.5)));
			Manifold outM = diff.SmoothByNormals(0).Refine(10);
			await Assert.That(Math.Abs(outM.Volume() - 2.46))
				.IsLessThan(0.01)
				.Because($"MissingNormals vol={outM.Volume()}");
			await Assert.That(Math.Abs(outM.SurfaceArea() - 12.45))
				.IsLessThan(0.01)
				.Because($"MissingNormals sa={outM.SurfaceArea()}");
		}

		/// <summary>
		/// C++ TEST(Smooth, MissingNormalsCone) — smooth cone with missing normals at cut.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothMissingNormalsCone()
		{
			Manifold cone = Manifold.Cylinder(10.0, 10.0, 0.0, 5).CalculateNormals(0, 60.0);
			Manifold cube = Manifold.Cube(Vec3.Splat(10.0), true)
				.Translate(new Vec3(0.0, 0.0, 10.0));
			Manifold diff = cone.Difference(cube);
			Manifold outM = diff.SmoothByNormals(0).Refine(20);
			await Assert.That(Math.Abs(outM.Volume() - 1009.0))
				.IsLessThan(1.0)
				.Because($"MissingNormalsCone vol={outM.Volume()}");
			await Assert.That(Math.Abs(outM.SurfaceArea() - 736.0))
				.IsLessThan(1.0)
				.Because($"MissingNormalsCone sa={outM.SurfaceArea()}");
		}

		/// <summary>
		/// C++ TEST(Smooth, InvalidTangents) — corrupted w=-1 tangents → InvalidTangents
		/// error.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothInvalidTangents()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false).SmoothOut(180.0, 0.0);
			MeshGL withTangents = cube.GetMeshGL(0);
			int sizeHalfedges = withTangents.HalfedgeTangent.Count;

			// Mark second half of tangents as kInsideQuad (-1), which is invalid
			MeshGL mesh = withTangents;
			int i = ((sizeHalfedges / 8) * 4) + 3;
			while (i < sizeHalfedges)
			{
				mesh.HalfedgeTangent[i] = -1.0f;
				i += 4;
			}

			Manifold cube2 = Manifold.FromMeshGL(mesh);
			Manifold smooth = cube2.Refine(10);
			await Assert.That(smooth.Status())
				.IsEqualTo(Error.InvalidTangents)
				.Because($"Expected InvalidTangents, got {smooth.Status()}");
		}

		/// <summary>
		/// C++ TEST(Manifold, MeshRelationRefine) — position colors preserved after
		/// RefineToLength.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMeshRelationRefine()
		{
			Manifold csaszarManifold = Manifold.FromMeshGL(CsaszarGl());
			Manifold csaszar = ManifoldTestHelpers.WithPositionColors(csaszarManifold).AsOriginal();
			MeshGL inGl = csaszar.GetMeshGL(0);

			await ManifoldTestHelpers.RelatedGl(csaszar, new[] { inGl });

			// Check mesh sizes after refine to length 1
			Manifold refined = csaszar.RefineToLength(1.0);
			await Assert.That(refined.IsEmpty())
				.IsFalse()
				.Because("MeshRelationRefine: result is empty");
			await Assert.That(refined.MatchesTriNormals())
				.IsTrue()
				.Because("MeshRelationRefine: normals don't match tris");
			await Assert.That(refined.NumVert())
				.IsEqualTo(9019)
				.Because($"MeshRelationRefine: expected 9019 verts, got {refined.NumVert()}");
			await Assert.That(refined.NumTri())
				.IsEqualTo(18038)
				.Because($"MeshRelationRefine: expected 18038 tris, got {refined.NumTri()}");
			await Assert.That(refined.NumProp())
				.IsEqualTo(3)
				.Because($"MeshRelationRefine: expected num_prop=3, got {refined.NumProp()}");
			await ManifoldTestHelpers.RelatedGl(refined, new[] { inGl });
		}

		/// <summary>
		/// C++ TEST(Manifold, MeshRelationRefinePrecision) — smooth mesh with
		/// RefineToTolerance.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMeshRelationRefinePrecision()
		{
			Manifold csaszarManifold = Manifold.FromMeshGL(CsaszarGl());
			MeshGL inGl = ManifoldTestHelpers.WithPositionColors(csaszarManifold).GetMeshGL(0);
			uint id = inGl.RunOriginalId[0];
			Manifold csaszar = Manifold.Smooth(inGl, Array.Empty<Smoothness>());

			Manifold refined = csaszar.RefineToTolerance(0.05);
			await Assert.That(refined.IsEmpty())
				.IsFalse()
				.Because("MeshRelationRefinePrecision: result is empty");
			await Assert.That(refined.MatchesTriNormals())
				.IsTrue()
				.Because("MeshRelationRefinePrecision: normals don't match tris");

			// C++ v3.5.0 expects {{2343, 4686, 3}} after the #1724/#1671 smoothing fixes.
			await Assert.That(refined.NumVert())
				.IsEqualTo(2343)
				.Because($"MeshRelationRefinePrecision: expected 2343 verts, got {refined.NumVert()}");
			await Assert.That(refined.NumTri())
				.IsEqualTo(4686)
				.Because($"MeshRelationRefinePrecision: expected 4686 tris, got {refined.NumTri()}");
			await Assert.That(refined.NumProp())
				.IsEqualTo(3)
				.Because($"MeshRelationRefinePrecision: expected num_prop=3, got {refined.NumProp()}");

			// Verify the run original ID is preserved
			MeshGL outGl = refined.GetMeshGL(0);
			await Assert.That(outGl.RunOriginalId.Count).IsEqualTo(1);
			await Assert.That(outGl.RunOriginalId[0])
				.IsEqualTo(id)
				.Because("MeshRelationRefinePrecision: original ID not preserved");
		}

		/// <summary>
		/// C++ TEST(Smooth, Sphere) — smoothed sphere vertices stay near radius 1.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothSphere()
		{
			int[] ns = { 4, 8, 16, 32, 64 };

			// Tests vertex precision of interpolation. Refine(6) makes a center point,
			// which is the worst case for deviation from the unit sphere.
			double[] precisions = { 0.04, 0.003, 0.003, 0.0005, 0.00006 };
			for (int i = 0; i < ns.Length; i++)
			{
				int n = ns[i];
				Manifold sphere = Manifold.Sphere(1.0, n);
				Manifold smoothed = Manifold
					.Smooth(sphere.GetMeshGL(0), Array.Empty<Smoothness>())
					.Refine(6);
				MeshGL64 mesh = smoothed.GetMeshGL64(0);
				int numVert = mesh.NumVert();
				double maxR2 = 0.0;
				double minR2 = 2.0;
				for (int v = 0; v < numVert; v++)
				{
					(double px, double py, double pz) = mesh.GetVertPos(v);
					double r2 = (px * px) + (py * py) + (pz * pz);
					if (r2 > maxR2)
					{
						maxR2 = r2;
					}

					if (r2 < minR2)
					{
						minR2 = r2;
					}
				}

				double prec = precisions[i];
				await Assert.That(Math.Abs(Math.Sqrt(minR2) - 1.0))
					.IsLessThan(prec)
					.Because($"Sphere n={n}: min_r={Math.Sqrt(minR2):F6} expected ≈1.0 within {prec}");
				await Assert.That(Math.Abs(Math.Sqrt(maxR2) - 1.0))
					.IsLessThan(prec)
					.Because($"Sphere n={n}: max_r={Math.Sqrt(maxR2):F6} expected ≈1.0 within {prec}");
			}
		}

		/// <summary>
		/// C++ TEST(Smooth, Fillet) — smoke test: Simplify+SmoothByNormals must not crash.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothFillet()
		{
			double depth = 3.0;
			Manifold cylinder = Manifold.Cylinder(40.0, 10.0, 10.0, 6).CalculateNormals(0, 80.0);
			CrossSection slice = cylinder.Slice(0.0);
			CrossSection section = new CrossSection(slice.ToPolygons()).Simplify(1e-6);
			Manifold chamfer = Manifold.Extrude(
					section.ToPolygons(),
					depth,
					0,
					0.0,
					new Vec2(1.2, 1.3))
				.Mirror(new Vec3(0.0, 0.0, 1.0));
			Manifold baseCube = Manifold.Cube(Vec3.Splat(40.0), true)
				.Translate(new Vec3(0.0, 0.0, -20.0 - depth + 0.001))
				.CalculateNormals(0, 60.0);
			Manifold chamfered = (cylinder + chamfer).Difference(baseCube);
			Manifold fillet = chamfered.Simplify(0.01).SmoothByNormals(0).Refine(10);
			await Assert.That(fillet.Status())
				.IsEqualTo(Error.NoError)
				.Because($"Fillet status={fillet.Status()}");
		}

		/// <summary>
		/// C++ TEST(Manifold, MeshRelation) — gyroid with position colors, RelatedGL after
		/// simplify.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMeshRelation()
		{
			Manifold gyroid = ManifoldTestHelpers.WithPositionColors(ManifoldTestHelpers.Gyroid());
			MeshGL gyroidGl = gyroid.GetMeshGL(0);
			Manifold simplified = gyroid.Simplify(0.0);
			await ManifoldTestHelpers.RelatedGl(simplified, new[] { gyroidGl });
		}
	}
}
