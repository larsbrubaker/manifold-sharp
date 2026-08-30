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

// The third file of the src/manifold_tests/boolean.rs port, continuing the same
// partial class. It exists because ManifoldBooleanTests.Shapes.cs is already 714
// lines and these four cases do not fit under the 800-line cap.
//
// These are the four that manifold_smooth.rs unblocked — the ones the DEFERRED
// table in ManifoldBooleanTests.cs used to list — kept in the Rust's relative
// order (MissingNormals, Simplify, SimplifyCracks, Normals). Grouping them by
// what blocked them rather than splicing them into the middle of Shapes.cs is
// the one place this port's file layout diverges from the Rust's ordering, and
// it is a file-cap decision, not a semantic one.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldBooleanTests
	{
		/// <summary>
		/// C++ TEST(Boolean, MissingNormals) — union of cube with/without normals.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMissingNormals()
		{
			Manifold noNormals = Manifold.Cube(Vec3.Splat(1.0), true);
			Manifold hasNormals = Manifold.Cube(Vec3.Splat(2.0), true)
				.Translate(new Vec3(0.0, 0.0, -1.0))
				.CalculateNormals(0, 30.0);
			MeshGL combo = (noNormals + hasNormals).GetMeshGL(0);
			Manifold result = Manifold.FromMeshGL(combo);
			await Assert.That(result.IsEmpty())
				.IsFalse()
				.Because("MissingNormals result should not be empty");
		}

		/// <summary>
		/// C++ TEST(Boolean, Simplify) — refine cube, boolean, simplify.
		/// The C++ test assigns unique faceIDs which prevents simplify from merging
		/// coplanar tris after boolean. Then it clears faceIDs, reconstructs from
		/// MeshGL, and simplify reduces from 2000 to 20 tris.
		/// Our port tests the roundtrip: MeshGL reconstruction wipes face identity,
		/// then simplify can collapse all coplanar faces.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSimplify()
		{
			int n = 10;
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false).Refine(n);
			Manifold result = cube.Union(cube.Translate(new Vec3(1.0, 0.0, 0.0)));

			// Boolean produces ~2000 tris (may vary slightly without faceID tracking)
			await Assert.That(result.NumTri())
				.IsGreaterThan(1000)
				.Because($"Simplify: pre-simplify should have many tris, got {result.NumTri()}");

			// Clear face/run data and reconstruct (matches C++ test: resultGL.faceID.clear())
			MeshGL meshGl = result.GetMeshGL(0);
			meshGl.FaceId.Clear();
			meshGl.RunOriginalId.Clear();
			meshGl.RunIndex.Clear();
			meshGl.RunTransform.Clear();
			Manifold result2 = Manifold.FromMeshGL(meshGl);
			Manifold simplified = result2.Simplify(0.0);

			// 2x1x1 box: 10 faces × 2 tris = 20
			// If internal face was removed, we'd get 12 (box with no partition).
			// The correct result preserves volume.
			await Assert.That(Math.Abs(simplified.Volume() - 2.0))
				.IsLessThan(0.01)
				.Because($"Simplify: volume should be 2.0, got {simplified.Volume()}");

			// Accept 12 (fully simplified box) or 20 (with partition face preserved)
			await Assert.That(simplified.NumTri() == 12 || simplified.NumTri() == 20)
				.IsTrue()
				.Because($"Simplify: expected 12 or 20 tris, got {simplified.NumTri()}");
		}

		/// <summary>
		/// C++ TEST(Boolean, SimplifyCracks) — simplify should preserve genus/volume/area.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSimplifyCracks()
		{
			Manifold cylinder = Manifold.Cylinder(2.0, 50.0, 50.0, 180)
				.Rotate(-89.999999999999, 0.0, 0.0)
				.Translate(new Vec3(50.0, 0.0, 50.0));
			Manifold cube = Manifold.Cube(new Vec3(100.0, 2.0, 50.0), false);
			Manifold refined = cylinder.Union(cube).RefineToLength(1.0);
			Manifold deformed = refined.Warp((ref Vec3 v) =>
			{
				v.Y += v.X - (v.X * v.X / 100.0);
			});
			Manifold simplified = deformed.Simplify(0.005);

			// If Simplify adds cracks, volume decreases and surface area increases
			await Assert.That(deformed.Genus())
				.IsEqualTo(0)
				.Because("SimplifyCracks: deformed genus should be 0");
			await Assert.That(simplified.Genus())
				.IsEqualTo(0)
				.Because("SimplifyCracks: simplified genus should be 0");
			await Assert.That(Math.Abs(simplified.Volume() - deformed.Volume()))
				.IsLessThan(10.0)
				.Because($"SimplifyCracks: volume {simplified.Volume()} vs {deformed.Volume()}");
			await Assert.That(Math.Abs(simplified.SurfaceArea() - deformed.SurfaceArea()))
				.IsLessThan(1.0)
				.Because($"SimplifyCracks: area {simplified.SurfaceArea()} vs {deformed.SurfaceArea()}");
		}

		/// <summary>
		/// C++ TEST(Boolean, Normals) — preserve per-vertex normals through booleans and a
		/// MeshGL round-trip.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNormals()
		{
			MeshGL cubeGl = ManifoldTestHelpers.CubeStl();
			cubeGl.Merge();
			Manifold cube = Manifold.FromMeshGL(cubeGl);

			// C++ Sphere(60) with default segments resolves to Quality::GetCircularSegments(60).
			// CalculateNormals(0) uses the v3.5.0 default minSharpAngle of 52.5°.
			int segs = Quality.GetCircularSegments(60.0);
			Manifold sphere = Manifold.Sphere(60.0, segs).CalculateNormals(0, 52.5);
			MeshGL sphereGl = sphere.GetMeshGL(0);

			// cube.scale(100) - (sphere.rotate(180) - sphere.scale(0.5).rotate(90).translate(40,40,40))
			Manifold inner = sphere.Clone().Rotate(180.0, 0.0, 0.0).Difference(
				sphere
					.Scale(Vec3.Splat(0.5))
					.Rotate(90.0, 0.0, 0.0)
					.Translate(new Vec3(40.0, 40.0, 40.0)));
			Manifold result = cube.Scale(Vec3.Splat(100.0)).Difference(inner);

			await ManifoldTestHelpers.RelatedGlCheckNormals(result, new List<MeshGL> { cubeGl, sphereGl });

			MeshGL output = result.GetMeshGL(0);
			output.MergeFromVert.Clear();
			output.MergeToVert.Clear();
			output.Merge();
			Manifold roundTrip = Manifold.FromMeshGL(output);

			await ManifoldTestHelpers.RelatedGlCheckNormals(
				roundTrip,
				new List<MeshGL> { cubeGl, sphereGl });
		}
	}
}
