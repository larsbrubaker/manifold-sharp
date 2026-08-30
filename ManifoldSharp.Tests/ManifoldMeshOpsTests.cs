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

// Port of src/manifold_tests/mesh_ops.rs — all 3 of its tests, same inputs, same
// expected values, same tolerances, in the same order. Nothing is deferred.
//
// Two files, for the 800-line cap: this one holds the tests,
// ManifoldMeshOpsTests.MergeRefineData.cs holds the 834-coordinate mesh literal
// the first test declares inline in the Rust.
//
// The module is small but it is the round-trip cover for the two import paths
// nothing else exercises end to end:
//
//   CppMergeRefine            MeshGL.Merge() on a real captured boolean result
//                             (open edges welded by BVH), then FromMeshGL and
//                             RefineToLength(1.0). Volume 31.21 ± 0.01.
//   MeshGl64BooleanRoundTripIsLossless
//                             the f64 import/export path, asserted BIT-exact.
//   CppObjRoundTrip           WriteObj → ReadObj.
//
// mesh_ops.rs does NOT use the mod.rs cpp-source parsing helpers. The note in
// ManifoldTestHelpers.cs that once named this module as their consumer was
// mistaken — `InterpolatedNormals` and `Ring` live in complex.rs — and it has
// been corrected there.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldMeshOpsTests
	{
		/// <summary>
		/// C++ TEST(Manifold, MergeRefine) — merge a tolerance mesh and refine to length.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMergeRefine()
		{
			MeshGL mesh = new MeshGL();
			mesh.Tolerance = 1e-5f;
			mesh.NumProp = 3;
			mesh.VertProperties = new List<float>(MergeRefineVertProperties);
			mesh.TriVerts = new List<uint>(MergeRefineTriVerts);
			mesh.Merge();
			Manifold manifold = Manifold.FromMeshGL(mesh);
			Manifold refined = manifold.RefineToLength(1.0);
			await Assert.That(Math.Abs(refined.Volume() - 31.21) < 0.01)
				.IsTrue()
				.Because($"MergeRefine: expected volume≈31.21, got {refined.Volume()}");
		}

		/// <summary>
		/// MeshGL64 must be lossless end to end: a coordinate that is NOT representable in
		/// f32 has to survive FromMeshGL64 → boolean → GetMeshGL64 bit-for-bit. This is the
		/// tripwire for any regression back to the old narrow-through-MeshGL import path.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task MeshGl64BooleanRoundTripIsLossless()
		{
			// Per-axis extents whose f64 values have more significand bits than f32
			// holds. Prove the premise before relying on it.
			double[] ext = { 1.0 / 3.0, 2.0 / 3.0, 5.0 / 7.0 };
			foreach (double v in ext)
			{
				await Assert.That((double)(float)v)
					.IsNotEqualTo(v)
					.Because($"test premise: {v} must not be f32-representable");
			}

			// Unit cube scaled per-axis in f64. 0 * k and 1 * k are exact, so the far
			// corner of the box carries exactly the ext values into the import.
			MeshGL64 mesh = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false).GetMeshGL64(-1);
			int np = (int)mesh.NumProp;
			for (int i = 0; i < mesh.NumVert(); i++)
			{
				for (int j = 0; j < 3; j++)
				{
					mesh.VertProperties[(i * np) + j] *= ext[j];
				}
			}

			Manifold a = Manifold.FromMeshGL64(mesh);
			await Assert.That(a.Status()).IsEqualTo(Error.NoError);

			// Import alone must already be lossless.
			MeshGL64 direct = a.GetMeshGL64(-1);
			for (int j = 0; j < 3; j++)
			{
				int dnp = (int)direct.NumProp;
				bool found = false;
				for (int i = 0; i < direct.NumVert(); i++)
				{
					if (direct.VertProperties[(i * dnp) + j] == ext[j])
					{
						found = true;
						break;
					}
				}

				await Assert.That(found)
					.IsTrue()
					.Because($"direct round trip lost axis-{j} coordinate {ext[j]}");
			}

			// Union with a small cube strictly inside A: the boolean pipeline runs,
			// and the result is exactly A's shell, so the precise corner survives.
			Manifold b = Manifold.Cube(new Vec3(0.05, 0.05, 0.05), false)
				.Translate(new Vec3(0.1, 0.1, 0.1));
			Manifold result = a.Union(b);
			await Assert.That(result.Status()).IsEqualTo(Error.NoError);

			MeshGL64 outMesh = result.GetMeshGL64(-1);
			int onp = (int)outMesh.NumProp;
			for (int j = 0; j < 3; j++)
			{
				bool found = false;
				for (int i = 0; i < outMesh.NumVert(); i++)
				{
					if (outMesh.VertProperties[(i * onp) + j] == ext[j])
					{
						found = true;
						break;
					}
				}

				await Assert.That(found)
					.IsTrue()
					.Because(
						$"boolean round trip lost axis-{j} coordinate {ext[j]} "
						+ "(bit-exact match required)");
			}

			// The f64 export must not floor tolerance at f32 epsilon the way the f32
			// export does (C++ GetMeshGL64 vs GetMeshGL).
			MeshGL out32 = result.GetMeshGL(-1);
			await Assert.That(outMesh.Tolerance < (double)out32.Tolerance)
				.IsTrue()
				.Because(
					$"f64 tolerance {outMesh.Tolerance} should be below the f32-floored "
					+ $"{out32.Tolerance}");
		}

		/// <summary>
		/// C++ TEST(Manifold, ObjRoundTrip) — cube → OBJ string → cube, volume preserved.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppObjRoundTrip()
		{
			Manifold m = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false);
			string obj = m.WriteObj();
			Manifold m2 = Manifold.ReadObj(obj);
			await Assert.That(m2.Status())
				.IsEqualTo(Error.NoError)
				.Because($"ObjRoundTrip: status={m2.Status()}");
			await Assert.That(Math.Abs(m2.Volume() - 1.0) < 1e-9)
				.IsTrue()
				.Because($"ObjRoundTrip: volume={m2.Volume()}, expected 1");
		}
	}
}
