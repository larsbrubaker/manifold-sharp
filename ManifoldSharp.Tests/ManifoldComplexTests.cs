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

// Port of src/manifold_tests/complex.rs — all 24 of its tests, in the Rust's
// order, with the same inputs, the same expected values and the same
// tolerances. Nothing is deferred: complex.rs is the C++ BooleanComplex suite,
// and every one of its booleans runs on the exact engine that is already ported.
//
// Six files, for the 800-line cap. The class is one `partial`; the split is by
// what a test needs, not by anything the Rust separates:
//
//   ManifoldComplexTests.cs                    the OBJ-model regressions and the
//                                              inline-mesh Subtract (tests 1–8)
//   ManifoldComplexTests.Shapes.cs             Cylinders, Close, LazyCollider,
//                                              Perturb2, Perturb3 (9–13)
//   ManifoldComplexTests.Relations.cs          the MeshGL round trips, Craycloud,
//                                              BooleanVolumes, Spiral,
//                                              OpenscadCrash, Sphere,
//                                              MeshRelation (14–21)
//   ManifoldComplexTests.Sweep.cs              Sweep and its 90 path points (22)
//   ManifoldComplexTests.InterpolatedNormals.cs  test 23 and its transcribed meshes
//   ManifoldComplexTests.Ring.cs                 test 24 and its transcribed meshes
//
// ── The two ignores, kept ────────────────────────────────────────────────────
// complex.rs carries two of the Rust suite's nine `#[ignore]`s, and both come
// over as [Skip] with the Rust reason string verbatim:
//
//   GenericTwinBooleanTest7081  debug-speed only. Verified: run here in Release
//                               with the [Skip] removed it passes in 22.3s,
//                               which is the reason string's claim.
//   Perturb3                    "Gear pattern requires BatchBoolean precision
//                               improvements". Verified with the [Skip] removed:
//                               it PASSES here, in both Debug and Release. So
//                               does the Rust — `cargo test --release
//                               test_cpp_perturb3 -- --ignored` is green on
//                               manifold-rust v0.14.0. The reason string is
//                               stale upstream, not a description of current
//                               behavior. The ignore is kept anyway, because
//                               this suite ends at the Rust's count with the
//                               Rust's ignores; un-skipping it is a decision for
//                               manifold-rust to make first, and it belongs in
//                               docs/RUST_DIVERGENCES.md if it is ever made here.
//
// ── Where the meshes come from ───────────────────────────────────────────────
// complex.rs reads two kinds of input the Rust pulls out of its cpp-reference
// submodule at test time. Neither dependency exists here (see
// CLAUDE.md, "Verification nets" #1):
//
//   OBJ models      checked in under TestData/models, loaded by
//                   ManifoldTestHelpers.ReadTestObj.
//   Inline C++ mesh literals (InterpolatedNormals, Ring)
//                   transcribed into the two data-carrying partials above, each
//                   of which documents its C++ file and commit pin.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	/// <summary>
	/// The C++ BooleanComplex suite, as ported in <c>manifold_tests/complex.rs</c>.
	/// </summary>
	public partial class ManifoldComplexTests
	{
		/// <summary>
		/// C++ TEST(BooleanComplex, SelfIntersect) — read_test_obj, union, get_mesh_gl
		/// (test crash).
		/// </summary>
		[Test]
		public void CppComplexSelfIntersect()
		{
			Manifold m1 = ManifoldTestHelpers.ReadTestObj("self_intersectA.obj");
			Manifold m2 = ManifoldTestHelpers.ReadTestObj("self_intersectB.obj");
			Manifold res = m1.Union(m2);
			res.GetMeshGL(0); // test that it doesn't crash
		}

		/// <summary>C++ TEST(BooleanComplex, GenericTwinBooleanTest7081).</summary>
		[Test]
		[Skip("Slow in debug only — passes in release (~18s; C++ takes ~50s). No longer hangs "
			+ "after the RecursiveEdgeSwap FormLoop / edge<0 fix in edge_op.rs. Kept ignored "
			+ "for debug-suite speed, like sdf_blobs/hull_sphere.")]
		public void CppComplexGenericTwin7081()
		{
			Manifold m1 = ManifoldTestHelpers.ReadTestObj("Generic_Twin_7081.1.t0_left.obj");
			Manifold m2 = ManifoldTestHelpers.ReadTestObj("Generic_Twin_7081.1.t0_right.obj");
			Manifold res = m1.Union(m2);
			res.GetMeshGL(0); // test crash
		}

		/// <summary>C++ TEST(BooleanComplex, GenericTwinBooleanTest7863).</summary>
		[Test]
		public void CppComplexGenericTwin7863()
		{
			Manifold m1 = ManifoldTestHelpers.ReadTestObj("Generic_Twin_7863.1.t0_left.obj");
			Manifold m2 = ManifoldTestHelpers.ReadTestObj("Generic_Twin_7863.1.t0_right.obj");
			Manifold res = m1.Union(m2);
			res.GetMeshGL(0); // test crash
		}

		/// <summary>C++ TEST(BooleanComplex, Havocglass8Bool).</summary>
		[Test]
		public void CppComplexHavocglass8()
		{
			Manifold m1 = ManifoldTestHelpers.ReadTestObj("Havocglass8_left.obj");
			Manifold m2 = ManifoldTestHelpers.ReadTestObj("Havocglass8_right.obj");
			Manifold res = m1.Union(m2);
			res.GetMeshGL(0); // test crash
		}

		/// <summary>C++ TEST(BooleanComplex, HullMask).</summary>
		[Test]
		public void CppComplexHullMask()
		{
			Manifold body = ManifoldTestHelpers.ReadTestObj("hull-body.obj");
			Manifold mask = ManifoldTestHelpers.ReadTestObj("hull-mask.obj");
			Manifold ret = body.Difference(mask);
			ret.GetMeshGL(0);
		}

		/// <summary>C++ TEST(BooleanComplex, OffsetTriangulationFailure).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppComplexOffsetTriangulationFailure()
		{
			Manifold a = ManifoldTestHelpers.ReadTestObj("Offset1.obj");
			Manifold b = ManifoldTestHelpers.ReadTestObj("Offset2.obj");
			Manifold result = a.Union(b);
			await Assert.That(result.Status())
				.IsEqualTo(Error.NoError)
				.Because($"OffsetTriangulationFailure: status {result.Status()}");
		}

		/// <summary>C++ TEST(BooleanComplex, OffsetSelfIntersect).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppComplexOffsetSelfIntersect()
		{
			Manifold a = ManifoldTestHelpers.ReadTestObj("Offset3.obj");
			Manifold b = ManifoldTestHelpers.ReadTestObj("Offset4.obj");
			Manifold result = a.Union(b);
			await Assert.That(result.Status())
				.IsEqualTo(Error.NoError)
				.Because($"OffsetSelfIntersect: status {result.Status()}");
		}

		/// <summary>C++ TEST(BooleanComplex, Subtract) — specific mesh subtract.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppComplexSubtract()
		{
			List<float> firstVerts = new List<float>
			{
				0.0f, 0.0f, 0.0f, 1540.0f, 0.0f, 0.0f, 1540.0f, 70.0f, 0.0f, 0.0f, 70.0f, 0.0f,
				0.0f, 0.0f, -278.282f, 1540.0f, 70.0f, -278.282f, 1540.0f, 0.0f, -278.282f,
				0.0f, 70.0f, -278.282f,
			};
			List<uint> firstTris = new List<uint>
			{
				0, 1, 2, 2, 3, 0, 4, 5, 6, 5, 4, 7, 6, 2, 1, 6, 5, 2, 5, 3, 2, 5, 7, 3, 7, 0, 3,
				7, 4, 0, 4, 1, 0, 4, 6, 1,
			};

			MeshGL firstMesh = new MeshGL();
			firstMesh.NumProp = 3;
			firstMesh.VertProperties = firstVerts;
			firstMesh.TriVerts = firstTris;

			List<float> secondVerts = new List<float>
			{
				2.04636e-12f,
				70.0f,
				50000.0f,
				2.04636e-12f,
				-1.27898e-13f,
				50000.0f,
				1470.0f,
				-1.27898e-13f,
				50000.0f,
				1540.0f,
				70.0f,
				50000.0f,
				2.04636e-12f,
				70.0f,
				-28.2818f,
				1470.0f,
				-1.27898e-13f,
				0.0f,
				2.04636e-12f,
				-1.27898e-13f,
				0.0f,
				1540.0f,
				70.0f,
				-28.2818f,
			};
			List<uint> secondTris = new List<uint>
			{
				0, 1, 2, 2, 3, 0, 4, 5, 6, 5, 4, 7, 6, 2, 1, 6, 5, 2, 5, 3, 2, 5, 7, 3, 7, 0, 3,
				7, 4, 0, 4, 1, 0, 4, 6, 1,
			};

			MeshGL secondMesh = new MeshGL();
			secondMesh.NumProp = 3;
			secondMesh.VertProperties = secondVerts;
			secondMesh.TriVerts = secondTris;

			Manifold first = Manifold.FromMeshGL(firstMesh);
			Manifold second = Manifold.FromMeshGL(secondMesh);

			first = first.Difference(second);
			first.GetMeshGL(0);
			await Assert.That(first.Status()).IsEqualTo(Error.NoError);
		}
	}
}
