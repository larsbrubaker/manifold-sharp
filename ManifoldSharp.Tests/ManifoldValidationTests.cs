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

// Port of src/manifold_tests/validation.rs — all 30 of its tests, in the Rust's
// order, with the same inputs, expected values and tolerances. Nothing is
// deferred: every boolean here runs on the exact engine, which is ported.
//
// Three files, for the 800-line cap. One `partial` class, split by what the Rust
// itself separates with banner comments:
//
//   ManifoldValidationTests.cs              the NumUnique helper, InvalidInput1–7,
//                                           Invalid, FaceIDRoundTrip and the
//                                           Manifold MeshGLRoundTrip (tests 1–10)
//   ManifoldValidationTests.Booleans.cs      Perturb, EdgeUnion, AlmostCoplanar,
//                                           Coplanar, the Boolean MeshGLRoundTrip,
//                                           and the constructor/geometry group —
//                                           Sphere, Slice, SliceEmptyObject,
//                                           Project (11–19)
//   ManifoldValidationTests.Regressions.cs   the OBJ-model booleans and the
//                                           remaining C++ regressions — Warp2,
//                                           Precision2, TreeTransforms, Normals,
//                                           EmptyOriginal (20–30)
//
// ── The one ignore, kept ─────────────────────────────────────────────────────
// validation.rs carries one of the Rust suite's nine `#[ignore]`s —
// GenericTwinBooleanTest7081, debug-speed only — and it comes over as [Skip]
// with the reason string verbatim. (complex.rs has the same test with the same
// ignore; both are ported, both stay skipped, because the Rust has both.)
//
// ── Shared fixtures ──────────────────────────────────────────────────────────
// The Rust reaches sideways into sibling test modules: `super::api::tet_gl()`,
// `super::api::cube_uv()`, `super::cube_stl()`, `super::related_gl*` and
// `super::read_test_obj`. Those land here as ManifoldApiTests.TetGl / .CubeUv
// (internal, exactly as `pub(super)` in the Rust) and ManifoldTestHelpers.*.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	/// <summary>
	/// The C++ input-validation, constructor and boolean-regression tests, as ported in
	/// <c>manifold_tests/validation.rs</c>.
	/// </summary>
	public partial class ManifoldValidationTests
	{
		/// <summary>How many distinct values a list holds — the Rust's <c>num_unique</c>.</summary>
		/// <param name="vals">The values.</param>
		/// <returns>The count of distinct values.</returns>
		private static int NumUnique(IReadOnlyList<uint> vals)
		{
			return new HashSet<uint>(vals).Count;
		}

		/// <summary>C++ TEST(Manifold, InvalidInput1) — NaN vertex.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidInput1()
		{
			MeshGL gl = ManifoldApiTests.TetGl();

			// Rust `f32::NAN` is the POSITIVE quiet NaN (0x7fc00000); C# `float.NaN`
			// carries the sign bit (0xffc00000). Same rule as the port's f64 NaN — build
			// it from bits so the import sees the byte pattern the Rust hands it.
			gl.VertProperties[(2 * 5) + 1] = BitConverter.Int32BitsToSingle(unchecked((int)0x7fc00000)); // 5 props per vert
			Manifold tet = Manifold.FromMeshGL(gl);
			await Assert.That(tet.IsEmpty()).IsTrue();
			await Assert.That(tet.Status()).IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>
		/// C++ TEST(Manifold, InvalidInput2) — swapped tri indices → not manifold.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidInput2()
		{
			MeshGL gl = ManifoldApiTests.TetGl();
			(gl.TriVerts[(2 * 3) + 1], gl.TriVerts[(2 * 3) + 2]) =
				(gl.TriVerts[(2 * 3) + 2], gl.TriVerts[(2 * 3) + 1]);
			Manifold tet = Manifold.FromMeshGL(gl);
			await Assert.That(tet.IsEmpty()).IsTrue();
			await Assert.That(tet.Status()).IsEqualTo(Error.NotManifold);
		}

		/// <summary>
		/// C++ TEST(Manifold, InvalidInput3) — vertex index = -2 (wraps to huge u32).
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidInput3()
		{
			MeshGL gl = ManifoldApiTests.TetGl();

			// C++ sets uint32_t to -2 which wraps; in Rust u32 wrapping: (-2i32 as u32)
			uint badVal = unchecked((uint)(-2));
			for (int i = 0; i < gl.TriVerts.Count; i++)
			{
				if (gl.TriVerts[i] == 2)
				{
					gl.TriVerts[i] = badVal;
				}
			}

			Manifold tet = Manifold.FromMeshGL(gl);
			await Assert.That(tet.IsEmpty()).IsTrue();
			await Assert.That(tet.Status()).IsEqualTo(Error.VertexOutOfBounds);
		}

		/// <summary>
		/// C++ TEST(Manifold, InvalidInput4) — vertex index = 4 (out of bounds) → not
		/// manifold.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidInput4()
		{
			MeshGL gl = ManifoldApiTests.TetGl();
			for (int i = 0; i < gl.TriVerts.Count; i++)
			{
				if (gl.TriVerts[i] == 2)
				{
					gl.TriVerts[i] = 4;
				}
			}

			Manifold tet = Manifold.FromMeshGL(gl);
			await Assert.That(tet.IsEmpty()).IsTrue();
			await Assert.That(tet.Status()).IsEqualTo(Error.NotManifold);
		}

		/// <summary>C++ TEST(Manifold, InvalidInput5) — merge index out of bounds.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidInput5()
		{
			MeshGL gl = ManifoldApiTests.TetGl();
			int last = gl.MergeFromVert.Count - 1;
			gl.MergeFromVert[last] = 7;
			Manifold tet = Manifold.FromMeshGL(gl);
			await Assert.That(tet.IsEmpty()).IsTrue();
			await Assert.That(tet.Status()).IsEqualTo(Error.MergeIndexOutOfBounds);
		}

		/// <summary>C++ TEST(Manifold, InvalidInput6) — tri_verts index out of bounds.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidInput6()
		{
			MeshGL gl = ManifoldApiTests.TetGl();
			int last = gl.TriVerts.Count - 1;
			gl.TriVerts[last] = 7;
			Manifold tet = Manifold.FromMeshGL(gl);
			await Assert.That(tet.IsEmpty()).IsTrue();
			await Assert.That(tet.Status()).IsEqualTo(Error.VertexOutOfBounds);
		}

		/// <summary>C++ TEST(Manifold, InvalidInput7) — run_index wrong length.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidInput7()
		{
			MeshGL gl = ManifoldApiTests.CubeUv();
			gl.RunIndex = new List<uint> { 0, 1, (uint)gl.TriVerts.Count };
			Manifold m = Manifold.FromMeshGL(gl);
			await Assert.That(m.IsEmpty()).IsTrue();
			await Assert.That(m.Status()).IsEqualTo(Error.RunIndexWrongLength);
		}

		/// <summary>C++ TEST(Manifold, Invalid) — invalid constructor parameters.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalid()
		{
			Error invalid = Error.InvalidConstruction;

			await Assert.That(Manifold.Sphere(0.0, 0).Status()).IsEqualTo(invalid).Because("Sphere(0)");
			await Assert.That(Manifold.Cylinder(0.0, 5.0, -1.0, 0).Status())
				.IsEqualTo(invalid)
				.Because("Cylinder(0,5)");
			await Assert.That(Manifold.Cylinder(2.0, -5.0, -1.0, 0).Status())
				.IsEqualTo(invalid)
				.Because("Cylinder(2,-5)");
			await Assert.That(Manifold.Cylinder(2.0, 0.0, -1.0, 0).Status())
				.IsEqualTo(invalid)
				.Because("Cylinder(2,0)");
			await Assert.That(Manifold.Cylinder(2.0, 0.0, 0.0, 0).Status())
				.IsEqualTo(invalid)
				.Because("Cylinder(2,0,0)");
			await Assert.That(Manifold.Cube(new Vec3(0.0, 0.0, 0.0), false).Status())
				.IsEqualTo(invalid)
				.Because("Cube(0)");
			await Assert.That(Manifold.Cube(new Vec3(-1.0, 1.0, 1.0), false).Status())
				.IsEqualTo(invalid)
				.Because("Cube(-1,1,1)");

			// Extrude with zero height
			CrossSection circ = CrossSection.Circle(10.0, 0);
			await Assert.That(
					Manifold.Extrude(circ.ToPolygons(), 0.0, 0, 0.0, new Vec2(1.0, 1.0)).Status())
				.IsEqualTo(invalid)
				.Because("Extrude(h=0)");

			// Extrude with empty cross-section (negative radius → empty)
			CrossSection emptyCirc = CrossSection.Circle(-2.0, 0);
			await Assert.That(
					Manifold.Extrude(emptyCirc.ToPolygons(), 10.0, 0, 0.0, new Vec2(1.0, 1.0)).Status())
				.IsEqualTo(invalid)
				.Because("Extrude(empty)");

			// Revolve with empty cross-section
			CrossSection emptySq = CrossSection.Square(0.0);
			await Assert.That(Manifold.Revolve(emptySq.ToPolygons(), 0, 360.0).Status())
				.IsEqualTo(invalid)
				.Because("Revolve(empty)");
		}

		/// <summary>C++ TEST(Manifold, FaceIDRoundTrip) — custom face IDs preserved.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppFaceIdRoundTrip()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			await Assert.That(cube.OriginalId() >= 0).IsTrue();
			MeshGL inGl = cube.GetMeshGL(0);
			await Assert.That(NumUnique(inGl.FaceId))
				.IsEqualTo(6)
				.Because("Cube should have 6 unique face IDs");

			// Set custom face IDs: first 6 tris = face 3, last 6 tris = face 5
			inGl.FaceId = new List<uint> { 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5 };

			Manifold cube2 = Manifold.FromMeshGL(inGl);
			MeshGL outGl = cube2.GetMeshGL(0);
			await Assert.That(NumUnique(outGl.FaceId))
				.IsEqualTo(2)
				.Because("Should have 2 unique face IDs after round-trip");
		}

		/// <summary>
		/// C++ TEST(Manifold, MeshGLRoundTrip) — cylinder round-trip preserves originalID.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppManifoldMeshglRoundTrip()
		{
			Manifold cylinder = Manifold.Cylinder(2.0, 1.0, -1.0, 0);
			await Assert.That(cylinder.OriginalId() >= 0).IsTrue();
			MeshGL inGl = cylinder.GetMeshGL(0);
			Manifold cylinder2 = Manifold.FromMeshGL(inGl);
			MeshGL outGl = cylinder2.GetMeshGL(0);

			await Assert.That(inGl.RunOriginalId.Count).IsEqualTo(1);
			await Assert.That(outGl.RunOriginalId.Count).IsEqualTo(1);
			await Assert.That(outGl.RunOriginalId[0]).IsEqualTo(inGl.RunOriginalId[0]);
		}
	}
}
