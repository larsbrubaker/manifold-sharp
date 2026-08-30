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

// Port of the tests module in face_op.rs — same 6 cases, same expected values,
// same tolerances, in the same order.
//
// face_op_triangulate.rs has no `#[cfg(test)]` module of its own; Face2Tri is
// exercised by the Rust's boolean-result tests. Those landed with Phase 5, so
// that coverage is here: boolean_result_assemble.rs — the only caller — is
// BooleanResultAssemble.cs, and every case in Boolean3Tests,
// ManifoldBooleanTests and ManifoldValidationTests runs Face2Tri over a real
// boolean's retriangulation. Face2Tri's parity is no longer held only by the
// differential harness face_op's own step ran against the compiled Rust.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class FaceOpTests
	{
		[Test]
		public async Task GetAxisAlignedProjectionZ()
		{
			// Normal primarily in Z: should project to XY plane
			Proj2x3 proj = FaceOp.GetAxisAlignedProjection(new Vec3(0.0, 0.0, 1.0));
			Vec3 v = new Vec3(3.0, 4.0, 5.0);
			Vec2 p = proj.Apply(v);
			await Assert.That(Math.Abs(p.X - 3.0) < 1e-12).IsTrue();
			await Assert.That(Math.Abs(p.Y - 4.0) < 1e-12).IsTrue();
		}

		[Test]
		public async Task GetAxisAlignedProjectionY()
		{
			// Normal primarily in Y: should project to ZX plane
			Proj2x3 proj = FaceOp.GetAxisAlignedProjection(new Vec3(0.0, 1.0, 0.0));
			Vec3 v = new Vec3(3.0, 4.0, 5.0);
			Vec2 p = proj.Apply(v);
			await Assert.That(Math.Abs(p.X - 5.0) < 1e-12).IsTrue().Because($"expected z=5, got {p.X}");
			await Assert.That(Math.Abs(p.Y - 3.0) < 1e-12).IsTrue().Because($"expected x=3, got {p.Y}");
		}

		[Test]
		public async Task GetAxisAlignedProjectionX()
		{
			// Normal primarily in X: should project to YZ plane
			Proj2x3 proj = FaceOp.GetAxisAlignedProjection(new Vec3(1.0, 0.0, 0.0));
			Vec3 v = new Vec3(3.0, 4.0, 5.0);
			Vec2 p = proj.Apply(v);
			await Assert.That(Math.Abs(p.X - 4.0) < 1e-12).IsTrue().Because($"expected y=4, got {p.X}");
			await Assert.That(Math.Abs(p.Y - 5.0) < 1e-12).IsTrue().Because($"expected z=5, got {p.Y}");
		}

		[Test]
		public async Task GetAxisAlignedProjectionNegativeZ()
		{
			// Normal primarily in -Z: row0 should be flipped
			Proj2x3 proj = FaceOp.GetAxisAlignedProjection(new Vec3(0.0, 0.0, -1.0));
			Vec3 v = new Vec3(3.0, 4.0, 5.0);
			Vec2 p = proj.Apply(v);

			// Flipped first row: x → -x
			await Assert.That(Math.Abs(p.X + 3.0) < 1e-12).IsTrue().Because($"expected -x=-3, got {p.X}");
			await Assert.That(Math.Abs(p.Y - 4.0) < 1e-12).IsTrue();
		}

		[Test]
		public async Task SetNormalsTetrahedron()
		{
			ManifoldImpl m = ManifoldImpl.Tetrahedron(Mat3x4.Identity());
			FaceOp.SetNormalsAndCoplanar(m);

			// Every face normal should be unit length
			foreach (Vec3 n in m.FaceNormal)
			{
				double len = Math.Sqrt(LinalgFunctions.LengthSquared(n));
				await Assert.That(Math.Abs(len - 1.0) < 1e-10)
					.IsTrue()
					.Because($"face normal not unit: len={len}");
			}

			// Every vert normal should be nonzero (tetrahedron has no degenerate verts)
			foreach (Vec3 n in m.VertNormal)
			{
				double len = Math.Sqrt(LinalgFunctions.LengthSquared(n));
				await Assert.That(len > 0.0).IsTrue().Because("vert normal is zero");
			}
		}

		[Test]
		public async Task SetNormalsCube()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			FaceOp.SetNormalsAndCoplanar(m);
			await Assert.That(m.FaceNormal.Count).IsEqualTo(m.NumTri());

			// Coplanar IDs should be assigned
			bool anyCoplanarIdSet = m.MeshRelation.TriRef.Any(r => r.CoplanarId >= 0);
			await Assert.That(anyCoplanarIdSet).IsTrue().Because("no coplanar IDs were set");
		}
	}
}
