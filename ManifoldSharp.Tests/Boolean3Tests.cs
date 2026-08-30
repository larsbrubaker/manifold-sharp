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

// Port of the tests module in boolean3.rs (src/boolean3_tests.rs) — the first
// 15 of its 41 cases, same inputs, same expected values, same tolerances, in
// the same order.
//
// The other 26 all open with `use crate::manifold::Manifold` and drive the
// public façade; they are ported in Boolean3Tests.Manifold.cs now that Phase 6
// has landed it (and Phase 7 the subdivide-backed `Manifold::sphere` that two of
// them need). Nothing in boolean3_tests.rs is deferred any more, so the DEFERRED
// table that used to stand here — a 26-row list keyed by the last missing
// fixture — is gone rather than kept as a record of finished work.
//
// The engine-level tests below and the façade-level ones next door overlap on
// several constructions and are deliberately both kept; the reason is in
// Boolean3Tests.Manifold.cs's header.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class Boolean3Tests
	{
		[Test]
		public async Task ComposeMeshesDisjointCubes()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(3.0, 0.0, 0.0);
			ManifoldImpl c = Boolean3Functions.ComposeMeshes(new[] { a, b });
			await Assert.That(c.NumTri()).IsEqualTo(24);
			await Assert.That(c.NumVert()).IsEqualTo(16);
		}

		[Test]
		public async Task BooleanAddDisjoint()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(3.0, 0.0, 0.0);
			ManifoldImpl c = Boolean3Functions.Boolean(a, b, OpType.Add);
			await Assert.That(c.NumTri()).IsEqualTo(24);
			await Assert.That(c.NumVert()).IsEqualTo(16);
		}

		[Test]
		public async Task BooleanIntersectDisjointEmpty()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(3.0, 0.0, 0.0);
			ManifoldImpl c = Boolean3Functions.Boolean(a, b, OpType.Intersect);
			await Assert.That(c.IsEmpty()).IsTrue();
		}

		[Test]
		public async Task BooleanSubtractDisjoint()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(3.0, 0.0, 0.0);
			ManifoldImpl c = Boolean3Functions.Boolean(a, b, OpType.Subtract);
			await Assert.That(c.NumTri()).IsEqualTo(12);
		}

		// Boolean3 intersection tests — verify data structures before result assembly
		[Test]
		public async Task Boolean3NoOverlap()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(3.0, 0.0, 0.0);
			Boolean3 bool3 = Boolean3.New(a, b, OpType.Add);
			await Assert.That(bool3.Valid).IsTrue();
			await Assert.That(bool3.Xv12.P1q2).IsEmpty();
			await Assert.That(bool3.Xv21.P1q2).IsEmpty();
			await Assert.That(bool3.W03.Count).IsEqualTo(a.NumVert());
			await Assert.That(bool3.W30.Count).IsEqualTo(b.NumVert());
			await Assert.That(bool3.W03.All(w => w == 0)).IsTrue();
			await Assert.That(bool3.W30.All(w => w == 0)).IsTrue();
		}

		[Test]
		public async Task Boolean3OverlappingCubes()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(0.5, 0.3, 0.2);
			Boolean3 bool3 = Boolean3.New(a, b, OpType.Add);
			await Assert.That(bool3.Valid).IsTrue();

			// With overlapping cubes, there should be edge-face intersections
			await Assert.That(bool3.Xv12.P1q2.Count > 0 || bool3.Xv21.P1q2.Count > 0)
				.IsTrue()
				.Because("Overlapping cubes should produce intersections");
		}

		[Test]
		public async Task CubeHasVertNormals()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			Console.WriteLine($"Cube has {a.VertNormal.Count} vert_normals for {a.NumVert()} verts");
			for (int i = 0; i < a.VertNormal.Count; i++)
			{
				Vec3 n = a.VertNormal[i];
				Console.WriteLine($"  normal[{i}] = ({n.X:F4}, {n.Y:F4}, {n.Z:F4})");
			}

			await Assert.That(a.VertNormal.Count)
				.IsEqualTo(a.NumVert())
				.Because("vert_normal should be populated");
		}

		/// <summary>Two unit cubes overlapping — offset avoids exact boundary alignment.</summary>
		[Test]
		public async Task BooleanUnionOverlappingCubes()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(0.5, 0.3, 0.2);
			ManifoldImpl result = Boolean3Functions.Boolean(a, b, OpType.Add);
			double expectedVol = 2.0 - (0.5 * 0.7 * 0.8); // 2 - overlap
			await Assert.That(result.IsEmpty()).IsFalse().Because("Union should not be empty");
			double vol = Math.Abs(result.GetProperty(Property.Volume));
			await Assert.That(Math.Abs(vol - expectedVol))
				.IsLessThan(0.05)
				.Because($"Union volume should be ~{expectedVol:F3}, got {vol}");
		}

		/// <summary>Two unit cubes, offset to avoid degenerate geometry.</summary>
		[Test]
		public async Task BooleanIntersectOverlappingCubes()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(0.5, 0.3, 0.2);
			ManifoldImpl result = Boolean3Functions.Boolean(a, b, OpType.Intersect);
			double expectedVol = 0.5 * 0.7 * 0.8;
			await Assert.That(result.IsEmpty()).IsFalse().Because("Intersection should not be empty");
			double vol = Math.Abs(result.GetProperty(Property.Volume));
			await Assert.That(Math.Abs(vol - expectedVol))
				.IsLessThan(0.05)
				.Because($"Intersection volume should be ~{expectedVol:F3}, got {vol}");
		}

		/// <summary>Two unit cubes, offset to avoid degenerate geometry.</summary>
		[Test]
		public async Task BooleanSubtractOverlappingCubes()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(0.5, 0.3, 0.2);
			ManifoldImpl result = Boolean3Functions.Boolean(a, b, OpType.Subtract);
			double expectedVol = 1.0 - (0.5 * 0.7 * 0.8);
			await Assert.That(result.IsEmpty()).IsFalse().Because("Difference should not be empty");
			double vol = Math.Abs(result.GetProperty(Property.Volume));
			await Assert.That(Math.Abs(vol - expectedVol))
				.IsLessThan(0.05)
				.Because($"Difference volume should be ~{expectedVol:F3}, got {vol}");
		}

		/// <summary>Union of two identical cubes at offset 0 (fully overlapping / degenerate).</summary>
		[Test]
		public async Task BooleanUnionSamePosition()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(0.0, 0.0, 0.0);
			ManifoldImpl result = Boolean3Functions.Boolean(a, b, OpType.Add);
			double vol = Math.Abs(result.GetProperty(Property.Volume));
			await Assert.That(Math.Abs(vol - 1.0))
				.IsLessThan(0.1)
				.Because($"Union of identical cubes should have volume ~1.0, got {vol}");
		}

		/// <summary>Intersection at offset=1.0 (cubes touching at a face).</summary>
		[Test]
		public async Task BooleanIntersectTouching()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(1.0, 0.0, 0.0);
			ManifoldImpl result = Boolean3Functions.Boolean(a, b, OpType.Intersect);

			// Touching cubes have zero-volume intersection
			double vol = Math.Abs(result.GetProperty(Property.Volume));
			await Assert.That(vol)
				.IsLessThan(0.01)
				.Because($"Intersection of touching cubes should have ~0 volume, got {vol}");
		}

		/// <summary>Intersection of non-overlapping cubes should return empty.</summary>
		[Test]
		public async Task BooleanIntersectDisjointReturnsEmpty()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(5.0, 0.0, 0.0);
			ManifoldImpl result = Boolean3Functions.Boolean(a, b, OpType.Intersect);
			await Assert.That(result.IsEmpty())
				.IsTrue()
				.Because("Intersection of disjoint cubes should be empty");
		}

		/// <summary>Union with small overlap (offset 0.9).</summary>
		[Test]
		public async Task BooleanUnionSmallOverlap()
		{
			ManifoldImpl a = Cube(0.0, 0.0, 0.0);
			ManifoldImpl b = Cube(0.9, 0.0, 0.0);
			ManifoldImpl result = Boolean3Functions.Boolean(a, b, OpType.Add);
			double expectedVol = 2.0 - 0.1; // overlap = 0.1 * 1 * 1
			await Assert.That(result.IsEmpty()).IsFalse().Because("Union should not be empty");
			double vol = Math.Abs(result.GetProperty(Property.Volume));
			await Assert.That(Math.Abs(vol - expectedVol))
				.IsLessThan(0.1)
				.Because($"Union volume should be ~{expectedVol:F3}, got {vol}");
		}

		/// <summary>Intersection at various offsets.</summary>
		/// <remarks>
		/// The Rust wraps each call in <c>catch_unwind</c> and only reports — it asserts
		/// nothing, and exists to show that no offset in the sweep aborts. The C#
		/// counterpart of "the panic did not escape" is "no exception escaped", so the
		/// try/catch is the port of the <c>catch_unwind</c>, not a swallowed failure:
		/// there is no assertion here to weaken.
		/// </remarks>
		[Test]
		public async Task BooleanIntersectVariousOffsets()
		{
			int swept = 0;
			foreach (double offset in new[] { 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5 })
			{
				swept += 1;
				ManifoldImpl a = Cube(0.0, 0.0, 0.0);
				ManifoldImpl b = Cube(offset, 0.0, 0.0);
				try
				{
					ManifoldImpl r = Boolean3Functions.Boolean(a, b, OpType.Intersect);
					double vol = Math.Abs(r.GetProperty(Property.Volume));
					Console.WriteLine(
						$"Intersect offset={offset}: vol={vol:F4} verts={r.NumVert()} "
						+ $"tris={r.NumTri()} empty={r.IsEmpty()}");
				}
				catch (Exception e)
				{
					Console.WriteLine($"Intersect offset={offset}: PANIC {e.Message}");
				}
			}

			// The Rust's loop body has no assertion. TUnit rejects asserting a constant, so
			// the one assertion here is the sweep's own completion count — still nothing
			// about geometry, which is the Rust's point.
			await Assert.That(swept).IsEqualTo(8);
		}

		private static ManifoldImpl Cube(double x, double y, double z)
		{
			return ManifoldImpl.Cube(LinalgFunctions.Mat4ToMat3x4(
				LinalgFunctions.TranslationMatrix(new Vec3(x, y, z))));
		}
	}
}
