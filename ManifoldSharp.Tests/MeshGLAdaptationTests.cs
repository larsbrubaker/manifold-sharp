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

// NOT PORTED FROM RUST. types_meshgl.rs has no `#[cfg(test)]` module and
// types_tests.rs carries no MeshGL coverage, so a test-for-test port of the Rust
// suite produces nothing for this file's subject — yet MeshGL.Merge and
// MeshGL.UpdateNormals are ~250 lines of real behaviour on the library's
// interchange boundary. These tests exist to cover them, and like
// InfrastructureAdaptationTests.cs they do NOT count toward the test-for-test
// tally in docs/PORTING_PLAN.md ("the C# suite ends at the same count as the Rust's
// 763"); that tally counts ported tests, and the ported mesh-core suites are
// ImplMeshTests.cs, SortTests.cs and ColliderTests.cs.
//
// ── Where the expected values come from ──────────────────────────────────────
// Every number below was read off a differential harness run against the
// compiled manifold-rust crate (scratchpad `meshcore_dump`, section `meshgl2`),
// not derived by hand and not recorded from this port's own output. So these are
// Rust-verified expectations, and a failure here is a real divergence.
//
// Float expectations are asserted as raw bit patterns, for two reasons: it is
// the form the harness emits, so an assertion can be traced to a dump line by
// eye; and the normalized results are values like 0.6f whose decimal spelling
// invites an approximate comparison, which is exactly the "close enough" this
// port forbids. `Bits()` below is the only indirection.
//
// ── One correction this file records ─────────────────────────────────────────
// `Merge` returns "true if new merges were found, false if the mesh was already
// fully merged", and "already fully merged" is decided by the `open_edges`
// set being *empty* — not by the merge having found nothing new. A mesh with an
// open boundary therefore keeps returning true on every call while producing an
// identical merge vector. Both halves of that are pinned below, because the
// intuitive reading (second call returns false) is wrong and would otherwise get
// "fixed" into the code some day.
//
// ── Collection ordering ──────────────────────────────────────────────────────
// IsEquivalentTo is passed CollectionOrdering.Matching explicitly throughout.
// TUnit's default is CollectionOrdering.Any — order-INSENSITIVE. That matters
// most for MergeFromVert / MergeToVert, which are *parallel* vectors whose
// pairing is positional: an order-insensitive comparison would accept a merge
// map that welds the wrong vertices onto each other.

using ManifoldSharp;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class MeshGLAdaptationTests
	{
		/// <summary>The raw bits of a float, the form the differential harness prints.</summary>
		private static uint Bits(float v)
		{
			return BitConverter.SingleToUInt32Bits(v);
		}

		/// <summary>The raw bits of a double, the form the differential harness prints.</summary>
		private static ulong Bits(double v)
		{
			return BitConverter.DoubleToUInt64Bits(v);
		}

		/// <summary>The closed unit cube as a MeshGL — every edge is shared, none are open.</summary>
		private static MeshGL ClosedCube()
		{
			MeshGL mesh = new MeshGL();
			mesh.NumProp = 3;
			mesh.VertProperties = new List<float>
			{
				0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
				0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
			};
			mesh.TriVerts = new List<uint>
			{
				0, 2, 1, 0, 3, 2,
				4, 5, 6, 4, 6, 7,
				0, 1, 5, 0, 5, 4,
				1, 2, 6, 1, 6, 5,
				2, 3, 7, 2, 7, 6,
				3, 0, 4, 3, 4, 7,
			};
			mesh.Tolerance = 0.0f;
			return mesh;
		}

		/// <summary>
		/// Two unit squares meeting along x = 1, but the shared edge's two vertices are
		/// duplicated: verts 4 and 5 are exact copies of verts 1 and 2.
		/// </summary>
		private static MeshGL SplitSeam()
		{
			MeshGL mesh = new MeshGL();
			mesh.NumProp = 3;
			mesh.VertProperties = new List<float>
			{
				0.0f, 0.0f, 0.0f,
				1.0f, 0.0f, 0.0f,
				1.0f, 1.0f, 0.0f,
				0.0f, 1.0f, 0.0f,
				1.0f, 0.0f, 0.0f,
				1.0f, 1.0f, 0.0f,
				2.0f, 0.0f, 0.0f,
				2.0f, 1.0f, 0.0f,
			};
			mesh.TriVerts = new List<uint> { 0, 1, 2, 0, 2, 3, 4, 6, 7, 4, 7, 5 };
			mesh.Tolerance = 0.0f;
			return mesh;
		}

		/// <summary>
		/// Six vertices in two runs: run 0 under a pure scale (2, 3, 0.5), run 1 under a
		/// 90-degree rotation about z and flagged backside.
		/// </summary>
		private static MeshGL TwoRuns()
		{
			MeshGL mesh = new MeshGL();
			mesh.NumProp = 6;
			mesh.VertProperties = new List<float>
			{
				0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
				1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
				0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 2.0f,
				0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
				1.0f, 1.0f, 0.0f, 1.0f, 2.0f, 2.0f,
				1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
			};
			mesh.TriVerts = new List<uint> { 0, 1, 2, 3, 4, 5 };
			mesh.RunIndex = new List<uint> { 0, 3, 6 };
			mesh.RunOriginalId = new List<uint> { 10, 20 };
			mesh.RunTransform = new List<float>
			{
				2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f,
				0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 5.0f, 6.0f, 7.0f,
			};
			mesh.RunFlags = new List<byte> { 0, 1 };
			return mesh;
		}

		[Test]
		public async Task MergeOnAClosedMeshReportsAlreadyMerged()
		{
			MeshGL mesh = ClosedCube();
			await Assert.That(mesh.NumVert()).IsEqualTo(8);
			await Assert.That(mesh.NumTri()).IsEqualTo(12);

			// No open edges at all, so Merge takes its early exit and reports false.
			await Assert.That(mesh.Merge()).IsFalse();
			await Assert.That(mesh.MergeFromVert.Count).IsEqualTo(0);
			await Assert.That(mesh.MergeToVert.Count).IsEqualTo(0);
		}

		[Test]
		public async Task MergeWeldsASplitSeamAndIsIdempotent()
		{
			MeshGL mesh = SplitSeam();
			await Assert.That(mesh.NumVert()).IsEqualTo(8);
			await Assert.That(mesh.NumTri()).IsEqualTo(4);

			// The duplicated seam verts 4 and 5 weld onto their originals 1 and 2.
			await Assert.That(mesh.Merge()).IsTrue();
			await Assert.That(mesh.MergeFromVert).IsEquivalentTo(
				new List<uint> { 4, 5 }, CollectionOrdering.Matching);
			await Assert.That(mesh.MergeToVert).IsEquivalentTo(
				new List<uint> { 1, 2 }, CollectionOrdering.Matching);

			// Idempotent in its RESULT. Not in its return value: these two squares have an
			// outer boundary, so open_edges is never empty and Merge never takes the
			// "already fully merged" exit — see the file header. Run it twice more to pin
			// that the merge vector does not grow or permute.
			for (int again = 0; again < 2; again++)
			{
				await Assert.That(mesh.Merge()).IsTrue();
				await Assert.That(mesh.MergeFromVert).IsEquivalentTo(
					new List<uint> { 4, 5 }, CollectionOrdering.Matching);
				await Assert.That(mesh.MergeToVert).IsEquivalentTo(
					new List<uint> { 1, 2 }, CollectionOrdering.Matching);
			}
		}

		[Test]
		public async Task UpdateNormalsAppliesEachRunsTransform()
		{
			MeshGL mesh = TwoRuns();
			await Assert.That(mesh.Backside(0)).IsFalse();
			await Assert.That(mesh.Backside(1)).IsTrue();

			mesh.UpdateNormals(3);

			// Positions (slots 0..2) are untouched; normals (slots 3..5) are transformed
			// and re-normalized, and run 1 is additionally negated for being backside.
			// Note vert 3's -0.0 components (0x80000000): `sign * 0.0 / len` with
			// sign = -1 produces negative zero, and that is the Rust's bit pattern too.
			uint[] expected = new uint[]
			{
				0x00000000, 0x00000000, 0x00000000, 0x3e752c1b, 0x3e2372bd, 0x3f752c1b,
				0x3f800000, 0x00000000, 0x00000000, 0x3f800000, 0x00000000, 0x00000000,
				0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000,
				0x00000000, 0x00000000, 0x3f800000, 0xbf800000, 0x80000000, 0x80000000,
				0x3f800000, 0x3f800000, 0x00000000, 0xbf2aaaab, 0x3eaaaaab, 0xbf2aaaab,
				0x3f800000, 0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000,
			};

			await Assert.That(mesh.VertProperties.Count).IsEqualTo(expected.Length);
			for (int i = 0; i < expected.Length; i++)
			{
				await Assert.That(Bits(mesh.VertProperties[i]))
					.IsEqualTo(expected[i])
					.Because($"vert_properties[{i}]");
			}

			// The happy path consumes the run transforms and flags.
			await Assert.That(mesh.RunTransform.Count).IsEqualTo(0);
			await Assert.That(mesh.RunFlags.Count).IsEqualTo(0);
		}

		[Test]
		public async Task UpdateNormalsFallsBackToIdentityWithoutARunTransform()
		{
			MeshGL mesh = new MeshGL();
			mesh.NumProp = 6;
			mesh.VertProperties = new List<float>
			{
				0.0f, 0.0f, 0.0f, 3.0f, 4.0f, 0.0f,
				1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -2.0f,
				0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
			};
			mesh.TriVerts = new List<uint> { 0, 1, 2 };
			mesh.RunIndex = new List<uint> { 0, 3 };
			mesh.RunOriginalId = new List<uint> { 1 };
			mesh.RunFlags = new List<byte> { 0 };

			mesh.UpdateNormals(3);

			// With no run_transform the normal transform is the identity, so this only
			// re-normalizes: (3,4,0) -> (0.6, 0.8, 0), (0,0,-2) -> (0,0,-1), and the zero
			// normal stays zero because SafeNormalize's length guard catches it.
			uint[] expected = new uint[]
			{
				0x00000000, 0x00000000, 0x00000000, 0x3f19999a, 0x3f4ccccd, 0x00000000,
				0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xbf800000,
				0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
			};

			await Assert.That(mesh.VertProperties.Count).IsEqualTo(expected.Length);
			for (int i = 0; i < expected.Length; i++)
			{
				await Assert.That(Bits(mesh.VertProperties[i]))
					.IsEqualTo(expected[i])
					.Because($"vert_properties[{i}]");
			}
		}

		[Test]
		public async Task UpdateNormalsFallsBackToIdentityForASingularTransform()
		{
			MeshGL mesh = new MeshGL();
			mesh.NumProp = 6;
			mesh.VertProperties = new List<float> { 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 4.0f };
			mesh.TriVerts = new List<uint> { 0, 0, 0 };
			mesh.RunIndex = new List<uint> { 0, 3 };
			mesh.RunOriginalId = new List<uint> { 1 };
			mesh.RunTransform = new List<float>();
			for (int i = 0; i < 12; i++)
			{
				mesh.RunTransform.Add(0.0f);
			}

			mesh.RunFlags = new List<byte> { 1 };

			mesh.UpdateNormals(3);

			// det == 0 takes the |det| < 1e-30 guard, so the normal transform is the
			// identity; the run is backside, so (0,3,4) normalizes to (0,0.6,0.8) and is
			// then negated — including the leading component, which becomes -0.0.
			uint[] expected = new uint[]
			{
				0x00000000, 0x00000000, 0x00000000, 0x80000000, 0xbf19999a, 0xbf4ccccd,
			};

			await Assert.That(mesh.VertProperties.Count).IsEqualTo(expected.Length);
			for (int i = 0; i < expected.Length; i++)
			{
				await Assert.That(Bits(mesh.VertProperties[i]))
					.IsEqualTo(expected[i])
					.Because($"vert_properties[{i}]");
			}
		}

		[Test]
		public async Task UpdateNormalsRejectsOutOfRangeSlots()
		{
			MeshGL mesh = TwoRuns();
			mesh.UpdateNormals(3);

			// Give it transforms and flags again so a wrongly-accepted call would be
			// visible in more than one place.
			mesh.RunTransform = new List<float>();
			for (int i = 0; i < 12; i++)
			{
				mesh.RunTransform.Add(1.0f);
			}

			mesh.RunFlags = new List<byte> { 1 };
			List<float> before = new List<float>(mesh.VertProperties);

			// normalIdx < 3 would overlap the position slots; normalIdx + 3 > numProp
			// would run off the end of a vertex. Both return before touching anything.
			mesh.UpdateNormals(0);
			mesh.UpdateNormals(2);
			mesh.UpdateNormals(4);

			await Assert.That(mesh.VertProperties).IsEquivalentTo(before, CollectionOrdering.Matching);
			await Assert.That(mesh.RunTransform.Count).IsEqualTo(12);
			await Assert.That(mesh.RunFlags.Count).IsEqualTo(1);
		}

		[Test]
		public async Task AccessorsReadTheInterleavedLayout()
		{
			MeshGL empty = new MeshGL();

			// numProp == 0 is the guarded case: a division would throw.
			await Assert.That(empty.NumVert()).IsEqualTo(0);
			await Assert.That(empty.NumTri()).IsEqualTo(0);

			MeshGL mesh = new MeshGL();
			mesh.NumProp = 5;
			mesh.VertProperties = new List<float>
			{
				1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
				-1.5f, -2.5f, -3.5f, -4.5f, -5.5f,
				10.25f, 20.25f, 30.25f, 40.25f, 50.25f,
			};
			mesh.TriVerts = new List<uint> { 2, 0, 1 };
			mesh.HalfedgeTangent = new List<float>
			{
				0.5f, 0.25f, 0.125f, 1.0f,
				-0.5f, -0.25f, -0.125f, 0.0f,
				7.0f, 8.0f, 9.0f, 0.5f,
			};

			await Assert.That(mesh.NumVert()).IsEqualTo(3);
			await Assert.That(mesh.NumTri()).IsEqualTo(1);

			// GetVertPos takes slots 0..2 of a 5-wide vertex, skipping the extra channels.
			await Assert.That(mesh.GetVertPos(0)).IsEqualTo((1.5f, 2.5f, 3.5f));
			await Assert.That(mesh.GetVertPos(1)).IsEqualTo((-1.5f, -2.5f, -3.5f));
			await Assert.That(mesh.GetVertPos(2)).IsEqualTo((10.25f, 20.25f, 30.25f));

			await Assert.That(mesh.GetTriVerts(0)).IsEqualTo((2u, 0u, 1u));

			await Assert.That(mesh.GetTangent(0)).IsEqualTo((0.5f, 0.25f, 0.125f, 1.0f));
			await Assert.That(mesh.GetTangent(1)).IsEqualTo((-0.5f, -0.25f, -0.125f, 0.0f));
			await Assert.That(mesh.GetTangent(2)).IsEqualTo((7.0f, 8.0f, 9.0f, 0.5f));
		}

		[Test]
		public async Task MeshGL64KeepsPrecisionMeshGLWouldLose()
		{
			MeshGL64 empty = new MeshGL64();
			await Assert.That(empty.NumVert()).IsEqualTo(0);
			await Assert.That(empty.NumTri()).IsEqualTo(0);

			// 0.1 + 0.2 is not representable in either width, and its two roundings
			// differ; 1e300 and -1e-300 have no float representation at all.
			double precise = 0.1 + 0.2;
			MeshGL64 mesh = new MeshGL64();
			mesh.NumProp = 4;
			mesh.VertProperties = new List<double>
			{
				precise, 1.0 / 3.0, 1e300, 0.0,
				-precise, 2.0 / 3.0, -1e-300, 1.0,
			};
			mesh.TriVerts = new List<ulong> { 1, 0, 1 };
			mesh.HalfedgeTangent = new List<double> { precise, 0.25, -0.5, 1.0 };
			mesh.Tolerance = precise;

			await Assert.That(mesh.NumVert()).IsEqualTo(2);
			await Assert.That(mesh.NumTri()).IsEqualTo(1);

			(double X, double Y, double Z) p0 = mesh.GetVertPos(0);
			await Assert.That(Bits(p0.X)).IsEqualTo(0x3fd3333333333334UL);
			await Assert.That(Bits(p0.Y)).IsEqualTo(0x3fd5555555555555UL);
			await Assert.That(Bits(p0.Z)).IsEqualTo(0x7e37e43c8800759cUL);

			(double X, double Y, double Z) p1 = mesh.GetVertPos(1);
			await Assert.That(Bits(p1.X)).IsEqualTo(0xbfd3333333333334UL);
			await Assert.That(Bits(p1.Y)).IsEqualTo(0x3fe5555555555555UL);
			await Assert.That(Bits(p1.Z)).IsEqualTo(0x81a56e1fc2f8f359UL);

			// 64-bit indices, and the wide index type is what distinguishes this
			// instantiation from MeshGL's u32.
			await Assert.That(mesh.GetTriVerts(0)).IsEqualTo((1UL, 0UL, 1UL));

			(double X, double Y, double Z, double W) t = mesh.GetTangent(0);
			await Assert.That(Bits(t.X)).IsEqualTo(0x3fd3333333333334UL);
			await Assert.That(Bits(t.Y)).IsEqualTo(Bits(0.25));
			await Assert.That(Bits(t.Z)).IsEqualTo(Bits(-0.5));
			await Assert.That(Bits(t.W)).IsEqualTo(Bits(1.0));

			await Assert.That(Bits(mesh.Tolerance)).IsEqualTo(0x3fd3333333333334UL);

			// The point of the two instantiations: the f32 one would have narrowed the
			// same value to 0x3e99999a. That is the MeshPrecision contract the MeshGL.cs
			// header describes, made observable.
			await Assert.That(Bits((float)precise)).IsEqualTo(0x3e99999au);
		}
	}
}
