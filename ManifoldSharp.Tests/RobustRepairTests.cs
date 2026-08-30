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

// RobustRepairTests.cs — port of robust/repair_tests.rs, whose header reads:
//
//   Unit tests for shell-level orientation repair (`robust/repair.rs`), on
//   constructed fixtures where the correct flip set is known exactly:
//   inverted bodies must rewind, legitimate cavities must survive, doubled
//   sheets must be left to the boolean's stack arithmetic.
//
// Same inputs, same expected values, same order as the Rust's eleven tests.

using ManifoldSharp;
using ManifoldSharp.Linalg;
using ManifoldSharp.Robust;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class RobustRepairTests
	{
		[Test]
		public async Task CorrectlyWoundCubeIsUntouched()
		{
			RepairPlan plan = Repair.PlanRepair(CubeTris(0.0, 2.0));
			await Assert.That(plan.NumShells).IsEqualTo(1);
			await Assert.That(plan.FlippedShells).IsEqualTo(0);
			await Assert.That(plan.Flip.All(f => !f)).IsTrue();
		}

		[Test]
		public async Task InvertedCubeIsFlipped()
		{
			RepairPlan plan = Repair.PlanRepair(Flipped(CubeTris(0.0, 2.0)));
			await Assert.That(plan.NumShells).IsEqualTo(1);
			await Assert.That(plan.FlippedShells).IsEqualTo(1);
			await Assert.That(plan.Flip.All(f => f)).IsTrue();
		}

		[Test]
		public async Task LegitimateCavityIsPreserved()
		{
			// Outer [0,6] outward + inner [2,4] inward = solid with a void: correct
			// as-is, and the classic failure case for signed-volume blanket flips.
			List<Vec3[]> tris = CubeTris(0.0, 6.0);
			tris.AddRange(Flipped(CubeTris(2.0, 4.0)));
			RepairPlan plan = Repair.PlanRepair(tris);
			await Assert.That(plan.NumShells).IsEqualTo(2);
			await Assert.That(plan.FlippedShells).IsEqualTo(0)
				.Because("cavity must not be flipped");
		}

		[Test]
		public async Task FullyInvertedNestedPairIsRepaired()
		{
			// Outer inverted + inner outward: both wrong for a solid-with-void.
			List<Vec3[]> tris = Flipped(CubeTris(0.0, 6.0));
			tris.AddRange(CubeTris(2.0, 4.0));
			RepairPlan plan = Repair.PlanRepair(tris);
			await Assert.That(plan.NumShells).IsEqualTo(2);
			await Assert.That(plan.FlippedShells).IsEqualTo(2);
			await Assert.That(plan.Flip.All(f => f)).IsTrue();
		}

		/// <summary>
		/// The other half of the ambiguity table above: same nesting, but the outer shell is
		/// *correct*. Two outward-wound shells, one inside the other, are a valid solid whose
		/// inner region simply winds 2 — there is no inversion to undo, and rewinding the
		/// inner one would carve real material out (the Thingi10K #61459 failure). Evidence of
		/// a mirrored stack is required before a material-removing flip, and there is none here.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task NestedOutwardSolidIsNotTurnedIntoACavity()
		{
			List<Vec3[]> tris = CubeTris(0.0, 6.0);
			tris.AddRange(CubeTris(2.0, 4.0));
			RepairPlan plan = Repair.PlanRepair(tris);
			await Assert.That(plan.NumShells).IsEqualTo(2);
			await Assert.That(plan.FlippedShells).IsEqualTo(0)
				.Because("a nested outward solid must keep its material");
		}

		[Test]
		public async Task SolidNestedInsideCavityWindsPositiveAgain()
		{
			// Depth 0 solid [0,10], depth 1 cavity [2,8], depth 2 solid [4,6] — the
			// innermost body must wind +1; here it arrives inverted.
			List<Vec3[]> tris = CubeTris(0.0, 10.0);
			tris.AddRange(Flipped(CubeTris(2.0, 8.0)));
			tris.AddRange(Flipped(CubeTris(4.0, 6.0)));
			RepairPlan plan = Repair.PlanRepair(tris);
			await Assert.That(plan.NumShells).IsEqualTo(3);
			await Assert.That(plan.FlippedShells).IsEqualTo(1);

			// Only the innermost 12 triangles (last shell appended) flip.
			await Assert.That(plan.Flip.Take(24).All(f => !f)).IsTrue();
			await Assert.That(plan.Flip.Skip(24).All(f => f)).IsTrue();
		}

		[Test]
		public async Task DisjointBodiesFlipIndependently()
		{
			// Body 1 correct, body 2 inverted, disjoint in x.
			List<Vec3[]> tris = CubeTris(0.0, 2.0);
			tris.AddRange(Flipped(CubeTris(5.0, 7.0)));
			RepairPlan plan = Repair.PlanRepair(tris);
			await Assert.That(plan.NumShells).IsEqualTo(2);
			await Assert.That(plan.FlippedShells).IsEqualTo(1);
			await Assert.That(plan.Flip.Take(12).All(f => !f)).IsTrue();
			await Assert.That(plan.Flip.Skip(12).All(f => f)).IsTrue();
		}

		[Test]
		public async Task DoubledSheetIsLeftAlone()
		{
			// A cube plus a coincident fully-doubled copy of itself (forward +
			// reversed = zero-thickness stack everywhere). The stack shell has no
			// clean orientation; repair must not touch it.
			List<Vec3[]> tris = CubeTris(0.0, 2.0);
			tris.AddRange(Flipped(CubeTris(5.0, 7.0)));
			tris.AddRange(CubeTris(5.0, 7.0));

			// Shells: [0,2] cube; the doubled [5,7] pair welds into one shell.
			RepairPlan plan = Repair.PlanRepair(tris);
			await Assert.That(plan.NumShells).IsEqualTo(2);
			await Assert.That(plan.FlippedShells).IsEqualTo(0)
				.Because("a doubled sheet has no orientation to repair");
		}

		[Test]
		public async Task ManifoldRepairOrientationInvertedCube()
		{
			Manifold m = MeshFromTris(Flipped(CubeTris(0.0, 2.0)));
			await Assert.That(m.Status()).IsEqualTo(Error.NoError);
			await Assert.That(SignedVolume(m)).IsLessThan(0.0)
				.Because("fixture must import inverted");
			Manifold repaired = m.RepairOrientation();
			await Assert.That(repaired.Status()).IsEqualTo(Error.NoError);
			await Assert.That(Math.Abs(SignedVolume(repaired) - 8.0)).IsLessThan(1e-9)
				.Because($"repaired cube must enclose +8 units³, got {SignedVolume(repaired)}");
			await Assert.That(repaired.NumTri()).IsEqualTo(m.NumTri());

			// Repairing again is a no-op (and returns a clone, not a rebuild).
			Manifold again = repaired.RepairOrientation();
			await Assert.That(Math.Abs(SignedVolume(again) - SignedVolume(repaired)) == 0.0).IsTrue();
		}

		[Test]
		public async Task ManifoldRepairPreservesCavityAndPairing()
		{
			List<Vec3[]> tris = Flipped(CubeTris(0.0, 6.0));
			tris.AddRange(CubeTris(2.0, 4.0));
			Manifold m = MeshFromTris(tris);
			await Assert.That(m.Status()).IsEqualTo(Error.NoError);
			Manifold repaired = m.RepairOrientation();

			// Solid [0,6] minus void [2,4]: 216 - 8 = 208.
			await Assert.That(Math.Abs(SignedVolume(repaired) - 208.0)).IsLessThan(1e-9)
				.Because($"expected 208, got {SignedVolume(repaired)}");

			// The rewound impl must still be a valid manifold (pairing intact).
			await Assert.That(repaired.AsImpl().IsSoup).IsFalse();
			await Assert.That(repaired.AsImpl().IsManifold()).IsTrue();

			// And it must boolean correctly as a standalone repaired solid.
			Manifold probe = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false)
				.Translate(new Vec3(2.5, 2.5, 2.5));
			Manifold inter = repaired.IntersectionWithEngine(probe, BooleanEngine.Robust);
			await Assert.That(inter.Volume()).IsLessThan(1e-9)
				.Because($"probe inside the cavity must intersect nothing, got {inter.Volume()}");
		}

		[Test]
		public async Task RepairIsAvailableBeforeAnyBooleanAndFixesUnion()
		{
			// Union of a solid with an inverted neighbor: without repair the
			// inverted body contributes no material.
			Manifold a = MeshFromTris(CubeTris(0.0, 2.0));
			Manifold b = MeshFromTris(Flipped(CubeTris(1.0, 3.0)));
			BooleanEngine engine = BooleanEngine.Robust;
			Manifold broken = a.UnionWithEngine(b, engine);
			Manifold fixedUnion = a.UnionWithEngine(b.RepairOrientation(), engine);
			await Assert.That(Math.Abs(broken.Volume() - 8.0)).IsLessThan(1e-9)
				.Because("inverted B adds nothing");
			await Assert.That(Math.Abs(fixedUnion.Volume() - 15.0)).IsLessThan(1e-9)
				.Because($"8 + 8 - 1 overlap = 15, got {fixedUnion.Volume()}");
		}

		/// <summary>Axis-aligned cube [lo,hi]³ as 12 outward-wound triangles.</summary>
		/// <param name="lo">The low coordinate on every axis.</param>
		/// <param name="hi">The high coordinate on every axis.</param>
		/// <returns>The cube's triangles.</returns>
		internal static List<Vec3[]> CubeTris(double lo, double hi)
		{
			double[][][] quads =
			{
				new[] { new[] { 0.0, 0.0, 0.0 }, new[] { 0.0, 1.0, 0.0 }, new[] { 1.0, 1.0, 0.0 }, new[] { 1.0, 0.0, 0.0 } }, // -z
				new[] { new[] { 0.0, 0.0, 1.0 }, new[] { 1.0, 0.0, 1.0 }, new[] { 1.0, 1.0, 1.0 }, new[] { 0.0, 1.0, 1.0 } }, // +z
				new[] { new[] { 0.0, 0.0, 0.0 }, new[] { 1.0, 0.0, 0.0 }, new[] { 1.0, 0.0, 1.0 }, new[] { 0.0, 0.0, 1.0 } }, // -y
				new[] { new[] { 0.0, 1.0, 0.0 }, new[] { 0.0, 1.0, 1.0 }, new[] { 1.0, 1.0, 1.0 }, new[] { 1.0, 1.0, 0.0 } }, // +y
				new[] { new[] { 0.0, 0.0, 0.0 }, new[] { 0.0, 0.0, 1.0 }, new[] { 0.0, 1.0, 1.0 }, new[] { 0.0, 1.0, 0.0 } }, // -x
				new[] { new[] { 1.0, 0.0, 0.0 }, new[] { 1.0, 1.0, 0.0 }, new[] { 1.0, 1.0, 1.0 }, new[] { 1.0, 0.0, 1.0 } }, // +x
			};
			double s = hi - lo;
			Vec3 M(double[] q) => new Vec3(lo + (q[0] * s), lo + (q[1] * s), lo + (q[2] * s));
			List<Vec3[]> outTris = new List<Vec3[]>();
			foreach (double[][] q in quads)
			{
				outTris.Add(new[] { M(q[0]), M(q[1]), M(q[2]) });
				outTris.Add(new[] { M(q[0]), M(q[2]), M(q[3]) });
			}

			return outTris;
		}

		/// <summary>Every triangle reversed.</summary>
		/// <param name="tris">The triangles.</param>
		/// <returns>The reversed triangles.</returns>
		internal static List<Vec3[]> Flipped(IReadOnlyList<Vec3[]> tris)
		{
			List<Vec3[]> outTris = new List<Vec3[]>(tris.Count);
			foreach (Vec3[] t in tris)
			{
				outTris.Add(new[] { t[0], t[2], t[1] });
			}

			return outTris;
		}

		/// <summary>
		/// Signed (divergence-theorem) volume of a Manifold — <c>Volume()</c> takes the
		/// absolute value, which hides exactly the inversions these tests are about.
		/// </summary>
		/// <param name="m">The manifold.</param>
		/// <returns>Its signed volume.</returns>
		internal static double SignedVolume(Manifold m)
		{
			List<Vec3[]> tris = Robust.Soup.ImplToTris(m.AsImpl());
			double sum = 0.0;
			foreach (Vec3[] t in tris)
			{
				sum += LinalgFunctions.Dot(t[0], LinalgFunctions.Cross(t[1], t[2])) / 6.0;
			}

			return sum;
		}

		/// <summary>
		/// The Rust's f32 <c>MeshGL</c> fixture import: one corner per vertex, merged, through
		/// the robust entry point.
		/// </summary>
		/// <param name="tris">The triangles.</param>
		/// <returns>The imported manifold.</returns>
		private static Manifold MeshFromTris(IReadOnlyList<Vec3[]> tris)
		{
			MeshGL mesh = new MeshGL();
			mesh.NumProp = 3;
			foreach (Vec3[] t in tris)
			{
				foreach (Vec3 p in t)
				{
					mesh.VertProperties.Add((float)p.X);
					mesh.VertProperties.Add((float)p.Y);
					mesh.VertProperties.Add((float)p.Z);
				}
			}

			for (uint i = 0; i < (uint)(tris.Count * 3); i++)
			{
				mesh.TriVerts.Add(i);
			}

			mesh.Merge();
			return Manifold.FromMeshGLRobust(mesh);
		}
	}
}
