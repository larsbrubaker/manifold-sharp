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

// ConvexDilationTests.Erosion.cs — NOT A PORT, a C#-only adaptation test like the rest of
// ConvexDilationTests: its subject is Manifold.TryErodeByConvex, ConvexDilation.cs's
// erosion leg (divergence ledger entry 6). The oracle is Minkowski.Difference — the ported
// sweep over the same hulls — and agreement is on volume and genus, not triangles, for the
// same reduction-order reason the dilation tests give. The sequential-versus-parallel bit
// identity is ParallelismTests.ConvexDilation.cs.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ConvexDilationErosionTests
	{
		/// <summary>Relative volume slop between the tree erosion and the ported sweep.</summary>
		private const double VolumeTolerance = 1e-9;

		/// <summary>
		/// A square frame: a slab with a square window cut through it — non-convex, genus 1,
		/// every face flat.
		/// </summary>
		/// <returns>The frame.</returns>
		internal static Manifold Frame()
		{
			return Manifold.Cube(new Vec3(4.0, 4.0, 1.0), true)
				.Difference(Manifold.Cube(new Vec3(2.0, 2.0, 2.0), true));
		}

		/// <summary>
		/// Two blocks joined by a wall 0.2 thick: a 0.3 ball erodes the wall away entirely,
		/// so the result splits into two pieces.
		/// </summary>
		/// <returns>The dumbbell.</returns>
		internal static Manifold ThinWallDumbbell()
		{
			return Manifold.Cube(new Vec3(2.0, 2.0, 2.0), false)
				.Union(Manifold.Cube(new Vec3(2.0, 2.0, 2.0), false).Translate(new Vec3(4.0, 0.0, 0.0)))
				.Union(Manifold.Cube(new Vec3(3.0, 0.2, 2.0), false).Translate(new Vec3(1.5, 0.9, 0.0)));
		}

		/// <summary>
		/// A bent plate 0.2 thick everywhere: a 0.3 ball erodes all of it, so nothing is left.
		/// </summary>
		/// <returns>The thin L.</returns>
		internal static Manifold ThinL()
		{
			return Manifold.Cube(new Vec3(3.0, 0.2, 2.0), false)
				.Union(Manifold.Cube(new Vec3(0.2, 3.0, 2.0), false));
		}

		private static Manifold Shape(string shape)
		{
			return shape switch
			{
				"L-shape" => ConvexDilationTests.LShape().Scale(Vec3.Splat(2.0)),
				"drilled" => ConvexDilationTests.DrilledPart(16),
				"frame" => Frame(),
				"thin-wall" => ThinWallDumbbell(),
				"thin-L" => ThinL(),
				"cube" => Manifold.Cube(Vec3.Splat(2.0), true),
				_ => throw new ArgumentException(shape),
			};
		}

		/// <summary>
		/// The tree erosion agrees with <see cref="Manifold.MinkowskiDifference"/> on volume
		/// and genus — the promise that makes it a drop-in for the ported sweep — including
		/// a wall eroded away (the part splits), a part eroded away entirely (empty), and a
		/// convex solid, which the tree takes because the sweep sweeps those too.
		/// </summary>
		/// <param name="shape">Which fixture to erode.</param>
		/// <returns>The test task.</returns>
		[Test]
		[Arguments("L-shape")]
		[Arguments("drilled")]
		[Arguments("frame")]
		[Arguments("thin-wall")]
		[Arguments("thin-L")]
		[Arguments("cube")]
		public async Task MatchesTheMinkowskiDifference(string shape)
		{
			Manifold solid = Shape(shape);
			Manifold ball = Manifold.Sphere(0.3, 8);

			Manifold reference = solid.MinkowskiDifference(ball);
			await Assert.That(solid.TryErodeByConvex(ball, null, null, out Manifold tree)).IsTrue();

			await Assert.That(tree.Status()).IsEqualTo(Error.NoError);
			await Assert.That(tree.Genus()).IsEqualTo(reference.Genus());
			await Assert.That(tree.Decompose().Count).IsEqualTo(reference.Decompose().Count);
			if (shape == "thin-L")
			{
				// Anti-vacuity for the vanishing case: the sweep really leaves nothing.
				await Assert.That(reference.IsEmpty()).IsTrue();
				await Assert.That(tree.IsEmpty()).IsTrue();
				return;
			}

			if (shape == "thin-wall")
			{
				// Anti-vacuity for the split case: the wall really is gone.
				await Assert.That(reference.Decompose().Count).IsEqualTo(2);
			}

			await Assert.That(reference.Volume()).IsGreaterThan(0.0);
			double relative = Math.Abs(tree.Volume() - reference.Volume()) / reference.Volume();
			await Assert.That(relative).IsLessThanOrEqualTo(VolumeTolerance)
				.Because($"tree {tree.Volume()} against the ported sweep's {reference.Volume()}");
		}

		/// <summary>
		/// An asymmetric, off-centre tool: MinkowskiDifference sweeps B, not -B, so a mirrored or
		/// re-centred tool would move the result. The box row moves it (a box is centrally
		/// symmetric, so -B is B shifted and only the bounding box sees the change); the
		/// tetrahedron row also changes the volume. The bounding box is compared for that.
		/// Both tools contain the origin off-centre: a box beside the origin, as first
		/// written, erodes the drilled part to a result whose box and volume a mirrored
		/// tool reproduces (the sweep leaves slivers on the original faces either way), so
		/// that row could not see a mirror.
		/// </summary>
		/// <param name="tool">Which tool.</param>
		/// <param name="shape">Which non-convex fixture.</param>
		/// <returns>The test task.</returns>
		[Test]
		[Arguments("box", "L-shape")]
		[Arguments("box", "drilled")]
		[Arguments("tetrahedron", "L-shape")]
		[Arguments("tetrahedron", "drilled")]
		public async Task AnAsymmetricOffCentreToolMatches(string tool, string shape)
		{
			Manifold solid = Shape(shape);
			Manifold b = tool == "box"
				? Manifold.Cube(new Vec3(0.3, 0.2, 0.1), false).Translate(new Vec3(-0.05, -0.05, -0.03))
				: Manifold.Tetrahedron().Scale(new Vec3(0.3, 0.2, 0.1)).Translate(new Vec3(0.05, 0.0, 0.0));

			Manifold reference = solid.MinkowskiDifference(b);
			await Assert.That(solid.TryErodeByConvex(b, null, null, out Manifold tree)).IsTrue();

			await Assert.That(tree.Status()).IsEqualTo(Error.NoError);
			await Assert.That(tree.Genus()).IsEqualTo(reference.Genus());
			double relative = Math.Abs(tree.Volume() - reference.Volume()) / reference.Volume();
			await Assert.That(relative).IsLessThanOrEqualTo(VolumeTolerance)
				.Because($"tree {tree.Volume()} against the ported path's {reference.Volume()}");
			Box treeBox = tree.BoundingBox();
			Box referenceBox = reference.BoundingBox();
			await Assert.That(Length(treeBox.Min - referenceBox.Min)).IsLessThanOrEqualTo(1e-9)
				.Because($"min corner {treeBox.Min} against {referenceBox.Min}");
			await Assert.That(Length(treeBox.Max - referenceBox.Max)).IsLessThanOrEqualTo(1e-9)
				.Because($"max corner {treeBox.Max} against {referenceBox.Max}");
		}

		/// <summary>
		/// A non-convex tool and an empty operand are declined, as they are for dilation; the
		/// ported sweep answers them. (Nested shells are ConvexDilationNestedTests'.)
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task NonConvexToolsAndEmptyOperandsAreDeclined()
		{
			Manifold ball = Manifold.Sphere(0.3, 8);
			await Assert.That(ConvexDilationTests.DrilledPart(8).TryErodeByConvex(ConvexDilationTests.LShape(), null, null, out Manifold result)).IsFalse();
			await Assert.That(result.IsEmpty()).IsTrue();

			await Assert.That(Manifold.Empty().TryErodeByConvex(ball, null, null, out _)).IsFalse();
			await Assert.That(ConvexDilationTests.LShape().TryErodeByConvex(Manifold.Empty(), null, null, out _)).IsFalse();
		}

		/// <summary>
		/// A pre-cancelled token buys no work and still answers true with a Cancelled result.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task APreCancelledTokenReturnsCancelled()
		{
			CancelToken token = new CancelToken();
			token.Cancel();

			await Assert.That(ConvexDilationTests.LShape().TryErodeByConvex(Manifold.Sphere(0.3, 8), token, null, out Manifold result)).IsTrue();
			await Assert.That(result.Status()).IsEqualTo(Error.Cancelled);
		}

		/// <summary>
		/// A watched erosion never goes backwards, ends on exactly 1.0, and spends exactly the
		/// units TryReduce's accounting promises — the closing subtraction's included.
		/// </summary>
		/// <remarks>
		/// The L is small enough that the total stays at or under 100, where the throttle's
		/// step is 1 and every unit emits, so the report just before CompletePhase's 1.0 is
		/// exactly (total - 1) / total and pins the total: a missing or extra unit moves it.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task ProgressIsMonotonicAndEndsAtOne()
		{
			List<double> fractions = new List<double>();
			ProgressReporter reporter = new ProgressReporter((phase, fraction) =>
			{
				lock (fractions)
				{
					fractions.Add(fraction!.Value);
				}
			});

			Manifold solid = ConvexDilationTests.LShape();

			// ConvexDilation's accounting: one unit per hull, per leaf (16 hulls each, no
			// solid leaf when eroding), per tree union (L - 1 for L leaves), the subtraction,
			// and the unit CompletePhase spends.
			int numTri = solid.NumTri();
			int numLeaves = (numTri + 15) / 16;
			double total = numTri + numLeaves + (numLeaves - 1) + 1 + 1;
			await Assert.That(total).IsLessThanOrEqualTo(100.0)
				.Because("above 100 the throttle skips units and the penultimate report proves nothing");

			bool restore = ManifoldParallel.Enabled;
			bool applied;
			try
			{
				ManifoldParallel.Enabled = false;
				applied = solid.TryErodeByConvex(Manifold.Sphere(0.3, 8), null, reporter, out _);
			}
			finally
			{
				ManifoldParallel.Enabled = restore;
			}

			await Assert.That(applied).IsTrue();
			await Assert.That(fractions.Count).IsGreaterThan(2);
			for (int i = 1; i < fractions.Count; i++)
			{
				await Assert.That(fractions[i]).IsGreaterThanOrEqualTo(fractions[i - 1]);
			}

			await Assert.That(fractions[fractions.Count - 1]).IsEqualTo(1.0);
			await Assert.That(fractions[fractions.Count - 2]).IsEqualTo((total - 1.0) / total)
				.Because($"{numTri} triangles in {numLeaves} leaves should cost {total} units");
		}


		/// <summary>
		/// Erosion's closing subtraction reports from inside: the last unit before the
		/// closing normals pass hears fractional values, strictly increasing.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task TheClosingSubtractionReportsFractionalProgress()
		{
			Manifold solid = ConvexDilationTests.DrilledPart(16);
			int numTri = solid.NumTri();
			int numLeaves = (numTri + 15) / 16;

			// Hulls, leaves, tree nodes, the subtraction and the normals pass.
			double total = numTri + numLeaves + (numLeaves - 1) + 1 + 1;

			List<double> fractions = new List<double>();
			ProgressReporter reporter = new ProgressReporter((_, fraction) => fractions.Add(fraction!.Value));

			bool restore = ManifoldParallel.Enabled;
			bool applied;
			try
			{
				ManifoldParallel.Enabled = false;
				applied = solid.TryErodeByConvex(Manifold.Sphere(0.3, 8), null, reporter, out _);
			}
			finally
			{
				ManifoldParallel.Enabled = restore;
			}

			await Assert.That(applied).IsTrue();
			await Assert.That(fractions[fractions.Count - 1]).IsEqualTo(1.0);

			double subtractionStart = (total - 2) / total;
			double subtractionEnd = (total - 1) / total;
			List<double> inside = fractions.Where(f => f > subtractionStart && f < subtractionEnd).ToList();
			await Assert.That(inside.Count).IsGreaterThanOrEqualTo(2)
				.Because("the closing subtraction must report between its unit boundaries");
			for (int i = 1; i < inside.Count; i++)
			{
				await Assert.That(inside[i]).IsGreaterThan(inside[i - 1]);
			}
		}
	}
}
