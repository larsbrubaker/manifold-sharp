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

// ConvexDilationTests.cs — NOT A PORT, and counted as a C#-only adaptation test the way
// CLAUDE.md requires: its subject is ConvexDilation.cs, which manifold-rust has no
// counterpart for (divergence ledger entry 6). The oracle is Minkowski.Sum — the ported
// path, which is the specification the tree reduction promises to agree with. Volumes
// agree at 1e-9 relative, not bit for bit: the same hulls unioned in a different order
// round their intersection vertices differently (ConvexDilation.cs's header). The
// sequential-versus-parallel bit identity is ParallelismTests.ConvexDilation.cs.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ConvexDilationTests
	{
		/// <summary>Relative volume slop between the tree reduction and the ported sum.</summary>
		private const double VolumeTolerance = 1e-9;

		/// <summary>
		/// An L: two boxes meeting at a re-entrant edge, the simplest non-convex solid.
		/// </summary>
		/// <returns>The L-shaped solid.</returns>
		internal static Manifold LShape()
		{
			return Manifold.Cube(new Vec3(4.0, 1.0, 1.0), false)
				.Union(Manifold.Cube(new Vec3(1.0, 3.0, 1.0), false));
		}

		/// <summary>
		/// A cube with a spherical bite out of one corner and a hole drilled clean through —
		/// genus 1, curved and flat faces, re-entrant everywhere the bite and the bore meet.
		/// </summary>
		/// <param name="segments">Circular segments for the bite and the bore.</param>
		/// <returns>The drilled part.</returns>
		internal static Manifold DrilledPart(int segments)
		{
			return Manifold.Cube(Vec3.Splat(4.0), true)
				.Difference(Manifold.Sphere(1.5, segments).Translate(new Vec3(2.0, 2.0, 2.0)))
				.Difference(Manifold.CylinderCentered(6.0, 0.8, -1.0, segments, true));
		}

		/// <summary>
		/// Non-convex ⊕ convex agrees with <see cref="Manifold.MinkowskiSum"/> on volume
		/// and genus — the promise that makes the tree a drop-in for the ported batches.
		/// </summary>
		/// <param name="shape">Which fixture to dilate.</param>
		/// <returns>The test task.</returns>
		[Test]
		[Arguments("L-shape")]
		[Arguments("drilled")]
		public async Task MatchesTheMinkowskiSumOnANonConvexSolid(string shape)
		{
			Manifold solid = shape == "L-shape" ? LShape() : DrilledPart(16);
			Manifold ball = Manifold.Sphere(0.3, 8);

			Manifold reference = solid.MinkowskiSum(ball);
			await Assert.That(solid.TryDilateByConvex(ball, null, null, out Manifold tree)).IsTrue();

			await Assert.That(tree.Status()).IsEqualTo(Error.NoError);
			await Assert.That(tree.Genus()).IsEqualTo(reference.Genus());
			double relative = Math.Abs(tree.Volume() - reference.Volume()) / reference.Volume();
			await Assert.That(relative).IsLessThanOrEqualTo(VolumeTolerance)
				.Because($"tree {tree.Volume()} against the ported sum's {reference.Volume()}");
		}

		/// <summary>
		/// An asymmetric, off-centre tool: MinkowskiSum adds B, not -B, so a mirrored or
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
			Manifold solid = shape == "L-shape" ? LShape() : DrilledPart(16);
			Manifold b = tool == "box"
				? Manifold.Cube(new Vec3(0.3, 0.2, 0.1), false).Translate(new Vec3(-0.05, -0.05, -0.03))
				: Manifold.Tetrahedron().Scale(new Vec3(0.3, 0.2, 0.1)).Translate(new Vec3(0.05, 0.0, 0.0));

			Manifold reference = solid.MinkowskiSum(b);
			await Assert.That(solid.TryDilateByConvex(b, null, null, out Manifold tree)).IsTrue();

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
		/// A convex solid — cube or sphere — is declined: convex ⊕ convex is one hull in the
		/// ported path already, and a tree of one hull per triangle would be slower.
		/// </summary>
		/// <param name="shape">Which convex fixture.</param>
		/// <returns>The test task.</returns>
		[Test]
		[Arguments("cube")]
		[Arguments("sphere")]
		public async Task AConvexSolidIsDeclined(string shape)
		{
			Manifold solid = shape == "cube" ? Manifold.Cube(Vec3.Splat(2.0), true) : Manifold.Sphere(1.0, 16);

			await Assert.That(solid.TryDilateByConvex(Manifold.Sphere(0.3, 8), null, null, out Manifold result)).IsFalse();
			await Assert.That(result.IsEmpty()).IsTrue();
		}

		/// <summary>
		/// A non-convex tool is declined: that is the ported path's non-convex ⊕ non-convex
		/// branch, a different algorithm.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ANonConvexToolIsDeclined()
		{
			await Assert.That(DrilledPart(8).TryDilateByConvex(LShape(), null, null, out _)).IsFalse();
		}

		/// <summary>
		/// A cancel that lands mid-run answers true with an empty Cancelled result — true so
		/// the caller does not go on to run the ported sum it was cancelled out of.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ACancelMidRunReturnsCancelled()
		{
			CancelToken token = new CancelToken();
			int reports = 0;

			// Cancel on the second report, after BeginPhase's opening one, so the cancel
			// lands inside the hull map rather than at the entry gate.
			ProgressReporter reporter = new ProgressReporter((_, _) =>
			{
				if (Interlocked.Increment(ref reports) == 2)
				{
					token.Cancel();
				}
			});

			bool applied = DrilledPart(16).TryDilateByConvex(Manifold.Sphere(0.3, 8), token, reporter, out Manifold result);

			await Assert.That(applied).IsTrue();
			await Assert.That(result.Status()).IsEqualTo(Error.Cancelled);
			await Assert.That(result.IsEmpty()).IsTrue();
		}

		/// <summary>
		/// A pre-cancelled token buys no work and still answers true.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task APreCancelledTokenReturnsCancelled()
		{
			CancelToken token = new CancelToken();
			token.Cancel();

			await Assert.That(LShape().TryDilateByConvex(Manifold.Sphere(0.3, 8), token, null, out Manifold result)).IsTrue();
			await Assert.That(result.Status()).IsEqualTo(Error.Cancelled);
		}

		/// <summary>
		/// A watched run reports only the Minkowski phase, never goes backwards, and ends on
		/// exactly 1.0; a decline reports nothing.
		/// </summary>
		/// <remarks>
		/// Run sequentially; <see cref="TopUnionsReportFractionalProgressFromInside"/> holds
		/// the parallel run monotone too.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task ProgressIsMonotonicAndEndsAtOne()
		{
			List<(string Name, double? Fraction)> events = new List<(string Name, double? Fraction)>();
			ProgressReporter reporter = new ProgressReporter((phase, fraction) =>
			{
				lock (events)
				{
					events.Add((phase.Name(), fraction));
				}
			});

			bool restore = ManifoldParallel.Enabled;
			bool applied;
			try
			{
				ManifoldParallel.Enabled = false;
				applied = DrilledPart(16).TryDilateByConvex(Manifold.Sphere(0.3, 8), null, reporter, out _);
			}
			finally
			{
				ManifoldParallel.Enabled = restore;
			}

			await Assert.That(applied).IsTrue();
			await Assert.That(events.Count).IsGreaterThan(2);
			double previous = -1.0;
			foreach ((string name, double? fraction) in events)
			{
				await Assert.That(name).IsEqualTo("minkowski");
				await Assert.That(fraction!.Value).IsGreaterThanOrEqualTo(previous)
					.Because($"the bar went backwards, {previous} then {fraction.Value}");
				previous = fraction.Value;
			}

			await Assert.That(events[events.Count - 1].Fraction).IsEqualTo(1.0);

			events.Clear();
			await Assert.That(Manifold.Cube(Vec3.Splat(2.0), true)
				.TryDilateByConvex(Manifold.Sphere(0.3, 8), null, reporter, out _)).IsFalse();
			await Assert.That(events.Count).IsEqualTo(0)
				.Because("a decline happens before the phase opens, so the ported sum's own phase is what a watcher sees");
		}


		/// <summary>
		/// The top unions report from inside: between two whole units the bar hears
		/// fractional values, strictly increasing, and the whole run stays monotone and
		/// ends on exactly 1.0 - sequentially and with the parallel switch on.
		/// </summary>
		/// <remarks>
		/// Monotone from the first report in both modes: the reporter drops a racing
		/// worker's stale fraction (ProgressOrderTests), and the top levels report through
		/// one lock-guarded tracker.
		/// </remarks>
		/// <param name="parallel">Whether the parallel switch is on.</param>
		/// <returns>The test task.</returns>
		[Test]
		[Arguments(false)]
		[Arguments(true)]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task TopUnionsReportFractionalProgressFromInside(bool parallel)
		{
			Manifold solid = DrilledPart(16);

			List<double> fractions = new List<double>();
			ProgressReporter reporter = new ProgressReporter((_, fraction) =>
			{
				lock (fractions)
				{
					fractions.Add(fraction!.Value);
				}
			});

			bool restore = ManifoldParallel.Enabled;
			bool applied;
			try
			{
				ManifoldParallel.Enabled = parallel;
				applied = solid.TryDilateByConvex(Manifold.Sphere(0.3, 8), null, reporter, out _);
			}
			finally
			{
				ManifoldParallel.Enabled = restore;
			}

			await Assert.That(applied).IsTrue();
			await Assert.That(fractions[fractions.Count - 1]).IsEqualTo(1.0);

			// One unit per hull - convex patches plus single triangles (ConvexPatches.cs) -
			// then per leaf, per union and the closing normals pass.
			int numHulls = ConvexDilation.LastHullCount;
			int numLeaves = ((numHulls + 15) / 16) + 1;
			double total = numHulls + numLeaves + (numLeaves - 1) + 1;

			// A whole-unit report sits on an integer multiple of 1/total; a report from
			// inside a boolean sits between two of them.
			static bool IsFractional(double fraction, double total)
			{
				double units = fraction * total;
				return Math.Abs(units - Math.Round(units)) > 1e-6;
			}

			int firstFractional = fractions.FindIndex(f => IsFractional(f, total));
			await Assert.That(firstFractional).IsGreaterThanOrEqualTo(0)
				.Because("the top unions must report between their unit boundaries");

			// The top union (the last node before the closing unit) must be heard inside.
			double topUnitStart = (total - 2) / total;
			double topUnitEnd = (total - 1) / total;
			List<double> insideTop = fractions.Where(f => f > topUnitStart && f < topUnitEnd).ToList();
			await Assert.That(insideTop.Count).IsGreaterThanOrEqualTo(2);
			for (int i = 1; i < insideTop.Count; i++)
			{
				await Assert.That(insideTop[i]).IsGreaterThan(insideTop[i - 1]);
			}

			for (int i = 1; i < fractions.Count; i++)
			{
				await Assert.That(fractions[i]).IsGreaterThanOrEqualTo(fractions[i - 1])
					.Because($"the bar went backwards at report {i}, {fractions[i - 1]} then {fractions[i]}");
			}
		}
	}
}
