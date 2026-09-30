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
		/// A closed shell nested inside another, both facing outward (Thingi10K 54229 and 54230 are
		/// two nested boxes), is declined, and the ported sum answers it.
		/// </summary>
		/// <remarks>
		/// Such a solid has winding number 2 inside the inner shell, and the exact engine's booleans
		/// assume 0 or 1: on 54229 the exact union of the part with itself comes out at 0.29 of its
		/// 1.27 volume. The tree unions the raw solid as a leaf and lost up to 1.5% of the dilation
		/// that way, with swept hulls sticking out of the result. The ported sum unions the same
		/// solid in a different order and measured right on both parts, so the tree stands down.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ANestedShellIsDeclined()
		{
			Manifold outer = LShape().Scale(new Vec3(2.0, 2.0, 2.0));
			Manifold inner = Manifold.Cube(Vec3.Splat(0.5), false).Translate(new Vec3(0.25, 0.25, 0.25));
			Manifold solid = Manifold.Compose(new[] { outer, inner });
			Manifold ball = Manifold.Sphere(0.3, 8);

			await Assert.That(solid.TryDilateByConvex(ball, null, null, out Manifold result)).IsFalse()
				.Because("the inner shell makes winding number 2, which the exact union is not defined for");
			await Assert.That(result.IsEmpty()).IsTrue();

			// Side by side is not nested: two separate parts still take the tree.
			Manifold apart = Manifold.Compose(new[] { outer, inner.Translate(new Vec3(20.0, 0.0, 0.0)) });
			await Assert.That(apart.TryDilateByConvex(ball, null, null, out _)).IsTrue();
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
		/// Run sequentially on purpose. Under the parallel switch two workers can cross the
		/// throttle together and report out of order (ProgressReporter.Advance's remarks
		/// call that a UI hint, not a ledger), so monotonicity is a sequential property.
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
	}
}
