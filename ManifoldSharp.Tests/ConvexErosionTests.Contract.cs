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

// ConvexErosionTests.Contract.cs — NOT A PORT; the second half of ConvexErosionTests,
// split off when the one file reached four lines under the 800-line cap. Same class,
// same C#-only adaptation-test status, and the helpers both halves use live in
// ConvexErosionTests.cs.
//
// The split is along a real seam rather than at a line count. ConvexErosionTests.cs
// asks "is the closed form the right answer" and answers it against two oracles. This
// file asks the questions that have nothing to do with the geometry:
//
//   - is the ported sweep still the ported sweep (TheGeneralDifferenceStillRunsTheSweep,
//     the regression test for divergence entry 5's whole "nothing is rerouted" claim);
//   - does it decline everything it promised to decline, and does the sweep's own
//     answer for those inputs stay exactly what it was;
//   - does it report and cancel the way the Phase conventions require.
//
// Every decline case here is a case where the closed form COULD have returned
// something. That is the point of testing them: a fast path that quietly answers an
// input it cannot prove itself on is worse than no fast path, and each of these was
// either measured disagreeing with the sweep or is one step away from it.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ConvexErosionTests
	{
		/// <summary>
		/// The ported entry point is not rerouted. <see cref="Manifold.MinkowskiDifference"/>
		/// still runs the sweep for a convex solid, which is what keeps every value in this
		/// library the Rust's.
		/// </summary>
		/// <remarks>
		/// Pinned on a shape where the two paths are observably different meshes of the same
		/// solid: the sweep reaches a 40x20x10 box through a union of twelve per-triangle
		/// hulls and leaves 36 triangles behind, while the closed form hulls eight corners
		/// and leaves 12. Equal volumes, different triangulations — so a future edit that
		/// quietly routed MinkowskiDifference through the fast path would fail here rather
		/// than pass unnoticed because the answers happen to match.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task TheGeneralDifferenceStillRunsTheSweep()
		{
			Manifold box = Manifold.Cube(new Vec3(40.0, 20.0, 10.0), true);
			Manifold ball = Manifold.Sphere(1.0, 12);

			Manifold swept = box.MinkowskiDifference(ball);
			await Assert.That(box.TryConvexErosion(ball, null, null, out Manifold closed)).IsTrue();

			await Assert.That(swept.Volume()).IsEqualTo(closed.Volume())
				.Because("38 x 18 x 8 either way");
			await Assert.That(swept.NumTri()).IsEqualTo(36)
				.Because("the sweep's triangulation, unchanged by this file existing");
			await Assert.That(closed.NumTri()).IsEqualTo(12)
				.Because("the closed form hulls the eight corners");
		}

		/// <summary>
		/// A non-convex solid is declined — the gate that makes the whole thing sound.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ANonConvexSolidIsDeclined()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(20.0), true);
			Manifold notch = Manifold.Cube(Vec3.Splat(10.0), true).Translate(new Vec3(10.0, 10.0, 10.0));
			Manifold lShape = cube.Difference(notch);
			await Assert.That(lShape.AsImpl().IsConvex()).IsFalse()
				.Because("the fixture has to actually be non-convex for the decline to mean anything");

			await Assert.That(lShape.TryConvexErosion(Manifold.Sphere(1.0, 12), null, null, out Manifold eroded)).IsFalse()
				.Because("the closed form has no answer for a solid that is not an intersection of its face halfspaces");
			await Assert.That(eroded.IsEmpty()).IsTrue()
				.Because("a declined call must not leave geometry a caller could mistake for an answer");
		}

		/// <summary>
		/// A non-convex tool is declined too, even though the mathematics does not need it
		/// to be.
		/// </summary>
		/// <remarks>
		/// Eroding a convex solid by B and by hull(B) are the same set, so the closed form
		/// would be right here. The general sweep would not: with a convex solid and a
		/// non-convex tool it swaps its operands and computes something else entirely. The
		/// gate is therefore about the promise — "this agrees with the sweep" — rather than
		/// about the geometry, and that is worth a test because it is the one decline that
		/// looks removable.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ANonConvexToolIsDeclined()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), true);
			Manifold lShapedTool = cube.Union(cube.Translate(new Vec3(0.6, 0.6, 0.0)));
			await Assert.That(lShapedTool.AsImpl().IsConvex()).IsFalse();

			await Assert.That(Manifold.Cube(Vec3.Splat(20.0), true)
				.TryConvexErosion(lShapedTool, null, null, out _)).IsFalse();
		}

		/// <summary>
		/// A tool that does not contain the origin is declined, because that is exactly where
		/// the closed form and the sweep stop being the same function.
		/// </summary>
		/// <remarks>
		/// Measured, and the numbers are the reason the gate exists: a 20-cube eroded by a
		/// unit ball centred at (2,0,0) comes back from the sweep as a 32-triangle solid of
		/// volume 5908 whose bounding box is still the whole cube — not an erosion of
		/// anything, because <c>A \ (boundary(A) (+) B)</c> keeps every point whose swept copy
		/// misses the boundary altogether. The closed form would answer the erosion, 5508.
		/// Being right and being different is still a behaviour change, so it declines.
		/// <para>
		/// The (0.8,0.8,0) case is the regression, and it is the one that decides how the gate
		/// has to be written. Its centre is only 1.13 out, so the tool does not reach past any
		/// of the CUBE's face planes, and a gate that tested the pushes against A's own
		/// normals — which is what this originally did — passed it: the closed form answered
		/// the true erosion, 5832, while the sweep answered 5832.547, and the two disagreed in
		/// silence. Sampling A's normals cannot see a tool that escapes in a direction A has
		/// no face in. Testing the origin against the TOOL's planes can, and exactly.
		/// </para>
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		[Arguments(2.0, 0.0, 5908.0)]
		[Arguments(0.8, 0.8, 5832.547441)]
		public async Task AToolThatMissesTheOriginIsDeclined(double x, double y, double sweptVolume)
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(20.0), true);
			Manifold offsetBall = Manifold.Sphere(1.0, 12).Translate(new Vec3(x, y, 0.0));

			await Assert.That(offsetBall.AsImpl().IsConvex()).IsTrue()
				.Because("convexity must not be what rejects this - the origin is the point");

			await Assert.That(cube.TryConvexErosion(offsetBall, null, null, out _)).IsFalse()
				.Because($"tool at ({x},{y},0): the sweep answers {sweptVolume} here, and disagreeing silently is worse than being slow");

			// And the sweep's answer is still exactly what it always was.
			Manifold swept = cube.MinkowskiDifference(offsetBall);
			await Assert.That(Relative(swept.Volume(), sweptVolume)).IsLessThan(1e-9)
				.Because($"the sweep's own answer moved: {swept.Volume()}");
			await Assert.That(swept.Volume()).IsNotEqualTo(5832.0)
				.Because("if the sweep agreed with the erosion here there would be nothing to gate");
		}

		/// <summary>
		/// A solid thinner than twice the radius is declined rather than answered — the
		/// centroid is not inside anything, so the dual is undefined.
		/// </summary>
		/// <remarks>
		/// The caller then runs the sweep, which answers the empty mesh that "the ball fits
		/// nowhere" is spelled as. Asserted here because the empty answer is the one a
		/// user-facing operation reads as a refusal, and it has to keep arriving.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ASolidTooThinToHoldTheToolIsDeclined()
		{
			Manifold thin = Manifold.Cube(Vec3.Splat(1.5), true);
			Manifold ball = Manifold.Sphere(1.0, 12);

			await Assert.That(thin.TryConvexErosion(ball, null, null, out _)).IsFalse();
			await Assert.That(thin.MinkowskiDifference(ball).IsEmpty()).IsTrue()
				.Because("the sweep is what the decline hands the question to, and it answers empty");
		}

		/// <summary>
		/// A pre-cancelled token answers <em>true</em> with a cancelled empty result, so the
		/// caller stops rather than falling through to the sweep it was cancelled out of.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task APreCancelledTokenAnswersCancelledRatherThanDeclining()
		{
			CancelToken token = new CancelToken();
			token.Cancel();

			bool applied = Manifold.Cube(Vec3.Splat(20.0), true)
				.TryConvexErosion(Manifold.Sphere(1.0, 12), token, null, out Manifold result);

			await Assert.That(applied).IsTrue()
				.Because("false would send a cancelled caller off to run the minutes-long path");
			await Assert.That(result.Status()).IsEqualTo(Error.Cancelled)
				.Because("the port's status-not-exception rule, same as the sweep's");
			await Assert.That(result.IsEmpty()).IsTrue();
		}

		/// <summary>
		/// A watched run reports the Minkowski phase and leaves the bar at exactly 1.0; a run
		/// that declines before doing any work reports nothing at all.
		/// </summary>
		/// <remarks>
		/// The second half is the load-bearing one, and its fixture matters: a non-convex
		/// solid is the decline that actually happens, and it is rejected before the phase
		/// opens, so the sweep the caller then runs is what the watcher sees start. The
		/// numeric declines that can happen after the phase has opened — a centroid outside
		/// the eroded body, a degenerate triple — leave a short bar on purpose, because the
		/// alternative is a run with no seam at which a watcher could cancel it. That trade is
		/// what the cancellation test below buys with it.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ProgressIsReportedOnSuccessAndNotOnADeclineBeforeAnyWork()
		{
			List<(string Name, double? Fraction)> events = new List<(string Name, double? Fraction)>();
			ProgressReporter reporter = new ProgressReporter((phase, fraction) => events.Add((phase.Name(), fraction)));

			await Assert.That(Manifold.Cube(Vec3.Splat(20.0), true)
				.TryConvexErosion(Manifold.Sphere(1.0, 12), null, reporter, out _)).IsTrue();

			await Assert.That(events.Count).IsGreaterThan(0);
			double previous = -1.0;
			foreach ((string name, double? fraction) in events)
			{
				await Assert.That(name).IsEqualTo("minkowski");
				await Assert.That(fraction!.Value).IsGreaterThanOrEqualTo(previous)
					.Because($"the bar went backwards, {previous} then {fraction.Value}");
				previous = fraction.Value;
			}

			await Assert.That(events[events.Count - 1].Fraction).IsEqualTo(1.0)
				.Because("a finished operation leaves a full bar, the rule CompletePhase exists for");

			events.Clear();
			Manifold lShape = Manifold.Cube(Vec3.Splat(20.0), true).Difference(
				Manifold.Cube(Vec3.Splat(10.0), true).Translate(new Vec3(10.0, 10.0, 10.0)));

			await Assert.That(lShape.TryConvexErosion(Manifold.Sphere(1.0, 12), null, reporter, out _)).IsFalse();
			await Assert.That(events.Count).IsEqualTo(0)
				.Because("the sweep the caller is about to run opens this phase itself");
		}

		/// <summary>
		/// A cancel raised part way through a convex erosion lands there, rather than being
		/// noticed only after the answer is already built.
		/// </summary>
		/// <remarks>
		/// The seam is the support pass — one dot product per face of the solid per vertex of
		/// the tool, and the only part of the closed form with any duration. It reports per
		/// face and polls the flag every 64, so a watcher can trip the token from a progress
		/// callback and have it observed within a poll. A 512-triangle solid is used because
		/// the throttle emits every <c>total / 100</c> units: at 513 units the step is 5, so
		/// the second report arrives around face 10 and the cancel is seen by face 64, a long
		/// way short of the 512 a completed pass would cover.
		/// <para>
		/// Answering <em>true</em> with a cancelled status is the contract, not an accident: a
		/// false here would send the caller into the sweep it was cancelled out of.
		/// </para>
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ACancelRaisedDuringTheSupportPassIsObservedThere()
		{
			Manifold solid = Manifold.Sphere(10.0, 32);
			Manifold ball = Manifold.Sphere(1.0, 12);
			await Assert.That(solid.NumTri()).IsEqualTo(512)
				.Because("the report arithmetic in the remarks is computed from this count");

			CancelToken token = new CancelToken();
			int reports = 0;
			ProgressReporter reporter = new ProgressReporter((_, _) =>
			{
				// Not the first: that one is BeginPhase, before a single face has been
				// measured, so cancelling on it would only re-test the entry gate.
				if (++reports >= 2)
				{
					token.Cancel();
				}
			});

			bool applied = solid.TryConvexErosion(ball, token, reporter, out Manifold result);

			await Assert.That(applied).IsTrue()
				.Because("a cancelled run is an applied run, or the caller falls into the sweep");
			await Assert.That(result.Status()).IsEqualTo(Error.Cancelled);
			await Assert.That(result.IsEmpty()).IsTrue();
			await Assert.That(reports).IsLessThan(30)
				.Because($"cancel was ignored for {reports} of the ~103 reports a completed pass emits");
		}

		/// <summary>
		/// Neither operand is touched. Both are read straight out of the caller's meshes,
		/// so a fast path that sorted or normalized one in place would corrupt a scene.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task NeitherOperandIsMutated()
		{
			Manifold solid = Polyhedron(2);
			Manifold ball = Manifold.Sphere(1.0, 12);
			long solidBefore = GeometryHash(solid);
			long ballBefore = GeometryHash(ball);

			await Assert.That(solid.TryConvexErosion(ball, null, null, out _)).IsTrue();

			await Assert.That(GeometryHash(solid)).IsEqualTo(solidBefore);
			await Assert.That(GeometryHash(ball)).IsEqualTo(ballBefore);
		}
	}
}
