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

// ConvexDilationTests.Nested.cs — NOT A PORT, a C#-only adaptation test like the rest of
// ConvexDilationTests: its subject is how ConvexDilation.cs's tree handles a solid with
// several shells whose bounding boxes overlap (divergence ledger entry 6). Such shells may
// nest or cross with the same orientation and wind 2, which the exact engine's unions are
// not defined for, so the tree runs on the robust engine's rebuild of the solid - the union
// of its shells. The fixtures cover every way boxes overlap: a cavity, a same-orientation
// shell inside another, interlocked parts, crossing shells, side-by-side overlapping shells
// and a ball in a cavity in a ball. The reference is always the sweep of that same rebuilt
// union; for a clean part it is the same solid as the raw one.
//
// Agreement is on volume and genus, for the reduction-order reason the other halves give.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ConvexDilationNestedTests
	{
		/// <summary>Relative volume slop between the tree and the ported sweep.</summary>
		private const double VolumeTolerance = 1e-9;

		/// <summary>The L of <see cref="ConvexDilationTests.LShape"/> at twice the size: arms 2 thick.</summary>
		/// <returns>The outer solid.</returns>
		private static Manifold Outer()
		{
			return ConvexDilationTests.LShape().Scale(new Vec3(2.0, 2.0, 2.0));
		}

		/// <summary>A 0.5 cube in the L's corner block, 0.75 clear of every wall.</summary>
		/// <returns>The inner shell, facing outward.</returns>
		private static Manifold Inner()
		{
			return Manifold.Cube(Vec3.Splat(0.5), false).Translate(new Vec3(0.75, 0.75, 0.75));
		}

		/// <summary>
		/// The fixture by name: "hollow" (a cavity), "nested" (same orientation, winding 2),
		/// "interlocked" (a cube in the L's notch: inside its box, outside the L), "crossing"
		/// (a cube straddling the L's notch wall: boxes nest, shells cross), "side-by-side"
		/// (two unmerged overlapping Ls, boxes overlap without nesting) and "ball-cavity-ball".
		/// </summary>
		/// <param name="shape">Which fixture.</param>
		/// <returns>The solid.</returns>
		private static Manifold Fixture(string shape)
		{
			return shape switch
			{
				"hollow" => Outer().Difference(Inner()),
				"nested" => Manifold.Compose(new[] { Outer(), Inner() }),
				"interlocked" => Manifold.Compose(new[] { Outer(), Manifold.Cube(Vec3.Splat(1.0), false).Translate(new Vec3(4.0, 3.5, 0.5)) }),
				"crossing" => Manifold.Compose(new[] { Outer(), Manifold.Cube(Vec3.Splat(1.0), false).Translate(new Vec3(1.5, 3.0, 0.5)) }),
				"side-by-side" => Manifold.Compose(new[] { Outer(), Outer().Translate(new Vec3(1.0, 1.0, 0.5)) }),
				_ => Manifold.Compose(new[]
				{
					Manifold.Sphere(3.0, 16).Difference(Manifold.Sphere(2.0, 16)),
					Manifold.Sphere(1.0, 16),
				}),
			};
		}

		/// <summary>
		/// What the sweep is compared on: the union of the solid's shells, which is what the
		/// tree reduces and what a user means.
		/// </summary>
		/// <param name="shape">Which fixture.</param>
		/// <returns>The sweep's operand.</returns>
		private static Manifold SweepOperand(string shape)
		{
			return Fixture(shape).RebuildSolid(WindingRule.Positive);
		}

		/// <summary>
		/// Each nesting fixture dilates through the tree and agrees with the sweep.
		/// </summary>
		/// <param name="shape">Which fixture.</param>
		/// <returns>The test task.</returns>
		[Test]
		[Arguments("hollow")]
		[Arguments("nested")]
		[Arguments("interlocked")]
		[Arguments("crossing")]
		[Arguments("side-by-side")]
		[Arguments("ball-cavity-ball")]
		public async Task ANestingSolidDilatesThroughTheTree(string shape)
		{
			Manifold ball = Manifold.Sphere(0.3, 8);
			Manifold reference = SweepOperand(shape).MinkowskiSum(ball);

			int before = ConvexDilation.RebuildsRun;
			bool applied = Fixture(shape).TryDilateByConvex(ball, null, null, out Manifold tree);
			int rebuilds = ConvexDilation.RebuildsRun - before;
			await Assert.That(applied).IsTrue()
				.Because("overlapping boxes no longer decline by themselves");
			await Assert.That(rebuilds).IsEqualTo(1)
				.Because("every fixture has overlapping component boxes, so the union is rebuilt");
			await Assert.That(tree.Status()).IsEqualTo(Error.NoError);
			await Assert.That(tree.Genus()).IsEqualTo(reference.Genus());
			double relative = Math.Abs(tree.Volume() - reference.Volume()) / reference.Volume();
			await Assert.That(relative).IsLessThanOrEqualTo(VolumeTolerance)
				.Because($"tree {tree.Volume()} against the sweep's {reference.Volume()}");
		}

		/// <summary>
		/// Each nesting fixture erodes through the tree and agrees with the sweep; for the hollow
		/// part the cavity grows by the tool, which the sweep's inner-shell hulls also carve.
		/// </summary>
		/// <param name="shape">Which fixture.</param>
		/// <returns>The test task.</returns>
		[Test]
		[Arguments("hollow")]
		[Arguments("nested")]
		[Arguments("interlocked")]
		[Arguments("crossing")]
		[Arguments("side-by-side")]
		[Arguments("ball-cavity-ball")]
		public async Task ANestingSolidErodesThroughTheTree(string shape)
		{
			Manifold ball = Manifold.Sphere(0.3, 8);
			Manifold reference = SweepOperand(shape).MinkowskiDifference(ball);

			int before = ConvexDilation.RebuildsRun;
			bool applied = Fixture(shape).TryErodeByConvex(ball, null, null, out Manifold tree);
			int rebuilds = ConvexDilation.RebuildsRun - before;
			await Assert.That(applied).IsTrue()
				.Because("overlapping boxes no longer decline by themselves");
			await Assert.That(rebuilds).IsEqualTo(1)
				.Because("every fixture has overlapping component boxes, so the union is rebuilt");
			await Assert.That(tree.Status()).IsEqualTo(Error.NoError);
			await Assert.That(tree.Genus()).IsEqualTo(reference.Genus());
			double relative = Math.Abs(tree.Volume() - reference.Volume()) / reference.Volume();
			await Assert.That(relative).IsLessThanOrEqualTo(VolumeTolerance)
				.Because($"tree {tree.Volume()} against the sweep's {reference.Volume()}");
		}

		/// <summary>
		/// Why same-orientation nesting is rebuilt first rather than swept raw: the raw erosion
		/// sweep carves the inner shell's boundary out of material the union keeps, so it is not
		/// the answer a user means. Pinned so the reason stays visible.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task TheRawSweepOfSameOrientationNestingCarvesTheInnerShell()
		{
			Manifold ball = Manifold.Sphere(0.3, 8);
			double raw = Fixture("nested").MinkowskiDifference(ball).Volume();
			await Assert.That(raw).IsGreaterThan(0.0);
			double union = SweepOperand("nested").MinkowskiDifference(ball).Volume();

			await Assert.That(raw).IsLessThan(union)
				.Because($"raw sweep {raw} against the union's {union}");
		}

		/// <summary>
		/// The rebuild runs exactly when component boxes overlap: several shells whose boxes
		/// are apart skip it, a single shell skips it, and an overlapping pair runs it.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task TheRebuildRunsOnlyWhenComponentBoxesOverlap()
		{
			Manifold ball = Manifold.Sphere(0.3, 8);
			Manifold apart = Manifold.Compose(new[] { Outer(), Inner().Translate(new Vec3(20.0, 0.0, 0.0)) });

			int before = ConvexDilation.RebuildsRun;
			bool apartApplied = apart.TryDilateByConvex(ball, null, null, out _);
			bool singleApplied = Outer().TryErodeByConvex(ball, null, null, out _);
			int skipped = ConvexDilation.RebuildsRun - before;

			before = ConvexDilation.RebuildsRun;
			bool overlapApplied = Fixture("side-by-side").TryErodeByConvex(ball, null, null, out _);
			int ran = ConvexDilation.RebuildsRun - before;

			await Assert.That(apartApplied && singleApplied && overlapApplied).IsTrue();
			await Assert.That(skipped).IsEqualTo(0)
				.Because("disjoint boxes cannot overlap in winding, so the raw solid is reduced");
			await Assert.That(ran).IsEqualTo(1)
				.Because("overlapping boxes rebuild the union once");
		}

		/// <summary>
		/// A watched Dilate of overlapping shells reports a monotone sequence that reaches 1.0
		/// only at its last report: the rebuild runs unreported, so its phases' closing 1.0
		/// cannot pin a caller's bar at full through the tree.
		/// </summary>
		/// <remarks>Sequential, for the reason ConvexDilationTests' progress test gives.</remarks>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task TheRebuildDoesNotReportAFinishedBar()
		{
			List<double> fractions = new List<double>();
			ProgressReporter reporter = new ProgressReporter((_, fraction) =>
			{
				lock (fractions)
				{
					fractions.Add(fraction ?? -1.0);
				}
			});

			bool restore = ManifoldParallel.Enabled;
			int before = ConvexDilation.RebuildsRun;
			bool applied;
			int rebuilds;
			try
			{
				ManifoldParallel.Enabled = false;
				applied = Fixture("side-by-side").TryDilateByConvex(Manifold.Sphere(0.3, 8), null, reporter, out _);
				rebuilds = ConvexDilation.RebuildsRun - before;
			}
			finally
			{
				ManifoldParallel.Enabled = restore;
			}

			await Assert.That(applied).IsTrue();
			await Assert.That(rebuilds).IsEqualTo(1);
			await Assert.That(fractions.Count).IsGreaterThan(2);
			for (int i = 0; i < fractions.Count - 1; i++)
			{
				await Assert.That(fractions[i]).IsLessThan(1.0)
					.Because($"report {i} of {fractions.Count} already read finished");
				await Assert.That(fractions[i + 1]).IsGreaterThanOrEqualTo(fractions[i])
					.Because($"the bar went backwards, {fractions[i]} then {fractions[i + 1]}");
			}

			await Assert.That(fractions[fractions.Count - 1]).IsEqualTo(1.0);
		}
	}
}
