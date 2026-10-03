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

// CsgTreeProgressTests.cs — C#-only adaptation tests (no Rust counterpart) for
// CsgNode.EvaluateWithToken(token, progress), divergence ledger entry 10: a CSG tree
// hands its reporter to every boolean it runs, and the reporter changes no bit.

using ManifoldSharp;
using ManifoldSharp.Linalg;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	/// <summary>
	/// Progress reporting through CSG tree evaluation.
	/// </summary>
	public class CsgTreeProgressTests
	{
		/// <summary>
		/// The tree's union reduction over the coplanar fixture, on the robust engine, reports
		/// the robust phases - coplanar overlaps included - and produces the same bits as the
		/// reporter-less run, sequentially and in parallel.
		/// </summary>
		/// <remarks>
		/// Pins the engine through the internal <see cref="CsgTree.BatchUnion"/> (the parameter
		/// ConvexDilation uses) rather than flipping <see cref="BooleanConfig"/>: a process-wide
		/// Robust default would leak into every concurrently running test that reads it.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task AUnionTreeReportsItsBooleansPhasesWithoutChangingTheResult()
		{
			bool restoreParallel = ManifoldParallel.Enabled;
			try
			{
				foreach (bool parallel in new[] { false, true })
				{
					ManifoldParallel.Enabled = parallel;
					List<string> phases = new List<string>();
					ProgressReporter reporter = new ProgressReporter((phase, fraction) =>
					{
						lock (phases)
						{
							phases.Add(phase.Name());
						}
					});

					string plain = CoplanarCrossCopyTests.Hash(Manifold.FromImpl(
						CsgTree.BatchUnion(Leaves(), null, BooleanEngine.Robust).GetImpl()));
					string watched = CoplanarCrossCopyTests.Hash(Manifold.FromImpl(
						CsgTree.BatchUnion(Leaves(), null, BooleanEngine.Robust, reporter).GetImpl()));

					await Assert.That(watched).IsEqualTo(plain).Because($"parallel={parallel}");
					await Assert.That(phases).Contains(Phase.NarrowPhase.Name());
					await Assert.That(phases).Contains(Phase.CoplanarOverlaps.Name())
						.Because($"parallel={parallel}: saw {string.Join(", ", phases.Distinct())}");
					await Assert.That(phases).Contains(Phase.Assemble.Name());
				}
			}
			finally
			{
				ManifoldParallel.Enabled = restoreParallel;
			}
		}

		/// <summary>
		/// The public route: a two-operand subtract tree on the default engine reports the
		/// boolean it runs and keeps its bits, and a null reporter is the plain evaluation.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(RobustEngineTests.BooleanConfigGlobalStateKey)]
		public async Task ASubtractTreeReportsItsBoolean()
		{
			Manifold a = Manifold.Cube(Vec3.Splat(2.0), true);
			Manifold b = Manifold.Sphere(1.2, 32);
			List<string> phases = new List<string>();
			ProgressReporter reporter = new ProgressReporter((phase, fraction) =>
			{
				lock (phases)
				{
					phases.Add(phase.Name());
				}
			});

			CsgOp Subtract() => new CsgOp(OpType.Subtract, new CsgLeaf(a.AsImpl().Clone()), new CsgLeaf(b.AsImpl().Clone()));
			string plain = CoplanarCrossCopyTests.Hash(Manifold.FromImpl(Subtract().EvaluateWithToken(null)));
			string watched = CoplanarCrossCopyTests.Hash(Manifold.FromImpl(Subtract().EvaluateWithToken(null, reporter)));
			string nullReporter = CoplanarCrossCopyTests.Hash(Manifold.FromImpl(Subtract().EvaluateWithToken(null, null)));

			await Assert.That(watched).IsEqualTo(plain);
			await Assert.That(nullReporter).IsEqualTo(plain);
			await Assert.That(phases.Count).IsGreaterThan(0);
		}

		/// <summary>
		/// The coplanar fixture's body and slab plus a cube overlapping the slab, so the union
		/// runs more than one boolean.
		/// </summary>
		private static List<CsgLeafNode> Leaves()
		{
			(Manifold body, Manifold slab) = CoplanarCrossCopyTests.Fixture();
			Manifold cap = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false).Translate(new Vec3(1.5, 1.5, 1.25));
			return new List<CsgLeafNode>
			{
				new CsgLeafNode(body.AsImpl().Clone()),
				new CsgLeafNode(slab.AsImpl().Clone()),
				new CsgLeafNode(cap.AsImpl().Clone()),
			};
		}
	}
}
