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

// NOT PORTED FROM RUST, and could not be: Rust gates parallelism behind the
// `parallel` Cargo feature, so "sequential vs parallel" there is two builds of
// the same test, compared by running `cargo test` twice. The C# analogue is a
// runtime switch (ManifoldParallel.Enabled), which makes the comparison
// expressible *inside* one test — run the operation with the switch off, run it
// again with the switch on, assert the outputs are bit-identical. None of these
// count toward the test-for-test tally in docs/PORTING_PLAN.md, for the reason
// InfrastructureAdaptationTests.cs's header gives.
//
// One test per determinism-preserving site — all eleven, not just the six
// docs/PORTING_PLAN.md blesses by name, because the robust engine's five
// per-triangle maps reach the same helper through Progress.MaybeParMapCtProgress
// and are inside the Rust feature's scope too:
//
//   Intersect12, Winding03, Face2Tri   BooleanPipelineIsBitIdenticalInParallel
//   Face2Tri (threshold, provably)     Face2TriIsBitIdenticalOverItsOwnThreshold
//   SDF voxel fill                     SdfVoxelFillIsBitIdenticalInParallel
//   CalculateVertNormals               VertNormalsAreBitIdenticalInParallel
//   Minkowski hulls                    MinkowskiGeometryIsBitIdenticalInParallel
//   the robust engine's five maps       RobustEnginePipelineIsBitIdenticalInParallel
//                                       (one robust boolean drives all five)
//
// Those six live in ParallelismTests.Sites.cs — same class, split off for the
// 800-line cap. This file carries the shared apparatus they use, plus the three
// cross-cutting properties Par.cs claims: that the switch has teeth (the map
// really enters Parallel.For), that cancellation keeps its contract under the
// parallel loop, and that a worker's exception still reaches the caller
// unwrapped.
//
// ── Anti-vacuity ─────────────────────────────────────────────────────────────
// A size below its site's threshold takes the sequential path with the switch
// on, and the test would then compare sequential against sequential and pass
// while proving nothing. Every site test therefore asserts the input crosses its
// site's threshold before comparing. Two of them — Face2Tri inside a boolean,
// and the robust engine's four inner maps — can only bound it indirectly,
// because the work-list sizes are internal; both say so at the assertion.
// ParallelismIsOffByDefaultAndReallyUsesTheParallelLoopWhenOn below is the other
// half of the same guarantee: it pins that the switch reaches Parallel.For at
// all, so "crossed the threshold" and "took the parallel branch" are both
// established rather than assumed.
//
// ── Why mesh-ID labels are excluded from the comparison ──────────────────────
// Mesh IDs come from a process-global monotonic counter, so an operation that
// mints fresh ones (LevelSet, Minkowski's per-face hulls, any impl freshly
// constructed) hands back different ID *values* on its second run — sequentially
// too. AssertSameGeometry therefore compares positions, indices, run boundaries
// and face IDs bit-for-bit and takes `compareRunLabels` for the ID labels, which
// only the boolean can honestly assert. MinkowskiGeometryIsBitIdenticalInParallel
// closes the loop by measuring the sequential-vs-sequential baseline for those
// labels, which is what shows parallelism adds nothing there.

using System.Collections.Concurrent;

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ParallelismTests
	{
		/// <summary>
		/// The constraint key that serializes every test touching
		/// <see cref="ManifoldParallel.Enabled"/>, mirroring
		/// <see cref="TypesTests.QualityGlobalStateKey"/>. TUnit's <c>[NotInParallel]</c>
		/// only promises that tests sharing a key never run concurrently <em>with each
		/// other</em>, so any future test that reads or writes the switch must carry this
		/// key — including the ones in InfrastructureAdaptationTests.cs that pin
		/// <see cref="Par"/>'s sequential call order.
		/// </summary>
		/// <remarks>
		/// The key does not stop the rest of the suite from running while the switch is
		/// briefly on; nothing can, short of serializing the whole run. That is sound
		/// precisely because of the claim under test — a site that goes parallel returns
		/// the same bits — and it is the same exposure
		/// <see cref="RobustEngineTests.BooleanConfigGlobalStateKey"/> already carries.
		/// </remarks>
		public const string ParallelismGlobalStateKey = "ManifoldParallelGlobalState";

		// ---------------------------------------------------------------------
		// The switch itself
		// ---------------------------------------------------------------------

		[Test]
		[NotInParallel(ParallelismGlobalStateKey)]
		public async Task ParallelismIsOffByDefaultAndReallyUsesTheParallelLoopWhenOn()
		{
			// The default is read from the environment once per process, so this states
			// the assumption it depends on rather than pretending it can change it — the
			// shape TimingIsOffUnlessTheEnvironmentAsks uses for MANIFOLD_TIMING. Under
			// the forced-on suite run (MANIFOLD_PARALLEL=1) the expected default flips,
			// which is exactly what makes that run a different configuration.
			string? configured = Environment.GetEnvironmentVariable(
				ManifoldParallel.EnabledEnvironmentVariable);
			bool expectedDefault = configured == "1"
				|| string.Equals(configured, "true", StringComparison.OrdinalIgnoreCase);

			bool restore = ManifoldParallel.Enabled;
			await Assert.That(restore).IsEqualTo(expectedDefault);

			try
			{
				// Below the threshold the switch must change nothing — `n < threshold` is
				// the autoPolicy half of the gate, and it wins over the switch.
				ManifoldParallel.Enabled = true;
				await Assert.That(RanInAParallelLoop(4, 8)).IsFalse();

				// Off, above the threshold: still the plain loop.
				ManifoldParallel.Enabled = false;
				await Assert.That(RanInAParallelLoop(64, 8)).IsFalse();

				// On, above the threshold: the parallel loop, which is what stops every
				// other test in this file from comparing sequential against sequential and
				// passing on nothing.
				ManifoldParallel.Enabled = true;
				await Assert.That(RanInAParallelLoop(64, 8))
					.IsTrue()
					.Because("the map must really enter Parallel.For above its threshold");
			}
			finally
			{
				ManifoldParallel.Enabled = restore;
			}
		}

		// ---------------------------------------------------------------------
		// Cancellation and exceptions under the parallel loop
		// ---------------------------------------------------------------------

		[Test]
		[NotInParallel(ParallelismGlobalStateKey)]
		public async Task CancellationUnderTheParallelLoopStillReturnsNull()
		{
			bool restore = ManifoldParallel.Enabled;
			try
			{
				ManifoldParallel.Enabled = true;

				CancelToken token = new CancelToken();
				int[]? result = Par.MaybeParMapCt(100_000, 8, token, i =>
				{
					if (i == 0)
					{
						// Index 0 is always the first iteration handed out, so the flag is
						// set before the bulk of the range is scheduled. Holding the worker
						// here afterwards removes the last theoretical race — the other
						// workers cannot finish 99_999 indices without one of them reading
						// the flag this thread has already written.
						token.Cancel();
						Thread.SpinWait(2_000_000);
					}

					return i;
				});

				await Assert.That(result)
					.IsNull()
					.Because("a cancellation any worker observed must discard the results");
			}
			finally
			{
				ManifoldParallel.Enabled = restore;
			}
		}

		[Test]
		[NotInParallel(ParallelismGlobalStateKey)]
		public async Task ACancelledParallelBooleanReportsCancelledAndReturnsNothing()
		{
			// The pipeline-level half of the contract, and the reason "torn state" is not
			// a thing a caller can reach: a cancelled map returns null, the caller turns
			// that into an EMPTY result with Error.Cancelled, and the partially filled
			// array is unreachable garbage. Pre-cancelled rather than mid-flight so the
			// assertion is deterministic; CancelTests covers the mid-flight latency.
			Manifold a = Manifold.Sphere(1.0, 96);
			Manifold b = Manifold.Sphere(1.0, 96).Translate(new Vec3(0.5, 0.0, 0.0));
			CancelToken token = new CancelToken();
			token.Cancel();

			Manifold cancelled = RunWith(true, () => a.BooleanWithToken(b, OpType.Add, token));

			await Assert.That(cancelled.Status()).IsEqualTo(Error.Cancelled);
			await Assert.That(cancelled.NumTri()).IsEqualTo(0);

			// And a live token changes nothing: same bits as the untokened parallel run.
			CancelToken live = new CancelToken();
			MeshGL64 tokened = RunWith(
				true,
				() => a.BooleanWithToken(b, OpType.Add, live).GetMeshGL64(-1));
			MeshGL64 untokened = RunWith(true, () => a.Union(b).GetMeshGL64(-1));
			await Assert.That(live.IsCancelled).IsFalse();
			await AssertSameGeometry("live token", tokened, untokened, compareRunLabels: true);
		}

		[Test]
		[NotInParallel(ParallelismGlobalStateKey)]
		public async Task AWorkerExceptionReachesTheCallerUnwrapped()
		{
			// Claim 3 in Par.cs's header. Parallel.For raises an AggregateException; a
			// caller that catches what the sequential loop threw must keep catching it
			// when the host flips the switch.
			static int Boom(int i)
			{
				return i == 40 ? throw new InvalidOperationException("boom") : i;
			}

			InvalidOperationException? sequential = null;
			InvalidOperationException? parallel = null;
			bool restore = ManifoldParallel.Enabled;
			try
			{
				ManifoldParallel.Enabled = false;
				try
				{
					Par.MaybeParMap(64, 8, Boom);
				}
				catch (InvalidOperationException e)
				{
					sequential = e;
				}

				ManifoldParallel.Enabled = true;
				try
				{
					Par.MaybeParMap(64, 8, Boom);
				}
				catch (InvalidOperationException e)
				{
					parallel = e;
				}
			}
			finally
			{
				ManifoldParallel.Enabled = restore;
			}

			await Assert.That(sequential).IsNotNull();
			await Assert.That(parallel).IsNotNull();
			await Assert.That(parallel!.Message).IsEqualTo(sequential!.Message);
		}

		// ---------------------------------------------------------------------
		// Helpers
		// ---------------------------------------------------------------------

		/// <summary>
		/// Runs <paramref name="operation"/> with the switch off and then on, restoring
		/// whatever the switch was on entry.
		/// </summary>
		/// <typeparam name="T">The operation's result type.</typeparam>
		/// <param name="operation">The operation to run twice.</param>
		/// <returns>The sequential and parallel results, in that order.</returns>
		private static (T Sequential, T Parallel) BothWays<T>(Func<T> operation)
		{
			bool restore = ManifoldParallel.Enabled;
			try
			{
				ManifoldParallel.Enabled = false;
				T sequential = operation();
				ManifoldParallel.Enabled = true;
				T parallel = operation();
				return (sequential, parallel);
			}
			finally
			{
				ManifoldParallel.Enabled = restore;
			}
		}

		/// <summary>One leg of <see cref="BothWays"/>, for tests that need three.</summary>
		/// <typeparam name="T">The operation's result type.</typeparam>
		/// <param name="parallel">Whether to run with parallelism enabled.</param>
		/// <param name="operation">The operation to run.</param>
		/// <returns>The operation's result.</returns>
		private static T RunWith<T>(bool parallel, Func<T> operation)
		{
			bool restore = ManifoldParallel.Enabled;
			try
			{
				ManifoldParallel.Enabled = parallel;
				return operation();
			}
			finally
			{
				ManifoldParallel.Enabled = restore;
			}
		}

		/// <summary>
		/// Whether the map body ran inside <c>Parallel.For</c> rather than the plain loop.
		/// </summary>
		/// <remarks>
		/// <para>
		/// Counting distinct threads would be the obvious probe and is the wrong one: when
		/// the pool is saturated — which it is whenever the rest of the suite is running —
		/// <c>Parallel.For</c> legitimately executes the whole range on the calling thread,
		/// so a thread count of 1 means "busy machine", not "switch broken". That probe was
		/// tried first and failed exactly that way. This one asks which loop ran, which is
		/// the actual question.
		/// </para>
		/// <para>
		/// It rests on two implementation details, neither of them a documented guarantee,
		/// both <em>verified here</em> rather than assumed: that <c>Parallel.For</c> invokes
		/// the body inside a TPL task (so <c>Task.CurrentId</c> is non-null), and that this
		/// test's own caller is not itself inside one (so the sequential branch reports
		/// null). The second is the fragile half — a future TUnit that ran test bodies
		/// inside tasks would break it. Both fail CLOSED: the parallel case would report
		/// false and the sequential case true, and either way the assertions above fail
		/// loudly rather than passing on a probe that has quietly stopped discriminating.
		/// </para>
		/// </remarks>
		/// <param name="n">Number of indices to map.</param>
		/// <param name="threshold">The threshold to pass to the map.</param>
		/// <returns>True when the parallel branch was taken.</returns>
		private static bool RanInAParallelLoop(int n, int threshold)
		{
			ConcurrentDictionary<int, byte> insideTask = new ConcurrentDictionary<int, byte>();
			Par.MaybeParMap(n, threshold, i =>
			{
				if (Task.CurrentId is not null)
				{
					insideTask[i] = 0;
				}

				return i;
			});
			return !insideTask.IsEmpty;
		}

		/// <summary>
		/// The AssertSameGeometry of the oracle lane, applied to two runs of this port
		/// instead of to this port and the native library: every position bit-for-bit in
		/// index order, every triangle row for row, no canonicalization.
		/// </summary>
		/// <param name="what">Label for failure messages.</param>
		/// <param name="sequential">The switch-off result.</param>
		/// <param name="parallel">The switch-on result.</param>
		/// <param name="compareRunLabels">
		/// Whether to compare the per-run mesh-ID labels too. False for operations that
		/// mint fresh IDs from the process-global counter, whose labels differ between two
		/// sequential runs as well — see this file's header.
		/// </param>
		/// <returns>A task representing the assertion.</returns>
		private static async Task AssertSameGeometry(
			string what,
			MeshGL64 sequential,
			MeshGL64 parallel,
			bool compareRunLabels)
		{
			await Assert.That(parallel.NumVert())
				.IsEqualTo(sequential.NumVert())
				.Because($"{what}: vertex count");
			await Assert.That(parallel.NumTri())
				.IsEqualTo(sequential.NumTri())
				.Because($"{what}: triangle count");
			await Assert.That(parallel.NumProp)
				.IsEqualTo(sequential.NumProp)
				.Because($"{what}: numProp");

			await Assert.That(parallel.VertProperties.Count)
				.IsEqualTo(sequential.VertProperties.Count)
				.Because($"{what}: vertex property count");
			for (int i = 0; i < sequential.VertProperties.Count; i++)
			{
				await AssertSameBits($"{what}: vertProperties[{i}]", sequential.VertProperties[i], parallel.VertProperties[i]);
			}

			await Assert.That(parallel.TriVerts.Count)
				.IsEqualTo(sequential.TriVerts.Count)
				.Because($"{what}: triangle index count");
			for (int t = 0; t + 2 < sequential.TriVerts.Count; t += 3)
			{
				await Assert.That($"{parallel.TriVerts[t]},{parallel.TriVerts[t + 1]},{parallel.TriVerts[t + 2]}")
					.IsEqualTo($"{sequential.TriVerts[t]},{sequential.TriVerts[t + 1]},{sequential.TriVerts[t + 2]}")
					.Because($"{what}: triangle {t / 3}");
			}

			await AssertSameSequence($"{what}: runIndex", sequential.RunIndex, parallel.RunIndex);
			await AssertSameSequence($"{what}: faceId", sequential.FaceId, parallel.FaceId);
			await AssertSameSequence($"{what}: mergeFromVert", sequential.MergeFromVert, parallel.MergeFromVert);
			await AssertSameSequence($"{what}: mergeToVert", sequential.MergeToVert, parallel.MergeToVert);
			await AssertSameSequence($"{what}: runFlags", sequential.RunFlags, parallel.RunFlags);
			await AssertSameBits($"{what}: tolerance", sequential.Tolerance, parallel.Tolerance);

			await Assert.That(parallel.HalfedgeTangent.Count)
				.IsEqualTo(sequential.HalfedgeTangent.Count)
				.Because($"{what}: tangent count");
			for (int i = 0; i < sequential.HalfedgeTangent.Count; i++)
			{
				await AssertSameBits($"{what}: halfedgeTangent[{i}]", sequential.HalfedgeTangent[i], parallel.HalfedgeTangent[i]);
			}

			if (compareRunLabels)
			{
				await AssertSameSequence($"{what}: runOriginalId", sequential.RunOriginalId, parallel.RunOriginalId);
				await Assert.That(parallel.RunTransform.Count)
					.IsEqualTo(sequential.RunTransform.Count)
					.Because($"{what}: run transform count");
				for (int i = 0; i < sequential.RunTransform.Count; i++)
				{
					await AssertSameBits($"{what}: runTransform[{i}]", sequential.RunTransform[i], parallel.RunTransform[i]);
				}
			}
		}

		/// <summary>The halfedge arena, entry for entry — the topology the map produced.</summary>
		/// <param name="what">Label for failure messages.</param>
		/// <param name="sequential">The switch-off result.</param>
		/// <param name="parallel">The switch-on result.</param>
		/// <returns>A task representing the assertion.</returns>
		private static async Task AssertSameTopology(
			string what,
			ManifoldImpl sequential,
			ManifoldImpl parallel)
		{
			await Assert.That(parallel.Halfedge.Count)
				.IsEqualTo(sequential.Halfedge.Count)
				.Because($"{what}: halfedge count");
			for (int h = 0; h < sequential.Halfedge.Count; h++)
			{
				Halfedge s = sequential.Halfedge[h];
				Halfedge p = parallel.Halfedge[h];
				await Assert.That($"{p.StartVert},{p.EndVert},{p.PairedHalfedge}")
					.IsEqualTo($"{s.StartVert},{s.EndVert},{s.PairedHalfedge}")
					.Because($"{what}: halfedge {h}");
			}

			await Assert.That(parallel.VertPos.Count)
				.IsEqualTo(sequential.VertPos.Count)
				.Because($"{what}: vertex count");
			for (int v = 0; v < sequential.VertPos.Count; v++)
			{
				await AssertSameBits($"{what}: vert[{v}].x", sequential.VertPos[v].X, parallel.VertPos[v].X);
				await AssertSameBits($"{what}: vert[{v}].y", sequential.VertPos[v].Y, parallel.VertPos[v].Y);
				await AssertSameBits($"{what}: vert[{v}].z", sequential.VertPos[v].Z, parallel.VertPos[v].Z);
			}
		}

		/// <summary>Triangle provenance, all four fields, triangle for triangle.</summary>
		/// <param name="what">Label for failure messages.</param>
		/// <param name="sequential">The switch-off result.</param>
		/// <param name="parallel">The switch-on result.</param>
		/// <returns>A task representing the assertion.</returns>
		private static async Task AssertSameTriRefs(
			string what,
			ManifoldImpl sequential,
			ManifoldImpl parallel)
		{
			IReadOnlyList<TriRef> s = sequential.MeshRelation.TriRef;
			IReadOnlyList<TriRef> p = parallel.MeshRelation.TriRef;
			await Assert.That(p.Count).IsEqualTo(s.Count).Because($"{what}: triRef count");
			for (int t = 0; t < s.Count; t++)
			{
				await Assert.That($"{p[t].MeshId},{p[t].OriginalId},{p[t].FaceId},{p[t].CoplanarId}")
					.IsEqualTo($"{s[t].MeshId},{s[t].OriginalId},{s[t].FaceId},{s[t].CoplanarId}")
					.Because($"{what}: triRef {t}");
			}
		}

		private static async Task AssertSameSequence<T>(
			string what,
			IReadOnlyList<T> sequential,
			IReadOnlyList<T> parallel)
		{
			await Assert.That(parallel.Count).IsEqualTo(sequential.Count).Because($"{what}: count");
			for (int i = 0; i < sequential.Count; i++)
			{
				await Assert.That(parallel[i]).IsEqualTo(sequential[i]).Because($"{what}[{i}]");
			}
		}

		/// <summary>
		/// Bit-for-bit, not approximately — the exactness bar is identical doubles, and
		/// <c>IsEqualTo</c> on a double would let a signed zero or a NaN payload through.
		/// The oracle lane's AssertSameBits, applied to two runs of this port.
		/// </summary>
		/// <param name="what">Label for failure messages.</param>
		/// <param name="sequential">The switch-off value.</param>
		/// <param name="parallel">The switch-on value.</param>
		/// <returns>A task representing the assertion.</returns>
		private static async Task AssertSameBits(string what, double sequential, double parallel)
		{
			await Assert.That(BitConverter.DoubleToUInt64Bits(parallel).ToString("x16"))
				.IsEqualTo(BitConverter.DoubleToUInt64Bits(sequential).ToString("x16"))
				.Because($"{what}: {parallel} vs {sequential}");
		}
	}
}
