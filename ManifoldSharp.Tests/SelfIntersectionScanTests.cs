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

// SelfIntersectionScanTests.cs — NOT A PORT; C#-only adaptation tests for the Auto
// engine's self-intersection pre-check (Soup.HasSelfIntersections and the operand order
// in Boolean3Functions.BooleanDispatchFull). They pin the verdicts the scan gives, so the
// speedups it carries (smaller operand first, the parallel per-triangle loop, the
// symmetric vertex-neighbour shortcut in Soup.GenuineContact) are held to "same answer".
//
// The selfisect-*.txt fixtures are MatterCAD scene parts (d20 assets) dumped from their
// .3mf through float, the way agg's Mesh stores them, and imported here the way agg's
// ManifoldKernel imports them: FromMeshGL64Robust, RepairOrientation, AsOriginal. The two
// "fold" parts carry ulp-scale folds, some between triangles sharing a single vertex.

using System.Globalization;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class SelfIntersectionScanTests
	{
		/// <summary>
		/// The fixtures keep their verdicts with the parallel loop on and off. Each run
		/// imports afresh, because the verdict is cached per impl.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task ImportedPartsKeepTheirVerdicts()
		{
			(string File, bool SelfIntersects)[] cases =
			{
				("selfisect-fold-F594AC8BB1542BE9.txt", true),
				("selfisect-fold-BC6E6DD496D38966.txt", true),
				("selfisect-clean-63F7A7FFD6A5F7C6.txt", false),
			};

			foreach (bool parallel in new[] { false, true })
			{
				foreach ((string file, bool expected) in cases)
				{
					Manifold m = LoadFixture(file);
					await Assert.That(m.Status()).IsEqualTo(Error.NoError).Because(file);
					bool verdict = WithParallelism(parallel, () => m.HasSelfIntersections());
					await Assert.That(verdict).IsEqualTo(expected).Because($"{file}, parallel={parallel}");
				}

				// A clean part big enough that the parallel run really runs in parallel.
				Manifold sphere = Manifold.Sphere(1.0, 64);
				bool sphereVerdict = WithParallelism(parallel, () => sphere.HasSelfIntersections());
				await Assert.That(sphereVerdict).IsFalse().Because($"sphere, parallel={parallel}");
			}

			// Anti-vacuity, with ParallelismTests' probe (Task.CurrentId inside the body):
			// the scan's own helper, at each part's triangle count and the scan's
			// threshold, really enters Parallel.For with the switch on and the plain loop
			// with it off, so the parallel verdicts above came from the parallel loop.
			// The parts that go parallel are the two folds and the sphere.
			int[] sizes =
			{
				LoadFixture(cases[0].File).NumTri(),
				LoadFixture(cases[1].File).NumTri(),
				Manifold.Sphere(1.0, 64).NumTri(),
			};
			foreach (int n in sizes)
			{
				await Assert.That(WithParallelism(true, () => AnyRanInAParallelLoop(n, Soup.SelfIntersectParThreshold)))
					.IsTrue().Because($"n={n}, switch on");
				await Assert.That(WithParallelism(false, () => AnyRanInAParallelLoop(n, Soup.SelfIntersectParThreshold)))
					.IsFalse().Because($"n={n}, switch off");
			}
		}

		/// <summary>
		/// A cancelled scan answers true (route to robust) and caches nothing, so the next
		/// uncancelled call computes the real verdict.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task CancelledScanAnswersTrueAndCachesNothing()
		{
			foreach (bool parallel in new[] { false, true })
			{
				ManifoldImpl imp = Manifold.Sphere(1.0, 64).AsImpl();
				CancelToken token = new CancelToken();
				token.Cancel();
				bool cancelled = WithParallelism(parallel, () => Soup.HasSelfIntersectionsWithToken(imp, token));
				await Assert.That(cancelled).IsTrue().Because($"parallel={parallel}");
				await Assert.That(imp.SelfIntersects.Get()).IsNull().Because($"parallel={parallel}");
				bool real = WithParallelism(parallel, () => Soup.HasSelfIntersections(imp));
				await Assert.That(real).IsFalse().Because($"parallel={parallel}");
				await Assert.That(imp.SelfIntersects.Get() == false).IsTrue();
			}
		}

		/// <summary>
		/// Two triangles sharing one vertex, where t2's corners straddle t1's plane but
		/// t1's two other corners are strictly above t2's plane: they meet only at the
		/// shared vertex. In this orientation RealSelfContact's own shortcut (t2's corners
		/// against t1's plane) cannot decide and pays for the full tri-tri test; the new
		/// mirror shortcut in GenuineContact decides it before RealSelfContact is reached,
		/// with the same verdict.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task VertexNeighbourWithOnlyItsOwnCornersOneSidedIsBenign()
		{
			Vec3[] t1 = { V(0, 0, 0), V(1, 0, 1), V(0, -1, 1) };
			Vec3[] t2 = { V(0, 0, 0), V(2, 0, 0), V(0, 2, 0) };

			// The old path alone: its shortcut misses, TriTri finds the point contact.
			SelfCutStats old = new SelfCutStats();
			await Assert.That(GraphSelfCut.RealSelfContact(t1, t2, old)).IsNull();
			await Assert.That(old.VertBenign).IsEqualTo(0).Because("RealSelfContact's shortcut cannot see this orientation");
			await Assert.That(old.Full).IsEqualTo(1);
			await Assert.That(old.FullPoint).IsEqualTo(1);

			// GenuineContact: the same verdict, from the new shortcut, without TriTri.
			SelfCutStats stats = new SelfCutStats();
			await Assert.That(Soup.GenuineContact(t1, t2, stats)).IsFalse();
			await Assert.That(stats.VertBenign).IsEqualTo(1).Because("the new shortcut counted it");
			await Assert.That(stats.Full).IsEqualTo(0).Because("the new shortcut, not TriTri, decided");
		}

		/// <summary>
		/// A vertex neighbour with one of t1's other corners exactly on t2's plane (sign
		/// Zero) is not "strictly on one side", so the new shortcut must fall through to
		/// RealSelfContact and give its verdict, by the same path.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task VertexNeighbourWithACornerOnThePlaneFallsThrough()
		{
			// (-1, -1, 0) lies on t2's plane z = 0, outside t2; (0, -1, 1) is above it.
			Vec3[] t1 = { V(0, 0, 0), V(-1, -1, 0), V(0, -1, 1) };
			Vec3[] t2 = { V(0, 0, 0), V(2, 0, 0), V(0, 2, 0) };

			SelfCutStats old = new SelfCutStats();
			bool oldVerdict = GraphSelfCut.RealSelfContact(t1, t2, old) is not null;

			SelfCutStats stats = new SelfCutStats();
			await Assert.That(Soup.GenuineContact(t1, t2, stats)).IsEqualTo(oldVerdict);
			await Assert.That(oldVerdict).IsFalse().Because("t1 meets z = 0 outside t2 except at the shared vertex");
			await Assert.That(stats.VertBenign).IsEqualTo(old.VertBenign).Because("the new shortcut did not count it");
			await Assert.That(stats.Full).IsEqualTo(1).Because("it fell through to RealSelfContact's full test");
			await Assert.That(stats.Full).IsEqualTo(old.Full);
		}

		/// <summary>
		/// A hit and a cancel together answer true: the hit is a genuine witness, so the
		/// "true" stands whether or not other workers then saw the cancel. The first
		/// predicate call cancels the token and reports a hit, so the cancel is always in
		/// flight with the hit, on both loops.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task AHitRacingACancelAnswersTrue()
		{
			foreach (bool parallel in new[] { false, true })
			{
				CancelToken token = new CancelToken();
				int claimed = 0;
				bool? answer = WithParallelism(parallel, () => Par.MaybeParAnyCt(
					100_000,
					Soup.SelfIntersectParThreshold,
					token,
					() => 0,
					(i, _) =>
					{
						if (Interlocked.Exchange(ref claimed, 1) != 0)
						{
							return false;
						}

						token.Cancel();
						return true;
					}));

				await Assert.That(token.IsCancelled).IsTrue().Because($"parallel={parallel}");
				await Assert.That(answer).IsNotNull().Because($"parallel={parallel}");
				await Assert.That(answer!.Value).IsTrue().Because($"parallel={parallel}");
			}

			// And the parallel leg above really ran the parallel loop.
			await Assert.That(WithParallelism(true, () => AnyRanInAParallelLoop(100_000, Soup.SelfIntersectParThreshold)))
				.IsTrue();
		}

		/// <summary>
		/// Two triangles sharing one vertex whose other corners straddle each other's
		/// planes and cross through each other's interior: still a genuine contact.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task VertexNeighbourThatCrossesIsAContact()
		{
			Vec3[] t1 = { V(0, 0, 0), V(1, 0.5, 1), V(0.5, 1, -1) };
			Vec3[] t2 = { V(0, 0, 0), V(2, 0, 0), V(0, 2, 0) };

			await Assert.That(Soup.GenuineContact(t1, t2, new SelfCutStats())).IsTrue();
			await Assert.That(Soup.GenuineContact(t2, t1, new SelfCutStats())).IsTrue();
		}

		/// <summary>
		/// Auto picks the same engine whichever operand comes first, and scans the smaller
		/// operand first: when that one self-intersects, the larger is never scanned.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task AutoPicksTheSameEngineWhicheverOperandComesFirst()
		{
			Manifold fold = LoadFixture("selfisect-fold-F594AC8BB1542BE9.txt");
			Manifold clean = Manifold.Sphere(1.0, 64);
			await Assert.That(clean.NumTri()).IsGreaterThan(fold.NumTri());

			await Assert.That(RanExactEngine(clean, fold)).IsFalse().Because("clean, fold");
			await Assert.That(clean.AsImpl().SelfIntersects.Get()).IsNull()
				.Because("the smaller, self-intersecting operand decided alone");
			await Assert.That(RanExactEngine(fold, clean)).IsFalse().Because("fold, clean");

			Manifold small = Manifold.Cube(V(1, 1, 1), false).Translate(V(0.5, 0, 0));
			await Assert.That(RanExactEngine(clean, small)).IsTrue().Because("clean, small");
			await Assert.That(RanExactEngine(small, clean)).IsTrue().Because("small, clean");
		}

		private static bool RanExactEngine(Manifold a, Manifold b)
		{
			List<string> phases = new List<string>();
			ProgressReporter reporter = new ProgressReporter((phase, _) =>
			{
				lock (phases)
				{
					phases.Add(phase.Name());
				}
			});
			Manifold r = a.BooleanWithEngineAndProgress(b, OpType.Add, BooleanEngine.Auto, null, reporter);
			if (r.Status() != Error.NoError)
			{
				throw new InvalidOperationException($"Auto union failed: {r.Status()}");
			}

			return phases.Contains(Phase.ExactBoolean.Name());
		}

		/// <summary>
		/// ParallelismTests.RanInAParallelLoop's probe, applied to Par.MaybeParAnyCt:
		/// whether the predicate ran inside a TPL task (Parallel.For) rather than the
		/// plain loop. See that helper's remarks for why it asks which loop ran instead of
		/// counting threads, and why it fails closed.
		/// </summary>
		/// <param name="n">Number of indices.</param>
		/// <param name="threshold">The threshold to pass to the helper.</param>
		/// <returns>True when the parallel branch was taken.</returns>
		private static bool AnyRanInAParallelLoop(int n, int threshold)
		{
			int insideTask = 0;
			Par.MaybeParAnyCt(n, threshold, null, () => 0, (i, _) =>
			{
				if (Task.CurrentId is not null)
				{
					Volatile.Write(ref insideTask, 1);
				}

				return false;
			});
			return Volatile.Read(ref insideTask) != 0;
		}

		private static T WithParallelism<T>(bool enabled, Func<T> operation)
		{
			bool restore = ManifoldParallel.Enabled;
			try
			{
				ManifoldParallel.Enabled = enabled;
				return operation();
			}
			finally
			{
				ManifoldParallel.Enabled = restore;
			}
		}

		private static Vec3 V(double x, double y, double z)
		{
			return new Vec3(x, y, z);
		}

		/// <summary>
		/// Reads a "numVert numTri" fixture (one "x y z" line per vertex, one "v0 v1 v2"
		/// line per triangle) and imports it the way agg's ManifoldKernel does.
		/// </summary>
		/// <param name="fileName">The fixture's file name under TestData/regressions.</param>
		/// <returns>The imported manifold.</returns>
		private static Manifold LoadFixture(string fileName)
		{
			string path = Path.Combine(AppContext.BaseDirectory, "TestData", "regressions", fileName);
			string[] lines = File.ReadAllLines(path);
			string[] header = lines[0].Split(' ');
			int numVert = int.Parse(header[0], CultureInfo.InvariantCulture);
			int numTri = int.Parse(header[1], CultureInfo.InvariantCulture);

			MeshGL64 mesh = new MeshGL64();
			mesh.NumProp = 3;
			mesh.VertProperties = new List<double>(3 * numVert);
			mesh.TriVerts = new List<ulong>(3 * numTri);
			for (int v = 0; v < numVert; v++)
			{
				foreach (string x in lines[1 + v].Split(' '))
				{
					mesh.VertProperties.Add(double.Parse(x, CultureInfo.InvariantCulture));
				}
			}

			for (int t = 0; t < numTri; t++)
			{
				foreach (string x in lines[1 + numVert + t].Split(' '))
				{
					mesh.TriVerts.Add(ulong.Parse(x, CultureInfo.InvariantCulture));
				}
			}

			Manifold imported = Manifold.FromMeshGL64Robust(mesh);
			Manifold repaired = imported.RepairOrientation();
			Manifold original = repaired.AsOriginal();
			return original.Status() == Error.NoError ? original : repaired;
		}
	}
}
