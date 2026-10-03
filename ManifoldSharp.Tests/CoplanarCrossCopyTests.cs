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

// CoplanarCrossCopyTests.cs — C#-only adaptation tests (no Rust counterpart) for the
// robust engine's phase 3, the coplanar cross-copy. They pin two things: that the
// cross-copy's optimizations (per-region clip setup, bounding-box reject, hashed
// dedupe) leave the robust result bit-for-bit where the straight transcription of the
// Rust put it, and that the step reports its own progress phase rather than running
// silently under "self intersections".

using System.Security.Cryptography;
using ManifoldSharp;
using ManifoldSharp.Linalg;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	/// <summary>
	/// The robust engine's coplanar cross-copy on a self-touching body with a coplanar slab.
	/// </summary>
	public class CoplanarCrossCopyTests
	{
		/// <summary>
		/// SHA-256 of the robust union's MeshGL64 (tri verts, then vertex property bits),
		/// captured from the straight transcription of the Rust's phase 3 before it was
		/// optimized. Any change here is a change to a computed result, which the
		/// maintenance contract forbids.
		/// </summary>
		private const string FrozenUnionHash = "8E910C9A57BC34E4421978487DDFFB679CB62180E3B94B5C162877B30EA5CE76";

		/// <summary>
		/// The union's bits are the ones the unoptimized cross-copy produced, sequentially
		/// and in parallel.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task RobustUnionOverCoplanarFacesIsBitIdenticalToTheFrozenResult()
		{
			bool restore = ManifoldParallel.Enabled;
			try
			{
				foreach (bool parallel in new[] { false, true })
				{
					ManifoldParallel.Enabled = parallel;
					(Manifold body, Manifold slab) = Fixture();
					Manifold union = body.BooleanWithEngine(slab, OpType.Add, BooleanEngine.Robust);
					await Assert.That(union.Status()).IsEqualTo(Error.NoError);
					await Assert.That(Hash(union)).IsEqualTo(FrozenUnionHash)
						.Because($"parallel={parallel}");
				}
			}
			finally
			{
				ManifoldParallel.Enabled = restore;
			}
		}

		/// <summary>
		/// A robust boolean with coplanar overlap regions reports the cross-copy as its own
		/// determinate phase, closed at exactly 1.0, between self intersections and
		/// candidate points.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task TheCrossCopyReportsItsOwnPhase()
		{
			(Manifold body, Manifold slab) = Fixture();
			List<(string Name, double? Fraction)> events = new List<(string, double?)>();
			ProgressReporter reporter = new ProgressReporter((phase, fraction) =>
			{
				lock (events)
				{
					events.Add((phase.Name(), fraction));
				}
			});

			body.BooleanWithEngineAndProgress(slab, OpType.Add, BooleanEngine.Robust, null, reporter);

			List<string> order = new List<string>();
			foreach ((string name, double? _) in events)
			{
				if (order.Count == 0 || order[order.Count - 1] != name)
				{
					order.Add(name);
				}
			}

			int cross = order.IndexOf(Phase.CoplanarOverlaps.Name());
			await Assert.That(cross).IsGreaterThan(0)
				.Because($"phases seen: {string.Join(", ", order)}");
			await Assert.That(order[cross - 1]).IsEqualTo(Phase.SelfIntersections.Name());
			await Assert.That(order[cross + 1]).IsEqualTo(Phase.CandidatePoints.Name());

			double? last = null;
			foreach ((string name, double? fraction) in events)
			{
				if (name == Phase.CoplanarOverlaps.Name())
				{
					last = fraction;
				}
			}

			await Assert.That(last).IsEqualTo(1.0);
		}

		/// <summary>
		/// A 4x4 grid of unit cubes composed without a boolean, so neighbours touch face to
		/// face (the self-contact that sends Auto to the robust engine), and a slab whose
		/// bottom is coplanar with the grid's tops, offset so its triangles straddle many
		/// grid triangles: every slab-bottom triangle collects segments from many coplanar
		/// overlap regions, which is the shape that made phase 3 expensive.
		/// </summary>
		internal static (Manifold Body, Manifold Slab) Fixture()
		{
			List<Manifold> cubes = new List<Manifold>();
			for (int x = 0; x < 4; x++)
			{
				for (int y = 0; y < 4; y++)
				{
					cubes.Add(Manifold.Cube(Vec3.Splat(1.0), false).Translate(new Vec3(x, y, 0.0)));
				}
			}

			Manifold body = Manifold.Compose(cubes);
			Manifold slab = Manifold.Cube(new Vec3(3.0, 3.0, 0.5), false).Translate(new Vec3(0.25, 0.75, 1.0));
			return (body, slab);
		}

		internal static string Hash(Manifold m)
		{
			MeshGL64 mesh = m.GetMeshGL64(-1);
			using MemoryStream bytes = new MemoryStream();
			using (BinaryWriter writer = new BinaryWriter(bytes, System.Text.Encoding.UTF8, leaveOpen: true))
			{
				writer.Write(mesh.TriVerts.Count);
				foreach (ulong v in mesh.TriVerts)
				{
					writer.Write(v);
				}

				writer.Write(mesh.VertProperties.Count);
				foreach (double p in mesh.VertProperties)
				{
					writer.Write(BitConverter.DoubleToInt64Bits(p));
				}
			}

			return Convert.ToHexString(SHA256.HashData(bytes.ToArray()));
		}
	}
}
