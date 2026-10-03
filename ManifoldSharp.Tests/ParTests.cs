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

// Port of the tests module in par.rs (par_tests.rs, manifold-rust 6e127b5 and
// cf5463e) — the parallel sites must give the sequential output. Same models,
// same pinned hashes, same order.
//
// The Rust runs each model on rayon pools of 1 and 8 threads, which with the
// `parallel` feature exercises the parallel sites and without it runs the
// sequential code twice. The C# parallel switch is a runtime flag
// (ManifoldParallel.Enabled, Par.cs), so `run_on` becomes "with the switch off,
// then on": both must give the hash, which the Rust pinned from main's
// sequential code before the parallel stages landed. Every test therefore runs
// under ParallelismTests.ParallelismGlobalStateKey.
//
// The fingerprint is FNV-1a 64 over every MeshGL64 field but RunOriginalId (it
// comes from the process-wide ID counter), byte for byte the Rust's: NumProp,
// then VertProperties, TriVerts, MergeFromVert, MergeToVert, RunIndex,
// RunTransform, FaceId, HalfedgeTangent and RunFlags, each as its length then its
// elements, then Tolerance. Lengths and integers are u64 little-endian, floats
// their IEEE bits as u64 little-endian, RunFlags one byte each. Every coordinate
// is an integer or a half and nothing calls sin or cos, so the hashes hold on
// every platform.

using System.Buffers.Binary;

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ParTests
	{
		/// <summary>
		/// Pins a union of 500 cubes touching along edges and corners, one at a time, so
		/// every union is below the parallel thresholds of the cleanup.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task CheckerboardUnionKeepsTheSequentialOutput()
		{
			static Manifold Model()
			{
				Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
				Manifold result = Manifold.Empty();
				for (int x = 0; x < 10; x++)
				{
					for (int y = 0; y < 10; y++)
					{
						for (int z = 0; z < 10; z++)
						{
							if ((x + y + z) % 2 == 0)
							{
								result = result.Union(cube.Translate(new Vec3(x, y, z)));
							}
						}
					}
				}

				return result;
			}

			foreach (bool parallel in new[] { false, true })
			{
				await Assert.That(RunOn(parallel, Model).Hash).IsEqualTo(0x63ff_dc74_1011_f536UL)
					.Because($"parallel {parallel}");
			}
		}

		/// <summary>
		/// Checkerboards of 20 cubes a side, over the 100,001-halfedge thresholds: with the
		/// switch on they take the parallel paths, off the sequential ones, and both must
		/// give main's hashes. Unit cubes only touch; cubes of 1.5 overlap, so the unions cut
		/// edges. Each of the four classes of cubes is disjoint, so Compose builds it and
		/// three unions join them.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task LargeCheckerboardUnionsKeepTheSequentialOutput()
		{
			foreach ((double size, ulong hash) in new[] { (1.0, 0x7802_34fb_b0d5_3e14UL), (1.5, 0x4a6e_f237_508a_1f18UL) })
			{
				Manifold Model()
				{
					Manifold cube = Manifold.Cube(Vec3.Splat(size), false);
					Manifold Class(int ox, int oy, int oz)
					{
						List<Manifold> cubes = new List<Manifold>();
						for (int x = ox; x < 20; x += 2)
						{
							for (int y = oy; y < 20; y += 2)
							{
								for (int z = oz; z < 20; z += 2)
								{
									cubes.Add(cube.Translate(new Vec3(x, y, z)));
								}
							}
						}

						return Manifold.Compose(cubes);
					}

					Manifold a = Class(0, 0, 0).Union(Class(0, 1, 1));
					Manifold b = Class(1, 0, 1).Union(Class(1, 1, 0));
					return a.Union(b);
				}

				foreach (bool parallel in new[] { false, true })
				{
					(ulong got, Manifold model) = RunOn(parallel, Model);
					await Assert.That(3 * model.NumTri() > 100_001).IsTrue()
						.Because($"{model.NumTri()} triangles");
					await Assert.That(got).IsEqualTo(hash)
						.Because($"size {size}, parallel {parallel}");
				}
			}
		}

		/// <summary>
		/// An imported checkerboard of 20 a side, over the 100,001-halfedge thresholds of
		/// the orbit and edge-flag scans. Runs in every configuration, against main's hash.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task PinchedCheckerboardImportKeepsTheSequentialOutput()
		{
			foreach (bool parallel in new[] { false, true })
			{
				(ulong got, Manifold model) = RunOn(parallel, () => PinchedCheckerboard(20, false));
				await Assert.That(3 * model.NumTri() > 100_001).IsTrue()
					.Because($"{model.NumTri()} triangles");
				await Assert.That(got).IsEqualTo(0x1c3d_e063_12f8_0576UL)
					.Because($"parallel {parallel}");
			}
		}

		/// <summary>
		/// The same import, 26 a side with halfedge tangents: over 100,000 triangles, so
		/// SortGeometry sorts the faces and gathers the tangents in parallel with the switch
		/// on.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[NotInParallel(ParallelismTests.ParallelismGlobalStateKey)]
		public async Task PinchedCheckerboardImportWithTangentsKeepsTheSequentialOutput()
		{
			foreach (bool parallel in new[] { false, true })
			{
				(ulong got, Manifold result) = RunOn(parallel, () => PinchedCheckerboard(26, true));
				await Assert.That(3 * result.NumTri() > 100_001).IsTrue()
					.Because($"{result.NumTri()} triangles");
				await Assert.That(result.NumTri() >= 100_000).IsTrue()
					.Because($"{result.NumTri()} triangles");
				await Assert.That(result.GetMeshGL64(-1).HalfedgeTangent.Count)
					.IsEqualTo(12 * result.NumTri());
				await Assert.That(got).IsEqualTo(0x3a43_20b2_3394_3b5dUL)
					.Because($"parallel {parallel}");
			}
		}

		/// <summary>
		/// The Rust's <c>fingerprint</c>: FNV-1a 64 over the output MeshGL64; see the file
		/// header for the byte layout.
		/// </summary>
		/// <param name="m">The manifold to hash.</param>
		/// <returns>The hash.</returns>
		internal static ulong Fingerprint(Manifold m)
		{
			MeshGL64 gl = m.GetMeshGL64(-1);
			ulong h = 0xcbf2_9ce4_8422_2325UL;
			Span<byte> buffer = stackalloc byte[8];

			void EatByte(byte b)
			{
				h ^= b;
				unchecked
				{
					h *= 0x0100_0000_01b3UL;
				}
			}

			void EatU64(ulong x, Span<byte> scratch)
			{
				BinaryPrimitives.WriteUInt64LittleEndian(scratch, x);
				foreach (byte b in scratch)
				{
					EatByte(b);
				}
			}

			void Ints(List<ulong> list, Span<byte> scratch)
			{
				EatU64((ulong)list.Count, scratch);
				foreach (ulong x in list)
				{
					EatU64(x, scratch);
				}
			}

			void Floats(List<double> list, Span<byte> scratch)
			{
				EatU64((ulong)list.Count, scratch);
				foreach (double x in list)
				{
					EatU64((ulong)BitConverter.DoubleToInt64Bits(x), scratch);
				}
			}

			EatU64(gl.NumProp, buffer);
			Floats(gl.VertProperties, buffer);
			Ints(gl.TriVerts, buffer);
			Ints(gl.MergeFromVert, buffer);
			Ints(gl.MergeToVert, buffer);
			Ints(gl.RunIndex, buffer);
			Floats(gl.RunTransform, buffer);
			Ints(gl.FaceId, buffer);
			Floats(gl.HalfedgeTangent, buffer);
			EatU64((ulong)gl.RunFlags.Count, buffer);
			foreach (byte b in gl.RunFlags)
			{
				EatByte(b);
			}

			EatU64((ulong)BitConverter.DoubleToInt64Bits(gl.Tolerance), buffer);
			return h;
		}

		/// <summary>
		/// The Rust's <c>run_on</c>: <paramref name="f"/>'s fingerprint, with the parallel
		/// switch set as given and restored after, with the model for the size checks the
		/// Rust makes inside its builders.
		/// </summary>
		private static (ulong Hash, Manifold Model) RunOn(bool parallel, Func<Manifold> f)
		{
			bool restore = ManifoldParallel.Enabled;
			try
			{
				ManifoldParallel.Enabled = parallel;
				Manifold model = f();
				return (Fingerprint(model), model);
			}
			finally
			{
				ManifoldParallel.Enabled = restore;
			}
		}

		/// <summary>
		/// A checkerboard of <paramref name="n"/> unit cubes a side imported from a MeshGL64
		/// whose cubes share the verts where they touch, so SplitPinchedVerts has thousands
		/// of orbits to split. With <paramref name="tangents"/>, every halfedge carries a
		/// tangent of small integers, which the import keeps and SortGeometry gathers.
		/// </summary>
		private static Manifold PinchedCheckerboard(uint n, bool tangents)
		{
			// Corners as x | y << 1 | z << 2, wound outward.
			uint[][] quads = new uint[][]
			{
				new uint[] { 0, 4, 6, 2 },
				new uint[] { 1, 3, 7, 5 },
				new uint[] { 0, 1, 5, 4 },
				new uint[] { 2, 6, 7, 3 },
				new uint[] { 0, 2, 3, 1 },
				new uint[] { 4, 5, 7, 6 },
			};
			uint side = n + 1;
			MeshGL64 gl = new MeshGL64();
			gl.NumProp = 3;
			for (uint z = 0; z < side; z++)
			{
				for (uint y = 0; y < side; y++)
				{
					for (uint x = 0; x < side; x++)
					{
						gl.VertProperties.AddRange(new double[] { x, y, z });
					}
				}
			}

			for (uint z = 0; z < n; z++)
			{
				for (uint y = 0; y < n; y++)
				{
					for (uint x = 0; x < n; x++)
					{
						if ((x + y + z) % 2 != 0)
						{
							continue;
						}

						ulong Corner(uint c)
						{
							uint cx = x + (c & 1);
							uint cy = y + ((c >> 1) & 1);
							uint cz = z + (c >> 2);
							return cx + (side * (cy + (side * cz)));
						}

						foreach (uint[] q in quads)
						{
							foreach (uint c in new[] { q[0], q[1], q[2], q[0], q[2], q[3] })
							{
								gl.TriVerts.Add(Corner(c));
							}
						}
					}
				}
			}

			if (tangents)
			{
				for (uint i = 0; i < (uint)gl.TriVerts.Count; i++)
				{
					gl.HalfedgeTangent.AddRange(new double[]
					{
						(i % 7) - 3.0,
						(i % 5) - 2.0,
						i % 3,
						(i % 4) * 0.5,
					});
				}
			}

			return Manifold.FromMeshGL64(gl);
		}
	}
}
