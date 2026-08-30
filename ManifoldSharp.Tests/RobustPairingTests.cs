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

// RobustPairingTests.cs — port of robust/pairing_tests.rs, whose header reads:
//
//   Unit tests for robust/pairing.rs — the geometric half-edge pairing of the
//   extracted boundary, exercised in isolation on hand-built closed meshes
//   whose correct answer is known by construction.
//
//   The load-bearing cases are: two solids meeting along one edge (the k = 2
//   fan the arrangement produces at a pinch, where the pairing must keep each
//   solid's own two faces together), the k = 3 generalization, the rejection
//   paths, and — the guarantee every clean mesh depends on — that an ordinary
//   closed mesh is left entirely alone.
//
// Same inputs, same expected values, same order as the Rust's five tests.
//
// `deg.to_radians()` is one multiply by the correctly rounded pi/180, the mirror
// image of Smoothing.ToDegrees; Types.Radians (`a * K_PI / 180.0`) is two
// operations and differs in the last ulp, so the fixture spells the Rust's form.

using ManifoldSharp;
using ManifoldSharp.Linalg;
using ManifoldSharp.Robust;
using ManifoldSharp.Robust.Exact;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class RobustPairingTests
	{
		/// <summary>
		/// An ordinary closed mesh must not be touched at all: every undirected edge already
		/// has exactly two half-edges, so the import's own pairing is already the geometric
		/// one and assembly has to stay byte-identical.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task OrdinaryClosedMeshNeedsNoSplit()
		{
			Verts v = new Verts();
			List<UVec3> tris = BoxTris(v, new[] { 0.0, 0.0, 0.0 }, new[] { 1.0, 1.0, 1.0 });
			await Assert.That(Plan(tris, v)).IsNull();
		}

		/// <summary>
		/// Two cubes meeting along one edge: that edge carries four half-edges, and the
		/// pairing must join each cube's *own* two faces (across the material wedge), not one
		/// cube's face to the other's. The split then gives each cube its own copy of the two
		/// shared vertices.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task TwoCubesSharingAnEdgePairWithinEachCube()
		{
			Verts v = new Verts();
			List<UVec3> tris = BoxTris(v, new[] { 0.0, 0.0, 0.0 }, new[] { 1.0, 1.0, 1.0 });
			int nA = tris.Count;
			tris.AddRange(BoxTris(v, new[] { -1.0, -1.0, 0.0 }, new[] { 0.0, 0.0, 1.0 }));
			uint lo = v.Id(0.0, 0.0, 0.0);
			uint hi = v.Id(0.0, 0.0, 1.0);

			uint[]? plan = Plan(tris, v);
			await Assert.That(plan).IsNotNull().Because("four half-edges on the shared edge");
			foreach (uint shared in new[] { lo, hi })
			{
				HashSet<uint> a = CopiesAt(tris, plan!, shared, t => t < nA);
				HashSet<uint> b = CopiesAt(tris, plan!, shared, t => t >= nA);
				await Assert.That(a.Count).IsEqualTo(1)
					.Because($"cube A's corners at {shared} must be one fan");
				await Assert.That(b.Count).IsEqualTo(1)
					.Because($"cube B's corners at {shared} must be one fan");
				await Assert.That(a.SetEquals(b)).IsFalse()
					.Because($"the two cubes must not share a copy of {shared}");
			}

			// Only the two shared vertices split; everything else keeps copy 0.
			for (int t = 0; t < tris.Count; t++)
			{
				for (int c = 0; c < 3; c++)
				{
					if (tris[t][c] != lo && tris[t][c] != hi)
					{
						await Assert.That(plan![(3 * t) + c]).IsEqualTo(0u)
							.Because($"unpinched vertex {tris[t][c]} split");
					}
				}
			}
		}

		/// <summary>
		/// The k = 3 generalization: three wedges around one edge, six half-edges, three
		/// material wedges. Each wedge's two faces must stay paired together.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ThreeWedgesAroundAnEdgePairWithinEachWedge()
		{
			Verts v = new Verts();
			List<UVec3> tris = new List<UVec3>();
			List<(int Start, int End)> spans = new List<(int Start, int End)>();
			foreach ((double A, double B) span in new[] { (0.0, 60.0), (120.0, 180.0), (240.0, 300.0) })
			{
				int start = tris.Count;
				tris.AddRange(WedgeTris(v, span.A, span.B));
				spans.Add((start, tris.Count));
			}

			uint lo = v.Id(0.0, 0.0, 0.0);
			uint hi = v.Id(0.0, 0.0, 1.0);

			uint[]? plan = Plan(tris, v);
			await Assert.That(plan).IsNotNull().Because("six half-edges on the axis edge");
			foreach (uint axis in new[] { lo, hi })
			{
				HashSet<uint> seen = new HashSet<uint>();
				foreach ((int Start, int End) span in spans)
				{
					HashSet<uint> c = CopiesAt(tris, plan!, axis, t => t >= span.Start && t < span.End);
					await Assert.That(c.Count).IsEqualTo(1)
						.Because($"a wedge's corners at {axis} must be one fan");
					seen.UnionWith(c);
				}

				await Assert.That(seen.Count).IsEqualTo(3)
					.Because($"each wedge needs its own copy of {axis}");
			}
		}

		/// <summary>
		/// A surface where the traversals around a fan do not alternate is not a consistently
		/// oriented boundary, so no wedge pairing exists: reversing one cube's winding leaves
		/// every other edge well formed but breaks the alternation on the shared edge.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task NonAlternatingFanIsRejected()
		{
			Verts v = new Verts();
			List<UVec3> tris = BoxTris(v, new[] { 0.0, 0.0, 0.0 }, new[] { 1.0, 1.0, 1.0 });
			foreach (UVec3 t in BoxTris(v, new[] { -1.0, -1.0, 0.0 }, new[] { 0.0, 0.0, 1.0 }))
			{
				tris.Add(new UVec3(t[0], t[2], t[1]));
			}

			await Assert.That(Plan(tris, v)).IsNull();
		}

		/// <summary>An odd number of half-edges on an edge cannot be a closed surface.</summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task OddFanIsRejected()
		{
			Verts v = new Verts();
			List<UVec3> tris = BoxTris(v, new[] { 0.0, 0.0, 0.0 }, new[] { 1.0, 1.0, 1.0 });
			tris.AddRange(BoxTris(v, new[] { -1.0, -1.0, 0.0 }, new[] { 0.0, 0.0, 1.0 }));
			uint lo = v.Id(0.0, 0.0, 0.0);
			uint hi = v.Id(0.0, 0.0, 1.0);
			uint loose = v.Id(2.0, 2.0, 2.0);
			tris.Add(new UVec3(lo, hi, loose));
			await Assert.That(Plan(tris, v)).IsNull();
		}

		/// <summary>The 12 outward-wound triangles of an axis-aligned box.</summary>
		/// <param name="v">The shared vertex table.</param>
		/// <param name="lo">The low corner.</param>
		/// <param name="hi">The high corner.</param>
		/// <returns>The box's triangles as interned ids.</returns>
		private static List<UVec3> BoxTris(Verts v, double[] lo, double[] hi)
		{
			uint[] c =
			{
				v.Id(lo[0], lo[1], lo[2]),
				v.Id(hi[0], lo[1], lo[2]),
				v.Id(hi[0], hi[1], lo[2]),
				v.Id(lo[0], hi[1], lo[2]),
				v.Id(lo[0], lo[1], hi[2]),
				v.Id(hi[0], lo[1], hi[2]),
				v.Id(hi[0], hi[1], hi[2]),
				v.Id(lo[0], hi[1], hi[2]),
			};
			int[][] idx =
			{
				new[] { 0, 3, 2 },
				new[] { 0, 2, 1 }, // -z
				new[] { 4, 5, 6 },
				new[] { 4, 6, 7 }, // +z
				new[] { 0, 1, 5 },
				new[] { 0, 5, 4 }, // -y
				new[] { 1, 2, 6 },
				new[] { 1, 6, 5 }, // +x
				new[] { 2, 3, 7 },
				new[] { 2, 7, 6 }, // +y
				new[] { 3, 0, 4 },
				new[] { 3, 4, 7 }, // -x
			};
			List<UVec3> outTris = new List<UVec3>(idx.Length);
			foreach (int[] t in idx)
			{
				outTris.Add(new UVec3(c[t[0]], c[t[1]], c[t[2]]));
			}

			return outTris;
		}

		/// <summary>
		/// The 8 outward-wound triangles of a triangular prism whose cross-section is the
		/// wedge from angle <paramref name="a"/> to angle <paramref name="b"/> (degrees,
		/// <c>b - a &lt; 180</c>) of the unit circle, extruded along z from 0 to 1. Its apex
		/// edge is the z axis, so several of these share that edge.
		/// </summary>
		/// <param name="v">The shared vertex table.</param>
		/// <param name="a">The wedge's start angle, in degrees.</param>
		/// <param name="b">The wedge's end angle, in degrees.</param>
		/// <returns>The prism's triangles as interned ids.</returns>
		private static List<UVec3> WedgeTris(Verts v, double a, double b)
		{
			static (double X, double Y) Pt(double deg)
			{
				double r = ToRadians(deg);
				return (DeterministicMath.Cos(r), DeterministicMath.Sin(r));
			}

			uint[] Ring(double z)
			{
				(double X, double Y) p = Pt(a);
				(double X, double Y) q = Pt(b);
				return new[] { v.Id(0.0, 0.0, z), v.Id(p.X, p.Y, z), v.Id(q.X, q.Y, z) };
			}

			uint[] bot = Ring(0.0);
			uint[] top = Ring(1.0);
			List<UVec3> tris = new List<UVec3>
			{
				new UVec3(bot[0], bot[2], bot[1]), // -z cap (reversed cross-section)
				new UVec3(top[0], top[1], top[2]), // +z cap
			};
			for (int i = 0; i < 3; i++)
			{
				int j = (i + 1) % 3;
				tris.Add(new UVec3(bot[i], bot[j], top[j]));
				tris.Add(new UVec3(bot[i], top[j], top[i]));
			}

			return tris;
		}

		/// <summary>Rust's <c>f64::to_radians</c>: one multiply by the correctly rounded pi/180.</summary>
		/// <param name="degrees">The angle in degrees.</param>
		/// <returns>The angle in radians.</returns>
		private static double ToRadians(double degrees)
		{
			return degrees * (Types.KPi / 180.0);
		}

		/// <summary>Runs the planner against a fixture's vertex tables.</summary>
		/// <param name="tris">The triangles.</param>
		/// <param name="v">The vertex table.</param>
		/// <returns>The split plan, or null.</returns>
		private static uint[]? Plan(IReadOnlyList<UVec3> tris, Verts v)
		{
			(List<R3> Verts, List<Vec3> VertsF64) tables = v.Tables();
			return Pairing.PlanVertexSplits(tris, new VertTables(tables.Verts, tables.VertsF64));
		}

		/// <summary>
		/// The distinct fan copies assigned to <paramref name="vid"/>'s corners among
		/// <paramref name="group"/>'s triangles.
		/// </summary>
		/// <param name="tris">The triangles.</param>
		/// <param name="plan">The split plan.</param>
		/// <param name="vid">The vertex id to look at.</param>
		/// <param name="group">Which triangles to consider.</param>
		/// <returns>The distinct copy indices seen.</returns>
		private static HashSet<uint> CopiesAt(
			IReadOnlyList<UVec3> tris,
			uint[] plan,
			uint vid,
			Func<int, bool> group)
		{
			HashSet<uint> outSet = new HashSet<uint>();
			for (int t = 0; t < tris.Count; t++)
			{
				if (!group(t))
				{
					continue;
				}

				for (int c = 0; c < 3; c++)
				{
					if (tris[t][c] == vid)
					{
						outSet.Add(plan[(3 * t) + c]);
					}
				}
			}

			return outSet;
		}

		/// <summary>
		/// Deduplicating vertex table, mirroring the graph's interned ids: two solids that
		/// touch share the id of every point they share.
		/// </summary>
		private sealed class Verts
		{
			private readonly List<Vec3> pts = new List<Vec3>();

			/// <summary>Interns a point, by exact f64 equality (the Rust's linear scan).</summary>
			/// <param name="x">The x coordinate.</param>
			/// <param name="y">The y coordinate.</param>
			/// <param name="z">The z coordinate.</param>
			/// <returns>Its id.</returns>
			public uint Id(double x, double y, double z)
			{
				Vec3 p = new Vec3(x, y, z);
				for (int i = 0; i < this.pts.Count; i++)
				{
					if (this.pts[i].X == p.X && this.pts[i].Y == p.Y && this.pts[i].Z == p.Z)
					{
						return (uint)i;
					}
				}

				this.pts.Add(p);
				return (uint)(this.pts.Count - 1);
			}

			/// <summary>The exact and rounded tables the planner reads.</summary>
			/// <returns>The two tables.</returns>
			public (List<R3> Verts, List<Vec3> VertsF64) Tables()
			{
				List<R3> exact = new List<R3>(this.pts.Count);
				foreach (Vec3 p in this.pts)
				{
					exact.Add(R3.FromVec3(p));
				}

				return (exact, new List<Vec3>(this.pts));
			}
		}
	}
}
