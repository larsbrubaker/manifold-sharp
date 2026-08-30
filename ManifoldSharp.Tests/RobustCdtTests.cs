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

// RobustCdtTests.cs — port of robust/cdt_tests.rs, whose header reads:
//
//   Unit tests for the exact constrained Delaunay triangulation
//   (robust/cdt.rs): structural validity (area conservation, Euler
//   characteristic, CCW orientation), the Delaunay property on unconstrained
//   edges, constraint preservation, and randomized fuzzing.
//
// Same inputs, same expected values, same order as the Rust's 11 tests. The two
// fuzz tests reproduce the Rust's LCG bit-for-bit (u64 wrapping arithmetic, same
// seeds, same draw order) so the point sets they exercise are the Rust's point
// sets; BTreeSet becomes SortedSet, whose iteration order is the same ascending
// order the Rust relies on when it pushes the drawn points.

using System.Numerics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Tests
{
	public class RobustCdtTests
	{
		private static R2 R2Of(double x, double y)
		{
			return R2.FromVec2(new Vec2(x, y));
		}

		/// <summary>Twice the signed area, exact.</summary>
		private static BigRational Area2(R2 a, R2 b, R2 c)
		{
			return b.Sub(a).Cross(c.Sub(a));
		}

		/// <summary>
		/// Full structural validation of a triangulation of <paramref name="points"/>
		/// (corners first): CCW triangles, exact area conservation, Euler characteristic
		/// V-E+T = 1, every point used, constraints present, and local Delaunay on every
		/// unconstrained internal edge.
		/// </summary>
		private static async Task Validate(
			IReadOnlyList<R2> points,
			IReadOnlyList<(int A, int B)> constraints,
			IReadOnlyList<IVec3> tris)
		{
			// CCW and area sum.
			BigRational total = Backend.RatZero();
			foreach (IVec3 t in tris)
			{
				BigRational a2 = Area2(points[t.X], points[t.Y], points[t.Z]);
				await Assert.That(SignFunctions.OfRat(a2)).IsEqualTo(Sign.Pos)
					.Because($"sub-triangle not CCW: [{t.X}, {t.Y}, {t.Z}]");
				total += a2;
			}

			BigRational baseArea = Backend.RatAbs(Area2(points[0], points[1], points[2]));
			await Assert.That(total).IsEqualTo(baseArea)
				.Because("sub-triangle areas do not sum to the base area");

			// Edge set + Euler.
			SortedSet<(int, int)> edges = new SortedSet<(int, int)>();
			SortedSet<int> used = new SortedSet<int>();
			foreach (IVec3 t in tris)
			{
				for (int e = 0; e < 3; e++)
				{
					int u = t[e];
					int v = t[(e + 1) % 3];
					edges.Add((Math.Min(u, v), Math.Max(u, v)));
					used.Add(u);
				}
			}

			await Assert.That(used.Count).IsEqualTo(points.Count)
				.Because("not every point appears in the output");
			long euler = points.Count - (long)edges.Count + tris.Count;
			await Assert.That(euler).IsEqualTo(1L).Because("Euler characteristic violated");

			// Constraints preserved.
			foreach ((int a, int b) in constraints)
			{
				await Assert.That(edges.Contains((Math.Min(a, b), Math.Max(a, b)))).IsTrue()
					.Because($"constraint ({a},{b}) missing from output");
			}

			// Local Delaunay on unconstrained internal edges: build edge → tris map.
			SortedSet<(int, int)> con = new SortedSet<(int, int)>();
			foreach ((int a, int b) in constraints)
			{
				con.Add((Math.Min(a, b), Math.Max(a, b)));
			}

			SortedDictionary<(int, int), List<int>> edgeTris = new SortedDictionary<(int, int), List<int>>();
			for (int i = 0; i < tris.Count; i++)
			{
				IVec3 t = tris[i];
				for (int e = 0; e < 3; e++)
				{
					int u = t[e];
					int v = t[(e + 1) % 3];
					(int, int) key = (Math.Min(u, v), Math.Max(u, v));
					if (!edgeTris.TryGetValue(key, out List<int>? owners))
					{
						owners = new List<int>();
						edgeTris[key] = owners;
					}

					owners.Add(i);
				}
			}

			foreach (KeyValuePair<(int, int), List<int>> entry in edgeTris)
			{
				(int, int) edge = entry.Key;
				List<int> owners = entry.Value;
				await Assert.That(owners.Count <= 2).IsTrue()
					.Because($"edge shared by {owners.Count} triangles");
				if (owners.Count != 2 || con.Contains(edge))
				{
					continue;
				}

				foreach ((int i, int j) in new[] { (0, 1), (1, 0) })
				{
					IVec3 t = tris[owners[i]];
					IVec3 other = tris[owners[j]];
					int d = -1;
					for (int k = 0; k < 3; k++)
					{
						if (other[k] != edge.Item1 && other[k] != edge.Item2)
						{
							d = other[k];
							break;
						}
					}

					await Assert.That(Predicates.IncircleR(points[t.X], points[t.Y], points[t.Z], points[d]))
						.IsNotEqualTo(Sign.Pos)
						.Because($"unconstrained edge ({edge.Item1}, {edge.Item2}) strictly violates Delaunay");
				}
			}
		}

		[Test]
		public async Task BareTrianglePassesThrough()
		{
			List<R2> pts = new List<R2> { R2Of(0.0, 0.0), R2Of(4.0, 0.0), R2Of(0.0, 4.0) };
			List<IVec3> tris = CdtFunctions.Triangulate(pts, Array.Empty<(int, int)>());
			await Assert.That(tris.Count).IsEqualTo(1);
			await Validate(pts, Array.Empty<(int, int)>(), tris);
		}

		[Test]
		public async Task CwCornerOrderIsNormalized()
		{
			List<R2> pts = new List<R2> { R2Of(0.0, 0.0), R2Of(0.0, 4.0), R2Of(4.0, 0.0) }; // CW
			List<IVec3> tris = CdtFunctions.Triangulate(pts, Array.Empty<(int, int)>());
			await Assert.That(tris.Count).IsEqualTo(1);
			await Validate(pts, Array.Empty<(int, int)>(), tris);
		}

		[Test]
		public async Task InteriorPointSplitsIntoThree()
		{
			List<R2> pts = new List<R2>
			{
				R2Of(0.0, 0.0), R2Of(4.0, 0.0), R2Of(0.0, 4.0), R2Of(1.0, 1.0),
			};
			List<IVec3> tris = CdtFunctions.Triangulate(pts, Array.Empty<(int, int)>());
			await Assert.That(tris.Count).IsEqualTo(3);
			await Validate(pts, Array.Empty<(int, int)>(), tris);
		}

		[Test]
		public async Task PointOnHullEdgeSplitsIntoTwo()
		{
			List<R2> pts = new List<R2>
			{
				R2Of(0.0, 0.0), R2Of(4.0, 0.0), R2Of(0.0, 4.0), R2Of(2.0, 0.0),
			};
			List<IVec3> tris = CdtFunctions.Triangulate(pts, Array.Empty<(int, int)>());
			await Assert.That(tris.Count).IsEqualTo(2);
			await Validate(pts, Array.Empty<(int, int)>(), tris);
		}

		[Test]
		public async Task PointOnInteriorEdgeSplitsBothSides()
		{
			// Interior point first creates internal edges; a second point on one of
			// them must split both adjacent triangles.
			List<R2> pts = new List<R2>
			{
				R2Of(0.0, 0.0),
				R2Of(8.0, 0.0),
				R2Of(0.0, 8.0),
				R2Of(2.0, 2.0),
				R2Of(1.0, 1.0), // on segment (0,0)-(2,2), an internal edge of the split
			};
			List<IVec3> tris = CdtFunctions.Triangulate(pts, Array.Empty<(int, int)>());
			await Validate(pts, Array.Empty<(int, int)>(), tris);
		}

		[Test]
		public async Task ConstraintForcesNonDelaunayDiagonal()
		{
			// Two interior points forming a quad with the corners where Delaunay
			// prefers one diagonal; constrain the other and verify it survives.
			List<R2> pts = new List<R2>
			{
				R2Of(0.0, 0.0),
				R2Of(10.0, 0.0),
				R2Of(0.0, 10.0),
				R2Of(3.0, 1.0),
				R2Of(1.0, 3.0),
			};
			(int, int)[] con = new[] { (3, 4) };
			List<IVec3> tris = CdtFunctions.Triangulate(pts, con);
			await Validate(pts, con, tris);
		}

		[Test]
		public async Task ConstraintChainOfCollinearPoints()
		{
			// A polyline of collinear points across the triangle, each sub-segment
			// constrained (as the arrangement emits them).
			List<R2> pts = new List<R2>
			{
				R2Of(0.0, 0.0),
				R2Of(8.0, 0.0),
				R2Of(0.0, 8.0),
				R2Of(1.0, 1.0),
				R2Of(2.0, 2.0),
				R2Of(3.0, 3.0),
			};
			(int, int)[] con = new[] { (3, 4), (4, 5) };
			List<IVec3> tris = CdtFunctions.Triangulate(pts, con);
			await Validate(pts, con, tris);
		}

		[Test]
		public async Task CrossingStarConstraintsSplitAtCenter()
		{
			// Two constraint chains through a shared center point (as the
			// arrangement produces for crossing intersection segments).
			List<R2> pts = new List<R2>
			{
				R2Of(0.0, 0.0),
				R2Of(12.0, 0.0),
				R2Of(0.0, 12.0),
				R2Of(2.0, 2.0), // center
				R2Of(1.0, 2.0),
				R2Of(3.0, 2.0),
				R2Of(2.0, 1.0),
				R2Of(2.0, 3.0),
			};
			(int, int)[] con = new[] { (4, 3), (3, 5), (6, 3), (3, 7) };
			List<IVec3> tris = CdtFunctions.Triangulate(pts, con);
			await Validate(pts, con, tris);
		}

		[Test]
		public async Task FuzzRandomInteriorPoints()
		{
			// Deterministic LCG; integer-coordinate points inside the triangle
			// x,y >= 1, x+y <= 62 of base triangle (0,0),(64,0),(0,64).
			ulong state = 0x853C49E6748FEA9BUL;
			ulong Next()
			{
				state = unchecked((state * 6364136223846793005UL) + 1442695040888963407UL);
				return state;
			}

			for (int round = 0; round < 30; round++)
			{
				List<R2> pts = new List<R2> { R2Of(0.0, 0.0), R2Of(64.0, 0.0), R2Of(0.0, 64.0) };
				SortedSet<(ulong X, ulong Y)> seen = new SortedSet<(ulong, ulong)>();
				int n = 3 + (int)(Next() % 40);
				while (seen.Count < n)
				{
					ulong x = 1 + (Next() % 60);
					ulong y = 1 + (Next() % (62 - Math.Min(x, 60UL)));
					if (x + y <= 62)
					{
						seen.Add((x, y));
					}
				}

				foreach ((ulong x, ulong y) in seen)
				{
					pts.Add(R2Of(x, y));
				}

				List<IVec3> tris = CdtFunctions.Triangulate(pts, Array.Empty<(int, int)>());
				await Validate(pts, Array.Empty<(int, int)>(), tris);
				await Assert.That(tris.Count).IsEqualTo((2 * seen.Count) + 1)
					.Because($"round {round}: T = 2*interior + 1");
			}
		}

		[Test]
		public async Task FuzzRandomConstraints()
		{
			// Random interior points plus non-crossing constraints built from a
			// random walk (consecutive pairs share endpoints only).
			ulong state = 0xDA3E39CB94B95BDBUL;
			ulong Next()
			{
				state = unchecked((state * 6364136223846793005UL) + 1442695040888963407UL);
				return state;
			}

			for (int round = 0; round < 20; round++)
			{
				List<R2> pts = new List<R2> { R2Of(0.0, 0.0), R2Of(64.0, 0.0), R2Of(0.0, 64.0) };
				SortedSet<(ulong X, ulong Y)> seen = new SortedSet<(ulong, ulong)>();
				while (seen.Count < 12)
				{
					ulong x = 1 + (Next() % 60);
					ulong y = 1 + (Next() % (62 - Math.Min(x, 60UL)));
					if (x + y <= 62)
					{
						seen.Add((x, y));
					}
				}

				foreach ((ulong x, ulong y) in seen)
				{
					pts.Add(R2Of(x, y));
				}

				// One constraint between a random pair, provided no other point lies
				// strictly inside the segment and it stays inside the triangle hull
				// (always true for interior endpoints).
				List<(int A, int B)> con = new List<(int, int)>();
				for (int attempt = 0; attempt < 8; attempt++)
				{
					int a = 3 + (int)(Next() % ((ulong)pts.Count - 3));
					int b = 3 + (int)(Next() % ((ulong)pts.Count - 3));
					if (a == b)
					{
						continue;
					}

					// Reject if any point lies strictly inside segment (a,b) or an
					// existing constraint properly crosses it — mirrors the
					// arrangement's preconditions.
					bool rejected = false;
					for (int i = 0; i < pts.Count; i++)
					{
						if (i == a || i == b)
						{
							continue;
						}

						if (Predicates.Orient2dR(pts[a], pts[b], pts[i]) == Sign.Zero)
						{
							R2 d = pts[b].Sub(pts[a]);
							BigRational t = pts[i].Sub(pts[a]).Dot(d);
							if (t > Backend.RatZero() && t < d.Dot(d))
							{
								rejected = true;
								break;
							}
						}
					}

					if (rejected)
					{
						continue;
					}

					foreach ((int c, int d2) in con)
					{
						Sign s1 = Predicates.Orient2dR(pts[a], pts[b], pts[c]);
						Sign s2 = Predicates.Orient2dR(pts[a], pts[b], pts[d2]);
						Sign s3 = Predicates.Orient2dR(pts[c], pts[d2], pts[a]);
						Sign s4 = Predicates.Orient2dR(pts[c], pts[d2], pts[b]);
						if (s1 != Sign.Zero
							&& s2 != Sign.Zero
							&& s1 != s2
							&& s3 != Sign.Zero
							&& s4 != Sign.Zero
							&& s3 != s4)
						{
							rejected = true;
							break;
						}
					}

					if (rejected)
					{
						continue;
					}

					con.Add((a, b));
				}

				List<IVec3> tris = CdtFunctions.Triangulate(pts, con);
				await Validate(pts, con, tris);
			}
		}

		/// <summary>
		/// <see cref="CdtFunctions.IsExact"/> gates the exact-input filters, so it must never
		/// call an inexactly-rounded rational exact. Checked against the correctly rounded
		/// conversion in both directions: every accepted value round-trips, and the values it
		/// rejects are only ever a lost optimization.
		/// </summary>
		[Test]
		public async Task IsExactOnlyAcceptsValuesThatRoundTrip()
		{
			int accepted = 0;
			ulong lcg = 0x1234_5678_9abc_def0UL;
			ulong Next()
			{
				lcg = unchecked((lcg * 6364136223846793005UL) + 1442695040888963407UL);
				return lcg;
			}

			for (ulong i = 0; i < 20_000UL; i++)
			{
				// A mix of exact f64s, integer ratios (mostly non-dyadic), and
				// deliberately huge-denominator constructions.
				BigRational r;
				switch (i % 4)
				{
					case 0:
						r = Backend.RatFromF64(BitConverter.UInt64BitsToDouble(Next() | (1UL << 52)) % 1e6)
							?? default(BigRational);
						break;
					case 1:
						r = Backend.RatNew(
							new BigInteger((long)(Next() % 100_000)),
							new BigInteger((long)((Next() % 100_000) + 1)));
						break;
					case 2:
						r = Backend.RatNew(
							new BigInteger((long)(Next() % 1000)),
							BigInteger.Pow(new BigInteger(3), 40));
						break;
					default:
						r = Backend.RatFromF64((((double)(Next() % 2000)) - 1000.0) / 64.0)
							?? throw new InvalidOperationException("finite value must convert");
						break;
				}

				double f = Rational.RatToF64(r);
				if (CdtFunctions.IsExact(r))
				{
					accepted++;
					bool roundTrips = double.IsFinite(f)
						&& Backend.RatEq(
							Backend.RatFromF64(f) ?? throw new InvalidOperationException("finite value must convert"),
							r);
					await Assert.That(roundTrips).IsTrue()
						.Because($"is_exact accepted a value that does not round-trip: {r}");
				}
			}

			await Assert.That(accepted > 5_000).IsTrue()
				.Because($"test should exercise the accept path ({accepted})");
		}
	}
}
