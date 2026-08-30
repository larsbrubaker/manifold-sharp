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

// RobustArrangementTests.cs — port of robust/arrangement_tests.rs, whose header
// reads:
//
//   Unit tests for the per-triangle 2D arrangement (robust/arrangement.rs):
//   segment splitting at crossings and at points, provenance tracking,
//   constraint preservation through the CDT, and exact area conservation.
//
// Same inputs, same expected values, same order as the Rust's 11 tests. The box
// sweep's fuzz test reproduces the Rust's LCG bit-for-bit (u64 wrapping
// arithmetic, same seed, same draw order) so the box layouts it exercises are
// the Rust's box layouts, and the pair list is compared with
// CollectionOrdering.Matching because the whole point of the test is the ORDER.

using System.Numerics;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using ManifoldSharp;
using ManifoldSharp.Linalg;
using ManifoldSharp.Robust;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Tests
{
	public class RobustArrangementTests
	{
		/// <summary>Rust's `TRI`: the axis-aligned test triangle every case but one uses.</summary>
		private static Vec3[] Tri
		{
			get
			{
				return new Vec3[]
				{
					new Vec3(0.0, 0.0, 0.0),
					new Vec3(8.0, 0.0, 0.0),
					new Vec3(0.0, 8.0, 0.0),
				};
			}
		}

		private static Vec3 V(double x, double y, double z)
		{
			return new Vec3(x, y, z);
		}

		private static R3 R3Of(double x, double y, double z)
		{
			return R3.FromVec3(V(x, y, z));
		}

		private static BigRational Area2(R2 a, R2 b, R2 c)
		{
			return b.Sub(a).Cross(c.Sub(a));
		}

		/// <summary>
		/// Shared checks: CCW sub-triangles, exact area conservation, every
		/// constraint edge realized in the triangulation.
		/// </summary>
		private static async Task Validate(Arrangement arr)
		{
			BigRational total = Backend.RatZero();
			foreach (IVec3 t in arr.Tris)
			{
				BigRational a2 = Area2(arr.Points2[t.X], arr.Points2[t.Y], arr.Points2[t.Z]);
				await Assert.That(SignFunctions.OfRat(a2)).IsEqualTo(Sign.Pos)
					.Because("sub-triangle not CCW");
				total += a2;
			}

			BigRational baseArea = Backend.RatAbs(Area2(arr.Points2[0], arr.Points2[1], arr.Points2[2]));
			await Assert.That(total).IsEqualTo(baseArea).Because("area not conserved");

			SortedSet<(int, int)> edges = new SortedSet<(int, int)>();
			foreach (IVec3 t in arr.Tris)
			{
				for (int e = 0; e < 3; e++)
				{
					int u = t[e];
					int w = t[(e + 1) % 3];
					edges.Add((Math.Min(u, w), Math.Max(u, w)));
				}
			}

			foreach ((int A, int B) edge in arr.Constraints.Keys)
			{
				await Assert.That(edges.Contains((edge.A, edge.B))).IsTrue()
					.Because($"constraint edge ({edge.A}, {edge.B}) not realized");
			}
		}

		[Test]
		public async Task EmptyInputYieldsSingleTriangle()
		{
			Arrangement arr = ArrangementFunctions.Build(Tri, new ArrangementInput(), null)!;
			await Assert.That(arr.Tris.Count).IsEqualTo(1);
			await Assert.That(arr.Points3.Count).IsEqualTo(3);
			await Assert.That(arr.Constraints.Count).IsEqualTo(0);
			await Validate(arr);
		}

		[Test]
		public async Task SingleSegmentBecomesOneConstraint()
		{
			ArrangementInput input = new ArrangementInput
			{
				Points = new List<(R3, int)>(),
				Segments = new List<(R3, R3, int)>
				{
					(R3Of(1.0, 1.0, 0.0), R3Of(3.0, 2.0, 0.0), 7),
				},
			};
			Arrangement arr = ArrangementFunctions.Build(Tri, input, null)!;
			await Assert.That(arr.Constraints.Count).IsEqualTo(1);
			List<int> provs = arr.Constraints.Values.First();
			await Assert.That(provs).IsEquivalentTo(new List<int> { 7 }, CollectionOrdering.Matching);
			await Validate(arr);
		}

		[Test]
		public async Task CrossingSegmentsSplitAtIntersection()
		{
			// Two segments crossing at (2,2); each splits in half → 4 constraints,
			// and the crossing point must exist as an arrangement vertex.
			ArrangementInput input = new ArrangementInput
			{
				Points = new List<(R3, int)>(),
				Segments = new List<(R3, R3, int)>
				{
					(R3Of(1.0, 1.0, 0.0), R3Of(3.0, 3.0, 0.0), 0),
					(R3Of(1.0, 3.0, 0.0), R3Of(3.0, 1.0, 0.0), 1),
				},
			};
			Arrangement arr = ArrangementFunctions.Build(Tri, input, null)!;
			await Assert.That(arr.Constraints.Count).IsEqualTo(4);
			await Assert.That(arr.Points3.Contains(R3Of(2.0, 2.0, 0.0))).IsTrue()
				.Because("crossing point missing");

			// Each sub-constraint carries exactly its parent's provenance.
			foreach (KeyValuePair<(int A, int B), List<int>> kv in arr.Constraints)
			{
				await Assert.That(kv.Value.Count).IsEqualTo(1)
					.Because($"edge ({kv.Key.A}, {kv.Key.B}) has provenance [{string.Join(", ", kv.Value)}]");
			}

			await Validate(arr);
		}

		[Test]
		public async Task PointPrimitiveSplitsSegment()
		{
			// An isolated point primitive lying on a segment splits it.
			ArrangementInput input = new ArrangementInput
			{
				Points = new List<(R3, int)> { (R3Of(2.0, 1.0, 0.0), 5) },
				Segments = new List<(R3, R3, int)>
				{
					(R3Of(1.0, 1.0, 0.0), R3Of(3.0, 1.0, 0.0), 9),
				},
			};
			Arrangement arr = ArrangementFunctions.Build(Tri, input, null)!;
			await Assert.That(arr.Constraints.Count).IsEqualTo(2)
				.Because("point on segment must split it");
			foreach (List<int> provs in arr.Constraints.Values)
			{
				await Assert.That(provs).IsEquivalentTo(new List<int> { 9 }, CollectionOrdering.Matching);
			}

			await Validate(arr);
		}

		[Test]
		public async Task CollinearOverlappingSegmentsMergeProvenance()
		{
			// Two overlapping collinear segments: [1,4] and [2,6] on y=1. Points
			// 1,2,4,6 → sub-edges [1,2](prov 0), [2,4](both), [4,6](prov 1).
			ArrangementInput input = new ArrangementInput
			{
				Points = new List<(R3, int)>(),
				Segments = new List<(R3, R3, int)>
				{
					(R3Of(1.0, 1.0, 0.0), R3Of(4.0, 1.0, 0.0), 0),
					(R3Of(2.0, 1.0, 0.0), R3Of(6.0, 1.0, 0.0), 1),
				},
			};
			Arrangement arr = ArrangementFunctions.Build(Tri, input, null)!;
			await Assert.That(arr.Constraints.Count).IsEqualTo(3);

			List<List<int>> provSets = arr.Constraints.Values.Select(v => new List<int>(v)).ToList();
			foreach (List<int> p in provSets)
			{
				p.Sort();
			}

			// Rust sorts a Vec<Vec<usize>>, whose Ord is lexicographic with the shorter
			// prefix ordering first.
			provSets.Sort(LexicographicIntLists);
			List<List<int>> want = new List<List<int>>
			{
				new List<int> { 0 },
				new List<int> { 0, 1 },
				new List<int> { 1 },
			};
			await Assert.That(provSets.Count).IsEqualTo(want.Count);
			for (int i = 0; i < want.Count; i++)
			{
				await Assert.That(provSets[i]).IsEquivalentTo(want[i], CollectionOrdering.Matching);
			}

			await Validate(arr);
		}

		[Test]
		public async Task SegmentTouchingTriangleEdgeAndCorner()
		{
			// Segment from a corner to the middle of the opposite edge — endpoints
			// dedup against the corner registry and split the hull edge.
			ArrangementInput input = new ArrangementInput
			{
				Points = new List<(R3, int)>(),
				Segments = new List<(R3, R3, int)>
				{
					(R3Of(0.0, 0.0, 0.0), R3Of(4.0, 4.0, 0.0), 3),
				},
			};
			Arrangement arr = ArrangementFunctions.Build(Tri, input, null)!;
			await Assert.That(arr.Points3.Count).IsEqualTo(4).Because("only the midpoint is new");
			await Assert.That(arr.Constraints.Count).IsEqualTo(1);
			await Validate(arr);
		}

		[Test]
		public async Task PolygonBoundaryRing()
		{
			// A coplanar-overlap hexagon boundary as six segments with one
			// provenance — mimics how the pipeline feeds Coplanar results.
			R3[] hex = new R3[]
			{
				R3Of(2.0, 1.0, 0.0),
				R3Of(3.0, 1.0, 0.0),
				R3Of(4.0, 2.0, 0.0),
				R3Of(3.0, 3.0, 0.0),
				R3Of(2.0, 3.0, 0.0),
				R3Of(1.0, 2.0, 0.0),
			};
			List<(R3, R3, int)> segments = new List<(R3, R3, int)>();
			for (int i = 0; i < 6; i++)
			{
				segments.Add((hex[i], hex[(i + 1) % 6], 11));
			}

			Arrangement arr = ArrangementFunctions.Build(
				Tri,
				new ArrangementInput { Points = new List<(R3, int)>(), Segments = segments },
				null)!;
			await Assert.That(arr.Constraints.Count).IsEqualTo(6);
			await Validate(arr);
		}

		[Test]
		public async Task SkewPlaneArrangementLiftsExactly()
		{
			// Non-axis-aligned triangle: verify all 3D points still lie exactly on
			// its plane after crossing construction + lifting.
			Vec3[] tri = new Vec3[] { V(0.1, 0.2, 0.3), V(6.7, -0.4, 0.9), V(-0.6, 6.1, 2.2) };

			// Two crossing segments built from midpoints (guaranteed on the plane).
			R3 a = R3.FromVec3(tri[0]);
			R3 b = R3.FromVec3(tri[1]);
			R3 c = R3.FromVec3(tri[2]);
			BigRational half = Backend.RatNew(BigInteger.One, new BigInteger(2));
			R3 mab = a.Add(b).Scale(half);
			R3 mbc = b.Add(c).Scale(half);
			R3 mca = c.Add(a).Scale(half);
			ArrangementInput input = new ArrangementInput
			{
				Points = new List<(R3, int)>(),
				Segments = new List<(R3, R3, int)>
				{
					(mab, mbc, 0),
					(mbc, mca, 1),
					(mca, mab, 2),
				},
			};
			Arrangement arr = ArrangementFunctions.Build(tri, input, null)!;
			foreach (R3 p in arr.Points3)
			{
				await Assert.That(Predicates.Orient3dR(a, b, c, p)).IsEqualTo(Sign.Zero)
					.Because("point off plane");
			}

			await Validate(arr);

			// The midpoint triangle splits the face into 4 regions ≥ 4 sub-tris.
			await Assert.That(arr.Tris.Count).IsGreaterThanOrEqualTo(4);
		}

		[Test]
		public async Task ConstraintEdgesReferenceValidPoints()
		{
			ArrangementInput input = new ArrangementInput
			{
				Points = new List<(R3, int)>(),
				Segments = new List<(R3, R3, int)>
				{
					(R3Of(1.0, 1.0, 0.0), R3Of(5.0, 1.0, 0.0), 0),
					(R3Of(2.0, 0.0, 0.0), R3Of(2.0, 4.0, 0.0), 1),
				},
			};
			Arrangement arr = ArrangementFunctions.Build(Tri, input, null)!;
			foreach ((int u, int w) in arr.Constraints.Keys)
			{
				await Assert.That(u < arr.Points3.Count && w < arr.Points3.Count).IsTrue();
				await Assert.That(arr.Points2[u]).IsNotEqualTo(arr.Points2[w]);
			}

			await Validate(arr);
		}

		/// <summary>
		/// The interval sweep that replaced the O(S²) all-pairs scans is only sound
		/// because it reproduces that scan's output <em>pair for pair and in order</em> —
		/// downstream point indices and CDT insertion order depend on the enumeration
		/// order, so a permuted (even if equal) pair set would change the result mesh.
		/// This pins both the set and the order against the naive reference, across
		/// sizes on both sides of the sweep's small-input cutoff and over box layouts
		/// (clustered, spanning, duplicated, degenerate) that stress the active-list
		/// purge and the sweep-axis choice.
		/// </summary>
		[Test]
		public async Task BoxPairSweepMatchesTheNaiveScanExactly()
		{
			Lcg rng = new Lcg(0x5EED_1234UL);

			// (count, extent scale) — tiny extents make the sweep prune almost
			// everything; extents near 1.0 make nearly every pair overlap.
			(int N, double Scale)[] cases = new (int, double)[]
			{
				(3, 0.5),
				(64, 0.02),
				(65, 0.02),
				(200, 0.005),
				(200, 0.5),
				(150, 1.0),
				(120, 0.001),
			};
			foreach ((int n, double scale) in cases)
			{
				List<ArrangementFunctions.Box2> boxes = new List<ArrangementFunctions.Box2>(n);
				for (int k = 0; k < n; k++)
				{
					(double cx, double cy) = (rng.Next(), rng.Next());

					// Every 11th box is a zero-extent point, every 17th a duplicate of
					// its predecessor: exact ties in the sweep-axis sort order.
					(double ex, double ey) = k % 11 == 0
						? (0.0, 0.0)
						: (rng.Next() * scale, rng.Next() * scale);
					if (k % 17 == 0 && k > 0)
					{
						boxes.Add(boxes[k - 1]);
					}
					else
					{
						boxes.Add(new ArrangementFunctions.Box2(cx - ex, cy - ey, cx + ex, cy + ey));
					}
				}

				ArrangementFunctions.Box2[] boxArray = boxes.ToArray();
				List<(int, int)> want = new List<(int, int)>();
				for (int i = 0; i < n; i++)
				{
					for (int j = i + 1; j < n; j++)
					{
						if (ArrangementFunctions.BoxesOverlap(boxArray[i], boxArray[j]))
						{
							want.Add((i, j));
						}
					}
				}

				List<(int, int)> got = ArrangementFunctions.OverlappingBoxPairs(boxArray, null)!
					.Iter()
					.Select(p => (p.I, p.J))
					.ToList();
				await Assert.That(got).IsEquivalentTo(want, CollectionOrdering.Matching)
					.Because($"n={n} scale={scale}");
			}
		}

		/// <summary>A cancelled token must abort the sweep rather than return a partial set.</summary>
		[Test]
		public async Task BoxPairSweepHonoursCancellation()
		{
			ArrangementFunctions.Box2[] boxes = new ArrangementFunctions.Box2[500];
			for (int k = 0; k < 500; k++)
			{
				double x = k * 0.001;
				boxes[k] = new ArrangementFunctions.Box2(x, 0.0, x + 0.5, 1.0);
			}

			CancelToken token = new CancelToken();
			token.Cancel();
			await Assert.That(ArrangementFunctions.OverlappingBoxPairs(boxes, token)).IsNull();
		}

		/// <summary>Rust's `Vec<usize>` Ord: lexicographic, shorter prefix first.</summary>
		private static int LexicographicIntLists(List<int> a, List<int> b)
		{
			int n = Math.Min(a.Count, b.Count);
			for (int i = 0; i < n; i++)
			{
				int c = a[i].CompareTo(b[i]);
				if (c != 0)
				{
					return c;
				}
			}

			return a.Count.CompareTo(b.Count);
		}

		/// <summary>The Rust test's LCG, bit for bit (u64 wrapping arithmetic).</summary>
		private sealed class Lcg
		{
			private ulong state;

			public Lcg(ulong seed)
			{
				this.state = seed;
			}

			public double Next()
			{
				unchecked
				{
					this.state = (this.state * 6364136223846793005UL) + 1442695040888963407UL;
				}

				return (this.state >> 11) / (double)(1UL << 53);
			}
		}
	}
}
