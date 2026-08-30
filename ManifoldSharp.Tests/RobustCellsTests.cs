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

// RobustCellsTests.cs — port of robust/cells_tests.rs, whose header reads:
//
//   Cell complex and winding propagation tests.
//
//   The invariants under test are the ones that make the arrangement
//   formulation robust: the number of cells matches the geometry, every cell
//   reachable from the outside gets a winding, and the winding in each region
//   equals how many operands actually contain it.
//
// Same inputs, same expected values, same order as the Rust's tests.
//
// ── Four of the Rust's eight tests are not here yet ──────────────────────────
// `nonzero_rule_keeps_an_inverted_operand`, `extracted_booleans_have_correct_volume`,
// `inverted_operand_yields_the_same_union` and
// `nonzero_rule_keeps_inside_out_geometry_through_the_public_api` all measure a
// *volume*, which means they run the extracted pieces through
// `robust::assemble::assemble` (the last three via the robust engine's public
// entry point). `robust/assemble.rs` and the `robust/pairing.rs` it depends on
// are later steps of Phase 10; those four tests land with them, unchanged, and
// this file's header note goes with them. The four below are exactly the Rust's
// tests that stop at the cell complex, its windings and `in_result`, and they
// carry the Rust's expected values verbatim.

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using ManifoldSharp;
using ManifoldSharp.Linalg;
using ManifoldSharp.Robust;

namespace ManifoldSharp.Tests
{
	public class RobustCellsTests
	{
		/// <summary>An axis-aligned box as 12 outward-oriented triangles.</summary>
		/// <param name="lo">The low corner.</param>
		/// <param name="hi">The high corner.</param>
		/// <returns>The box's twelve triangles.</returns>
		public static List<Vec3[]> Cube(Vec3 lo, Vec3 hi)
		{
			Vec3[] v =
			{
				new Vec3(lo.X, lo.Y, lo.Z),
				new Vec3(hi.X, lo.Y, lo.Z),
				new Vec3(hi.X, hi.Y, lo.Z),
				new Vec3(lo.X, hi.Y, lo.Z),
				new Vec3(lo.X, lo.Y, hi.Z),
				new Vec3(hi.X, lo.Y, hi.Z),
				new Vec3(hi.X, hi.Y, hi.Z),
				new Vec3(lo.X, hi.Y, hi.Z),
			};
			int[][] idx =
			{
				new[] { 0, 3, 2 },
				new[] { 0, 2, 1 }, // −z
				new[] { 4, 5, 6 },
				new[] { 4, 6, 7 }, // +z
				new[] { 0, 1, 5 },
				new[] { 0, 5, 4 }, // −y
				new[] { 3, 7, 6 },
				new[] { 3, 6, 2 }, // +y
				new[] { 0, 4, 7 },
				new[] { 0, 7, 3 }, // −x
				new[] { 1, 2, 6 },
				new[] { 1, 6, 5 }, // +x
			};
			List<Vec3[]> outTris = new List<Vec3[]>(idx.Length);
			foreach (int[] t in idx)
			{
				outTris.Add(new[] { v[t[0]], v[t[1]], v[t[2]] });
			}

			return outTris;
		}

		[Test]
		public async Task DisjointCubesHaveThreeCells()
		{
			List<Vec3[]> p = Cube(new Vec3(0.0, 0.0, 0.0), new Vec3(1.0, 1.0, 1.0));
			List<Vec3[]> q = Cube(new Vec3(2.0, 0.0, 0.0), new Vec3(3.0, 1.0, 1.0));
			IntersectionGraph graph = IntersectionGraphFunctions.BuildGraph(p.ToArray(), q.ToArray());
			CellComplex complex = Cells.BuildCells(graph);

			// Each shell bounds its own inside/outside pair. The two outer cells are
			// one region in space, but nothing connects them combinatorially — that
			// is exactly what the per-component seeding resolves.
			await Assert.That(complex.NumCells).IsEqualTo(4)
				.Because("two disjoint shells, two cells each");

			Windings wind = AllWindings(graph, complex, p, q);
			await Assert.That(wind.Known.All(k => k)).IsTrue()
				.Because("seeding reaches every cell");
			await Assert.That(wind.W.Any(w => w == (0, 0))).IsTrue()
				.Because("the region outside both must be present");
			await Assert.That(InsideWinding(graph, complex, wind, 0))
				.IsEquivalentTo(new[] { (1, 0) }, CollectionOrdering.Matching);
			await Assert.That(InsideWinding(graph, complex, wind, 1))
				.IsEquivalentTo(new[] { (0, 1) }, CollectionOrdering.Matching);
		}

		[Test]
		public async Task OverlappingCubesWindToTwoInTheLens()
		{
			List<Vec3[]> p = Cube(new Vec3(0.0, 0.0, 0.0), new Vec3(2.0, 2.0, 2.0));
			List<Vec3[]> q = Cube(new Vec3(1.0, 1.0, 1.0), new Vec3(3.0, 3.0, 3.0));
			IntersectionGraph graph = IntersectionGraphFunctions.BuildGraph(p.ToArray(), q.ToArray());
			CellComplex complex = Cells.BuildCells(graph);

			Windings wind = AllWindings(graph, complex, p, q);

			// The three material regions: P only, Q only, and the shared lens.
			List<(int P, int Q)> regions = new List<(int P, int Q)>();
			for (int c = 0; c < complex.NumCells; c++)
			{
				if (wind.Known[c] && !regions.Contains(wind.W[c]))
				{
					regions.Add(wind.W[c]);
				}
			}

			regions.Sort();
			await Assert.That(regions)
				.IsEquivalentTo(
					new[] { (0, 0), (0, 1), (1, 0), (1, 1) },
					CollectionOrdering.Matching)
				.Because("outside, Q only, P only, and the overlap where both wind to one");
		}

		/// <summary>
		/// The containment predicate under both rules, spelled out per operation.
		/// </summary>
		/// <remarks>
		/// Expected truth is derived from the rule definitions rather than copied from the
		/// implementation: <c>Positive</c> means <c>w &gt;= 1</c>, <c>Nonzero</c> means
		/// <c>w != 0</c>, and each operation is the obvious boolean combination of the two
		/// operands' insideness.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task InResultHonorsTheWindingRule()
		{
			(int P, int Q)[] vectors = { (1, 0), (-1, 0), (2, 0), (0, -1), (-1, -1), (0, 0), (1, 1) };
			foreach ((int P, int Q) w in vectors)
			{
				foreach ((WindingRule Rule, Func<int, bool> Inside) pair in new[]
				{
					(WindingRule.Positive, (Func<int, bool>)(v => v >= 1)),
					(WindingRule.Nonzero, (Func<int, bool>)(v => v != 0)),
				})
				{
					bool a = pair.Inside(w.P);
					bool b = pair.Inside(w.Q);
					foreach ((OpType Op, bool Want) expect in new[]
					{
						(OpType.Add, a || b),
						(OpType.Intersect, a && b),
						(OpType.Subtract, a && !b),
					})
					{
						await Assert.That(Cells.InResult(expect.Op, pair.Rule, w))
							.IsEqualTo(expect.Want)
							.Because($"{expect.Op} {pair.Rule} on {w}");
					}
				}
			}

			// Spot-check the cases the rules actually disagree on, so a predicate
			// that ignored its rule argument could not pass this test.
			await Assert.That(Cells.InResult(OpType.Add, WindingRule.Positive, (-1, 0))).IsFalse();
			await Assert.That(Cells.InResult(OpType.Add, WindingRule.Nonzero, (-1, 0))).IsTrue();
			await Assert.That(Cells.InResult(OpType.Subtract, WindingRule.Nonzero, (1, -1))).IsFalse();
			await Assert.That(Cells.InResult(OpType.Subtract, WindingRule.Positive, (1, -1))).IsTrue();
			await Assert.That(Cells.InResult(OpType.Intersect, WindingRule.Nonzero, (-1, -1))).IsTrue();
			await Assert.That(Cells.InResult(OpType.Intersect, WindingRule.Positive, (-1, -1))).IsFalse();
		}

		/// <summary>
		/// A doubled shell must step the winding by two, not one — this is the multiplicity
		/// behaviour that lets self-overlapping scans classify correctly without an explicit
		/// regularization pass.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task DoubledShellStepsWindingByTwo()
		{
			List<Vec3[]> p = Cube(new Vec3(0.0, 0.0, 0.0), new Vec3(1.0, 1.0, 1.0));
			List<Vec3[]> dup = new List<Vec3[]>(p);
			p.AddRange(dup); // every facet present twice, same orientation
			List<Vec3[]> q = Cube(new Vec3(5.0, 0.0, 0.0), new Vec3(6.0, 1.0, 1.0));
			IntersectionGraph graph = IntersectionGraphFunctions.BuildGraph(p.ToArray(), q.ToArray());
			CellComplex complex = Cells.BuildCells(graph);

			Windings wind = AllWindings(graph, complex, p, q);
			await Assert.That(InsideWinding(graph, complex, wind, 0))
				.IsEquivalentTo(new[] { (2, 0) }, CollectionOrdering.Matching)
				.Because("a double cover winds to two inside");
		}

		/// <summary>Winding numbers for every cell, each component anchored by an exact query.</summary>
		/// <param name="graph">The intersection graph.</param>
		/// <param name="complex">Its cell complex.</param>
		/// <param name="p">The first operand's triangles.</param>
		/// <param name="q">The second operand's triangles.</param>
		/// <returns>The per-cell windings.</returns>
		private static Windings AllWindings(
			IntersectionGraph graph,
			CellComplex complex,
			IReadOnlyList<Vec3[]> p,
			IReadOnlyList<Vec3[]> q)
		{
			return Cells.Windings(graph, complex, new[] { p, q });
		}

		/// <summary>
		/// Winding of the region just inside <paramref name="mesh"/>'s surface, by looking at
		/// the anti side of any piece belonging to it.
		/// </summary>
		/// <param name="graph">The intersection graph.</param>
		/// <param name="complex">Its cell complex.</param>
		/// <param name="wind">The per-cell windings.</param>
		/// <param name="mesh">0 for P, 1 for Q.</param>
		/// <returns>The distinct winding vectors seen, in discovery order.</returns>
		private static List<(int P, int Q)> InsideWinding(
			IntersectionGraph graph,
			CellComplex complex,
			Windings wind,
			byte mesh)
		{
			List<(int P, int Q)> seen = new List<(int P, int Q)>();
			for (int pi = 0; pi < graph.Pieces.Count; pi++)
			{
				if (graph.Pieces[pi].Mesh != mesh)
				{
					continue;
				}

				int c = complex.Cell(pi, Cells.Anti);
				if (wind.Known[c] && !seen.Contains(wind.W[c]))
				{
					seen.Add(wind.W[c]);
				}
			}

			return seen;
		}
	}
}
