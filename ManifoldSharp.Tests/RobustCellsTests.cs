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
// Same inputs, same expected values, same order as the Rust's eight tests.

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

		[Test]
		public async Task NonzeroRuleKeepsAnInvertedOperand()
		{
			List<Vec3[]> p = Cube(new Vec3(0.0, 0.0, 0.0), new Vec3(2.0, 2.0, 2.0));
			List<Vec3[]> q = Cube(new Vec3(5.0, 0.0, 0.0), new Vec3(6.0, 1.0, 1.0));
			List<Vec3[]> flipped = new List<Vec3[]>(q.Count);
			foreach (Vec3[] t in q)
			{
				flipped.Add(new[] { t[0], t[2], t[1] });
			}

			Manifold positive = BooleanViaCellsRule(p, flipped, OpType.Add, WindingRule.Positive);
			await Assert.That(positive.Status()).IsEqualTo(Error.NoError);
			await Assert.That(Math.Abs(positive.Volume() - 8.0)).IsLessThan(1e-9)
				.Because($"{positive.Volume()}");

			Manifold nonzero = BooleanViaCellsRule(p, flipped, OpType.Add, WindingRule.Nonzero);
			await Assert.That(nonzero.Status()).IsEqualTo(Error.NoError);
			await Assert.That(nonzero.AsImpl().IsSoup).IsFalse().Because("nonzero result must close");
			await Assert.That(Math.Abs(nonzero.Volume() - 9.0)).IsLessThan(1e-9)
				.Because($"{nonzero.Volume()}");
		}

		/// <summary>
		/// Two 2³ cubes overlapping in a 1³ corner: union 15, intersection 1, difference 7.
		/// Each result must also be a closed manifold — that is the property derived
		/// orientation is supposed to guarantee.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ExtractedBooleansHaveCorrectVolume()
		{
			List<Vec3[]> p = Cube(new Vec3(0.0, 0.0, 0.0), new Vec3(2.0, 2.0, 2.0));
			List<Vec3[]> q = Cube(new Vec3(1.0, 1.0, 1.0), new Vec3(3.0, 3.0, 3.0));

			foreach ((OpType Op, double Want) c in new[]
			{
				(OpType.Add, 15.0),
				(OpType.Intersect, 1.0),
				(OpType.Subtract, 7.0),
			})
			{
				Manifold m = BooleanViaCells(p, q, c.Op);
				await Assert.That(m.Status()).IsEqualTo(Error.NoError).Because($"{c.Op} status");
				await Assert.That(m.AsImpl().IsSoup).IsFalse()
					.Because($"{c.Op} must close into a manifold");
				await Assert.That(Math.Abs(m.Volume() - c.Want)).IsLessThan(1e-9)
					.Because($"{c.Op} volume {m.Volume()}, want {c.Want}");
			}
		}

		/// <summary>
		/// Inverting one operand's winding must not change the result: the cell labels decide
		/// orientation, so a reversed input shell still yields the same solid. This is the
		/// property that fixes the inverted-orientation class of NotClosed failures.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task InvertedOperandYieldsTheSameUnion()
		{
			List<Vec3[]> p = Cube(new Vec3(0.0, 0.0, 0.0), new Vec3(2.0, 2.0, 2.0));
			List<Vec3[]> q = Cube(new Vec3(1.0, 1.0, 1.0), new Vec3(3.0, 3.0, 3.0));
			List<Vec3[]> flipped = new List<Vec3[]>(q.Count);
			foreach (Vec3[] t in q)
			{
				flipped.Add(new[] { t[0], t[2], t[1] });
			}

			Manifold m = BooleanViaCells(p, flipped, OpType.Add);
			await Assert.That(m.Status()).IsEqualTo(Error.NoError)
				.Because("inverted-operand union status");

			// The reversed shell bounds {w <= -1}, so it contributes no material;
			// the union is P alone, and critically the result still closes.
			await Assert.That(m.AsImpl().IsSoup).IsFalse().Because("result must still be a manifold");
			await Assert.That(Math.Abs(m.Volume() - 8.0)).IsLessThan(1e-9)
				.Because($"volume {m.Volume()}");
		}

		/// <summary>
		/// End-to-end through the public API: one operand carries a correctly wound cube and
		/// an inside-out cube in the same soup — the shape of Thingi10K #51360 — and B is a
		/// bar crossing both.
		/// </summary>
		/// <remarks>
		/// Positive rule: A's solid is the wound cube alone (8), B adds 4 and overlaps it in
		/// 1 → 11. Nonzero rule: the inside-out cube is material too (16), B overlaps each of
		/// them in 1 → 18. Both are exact, so this pins the rule's effect rather than just
		/// its direction.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task NonzeroRuleKeepsInsideOutGeometryThroughThePublicApi()
		{
			List<Vec3[]> soup = Cube(new Vec3(0.0, 0.0, 0.0), new Vec3(2.0, 2.0, 2.0));
			List<Vec3[]> inverted = Cube(new Vec3(4.0, 0.0, 0.0), new Vec3(6.0, 2.0, 2.0));
			foreach (Vec3[] t in inverted)
			{
				soup.Add(new[] { t[0], t[2], t[1] });
			}

			Manifold a = MeshFromTris(soup);

			// A bar from x=1 to x=5 through both cubes' interiors.
			Manifold b = MeshFromTris(Cube(new Vec3(1.0, 0.5, 0.5), new Vec3(5.0, 1.5, 1.5)));

			foreach ((WindingRule Rule, double Want) c in new[]
			{
				(WindingRule.Positive, 11.0),
				(WindingRule.Nonzero, 18.0),
			})
			{
				Manifold outManifold = a.BooleanWithEngineAndRule(
					b, OpType.Add, BooleanEngine.Robust, c.Rule);
				await Assert.That(outManifold.Status()).IsEqualTo(Error.NoError)
					.Because($"{c.Rule} status");
				await Assert.That(Math.Abs(outManifold.Volume() - c.Want)).IsLessThan(1e-9)
					.Because($"{c.Rule} volume {outManifold.Volume()}, want {c.Want}");
			}
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

		/// <summary>Run one operation end to end through the cell complex and assemble it.</summary>
		/// <param name="p">The first operand's triangles.</param>
		/// <param name="q">The second operand's triangles.</param>
		/// <param name="op">The boolean operation.</param>
		/// <returns>The assembled result.</returns>
		private static Manifold BooleanViaCells(List<Vec3[]> p, List<Vec3[]> q, OpType op)
		{
			return BooleanViaCellsRule(p, q, op, WindingRule.Positive);
		}

		/// <summary><see cref="BooleanViaCells"/> with an explicit winding rule.</summary>
		/// <param name="p">The first operand's triangles.</param>
		/// <param name="q">The second operand's triangles.</param>
		/// <param name="op">The boolean operation.</param>
		/// <param name="rule">The winding rule.</param>
		/// <returns>The assembled result.</returns>
		private static Manifold BooleanViaCellsRule(
			List<Vec3[]> p,
			List<Vec3[]> q,
			OpType op,
			WindingRule rule)
		{
			IntersectionGraph graph = IntersectionGraphFunctions.BuildGraph(p.ToArray(), q.ToArray());
			CellComplex complex = Cells.BuildCells(graph);
			Windings wind = AllWindings(graph, complex, p, q);
			List<Piece> pieces = Cells.Extract(graph, complex, wind, op, rule);
			return AssembleFunctions.Assemble(pieces, graph.Verts, graph.VertsF64, _ => true, null);
		}

		/// <summary>Build a Manifold from a raw triangle soup, the way the demo imports STL.</summary>
		/// <param name="tris">The triangles.</param>
		/// <returns>The imported manifold.</returns>
		private static Manifold MeshFromTris(IReadOnlyList<Vec3[]> tris)
		{
			MeshGL mesh = new MeshGL();
			mesh.NumProp = 3;
			foreach (Vec3[] t in tris)
			{
				foreach (Vec3 p in t)
				{
					mesh.VertProperties.Add((float)p.X);
					mesh.VertProperties.Add((float)p.Y);
					mesh.VertProperties.Add((float)p.Z);
				}
			}

			for (uint i = 0; i < (uint)(tris.Count * 3); i++)
			{
				mesh.TriVerts.Add(i);
			}

			mesh.Merge();
			return Manifold.FromMeshGLRobust(mesh);
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
