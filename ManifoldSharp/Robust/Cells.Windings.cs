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

// Cells.Windings.cs — the second half of robust/cells.rs: the walls, the cell
// adjacency they induce, and the winding propagation over it. The module header
// is on Cells.cs; this file exists only so both stay inside the 800-line cap.
//
// ── `[&[[Vec3; 3]]; 2]` is a two-element array of lists ──────────────────────
// The Rust passes both operands' triangle tables as one fixed-size array so the
// body can write `tris[m]` inside `for m in 0..2`. The port keeps that shape —
// `IReadOnlyList<Vec3[]>[]` with exactly two entries — rather than splitting
// each into a P and a Q parameter, because the `for m in 0..2` winding loop is
// the whole point of the array and splitting it would turn one loop into two
// hand-unrolled calls.

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Robust
{
	/// <summary>Per-cell winding numbers, one entry per operand.</summary>
	public sealed class Windings
	{
		/// <summary>
		/// <c>W[cell] = (w_P, w_Q)</c>, valid only where <see cref="Known"/> is set.
		/// </summary>
		public (int P, int Q)[] W = Array.Empty<(int, int)>();

		/// <summary>Which cells the traversal has resolved.</summary>
		public bool[] Known = Array.Empty<bool>();

		/// <summary>Every cell resolved — no residual point queries needed.</summary>
		/// <returns>True when every cell has a winding.</returns>
		public bool Complete()
		{
			foreach (bool k in this.Known)
			{
				if (!k)
				{
					return false;
				}
			}

			return true;
		}
	}

	/// <summary>
	/// One distinct triangle of the arrangement, with the coincident stack that occupies it
	/// collapsed into a single winding step.
	/// </summary>
	public readonly struct Wall
	{
		/// <summary>Representative piece; its winding fixes the wall's normal side.</summary>
		public readonly int Rep;

		/// <summary>
		/// Winding change per operand, crossing the representative's normal side to its anti
		/// side.
		/// </summary>
		public readonly (int P, int Q) Delta;

		/// <summary>Creates a wall.</summary>
		/// <param name="rep">The representative piece index.</param>
		/// <param name="delta">The per-operand winding step.</param>
		public Wall(int rep, (int P, int Q) delta)
		{
			this.Rep = rep;
			this.Delta = delta;
		}
	}

	public static partial class Cells
	{
		/// <summary>
		/// The two sides, in the order the Rust's <c>for side in [NORMAL, ANTI]</c> visits
		/// them. A <see cref="ReadOnlySpan{T}"/> property over an array initializer rather
		/// than a static readonly array, which would still be writable through its elements —
		/// the same call DeterministicMath.cs's kernel-coefficient tables make, and lowered
		/// the same way, to an immutable metadata blob with no allocation.
		/// </summary>
		private static ReadOnlySpan<int> Sides => [Normal, Anti];

		/// <summary>Winding numbers for every cell of the arrangement.</summary>
		/// <remarks>
		/// The combinatorial BFS only reaches cells connected through shared arrangement
		/// edges, so disjoint or nested components need a seed each: exactly the residual
		/// ray-shooting libigl's <c>propagate_winding_numbers</c> performs. One exact query
		/// pair per component, not per surface region — so the expensive part scales with the
		/// number of components rather than the number of regions.
		/// <para>
		/// The anchor is deliberately measured rather than deduced. Identifying the unbounded
		/// cell combinatorially — take the lexicographically extreme vertex, pick the incident
		/// face most nearly perpendicular to x, call its outward side unbounded — is wrong
		/// whenever that face's outward side holds material, which happens on real scans (a
		/// thin shell whose rim reaches the extreme vertex). The failure is silent and total:
		/// anchoring an interior cell at zero shifts every winding by a constant, and
		/// <c>w ≥ 1</c> then excludes almost the whole model. Two Thingi10K unions collapsed
		/// from 51372 and 5856 triangles to 4 and 40 that way.
		/// </para>
		/// </remarks>
		/// <param name="graph">The intersection graph.</param>
		/// <param name="complex">Its cell complex.</param>
		/// <param name="tris">Both operands' original triangles, P then Q.</param>
		/// <returns>The per-cell winding numbers.</returns>
		/// <exception cref="ArgumentException"><paramref name="tris"/> is not two operands.</exception>
		public static Windings Windings(
			IntersectionGraph graph,
			CellComplex complex,
			IReadOnlyList<Vec3[]>[] tris)
		{
			ArgumentNullException.ThrowIfNull(complex);
			RequireTwoOperands(tris, nameof(tris));
			IReadOnlyList<R3[]>[] rat = { ToRational(tris[0]), ToRational(tris[1]) };
			IReadOnlyList<Box>[] bx = { TriBoxes(tris[0]), TriBoxes(tris[1]) };
			Windings result = new Windings
			{
				W = new (int, int)[complex.NumCells],
				Known = new bool[complex.NumCells],
			};
			SeedUnreached(graph, complex, result, tris, rat, bx);
			return result;
		}

		/// <summary>
		/// Resolve whatever the outer traversal could not reach — disjoint or nested
		/// components — with one exact query pair each.
		/// </summary>
		/// <remarks>
		/// Returns immediately when everything is already known, so callers can hold off
		/// building the rational and bounding-box tables until this says it needs them.
		/// </remarks>
		/// <param name="graph">The intersection graph.</param>
		/// <param name="complex">Its cell complex.</param>
		/// <param name="result">The windings to fill in (Rust's <c>&amp;mut out</c>).</param>
		/// <param name="trisF64">Both operands' triangles, P then Q.</param>
		/// <param name="trisR">The same triangles exactly, P then Q.</param>
		/// <param name="boxes">One bounding box per triangle, P then Q.</param>
		/// <exception cref="ArgumentException">Some argument is not two operands.</exception>
		public static void SeedUnreached(
			IntersectionGraph graph,
			CellComplex complex,
			Windings result,
			IReadOnlyList<Vec3[]>[] trisF64,
			IReadOnlyList<R3[]>[] trisR,
			IReadOnlyList<Box>[] boxes)
		{
			ArgumentNullException.ThrowIfNull(graph);
			ArgumentNullException.ThrowIfNull(complex);
			ArgumentNullException.ThrowIfNull(result);
			RequireTwoOperands(trisF64, nameof(trisF64));
			RequireTwoOperands(trisR, nameof(trisR));
			RequireTwoOperands(boxes, nameof(boxes));
			if (result.Complete())
			{
				return;
			}

			List<(int Cell, (int P, int Q) Delta)>[] adj = CellAdjacency(complex);

			// A representative (piece, side) per cell, for seeding by point query.
			(int Piece, int Side)?[] rep = new (int, int)?[complex.NumCells];
			for (int pi = 0; pi < graph.Pieces.Count; pi++)
			{
				foreach (int side in Sides)
				{
					int c = complex.Cell(pi, side);
					if (rep[c] is null)
					{
						rep[c] = (pi, side);
					}
				}
			}

			for (int c = 0; c < complex.NumCells; c++)
			{
				if (result.Known[c])
				{
					continue;
				}

				if (rep[c] is null)
				{
					continue; // Rust's `let Some((pi, side)) = rep[c] else { continue }`
				}

				(int pi, int side) = rep[c]!.Value;
				R3[] pv = graph.PieceVerts(pi);
				R3 point = RayShoot.PieceCentroid(pv);
				R3 n = pv[1].Sub(pv[0]).Cross(pv[2].Sub(pv[0]));
				R3 outward = side == Normal ? n : new R3(-n.X, -n.Y, -n.Z);
				int[] w = new int[2];
				for (int m = 0; m < 2; m++)
				{
					w[m] = RayShoot.WindingOffSurface(point, outward, trisR[m], trisF64[m], boxes[m]);
				}

				Seed(result, c, (w[0], w[1]));
				Bfs(adj, result, c);
			}
		}

		/// <summary>
		/// Walls whose winding step disagrees with the cells' resolved windings.
		/// </summary>
		/// <remarks>
		/// The difference between two cells is well defined, so a disagreement means the
		/// complex merged cells that the geometry keeps apart. Used by tests and diagnostics;
		/// a clean arrangement returns an empty list.
		/// </remarks>
		/// <param name="complex">The cell complex.</param>
		/// <param name="wind">Its resolved windings.</param>
		/// <returns>One entry per disagreeing wall: representative, claimed step, actual step.</returns>
		public static List<(int Rep, (int P, int Q) Delta, (int P, int Q) Actual)> InconsistentWalls(
			CellComplex complex,
			Windings wind)
		{
			ArgumentNullException.ThrowIfNull(complex);
			ArgumentNullException.ThrowIfNull(wind);
			List<(int Rep, (int P, int Q) Delta, (int P, int Q) Actual)> bad
				= new List<(int, (int, int), (int, int))>();
			foreach (Wall wall in complex.Walls)
			{
				int cn = complex.Cell(wall.Rep, Normal);
				int ca = complex.Cell(wall.Rep, Anti);
				if (cn == ca || !wind.Known[cn] || !wind.Known[ca])
				{
					continue;
				}

				(int P, int Q) actual = (wind.W[ca].P - wind.W[cn].P, wind.W[ca].Q - wind.W[cn].Q);
				if (actual != wall.Delta)
				{
					bad.Add((wall.Rep, wall.Delta, actual));
				}
			}

			return bad;
		}

		/// <summary>Rust's <c>to_rational</c>: both operands' triangles as exact points.</summary>
		/// <param name="tris">One operand's triangles.</param>
		/// <returns>The same triangles exactly.</returns>
		private static R3[][] ToRational(IReadOnlyList<Vec3[]> tris)
		{
			R3[][] outTris = new R3[tris.Count][];
			for (int i = 0; i < tris.Count; i++)
			{
				Vec3[] t = tris[i];
				outTris[i] = new R3[] { R3.FromVec3(t[0]), R3.FromVec3(t[1]), R3.FromVec3(t[2]) };
			}

			return outTris;
		}

		/// <summary>Rust's <c>tri_boxes</c>: one bounding box per triangle.</summary>
		/// <param name="tris">One operand's triangles.</param>
		/// <returns>Their bounding boxes.</returns>
		private static Box[] TriBoxes(IReadOnlyList<Vec3[]> tris)
		{
			Box[] outBoxes = new Box[tris.Count];
			for (int i = 0; i < tris.Count; i++)
			{
				Vec3[] t = tris[i];
				Box b = Box.FromPoints(t[0], t[1]);
				b.UnionPoint(t[2]);
				outBoxes[i] = b;
			}

			return outBoxes;
		}

		/// <summary>Record one cell's winding.</summary>
		/// <param name="result">The windings being filled in.</param>
		/// <param name="cell">The cell id.</param>
		/// <param name="w">Its winding vector.</param>
		private static void Seed(Windings result, int cell, (int P, int Q) w)
		{
			result.W[cell] = w;
			result.Known[cell] = true;
		}

		/// <summary>
		/// Winding step between adjacent cells, <b>summed</b> over every piece that separates
		/// them.
		/// </summary>
		/// <remarks>
		/// Aggregation is what gives coincident stacks their multiplicity: a doubled sheet
		/// contributes +2 and a fold cancels to 0, so self-overlapping input classifies
		/// correctly with no separate regularization pass. Treating the coincident pieces as
		/// independent adjacencies would apply only the first and silently lose the rest.
		/// </remarks>
		/// <param name="complex">The cell complex.</param>
		/// <returns>Per cell, its neighbours and the step to each.</returns>
		private static List<(int Cell, (int P, int Q) Delta)>[] CellAdjacency(CellComplex complex)
		{
			List<(int Cell, (int P, int Q) Delta)>[] adj
				= new List<(int Cell, (int P, int Q) Delta)>[complex.NumCells];
			for (int i = 0; i < adj.Length; i++)
			{
				adj[i] = new List<(int Cell, (int P, int Q) Delta)>();
			}

			foreach (Wall wall in complex.Walls)
			{
				int cn = complex.Cell(wall.Rep, Normal);
				int ca = complex.Cell(wall.Rep, Anti);
				if (cn == ca)
				{
					continue; // both sides in one cell: the sheet bounds nothing
				}

				adj[cn].Add((ca, wall.Delta));
				adj[ca].Add((cn, (-wall.Delta.P, -wall.Delta.Q)));
			}

			// Every wall stays its own edge — collapsing them by cell pair would let
			// an arbitrary one win, which is both order-dependent and lossy. Sorting
			// makes the traversal deterministic regardless of hash iteration order.
			//
			// Rust `sort_unstable` then `dedup`: equal elements are equal in all three
			// fields, so no ordering of the ties is observable, and the dedup that
			// follows removes them anyway.
			foreach (List<(int Cell, (int P, int Q) Delta)> a in adj)
			{
				a.Sort(CompareAdjacency);
				Dedup(a);
			}

			return adj;
		}

		/// <summary>The lexicographic order of Rust's <c>(usize, [i32; 2])</c>.</summary>
		/// <param name="x">The left edge.</param>
		/// <param name="y">The right edge.</param>
		/// <returns>Negative, zero or positive.</returns>
		private static int CompareAdjacency(
			(int Cell, (int P, int Q) Delta) x,
			(int Cell, (int P, int Q) Delta) y)
		{
			int c = x.Cell.CompareTo(y.Cell);
			if (c != 0)
			{
				return c;
			}

			c = x.Delta.P.CompareTo(y.Delta.P);
			if (c != 0)
			{
				return c;
			}

			return x.Delta.Q.CompareTo(y.Delta.Q);
		}

		/// <summary>Rust's <c>Vec::dedup</c>: drop runs of consecutive equal elements.</summary>
		/// <param name="list">The sorted list, edited in place.</param>
		private static void Dedup(List<(int Cell, (int P, int Q) Delta)> list)
		{
			int write = 0;
			for (int read = 0; read < list.Count; read++)
			{
				if (write == 0 || !list[read].Equals(list[write - 1]))
				{
					list[write] = list[read];
					write++;
				}
			}

			list.RemoveRange(write, list.Count - write);
		}

		/// <summary>
		/// Group pieces into walls by exact triangle identity.
		/// </summary>
		/// <remarks>
		/// Pieces occupying the same triangle are a coincident stack whose contributions add —
		/// a doubled sheet steps by two, a fold cancels to zero. Two *different* triangles
		/// between the same pair of cells are alternative crossings of one boundary and are
		/// never summed; that distinction is why aggregation keys on the triangle rather than
		/// the cell pair.
		/// <para>
		/// Order invariance: <c>byTri</c> is iterated, but its output is sorted by the unique
		/// representative piece index right after, so hash order cannot reach the result.
		/// </para>
		/// </remarks>
		/// <param name="graph">The intersection graph.</param>
		/// <returns>The walls, in representative-piece order.</returns>
		private static List<Wall> Walls(IntersectionGraph graph)
		{
			Dictionary<(uint A, uint B, uint C), WallAccum> byTri
				= new Dictionary<(uint, uint, uint), WallAccum>();
			for (int pi = 0; pi < graph.Pieces.Count; pi++)
			{
				Piece piece = graph.Pieces[pi];
				((uint A, uint B, uint C) key, bool parity) = Canonical(piece.Vi);
				int m = piece.Mesh;
				if (!byTri.TryGetValue(key, out WallAccum? entry))
				{
					entry = new WallAccum(pi, parity);
					byTri.Add(key, entry);
				}

				// Opposite winding means this piece's normal side is the
				// representative's anti side, so it steps the other way.
				int step = parity == entry.Parity ? 1 : -1;
				if (m == 0)
				{
					entry.DeltaP += step;
				}
				else if (m == 1)
				{
					entry.DeltaQ += step;
				}
				else
				{
					// Rust writes `entry.2[m] += …` into a `[i32; 2]`, so a mesh id
					// outside {0, 1} aborts on the bounds check. `Piece`'s constructor
					// is public and takes a bare byte, so the same malformed piece is
					// constructible here — the third arm exists to fail just as loudly
					// instead of silently folding a third operand into Q's winding.
					throw new InvalidOperationException(
						$"piece {pi} has mesh id {m}; a boolean has exactly two operands");
				}
			}

			List<Wall> outWalls = new List<Wall>(byTri.Count);
			foreach (WallAccum e in byTri.Values)
			{
				outWalls.Add(new Wall(e.Rep, (e.DeltaP, e.DeltaQ)));
			}

			// Hash iteration order must not reach the output: sort so extraction
			// emits triangles in a stable order across runs. Rust `sort_unstable_by_key`,
			// and unstable is safe because one wall owns each representative index, so
			// the key is unique and no ties exist.
			outWalls.Sort((x, y) => x.Rep.CompareTo(y.Rep));
			return outWalls;
		}

		/// <summary>
		/// Canonical key for a triangle (sorted vertex ids) plus the parity of the piece's
		/// winding against that order — the same identity the coincident binding uses, so both
		/// agree on what "the same triangle" means.
		/// </summary>
		/// <param name="vi">The piece's three interned vertex ids, in its own winding.</param>
		/// <returns>The sorted ids and whether the piece's rotation matches them.</returns>
		private static ((uint A, uint B, uint C) Key, bool Parity) Canonical(UVec3 vi)
		{
			uint[] sorted = { vi[0], vi[1], vi[2] };
			Array.Sort(sorted);

			// Rust's `min_by_key`, which returns the *first* of several equal minima.
			int i = 0;
			if (vi[1] < vi[i])
			{
				i = 1;
			}

			if (vi[2] < vi[i])
			{
				i = 2;
			}

			uint[] rotated = { vi[i], vi[(i + 1) % 3], vi[(i + 2) % 3] };
			bool parity = rotated[0] == sorted[0] && rotated[1] == sorted[1] && rotated[2] == sorted[2];
			return ((sorted[0], sorted[1], sorted[2]), parity);
		}

		/// <summary>Breadth-first winding propagation out of one seeded cell.</summary>
		/// <param name="adj">The cell adjacency.</param>
		/// <param name="result">The windings being filled in.</param>
		/// <param name="start">The seeded cell.</param>
		private static void Bfs(
			List<(int Cell, (int P, int Q) Delta)>[] adj,
			Windings result,
			int start)
		{
			Queue<int> queue = new Queue<int>();
			queue.Enqueue(start);
			while (queue.Count > 0)
			{
				int c = queue.Dequeue();
				(int P, int Q) baseW = result.W[c];
				foreach ((int next, (int P, int Q) d) in adj[c])
				{
					if (result.Known[next])
					{
						continue;
					}

					Seed(result, next, (baseW.P + d.P, baseW.Q + d.Q));
					queue.Enqueue(next);
				}
			}
		}

		/// <summary>The length check that stands in for Rust's <c>[…; 2]</c> parameter type.</summary>
		/// <typeparam name="T">The per-operand table type.</typeparam>
		/// <param name="operands">The two-element array.</param>
		/// <param name="name">The parameter name to blame.</param>
		/// <exception cref="ArgumentException">The array is not exactly two operands.</exception>
		private static void RequireTwoOperands<T>(T[] operands, string name)
		{
			ArgumentNullException.ThrowIfNull(operands, name);
			if (operands.Length != 2)
			{
				throw new ArgumentException("a boolean has exactly two operands", name);
			}
		}

		/// <summary>
		/// Rust's <c>by_tri</c> value <c>(usize, bool, [i32; 2])</c>. A mutable class rather
		/// than a value tuple because the accumulation edits the entry in place, keyed on a
		/// runtime mesh index.
		/// </summary>
		private sealed class WallAccum
		{
			/// <summary>The first piece seen on this triangle.</summary>
			public readonly int Rep;

			/// <summary>That piece's winding parity against the sorted key.</summary>
			public readonly bool Parity;

			/// <summary>Accumulated winding step for operand P.</summary>
			public int DeltaP;

			/// <summary>Accumulated winding step for operand Q.</summary>
			public int DeltaQ;

			public WallAccum(int rep, bool parity)
			{
				this.Rep = rep;
				this.Parity = parity;
			}
		}
	}
}
