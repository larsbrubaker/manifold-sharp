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

// Cells.cs — port of robust/cells.rs, whose header reads:
//
//   Arrangement cell complex and combinatorial winding propagation (Zhou,
//   Grinspun, Zorin, Jacobson 2016, "Mesh Arrangements for Solid Geometry" —
//   the formulation libigl's mesh_boolean uses).
//
//   This replaces the per-component winding queries of robust/mod.rs with the
//   structure that makes local inconsistency unrepresentable. The pieces of
//   robust/intersection_graph.rs already form an intersection-free arrangement
//   of both operands; what this module adds is the *dual*:
//
//     1. Every piece has two sides (normal / anti). Around each arrangement
//        edge the incident half-faces are radially sorted (the same exact
//        basis + angle_cmp the ring regularization uses), and the wedge
//        between consecutive radial positions unions the sides it touches.
//        The resulting equivalence classes are the 3D cells of the
//        arrangement.
//     2. Winding numbers are then propagated cell-to-cell by breadth-first
//        search: crossing a piece from its normal side to its anti side enters
//        the solid its operand bounds, so w[mesh] increases by one. Because
//        every cell's winding is *derived from one traversal*, adjacent
//        regions cannot disagree — the failure mode of independent per-region
//        queries (two pieces meeting along a segment both classified "outside",
//        leaving a surface that cannot close) does not exist here.
//
//   Coincident duplicate faces fall out correctly: they share a radial
//   position, so they form one "wall" whose winding step is the signed sum of
//   the stack. A doubled sheet steps the winding by two and a fold cancels to
//   zero, without any explicit regularization pass.
//
// ── The module is one partial class across four files ────────────────────────
// The Rust splits the same module in two — `cells.rs` and `cells_extract.rs` —
// purely for its file-length limit, and re-exports the second half so callers
// still say `cells::extract` / `cells::in_result`. This port keeps that shape
// and applies the same rule one cut deeper, because C# spends more lines per
// item than Rust does: `Cells` is one `public static partial class` spelled
// across Cells.cs (the complex and its construction), Cells.Fan.cs (the radial
// sort and its filtered predicates), Cells.Windings.cs (walls, adjacency and
// the winding propagation) and CellsExtract.cs (the containment predicate and
// the boundary walk). One API, four files, same reason.
//
// ── The Rust's private `struct Approx` keeps its name ────────────────────────
// It collides with the exact tier's `Exact.Approx` module, which the fan also
// calls. The error-tracked value stays `Approx` (nested in `Cells`, so it wins
// the simple name inside the class) and the exact tier's filtered predicate is
// spelled `Exact.Approx.Orient3dA` at its two call sites. Renaming the Rust's
// type would have cost more in diffability than the qualification costs here.
//
// ── `[f64; 3]` is a Vec3 here ────────────────────────────────────────────────
// Same call RayShoot.cs and GraphGeom.cs made: the exact tier settled on Vec3
// for the Rust's bare `[f64; 3]` approximations, so the Rust's `pt` closures
// (`let p = vt.verts_f64[v]; [p.x, p.y, p.z]`) collapse to a plain table read.
//
// ── `[i32; 2]` is an `(int P, int Q)` value tuple ────────────────────────────
// Same call GraphTypes.cs made for `[u64; 3]`: a fixed-size array of scalars
// becomes a value tuple, never an `int[]`, whose equality and ordering are by
// reference. The tuple's lexicographic `CompareTo` is Rust's array `Ord`, which
// `cell_adjacency`'s sort-then-dedup depends on. The one site that indexes it
// with a runtime operand (`entry.2[m] += …`, keyed on `piece.mesh`) becomes an
// explicit two-way branch inside `Walls`, with a third arm that throws where
// Rust's own bounds check would abort.
//
// ── `Cells.Windings` the method and `Windings` the type ──────────────────────
// The Rust has `fn windings` beside `struct Windings`, and this port keeps both
// spellings rather than inventing a `ComputeWindings` the Rust has no name for.
// That is legal C# — a *type* context (`Windings result = …`, the method's own
// return type) only ever considers types, so the method never hides the class.
// The hazard to know about before adding to these partials: in a *member*
// context the binding flips, so a bare `Windings` on the right-hand side of an
// expression resolves to the method group, not the type. Any future partial of
// this class that wants the type in an expression position (`typeof`, a
// pattern, `nameof`) should say so explicitly rather than assume the simple
// name still means the class.

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

// Rust's `type Half = (EdgeKey, usize, bool, u32)` from pairing.rs, which names
// the tuple `radial_fan` consumes. C# aliases are file-scoped, so downstream
// files spell the tuple out; the alias is written here exactly once.
using Half = ((uint A, uint B) Key, int Id, bool Forward, uint Apex);

namespace ManifoldSharp.Robust
{
	/// <summary>The arrangement's cell decomposition.</summary>
	public sealed class CellComplex
	{
		/// <summary>Compact cell id per (piece, side); index with <c>Cells.Node</c>.</summary>
		public uint[] CellOf = Array.Empty<uint>();

		/// <summary>The number of distinct cells.</summary>
		public int NumCells;

		/// <summary>
		/// Distinct triangles of the arrangement, each with its coincident stack
		/// collapsed into one winding step. Computed once here because both the
		/// winding propagation and the extraction need it.
		/// </summary>
		public List<Wall> Walls = new List<Wall>();

		/// <summary>The cell on one side of one piece.</summary>
		/// <param name="piece">Index into <see cref="IntersectionGraph.Pieces"/>.</param>
		/// <param name="side">, <see cref="Cells.Normal"/> or <see cref="Cells.Anti"/>.</param>
		/// <returns>The compact cell id.</returns>
		public int Cell(int piece, int side)
		{
			return (int)this.CellOf[(2 * piece) + side];
		}
	}

	/// <summary>
	/// The vertex tables the radial machinery reads: exact coordinates plus their cached
	/// correctly rounded f64 approximations. Passing them explicitly (rather than the whole
	/// <see cref="IntersectionGraph"/>) lets the output assembly reuse the fan sort on the
	/// extracted boundary, which has the same interned vertex ids but no graph of its own.
	/// </summary>
	/// <remarks>A readonly struct because the Rust derives <c>Clone, Copy</c>.</remarks>
	public readonly struct VertTables
	{
		/// <summary>Exact coordinates, indexed by interned vertex id.</summary>
		public readonly IReadOnlyList<R3> Verts;

		/// <summary>Their correctly rounded f64 approximations, same indices.</summary>
		public readonly IReadOnlyList<Vec3> VertsF64;

		/// <summary>Creates the pair of tables (Rust's struct literal).</summary>
		/// <param name="verts">Exact coordinates.</param>
		/// <param name="vertsF64">Their rounded approximations.</param>
		public VertTables(IReadOnlyList<R3> verts, IReadOnlyList<Vec3> vertsF64)
		{
			this.Verts = verts;
			this.VertsF64 = vertsF64;
		}

		/// <summary>The tables a built graph carries.</summary>
		/// <param name="graph">The intersection graph.</param>
		/// <returns>Its vertex tables.</returns>
		public static VertTables Of(IntersectionGraph graph)
		{
			ArgumentNullException.ThrowIfNull(graph);
			return new VertTables(graph.Verts, graph.VertsF64);
		}
	}

	/// <summary>One half-face incident to an arrangement edge.</summary>
	/// <remarks>A readonly struct: the fan holds these in a list and copies them out.</remarks>
	public readonly struct Inc
	{
		/// <summary>
		/// Caller-defined id of the half-face. <see cref="Cells.BuildCells"/> passes the piece
		/// index; the output pairing (<c>robust::pairing</c>) passes the half-edge index — the
		/// fan sort itself never interprets it.
		/// </summary>
		public readonly int Id;

		/// <summary>Traversal runs key.0 → key.1 (the edge's canonical direction).</summary>
		public readonly bool Forward;

		/// <summary>Vertex id of the opposite (apex) vertex.</summary>
		public readonly uint Apex;

		/// <summary>Creates a half-face record.</summary>
		/// <param name="id">Caller-defined half-face id.</param>
		/// <param name="forward">True when the face traverses the edge's canonical direction.</param>
		/// <param name="apex">The opposite vertex's id.</param>
		public Inc(int id, bool forward, uint apex)
		{
			this.Id = id;
			this.Forward = forward;
			this.Apex = apex;
		}

		/// <summary>The side of this half-face facing counter-clockwise about its edge.</summary>
		internal int CcwSide()
		{
			return Cells.CcwSide(this.Forward);
		}

		/// <summary>The side of this half-face facing clockwise about its edge.</summary>
		internal int CwSide()
		{
			return Cells.CwSide(this.Forward);
		}
	}

	/// <summary>
	/// The free functions of <c>robust/cells.rs</c> and <c>robust/cells_extract.rs</c>: the
	/// arrangement's cell complex, its combinatorial winding propagation, and the boundary
	/// extraction that turns cell labels back into output pieces.
	/// </summary>
	public static partial class Cells
	{
		/// <summary>
		/// Side of a piece: the half-space its outward normal points into.
		/// </summary>
		public const int Normal = 0;

		/// <summary>
		/// Side of a piece: the one behind it (inside the solid its operand bounds).
		/// </summary>
		public const int Anti = 1;

		/// <summary>Node id in the side union-find: two per piece.</summary>
		/// <param name="piece">The piece index.</param>
		/// <param name="side">, <see cref="Normal"/> or <see cref="Anti"/>.</param>
		/// <returns>The union-find node id.</returns>
		private static uint Node(int piece, int side)
		{
			return (uint)((2 * piece) + side);
		}

		/// <summary>
		/// The side of a half-face that faces counter-clockwise (increasing radial angle)
		/// around its edge.
		/// </summary>
		/// <remarks>
		/// A forward-traversing face has normal ∝ w × d, which sits 90° CCW of its apex
		/// direction, so its normal side is the CCW one; a backward face's normal is 90° CW,
		/// so the relationship inverts.
		/// </remarks>
		/// <param name="forward">True when the face traverses the edge's canonical direction.</param>
		/// <returns>, <see cref="Normal"/> or <see cref="Anti"/>.</returns>
		internal static int CcwSide(bool forward)
		{
			return forward ? Normal : Anti;
		}

		/// <summary>The other side from <see cref="CcwSide(bool)"/>.</summary>
		/// <param name="forward">True when the face traverses the edge's canonical direction.</param>
		/// <returns>, <see cref="Normal"/> or <see cref="Anti"/>.</returns>
		internal static int CwSide(bool forward)
		{
			return 1 - CcwSide(forward);
		}

		/// <summary>
		/// Build the cell complex over every piece of the graph.
		/// </summary>
		/// <remarks>
		/// Discarded/regularized pieces are deliberately *not* excluded: thin material
		/// cancels arithmetically in the winding sum, which is both simpler and more robust
		/// than deciding up front which sheets are real.
		/// </remarks>
		/// <param name="graph">The intersection graph.</param>
		/// <returns>The cell complex.</returns>
		public static CellComplex BuildCells(IntersectionGraph graph)
		{
			return BuildCellsWithToken(graph, null)
				?? throw new InvalidOperationException("uncancellable build_cells cannot cancel");
		}

		/// <summary>
		/// <see cref="BuildCells"/> with cooperative cancellation, checked once per arrangement
		/// edge. Returns null when the token fires.
		/// </summary>
		/// <param name="graph">The intersection graph.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The cell complex, or null when cancelled.</returns>
		public static CellComplex? BuildCellsWithToken(IntersectionGraph graph, CancelToken? token)
		{
			return BuildCellsWithProgress(graph, token, null);
		}

		/// <summary>
		/// <see cref="BuildCellsWithToken"/> that also reports the arrangement-edge sweep's
		/// fraction to <paramref name="progress"/>. Null costs nothing (see
		/// <see cref="ProgressReporter"/>).
		/// </summary>
		/// <param name="graph">The intersection graph.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>The cell complex, or null when cancelled.</returns>
		public static CellComplex? BuildCellsWithProgress(
			IntersectionGraph graph,
			CancelToken? token,
			ProgressReporter? progress)
		{
			ArgumentNullException.ThrowIfNull(graph);
			int n = graph.Pieces.Count;
			VertTables vt = VertTables.Of(graph);
			DisjointSets ds = new DisjointSets((uint)Math.Max(2 * n, 1));

			// Incident half-faces per edge, as one flat array sorted by edge rather
			// than a hash entry owning its own Vec: the allocation churn of ~3n tiny
			// Vecs dominated cell construction on large arrangements.
			Half[] incident = new Half[3 * n];
			for (int pi = 0; pi < n; pi++)
			{
				UVec3 vi = graph.Pieces[pi].Vi;
				for (int e = 0; e < 3; e++)
				{
					uint a = vi[e];
					uint b = vi[(e + 1) % 3];
					incident[(3 * pi) + e] = (GraphTypes.EdgeKey(a, b), pi, a < b, vi[(e + 2) % 3]);
				}
			}

			// Rust `sort_unstable`, and it stays unstable here: two entries compare equal
			// only when all four fields match, and four matching fields make them the same
			// record, so no ordering of the ties is observable.
			Array.Sort(incident, CompareHalf);

			Progress.BeginPhase(progress, Phase.Cells, (ulong)incident.Length);
			int at = 0;
			while (at < incident.Length)
			{
				if (Cancel.IsCancelled(token))
				{
					return null;
				}

				(uint A, uint B) key = incident[at].Key;
				int end = at + 1;
				while (end < incident.Length && incident[end].Key == key)
				{
					end++;
				}

				int start = at;
				int count = end - at;
				progress?.Advance((ulong)count);
				at = end;
				if (count < 2)
				{
					continue; // a boundary edge bounds no wedge
				}

				// Ordinary manifold edge whose two faces are provably in different
				// radial directions: the cyclic order is trivial, so both wedges
				// link with no exact angle work at all. These edges outnumber the
				// rest by a wide margin, and computing rational radial directions
				// for them dominated cell construction. The filter must certify
				// non-coplanarity — a coincident pair has one radial position, not
				// two, and linking it as two would fuse the sheet's own sides.
				if (count == 2)
				{
					Sign? sign = Exact.Approx.Orient3dA(
						vt.VertsF64[(int)key.A],
						vt.VertsF64[(int)key.B],
						vt.VertsF64[(int)incident[start].Apex],
						vt.VertsF64[(int)incident[start + 1].Apex]);
					if (sign == Sign.Neg || sign == Sign.Pos)
					{
						int p0 = incident[start].Id;
						bool fw0 = incident[start].Forward;
						int p1 = incident[start + 1].Id;
						bool fw1 = incident[start + 1].Forward;
						ds.Unite(Node(p0, CcwSide(fw0)), Node(p1, CwSide(fw1)));
						ds.Unite(Node(p1, CcwSide(fw1)), Node(p0, CwSide(fw0)));
						continue;
					}
				}

				(List<Inc> Incs, List<(int Start, int End)> Groups)? fan
					= RadialFan(key.A, key.B, new ArraySegment<Half>(incident, start, count), vt);
				if (fan is null)
				{
					continue;
				}

				List<Inc> incs = fan.Value.Incs;
				List<(int Start, int End)> groups = fan.Value.Groups;
				for (int gi = 0; gi < groups.Count; gi++)
				{
					(int s, int e) = groups[gi];

					// All faces of a wall share the cell on each of its two sides.
					for (int k = s; k < e; k++)
					{
						ds.Unite(
							Node(incs[s].Id, incs[s].CcwSide()),
							Node(incs[k].Id, incs[k].CcwSide()));
						ds.Unite(
							Node(incs[s].Id, incs[s].CwSide()),
							Node(incs[k].Id, incs[k].CwSide()));
					}

					// The wedge between this wall and the next: CCW side of this
					// wall meets the CW side of the next.
					int ns = groups[(gi + 1) % groups.Count].Start;
					if (groups.Count > 1)
					{
						ds.Unite(
							Node(incs[s].Id, incs[s].CcwSide()),
							Node(incs[ns].Id, incs[ns].CwSide()));
					}
				}
			}

			// Every incident half-face was advanced for, cancel-free, so close the bar at
			// exactly 1.0 — the throttle cannot, since it emits only on multiples of
			// `total / 100` and the group that spends the last units rarely lands on one.
			// Success only: the cancelled return above skips it, because a full bar is a
			// claim that the work was done. C#-only, divergence ledger entry 4.
			Progress.CompletePhase(progress);

			// Compact the union-find roots into dense cell ids. Roots are already
			// node ids, so a flat table beats hashing here.
			uint[] cellOf = new uint[2 * n];
			uint[] remap = new uint[2 * n];
			Array.Fill(remap, uint.MaxValue);
			uint numCells = 0;
			for (int i = 0; i < 2 * n; i++)
			{
				int root = (int)ds.Find((uint)i);
				if (remap[root] == uint.MaxValue)
				{
					remap[root] = numCells;
					numCells++;
				}

				cellOf[i] = remap[root];
			}

			return new CellComplex
			{
				NumCells = (int)numCells,
				CellOf = cellOf,
				Walls = Walls(graph),
			};
		}

		/// <summary>Rust's <c>incident.sort_unstable()</c>: the tuple's lexicographic order.</summary>
		/// <param name="x">The left record.</param>
		/// <param name="y">The right record.</param>
		/// <returns>Negative, zero or positive.</returns>
		private static int CompareHalf(Half x, Half y)
		{
			int c = x.Key.A.CompareTo(y.Key.A);
			if (c != 0)
			{
				return c;
			}

			c = x.Key.B.CompareTo(y.Key.B);
			if (c != 0)
			{
				return c;
			}

			c = x.Id.CompareTo(y.Id);
			if (c != 0)
			{
				return c;
			}

			c = x.Forward.CompareTo(y.Forward);
			if (c != 0)
			{
				return c;
			}

			return x.Apex.CompareTo(y.Apex);
		}
	}
}
