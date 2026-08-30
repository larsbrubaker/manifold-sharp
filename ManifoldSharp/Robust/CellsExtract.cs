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

// CellsExtract.cs — port of robust/cells_extract.rs, whose header reads:
//
//   Turning the arrangement's cell labels into the result's boundary.
//
//   Split out of `cells.rs` (which builds the cell complex and propagates the
//   winding numbers) purely to keep both files within the project's
//   file-length limit; the two halves are re-exported as one `cells` API.
//   Everything here depends on `cells` for the complex, the walls, and the
//   per-cell windings, and on `intersection_graph` for the pieces it emits.
//
// The Rust's re-export (`pub use super::cells_extract::{extract, in_result};`)
// is why these two land on `Cells` rather than a class of their own: callers
// say `cells::extract` there and `Cells.Extract` here.

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Robust
{
	public static partial class Cells
	{
		/// <summary>Does a cell with this winding vector lie inside the operation's result?</summary>
		/// <remarks>
		/// The winding rule decides what "inside an operand" means: under
		/// <see cref="WindingRule.Positive"/> each operand's solid is <c>{w ≥ 1}</c>, so a
		/// negative winding is inverted geometry and never material; under
		/// <see cref="WindingRule.Nonzero"/> it is <c>{w ≠ 0}</c>, so an inside-out region
		/// counts as solid. Only this predicate changes — the winding numbers themselves are
		/// rule-independent.
		/// <para>
		/// Expressing every operation as one predicate on the winding vector is what lets
		/// subtraction drop its operand-flip trick: P − Q is just "inside P and not inside Q".
		/// </para>
		/// </remarks>
		/// <param name="op">The boolean operation.</param>
		/// <param name="rule">The winding rule.</param>
		/// <param name="w">The cell's winding vector.</param>
		/// <returns>True when the cell is material in the result.</returns>
		/// <exception cref="ArgumentOutOfRangeException">The rule or operation is not a declared value.</exception>
		public static bool InResult(OpType op, WindingRule rule, (int P, int Q) w)
		{
			bool Inside(int v)
			{
				switch (rule)
				{
					case WindingRule.Positive:
						return v >= 1;
					case WindingRule.Nonzero:
						return v != 0;
					default:
						throw new ArgumentOutOfRangeException(nameof(rule), rule, "unknown winding rule");
				}
			}

			bool a = Inside(w.P);
			bool b = Inside(w.Q);
			switch (op)
			{
				case OpType.Add:
					return a || b;
				case OpType.Intersect:
					return a && b;
				case OpType.Subtract:
					return a && !b;
				default:
					throw new ArgumentOutOfRangeException(nameof(op), op, "unknown operation");
			}
		}

		/// <summary>
		/// The boundary of the result: every wall whose two cells disagree about containment,
		/// wound so its normal points out of the solid.
		/// </summary>
		/// <remarks>
		/// Orientation is *derived* from the cell labels rather than inherited from the input
		/// face. That is what makes the output closed and consistently oriented no matter how
		/// the input was wound — an inverted region of a self-intersecting scan still lands in
		/// the right cells, and its emitted orientation is corrected on the way out. One
		/// representative per wall is emitted, so a coincident stack contributes a single face.
		/// </remarks>
		/// <param name="graph">The intersection graph the pieces come from.</param>
		/// <param name="complex">Its cell complex.</param>
		/// <param name="wind">The per-cell windings.</param>
		/// <param name="op">The boolean operation.</param>
		/// <param name="rule">The winding rule.</param>
		/// <returns>The result's boundary pieces.</returns>
		public static List<Piece> Extract(
			IntersectionGraph graph,
			CellComplex complex,
			Windings wind,
			OpType op,
			WindingRule rule)
		{
			ArgumentNullException.ThrowIfNull(graph);
			ArgumentNullException.ThrowIfNull(complex);
			ArgumentNullException.ThrowIfNull(wind);
			List<Piece> outPieces = new List<Piece>();
			foreach (Wall wall in complex.Walls)
			{
				int rep = wall.Rep;
				int cn = complex.Cell(rep, Normal);
				int ca = complex.Cell(rep, Anti);
				if (cn == ca || !wind.Known[cn] || !wind.Known[ca])
				{
					continue;
				}

				bool inN = InResult(op, rule, wind.W[cn]);
				bool inA = InResult(op, rule, wind.W[ca]);
				if (inN == inA)
				{
					continue; // same material both sides: not a boundary
				}

				Piece piece = graph.Pieces[rep];

				// The representative's normal points from its anti side toward its
				// normal side. Material belongs behind the emitted normal, so the
				// winding reverses when the solid is on the normal side instead.
				UVec3 vi = inA
					? piece.Vi
					: new UVec3(piece.Vi[0], piece.Vi[2], piece.Vi[1]);
				outPieces.Add(new Piece(piece.Mesh, piece.Tri, vi));
			}

			return outPieces;
		}
	}
}
