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

// EdgeOp.Orbits.cs — port of edge_op_orbits.rs — vertex-orbit scans for
// `SplitPinchedVerts` and `DedupeEdges` (EdgeOp.Dedupe.cs): the owner search that
// lets both run in parallel, and the per-orbit duplicate search they share.
// manifold-rust 6e127b5 (PR #8) and e91eeee (the 64-step walk cap).
//
// Both scans walk each vertex orbit (stepping `NextHalfedge(PairedHalfedge)`,
// step A in EdgeOp.Dedupe.cs's header) once, from its smallest eligible
// halfedge, its owner, using visited flags in ascending order. C++ runs both in
// parallel (edge_op.cpp:722-796, 903-924).
//
// If every orbit is a closed cycle, a halfedge owns its orbit iff its walk
// returns to it without meeting a smaller eligible halfedge. If
// `PairedHalfedge` is an involution the step is injective, so a walk either
// returns to its start or reaches a halfedge with no pair, which an open
// orbit's smallest eligible halfedge must do. Otherwise, or on an open orbit,
// the callers fall back to their sequential scans.
//
// Walking a whole orbit from every eligible halfedge costs the sum of the
// squared orbit lengths, Θ(n²) for one high-valence vertex. So the parallel
// walks stop after `OrbitWalkCap` steps, and the orbits they leave
// unresolved are settled by one sequential visited-flag pass, which walks
// each such orbit once. Why the owners are still exactly the sequential
// scan's, in its order, given closed orbits:
//
// - An orbit of length L <= cap: every walk from an eligible halfedge returns
//   to its start within L steps unless it first meets a smaller eligible one,
//   so the walks alone mark exactly its smallest eligible halfedge owner.
// - An orbit of length L > cap: no walk returns within the cap. A walk from a
//   halfedge that is not the smallest eligible one ends `NotOwner` or
//   `Unresolved`; the walk from the smallest one meets nothing smaller and
//   ends `Unresolved`. So the orbit's `Unresolved` halfedges are eligible and
//   include its smallest eligible halfedge, which is therefore the smallest
//   `Unresolved` one. The ascending pass reaches it first, makes it the owner
//   and marks the whole orbit visited, so no other halfedge of the orbit
//   becomes an owner.
// - The pass walks only from `Unresolved` halfedges, and a walk never leaves
//   its orbit, so short orbits keep the roles their walks gave them.
//
// Every orbit with an eligible halfedge thus has exactly one owner, its
// smallest eligible halfedge, as in the sequential scan, and the owners are
// collected ascending, the order in which that scan reaches them. An open
// orbit with an eligible halfedge is still caught: its smallest eligible
// halfedge's walk meets no smaller one, so it either reaches the open end
// (`Open`) or the cap, and then the pass reaches it first and its full walk
// reaches the open end.
//
// ── C# port notes ────────────────────────────────────────────────────────────
// The Rust returns `None` unless the `parallel` feature is compiled in
// (`!cfg!(feature = "parallel")`); here that is `!ManifoldParallel.Enabled`, the
// runtime switch Par.cs's header describes, so with the switch off every caller
// runs exactly the sequential scan it ran before. The per-halfedge walks are a
// Par.MaybeParMap and the involution check a Par.MaybeParFilter, at the Rust's
// thresholds, so each is bit-identical to its own sequential loop (Par.cs claim 1)
// and the owner list is the sequential scan's by the argument above.

namespace ManifoldSharp
{
	/// <content>
	/// The orbit-owner search and the per-orbit duplicate search of edge_op_orbits.rs.
	/// </content>
	public static partial class EdgeOp
	{
		/// <summary>
		/// Halfedge count from which the orbit scans run in parallel. Above C++'s 1e4:
		/// every halfedge walks its own orbit, and below 100k that costs more than it saves.
		/// </summary>
		internal const int OrbitParThreshold = 100_001;

		/// <summary>
		/// Steps after which a parallel orbit walk stops and leaves its orbit to the
		/// sequential pass. Vertex valences rarely exceed it, so that pass is usually
		/// empty, and it bounds the parallel work at 64 steps per halfedge.
		/// </summary>
		internal const int OrbitWalkCap = 64;

		private const byte NotOwner = 0;
		private const byte Owner = 1;
		private const byte Open = 2;
		private const byte Unresolved = 3;

		/// <summary>
		/// The owner of every orbit with an eligible halfedge, ascending, or null below
		/// <paramref name="threshold"/>, with <see cref="ManifoldParallel.Enabled"/> off, or
		/// if an orbit is not a closed cycle.
		/// </summary>
		/// <param name="halfedge">The halfedge arena.</param>
		/// <param name="threshold">Halfedge count from which the search runs.</param>
		/// <param name="eligible">Which halfedges may own an orbit. Must be pure.</param>
		/// <returns>The owners ascending, or null for "run the sequential scan".</returns>
		internal static int[]? OrbitOwners(
			List<Halfedge> halfedge,
			int threshold,
			Func<Halfedge, bool> eligible)
		{
			int n = halfedge.Count;
			if (n < threshold || !ManifoldParallel.Enabled)
			{
				return null;
			}

			int[] pairedBack = Par.MaybeParFilter(n, threshold, i =>
			{
				int p = halfedge[i].PairedHalfedge;
				return p >= 0 && (p >= n || halfedge[p].PairedHalfedge != i);
			});
			if (pairedBack.Length != 0)
			{
				return null;
			}

			byte[] role = Par.MaybeParMap(n, threshold, i =>
			{
				if (!eligible(halfedge[i]))
				{
					return NotOwner;
				}

				int current = i;
				for (int step = 0; step < OrbitWalkCap; step++)
				{
					if (!TryStep(halfedge, current, out int next))
					{
						return Open;
					}

					current = next;
					if (current == i)
					{
						return Owner;
					}

					if (current < i && eligible(halfedge[current]))
					{
						return NotOwner;
					}
				}

				return Unresolved;
			});
			if (Array.IndexOf(role, Open) >= 0)
			{
				return null;
			}

			// The sequential pass over the orbits longer than the cap (see the header).
			bool[]? visited = null;
			for (int i = 0; i < n; i++)
			{
				if (role[i] != Unresolved)
				{
					continue;
				}

				visited ??= new bool[n];
				if (visited[i])
				{
					role[i] = NotOwner;
					continue;
				}

				role[i] = Owner;
				int current = i;
				while (true)
				{
					visited[current] = true;
					if (!TryStep(halfedge, current, out current))
					{
						return null;
					}

					if (current == i)
					{
						break;
					}
				}
			}

			List<int> owners = new List<int>();
			for (int i = 0; i < n; i++)
			{
				if (role[i] == Owner)
				{
					owners.Add(i);
				}
			}

			return owners.ToArray();
		}

		/// <summary>
		/// The Rust's <c>step</c> closure: the orbit step from <paramref name="current"/>,
		/// or false at an open end (no pair, or a count not a multiple of 3), which is left
		/// to the sequential scan.
		/// </summary>
		private static bool TryStep(List<Halfedge> halfedge, int current, out int next)
		{
			int p = halfedge[current].PairedHalfedge;
			if (p < 0)
			{
				next = -1;
				return false;
			}

			next = Types.NextHalfedge(p);
			return next < halfedge.Count;
		}

		/// <summary>
		/// Appends the duplicate edges of the orbit walked from <paramref name="i"/>, in walk
		/// order: halfedges whose end vertex a smaller halfedge of the orbit also ends at.
		/// <paramref name="processed"/>, when given, is marked for each halfedge the walk
		/// visits (the Rust's <c>mark</c> closure; the parallel caller passes none).
		/// </summary>
		/// <param name="mesh">The mesh, read only.</param>
		/// <param name="i">The orbit's first halfedge.</param>
		/// <param name="nEdges">The safety bound on the walk.</param>
		/// <param name="processed">The sequential scan's visited flags, or null.</param>
		/// <param name="duplicates">Receives the duplicates.</param>
		internal static void DedupeOrbit(
			ManifoldImpl mesh,
			int i,
			int nEdges,
			bool[]? processed,
			List<int> duplicates)
		{
			// Track all endVerts seen in this vertex's orbit, keeping smallest edge idx.
			// Uses ForVert traversal: current = next_halfedge(halfedge[current].paired_halfedge)
			List<EndVertEntry> endVerts = new List<EndVertEntry>(); // (endVert, min_edge_idx)

			// Process i itself first
			if (processed is not null)
			{
				processed[i] = true;
			}

			int cEv0 = mesh.Halfedge[i].EndVert;
			if (cEv0 >= 0)
			{
				endVerts.Add(new EndVertEntry(cEv0, i));
			}

			// Then orbit (with safety bound to prevent infinite loops)
			int current = i;
			int orbitSteps = 0;
			while (true)
			{
				int pair = mesh.Halfedge[current].PairedHalfedge;
				if (pair < 0)
				{
					break;
				}

				current = Types.NextHalfedge(pair);
				if (current == i)
				{
					break;
				}

				orbitSteps += 1;
				if (orbitSteps > nEdges)
				{
					break;
				} // safety

				if (processed is not null)
				{
					processed[current] = true;
				}

				int cSv = mesh.Halfedge[current].StartVert;
				int cEv = mesh.Halfedge[current].EndVert;
				if (cSv >= 0 && cEv >= 0)
				{
					// Rust `iter_mut().find(...)` then mutates through the &mut;
					// EndVertEntry is a struct in a List, so the C# equivalent is a
					// scan for the index and a store back (list[j].Field = x is
					// CS1612).
					int found = FindEndVert(endVerts, cEv);
					if (found >= 0)
					{
						if (current < endVerts[found].MinEdge)
						{
							endVerts[found] = new EndVertEntry(endVerts[found].EndVert, current);
						}
					}
					else
					{
						endVerts.Add(new EndVertEntry(cEv, current));
					}
				}
			}

			// Second pass: find edges that aren't the minimum for their endVert
			cEv0 = mesh.Halfedge[i].EndVert;
			if (cEv0 >= 0)
			{
				int found = FindEndVert(endVerts, cEv0);
				if (found >= 0 && endVerts[found].MinEdge != i)
				{
					duplicates.Add(i);
				}
			}

			current = i;
			orbitSteps = 0;
			while (true)
			{
				int pair = mesh.Halfedge[current].PairedHalfedge;
				if (pair < 0)
				{
					break;
				}

				current = Types.NextHalfedge(pair);
				if (current == i)
				{
					break;
				}

				orbitSteps += 1;
				if (orbitSteps > nEdges)
				{
					break;
				} // safety

				int cEv = mesh.Halfedge[current].EndVert;
				if (cEv >= 0)
				{
					int found = FindEndVert(endVerts, cEv);
					if (found >= 0 && endVerts[found].MinEdge != current)
					{
						duplicates.Add(current);
					}
				}
			}
		}
	}
}
