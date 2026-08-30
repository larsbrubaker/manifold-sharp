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

// Pairing.cs — port of robust/pairing.rs, whose header reads:
//
//   Geometrically correct half-edge pairing for the extracted boundary (used
//   by robust/assemble.rs).
//
//   `cells::extract` can legitimately emit a boundary that touches itself
//   along an edge: the solid occupies two separate wedges around one
//   arrangement edge, so that undirected vertex-id edge carries four (in
//   general 2k) half-edges. `ManifoldImpl::create_halfedges` pairs half-edges
//   by vertex ids alone, which on such an edge picks an arbitrary
//   forward/backward partner. A crosslinked guess fuses the two geometric
//   fans into one combinatorial orbit, and `edge_op::cleanup_topology` then
//   "repairs" the fused orbit by repointing corners of unrelated faces onto
//   another vertex's position — moving geometry and destroying volume (on
//   Thingi10K #301921 ∪ its rotated copy, one `dedupe_edge` call cost 7% of
//   the model).
//
//   This module removes the guess. It radially sorts the half-faces around
//   every such edge with the same filtered orient3d machinery the cell
//   complex uses (`cells::radial_fan`), pairs each half-edge with the
//   radially adjacent one bounding the *same solid wedge*, and reports the
//   split of each pinched vertex into one copy per fan that the pairing
//   implies. The assembly emits those copies as distinct output vertices, so
//   every undirected edge carries exactly two half-edges again and the
//   import's id-based pairing reproduces the geometric one — leaving
//   `cleanup_topology` with nothing to repair.
//
//   An accepted plan splits *every* pinched vertex orbit in the mesh, not
//   only the ones a multi-edge fan touches: the copy index comes from the
//   connected components of the corner graph induced by the whole pairing, so
//   a vertex whose star is two cones meeting at a single point, far from any
//   multi-edge, is separated as well.
//
//   Only meshes that actually carry such an edge take this path; everything
//   else keeps using the generic MeshGL import unchanged. Anything the fan
//   cannot certify (odd fans, coincident radial directions, apexes on the
//   edge axis, traversals that do not alternate, a split that fails to
//   separate the fans) yields `None`, and the caller falls back to that same
//   generic import.
//
// ── The Rust's rustc-hash note, kept with its maps ───────────────────────────
// Fx hashing (unseeded): every map here is probe-only — fan-copy ordinals are
// assigned walking half-edges in index order, and the edge-count table is only
// ever looked up — so hash order cannot reach the split plan. Plain
// `Dictionary` is therefore the faithful replacement (docs/PORTING_PLAN.md's
// dependency-replacement table).
//
// ── `usize::MAX` becomes `Unpaired` ──────────────────────────────────────────
// The Rust's "no partner yet" sentinel is `usize::MAX`, and `split_from_partners`
// leans on `usize::MAX < h` being false. `int.MaxValue` reproduces both the
// equality test and that ordering; -1 would reproduce only the first.
//
// ── `Option<SplitPlan>` is a nullable array ──────────────────────────────────
// `SplitPlan = Vec<u32>` is `uint[]`, and the Rust's `None` — "no split needed,
// or the geometry could not be certified" — is `null`. Both readings are the
// same instruction to the caller: take the untouched import path.

using ManifoldSharp.Linalg;

// Rust's `type Half = (EdgeKey, usize, bool, u32)`, declared in this very file
// (pairing.rs) and consumed by `radial_fan`. C# aliases are file-scoped, so
// this repeats what Cells.cs and Cells.Fan.cs each declare.
using Half = ((uint A, uint B) Key, int Id, bool Forward, uint Apex);

namespace ManifoldSharp.Robust
{
	/// <summary>
	/// The free functions of <c>robust/pairing.rs</c>: the geometric half-edge pairing of
	/// the extracted boundary and the vertex split it implies.
	/// </summary>
	public static class Pairing
	{
		/// <summary>
		/// The Rust's <c>usize::MAX</c> sentinel for "this half-edge has no partner yet".
		/// </summary>
		private const int Unpaired = int.MaxValue;

		/// <summary>
		/// Whole attempts (one radial pass over every fan plus one split) are bounded by a
		/// small constant rather than by the fan count: the fan predicates are the expensive
		/// part of assembly, and a mesh that needs more rounds than this is better served by
		/// degrading to the untouched import path than by paying O(fans) passes to maybe
		/// salvage it.
		/// </summary>
		private const int MaxAttempts = 4;

		/// <summary>
		/// Plan the vertex splits that make id-based half-edge pairing reproduce the
		/// geometric pairing of <paramref name="tris"/> (triangles of interned vertex ids,
		/// wound outward).
		/// </summary>
		/// <remarks>
		/// Returns <c>null</c> when no split is needed (no edge carries more than two
		/// half-edges — the overwhelmingly common case, which must stay on the untouched
		/// import path) and also when the geometry cannot be certified, so callers get
		/// today's behavior rather than a guess.
		/// <para>
		/// The geometric pairing (<c>PairFan</c>) is preferred everywhere. It is not always
		/// expressible through vertex ids: when the two sheets meeting along an edge
		/// reconnect around *both* of its endpoints, no vertex split can separate them, and
		/// leaving the duplicate id-edge in place hands the mesh to <c>DedupeEdge</c>, which
		/// splits the pinched *start* vertex onto the end vertex's position (upstream
		/// <c>Impl::DedupeEdge</c>) and wrecks the geometry. Such a fan falls back to the
		/// other radially adjacent pairing — also a closed, consistently oriented surface
		/// over the identical triangles, differing only in which sheets are joined — and the
		/// plan is rejected outright if even that leaves the edge duplicated.
		/// </para>
		/// </remarks>
		/// <param name="tris">Triangles of interned vertex ids, wound outward.</param>
		/// <param name="vt">The vertex tables the radial machinery reads.</param>
		/// <returns>The per-corner copy index, or null.</returns>
		public static uint[]? PlanVertexSplits(IReadOnlyList<UVec3> tris, VertTables vt)
		{
			ArgumentNullException.ThrowIfNull(tris);

			// NARROWING (usize -> int, this line and the `basePartner` allocation below).
			// The Rust computes `3 * tris.len()` in usize; here both operands are int, so
			// the product wraps (unchecked) above 2^31/3 ≈ 715M triangles. That bound is
			// unreachable: `tris` comes from the extracted boundary, whose pieces are
			// already indexed by int throughout the graph, so a soup that large could not
			// have been built to reach this call.
			List<Half> incident = new List<Half>(3 * tris.Count);
			for (int t = 0; t < tris.Count; t++)
			{
				UVec3 vi = tris[t];
				for (int c = 0; c < 3; c++)
				{
					uint a = vi[c];
					uint b = vi[(c + 1) % 3];
					if (a == b)
					{
						return null; // degenerate corner: no radial direction
					}

					incident.Add((GraphTypes.EdgeKey(a, b), (3 * t) + c, a < b, vi[(c + 2) % 3]));
				}
			}

			// Rust `sort_unstable` on a tuple whose second field is the globally unique
			// half-edge index: no two entries can tie, so the unstable sort is
			// deterministic and `List.Sort` is the faithful spelling.
			incident.Sort();

			// Partner half-edge of every half-edge on an ordinary (two half-edge)
			// arrangement edge; the fans fill in the rest on each attempt.
			int[] basePartner = new int[3 * tris.Count];
			Array.Fill(basePartner, Unpaired);
			List<Fan> fans = new List<Fan>();
			int at = 0;
			while (at < incident.Count)
			{
				(uint A, uint B) key = incident[at].Key;
				int end = at + 1;
				while (end < incident.Count && incident[end].Key == key)
				{
					end++;
				}

				int rawStart = at;
				int rawCount = end - at;
				at = end;
				if (rawCount == 2)
				{
					// An ordinary edge: the two half-edges must run opposite ways.
					Half r0 = incident[rawStart];
					Half r1 = incident[rawStart + 1];
					if (r0.Forward == r1.Forward)
					{
						return null;
					}

					basePartner[r0.Id] = r1.Id;
					basePartner[r1.Id] = r0.Id;
					continue;
				}

				if (rawCount % 2 != 0)
				{
					return null; // a boundary or odd fan: not a closed surface
				}

				fans.Add(new Fan(key, incident.GetRange(rawStart, rawCount)));
			}

			if (fans.Count == 0)
			{
				return null;
			}

			int budget = MaxAttempts;
			while (true)
			{
				Attempt? a = TryAttempt(tris, basePartner, fans, vt, ref budget);
				if (a == null)
				{
					return null;
				}

				if (a.Separated.All(s => s))
				{
					return Settle(tris, basePartner, fans, vt, ref budget, a);
				}

				// Flip the fans the geometric pairing could not separate. Flips
				// latch within this loop — a fan flipped to unblock one round is
				// not reconsidered here even if a later round would have separated
				// it geometrically — which is what `Settle` exists to undo.
				bool progress = false;
				for (int i = 0; i < fans.Count; i++)
				{
					if (!a.Separated[i] && !fans[i].Flip)
					{
						fans[i].Flip = true;
						progress = true;
					}
				}

				if (!progress)
				{
					return null;
				}
			}
		}

		/// <summary>
		/// One pairing pass over every fan plus the split it implies. Consumes one unit of
		/// <paramref name="budget"/>; null once that runs out or when a fan cannot be
		/// certified.
		/// </summary>
		/// <param name="tris">The boundary triangles.</param>
		/// <param name="basePartner">Partners of the ordinary edges' half-edges.</param>
		/// <param name="fans">The multi-half-edge arrangement edges.</param>
		/// <param name="vt">The vertex tables.</param>
		/// <param name="budget">The remaining attempt budget.</param>
		/// <returns>The attempt's outcome, or null.</returns>
		private static Attempt? TryAttempt(
			IReadOnlyList<UVec3> tris,
			int[] basePartner,
			IReadOnlyList<Fan> fans,
			VertTables vt,
			ref int budget)
		{
			if (budget == 0)
			{
				return null;
			}

			budget--;
			int[] partner = (int[])basePartner.Clone();
			foreach (Fan fan in fans)
			{
				if (!PairFan(fan, vt, partner))
				{
					return null;
				}
			}

			uint[] plan = SplitFromPartners(tris, partner);
			Dictionary<((uint, uint) A, (uint, uint) B), (uint F, uint B)>? counts
				= SplitEdgeCounts(tris, plan);
			if (counts == null)
			{
				return null;
			}

			bool[] separated = new bool[fans.Count];
			for (int i = 0; i < fans.Count; i++)
			{
				separated[i] = FanSeparated(fans[i], tris, plan, counts);
			}

			return new Attempt(plan, separated, counts.Values.All(e => e.F == 1 && e.B == 1));
		}

		/// <summary>
		/// Drop flips that are no longer needed.
		/// </summary>
		/// <remarks>
		/// A fan flipped early can owe its flip to another fan that has since been flipped
		/// too, so re-test each flipped fan geometrically and keep the geometric pairing
		/// wherever it now separates. Best effort within the remaining attempt budget: an
		/// un-flip that cannot be re-tested stays.
		/// </remarks>
		/// <param name="tris">The boundary triangles.</param>
		/// <param name="basePartner">Partners of the ordinary edges' half-edges.</param>
		/// <param name="fans">The multi-half-edge arrangement edges.</param>
		/// <param name="vt">The vertex tables.</param>
		/// <param name="budget">The remaining attempt budget.</param>
		/// <param name="accepted">The attempt that already separated every fan.</param>
		/// <returns>The settled split plan, or null when it leaves an edge duplicated.</returns>
		private static uint[]? Settle(
			IReadOnlyList<UVec3> tris,
			int[] basePartner,
			List<Fan> fans,
			VertTables vt,
			ref int budget,
			Attempt accepted)
		{
			for (int i = 0; i < fans.Count; i++)
			{
				if (!fans[i].Flip || budget == 0)
				{
					continue;
				}

				fans[i].Flip = false;
				Attempt? a = TryAttempt(tris, basePartner, fans, vt, ref budget);
				if (a != null && a.Separated.All(s => s))
				{
					accepted = a;
				}
				else
				{
					fans[i].Flip = true;
				}
			}

			return accepted.EdgesOk ? accepted.Plan : null;
		}

		/// <summary>
		/// Pair the half-edges of one fan across their shared solid wedges.
		/// </summary>
		/// <remarks>
		/// <see cref="Cells.RadialFan"/> orders the half-faces counter-clockwise about the
		/// canonical edge direction <c>k0 → k1</c>. A forward-traversing face is wound
		/// <c>(k0, k1, apex)</c>, so its normal sits 90° counter-clockwise of its apex
		/// direction (<c>cells::ccw_side</c>), and the extracted boundary's normals point
		/// *away* from material (<c>cells::extract</c>) — so a forward face has material in
		/// the wedge clockwise of it and a backward face in the wedge counter-clockwise of
		/// it. Traversals therefore alternate around the fan, and the wedge between radial
		/// positions <c>i</c> and <c>i + 1</c> is solid exactly when face <c>i</c> is
		/// backward. Those two faces bound one solid wedge, which makes them the adjacent
		/// pair of the surface enclosing it — for two cubes meeting along an edge, exactly
		/// the two faces of the same cube.
		/// <para>
		/// <c>fan.Flip</c> pairs across the empty wedges instead; see
		/// <see cref="PlanVertexSplits"/> for when that is used.
		/// </para>
		/// </remarks>
		/// <param name="fan">The fan to pair.</param>
		/// <param name="vt">The vertex tables.</param>
		/// <param name="partner">The partner table, filled in for this fan's half-edges.</param>
		/// <returns>True on success; false is the Rust's <c>None</c>.</returns>
		private static bool PairFan(Fan fan, VertTables vt, int[] partner)
		{
			(List<Inc> Incs, List<(int Start, int End)> Groups)? fanned
				= Cells.RadialFan(fan.Key.A, fan.Key.B, fan.Halfs, vt);
			if (fanned == null)
			{
				return false;
			}

			List<Inc> incs = fanned.Value.Incs;
			List<(int Start, int End)> groups = fanned.Value.Groups;

			// Every half-face must survive the fan (none on the edge axis) and hold
			// its own radial direction (no coincident pair to disambiguate).
			if (incs.Count != fan.Halfs.Count || groups.Count != incs.Count)
			{
				return false;
			}

			int n = incs.Count;
			for (int i = 0; i < n; i++)
			{
				if (incs[i].Forward != fan.Flip)
				{
					continue;
				}

				int j = (i + 1) % n;
				if (incs[j].Forward == fan.Flip)
				{
					return false; // traversals must alternate around the fan
				}

				partner[incs[i].Id] = incs[j].Id;
				partner[incs[j].Id] = incs[i].Id;
			}

			foreach (Half h in fan.Halfs)
			{
				if (partner[h.Id] == Unpaired)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>Corners that the pairing keeps in one fan become one output vertex.</summary>
		/// <param name="tris">The boundary triangles.</param>
		/// <param name="partner">The half-edge pairing.</param>
		/// <returns>The per-corner copy index.</returns>
		private static uint[] SplitFromPartners(IReadOnlyList<UVec3> tris, int[] partner)
		{
			int n = 3 * tris.Count;

			// NARROWING (int -> uint): the Rust is `n.max(1) as u32`. Both sides wrap the
			// same way on overflow, and `Math.Max(n, 1)` is non-negative by construction,
			// so the cast is value-preserving for every reachable n (bounded by the corner
			// count, which the caller already holds in an int-indexed array).
			DisjointSets ds = new DisjointSets((uint)Math.Max(n, 1));
			for (int h = 0; h < n; h++)
			{
				int p = partner[h];
				if (p == Unpaired || p < h)
				{
					continue;
				}

				// `h` runs a→b inside its triangle and `p` runs b→a inside its own,
				// so they meet at a via corners (h, p+1) and at b via (h+1, p).
				int ht = h / 3;
				int hc = h % 3;
				int pt = p / 3;
				int pc = p % 3;
				ds.Unite((uint)h, (uint)((3 * pt) + ((pc + 1) % 3)));
				ds.Unite((uint)((3 * ht) + ((hc + 1) % 3)), (uint)p);
			}

			// Probe-only maps (see the header): `ordinal` is keyed by union-find root and
			// `next` by vertex id, and the ordinals are handed out walking half-edges in
			// index order, so no enumeration order can reach the plan.
			Dictionary<uint, uint> ordinal = new Dictionary<uint, uint>();
			Dictionary<uint, uint> next = new Dictionary<uint, uint>();
			uint[] plan = new uint[n];
			for (int h = 0; h < n; h++)
			{
				uint root = ds.Find((uint)h);
				uint vid = tris[h / 3][h % 3];
				if (!ordinal.TryGetValue(root, out uint copy))
				{
					next.TryGetValue(vid, out uint slot);
					copy = slot;
					next[vid] = slot + 1;
					ordinal.Add(root, copy);
				}

				plan[h] = copy;
			}

			return plan;
		}

		/// <summary>Identity of one corner after splitting: its vertex id and fan copy.</summary>
		/// <param name="tris">The boundary triangles.</param>
		/// <param name="plan">The per-corner copy index.</param>
		/// <param name="h">The corner (half-edge) index.</param>
		/// <returns>The vertex id and its fan copy.</returns>
		private static (uint Vid, uint Copy) SplitVert(IReadOnlyList<UVec3> tris, uint[] plan, int h)
		{
			return (tris[h / 3][h % 3], plan[h]);
		}

		/// <summary>
		/// Forward/backward half-edge counts per undirected edge of the split mesh.
		/// </summary>
		/// <remarks>
		/// Null if a corner pair collapsed, which would make the counts meaningless. That is
		/// belt and braces: <see cref="PlanVertexSplits"/> already rejects a triangle with a
		/// repeated vertex id up front, and splitting only refines vertex identity, so it
		/// cannot merge two distinct corners.
		/// </remarks>
		/// <param name="tris">The boundary triangles.</param>
		/// <param name="plan">The per-corner copy index.</param>
		/// <returns>The per-edge counts, or null.</returns>
		private static Dictionary<((uint, uint) A, (uint, uint) B), (uint F, uint B)>? SplitEdgeCounts(
			IReadOnlyList<UVec3> tris,
			uint[] plan)
		{
			Dictionary<((uint, uint) A, (uint, uint) B), (uint F, uint B)> counts
				= new Dictionary<((uint, uint) A, (uint, uint) B), (uint F, uint B)>();
			for (int t = 0; t < tris.Count; t++)
			{
				for (int c = 0; c < 3; c++)
				{
					(uint Vid, uint Copy) a = SplitVert(tris, plan, (3 * t) + c);
					(uint Vid, uint Copy) b = SplitVert(tris, plan, (3 * t) + ((c + 1) % 3));
					if (a == b)
					{
						return null;
					}

					bool aFirst = Less(a, b);
					((uint, uint) A, (uint, uint) B) key = aFirst ? (a, b) : (b, a);
					counts.TryGetValue(key, out (uint F, uint B) e);
					counts[key] = aFirst ? (e.F + 1, e.B) : (e.F, e.B + 1);
				}
			}

			return counts;
		}

		/// <summary>Did the split actually separate this fan into ordinary edges?</summary>
		/// <param name="fan">The fan.</param>
		/// <param name="tris">The boundary triangles.</param>
		/// <param name="plan">The per-corner copy index.</param>
		/// <param name="counts">The per-edge counts of the split mesh.</param>
		/// <returns>True when every one of the fan's edges now carries exactly one half-edge each way.</returns>
		private static bool FanSeparated(
			Fan fan,
			IReadOnlyList<UVec3> tris,
			uint[] plan,
			Dictionary<((uint, uint) A, (uint, uint) B), (uint F, uint B)> counts)
		{
			foreach (Half half in fan.Halfs)
			{
				int h = half.Id;
				(uint Vid, uint Copy) a = SplitVert(tris, plan, h);
				(uint Vid, uint Copy) b = SplitVert(tris, plan, (3 * (h / 3)) + (((h % 3) + 1) % 3));
				((uint, uint) A, (uint, uint) B) key = Less(a, b) ? (a, b) : (b, a);
				if (!counts.TryGetValue(key, out (uint F, uint B) e) || e != (1u, 1u))
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// Rust's <c>&lt;</c> on a <c>(u32, u32)</c> tuple: lexicographic. C# value tuples
		/// carry no relational operators, so the comparison is spelled out rather than
		/// routed through <c>Comparer</c>, which would allocate on every call.
		/// </summary>
		/// <param name="a">The left tuple.</param>
		/// <param name="b">The right tuple.</param>
		/// <returns>True when <paramref name="a"/> sorts before <paramref name="b"/>.</returns>
		private static bool Less((uint Vid, uint Copy) a, (uint Vid, uint Copy) b)
		{
			return a.Vid != b.Vid ? a.Vid < b.Vid : a.Copy < b.Copy;
		}

		/// <summary>One arrangement edge carrying more than two half-edges.</summary>
		private sealed class Fan
		{
			/// <summary>The canonical edge key.</summary>
			public readonly (uint A, uint B) Key;

			/// <summary>Its incident half-face records.</summary>
			public readonly List<Half> Halfs;

			/// <summary>
			/// Pair across the empty wedges instead of the solid ones. Only set when the
			/// geometric pairing leaves the fans inseparable — see
			/// <see cref="PlanVertexSplits"/>.
			/// </summary>
			public bool Flip;

			/// <summary>Creates a fan with <see cref="Flip"/> clear (the Rust's literal).</summary>
			/// <param name="key">The canonical edge key.</param>
			/// <param name="halfs">Its incident half-face records.</param>
			public Fan((uint A, uint B) key, List<Half> halfs)
			{
				this.Key = key;
				this.Halfs = halfs;
				this.Flip = false;
			}
		}

		/// <summary>
		/// The outcome of one pairing pass: the split it implies and which fans it managed
		/// to separate.
		/// </summary>
		private sealed class Attempt
		{
			/// <summary>The per-corner copy index this pass implies.</summary>
			public readonly uint[] Plan;

			/// <summary>Which fans the split separated into ordinary edges.</summary>
			public readonly bool[] Separated;

			/// <summary>
			/// Every edge of the split mesh is an ordinary one (exactly one half-edge each
			/// way) — the condition the caller must hand the import.
			/// </summary>
			public readonly bool EdgesOk;

			/// <summary>Creates an attempt record.</summary>
			/// <param name="plan">The split plan.</param>
			/// <param name="separated">Per-fan separation verdicts.</param>
			/// <param name="edgesOk">Whether every split edge is ordinary.</param>
			public Attempt(uint[] plan, bool[] separated, bool edgesOk)
			{
				this.Plan = plan;
				this.Separated = separated;
				this.EdgesOk = edgesOk;
			}
		}
	}
}
