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

// Repair.cs — port of robust/repair.rs, whose header reads:
//
//   Shell-level orientation repair for closed, orientable triangle meshes
//   (manifold or soup).
//
//   Real-world scans often contain whole bodies wound inside-out. The robust
//   engine's solid semantics are {winding >= 1}, so an inverted body bounds no
//   material and silently vanishes from every boolean. This module decides,
//   per connected shell, whether the shell is wound the way its nesting
//   demands — outermost shells wind +1, first-level cavities wind -1, solids
//   inside cavities wind +1 again — and reports which shells to flip.
//
//   The decisions are made with the exact winding machinery of
//   `robust/ray_shoot.rs`, never with heuristics like signed volume:
//     * A shell's *current* orientation comes from the exact winding of a
//       point just off one of its faces, against the shell alone: an
//       outward-wound shell has winding 0 outside / 1 inside its own surface,
//       an inverted one -1 / 0. Anything else (a doubled sheet, a
//       multiply-wrapped surface) is deliberately left alone — the boolean's
//       coincident-stack arithmetic already classifies those correctly, and
//       "repairing" them would destroy information.
//     * A shell's *nesting depth* is the number of other shells that strictly
//       contain a point just off the shell's geometric exterior, measured by
//       the same exact query per shell — orientation-independently (winding
//       != 0), so already-broken neighbors cannot corrupt the count.
//
//   Blanket-flipping negative-signed-volume shells would instead turn every
//   legitimate cavity inside out; the depth rule is what keeps voids voids.
//
//   The depth rule is applied asymmetrically, because only one of its two
//   verdicts can restore lost material. Rewinding an *inverted* shell where the
//   nesting wants a solid gives material back, so it is always done. Rewinding
//   an *outward* shell where the nesting wants a cavity takes material away —
//   and a nested outward shell is genuinely ambiguous ({w >= 1} reads its
//   interior as solid winding 2, which is a perfectly valid mesh). We only do
//   that on strong evidence that the whole stack arrived mirrored: some
//   container of the shell is itself inverted at a depth that wants a solid.
//   That is a prior, not a proof — {-1 at depth 0, +1 at depth 1} could equally
//   be a lone inverted body with a legitimate inner solid — but it is the
//   reading that keeps the two shells consistent with each other.
//
//   The invariant this buys: every material-*removing* flip requires a
//   container with sign -1 at even depth, and such a container always takes the
//   unconditional restoring flip. So a removal never happens without a strictly
//   larger addition above it in the same containment stack, and the net effect
//   of a repair on a stack is never a loss. Thingi10K #61459 — a correctly
//   wound body carrying a nested outward shell, 80% of whose material the old
//   symmetric rule destroyed — cannot recur.
//
//   Known conservative false negative: a mirrored sub-assembly hanging under a
//   correctly wound parent ({+1 at depth 0, +1 at depth 1, -1 at depth 2}) is
//   read as two nested solids with an inverted island, not as a void containing
//   an island, because evidence is only searched *upward* through containers.
//   Nothing is destroyed — the mesh keeps all its material — but the intended
//   void stays filled. Propagating a descendant-inverted signal downward is the
//   fix if that class ever shows up in practice.
//
//   `Manifold::repair_orientation` applies the plan by rewinding whole shells
//   in place (`apply_flips`), which preserves halfedge pairing exactly because
//   a paired edge always joins two triangles of the same shell.
//
// ── `BTreeMap` becomes `SortedDictionary` ────────────────────────────────────
// The Rust reaches for `BTreeMap` twice here, and both times the ORDER is
// load-bearing rather than incidental: `connected_shells` hands out vertex ids
// and dense shell ids by first-insertion, and the shell ids decide which
// entries of the returned `flip` vector belong together. Insertion order is what
// actually pins those, not key order, but a `SortedDictionary` is the literal
// transcription and costs nothing here (both maps are tiny relative to the
// exact winding queries).
//
// ── `[Vec3; 3]` is a `Vec3[]` of length 3 ────────────────────────────────────
// The same call RayShoot.cs and Soup.cs made: `Soup.ImplToTris` already returns
// `List<Vec3[]>`, so a triangle is a three-element array here and every method
// below takes the soup in that shape.

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Robust
{
	/// <summary>
	/// Result of <see cref="Repair.PlanRepair"/>: which triangles to rewind, plus shell
	/// counts for reporting.
	/// </summary>
	public sealed class RepairPlan
	{
		/// <summary>Per-input-triangle flip flag, aligned with the <c>tris</c> argument.</summary>
		public bool[] Flip;

		/// <summary>How many connected shells the soup decomposed into.</summary>
		public int NumShells;

		/// <summary>How many of them the plan rewinds.</summary>
		public int FlippedShells;

		/// <summary>Creates a plan.</summary>
		/// <param name="flip">Per-triangle flip flags.</param>
		/// <param name="numShells">The shell count.</param>
		/// <param name="flippedShells">The rewound shell count.</param>
		public RepairPlan(bool[] flip, int numShells, int flippedShells)
		{
			this.Flip = flip;
			this.NumShells = numShells;
			this.FlippedShells = flippedShells;
		}

		/// <summary>True when the plan rewinds nothing.</summary>
		/// <returns>Whether the repair is a no-op.</returns>
		public bool IsNoop()
		{
			return this.FlippedShells == 0;
		}
	}

	/// <summary>
	/// The free functions of <c>robust/repair.rs</c>: shell decomposition, exact
	/// orientation classification, the flip plan and its in-place application.
	/// </summary>
	public static class Repair
	{
		/// <summary>
		/// How many candidate faces per shell to try before declaring the shell's
		/// orientation ambiguous. A simple closed shell classifies on its first
		/// non-degenerate face; only pathological sheets (every sampled face part of a
		/// coincident stack) need retries.
		/// </summary>
		private const int MaxCandidateFaces = 16;

		/// <summary>
		/// True when the soup's surface is *already* the boundary of the solid it denotes:
		/// every shell classifies cleanly (no coincident stacks, no multiple wraps) and winds
		/// the way its nesting demands — +1 at even containment depth, -1 at odd.
		/// </summary>
		/// <remarks>
		/// That is exactly the condition under which no winding classification can change the
		/// surface, for either winding rule: every point is enclosed 0 or 1 times, so
		/// {w &gt;= 1} and {w != 0} agree, and every face separates material from void.
		/// <para>
		/// Used by the robust engine to decide whether its bbox-disjoint union may take the
		/// concatenating fast path (which classifies nothing) or must go through the full
		/// pipeline. Deliberately conservative: <c>false</c> only ever costs the pipeline,
		/// never correctness.
		/// </para>
		/// </remarks>
		/// <param name="tris">The triangle soup.</param>
		/// <returns>Whether every shell already winds as its nesting demands.</returns>
		public static bool ShellsWellNested(IReadOnlyList<Vec3[]> tris)
		{
			Analysis a = Analyze(tris);
			for (int s = 0; s < a.NumShells; s++)
			{
				Classified? c = a.Classified[s];
				if (c == null)
				{
					return false;
				}

				if (c.Sign != (a.Containers[s].Count % 2 == 0 ? 1 : -1))
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// Decide which shells to flip so the mesh's solid reads correctly under
		/// {winding &gt;= 1}.
		/// </summary>
		/// <remarks>
		/// The target from containment depth is the classic one — even depth winds outward
		/// (+1), odd depth (cavity boundaries) winds inward (-1) — but it is enforced
		/// *asymmetrically*: an inverted shell that should be solid is always rewound (that
		/// restores material), while an outward shell that a cavity target disagrees with is
		/// only rewound on evidence that its containment stack arrived mirrored (that removes
		/// material, and a nested outward shell is a legitimate mesh on its own). See the
		/// module header for the full rule and the invariant it guarantees.
		/// </remarks>
		/// <param name="tris">The triangle soup.</param>
		/// <returns>The flip plan.</returns>
		public static RepairPlan PlanRepair(IReadOnlyList<Vec3[]> tris)
		{
			Analysis analysis = Analyze(tris);
			int[] shellOf = analysis.ShellOf;
			int numShells = analysis.NumShells;
			Classified?[] classified = analysis.Classified;
			List<int>[] containers = analysis.Containers;
			int Depth(int s) => containers[s].Count;

			bool[] flipShell = new bool[numShells];
			int flippedShells = 0;
			for (int s = 0; s < numShells; s++)
			{
				Classified? c = classified[s];
				if (c == null)
				{
					continue;
				}

				int target = Depth(s) % 2 == 0 ? 1 : -1;
				if (c.Sign == target)
				{
					continue;
				}

				// An outward-wound shell at odd depth is ambiguous: it may be a
				// cavity that was saved inside-out, or a nested solid that simply
				// winds its interior to 2 — both read the same. Rewinding it can only
				// *remove* material that the mesh currently has, so we only do it when
				// the shell sits under demonstrably inside-out geometry: a container
				// that is itself inverted where it should be solid (an even-depth
				// shell with sign -1), i.e. the whole nesting stack arrived mirrored.
				// Without that evidence we leave the nested solid alone — Thingi10K
				// #61459 is a correctly wound body with an inner shell, and flipping
				// it carved away 80% of the model.
				if (c.Sign == 1)
				{
					bool mirroredStack = containers[s].Any(
						o => classified[o] != null && classified[o]!.Sign == -1 && Depth(o) % 2 == 0);
					if (!mirroredStack)
					{
						continue;
					}
				}

				flipShell[s] = true;
				flippedShells++;
			}

			bool[] flip = new bool[shellOf.Length];
			for (int t = 0; t < shellOf.Length; t++)
			{
				flip[t] = flipShell[shellOf[t]];
			}

			return new RepairPlan(flip, numShells, flippedShells);
		}

		/// <summary>
		/// Rewind the flagged triangles of <paramref name="imp"/> in place: (v0, v1, v2)
		/// becomes (v0, v2, v1), with halfedge pairing remapped rather than rebuilt.
		/// </summary>
		/// <remarks>
		/// Sound because <paramref name="flip"/> comes from a per-shell decision: a paired
		/// edge joins two triangles that share an exact-position edge, hence the same shell,
		/// hence the same flag — so every pair is either reversed on both sides or untouched,
		/// never mixed. Works identically for soup impls (unpaired halfedges stay -1).
		/// </remarks>
		/// <param name="imp">The impl to rewind.</param>
		/// <param name="flip">One flag per triangle.</param>
		public static void ApplyFlips(ManifoldImpl imp, IReadOnlyList<bool> flip)
		{
			ArgumentNullException.ThrowIfNull(imp);
			ArgumentNullException.ThrowIfNull(flip);
			System.Diagnostics.Debug.Assert(imp.NumTri() == flip.Count, "one flag per triangle");

			// Reversing (v0, v1, v2) to (v0, v2, v1) turns each directed edge into
			// its reverse, relocated within the triangle: e0: v0→v1 reappears as
			// slot 2 (v1→v0), e1: v1→v2 as slot 1 (v2→v1), e2: v2→v0 as slot 0
			// (v0→v2). The prop_vert follows its start corner.
			int[] slotOfEdge = { 2, 1, 0 };
			int NewSlot(int he)
			{
				int t = he / 3;
				int e = he % 3;
				return flip[t] ? (3 * t) + slotOfEdge[e] : he;
			}

			List<Halfedge> old = new List<Halfedge>(imp.Halfedge);
			for (int t = 0; t < flip.Count; t++)
			{
				if (!flip[t])
				{
					continue;
				}

				Halfedge e0 = old[3 * t];
				Halfedge e1 = old[(3 * t) + 1];
				Halfedge e2 = old[(3 * t) + 2];
				imp.Halfedge[3 * t] = new Halfedge(e0.StartVert, e2.StartVert, -1, e0.PropVert);
				imp.Halfedge[(3 * t) + 1] = new Halfedge(e2.StartVert, e1.StartVert, -1, e2.PropVert);
				imp.Halfedge[(3 * t) + 2] = new Halfedge(e1.StartVert, e0.StartVert, -1, e1.PropVert);
			}

			// Second pass: remap every pairing through the slot relocation (including
			// pairs between two untouched triangles, whose slots don't move).
			for (int i = 0; i < old.Count; i++)
			{
				int p = old[i].PairedHalfedge;
				Halfedge he = imp.Halfedge[NewSlot(i)];
				he.PairedHalfedge = p < 0 ? p : NewSlot(p);
				imp.Halfedge[NewSlot(i)] = he;
			}

			if (imp.IsSoup)
			{
				// Soup impls carry per-face normals only (see soup::soupify).
				for (int t = 0; t < flip.Count; t++)
				{
					if (flip[t] && t < imp.FaceNormal.Count)
					{
						imp.FaceNormal[t] = -imp.FaceNormal[t];
					}
				}

				imp.VertNormal.Clear();
			}
			else
			{
				// Recompute normals and coplanar grouping from the rewound topology.
				imp.SetNormalsAndCoplanar();
			}
		}

		/// <summary>
		/// Exact-position weld key with -0.0 normalized (same identity <c>soup.rs</c> uses,
		/// so "connected" means the same thing in both places).
		/// </summary>
		/// <param name="v">The position.</param>
		/// <returns>The three coordinates' normalized bit patterns.</returns>
		private static (ulong X, ulong Y, ulong Z) PosKey(Vec3 v)
		{
			static ulong Norm(double x)
			{
				return BitConverter.DoubleToUInt64Bits(x == 0.0 ? 0.0 : x);
			}

			return (Norm(v.X), Norm(v.Y), Norm(v.Z));
		}

		/// <summary>
		/// Connected shells of a triangle soup: triangles are joined when they share an
		/// undirected edge on exactly-welded vertex positions.
		/// </summary>
		/// <param name="tris">The triangle soup.</param>
		/// <returns>The dense 0-based shell id per triangle, and the shell count.</returns>
		private static (int[] Shell, int Count) ConnectedShells(IReadOnlyList<Vec3[]> tris)
		{
			SortedDictionary<(ulong X, ulong Y, ulong Z), uint> weld
				= new SortedDictionary<(ulong X, ulong Y, ulong Z), uint>();
			uint next = 0;
			uint VertId(Vec3 v)
			{
				(ulong X, ulong Y, ulong Z) key = PosKey(v);
				if (weld.TryGetValue(key, out uint hit))
				{
					return hit;
				}

				uint id = next;
				next++;
				weld.Add(key, id);
				return id;
			}

			uint[][] triVerts = new uint[tris.Count][];
			for (int t = 0; t < tris.Count; t++)
			{
				Vec3[] tri = tris[t];
				triVerts[t] = new[] { VertId(tri[0]), VertId(tri[1]), VertId(tri[2]) };
			}

			DisjointSets ds = new DisjointSets((uint)Math.Max(tris.Count, 1));
			List<((uint A, uint B) Edge, uint Tri)> byEdge
				= new List<((uint A, uint B) Edge, uint Tri)>(3 * tris.Count);
			for (int t = 0; t < tris.Count; t++)
			{
				uint[] tv = triVerts[t];
				for (int e = 0; e < 3; e++)
				{
					uint a = tv[e];
					uint b = tv[(e + 1) % 3];
					if (a != b)
					{
						byEdge.Add(((Math.Min(a, b), Math.Max(a, b)), (uint)t));
					}
				}
			}

			// Rust `sort_unstable` on ((u32, u32), u32) — every entry carries its own
			// triangle index, so ties across the whole tuple are impossible and the
			// unstable sort is deterministic.
			byEdge.Sort();
			for (int w = 0; w + 1 < byEdge.Count; w++)
			{
				if (byEdge[w].Edge == byEdge[w + 1].Edge)
				{
					ds.Unite(byEdge[w].Tri, byEdge[w + 1].Tri);
				}
			}

			SortedDictionary<uint, int> remap = new SortedDictionary<uint, int>();
			int[] shell = new int[tris.Count];
			for (int t = 0; t < tris.Count; t++)
			{
				uint root = ds.Find((uint)t);
				if (!remap.TryGetValue(root, out int id))
				{
					id = remap.Count;
					remap.Add(root, id);
				}

				shell[t] = id;
			}

			return (shell, remap.Count);
		}

		/// <summary>
		/// Current orientation of one shell, from exact winding just off a face.
		/// </summary>
		/// <remarks>
		/// Tries up to <see cref="MaxCandidateFaces"/> faces: a face inside a coincident
		/// stack (doubled sheet, fold) yields winding pairs other than the two clean
		/// signatures and is skipped. Null means no sampled face gave a clean answer — the
		/// shell is left untouched.
		/// </remarks>
		/// <param name="geom">The shell's geometry tables.</param>
		/// <returns>The classification, or null.</returns>
		private static Classified? ClassifyShell(ShellGeom geom)
		{
			int limit = Math.Min(geom.TrisR.Count, MaxCandidateFaces);
			for (int i = 0; i < limit; i++)
			{
				R3[] t = geom.TrisR[i];
				R3 n = Predicates.TriNormalR(t[0], t[1], t[2]);
				if (n.IsZero())
				{
					continue; // exactly degenerate: no sides to speak of
				}

				R3 probe = RayShoot.PieceCentroid(new[] { t[0], t[1], t[2] });
				int wOut = RayShoot.WindingOffSurface(probe, n, geom.TrisR, geom.TrisF64, geom.Boxes);
				R3 neg = new R3(-n.X, -n.Y, -n.Z);
				int wIn = RayShoot.WindingOffSurface(probe, neg, geom.TrisR, geom.TrisF64, geom.Boxes);

				// Normal side empty, anti side solid: wound outward.
				if (wOut == 0 && wIn == 1)
				{
					return new Classified(1, probe, n);
				}

				// Normal side inverted-solid, anti side empty: wound inward.
				if (wOut == -1 && wIn == 0)
				{
					return new Classified(-1, probe, neg);
				}

				// Coincident stack or multiple wrap at this face: try another.
			}

			return null;
		}

		/// <summary>
		/// Shell decomposition plus the two exact verdicts every consumer here needs: each
		/// shell's current orientation and the set of other shells containing it.
		/// </summary>
		/// <param name="tris">The triangle soup.</param>
		/// <returns>The analysis.</returns>
		private static Analysis Analyze(IReadOnlyList<Vec3[]> tris)
		{
			ArgumentNullException.ThrowIfNull(tris);
			(int[] shellOf, int numShells) = ConnectedShells(tris);
			List<int>[] members = new List<int>[numShells];
			for (int s = 0; s < numShells; s++)
			{
				members[s] = new List<int>();
			}

			for (int t = 0; t < shellOf.Length; t++)
			{
				members[shellOf[t]].Add(t);
			}

			ShellGeom[] geoms = new ShellGeom[numShells];
			Classified?[] classified = new Classified?[numShells];
			for (int s = 0; s < numShells; s++)
			{
				geoms[s] = new ShellGeom(tris, members[s]);
				classified[s] = ClassifyShell(geoms[s]);
			}

			// Containment: which *other* shells hold the point just off this shell's
			// exterior. Winding != 0 rather than >= 1, so an inverted (not-yet-
			// repaired) container still counts as containing.
			List<int>[] containers = new List<int>[numShells];
			for (int s = 0; s < numShells; s++)
			{
				containers[s] = new List<int>();
			}

			for (int s = 0; s < numShells; s++)
			{
				Classified? c = classified[s];
				if (c == null)
				{
					continue;
				}

				for (int o = 0; o < numShells; o++)
				{
					if (o == s)
					{
						continue;
					}

					ShellGeom g = geoms[o];
					if (RayShoot.WindingOffSurface(c.Probe, c.Exterior, g.TrisR, g.TrisF64, g.Boxes) != 0)
					{
						containers[s].Add(o);
					}
				}
			}

			return new Analysis(shellOf, numShells, classified, containers);
		}

		/// <summary>Per-shell exact/f64/bbox triangle tables for the winding queries.</summary>
		private sealed class ShellGeom
		{
			/// <summary>The shell's triangles, rounded to f64.</summary>
			public readonly List<Vec3[]> TrisF64;

			/// <summary>The same triangles, exact.</summary>
			public readonly List<R3[]> TrisR;

			/// <summary>One bounding box per triangle.</summary>
			public readonly List<Box> Boxes;

			/// <summary>Builds the tables for one shell's member triangles.</summary>
			/// <param name="tris">The whole soup.</param>
			/// <param name="members">Indices of this shell's triangles.</param>
			public ShellGeom(IReadOnlyList<Vec3[]> tris, IReadOnlyList<int> members)
			{
				this.TrisF64 = new List<Vec3[]>(members.Count);
				foreach (int t in members)
				{
					this.TrisF64.Add(tris[t]);
				}

				this.TrisR = new List<R3[]>(this.TrisF64.Count);
				this.Boxes = new List<Box>(this.TrisF64.Count);
				foreach (Vec3[] t in this.TrisF64)
				{
					this.TrisR.Add(new[] { R3.FromVec3(t[0]), R3.FromVec3(t[1]), R3.FromVec3(t[2]) });
					Box b = Box.FromPoints(t[0], t[1]);
					b.UnionPoint(t[2]);
					this.Boxes.Add(b);
				}
			}
		}

		/// <summary>A shell whose orientation the exact queries could pin down.</summary>
		private sealed class Classified
		{
			/// <summary>+1 = currently outward-wound, -1 = currently inverted.</summary>
			public readonly int Sign;

			/// <summary>A point on the shell's surface (a face centroid) …</summary>
			public readonly R3 Probe;

			/// <summary>
			/// … and the direction off that face toward the shell's geometric exterior
			/// (where its own winding is 0).
			/// </summary>
			public readonly R3 Exterior;

			/// <summary>Creates a classification.</summary>
			/// <param name="sign">The current orientation.</param>
			/// <param name="probe">The surface probe point.</param>
			/// <param name="exterior">The outward direction at it.</param>
			public Classified(int sign, R3 probe, R3 exterior)
			{
				this.Sign = sign;
				this.Probe = probe;
				this.Exterior = exterior;
			}
		}

		/// <summary>The shell decomposition and both exact verdicts.</summary>
		private sealed class Analysis
		{
			/// <summary>The shell id of every triangle.</summary>
			public readonly int[] ShellOf;

			/// <summary>How many shells there are.</summary>
			public readonly int NumShells;

			/// <summary>Each shell's orientation, or null when it could not be pinned down.</summary>
			public readonly Classified?[] Classified;

			/// <summary>The other shells containing each shell.</summary>
			public readonly List<int>[] Containers;

			/// <summary>Creates an analysis.</summary>
			/// <param name="shellOf">Per-triangle shell ids.</param>
			/// <param name="numShells">The shell count.</param>
			/// <param name="classified">Per-shell orientation verdicts.</param>
			/// <param name="containers">Per-shell containment lists.</param>
			public Analysis(
				int[] shellOf,
				int numShells,
				Classified?[] classified,
				List<int>[] containers)
			{
				this.ShellOf = shellOf;
				this.NumShells = numShells;
				this.Classified = classified;
				this.Containers = containers;
			}
		}
	}
}
