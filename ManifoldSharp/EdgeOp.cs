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

// edge_op.rs — Phase 7b: Edge collapse, degenerate removal, topology cleanup
//
// Ports src/edge_op.cpp from the Manifold C++ library.
// All algorithms are sequential (rayon deferred to later).
//
// ── C# port notes ────────────────────────────────────────────────────────────
// The Rust module's free functions land on a static class named for the module
// (`EdgeOp`), per the naming rule in CLAUDE.md. edge_op.rs declares no types and
// has no `impl ManifoldImpl` block, so nothing here is a partial of that class;
// every entry point takes the mesh as its first argument, exactly as the Rust
// does.
//
// ── File split, and why the coupling survives it ─────────────────────────────
// edge_op.rs is 1,144 lines and is a documented exception to the 800-line cap
// (docs/PORTING_PLAN.md) *by design*: its algorithms are mutually recursive through
// the mesh arena and are meant to be read together. The C# expansion cannot fit
// one file, so it lands as five, all continuing one `static partial class
// EdgeOp`. The split is by call depth, not by concern, and the cycle it cuts
// through is:
//
//   CollapseEdge ──> FormLoop ──> RemoveIfFolded
//        ^                ^             ^
//        └── RecursiveEdgeSwap ─────────┘   (and back into CollapseEdge)
//
//   EdgeOp.cs           this file — TriOf, Is01Longest, the halfedge field
//                       writers, PairUp, UpdateVert, FormLoop, CollapseTri,
//                       RemoveIfFolded: the primitives every other file calls
//   EdgeOp.Collapse.cs  CollapseEdge
//   EdgeOp.Swap.cs      RecursiveEdgeSwap
//   EdgeOp.Dedupe.cs    SplitPinchedVerts, DedupeEdge, DedupeEdges
//   EdgeOp.Simplify.cs  CleanupTopology, SimplifyTopology, RemoveDegenerates,
//                       CollapseShortEdges, CollapseColinearEdges,
//                       SwapDegenerates — the driver layer
//
// Reading order is that list; a change to any of the first three has to be
// checked against the other two, because they edit the same halfedge arena
// mid-walk and each one's guards assume the shape the others leave behind.
//
// ── Orbit walks that look alike and are not ──────────────────────────────────
// Fourteen sites here walk the halfedges around a vertex: six outside
// EdgeOp.Dedupe.cs (UpdateVert below; CollapseEdge's three — the inversion
// check, the endVert collection and the startVert relabel; DoSwap's
// duplicate-edge scan; CollapseColinearEdges' FlagEdge scan) and eight inside
// EdgeOp.Dedupe.cs, which enumerates its own. None of them calls
// ManifoldImpl.ForVert, for the reason FaceOp.cs already documents: ForVert has
// no `paired < 0` guard and walks off the arena on any mesh with an unpaired
// halfedge — and edge_op runs *on* meshes that have those. Each site's guards
// differ (some break, some return, some bound the step count), so every walk is
// transcribed inline from its own Rust, and the differences below are
// deliberate, not drift.
//
// ── Field writes go through helpers, not through a span ──────────────────────
// Rust's `mesh.halfedge[i].field = x` has no direct C# spelling: writing through
// a List<T> indexer is CS1612. Elsewhere in the port that becomes
// CollectionsMarshal.AsSpan, which is NOT safe here — DedupeEdge appends to
// mesh.Halfedge while it is walking it, and any span taken before that append
// points at the old backing array. The read-modify-store helpers at the bottom
// of this file are used instead, at every site.

using System.Diagnostics;

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// Edge collapse, degenerate removal and topology cleanup — the free functions of
	/// <c>edge_op.rs</c>.
	/// </summary>
	public static partial class EdgeOp
	{
		// -----------------------------------------------------------------------
		// Private helpers (mirrors anonymous-namespace functions in edge_op.cpp)
		// -----------------------------------------------------------------------

		/// <summary>
		/// Returns the 3 halfedge indices forming the triangle containing <c>edge</c>.
		/// tri_of(e) = [e, next(e), next(next(e))]
		/// </summary>
		private static int[] TriOf(int edge)
		{
			int e1 = Types.NextHalfedge(edge);
			int e2 = Types.NextHalfedge(e1);
			return new int[] { edge, e1, e2 };
		}

		/// <summary>Returns true if edge 0–1 is longer than 0–2 and 1–2.</summary>
		private static bool Is01Longest(Vec2 v0, Vec2 v1, Vec2 v2)
		{
			Vec2[] e = new Vec2[] { v1 - v0, v2 - v1, v0 - v2 };
			double[] l = new double[] { Dot(e[0], e[0]), Dot(e[1], e[1]), Dot(e[2], e[2]) };
			return l[0] > l[1] && l[0] > l[2];
		}

		/// <summary>
		/// Rust <c>Vec3::new(f64::NAN, f64::NAN, f64::NAN)</c>, the "this vertex is gone"
		/// marker written by <see cref="CollapseEdge"/> and <see cref="RemoveIfFolded"/>.
		/// </summary>
		/// <remarks>
		/// The NaN must be the positive quiet one Rust's <c>f64::NAN</c> is; C#'s
		/// <see cref="double.NaN"/> has the sign bit set and would diverge in any bit-based
		/// compare downstream. A method rather than a <c>static readonly</c> field because
		/// the Linalg folder header forbids holding a ref-indexable vector type in readonly
		/// storage.
		/// </remarks>
		private static Vec3 NanVertPos()
		{
			double nan = DeterministicMath.PositiveQuietNaN;
			return new Vec3(nan, nan, nan);
		}

		// -----------------------------------------------------------------------
		// pair_up / update_vert / form_loop / collapse_tri / remove_if_folded
		// -----------------------------------------------------------------------

		/// <summary>Pairs two halfedges bidirectionally.</summary>
		/// <param name="halfedge">The halfedge arena to edit.</param>
		/// <param name="e0">The first halfedge index.</param>
		/// <param name="e1">The second halfedge index.</param>
		public static void PairUp(List<Halfedge> halfedge, int e0, int e1)
		{
			SetPairedHalfedge(halfedge, e0, e1);
			SetPairedHalfedge(halfedge, e1, e0);
		}

		/// <summary>
		/// Traverses CW from <paramref name="startEdge"/> to <paramref name="endEdge"/>
		/// (exclusive) around <c>startEdge.endVert</c>, repointing all traversed halfedges to
		/// <paramref name="vert"/>.
		/// </summary>
		/// <param name="mesh">The mesh to edit.</param>
		/// <param name="vert">The vertex index the traversed halfedges are repointed to.</param>
		/// <param name="startEdge">The halfedge to start at.</param>
		/// <param name="endEdge">The halfedge to stop before.</param>
		public static void UpdateVert(ManifoldImpl mesh, int vert, int startEdge, int endEdge)
		{
			// C++ UpdateVert: start_edge == end_edge is a legitimate no-op; the
			// infinite-loop check is against *start_edge* after stepping (i.e. we
			// wrapped all the way around without hitting end_edge).
			int current = startEdge;
			while (current != endEdge)
			{
				SetEndVert(mesh.Halfedge, current, vert);
				int next = Types.NextHalfedge(current);
				SetStartVert(mesh.Halfedge, next, vert);

				// An unpaired -1 here is a caller bug in both ports: the Rust widens it to
				// usize::MAX, which never equals end_edge and then panics on the next
				// index; a C# -1 never equals end_edge either and throws on the same step.
				current = mesh.Halfedge[next].PairedHalfedge;
				Debug.Assert(current != startEdge, "infinite loop in update_vert!");
			}
		}

		/// <summary>
		/// When an edge collapse would create a non-manifold edge, instead duplicate both
		/// endpoints and re-attach the manifold the other way across this edge.
		/// </summary>
		/// <param name="mesh">The mesh to edit.</param>
		/// <param name="current">The halfedge the loop is formed from.</param>
		/// <param name="end">The halfedge the loop is closed at.</param>
		public static void FormLoop(ManifoldImpl mesh, int current, int end)
		{
			int startVert = mesh.VertPos.Count;
			mesh.VertPos.Add(mesh.VertPos[mesh.Halfedge[current].StartVert]);
			int endVert = mesh.VertPos.Count;
			mesh.VertPos.Add(mesh.VertPos[mesh.Halfedge[current].EndVert]);

			int oldMatch = mesh.Halfedge[current].PairedHalfedge;
			int newMatch = mesh.Halfedge[end].PairedHalfedge;

			UpdateVert(mesh, startVert, oldMatch, newMatch);
			UpdateVert(mesh, endVert, end, current);

			SetPairedHalfedge(mesh.Halfedge, current, newMatch);
			SetPairedHalfedge(mesh.Halfedge, newMatch, current);
			SetPairedHalfedge(mesh.Halfedge, end, oldMatch);
			SetPairedHalfedge(mesh.Halfedge, oldMatch, end);

			RemoveIfFolded(mesh, end);
		}

		/// <summary>
		/// Collapse the two non-<c>edge</c>-index halfedges of a triangle by linking their
		/// paired partners directly. Marks the three halfedges as removed.
		/// </summary>
		/// <param name="halfedge">The halfedge arena to edit.</param>
		/// <param name="triEdge">The triangle's three halfedge indices, from <see cref="TriOf"/>.</param>
		public static void CollapseTri(List<Halfedge> halfedge, int[] triEdge)
		{
			if (halfedge[triEdge[1]].PairedHalfedge == -1)
			{
				return;
			}

			int pair1 = halfedge[triEdge[1]].PairedHalfedge;
			int pair2 = halfedge[triEdge[2]].PairedHalfedge;
			SetPairedHalfedge(halfedge, pair1, pair2);
			SetPairedHalfedge(halfedge, pair2, pair1);
			for (int i = 0; i < 3; i++)
			{
				// The prop vert is read out and written back: the Rust rebuilds the whole
				// Halfedge and carries prop_vert across, unlike RemoveIfFolded below, which
				// clears it to -1. That asymmetry is the C++'s.
				int propVert = halfedge[triEdge[i]].PropVert;
				halfedge[triEdge[i]] = new Halfedge(-1, -1, -1, propVert);
			}
		}

		/// <summary>
		/// Removes a pair of triangles that have folded onto each other (degenerate), by
		/// patching their external paired halfedges together.
		/// </summary>
		/// <param name="mesh">The mesh to edit.</param>
		/// <param name="edge">A halfedge of the first of the two folded triangles.</param>
		public static void RemoveIfFolded(ManifoldImpl mesh, int edge)
		{
			int[] tri0edge = TriOf(edge);
			int pair0 = mesh.Halfedge[edge].PairedHalfedge;
			if (pair0 < 0)
			{
				return;
			}

			int[] tri1edge = TriOf(pair0);
			if (mesh.Halfedge[tri0edge[1]].PairedHalfedge == -1)
			{
				return;
			}

			if (mesh.Halfedge[tri0edge[1]].EndVert == mesh.Halfedge[tri1edge[1]].EndVert)
			{
				if (mesh.Halfedge[tri0edge[1]].PairedHalfedge == tri1edge[2])
				{
					if (mesh.Halfedge[tri0edge[2]].PairedHalfedge == tri1edge[1])
					{
						// Both outer edges are paired: degenerate two-triangle island — mark all verts NaN
						for (int i = 0; i < 3; i++)
						{
							int sv = mesh.Halfedge[tri0edge[i]].StartVert;
							if (sv >= 0)
							{
								mesh.VertPos[sv] = NanVertPos();
							}
						}
					}
					else
					{
						int sv = mesh.Halfedge[tri0edge[1]].StartVert;
						if (sv >= 0)
						{
							mesh.VertPos[sv] = NanVertPos();
						}
					}
				}
				else if (mesh.Halfedge[tri0edge[2]].PairedHalfedge == tri1edge[1])
				{
					int sv = mesh.Halfedge[tri1edge[1]].StartVert;
					if (sv >= 0)
					{
						mesh.VertPos[sv] = NanVertPos();
					}
				}
				else
				{
					return;
				}

				int p01 = mesh.Halfedge[tri0edge[1]].PairedHalfedge;
				int p02 = mesh.Halfedge[tri0edge[2]].PairedHalfedge;
				int p11 = mesh.Halfedge[tri1edge[1]].PairedHalfedge;
				int p12 = mesh.Halfedge[tri1edge[2]].PairedHalfedge;
				PairUp(mesh.Halfedge, p01, p12);
				PairUp(mesh.Halfedge, p02, p11);
				for (int i = 0; i < 3; i++)
				{
					mesh.Halfedge[tri0edge[i]] = new Halfedge(-1, -1, -1, -1);
					mesh.Halfedge[tri1edge[i]] = new Halfedge(-1, -1, -1, -1);
				}
			}
		}

		// -----------------------------------------------------------------------
		// Halfedge field writers — see the "Field writes go through helpers" note
		// in the file header for why these exist rather than a span.
		// -----------------------------------------------------------------------

		/// <summary>Rust <c>halfedge[index].start_vert = value</c>.</summary>
		private static void SetStartVert(List<Halfedge> halfedge, int index, int value)
		{
			Halfedge h = halfedge[index];
			h.StartVert = value;
			halfedge[index] = h;
		}

		/// <summary>Rust <c>halfedge[index].end_vert = value</c>.</summary>
		private static void SetEndVert(List<Halfedge> halfedge, int index, int value)
		{
			Halfedge h = halfedge[index];
			h.EndVert = value;
			halfedge[index] = h;
		}

		/// <summary>Rust <c>halfedge[index].paired_halfedge = value</c>.</summary>
		private static void SetPairedHalfedge(List<Halfedge> halfedge, int index, int value)
		{
			Halfedge h = halfedge[index];
			h.PairedHalfedge = value;
			halfedge[index] = h;
		}

		/// <summary>Rust <c>halfedge[index].prop_vert = value</c>.</summary>
		private static void SetPropVert(List<Halfedge> halfedge, int index, int value)
		{
			Halfedge h = halfedge[index];
			h.PropVert = value;
			halfedge[index] = h;
		}
	}
}
