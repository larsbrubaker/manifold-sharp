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

// EdgeOp.Swap.cs — RecursiveEdgeSwap and its inner DoSwap. The module header
// and the file-split map live in EdgeOp.cs.
//
// Three things in here are load-bearing and have each been got wrong before:
//
//   1. `tag` is passed BY REFERENCE (`ref int`, C++ `int&`). The facing-
//      degenerates branch increments it and the increment must reach the
//      caller's stack-drain loop; taking it by value was a real bug.
//   2. The `edge` parameter and the swap stack are `int` (i32), not an index
//      type, and the `edge < 0` guard at the top is what makes that safe — the
//      normal-swap path pushes paired-halfedge values that are legitimately -1.
//   3. DoSwap's duplicate-edge tail (FormLoop + RemoveIfFolded) is not
//      optional; without it the swap can leave a non-manifold edge behind.
//
// Note also which triangle's verts get projected: RecursiveEdgeSwap projects
// *tri0's* verts through the neighbour's projection, while SwapDegenerates in
// EdgeOp.Simplify.cs projects the PAIR triangle's. That difference is in the
// C++ and is deliberate at both sites.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <content>
	/// The edge-swap half of edge_op.rs.
	/// </content>
	public static partial class EdgeOp
	{
		// -----------------------------------------------------------------------
		// RecursiveEdgeSwap
		// -----------------------------------------------------------------------

		/// <summary>
		/// Swaps the long edge of a degenerate triangle with its neighbor, propagating the
		/// swap recursively as needed.
		/// </summary>
		/// <param name="mesh">The mesh to edit.</param>
		/// <param name="edge">The halfedge to consider swapping; negative values are skipped.</param>
		/// <param name="tag">
		/// The visit tag, incremented in place by the facing-degenerates branch — see the
		/// remarks.
		/// </param>
		/// <param name="visited">Per-halfedge tag of the last visit, sized to the arena.</param>
		/// <param name="edgeSwapStack">The caller's worklist; this function pushes onto it.</param>
		/// <param name="scratch">Scratch space handed on to <see cref="CollapseEdge"/>.</param>
		/// <remarks>
		/// C++ takes <c>int&amp; tag</c> and increments it in the facing-degenerates branch;
		/// the increment must persist into the caller's stack-drain loop so edges visited
		/// before a collapse can be re-processed after it.
		/// </remarks>
		public static void RecursiveEdgeSwap(
			ManifoldImpl mesh,
			int edge,
			ref int tag,
			List<int> visited,
			List<int> edgeSwapStack,
			List<int> scratch)
		{
			// C++ guards `if (edge < 0) return;` first — the normal-swap path pushes
			// paired-halfedge values that can legitimately be -1, and the stack stores
			// them as-is (int). Keeping the stack/param as i32 preserves that sentinel
			// instead of wrapping -1 to usize::MAX and indexing out of bounds.
			if (edge < 0)
			{
				return;
			}

			if (mesh.Halfedge[edge].PairedHalfedge < 0)
			{
				return;
			}

			int pair = mesh.Halfedge[edge].PairedHalfedge;

			// Avoid infinite recursion via visited tag. Rust `visited.get(i)` returns None
			// past the end, which never compares equal, so an out-of-range index does not
			// short-circuit the swap.
			if (edge < visited.Count && visited[edge] == tag
				&& pair < visited.Count && visited[pair] == tag)
			{
				return;
			}

			int[] tri0edge = TriOf(edge);
			int[] tri1edge = TriOf(pair);

			if (edge / 3 >= mesh.FaceNormal.Count || pair / 3 >= mesh.FaceNormal.Count)
			{
				return;
			}

			Proj2x3 proj0 = FaceOp.GetAxisAlignedProjection(mesh.FaceNormal[edge / 3]);
			Vec2[] v = new Vec2[4];
			for (int i = 0; i < 4; i++)
			{
				v[i] = new Vec2(0.0, 0.0);
			}

			for (int i = 0; i < 3; i++)
			{
				// The Rust guard is `sv as usize < vert_pos.len()`, which also rejects the
				// -1 sentinel; a signed `<` would accept it and index out of bounds, so it
				// is transcribed unsigned (the pattern the FaceOp.cs header names).
				int sv = mesh.Halfedge[tri0edge[i]].StartVert;
				if ((uint)sv < (uint)mesh.VertPos.Count)
				{
					v[i] = proj0.Apply(mesh.VertPos[sv]);
				}
			}

			// Only operate on the long edge of a degenerate triangle
			if (Polygon.Ccw(v[0], v[1], v[2], mesh.Tolerance) > 0 || !Is01Longest(v[0], v[1], v[2]))
			{
				return;
			}

			// Switch to neighbor's projection. Note this re-projects TRI0's verts (the
			// same start verts as above) through tri1's projection — SwapDegenerates does
			// the corresponding step on the pair triangle's verts instead, and the two
			// sites differ on purpose.
			Proj2x3 proj1 = FaceOp.GetAxisAlignedProjection(mesh.FaceNormal[pair / 3]);
			for (int i = 0; i < 3; i++)
			{
				int sv = mesh.Halfedge[tri0edge[i]].StartVert;
				if ((uint)sv < (uint)mesh.VertPos.Count)
				{
					v[i] = proj1.Apply(mesh.VertPos[sv]);
				}
			}

			int sv3 = mesh.Halfedge[tri1edge[2]].StartVert;
			if ((uint)sv3 < (uint)mesh.VertPos.Count)
			{
				v[3] = proj1.Apply(mesh.VertPos[sv3]);
			}

			// Swap edge logic
			void DoSwap()
			{
				int v0 = mesh.Halfedge[tri0edge[2]].StartVert;
				int v1 = mesh.Halfedge[tri1edge[2]].StartVert;
				SetStartVert(mesh.Halfedge, tri0edge[0], v1);
				SetEndVert(mesh.Halfedge, tri0edge[2], v1);
				SetStartVert(mesh.Halfedge, tri1edge[0], v0);
				SetEndVert(mesh.Halfedge, tri1edge[2], v0);
				int tri1e2Paired = mesh.Halfedge[tri1edge[2]].PairedHalfedge;
				int tri0e2Paired = mesh.Halfedge[tri0edge[2]].PairedHalfedge;
				PairUp(mesh.Halfedge, tri0edge[0], tri1e2Paired);
				PairUp(mesh.Halfedge, tri1edge[0], tri0e2Paired);
				PairUp(mesh.Halfedge, tri0edge[2], tri1edge[2]);

				int tri0 = tri0edge[0] / 3;
				int tri1 = tri1edge[0] / 3;
				if (tri1 < mesh.FaceNormal.Count && tri0 < mesh.FaceNormal.Count)
				{
					mesh.FaceNormal[tri0] = mesh.FaceNormal[tri1];
				}

				if (tri1 < mesh.MeshRelation.TriRef.Count && tri0 < mesh.MeshRelation.TriRef.Count)
				{
					mesh.MeshRelation.TriRef[tri0] = mesh.MeshRelation.TriRef[tri1];
				}

				// Update properties if applicable. Mirrors the C++ assignment order:
				// tri0's props are copied from tri1's ORIGINAL props first, then the
				// interpolated new prop vert replaces tri1edge[0]/tri0edge[2].
				if (mesh.Properties.Count != 0)
				{
					int numProp = mesh.NumProp;
					double l01 = Math.Sqrt(LengthSquared(v[1] - v[0]));
					double l02 = Math.Sqrt(LengthSquared(v[2] - v[0]));

					// Rust `f64::clamp`, transcribed as a pair of comparisons so that a NaN
					// stays NaN. LinalgFunctions.Clamp is max-then-min and would turn a NaN
					// into `lo`, so it is not a substitute (the trap FaceOp.cs documents).
					double a = l02 / l01;
					a = a < 0.0 ? 0.0 : (a > 1.0 ? 1.0 : a);

					SetPropVert(mesh.Halfedge, tri0edge[1], mesh.Halfedge[tri1edge[0]].PropVert);
					SetPropVert(mesh.Halfedge, tri0edge[0], mesh.Halfedge[tri1edge[2]].PropVert);
					SetPropVert(mesh.Halfedge, tri0edge[2], mesh.Halfedge[tri1edge[2]].PropVert);
					int newProp = mesh.Properties.Count / numProp;
					int idx0 = mesh.Halfedge[tri1edge[0]].PropVert;
					int idx1 = mesh.Halfedge[tri1edge[1]].PropVert;
					for (int p = 0; p < numProp; p++)
					{
						double val = (a * mesh.Properties[(numProp * idx0) + p])
							+ ((1.0 - a) * mesh.Properties[(numProp * idx1) + p]);
						mesh.Properties.Add(val);
					}

					SetPropVert(mesh.Halfedge, tri1edge[0], newProp);
					SetPropVert(mesh.Halfedge, tri0edge[2], newProp);
				}

				// If the new (swapped) edge already exists elsewhere, the swap has
				// created a duplicate edge. Duplicate the verts and split the mesh the
				// other way so the result stays manifold. Omitting this (as the Rust
				// port previously did) leaves a non-manifold edge that later trips
				// collapse_edge/update_vert into walking an unpaired (-1) halfedge.
				// `current` is kept as i32 to mirror the C++ `int` orbit exactly.
				int endVert = mesh.Halfedge[tri1edge[1]].EndVert;
				int current = mesh.Halfedge[tri1edge[0]].PairedHalfedge;
				while (current != tri0edge[1])
				{
					current = Types.NextHalfedge(current);
					if (mesh.Halfedge[current].EndVert == endVert)
					{
						FormLoop(mesh, tri0edge[2], current);
						RemoveIfFolded(mesh, tri0edge[2]);
						return;
					}

					current = mesh.Halfedge[current].PairedHalfedge;
				}
			}

			if (Polygon.Ccw(v[1], v[0], v[3], mesh.Tolerance) <= 0)
			{
				if (!Is01Longest(v[1], v[0], v[3]))
				{
					return;
				}

				// Two facing long-edge degenerates can swap
				DoSwap();
				Vec2 e23 = v[3] - v[2];
				if (Dot(e23, e23) < mesh.Tolerance * mesh.Tolerance)
				{
					// Also collapse the resulting short edge. C++ bumps the tag here so
					// previously-visited edges become re-processable after the mesh
					// changed under them.
					tag += 1;
					CollapseEdge(mesh, tri0edge[2], scratch, -1.0, 0);
					scratch.Clear();
				}
				else
				{
					if (edge < visited.Count)
					{
						visited[edge] = tag;
					}

					if (pair < visited.Count)
					{
						visited[pair] = tag;
					}

					foreach (int e in new int[] { tri1edge[1], tri1edge[0], tri0edge[1], tri0edge[0] })
					{
						edgeSwapStack.Add(e);
					}
				}
			}
			else if (Polygon.Ccw(v[0], v[3], v[2], mesh.Tolerance) <= 0
				|| Polygon.Ccw(v[1], v[2], v[3], mesh.Tolerance) <= 0)
			{
				return;
			}
			else
			{
				// Normal swap path
				DoSwap();
				if (edge < visited.Count)
				{
					visited[edge] = tag;
				}

				if (pair < visited.Count)
				{
					visited[pair] = tag;
				}

				// These pair lookups can be -1; C++ pushes them as-is and relies on the
				// `edge < 0` guard at the top of the next call to skip them.
				int p1 = mesh.Halfedge[tri1edge[0]].PairedHalfedge;
				int p2 = mesh.Halfedge[tri0edge[1]].PairedHalfedge;
				edgeSwapStack.Add(p1);
				edgeSwapStack.Add(p2);
			}
		}
	}
}
