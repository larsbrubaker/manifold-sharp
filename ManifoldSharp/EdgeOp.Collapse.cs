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

// EdgeOp.Collapse.cs — CollapseEdge, the largest single function of edge_op.rs.
// The module header, the primitives it calls (UpdateVert, CollapseTri,
// FormLoop, RemoveIfFolded) and the file-split map live in EdgeOp.cs.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <content>
	/// The edge-collapse half of edge_op.rs.
	/// </content>
	public static partial class EdgeOp
	{
		// -----------------------------------------------------------------------
		// CollapseEdge
		// -----------------------------------------------------------------------

		/// <summary>
		/// Collapses the given edge by merging <c>startVert</c> into <c>endVert</c>.
		/// Returns false if the collapse cannot be done safely.
		/// May form loops (topological splits) to avoid non-manifold configurations.
		/// </summary>
		/// <param name="mesh">The mesh to edit.</param>
		/// <param name="edge">The halfedge to collapse.</param>
		/// <param name="scratch">
		/// Caller-owned scratch list of halfedge indices, reused across calls to avoid
		/// reallocating; the callers clear it before each call and this function appends the
		/// endVert orbit to it.
		/// </param>
		/// <param name="tol">
		/// The collapse tolerance; negative means "use <see cref="ManifoldImpl.Epsilon"/>".
		/// </param>
		/// <param name="firstNewVert">
		/// The first vertex index considered new; verts below it may only move by epsilon.
		/// </param>
		/// <returns>True when the collapse was performed.</returns>
		public static bool CollapseEdge(
			ManifoldImpl mesh,
			int edge,
			List<int> scratch,
			double tol,
			int firstNewVert)
		{
			// Per #1671: `tol` defaults to epsilon when negative. In a Boolean
			// (first_new_vert != 0) callers pass tolerance_ so newly-created verts may
			// move up to tolerance, but an edge whose far end is an *old* vert is still
			// limited to epsilon (old verts may only move by epsilon — avoids error
			// stacking).
			tol = tol < 0.0 ? mesh.Epsilon : tol;
			Halfedge toRemove = mesh.Halfedge[edge];
			if (toRemove.PairedHalfedge < 0)
			{
				return false;
			}

			int endVert = toRemove.EndVert;
			int[] tri0edge = TriOf(edge);
			int paired = toRemove.PairedHalfedge;
			int[] tri1edge = TriOf(paired);

			Vec3 pNew = mesh.VertPos[endVert];
			Vec3 pOld = mesh.VertPos[toRemove.StartVert];
			Vec3 delta = pNew - pOld;
			double maxLen = endVert < firstNewVert
				? tol * tol
				: mesh.Epsilon * mesh.Epsilon;
			bool shortEdge = Dot(delta, delta) < maxLen;

			// Orbit startVert: check that collapse does not invert any triangle
			int startOrb = mesh.Halfedge[tri1edge[1]].PairedHalfedge;
			if (startOrb < 0)
			{
				return false;
			}

			int start = startOrb;
			int current;

			if (!shortEdge)
			{
				current = start;
				TriRef refCheck = mesh.MeshRelation.TriRef[paired / 3];
				Vec3 pLast = mesh.VertPos[mesh.Halfedge[tri1edge[1]].EndVert];

				while (current != tri1edge[0])
				{
					current = Types.NextHalfedge(current);
					Vec3 pNext = mesh.VertPos[mesh.Halfedge[current].EndVert];
					int tri = current / 3;
					TriRef refTri = mesh.MeshRelation.TriRef[tri];
					Proj2x3 projection = FaceOp.GetAxisAlignedProjection(mesh.FaceNormal[tri]);

					// Don't collapse if the edge is not redundant (this may have changed
					// due to the collapse of neighbors).
					if (!refTri.SameFace(refCheck))
					{
						TriRef oldRef = refCheck;
						refCheck = mesh.MeshRelation.TriRef[edge / 3];
						if (!refTri.SameFace(refCheck))
						{
							return false;
						}

						// Restrict collapse to colinear edges when the edge separates
						// faces or the edge is sharp.
						if (refTri.MeshId != oldRef.MeshId
							|| refTri.FaceId != oldRef.FaceId
							|| Dot(mesh.FaceNormal[paired / 3], mesh.FaceNormal[tri]) < -0.5)
						{
							Vec2 pPLast = projection.Apply(pLast);
							Vec2 pPOld = projection.Apply(pOld);
							Vec2 pPNew = projection.Apply(pNew);

							// Per #1671: this colinear-restrict check uses `tol` (the
							// triangle-inversion CCW below stays at epsilon).
							if (Polygon.Ccw(pPLast, pPOld, pPNew, tol) != 0)
							{
								return false;
							}
						}
					}

					// Don't collapse edge if it would cause a triangle to invert.
					if (Polygon.Ccw(
							projection.Apply(pNext),
							projection.Apply(pLast),
							projection.Apply(pNew),
							mesh.Epsilon) < 0)
					{
						return false;
					}

					pLast = pNext;
					int pair = mesh.Halfedge[current].PairedHalfedge;
					if (pair < 0)
					{
						break;
					}

					current = pair;
				}
			}

			// Orbit endVert: collect edges that share endVert (for loop detection)
			{
				int cur = mesh.Halfedge[tri0edge[1]].PairedHalfedge;
				while (cur != tri1edge[2])
				{
					cur = Types.NextHalfedge(cur);
					scratch.Add(cur);
					int pair = mesh.Halfedge[cur].PairedHalfedge;
					if (pair < 0)
					{
						break;
					}

					cur = pair;
				}
			}

			// Remove startVert: mark position NaN
			mesh.VertPos[toRemove.StartVert] = NanVertPos();
			CollapseTri(mesh.Halfedge, tri1edge);

			// Orbit startVert and update to endVert; detect and break loops
			current = start;
			while (current != tri0edge[2])
			{
				current = Types.NextHalfedge(current);

				// Update propVert for property meshes. C++ is an else-if chain: when a
				// triangle matches BOTH tri0's and tri1's face group, only tri0's prop
				// is taken.
				if (mesh.NumProp > 0 && mesh.MeshRelation.TriRef.Count != 0)
				{
					int tri = current / 3;
					if (tri < mesh.MeshRelation.TriRef.Count)
					{
						TriRef refTri = mesh.MeshRelation.TriRef[tri];
						if (refTri.SameFace(mesh.MeshRelation.TriRef[edge / 3]))
						{
							int propV = mesh.Halfedge[Types.NextHalfedge(edge)].PropVert;
							SetPropVert(mesh.Halfedge, current, propV);
						}
						else if (refTri.SameFace(mesh.MeshRelation.TriRef[paired / 3]))
						{
							int propV = mesh.Halfedge[paired].PropVert;
							SetPropVert(mesh.Halfedge, current, propV);
						}
					}
				}

				int vert = mesh.Halfedge[current].EndVert;
				int nextPair = mesh.Halfedge[current].PairedHalfedge;
				if (nextPair < 0)
				{
					break;
				}

				int nextEdge = nextPair;

				// Check if this creates a loop (edge to an already-encountered vert)
				bool formedLoop = false;
				for (int k = 0; k < scratch.Count; k++)
				{
					if (vert == mesh.Halfedge[scratch[k]].EndVert)
					{
						FormLoop(mesh, scratch[k], current);
						start = nextEdge;
						scratch.RemoveRange(k, scratch.Count - k);
						current = nextEdge;
						formedLoop = true;
						break;
					}
				}

				if (formedLoop)
				{
					continue;
				}

				current = nextEdge;
			}

			UpdateVert(mesh, endVert, start, tri0edge[2]);
			CollapseTri(mesh.Halfedge, tri0edge);
			RemoveIfFolded(mesh, start);
			return true;
		}
	}
}
