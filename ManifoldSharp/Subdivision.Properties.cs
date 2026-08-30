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

// The `if self.num_prop > 0` tail of Manifold::Impl::Subdivide, split out of
// Subdivision.cs for the 800-line cap. The module header lives there.
//
// This is one straight-line block in subdivision.rs, not a function of its own,
// so everything it reads comes in as a parameter — the split must not change
// which values it sees or the order it writes them. It ends by calling
// CreateHalfedges, exactly as the Rust block does, which is why the caller does
// not.
//
// ── The four property-vertex regions ─────────────────────────────────────────
// The new property array is laid out as, in order:
//   [0, numPropVert)                              the retained property verts,
//                                                 copied across untouched
//   [numPropVert, numPropVert + addedVerts)       interior verts and *forward*
//                                                 edge verts, interpolated from
//                                                 the face's corners
//   [.. + totalEdgeVerts)                         *backward* edge verts, which
//                                                 exist only where the two sides
//                                                 of an edge disagree about
//                                                 their property vertices
// `prop_offset` (numPropVert - numVert) is what converts a vertex index from the
// position-space numbering used above into this array's numbering.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Types;

namespace ManifoldSharp
{
	/// <content>
	/// The property-interpolation tail of <see cref="ManifoldImpl.Subdivide"/>.
	/// </content>
	public sealed partial class ManifoldImpl
	{
		/// <summary>
		/// Interpolates the property vertices across the subdivision and rebuilds the
		/// halfedges from the property and position triangles together.
		/// </summary>
		private void SubdivideProperties(
			List<Barycentric> vertBary,
			List<TmpEdge> edges,
			IVec4[] faceHalfedges,
			Partition[] subTris,
			int[] half2edge,
			int[] edgeAdded,
			int[] edgeOffset,
			int[] triOffset,
			int[] interiorOffset,
			List<IVec3> triVerts,
			int numVert,
			int totalEdgeVerts,
			int totalNewTris)
		{
			int numEdge = edges.Count;
			int numTri = faceHalfedges.Length;
			int numPropVert = this.NumPropVert();
			int addedVerts = this.NumVert() - numVert;
			int propOffset = numPropVert - numVert;
			int numProp = this.NumProp;

			// Allocate new property array
			List<double> prop = new List<double>(numProp * (numPropVert + addedVerts + totalEdgeVerts));
			prop.Resize(numProp * (numPropVert + addedVerts + totalEdgeVerts), 0.0);

			// Copy retained prop verts
			for (int i = 0; i < this.Properties.Count; i++)
			{
				prop[i] = this.Properties[i];
			}

			// Copy interior prop verts and forward edge prop verts
			for (int i = 0; i < addedVerts; i++)
			{
				int vert = numPropVert + i;
				Barycentric bary = vertBary[numVert + i];
				IVec4 halfedges = faceHalfedges[bary.Tri];

				for (int p = 0; p < numProp; p++)
				{
					if (halfedges[3] < 0)
					{
						// triangle
						Vec3 triProp = Vec3.Splat(0.0);
						for (int k = 0; k < 3; k++)
						{
							triProp[k] = this.Properties[(this.Halfedge[(3 * bary.Tri) + k].PropVert * numProp) + p];
						}

						prop[(vert * numProp) + p] = (triProp.X * bary.Uvw.X)
							+ (triProp.Y * bary.Uvw.Y)
							+ (triProp.Z * bary.Uvw.Z);
					}
					else
					{
						// quad
						Vec4 quadProp = Vec4.Splat(0.0);
						for (int k = 0; k < 4; k++)
						{
							quadProp[k] = this.Properties[(this.Halfedge[halfedges[k]].PropVert * numProp) + p];
						}

						prop[(vert * numProp) + p] = (quadProp.X * bary.Uvw.X)
							+ (quadProp.Y * bary.Uvw.Y)
							+ (quadProp.Z * bary.Uvw.Z)
							+ (quadProp.W * bary.Uvw.W);
					}
				}
			}

			// Copy backward edge prop verts
			for (int i = 0; i < numEdge; i++)
			{
				int n = edgeAdded[i];
				int offset = edgeOffset[i] + propOffset + addedVerts;
				double frac = 1.0 / ((double)n + 1.0);
				int halfedgeIdx = this.Halfedge[edges[i].HalfedgeIdx].PairedHalfedge;
				int prop0 = this.Halfedge[halfedgeIdx].PropVert;
				int prop1 = this.Halfedge[NextHalfedge(halfedgeIdx)].PropVert;
				for (int j = 0; j < n; j++)
				{
					for (int p = 0; p < numProp; p++)
					{
						double t = (double)(j + 1) * frac;
						prop[((offset + j) * numProp) + p] = this.Properties[(prop0 * numProp) + p]
							+ ((this.Properties[(prop1 * numProp) + p]
								- this.Properties[(prop0 * numProp) + p])
								* t);
					}
				}
			}

			// Build property triangles
			List<IVec3> triPropOut = new List<IVec3>(totalNewTris);
			triPropOut.Resize(totalNewTris, default(IVec3));
			for (int tri = 0; tri < numTri; tri++)
			{
				IVec4 halfedges = faceHalfedges[tri];
				if (halfedges[0] < 0)
				{
					continue;
				}

				IVec4 tri3 = default(IVec4);
				IVec4 edgeOffs = default(IVec4);
				BVec4 edgeFwd = BVec4.Splat(true);
				for (int i = 0; i < 4; i++)
				{
					if (halfedges[i] < 0)
					{
						tri3[i] = -1;
						continue;
					}

					Halfedge he = this.Halfedge[halfedges[i]];
					tri3[i] = he.PropVert;
					edgeOffs[i] = edgeOffset[half2edge[halfedges[i]]];
					if (!he.IsForward())
					{
						int paired = he.PairedHalfedge;
						if (this.Halfedge[paired].PropVert
								!= this.Halfedge[NextHalfedge(halfedges[i])].PropVert
							|| this.Halfedge[NextHalfedge(paired)].PropVert
								!= he.PropVert)
						{
							// edge doesn't match, point to backward edge propverts
							edgeOffs[i] += addedVerts;
						}
						else
						{
							edgeFwd[i] = false;
						}
					}
				}

				// Add prop_offset to edge offsets
				IVec4 propEdgeOffs = new IVec4(
					edgeOffs[0] + propOffset,
					edgeOffs[1] + propOffset,
					edgeOffs[2] + propOffset,
					edgeOffs[3] + propOffset);

				List<IVec3> newTris = subTris[tri].Reindex(
					tri3,
					propEdgeOffs,
					edgeFwd,
					interiorOffset[tri] + propOffset);

				int start = triOffset[tri];
				for (int j = 0; j < newTris.Count; j++)
				{
					triPropOut[start + j] = newTris[j];
				}
			}

			this.Properties = prop;
			this.CreateHalfedges(triPropOut, triVerts);
		}
	}
}
