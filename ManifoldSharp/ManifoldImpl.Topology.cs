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

// ManifoldImpl.Topology.cs — the halfedge construction and bookkeeping half of
// impl_mesh.rs: CreateHalfedges, InitializeOriginal, the #1718 normal
// bookkeeping, IncrementMeshIds, DedupePropVerts and RemoveUnreferencedVerts.
// The module header lives in ManifoldImpl.cs.

using System.Runtime.InteropServices;

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <content>
	/// Halfedge construction, mesh-relation bookkeeping and property dedup.
	/// </content>
	public sealed partial class ManifoldImpl
	{
		/// <summary>
		/// Build the halfedge data structure from triangle lists.
		/// </summary>
		/// <remarks>
		/// When <paramref name="triVert"/> is empty, <paramref name="triProp"/> is used for
		/// both geometry and properties. When <paramref name="triVert"/> is present,
		/// <c>triProp[i][j]</c> = <c>propVert</c> and <c>triVert[i][j]</c> = <c>startVert</c>.
		/// </remarks>
		/// <param name="triProp">
		/// Property vertex indices per triangle (also geometry if
		/// <paramref name="triVert"/> is empty).
		/// </param>
		/// <param name="triVert">Geometry vertex indices per triangle (may be empty).</param>
		public void CreateHalfedges(IReadOnlyList<IVec3> triProp, IReadOnlyList<IVec3> triVert)
		{
			// The triangle set is being (re)built, so any earlier verdict about
			// self-intersection no longer describes this geometry.
			this.InvalidateSelfIntersects();
			int numTri = triProp.Count;
			if (numTri == 0)
			{
				this.Halfedge.Clear();
				return;
			}

			int numHalfedge = 3 * numTri;
			int numEdge = numHalfedge / 2;

			this.Halfedge.Clear();
			this.Halfedge.Resize(numHalfedge, new Halfedge(-1, -1, -1, -1));

			bool useProp = triVert.Count == 0;

			// Build halfedges and compute edge sort key
			// key = [forward_bit:1][min_vert:31][max_vert:32]
			// forward: v0 < v1 → bit=1; backward: v0 > v1 → bit=0
			// After sorting: backward halfedges first, then forward, both by (min,max)
			ulong[] edgeKeys = new ulong[numHalfedge];

			// Rust's `self.halfedge[e] = ...` and later `self.halfedge[pair].field = ...`
			// both need in-place writes; CollectionsMarshal.AsSpan is the port of that.
			// The list is not resized again below, which is what would invalidate it.
			Span<Halfedge> halfedge = CollectionsMarshal.AsSpan(this.Halfedge);

			for (int tri = 0; tri < numTri; tri++)
			{
				// Read the triangles into locals: IVec3's indexer returns `ref int`, and a
				// ref-returning member wants an addressable receiver, which an
				// IReadOnlyList indexer result is not.
				IVec3 propTri = triProp[tri];
				IVec3 vertTri = useProp ? default(IVec3) : triVert[tri];
				for (int i = 0; i < 3; i++)
				{
					int j = Next3(i);
					int e = (3 * tri) + i;
					int v0 = useProp ? propTri[i] : vertTri[i];
					int v1 = useProp ? propTri[j] : vertTri[j];
					halfedge[e] = new Halfedge(v0, v1, -1, propTri[i]);
					ulong fwd = v0 < v1 ? 1UL : 0UL;

					// Rust `as u64` from i32 sign-extends, which C#'s unchecked int→ulong
					// conversion also does. Vertex indices are non-negative in every
					// production path, so the two only have to agree on the degenerate
					// input — and they do.
					ulong minV = unchecked((ulong)Math.Min(v0, v1));
					ulong maxV = unchecked((ulong)Math.Max(v0, v1));
					edgeKeys[e] = (fwd << 63) | (minV << 32) | maxV;
				}
			}

			// Sort halfedge indices by edge key.
			// SORT AUDIT (impl_mesh.rs:403): C++ CreateHalfedges uses a STABLE sort here
			// (impl.cpp), and the #1687 fix ensures its parallel stable_sort matches
			// std::stable_sort. When two halfedges share an edge key (duplicate directed
			// edges in degenerate/intermediate meshes) the tie must break on original
			// halfedge-index order, so we use a stable sort to stay bit-identical to C++.
			// LINQ OrderBy is documented stable; Array.Sort is not usable here.
			int[] ids = Enumerable.Range(0, numHalfedge).OrderBy(i => edgeKeys[i]).ToArray();

			// ids[0..num_edge] = backward halfedges (startVert > endVert), sorted by (min,max)
			// ids[num_edge..] = forward halfedges (startVert < endVert), sorted by (min,max)

			// Sequential pairing with opposed-triangle detection
			int segmentEnd = numEdge;
			int consecutiveStart = 0;

			for (int i = 0; i < numEdge; i++)
			{
				int pair0 = ids[i];
				int h0Sv = halfedge[pair0].StartVert;
				int h0Ev = halfedge[pair0].EndVert;

				int k = consecutiveStart + numEdge;
				while (true)
				{
					if (k >= segmentEnd + numEdge)
					{
						break;
					}

					int pair1 = ids[k];
					int h1Sv = halfedge[pair1].StartVert;
					int h1Ev = halfedge[pair1].EndVert;

					if (h0Sv != h1Ev || h0Ev != h1Sv)
					{
						break; // Different edge direction — no match
					}

					if (halfedge[pair1].PairedHalfedge != KRemovedHalfedge)
					{
						// Check for opposed triangle: same undirected edge, same third vertex
						int next0 = NextHalfedge(pair0);
						int next1 = NextHalfedge(pair1);
						if (halfedge[next0].EndVert == halfedge[next1].EndVert)
						{
							// Opposed triangles: mark both for removal.
							// Reorder ids so the remaining valid forward halfedge (at i+num_edge)
							// moves to position k, and pair1 (the opposed one) goes to i+num_edge.
							// This matches C++ which does: ids[k] = ids[i+numEdge]; ids[i+numEdge] = pair1;
							halfedge[pair0].PairedHalfedge = KRemovedHalfedge;
							halfedge[pair1].PairedHalfedge = KRemovedHalfedge;
							if (i + numEdge != k)
							{
								(ids[k], ids[i + numEdge]) = (ids[i + numEdge], ids[k]);
							}

							break;
						}
					}

					k += 1;
				}

				// Update consecutive_start for next iteration
				if (i + 1 < segmentEnd)
				{
					int nextSv = halfedge[ids[i + 1]].StartVert;
					int nextEv = halfedge[ids[i + 1]].EndVert;
					if (nextSv != h0Sv || nextEv != h0Ev)
					{
						consecutiveStart = i + 1;
					}
				}
			}

			// Final pairing pass
			for (int i = 0; i < numEdge; i++)
			{
				int pair0 = ids[i];
				int pair1 = ids[i + numEdge];
				if (halfedge[pair0].PairedHalfedge != KRemovedHalfedge)
				{
					halfedge[pair0].PairedHalfedge = pair1;
					halfedge[pair1].PairedHalfedge = pair0;
				}
				else
				{
					// Invalidate both (opposed triangles removed)
					halfedge[pair0] = new Halfedge(-1, -1, -1, 0);
					halfedge[pair1] = new Halfedge(-1, -1, -1, 0);
				}
			}
		}

		// ---------------------------------------------------------------------
		// InitializeOriginal
		// ---------------------------------------------------------------------

		/// <summary>Set up the mesh relation for a newly created original mesh.</summary>
		public void InitializeOriginal()
		{
			// Per C++ #1718: preserve the AND-across-old-Relations hasNormals state
			// so AsOriginal keeps the recording when it builds a fresh Relation.
			// Primitives start with an empty map → all_have_normals() is false.
			bool hadNormals = this.AllHaveNormals();
			int meshId = (int)ReserveIds(1);
			this.MeshRelation.OriginalId = meshId;
			int numTri = this.NumTri();
			this.MeshRelation.TriRef.Resize(numTri, default(TriRef));
			for (int tri = 0; tri < numTri; tri++)
			{
				TriRef triRef = this.MeshRelation.TriRef[tri];
				triRef.MeshId = meshId;
				triRef.OriginalId = meshId;
				triRef.FaceId = -1;
				triRef.CoplanarId = tri;
				this.MeshRelation.TriRef[tri] = triRef;
			}

			this.MeshRelation.MeshIdTransform.Clear();
			Relation relation = new Relation();
			relation.OriginalId = meshId;
			relation.Transform = Mat3x4.Identity();
			relation.BackSide = false;
			relation.HasNormals = hadNormals;
			this.MeshRelation.MeshIdTransform.Add(meshId, relation);
		}

		/// <summary>
		/// True only when every meshID carries normals at slot 0..2 — the condition under
		/// which <c>GetMeshGL(-1)</c> can safely auto-substitute that slot.
		/// </summary>
		/// <remarks>
		/// A mixed Boolean output (some meshIDs with normals, some without) returns false;
		/// the output MeshGL's per-run bit 1 still marks the with-normals runs individually.
		/// AND semantics across meshIDs. Per C++ #1718.
		/// </remarks>
		/// <returns>True when every meshID records normals.</returns>
		public bool AllHaveNormals()
		{
			SortedDictionary<int, Relation> map = this.MeshRelation.MeshIdTransform;
			if (map.Count == 0)
			{
				return false;
			}

			foreach (Relation m in map.Values)
			{
				if (!m.HasNormals)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// True iff the meshID owning <paramref name="tri"/> has hasNormals set. False when
		/// the meshID isn't in <c>MeshIdTransform</c> (treat as no-normals). Per C++ #1718.
		/// </summary>
		/// <param name="tri">The triangle index.</param>
		/// <returns>True when the owning meshID records normals.</returns>
		public bool TriHasNormals(int tri)
		{
			int meshId = this.MeshRelation.TriRef[tri].MeshId;
			if (this.MeshRelation.MeshIdTransform.TryGetValue(meshId, out Relation m))
			{
				return m.HasNormals;
			}

			return false;
		}

		/// <summary>
		/// Eager-transform slot 0..2 of <paramref name="properties"/> for propVerts whose
		/// meshID carries hasNormals.
		/// </summary>
		/// <remarks>
		/// Used by both <see cref="Transform"/> and Compose so world-frame normals stay in
		/// sync with vertPos / faceNormal across any sequence of transforms (including
		/// mixed-input Boolean/Compose outputs where some meshIDs carry normals and others
		/// don't). Per C++ #1718.
		/// <para>
		/// <paramref name="properties"/> is laid out as
		/// <c>properties[(offset + prop) * stride + i]</c>, so callers can target an
		/// in-place properties vector (offset=0) or a per-node slice of a combined array
		/// (offset=propVertIndices, stride=numPropOut). Re-normalizes as it transforms so
		/// non-orthogonal transforms (scale) and upstream barycentric interpolation don't
		/// leave non-unit values that compound downstream.
		/// </para>
		/// </remarks>
		/// <param name="halfedge">The halfedge arena the prop verts are reached through.</param>
		/// <param name="meshRelation">The relation table naming which meshIDs carry normals.</param>
		/// <param name="normalTransform">The normal transform to apply.</param>
		/// <param name="properties">The property array to rewrite in place.</param>
		/// <param name="numPropVert">The number of property vertices.</param>
		/// <param name="stride">The property stride.</param>
		/// <param name="offset">The property-vertex offset into the array.</param>
		public static void EagerTransformPropNormals(
			List<Halfedge> halfedge,
			MeshRelationD meshRelation,
			Mat3 normalTransform,
			List<double> properties,
			int numPropVert,
			int stride,
			int offset)
		{
			// OR semantics (any meshID has normals), unlike AllHaveNormals():
			// mixed inputs still need the per-meshID iteration below to rotate the
			// with-normals subset.
			bool anyHasNormals = false;
			foreach (Relation m in meshRelation.MeshIdTransform.Values)
			{
				if (m.HasNormals)
				{
					anyHasNormals = true;
					break;
				}
			}

			if (!anyHasNormals)
			{
				return;
			}

			bool TriHasNormalsLocal(int tri)
			{
				int mid = meshRelation.TriRef[tri].MeshId;
				if (meshRelation.MeshIdTransform.TryGetValue(mid, out Relation m))
				{
					return m.HasNormals;
				}

				return false;
			}

			bool[] visited = new bool[numPropVert];
			for (int e = 0; e < halfedge.Count; e++)
			{
				if (!TriHasNormalsLocal(e / 3))
				{
					continue;
				}

				int prop = halfedge[e].PropVert;
				if (prop < 0)
				{
					continue;
				}

				if (visited[prop])
				{
					continue;
				}

				visited[prop] = true;
				int baseIdx = (offset + prop) * stride;
				Vec3 n = new Vec3(properties[baseIdx], properties[baseIdx + 1], properties[baseIdx + 2]);
				Vec3 nt = SafeNormalize(normalTransform * n);
				properties[baseIdx] = nt.X;
				properties[baseIdx + 1] = nt.Y;
				properties[baseIdx + 2] = nt.Z;
			}
		}

		// ---------------------------------------------------------------------
		// IncrementMeshIDs — port of C++ Manifold::Impl::IncrementMeshIDs()
		// ---------------------------------------------------------------------

		/// <summary>
		/// Allocates fresh unique mesh IDs and remaps all triRef.meshID values. This ensures
		/// boolean results don't collide with source mesh IDs.
		/// </summary>
		public void IncrementMeshIds()
		{
			// Build old -> new ID mapping. Iteration order determines which old
			// ID gets which fresh ID, so it must be sorted like C++ std::map —
			// SortedDictionary is, which is why MeshIdTransform is one.
			List<KeyValuePair<int, Relation>> oldTransforms =
				new List<KeyValuePair<int, Relation>>(this.MeshRelation.MeshIdTransform);
			this.MeshRelation.MeshIdTransform.Clear();
			uint numMeshIds = (uint)oldTransforms.Count;
			if (numMeshIds == 0)
			{
				return;
			}

			int nextMeshId = (int)ReserveIds(numMeshIds);
			Dictionary<int, int> old2New = new Dictionary<int, int>();
			foreach (KeyValuePair<int, Relation> entry in oldTransforms)
			{
				old2New.Add(entry.Key, nextMeshId);
				this.MeshRelation.MeshIdTransform.Add(nextMeshId, entry.Value);
				nextMeshId += 1;
			}

			// Update all triRef.meshID
			for (int i = 0; i < this.MeshRelation.TriRef.Count; i++)
			{
				TriRef triRef = this.MeshRelation.TriRef[i];
				if (old2New.TryGetValue(triRef.MeshId, out int newId))
				{
					triRef.MeshId = newId;
					this.MeshRelation.TriRef[i] = triRef;
				}
			}
		}

		// ---------------------------------------------------------------------
		// DedupePropVerts — port of C++ Manifold::Impl::DedupePropVerts()
		// ---------------------------------------------------------------------

		/// <summary>
		/// Deduplicates property vertices that share identical property values across paired
		/// halfedges within the same mesh.
		/// </summary>
		public void DedupePropVerts()
		{
			int numProp = this.NumProp;
			if (numProp == 0)
			{
				return;
			}

			int nEdges = this.Halfedge.Count;

			// Collect (prop0, prop1) pairs for edges where properties match
			int[] vert2VertA = new int[nEdges];
			int[] vert2VertB = new int[nEdges];
			for (int i = 0; i < nEdges; i++)
			{
				vert2VertA[i] = -1;
				vert2VertB[i] = -1;
			}

			for (int edgeIdx = 0; edgeIdx < nEdges; edgeIdx++)
			{
				Halfedge edge = this.Halfedge[edgeIdx];
				if (edge.PairedHalfedge < 0)
				{
					continue;
				}

				int edgeFace = edgeIdx / 3;
				int pairFace = edge.PairedHalfedge / 3;

				if (this.MeshRelation.TriRef[edgeFace].MeshId != this.MeshRelation.TriRef[pairFace].MeshId)
				{
					continue;
				}

				int prop0 = this.Halfedge[edgeIdx].PropVert;
				int prop1 = this.Halfedge[NextHalfedge(edge.PairedHalfedge)].PropVert;
				if (prop0 < 0 || prop1 < 0)
				{
					continue;
				}

				bool propEqual = true;
				for (int p = 0; p < numProp; p++)
				{
					int idx0 = (numProp * prop0) + p;
					int idx1 = (numProp * prop1) + p;
					if (idx0 >= this.Properties.Count || idx1 >= this.Properties.Count)
					{
						propEqual = false;
						break;
					}

					if (this.Properties[idx0] != this.Properties[idx1])
					{
						propEqual = false;
						break;
					}
				}

				if (propEqual)
				{
					vert2VertA[edgeIdx] = prop0;
					vert2VertB[edgeIdx] = prop1;
				}
			}

			// Use union-find to merge equivalent property vertices
			int numPropVert = this.NumPropVert();
			DisjointSets ds = new DisjointSets((uint)numPropVert);
			for (int i = 0; i < nEdges; i++)
			{
				int a = vert2VertA[i];
				int b = vert2VertB[i];
				if (a >= 0 && b >= 0)
				{
					ds.Unite((uint)a, (uint)b);
				}
			}

			List<int> vertLabels = new List<int>();
			int numLabels = ds.ConnectedComponents(vertLabels);

			// Build label -> canonical vert mapping
			int[] label2Vert = new int[numLabels];
			for (int v = 0; v < numPropVert; v++)
			{
				label2Vert[vertLabels[v]] = v;
			}

			// Remap all prop_vert indices
			foreach (ref Halfedge edge in CollectionsMarshal.AsSpan(this.Halfedge))
			{
				if (edge.PropVert >= 0 && edge.PropVert < numPropVert)
				{
					edge.PropVert = label2Vert[vertLabels[edge.PropVert]];
				}
			}
		}

		// ---------------------------------------------------------------------
		// RemoveUnreferencedVerts
		// ---------------------------------------------------------------------

		/// <summary>
		/// Mark unreferenced vertices as NaN (to be cleaned up by later passes).
		/// </summary>
		public void RemoveUnreferencedVerts()
		{
			int numVert = this.NumVert();
			bool[] keep = new bool[numVert];
			foreach (Halfedge h in this.Halfedge)
			{
				if (h.StartVert >= 0)
				{
					keep[h.StartVert] = true;
				}
			}

			for (int i = 0; i < numVert; i++)
			{
				if (!keep[i])
				{
					// Rust's f64::NAN is the *positive* quiet NaN; C#'s double.NaN has the
					// sign bit set. Sort.MortonCode and CalculateBBox only test IsNaN, but
					// these values reach bit-level welds downstream, so use the Rust value.
					double nan = DeterministicMath.PositiveQuietNaN;
					this.VertPos[i] = new Vec3(nan, nan, nan);
				}
			}
		}

		// ---------------------------------------------------------------------
		// SortGeometry
		// ---------------------------------------------------------------------

		/// <summary>Reorder mesh geometry for cache efficiency using Morton codes.</summary>
		public void SortGeometry()
		{
			Sort.SortGeometry(this);
		}
	}
}
