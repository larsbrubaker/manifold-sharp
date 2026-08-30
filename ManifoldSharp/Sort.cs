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

// Sort.cs — port of sort.rs — Phase 5: SortGeometry, Morton codes, vertex/face
// sorting.
//
// Ports src/sort.cpp from the Manifold C++ library.
//
// (The Rust header ends "The Collider is stubbed (Phase 10 will implement it
// fully)"; that sentence is stale in the Rust tree — collider.rs is complete and
// sort_geometry caches a real BVH — and it is not carried over.)
//
// ── Sort audit ───────────────────────────────────────────────────────────────
// CLAUDE.md requires every sort site to be justified in place. Both of
// this module's sorts are Rust `sort_by`, which is STABLE, and both are
// numerical-parity surfaces: ties in the Morton code are common (the code is
// only 30 bits, and degenerate/removed elements all carry K_NO_CODE), and the
// permutation they resolve to decides the final vertex and triangle numbering of
// every mesh in the library. They are therefore LINQ `OrderBy`, which the BCL
// documents as stable. `Array.Sort` / `List<T>.Sort` are unstable introsort and
// must not be substituted. The two sites are marked SORT AUDIT below.
//
// ── Arrays, not lists, for the box buffers ───────────────────────────────────
// `face_box` is a `Vec<BBox>` that `get_face_box_morton` grows with
// `face_box[face].union_point(pos)`. Through a `List<Box>` indexer that call
// compiles and silently updates a temporary — the trap the Bounds.cs header
// names — so the box buffers are `Box[]`, where an element is a real variable.
// Their `resize` (the sort.rs:164 site) goes through RustVec.Resize, because
// `Array.Resize` pads with `default(Box)`, a box at the origin, and the Rust
// pads with `BBox::default()`, the inverted empty box.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// The free functions of <c>sort.rs</c>: Morton codes and the vertex/face
	/// reordering that finalizes a mesh after <see cref="ManifoldImpl.CreateHalfedges"/>.
	/// </summary>
	public static class Sort
	{
		/// <summary>
		/// The Morton code given to elements flagged for removal, so they sort last and
		/// can be trimmed by a partition point.
		/// </summary>
		internal const uint KNoCode = 0xFFFFFFFFu;

		// ---------------------------------------------------------------------
		// Morton code (30-bit, 10 bits per axis)
		// ---------------------------------------------------------------------

		/// <summary>
		/// Compute a 30-bit Morton code for a position within the given bounding box.
		/// Returns K_NO_CODE for NaN positions (unreferenced vertices).
		/// </summary>
		/// <param name="position">The position to encode.</param>
		/// <param name="bbox">The box the position is normalized against.</param>
		/// <returns>The Morton code, or K_NO_CODE for a NaN position.</returns>
		public static uint MortonCode(Vec3 position, Box bbox)
		{
			if (double.IsNaN(position.X))
			{
				return KNoCode;
			}

			return MortonCodeImpl(position, bbox);
		}

		/// <summary>
		/// Spread the low 10 bits of v into bits 0,3,6,9,...,27 (every 3rd bit).
		/// This is the inverse of the interleaving needed for a 3D Morton code.
		/// </summary>
		internal static uint SpreadBits3(uint v)
		{
			// Rust `wrapping_mul`. C# `uint` arithmetic is unchecked by default and so
			// already wraps; `unchecked` is written out so the intent survives a project
			// that ever turns on CheckForOverflowUnderflow.
			unchecked
			{
				v = 0xFF0000FFu & (v * 0x00010001u);
				v = 0x0F00F00Fu & (v * 0x00000101u);
				v = 0xC30C30C3u & (v * 0x00000011u);
				v = 0x49249249u & (v * 0x00000005u);
				return v;
			}
		}

		private static uint MortonCodeImpl(Vec3 position, Box bbox)
		{
			Vec3 range = bbox.Max - bbox.Min;
			Vec3 xyz = (position - bbox.Min) / range;

			// Rust `.min(1023.0).max(0.0)` is f64::min / f64::max, where a NaN operand
			// loses — so a NaN component clamps to 1023.0 then to itself, and only the
			// x-NaN early-out above rejects codes. MinF64/MaxF64 reproduce that;
			// Math.Min/Math.Max would propagate the NaN.
			double xF = MaxF64(MinF64(1024.0 * xyz.X, 1023.0), 0.0);
			double yF = MaxF64(MinF64(1024.0 * xyz.Y, 1023.0), 0.0);
			double zF = MaxF64(MinF64(1024.0 * xyz.Z, 1023.0), 0.0);
			uint x = SpreadBits3((uint)xF);
			uint y = SpreadBits3((uint)yF);
			uint z = SpreadBits3((uint)zF);
			return (x * 4) + (y * 2) + z;
		}

		// ---------------------------------------------------------------------
		// SortVerts
		// ---------------------------------------------------------------------

		/// <summary>
		/// Sorts vertices by their Morton code and removes NaN-flagged vertices.
		/// Updates all halfedge vertex references accordingly.
		/// </summary>
		/// <param name="mesh">The mesh to sort in place.</param>
		public static void SortVerts(ManifoldImpl mesh)
		{
			int numVert = mesh.VertPos.Count;
			Box bbox = mesh.Bbox;

			// Compute Morton code for each vertex
			uint[] vertMorton = new uint[numVert];
			for (int i = 0; i < numVert; i++)
			{
				vertMorton[i] = MortonCode(mesh.VertPos[i], bbox);
			}

			// Build sorted index array.
			// SORT AUDIT (sort.rs:67): Rust `sort_by`, which is STABLE. Vertices sharing a
			// Morton code — common, since the code is a 30-bit quantization — must keep
			// ascending original index, because that permutation becomes the mesh's vertex
			// numbering and every downstream position, halfedge and property index with
			// it. OrderBy is the stable C# sort; Array.Sort is not usable here.
			int[] vertNew2Old = Enumerable.Range(0, numVert).OrderBy(i => vertMorton[i]).ToArray();

			// Find how many survive (NaN verts get K_NO_CODE, sort to end)
			int newNumVert = PartitionPoint(vertNew2Old, v => vertMorton[v] < KNoCode);
			int[] vertNew2OldTrimmed = new int[newNumVert];
			Array.Copy(vertNew2Old, vertNew2OldTrimmed, newNumVert);

			ReindexVerts(mesh, vertNew2OldTrimmed, numVert);

			// Permute vert positions (only surviving verts)
			List<Vec3> oldPos = new List<Vec3>(mesh.VertPos);
			mesh.VertPos.Resize(newNumVert, new Vec3(0.0, 0.0, 0.0));
			for (int newIdx = 0; newIdx < newNumVert; newIdx++)
			{
				mesh.VertPos[newIdx] = oldPos[vertNew2OldTrimmed[newIdx]];
			}

			// Permute vert normals if present
			if (mesh.VertNormal.Count == numVert)
			{
				List<Vec3> oldN = new List<Vec3>(mesh.VertNormal);
				mesh.VertNormal.Resize(newNumVert, new Vec3(0.0, 0.0, 0.0));
				for (int newIdx = 0; newIdx < newNumVert; newIdx++)
				{
					mesh.VertNormal[newIdx] = oldN[vertNew2OldTrimmed[newIdx]];
				}
			}
		}

		/// <summary>
		/// Updates halfedge start/end vert indices from old-&gt;new index mapping.
		/// <c>vert_new2old[new] = old</c> — we invert to get <c>vert_old2new[old] = new</c>.
		/// </summary>
		/// <param name="mesh">The mesh whose halfedges are reindexed.</param>
		/// <param name="vertNew2Old">The new-to-old vertex permutation.</param>
		/// <param name="oldNumVert">The vertex count before the permutation.</param>
		public static void ReindexVerts(ManifoldImpl mesh, int[] vertNew2Old, int oldNumVert)
		{
			int[] vertOld2New = new int[oldNumVert];
			for (int i = 0; i < oldNumVert; i++)
			{
				vertOld2New[i] = -1;
			}

			for (int newIdx = 0; newIdx < vertNew2Old.Length; newIdx++)
			{
				vertOld2New[vertNew2Old[newIdx]] = newIdx;
			}

			bool hasProp = mesh.NumProp > 0;

			// Rust `for edge in mesh.halfedge.iter_mut()`. CollectionsMarshal.AsSpan hands
			// out the List's own storage, so `ref edge` writes land in the list; a plain
			// foreach would iterate copies. Nothing here resizes the list, which is the
			// one thing that would invalidate the span.
			foreach (ref Halfedge edge in System.Runtime.InteropServices.CollectionsMarshal.AsSpan(mesh.Halfedge))
			{
				if (edge.StartVert < 0)
				{
					continue;
				}

				edge.StartVert = vertOld2New[edge.StartVert];
				edge.EndVert = vertOld2New[edge.EndVert];
				if (!hasProp)
				{
					edge.PropVert = edge.StartVert;
				}
			}
		}

		// ---------------------------------------------------------------------
		// GetFaceBoxMorton
		// ---------------------------------------------------------------------

		/// <summary>
		/// Computes per-face bounding boxes and Morton codes.
		/// Faces with removed halfedges (pairedHalfedge &lt; 0) get K_NO_CODE.
		/// </summary>
		/// <param name="mesh">The mesh to measure.</param>
		/// <returns>The per-face boxes and their Morton codes.</returns>
		public static (Box[] FaceBox, uint[] FaceMorton) GetFaceBoxMorton(ManifoldImpl mesh)
		{
			int numTri = mesh.NumTri();
			Box bbox = mesh.Bbox;

			// sort.rs:122 — `vec![BBox::default(); num_tri]`. Box.Filled, never
			// `new Box[numTri]`: faces whose halfedge was removed keep this default, and
			// the zeroed default would be a box at the origin instead of the empty box.
			Box[] faceBox = Box.Filled(numTri);
			uint[] faceMorton = new uint[numTri];

			for (int face = 0; face < numTri; face++)
			{
				if (mesh.Halfedge[3 * face].PairedHalfedge < 0)
				{
					faceMorton[face] = KNoCode;
					continue;
				}

				Vec3 center = new Vec3(0.0, 0.0, 0.0);
				for (int i = 0; i < 3; i++)
				{
					Vec3 pos = mesh.VertPos[mesh.Halfedge[(3 * face) + i].StartVert];
					center = center + pos;
					faceBox[face].UnionPoint(pos);
				}

				center = center / 3.0;
				faceMorton[face] = MortonCodeImpl(center, bbox);
			}

			return (faceBox, faceMorton);
		}

		// ---------------------------------------------------------------------
		// SortFaces / GatherFaces
		// ---------------------------------------------------------------------

		/// <summary>
		/// Sorts faces by Morton code, removing faces flagged for removal (K_NO_CODE).
		/// Updates <paramref name="faceBox"/> and <paramref name="faceMorton"/> in-place.
		/// </summary>
		/// <param name="mesh">The mesh whose faces are reordered.</param>
		/// <param name="faceBox">The per-face boxes, resized to the surviving faces.</param>
		/// <param name="faceMorton">The per-face codes, resized to the surviving faces.</param>
		public static void SortFaces(ManifoldImpl mesh, ref Box[] faceBox, ref uint[] faceMorton)
		{
			int numTri = faceBox.Length;

			// Stable sort by Morton code (removed tris get K_NO_CODE → sorted last).
			// SORT AUDIT (sort.rs:154): Rust `sort_by`, STABLE. Coplanar and
			// near-coincident triangles routinely share a Morton code, and the tie order
			// becomes the output triangle order — which the boolean engine's face IDs and
			// every ported test's expected triangle list depend on. OrderBy, not
			// Array.Sort.
			uint[] mortonKey = faceMorton;
			int[] faceNew2OldFull = Enumerable.Range(0, numTri).OrderBy(f => mortonKey[f]).ToArray();

			// Trim removed faces
			int newNumTri = PartitionPoint(faceNew2OldFull, f => mortonKey[f] < KNoCode);
			int[] faceNew2Old = new int[newNumTri];
			Array.Copy(faceNew2OldFull, faceNew2Old, newNumTri);

			// Permute face_morton and face_box to match new order
			uint[] oldMorton = (uint[])faceMorton.Clone();
			Box[] oldBox = (Box[])faceBox.Clone();
			RustVec.Resize(ref faceMorton, newNumTri, 0u);

			// sort.rs:164 — the grow half of `resize(new_num_tri, BBox::default())`. The
			// fill has to be `new Box()` (the inverted empty box), which is what
			// Array.Resize alone would get wrong.
			RustVec.Resize(ref faceBox, newNumTri, new Box());

			for (int newF = 0; newF < newNumTri; newF++)
			{
				int oldF = faceNew2Old[newF];
				faceMorton[newF] = oldMorton[oldF];
				faceBox[newF] = oldBox[oldF];
			}

			GatherFaces(mesh, faceNew2Old);
		}

		/// <summary>
		/// Reorders halfedges (and related arrays) according to the
		/// <paramref name="faceNew2Old"/> permutation.
		/// </summary>
		/// <param name="mesh">The mesh to reorder in place.</param>
		/// <param name="faceNew2Old">The new-to-old face permutation.</param>
		public static void GatherFaces(ManifoldImpl mesh, int[] faceNew2Old)
		{
			int numTri = faceNew2Old.Length;
			int oldNumTri = mesh.NumTri();

			// Permute tri_ref if present
			if (mesh.MeshRelation.TriRef.Count == oldNumTri)
			{
				List<TriRef> oldTriRef = new List<TriRef>(mesh.MeshRelation.TriRef);
				mesh.MeshRelation.TriRef.Resize(numTri, default(TriRef));
				for (int newF = 0; newF < numTri; newF++)
				{
					mesh.MeshRelation.TriRef[newF] = oldTriRef[faceNew2Old[newF]];
				}
			}

			// Permute face normals if present
			if (mesh.FaceNormal.Count == oldNumTri)
			{
				List<Vec3> oldNormals = new List<Vec3>(mesh.FaceNormal);
				mesh.FaceNormal.Resize(numTri, new Vec3(0.0, 0.0, 0.0));
				for (int newF = 0; newF < numTri; newF++)
				{
					mesh.FaceNormal[newF] = oldNormals[faceNew2Old[newF]];
				}
			}

			// Build faceOld2New for pairedHalfedge remapping
			int[] faceOld2New = new int[oldNumTri];
			for (int i = 0; i < oldNumTri; i++)
			{
				faceOld2New[i] = -1;
			}

			for (int newF = 0; newF < numTri; newF++)
			{
				faceOld2New[faceNew2Old[newF]] = newF;
			}

			// Gather halfedges from old layout into new
			List<Halfedge> oldHalfedge = new List<Halfedge>(mesh.Halfedge);
			List<Vec4> oldTangent = new List<Vec4>(mesh.HalfedgeTangent);
			bool hasTangent = oldTangent.Count != 0;

			mesh.Halfedge.Resize(3 * numTri, default(Halfedge));
			if (hasTangent)
			{
				mesh.HalfedgeTangent.Resize(3 * numTri, default(Vec4));
			}

			for (int newFace = 0; newFace < numTri; newFace++)
			{
				int oldFace = faceNew2Old[newFace];
				for (int i = 0; i < 3; i++)
				{
					int oldEdgeIdx = (3 * oldFace) + i;
					int newEdgeIdx = (3 * newFace) + i;
					Halfedge edge = oldHalfedge[oldEdgeIdx];

					// Remap pairedHalfedge
					if (edge.PairedHalfedge >= 0)
					{
						int pairedOldFace = edge.PairedHalfedge / 3;
						int offset = edge.PairedHalfedge % 3;
						edge.PairedHalfedge = (3 * faceOld2New[pairedOldFace]) + offset;
					}

					mesh.Halfedge[newEdgeIdx] = edge;
					if (hasTangent)
					{
						mesh.HalfedgeTangent[newEdgeIdx] = oldTangent[oldEdgeIdx];
					}
				}
			}
		}

		// ---------------------------------------------------------------------
		// CompactProps
		// ---------------------------------------------------------------------

		/// <summary>Removes unreferenced property vertices and reindexes propVerts.</summary>
		/// <param name="mesh">The mesh to compact in place.</param>
		public static void CompactProps(ManifoldImpl mesh)
		{
			if (mesh.NumProp == 0)
			{
				return;
			}

			int numProp = mesh.NumProp;
			int numPropVerts = mesh.Properties.Count / numProp;

			// Mark which prop verts are referenced
			bool[] keep = new bool[numPropVerts];
			foreach (Halfedge edge in mesh.Halfedge)
			{
				if (edge.PropVert >= 0 && edge.PropVert < numPropVerts)
				{
					keep[edge.PropVert] = true;
				}
			}

			// Build prefix sum for old→new mapping
			int[] propOld2New = new int[numPropVerts + 1];
			for (int i = 0; i < numPropVerts; i++)
			{
				propOld2New[i + 1] = propOld2New[i] + (keep[i] ? 1 : 0);
			}

			int newNumPropVerts = propOld2New[numPropVerts];

			// Compact properties array
			List<double> oldProp = new List<double>(mesh.Properties);
			mesh.Properties.Resize(numProp * newNumPropVerts, 0.0);
			for (int oldIdx = 0; oldIdx < numPropVerts; oldIdx++)
			{
				if (!keep[oldIdx])
				{
					continue;
				}

				int newIdx = propOld2New[oldIdx];
				for (int p = 0; p < numProp; p++)
				{
					mesh.Properties[(newIdx * numProp) + p] = oldProp[(oldIdx * numProp) + p];
				}
			}

			// Remap propVert indices in halfedges
			foreach (ref Halfedge edge in System.Runtime.InteropServices.CollectionsMarshal.AsSpan(mesh.Halfedge))
			{
				if (edge.PropVert >= 0)
				{
					edge.PropVert = propOld2New[edge.PropVert];
				}
			}
		}

		// ---------------------------------------------------------------------
		// SortGeometry — main entry point
		// ---------------------------------------------------------------------

		/// <summary>
		/// Sorts vertices and faces by Morton code, removes flagged-for-deletion elements,
		/// and compacts property arrays. Should be called after
		/// <see cref="ManifoldImpl.CreateHalfedges"/> to finalize the mesh topology.
		/// </summary>
		/// <param name="mesh">The mesh to sort in place.</param>
		public static void SortGeometry(ManifoldImpl mesh)
		{
			if (mesh.Halfedge.Count == 0)
			{
				mesh.Collider = new Collider();
				return;
			}

			SortVerts(mesh);
			(Box[] faceBox, uint[] faceMorton) = GetFaceBoxMorton(mesh);
			SortFaces(mesh, ref faceBox, ref faceMorton);
			if (mesh.Halfedge.Count == 0)
			{
				mesh.Collider = new Collider();
				return;
			}

			// Cache the face BVH on the mesh (C++ builds collider_ here in
			// SortGeometry); query sites reuse it instead of rebuilding per query.
			mesh.Collider = new Collider(faceBox, faceMorton);
			CompactProps(mesh);

			System.Diagnostics.Debug.Assert(
				mesh.Halfedge.Count % 6 == 0,
				$"Not an even number of halfedges after sorting (expected multiple of 6, got {mesh.Halfedge.Count})");
		}

		/// <summary>
		/// Rust <c>slice::partition_point</c>: the index of the first element for which
		/// <paramref name="predicate"/> is false, over a slice already partitioned so every
		/// true precedes every false.
		/// </summary>
		/// <remarks>
		/// Written out because the BCL has no equivalent. Both call sites feed it a
		/// Morton-sorted permutation, so the partition precondition holds by construction;
		/// the linear scan is what a binary search would find, and is transcribed linearly
		/// so a violated precondition degrades into an obvious answer rather than a
		/// plausible wrong one.
		/// </remarks>
		private static int PartitionPoint(int[] items, Func<int, bool> predicate)
		{
			int count = 0;
			while (count < items.Length && predicate(items[count]))
			{
				count++;
			}

			return count;
		}
	}
}
