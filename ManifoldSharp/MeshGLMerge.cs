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

// MeshGLMerge.cs — port of `MeshGLP::<P, I>::merge` in types_meshgl.rs, itself the
// port of the C++ `MergeMeshGLP` template (src/sort.cpp), shared by `MeshGL::Merge`
// and `MeshGL64::Merge`. MeshGL.Merge and MeshGL64.Merge are one-line entry points
// into the single body here.
//
// The Rust merge is generic over MeshPrecision / MeshIndex (manifold-rust a13d0bf;
// before that it existed only on `MeshGLP<f32, u32>`), so the C# body is written
// once against IMeshGLAccess — the narrow conversion interface MeshGLAccess.cs
// introduced for the other two generic bodies — and every `to_f64` / `to_u64` /
// `I::from_usize` / `P::IS_SINGLE` in the Rust is a call on that interface.
//
// ── Why the open-edge bookkeeping is a counted map ───────────────────────────
// C++ keeps open halfedges in a std::multiset<std::pair<int,int>>: a halfedge adds
// one copy of itself unless a copy of its reverse is present, in which case exactly
// one copy of the reverse is erased; every remaining copy then contributes its start
// vertex to openVerts, duplicates kept. The Rust stands a BTreeMap<(usize, usize),
// usize> of counts in for that multiset, and this port stands a SortedDictionary
// keyed by a (ulong, ulong) ValueTuple in for the BTreeMap. ValueTuple's default
// comparer is lexicographic over Comparer<ulong>.Default, which is exactly the Rust
// tuple `Ord` over usize, and enumeration is ascending — so open verts come out in
// the same key order. A plain set here (what both ports used to have) drops the
// duplicate copies, which loses merges on doubled faces and changes the BVH shape,
// and hence the union-by-rank representative, on pinched boundaries.
//
// Keys are ulong, not int, because the Rust holds `to_u64() as usize` values there:
// narrowing them would make distinct out-of-range u64 indices collide in the map.
// Where one of them is used as an index it goes through a `checked` cast, so an
// index the Rust would panic on (out of bounds) throws here instead of wrapping
// onto a valid vertex.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <summary>The single body of <c>MeshGLP::merge</c> for both MeshGL instantiations.</summary>
	internal static class MeshGLMerge
	{
		// Rust `f32::EPSILON as f64` — 2^-23, the gap between 1.0f and the next float.
		// C# `float.Epsilon` is the smallest *subnormal* float and is wrong here; see
		// the f64 counterpart of this rule in CLAUDE.md.
		private const double F32Epsilon = 1.1920928955078125E-07;

		/// <summary>
		/// Merges coincident vertices based on position within tolerance. Open halfedges
		/// are found with a counted multiset, their start vertices are boxed and
		/// Morton-sorted, and coincident pairs from the BVH are grouped via union-find.
		/// </summary>
		/// <param name="mesh">The mesh, through its conversion interface.</param>
		/// <returns>
		/// False (leaving the merge vectors untouched) if the mesh has no open edges, true
		/// otherwise.
		/// </returns>
		public static bool Merge(IMeshGLAccess mesh)
		{
			int numVert = mesh.NumVert();
			int numTri = mesh.NumTri();

			// Build initial merge map from existing merge vectors
			ulong[] mergeMap = new ulong[numVert];
			for (int i = 0; i < numVert; i++)
			{
				mergeMap[i] = (ulong)i;
			}

			for (int i = 0; i < mesh.MergeFromVertCount; i++)
			{
				mergeMap[checked((int)mesh.MergeFromVert(i))] = mesh.MergeToVert(i);
			}

			// Open halfedges as (start, end) pairs in a counted multiset, standing in for
			// C++'s std::multiset<std::pair<int,int>>: a halfedge adds one copy of itself
			// unless a copy of its reverse is present, in which case exactly one copy of
			// the reverse is erased. Keeping the counts (rather than a set) means a
			// halfedge listed twice survives a single reverse match, as it does in the C++.
			int[] next = new int[] { 1, 2, 0 };
			SortedDictionary<(ulong, ulong), int> openEdges = new SortedDictionary<(ulong, ulong), int>();
			for (int tri = 0; tri < numTri; tri++)
			{
				for (int i = 0; i < 3; i++)
				{
					ulong start = mergeMap[checked((int)mesh.TriVert((3 * tri) + i))];
					ulong end = mergeMap[checked((int)mesh.TriVert((3 * tri) + next[i]))];
					if (openEdges.TryGetValue((end, start), out int count))
					{
						count -= 1;
						if (count == 0)
						{
							openEdges.Remove((end, start));
						}
						else
						{
							openEdges[(end, start)] = count;
						}
					}
					else
					{
						openEdges.TryGetValue((start, end), out int existing);
						openEdges[(start, end)] = existing + 1;
					}
				}
			}

			if (openEdges.Count == 0)
			{
				return false;
			}

			// One entry per open halfedge (duplicates kept), in multiset order: ascending
			// (start, end), equal pairs adjacent. The entry is the halfedge's start
			// vertex, as C++ takes edge.first.
			List<ulong> openVerts = new List<ulong>();
			foreach (KeyValuePair<(ulong, ulong), int> entry in openEdges)
			{
				for (int c = 0; c < entry.Value; c++)
				{
					openVerts.Add(entry.Key.Item1);
				}
			}

			int numOpen = openVerts.Count;

			// Compute bounding box
			Box bbox = new Box();
			for (int v = 0; v < numVert; v++)
			{
				(double X, double Y, double Z) pos = mesh.GetVertPos(v);
				Vec3 p = new Vec3(pos.X, pos.Y, pos.Z);
				bbox.UnionPoint(p);
			}

			// std::max(tolerance, eps * scale): float epsilon for the f32 instantiation,
			// kPrecision for f64, as in the C++ template. Spelled as the Rust's explicit
			// comparison, not MaxF64: a NaN mesh tolerance is kept, as std::max keeps it.
			double eps = mesh.IsSinglePrecision ? F32Epsilon : Types.KPrecision;
			double floor = eps * bbox.Scale();
			double meshTol = mesh.Tolerance;
			double tolerance = meshTol < floor ? floor : meshTol;

			// Build BVH boxes and morton codes for open vertices
			double halfTol = tolerance / 2.0;
			Box[] vertBox = new Box[numOpen];
			uint[] vertMorton = new uint[numOpen];
			for (int k = 0; k < numOpen; k++)
			{
				(double X, double Y, double Z) pos = mesh.GetVertPos(checked((int)openVerts[k]));
				Vec3 center = new Vec3(pos.X, pos.Y, pos.Z);
				vertBox[k] = new Box(
					center - new Vec3(halfTol, halfTol, halfTol),
					center + new Vec3(halfTol, halfTol, halfTol));
				vertMorton[k] = Sort.MortonCode(center, bbox);
			}

			// Stable sort by morton code (C++ stable_sort), so duplicate entries of one
			// vertex stay adjacent in multiset order.
			// SORT AUDIT (types_meshgl.rs merge): Rust `sort_by_key`, which is STABLE, and
			// the order reaches the collider's leaf order and hence the (a, b) argument
			// order of every union below, so it is a numerical-parity surface. LINQ OrderBy
			// is the documented-stable C# sort; Array.Sort is not usable here.
			int[] order = Enumerable.Range(0, numOpen).OrderBy(i => vertMorton[i]).ToArray();

			Box[] sortedBox = new Box[numOpen];
			uint[] sortedMorton = new uint[numOpen];
			ulong[] sortedVerts = new ulong[numOpen];
			for (int k = 0; k < numOpen; k++)
			{
				sortedBox[k] = vertBox[order[k]];
				sortedMorton[k] = vertMorton[order[k]];
				sortedVerts[k] = openVerts[order[k]];
			}

			// Build collider and find coincident vertex pairs. The Rust passes
			// `sorted_box.clone()` only because `Collider::new` takes the Vec by value and
			// the original is queried below; the C# constructor copies the boxes it needs
			// into its node array and keeps no reference, so the array is shared safely.
			// Self-collision: the query set is the leaf set, so each leaf skips itself,
			// as the C++ self-collision query does.
			Collider collider = new Collider(sortedBox, sortedMorton);
			DisjointSets uf = new DisjointSets((uint)numVert);

			// NARROWING AUDIT: Rust `sorted_verts[a] as u32` truncates usize to u32; the
			// unchecked (uint) cast of a ulong is that truncation.
			collider.CollisionsWithBoxes(
				sortedBox,
				true,
				(a, b) => uf.Unite((uint)sortedVerts[a], (uint)sortedVerts[b]));

			// Also merge from existing merge vectors (Rust `to_u64() as u32`, truncating).
			for (int i = 0; i < mesh.MergeFromVertCount; i++)
			{
				uf.Unite((uint)mesh.MergeFromVert(i), (uint)mesh.MergeToVert(i));
			}

			// Rebuild merge vectors
			mesh.ClearMergeVerts();
			for (int v = 0; v < numVert; v++)
			{
				int mergeTo = (int)uf.Find((uint)v);
				if (mergeTo != v)
				{
					mesh.AddMergeFromVertFromUSize(v);
					mesh.AddMergeToVertFromUSize(mergeTo);
				}
			}

			return true;
		}
	}
}
