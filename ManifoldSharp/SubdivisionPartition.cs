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

// Partition — cached topological triangulation for subdivision
// Split from subdivision.rs
// Port of C++ Partition class (lines 31-380)
//
// ── C# port notes ────────────────────────────────────────────────────────────
// subdivision_partition.rs holds two types (`Partition`, `BaryIndices`) and six
// free functions. Per CLAUDE.md's naming rule the module's *public* free
// functions — `next3` and `lerp_vec4` — land on a static class named for the
// module (<see cref="SubdivisionPartition"/>); the file-private ones
// (`swap_i32`, `partition_fan`, `partition_quad`, `partition_cache`) have exactly
// one caller between them and become private statics on <see cref="Partition"/>.
//
// `Partition` is a class, not a struct: the Rust is a `#[derive(Clone)]` struct
// with two `Vec` fields, and every site that hands one out hands out a *deep
// clone*. Modelling that as a C# class with an explicit `Clone()` keeps the copy
// visible at the call site instead of hiding an aliasing bug behind a shallow
// struct copy of two `List` references.
//
// `lerp_vec4` is written `a + (b - a) * t`, deliberately, against the general
// translation rule in CLAUDE.md that says `lerp` is
// `a * (1 - t) + b * t`. That rule is about the *linalg* `lerp`; this module
// spells its own out longhand in the other form, and the two disagree in the
// last bit. The Rust source is the specification, so the longhand is ported
// exactly as written.

using System.Diagnostics;

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	// ---------------------------------------------------------------------------
	// BaryIndices — maps halfedge to triangle + 4D index pair
	// Port of C++ Manifold::Impl::BaryIndices
	// ---------------------------------------------------------------------------

	/// <summary>
	/// The triangle a halfedge belongs to, plus the pair of 4D corner indices its
	/// endpoints occupy in that face's barycentric frame.
	/// </summary>
	/// <remarks>
	/// A <c>tri</c> of -1 means the halfedge is interior to a quad and has no
	/// barycentric identity of its own.
	/// </remarks>
	internal struct BaryIndices
	{
		/// <summary>Index of the tri (or quad-representing lower tri) this halfedge belongs to.</summary>
		public int Tri;

		/// <summary>Corner index in 0..3 of the halfedge's start vertex.</summary>
		public int Start4;

		/// <summary>Corner index in 0..3 of the halfedge's end vertex.</summary>
		public int End4;
	}

	// ---------------------------------------------------------------------------
	// Partition — cached topological triangulation
	// ---------------------------------------------------------------------------

	/// <summary>
	/// A purely topological triangulation of one tri or quad face, keyed by how many
	/// divisions each of its edges gets. Independent of position, so it is computed once
	/// per distinct division tuple and reused for every face that shares it.
	/// </summary>
	internal sealed class Partition
	{
		/// <summary>
		/// The process-global partition cache and the lock that guards it.
		/// </summary>
		/// <remarks>
		/// Ports the Rust's <c>OnceLock&lt;Mutex&lt;HashMap&lt;(i32,i32,i32,i32),
		/// Partition&gt;&gt;&gt;</c>. C# static field initializers already give the
		/// once-only construction <c>OnceLock</c> provides, so only the mutex needs
		/// spelling out.
		/// <para>
		/// Key: the *sorted* divisions vector, the Rust's 4-tuple of the four components.
		/// <see cref="IVec4"/> carries the value equality and hash that tuple did.
		/// </para>
		/// <para>
		/// Value: a partition whose <see cref="Idx"/> is still <c>default</c> — the caller
		/// stamps <see cref="Idx"/> on its own copy *after*
		/// <see cref="GetCachedPartition"/> returns, so the stored entry never carries one.
		/// Entries go in and come out deep-cloned, so no caller can reach into the cache.
		/// </para>
		/// <para>
		/// Locking follows the Rust exactly, and the Rust deliberately does *not* hold the
		/// lock across the build: it locks to probe, unlocks, computes, then locks again to
		/// insert. Two threads racing on a cold key therefore both build, both insert
		/// (last writer wins) and each returns its own equal-valued result. That is
		/// harmless — the partition is a pure function of the key — and reproducing it
		/// keeps this port's contention behaviour the same as the Rust's.
		/// </para>
		/// </remarks>
		private static readonly Dictionary<IVec4, Partition> PartitionCache = new Dictionary<IVec4, Partition>();

		private static readonly object PartitionCacheLock = new object();

		/// <summary>Creates the empty partition — the Rust <c>Partition::new()</c>.</summary>
		public Partition()
		{
			this.Idx = default(IVec4);
			this.SortedDivisions = default(IVec4);
			this.VertBary = new List<Vec4>();
			this.TriVert = new List<IVec3>();
		}

		/// <summary>Permutation taking this face's corners to the sorted/rotated canonical order.</summary>
		public IVec4 Idx { get; set; }

		/// <summary>The divisions per edge in canonical order (descending for tris, rotated for quads).</summary>
		public IVec4 SortedDivisions { get; set; }

		/// <summary>Barycentric coordinates of every vertex of the triangulation.</summary>
		public List<Vec4> VertBary { get; }

		/// <summary>The triangulation, as triples of indices into <see cref="VertBary"/>.</summary>
		public List<IVec3> TriVert { get; }

		/// <summary>
		/// Index in <see cref="VertBary"/> of the first interior vertex: corners and edge
		/// vertices come first, and there are exactly <see cref="SortedDivisions"/>-many of them.
		/// </summary>
		/// <returns>The count of corner plus edge vertices.</returns>
		public int InteriorOffset()
		{
			return this.SortedDivisions[0]
				+ this.SortedDivisions[1]
				+ this.SortedDivisions[2]
				+ this.SortedDivisions[3];
		}

		/// <summary>The number of vertices strictly inside the face.</summary>
		/// <returns>Vertex count less <see cref="InteriorOffset"/>.</returns>
		public int NumInterior()
		{
			return this.VertBary.Count - this.InteriorOffset();
		}

		/// <summary>Port of C++ Partition::GetPartition(ivec4 divisions).</summary>
		/// <param name="divisions">Divisions per edge; a zero w means a triangle.</param>
		/// <returns>A private copy of the cached partition, stamped with its corner permutation.</returns>
		public static Partition GetPartition(IVec4 divisions)
		{
			if (divisions[0] == 0)
			{
				return new Partition(); // skip wrong side of quad
			}

			IVec4 sortedDiv = divisions;
			IVec4 triIdx = new IVec4(0, 1, 2, 3);

			if (divisions[3] == 0)
			{
				// triangle: sort descending
				if (sortedDiv[2] > sortedDiv[1])
				{
					SwapI32(ref sortedDiv, ref triIdx, 2, 1);
				}

				if (sortedDiv[1] > sortedDiv[0])
				{
					SwapI32(ref sortedDiv, ref triIdx, 1, 0);
					if (sortedDiv[2] > sortedDiv[1])
					{
						SwapI32(ref sortedDiv, ref triIdx, 2, 1);
					}
				}
			}
			else
			{
				// quad: rotate to canonical form
				int minIdx = 0;
				int min = divisions[0];
				int next = divisions[1];
				for (int i = 1; i < 4; i++)
				{
					int n = divisions[(i + 1) % 4];
					if (divisions[i] < min || (divisions[i] == min && n < next))
					{
						minIdx = i;
						min = divisions[i];
						next = n;
					}
				}

				IVec4 tmp = sortedDiv;
				for (int i = 0; i < 4; i++)
				{
					triIdx[i] = (i + minIdx) % 4;
					sortedDiv[i] = tmp[triIdx[i]];
				}
			}

			Partition partition = GetCachedPartition(sortedDiv);
			partition.Idx = triIdx;
			return partition;
		}

		/// <summary>Port of C++ Partition::Reindex().</summary>
		/// <param name="triVerts">The face's corner vertex indices; -1 in w marks a triangle.</param>
		/// <param name="edgeOffsets">First new vertex index contributed by each edge.</param>
		/// <param name="edgeFwd">Whether each edge's new vertices run in increasing index order.</param>
		/// <param name="interiorOffset">First new vertex index for this face's interior vertices.</param>
		/// <returns>The triangulation rewritten in the mesh's own vertex indices.</returns>
		public List<IVec3> Reindex(IVec4 triVerts, IVec4 edgeOffsets, BVec4 edgeFwd, int interiorOffset)
		{
			List<int> newVerts = new List<int>(this.VertBary.Count);
			IVec4 triIdx = this.Idx;
			IVec4 outTri = new IVec4(0, 1, 2, 3);

			if (triVerts[3] < 0 && this.Idx[1] != SubdivisionPartition.Next3(this.Idx[0]))
			{
				triIdx = new IVec4(this.Idx[2], this.Idx[0], this.Idx[1], this.Idx[3]);
				edgeFwd = !edgeFwd;

				// swap outTri[0] and outTri[1]
				int tmp = outTri[0];
				outTri[0] = outTri[1];
				outTri[1] = tmp;
			}

			for (int i = 0; i < 4; i++)
			{
				if (triVerts[triIdx[i]] >= 0)
				{
					newVerts.Add(triVerts[triIdx[i]]);
				}
			}

			for (int i = 0; i < 4; i++)
			{
				int n = this.SortedDivisions[i] - 1;
				int offset = edgeOffsets[this.Idx[i]] + (edgeFwd[this.Idx[i]] ? 0 : n - 1);
				for (int j = 0; j < n; j++)
				{
					newVerts.Add(offset);
					offset += edgeFwd[this.Idx[i]] ? 1 : -1;
				}
			}

			int tailOffset = interiorOffset - newVerts.Count;
			int old = newVerts.Count;
			newVerts.Resize(this.VertBary.Count, 0);

			// The Rust writes this as `new_verts[old..]`, which panics if the resize
			// truncated (VertBary.Count < old); the loop below silently does nothing in
			// that case. The difference is unreachable: `old` is always exactly
			// InteriorOffset() — the four corner pushes plus the per-edge
			// SortedDivisions[i] - 1 pushes sum to it — and VertBary.Count is
			// InteriorOffset() + NumInterior() by construction, with NumInterior() never
			// negative. The one Partition that could break that arithmetic is the empty
			// one GetPartition returns for the wrong side of a quad, and it never reaches
			// Reindex: Subdivide skips those faces on `halfedges[0] < 0`.
			for (int j = 0; old + j < newVerts.Count; j++)
			{
				newVerts[old + j] = old + j + tailOffset;
			}

			int numTri = this.TriVert.Count;
			List<IVec3> newTriVert = new List<IVec3>(numTri);
			newTriVert.Resize(numTri, default(IVec3));
			for (int tri = 0; tri < numTri; tri++)
			{
				// Both sides are pulled into locals first: List<T>'s indexer hands back a
				// copy of a struct element, so writing through it would update a temporary.
				IVec3 source = this.TriVert[tri];
				IVec3 t = newTriVert[tri];
				for (int j = 0; j < 3; j++)
				{
					t[outTri[j]] = newVerts[source[j]];
				}

				newTriVert[tri] = t;
			}

			return newTriVert;
		}

		/// <summary>
		/// The Rust's derived <c>Clone</c>: a deep copy, both vectors included.
		/// </summary>
		/// <returns>A partition sharing no mutable state with this one.</returns>
		public Partition Clone()
		{
			Partition copy = new Partition();
			copy.Idx = this.Idx;
			copy.SortedDivisions = this.SortedDivisions;
			copy.VertBary.AddRange(this.VertBary);
			copy.TriVert.AddRange(this.TriVert);
			return copy;
		}

		/// <summary>Port of C++ Partition::GetCachedPartition(ivec4 n).</summary>
		private static Partition GetCachedPartition(IVec4 n)
		{
			// Check cache
			lock (PartitionCacheLock)
			{
				if (PartitionCache.TryGetValue(n, out Partition? cached))
				{
					return cached.Clone();
				}
			}

			Partition partition = new Partition();
			partition.SortedDivisions = n;

			if (n[3] > 0)
			{
				// quad
				partition.VertBary.Add(new Vec4(1.0, 0.0, 0.0, 0.0));
				partition.VertBary.Add(new Vec4(0.0, 1.0, 0.0, 0.0));
				partition.VertBary.Add(new Vec4(0.0, 0.0, 1.0, 0.0));
				partition.VertBary.Add(new Vec4(0.0, 0.0, 0.0, 1.0));

				IVec4 edgeOffsets = default(IVec4);
				edgeOffsets[0] = 4;
				for (int i = 0; i < 4; i++)
				{
					if (i > 0)
					{
						edgeOffsets[i] = edgeOffsets[i - 1] + n[i - 1] - 1;
					}

					Vec4 nextBary = partition.VertBary[(i + 1) % 4];
					for (int j = 1; j < n[i]; j++)
					{
						partition.VertBary.Add(SubdivisionPartition.LerpVec4(
							partition.VertBary[i],
							nextBary,
							(double)j / (double)n[i]));
					}
				}

				IVec4 nMinus1 = new IVec4(n[0] - 1, n[1] - 1, n[2] - 1, n[3] - 1);
				PartitionQuad(
					partition.TriVert,
					partition.VertBary,
					new IVec4(0, 1, 2, 3),
					edgeOffsets,
					nMinus1,
					BVec4.Splat(true));
			}
			else
			{
				// triangle
				partition.VertBary.Add(new Vec4(1.0, 0.0, 0.0, 0.0));
				partition.VertBary.Add(new Vec4(0.0, 1.0, 0.0, 0.0));
				partition.VertBary.Add(new Vec4(0.0, 0.0, 1.0, 0.0));

				for (int i = 0; i < 3; i++)
				{
					Vec4 nextBary = partition.VertBary[(i + 1) % 3];
					for (int j = 1; j < n[i]; j++)
					{
						partition.VertBary.Add(SubdivisionPartition.LerpVec4(
							partition.VertBary[i],
							nextBary,
							(double)j / (double)n[i]));
					}
				}

				IVec4 edgeOffsets = new IVec4(3, 3 + n[0] - 1, 3 + n[0] - 1 + n[1] - 1, 0);

				double f = (double)((n[2] * n[2]) + (n[0] * n[0]));
				if (n[1] == 1)
				{
					if (n[0] == 1)
					{
						partition.TriVert.Add(new IVec3(0, 1, 2));
					}
					else
					{
						PartitionFan(
							partition.TriVert,
							new IVec3(0, 1, 2),
							n[0] - 1,
							edgeOffsets[0]);
					}
				}
				else if ((double)(n[1] * n[1]) > f - (Math.Sqrt(2.0) * (double)n[0] * (double)n[2]))
				{
					// acute-ish
					partition.TriVert.Add(new IVec3(edgeOffsets[1] - 1, 1, edgeOffsets[1]));
					PartitionQuad(
						partition.TriVert,
						partition.VertBary,
						new IVec4(edgeOffsets[1] - 1, edgeOffsets[1], 2, 0),
						new IVec4(-1, edgeOffsets[1] + 1, edgeOffsets[2], edgeOffsets[0]),
						new IVec4(0, n[1] - 2, n[2] - 1, n[0] - 2),
						BVec4.Splat(true));
				}
				else
				{
					// obtuse -> split into two acute
					int ns = Math.Min(
						n[0] - 2,
						(int)Math.Round((f - (double)(n[1] * n[1])) / (2.0 * (double)n[0]), MidpointRounding.AwayFromZero));

					// MaxF64, not Math.Max: Rust's f64::max returns the *other* operand when
					// one is NaN, and the sqrt below goes NaN whenever ns exceeds n[2].
					int nh = (int)MaxF64(1.0, Math.Round(Math.Sqrt((double)((n[2] * n[2]) - (ns * ns))), MidpointRounding.AwayFromZero));

					int hOffset = partition.VertBary.Count;
					Vec4 middleBary = partition.VertBary[edgeOffsets[0] + ns - 1];
					for (int j = 1; j < nh; j++)
					{
						partition.VertBary.Add(SubdivisionPartition.LerpVec4(
							partition.VertBary[2],
							middleBary,
							(double)j / (double)nh));
					}

					partition.TriVert.Add(new IVec3(edgeOffsets[1] - 1, 1, edgeOffsets[1]));
					PartitionQuad(
						partition.TriVert,
						partition.VertBary,
						new IVec4(
							edgeOffsets[1] - 1,
							edgeOffsets[1],
							2,
							edgeOffsets[0] + ns - 1),
						new IVec4(-1, edgeOffsets[1] + 1, hOffset, edgeOffsets[0] + ns),
						new IVec4(0, n[1] - 2, nh - 1, n[0] - ns - 2),
						BVec4.Splat(true));

					if (n[2] == 1)
					{
						PartitionFan(
							partition.TriVert,
							new IVec3(0, edgeOffsets[0] + ns - 1, 2),
							ns - 1,
							edgeOffsets[0]);
					}
					else if (ns == 1)
					{
						partition.TriVert.Add(new IVec3(hOffset, 2, edgeOffsets[2]));
						PartitionQuad(
							partition.TriVert,
							partition.VertBary,
							new IVec4(hOffset, edgeOffsets[2], 0, edgeOffsets[0]),
							new IVec4(-1, edgeOffsets[2] + 1, -1, hOffset + nh - 2),
							new IVec4(0, n[2] - 2, ns - 1, nh - 2),
							new BVec4(true, true, true, false));
					}
					else
					{
						partition.TriVert.Add(new IVec3(hOffset - 1, 0, edgeOffsets[0]));
						PartitionQuad(
							partition.TriVert,
							partition.VertBary,
							new IVec4(hOffset - 1, edgeOffsets[0], edgeOffsets[0] + ns - 1, 2),
							new IVec4(-1, edgeOffsets[0] + 1, hOffset + nh - 2, edgeOffsets[2]),
							new IVec4(0, ns - 2, nh - 1, n[2] - 2),
							new BVec4(true, true, false, true));
					}
				}
			}

			// Store in cache
			lock (PartitionCacheLock)
			{
				PartitionCache[n] = partition.Clone();
			}

			return partition;
		}

		private static void SwapI32(ref IVec4 sortedDiv, ref IVec4 triIdx, int a, int b)
		{
			int tmp = sortedDiv[a];
			sortedDiv[a] = sortedDiv[b];
			sortedDiv[b] = tmp;
			tmp = triIdx[a];
			triIdx[a] = triIdx[b];
			triIdx[b] = tmp;
		}

		/// <summary>Port of C++ Partition::PartitionFan().</summary>
		private static void PartitionFan(List<IVec3> triVert, IVec3 cornerVerts, int added, int edgeOffset)
		{
			int last = cornerVerts[0];
			for (int i = 0; i < added; i++)
			{
				int next = edgeOffset + i;
				triVert.Add(new IVec3(last, next, cornerVerts[2]));
				last = next;
			}

			triVert.Add(new IVec3(last, cornerVerts[1], cornerVerts[2]));
		}

		/// <summary>Port of C++ Partition::PartitionQuad().</summary>
		private static void PartitionQuad(
			List<IVec3> triVert,
			List<Vec4> vertBary,
			IVec4 cornerVerts,
			IVec4 edgeOffsets,
			IVec4 edgeAdded,
			BVec4 edgeFwd)
		{
			int GetEdgeVert(int edge, int idx)
			{
				return edgeOffsets[edge] + ((edgeFwd[edge] ? 1 : -1) * idx);
			}

			Debug.Assert(
				edgeAdded[0] >= 0 && edgeAdded[1] >= 0 && edgeAdded[2] >= 0 && edgeAdded[3] >= 0,
				"negative divisions!");

			int corner = -1;
			int last = 3;
			int maxEdge = -1;
			for (int i = 0; i < 4; i++)
			{
				if (corner == -1 && edgeAdded[i] == 0 && edgeAdded[last] == 0)
				{
					corner = i;
				}

				if (edgeAdded[i] > 0)
				{
					maxEdge = maxEdge == -1 ? i : -2;
				}

				last = i;
			}

			if (corner >= 0)
			{
				// terminate
				if (maxEdge >= 0)
				{
					IVec4 edge = default(IVec4);
					for (int j = 0; j < 4; j++)
					{
						edge[j] = (j + maxEdge) % 4;
					}

					int middle = edgeAdded[maxEdge] / 2;
					triVert.Add(new IVec3(
						cornerVerts[edge[2]],
						cornerVerts[edge[3]],
						GetEdgeVert(maxEdge, middle)));
					int lastV = cornerVerts[edge[0]];
					for (int i = 0; i <= middle; i++)
					{
						int next = GetEdgeVert(maxEdge, i);
						triVert.Add(new IVec3(cornerVerts[edge[3]], lastV, next));
						lastV = next;
					}

					lastV = cornerVerts[edge[1]];
					for (int i = edgeAdded[maxEdge] - 1; i >= middle; i--)
					{
						int next = GetEdgeVert(maxEdge, i);
						triVert.Add(new IVec3(cornerVerts[edge[2]], next, lastV));
						lastV = next;
					}
				}
				else
				{
					int cornerU = corner;
					int sideVert = cornerVerts[0]; // initial value unused
					for (int j = 1; j <= 2; j++)
					{
						int side = (cornerU + j) % 4;
						if (j == 2 && edgeAdded[side] > 0)
						{
							triVert.Add(new IVec3(
								cornerVerts[side],
								GetEdgeVert(side, 0),
								sideVert));
						}
						else
						{
							sideVert = cornerVerts[side];
						}

						for (int i = 0; i < edgeAdded[side]; i++)
						{
							int nextVert = GetEdgeVert(side, i);
							triVert.Add(new IVec3(cornerVerts[cornerU], sideVert, nextVert));
							sideVert = nextVert;
						}

						if (j == 2 || edgeAdded[side] == 0)
						{
							triVert.Add(new IVec3(
								cornerVerts[cornerU],
								sideVert,
								cornerVerts[(cornerU + j + 1) % 4]));
						}
					}
				}

				return;
			}

			// recursively partition
			int partitions = 1 + Math.Min(edgeAdded[1], edgeAdded[3]);
			IVec4 newCornerVerts = new IVec4(cornerVerts[1], -1, -1, cornerVerts[0]);
			IVec4 newEdgeOffsets = new IVec4(
				edgeOffsets[1],
				-1,
				GetEdgeVert(3, edgeAdded[3] + 1),
				edgeOffsets[0]);
			IVec4 newEdgeAdded = new IVec4(0, -1, 0, edgeAdded[0]);
			BVec4 newEdgeFwd = new BVec4(edgeFwd[1], true, edgeFwd[3], edgeFwd[0]);

			for (int i = 1; i < partitions; i++)
			{
				int cornerOffset1 = (edgeAdded[1] * i) / partitions;
				int cornerOffset3 = edgeAdded[3] - 1 - ((edgeAdded[3] * i) / partitions);
				int nextOffset1 = GetEdgeVert(1, cornerOffset1 + 1);
				int nextOffset3 = GetEdgeVert(3, cornerOffset3 + 1);
				int added = (int)Math.Round(
					(double)edgeAdded[0] + (((double)edgeAdded[2] - (double)edgeAdded[0]) * ((double)i / (double)partitions)),
					MidpointRounding.AwayFromZero);

				newCornerVerts[1] = GetEdgeVert(1, cornerOffset1);
				newCornerVerts[2] = GetEdgeVert(3, cornerOffset3);
				newEdgeAdded[0] = Math.Abs(nextOffset1 - newEdgeOffsets[0]) - 1;
				newEdgeAdded[1] = added;
				newEdgeAdded[2] = Math.Abs(nextOffset3 - newEdgeOffsets[2]) - 1;
				newEdgeOffsets[1] = vertBary.Count;
				newEdgeOffsets[2] = nextOffset3;

				for (int j = 0; j < added; j++)
				{
					vertBary.Add(SubdivisionPartition.LerpVec4(
						vertBary[newCornerVerts[1]],
						vertBary[newCornerVerts[2]],
						(double)(j + 1) / (double)(added + 1)));
				}

				PartitionQuad(
					triVert,
					vertBary,
					newCornerVerts,
					newEdgeOffsets,
					newEdgeAdded,
					newEdgeFwd);

				newCornerVerts[0] = newCornerVerts[1];
				newCornerVerts[3] = newCornerVerts[2];
				newEdgeAdded[3] = newEdgeAdded[1];
				newEdgeOffsets[0] = nextOffset1;
				newEdgeOffsets[3] = newEdgeOffsets[1] + newEdgeAdded[1] - 1;
				newEdgeFwd[3] = false;
			}

			newCornerVerts[1] = cornerVerts[2];
			newCornerVerts[2] = cornerVerts[3];
			newEdgeOffsets[1] = edgeOffsets[2];
			newEdgeAdded[0] = edgeAdded[1] - Math.Abs(newEdgeOffsets[0] - edgeOffsets[1]);
			newEdgeAdded[1] = edgeAdded[2];
			newEdgeAdded[2] = Math.Abs(newEdgeOffsets[2] - edgeOffsets[3]) - 1;
			newEdgeOffsets[2] = edgeOffsets[3];
			newEdgeFwd[1] = edgeFwd[2];

			PartitionQuad(
				triVert,
				vertBary,
				newCornerVerts,
				newEdgeOffsets,
				newEdgeAdded,
				newEdgeFwd);
		}
	}

	// ---------------------------------------------------------------------------
	// Helper functions
	// ---------------------------------------------------------------------------

	/// <summary>
	/// The module-level free functions of subdivision_partition.rs.
	/// </summary>
	internal static class SubdivisionPartition
	{
		/// <summary>
		/// Next index in a triangle (0→1→2→0).
		/// </summary>
		/// <remarks>
		/// Computes the same thing as <see cref="Types.Next3(int)"/>; it exists separately
		/// because subdivision_partition.rs declares its own <c>next3</c> and
		/// subdivision.rs imports *that* one, so the call graph is kept as the Rust has it.
		/// </remarks>
		/// <param name="i">The index.</param>
		/// <returns>The next index.</returns>
		public static int Next3(int i)
		{
			return i == 2 ? 0 : i + 1;
		}

		/// <summary>
		/// Linear interpolation between two barycentric coordinate vectors.
		/// </summary>
		/// <remarks>
		/// Written <c>a + (b - a) * t</c>, which is *not* the <c>a * (1 - t) + b * t</c>
		/// form the porting rules mandate for linalg's <c>lerp</c>. See this file's header:
		/// subdivision_partition.rs spells this out longhand in the other form, and the two
		/// disagree in the last bit, so the Rust's spelling is the specification here.
		/// </remarks>
		/// <param name="a">Value at t = 0.</param>
		/// <param name="b">Value at t = 1.</param>
		/// <param name="t">The interpolation parameter.</param>
		/// <returns>The interpolated vector.</returns>
		public static Vec4 LerpVec4(Vec4 a, Vec4 b, double t)
		{
			return new Vec4(
				a.X + ((b.X - a.X) * t),
				a.Y + ((b.Y - a.Y) * t),
				a.Z + ((b.Z - a.Z) * t),
				a.W + ((b.W - a.W) * t));
		}
	}
}
