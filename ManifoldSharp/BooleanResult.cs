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

// Phase 11: Boolean Result — face assembly from intersection data
//
// C++ source: src/boolean_result.cpp (889 lines)
//
// Takes intersection data from Boolean3 and assembles the output mesh faces.
// The algorithm:
// 1. Convert winding numbers to inclusion values based on operation type
// 2. Compute vertex remapping via exclusive scan (with absolute sum)
// 3. Create output vertices (retained + new intersection verts)
// 4. Build edge maps for partial and new edges
// 5. Size output (count sides per face, allocate halfedges)
// 6. Assemble edges (partial, new, whole)
// 7. Triangulate polygonal faces (Face2Tri)
// 8. Create properties via barycentric interpolation
// 9. Finalize: simplify topology, sort geometry
//
// (The header above is boolean_result.rs's, verbatim; its phase number is the
// Rust port's own — it names manifold-rust's plan, not this port's.)
//
// ── C# port notes ────────────────────────────────────────────────────────────
// boolean_result.rs ends by pulling in boolean_result_assemble.rs and
// re-exporting its `boolean_result` / `boolean_result_with_token`. C# has no
// re-export, so those two stay members of `BooleanResultAssemble` and callers
// name that class directly. Everything else in boolean_result.rs lands on
// `BooleanResult` (this file and BooleanResult.Edges.cs).
//
// The edge maps were Rust `BTreeMap<K, Vec<EdgePos>>` (here `SortedDictionary`)
// until manifold-rust 6e127b5. They are now `(key, EdgePos)` lists in push
// order, stably sorted by key and cut into runs (EdgeGroups below); the
// consumers read keys in ascending order and each run in push order, exactly as
// they read the map. Key order still reaches the output — every one of these is
// iterated to emit halfedges — which is why the sort is stable and by key, never
// a hash map's order.
//
// The Rust's `OrderedF64` newtype (an `Ord` wrapper whose `cmp` is
// `partial_cmp(..).unwrap_or(Equal)`) is `Polygon.PartialCmp` here, the same
// comparator the polygon triangulator already uses.
//
// ── File split ───────────────────────────────────────────────────────────────
//   BooleanResult.cs        this file — EdgePos, the exclusive scan, SizeOutput
//                           and AddNewEdgeVerts
//   BooleanResult.Edges.cs  CppPartition, PairUp and the three Append*Edges

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	// -----------------------------------------------------------------------
	// EdgePos — position of a vertex along an edge for pairing
	// -----------------------------------------------------------------------

	/// <summary>
	/// One endpoint candidate along an edge of the output, carrying where it sits on
	/// that edge and whether it starts or ends a halfedge.
	/// </summary>
	internal struct EdgePos
	{
		/// <summary>
		/// The Rust's <c>edge_pos</c>: the projection of this vertex along the edge.
		/// Renamed because C# forbids a member spelled like its enclosing type.
		/// </summary>
		public double EdgePosition;

		/// <summary>The output vertex index.</summary>
		public int Vert;

		/// <summary>Which intersection produced it; <c>int.MaxValue</c> for retained verts.</summary>
		public int CollisionId;

		/// <summary>True when this endpoint starts a halfedge rather than ending one.</summary>
		public bool IsStart;

		/// <summary>
		/// Creates an entry.
		/// </summary>
		/// <param name="edgePosition">The projection along the edge.</param>
		/// <param name="vert">The output vertex index.</param>
		/// <param name="collisionId">The producing intersection, or <c>int.MaxValue</c>.</param>
		/// <param name="isStart">Whether this endpoint starts a halfedge.</param>
		public EdgePos(double edgePosition, int vert, int collisionId, bool isStart)
		{
			this.EdgePosition = edgePosition;
			this.Vert = vert;
			this.CollisionId = collisionId;
			this.IsStart = isStart;
		}
	}

	/// <summary>
	/// The free functions of <c>boolean_result.rs</c>: output sizing and edge assembly.
	/// </summary>
	internal static partial class BooleanResult
	{
		// -------------------------------------------------------------------
		// AbsSum — exclusive scan combiner
		// -------------------------------------------------------------------

		internal static int AbsSum(int a, int b)
		{
			return Math.Abs(a) + Math.Abs(b);
		}

		/// <summary>
		/// Exclusive scan with abs_sum combiner. Output[i] = init + sum(|input[0..i]|).
		/// </summary>
		internal static List<int> ExclusiveScanAbs(IReadOnlyList<int> input, int init)
		{
			List<int> result = new List<int>(input.Count);
			int acc = init;
			foreach (int v in input)
			{
				result.Add(acc);
				acc = AbsSum(acc, v);
			}

			return result;
		}

		// -------------------------------------------------------------------
		// SizeOutput — compute output face sizes and allocate halfedges
		// -------------------------------------------------------------------

		/// <summary>
		/// Counts sides per face and allocates output halfedges.
		/// Returns (face_edge, face_pq2r).
		/// </summary>
		internal static (List<int> FaceEdge, List<int> FacePq2r) SizeOutput(
			ManifoldImpl outR,
			ManifoldImpl inP,
			ManifoldImpl inQ,
			IReadOnlyList<int> i03,
			IReadOnlyList<int> i30,
			IReadOnlyList<int> i12,
			IReadOnlyList<int> i21,
			IReadOnlyList<IVec2> p1q2,
			IReadOnlyList<IVec2> p2q1,
			bool invertQ)
		{
			int numTriP = inP.NumTri();
			int numTriQ = inQ.NumTri();
			int[] sidesPerFace = new int[numTriP + numTriQ];

			// Count retained vertex contributions
			for (int face = 0; face < numTriP; face++)
			{
				for (int j = 0; j < 3; j++)
				{
					int v = inP.Halfedge[(3 * face) + j].StartVert;
					sidesPerFace[face] += Math.Abs(i03[v]);
				}
			}

			for (int face = 0; face < numTriQ; face++)
			{
				for (int j = 0; j < 3; j++)
				{
					int v = inQ.Halfedge[(3 * face) + j].StartVert;
					sidesPerFace[numTriP + face] += Math.Abs(i30[v]);
				}
			}

			// Count new intersection vertex contributions
			for (int idx = 0; idx < p1q2.Count; idx++)
			{
				IVec2 pair = p1q2[idx];
				int edgeP = pair.X;
				int faceQ = pair.Y;
				int inclusion = Math.Abs(i12[idx]);
				sidesPerFace[numTriP + faceQ] += inclusion;
				Halfedge half = inP.Halfedge[edgeP];
				sidesPerFace[edgeP / 3] += inclusion;
				sidesPerFace[half.PairedHalfedge / 3] += inclusion;
			}

			for (int idx = 0; idx < p2q1.Count; idx++)
			{
				IVec2 pair = p2q1[idx];
				int faceP = pair.X;
				int edgeQ = pair.Y;
				int inclusion = Math.Abs(i21[idx]);
				sidesPerFace[faceP] += inclusion;
				Halfedge half = inQ.Halfedge[edgeQ];
				sidesPerFace[numTriP + (edgeQ / 3)] += inclusion;
				sidesPerFace[numTriP + (half.PairedHalfedge / 3)] += inclusion;
			}

			// Build face_pq2r: maps input face → output face index
			List<int> facePq2r = new List<int>(numTriP + numTriQ);
			facePq2r.Resize(numTriP + numTriQ, 0);
			int count = 0;
			for (int i = 0; i < sidesPerFace.Length; i++)
			{
				facePq2r[i] = count;
				if (sidesPerFace[i] > 0)
				{
					count += 1;
				}
			}

			int numFaceR = count;

			// Build face normals for output
			outR.FaceNormal.Resize(numFaceR, Vec3.Splat(0.0));
			int outIdx = 0;
			for (int i = 0; i < numTriP; i++)
			{
				if (sidesPerFace[i] > 0)
				{
					outR.FaceNormal[outIdx] = inP.FaceNormal[i];
					outIdx += 1;
				}
			}

			for (int i = 0; i < numTriQ; i++)
			{
				if (sidesPerFace[numTriP + i] > 0)
				{
					Vec3 normal = inQ.FaceNormal[i];
					outR.FaceNormal[outIdx] = invertQ
						? new Vec3(-normal.X, -normal.Y, -normal.Z)
						: normal;
					outIdx += 1;
				}
			}

			// Build face_edge: cumulative edge counts for each output face
			List<int> activeSides = new List<int>();
			foreach (int s in sidesPerFace)
			{
				if (s > 0)
				{
					activeSides.Add(s);
				}
			}

			List<int> faceEdge = new List<int>(activeSides.Count + 1);
			faceEdge.Resize(activeSides.Count + 1, 0);
			for (int i = 0; i < activeSides.Count; i++)
			{
				faceEdge[i + 1] = faceEdge[i] + activeSides[i];
			}

			int numHalfedge = faceEdge[faceEdge.Count - 1];
			outR.Halfedge.Resize(numHalfedge, new Halfedge(-1, -1, -1, -1));

			return (faceEdge, facePq2r);
		}

		// -------------------------------------------------------------------
		// AddNewEdgeVerts — populate edge maps with intersection vertices
		// -------------------------------------------------------------------

		internal static void AddNewEdgeVerts(
			List<(int Key, EdgePos Pos)> edgesP,
			List<((int FaceP, int FaceQ) Key, EdgePos Pos)> edgesNew,
			IReadOnlyList<IVec2> p1q2,
			IReadOnlyList<int> i12,
			IReadOnlyList<int> v12r,
			IReadOnlyList<Halfedge> halfedgeP,
			bool forward,
			int offset,
			CancelToken? token)
		{
			for (int i = 0; i < p1q2.Count; i++)
			{
				// As C++ AddNewEdgeVerts (boolean_result.cpp:276-280); the caller's
				// check after the call discards the partial lists (manifold-rust e81b00f).
				if (Cancel.IsCancelled(token))
				{
					return;
				}

				IVec2 pair = p1q2[i];
				int edgeP = forward ? pair.X : pair.Y;
				int faceQ = forward ? pair.Y : pair.X;
				int vert = v12r[i];
				int inclusion = i12[i];

				Halfedge halfedge = halfedgeP[edgeP];
				(int, int) keyRight = (halfedge.PairedHalfedge / 3, faceQ);
				if (!forward)
				{
					keyRight = (keyRight.Item2, keyRight.Item1);
				}

				(int, int) keyLeft = (edgeP / 3, faceQ);
				if (!forward)
				{
					keyLeft = (keyLeft.Item2, keyLeft.Item1);
				}

				bool direction = inclusion < 0;
				int collisionId = i + offset;

				// C++ captures all three is_start values at array creation time:
				// edgesP: direction
				// edgesNew[keyRight]: direction ^ !forward
				// edgesNew[keyLeft]: direction ^ forward
				bool dirP = direction;
				bool dirRight = direction ^ !forward;
				bool dirLeft = direction ^ forward;

				// Add to edge P's list
				for (int j = 0; j < Math.Abs(inclusion); j++)
				{
					edgesP.Add((edgeP, new EdgePos(0.0, vert + j, collisionId, dirP)));
				}

				// Add to right new edge
				for (int j = 0; j < Math.Abs(inclusion); j++)
				{
					edgesNew.Add((keyRight, new EdgePos(0.0, vert + j, collisionId, dirRight)));
				}

				// Add to left new edge
				for (int j = 0; j < Math.Abs(inclusion); j++)
				{
					edgesNew.Add((keyLeft, new EdgePos(0.0, vert + j, collisionId, dirLeft)));
				}
			}
		}
	}

	/// <summary>
	/// Runs of <see cref="EdgePos"/> sharing a key, in ascending key order: the Rust's
	/// <c>EdgeGroups</c>, standing where the <c>BTreeMap&lt;K, Vec&lt;EdgePos&gt;&gt;</c>
	/// stood.
	/// </summary>
	/// <typeparam name="TKey">The edge key: a halfedge index or a (faceP, faceQ) pair.</typeparam>
	internal sealed class EdgeGroups<TKey>
		where TKey : IComparable<TKey>
	{
		private readonly (TKey Key, EdgePos Pos)[] entries;

		/// <summary><c>starts[g]..starts[g + 1]</c> is run <c>g</c> of <c>entries</c>.</summary>
		private readonly List<int> starts;

		/// <summary>
		/// Stably sorts <paramref name="entries"/> (in push order) by key — in parallel from
		/// 10,000 entries, Par.MaybeParSortByKey — and cuts them into runs.
		/// </summary>
		/// <param name="entries">The (key, EdgePos) pairs in push order.</param>
		public EdgeGroups(List<(TKey Key, EdgePos Pos)> entries)
		{
			this.entries = entries.ToArray();
			Par.MaybeParSortByKey(this.entries, 10_000, e => e.Key);
			this.starts = new List<int>();
			for (int i = 0; i < this.entries.Length; i++)
			{
				if (i == 0 || this.entries[i].Key.CompareTo(this.entries[i - 1].Key) != 0)
				{
					this.starts.Add(i);
				}
			}

			this.starts.Add(this.entries.Length);
		}

		/// <summary>The number of runs.</summary>
		public int Count
		{
			get { return this.starts.Count - 1; }
		}

		/// <summary>The key of run <paramref name="g"/>.</summary>
		/// <param name="g">The run, in ascending key order.</param>
		/// <returns>The run's key.</returns>
		public TKey KeyAt(int g)
		{
			return this.entries[this.starts[g]].Key;
		}

		/// <summary>
		/// Replaces <paramref name="run"/>'s contents with run <paramref name="g"/>'s entries,
		/// in push order: the Rust's reused buffer in <c>for_each</c>, which the caller may
		/// modify.
		/// </summary>
		/// <param name="g">The run.</param>
		/// <param name="run">The buffer to fill.</param>
		public void CopyRun(int g, List<EdgePos> run)
		{
			run.Clear();
			for (int i = this.starts[g]; i < this.starts[g + 1]; i++)
			{
				run.Add(this.entries[i].Pos);
			}
		}
	}
}
