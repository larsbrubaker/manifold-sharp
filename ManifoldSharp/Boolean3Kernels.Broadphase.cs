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

// Boolean3Kernels.Broadphase.cs — the second half of boolean3_kernels.rs:
// Intersect12 and Winding03, the two collider-driven drivers built on the
// kernels in Boolean3Kernels.cs. See that file's header for the module as a
// whole and for why it is split.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <content>
	/// Intersect12 and Winding03 — the broadphase drivers.
	/// </content>
	internal static partial class Boolean3Kernels
	{
		/// <summary>
		/// How many chunks <see cref="Intersect12"/> aims to split its halfedges into.
		/// </summary>
		/// <remarks>
		/// <para>
		/// <b>Nothing here is load-bearing for correctness.</b> Any chunk size produces the
		/// identical output, because chunks are concatenated in index order and each walks
		/// its own edges in ascending order. These two constants buy throughput only, and
		/// they were both chosen by measurement rather than by argument.
		/// </para>
		/// <para>
		/// The chunk is the atomic unit of <em>parallel scheduling</em>, and Intersect12's
		/// per-edge cost is wildly non-uniform: an edge costs one Kernel12 call per collider
		/// candidate, and candidate counts span orders of magnitude between a mesh that
		/// barely touches its partner (2M-triangle sphere difference: 6 291 456 halfedges,
		/// 21 936 candidates in total) and one that is nearly coincident with it
		/// (Generic_Twin_7081: 59 112 halfedges and enough candidates to spend 9 s in the
		/// stage, for 1 024 intersections). So the makespan is set by the unluckiest chunk,
		/// and the chunk count has to stay large relative to the core count. Measured on the
		/// twins union at 10 cores, a fixed 1024-edge chunk cost 10 % against the per-edge
		/// shape it replaced and a 64-edge chunk cost nothing; on the 2M sphere the ordering
		/// reverses and the bigger chunk is 1.6 % better, because there the cost is
		/// per-chunk overhead rather than scheduling.
		/// </para>
		/// <para>
		/// Targeting a fixed <em>number</em> of chunks, with a floor on their size, satisfies
		/// both ends: a mesh gets ~4 096 chunks (615 per core at 10 cores) until that would
		/// make them smaller than 64 edges, below which per-chunk overhead starts to matter
		/// more than balance. The floor is what the small, dense meshes land on and the
		/// target is what the large, sparse ones land on.
		/// </para>
		/// </remarks>
		private const int TargetChunks = 4096;

		/// <summary>Smallest chunk <see cref="Intersect12"/> will use. See <see cref="TargetChunks"/>.</summary>
		private const int MinEdgeChunk = 64;

		/// <summary>
		/// <see cref="Intersect12"/>'s <c>autoPolicy</c> size threshold, in halfedges.
		/// </summary>
		/// <remarks>
		/// The site's threshold has always been 10 000 halfedges and still is; it is divided
		/// by the chunk size at the call site because chunks, not edges, are what the map
		/// iterates now. The division rounds the boundary by less than one chunk, which
		/// cannot be observed in a result: both loops produce bit-identical output, so the
		/// threshold decides only who pays for thread dispatch, never what comes out.
		/// </remarks>
		private const int ParallelEdgeThreshold = 10_000;

		// -------------------------------------------------------------------
		// Intersect12 — find all edge-face intersections using collider broadphase
		// -------------------------------------------------------------------

		/// <summary>
		/// Null means <paramref name="token"/> was cancelled part-way through; the partial
		/// results are discarded, mirroring C++ <c>Intersect12</c> returning
		/// <c>Intersections{}</c> after its post-<c>for_each</c> <c>IsCancelled</c> check
		/// (boolean3.cpp:380-381).
		/// </summary>
		internal static Intersections? Intersect12(
			ManifoldImpl inP,
			ManifoldImpl inQ,
			bool expandP,
			bool forward,
			CancelToken? token)
		{
			// a: edge mesh, b: face mesh
			ManifoldImpl a = forward ? inP : inQ;
			ManifoldImpl b = forward ? inQ : inP;

			// Query b's cached face BVH (built in sort_geometry), as C++ queries
			// b.collider_ — rebuilding here dominated the intersection stage.
			Collider collider = b.Collider;

			Intersections result = new Intersections();

			// For each forward edge of a, query its bounding box against b's face BVH
			// and run kernel12 on each candidate. Per-edge work is independent and the
			// final stable sort below fully orders the unique (edge, face) pairs, so
			// the parallel path is bit-identical to the sequential one.
			int n = a.Halfedge.Count;

			// OUTPUT SHAPE: PER-CHUNK COLLECT, WHERE THE RUST COLLECTS PER EDGE.
			//
			// The Rust (boolean3_kernels.rs:396) maps each halfedge to its own
			// `Vec<([i32;2], i32, Vec3)>` and concatenates the lot; its own PORTING_PLAN
			// names "intersect12's per-edge Vec-of-Vecs (C++ uses counted two-pass
			// output)" as the port's remaining memory overhang against C++. A GC runtime
			// pays far more for that shape than Rust does. docs/BENCHMARKS.md measured the
			// stage at 13.6x/15.3x the Rust on the 2M-triangle sphere difference and 63% of
			// the whole C# boolean, where every other stage sat between 1.5x and 3.4x. The
			// counts say the cost is not geometry: on that case 6 291 456 halfedges produce
			// 21 936 collider candidates and 3 504 intersections. What the shape costs is
			// one `List`, one closure and one delegate allocated per halfedge, and — the
			// expensive part — a live 6.3-million-element array of *references* that every
			// Gen0 collection in the rest of the boolean then has to mark through.
			//
			// The fix is to move the unit of collection from the edge to a CHUNK of edges:
			// one list, one closure and one delegate per chunk (4 096 of each here instead of
			// 6.3 million), and a reference array small enough that marking it costs nothing.
			// The recorder is handed the edge index, so it needs no per-edge state — that is
			// what lets it be hoisted out of the inner loop. TargetChunks documents how the
			// chunk size is picked and why it is not a constant.
			//
			// C++ instead uses a counted two-pass output (count candidates, exclusive-scan
			// into offsets, fill flat arrays), and that form was built and measured here
			// first. It wins the same way on the sphere ladder, but its counting pass is a
			// SECOND BVH traversal, and a boolean whose edges have very many candidates pays
			// for that traversal without getting the allocation back: Generic_Twin_7081
			// regressed 6.9 %/15.7 % under it and does not under this shape.
			// docs/BENCHMARKS.md carries the three-way table.
			//
			// RESULTS ARE BIT-IDENTICAL, and that is structural rather than hoped-for: chunk
			// c walks its own contiguous, ascending block of edges appending to its own list,
			// so concatenating the lists in chunk order yields precisely the edge-ascending,
			// traversal-ordered sequence that concatenating the per-edge lists yielded. Same
			// elements, same order, hence the same tie-break for the stable sort below. No
			// RUST_DIVERGENCES.md entry is due — that ledger records deliberate *behavioral*
			// divergence, and this changes only internal shape.
			//
			// Par's determinism contract (Par.cs, claim 1) is unchanged in substance: each
			// mapped index still writes only what it owns and reads nothing another index
			// writes; the index is now a chunk rather than an edge. What does change is
			// cancellation granularity, addressed at ScanChunk.
			int chunkSize = Math.Max(MinEdgeChunk, n / TargetChunks);
			int chunkCount = (n + chunkSize - 1) / chunkSize;
			List<(IVec2 Pair, int X, Vec3 V)>[]? perChunk = Par.MaybeParMapCt(
				chunkCount,
				(ParallelEdgeThreshold + chunkSize - 1) / chunkSize,
				token,
				chunkIdx => ScanChunk(a, b, inP, inQ, collider, n, chunkIdx, chunkSize, expandP, forward, token));
			if (perChunk is null)
			{
				return null;
			}

			// C++'s stated invariant is "every ctx-passing parallel op is followed by
			// IsCancelled to discard partial results" (boolean3.cpp:364). Honouring it
			// here also skips the concatenation and sort below, which together are the
			// longest uninterruptible stretch in this function — and it is what makes
			// ScanChunk's mid-chunk bail safe, since a chunk that stopped early leaves a
			// short list that this check throws away wholesale.
			if (Cancel.IsCancelled(token))
			{
				return null;
			}

			int total = 0;
			foreach (List<(IVec2 Pair, int X, Vec3 V)> chunk in perChunk)
			{
				total += chunk.Count;
			}

			IVec2[] pairs = new IVec2[total];
			int[] x12 = new int[total];
			Vec3[] v12 = new Vec3[total];
			int w = 0;
			foreach (List<(IVec2 Pair, int X, Vec3 V)> chunk in perChunk)
			{
				foreach ((IVec2 pair, int x, Vec3 v) in chunk)
				{
					pairs[w] = pair;
					x12[w] = x;
					v12[w] = v;
					w++;
				}
			}

			// Sort by edge index for deterministic results.
			//
			// SORT AUDIT (boolean3_kernels.rs:438): Rust `sort_by`, which is STABLE. The
			// comparator is a two-level key on (pair[sortIdx], pair[1 - sortIdx]) and the
			// header claims it "fully orders the unique (edge, face) pairs" — but that is a
			// claim about the *input* having no duplicate pairs, not a property the sort
			// enforces, and a duplicate would land the tie-break back on the collect order.
			// LINQ OrderBy/ThenBy is the documented-stable C# sort and preserves it;
			// List<T>.Sort is introsort and is not usable here.
			int sortIdx = forward ? 0 : 1;
			int[] indices = Enumerable.Range(0, total)
				.OrderBy(i => Component(pairs[i], sortIdx))
				.ThenBy(i => Component(pairs[i], 1 - sortIdx))
				.ToArray();

			// The Rust permutes three Vecs in place against three clones of themselves.
			// Filling the (still empty) result lists in sorted order is the same
			// permutation with the clones deleted: the flat arrays above are already the
			// "old" copies the in-place rewrite needed.
			result.P1q2.Capacity = total;
			result.X12.Capacity = total;
			result.V12.Capacity = total;
			for (int newI = 0; newI < indices.Length; newI++)
			{
				int oldI = indices[newI];
				result.P1q2.Add(pairs[oldI]);
				result.X12.Add(x12[oldI]);
				result.V12.Add(v12[oldI]);
			}

			return result;
		}

		/// <summary>
		/// <see cref="Intersect12"/>'s unit of collection: the intersections found by one
		/// contiguous block of <paramref name="chunkSize"/> halfedges, in edge order.
		/// </summary>
		/// <remarks>
		/// <para>
		/// The whole point of the chunk is that <c>record</c> below is built ONCE here rather
		/// than once per halfedge. It can be, because <c>CollisionsOne</c> hands the edge
		/// index back to the recorder, so the recorder needs no per-edge state — the closure
		/// captures only chunk-invariant things plus the chunk's own output list.
		/// </para>
		/// <para>
		/// <b>Cancellation.</b> <see cref="Par.MaybeParMapCt"/> polls the token once per
		/// mapped index, which is now once per chunk (64 to 1 536 edges, per
		/// <see cref="TargetChunks"/>) rather than once per edge. That is the granularity
		/// C++'s <c>kSeqCancelChunk</c> = 1024 already settles for (parallel.h:424) and the
		/// same responsiveness class — but Par.cs claims something stronger than C++ for its
		/// own loop, so rather than drop that claim at this site the per-element poll is kept
		/// here: the loop below re-checks the flag every edge and stops. Stopping mid-chunk
		/// leaves this list SHORT, which is safe only because the caller re-checks
		/// <c>Cancel.IsCancelled</c> before it reads any of these lists and discards the whole
		/// array when the flag is set. That check is not optional and the call site says so.
		/// </para>
		/// <para>
		/// The poll is hoisted behind <c>token is not null</c> so the uncancellable path pays
		/// one predictable branch per edge rather than a volatile load, the same way
		/// <see cref="Winding03"/>'s union-find loop does.
		/// </para>
		/// </remarks>
		/// <param name="a">The edge mesh.</param>
		/// <param name="b">The face mesh.</param>
		/// <param name="inP">The P operand, as Kernel12 expects it.</param>
		/// <param name="inQ">The Q operand, as Kernel12 expects it.</param>
		/// <param name="collider">The face mesh's cached BVH.</param>
		/// <param name="n">Halfedge count of <paramref name="a"/>.</param>
		/// <param name="chunkIdx">Which block of halfedges to scan.</param>
		/// <param name="chunkSize">How many halfedges a block spans; see <see cref="TargetChunks"/>.</param>
		/// <param name="expandP">Kernel12's expand-P flag.</param>
		/// <param name="forward">True when a is P and b is Q.</param>
		/// <param name="token">Cancellation token, or null.</param>
		/// <returns>The chunk's intersections in edge order, possibly short if the token was
		/// cancelled part-way through.</returns>
		private static List<(IVec2 Pair, int X, Vec3 V)> ScanChunk(
			ManifoldImpl a,
			ManifoldImpl b,
			ManifoldImpl inP,
			ManifoldImpl inQ,
			Collider collider,
			int n,
			int chunkIdx,
			int chunkSize,
			bool expandP,
			bool forward,
			CancelToken? token)
		{
			List<(IVec2 Pair, int X, Vec3 V)> local = new List<(IVec2, int, Vec3)>();
			Action<int, int> record = (qi, faceIdx) =>
			{
				(int x, Vec3 v) = Kernel12(qi, faceIdx, a, b, inP, inQ, expandP, forward);
				if (double.IsFinite(v.X))
				{
					IVec2 pair = forward
						? new IVec2(qi, faceIdx)
						: new IVec2(faceIdx, qi);
					local.Add((pair, x, v));
				}
			};

			bool cancellable = token is not null;
			int start = chunkIdx * chunkSize;
			int end = Math.Min(n, start + chunkSize);
			for (int queryIdx = start; queryIdx < end; queryIdx++)
			{
				if (cancellable && Cancel.IsCancelled(token))
				{
					break;
				}

				if (!a.Halfedge[queryIdx].IsForward())
				{
					continue;
				}

				Box query = Box.FromPoints(
					a.VertPos[a.Halfedge[queryIdx].StartVert],
					a.VertPos[a.Halfedge[queryIdx].EndVert]);
				collider.CollisionsOne(query, queryIdx, record);
			}

			return local;
		}

		// -------------------------------------------------------------------
		// Winding03 — compute winding numbers via flood-fill
		// -------------------------------------------------------------------
		// Groups vertices into connected components along unbroken edges (edges not
		// cut by any intersection). For each component, picks a representative vertex
		// and computes its winding number via kernel02 against all overlapping faces
		// of the other mesh. Then flood-fills that winding number to all vertices in
		// the component.

		/// <summary>
		/// Null means <paramref name="token"/> was cancelled part-way through; the partial
		/// winding numbers are discarded, mirroring C++ <c>Winding03</c> returning
		/// <c>Vec&lt;int&gt;{}</c> at each of its <c>IsCancelled</c> checks
		/// (boolean3.cpp:437, 456, 472, 480).
		/// </summary>
		internal static List<int>? Winding03(
			ManifoldImpl inP,
			ManifoldImpl inQ,
			List<IVec2> p1q2,
			bool expandP,
			bool forward,
			CancelToken? token)
		{
			ManifoldImpl a = forward ? inP : inQ;
			ManifoldImpl b = forward ? inQ : inP;
			int sortIdx = forward ? 0 : 1;

			// Build union-find: unite vertices along unbroken edges. This loop is the
			// Rust counterpart of the ctx-passing `for_each` at boolean3.cpp:425, so it
			// gets the same bounded-latency check — but on a chunk boundary rather than
			// per element, because the body is a cheap binary search and `unite`.
			DisjointSets uA = new DisjointSets((uint)a.VertPos.Count);

			// Hoisted so the uncancellable path pays one predictable branch per edge
			// instead of a modulo: C++ gets the same effect from `ctx == nullptr`
			// folding the check out of the loop entirely (parallel.h:427-430).
			bool cancellable = token is not null;
			for (int edge = 0; edge < a.Halfedge.Count; edge++)
			{
				// C++ `for_each` checks every kSeqCancelChunk (= 1024) elements on the
				// sequential branch (parallel.h:424); same constant, same reason.
				if (cancellable && edge % 1024 == 0 && Cancel.IsCancelled(token))
				{
					return null;
				}

				Halfedge he = a.Halfedge[edge];
				if (!he.IsForward())
				{
					continue;
				}

				// Check if this edge is broken (has an intersection)
				bool isBroken = ContainsKey(p1q2, sortIdx, edge);
				if (!isBroken)
				{
					uA.Unite((uint)he.StartVert, (uint)he.EndVert);
				}
			}

			// Post-loop check, matching C++ boolean3.cpp:437.
			if (Cancel.IsCancelled(token))
			{
				return null;
			}

			// Find unique component representatives.
			//
			// The Rust collects into a `std::collections::HashSet` and iterates it, so
			// `verts` is in RandomState order there and in C# insertion order here. That is
			// not a divergence in results: `w03` below is written by vertex index, each
			// representative appears exactly once, and kernel02 is a pure function of the
			// vertex — so the order this set is drained in reaches nothing.
			HashSet<uint> components = new HashSet<uint>();
			for (int v = 0; v < a.VertPos.Count; v++)
			{
				components.Add(uA.Find((uint)v));
			}

			List<int> verts = components.Select(v => (int)v).ToList();

			// Post-scan check, matching C++ boolean3.cpp:456.
			if (Cancel.IsCancelled(token))
			{
				return null;
			}

			// Query b's cached face BVH (built in sort_geometry), as C++ queries
			// b.collider_.
			Collider collider = b.Collider;

			// For each representative vertex, compute winding number via kernel02
			List<int> w03 = new List<int>(a.VertPos.Count);
			w03.Resize(a.VertPos.Count, 0);

			// Use BVH for winding number queries.
			// The winding number shoots a Z-ray, so we need XY overlap with infinite Z.
			// Build query boxes with the vertex XY position and infinite Z extent.
			List<(int Vi, Box QBox)> queryBoxes = verts
				.Select(vi =>
				{
					Vec3 pt = a.VertPos[vi];
					Box qbox = Box.FromPoints(
						new Vec3(pt.X, pt.Y, double.NegativeInfinity),
						new Vec3(pt.X, pt.Y, double.PositiveInfinity));
					return (vi, qbox);
				})
				.ToList();

			// For each representative vert, query the BVH and sum kernel02 winding
			// contributions. The sums are integers, so accumulation order is
			// irrelevant and the per-vert work can run in parallel bit-exactly.
			(int Vi, int Sum)[]? sums = Par.MaybeParMapCt(queryBoxes.Count, 1_000, token, qi =>
			{
				(int vi, Box qbox) = queryBoxes[qi];
				int sum = 0;
				collider.CollisionsOne(qbox, 0, (_, faceIdx) =>
				{
					(int s02, double z02) = Kernel02(vi, faceIdx, a, b, expandP, forward);
					if (double.IsFinite(z02))
					{
						sum += s02 * (forward ? 1 : -1);
					}
				});
				return (vi, sum);
			});
			if (sums is null)
			{
				return null;
			}

			foreach ((int vi, int sum) in sums)
			{
				w03[vi] += sum;
			}

			// Flood fill: propagate representative's winding number to all component members
			for (int i = 0; i < w03.Count; i++)
			{
				int root = (int)uA.Find((uint)i);
				if (root != i)
				{
					w03[i] = w03[root];
				}
			}

			return w03;
		}

		/// <summary>
		/// One component of a pair. A local because <see cref="IVec2"/>'s indexer returns
		/// <c>ref int</c>, which C# will not resolve against the temporary a
		/// <c>List&lt;IVec2&gt;</c> lookup produces.
		/// </summary>
		private static int Component(IVec2 pair, int i)
		{
			return pair[i];
		}

		/// <summary>
		/// The Rust's <c>p1q2.binary_search_by(|pair| pair[sortIdx].cmp(&amp;edge)).is_ok()</c>:
		/// does any pair in the (already sorted by that component) list carry this edge index?
		/// </summary>
		private static bool ContainsKey(List<IVec2> p1q2, int sortIdx, int edge)
		{
			int lo = 0;
			int hi = p1q2.Count;
			while (lo < hi)
			{
				int mid = lo + ((hi - lo) / 2);
				IVec2 pair = p1q2[mid];
				int key = pair[sortIdx];
				if (key == edge)
				{
					return true;
				}

				if (key < edge)
				{
					lo = mid + 1;
				}
				else
				{
					hi = mid;
				}
			}

			return false;
		}
	}
}
