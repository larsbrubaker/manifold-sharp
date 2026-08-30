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
			List<(IVec2 Pair, int X, Vec3 V)>[]? perEdge =
				Par.MaybeParMapCt(n, 10_000, token, queryIdx =>
				{
					List<(IVec2 Pair, int X, Vec3 V)> local = new List<(IVec2, int, Vec3)>();
					if (!a.Halfedge[queryIdx].IsForward())
					{
						return local;
					}

					Box query = Box.FromPoints(
						a.VertPos[a.Halfedge[queryIdx].StartVert],
						a.VertPos[a.Halfedge[queryIdx].EndVert]);
					collider.CollisionsOne(query, queryIdx, (qi, faceIdx) =>
					{
						(int x, Vec3 v) = Kernel12(qi, faceIdx, a, b, inP, inQ, expandP, forward);
						if (double.IsFinite(v.X))
						{
							IVec2 pair = forward
								? new IVec2(qi, faceIdx)
								: new IVec2(faceIdx, qi);
							local.Add((pair, x, v));
						}
					});
					return local;
				});
			if (perEdge is null)
			{
				return null;
			}

			foreach (List<(IVec2 Pair, int X, Vec3 V)> local in perEdge)
			{
				foreach ((IVec2 pair, int x, Vec3 v) in local)
				{
					result.P1q2.Add(pair);
					result.X12.Add(x);
					result.V12.Add(v);
				}
			}

			// C++'s stated invariant is "every ctx-passing parallel op is followed by
			// IsCancelled to discard partial results" (boolean3.cpp:364). Honouring it
			// here also skips the sort below, which is the longest uninterruptible
			// stretch in this function.
			if (Cancel.IsCancelled(token))
			{
				return null;
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
			List<IVec2> p1q2 = result.P1q2;
			int[] indices = Enumerable.Range(0, result.P1q2.Count)
				.OrderBy(i => Component(p1q2[i], sortIdx))
				.ThenBy(i => Component(p1q2[i], 1 - sortIdx))
				.ToArray();

			List<IVec2> oldP1q2 = new List<IVec2>(result.P1q2);
			List<int> oldX12 = new List<int>(result.X12);
			List<Vec3> oldV12 = new List<Vec3>(result.V12);
			for (int newI = 0; newI < indices.Length; newI++)
			{
				int oldI = indices[newI];
				result.P1q2[newI] = oldP1q2[oldI];
				result.X12[newI] = oldX12[oldI];
				result.V12[newI] = oldV12[oldI];
			}

			return result;
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
