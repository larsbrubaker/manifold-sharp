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

// Par.Stages.cs — the par.rs helpers manifold-rust 6e127b5 (PR #8, "Parallel
// boolean stages whose output is the sequential output") added for the boolean
// stages: an ordered filter, a stable sort by key, a for-each over items that own
// disjoint output, and an in-place per-element update. Split from Par.cs, whose
// header carries the contract and the site list, for the 800-line cap.
//
// Each helper keeps Par.cs's claim 1 the way the Rust's does, adapted to Par's
// indexed-write model:
//
//   MaybeParFilter         rayon's `filter(..).collect()` keeps index order; here
//                          the predicate fills a bool per index in parallel and one
//                          sequential pass compacts the hits ascending, so the list
//                          is the sequential filter's by construction.
//   MaybeParSortByKey      a stable sort by key is the unique sort by (key, original
//                          index), a total order, so ANY correct sort of those pairs
//                          gives the same permutation: the sequential branch sorts
//                          them with Array.Sort (which may then be unstable without
//                          harm — CLAUDE.md's "explicit index tie-break comparator"),
//                          the parallel branch sorts fixed chunks the same way and
//                          merges them pairwise. Same pairs, same order.
//   MaybeParForEach        each index owns disjoint output (the Rust hands each item
//                          its own `split_at_mut` slices; C# arrays are shared, so the
//                          disjointness is the call site's proof, stated there).
//   MaybeParFill /         a gather into a List and an in-place update of a List's
//   MaybeParForEachRef     elements: each index writes only its own slot. Workers
//                          take the List's span inside their own chunk (a span cannot
//                          be captured), and nothing resizes the list while they run.

using System.Runtime.InteropServices;

namespace ManifoldSharp
{
	/// <summary>
	/// An update of one element in place, for <see cref="Par.MaybeParForEachRef{T}"/>.
	/// </summary>
	/// <typeparam name="T">The element type.</typeparam>
	/// <param name="item">The element, updated in place.</param>
	internal delegate void RefAction<T>(ref T item);

	/// <content>
	/// The ordered filter, stable sort, for-each and list helpers of par.rs.
	/// </content>
	public static partial class Par
	{
		/// <summary>
		/// Indices per worker chunk in <see cref="MaybeParFill{T}"/> and
		/// <see cref="MaybeParForEachRef{T}"/>, so each worker takes the list's span once
		/// per chunk rather than once per element.
		/// </summary>
		private const int ListChunk = 4096;

		/// <summary>
		/// The indices in <c>0..n</c> where <paramref name="predicate"/> holds, ascending,
		/// testing in parallel when <see cref="ManifoldParallel.Enabled"/> is set and
		/// <c>n &gt;= threshold</c>: the flag half of C++ <c>FlagStore::run_par</c>
		/// (edge_op.cpp:54-84), as Rust <c>maybe_par_filter</c>.
		/// </summary>
		/// <param name="n">Number of indices to test, <c>0..n</c>.</param>
		/// <param name="threshold">Size at or above which an enabled parallel run goes parallel.</param>
		/// <param name="predicate">The per-index test. Must be pure: the parallel loop calls it concurrently.</param>
		/// <returns>The indices that pass, ascending.</returns>
		internal static int[] MaybeParFilter(int n, int threshold, Func<int, bool> predicate)
		{
			if (!ManifoldParallel.ShouldGoParallel(n, threshold))
			{
				List<int> hits = new List<int>();
				for (int i = 0; i < n; i++)
				{
					if (predicate(i))
					{
						hits.Add(i);
					}
				}

				return hits.ToArray();
			}

			bool[] flag = new bool[n];
			RunParallel(0, n, i => flag[i] = predicate(i));
			int count = 0;
			for (int i = 0; i < n; i++)
			{
				if (flag[i])
				{
					count++;
				}
			}

			int[] result = new int[count];
			int w = 0;
			for (int i = 0; i < n; i++)
			{
				if (flag[i])
				{
					result[w++] = i;
				}
			}

			return result;
		}

		/// <summary>
		/// A stable sort of <paramref name="items"/> by <paramref name="key"/>, in parallel
		/// when <see cref="ManifoldParallel.Enabled"/> is set and
		/// <c>items.Length &gt;= threshold</c>: Rust <c>maybe_par_sort_by_key</c>.
		/// </summary>
		/// <remarks>
		/// Both branches sort (key, original index) pairs, a total order, so the
		/// permutation is the unique stable one whichever algorithm finds it, and the
		/// two branches agree element for element. <typeparamref name="TKey"/>'s
		/// <c>CompareTo</c> must itself be a total order (integers and tuples of them, at
		/// every call site).
		/// </remarks>
		/// <typeparam name="T">The element type.</typeparam>
		/// <typeparam name="TKey">The key type.</typeparam>
		/// <param name="items">The array to sort in place.</param>
		/// <param name="threshold">Size at or above which an enabled parallel run goes parallel.</param>
		/// <param name="key">The sort key. Must be pure.</param>
		internal static void MaybeParSortByKey<T, TKey>(T[] items, int threshold, Func<T, TKey> key)
			where TKey : IComparable<TKey>
		{
			int n = items.Length;
			bool parallel = ManifoldParallel.ShouldGoParallel(n, threshold);
			KeyIndex<TKey>[] pairs = new KeyIndex<TKey>[n];
			if (parallel)
			{
				RunParallel(0, n, i => pairs[i] = new KeyIndex<TKey>(key(items[i]), i));
			}
			else
			{
				for (int i = 0; i < n; i++)
				{
					pairs[i] = new KeyIndex<TKey>(key(items[i]), i);
				}
			}

			KeyIndexComparer<TKey> comparer = default;
			if (parallel)
			{
				pairs = ParallelMergeSort(pairs, comparer);
			}
			else
			{
				Array.Sort(pairs, comparer);
			}

			T[] old = (T[])items.Clone();
			if (parallel)
			{
				RunParallel(0, n, i => items[i] = old[pairs[i].Index]);
			}
			else
			{
				for (int i = 0; i < n; i++)
				{
					items[i] = old[pairs[i].Index];
				}
			}
		}

		/// <summary>
		/// Run <paramref name="body"/> for every index in <c>0..n</c>, in parallel when
		/// <see cref="ManifoldParallel.Enabled"/> is set and <c>n &gt;= threshold</c>:
		/// Rust <c>maybe_par_for_each</c>. Every index must write output no other index
		/// reads or writes, so the run order cannot change what is written.
		/// </summary>
		/// <param name="n">Number of indices, <c>0..n</c>.</param>
		/// <param name="threshold">Size at or above which an enabled parallel run goes parallel.</param>
		/// <param name="body">The per-index body, writing only output it owns.</param>
		internal static void MaybeParForEach(int n, int threshold, Action<int> body)
		{
			if (ManifoldParallel.ShouldGoParallel(n, threshold))
			{
				RunParallel(0, n, body);
				return;
			}

			for (int i = 0; i < n; i++)
			{
				body(i);
			}
		}

		/// <summary>
		/// Set <paramref name="list"/> to <c>f(0), .., f(n - 1)</c>, mapping in parallel when
		/// <see cref="ManifoldParallel.Enabled"/> is set and <c>n &gt;= threshold</c>: the
		/// Rust's <c>*v = maybe_par_map(n, threshold, f)</c> written into an existing list.
		/// </summary>
		/// <remarks>
		/// <paramref name="f"/> must not read <paramref name="list"/>: the gathers that use
		/// this read a copy of the old contents.
		/// </remarks>
		/// <typeparam name="T">The element type.</typeparam>
		/// <param name="list">The list to overwrite; its count becomes <paramref name="n"/>.</param>
		/// <param name="n">The new element count.</param>
		/// <param name="threshold">Size at or above which an enabled parallel run goes parallel.</param>
		/// <param name="f">The per-index value. Must be pure.</param>
		internal static void MaybeParFill<T>(List<T> list, int n, int threshold, Func<int, T> f)
		{
			CollectionsMarshal.SetCount(list, n);
			if (!ManifoldParallel.ShouldGoParallel(n, threshold))
			{
				Span<T> span = CollectionsMarshal.AsSpan(list);
				for (int i = 0; i < n; i++)
				{
					span[i] = f(i);
				}

				return;
			}

			RunParallel(0, (n + ListChunk - 1) / ListChunk, c =>
			{
				Span<T> span = CollectionsMarshal.AsSpan(list);
				int end = Math.Min(n, (c + 1) * ListChunk);
				for (int i = c * ListChunk; i < end; i++)
				{
					span[i] = f(i);
				}
			});
		}

		/// <summary>
		/// Apply <paramref name="f"/> to every element of <paramref name="list"/> in place,
		/// in parallel when <see cref="ManifoldParallel.Enabled"/> is set and
		/// <c>list.Count &gt;= threshold</c>: Rust <c>maybe_par_for_each_mut</c>.
		/// <paramref name="f"/> sees only its own element, so the order cannot matter.
		/// </summary>
		/// <typeparam name="T">The element type.</typeparam>
		/// <param name="list">The list whose elements are updated; never resized.</param>
		/// <param name="threshold">Size at or above which an enabled parallel run goes parallel.</param>
		/// <param name="f">The per-element update.</param>
		internal static void MaybeParForEachRef<T>(List<T> list, int threshold, RefAction<T> f)
		{
			int n = list.Count;
			if (!ManifoldParallel.ShouldGoParallel(n, threshold))
			{
				foreach (ref T item in CollectionsMarshal.AsSpan(list))
				{
					f(ref item);
				}

				return;
			}

			RunParallel(0, (n + ListChunk - 1) / ListChunk, c =>
			{
				Span<T> span = CollectionsMarshal.AsSpan(list);
				int end = Math.Min(n, (c + 1) * ListChunk);
				for (int i = c * ListChunk; i < end; i++)
				{
					f(ref span[i]);
				}
			});
		}

		/// <summary>
		/// Sorts <paramref name="pairs"/> by a total order: fixed chunks sorted in parallel,
		/// then merged pairwise in rounds, each round's merges in parallel.
		/// </summary>
		private static KeyIndex<TKey>[] ParallelMergeSort<TKey>(
			KeyIndex<TKey>[] pairs,
			KeyIndexComparer<TKey> comparer)
			where TKey : IComparable<TKey>
		{
			int n = pairs.Length;
			int chunks = Math.Max(1, Math.Min(Environment.ProcessorCount * 2, n / 4096));
			int run = (n + chunks - 1) / chunks;
			RunParallel(0, chunks, c =>
			{
				int start = c * run;
				int length = Math.Min(n, start + run) - start;
				if (length > 1)
				{
					Array.Sort(pairs, start, length, comparer);
				}
			});

			KeyIndex<TKey>[] src = pairs;
			KeyIndex<TKey>[] dst = new KeyIndex<TKey>[n];
			for (int width = run; width < n; width *= 2)
			{
				int w = width;
				KeyIndex<TKey>[] from = src;
				KeyIndex<TKey>[] to = dst;
				int merges = (n + (2 * w) - 1) / (2 * w);
				RunParallel(0, merges, m =>
				{
					int lo = m * 2 * w;
					int mid = Math.Min(n, lo + w);
					int hi = Math.Min(n, lo + (2 * w));
					int a = lo;
					int b = mid;
					int o = lo;
					while (a < mid && b < hi)
					{
						to[o++] = comparer.Compare(from[b], from[a]) < 0 ? from[b++] : from[a++];
					}

					while (a < mid)
					{
						to[o++] = from[a++];
					}

					while (b < hi)
					{
						to[o++] = from[b++];
					}
				});
				(src, dst) = (dst, src);
			}

			return src;
		}

		/// <summary>A sort key with the element's original index, its tie-break.</summary>
		private readonly struct KeyIndex<TKey>
		{
			public readonly TKey Key;
			public readonly int Index;

			public KeyIndex(TKey key, int index)
			{
				this.Key = key;
				this.Index = index;
			}
		}

		/// <summary>Key first, then original index: a total order on distinct indices.</summary>
		private readonly struct KeyIndexComparer<TKey> : IComparer<KeyIndex<TKey>>
			where TKey : IComparable<TKey>
		{
			public int Compare(KeyIndex<TKey> x, KeyIndex<TKey> y)
			{
				int c = x.Key.CompareTo(y.Key);
				return c != 0 ? c : x.Index.CompareTo(y.Index);
			}
		}
	}
}
