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

// Par.cs — port of par.rs — determinism-preserving parallel execution helpers
//
// Mirrors the C++ `autoPolicy` pattern (parallel.h): each site opts into
// parallelism above a size threshold, staying sequential for small inputs.
// Unlike upstream MANIFOLD_PAR (which allows nondeterministic vertex order in
// some phases), only sites whose output is provably identical to the
// sequential build are parallelized — per-index maps with indexed writes, and
// collect-then-sort pipelines whose final sort is a total order. This keeps
// the `parallel` feature bit-exact with the sequential reference.
//
// Both bodies here are the Rust's `#[cfg(not(feature = "parallel"))]` variants:
// sequential, in index order. Parallelism arrives in Phase 11 (see
// docs/PORTING_PLAN.md), at which point only the bodies change — the `threshold`
// parameter and the cancellation short-circuit are already in the signatures so
// that every call site ported before then ports unchanged. The doc comments of
// the parallel variants are kept verbatim, because they are the specification
// Phase 11 has to satisfy.
//
// Rust's `Vec<T>` becomes `T[]`, not `List<T>`, and the bodies fill a
// pre-allocated array *by index* rather than pushing. That is deliberate: it is
// already the shape Phase 11 needs, where the determinism contract is "indexed
// writes into a pre-allocated array", so going parallel becomes a change of
// loop and nothing else — no growth, no append order to reason about.

namespace ManifoldSharp
{
	/// <summary>
	/// Determinism-preserving map helpers, the port of Rust's <c>par</c> module.
	/// </summary>
	public static class Par
	{
		/// <summary>
		/// Map <paramref name="f"/> over <c>0..n</c>, in parallel when the parallel
		/// build is enabled and <c>n &gt;= threshold</c>. Results are returned in index
		/// order either way.
		/// </summary>
		/// <typeparam name="T">The mapped element type.</typeparam>
		/// <param name="n">Number of indices to map, <c>0..n</c>.</param>
		/// <param name="threshold">Size at or above which a parallel build goes parallel.</param>
		/// <param name="f">The per-index map. Must be pure: the parallel build calls it concurrently.</param>
		/// <returns>The mapped values, in index order.</returns>
		public static T[] MaybeParMap<T>(int n, int threshold, Func<int, T> f)
		{
			// `threshold` is unused while the port is sequential-only; it is part of
			// the signature so Phase 11 is a body change and not a call-site change.
			_ = threshold;

			T[] result = new T[n];
			for (int i = 0; i < n; i++)
			{
				result[i] = f(i);
			}

			return result;
		}

		/// <summary>
		/// <see cref="MaybeParMap"/> with cooperative cancellation: null means the token
		/// was cancelled and the (necessarily incomplete) results were discarded.
		/// </summary>
		/// <remarks>
		/// <para>
		/// Mirrors the ctx-aware <c>for_each</c> overload in C++ <c>parallel.h:400-437</c>,
		/// whose contract is the same — "only safe when <em>skip the rest of the range</em>
		/// produces a result the caller will discard via a post-loop <c>IsCancelled</c>
		/// check".
		/// </para>
		/// <para>
		/// Two deliberate differences from the C++:
		/// </para>
		/// <list type="bullet">
		/// <item><description>
		/// A null token dispatches straight to <see cref="MaybeParMap"/>, so the
		/// uncancellable path is byte-for-byte the code that ran before cancellation
		/// existed — stricter than C++, which still branches on <c>ctx != nullptr</c>
		/// inside the loop.
		/// </description></item>
		/// <item><description>
		/// With a token we check the flag on <em>every</em> element rather than once per
		/// <c>kSeqCancelChunk</c> (1024) elements. The flag is written at most once, so
		/// its cache line stays shared and the relaxed load is an L1 hit; paying it per
		/// element buys strictly better cancellation latency than the C++ chunking.
		/// </description></item>
		/// </list>
		/// <para>
		/// C# caveat on that second point, which Phase 11 leans on: the flag is
		/// <c>volatile</c>, because .NET has no relaxed atomic load and a plain field read
		/// may be hoisted out of this very loop. Each element therefore pays an
		/// <em>acquire</em> load (<c>ldar</c> on ARM64), not the free <c>ldr</c> the Rust
		/// gets from <c>Relaxed</c>. The cache-line half of the argument still holds — the
		/// flag is written at most once, so it stays shared and the load stays an L1 hit —
		/// but the barrier is not free, and "cheaper than chunking" is a claim to measure
		/// here rather than inherit.
		/// </para>
		/// </remarks>
		/// <typeparam name="T">The mapped element type.</typeparam>
		/// <param name="n">Number of indices to map, <c>0..n</c>.</param>
		/// <param name="threshold">Size at or above which a parallel build goes parallel.</param>
		/// <param name="token">The cancellation token, or null for an uncancellable run.</param>
		/// <param name="f">The per-index map. Must be pure: the parallel build calls it concurrently.</param>
		/// <returns>The mapped values in index order, or null if the token was cancelled.</returns>
		public static T[]? MaybeParMapCt<T>(int n, int threshold, CancelToken? token, Func<int, T> f)
		{
			if (token is null)
			{
				return MaybeParMap(n, threshold, f);
			}

			// Checked BEFORE the allocation, not just inside the loop. Rust's
			// short-circuiting collect never reserves anything for a sequence whose
			// first item is `None`, so a pre-cancelled token costs it nothing; hoisting
			// the check keeps that true here, where `new T[n]` would otherwise zero-fill
			// the whole buffer first (80 MB at n = 10M) before discarding it. The
			// deferred `pre_cancelled_token_returns_cancelled_promptly` test measures
			// exactly this ratio, so the allocation is not merely wasteful — it is the
			// thing that test fails on.
			if (token.IsCancelled)
			{
				return null;
			}

			// Rust collects `Option<T>` into `Option<Vec<T>>`, which short-circuits on
			// the first `None` in both rayon and std, so a cancel stops the remaining
			// work instead of merely skipping each element's body. The early return
			// below is that short circuit; the partially filled array is dropped with
			// it, matching the Rust's "the incomplete results were discarded".
			T[] result = new T[n];
			for (int i = 0; i < n; i++)
			{
				if (token.IsCancelled)
				{
					return null;
				}

				result[i] = f(i);
			}

			return result;
		}
	}
}
