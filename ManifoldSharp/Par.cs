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
// Rust's `Vec<T>` becomes `T[]`, not `List<T>`, and the bodies fill a
// pre-allocated array *by index* rather than pushing. That is the whole
// determinism contract: indexed writes into a pre-allocated array, so going
// parallel is a change of loop and nothing else — no growth, no append order to
// reason about.
//
// HOW THE FEATURE IS SPELLED IN C#. Rust gates this behind the compile feature
// `parallel` (Cargo.toml: `parallel = ["dep:rayon"]`), so the choice is frozen
// at build time and both variants of every function exist in the source. A C#
// class library ships one assembly to every consumer, so the analogue is a
// runtime switch instead: `ManifoldParallel.Enabled`, default OFF, set once by
// the host. "Default off" preserves the behavior of every caller that predates
// this, and makes the parallel path something a host opts into knowingly — the
// same posture as a Cargo feature nobody enables by accident.
//
// THE THREE CLAIMS THIS FILE MAKES, in the order they matter:
//
//   1. Results are BIT-IDENTICAL with the switch on or off. Every parallel body
//      writes `result[i]` for its own `i` and reads nothing another index
//      writes, and the map is required to be pure — so the array a parallel run
//      produces is the array a sequential run produces, value for value and bit
//      for bit. This is stricter than upstream C++ MANIFOLD_PAR, which permits
//      nondeterministic vertex ordering in some phases, and it is why routing a
//      site through here is a decision and not a convenience: the six sites
//      docs/PORTING_PLAN.md blesses, plus the robust engine's per-triangle maps,
//      which reach this helper through Progress.MaybeParMapCtProgress exactly as
//      they do in the Rust.
//   2. Cancellation returns the same thing: null when a worker observed the
//      cancelled flag, the complete array otherwise. What differs is *which*
//      indices ran before the stop — see MaybeParMapCt's remarks. Every caller
//      treats null as "discard and report Error.Cancelled", and every caller
//      re-checks the token after the map, so no caller can tell the difference.
//   3. Exceptions propagate unwrapped. Parallel.For would surface a worker's
//      exception inside an AggregateException; the sequential body throws it
//      bare. Sites like the triangulator can throw, so the parallel body
//      unwraps a single inner exception with its original stack rather than
//      changing what a caller catches. Two caveats, both inherent rather than
//      chosen. A throw does not un-run the iterations already in flight, exactly
//      as a cancel does not — same two side effects, same reasoning, spelled out
//      in MaybeParMapCt's remarks. And SEVERAL workers faulting at once has no
//      sequential counterpart at all (the sequential loop stops at the first),
//      so that case keeps the AggregateException: a caller catching a specific
//      type must handle it, and the only site whose map runs caller-supplied
//      code is the SDF voxel fill, whose public entry points say so.
//
// The one documented exception to claim 1 lives at the Minkowski call site, not
// here: parallel hulls consume mesh-ID *values* from the global counter in
// worker order rather than face order. Those handles are opaque and
// process-global (already test-order dependent), and reach no geometry, no
// topology and no assertion. Minkowski.cs's header states it in full.

using System.Runtime.ExceptionServices;

namespace ManifoldSharp
{
	/// <summary>
	/// The process-global switch that stands in for Rust's <c>parallel</c> Cargo feature.
	/// </summary>
	/// <remarks>
	/// <para>
	/// Off by default. Turning it on makes the six determinism-preserving sites in
	/// docs/PORTING_PLAN.md — <c>Intersect12</c>, <c>Winding03</c>, <c>Face2Tri</c>, the SDF
	/// voxel fill, the Minkowski per-face hulls and <c>CalculateVertNormals</c>, plus the
	/// robust engine's per-triangle maps, which share the same helper — spread their
	/// per-index work across the thread pool. Results stay bit-identical either way; see
	/// this file's header for the three claims that carries.
	/// </para>
	/// <para>
	/// <b>Set it once, at startup, before any geometry runs.</b> Rust's is a compile-time
	/// feature and cannot change mid-process; this one can, and nothing synchronizes a flip
	/// against an operation already inside a map. Flipping it while work is in flight is not
	/// unsafe — each map reads the flag once, before it starts — but it makes which loop ran
	/// a race, which is exactly the thing this port refuses to leave unpinned.
	/// </para>
	/// </remarks>
	public static class ManifoldParallel
	{
		/// <summary>
		/// The environment variable that seeds <see cref="Enabled"/> at process start:
		/// <c>1</c> or <c>true</c> (case-insensitive) turns parallelism on, anything else
		/// leaves it off.
		/// </summary>
		/// <remarks>
		/// This is how the whole test suite is run with parallelism forced on without
		/// editing a line of test code — the strongest determinism net the port has, since
		/// every ported expected value then has to survive the parallel loop. Mirrors the
		/// env gating <see cref="Timing"/> already uses.
		/// <para>
		/// Setting it to anything else — <c>0</c>, say — is not the same as leaving it
		/// unset, even though both leave <see cref="Enabled"/> false: see
		/// <see cref="ConfiguredByEnvironment"/>, which is what stops a host's own default
		/// from overwriting an explicit "off".
		/// </para>
		/// </remarks>
		public const string EnabledEnvironmentVariable = "MANIFOLD_PARALLEL";

		// Read once, with the flag, so the two can never disagree about the same startup.
		private static readonly bool SeededByEnvironment =
			Environment.GetEnvironmentVariable(EnabledEnvironmentVariable) is not null;

		// `volatile` for the same reason CancelToken's flag is: .NET has no relaxed
		// atomic, and a plain static bool read may be hoisted out of a caller's loop.
		// The extra ordering is free here — the flag is written approximately once per
		// process and read once per map, never per element.
		private static volatile bool enabled = ReadEnvironmentDefault();

		/// <summary>
		/// Whether the blessed sites run their per-index work in parallel.
		/// </summary>
		/// <value>
		/// False unless <see cref="EnabledEnvironmentVariable"/> asked otherwise at process
		/// start, or a host has set it.
		/// </value>
		public static bool Enabled
		{
			get { return enabled; }
			set { enabled = value; }
		}

		/// <summary>
		/// Whether <see cref="EnabledEnvironmentVariable"/> was <em>present</em> in the
		/// environment when this class initialized — regardless of what it was set to.
		/// </summary>
		/// <remarks>
		/// <para>
		/// The distinction a host needs, and the one <see cref="Enabled"/> alone cannot
		/// make: <c>MANIFOLD_PARALLEL=0</c> and an unset variable both leave
		/// <see cref="Enabled"/> false, but only the first is somebody asking for
		/// sequential execution. A host that installs its own default — agg-sharp's
		/// <c>ManifoldKernel</c> turns parallelism on for every non-browser process — must
		/// not overwrite an explicit request, or the environment variable silently stops
		/// working and the "both configurations must be green" discipline in CLAUDE.md
		/// becomes a run of the same configuration twice.
		/// </para>
		/// <para>
		/// So the contract is: when this is true the environment has spoken and
		/// <see cref="Enabled"/> already reflects it; when it is false, a host default is
		/// free to apply. Read it before assigning <see cref="Enabled"/>, not after.
		/// </para>
		/// </remarks>
		public static bool ConfiguredByEnvironment
		{
			get { return SeededByEnvironment; }
		}

		/// <summary>
		/// The <c>autoPolicy</c> decision: parallel only when the host asked for it
		/// <em>and</em> the input is big enough to pay for the dispatch.
		/// </summary>
		/// <param name="n">Number of indices to map.</param>
		/// <param name="threshold">The site's size threshold, from its call site.</param>
		/// <returns>True when the parallel loop should be used.</returns>
		internal static bool ShouldGoParallel(int n, int threshold)
		{
			return enabled && n >= threshold;
		}

		private static bool ReadEnvironmentDefault()
		{
			string? configured = Environment.GetEnvironmentVariable(EnabledEnvironmentVariable);
			return string.Equals(configured, "1", StringComparison.Ordinal)
				|| string.Equals(configured, "true", StringComparison.OrdinalIgnoreCase);
		}
	}

	/// <summary>
	/// Determinism-preserving map helpers, the port of Rust's <c>par</c> module.
	/// </summary>
	public static class Par
	{
		/// <summary>
		/// Map <paramref name="f"/> over <c>0..n</c>, in parallel when
		/// <see cref="ManifoldParallel.Enabled"/> is set and <c>n &gt;= threshold</c>.
		/// Results are returned in index order either way.
		/// </summary>
		/// <typeparam name="T">The mapped element type.</typeparam>
		/// <param name="n">Number of indices to map, <c>0..n</c>.</param>
		/// <param name="threshold">Size at or above which an enabled parallel run goes parallel.</param>
		/// <param name="f">The per-index map. Must be pure: the parallel loop calls it concurrently.</param>
		/// <returns>The mapped values, in index order.</returns>
		public static T[] MaybeParMap<T>(int n, int threshold, Func<int, T> f)
		{
			T[] result = new T[n];
			if (ManifoldParallel.ShouldGoParallel(n, threshold))
			{
				RunParallel(0, n, i => result[i] = f(i));
			}
			else
			{
				for (int i = 0; i < n; i++)
				{
					result[i] = f(i);
				}
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
		/// C# caveat on that second point, which the parallel loop leans on: the flag is
		/// <c>volatile</c>, because .NET has no relaxed atomic load and a plain field read
		/// may be hoisted out of this very loop. Each element therefore pays an
		/// <em>acquire</em> load (<c>ldar</c> on ARM64), not the free <c>ldr</c> the Rust
		/// gets from <c>Relaxed</c>. The cache-line half of the argument still holds — the
		/// flag is written at most once, so it stays shared and the load stays an L1 hit —
		/// but the barrier is not free, and "cheaper than chunking" is a claim to measure
		/// here rather than inherit.
		/// </para>
		/// <para>
		/// <b>What the parallel loop changes, and what it does not.</b> Rust's short-circuit
		/// is rayon's <c>collect::&lt;Option&lt;Vec&lt;_&gt;&gt;&gt;</c>, which stops feeding
		/// new work to every thread once one yields <c>None</c> — but cannot recall the
		/// items already running. <c>Parallel.For</c> plus
		/// <c>ParallelLoopState.Stop</c> behaves the same way: iterations that have not
		/// started are abandoned, iterations in flight run to completion, and indices
		/// <em>past</em> the cancelled one may therefore have been mapped. Neither
		/// implementation promises "index k and nothing after it".
		/// </para>
		/// <para>
		/// The observable contract, which both loops honour exactly, is narrower and is
		/// what the six call sites actually depend on:
		/// </para>
		/// <list type="number">
		/// <item><description>
		/// A cancellation any worker observed returns <c>null</c>, and the partially filled
		/// array is dropped with it. No caller ever sees a half-filled result, so there is
		/// no torn state to reason about — the array is local to this call and unreachable
		/// once it is discarded.
		/// </description></item>
		/// <item><description>
		/// A run in which no worker observed the flag returns the complete, bit-identical
		/// array. This includes a cancel that lands after the last element: the sequential
		/// loop has the same window, and every call site closes it with the post-map
		/// <c>Cancel.IsCancelled</c> check the C++ mandates (boolean3.cpp:364), which is why
		/// "a cancelled token can never produce a NoError result" survives here.
		/// </description></item>
		/// <item><description>
		/// Cancellation latency stays in the same class: the flag is polled once per
		/// element on every worker, so the bound is one element's work per core rather than
		/// one element's work.
		/// </description></item>
		/// </list>
		/// <para>
		/// One consequence worth stating because it is invisible from the return value: the
		/// extra in-flight indices mean <paramref name="f"/> may run more times after a
		/// cancel than the sequential loop would. Only side effects could notice, and the
		/// two that exist — the progress counter and the Minkowski mesh-ID counter — are
		/// both already order-independent and both discarded along with the cancelled
		/// result.
		/// </para>
		/// <para>
		/// A <em>throw</em> out of <paramref name="f"/> has the identical shape and the
		/// identical answer: <c>Parallel.For</c> stops handing out iterations but cannot
		/// recall the ones running, so more indices may have been mapped than the
		/// sequential loop would have mapped before it stopped at the first throw. The same
		/// two side effects are the only observers, and the whole result is abandoned
		/// either way. Worth naming here rather than only in <c>RunParallel</c>, because
		/// this is where the reasoning about in-flight iterations lives.
		/// </para>
		/// </remarks>
		/// <typeparam name="T">The mapped element type.</typeparam>
		/// <param name="n">Number of indices to map, <c>0..n</c>.</param>
		/// <param name="threshold">Size at or above which an enabled parallel run goes parallel.</param>
		/// <param name="token">The cancellation token, or null for an uncancellable run.</param>
		/// <param name="f">The per-index map. Must be pure: the parallel loop calls it concurrently.</param>
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
			// the whole buffer first (80 MB at n = 10M) before discarding it.
			// CancelTests.PreCancelledTokenReturnsCancelledPromptly measures exactly this
			// ratio, so the allocation is not merely wasteful — it is the thing that test
			// fails on. (It is no longer deferred; this comment used to say it was.)
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
			if (ManifoldParallel.ShouldGoParallel(n, threshold))
			{
				// `Stop` is rayon's short circuit: it tells Parallel.For to hand out no
				// further iterations, which is precisely as far as either runtime can go
				// (see the remarks above on in-flight iterations). `IsCompleted` is false
				// exactly when some worker called it, so it *is* the "was this cancelled"
				// answer — no second flag, and no way for the two to disagree.
				bool completed = RunParallel(0, n, (i, state) =>
				{
					if (token.IsCancelled)
					{
						state.Stop();
						return;
					}

					result[i] = f(i);
				});

				return completed ? result : null;
			}

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

		/// <summary>
		/// <c>Parallel.For</c> with the sequential loop's exception behavior: a worker's
		/// exception comes out bare, with its original stack, instead of wrapped in the
		/// <see cref="AggregateException"/> the TPL would raise.
		/// </summary>
		/// <remarks>
		/// Claim 3 of this file's header. The maps here call into code that throws — the
		/// triangulator and the hull among them — and a caller that catches a specific
		/// exception type must not start missing it because the host flipped a switch. Only
		/// a single inner exception is unwrapped: several workers failing at once has no
		/// sequential counterpart to be faithful to, so the aggregate goes out intact
		/// rather than picking a winner. <see cref="ExceptionDispatchInfo"/> rather than
		/// <c>throw inner</c>, so the stack trace still points at the worker.
		/// </remarks>
		/// <param name="fromInclusive">First index.</param>
		/// <param name="toExclusive">One past the last index.</param>
		/// <param name="body">The per-index body.</param>
		private static void RunParallel(int fromInclusive, int toExclusive, Action<int> body)
		{
			try
			{
				Parallel.For(fromInclusive, toExclusive, body);
			}
			catch (AggregateException aggregate) when (aggregate.InnerExceptions.Count == 1)
			{
				ExceptionDispatchInfo.Capture(aggregate.InnerExceptions[0]).Throw();
			}
		}

		/// <summary>
		/// <see cref="RunParallel(int, int, Action{int})"/> for a body that can stop the
		/// loop.
		/// </summary>
		/// <param name="fromInclusive">First index.</param>
		/// <param name="toExclusive">One past the last index.</param>
		/// <param name="body">The per-index body, given the loop state.</param>
		/// <returns>True when every index ran; false when a body called <c>Stop</c>.</returns>
		private static bool RunParallel(
			int fromInclusive,
			int toExclusive,
			Action<int, ParallelLoopState> body)
		{
			try
			{
				return Parallel.For(fromInclusive, toExclusive, body).IsCompleted;
			}
			catch (AggregateException aggregate) when (aggregate.InnerExceptions.Count == 1)
			{
				ExceptionDispatchInfo.Capture(aggregate.InnerExceptions[0]).Throw();
				throw; // Unreachable: Throw() above never returns. Satisfies definite assignment.
			}
		}
	}
}
