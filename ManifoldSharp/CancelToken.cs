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

// CancelToken.cs — port of cancel.rs — cooperative cancellation for the
// long-running kernel entry points (boolean, CSG tree evaluation).
//
// Port of the *cancellation* half of the C++ `ExecutionContext` mechanism. The
// progress-reporting half (donePhases / totalPhases / Progress()) is not
// ported: Progress.cs provides progress reporting natively instead, reporting
// named phases plus an intra-phase fraction rather than the C++'s
// whole-pipeline phase count. Both are threaded through the kernel the same
// way — as an `Option<&_>` whose `None` path is the pre-existing code. In C#
// that `Option<&CancelToken>` is a nullable reference, `CancelToken?`.
//
// What the C++ does (cpp-reference/manifold/src/execution_impl.h:81-114):
//   - `ExecutionContext::Impl` holds a single `std::atomic<bool> cancel`,
//     shared through a `shared_ptr` so copies of a context observe each other.
//   - `ExecutionContext::Cancel()` stores `true` with `memory_order_relaxed`
//     (execution_impl.cpp:91-93); cancel is advisory, so it needs no
//     synchronisation with the surrounding data.
//   - `IsCancelled(ctx)` (execution_impl.h:112-114) is the single canonical
//     reader. It returns `false` for a null ctx, which is how the "no
//     cancellation requested" path stays free: `ctx == nullptr` folds the
//     atomic load out of the loop entirely.
//   - Cancel is *sticky*: it is never reset, so once a context is cancelled
//     every later operation through it short-circuits
//     (execution_impl.cpp:30-33 explicitly preserves it across resets).
//   - The observable result of an interrupted operation is an *empty* manifold
//     whose status is `Manifold::Error::Cancelled` — the last enum value, added
//     at the end of the list (manifold.h:124-140). See `ADVANCE_PHASE_OR_RETURN`
//     (execution_impl.h:150-160) and `Boolean3::Result`'s `phase()` lambda
//     (boolean_result.cpp:758-770), both of which do `MakeEmpty(Cancelled)`.
//
// The Rust mirror of `ExecutionContext::Impl*` is `Option<&CancelToken>`:
// `None` is C++'s `nullptr` and costs nothing (no atomic is touched), `Some`
// carries an `Arc<AtomicBool>` that any number of threads may share. The C#
// mirror is `CancelToken?`: `null` is `None`, and a class reference *is* the
// `Arc` — see the ordering note on the flag field below.
//
// Where the checks live (mirroring the C++ sites named above; the C# file names
// are the ones those Rust modules port to):
//   CsgTree.cs            `ToLeafNode` (per stack step), `SimpleBoolean`
//                         (entry), `BatchBoolean` (per round), `BatchUnion`
//                         (per chunk)          <- csg_tree.cpp:172/460/511/752
//   Boolean3.cs           `BooleanWithToken` (entry), `Boolean3.NewWithToken`
//                         (four stage boundaries), plus intra-stage checks in
//                         `Intersect12` / `Winding03`
//                                              <- boolean3.cpp:380/437/456/472/
//                                                 480/530/536/552/558
//   BooleanResult         all eleven phase boundaries between the assembly
//     Assemble.cs         stages, including the final one after SortGeometry
//                                              <- boolean_result.cpp:758-963
//   FaceOp.cs             `Face2TriCt` entry plus per-face triangulation
//                                              <- face_op.cpp:192/290
//
// The invariant those sites buy: **a cancelled token can never produce a
// NoError result.** Every stage of the boolean pipeline is bracketed by a
// check, so a cancel that lands inside a stage is always observed at the next
// boundary and converted to `Error.Cancelled` before the value escapes. It is
// not enough to check "often enough for good latency" — a missed *final* check
// would report success for an operation the caller cancelled.
//
// Deviations from the C++, all in the "checks fewer places" direction, none
// affecting the uncancelled result and none breaking the invariant above:
//   - Progress reporting (donePhases/totalPhases/Progress) is not ported.
//   - C++ threads ctx *into* `SortGeometry`, `ReorderHalfedges` and
//     `SimplifyTopology`; we only bracket them. The cost is latency (one run of
//     the trailing simplify + sort block), not a wrong status.
//   - C++ also threads ctx into the non-Boolean entry points (`FromMeshGL`,
//     `Smooth`, `LevelSet`, `Hull`, `Minkowski`, `Refine`). Here only the
//     boolean / CSG pipeline is cancellable at all; those entry points ignore
//     tokens rather than reporting a stale status, since they take none.

using System.Runtime.CompilerServices;

namespace ManifoldSharp
{
	/// <summary>
	/// A cheaply shared, thread-safe cancellation flag.
	/// </summary>
	/// <remarks>
	/// <para>
	/// The Rust source is a <c>Clone</c> struct wrapping an <c>Arc&lt;AtomicBool&gt;</c>,
	/// so clones share one flag (the C++ <c>ExecutionContext</c> pimpl semantics) and a
	/// token handed to a worker thread can be cancelled from anywhere. C# reference
	/// semantics <em>are</em> that clone: handing the same <see cref="CancelToken"/>
	/// object to another thread shares the flag, and there is no value copy to get wrong.
	/// </para>
	/// <para>
	/// Cancellation is <b>sticky</b>: there is no way to un-cancel a token, matching C++,
	/// where the flag is deliberately preserved across operation resets. Start a fresh
	/// token for work that should be allowed to complete.
	/// </para>
	/// <para>
	/// That shape is the Rust's, and it is the native one here: sticky, no reset, no
	/// disposal, polled at documented checkpoints. It is deliberately <em>not</em>
	/// <see cref="System.Threading.CancellationToken"/>, whose contract — throwing
	/// <c>OperationCanceledException</c>, linked sources, disposable registrations — fits
	/// badly a kernel that reports cancellation as a status on an empty result rather
	/// than by unwinding. <see cref="CancelToken(System.Threading.CancellationToken)"/>
	/// is the bridge between the two, so callers already threading a BCL token (all of
	/// agg-sharp's <c>ManifoldKernel</c>) convert in one expression instead of open-coding
	/// a registration at every call site — which is what kept agg-sharp's swap off
	/// the P/Invoke binding mechanical.
	/// </para>
	/// </remarks>
	public sealed class CancelToken
	{
		// The Rust stores this flag with `Ordering::Relaxed` on both the load and the
		// store, matching C++ `cancel.store(true, std::memory_order_relaxed)`: the flag
		// is advisory, orders nothing else, and readers only use it to decide whether to
		// stop early.
		//
		// C# has no relaxed atomic. `volatile` is the weakest ordering the CLR memory
		// model exposes that still forbids the two things a cancellation flag cannot
		// survive: caching the read in a register across a loop, and sinking the write
		// past the point the canceller returns. It is *stronger* than Relaxed
		// (acquire on read, release on write), which is sound in one direction only —
		// Relaxed is the weakest requirement, so satisfying more of the memory model
		// than the Rust asked for cannot change which values are observed. Nothing here
		// relies on the extra ordering, so a future move to a genuinely relaxed
		// primitive would also be correct.
		private volatile bool flag;

		/// <summary>
		/// Creates a fresh, uncancelled token.
		/// </summary>
		public CancelToken()
		{
			this.flag = false;
		}

		/// <summary>
		/// Creates a token that cancels when <paramref name="external"/> does — the
		/// bridge from the BCL's cancellation model into this one.
		/// </summary>
		/// <remarks>
		/// <para>
		/// The registration is deliberately never disposed. A <see cref="CancelToken"/>
		/// has no <c>Dispose</c> and is not meant to outlive the operation it was created
		/// for: the intended shape is one token per kernel call, constructed at the call
		/// site and unreferenced when the call returns, at which point the registration
		/// and the token are collected together. Holding a bridged token alive for the
		/// lifetime of a long-lived <see cref="System.Threading.CancellationTokenSource"/>
		/// would keep it rooted in that source's callback list, so don't — build one per
		/// operation, as the sticky semantics already require.
		/// </para>
		/// <para>
		/// An already-cancelled <paramref name="external"/> is handled by the same line:
		/// <c>Register</c> invokes the callback synchronously in that case, so the token
		/// is born cancelled. A non-cancellable token
		/// (<see cref="System.Threading.CancellationToken.None"/>) registers nothing.
		/// </para>
		/// </remarks>
		/// <param name="external">The BCL token to follow.</param>
		public CancelToken(CancellationToken external)
		{
			this.flag = false;
			external.Register(static state => ((CancelToken)state!).Cancel(), this);
		}

		/// <summary>
		/// Whether cancellation has been requested.
		/// </summary>
		public bool IsCancelled
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			get { return this.flag; }
		}

		/// <summary>
		/// Requests cancellation. Callable from any thread, including while another
		/// thread is inside an operation holding this token. Calling it more than once
		/// is a no-op — cancellation is sticky and never clears.
		/// </summary>
		public void Cancel()
		{
			this.flag = true;
		}

		/// <summary>
		/// Debugger-facing text, mirroring the Rust <c>#[derive(Debug)]</c>.
		/// </summary>
		public override string ToString()
		{
			return this.flag ? "CancelToken { cancelled }" : "CancelToken { live }";
		}
	}

	/// <summary>
	/// Module-level helpers from <c>cancel.rs</c> that are free functions in Rust.
	/// </summary>
	public static class Cancel
	{
		/// <summary>
		/// Canonical reader, mirroring C++ <c>IsCancelled(ExecutionContext::Impl*)</c>
		/// and Rust <c>cancel::is_cancelled</c>.
		/// </summary>
		/// <remarks>
		/// <c>null</c> (Rust's <c>None</c>, C++'s <c>nullptr</c> ctx) returns false
		/// without touching the flag, so the uncancellable path — every existing caller
		/// — reads exactly as it did before this module existed.
		/// </remarks>
		/// <param name="token">The token to poll, or null for "no cancellation requested".</param>
		/// <returns>True when a token is present and has been cancelled.</returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool IsCancelled(CancelToken? token)
		{
			return token is not null && token.IsCancelled;
		}
	}
}
