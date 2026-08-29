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

// Progress.cs — port of progress.rs — optional, throttled progress reporting
// for the long-running boolean pipelines.
//
// This is the sibling of CancelToken.cs: both are threaded through the kernel
// as an `Option<&_>` (a nullable reference in C#) so that the "nobody is
// watching" path — every pre-existing caller — is byte-for-byte the code that
// ran before the feature existed. `None` touches no atomic and takes no lock;
// the branch folds out of the hot loops entirely because the *decision* is made
// once per phase, at the call site of a map, not per element.
//
// The C++ reference has an equivalent (`ExecutionContext`'s donePhases /
// totalPhases / Progress()), which CancelToken.cs deliberately did not port.
// This module is not a port of it: the C++ counts whole pipeline phases, while
// the robust engine's phases are wildly unequal in cost, so we report a *named*
// phase plus an intra-phase fraction instead. Nothing here can change a
// computed value — the reporter is write-only from the kernel's point of view.
//
// Who reports what (Rust module names; the C# files they port to):
//   robust/intersection_graph.rs  NarrowPhase, SelfIntersections,
//                                 CandidatePoints, Registries, Arrangements
//   robust/cells.rs               Cells (per arrangement edge)
//   robust/mod.rs                 Winding, Assemble (phase transitions only)
//   boolean3.rs                   ExactBoolean (one indeterminate phase; the
//                                 exact engine's internals are not
//                                 instrumented, so its timing stays exactly
//                                 what it was)
//
// Threading model: the callback is invoked under a lock, so it is never
// re-entered concurrently even when Phase 11's parallel maps have workers
// driving `Advance`. It *can* be invoked from a worker thread rather than the
// caller's; consumers that need a specific thread must marshal themselves.

using System.Collections.ObjectModel;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace ManifoldSharp
{
	/// <summary>
	/// Coarse pipeline stages, in the order the robust engine runs them.
	/// </summary>
	/// <remarks>
	/// Ids are part of the FFI surface of the Rust source
	/// (<c>manifold_rs_progress_phase_name</c>), so new phases are appended rather than
	/// inserted, and the explicit discriminants are the ones the Rust <c>#[repr(u32)]</c>
	/// enum assigns.
	/// </remarks>
	public enum Phase : uint
	{
		/// <summary>Broad-phase pair culling.</summary>
		NarrowPhase = 0,

		/// <summary>Self-intersection detection.</summary>
		SelfIntersections = 1,

		/// <summary>Candidate intersection point generation.</summary>
		CandidatePoints = 2,

		/// <summary>Registry construction.</summary>
		Registries = 3,

		/// <summary>Per-face arrangement construction.</summary>
		Arrangements = 4,

		/// <summary>Cell extraction, per arrangement edge.</summary>
		Cells = 5,

		/// <summary>Winding number evaluation.</summary>
		Winding = 6,

		/// <summary>Output mesh assembly.</summary>
		Assemble = 7,

		/// <summary>The exact engine, reported as one indeterminate phase.</summary>
		ExactBoolean = 8,
	}

	/// <summary>
	/// The <see cref="Phase"/> members Rust declares as inherent methods and constants.
	/// </summary>
	public static class Phases
	{
		private static readonly Phase[] AllPhases =
		{
			Phase.NarrowPhase,
			Phase.SelfIntersections,
			Phase.CandidatePoints,
			Phase.Registries,
			Phase.Arrangements,
			Phase.Cells,
			Phase.Winding,
			Phase.Assemble,
			Phase.ExactBoolean,
		};

		// Rust's `Phase::ALL` is a `const [Phase; 9]` — every reader gets a copy and
		// nobody can write through it. An `IReadOnlyList<Phase>` that is really the
		// backing array is not that: a caller can cast it back to `Phase[]` and mutate
		// the table every phase lookup reads. Wrapping once at startup restores the
		// Rust's guarantee for the cost of one allocation.
		private static readonly ReadOnlyCollection<Phase> AllPhasesView = Array.AsReadOnly(AllPhases);

		/// <summary>
		/// Every phase, in pipeline order — Rust's <c>Phase::ALL</c>. Index equals id.
		/// </summary>
		public static IReadOnlyList<Phase> All
		{
			get { return AllPhasesView; }
		}

		/// <summary>
		/// The phase with this id, or null when the id names no phase.
		/// </summary>
		/// <param name="id">A phase id, as <see cref="Id"/> returns.</param>
		/// <returns>The phase, or null if out of range.</returns>
		public static Phase? FromId(uint id)
		{
			return id < (uint)AllPhases.Length ? AllPhases[id] : null;
		}

		/// <summary>
		/// Stable display name. A constant string so a reporter callback never has to
		/// allocate to forward it.
		/// </summary>
		/// <param name="phase">The phase to name.</param>
		/// <returns>The phase's stable display name.</returns>
		public static string Name(this Phase phase)
		{
			switch (phase)
			{
				case Phase.NarrowPhase: return "narrow phase";
				case Phase.SelfIntersections: return "self intersections";
				case Phase.CandidatePoints: return "candidate points";
				case Phase.Registries: return "registries";
				case Phase.Arrangements: return "arrangements";
				case Phase.Cells: return "cells";
				case Phase.Winding: return "winding";
				case Phase.Assemble: return "assemble";
				case Phase.ExactBoolean: return "exact boolean";

				// Rust's match is exhaustive over the enum; C# cannot prove a `Phase`
				// holds a declared value, so the unreachable arm says so loudly rather
				// than inventing a name.
				default: throw new ArgumentOutOfRangeException(nameof(phase));
			}
		}

		/// <summary>
		/// The phase's stable id, the value of its <c>#[repr(u32)]</c> discriminant.
		/// </summary>
		/// <param name="phase">The phase to identify.</param>
		/// <returns>The phase id.</returns>
		public static uint Id(this Phase phase)
		{
			return (uint)phase;
		}
	}

	/// <summary>
	/// A throttled sink for pipeline progress.
	/// </summary>
	/// <remarks>
	/// Pass a reporter to a <c>*WithProgress</c> entry point; the reporter may be shared
	/// across threads and outlive the call. The Rust bounds its callback
	/// <c>Send + Sync + 'static</c> because rayon workers may drive it; C# delegates carry
	/// no such bound, so the requirement becomes a documented one — a callback handed here
	/// must tolerate being invoked on a worker thread.
	/// </remarks>
	public sealed class ProgressReporter
	{
		/// <summary>
		/// How many callbacks a determinate phase emits, at most. Chosen so the
		/// per-item cost stays a relaxed increment plus one compare against a cached
		/// threshold: the lock and the callback itself are amortized over
		/// <c>total / 100</c> items.
		/// </summary>
		/// <remarks>
		/// The Rust's "relaxed increment" does not survive the port intact, and the cost
		/// model above is the Rust's, not this one's:
		/// <see cref="Interlocked.Add(ref ulong, ulong)"/> is a full sequentially-consistent
		/// read-modify-write (a <c>lock xadd</c> on x64, <c>ldaddal</c> on ARM64), where Rust's
		/// <c>fetch_add(Relaxed)</c> compiles to a barrier-free <c>ldadd</c>. .NET exposes no
		/// relaxed RMW, so an instrumented item costs strictly more here than in the Rust.
		/// It is still O(1) and still amortized against the callback, but Phase 11 should
		/// measure this rather than inherit the Rust's number.
		/// </remarks>
		private const ulong ReportsPerPhase = 100;

		private readonly Action<Phase, double?> callback;

		// Rust holds the callback *inside* a `Mutex<Callback>`; C# delegates need no
		// such wrapper, so the mutex becomes a plain lock object guarding the call.
		private readonly object callbackGate = new object();

		/// <summary>Current phase id, as <see cref="Phases.Id"/>.</summary>
		private uint phase;

		/// <summary>Items completed in the current phase.</summary>
		private ulong done;

		/// <summary>Items the current phase expects; 0 means "indeterminate".</summary>
		private ulong total;

		/// <summary><see cref="done"/> value at which the next callback fires.</summary>
		private ulong next;

		/// <summary>Items between callbacks.</summary>
		private ulong step;

		// Set once a callback has thrown; see Emit.
		private bool callbackFaulted;

		/// <summary>
		/// Creates a reporter that forwards each update to <paramref name="callback"/>.
		/// </summary>
		/// <param name="callback">
		/// The kernel-facing callback: the phase entered (its display name is
		/// <see cref="Phases.Name"/>) plus either a fraction in <c>[0, 1]</c> for a
		/// determinate bar, or null when the phase has no meaningful total.
		/// </param>
		public ProgressReporter(Action<Phase, double?> callback)
		{
			ArgumentNullException.ThrowIfNull(callback);

			this.callback = callback;
			this.phase = Phase.NarrowPhase.Id();
			this.done = 0;
			this.total = 0;
			this.next = ulong.MaxValue;
			this.step = ulong.MaxValue;
		}

		/// <summary>
		/// Enter <paramref name="phase"/>, expecting <paramref name="total"/> work items
		/// (0 = no total known, which reports as an indeterminate phase). Always emits a
		/// callback, so a phase transition is never throttled away.
		/// </summary>
		/// <param name="phase">The phase being entered.</param>
		/// <param name="total">Work items the phase expects, or 0 for indeterminate.</param>
		public void BeginPhase(Phase phase, ulong total)
		{
			ulong step = Math.Max(total / ReportsPerPhase, 1);
			Volatile.Write(ref this.phase, phase.Id());
			Volatile.Write(ref this.total, total);
			Volatile.Write(ref this.done, 0);
			Volatile.Write(ref this.step, step);
			Volatile.Write(ref this.next, total == 0 ? ulong.MaxValue : step);
			this.Emit(phase, total == 0 ? null : 0.0);
		}

		/// <summary>
		/// Record <paramref name="n"/> completed work items in the current phase,
		/// emitting a callback only when the throttle threshold is crossed.
		/// </summary>
		/// <remarks>
		/// Safe to call from several threads at once; the counter is atomic and the
		/// callback is serialized. Under contention two threads can both cross the
		/// threshold and both report, which is harmless — this is a UI hint, not a
		/// ledger.
		/// </remarks>
		/// <param name="n">Work items completed since the last call.</param>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public void Advance(ulong n)
		{
			ulong completed = Interlocked.Add(ref this.done, n);
			if (completed < Volatile.Read(ref this.next))
			{
				return;
			}

			this.ReportAt(completed);
		}

		/// <summary>
		/// Debugger-facing text, mirroring the Rust <c>Debug</c> impl.
		/// </summary>
		public override string ToString()
		{
			return string.Create(
				CultureInfo.InvariantCulture,
				$"ProgressReporter {{ phase: {Phases.FromId(Volatile.Read(ref this.phase))}, done: {Volatile.Read(ref this.done)}, total: {Volatile.Read(ref this.total)} }}");
		}

		// Cold half of Advance, kept out of line so the common case is an increment and
		// a compare. Rust marks it #[cold]; NoInlining is the C# lever that keeps it out
		// of the caller's hot path.
		[MethodImpl(MethodImplOptions.NoInlining)]
		private void ReportAt(ulong completed)
		{
			ulong step = Volatile.Read(ref this.step);
			// Rust's saturating_add: the sentinel for "never report again" is MaxValue,
			// so the throttle must never wrap past it. `unchecked` is explicit because
			// the wrap is the detection mechanism — under /checked this addition is the
			// one place in the file that would throw instead of saturating.
			ulong sum = unchecked(completed + step);
			ulong nextAt = sum < completed ? ulong.MaxValue : sum;
			Volatile.Write(ref this.next, nextAt);
			ulong total = Volatile.Read(ref this.total);
			Phase? phase = Phases.FromId(Volatile.Read(ref this.phase));
			if (phase is null)
			{
				return;
			}

			double? fraction = total == 0
				? null
				: Math.Clamp((double)completed / total, 0.0, 1.0);
			this.Emit(phase.Value, fraction);
		}

		// Invoke the callback. Rust guards it with a Mutex and deliberately ignores a
		// poisoned lock (meaning a previous callback panicked) rather than propagating
		// it: a broken progress sink must not take down a geometry operation. C# locks
		// do not poison, so the latch below stands in for the poison flag — the first
		// exception unwinds out of Emit the way a Rust panic would, and every later
		// Emit is the no-op a poisoned `lock()` produces.
		//
		// The latch is read TWICE, and the second read is the load-bearing one. Rust
		// reads the poison bit while acquiring the mutex, so a thread queued behind the
		// panicking one still sees it. A single check before the lock does not: a thread
		// that passed it a nanosecond before another thread faulted is already committed,
		// waits out the lock, and calls a sink that has been declared broken — sending a
		// second exception into the kernel that Rust would have swallowed. The check
		// inside the lock closes that window; the one outside is only the free fast path
		// for the overwhelmingly common already-broken case.
		private void Emit(Phase phase, double? fraction)
		{
			if (Volatile.Read(ref this.callbackFaulted))
			{
				return;
			}

			lock (this.callbackGate)
			{
				if (this.callbackFaulted)
				{
					return;
				}

				try
				{
					this.callback(phase, fraction);
				}
				catch
				{
					// Published with a release (the lock exit) and read with an acquire
					// (the Volatile.Read above), so the fast path cannot miss it.
					Volatile.Write(ref this.callbackFaulted, true);
					throw;
				}
			}
		}
	}

	/// <summary>
	/// Module-level helpers from <c>progress.rs</c> that are free functions in Rust.
	/// </summary>
	public static class Progress
	{
		/// <summary>
		/// Null-aware <see cref="ProgressReporter.BeginPhase"/>, mirroring how
		/// <see cref="Cancel.IsCancelled"/> handles the absent case.
		/// </summary>
		/// <param name="progress">The reporter, or null for an uninstrumented run.</param>
		/// <param name="phase">The phase being entered.</param>
		/// <param name="total">Work items the phase expects, or 0 for indeterminate.</param>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static void BeginPhase(ProgressReporter? progress, Phase phase, ulong total)
		{
			if (progress is not null)
			{
				progress.BeginPhase(phase, total);
			}
		}

		/// <summary>
		/// <see cref="Par.MaybeParMapCt"/> that also counts completed items into
		/// <paramref name="progress"/>.
		/// </summary>
		/// <remarks>
		/// With a null reporter this <em>is</em> <c>MaybeParMapCt</c> — the same closure,
		/// no wrapper — so the uninstrumented path keeps its exact codegen. With a
		/// reporter the only added work per item is one increment; results are still
		/// collected in index order, so the output is bit-identical either way.
		/// </remarks>
		/// <remarks>
		/// The Rust calls that increment "relaxed" and it is not, here: see the cost note
		/// on <c>ReportsPerPhase</c>. .NET has no relaxed read-modify-write, so the added
		/// per-item work is a full barrier rather than Rust's barrier-free
		/// <c>fetch_add</c>. It cannot change a value — the reporter is still write-only
		/// from the kernel's point of view — but it is not the free ride the ported
		/// sentence implies.
		/// </remarks>
		/// <typeparam name="T">The mapped element type.</typeparam>
		/// <param name="n">Number of indices to map, <c>0..n</c>.</param>
		/// <param name="threshold">Size at or above which a parallel build goes parallel.</param>
		/// <param name="token">The cancellation token, or null for an uncancellable run.</param>
		/// <param name="progress">The reporter, or null for an uninstrumented run.</param>
		/// <param name="f">The per-index map.</param>
		/// <returns>The mapped values in index order, or null if the token was cancelled.</returns>
		public static T[]? MaybeParMapCtProgress<T>(
			int n,
			int threshold,
			CancelToken? token,
			ProgressReporter? progress,
			Func<int, T> f)
		{
			if (progress is null)
			{
				return Par.MaybeParMapCt(n, threshold, token, f);
			}

			return Par.MaybeParMapCt(n, threshold, token, i =>
			{
				T output = f(i);
				progress.Advance(1);
				return output;
			});
		}
	}
}
