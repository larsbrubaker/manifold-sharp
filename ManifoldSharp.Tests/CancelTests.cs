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

// Port of cancel_tests.rs: the cancellation mechanism (CancelToken.cs) and its
// threading through the boolean / CSG-tree pipeline. The C++ counterparts live
// in cpp-reference/manifold/test/context_test.cpp.
//
// DEFERRED, and why. cancel_tests.rs has nine tests; eight of them drive a real
// boolean or a real CSG tree through the `Manifold` façade, which is Phase 6.
// (The engine under it is no longer the obstacle: the boolean landed in Phase 5's
// first step and CsgTree in its second.) They are listed here by their Rust names
// so the phase that unblocks them can grep for this block rather than rediscover
// the list:
//
//   no_token_path_is_unchanged                                    Phase 5/6
//   pre_cancelled_token_returns_cancelled_promptly                Phase 5/6
//   pre_cancelled_token_short_circuits_csg_tree_evaluation        Phase 5/6
//   empty_input_fast_paths_still_honour_a_pre_cancelled_token     Phase 5/6
//   cancel_from_another_thread_interrupts_a_boolean_in_flight     Phase 5/6
//   a_used_then_cancelled_token_does_not_affect_later_ops_with_
//     a_fresh_token                                               Phase 5/6
//   a_live_token_produces_identical_geometry_above_the_parallel_
//     thresholds                                                  Phase 5/6 (asserts
//                                                                 the Phase 11 threshold)
//   cancelled_status_survives_the_csg_tree_root                   Phase 5/6
//
// The `slow_pair()` fixture (two 256-segment spheres offset by 0.5) they share
// is deferred with them. Nothing is stubbed: when the dependency lands, these
// get written with their Rust expected values intact.
//
// RECHECKED at Phase 5 (the boolean engine landed: Boolean3.cs,
// BooleanResult*.cs). None of the eight became portable — every one of them
// opens with `Manifold::cube`/`sphere`/`empty` and asserts on `status()` or
// `get_mesh_gl(-1)`, so the façade (Phase 6) is a blocker for all eight.
//
// SIX of the eight additionally reach `Manifold::sphere`, which is
// subdivide-backed and therefore Phase 7 — do not start any of these expecting a
// façade-only blocker:
//     no_token_path_is_unchanged                          sphere(0.6, 32)
//                                                         (cancel_tests.rs:58)
//     pre_cancelled_token_returns_cancelled_promptly      slow_pair()
//     pre_cancelled_token_short_circuits_csg_tree_eval..  sphere(1.0, 64) (:105)
//     cancel_from_another_thread_interrupts_a_boolean..   slow_pair()
//     a_used_then_cancelled_token_does_not_affect..       sphere(0.6, 32) (:199)
//     a_live_token_produces_identical_geometry_above..    sphere(1.0, 128)
//
// The two that are genuinely façade-only are
// `empty_input_fast_paths_still_honour_a_pre_cancelled_token` (cube + empty) and
// `cancelled_status_survives_the_csg_tree_root` (cube + translate). Both, along
// with `pre_cancelled_token_short_circuits_csg_tree_evaluation`, also wanted
// CsgTree — that landed with Phase 5's second step (CsgTree.cs,
// CsgTree.Batch.cs, Minkowski.cs), so the Phase 6 façade is now the sole
// remaining blocker for the pair above and the only non-sphere one for the third.
//
// What the engine step did cover, differentially rather than by a checked-in
// test: the entry gate beating the empty-input fast paths, a live-but-uncancelled
// token producing byte-identical geometry to the untokened path, and a
// pre-cancelled token yielding an empty `Error::Cancelled` impl — all three run
// over 31 mesh pairs x 3 ops against the compiled Rust with zero diffs.
//
// The CSG step covered its own cancel branches the same way (see the header of
// CsgTreeTests.cs): pre-cancelled Add/Subtract/leaf evaluations and a live token,
// dumped bit-for-bit against the compiled Rust. So `EvaluateWithToken`'s
// Cancelled paths have differential coverage but still no checked-in C# test.

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class CancelTests
	{
		[Test]
		public async Task TokenStartsUncancelledAndCancelIsObservableThroughClones()
		{
			CancelToken token = new CancelToken();
			await Assert.That(token.IsCancelled).IsFalse();

			// The Rust clones the token here to show that clones share one flag,
			// matching the C++ ExecutionContext pimpl. In C# the reference *is* the
			// clone: `Arc<AtomicBool>` maps onto a class reference, so aliasing the
			// token is the port of `token.clone()`.
			CancelToken clone = token;
			await Assert.That(clone.IsCancelled).IsFalse();
			token.Cancel();
			await Assert.That(token.IsCancelled).IsTrue();
			await Assert.That(clone.IsCancelled).IsTrue();

			// Cancel is sticky: there is deliberately no way to clear it.
			token.Cancel();
			await Assert.That(token.IsCancelled).IsTrue();

			// The canonical reader, Rust's free `cancel::is_cancelled`: a null token is
			// C++'s nullptr ctx and reports uncancelled without touching a flag.
			await Assert.That(Cancel.IsCancelled(null)).IsFalse();
			await Assert.That(Cancel.IsCancelled(token)).IsTrue();
			await Assert.That(Cancel.IsCancelled(new CancelToken())).IsFalse();
		}
	}
}
