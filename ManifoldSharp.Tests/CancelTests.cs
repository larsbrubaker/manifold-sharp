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
// boolean or a real CSG tree, neither of which exists yet (Manifold arrives in
// Phase 6, the boolean engine in Phase 5, CsgTree in Phase 5). They are listed
// here by their Rust names so the phase that unblocks them can grep for this
// block rather than rediscover the list:
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
