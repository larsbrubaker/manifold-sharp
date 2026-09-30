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

// ProgressOrderTests.cs — NOT A PORT, a C#-only adaptation test counted separately as
// CLAUDE.md requires. It pins that a ProgressReporter never emits a phase's fraction
// below one it already emitted, even when workers race: two workers whose increments
// land at 50 and 51 can reach the callback lock in the other order, and the stale 50
// must be dropped rather than sent backwards (divergence ledger entry 4, item 3).

using ManifoldSharp;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ProgressOrderTests
	{
		/// <summary>
		/// Many workers advancing one phase of 100 units (a report per unit) see a
		/// never-decreasing stream, round after round.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task RacingWorkersNeverSendTheBarBackwards()
		{
			const int Rounds = 2000;
			const int Workers = 4;
			List<double> fractions = new List<double>();
			ProgressReporter reporter = new ProgressReporter((_, fraction) => fractions.Add(fraction!.Value));

			for (int round = 0; round < Rounds; round++)
			{
				// The callback runs under the reporter's own lock, so the list needs no lock.
				fractions.Clear();
				reporter.BeginPhase(Phase.Minkowski, 100);
				Parallel.For(0, Workers, _ =>
				{
					for (int i = 0; i < 100 / Workers; i++)
					{
						reporter.Advance(1);
					}
				});

				for (int i = 1; i < fractions.Count; i++)
				{
					if (fractions[i] < fractions[i - 1])
					{
						await Assert.That(fractions[i]).IsGreaterThanOrEqualTo(fractions[i - 1])
							.Because($"round {round}: the bar went backwards at report {i}");
					}
				}
			}
		}
	}
}
