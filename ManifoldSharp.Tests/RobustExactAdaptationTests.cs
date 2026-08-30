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

// NOT PORTED FROM RUST. Every test in this file covers a state or a precondition
// that exists only in the C# port of robust/exact, so none of them counts toward
// the test-for-test tally in docs/PORTING_PLAN.md. They live in their own file so
// that tally stays auditable: RobustExactTests.cs and its five partials hold
// exactly the 36 ported tests (tests.rs's 18, approx.rs's 8, intpred.rs's 10).
//
// The rule that produced this file: the Rust suite cannot cover states the Rust
// does not have. Both of these come from the same source — dashu's type system
// carried an invariant that BigInteger's does not:
//
//   default(R2Key)/default(R3Key)   Rust tuple structs have no default form;
//                                   every C# struct does, and R2/R3 are classes,
//                                   so the empty key is reachable here alone.
//   Backend.UintBits/UintBit        dashu's UBig made a negative argument
//                                   unrepresentable; BigInteger plays both roles,
//                                   so the check has to be executable.

using System.Numerics;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Tests
{
	public class RobustExactAdaptationTests
	{
		/// <summary>
		/// A default-initialized key must behave as a value distinct from every real
		/// key — not throw. A Phase 10 site that pre-allocates a key array, or reads a
		/// failed TryGetValue's out parameter, holds one of these; the failure mode
		/// being guarded against is a NullReferenceException raised inside a dictionary
		/// probe, which reports the bug at the collection rather than at the caller.
		/// </summary>
		[Test]
		public async Task DefaultKeysProbeAsAMissRatherThanThrowing()
		{
			R2Key empty2 = default(R2Key);
			R2Key alsoEmpty2 = default(R2Key);
			R2Key real2 = new R2Key(new R2(Rational.Rat(1.0), Rational.Rat(2.0)));

			await Assert.That(empty2.Equals(alsoEmpty2)).IsTrue();
			await Assert.That(empty2.Equals(real2)).IsFalse();
			await Assert.That(real2.Equals(empty2)).IsFalse();
			await Assert.That(empty2.GetHashCode()).IsEqualTo(alsoEmpty2.GetHashCode());

			Dictionary<R2Key, int> map2 = new Dictionary<R2Key, int> { [real2] = 7 };
			await Assert.That(map2.ContainsKey(empty2)).IsFalse();
			await Assert.That(map2.TryGetValue(empty2, out int _)).IsFalse();

			R3Key empty3 = default(R3Key);
			R3Key alsoEmpty3 = default(R3Key);
			R3Key real3 = new R3Key(new R3(Rational.Rat(1.0), Rational.Rat(2.0), Rational.Rat(3.0)));

			await Assert.That(empty3.Equals(alsoEmpty3)).IsTrue();
			await Assert.That(empty3.Equals(real3)).IsFalse();
			await Assert.That(real3.Equals(empty3)).IsFalse();
			await Assert.That(empty3.GetHashCode()).IsEqualTo(alsoEmpty3.GetHashCode());

			Dictionary<R3Key, int> map3 = new Dictionary<R3Key, int> { [real3] = 7 };
			await Assert.That(map3.ContainsKey(empty3)).IsFalse();
			await Assert.That(map3.TryGetValue(empty3, out int _)).IsFalse();

			// And real keys still hit, so the guard did not swallow the normal path.
			await Assert.That(map2[new R2Key(new R2(Rational.Rat(1.0), Rational.Rat(2.0)))]).IsEqualTo(7);
			await Assert.That(map3[new R3Key(new R3(Rational.Rat(1.0), Rational.Rat(2.0), Rational.Rat(3.0)))])
				.IsEqualTo(7);
		}

		/// <summary>
		/// The magnitude precondition is executable, not commentary. Passing a signed
		/// value would otherwise be silently wrong rather than loud:
		/// <c>BigInteger.GetBitLength</c> reports the two's-complement length, which is
		/// one short of the magnitude's at every power of two, and a right shift sign-
		/// extends. Inside RatToF64 that is a moved output vertex, not a crash — the
		/// one failure mode this port is least able to detect after the fact.
		/// </summary>
		[Test]
		public async Task BitHelpersRejectSignedValues()
		{
			// The silent-wrong-answer these guards replace, stated as the reason: the
			// two's-complement length of -8 is 3 where its magnitude needs 4 bits.
			await Assert.That(Backend.UintBits(new BigInteger(8))).IsEqualTo(4L);
			await Assert.That((long)new BigInteger(-8).GetBitLength()).IsEqualTo(3L);

			await Assert.That(() => Backend.UintBits(new BigInteger(-8)))
				.Throws<ArgumentOutOfRangeException>();
			await Assert.That(() => Backend.UintBit(new BigInteger(-8), 0))
				.Throws<ArgumentOutOfRangeException>();

			// Zero and positive magnitudes are unaffected.
			await Assert.That(Backend.UintBits(BigInteger.Zero)).IsEqualTo(0L);
			await Assert.That(Backend.UintBit(new BigInteger(6), 0)).IsFalse();
			await Assert.That(Backend.UintBit(new BigInteger(6), 1)).IsTrue();
		}
	}
}
