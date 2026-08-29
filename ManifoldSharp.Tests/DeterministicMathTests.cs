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

// Port of the tests module in math.rs — same cases, same expected values, in
// the same order.
//
// The agreement tests call System.Math.Sin/Cos/Tan/Acos/Atan/Atan2 on purpose:
// they are the port of Rust's `test_agreement_with_std`, whose whole job is to
// pin the deterministic implementations to within a few ULP of the platform's
// libm. This is the one place in the repo where calling those is correct —
// DeterministicMath itself must never do it.

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class DeterministicMathTests
	{
		private const double PI = Math.PI;
		private const double FRAC_PI_2 = Math.PI / 2.0;
		private const double FRAC_PI_4 = Math.PI / 4.0;

		[Test]
		public async Task SinBasic()
		{
			await Assert.That(DeterministicMath.Sin(0.0)).IsEqualTo(0.0);
			await Assert.That(Math.Abs(DeterministicMath.Sin(FRAC_PI_2) - 1.0) < 1e-15).IsTrue();
			await Assert.That(Math.Abs(DeterministicMath.Sin(PI)) < 1e-15).IsTrue();
			await Assert.That(Math.Abs(DeterministicMath.Sin(-FRAC_PI_2) + 1.0) < 1e-15).IsTrue();
		}

		[Test]
		public async Task CosBasic()
		{
			await Assert.That(DeterministicMath.Cos(0.0)).IsEqualTo(1.0);
			await Assert.That(Math.Abs(DeterministicMath.Cos(FRAC_PI_2)) < 1e-15).IsTrue();
			await Assert.That(Math.Abs(DeterministicMath.Cos(PI) + 1.0) < 1e-15).IsTrue();
		}

		[Test]
		public async Task TanBasic()
		{
			await Assert.That(DeterministicMath.Tan(0.0)).IsEqualTo(0.0);
			await Assert.That(Math.Abs(DeterministicMath.Tan(FRAC_PI_4) - 1.0) < 1e-15).IsTrue();
		}

		[Test]
		public async Task AcosBasic()
		{
			await Assert.That(DeterministicMath.Acos(1.0)).IsEqualTo(0.0);
			await Assert.That(Math.Abs(DeterministicMath.Acos(0.0) - FRAC_PI_2) < 1e-15).IsTrue();
			await Assert.That(Math.Abs(DeterministicMath.Acos(-1.0) - PI) < 1e-15).IsTrue();
		}

		[Test]
		public async Task AsinBasic()
		{
			await Assert.That(DeterministicMath.Asin(0.0)).IsEqualTo(0.0);
			await Assert.That(Math.Abs(DeterministicMath.Asin(1.0) - FRAC_PI_2) < 1e-15).IsTrue();
		}

		[Test]
		public async Task AtanBasic()
		{
			await Assert.That(DeterministicMath.Atan(0.0)).IsEqualTo(0.0);
			await Assert.That(Math.Abs(DeterministicMath.Atan(1.0) - FRAC_PI_4) < 1e-15).IsTrue();
		}

		[Test]
		public async Task Atan2Basic()
		{
			await Assert.That(Math.Abs(DeterministicMath.Atan2(1.0, 1.0) - FRAC_PI_4) < 1e-15).IsTrue();
			await Assert.That(Math.Abs(DeterministicMath.Atan2(0.0, 1.0)) < 1e-15).IsTrue();
			await Assert.That(Math.Abs(DeterministicMath.Atan2(1.0, 0.0) - FRAC_PI_2) < 1e-15).IsTrue();
		}

		/// <summary>
		/// Compare our deterministic trig functions against the platform's for a range
		/// of values. They should agree to within a few ULP for normal inputs.
		/// </summary>
		[Test]
		public async Task AgreementWithStd()
		{
			for (int i = -100; i <= 100; i++)
			{
				double v = i * 0.1;

				double ourSin = DeterministicMath.Sin(v);
				double stdSin = Math.Sin(v);
				await Assert.That(Math.Abs(ourSin - stdSin) < 1e-12)
					.IsTrue()
					.Because($"sin({v}): ours={ourSin}, std={stdSin}");

				double ourCos = DeterministicMath.Cos(v);
				double stdCos = Math.Cos(v);
				await Assert.That(Math.Abs(ourCos - stdCos) < 1e-12)
					.IsTrue()
					.Because($"cos({v}): ours={ourCos}, std={stdCos}");

				double ourTan = DeterministicMath.Tan(v);
				double stdTan = Math.Tan(v);

				// tan can be very large near asymptotes; only check moderate values
				if (Math.Abs(stdTan) < 1e6)
				{
					await Assert.That(Math.Abs(ourTan - stdTan) < 1e-10)
						.IsTrue()
						.Because($"tan({v}): ours={ourTan}, std={stdTan}");
				}
			}
		}

		[Test]
		public async Task AcosAgreementWithStd()
		{
			for (int i = -100; i <= 100; i++)
			{
				double v = i * 0.01; // range [-1, 1]
				double ours = DeterministicMath.Acos(v);
				double stdValue = Math.Acos(v);
				await Assert.That(Math.Abs(ours - stdValue) < 1e-14)
					.IsTrue()
					.Because($"acos({v}): ours={ours}, std={stdValue}");
			}
		}

		[Test]
		public async Task AtanAgreementWithStd()
		{
			for (int i = -100; i <= 100; i++)
			{
				double v = i * 0.1;
				double ours = DeterministicMath.Atan(v);
				double stdValue = Math.Atan(v);
				await Assert.That(Math.Abs(ours - stdValue) < 1e-14)
					.IsTrue()
					.Because($"atan({v}): ours={ours}, std={stdValue}");
			}
		}

		[Test]
		public async Task Atan2AgreementWithStd()
		{
			double[] vals = { -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0 };
			foreach (double y in vals)
			{
				foreach (double x in vals)
				{
					if (x == 0.0 && y == 0.0)
					{
						continue;
					}

					double ours = DeterministicMath.Atan2(y, x);
					double stdValue = Math.Atan2(y, x);
					await Assert.That(Math.Abs(ours - stdValue) < 1e-14)
						.IsTrue()
						.Because($"atan2({y}, {x}): ours={ours}, std={stdValue}");
				}
			}
		}

		[Test]
		public async Task SpecialValues()
		{
			// NaN propagation
			await Assert.That(double.IsNaN(DeterministicMath.Sin(double.NaN))).IsTrue();
			await Assert.That(double.IsNaN(DeterministicMath.Cos(double.NaN))).IsTrue();
			await Assert.That(double.IsNaN(DeterministicMath.Tan(double.NaN))).IsTrue();

			// Infinity -> NaN
			await Assert.That(double.IsNaN(DeterministicMath.Sin(double.PositiveInfinity))).IsTrue();
			await Assert.That(double.IsNaN(DeterministicMath.Cos(double.PositiveInfinity))).IsTrue();
			await Assert.That(double.IsNaN(DeterministicMath.Tan(double.PositiveInfinity))).IsTrue();
		}

		// ------------------------------------------------------------------
		// C#-port regression tests
		//
		// Everything above this line is a test-for-test port of math.rs's
		// `#[cfg(test)]` module - 12 tests, same order, same expected values - and
		// that count is meant to stay auditable against the Rust suite. The two
		// below have no Rust counterpart: they pin C#-specific traps that a
		// fuzz-diff against manifold-rust caught and that no ported test can see.
		// Count them separately.
		// ------------------------------------------------------------------

		/// <summary>
		/// Asin's out-of-domain NaN must be Rust's f64::NAN (0x7ff8000000000000),
		/// not C#'s double.NaN (0xfff8000000000000, sign bit set).
		/// </summary>
		/// <remarks>
		/// double.IsNaN cannot see the sign bit, so every ported test stays green if
		/// someone "simplifies" this back to double.NaN - while every downstream
		/// DoubleToInt64Bits hash or weld silently diverges from the Rust. That is
		/// why the assertion is on the bits.
		/// </remarks>
		[Test]
		public async Task AsinOutOfDomainReturnsRustsPositiveQuietNaN()
		{
			await Assert.That(BitConverter.DoubleToUInt64Bits(DeterministicMath.Asin(2.0)))
				.IsEqualTo(0x7ff8000000000000UL);
			await Assert.That(BitConverter.DoubleToUInt64Bits(DeterministicMath.Asin(-2.0)))
				.IsEqualTo(0x7ff8000000000000UL);
			await Assert.That(BitConverter.DoubleToUInt64Bits(DeterministicMath.Asin(double.PositiveInfinity)))
				.IsEqualTo(0x7ff8000000000000UL);
		}

		/// <summary>
		/// The huge-argument path must saturate its float-to-int conversion the way
		/// Rust's <c>as i32</c> does. Sin(1e22) is exactly -1.0 in manifold-rust.
		/// </summary>
		/// <remarks>
		/// This pins DeterministicMath's SaturatingToInt32 in place. ARM64 saturates
		/// float-to-int in hardware, so replacing the helper with a bare (int) cast
		/// passes every other test on an Apple-silicon machine while returning a
		/// different quadrant on x64, where that cast yields int.MinValue. Only an
		/// argument past the 2^20*(pi/2) reduction threshold reaches it.
		/// </remarks>
		[Test]
		public async Task HugeArgumentReductionSaturatesLikeRust()
		{
			await Assert.That(DeterministicMath.Sin(1e22)).IsEqualTo(-1.0);
			await Assert.That(DeterministicMath.Cos(1e22)).IsEqualTo(0.0);
			await Assert.That(DeterministicMath.Sin(double.MaxValue)).IsEqualTo(-1.0);
			await Assert.That(DeterministicMath.Sin(double.MinValue)).IsEqualTo(0.0);
		}
	}
}
