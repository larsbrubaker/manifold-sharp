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

// Deterministic trigonometric helpers.
//
// Adapted from FreeBSD msun implementations via musl libc sources.
// These produce bit-identical results across all platforms, unlike
// the platform-dependent System.Math.Asin/Acos/etc.
//
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro/SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

// Port of math.rs, second half: the inverse functions Acos, Asin, Atan and
// Atan2. DeterministicMath.cs holds the bit helpers, the kernels, the pi/2
// argument reduction and the forward functions, plus the shared file notes.
//
// Math.Sqrt (in Acos) and Math.Abs (in Atan and Atan2) are the only System.Math
// calls here; both are IEEE-exact bit operations standing in for the identical
// Rust primitive, and both are commented at their call sites.

using System.Runtime.CompilerServices;

namespace ManifoldSharp
{
	public static partial class DeterministicMath
	{
		// PositiveQuietNaN — Rust's f64::NAN, bit for bit — lives in DeterministicMath.cs
		// (same partial class) so this file and Types.TrigDegrees.cs share one constant.
		// See the note in Asin for why double.NaN is not the same value.

		// The rational approximation shared by both halves of acos. In Rust this
		// is a nested `fn r(z)` inside acos; C# has no nested functions that can
		// hold consts, so it becomes a private method and carries the PS/QS
		// coefficients with it.
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static double AcosR(double z)
		{
			const double PS0 = 1.66666666666666657415e-01;
			const double PS1 = -3.25565818622400915405e-01;
			const double PS2 = 2.01212532134862925881e-01;
			const double PS3 = -4.00555345006794114027e-02;
			const double PS4 = 7.91534994289814532176e-04;
			const double PS5 = 3.47933107596021167570e-05;
			const double QS1 = -2.40339491173441421878e+00;
			const double QS2 = 2.02094576023350569471e+00;
			const double QS3 = -6.88283971605453293030e-01;
			const double QS4 = 7.70381505559019352791e-02;

			double p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))));
			double q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)));
			return p / q;
		}

		/// <summary>
		/// Deterministic arccosine: the same bits on every platform, and the same
		/// bits as manifold-rust, for every |x| &lt;= 1.
		/// </summary>
		/// <remarks>
		/// The one exception is |x| &gt; 1, which returns the hardware's default NaN
		/// and therefore differs in the sign bit between architectures. That
		/// divergence is faithful - the Rust does the identical thing - and it is
		/// documented at the call site rather than papered over.
		/// </remarks>
		public static double Acos(double x)
		{
			const double PIO2_HI = 1.57079632679489655800e+00;
			const double PIO2_LO = 6.12323399573676603587e-17;

			ulong xx = BitConverter.DoubleToUInt64Bits(x);
			uint hx = (uint)(xx >> 32);
			uint ix = hx & 0x7fff_ffffu;

			if (ix >= 0x3ff0_0000u)
			{
				uint lx = unchecked((uint)xx);
				if (unchecked((ix - 0x3ff0_0000u) | lx) == 0)
				{
					if ((hx >> 31) != 0)
					{
						return 2.0 * PIO2_HI + BitConverter.UInt64BitsToDouble(0x3987_0000_0000_0000UL); // 0x1p-120
					}

					return 0.0;
				}

				// |x| > 1: NaN. Deliberately the *platform default* NaN that 0/0
				// generates - positive quiet on ARM64, sign-bit-set on x86 SSE - so
				// unlike the rest of this file the bits here are architecture
				// dependent. Rust's `0.0 / (x - x)` has exactly the same dependence,
				// so this is bug-for-bug faithful and must stay that way: pinning it
				// to a fixed NaN on the C# side alone would create the divergence it
				// looks like it fixes. Contrast Asin below, where Rust names a
				// concrete constant and so must this port.
				return 0.0 / (x - x);
			}

			if (ix < 0x3fe0_0000u)
			{
				// |x| < 0.5
				if (ix <= 0x3c60_0000u)
				{
					return PIO2_HI + BitConverter.UInt64BitsToDouble(0x3987_0000_0000_0000UL);
				}

				return PIO2_HI - (x - (PIO2_LO - x * AcosR(x * x)));
			}

			if ((hx >> 31) != 0)
			{
				// x < -0.5
				double zNeg = (1.0 + x) * 0.5;

				// f64::sqrt and Math.Sqrt are both the IEEE-754 correctly-rounded
				// square root, so this one is bit-identical everywhere by mandate
				// of the standard - unlike the transcendentals this file replaces.
				double sNeg = Math.Sqrt(zNeg);
				double wNeg = AcosR(zNeg) * sNeg - PIO2_LO;
				return 2.0 * (PIO2_HI - (sNeg + wNeg));
			}

			// x >= 0.5
			double z = (1.0 - x) * 0.5;
			double s = Math.Sqrt(z); // correctly rounded; see the note above
			double df = BitConverter.UInt64BitsToDouble(BitConverter.DoubleToUInt64Bits(s) & 0xffff_ffff_0000_0000UL);
			double c = (z - df * df) / (s + df);
			double w = AcosR(z) * s + c;
			return 2.0 * (df + w);
		}

		/// <summary>
		/// Deterministic arcsine. Produces bit-identical results on all platforms.
		/// </summary>
		public static double Asin(double x)
		{
			const double HALF_PI = 1.57079632679489661923132169163975144;
			if (!double.IsFinite(x) || x < -1.0 || x > 1.0)
			{
				// Rust's f64::NAN is the *positive* quiet NaN, 0x7ff8000000000000;
				// C#'s double.NaN has the sign bit set, 0xfff8000000000000. Returning
				// double.NaN here would be the only bit-level divergence in the file,
				// and one that would survive into any DoubleToInt64Bits hash or weld
				// downstream, so the constant is spelled out.
				return PositiveQuietNaN;
			}

			if (x == 1.0)
			{
				return HALF_PI;
			}

			if (x == -1.0)
			{
				return -HALF_PI;
			}

			return HALF_PI - Acos(x);
		}

		// atan's per-interval endpoints, high and low halves, and the odd/even
		// polynomial coefficients. Rust holds these in `const [f64; N]` arrays;
		// C# has no const array, so these are ReadOnlySpan properties over array
		// initializers - immutable metadata blobs rather than writable statics,
		// allocation-free, with the msun indices preserved. Same reasoning as the
		// tan kernel's T table.
		private static ReadOnlySpan<double> AtanHi => new[]
		{
			4.63647609000806093515e-01,
			7.85398163397448278999e-01,
			9.82793723247329054082e-01,
			1.57079632679489655800e+00,
		};

		private static ReadOnlySpan<double> AtanLo => new[]
		{
			2.26987774529616870924e-17,
			3.06161699786838301793e-17,
			1.39033110312309984516e-17,
			6.12323399573676603587e-17,
		};

		private static ReadOnlySpan<double> AT => new[]
		{
			3.33333333333329318027e-01,
			-1.99999999998764832476e-01,
			1.42857142725034663711e-01,
			-1.11111104054623557880e-01,
			9.09088713343650656196e-02,
			-7.69187620504482999495e-02,
			6.66107313738753120669e-02,
			-5.83357013379057348645e-02,
			4.97687799461593236017e-02,
			-3.65315727442169155270e-02,
			1.62858201153657823623e-02,
		};

		/// <summary>
		/// Deterministic arctangent. Produces bit-identical results on all platforms.
		/// </summary>
		public static double Atan(double x)
		{
			uint ix = HighWord(x);
			uint sign = ix >> 31;
			ix &= 0x7fff_ffffu;

			if (ix >= 0x4410_0000u)
			{
				// |x| >= 2^66
				if (double.IsNaN(x))
				{
					return x;
				}

				double zBig = AtanHi[3] + BitConverter.UInt64BitsToDouble(0x3987_0000_0000_0000UL); // 0x1p-120
				return sign != 0 ? -zBig : zBig;
			}

			int id;
			if (ix < 0x3fdc_0000u)
			{
				// |x| < 0.4375
				if (ix < 0x3e40_0000u)
				{
					return x; // |x| < 2^-27
				}

				id = -1;
			}
			else
			{
				// f64::abs and Math.Abs are the same sign-bit clear, exact on every
				// input including zeros and NaN payloads.
				x = Math.Abs(x);
				if (ix < 0x3ff3_0000u)
				{
					// |x| < 1.1875
					if (ix < 0x3fe6_0000u)
					{
						// 7/16 <= |x| < 11/16
						id = 0;
						x = (2.0 * x - 1.0) / (2.0 + x);
					}
					else
					{
						// 11/16 <= |x| < 19/16
						id = 1;
						x = (x - 1.0) / (x + 1.0);
					}
				}
				else if (ix < 0x4003_8000u)
				{
					// |x| < 2.4375
					id = 2;
					x = (x - 1.5) / (1.0 + 1.5 * x);
				}
				else
				{
					// 2.4375 <= |x| < 2^66
					id = 3;
					x = -1.0 / x;
				}
			}

			double z = x * x;
			double w = z * z;
			double s1 = z * (AT[0] + w * (AT[2] + w * (AT[4] + w * (AT[6] + w * (AT[8] + w * AT[10])))));
			double s2 = w * (AT[1] + w * (AT[3] + w * (AT[5] + w * (AT[7] + w * AT[9]))));

			if (id < 0)
			{
				return x - x * (s1 + s2);
			}

			double zz = AtanHi[id] - (x * (s1 + s2) - AtanLo[id] - x);
			return sign != 0 ? -zz : zz;
		}

		/// <summary>
		/// Deterministic atan2. Produces bit-identical results on all platforms.
		/// </summary>
		public static double Atan2(double y, double x)
		{
			const double PI = 3.1415926535897931160E+00;
			const double PI_LO = 1.2246467991473531772E-16;

			if (double.IsNaN(x) || double.IsNaN(y))
			{
				return x + y;
			}

			uint ix = HighWord(x);
			uint iy = HighWord(y);
			uint lx = LowWord(x);
			uint ly = LowWord(y);

			if (unchecked((ix - 0x3ff0_0000u) | lx) == 0)
			{
				return Atan(y); // x = 1.0
			}

			uint m = ((iy >> 31) & 1u) | ((ix >> 30) & 2u);
			ix &= 0x7fff_ffffu;
			iy &= 0x7fff_ffffu;

			if ((iy | ly) == 0)
			{
				// y = 0
				switch (m)
				{
					case 0:
					case 1:
						return y;
					case 2:
						return PI;
					default:
						return -PI;
				}
			}

			if ((ix | lx) == 0)
			{
				// x = 0
				return (m & 1) != 0 ? -PI / 2.0 : PI / 2.0;
			}

			if (ix == 0x7ff0_0000u)
			{
				// x is inf
				if (iy == 0x7ff0_0000u)
				{
					// y is also inf
					switch (m)
					{
						case 0:
							return PI / 4.0;
						case 1:
							return -PI / 4.0;
						case 2:
							return 3.0 * PI / 4.0;
						default:
							return -3.0 * PI / 4.0;
					}
				}

				switch (m)
				{
					case 0:
						return 0.0;
					case 1:
						return -0.0;
					case 2:
						return PI;
					default:
						return -PI;
				}
			}

			// Neither ix nor iy can overflow a uint here: both are masked to
			// 0x7fffffff and 64 << 20 is 0x4000000, so the Rust's checked add
			// never trips and the unchecked C# add is the same value.
			if (unchecked(ix + (64u << 20)) < iy || iy == 0x7ff0_0000u)
			{
				// |y/x| > 2^64
				return (m & 1) != 0 ? -PI / 2.0 : PI / 2.0;
			}

			double z;
			if ((m & 2) != 0 && unchecked(iy + (64u << 20)) < ix)
			{
				z = 0.0; // |y/x| < 2^-64 and x < 0
			}
			else
			{
				z = Atan(Math.Abs(y / x)); // Math.Abs: sign-bit clear, see Atan
			}

			switch (m)
			{
				case 0:
					return z;
				case 1:
					return -z;
				case 2:
					return PI - (z - PI_LO);
				default:
					return (z - PI_LO) - PI;
			}
		}
	}
}
