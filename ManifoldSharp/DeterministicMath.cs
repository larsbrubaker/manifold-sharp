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
// the platform-dependent System.Math.Sin/Cos/etc.
//
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro/SunSoft, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.

// Port of math.rs. Split across two files to stay under the 800-line cap: this
// one holds the bit-manipulation helpers, the reduced-range kernels, the pi/2
// argument reduction and the forward functions (Sin, Cos, Tan);
// DeterministicMath.Inverse.cs holds Acos, Asin, Atan and Atan2.
//
// Nothing in either file may call a System.Math transcendental - avoiding
// platform libm is the entire reason the file exists. Three System.Math methods
// do appear, across five call sites: Math.Round once here, and in the inverse
// file Math.Sqrt twice (both in Acos) and Math.Abs twice (Atan, Atan2). All
// three are IEEE-exact primitives standing in for the identical Rust primitive,
// and each is commented at its call site.
//
// The msun constant names (PIO2_1, S1..S6, T[..]) are kept verbatim rather than
// renamed to C# casing, so this stays diffable against musl, the C source and
// the Rust port.

using System.Runtime.CompilerServices;

namespace ManifoldSharp
{
	/// <summary>
	/// Deterministic replacements for the platform's trigonometric functions,
	/// transcribed from FreeBSD msun via musl libc so that every platform - and
	/// this port versus manifold-rust - produces bit-identical results.
	/// </summary>
	public static partial class DeterministicMath
	{
		// -------------------------------------------------------------------
		// Bit-manipulation helpers
		// -------------------------------------------------------------------

		// Rust's f64::to_bits/from_bits are unsigned and every mask below is
		// written against a u64, so BitConverter's unsigned overloads are the
		// literal translation. They are also the safe one: the signed
		// DoubleToInt64Bits would sign-extend into the bitwise ors here, which
		// is the CS0675 class of silent corruption this repo turns into an error.
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static uint HighWord(double x)
		{
			return (uint)(BitConverter.DoubleToUInt64Bits(x) >> 32);
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static uint LowWord(double x)
		{
			return unchecked((uint)BitConverter.DoubleToUInt64Bits(x));
		}

		/// <summary>
		/// Replace the lower 32 bits of <paramref name="x"/> with
		/// <paramref name="low"/>, preserving the upper 32 bits.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static double WithLowWord(double x, uint low)
		{
			ulong u = (BitConverter.DoubleToUInt64Bits(x) & 0xffff_ffff_0000_0000UL) | low;
			return BitConverter.UInt64BitsToDouble(u);
		}

		/// <summary>
		/// Rust's <c>as i32</c> on an <c>f64</c>: truncate toward zero, saturating at
		/// the <see cref="int"/> bounds, with NaN mapping to zero.
		/// </summary>
		/// <remarks>
		/// C#'s <c>(int)</c> cast on an out-of-range double is architecture-dependent
		/// in practice (x64 yields <see cref="int.MinValue"/>, ARM64 saturates), so a
		/// bare cast would break bit-identity across platforms on exactly the huge
		/// arguments the last branch of <see cref="RemPio2"/> exists to handle.
		/// </remarks>
		private static int SaturatingToInt32(double value)
		{
			if (double.IsNaN(value))
			{
				return 0;
			}

			if (value >= 2147483647.0)
			{
				return int.MaxValue;
			}

			if (value <= -2147483648.0)
			{
				return int.MinValue;
			}

			return (int)value;
		}

		// -------------------------------------------------------------------
		// Kernel functions (reduced-range polynomial approximations)
		// -------------------------------------------------------------------

		/// <summary>
		/// Kernel sin for |x| in [-pi/4, pi/4], y is the tail of x.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static double SinKernel(double x, double y, int iy)
		{
			const double S1 = -1.66666666666666324348e-01;
			const double S2 = 8.33333333332248946124e-03;
			const double S3 = -1.98412698298579493134e-04;
			const double S4 = 2.75573137070700676789e-06;
			const double S5 = -2.50507602534068634195e-08;
			const double S6 = 1.58969099521155010221e-10;

			double z = x * x;
			double w = z * z;
			double r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6);
			double v = z * x;
			if (iy == 0)
			{
				return x + v * (S1 + z * r);
			}

			return x - ((z * (0.5 * y - v * r) - y) - v * S1);
		}

		/// <summary>
		/// Kernel cos for |x| in [-pi/4, pi/4], y is the tail of x.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static double CosKernel(double x, double y)
		{
			const double C1 = 4.16666666666666019037e-02;
			const double C2 = -1.38888888888741095749e-03;
			const double C3 = 2.48015872894767294178e-05;
			const double C4 = -2.75573143513906633035e-07;
			const double C5 = 2.08757232129817482790e-09;
			const double C6 = -1.13596475577881948265e-11;

			double z = x * x;
			double w = z * z;
			double r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
			double hz = 0.5 * z;
			double w1 = 1.0 - hz;
			return w1 + (((1.0 - w1) - hz) + (z * r - x * y));
		}

		// Coefficients of the tan kernel's polynomial. Rust holds these in a
		// `const [f64; 13]`; C# has no const array, and a static readonly array
		// would still be writable through its elements. A ReadOnlySpan property
		// over an array initializer is the closest thing available: the compiler
		// lowers it to an immutable metadata blob with no allocation and no
		// mutable static to corrupt. The indices below are the msun ones
		// (T[0]..T[12]) so the odd/even interleaving still reads as it does in
		// the C source.
		private static ReadOnlySpan<double> T => new[]
		{
			3.33333333333334091986e-01,
			1.33333333333201242699e-01,
			5.39682539762260521377e-02,
			2.18694882948595424599e-02,
			8.86323982359930005737e-03,
			3.59207910759131235356e-03,
			1.45620945432529025516e-03,
			5.88041240820264096874e-04,
			2.46463134818469906812e-04,
			7.81794442939557092300e-05,
			7.14072491382608190305e-05,
			-1.85586374855275456654e-05,
			2.59073051863633712884e-05,
		};

		/// <summary>
		/// Kernel tan for |x| in [-pi/4, pi/4]. <paramref name="odd"/> is 1 for
		/// computing -1/tan(x).
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static double TanKernel(double x, double y, int odd)
		{
			const double PIO4 = 7.85398163397448278999e-01;
			const double PIO4LO = 3.06161699786838301793e-17;

			uint hx = HighWord(x);
			bool big = (hx & 0x7fff_ffffu) >= 0x3FE5_9428u; // |x| >= 0.6744
			bool sign = false;
			if (big)
			{
				sign = (hx >> 31) != 0;
				if (sign)
				{
					x = -x;
					y = -y;
				}

				x = (PIO4 - x) + (PIO4LO - y);
				y = 0.0;
			}

			double z = x * x;
			double w = z * z;
			double r = T[1] + w * (T[3] + w * (T[5] + w * (T[7] + w * (T[9] + w * T[11]))));
			double v = z * (T[2] + w * (T[4] + w * (T[6] + w * (T[8] + w * (T[10] + w * T[12])))));
			double s = z * x;
			double rr = y + z * (s * (r + v) + y) + s * T[0];
			double ww = x + rr;
			if (big)
			{
				double s2 = 1.0 - 2.0 * (double)odd;
				double vvBig = s2 - 2.0 * (x + (rr - ww * ww / (ww + s2)));
				return sign ? -vvBig : vvBig;
			}

			if (odd == 0)
			{
				return ww;
			}

			// Compute -1/(x+r) with reduced cancellation error.
			double w0 = WithLowWord(ww, 0);
			double vv = rr - (w0 - x);
			double aa = -1.0 / ww;
			double a0 = WithLowWord(aa, 0);
			return a0 + aa * (1.0 + a0 * w0 + a0 * vv);
		}

		// -------------------------------------------------------------------
		// Argument reduction: reduce x to y0+y1 in [-pi/4, pi/4]
		// -------------------------------------------------------------------

		/// <summary>
		/// Reduce <paramref name="x"/> modulo pi/2. Returns quadrant n and sets
		/// <paramref name="y0"/>, <paramref name="y1"/> such that
		/// x = n * pi/2 + y0 + y1 with |y0+y1| &lt;= pi/4.
		/// </summary>
		private static int RemPio2(double x, out double y0, out double y1)
		{
			const double PIO2_1 = 1.57079632673412561417e+00;
			const double PIO2_1T = 6.07710050650619224932e-11;
			const double HALF_PI = 1.57079632679489661923132169163975144;

			ulong ux = BitConverter.DoubleToUInt64Bits(x);
			bool sign = (ux >> 63) != 0;
			uint ix = (uint)((ux >> 32) & 0x7fff_ffffUL);
			double z;

			if (ix <= 0x400f_6a7au)
			{
				// |x| ~<= 5pi/4
				if ((ix & 0xf_ffffu) != 0x9_21fbu)
				{
					// not near pi/2 multiples — try fast paths
					if (ix <= 0x4002_d97cu)
					{
						// |x| ~<= 3pi/4
						if (!sign)
						{
							z = x - PIO2_1;
							y0 = z - PIO2_1T;
							y1 = (z - y0) - PIO2_1T;
							return 1;
						}

						z = x + PIO2_1;
						y0 = z + PIO2_1T;
						y1 = (z - y0) + PIO2_1T;
						return -1;
					}

					if (!sign)
					{
						z = x - 2.0 * PIO2_1;
						y0 = z - 2.0 * PIO2_1T;
						y1 = (z - y0) - 2.0 * PIO2_1T;
						return 2;
					}

					z = x + 2.0 * PIO2_1;
					y0 = z + 2.0 * PIO2_1T;
					y1 = (z - y0) + 2.0 * PIO2_1T;
					return -2;
				}

				// Fall through to "medium" path below
				return RemPio2Medium(x, ix, out y0, out y1);
			}

			if (ix <= 0x401c_463bu)
			{
				// |x| ~<= 9pi/4
				if (ix <= 0x4015_fdbcu)
				{
					// |x| ~<= 7pi/4
					if (ix == 0x4012_d97cu)
					{
						return RemPio2Medium(x, ix, out y0, out y1);
					}

					if (!sign)
					{
						z = x - 3.0 * PIO2_1;
						y0 = z - 3.0 * PIO2_1T;
						y1 = (z - y0) - 3.0 * PIO2_1T;
						return 3;
					}

					z = x + 3.0 * PIO2_1;
					y0 = z + 3.0 * PIO2_1T;
					y1 = (z - y0) + 3.0 * PIO2_1T;
					return -3;
				}

				if (ix == 0x4019_21fbu)
				{
					return RemPio2Medium(x, ix, out y0, out y1);
				}

				if (!sign)
				{
					z = x - 4.0 * PIO2_1;
					y0 = z - 4.0 * PIO2_1T;
					y1 = (z - y0) - 4.0 * PIO2_1T;
					return 4;
				}

				z = x + 4.0 * PIO2_1;
				y0 = z + 4.0 * PIO2_1T;
				y1 = (z - y0) + 4.0 * PIO2_1T;
				return -4;
			}

			if (ix < 0x4139_21fbu)
			{
				// |x| ~< 2^20*(pi/2), medium size
				return RemPio2Medium(x, ix, out y0, out y1);
			}

			if (ix >= 0x7ff0_0000u)
			{
				// x is inf or NaN
				double v = x - x;
				y0 = v;
				y1 = v;
				return 0;
			}

			// Very large arguments: fall back to round-based reduction.
			// Rust doesn't have remquo in std, so we use the equivalent.
			//
			// "Equivalent" overstates it, and the overstatement is inherited from
			// the Rust comment rather than introduced here. musl reduces this range
			// with Payne-Hanek, which carries enough bits of 2/pi to stay accurate;
			// the Rust port replaced it with the naive round-and-subtract below, so
			// past |x| >= 2^20*(pi/2) the reduction loses every significant bit and
			// accuracy falls off a cliff: Sin(1e22) returns exactly -1.0 where the
			// true value is about -0.852, and Tan of a huge argument comes back NaN.
			// That is not a defect to fix on this side - it is what manifold-rust
			// computes, it matches bit for bit, and diverging here would break the
			// exactness bar. Anything needing accurate huge-argument trig must not
			// route through this file.
			//
			// Rust's f64::round is ties-away-from-zero; C#'s one-argument
			// Math.Round is ties-to-even, so naming the mode is not optional. With
			// the mode named this is exact integral rounding of a double - no libm,
			// no platform variance.
			double quotient = Math.Round(x / HALF_PI, MidpointRounding.AwayFromZero);
			int q = SaturatingToInt32(quotient);
			y0 = x - quotient * HALF_PI;
			y1 = 0.0;
			return q;
		}

		/// <summary>
		/// Medium-range argument reduction for <see cref="RemPio2"/>.
		/// </summary>
		private static int RemPio2Medium(double x, uint ix, out double y0, out double y1)
		{
			// f64::EPSILON is 2^-52. C#'s double.Epsilon is the smallest subnormal
			// and would be catastrophically wrong here, so the literal is spelled out.
			const double TOINT = 1.5 / 2.220446049250313E-16;
			const double PIO4 = 7.85398163397448278999e-01; // 0x1.921fb54442d18p-1
			const double INVPIO2 = 6.36619772367581382433e-01;
			const double PIO2_1 = 1.57079632673412561417e+00;
			const double PIO2_1T = 6.07710050650619224932e-11;
			const double PIO2_2 = 6.07710050630396597660e-11;
			const double PIO2_2T = 2.02226624879595063154e-21;
			const double PIO2_3 = 2.02226624871116645580e-21;
			const double PIO2_3T = 8.47842766036889956997e-32;

			double fn = x * INVPIO2 + TOINT - TOINT;
			int n = SaturatingToInt32(fn);
			double r = x - fn * PIO2_1;
			double w = fn * PIO2_1T;

			if (r - w < -PIO4)
			{
				n -= 1;
				fn -= 1.0;
				r = x - fn * PIO2_1;
				w = fn * PIO2_1T;
			}
			else if (r - w > PIO4)
			{
				n += 1;
				fn += 1.0;
				r = x - fn * PIO2_1;
				w = fn * PIO2_1T;
			}

			y0 = r - w;
			ulong uy0 = BitConverter.DoubleToUInt64Bits(y0);
			int ey = (int)((uy0 >> 52) & 0x7ffUL);
			int ex = (int)(ix >> 20);

			if (ex - ey > 16)
			{
				double t = r;
				w = fn * PIO2_2;
				r = t - w;
				w = fn * PIO2_2T - ((t - r) - w);
				y0 = r - w;
				ulong uy0_2 = BitConverter.DoubleToUInt64Bits(y0);
				int ey2 = (int)((uy0_2 >> 52) & 0x7ffUL);
				if (ex - ey2 > 49)
				{
					double t2 = r;
					w = fn * PIO2_3;
					r = t2 - w;
					w = fn * PIO2_3T - ((t2 - r) - w);
					y0 = r - w;
				}
			}

			y1 = (r - y0) - w;
			return n;
		}

		// -------------------------------------------------------------------
		// Public trigonometric functions
		// -------------------------------------------------------------------

		/// <summary>
		/// Deterministic sine: the same bits on every platform, and the same bits as
		/// manifold-rust.
		/// </summary>
		/// <remarks>
		/// Reproducibility is the guarantee here, not accuracy. For
		/// |x| &gt;= 2^20*(pi/2) the argument reduction is the naive round-based one
		/// inherited from the Rust port and the result is badly wrong (though wrong
		/// identically everywhere) - see the huge-argument note in
		/// <see cref="RemPio2"/>.
		/// </remarks>
		public static double Sin(double x)
		{
			uint ix = (uint)((BitConverter.DoubleToUInt64Bits(x) >> 32) & 0x7fff_ffffUL);
			if (ix <= 0x3fe9_21fbu)
			{
				// |x| ~<= pi/4
				if (ix < 0x3e50_0000u)
				{
					return x; // |x| < 2^-26
				}

				return SinKernel(x, 0.0, 0);
			}

			if (ix >= 0x7ff0_0000u)
			{
				return x - x; // NaN or Inf
			}

			int n = RemPio2(x, out double y0, out double y1);
			switch (n & 3)
			{
				case 0:
					return SinKernel(y0, y1, 1);
				case 1:
					return CosKernel(y0, y1);
				case 2:
					return -SinKernel(y0, y1, 1);
				default:
					return -CosKernel(y0, y1);
			}
		}

		/// <summary>
		/// Deterministic cosine: the same bits on every platform, and the same bits
		/// as manifold-rust. Reproducible, not accurate past
		/// |x| &gt;= 2^20*(pi/2) - see <see cref="Sin"/>.
		/// </summary>
		public static double Cos(double x)
		{
			uint ix = (uint)((BitConverter.DoubleToUInt64Bits(x) >> 32) & 0x7fff_ffffUL);
			if (ix <= 0x3fe9_21fbu)
			{
				// |x| ~<= pi/4
				if (ix < 0x3e46_a09eu)
				{
					return 1.0;
				}

				return CosKernel(x, 0.0);
			}

			if (ix >= 0x7ff0_0000u)
			{
				return x - x; // NaN or Inf
			}

			int n = RemPio2(x, out double y0, out double y1);
			switch (n & 3)
			{
				case 0:
					return CosKernel(y0, y1);
				case 1:
					return -SinKernel(y0, y1, 1);
				case 2:
					return -CosKernel(y0, y1);
				default:
					return SinKernel(y0, y1, 1);
			}
		}

		/// <summary>
		/// Deterministic tangent: the same bits on every platform, and the same bits
		/// as manifold-rust. Reproducible, not accurate past
		/// |x| &gt;= 2^20*(pi/2), where huge arguments return NaN - see
		/// <see cref="Sin"/>.
		/// </summary>
		public static double Tan(double x)
		{
			uint ix = HighWord(x) & 0x7fff_ffffu;
			if (ix <= 0x3fe9_21fbu)
			{
				// |x| ~<= pi/4
				if (ix < 0x3e40_0000u)
				{
					return x;
				}

				return TanKernel(x, 0.0, 0);
			}

			if (ix >= 0x7ff0_0000u)
			{
				return x - x; // NaN or Inf
			}

			int n = RemPio2(x, out double y0, out double y1);
			return TanKernel(y0, y1, n & 1);
		}
	}
}
