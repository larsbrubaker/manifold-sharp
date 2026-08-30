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

// IntPred.cs — port of robust/exact/intpred.rs. Exact integer evaluation of the
// raw-f64 predicates, the fast escalation tier between the float filters
// (Filtered.cs) and the exact rational ground truth (Predicates.cs).
//
// Every finite f64 is a dyadic rational m·2^e. Scaling all four inputs of a
// predicate *per coordinate axis* by a common power of two turns them into
// integers without changing the determinant's sign (the scale factors are
// positive and factor out of each column). The determinant is then evaluated
// in the narrowest integer width a static bit-length budget proves cannot
// overflow — long, then Int128, then BigInteger — either way the sign is exact,
// with no rational normalization (gcds) anywhere. On the degenerate-heavy meshes
// that defeat the float filters (doubled surfaces, large flat regions) this
// tier is 10–50× cheaper than the rational fallback it replaces.
//
// The i64 tier exists mainly for wasm, where a 64×64→128 multiply lowers to a
// compiler-rt `__multi3` call while an i64 multiply is a single `i64.mul`.
// Every budget below is derived from worst-case magnitudes, not measured, and
// the tier declines (falls through to Int128) the moment any operand exceeds it.
//
// Two C# shape notes:
//   * Rust's `[f64; N]` per-axis inputs are ReadOnlySpan<double> here, so the
//     callers pass stackalloc buffers and nothing allocates; the Rust's
//     `Option<[i128; N]>` return becomes a bool plus a caller-owned output span
//     for the same reason.
//   * The Rust's tier counter is `#[cfg(test)]`. This one is always compiled:
//     the C# suite runs against the same Release assembly it ships, so a
//     conditionally compiled counter would leave the tier-hit tests unable to
//     observe anything in exactly the configuration that matters. It costs one
//     thread-local increment on the i64 tier and nothing anywhere else.

using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Robust.Exact
{
	/// <summary>
	/// Per-thread count of predicate calls the i64 tier resolved, so tests can prove
	/// their inputs actually reach that tier instead of silently checking the Int128
	/// path twice. Thread-local (not atomic) to stay race-free while the test
	/// harness runs tests in parallel, and internal rather than public because it is
	/// instrumentation, not API — the Rust compiles it out of non-test builds
	/// entirely, which this port cannot do (see the file header).
	/// </summary>
	internal static class TierStats
	{
		[ThreadStatic]
		private static ulong i64Hits;

		/// <summary>Zeroes this thread's counter.</summary>
		internal static void Reset()
		{
			i64Hits = 0;
		}

		/// <summary>This thread's count of i64-tier resolutions since the last reset.</summary>
		internal static ulong I64Hits()
		{
			return i64Hits;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		internal static void NoteI64()
		{
			i64Hits++;
		}
	}

	/// <summary>
	/// The integer tier of the exact predicates: per-axis power-of-two scaling into
	/// long, Int128 or BigInteger, whichever the static bit-length budget admits.
	/// </summary>
	public static class IntPred
	{
		/// <summary>
		/// Exact dyadic decomposition: returns (m, e) with v == m·2^e and trailing
		/// zeros stripped from m (minimizing later shift widths). v must be finite.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static (long M, int E) Decomp(double v)
		{
			Debug.Assert(double.IsFinite(v), "Decomp requires a finite value");
			if (v == 0.0)
			{
				return (0L, 0);
			}

			ulong bits = (ulong)BitConverter.DoubleToInt64Bits(v);
			int biased = (int)((bits >> 52) & 0x7ff);
			ulong frac = bits & ((1UL << 52) - 1);
			ulong m;
			int e;
			if (biased == 0)
			{
				m = frac; // subnormal: no hidden bit
				e = -1074;
			}
			else
			{
				m = frac | (1UL << 52);
				e = biased - 1075;
			}

			int tz = BitOperations.TrailingZeroCount(m);
			m >>= tz;
			e += tz;
			return (v < 0.0 ? -(long)m : (long)m, e);
		}

		/// <summary>
		/// The common scale exponent of one coordinate axis: the smallest dyadic
		/// exponent among the nonzero values (zeros never constrain it).
		/// </summary>
		private static int MinExponent(ReadOnlySpan<long> ms, ReadOnlySpan<int> es)
		{
			int emin = 0;
			bool any = false;
			for (int i = 0; i < ms.Length; i++)
			{
				if (ms[i] == 0)
				{
					continue;
				}

				if (!any || es[i] < emin)
				{
					emin = es[i];
					any = true;
				}
			}

			return emin;
		}

		/// <summary>
		/// The values of one coordinate axis as exactly scaled Int128 integers, or
		/// false when any scaled magnitude would exceed <paramref name="budget"/> bits
		/// (the caller then takes the BigInteger path). Zeros scale to zero regardless
		/// of exponent.
		/// </summary>
		private static bool ScaledI128(ReadOnlySpan<double> vs, uint budget, Span<Int128> result)
		{
			Debug.Assert(result.Length >= vs.Length, "the caller's output span is shorter than the axis");
			int n = vs.Length;
			Span<long> ms = stackalloc long[n];
			Span<int> es = stackalloc int[n];
			for (int i = 0; i < n; i++)
			{
				(ms[i], es[i]) = Decomp(vs[i]);
			}

			int emin = MinExponent(ms, es);
			for (int i = 0; i < n; i++)
			{
				result[i] = Int128.Zero;
				long m = ms[i];
				if (m == 0)
				{
					continue;
				}

				uint shift = (uint)(es[i] - emin);
				uint bits = (uint)(64 - BitOperations.LeadingZeroCount(UnsignedAbs(m))) + shift;
				if (bits > budget)
				{
					return false;
				}

				result[i] = (Int128)m << (int)shift;
			}

			return true;
		}

		/// <summary>
		/// The same per-axis scaling into long, or false when any scaled magnitude
		/// would exceed <paramref name="budget"/> bits. Used by the i64 fast tier,
		/// which thereby avoids touching 128-bit arithmetic at all (see
		/// <see cref="Orient2dI"/>).
		/// </summary>
		internal static bool ScaledI64(ReadOnlySpan<double> vs, uint budget, Span<long> result)
		{
			Debug.Assert(budget < 63, "the i64 budget must leave room for the accumulation");
			Debug.Assert(result.Length >= vs.Length, "the caller's output span is shorter than the axis");
			int n = vs.Length;
			Span<long> ms = stackalloc long[n];
			Span<int> es = stackalloc int[n];
			for (int i = 0; i < n; i++)
			{
				(ms[i], es[i]) = Decomp(vs[i]);
			}

			int emin = MinExponent(ms, es);
			for (int i = 0; i < n; i++)
			{
				result[i] = 0L;
				long m = ms[i];
				if (m == 0)
				{
					continue;
				}

				uint shift = (uint)(es[i] - emin);

				// `shift` is unbounded here, so test the bit length before shifting.
				uint bits = (uint)(64 - BitOperations.LeadingZeroCount(UnsignedAbs(m))) + shift;
				if (bits > budget)
				{
					return false;
				}

				result[i] = m << (int)shift;
			}

			return true;
		}

		/// <summary>
		/// The same scaling with unbounded BigInteger magnitudes. Public: the tri-tri
		/// interval overlap (robust/tri_tri.rs) reuses it to compare interval endpoints
		/// along the common plane-intersection line in pure integers.
		/// </summary>
		public static BigInteger[] ScaledBig(ReadOnlySpan<double> vs)
		{
			int n = vs.Length;
			Span<long> ms = stackalloc long[n];
			Span<int> es = stackalloc int[n];
			for (int i = 0; i < n; i++)
			{
				(ms[i], es[i]) = Decomp(vs[i]);
			}

			int emin = MinExponent(ms, es);
			BigInteger[] result = new BigInteger[n];
			for (int i = 0; i < n; i++)
			{
				result[i] = ms[i] == 0 ? BigInteger.Zero : (BigInteger)ms[i] << (es[i] - emin);
			}

			return result;
		}

		/// <summary>Magnitude of a long as an unsigned value.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static ulong UnsignedAbs(long v)
		{
			return v < 0 ? (ulong)(-v) : (ulong)v;
		}

		/// <summary>
		/// True when |v| &lt; 2^bits, i.e. v fits in <paramref name="bits"/> magnitude
		/// bits.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static bool Fits(Int128 v, uint bits)
		{
			UInt128 mag = v < Int128.Zero ? (UInt128)(-v) : (UInt128)v;
			return mag < ((UInt128)1 << (int)bits);
		}

		/// <summary>
		/// Narrows three Int128 values to long, or false if any exceeds
		/// <paramref name="bits"/> magnitude bits. Declining is always safe: the caller
		/// stays on the Int128 path.
		/// </summary>
		internal static bool Narrow3(ReadOnlySpan<Int128> vs, uint bits, Span<long> result)
		{
			// The Rust takes [i128; 3] and returns Option<[i64; 3]>, so both lengths are
			// the type. Internal and called from Orient3dI alone, so an assert carries
			// the invariant the type used to.
			Debug.Assert(vs.Length == 3 && result.Length == 3, "Narrow3 is exactly three values wide");
			Debug.Assert(bits < 63, "the narrowed values must still multiply inside i64");
			if (Fits(vs[0], bits) && Fits(vs[1], bits) && Fits(vs[2], bits))
			{
				result[0] = (long)vs[0];
				result[1] = (long)vs[1];
				result[2] = (long)vs[2];
				return true;
			}

			return false;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static Sign SignI64(long v)
		{
			if (v < 0)
			{
				return Sign.Neg;
			}
			else if (v == 0)
			{
				return Sign.Zero;
			}
			else
			{
				return Sign.Pos;
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static Sign SignI128(Int128 v)
		{
			if (v < Int128.Zero)
			{
				return Sign.Neg;
			}
			else if (v == Int128.Zero)
			{
				return Sign.Zero;
			}
			else
			{
				return Sign.Pos;
			}
		}

		private static Sign SignBig(in BigInteger v)
		{
			if (v.IsZero)
			{
				return Sign.Zero;
			}
			else if (v.Sign < 0)
			{
				return Sign.Neg;
			}
			else
			{
				return Sign.Pos;
			}
		}

		/// <summary>
		/// Exact sign of the orient2d determinant cross(b-a, c-a) on raw f64 points.
		/// </summary>
		/// <remarks>
		/// i64 budget 30: scaled values satisfy |v| ≤ 2^30−1, so each difference is
		/// |d| ≤ 2^31−2, each product ≤ (2^31−2)^2 = 2^62 − 2^33 + 4, and the final
		/// subtraction of two such products is bounded by
		///   2·(2^62 − 2^33 + 4) = 2^63 − 2^34 + 8 &lt; 2^63 − 1 = long.MaxValue.
		/// Budget 31 would not do: 2·(2^32−2)^2 = 2^65 − 2^36 + 8 overflows.
		/// <para>
		/// i128 budget: scaled values ≤ 61 bits ⇒ differences ≤ 62, products ≤ 124,
		/// final subtraction ≤ 125 bits — safely inside Int128.
		/// </para>
		/// </remarks>
		public static Sign Orient2dI(Vec2 a, Vec2 b, Vec2 c)
		{
			Span<long> lx = stackalloc long[3];
			Span<long> ly = stackalloc long[3];
			if (ScaledI64(stackalloc double[] { a.X, b.X, c.X }, 30, lx)
				&& ScaledI64(stackalloc double[] { a.Y, b.Y, c.Y }, 30, ly))
			{
				TierStats.NoteI64();
				return SignI64(((lx[1] - lx[0]) * (ly[2] - ly[0])) - ((ly[1] - ly[0]) * (lx[2] - lx[0])));
			}

			Span<Int128> wx = stackalloc Int128[3];
			Span<Int128> wy = stackalloc Int128[3];
			if (ScaledI128(stackalloc double[] { a.X, b.X, c.X }, 61, wx)
				&& ScaledI128(stackalloc double[] { a.Y, b.Y, c.Y }, 61, wy))
			{
				return SignI128(((wx[1] - wx[0]) * (wy[2] - wy[0])) - ((wy[1] - wy[0]) * (wx[2] - wx[0])));
			}

			BigInteger[] bx = ScaledBig(stackalloc double[] { a.X, b.X, c.X });
			BigInteger[] by = ScaledBig(stackalloc double[] { a.Y, b.Y, c.Y });
			return SignBig(((bx[1] - bx[0]) * (by[2] - by[0])) - ((by[1] - by[0]) * (bx[2] - bx[0])));
		}

		/// <summary>
		/// Exact sign of the orient3d determinant dot(cross(b-a, c-a), d-a) on raw f64
		/// points.
		/// </summary>
		/// <remarks>
		/// i128 budget: scaled values ≤ 40 bits ⇒ differences ≤ 41, 2-products ≤ 82,
		/// their difference ≤ 83, 3-products ≤ 124, 3-term sum ≤ 126 bits.
		/// <para>
		/// i64 tier, budget 20 on the *edge differences* rather than on the scaled
		/// values: with D = 2^20−1 bounding each of u, v, w componentwise,
		///   2-product        ≤ D²        = 2^40 − 2^21 + 1;
		///   2×2 minor        ≤ 2D²       = 2^41 − 2^22 + 2;
		///   minor × 3rd term ≤ 2D³;
		///   3-term sum       ≤ 6D³       = 6·1152918206075109375
		///                                = 6917509236450656250 &lt; 2^63 − 1
		/// (partial sums are bounded by 4D³, so no intermediate overflows either).
		/// Budget 21 fails: 6·(2^21−1)³ ≈ 5.5·10^19 ≫ long.MaxValue.
		/// </para>
		/// <para>
		/// The check is on the differences, not on the scaled values, deliberately:
		/// the equivalent value budget is 19 bits (values ≤ 2^19−1 ⇒ differences
		/// ≤ 2^20−2), which instrumentation on the 1075458 ∪ 91115 workload showed
		/// never fires — real mesh coordinates carry ≥ 24-bit mantissas — whereas the
		/// difference form fires on the calls where the four points are near each
		/// other, which is the common geometric case.
		/// </para>
		/// </remarks>
		public static Sign Orient3dI(Vec3 a, Vec3 b, Vec3 c, Vec3 d)
		{
			Span<Int128> xs = stackalloc Int128[4];
			Span<Int128> ys = stackalloc Int128[4];
			Span<Int128> zs = stackalloc Int128[4];
			if (ScaledI128(stackalloc double[] { a.X, b.X, c.X, d.X }, 40, xs)
				&& ScaledI128(stackalloc double[] { a.Y, b.Y, c.Y, d.Y }, 40, ys)
				&& ScaledI128(stackalloc double[] { a.Z, b.Z, c.Z, d.Z }, 40, zs))
			{
				Int128 ux = xs[1] - xs[0];
				Int128 uy = ys[1] - ys[0];
				Int128 uz = zs[1] - zs[0];
				Int128 vx = xs[2] - xs[0];
				Int128 vy = ys[2] - ys[0];
				Int128 vz = zs[2] - zs[0];
				Int128 wx = xs[3] - xs[0];
				Int128 wy = ys[3] - ys[0];
				Int128 wz = zs[3] - zs[0];

				Span<long> nu = stackalloc long[3];
				Span<long> nv = stackalloc long[3];
				Span<long> nw = stackalloc long[3];
				if (Narrow3(stackalloc Int128[] { ux, uy, uz }, 20, nu)
					&& Narrow3(stackalloc Int128[] { vx, vy, vz }, 20, nv)
					&& Narrow3(stackalloc Int128[] { wx, wy, wz }, 20, nw))
				{
					long ndet = (nu[0] * ((nv[1] * nw[2]) - (nv[2] * nw[1])))
						+ (nu[1] * ((nv[2] * nw[0]) - (nv[0] * nw[2])))
						+ (nu[2] * ((nv[0] * nw[1]) - (nv[1] * nw[0])));
					TierStats.NoteI64();
					return SignI64(ndet);
				}

				Int128 det = (ux * ((vy * wz) - (vz * wy)))
					+ (uy * ((vz * wx) - (vx * wz)))
					+ (uz * ((vx * wy) - (vy * wx)));
				return SignI128(det);
			}

			BigInteger[] bx = ScaledBig(stackalloc double[] { a.X, b.X, c.X, d.X });
			BigInteger[] by = ScaledBig(stackalloc double[] { a.Y, b.Y, c.Y, d.Y });
			BigInteger[] bz = ScaledBig(stackalloc double[] { a.Z, b.Z, c.Z, d.Z });
			BigInteger gux = bx[1] - bx[0];
			BigInteger guy = by[1] - by[0];
			BigInteger guz = bz[1] - bz[0];
			BigInteger gvx = bx[2] - bx[0];
			BigInteger gvy = by[2] - by[0];
			BigInteger gvz = bz[2] - bz[0];
			BigInteger gwx = bx[3] - bx[0];
			BigInteger gwy = by[3] - by[0];
			BigInteger gwz = bz[3] - bz[0];
			BigInteger bdet = (gux * ((gvy * gwz) - (gvz * gwy)))
				+ (guy * ((gvz * gwx) - (gvx * gwz)))
				+ (guz * ((gvx * gwy) - (gvy * gwx)));
			return SignBig(bdet);
		}
	}
}
