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

// Backend.cs — port of robust/exact/backend.rs. The single seam between the
// robust engine and its arbitrary-precision arithmetic library.
//
// Every module in ManifoldSharp.Robust that needs unbounded integers or
// rationals goes through the aliases and helpers here; nothing else in the
// assembly names System.Numerics.BigInteger's own constructors or accessors
// directly. Swapping the backend is therefore a change to this file alone,
// plus a re-verification of the backend-coupled hot spots inventoried below.
//
// The Rust used dashu-int/dashu-ratio; this port uses System.Numerics.BigInteger
// plus the hand-written canonical BigRational below. The one structural
// difference that ripples outward: dashu has a distinct unsigned magnitude type
// (`UBig`) that the Rust names `Uint`, and BigInteger has no unsigned twin — so
// `Int` and `Uint` both land on BigInteger and the "always positive" property of
// a denominator is an invariant this file maintains rather than one the type
// system enforces. The bridging helpers (MulIntUint, MulUint, IntFromUint,
// IntMag) are kept anyway, exactly where the Rust has them: they are where a
// future reader looks to find every place the two roles meet, and they document
// which operand is which.
//
// ─── Phase-2 checklist: backend-coupled hot spots ───────────────────────────
//
// These are the places whose *correctness* (not just compilation) depends on
// how the backend represents and normalizes values. A backend swap must
// re-verify each one.
//
//  1. HashRational / RatEq (below), backing Rational.cs's R2Key / R3Key.
//     Structural hashing that reaches into the backend's representation: the
//     sign, then the little-endian bytes of the numerator magnitude, then the
//     bytes of the denominator. A replacement backend must expose an
//     equivalent canonical limb/word/byte view (or the hash must be
//     rewritten). Note the byte count is not hashed; the 0xfeed separator
//     between numerator and denominator is what keeps the stream unambiguous.
//     Preserve that if the unit width changes. The Rust hashes target-width
//     limbs where this hashes bytes, so the two produce DIFFERENT hash
//     streams — sanctioned, because these hashes are process-local (map
//     probing only) and never reach the output, and the maps that use them
//     document their order-independence. EQUALITY, by contrast, must be
//     exactly the Rust's.
//
//  2. Canonicality invariant behind (1). R2Key/R3Key equality is RatEq, i.e.
//     *field* equality (numerator and denominator compared independently),
//     which equals value equality only if every stored rational is fully
//     reduced with a positive denominator. Today that holds because all stored
//     rationals come from RatNew (gcd-reducing), RatFromF64, or the arithmetic
//     operators — all of which canonicalize. A replacement backend must
//     auto-reduce in its equivalent of RatNew and must normalize the sign onto
//     the numerator, or (1) breaks silently.
//
//  3. Predicates.cs construction sites that rely on "exactly one gcd":
//     LinePlaneIntersect, LineLineIntersect2d, LiftToPlane and SegmentParam
//     each build every output coordinate with a single RatNew, so the
//     reduction cost is paid exactly once per coordinate. The gcd is a
//     measured hot spot; re-measure it on a swap.
//
//  4. IntPred.cs BigInteger fallback tier: ScaledBig (exact dyadic scaling via
//     BigInteger shifts) and SignBig. This is the unbounded tier below
//     long/Int128 and is where big-integer performance shows up on degenerate
//     meshes.
//
//  5. Rational.cs RatToF64 — correctly rounded (nearest, ties to even)
//     rational -> f64. It works on the *unsigned magnitude* through
//     NumerMag/Denom, UintBits, UintBit, shifts, comparison, division,
//     subtraction, multiplication and addition.
//     It must NOT be replaced by any backend-provided conversion without a
//     differential test — some libraries' rational->float conversions are not
//     correctly rounded (num-rational's `ToPrimitive for Ratio` is not), and
//     the BCL has no rational->double at all: BigInteger's explicit double
//     conversion truncates toward zero, so numerator/denominator computed
//     through it would be doubly rounded and wrong.
//     (RatToF64 is the only rational->f64 rounding path in the engine.)
//
//  6. Predicates.Homog2 — a homogeneous 2D point holding backend integers
//     directly, and the BigInteger[3] normals from TriNormalInt / DotPointRaw /
//     DotDiffRaw, plus tri_tri.rs's `type Frac = (Int, Int)`. These are
//     assembly-internal API surfaces whose signatures change with the backend
//     type. (They stay inside the assembly: the public façade only ever sees
//     meshes and f64s.)
//
//  7. RatFromF64 (Rational.Rat) must be exact for every finite f64 and must
//     return null for NaN and infinity — the engine relies on the throw path
//     never triggering for finite input.

using System.Numerics;
using System.Runtime.CompilerServices;

namespace ManifoldSharp.Robust.Exact
{
	/// <summary>
	/// Exact rational, always stored fully reduced with the sign on the numerator
	/// and a strictly positive denominator. Every constructor and operator
	/// canonicalizes, which is what makes the field-wise <see cref="Backend.RatEq"/>
	/// and <see cref="Backend.HashRational"/> sound — see items (1) and (2) in the
	/// file header.
	/// </summary>
	/// <remarks>
	/// The Rust's <c>Rational</c> is the type alias for dashu's <c>RBig</c>, which
	/// enforces canonical form in its type (the non-reducing variant is a separate
	/// type, <c>Relaxed</c>, which the Rust never uses). Here the invariant is
	/// maintained by construction instead: the raw constructor is private, and
	/// <c>default(BigRational)</c> reads back as 0/1 rather than as the invalid 0/0
	/// its zeroed fields literally hold.
	/// </remarks>
	public readonly struct BigRational : IEquatable<BigRational>, IComparable<BigRational>
	{
		private readonly BigInteger numerator;

		/// <summary>
		/// The denominator, except that a zero field means one: a default-initialized
		/// struct has all-zero fields, and reading that back as 0/0 would be the one
		/// way to obtain a non-canonical value. Always read through
		/// <see cref="Denominator"/>.
		/// </summary>
		private readonly BigInteger denominator;

		/// <summary>Assumes the pair is already canonical.</summary>
		private BigRational(BigInteger numerator, BigInteger denominator)
		{
			this.numerator = numerator;
			this.denominator = denominator;
		}

		/// <summary>Signed numerator of the canonical form.</summary>
		public BigInteger Numerator
		{
			get { return this.numerator; }
		}

		/// <summary>Denominator of the canonical form; always strictly positive.</summary>
		public BigInteger Denominator
		{
			get { return this.denominator.IsZero ? BigInteger.One : this.denominator; }
		}

		/// <summary>The rational 0/1.</summary>
		public static BigRational Zero
		{
			get { return default(BigRational); }
		}

		/// <summary>The rational 1/1.</summary>
		public static BigRational One
		{
			get { return new BigRational(BigInteger.One, BigInteger.One); }
		}

		/// <summary>True when the value is exactly zero.</summary>
		public bool IsZero
		{
			get { return this.numerator.IsZero; }
		}

		/// <summary>True when the value is strictly negative.</summary>
		public bool IsNegative
		{
			get { return this.numerator.Sign < 0; }
		}

		/// <summary>True when the value is strictly positive.</summary>
		public bool IsPositive
		{
			get { return this.numerator.Sign > 0; }
		}

		/// <summary>
		/// <paramref name="numerator"/> / <paramref name="denominator"/>, fully
		/// reduced with the sign normalized onto the numerator. This is the single
		/// gcd whose cost item (3) in the file header accounts for.
		/// </summary>
		/// <exception cref="DivideByZeroException">The denominator is zero.</exception>
		public static BigRational Canonical(BigInteger numerator, BigInteger denominator)
		{
			if (denominator.IsZero)
			{
				throw new DivideByZeroException("BigRational: zero denominator");
			}

			if (numerator.IsZero)
			{
				return Zero;
			}

			if (denominator.Sign < 0)
			{
				numerator = -numerator;
				denominator = -denominator;
			}

			BigInteger g = BigInteger.GreatestCommonDivisor(numerator, denominator);
			if (!g.IsOne)
			{
				numerator /= g;
				denominator /= g;
			}

			return new BigRational(numerator, denominator);
		}

		/// <summary>The integer <paramref name="v"/> as a rational (denominator 1).</summary>
		public static BigRational FromInteger(BigInteger v)
		{
			// Already canonical: gcd(v, 1) == 1 for every v, so no reduction is owed.
			return v.IsZero ? Zero : new BigRational(v, BigInteger.One);
		}

		/// <summary>
		/// Exact conversion of a finite f64. Every finite f64 is a dyadic rational
		/// m·2^e, so nothing is lost; stripping m's trailing zeros first makes the
		/// pair coprime by construction, so this costs no gcd.
		/// </summary>
		public static BigRational FromDouble(double v)
		{
			long bits = BitConverter.DoubleToInt64Bits(v);
			int biased = (int)((bits >> 52) & 0x7ff);
			ulong frac = (ulong)bits & 0xF_FFFF_FFFF_FFFFUL;
			ulong m;
			int e;
			if (biased == 0)
			{
				// Subnormal: no hidden bit.
				m = frac;
				e = -1074;
			}
			else
			{
				m = frac | (1UL << 52);
				e = biased - 1075;
			}

			if (m == 0)
			{
				// Both zeroes land here: rational zero is unsigned, so -0.0's sign bit is
				// (harmlessly) lost, exactly as in the Rust.
				return Zero;
			}

			int tz = BitOperations.TrailingZeroCount(m);
			m >>= tz;
			e += tz;
			BigInteger n = v < 0.0 ? -(BigInteger)m : (BigInteger)m;
			if (e >= 0)
			{
				return new BigRational(n << e, BigInteger.One);
			}

			// m is odd and the denominator is a power of two, hence already coprime.
			return new BigRational(n, BigInteger.One << (-e));
		}

		/// <summary>Absolute value.</summary>
		public BigRational Abs()
		{
			return this.IsNegative ? new BigRational(-this.numerator, this.Denominator) : this;
		}

		/// <summary>Negation. Canonical form is preserved by construction.</summary>
		public static BigRational operator -(BigRational a)
		{
			return a.IsZero ? Zero : new BigRational(-a.numerator, a.Denominator);
		}

		/// <summary>Exact sum, canonicalized.</summary>
		public static BigRational operator +(BigRational a, BigRational b)
		{
			BigInteger ad = a.Denominator;
			BigInteger bd = b.Denominator;
			return Canonical((a.numerator * bd) + (b.numerator * ad), ad * bd);
		}

		/// <summary>Exact difference, canonicalized.</summary>
		public static BigRational operator -(BigRational a, BigRational b)
		{
			BigInteger ad = a.Denominator;
			BigInteger bd = b.Denominator;
			return Canonical((a.numerator * bd) - (b.numerator * ad), ad * bd);
		}

		/// <summary>Exact product, canonicalized.</summary>
		public static BigRational operator *(BigRational a, BigRational b)
		{
			return Canonical(a.numerator * b.numerator, a.Denominator * b.Denominator);
		}

		/// <summary>Exact quotient, canonicalized.</summary>
		/// <exception cref="DivideByZeroException"><paramref name="b"/> is zero.</exception>
		public static BigRational operator /(BigRational a, BigRational b)
		{
			return Canonical(a.numerator * b.Denominator, a.Denominator * b.numerator);
		}

		/// <summary>
		/// Field-wise equality — see <see cref="Backend.RatEq"/> and item (2) in the
		/// file header. This is value equality because every stored value is canonical.
		/// </summary>
		public bool Equals(BigRational other)
		{
			return this.numerator == other.numerator && this.Denominator == other.Denominator;
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is BigRational other && this.Equals(other);
		}

		/// <summary>Field-wise equality; see <see cref="Equals(BigRational)"/>.</summary>
		public static bool operator ==(BigRational a, BigRational b)
		{
			return a.Equals(b);
		}

		/// <summary>Field-wise inequality; see <see cref="Equals(BigRational)"/>.</summary>
		public static bool operator !=(BigRational a, BigRational b)
		{
			return !a.Equals(b);
		}

		/// <summary>The structural hash from item (1); see <see cref="Backend.HashRational"/>.</summary>
		public override int GetHashCode()
		{
			return Backend.HashRational(this);
		}

		/// <summary>
		/// Value ordering — the Rust's derived <c>Ord</c> on <c>RBig</c>, which R2/R3
		/// inherit for the lexicographic point ordering the arrangement's exact dedup
		/// maps rely on. Denominators are positive, so cross-multiplication cannot
		/// flip the comparison.
		/// </summary>
		public int CompareTo(BigRational other)
		{
			return BigInteger.Compare(this.numerator * other.Denominator, other.numerator * this.Denominator);
		}

		/// <summary>Value ordering; see <see cref="CompareTo"/>.</summary>
		public static bool operator <(BigRational a, BigRational b)
		{
			return a.CompareTo(b) < 0;
		}

		/// <summary>Value ordering; see <see cref="CompareTo"/>.</summary>
		public static bool operator >(BigRational a, BigRational b)
		{
			return a.CompareTo(b) > 0;
		}

		/// <summary>Value ordering; see <see cref="CompareTo"/>.</summary>
		public static bool operator <=(BigRational a, BigRational b)
		{
			return a.CompareTo(b) <= 0;
		}

		/// <summary>Value ordering; see <see cref="CompareTo"/>.</summary>
		public static bool operator >=(BigRational a, BigRational b)
		{
			return a.CompareTo(b) >= 0;
		}

		/// <summary>The canonical <c>numerator/denominator</c> form, for debugging.</summary>
		public override string ToString()
		{
			return this.numerator.ToString() + "/" + this.Denominator.ToString();
		}
	}

	/// <summary>
	/// The helper layer named in the file header: the only code that has to change
	/// when the arbitrary-precision backend does. Call sites use these instead of
	/// naming <see cref="BigInteger"/>'s or <see cref="BigRational"/>'s own
	/// constructors, accessors and trait methods.
	/// </summary>
	public static class Backend
	{
		// ─── Rational construction ───────────────────────────────────────────────

		/// <summary>
		/// <paramref name="numer"/> / <paramref name="denom"/>, fully reduced with the
		/// sign normalized onto the numerator. This is the single gcd whose cost item
		/// (3) in the file header accounts for.
		/// </summary>
		/// <exception cref="DivideByZeroException">The denominator is zero, as with
		/// every backend's equivalent.</exception>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BigRational RatNew(BigInteger numer, BigInteger denom)
		{
			return BigRational.Canonical(numer, denom);
		}

		/// <summary>The integer <paramref name="v"/> as a rational (denominator 1).</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BigRational RatFromInt(BigInteger v)
		{
			return BigRational.FromInteger(v);
		}

		/// <summary>
		/// Exact conversion of an f64; null for NaN and infinity. Every finite f64 is
		/// a dyadic rational, so no information is lost — see item (7).
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BigRational? RatFromF64(double v)
		{
			if (!double.IsFinite(v))
			{
				return null;
			}

			return BigRational.FromDouble(v);
		}

		/// <summary>The rational zero.</summary>
		public static BigRational RatZero()
		{
			return BigRational.Zero;
		}

		/// <summary>The rational one.</summary>
		public static BigRational RatOne()
		{
			return BigRational.One;
		}

		// ─── Rational inspection ─────────────────────────────────────────────────

		/// <summary>True when the value is exactly zero.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool RatIsZero(in BigRational r)
		{
			return r.IsZero;
		}

		/// <summary>True when the value is strictly negative.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool RatIsNegative(in BigRational r)
		{
			return r.IsNegative;
		}

		/// <summary>True when the value is strictly positive.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool RatIsPositive(in BigRational r)
		{
			return r.IsPositive;
		}

		/// <summary>Absolute value.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BigRational RatAbs(in BigRational r)
		{
			return r.Abs();
		}

		/// <summary>Signed numerator of a canonical rational.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BigInteger Numer(in BigRational r)
		{
			return r.Numerator;
		}

		/// <summary>
		/// Denominator of a canonical rational, always strictly positive — which is
		/// what the homogenizing predicates and RatToF64 both want. (In dashu this is
		/// carried by the unsigned type; here it is an invariant, see the file header.)
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BigInteger Denom(in BigRational r)
		{
			return r.Denominator;
		}

		/// <summary>Magnitude of the numerator.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BigInteger NumerMag(in BigRational r)
		{
			return BigInteger.Abs(r.Numerator);
		}

		/// <summary>
		/// Field-wise equality of two *canonical* rationals — value equality without a
		/// general (unreduced-tolerant) comparison, which would be most expensive
		/// exactly when the values ARE equal. See item (2).
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool RatEq(in BigRational a, in BigRational b)
		{
			return a.Equals(b);
		}

		/// <summary>
		/// Structural hash of a canonical rational — see item (1): the sign, then the
		/// little-endian bytes of the numerator magnitude, the 0xfeed separator, then
		/// the bytes of the denominator.
		/// </summary>
		/// <remarks>
		/// The Rust hashes target-width limbs (u64, or u32 on wasm32) where this hashes
		/// bytes, so the two hash streams differ — and the Rust's already differs
		/// between its own targets. That is harmless and sanctioned: these hashes are
		/// process-local (map probing only), never reach the output, and the maps that
		/// use them document their order-independence. Equality is what must be exact.
		/// </remarks>
		public static int HashRational(in BigRational r)
		{
			HashCode state = default(HashCode);
			state.Add(r.IsNegative);
			AddMagnitudeBytes(ref state, BigInteger.Abs(r.Numerator));
			state.Add(0xfeedUL); // separator between numerator and denominator
			AddMagnitudeBytes(ref state, r.Denominator);
			return state.ToHashCode();
		}

		/// <summary>
		/// Feeds the minimal little-endian byte view of a non-negative magnitude into
		/// the hash. Minimal means no leading zero byte, so equal magnitudes always
		/// produce identical streams (the byte count itself is deliberately not
		/// hashed — the 0xfeed separator is what keeps the stream unambiguous).
		/// </summary>
		private static void AddMagnitudeBytes(ref HashCode state, BigInteger magnitude)
		{
			int count = magnitude.GetByteCount(isUnsigned: true);

			// `scoped` is what lets the small case be a stackalloc: without it the
			// compiler must assume a Span local declared out here could escape.
			scoped Span<byte> buffer;
			if (count <= 64)
			{
				buffer = stackalloc byte[64];
			}
			else
			{
				buffer = new byte[count];
			}

			bool ok = magnitude.TryWriteBytes(buffer, out int written, isUnsigned: true, isBigEndian: false);
			System.Diagnostics.Debug.Assert(ok, "magnitude byte buffer was too small");
			state.AddBytes(buffer.Slice(0, written));
		}

		// ─── Int / Uint bridging ─────────────────────────────────────────────────
		//
		// Denominators are unsigned but the homogenized predicate coordinates are
		// signed, so the two roles meet in the homogenization helpers. dashu needed
		// these because its mixed-type operator impls are selective; BigInteger plays
		// both roles, so here they are documentation of which operand is which — and
		// the place a future backend swap re-introduces the distinction.

		/// <summary>Signed times unsigned magnitude.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BigInteger MulIntUint(in BigInteger a, in BigInteger b)
		{
			return a * b;
		}

		/// <summary>Unsigned magnitude times unsigned magnitude.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BigInteger MulUint(in BigInteger a, in BigInteger b)
		{
			return a * b;
		}

		/// <summary>Widens an unsigned magnitude into the signed integer type.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BigInteger IntFromUint(BigInteger v)
		{
			return v;
		}

		/// <summary>
		/// Magnitude of a signed integer, for the unsigned core of RatToF64 and its
		/// IntRatioToF64 sibling.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BigInteger IntMag(in BigInteger v)
		{
			return BigInteger.Abs(v);
		}

		/// <summary>Number of bits in the magnitude (0 for zero).</summary>
		/// <exception cref="ArgumentOutOfRangeException">
		/// The value is negative. dashu's <c>UBig</c> made that unrepresentable; here
		/// the check stands in for the type, and it throws rather than asserting
		/// because both wrong answers are silent. <c>GetBitLength</c> on a negative
		/// returns the two's-complement length, which is one short of the magnitude's
		/// at every power of two (-8 reads 3, not 4) — inside RatToF64 that is an
		/// off-by-one binade, i.e. a moved output vertex, not a crash.
		/// </exception>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static long UintBits(in BigInteger v)
		{
			if (v.Sign < 0)
			{
				throw new ArgumentOutOfRangeException(nameof(v), "UintBits expects a magnitude, not a signed value");
			}

			// GetBitLength is the shortest two's-complement length without the sign
			// bit, which for a non-negative value is exactly dashu's bit_len.
			return (long)v.GetBitLength();
		}

		/// <summary>Value of bit <paramref name="n"/> (little-endian) of the magnitude.</summary>
		/// <exception cref="ArgumentOutOfRangeException">
		/// The value is negative — see <see cref="UintBits"/>. A negative shifts as an
		/// infinite two's-complement sign extension, so every bit reads back the
		/// complement's, not the magnitude's.
		/// </exception>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool UintBit(in BigInteger v, int n)
		{
			if (v.Sign < 0)
			{
				throw new ArgumentOutOfRangeException(nameof(v), "UintBit expects a magnitude, not a signed value");
			}

			return !((v >> n) & BigInteger.One).IsZero;
		}
	}

	/// <summary>
	/// The <see cref="IEqualityComparer{T}"/> face of item (1): the structural hash
	/// paired with the field-wise equality, for dictionaries keyed on canonical
	/// rationals. (The Rust reaches these through <c>Hash</c>/<c>PartialEq</c> impls;
	/// C# hash maps take a comparer, and <see cref="BigRational"/>'s own
	/// Equals/GetHashCode delegate to the same two functions.)
	/// </summary>
	public sealed class BigRationalComparer : IEqualityComparer<BigRational>
	{
		/// <summary>The stateless shared instance.</summary>
		public static readonly BigRationalComparer Instance = new BigRationalComparer();

		private BigRationalComparer()
		{
		}

		/// <summary>Field-wise equality of canonical rationals.</summary>
		public bool Equals(BigRational x, BigRational y)
		{
			return Backend.RatEq(x, y);
		}

		/// <summary>The structural hash from item (1).</summary>
		public int GetHashCode(BigRational obj)
		{
			return Backend.HashRational(obj);
		}
	}
}
