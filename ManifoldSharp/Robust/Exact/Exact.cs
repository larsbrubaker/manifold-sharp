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

// Exact.cs — port of robust/exact/mod.rs. Exact geometric arithmetic for the
// robust boolean engine (src/robust).
//
// Layered design:
//   Backend.cs    — the single seam to the bignum library (System.Numerics):
//                   the BigInteger/BigRational types and the small helper
//                   layer every other module uses instead of naming the
//                   backend types' own constructors and accessors.
//   Rational.cs   — Rational point types (R2/R3) and correctly rounded
//                   rational→f64 conversion.
//   Predicates.cs — fully exact predicates and geometric constructions on
//                   rational points; ground truth for everything.
//   Filtered.cs   — f64 entry points with Shewchuk-style static error-bound
//                   filters that escalate to Predicates.cs only when the
//                   float computation cannot certify a sign.
//   IntPred.cs    — the integer escalation tier between the float filters and
//                   the rational ground truth.
//   Approx.cs     — Attene-style semi-static filters over approximate
//                   coordinates of exact points.
//
// The exact boolean pipeline (Boolean3.cs) never calls into this namespace.
//
// ── Rust's [f64; 2] / [f64; 3] parameters ────────────────────────────────────
// The Rust predicate entry points in intpred.rs and approx.rs take raw
// `[f64; 2]` / `[f64; 3]` arrays where filtered.rs takes `Vec2`/`Vec3`. A C#
// array parameter would allocate on every call, so those land as
// Linalg.Vec2/Vec3 — the same three doubles in the same order, so `a[0]`
// becomes `a.X` with no change of meaning, and the filtered.rs call sites that
// re-wrapped their Vec2/Vec3 into arrays simply pass them through. Only the
// genuinely variable-length arrays (the per-axis scaling helpers' `[f64; N]`)
// stay array-shaped, as ReadOnlySpan<double>.

namespace ManifoldSharp.Robust.Exact
{
	/// <summary>
	/// Sign of an exactly evaluated quantity. The whole robust pipeline reasons
	/// in terms of signs; magnitudes only matter inside constructions.
	/// </summary>
	public enum Sign
	{
		/// <summary>The quantity is strictly negative.</summary>
		Neg,

		/// <summary>The quantity is exactly zero.</summary>
		Zero,

		/// <summary>The quantity is strictly positive.</summary>
		Pos,
	}

	/// <summary>
	/// The free functions and inherent methods of <see cref="Sign"/> (Rust's
	/// <c>impl Sign</c>). They live in a companion static class because a C# enum
	/// cannot carry members.
	/// </summary>
	public static class SignFunctions
	{
		/// <summary>
		/// Sign of a finite f64 (<c>-0.0</c> is <see cref="Sign.Zero"/>). Only
		/// meaningful when the value is known to carry the correct sign of the exact
		/// quantity.
		/// </summary>
		public static Sign OfF64(double v)
		{
			if (v > 0.0)
			{
				return Sign.Pos;
			}
			else if (v < 0.0)
			{
				return Sign.Neg;
			}
			else
			{
				return Sign.Zero;
			}
		}

		/// <summary>Sign of an exact rational.</summary>
		public static Sign OfRat(in BigRational r)
		{
			if (Backend.RatIsPositive(r))
			{
				return Sign.Pos;
			}
			else if (Backend.RatIsNegative(r))
			{
				return Sign.Neg;
			}
			else
			{
				return Sign.Zero;
			}
		}

		/// <summary>The opposite sign (<see cref="Sign.Zero"/> stays Zero).</summary>
		public static Sign Flip(this Sign self)
		{
			switch (self)
			{
				case Sign.Neg:
					return Sign.Pos;
				case Sign.Pos:
					return Sign.Neg;
				default:
					return Sign.Zero;
			}
		}

		/// <summary>-1, 0 or +1.</summary>
		public static int AsI32(this Sign self)
		{
			switch (self)
			{
				case Sign.Neg:
					return -1;
				case Sign.Pos:
					return 1;
				default:
					return 0;
			}
		}

		/// <summary>True for <see cref="Sign.Zero"/>.</summary>
		public static bool IsZero(this Sign self)
		{
			return self == Sign.Zero;
		}
	}
}
