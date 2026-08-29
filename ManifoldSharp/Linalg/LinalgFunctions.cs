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

// Ports the free functions of linalg.rs: vector algebra, the component-wise
// math wrappers, min/max/clamp/lerp and the reductions. The quaternion
// functions, the Mat3x4/Mat4 conversions and the matrix factories are in
// Quat.cs, which continues this same partial class. The module header for the
// whole Linalg folder lives in Vec3.cs.
//
// ── Naming ───────────────────────────────────────────────────────────────────
// Rust cannot overload, so linalg.rs disambiguates by suffix; C# can, so the
// suffixes collapse into overloads. Two of those suffixes mean different things
// (`dot2` is 2-D, `length2` is squared) and would have become the confusable
// pair Length2/Length_2, so the squared forms are spelled out instead:
//
//   dot2/dot/dot4                     -> Dot
//   length2_2/length2/length2_4       -> LengthSquared
//   length_2/length/length_4          -> Length
//   normalize2/normalize/normalize4   -> Normalize
//   distance2_2/distance2             -> DistanceSquared
//   distance_2/distance               -> Distance
//   abs2/3/4, floor2/3/4, ceil2/3,
//   round3/4, sqrt3/4, isfinite3/4    -> Abs, Floor, Ceil, Round, Sqrt, IsFinite
//   min2/3/4, max2/3/4, clamp3/4      -> Min, Max, Clamp
//   clamp_s                           -> Clamp(double, double, double)
//   lerp2/3/4                         -> Lerp
//   minelem2/3/4, maxelem2/3/4        -> MinElem, MaxElem
//   sum3/4, argmin3, argmax3/4        -> Sum, ArgMin, ArgMax
//
// cross2, cross_sv and cross_vs keep their Rust names, because there the suffix
// is naming the operation and not just the arity.
//
// ── Which math ───────────────────────────────────────────────────────────────
// Every transcendental the Rust routes through `crate::math` routes through
// DeterministicMath here — acos in UAngle, sin/cos in Rot2/RotX/RotY/RotZ and
// Slerp. System.Math appears only for Abs/Sqrt/Floor/Ceiling/Round, which are
// IEEE-exact and identical to the Rust primitives they stand in for. Note that
// Round must ask for MidpointRounding.AwayFromZero: Rust's f64::round rounds
// halves away from zero, while C#'s default is banker's rounding.

using System.Runtime.CompilerServices;

namespace ManifoldSharp.Linalg
{
	/// <summary>
	/// The free functions of <c>linalg.rs</c> — vector algebra, component-wise math and
	/// reductions over the vector types.
	/// </summary>
	public static partial class LinalgFunctions
	{
		// ─── Scalar helpers ──────────────────────────────────────────────────────

		/// <summary>
		/// Rust's <c>f64::min</c> (libm <c>fmin</c>): a NaN operand loses, so the other
		/// argument is returned. Every Rust <c>.min()</c> in this port comes here rather
		/// than calling <see cref="Math.Min(double, double)"/> directly, because
		/// <see cref="Math.Min(double, double)"/> propagates NaN and Rust does not.
		/// </summary>
		/// <remarks>
		/// The two NaN guards are the whole reason this wrapper exists; once they have
		/// run, <see cref="Math.Min(double, double)"/> is exactly the semantics wanted,
		/// including the signed-zero tie. That tie is worth spelling out: Rust documents
		/// <c>f64::min</c> as returning "either input" when the arguments compare equal,
		/// which makes +0.0 against -0.0 genuinely target-dependent — x86's <c>minsd</c>
		/// returns the second operand regardless of sign, while arm64's <c>FMIN</c> is
		/// sign-of-zero aware and returns -0.0 either way. This port pins the tie to
		/// .NET's specified behaviour, which is sign-of-zero aware in both operand orders
		/// and therefore matches the arm64 Rust build the oracle lane compares against.
		/// Recorded in docs/RUST_DIVERGENCES.md.
		/// </remarks>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double MinF64(double a, double b)
		{
			if (double.IsNaN(a))
			{
				return b;
			}

			if (double.IsNaN(b))
			{
				return a;
			}

			return Math.Min(a, b);
		}

		/// <summary>
		/// Rust's <c>f64::max</c> (libm <c>fmax</c>): a NaN operand loses, so the other
		/// argument is returned. See <see cref="MinF64"/> for why the NaN guards are
		/// needed and how the signed-zero tie is pinned.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double MaxF64(double a, double b)
		{
			if (double.IsNaN(a))
			{
				return b;
			}

			if (double.IsNaN(b))
			{
				return a;
			}

			return Math.Max(a, b);
		}

		// ─── Vector algebra functions ────────────────────────────────────────────

		/// <summary>2D cross product: <c>a.x*b.y - a.y*b.x</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Cross2(Vec2 a, Vec2 b)
		{
			return a.X * b.Y - a.Y * b.X;
		}

		/// <summary>2D: rotate vector 90 degrees CCW by scalar <paramref name="a"/> (scalar x vec2 cross).</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 CrossSv(double a, Vec2 b)
		{
			return new Vec2(-a * b.Y, a * b.X);
		}

		/// <summary>2D: vec2 x scalar.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 CrossVs(Vec2 a, double b)
		{
			return new Vec2(a.Y * b, -a.X * b);
		}

		/// <summary>3D cross product.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 Cross(Vec3 a, Vec3 b)
		{
			return new Vec3(
				a.Y * b.Z - a.Z * b.Y,
				a.Z * b.X - a.X * b.Z,
				a.X * b.Y - a.Y * b.X);
		}

		/// <summary>Dot product of two <see cref="Vec2"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Dot(Vec2 a, Vec2 b)
		{
			return a.X * b.X + a.Y * b.Y;
		}

		/// <summary>Dot product of two <see cref="Vec3"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Dot(Vec3 a, Vec3 b)
		{
			return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
		}

		/// <summary>Dot product of two <see cref="Vec4"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Dot(Vec4 a, Vec4 b)
		{
			return a.X * b.X + a.Y * b.Y + a.Z * b.Z + a.W * b.W;
		}

		/// <summary>Squared length of a <see cref="Vec2"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double LengthSquared(Vec2 a)
		{
			return Dot(a, a);
		}

		/// <summary>Squared length of a <see cref="Vec3"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double LengthSquared(Vec3 a)
		{
			return Dot(a, a);
		}

		/// <summary>Squared length of a <see cref="Vec4"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double LengthSquared(Vec4 a)
		{
			return Dot(a, a);
		}

		/// <summary>Length of a <see cref="Vec2"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Length(Vec2 a)
		{
			return Math.Sqrt(LengthSquared(a));
		}

		/// <summary>Length of a <see cref="Vec3"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Length(Vec3 a)
		{
			return Math.Sqrt(LengthSquared(a));
		}

		/// <summary>Length of a <see cref="Vec4"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Length(Vec4 a)
		{
			return Math.Sqrt(LengthSquared(a));
		}

		/// <summary>Unit vector in the direction of a <see cref="Vec2"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 Normalize(Vec2 a)
		{
			return a / Length(a);
		}

		/// <summary>Unit vector in the direction of a <see cref="Vec3"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 Normalize(Vec3 a)
		{
			return a / Length(a);
		}

		/// <summary>Unit vector in the direction of a <see cref="Vec4"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 Normalize(Vec4 a)
		{
			return a / Length(a);
		}

		/// <summary>Squared distance between two <see cref="Vec2"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double DistanceSquared(Vec2 a, Vec2 b)
		{
			return LengthSquared(b - a);
		}

		/// <summary>Squared distance between two <see cref="Vec3"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double DistanceSquared(Vec3 a, Vec3 b)
		{
			return LengthSquared(b - a);
		}

		/// <summary>Distance between two <see cref="Vec2"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Distance(Vec2 a, Vec2 b)
		{
			return Length(b - a);
		}

		/// <summary>Distance between two <see cref="Vec3"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Distance(Vec3 a, Vec3 b)
		{
			return Length(b - a);
		}

		/// <summary>Angle between two unit vectors (clamped to [0, pi]).</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double UAngle(Vec3 a, Vec3 b)
		{
			double d = Dot(a, b);
			if (d > 1.0)
			{
				return 0.0;
			}

			return DeterministicMath.Acos(d < -1.0 ? -1.0 : d);
		}

		/// <summary>Angle between two non-unit vectors.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Angle(Vec3 a, Vec3 b)
		{
			return UAngle(Normalize(a), Normalize(b));
		}

		/// <summary>2D rotation: rotate <paramref name="v"/> CCW by angle <paramref name="a"/> (radians).</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 Rot2(double a, Vec2 v)
		{
			double s = DeterministicMath.Sin(a);
			double c = DeterministicMath.Cos(a);
			return new Vec2(v.X * c - v.Y * s, v.X * s + v.Y * c);
		}

		/// <summary>Rotate <paramref name="v"/> CCW around the X axis by <paramref name="a"/> radians.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 RotX(double a, Vec3 v)
		{
			double s = DeterministicMath.Sin(a);
			double c = DeterministicMath.Cos(a);
			return new Vec3(v.X, v.Y * c - v.Z * s, v.Y * s + v.Z * c);
		}

		/// <summary>Rotate <paramref name="v"/> CCW around the Y axis by <paramref name="a"/> radians.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 RotY(double a, Vec3 v)
		{
			double s = DeterministicMath.Sin(a);
			double c = DeterministicMath.Cos(a);
			return new Vec3(v.X * c + v.Z * s, v.Y, -v.X * s + v.Z * c);
		}

		/// <summary>Rotate <paramref name="v"/> CCW around the Z axis by <paramref name="a"/> radians.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 RotZ(double a, Vec3 v)
		{
			double s = DeterministicMath.Sin(a);
			double c = DeterministicMath.Cos(a);
			return new Vec3(v.X * c - v.Y * s, v.X * s + v.Y * c, v.Z);
		}

		/// <summary>Normalized linear interpolation.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 NLerp(Vec3 a, Vec3 b, double t)
		{
			return Normalize(Lerp(a, b, t));
		}

		/// <summary>Spherical linear interpolation between unit vectors.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 Slerp(Vec3 a, Vec3 b, double t)
		{
			double th = UAngle(a, b);
			if (th == 0.0)
			{
				return a;
			}

			return a * DeterministicMath.Sin(th * (1.0 - t)) / DeterministicMath.Sin(th)
				+ b * DeterministicMath.Sin(th * t) / DeterministicMath.Sin(th);
		}

		// ─── Component-wise math functions ───────────────────────────────────────

		/// <summary>Component-wise absolute value.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 Abs(Vec2 a)
		{
			return new Vec2(Math.Abs(a.X), Math.Abs(a.Y));
		}

		/// <summary>Component-wise absolute value.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 Abs(Vec3 a)
		{
			return new Vec3(Math.Abs(a.X), Math.Abs(a.Y), Math.Abs(a.Z));
		}

		/// <summary>Component-wise absolute value.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 Abs(Vec4 a)
		{
			return new Vec4(Math.Abs(a.X), Math.Abs(a.Y), Math.Abs(a.Z), Math.Abs(a.W));
		}

		/// <summary>Component-wise floor.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 Floor(Vec2 a)
		{
			return new Vec2(Math.Floor(a.X), Math.Floor(a.Y));
		}

		/// <summary>Component-wise floor.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 Floor(Vec3 a)
		{
			return new Vec3(Math.Floor(a.X), Math.Floor(a.Y), Math.Floor(a.Z));
		}

		/// <summary>Component-wise floor.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 Floor(Vec4 a)
		{
			return new Vec4(Math.Floor(a.X), Math.Floor(a.Y), Math.Floor(a.Z), Math.Floor(a.W));
		}

		/// <summary>Component-wise ceiling.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 Ceil(Vec2 a)
		{
			return new Vec2(Math.Ceiling(a.X), Math.Ceiling(a.Y));
		}

		/// <summary>Component-wise ceiling.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 Ceil(Vec3 a)
		{
			return new Vec3(Math.Ceiling(a.X), Math.Ceiling(a.Y), Math.Ceiling(a.Z));
		}

		/// <summary>
		/// Component-wise rounding, halves away from zero — Rust's <c>f64::round</c>.
		/// C#'s default <see cref="Math.Round(double)"/> is banker's rounding and would
		/// diverge on every exact .5.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 Round(Vec3 a)
		{
			return new Vec3(
				Math.Round(a.X, MidpointRounding.AwayFromZero),
				Math.Round(a.Y, MidpointRounding.AwayFromZero),
				Math.Round(a.Z, MidpointRounding.AwayFromZero));
		}

		/// <summary>Component-wise rounding, halves away from zero — Rust's <c>f64::round</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 Round(Vec4 a)
		{
			return new Vec4(
				Math.Round(a.X, MidpointRounding.AwayFromZero),
				Math.Round(a.Y, MidpointRounding.AwayFromZero),
				Math.Round(a.Z, MidpointRounding.AwayFromZero),
				Math.Round(a.W, MidpointRounding.AwayFromZero));
		}

		/// <summary>Component-wise square root.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 Sqrt(Vec3 a)
		{
			return new Vec3(Math.Sqrt(a.X), Math.Sqrt(a.Y), Math.Sqrt(a.Z));
		}

		/// <summary>Component-wise square root.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 Sqrt(Vec4 a)
		{
			return new Vec4(Math.Sqrt(a.X), Math.Sqrt(a.Y), Math.Sqrt(a.Z), Math.Sqrt(a.W));
		}

		/// <summary>True when every component is finite (neither infinite nor NaN).</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool IsFinite(Vec3 a)
		{
			return double.IsFinite(a.X) && double.IsFinite(a.Y) && double.IsFinite(a.Z);
		}

		/// <summary>True when every component is finite (neither infinite nor NaN).</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool IsFinite(Vec4 a)
		{
			return double.IsFinite(a.X) && double.IsFinite(a.Y) && double.IsFinite(a.Z) && double.IsFinite(a.W);
		}

		// ─── Component-wise min/max/clamp ────────────────────────────────────────

		/// <summary>Component-wise minimum.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 Min(Vec2 a, Vec2 b)
		{
			return new Vec2(MinF64(a.X, b.X), MinF64(a.Y, b.Y));
		}

		/// <summary>Component-wise minimum.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 Min(Vec3 a, Vec3 b)
		{
			return new Vec3(MinF64(a.X, b.X), MinF64(a.Y, b.Y), MinF64(a.Z, b.Z));
		}

		/// <summary>Component-wise minimum.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 Min(Vec4 a, Vec4 b)
		{
			return new Vec4(MinF64(a.X, b.X), MinF64(a.Y, b.Y), MinF64(a.Z, b.Z), MinF64(a.W, b.W));
		}

		/// <summary>Component-wise maximum.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 Max(Vec2 a, Vec2 b)
		{
			return new Vec2(MaxF64(a.X, b.X), MaxF64(a.Y, b.Y));
		}

		/// <summary>Component-wise maximum.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 Max(Vec3 a, Vec3 b)
		{
			return new Vec3(MaxF64(a.X, b.X), MaxF64(a.Y, b.Y), MaxF64(a.Z, b.Z));
		}

		/// <summary>Component-wise maximum.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 Max(Vec4 a, Vec4 b)
		{
			return new Vec4(MaxF64(a.X, b.X), MaxF64(a.Y, b.Y), MaxF64(a.Z, b.Z), MaxF64(a.W, b.W));
		}

		/// <summary>
		/// Clamp <paramref name="x"/> component-wise between <paramref name="lo"/> and
		/// <paramref name="hi"/>. Max first, then min, exactly as the Rust writes it.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 Clamp(Vec3 x, Vec3 lo, Vec3 hi)
		{
			return new Vec3(
				MinF64(MaxF64(x.X, lo.X), hi.X),
				MinF64(MaxF64(x.Y, lo.Y), hi.Y),
				MinF64(MaxF64(x.Z, lo.Z), hi.Z));
		}

		/// <summary>
		/// Clamp <paramref name="x"/> component-wise between <paramref name="lo"/> and
		/// <paramref name="hi"/>. Max first, then min, exactly as the Rust writes it.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 Clamp(Vec4 x, Vec4 lo, Vec4 hi)
		{
			return new Vec4(
				MinF64(MaxF64(x.X, lo.X), hi.X),
				MinF64(MaxF64(x.Y, lo.Y), hi.Y),
				MinF64(MaxF64(x.Z, lo.Z), hi.Z),
				MinF64(MaxF64(x.W, lo.W), hi.W));
		}

		/// <summary>Scalar clamp — the Rust <c>clamp_s</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Clamp(double x, double lo, double hi)
		{
			return MinF64(MaxF64(x, lo), hi);
		}

		/// <summary>
		/// Linear interpolation: <c>a*(1-t) + b*t</c>. Never <c>a + (b-a)*t</c> — the two
		/// differ in the last bits, and at t=1 only this form is guaranteed to return b
		/// exactly.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 Lerp(Vec2 a, Vec2 b, double t)
		{
			return a * (1.0 - t) + b * t;
		}

		/// <summary>
		/// Linear interpolation: <c>a*(1-t) + b*t</c>. Never <c>a + (b-a)*t</c> — the two
		/// differ in the last bits, and at t=1 only this form is guaranteed to return b
		/// exactly.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 Lerp(Vec3 a, Vec3 b, double t)
		{
			return a * (1.0 - t) + b * t;
		}

		/// <summary>
		/// Linear interpolation: <c>a*(1-t) + b*t</c>. Never <c>a + (b-a)*t</c> — the two
		/// differ in the last bits, and at t=1 only this form is guaranteed to return b
		/// exactly.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 Lerp(Vec4 a, Vec4 b, double t)
		{
			return a * (1.0 - t) + b * t;
		}

		// ─── Reductions ──────────────────────────────────────────────────────────

		/// <summary>Smallest component.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double MinElem(Vec2 a)
		{
			return MinF64(a.X, a.Y);
		}

		/// <summary>Smallest component.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double MinElem(Vec3 a)
		{
			return MinF64(MinF64(a.X, a.Y), a.Z);
		}

		/// <summary>Smallest component.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double MinElem(Vec4 a)
		{
			return MinF64(MinF64(MinF64(a.X, a.Y), a.Z), a.W);
		}

		/// <summary>Largest component.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double MaxElem(Vec2 a)
		{
			return MaxF64(a.X, a.Y);
		}

		/// <summary>Largest component.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double MaxElem(Vec3 a)
		{
			return MaxF64(MaxF64(a.X, a.Y), a.Z);
		}

		/// <summary>Largest component.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double MaxElem(Vec4 a)
		{
			return MaxF64(MaxF64(MaxF64(a.X, a.Y), a.Z), a.W);
		}

		/// <summary>Sum of the components, added left to right.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Sum(Vec3 a)
		{
			return a.X + a.Y + a.Z;
		}

		/// <summary>Sum of the components, added left to right.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Sum(Vec4 a)
		{
			return a.X + a.Y + a.Z + a.W;
		}

		/// <summary>Index of the minimum element (argmin), ties going to the lower index.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static int ArgMin(Vec3 a)
		{
			if (a.X <= a.Y && a.X <= a.Z)
			{
				return 0;
			}

			return a.Y <= a.Z ? 1 : 2;
		}

		/// <summary>Index of the maximum element (argmax), ties going to the lower index.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static int ArgMax(Vec3 a)
		{
			if (a.X >= a.Y && a.X >= a.Z)
			{
				return 0;
			}

			return a.Y >= a.Z ? 1 : 2;
		}

		/// <summary>
		/// Index of the maximum element (argmax). The strict <c>&gt;</c> and the ascending
		/// scan mean ties keep the lowest index, which is what
		/// <c>RotationQuatMat</c> relies on when it picks a sign column.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static int ArgMax(Vec4 a)
		{
			int j = 0;
			for (int i = 1; i < 4; i++)
			{
				if (a[i] > a[j])
				{
					j = i;
				}
			}

			return j;
		}
	}
}
