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

// Linear algebra types for ManifoldSharp — port of linalg.rs.
// Ports: include/manifold/linalg.h and the type aliases in include/manifold/common.h
//
// C++ naming:  mat<T, M, N> = M rows, N cols, column-major (N columns of vec<T,M>)
// The C# types use the same naming as the C++ `using` aliases in common.h.
//
// All double types match C++ `double`; float appears only at the MeshGL boundary.
//
// ── Folder layout ────────────────────────────────────────────────────────────
// linalg.rs is one 2,230-line file; the port splits it by type family to keep
// the 800-line cap. This file holds Vec3 and carries the module header.
// Siblings: Vec2.cs, Vec4.cs, IVec.cs (IVec2/IVec3/IVec4/BVec4/UVec3), Mat2.cs,
// Mat3.cs, Mat4.cs, Mat3x4.cs, Mat2x3.cs (Mat4x3/Mat3x2/Mat2x3),
// LinalgFunctions.cs (free functions) and Quat.cs (quaternion functions, the
// Mat3x4/Mat4 conversions and the matrix factories).
//
// ── Struct-shape rule (folder-wide) ──────────────────────────────────────────
// Every type here is a value type, because every Rust type here is Copy.
//   * Types the Rust gives `IndexMut` — Vec2/3/4, IVec2/3/4, BVec4, Mat2/3/4,
//     Mat3x4 — are plain structs with mutable public fields and a
//     ref-returning indexer, so both `v[i]` and `v[i] = x` port literally.
//     `return ref this.X` from a struct member is only legal because of
//     [UnscopedRef] (C# 11); without it the indexer would have to degrade to a
//     switch-based get/set pair.
//   * UVec3 has `Index` but no `IndexMut` in the Rust, so it is a readonly
//     struct with a by-value indexer.
//   * Mat4x3, Mat3x2 and Mat2x3 have no indexer in the Rust and get none here.
//
// NEVER declare a `readonly` field — nor a field of a `readonly struct` — whose
// type is one of the ref-indexer types above. `readonlyField[i] = x` compiles
// clean, with no warning, so warnings-as-errors does not catch it; the write
// then lands in the defensive copy C# makes at the access and the real field is
// never touched. Rust's `&mut` made this a compile error. Here it is a silent
// no-op that will present as "the arena did not update", which is a miserable
// bug to find. The arena structs of Phases 3-10 are exactly where this will be
// tempting: hold these types in plain mutable fields or in arrays (array
// elements are real storage, so `arr[k][i] = x` does mutate), never behind
// `readonly`.
//
// ── Equality rule (folder-wide) ──────────────────────────────────────────────
// The Rust derives `PartialEq` (IEEE component-wise ==) and separately
// hand-writes `Hash` over `to_bits`; it never implements `Eq`, so those two
// never have to agree there. In C# they must, because Equals and GetHashCode
// are a contract. So: operator== / operator!= are the IEEE comparison, and
// Equals/GetHashCode are bit-based over BitConverter.DoubleToUInt64Bits. The
// two disagree exactly where IEEE and bits disagree: identically-encoded NaNs
// (bit-equal, IEEE-unequal) and +0.0 vs -0.0 (IEEE-equal, bit-unequal). Note
// "identically-encoded" — two NaNs with different payloads or different sign
// bits are bit-unequal here, and that distinction is live in this port, since
// Rust's f64::NAN is 0x7ff8000000000000 while C#'s double.NaN is 0xfff8...
//
// ── Numerics rule (folder-wide) ──────────────────────────────────────────────
// No FMA, no reassociation, no SIMD: every expression is transcribed in the
// Rust's operation order, because the bits of the result depend on it. Trig
// goes through DeterministicMath, never System.Math; System.Math is used only
// for the IEEE-exact primitives Abs/Sqrt/Floor/Ceiling/Round/CopySign.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace ManifoldSharp.Linalg
{
	/// <summary>
	/// <c>la::vec&lt;double, 3&gt;</c> — 3-component double column vector.
	/// </summary>
	/// <remarks>
	/// <c>operator==</c> is the IEEE component-wise comparison; <see cref="Equals(Vec3)"/>
	/// and <see cref="GetHashCode"/> are bit-based. See the folder header above.
	/// </remarks>
	public struct Vec3 : IEquatable<Vec3>
	{
		/// <summary>The x component.</summary>
		public double X;

		/// <summary>The y component.</summary>
		public double Y;

		/// <summary>The z component.</summary>
		public double Z;

		/// <summary>Creates a vector from its components.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Vec3(double x, double y, double z)
		{
			this.X = x;
			this.Y = y;
			this.Z = z;
		}

		/// <summary>Creates a vector with all three components set to <paramref name="v"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 Splat(double v)
		{
			return new Vec3(v, v, v);
		}

		/// <summary>The <c>xy</c> swizzle.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Vec2 Xy()
		{
			return new Vec2(this.X, this.Y);
		}

		/// <summary>The <c>yz</c> swizzle.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Vec2 Yz()
		{
			return new Vec2(this.Y, this.Z);
		}

		/// <summary>
		/// Component by index: 0 is <see cref="X"/>, 1 is <see cref="Y"/>, 2 is
		/// <see cref="Z"/>. Returns a <c>ref</c> so both the Rust <c>Index</c> and
		/// <c>IndexMut</c> uses port literally.
		/// </summary>
		/// <exception cref="ArgumentOutOfRangeException">The index is not 0, 1 or 2.</exception>
		[UnscopedRef]
		public ref double this[int i]
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			get
			{
				switch (i)
				{
					case 0:
						return ref this.X;
					case 1:
						return ref this.Y;
					case 2:
						return ref this.Z;
					default:
						throw new ArgumentOutOfRangeException(nameof(i), $"Vec3 index out of range: {i}");
				}
			}
		}

		/// <summary>Ports the Rust <c>From&lt;[f64; 3]&gt;</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 FromArray(double[] a)
		{
			ArgumentNullException.ThrowIfNull(a);
			return new Vec3(a[0], a[1], a[2]);
		}

		/// <summary>Ports the Rust <c>From&lt;Vec3&gt; for [f64; 3]</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public double[] ToArray()
		{
			return new double[] { this.X, this.Y, this.Z };
		}

		/// <summary>Ports the Rust <c>From&lt;(Vec2, f64)&gt;</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 From(Vec2 xy, double z)
		{
			return new Vec3(xy.X, xy.Y, z);
		}

		// The Rust also has From<(f64, f64, f64)>; in C# that is the constructor.

		/// <summary>Component-wise negation.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 operator -(Vec3 a)
		{
			return new Vec3(-a.X, -a.Y, -a.Z);
		}

		/// <summary>Component-wise addition.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 operator +(Vec3 a, Vec3 b)
		{
			return new Vec3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
		}

		/// <summary>Component-wise subtraction.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 operator -(Vec3 a, Vec3 b)
		{
			return new Vec3(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
		}

		/// <summary>Component-wise multiplication (C++ <c>cmul</c>), not a dot product.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 operator *(Vec3 a, Vec3 b)
		{
			return new Vec3(a.X * b.X, a.Y * b.Y, a.Z * b.Z);
		}

		/// <summary>Component-wise division.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 operator /(Vec3 a, Vec3 b)
		{
			return new Vec3(a.X / b.X, a.Y / b.Y, a.Z / b.Z);
		}

		/// <summary>Scales by a scalar on the right.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 operator *(Vec3 a, double s)
		{
			return new Vec3(a.X * s, a.Y * s, a.Z * s);
		}

		/// <summary>Scales by a scalar on the left; the Rust writes <c>s * v.x</c>, kept in that order.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 operator *(double s, Vec3 v)
		{
			return new Vec3(s * v.X, s * v.Y, s * v.Z);
		}

		/// <summary>Divides every component by a scalar.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 operator /(Vec3 a, double s)
		{
			return new Vec3(a.X / s, a.Y / s, a.Z / s);
		}

		/// <summary>Adds a scalar to every component.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 operator +(Vec3 a, double s)
		{
			return new Vec3(a.X + s, a.Y + s, a.Z + s);
		}

		/// <summary>Subtracts a scalar from every component.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 operator -(Vec3 a, double s)
		{
			return new Vec3(a.X - s, a.Y - s, a.Z - s);
		}

		// AddAssign/SubAssign/MulAssign/DivAssign need no port: C# synthesizes +=, -=,
		// *= and /= from the binary operators above.

		// Lexicographic ordering. A NaN component yields false in every direction,
		// which is what Rust's partial_cmp returning None does.

		/// <summary>Lexicographic less-than.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator <(Vec3 a, Vec3 b)
		{
			return a.X != b.X ? a.X < b.X : (a.Y != b.Y ? a.Y < b.Y : a.Z < b.Z);
		}

		/// <summary>Lexicographic greater-than.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator >(Vec3 a, Vec3 b)
		{
			return a.X != b.X ? a.X > b.X : (a.Y != b.Y ? a.Y > b.Y : a.Z > b.Z);
		}

		/// <summary>Lexicographic less-than-or-equal.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator <=(Vec3 a, Vec3 b)
		{
			return a.X != b.X ? a.X < b.X : (a.Y != b.Y ? a.Y < b.Y : a.Z <= b.Z);
		}

		/// <summary>Lexicographic greater-than-or-equal.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator >=(Vec3 a, Vec3 b)
		{
			return a.X != b.X ? a.X > b.X : (a.Y != b.Y ? a.Y > b.Y : a.Z >= b.Z);
		}

		/// <summary>IEEE component-wise equality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(Vec3 a, Vec3 b)
		{
			return a.X == b.X && a.Y == b.Y && a.Z == b.Z;
		}

		/// <summary>IEEE component-wise inequality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(Vec3 a, Vec3 b)
		{
			return !(a == b);
		}

		/// <summary>
		/// Bit-exact equality: identically-encoded NaNs are equal, and +0.0 does not
		/// equal -0.0. "Identically-encoded" is load-bearing — NaNs differing in sign
		/// bit or payload are unequal here, and Rust's <c>f64::NAN</c>
		/// (0x7ff8000000000000) and C#'s <c>double.NaN</c> (0xfff8000000000000) differ
		/// in exactly that way. This is the partner of <see cref="GetHashCode"/>, not of
		/// <c>operator==</c>.
		/// </summary>
		public bool Equals(Vec3 other)
		{
			return BitConverter.DoubleToUInt64Bits(this.X) == BitConverter.DoubleToUInt64Bits(other.X)
				&& BitConverter.DoubleToUInt64Bits(this.Y) == BitConverter.DoubleToUInt64Bits(other.Y)
				&& BitConverter.DoubleToUInt64Bits(this.Z) == BitConverter.DoubleToUInt64Bits(other.Z);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Vec3 other && this.Equals(other);
		}

		/// <summary>
		/// Hash over the bit representation of each component, in field order — the
		/// Rust's "simpler hash that is consistent: just hash all fields sequentially".
		/// The mixing function differs from Rust's SipHash; only the equality it pairs
		/// with has to match, since no hash value reaches an output.
		/// </summary>
		public override int GetHashCode()
		{
			return HashCode.Combine(
				BitConverter.DoubleToUInt64Bits(this.X),
				BitConverter.DoubleToUInt64Bits(this.Y),
				BitConverter.DoubleToUInt64Bits(this.Z));
		}
	}
}
