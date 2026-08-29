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

// Ports the Vec2 section of linalg.rs. The module header for the whole Linalg
// folder — including the struct-shape and equality rules every file here obeys
// — lives in Vec3.cs.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace ManifoldSharp.Linalg
{
	/// <summary>
	/// <c>la::vec&lt;double, 2&gt;</c> — 2-component double column vector.
	/// </summary>
	/// <remarks>
	/// <c>operator==</c> is the IEEE component-wise comparison the Rust derives via
	/// <c>PartialEq</c>; <see cref="Equals(Vec2)"/> and <see cref="GetHashCode"/> are
	/// bit-based, mirroring the Rust's hand-written <c>Hash</c> over <c>to_bits</c>.
	/// See the folder header in Vec3.cs.
	/// </remarks>
	public struct Vec2 : IEquatable<Vec2>
	{
		/// <summary>The x component.</summary>
		public double X;

		/// <summary>The y component.</summary>
		public double Y;

		/// <summary>Creates a vector from its components.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Vec2(double x, double y)
		{
			this.X = x;
			this.Y = y;
		}

		/// <summary>Creates a vector with both components set to <paramref name="v"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 Splat(double v)
		{
			return new Vec2(v, v);
		}

		/// <summary>The <c>xy</c> swizzle, which for a 2-vector is the vector itself.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Vec2 Xy()
		{
			return this;
		}

		/// <summary>
		/// Component by index: 0 is <see cref="X"/>, 1 is <see cref="Y"/>. Returns a
		/// <c>ref</c> so that both the Rust <c>Index</c> and <c>IndexMut</c> uses port
		/// literally.
		/// </summary>
		/// <exception cref="ArgumentOutOfRangeException">The index is not 0 or 1.</exception>
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
					default:
						throw new ArgumentOutOfRangeException(nameof(i), $"Vec2 index out of range: {i}");
				}
			}
		}

		/// <summary>Ports the Rust <c>From&lt;[f64; 2]&gt;</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 FromArray(double[] a)
		{
			ArgumentNullException.ThrowIfNull(a);
			return new Vec2(a[0], a[1]);
		}

		/// <summary>Ports the Rust <c>From&lt;Vec2&gt; for [f64; 2]</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public double[] ToArray()
		{
			return new double[] { this.X, this.Y };
		}

		// The Rust also has From<(f64, f64)>; in C# that is the constructor, so it
		// does not get a second spelling here.

		/// <summary>Component-wise negation.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 operator -(Vec2 a)
		{
			return new Vec2(-a.X, -a.Y);
		}

		/// <summary>Component-wise addition.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 operator +(Vec2 a, Vec2 b)
		{
			return new Vec2(a.X + b.X, a.Y + b.Y);
		}

		/// <summary>Component-wise subtraction.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 operator -(Vec2 a, Vec2 b)
		{
			return new Vec2(a.X - b.X, a.Y - b.Y);
		}

		/// <summary>Component-wise multiplication (C++ <c>cmul</c>), not a dot product.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 operator *(Vec2 a, Vec2 b)
		{
			return new Vec2(a.X * b.X, a.Y * b.Y);
		}

		/// <summary>Component-wise division.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 operator /(Vec2 a, Vec2 b)
		{
			return new Vec2(a.X / b.X, a.Y / b.Y);
		}

		/// <summary>Scales by a scalar on the right.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 operator *(Vec2 a, double s)
		{
			return new Vec2(a.X * s, a.Y * s);
		}

		/// <summary>Scales by a scalar on the left; the Rust writes <c>s * v.x</c>, kept in that order.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 operator *(double s, Vec2 v)
		{
			return new Vec2(s * v.X, s * v.Y);
		}

		/// <summary>Divides every component by a scalar.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 operator /(Vec2 a, double s)
		{
			return new Vec2(a.X / s, a.Y / s);
		}

		/// <summary>Adds a scalar to every component.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 operator +(Vec2 a, double s)
		{
			return new Vec2(a.X + s, a.Y + s);
		}

		/// <summary>Subtracts a scalar from every component.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 operator -(Vec2 a, double s)
		{
			return new Vec2(a.X - s, a.Y - s);
		}

		// The Rust's AddAssign/SubAssign/MulAssign/DivAssign impls need no port: C#
		// synthesizes +=, -=, *= and /= from the binary operators above.

		// Lexicographic ordering — matches C++ compare(vec<T,2>, vec<T,2>). Written so
		// that a NaN component yields false in every direction, which is what Rust's
		// partial_cmp returning None does.

		/// <summary>Lexicographic less-than.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator <(Vec2 a, Vec2 b)
		{
			return a.X != b.X ? a.X < b.X : a.Y < b.Y;
		}

		/// <summary>Lexicographic greater-than.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator >(Vec2 a, Vec2 b)
		{
			return a.X != b.X ? a.X > b.X : a.Y > b.Y;
		}

		/// <summary>Lexicographic less-than-or-equal.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator <=(Vec2 a, Vec2 b)
		{
			return a.X != b.X ? a.X < b.X : a.Y <= b.Y;
		}

		/// <summary>Lexicographic greater-than-or-equal.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator >=(Vec2 a, Vec2 b)
		{
			return a.X != b.X ? a.X > b.X : a.Y >= b.Y;
		}

		/// <summary>IEEE component-wise equality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(Vec2 a, Vec2 b)
		{
			return a.X == b.X && a.Y == b.Y;
		}

		/// <summary>IEEE component-wise inequality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(Vec2 a, Vec2 b)
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
		public bool Equals(Vec2 other)
		{
			return BitConverter.DoubleToUInt64Bits(this.X) == BitConverter.DoubleToUInt64Bits(other.X)
				&& BitConverter.DoubleToUInt64Bits(this.Y) == BitConverter.DoubleToUInt64Bits(other.Y);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Vec2 other && this.Equals(other);
		}

		/// <summary>
		/// Hash over the bit representation, so NaN hashes consistently — the Rust's
		/// comment on <c>hash_f64</c>. The Rust's Vec2 impl additionally mixes through a
		/// freshly seeded <c>RandomState</c>, which makes it non-reproducible even
		/// between two runs of the same Rust binary; that part is deliberately not
		/// ported. Nothing downstream lets a hash value reach an output, so only the
		/// equality it pairs with has to match.
		/// </summary>
		public override int GetHashCode()
		{
			return HashCode.Combine(
				BitConverter.DoubleToUInt64Bits(this.X),
				BitConverter.DoubleToUInt64Bits(this.Y));
		}
	}
}
