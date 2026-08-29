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

// Ports the Vec4 section of linalg.rs. Vec4 doubles as the quaternion type —
// the Rust `pub type Quat = Vec4;` — and the quaternion functions that operate
// on it live in Quat.cs. The module header for the whole Linalg folder lives in
// Vec3.cs.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace ManifoldSharp.Linalg
{
	/// <summary>
	/// <c>la::vec&lt;double, 4&gt;</c> / <c>quat</c> — 4-component double column vector.
	/// </summary>
	/// <remarks>
	/// The Rust aliases <c>Quat</c> to this type, so a quaternion is the same bits used
	/// where the value represents xi+yj+zk+w. C# type aliases are file-scoped, so files
	/// that want to spell it <c>Quat</c> declare their own
	/// <c>using Quat = ManifoldSharp.Linalg.Vec4;</c>.
	/// <c>operator==</c> is the IEEE component-wise comparison; <see cref="Equals(Vec4)"/>
	/// and <see cref="GetHashCode"/> are bit-based. See the folder header in Vec3.cs.
	/// </remarks>
	public struct Vec4 : IEquatable<Vec4>
	{
		/// <summary>The x component.</summary>
		public double X;

		/// <summary>The y component.</summary>
		public double Y;

		/// <summary>The z component.</summary>
		public double Z;

		/// <summary>The w component.</summary>
		public double W;

		/// <summary>Creates a vector from its components.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Vec4(double x, double y, double z, double w)
		{
			this.X = x;
			this.Y = y;
			this.Z = z;
			this.W = w;
		}

		/// <summary>Creates a vector with all four components set to <paramref name="v"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 Splat(double v)
		{
			return new Vec4(v, v, v, v);
		}

		/// <summary>The <c>xy</c> swizzle.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Vec2 Xy()
		{
			return new Vec2(this.X, this.Y);
		}

		/// <summary>The <c>xyz</c> swizzle — for a quaternion, its vector part.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Vec3 Xyz()
		{
			return new Vec3(this.X, this.Y, this.Z);
		}

		/// <summary>
		/// Component by index: 0 is <see cref="X"/> through 3 is <see cref="W"/>. Returns
		/// a <c>ref</c> so both the Rust <c>Index</c> and <c>IndexMut</c> uses port
		/// literally.
		/// </summary>
		/// <exception cref="ArgumentOutOfRangeException">The index is not in [0, 3].</exception>
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
					case 3:
						return ref this.W;
					default:
						throw new ArgumentOutOfRangeException(nameof(i), $"Vec4 index out of range: {i}");
				}
			}
		}

		/// <summary>Ports the Rust <c>From&lt;[f64; 4]&gt;</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 FromArray(double[] a)
		{
			ArgumentNullException.ThrowIfNull(a);
			return new Vec4(a[0], a[1], a[2], a[3]);
		}

		/// <summary>Ports the Rust <c>From&lt;Vec4&gt; for [f64; 4]</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public double[] ToArray()
		{
			return new double[] { this.X, this.Y, this.Z, this.W };
		}

		/// <summary>Ports the Rust <c>From&lt;(Vec3, f64)&gt;</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 From(Vec3 xyz, double w)
		{
			return new Vec4(xyz.X, xyz.Y, xyz.Z, w);
		}

		/// <summary>Ports the Rust <c>From&lt;(Vec2, f64, f64)&gt;</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 From(Vec2 xy, double z, double w)
		{
			return new Vec4(xy.X, xy.Y, z, w);
		}

		/// <summary>Component-wise negation.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 operator -(Vec4 a)
		{
			return new Vec4(-a.X, -a.Y, -a.Z, -a.W);
		}

		/// <summary>Component-wise addition.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 operator +(Vec4 a, Vec4 b)
		{
			return new Vec4(a.X + b.X, a.Y + b.Y, a.Z + b.Z, a.W + b.W);
		}

		/// <summary>Component-wise subtraction.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 operator -(Vec4 a, Vec4 b)
		{
			return new Vec4(a.X - b.X, a.Y - b.Y, a.Z - b.Z, a.W - b.W);
		}

		/// <summary>
		/// Component-wise multiplication (C++ <c>cmul</c>). This is <em>not</em> the
		/// Hamilton product; quaternion multiplication is <c>LinalgFunctions.QMul</c>.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 operator *(Vec4 a, Vec4 b)
		{
			return new Vec4(a.X * b.X, a.Y * b.Y, a.Z * b.Z, a.W * b.W);
		}

		/// <summary>Component-wise division.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 operator /(Vec4 a, Vec4 b)
		{
			return new Vec4(a.X / b.X, a.Y / b.Y, a.Z / b.Z, a.W / b.W);
		}

		/// <summary>Scales by a scalar on the right.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 operator *(Vec4 a, double s)
		{
			return new Vec4(a.X * s, a.Y * s, a.Z * s, a.W * s);
		}

		/// <summary>Scales by a scalar on the left; the Rust writes <c>s * v.x</c>, kept in that order.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 operator *(double s, Vec4 v)
		{
			return new Vec4(s * v.X, s * v.Y, s * v.Z, s * v.W);
		}

		/// <summary>Divides every component by a scalar.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 operator /(Vec4 a, double s)
		{
			return new Vec4(a.X / s, a.Y / s, a.Z / s, a.W / s);
		}

		/// <summary>Adds a scalar to every component.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 operator +(Vec4 a, double s)
		{
			return new Vec4(a.X + s, a.Y + s, a.Z + s, a.W + s);
		}

		/// <summary>Subtracts a scalar from every component.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 operator -(Vec4 a, double s)
		{
			return new Vec4(a.X - s, a.Y - s, a.Z - s, a.W - s);
		}

		// AddAssign/SubAssign/MulAssign/DivAssign need no port: C# synthesizes +=, -=,
		// *= and /= from the binary operators above.

		// Lexicographic ordering. A NaN component yields false in every direction,
		// which is what Rust's partial_cmp returning None does.

		/// <summary>Lexicographic less-than.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator <(Vec4 a, Vec4 b)
		{
			return a.X != b.X ? a.X < b.X : (a.Y != b.Y ? a.Y < b.Y : (a.Z != b.Z ? a.Z < b.Z : a.W < b.W));
		}

		/// <summary>Lexicographic greater-than.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator >(Vec4 a, Vec4 b)
		{
			return a.X != b.X ? a.X > b.X : (a.Y != b.Y ? a.Y > b.Y : (a.Z != b.Z ? a.Z > b.Z : a.W > b.W));
		}

		/// <summary>Lexicographic less-than-or-equal.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator <=(Vec4 a, Vec4 b)
		{
			return a.X != b.X ? a.X < b.X : (a.Y != b.Y ? a.Y < b.Y : (a.Z != b.Z ? a.Z < b.Z : a.W <= b.W));
		}

		/// <summary>Lexicographic greater-than-or-equal.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator >=(Vec4 a, Vec4 b)
		{
			return a.X != b.X ? a.X > b.X : (a.Y != b.Y ? a.Y > b.Y : (a.Z != b.Z ? a.Z > b.Z : a.W >= b.W));
		}

		/// <summary>IEEE component-wise equality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(Vec4 a, Vec4 b)
		{
			return a.X == b.X && a.Y == b.Y && a.Z == b.Z && a.W == b.W;
		}

		/// <summary>IEEE component-wise inequality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(Vec4 a, Vec4 b)
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
		public bool Equals(Vec4 other)
		{
			return BitConverter.DoubleToUInt64Bits(this.X) == BitConverter.DoubleToUInt64Bits(other.X)
				&& BitConverter.DoubleToUInt64Bits(this.Y) == BitConverter.DoubleToUInt64Bits(other.Y)
				&& BitConverter.DoubleToUInt64Bits(this.Z) == BitConverter.DoubleToUInt64Bits(other.Z)
				&& BitConverter.DoubleToUInt64Bits(this.W) == BitConverter.DoubleToUInt64Bits(other.W);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Vec4 other && this.Equals(other);
		}

		/// <summary>
		/// Hash over the bit representation of each component, in field order, matching
		/// the Rust's Vec4 <c>Hash</c> impl. The mixing function differs from Rust's
		/// SipHash; only the equality it pairs with has to match.
		/// </summary>
		public override int GetHashCode()
		{
			return HashCode.Combine(
				BitConverter.DoubleToUInt64Bits(this.X),
				BitConverter.DoubleToUInt64Bits(this.Y),
				BitConverter.DoubleToUInt64Bits(this.Z),
				BitConverter.DoubleToUInt64Bits(this.W));
		}
	}
}
