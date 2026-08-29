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

// Ports the IVec2 / IVec3 / IVec4 / BVec4 / UVec3 sections of linalg.rs — the
// non-double vectors. The module header for the whole Linalg folder lives in
// Vec3.cs.
//
// The operator matrix here is deliberately asymmetric, because the Rust's is:
// only IVec2 has a scalar-on-the-left multiply, and only IVec2 and IVec3 have a
// vector*vector multiply. Those gaps are preserved — an unported operator is a
// compile error at the call site, which is the signal we want.
//
// One asymmetry is not expressible. The Rust implements AddAssign for IVec2 and
// IVec3 but not for IVec4, whereas C# synthesizes `+=` from `operator+` and
// offers no way to withhold it. So `a += b` on an IVec4 compiles here and does
// not in Rust. It computes the right value — only the restriction is lost, not
// the arithmetic — and no ported call site depends on the Rust side rejecting
// it.
//
// Integer arithmetic is written `unchecked` throughout: Rust release builds wrap
// on i32 overflow, and this port must wrap identically even if a project ever
// turns on CheckForOverflowUnderflow.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace ManifoldSharp.Linalg
{
	/// <summary>
	/// <c>la::vec&lt;int, 2&gt;</c> — 2-component int vector with a total (lexicographic) order.
	/// </summary>
	public struct IVec2 : IEquatable<IVec2>, IComparable<IVec2>
	{
		/// <summary>The x component.</summary>
		public int X;

		/// <summary>The y component.</summary>
		public int Y;

		/// <summary>Creates a vector from its components.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public IVec2(int x, int y)
		{
			this.X = x;
			this.Y = y;
		}

		/// <summary>Creates a vector with both components set to <paramref name="v"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec2 Splat(int v)
		{
			return new IVec2(v, v);
		}

		/// <summary>Component by index: 0 is <see cref="X"/>, 1 is <see cref="Y"/>.</summary>
		/// <exception cref="ArgumentOutOfRangeException">The index is not 0 or 1.</exception>
		[UnscopedRef]
		public ref int this[int i]
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
						throw new ArgumentOutOfRangeException(nameof(i), $"IVec2 index out of range: {i}");
				}
			}
		}

		/// <summary>Ports the Rust <c>From&lt;[i32; 2]&gt;</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec2 FromArray(int[] a)
		{
			ArgumentNullException.ThrowIfNull(a);
			return new IVec2(a[0], a[1]);
		}

		/// <summary>Component-wise negation.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec2 operator -(IVec2 a)
		{
			return new IVec2(unchecked(-a.X), unchecked(-a.Y));
		}

		/// <summary>Component-wise addition.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec2 operator +(IVec2 a, IVec2 b)
		{
			return new IVec2(unchecked(a.X + b.X), unchecked(a.Y + b.Y));
		}

		/// <summary>Component-wise subtraction.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec2 operator -(IVec2 a, IVec2 b)
		{
			return new IVec2(unchecked(a.X - b.X), unchecked(a.Y - b.Y));
		}

		/// <summary>Component-wise multiplication.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec2 operator *(IVec2 a, IVec2 b)
		{
			return new IVec2(unchecked(a.X * b.X), unchecked(a.Y * b.Y));
		}

		/// <summary>Scales by a scalar on the right.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec2 operator *(IVec2 a, int s)
		{
			return new IVec2(unchecked(a.X * s), unchecked(a.Y * s));
		}

		/// <summary>Scales by a scalar on the left; the Rust writes <c>s * v.x</c>, kept in that order.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec2 operator *(int s, IVec2 v)
		{
			return new IVec2(unchecked(s * v.X), unchecked(s * v.Y));
		}

		/// <summary>Lexicographic less-than.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator <(IVec2 a, IVec2 b)
		{
			return a.CompareTo(b) < 0;
		}

		/// <summary>Lexicographic greater-than.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator >(IVec2 a, IVec2 b)
		{
			return a.CompareTo(b) > 0;
		}

		/// <summary>Lexicographic less-than-or-equal.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator <=(IVec2 a, IVec2 b)
		{
			return a.CompareTo(b) <= 0;
		}

		/// <summary>Lexicographic greater-than-or-equal.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator >=(IVec2 a, IVec2 b)
		{
			return a.CompareTo(b) >= 0;
		}

		/// <summary>Component-wise equality.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(IVec2 a, IVec2 b)
		{
			return a.X == b.X && a.Y == b.Y;
		}

		/// <summary>Component-wise inequality.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(IVec2 a, IVec2 b)
		{
			return !(a == b);
		}

		/// <summary>Lexicographic comparison — the Rust <c>Ord</c> impl, x then y.</summary>
		public int CompareTo(IVec2 other)
		{
			int c = this.X.CompareTo(other.X);
			return c != 0 ? c : this.Y.CompareTo(other.Y);
		}

		/// <inheritdoc/>
		public bool Equals(IVec2 other)
		{
			return this == other;
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is IVec2 other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.X, this.Y);
		}
	}

	/// <summary>
	/// <c>la::vec&lt;int, 3&gt;</c> — 3-component int vector with a total (lexicographic) order.
	/// </summary>
	public struct IVec3 : IEquatable<IVec3>, IComparable<IVec3>
	{
		/// <summary>The x component.</summary>
		public int X;

		/// <summary>The y component.</summary>
		public int Y;

		/// <summary>The z component.</summary>
		public int Z;

		/// <summary>Creates a vector from its components.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public IVec3(int x, int y, int z)
		{
			this.X = x;
			this.Y = y;
			this.Z = z;
		}

		/// <summary>Creates a vector with all three components set to <paramref name="v"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec3 Splat(int v)
		{
			return new IVec3(v, v, v);
		}

		/// <summary>The <c>xy</c> swizzle.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public IVec2 Xy()
		{
			return new IVec2(this.X, this.Y);
		}

		/// <summary>Component by index: 0 is <see cref="X"/>, 1 is <see cref="Y"/>, 2 is <see cref="Z"/>.</summary>
		/// <exception cref="ArgumentOutOfRangeException">The index is not 0, 1 or 2.</exception>
		[UnscopedRef]
		public ref int this[int i]
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
						throw new ArgumentOutOfRangeException(nameof(i), $"IVec3 index out of range: {i}");
				}
			}
		}

		/// <summary>Ports the Rust <c>From&lt;[i32; 3]&gt;</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec3 FromArray(int[] a)
		{
			ArgumentNullException.ThrowIfNull(a);
			return new IVec3(a[0], a[1], a[2]);
		}

		/// <summary>Ports the Rust <c>From&lt;IVec3&gt; for [i32; 3]</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public int[] ToArray()
		{
			return new int[] { this.X, this.Y, this.Z };
		}

		/// <summary>Component-wise negation.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec3 operator -(IVec3 a)
		{
			return new IVec3(unchecked(-a.X), unchecked(-a.Y), unchecked(-a.Z));
		}

		/// <summary>Component-wise addition.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec3 operator +(IVec3 a, IVec3 b)
		{
			return new IVec3(unchecked(a.X + b.X), unchecked(a.Y + b.Y), unchecked(a.Z + b.Z));
		}

		/// <summary>Component-wise subtraction.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec3 operator -(IVec3 a, IVec3 b)
		{
			return new IVec3(unchecked(a.X - b.X), unchecked(a.Y - b.Y), unchecked(a.Z - b.Z));
		}

		/// <summary>Component-wise multiplication.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec3 operator *(IVec3 a, IVec3 b)
		{
			return new IVec3(unchecked(a.X * b.X), unchecked(a.Y * b.Y), unchecked(a.Z * b.Z));
		}

		/// <summary>Scales by a scalar on the right. The Rust defines no left-hand form for IVec3.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec3 operator *(IVec3 a, int s)
		{
			return new IVec3(unchecked(a.X * s), unchecked(a.Y * s), unchecked(a.Z * s));
		}

		/// <summary>Lexicographic less-than.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator <(IVec3 a, IVec3 b)
		{
			return a.CompareTo(b) < 0;
		}

		/// <summary>Lexicographic greater-than.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator >(IVec3 a, IVec3 b)
		{
			return a.CompareTo(b) > 0;
		}

		/// <summary>Lexicographic less-than-or-equal.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator <=(IVec3 a, IVec3 b)
		{
			return a.CompareTo(b) <= 0;
		}

		/// <summary>Lexicographic greater-than-or-equal.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator >=(IVec3 a, IVec3 b)
		{
			return a.CompareTo(b) >= 0;
		}

		/// <summary>Component-wise equality.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(IVec3 a, IVec3 b)
		{
			return a.X == b.X && a.Y == b.Y && a.Z == b.Z;
		}

		/// <summary>Component-wise inequality.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(IVec3 a, IVec3 b)
		{
			return !(a == b);
		}

		/// <summary>Lexicographic comparison — the Rust <c>Ord</c> impl, x then y then z.</summary>
		public int CompareTo(IVec3 other)
		{
			int c = this.X.CompareTo(other.X);
			if (c != 0)
			{
				return c;
			}

			c = this.Y.CompareTo(other.Y);
			return c != 0 ? c : this.Z.CompareTo(other.Z);
		}

		/// <inheritdoc/>
		public bool Equals(IVec3 other)
		{
			return this == other;
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is IVec3 other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.X, this.Y, this.Z);
		}
	}

	/// <summary>
	/// <c>la::vec&lt;int, 4&gt;</c> — 4-component int vector with a total (lexicographic) order.
	/// </summary>
	public struct IVec4 : IEquatable<IVec4>, IComparable<IVec4>
	{
		/// <summary>The x component.</summary>
		public int X;

		/// <summary>The y component.</summary>
		public int Y;

		/// <summary>The z component.</summary>
		public int Z;

		/// <summary>The w component.</summary>
		public int W;

		/// <summary>Creates a vector from its components.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public IVec4(int x, int y, int z, int w)
		{
			this.X = x;
			this.Y = y;
			this.Z = z;
			this.W = w;
		}

		/// <summary>Creates a vector with all four components set to <paramref name="v"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec4 Splat(int v)
		{
			return new IVec4(v, v, v, v);
		}

		/// <summary>The <c>xyz</c> swizzle.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public IVec3 Xyz()
		{
			return new IVec3(this.X, this.Y, this.Z);
		}

		/// <summary>Component by index: 0 is <see cref="X"/> through 3 is <see cref="W"/>.</summary>
		/// <exception cref="ArgumentOutOfRangeException">The index is not in [0, 3].</exception>
		[UnscopedRef]
		public ref int this[int i]
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
						throw new ArgumentOutOfRangeException(nameof(i), $"IVec4 index out of range: {i}");
				}
			}
		}

		/// <summary>Ports the Rust <c>From&lt;[i32; 4]&gt;</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec4 FromArray(int[] a)
		{
			ArgumentNullException.ThrowIfNull(a);
			return new IVec4(a[0], a[1], a[2], a[3]);
		}

		/// <summary>Component-wise negation.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec4 operator -(IVec4 a)
		{
			return new IVec4(unchecked(-a.X), unchecked(-a.Y), unchecked(-a.Z), unchecked(-a.W));
		}

		/// <summary>Component-wise addition.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec4 operator +(IVec4 a, IVec4 b)
		{
			return new IVec4(unchecked(a.X + b.X), unchecked(a.Y + b.Y), unchecked(a.Z + b.Z), unchecked(a.W + b.W));
		}

		/// <summary>Component-wise subtraction.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec4 operator -(IVec4 a, IVec4 b)
		{
			return new IVec4(unchecked(a.X - b.X), unchecked(a.Y - b.Y), unchecked(a.Z - b.Z), unchecked(a.W - b.W));
		}

		/// <summary>Scales by a scalar on the right. The Rust defines no vector*vector or left-hand form for IVec4.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static IVec4 operator *(IVec4 a, int s)
		{
			return new IVec4(unchecked(a.X * s), unchecked(a.Y * s), unchecked(a.Z * s), unchecked(a.W * s));
		}

		/// <summary>Lexicographic less-than.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator <(IVec4 a, IVec4 b)
		{
			return a.CompareTo(b) < 0;
		}

		/// <summary>Lexicographic greater-than.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator >(IVec4 a, IVec4 b)
		{
			return a.CompareTo(b) > 0;
		}

		/// <summary>Lexicographic less-than-or-equal.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator <=(IVec4 a, IVec4 b)
		{
			return a.CompareTo(b) <= 0;
		}

		/// <summary>Lexicographic greater-than-or-equal.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator >=(IVec4 a, IVec4 b)
		{
			return a.CompareTo(b) >= 0;
		}

		/// <summary>Component-wise equality.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(IVec4 a, IVec4 b)
		{
			return a.X == b.X && a.Y == b.Y && a.Z == b.Z && a.W == b.W;
		}

		/// <summary>Component-wise inequality.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(IVec4 a, IVec4 b)
		{
			return !(a == b);
		}

		/// <summary>Lexicographic comparison — the Rust <c>Ord</c> impl, x then y then z then w.</summary>
		public int CompareTo(IVec4 other)
		{
			int c = this.X.CompareTo(other.X);
			if (c != 0)
			{
				return c;
			}

			c = this.Y.CompareTo(other.Y);
			if (c != 0)
			{
				return c;
			}

			c = this.Z.CompareTo(other.Z);
			return c != 0 ? c : this.W.CompareTo(other.W);
		}

		/// <inheritdoc/>
		public bool Equals(IVec4 other)
		{
			return this == other;
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is IVec4 other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.X, this.Y, this.Z, this.W);
		}
	}

	/// <summary>
	/// <c>la::vec&lt;bool, 4&gt;</c> — 4-component bool vector.
	/// </summary>
	public struct BVec4 : IEquatable<BVec4>
	{
		/// <summary>The x component.</summary>
		public bool X;

		/// <summary>The y component.</summary>
		public bool Y;

		/// <summary>The z component.</summary>
		public bool Z;

		/// <summary>The w component.</summary>
		public bool W;

		/// <summary>Creates a vector from its components.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public BVec4(bool x, bool y, bool z, bool w)
		{
			this.X = x;
			this.Y = y;
			this.Z = z;
			this.W = w;
		}

		/// <summary>Creates a vector with all four components set to <paramref name="v"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BVec4 Splat(bool v)
		{
			return new BVec4(v, v, v, v);
		}

		/// <summary>Component by index: 0 is <see cref="X"/> through 3 is <see cref="W"/>.</summary>
		/// <exception cref="ArgumentOutOfRangeException">The index is not in [0, 3].</exception>
		[UnscopedRef]
		public ref bool this[int i]
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
						throw new ArgumentOutOfRangeException(nameof(i), $"BVec4 index out of range: {i}");
				}
			}
		}

		/// <summary>Component-wise logical negation — the Rust <c>Not</c> impl.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BVec4 operator !(BVec4 a)
		{
			return new BVec4(!a.X, !a.Y, !a.Z, !a.W);
		}

		/// <summary>Component-wise conjunction — the Rust <c>BitAnd</c> impl (non-short-circuiting).</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BVec4 operator &(BVec4 a, BVec4 b)
		{
			return new BVec4(a.X & b.X, a.Y & b.Y, a.Z & b.Z, a.W & b.W);
		}

		/// <summary>Component-wise disjunction — the Rust <c>BitOr</c> impl (non-short-circuiting).</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static BVec4 operator |(BVec4 a, BVec4 b)
		{
			return new BVec4(a.X | b.X, a.Y | b.Y, a.Z | b.Z, a.W | b.W);
		}

		/// <summary>Component-wise equality.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(BVec4 a, BVec4 b)
		{
			return a.X == b.X && a.Y == b.Y && a.Z == b.Z && a.W == b.W;
		}

		/// <summary>Component-wise inequality.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(BVec4 a, BVec4 b)
		{
			return !(a == b);
		}

		/// <inheritdoc/>
		public bool Equals(BVec4 other)
		{
			return this == other;
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is BVec4 other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.X, this.Y, this.Z, this.W);
		}
	}

	/// <summary>
	/// uint index vector — used for triangle index triples in Morton sorting.
	/// </summary>
	/// <remarks>
	/// A readonly struct because the Rust gives it <c>Index</c> but deliberately no
	/// <c>IndexMut</c>, and no arithmetic operators at all.
	/// </remarks>
	public readonly struct UVec3 : IEquatable<UVec3>
	{
		/// <summary>The x component.</summary>
		public readonly uint X;

		/// <summary>The y component.</summary>
		public readonly uint Y;

		/// <summary>The z component.</summary>
		public readonly uint Z;

		/// <summary>Creates a vector from its components.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public UVec3(uint x, uint y, uint z)
		{
			this.X = x;
			this.Y = y;
			this.Z = z;
		}

		/// <summary>Component by index: 0 is <see cref="X"/>, 1 is <see cref="Y"/>, 2 is <see cref="Z"/>.</summary>
		/// <exception cref="ArgumentOutOfRangeException">The index is not 0, 1 or 2.</exception>
		public uint this[int i]
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			get
			{
				switch (i)
				{
					case 0:
						return this.X;
					case 1:
						return this.Y;
					case 2:
						return this.Z;
					default:
						throw new ArgumentOutOfRangeException(nameof(i), $"UVec3 index out of range: {i}");
				}
			}
		}

		/// <summary>Component-wise equality.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(UVec3 a, UVec3 b)
		{
			return a.X == b.X && a.Y == b.Y && a.Z == b.Z;
		}

		/// <summary>Component-wise inequality.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(UVec3 a, UVec3 b)
		{
			return !(a == b);
		}

		/// <inheritdoc/>
		public bool Equals(UVec3 other)
		{
			return this == other;
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is UVec3 other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.X, this.Y, this.Z);
		}
	}
}
