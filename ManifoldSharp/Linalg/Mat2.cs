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

// Ports the Mat2 section of linalg.rs. mat<T, M, N> is M rows by N cols, stored
// as N column vectors of length M — so Mat2's two fields are its two columns,
// not its two rows. The module header for the whole Linalg folder lives in
// Vec3.cs.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace ManifoldSharp.Linalg
{
	/// <summary>
	/// <c>la::mat&lt;double, 2, 2&gt;</c> — 2x2 matrix stored as 2 columns of <see cref="Vec2"/>.
	/// </summary>
	public struct Mat2 : IEquatable<Mat2>
	{
		/// <summary>Column 0.</summary>
		public Vec2 X;

		/// <summary>Column 1.</summary>
		public Vec2 Y;

		/// <summary>Creates a matrix from its two columns.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat2 FromCols(Vec2 x, Vec2 y)
		{
			return new Mat2 { X = x, Y = y };
		}

		/// <summary>The identity matrix.</summary>
		public static Mat2 Identity()
		{
			return FromCols(new Vec2(1.0, 0.0), new Vec2(0.0, 1.0));
		}

		/// <summary>Row <paramref name="i"/>, gathered across the columns.</summary>
		public Vec2 Row(int i)
		{
			return new Vec2(this.X[i], this.Y[i]);
		}

		/// <summary>The transpose — columns become rows.</summary>
		public Mat2 Transpose()
		{
			return FromCols(this.Row(0), this.Row(1));
		}

		/// <summary>The determinant.</summary>
		public double Determinant()
		{
			return this.X.X * this.Y.Y - this.X.Y * this.Y.X;
		}

		/// <summary>Column by index: 0 is <see cref="X"/>, 1 is <see cref="Y"/>.</summary>
		/// <exception cref="ArgumentOutOfRangeException">The index is not 0 or 1.</exception>
		[UnscopedRef]
		public ref Vec2 this[int j]
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			get
			{
				switch (j)
				{
					case 0:
						return ref this.X;
					case 1:
						return ref this.Y;
					default:
						throw new ArgumentOutOfRangeException(nameof(j), $"Mat2 column index out of range: {j}");
				}
			}
		}

		/// <summary>Matrix-vector product.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 operator *(Mat2 a, Vec2 b)
		{
			return a.X * b.X + a.Y * b.Y;
		}

		/// <summary>Matrix-matrix product.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat2 operator *(Mat2 a, Mat2 b)
		{
			return FromCols(a * b.X, a * b.Y);
		}

		/// <summary>Scales every entry by a scalar.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat2 operator *(Mat2 a, double s)
		{
			return FromCols(a.X * s, a.Y * s);
		}

		/// <summary>Entry-wise addition.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat2 operator +(Mat2 a, Mat2 b)
		{
			return FromCols(a.X + b.X, a.Y + b.Y);
		}

		/// <summary>Entry-wise subtraction.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat2 operator -(Mat2 a, Mat2 b)
		{
			return FromCols(a.X - b.X, a.Y - b.Y);
		}

		/// <summary>Entry-wise negation.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat2 operator -(Mat2 a)
		{
			return FromCols(-a.X, -a.Y);
		}

		/// <summary>IEEE entry-wise equality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(Mat2 a, Mat2 b)
		{
			return a.X == b.X && a.Y == b.Y;
		}

		/// <summary>IEEE entry-wise inequality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(Mat2 a, Mat2 b)
		{
			return !(a == b);
		}

		/// <summary>Bit-exact equality, column by column. See the folder header in Vec3.cs.</summary>
		public bool Equals(Mat2 other)
		{
			return this.X.Equals(other.X) && this.Y.Equals(other.Y);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Mat2 other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.X, this.Y);
		}
	}
}
