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

// Ports the Mat4 section of linalg.rs. Column-major: the four fields are the
// four columns. The determinant and the sixteen adjugate entries are
// transcribed term for term, in the Rust's term order and with the Rust's
// left-to-right association — every one is a sum of three products minus three
// products, and rebalancing any of them changes the low bits of a result that
// Inverse() then multiplies through the whole matrix. The module header for the
// whole Linalg folder lives in Vec3.cs.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace ManifoldSharp.Linalg
{
	/// <summary>
	/// <c>la::mat&lt;double, 4, 4&gt;</c> — 4x4 matrix stored as 4 columns of <see cref="Vec4"/>.
	/// </summary>
	public struct Mat4 : IEquatable<Mat4>
	{
		/// <summary>Column 0.</summary>
		public Vec4 X;

		/// <summary>Column 1.</summary>
		public Vec4 Y;

		/// <summary>Column 2.</summary>
		public Vec4 Z;

		/// <summary>Column 3.</summary>
		public Vec4 W;

		/// <summary>Creates a matrix from its four columns.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat4 FromCols(Vec4 x, Vec4 y, Vec4 z, Vec4 w)
		{
			return new Mat4 { X = x, Y = y, Z = z, W = w };
		}

		/// <summary>The identity matrix.</summary>
		public static Mat4 Identity()
		{
			return FromCols(
				new Vec4(1.0, 0.0, 0.0, 0.0),
				new Vec4(0.0, 1.0, 0.0, 0.0),
				new Vec4(0.0, 0.0, 1.0, 0.0),
				new Vec4(0.0, 0.0, 0.0, 1.0));
		}

		/// <summary>Row <paramref name="i"/>, gathered across the columns.</summary>
		public Vec4 Row(int i)
		{
			return new Vec4(this.X[i], this.Y[i], this.Z[i], this.W[i]);
		}

		/// <summary>The transpose — columns become rows.</summary>
		public Mat4 Transpose()
		{
			return FromCols(this.Row(0), this.Row(1), this.Row(2), this.Row(3));
		}

		/// <summary>The determinant, expanded along the first row.</summary>
		public double Determinant()
		{
			Mat4 a = this;
			return (a.X.X
					* (a.Y.Y * a.Z.Z * a.W.W + a.W.Y * a.Y.Z * a.Z.W + a.Z.Y * a.W.Z * a.Y.W
						- a.Y.Y * a.W.Z * a.Z.W
						- a.Z.Y * a.Y.Z * a.W.W
						- a.W.Y * a.Z.Z * a.Y.W))
				+ (a.X.Y
					* (a.Y.Z * a.W.W * a.Z.X + a.Z.Z * a.Y.W * a.W.X + a.W.Z * a.Z.W * a.Y.X
						- a.Y.Z * a.Z.W * a.W.X
						- a.W.Z * a.Y.W * a.Z.X
						- a.Z.Z * a.W.W * a.Y.X))
				+ (a.X.Z
					* (a.Y.W * a.Z.X * a.W.Y + a.W.W * a.Y.X * a.Z.Y + a.Z.W * a.W.X * a.Y.Y
						- a.Y.W * a.W.X * a.Z.Y
						- a.Z.W * a.Y.X * a.W.Y
						- a.W.W * a.Z.X * a.Y.Y))
				+ (a.X.W
					* (a.Y.X * a.W.Y * a.Z.Z + a.Z.X * a.Y.Y * a.W.Z + a.W.X * a.Z.Y * a.Y.Z
						- a.Y.X * a.Z.Y * a.W.Z
						- a.W.X * a.Y.Y * a.Z.Z
						- a.Z.X * a.W.Y * a.Y.Z));
		}

		/// <summary>The adjugate — the transpose of the cofactor matrix.</summary>
		public Mat4 Adjugate()
		{
			Mat4 a = this;
			return FromCols(
				new Vec4(
					a.Y.Y * a.Z.Z * a.W.W + a.W.Y * a.Y.Z * a.Z.W + a.Z.Y * a.W.Z * a.Y.W
						- a.Y.Y * a.W.Z * a.Z.W
						- a.Z.Y * a.Y.Z * a.W.W
						- a.W.Y * a.Z.Z * a.Y.W,
					a.X.Y * a.W.Z * a.Z.W + a.Z.Y * a.X.Z * a.W.W + a.W.Y * a.Z.Z * a.X.W
						- a.W.Y * a.X.Z * a.Z.W
						- a.Z.Y * a.W.Z * a.X.W
						- a.X.Y * a.Z.Z * a.W.W,
					a.X.Y * a.Y.Z * a.W.W + a.W.Y * a.X.Z * a.Y.W + a.Y.Y * a.W.Z * a.X.W
						- a.X.Y * a.W.Z * a.Y.W
						- a.Y.Y * a.X.Z * a.W.W
						- a.W.Y * a.Y.Z * a.X.W,
					a.X.Y * a.Z.Z * a.Y.W + a.Y.Y * a.X.Z * a.Z.W + a.Z.Y * a.Y.Z * a.X.W
						- a.X.Y * a.Y.Z * a.Z.W
						- a.Z.Y * a.X.Z * a.Y.W
						- a.Y.Y * a.Z.Z * a.X.W),
				new Vec4(
					a.Y.Z * a.W.W * a.Z.X + a.Z.Z * a.Y.W * a.W.X + a.W.Z * a.Z.W * a.Y.X
						- a.Y.Z * a.Z.W * a.W.X
						- a.W.Z * a.Y.W * a.Z.X
						- a.Z.Z * a.W.W * a.Y.X,
					a.X.Z * a.Z.W * a.W.X + a.W.Z * a.X.W * a.Z.X + a.Z.Z * a.W.W * a.X.X
						- a.X.Z * a.W.W * a.Z.X
						- a.Z.Z * a.X.W * a.W.X
						- a.W.Z * a.Z.W * a.X.X,
					a.X.Z * a.W.W * a.Y.X + a.Y.Z * a.X.W * a.W.X + a.W.Z * a.Y.W * a.X.X
						- a.X.Z * a.Y.W * a.W.X
						- a.W.Z * a.X.W * a.Y.X
						- a.Y.Z * a.W.W * a.X.X,
					a.X.Z * a.Y.W * a.Z.X + a.Z.Z * a.X.W * a.Y.X + a.Y.Z * a.Z.W * a.X.X
						- a.X.Z * a.Z.W * a.Y.X
						- a.Y.Z * a.X.W * a.Z.X
						- a.Z.Z * a.Y.W * a.X.X),
				new Vec4(
					a.Y.W * a.Z.X * a.W.Y + a.W.W * a.Y.X * a.Z.Y + a.Z.W * a.W.X * a.Y.Y
						- a.Y.W * a.W.X * a.Z.Y
						- a.Z.W * a.Y.X * a.W.Y
						- a.W.W * a.Z.X * a.Y.Y,
					a.X.W * a.W.X * a.Z.Y + a.Z.W * a.X.X * a.W.Y + a.W.W * a.Z.X * a.X.Y
						- a.X.W * a.Z.X * a.W.Y
						- a.W.W * a.X.X * a.Z.Y
						- a.Z.W * a.W.X * a.X.Y,
					a.X.W * a.Y.X * a.W.Y + a.W.W * a.X.X * a.Y.Y + a.Y.W * a.W.X * a.X.Y
						- a.X.W * a.W.X * a.Y.Y
						- a.Y.W * a.X.X * a.W.Y
						- a.W.W * a.Y.X * a.X.Y,
					a.X.W * a.Z.X * a.Y.Y + a.Y.W * a.X.X * a.Z.Y + a.Z.W * a.Y.X * a.X.Y
						- a.X.W * a.Y.X * a.Z.Y
						- a.Z.W * a.X.X * a.Y.Y
						- a.Y.W * a.Z.X * a.X.Y),
				new Vec4(
					a.Y.X * a.W.Y * a.Z.Z + a.Z.X * a.Y.Y * a.W.Z + a.W.X * a.Z.Y * a.Y.Z
						- a.Y.X * a.Z.Y * a.W.Z
						- a.W.X * a.Y.Y * a.Z.Z
						- a.Z.X * a.W.Y * a.Y.Z,
					a.X.X * a.Z.Y * a.W.Z + a.W.X * a.X.Y * a.Z.Z + a.Z.X * a.W.Y * a.X.Z
						- a.X.X * a.W.Y * a.Z.Z
						- a.Z.X * a.X.Y * a.W.Z
						- a.W.X * a.Z.Y * a.X.Z,
					a.X.X * a.W.Y * a.Y.Z + a.Y.X * a.X.Y * a.W.Z + a.W.X * a.Y.Y * a.X.Z
						- a.X.X * a.Y.Y * a.W.Z
						- a.W.X * a.X.Y * a.Y.Z
						- a.Y.X * a.W.Y * a.X.Z,
					a.X.X * a.Y.Y * a.Z.Z + a.Z.X * a.X.Y * a.Y.Z + a.Y.X * a.Z.Y * a.X.Z
						- a.X.X * a.Z.Y * a.Y.Z
						- a.Y.X * a.X.Y * a.Z.Z
						- a.Z.X * a.Y.Y * a.X.Z));
		}

		/// <summary>
		/// The inverse, as <c>adjugate * (1 / determinant)</c> — the reciprocal is formed
		/// once and multiplied through, exactly as the Rust does, not divided per entry.
		/// </summary>
		public Mat4 Inverse()
		{
			return this.Adjugate() * (1.0 / this.Determinant());
		}

		/// <summary>Column by index: 0 is <see cref="X"/> through 3 is <see cref="W"/>.</summary>
		/// <exception cref="ArgumentOutOfRangeException">The index is not in [0, 3].</exception>
		[UnscopedRef]
		public ref Vec4 this[int j]
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
					case 2:
						return ref this.Z;
					case 3:
						return ref this.W;
					default:
						throw new ArgumentOutOfRangeException(nameof(j), $"Mat4 column index out of range: {j}");
				}
			}
		}

		/// <summary>Matrix-vector product.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 operator *(Mat4 a, Vec4 b)
		{
			return a.X * b.X + a.Y * b.Y + a.Z * b.Z + a.W * b.W;
		}

		/// <summary>Matrix-matrix product.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat4 operator *(Mat4 a, Mat4 b)
		{
			return FromCols(a * b.X, a * b.Y, a * b.Z, a * b.W);
		}

		/// <summary>Scales every entry by a scalar.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat4 operator *(Mat4 a, double s)
		{
			return FromCols(a.X * s, a.Y * s, a.Z * s, a.W * s);
		}

		/// <summary>Entry-wise addition.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat4 operator +(Mat4 a, Mat4 b)
		{
			return FromCols(a.X + b.X, a.Y + b.Y, a.Z + b.Z, a.W + b.W);
		}

		/// <summary>Entry-wise subtraction.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat4 operator -(Mat4 a, Mat4 b)
		{
			return FromCols(a.X - b.X, a.Y - b.Y, a.Z - b.Z, a.W - b.W);
		}

		/// <summary>Entry-wise negation.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat4 operator -(Mat4 a)
		{
			return FromCols(-a.X, -a.Y, -a.Z, -a.W);
		}

		/// <summary>IEEE entry-wise equality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(Mat4 a, Mat4 b)
		{
			return a.X == b.X && a.Y == b.Y && a.Z == b.Z && a.W == b.W;
		}

		/// <summary>IEEE entry-wise inequality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(Mat4 a, Mat4 b)
		{
			return !(a == b);
		}

		/// <summary>Bit-exact equality, column by column. See the folder header in Vec3.cs.</summary>
		public bool Equals(Mat4 other)
		{
			return this.X.Equals(other.X)
				&& this.Y.Equals(other.Y)
				&& this.Z.Equals(other.Z)
				&& this.W.Equals(other.W);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Mat4 other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.X, this.Y, this.Z, this.W);
		}
	}
}
