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

// Ports the Mat3 section of linalg.rs. Column-major: the three fields are the
// three columns. The adjugate and determinant expressions are transcribed term
// for term in the Rust's order — reassociating them changes the low bits, and
// Inverse() divides by the determinant, so a changed bit there moves every
// entry. The module header for the whole Linalg folder lives in Vec3.cs.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace ManifoldSharp.Linalg
{
	/// <summary>
	/// <c>la::mat&lt;double, 3, 3&gt;</c> — 3x3 matrix stored as 3 columns of <see cref="Vec3"/>.
	/// </summary>
	public struct Mat3 : IEquatable<Mat3>
	{
		/// <summary>Column 0.</summary>
		public Vec3 X;

		/// <summary>Column 1.</summary>
		public Vec3 Y;

		/// <summary>Column 2.</summary>
		public Vec3 Z;

		/// <summary>Creates a matrix from its three columns.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat3 FromCols(Vec3 x, Vec3 y, Vec3 z)
		{
			return new Mat3 { X = x, Y = y, Z = z };
		}

		/// <summary>The identity matrix.</summary>
		public static Mat3 Identity()
		{
			return FromCols(
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
				new Vec3(0.0, 0.0, 1.0));
		}

		/// <summary>Row <paramref name="i"/>, gathered across the columns.</summary>
		public Vec3 Row(int i)
		{
			return new Vec3(this.X[i], this.Y[i], this.Z[i]);
		}

		/// <summary>The transpose — columns become rows.</summary>
		public Mat3 Transpose()
		{
			return FromCols(this.Row(0), this.Row(1), this.Row(2));
		}

		/// <summary>C++ <c>adjugate(mat&lt;T,3,3&gt;)</c> — the transpose of the cofactor matrix.</summary>
		public Mat3 Adjugate()
		{
			Mat3 a = this;
			return FromCols(
				new Vec3(
					a.Y.Y * a.Z.Z - a.Z.Y * a.Y.Z,
					a.Z.Y * a.X.Z - a.X.Y * a.Z.Z,
					a.X.Y * a.Y.Z - a.Y.Y * a.X.Z),
				new Vec3(
					a.Y.Z * a.Z.X - a.Z.Z * a.Y.X,
					a.Z.Z * a.X.X - a.X.Z * a.Z.X,
					a.X.Z * a.Y.X - a.Y.Z * a.X.X),
				new Vec3(
					a.Y.X * a.Z.Y - a.Z.X * a.Y.Y,
					a.Z.X * a.X.Y - a.X.X * a.Z.Y,
					a.X.X * a.Y.Y - a.Y.X * a.X.Y));
		}

		/// <summary>The determinant.</summary>
		public double Determinant()
		{
			Mat3 a = this;
			return (a.X.X * (a.Y.Y * a.Z.Z - a.Z.Y * a.Y.Z))
				+ (a.X.Y * (a.Y.Z * a.Z.X - a.Z.Z * a.Y.X))
				+ (a.X.Z * (a.Y.X * a.Z.Y - a.Z.X * a.Y.Y));
		}

		/// <summary>
		/// The inverse, as <c>adjugate * (1 / determinant)</c> — the reciprocal is formed
		/// once and multiplied through, exactly as the Rust does, not divided per entry.
		/// </summary>
		public Mat3 Inverse()
		{
			return this.Adjugate() * (1.0 / this.Determinant());
		}

		/// <summary>The diagonal entries.</summary>
		public Vec3 Diagonal()
		{
			return new Vec3(this.X.X, this.Y.Y, this.Z.Z);
		}

		/// <summary>The trace — the sum of the diagonal entries.</summary>
		public double Trace()
		{
			return this.X.X + this.Y.Y + this.Z.Z;
		}

		/// <summary>Column by index: 0 is <see cref="X"/>, 1 is <see cref="Y"/>, 2 is <see cref="Z"/>.</summary>
		/// <exception cref="ArgumentOutOfRangeException">The index is not 0, 1 or 2.</exception>
		[UnscopedRef]
		public ref Vec3 this[int j]
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
					default:
						throw new ArgumentOutOfRangeException(nameof(j), $"Mat3 column index out of range: {j}");
				}
			}
		}

		/// <summary>Matrix-vector product.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 operator *(Mat3 a, Vec3 b)
		{
			return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
		}

		/// <summary>Matrix-matrix product.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat3 operator *(Mat3 a, Mat3 b)
		{
			return FromCols(a * b.X, a * b.Y, a * b.Z);
		}

		/// <summary>Scales every entry by a scalar.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat3 operator *(Mat3 a, double s)
		{
			return FromCols(a.X * s, a.Y * s, a.Z * s);
		}

		/// <summary>Entry-wise addition.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat3 operator +(Mat3 a, Mat3 b)
		{
			return FromCols(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
		}

		/// <summary>Entry-wise subtraction.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat3 operator -(Mat3 a, Mat3 b)
		{
			return FromCols(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
		}

		/// <summary>Entry-wise negation.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat3 operator -(Mat3 a)
		{
			return FromCols(-a.X, -a.Y, -a.Z);
		}

		/// <summary>IEEE entry-wise equality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(Mat3 a, Mat3 b)
		{
			return a.X == b.X && a.Y == b.Y && a.Z == b.Z;
		}

		/// <summary>IEEE entry-wise inequality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(Mat3 a, Mat3 b)
		{
			return !(a == b);
		}

		/// <summary>Bit-exact equality, column by column. See the folder header in Vec3.cs.</summary>
		public bool Equals(Mat3 other)
		{
			return this.X.Equals(other.X) && this.Y.Equals(other.Y) && this.Z.Equals(other.Z);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Mat3 other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.X, this.Y, this.Z);
		}
	}
}
