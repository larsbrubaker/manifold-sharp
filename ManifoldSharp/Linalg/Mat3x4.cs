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

// Ports the Mat3x4 section of linalg.rs — the affine transform type, 3 rows by
// 4 columns. Both mixed products with Mat4 live here rather than in Mat4.cs,
// because C# lets an operator be declared on either operand's type and keeping
// them together is what makes the asymmetry legible: Mat3x4 * Mat4 stays a
// Mat3x4, while Mat4 * Mat3x4 promotes and returns a Mat4. The module header
// for the whole Linalg folder lives in Vec3.cs.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace ManifoldSharp.Linalg
{
	/// <summary>
	/// <c>la::mat&lt;double, 3, 4&gt;</c> — 3 rows, 4 columns (affine transform type).
	/// Stored as 4 columns each of <see cref="Vec3"/>: [x|y|z|translation].
	/// </summary>
	public struct Mat3x4 : IEquatable<Mat3x4>
	{
		/// <summary>Column 0.</summary>
		public Vec3 X;

		/// <summary>Column 1.</summary>
		public Vec3 Y;

		/// <summary>Column 2.</summary>
		public Vec3 Z;

		/// <summary>Column 3 (translation).</summary>
		public Vec3 W;

		/// <summary>Creates a matrix from its four columns.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat3x4 FromCols(Vec3 x, Vec3 y, Vec3 z, Vec3 w)
		{
			return new Mat3x4 { X = x, Y = y, Z = z, W = w };
		}

		/// <summary>Identity: rotation=I, translation=0.</summary>
		public static Mat3x4 Identity()
		{
			return FromCols(
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
				new Vec3(0.0, 0.0, 1.0),
				new Vec3(0.0, 0.0, 0.0));
		}

		/// <summary>Row <paramref name="i"/>, gathered across all four columns.</summary>
		public Vec4 Row(int i)
		{
			return new Vec4(this.X[i], this.Y[i], this.Z[i], this.W[i]);
		}

		/// <summary>Extract the 3x3 rotation/scale submatrix (first 3 columns).</summary>
		public Mat3 Rotation()
		{
			return Mat3.FromCols(this.X, this.Y, this.Z);
		}

		/// <summary>Extract the translation column.</summary>
		public Vec3 Translation()
		{
			return this.W;
		}

		/// <summary>Column by index: 0 is <see cref="X"/> through 3 is <see cref="W"/>.</summary>
		/// <exception cref="ArgumentOutOfRangeException">The index is not in [0, 3].</exception>
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
					case 3:
						return ref this.W;
					default:
						throw new ArgumentOutOfRangeException(nameof(j), $"Mat3x4 column index out of range: {j}");
				}
			}
		}

		/// <summary>mat3x4 * vec4 -&gt; vec3 (matrix-vector multiplication).</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 operator *(Mat3x4 a, Vec4 b)
		{
			return a.X * b.X + a.Y * b.Y + a.Z * b.Z + a.W * b.W;
		}

		/// <summary>mat3x4 * mat4x4 -&gt; mat3x4 (chain transforms).</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat3x4 operator *(Mat3x4 a, Mat4 b)
		{
			return FromCols(a * b.X, a * b.Y, a * b.Z, a * b.W);
		}

		/// <summary>mat4x4 * mat3x4 — promotes mat3x4 to mat4x4 then multiplies; returns Mat4.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat4 operator *(Mat4 a, Mat3x4 b)
		{
			// Treat mat3x4 as mat4x4 with bottom row [0,0,0,1]
			Mat4 b4 = LinalgFunctions.Mat3x4ToMat4(b);
			return a * b4;
		}

		/// <summary>Scales every entry by a scalar.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat3x4 operator *(Mat3x4 a, double s)
		{
			return FromCols(a.X * s, a.Y * s, a.Z * s, a.W * s);
		}

		/// <summary>Entry-wise addition.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat3x4 operator +(Mat3x4 a, Mat3x4 b)
		{
			return FromCols(a.X + b.X, a.Y + b.Y, a.Z + b.Z, a.W + b.W);
		}

		/// <summary>Entry-wise subtraction.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat3x4 operator -(Mat3x4 a, Mat3x4 b)
		{
			return FromCols(a.X - b.X, a.Y - b.Y, a.Z - b.Z, a.W - b.W);
		}

		/// <summary>Entry-wise negation.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat3x4 operator -(Mat3x4 a)
		{
			return FromCols(-a.X, -a.Y, -a.Z, -a.W);
		}

		/// <summary>IEEE entry-wise equality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(Mat3x4 a, Mat3x4 b)
		{
			return a.X == b.X && a.Y == b.Y && a.Z == b.Z && a.W == b.W;
		}

		/// <summary>IEEE entry-wise inequality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(Mat3x4 a, Mat3x4 b)
		{
			return !(a == b);
		}

		/// <summary>Bit-exact equality, column by column. See the folder header in Vec3.cs.</summary>
		public bool Equals(Mat3x4 other)
		{
			return this.X.Equals(other.X)
				&& this.Y.Equals(other.Y)
				&& this.Z.Equals(other.Z)
				&& this.W.Equals(other.W);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Mat3x4 other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.X, this.Y, this.Z, this.W);
		}
	}
}
