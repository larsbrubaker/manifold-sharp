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

// Ports the Mat4x3 / Mat3x2 / Mat2x3 sections of linalg.rs — the three
// rectangular matrices that exist only to be multiplied by a vector. The Rust
// gives them no indexer, no scalar multiply and no addition, and neither does
// this port; each carries exactly the one product the Rust defines. The module
// header for the whole Linalg folder lives in Vec3.cs.

using System.Runtime.CompilerServices;

namespace ManifoldSharp.Linalg
{
	/// <summary>
	/// <c>la::mat&lt;double, 4, 3&gt;</c> — 4 rows, 3 cols, stored as 3 columns of <see cref="Vec4"/>.
	/// </summary>
	public struct Mat4x3 : IEquatable<Mat4x3>
	{
		/// <summary>Column 0.</summary>
		public Vec4 X;

		/// <summary>Column 1.</summary>
		public Vec4 Y;

		/// <summary>Column 2.</summary>
		public Vec4 Z;

		/// <summary>Creates a matrix from its three columns.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat4x3 FromCols(Vec4 x, Vec4 y, Vec4 z)
		{
			return new Mat4x3 { X = x, Y = y, Z = z };
		}

		/// <summary>Row <paramref name="i"/>, gathered across the columns.</summary>
		public Vec3 Row(int i)
		{
			return new Vec3(this.X[i], this.Y[i], this.Z[i]);
		}

		/// <summary>Matrix-vector product: mat4x3 * vec3 -&gt; vec4.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec4 operator *(Mat4x3 a, Vec3 b)
		{
			return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
		}

		/// <summary>IEEE entry-wise equality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(Mat4x3 a, Mat4x3 b)
		{
			return a.X == b.X && a.Y == b.Y && a.Z == b.Z;
		}

		/// <summary>IEEE entry-wise inequality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(Mat4x3 a, Mat4x3 b)
		{
			return !(a == b);
		}

		/// <summary>Bit-exact equality, column by column. See the folder header in Vec3.cs.</summary>
		public bool Equals(Mat4x3 other)
		{
			return this.X.Equals(other.X) && this.Y.Equals(other.Y) && this.Z.Equals(other.Z);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Mat4x3 other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.X, this.Y, this.Z);
		}
	}

	/// <summary>
	/// <c>la::mat&lt;double, 3, 2&gt;</c> — 3 rows, 2 cols, stored as 2 columns of <see cref="Vec3"/>.
	/// </summary>
	public struct Mat3x2 : IEquatable<Mat3x2>
	{
		/// <summary>Column 0.</summary>
		public Vec3 X;

		/// <summary>Column 1.</summary>
		public Vec3 Y;

		/// <summary>Creates a matrix from its two columns.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat3x2 FromCols(Vec3 x, Vec3 y)
		{
			return new Mat3x2 { X = x, Y = y };
		}

		/// <summary>Row <paramref name="i"/>, gathered across the columns.</summary>
		public Vec2 Row(int i)
		{
			return new Vec2(this.X[i], this.Y[i]);
		}

		/// <summary>Matrix-vector product: mat3x2 * vec2 -&gt; vec3.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 operator *(Mat3x2 a, Vec2 b)
		{
			return a.X * b.X + a.Y * b.Y;
		}

		/// <summary>IEEE entry-wise equality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(Mat3x2 a, Mat3x2 b)
		{
			return a.X == b.X && a.Y == b.Y;
		}

		/// <summary>IEEE entry-wise inequality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(Mat3x2 a, Mat3x2 b)
		{
			return !(a == b);
		}

		/// <summary>Bit-exact equality, column by column. See the folder header in Vec3.cs.</summary>
		public bool Equals(Mat3x2 other)
		{
			return this.X.Equals(other.X) && this.Y.Equals(other.Y);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Mat3x2 other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.X, this.Y);
		}
	}

	/// <summary>
	/// <c>la::mat&lt;double, 2, 3&gt;</c> — 2 rows, 3 cols, stored as 3 columns of <see cref="Vec2"/>.
	/// </summary>
	public struct Mat2x3 : IEquatable<Mat2x3>
	{
		/// <summary>Column 0.</summary>
		public Vec2 X;

		/// <summary>Column 1.</summary>
		public Vec2 Y;

		/// <summary>Column 2.</summary>
		public Vec2 Z;

		/// <summary>Creates a matrix from its three columns.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat2x3 FromCols(Vec2 x, Vec2 y, Vec2 z)
		{
			return new Mat2x3 { X = x, Y = y, Z = z };
		}

		/// <summary>Matrix-vector product: mat2x3 * vec3 -&gt; vec2.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec2 operator *(Mat2x3 a, Vec3 b)
		{
			return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
		}

		/// <summary>IEEE entry-wise equality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(Mat2x3 a, Mat2x3 b)
		{
			return a.X == b.X && a.Y == b.Y && a.Z == b.Z;
		}

		/// <summary>IEEE entry-wise inequality, the Rust's derived <c>PartialEq</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(Mat2x3 a, Mat2x3 b)
		{
			return !(a == b);
		}

		/// <summary>Bit-exact equality, column by column. See the folder header in Vec3.cs.</summary>
		public bool Equals(Mat2x3 other)
		{
			return this.X.Equals(other.X) && this.Y.Equals(other.Y) && this.Z.Equals(other.Z);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Mat2x3 other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.X, this.Y, this.Z);
		}
	}
}
