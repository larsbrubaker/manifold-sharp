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

// Rational.Points.cs — the R2/R3 exact point types and their hash keys, from
// robust/exact/rational.rs. The module header is on Rational.cs.
//
// R2/R3 are reference types, not structs: the Rust passes them everywhere as
// `&R2`/`&R3` and returns `Option<R3>` from the constructions, and a struct
// holding two or three BigRationals (four to six BigIntegers) would be copied by
// value at every one of those call sites. They are immutable, so sharing a
// reference is exactly what `&` meant.

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Robust.Exact
{
	/// <summary>
	/// Exact 2D point (or vector). <see cref="CompareTo"/> is lexicographic, which
	/// the arrangement code uses for exact point dedup in ordered maps.
	/// </summary>
	public sealed class R2 : IEquatable<R2>, IComparable<R2>
	{
		/// <summary>The exact x coordinate.</summary>
		public readonly BigRational X;

		/// <summary>The exact y coordinate.</summary>
		public readonly BigRational Y;

		/// <summary>Creates a point from two canonical rationals.</summary>
		public R2(BigRational x, BigRational y)
		{
			this.X = x;
			this.Y = y;
		}

		/// <summary>Exact conversion of a finite f64 point.</summary>
		public static R2 FromVec2(Vec2 v)
		{
			return new R2(Rational.Rat(v.X), Rational.Rat(v.Y));
		}

		/// <summary>Both coordinates correctly rounded back to f64.</summary>
		public Vec2 ToVec2Rounded()
		{
			return new Vec2(Rational.RatToF64(this.X), Rational.RatToF64(this.Y));
		}

		/// <summary>Exact difference.</summary>
		public R2 Sub(R2 o)
		{
			return new R2(this.X - o.X, this.Y - o.Y);
		}

		/// <summary>Exact sum.</summary>
		public R2 Add(R2 o)
		{
			return new R2(this.X + o.X, this.Y + o.Y);
		}

		/// <summary>Exact scalar multiple.</summary>
		public R2 Scale(in BigRational s)
		{
			return new R2(this.X * s, this.Y * s);
		}

		/// <summary>Exact dot product.</summary>
		public BigRational Dot(R2 o)
		{
			return (this.X * o.X) + (this.Y * o.Y);
		}

		/// <summary>
		/// 2D cross product (z of the 3D cross of the embedded vectors).
		/// </summary>
		public BigRational Cross(R2 o)
		{
			return (this.X * o.Y) - (this.Y * o.X);
		}

		/// <summary>True when both coordinates are exactly zero.</summary>
		public bool IsZero()
		{
			return Backend.RatIsZero(this.X) && Backend.RatIsZero(this.Y);
		}

		/// <summary>Exact coordinate-wise equality.</summary>
		public bool Equals(R2? other)
		{
			return other != null && this.X.Equals(other.X) && this.Y.Equals(other.Y);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return this.Equals(obj as R2);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.X.GetHashCode(), this.Y.GetHashCode());
		}

		/// <summary>Lexicographic (x, then y) ordering.</summary>
		public int CompareTo(R2? other)
		{
			if (other == null)
			{
				return 1;
			}

			int c = this.X.CompareTo(other.X);
			return c != 0 ? c : this.Y.CompareTo(other.Y);
		}

		/// <summary>The pair of canonical fractions, for debugging.</summary>
		public override string ToString()
		{
			return "(" + this.X.ToString() + ", " + this.Y.ToString() + ")";
		}
	}

	/// <summary>
	/// Exact 3D point (or vector). <see cref="CompareTo"/> is lexicographic
	/// (x, y, z) for exact vertex welding in output assembly.
	/// </summary>
	public sealed class R3 : IEquatable<R3>, IComparable<R3>
	{
		/// <summary>The exact x coordinate.</summary>
		public readonly BigRational X;

		/// <summary>The exact y coordinate.</summary>
		public readonly BigRational Y;

		/// <summary>The exact z coordinate.</summary>
		public readonly BigRational Z;

		/// <summary>Creates a point from three canonical rationals.</summary>
		public R3(BigRational x, BigRational y, BigRational z)
		{
			this.X = x;
			this.Y = y;
			this.Z = z;
		}

		/// <summary>Exact conversion of a finite f64 point.</summary>
		public static R3 FromVec3(Vec3 v)
		{
			return new R3(Rational.Rat(v.X), Rational.Rat(v.Y), Rational.Rat(v.Z));
		}

		/// <summary>All three coordinates correctly rounded back to f64.</summary>
		public Vec3 ToVec3Rounded()
		{
			return new Vec3(
				Rational.RatToF64(this.X),
				Rational.RatToF64(this.Y),
				Rational.RatToF64(this.Z));
		}

		/// <summary>Exact difference.</summary>
		public R3 Sub(R3 o)
		{
			return new R3(this.X - o.X, this.Y - o.Y, this.Z - o.Z);
		}

		/// <summary>Exact sum.</summary>
		public R3 Add(R3 o)
		{
			return new R3(this.X + o.X, this.Y + o.Y, this.Z + o.Z);
		}

		/// <summary>Exact scalar multiple.</summary>
		public R3 Scale(in BigRational s)
		{
			return new R3(this.X * s, this.Y * s, this.Z * s);
		}

		/// <summary>Exact dot product.</summary>
		public BigRational Dot(R3 o)
		{
			return (this.X * o.X) + (this.Y * o.Y) + (this.Z * o.Z);
		}

		/// <summary>Exact cross product.</summary>
		public R3 Cross(R3 o)
		{
			return new R3(
				(this.Y * o.Z) - (this.Z * o.Y),
				(this.Z * o.X) - (this.X * o.Z),
				(this.X * o.Y) - (this.Y * o.X));
		}

		/// <summary>True when all three coordinates are exactly zero.</summary>
		public bool IsZero()
		{
			return Backend.RatIsZero(this.X) && Backend.RatIsZero(this.Y) && Backend.RatIsZero(this.Z);
		}

		/// <summary>
		/// Drops the coordinate at <paramref name="axis"/> (0=x, 1=y, 2=z), keeping the
		/// other two in cyclic order — the paper's bijective dominant-axis projection
		/// used to embed per-triangle 2D arrangements.
		/// </summary>
		/// <exception cref="ArgumentOutOfRangeException">The axis is not 0, 1 or 2.</exception>
		public R2 ProjectDrop(int axis)
		{
			switch (axis)
			{
				case 0:
					return new R2(this.Y, this.Z);
				case 1:
					return new R2(this.Z, this.X);
				case 2:
					return new R2(this.X, this.Y);
				default:
					throw new ArgumentOutOfRangeException(nameof(axis), "axis must be 0, 1, or 2");
			}
		}

		/// <summary>Exact coordinate-wise equality.</summary>
		public bool Equals(R3? other)
		{
			return other != null
				&& this.X.Equals(other.X)
				&& this.Y.Equals(other.Y)
				&& this.Z.Equals(other.Z);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return this.Equals(obj as R3);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.X.GetHashCode(), this.Y.GetHashCode(), this.Z.GetHashCode());
		}

		/// <summary>Lexicographic (x, then y, then z) ordering.</summary>
		public int CompareTo(R3? other)
		{
			if (other == null)
			{
				return 1;
			}

			int c = this.X.CompareTo(other.X);
			if (c != 0)
			{
				return c;
			}

			c = this.Y.CompareTo(other.Y);
			return c != 0 ? c : this.Z.CompareTo(other.Z);
		}

		/// <summary>The three canonical fractions, for debugging.</summary>
		public override string ToString()
		{
			return "(" + this.X.ToString() + ", " + this.Y.ToString() + ", " + this.Z.ToString() + ")";
		}
	}

	// ─── Cheap exact hash keys ───────────────────────────────────────────────────
	//
	// A general-purpose rational Hash/Eq must stay consistent for UNREDUCED ratios,
	// which costs at least a cross-multiplication (and, in some libraries, a
	// Euclidean recursion). Every rational this pipeline stores is canonical — built
	// by RatNew, the arithmetic operators, or RatFromF64, all of which reduce — so
	// field identity IS value identity, and hashing the raw sign/byte data is both
	// exact and division-free. The wrappers below are drop-in hash-map keys. See
	// Backend.cs items (1) and (2).
	//
	// One port note: in the Rust these wrappers are also a *performance* seam,
	// because R2's derived Hash goes through the backend's own (more expensive)
	// Hash impl while R2Key's goes through hash_rational. Here BigRational has only
	// one hash — hash_rational's — so the wrappers are a naming and documentation
	// seam that keeps the Phase 10 call sites transcribable, not a second algorithm.

	// ── The empty key, which the Rust does not have ─────────────────────────────
	//
	// `R2Key(pub R2)` is a Rust tuple struct: it has no default form, so the only way
	// to hold one is to have built it around a point. Every C# struct has a
	// default(T), and R2/R3 are reference types, so `default(R2Key)` — a
	// `new R2Key[n]` awaiting fill, a `TryGetValue` miss's `out` value, a field on a
	// not-yet-initialized struct — carries a null Point. Equals and GetHashCode
	// therefore guard it and treat it as a value distinct from every real key rather
	// than throwing NullReferenceException at a dictionary probe: a Phase 10 caller
	// that probes with an empty key should get a miss, and one that reads `.Point`
	// off it should get the NRE at its own bug site, not inside a hash bucket.

	/// <summary>Hash-map key wrapper around a canonical <see cref="R2"/>.</summary>
	public readonly struct R2Key : IEquatable<R2Key>
	{
		/// <summary>
		/// The wrapped point. Non-null for every key built through the constructor;
		/// null only for <c>default(R2Key)</c>, which is not a key for any point.
		/// </summary>
		public readonly R2 Point;

		/// <summary>Wraps a canonical point as a hash key.</summary>
		public R2Key(R2 point)
		{
			this.Point = point;
		}

		/// <summary>
		/// Field-wise equality of the two canonical points. Two empty keys are equal;
		/// an empty key equals no real one.
		/// </summary>
		public bool Equals(R2Key other)
		{
			if (this.Point is null || other.Point is null)
			{
				return this.Point is null && other.Point is null;
			}

			return Rational.R2Eq(this.Point, other.Point);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is R2Key other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			if (this.Point is null)
			{
				return 0;
			}

			return HashCode.Combine(
				Backend.HashRational(this.Point.X),
				Backend.HashRational(this.Point.Y));
		}
	}

	/// <summary>Hash-map key wrapper around a canonical <see cref="R3"/>.</summary>
	public readonly struct R3Key : IEquatable<R3Key>
	{
		/// <summary>
		/// The wrapped point. Non-null for every key built through the constructor;
		/// null only for <c>default(R3Key)</c>, which is not a key for any point.
		/// </summary>
		public readonly R3 Point;

		/// <summary>Wraps a canonical point as a hash key.</summary>
		public R3Key(R3 point)
		{
			this.Point = point;
		}

		/// <summary>
		/// Field-wise equality of the two canonical points. Two empty keys are equal;
		/// an empty key equals no real one.
		/// </summary>
		public bool Equals(R3Key other)
		{
			if (this.Point is null || other.Point is null)
			{
				return this.Point is null && other.Point is null;
			}

			return Rational.R3Eq(this.Point, other.Point);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is R3Key other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			if (this.Point is null)
			{
				return 0;
			}

			return HashCode.Combine(
				Backend.HashRational(this.Point.X),
				Backend.HashRational(this.Point.Y),
				Backend.HashRational(this.Point.Z));
		}
	}
}
