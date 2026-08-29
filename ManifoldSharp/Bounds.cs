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

// Bounds.cs — port of types_bounds.rs — axis-aligned bounding volumes: Box (3D)
// and Rect (2D).
//
// Ported from include/manifold/common.h. Extracted from types.rs, which
// re-exports both types so external paths (`crate::types::Box`,
// `crate::types::Rect`) are unchanged. The collider (Collider.cs) and the
// broadphase queries in Boolean3.cs are the main consumers of Box; Rect backs
// the 2D cross-section pipeline (CrossSection.cs, Tree2d.cs).
//
// `Box` keeps its Rust name: nothing in the BCL or in this namespace is called
// Box, and C#'s boxing has no type of that name.
//
// Every `.min()` / `.max()` in the Rust is f64::min / f64::max, where a NaN
// operand loses and the other one is returned. They are therefore
// LinalgFunctions.MinF64 / MaxF64 here, never Math.Min / Math.Max, which
// propagate NaN and would let a single NaN coordinate poison a bound that the
// Rust would have kept finite.
//
// ── Two traps these value types set for their C# callers ─────────────────────
//
// 1. Bulk default. The Rust default is the *inverted* box (min +inf, max -inf),
//    which is the identity for union: `vec![BBox::default(); n]` yields n boxes
//    that report IsEmpty() == true and grow correctly on the first UnionPoint.
//    `new Box[n]` yields no such thing — every element is zeroed, so min == max
//    == the origin: a degenerate box whose IsEmpty() answers *false* and which
//    drags any bound it is unioned into back to the origin. Box.Filled and
//    Rect.Filled below are the port of `vec![T::default(); n]`; three Rust sites
//    need them when they are ported:
//      collider.rs:79   `vec![BBox::default(); num_nodes]` — the node bboxes the
//                       BVH refit writes into
//      sort.rs:122      `vec![BBox::default(); num_tri]` in get_face_box_morton,
//                       where faces whose halfedge was removed keep the default
//      sort.rs:164      `face_box.resize(new_num_tri, BBox::default())` — the
//                       grow half of a resize, which Filled does not cover: the
//                       fill value there has to be `new Box()`.
//
// 2. Mutating methods through an indexer. UnionPoint mutates `this`, so it works
//    on a local, on a field, and on an *array element* (`boxes[i].UnionPoint(p)`
//    mutates in place, because an array indexer yields a variable). Through a
//    List<Box> indexer, or any property, it compiles and does nothing: the
//    indexer returns a copy, UnionPoint grows the copy, the copy is dropped.
//    Phase 3 should hold these in arrays — which the arena rule asks for anyway —
//    or read-modify-store explicitly.

using System.Runtime.CompilerServices;

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// A 3D axis-aligned bounding box.
	/// </summary>
	/// <remarks>
	/// <c>new Box()</c> reproduces the Rust <c>Default</c>/<c>Box::new</c>: an inverted,
	/// empty box (min +inf, max -inf) that unions correctly from nothing.
	/// <c>default(Box)</c> and zeroed array elements do <em>not</em> — they are a box at
	/// the origin. Construct these, never default them.
	/// </remarks>
	public struct Box : IEquatable<Box>
	{
		/// <summary>The minimum corner.</summary>
		public Vec3 Min;

		/// <summary>The maximum corner.</summary>
		public Vec3 Max;

		/// <summary>Default is an inverted (empty) box, ready to be unioned into.</summary>
		public Box()
		{
			this.Min = Vec3.Splat(double.PositiveInfinity);
			this.Max = Vec3.Splat(double.NegativeInfinity);
		}

		/// <summary>Creates a box from its two corners, taken as given.</summary>
		/// <param name="min">The minimum corner.</param>
		/// <param name="max">The maximum corner.</param>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Box(Vec3 min, Vec3 max)
		{
			this.Min = min;
			this.Max = max;
		}

		/// <summary>Box containing the two given points.</summary>
		/// <param name="p1">One corner.</param>
		/// <param name="p2">The opposite corner.</param>
		/// <returns>The bounding box of both points.</returns>
		public static Box FromPoints(Vec3 p1, Vec3 p2)
		{
			return new Box(
				new Vec3(MinF64(p1.X, p2.X), MinF64(p1.Y, p2.Y), MinF64(p1.Z, p2.Z)),
				new Vec3(MaxF64(p1.X, p2.X), MaxF64(p1.Y, p2.Y), MaxF64(p1.Z, p2.Z)));
		}

		/// <summary>A box containing a single point.</summary>
		/// <param name="p">The point.</param>
		/// <returns>A degenerate box at <paramref name="p"/>.</returns>
		public static Box FromPoint(Vec3 p)
		{
			return new Box(p, p);
		}

		/// <summary>
		/// An array of <paramref name="n"/> default (inverted, empty) boxes — the port of
		/// the Rust <c>vec![BBox::default(); n]</c>. C#-only: <c>new Box[n]</c> would give
		/// zeroed boxes at the origin instead, which is the trap the file header names.
		/// </summary>
		/// <param name="n">The element count.</param>
		/// <returns>The filled array.</returns>
		internal static Box[] Filled(int n)
		{
			Box[] boxes = new Box[n];
			Box fill = new Box();
			for (int i = 0; i < n; i++)
			{
				boxes[i] = fill;
			}

			return boxes;
		}

		/// <summary>True when the box has no volume (min &gt; max on any axis).</summary>
		/// <returns>True when the box is empty.</returns>
		public bool IsEmpty()
		{
			return this.Min.X > this.Max.X || this.Min.Y > this.Max.Y || this.Min.Z > this.Max.Z;
		}

		/// <summary>The extent of the box along each axis.</summary>
		/// <returns>max - min.</returns>
		public Vec3 Size()
		{
			return this.Max - this.Min;
		}

		/// <summary>The midpoint of the box.</summary>
		/// <returns>(max + min) * 0.5.</returns>
		public Vec3 Center()
		{
			return (this.Max + this.Min) * 0.5;
		}

		/// <summary>Absolute-largest coordinate value.</summary>
		/// <returns>The largest magnitude of any corner coordinate.</returns>
		public double Scale()
		{
			Vec3 absMin = new Vec3(Math.Abs(this.Min.X), Math.Abs(this.Min.Y), Math.Abs(this.Min.Z));
			Vec3 absMax = new Vec3(Math.Abs(this.Max.X), Math.Abs(this.Max.Y), Math.Abs(this.Max.Z));
			Vec3 m = new Vec3(
				MaxF64(absMin.X, absMax.X),
				MaxF64(absMin.Y, absMax.Y),
				MaxF64(absMin.Z, absMax.Z));
			return MaxF64(MaxF64(m.X, m.Y), m.Z);
		}

		/// <summary>Whether the point lies inside or on the box.</summary>
		/// <param name="p">The point to test.</param>
		/// <returns>True when the point is contained.</returns>
		public bool ContainsPoint(Vec3 p)
		{
			return p.X >= this.Min.X
				&& p.X <= this.Max.X
				&& p.Y >= this.Min.Y
				&& p.Y <= this.Max.Y
				&& p.Z >= this.Min.Z
				&& p.Z <= this.Max.Z;
		}

		/// <summary>Whether the other box lies entirely inside or on this one.</summary>
		/// <param name="other">The box to test.</param>
		/// <returns>True when the other box is contained.</returns>
		public bool ContainsBox(Box other)
		{
			return other.Min.X >= this.Min.X
				&& other.Max.X <= this.Max.X
				&& other.Min.Y >= this.Min.Y
				&& other.Max.Y <= this.Max.Y
				&& other.Min.Z >= this.Min.Z
				&& other.Max.Z <= this.Max.Z;
		}

		/// <summary>Expand in-place to include the given point.</summary>
		/// <param name="p">The point to include.</param>
		public void UnionPoint(Vec3 p)
		{
			this.Min.X = MinF64(this.Min.X, p.X);
			this.Min.Y = MinF64(this.Min.Y, p.Y);
			this.Min.Z = MinF64(this.Min.Z, p.Z);
			this.Max.X = MaxF64(this.Max.X, p.X);
			this.Max.Y = MaxF64(this.Max.Y, p.Y);
			this.Max.Z = MaxF64(this.Max.Z, p.Z);
		}

		/// <summary>Return the union of this box with another.</summary>
		/// <param name="other">The box to union with.</param>
		/// <returns>The smallest box containing both.</returns>
		public Box UnionBox(Box other)
		{
			return new Box(
				new Vec3(
					MinF64(this.Min.X, other.Min.X),
					MinF64(this.Min.Y, other.Min.Y),
					MinF64(this.Min.Z, other.Min.Z)),
				new Vec3(
					MaxF64(this.Max.X, other.Max.X),
					MaxF64(this.Max.Y, other.Max.Y),
					MaxF64(this.Max.Z, other.Max.Z)));
		}

		/// <summary>
		/// Transform by axis-aligned affine transform (Mat3x4 * vec4(pt, 1)).
		/// </summary>
		/// <param name="t">The transform.</param>
		/// <returns>The transformed box.</returns>
		public Box Transform(Mat3x4 t)
		{
			Vec3 minT = t * new Vec4(this.Min.X, this.Min.Y, this.Min.Z, 1.0);
			Vec3 maxT = t * new Vec4(this.Max.X, this.Max.Y, this.Max.Z, 1.0);
			return new Box(
				new Vec3(MinF64(minT.X, maxT.X), MinF64(minT.Y, maxT.Y), MinF64(minT.Z, maxT.Z)),
				new Vec3(MaxF64(minT.X, maxT.X), MaxF64(minT.Y, maxT.Y), MaxF64(minT.Z, maxT.Z)));
		}

		/// <summary>Whether the two boxes touch or overlap.</summary>
		/// <param name="other">The box to test against.</param>
		/// <returns>True when they overlap.</returns>
		public bool DoesOverlapBox(Box other)
		{
			return this.Min.X <= other.Max.X
				&& this.Min.Y <= other.Max.Y
				&& this.Min.Z <= other.Max.Z
				&& this.Max.X >= other.Min.X
				&& this.Max.Y >= other.Min.Y
				&& this.Max.Z >= other.Min.Z;
		}

		/// <summary>Does the given point project within the XY extent (including equality)?</summary>
		/// <param name="p">The point to test.</param>
		/// <returns>True when the point's XY projection is inside.</returns>
		public bool DoesOverlapPointXy(Vec3 p)
		{
			return p.X >= this.Min.X && p.X <= this.Max.X && p.Y >= this.Min.Y && p.Y <= this.Max.Y;
		}

		/// <summary>Whether every corner coordinate is finite.</summary>
		/// <returns>True when neither corner is infinite or NaN.</returns>
		public bool IsFinite()
		{
			return double.IsFinite(this.Min.X)
				&& double.IsFinite(this.Min.Y)
				&& double.IsFinite(this.Min.Z)
				&& double.IsFinite(this.Max.X)
				&& double.IsFinite(this.Max.Y)
				&& double.IsFinite(this.Max.Z);
		}

		// C# synthesizes += and *= from these two, so the Rust AddAssign/MulAssign
		// need no port.

		/// <summary>Translate the box.</summary>
		public static Box operator +(Box b, Vec3 shift)
		{
			return new Box(b.Min + shift, b.Max + shift);
		}

		/// <summary>Scale the box component-wise about the origin.</summary>
		public static Box operator *(Box b, Vec3 scale)
		{
			return new Box(b.Min * scale, b.Max * scale);
		}

		/// <summary>IEEE equality of both corners, the Rust's derived <c>PartialEq</c>.</summary>
		public static bool operator ==(Box a, Box b)
		{
			return a.Min == b.Min && a.Max == b.Max;
		}

		/// <summary>IEEE inequality of both corners, the Rust's derived <c>PartialEq</c>.</summary>
		public static bool operator !=(Box a, Box b)
		{
			return !(a == b);
		}

		/// <summary>Bit-exact equality; the partner of <see cref="GetHashCode"/>.</summary>
		public bool Equals(Box other)
		{
			return this.Min.Equals(other.Min) && this.Max.Equals(other.Max);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Box other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.Min, this.Max);
		}
	}

	/// <summary>
	/// A 2D axis-aligned bounding box.
	/// </summary>
	/// <remarks>
	/// <c>new Rect()</c> reproduces the Rust <c>Default</c>/<c>Rect::new</c>: an inverted,
	/// empty rect (min +inf, max -inf). <c>default(Rect)</c> does not — see the same note
	/// on <see cref="Box"/>.
	/// </remarks>
	public struct Rect : IEquatable<Rect>
	{
		/// <summary>The minimum corner.</summary>
		public Vec2 Min;

		/// <summary>The maximum corner.</summary>
		public Vec2 Max;

		/// <summary>Default is an inverted (empty) rect, ready to be unioned into.</summary>
		public Rect()
		{
			this.Min = Vec2.Splat(double.PositiveInfinity);
			this.Max = Vec2.Splat(double.NegativeInfinity);
		}

		/// <summary>Creates a rect from its two corners, taken as given.</summary>
		/// <param name="min">The minimum corner.</param>
		/// <param name="max">The maximum corner.</param>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Rect(Vec2 min, Vec2 max)
		{
			this.Min = min;
			this.Max = max;
		}

		/// <summary>Rect containing the two given points.</summary>
		/// <param name="a">One corner.</param>
		/// <param name="b">The opposite corner.</param>
		/// <returns>The bounding rect of both points.</returns>
		public static Rect FromPoints(Vec2 a, Vec2 b)
		{
			return new Rect(
				new Vec2(MinF64(a.X, b.X), MinF64(a.Y, b.Y)),
				new Vec2(MaxF64(a.X, b.X), MaxF64(a.Y, b.Y)));
		}

		/// <summary>
		/// An array of <paramref name="n"/> default (inverted, empty) rects — the port of
		/// the Rust <c>vec![Rect::default(); n]</c>. C#-only, for the same reason as
		/// <see cref="Box.Filled"/>.
		/// </summary>
		/// <param name="n">The element count.</param>
		/// <returns>The filled array.</returns>
		internal static Rect[] Filled(int n)
		{
			Rect[] rects = new Rect[n];
			Rect fill = new Rect();
			for (int i = 0; i < n; i++)
			{
				rects[i] = fill;
			}

			return rects;
		}

		/// <summary>The extent of the rect along each axis.</summary>
		/// <returns>max - min.</returns>
		public Vec2 Size()
		{
			return this.Max - this.Min;
		}

		/// <summary>The area of the rect.</summary>
		/// <returns>width * height.</returns>
		public double Area()
		{
			Vec2 sz = this.Size();
			return sz.X * sz.Y;
		}

		/// <summary>Absolute-largest coordinate value.</summary>
		/// <returns>The largest magnitude of any corner coordinate.</returns>
		public double Scale()
		{
			Vec2 absMin = new Vec2(Math.Abs(this.Min.X), Math.Abs(this.Min.Y));
			Vec2 absMax = new Vec2(Math.Abs(this.Max.X), Math.Abs(this.Max.Y));
			Vec2 m = new Vec2(MaxF64(absMin.X, absMax.X), MaxF64(absMin.Y, absMax.Y));
			return MaxF64(m.X, m.Y);
		}

		/// <summary>The midpoint of the rect.</summary>
		/// <returns>(max + min) * 0.5.</returns>
		public Vec2 Center()
		{
			return (this.Max + this.Min) * 0.5;
		}

		/// <summary>Whether the point lies inside or on the rect.</summary>
		/// <param name="p">The point to test.</param>
		/// <returns>True when the point is contained.</returns>
		public bool ContainsPoint(Vec2 p)
		{
			return p.X >= this.Min.X && p.X <= this.Max.X && p.Y >= this.Min.Y && p.Y <= this.Max.Y;
		}

		/// <summary>Whether the other rect lies entirely inside or on this one.</summary>
		/// <param name="other">The rect to test.</param>
		/// <returns>True when the other rect is contained.</returns>
		public bool ContainsRect(Rect other)
		{
			return other.Min.X >= this.Min.X
				&& other.Max.X <= this.Max.X
				&& other.Min.Y >= this.Min.Y
				&& other.Max.Y <= this.Max.Y;
		}

		/// <summary>Whether the two rects touch or overlap.</summary>
		/// <param name="other">The rect to test against.</param>
		/// <returns>True when they overlap.</returns>
		public bool DoesOverlap(Rect other)
		{
			return this.Min.X <= other.Max.X
				&& this.Min.Y <= other.Max.Y
				&& this.Max.X >= other.Min.X
				&& this.Max.Y >= other.Min.Y;
		}

		/// <summary>
		/// True when the rect has no area. Note the strict-or-equal comparison: unlike
		/// <see cref="Box.IsEmpty"/>, a zero-width or zero-height rect counts as empty.
		/// </summary>
		/// <returns>True when the rect is empty.</returns>
		public bool IsEmpty()
		{
			return this.Max.Y <= this.Min.Y || this.Max.X <= this.Min.X;
		}

		/// <summary>Whether every corner coordinate is finite.</summary>
		/// <returns>True when neither corner is infinite or NaN.</returns>
		public bool IsFinite()
		{
			return double.IsFinite(this.Min.X)
				&& double.IsFinite(this.Min.Y)
				&& double.IsFinite(this.Max.X)
				&& double.IsFinite(this.Max.Y);
		}

		/// <summary>Expand in-place to include the given point.</summary>
		/// <param name="p">The point to include.</param>
		public void UnionPoint(Vec2 p)
		{
			this.Min.X = MinF64(this.Min.X, p.X);
			this.Min.Y = MinF64(this.Min.Y, p.Y);
			this.Max.X = MaxF64(this.Max.X, p.X);
			this.Max.Y = MaxF64(this.Max.Y, p.Y);
		}

		/// <summary>Return the union of this rect with another.</summary>
		/// <param name="other">The rect to union with.</param>
		/// <returns>The smallest rect containing both.</returns>
		public Rect UnionRect(Rect other)
		{
			return new Rect(
				new Vec2(MinF64(this.Min.X, other.Min.X), MinF64(this.Min.Y, other.Min.Y)),
				new Vec2(MaxF64(this.Max.X, other.Max.X), MaxF64(this.Max.Y, other.Max.Y)));
		}

		/// <summary>Translate the rect.</summary>
		public static Rect operator +(Rect r, Vec2 shift)
		{
			return new Rect(r.Min + shift, r.Max + shift);
		}

		/// <summary>Scale the rect component-wise about the origin.</summary>
		public static Rect operator *(Rect r, Vec2 scale)
		{
			return new Rect(r.Min * scale, r.Max * scale);
		}

		/// <summary>IEEE equality of both corners, the Rust's derived <c>PartialEq</c>.</summary>
		public static bool operator ==(Rect a, Rect b)
		{
			return a.Min == b.Min && a.Max == b.Max;
		}

		/// <summary>IEEE inequality of both corners, the Rust's derived <c>PartialEq</c>.</summary>
		public static bool operator !=(Rect a, Rect b)
		{
			return !(a == b);
		}

		/// <summary>Bit-exact equality; the partner of <see cref="GetHashCode"/>.</summary>
		public bool Equals(Rect other)
		{
			return this.Min.Equals(other.Min) && this.Max.Equals(other.Max);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Rect other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.Min, this.Max);
		}
	}
}
