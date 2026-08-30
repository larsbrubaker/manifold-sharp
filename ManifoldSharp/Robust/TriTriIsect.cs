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

// TriTriIsect.cs — the result type of robust/tri_tri.rs's tri_tri_intersect.
// The module header, and everything that produces one of these, is on TriTri.cs.

using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Robust
{
	/// <summary>The four cases of <see cref="TriTriIsect"/>.</summary>
	public enum TriTriIsectKind
	{
		/// <summary>The triangles do not meet.</summary>
		None,

		/// <summary>Single-point contact.</summary>
		Point,

		/// <summary>Contact with positive length.</summary>
		Segment,

		/// <summary>Coplanar triangles overlapping with positive area.</summary>
		Coplanar,
	}

	/// <summary>
	/// Exact intersection of two triangles.
	/// </summary>
	/// <remarks>
	/// The Rust is an enum with payloads (<c>Point(R3)</c>, <c>Segment(R3, R3)</c>,
	/// <c>Coplanar { polygon, same_orientation }</c>). This is a struct with a
	/// <see cref="Kind"/> tag and every payload alongside it, like
	/// <see cref="TriLoc"/> — not the abstract-base-plus-subclasses shape the plan
	/// allows for cold enums-with-data. The deciding measurement is where the
	/// payloads actually appear: <see cref="TriTriIsectKind.None"/> is the dominant
	/// return (both plane rejects, the SAT reject, the coplanar SAT reject and the
	/// empty clip all take it), and it must not allocate; the polygon
	/// <see cref="System.Collections.Generic.List{T}"/> is built at exactly one
	/// site — the <c>_ =&gt; …</c> arm of the coplanar clip, the rarest exit of the
	/// rarest branch — so keeping it as a reference field costs nothing on the hot
	/// path. The struct is five fields wide (a tag, two point references, a list
	/// reference and a bool), so returning it copies about as much as returning a
	/// class reference plus the allocation it avoids.
	/// </remarks>
	public readonly struct TriTriIsect : IEquatable<TriTriIsect>
	{
		/// <summary>Which of the four cases this is.</summary>
		public readonly TriTriIsectKind Kind;

		/// <summary>
		/// The <see cref="TriTriIsectKind.Point"/> point, or the first
		/// <see cref="TriTriIsectKind.Segment"/> endpoint; null otherwise.
		/// </summary>
		public readonly R3? P0;

		/// <summary>
		/// The second <see cref="TriTriIsectKind.Segment"/> endpoint; null otherwise.
		/// </summary>
		public readonly R3? P1;

		/// <summary>
		/// The convex overlap region for <see cref="TriTriIsectKind.Coplanar"/>
		/// (distinct vertices, no three collinear, no guaranteed winding); null
		/// otherwise.
		/// </summary>
		public readonly IReadOnlyList<R3>? Polygon;

		/// <summary>
		/// For <see cref="TriTriIsectKind.Coplanar"/>: true when the two triangles'
		/// normals point the same way, false for opposite planes. Meaningless for the
		/// other cases.
		/// </summary>
		public readonly bool SameOrientation;

		private TriTriIsect(
			TriTriIsectKind kind,
			R3? p0,
			R3? p1,
			IReadOnlyList<R3>? polygon,
			bool sameOrientation)
		{
			this.Kind = kind;
			this.P0 = p0;
			this.P1 = p1;
			this.Polygon = polygon;
			this.SameOrientation = sameOrientation;
		}

		/// <summary>No intersection. This is also <c>default(TriTriIsect)</c>.</summary>
		public static TriTriIsect None
		{
			get { return default; }
		}

		/// <summary>
		/// Single-point contact (vertex-on-face, vertex-on-edge, edge-through-edge, or
		/// interval intersection collapsing to one point).
		/// </summary>
		public static TriTriIsect Point(R3 p)
		{
			return new TriTriIsect(TriTriIsectKind.Point, p, null, null, false);
		}

		/// <summary>
		/// Proper crossing (or edge/vertex contact with positive length).
		/// </summary>
		public static TriTriIsect Segment(R3 p0, R3 p1)
		{
			return new TriTriIsect(TriTriIsectKind.Segment, p0, p1, null, false);
		}

		/// <summary>
		/// Coplanar triangles overlapping with positive area.
		/// </summary>
		public static TriTriIsect Coplanar(IReadOnlyList<R3> polygon, bool sameOrientation)
		{
			return new TriTriIsect(TriTriIsectKind.Coplanar, null, null, polygon, sameOrientation);
		}

		/// <summary>Case and payload equality (Rust's derived <c>PartialEq</c>).</summary>
		public static bool operator ==(TriTriIsect a, TriTriIsect b)
		{
			return a.Equals(b);
		}

		/// <summary>Case or payload inequality.</summary>
		public static bool operator !=(TriTriIsect a, TriTriIsect b)
		{
			return !a.Equals(b);
		}

		/// <summary>
		/// Case and payload equality. The polygon compares element-wise in order, as
		/// the Rust's derived <c>PartialEq</c> on a <c>Vec&lt;R3&gt;</c> does.
		/// </summary>
		public bool Equals(TriTriIsect other)
		{
			if (this.Kind != other.Kind)
			{
				return false;
			}

			switch (this.Kind)
			{
				case TriTriIsectKind.Point:
					return this.P0!.Equals(other.P0);
				case TriTriIsectKind.Segment:
					return this.P0!.Equals(other.P0) && this.P1!.Equals(other.P1);
				case TriTriIsectKind.Coplanar:
					if (this.SameOrientation != other.SameOrientation)
					{
						return false;
					}

					IReadOnlyList<R3> a = this.Polygon!;
					IReadOnlyList<R3> b = other.Polygon!;
					if (a.Count != b.Count)
					{
						return false;
					}

					for (int i = 0; i < a.Count; i++)
					{
						if (!a[i].Equals(b[i]))
						{
							return false;
						}
					}

					return true;
				default:
					return true;
			}
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is TriTriIsect other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			HashCode h = default;
			h.Add((int)this.Kind);
			switch (this.Kind)
			{
				case TriTriIsectKind.Point:
					h.Add(this.P0);
					break;
				case TriTriIsectKind.Segment:
					h.Add(this.P0);
					h.Add(this.P1);
					break;
				case TriTriIsectKind.Coplanar:
					h.Add(this.SameOrientation);
					foreach (R3 p in this.Polygon!)
					{
						h.Add(p);
					}

					break;
				default:
					break;
			}

			return h.ToHashCode();
		}

		/// <summary>The case with its payload, for the ported assertion messages.</summary>
		public override string ToString()
		{
			switch (this.Kind)
			{
				case TriTriIsectKind.Point:
					return "Point(" + this.P0!.ToString() + ")";
				case TriTriIsectKind.Segment:
					return "Segment(" + this.P0!.ToString() + ", " + this.P1!.ToString() + ")";
				case TriTriIsectKind.Coplanar:
					return "Coplanar { polygon: [" + string.Join(", ", this.Polygon!)
						+ "], same_orientation: " + this.SameOrientation.ToString() + " }";
				default:
					return "None";
			}
		}
	}
}
