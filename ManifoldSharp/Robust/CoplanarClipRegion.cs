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

// CoplanarClipRegion.cs — C#-only (no Rust file): graph_geom.rs's
// `clip_segment_to_polygon` and `point_in_polygon_coplanar` with the polygon's
// side of the work done once instead of once per call.
//
// The Rust recomputes the polygon's normal, dominant axis, 2D projection and
// orientation on every segment it clips. Phase 3 of the intersection-graph build
// clips every primitive of a triangle against the same overlap polygon, and a
// coplanar triangle can carry hundreds of primitives across hundreds of regions
// (a 12-triangle cube resting on an 11,652-triangle self-touching body spent
// 6.7 s here), so that setup — all BigRational — was most of the step's cost.
//
// Bit-identity argument (RUST_DIVERGENCES entry 9):
// - Prepare computes exactly the values the Rust computes per call, from the
//   polygon alone, in the same operations and order; Clip and Contains then run
//   the Rust's loops unchanged over them. Exact rationals have no rounding, so
//   computing a value once or N times gives the same value.
// - The bounding-box reject returns "empty" only for a segment (or point) whose
//   projected box misses the polygon's projected box. When the polygon has
//   positive area, a convex CCW polygon is the intersection of its edge
//   half-planes, so the parametric clip's interval contains only parameters whose
//   points lie in the polygon, hence in its box, hence in the segment's box too —
//   disjoint boxes mean the clip's interval is empty (null) and the containment
//   test fails. A zero-area polygon's half-planes intersect in an unbounded line,
//   which a box cannot bound, so the reject is disabled for it.

using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Robust
{
	/// <summary>
	/// A convex coplanar polygon prepared for repeated exact clipping and containment.
	/// </summary>
	internal sealed class CoplanarClipRegion
	{
		private readonly int axis;
		private readonly List<R2> pts2;
		private readonly bool rejectByBox;
		private readonly BigRational minX;
		private readonly BigRational minY;
		private readonly BigRational maxX;
		private readonly BigRational maxY;

		private CoplanarClipRegion(IReadOnlyList<R3> poly)
		{
			R3 n = Predicates.TriNormalR(poly[0], poly[1], poly[2]);
			this.axis = TriTri.DominantAxis(n);
			this.pts2 = new List<R2>(poly.Count);
			for (int i = 0; i < poly.Count; i++)
			{
				this.pts2.Add(poly[i].ProjectDrop(this.axis));
			}

			if (Predicates.Orient2dR(this.pts2[0], this.pts2[1], this.pts2[2]) == Sign.Neg)
			{
				this.pts2.Reverse();
			}

			for (int i = 1; i + 1 < this.pts2.Count && !this.rejectByBox; i++)
			{
				this.rejectByBox = Predicates.Orient2dR(this.pts2[0], this.pts2[i], this.pts2[i + 1]) != Sign.Zero;
			}

			this.minX = this.maxX = this.pts2[0].X;
			this.minY = this.maxY = this.pts2[0].Y;
			foreach (R2 p in this.pts2)
			{
				this.minX = p.X < this.minX ? p.X : this.minX;
				this.maxX = p.X > this.maxX ? p.X : this.maxX;
				this.minY = p.Y < this.minY ? p.Y : this.minY;
				this.maxY = p.Y > this.maxY ? p.Y : this.maxY;
			}
		}

		/// <summary>Prepares a convex coplanar polygon of at least three vertices.</summary>
		/// <param name="poly">The polygon.</param>
		/// <returns>The prepared region.</returns>
		internal static CoplanarClipRegion Prepare(IReadOnlyList<R3> poly)
		{
			return new CoplanarClipRegion(poly);
		}

		/// <summary>
		/// <see cref="GraphGeom.ClipSegmentToPolygon"/> against this region: the positive-
		/// length sub-segment inside it, or null.
		/// </summary>
		/// <param name="a">The segment's start.</param>
		/// <param name="b">The segment's end.</param>
		/// <returns>The clipped sub-segment, or null when the clip is empty.</returns>
		internal (R3 A, R3 B)? Clip(R3 a, R3 b)
		{
			R2 a2 = a.ProjectDrop(this.axis);
			R2 b2 = b.ProjectDrop(this.axis);
			if (this.MissesBox(a2, b2))
			{
				return null;
			}

			R2 dir = b2.Sub(a2);

			// Parametric clip of [0,1] against each CCW edge halfplane.
			BigRational t0 = Backend.RatZero();
			BigRational t1 = Backend.RatOne();
			for (int i = 0; i < this.pts2.Count; i++)
			{
				R2 e0 = this.pts2[i];
				R2 e1 = this.pts2[(i + 1) % this.pts2.Count];
				R2 edge = e1.Sub(e0);

				// Signed distance numerators of a2 + t*dir against the edge line:
				// f(t) = cross(edge, a2 + t*dir - e0) = fa + t * fd.
				BigRational fa = edge.Cross(a2.Sub(e0));
				BigRational fd = edge.Cross(dir);
				if (Backend.RatIsZero(fd))
				{
					if (fa < Backend.RatZero())
					{
						return null; // parallel and strictly outside
					}

					continue;
				}

				BigRational tHit = -fa / fd;
				if (fd > Backend.RatZero())
				{
					// entering: f grows with t → require t >= t_hit
					if (tHit > t0)
					{
						t0 = tHit;
					}
				}
				else if (tHit < t1)
				{
					t1 = tHit;
				}

				if (t0 >= t1)
				{
					return null;
				}
			}

			if (t0 >= t1)
			{
				return null;
			}

			return (Seg(t0), Seg(t1));

			R3 Seg(in BigRational t)
			{
				return a.Add(b.Sub(a).Scale(t));
			}
		}

		/// <summary>
		/// <see cref="GraphGeom.PointInPolygonCoplanar"/> against this region: true when the
		/// point, assumed on the polygon's plane, is inside or on it.
		/// </summary>
		/// <param name="p">The point.</param>
		/// <returns>True when the point is inside or on the polygon.</returns>
		internal bool Contains(R3 p)
		{
			R2 p2 = p.ProjectDrop(this.axis);
			if (this.MissesBox(p2, p2))
			{
				return false;
			}

			for (int i = 0; i < this.pts2.Count; i++)
			{
				if (Predicates.Orient2dR(this.pts2[i], this.pts2[(i + 1) % this.pts2.Count], p2) == Sign.Neg)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// True when the projected segment's box is disjoint from the polygon's box, which
		/// (see the file header) proves the segment misses a positive-area polygon.
		/// </summary>
		private bool MissesBox(R2 a2, R2 b2)
		{
			if (!this.rejectByBox)
			{
				return false;
			}

			return (a2.X < this.minX && b2.X < this.minX)
				|| (a2.X > this.maxX && b2.X > this.maxX)
				|| (a2.Y < this.minY && b2.Y < this.minY)
				|| (a2.Y > this.maxY && b2.Y > this.maxY);
		}
	}
}
