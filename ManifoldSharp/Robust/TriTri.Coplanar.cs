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

// TriTri.Coplanar.cs — the "Coplanar overlap" section of robust/tri_tri.rs.
// The module header is on TriTri.cs.

using System.Diagnostics;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Robust
{
	/// <content>The coplanar-pair overlap clip.</content>
	public static partial class TriTri
	{
		/// <summary>
		/// Certified separating-edge pre-reject for coplanar pairs, on the raw f64
		/// projection dropping coordinate <paramref name="axis"/>. Sound for ANY
		/// choice of axis: a projection is linear, so a shared 3D point would project
		/// into both projected triangles — strict 2D separation therefore proves 3D
		/// disjointness even when the projection degenerates the triangles. Signs
		/// come from the exact filtered orient2d, so a <c>true</c> answer is certain.
		/// </summary>
		private static bool CoplanarSeparated2d(Vec3[] t1, Vec3[] t2, int axis)
		{
			// Same cyclic drop-axis convention as R3.ProjectDrop.
			Vec2 Proj(Vec3 v)
			{
				switch (axis)
				{
					case 0:
						return new Vec2(v.Y, v.Z);
					case 1:
						return new Vec2(v.Z, v.X);
					default:
						return new Vec2(v.X, v.Y);
				}
			}

			Vec2[] p1 = { Proj(t1[0]), Proj(t1[1]), Proj(t1[2]) };
			Vec2[] p2 = { Proj(t2[0]), Proj(t2[1]), Proj(t2[2]) };
			static bool Separates(Vec2[] tri, Vec2[] other)
			{
				for (int i = 0; i < 3; i++)
				{
					Vec2 a = tri[i];
					Vec2 b = tri[(i + 1) % 3];
					Sign sRef = Filtered.Orient2d(a, b, tri[(i + 2) % 3]);
					if (sRef == Sign.Zero)
					{
						continue;
					}

					bool allOutside = true;
					foreach (Vec2 q in other)
					{
						Sign s = Filtered.Orient2d(a, b, q);
						if (s == Sign.Zero || s == sRef)
						{
							allOutside = false;
							break;
						}
					}

					if (allOutside)
					{
						return true;
					}
				}

				return false;
			}

			return Separates(p1, p2) || Separates(p2, p1);
		}

		/// <summary>
		/// Intersection of two coplanar triangles: Sutherland–Hodgman clip of t2
		/// against t1 in the exact 2D projection, classified by the dimension of the
		/// result (empty / point / segment / convex polygon).
		/// </summary>
		private static TriTriIsect CoplanarOverlap(Vec3[] t1, Vec3[] t2)
		{
			{
				Vec3 n = LinalgFunctions.Cross(t1[1] - t1[0], t1[2] - t1[0]);
				double ax = Math.Abs(n.X);
				double ay = Math.Abs(n.Y);
				double az = Math.Abs(n.Z);
				int axis;
				if (az >= ax && az >= ay)
				{
					axis = 2;
				}
				else if (ay >= ax)
				{
					axis = 1;
				}
				else
				{
					axis = 0;
				}

				if (CoplanarSeparated2d(t1, t2, axis))
				{
					Stats.AddCoplanarSat();
					return TriTriIsect.None;
				}
			}

			Timing.Stopwatch tClip = Timing.Stopwatch.Start();
			TriTriIsect clipOut = CoplanarClip(t1, t2);
			Stats.AddCoplanarClipNs(tClip.ElapsedNs());
			return clipOut;
		}

		/// <summary>
		/// A clip vertex carried with the two auxiliary forms the sign tests want:
		/// the homogenized integer triple (exact fallback) and the correctly rounded
		/// f64 approximation (semi-static filter). Both are pure functions of
		/// <see cref="R"/>, so nothing here changes which points the clip produces —
		/// only how many BigInteger operations decide the signs along the way.
		/// Building them once per vertex replaces the three homogenizations
		/// <c>Orient2dR</c> did on <em>every</em> call.
		/// </summary>
		private sealed class ClipPt
		{
			/// <summary>The exact projected point.</summary>
			public readonly R2 R;

			/// <summary>Its correctly rounded f64 approximation.</summary>
			public readonly Vec2 A;

			// Homogenized lazily: coplanar overlaps are degeneracy-rich, so the f64
			// filter fails often enough to be worth caching, but plenty of vertices
			// never need the exact form at all. Rust's OnceCell; here a nullable
			// field, since the clip is single-threaded.
			private Homog2? h;

			public ClipPt(R2 r)
			{
				this.R = r;
				this.A = new Vec2(Rational.RatToF64(r.X), Rational.RatToF64(r.Y));
			}

			public Homog2 H()
			{
				this.h ??= Predicates.Homog2Of(this.R);
				return this.h.Value;
			}
		}

		/// <summary>
		/// Filtered orient2d over clip vertices: certified f64 sign when the
		/// semi-static bound allows, exact homogeneous sign otherwise. Identical
		/// result to <c>Orient2dR</c> by construction.
		/// </summary>
		private static Sign O2p(ClipPt a, ClipPt b, ClipPt c)
		{
			return Approx.Orient2dA(a.A, b.A, c.A) ?? Predicates.Orient2dH(a.H(), b.H(), c.H());
		}

		/// <summary>
		/// Field-wise equality of two canonical projected points — same answer as
		/// <see cref="R2.Equals(R2)"/> (see the canonicality argument in
		/// Robust/Exact/Rational.cs) without the backend's general
		/// (unreduced-tolerant) comparison.
		/// </summary>
		private static bool ClipPtEq(ClipPt a, ClipPt b)
		{
			return Rational.R2Eq(a.R, b.R);
		}

		/// <summary>
		/// The exact part of <see cref="CoplanarOverlap"/>: Sutherland–Hodgman clip
		/// in the rational 2D projection, once the f64 pre-reject has failed.
		/// </summary>
		private static TriTriIsect CoplanarClip(Vec3[] t1, Vec3[] t2)
		{
			R3[] r1 =
			{
				R3.FromVec3(t1[0]),
				R3.FromVec3(t1[1]),
				R3.FromVec3(t1[2]),
			};
			R3[] r2 =
			{
				R3.FromVec3(t2[0]),
				R3.FromVec3(t2[1]),
				R3.FromVec3(t2[2]),
			};
			R3 n1 = Predicates.TriNormalR(r1[0], r1[1], r1[2]);
			Debug.Assert(!n1.IsZero(), "degenerate input triangle");

			int axis = DominantAxis(n1);
			List<ClipPt> clip = new List<ClipPt>(3);
			foreach (R3 p in r1)
			{
				clip.Add(new ClipPt(p.ProjectDrop(axis)));
			}

			// Normalize the clip triangle to CCW in projection space.
			if (O2p(clip[0], clip[1], clip[2]) == Sign.Neg)
			{
				(clip[1], clip[2]) = (clip[2], clip[1]);
			}

			List<ClipPt> subject = new List<ClipPt>(3);
			foreach (R3 p in r2)
			{
				subject.Add(new ClipPt(p.ProjectDrop(axis)));
			}

			if (O2p(subject[0], subject[1], subject[2]) == Sign.Neg)
			{
				(subject[1], subject[2]) = (subject[2], subject[1]);
			}

			// Clip `subject` against each closed halfplane left of the CCW clip edges.
			List<ClipPt> poly = subject;
			for (int i = 0; i < 3; i++)
			{
				if (poly.Count == 0)
				{
					break;
				}

				ClipPt c0 = clip[i];
				ClipPt c1 = clip[(i + 1) % 3];
				List<ClipPt> outPts = new List<ClipPt>(poly.Count + 2);

				// Each vertex's side is needed twice (as `e` then as `s`); computing
				// it once per vertex halves the sign tests.
				bool[] sides = new bool[poly.Count];
				for (int k = 0; k < poly.Count; k++)
				{
					sides[k] = O2p(c0, c1, poly[k]) != Sign.Neg;
				}

				for (int k = 0; k < poly.Count; k++)
				{
					int kn = (k + 1) % poly.Count;
					ClipPt s = poly[k];
					ClipPt e = poly[kn];
					if (sides[k] && sides[kn])
					{
						// Pass-through vertices are shared, not rebuilt: the cached
						// approximation (and homogenization, if already forced) is
						// valid for the identical point. The Rust clones the struct;
						// ClipPt is a reference type whose only mutable state is that
						// cache, so sharing the reference is the same value with one
						// fewer conversion — and one more cache hit.
						outPts.Add(e);
					}
					else if (sides[k] && !sides[kn])
					{
						outPts.Add(new ClipPt(CrossingPoint(c0, c1, s, e)));
					}
					else if (!sides[k] && sides[kn])
					{
						outPts.Add(new ClipPt(CrossingPoint(c0, c1, s, e)));
						outPts.Add(e);
					}
				}

				poly = outPts;
			}

			// Canonicalize: drop consecutive duplicates (exact equality) and
			// collinear intermediate vertices.
			List<ClipPt> canon = CanonicalPolygon(poly);
			switch (canon.Count)
			{
				case 0:
					return TriTriIsect.None;
				case 1:
					return TriTriIsect.Point(LiftToPlane(canon[0].R, axis, r1[0], n1));
				case 2:
					return TriTriIsect.Segment(
						LiftToPlane(canon[0].R, axis, r1[0], n1),
						LiftToPlane(canon[1].R, axis, r1[0], n1));

				// `same_orientation` costs a rational cross product and a dot, and
				// only the polygon case consumes it — so it is computed here rather
				// than up front, where the (far more common) empty/point/segment
				// exits would pay for it too.
				default:
					R3 n2 = Predicates.TriNormalR(r2[0], r2[1], r2[2]);
					Debug.Assert(!n2.IsZero(), "degenerate input triangle");
					Sign dot = SignFunctions.OfRat(n1.Dot(n2));
					bool sameOrientation;
					if (dot == Sign.Pos)
					{
						sameOrientation = true;
					}
					else if (dot == Sign.Neg)
					{
						sameOrientation = false;
					}
					else
					{
						throw new InvalidOperationException("coplanar triangles have parallel normals");
					}

					List<R3> polygon = new List<R3>(canon.Count);
					foreach (ClipPt p in canon)
					{
						polygon.Add(LiftToPlane(p.R, axis, r1[0], n1));
					}

					return TriTriIsect.Coplanar(polygon, sameOrientation);
			}
		}

		/// <summary>
		/// The Rust's <c>line_line_intersect_2d(…).expect("strictly crossing edge is
		/// not parallel to clip line")</c>, factored out because both crossing arms
		/// of the clip call it identically.
		/// </summary>
		private static R2 CrossingPoint(ClipPt c0, ClipPt c1, ClipPt s, ClipPt e)
		{
			R2? x = Predicates.LineLineIntersect2d(c0.R, c1.R, s.R, e.R);
			if (x is null)
			{
				throw new InvalidOperationException(
					"strictly crossing edge is not parallel to clip line");
			}

			return x;
		}

		/// <summary>
		/// Remove exact duplicates and collinear intermediate vertices from a closed
		/// polygon; a fully collinear result collapses to its two extreme points, a
		/// single repeated point to one point.
		/// </summary>
		private static List<ClipPt> CanonicalPolygon(List<ClipPt> poly)
		{
			// Dedup (cyclic).
			List<ClipPt> pts = new List<ClipPt>(poly.Count);
			foreach (ClipPt p in poly)
			{
				if (pts.Count == 0 || !ClipPtEq(pts[pts.Count - 1], p))
				{
					pts.Add(p);
				}
			}

			while (pts.Count > 1 && ClipPtEq(pts[0], pts[pts.Count - 1]))
			{
				pts.RemoveAt(pts.Count - 1);
			}

			if (pts.Count <= 2)
			{
				return pts;
			}

			// Fully collinear (possible when the overlap is a shared edge segment
			// that SH clipping walked over several vertices): keep the two extremes.
			bool allCollinear = true;
			for (int i = 0; i < pts.Count; i++)
			{
				ClipPt a = pts[i];
				ClipPt b = pts[(i + 1) % pts.Count];
				ClipPt c = pts[(i + 2) % pts.Count];
				if (O2p(a, b, c) != Sign.Zero)
				{
					allCollinear = false;
					break;
				}
			}

			if (allCollinear)
			{
				// Order along the dominant direction of the point spread. Rare exit
				// (a degenerate, zero-area overlap), so it stays fully rational.
				R2? dir = null;
				for (int i = 1; i < pts.Count; i++)
				{
					R2 d = pts[i].R.Sub(pts[0].R);
					if (!d.IsZero())
					{
						dir = d;
						break;
					}
				}

				if (dir is null)
				{
					throw new InvalidOperationException("at least two distinct points");
				}

				R2 spread = dir;
				BigRational Param(R2 p)
				{
					return p.Sub(pts[0].R).Dot(spread);
				}

				int lo = 0;
				int hi = 0;
				BigRational loT = Backend.RatZero();
				BigRational hiT = Backend.RatZero();
				for (int i = 0; i < pts.Count; i++)
				{
					BigRational t = Param(pts[i].R);
					if (t < loT)
					{
						loT = t;
						lo = i;
					}

					if (t > hiT)
					{
						hiT = t;
						hi = i;
					}
				}

				if (ClipPtEq(pts[lo], pts[hi]))
				{
					return new List<ClipPt> { pts[lo] };
				}

				return new List<ClipPt> { pts[lo], pts[hi] };
			}

			// Drop collinear intermediate vertices.
			int n = pts.Count;
			List<ClipPt> keep = new List<ClipPt>(n);
			for (int i = 0; i < n; i++)
			{
				ClipPt prev = pts[(i + n - 1) % n];
				ClipPt next = pts[(i + 1) % n];
				if (O2p(prev, pts[i], next) != Sign.Zero)
				{
					keep.Add(pts[i]);
				}
			}

			return keep;
		}
	}
}
