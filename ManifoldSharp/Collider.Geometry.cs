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

// Collider.Geometry.cs — the second half of the collider.rs port: the narrow
// phase (edge/edge distance, triangle/triangle distance, ray/triangle
// intersection) and the two ManifoldImpl methods the Rust file carries in its
// `impl ManifoldImpl` block (is_self_intersecting, min_gap). The BVH itself and
// the module header are in Collider.cs; this split exists only for the 800-line
// cap.
//
// ── `[Vec3; 3]` is a value, `Vec3[]` is a reference ──────────────────────────
// The Rust passes triangles as `[Vec3; 3]`, which is Copy: `let mut tmp = tri;`
// makes an independent copy. The C# equivalent shape is `Vec3[]`, which is not —
// the same line would alias, and the loops below that write `tmp[i] = tri[i] +
// n * ep` would then be reading values they had already overwritten. Every one
// of those Rust copies is spelled `(Vec3[])x.Clone()` here, and that is not
// optional.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// The free functions of <c>collider.rs</c> — the narrow-phase primitives the
	/// broadphase feeds. Named with the <c>Functions</c> suffix because the bare module
	/// name is taken by <see cref="Collider"/> itself.
	/// </summary>
	public static class ColliderFunctions
	{
		/// <summary>
		/// The closest pair of points on two segments, given as origin plus direction.
		/// </summary>
		/// <param name="p">Origin of the first segment.</param>
		/// <param name="a">Direction (and length) of the first segment.</param>
		/// <param name="q">Origin of the second segment.</param>
		/// <param name="b">Direction (and length) of the second segment.</param>
		/// <returns>The closest point on each segment.</returns>
		public static (Vec3 OnFirst, Vec3 OnSecond) EdgeEdgeDist(Vec3 p, Vec3 a, Vec3 q, Vec3 b)
		{
			Vec3 tVec = q - p;
			double aDotA = Dot(a, a);
			double bDotB = Dot(b, b);
			double aDotB = Dot(a, b);
			double aDotT = Dot(a, tVec);
			double bDotT = Dot(b, tVec);

			double denom = (aDotA * bDotB) - (aDotB * aDotB);
			double t = denom != 0.0
				? ClampF64(((aDotT * bDotB) - (bDotT * aDotB)) / denom, 0.0, 1.0)
				: 0.0;

			double u;
			if (bDotB != 0.0)
			{
				double u0 = ((t * aDotB) - bDotT) / bDotB;
				if (u0 < 0.0)
				{
					t = aDotA != 0.0 ? ClampF64(aDotT / aDotA, 0.0, 1.0) : 0.0;
					u = 0.0;
				}
				else if (u0 > 1.0)
				{
					t = aDotA != 0.0 ? ClampF64((aDotB + aDotT) / aDotA, 0.0, 1.0) : 0.0;
					u = 1.0;
				}
				else
				{
					u = u0;
				}
			}
			else
			{
				t = aDotA != 0.0 ? ClampF64(aDotT / aDotA, 0.0, 1.0) : 0.0;
				u = 0.0;
			}

			return (p + (a * t), q + (b * u));
		}

		/// <summary>
		/// The squared distance between two triangles; exactly 0 when they touch or
		/// intersect.
		/// </summary>
		/// <param name="p">The first triangle's three vertices.</param>
		/// <param name="q">The second triangle's three vertices.</param>
		/// <returns>The squared distance.</returns>
		public static double DistanceTriangleTriangleSquared(Vec3[] p, Vec3[] q)
		{
			Vec3[] sv = new Vec3[] { p[1] - p[0], p[2] - p[1], p[0] - p[2] };
			Vec3[] tv = new Vec3[] { q[1] - q[0], q[2] - q[1], q[0] - q[2] };

			bool shownDisjoint = false;
			double mindd = double.MaxValue;

			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					(Vec3 cp, Vec3 cq) = EdgeEdgeDist(p[i], sv[i], q[j], tv[j]);
					Vec3 v = cq - cp;
					double dd = Dot(v, v);

					if (dd <= mindd)
					{
						mindd = dd;

						int id = i + 2;
						if (id >= 3)
						{
							id -= 3;
						}

						Vec3 z = p[id] - cp;
						double a = Dot(z, v);

						id = j + 2;
						if (id >= 3)
						{
							id -= 3;
						}

						Vec3 z2 = q[id] - cq;
						double b = Dot(z2, v);

						if (a <= 0.0 && b >= 0.0)
						{
							return Dot(v, v);
						}

						if (a <= 0.0)
						{
							a = 0.0;
						}
						else if (b > 0.0)
						{
							b = 0.0;
						}

						if (mindd - a + b > 0.0)
						{
							shownDisjoint = true;
						}
					}
				}
			}

			Vec3 sn = Cross(sv[0], sv[1]);
			double snl = Dot(sn, sn);
			if (snl > 1e-15)
			{
				Vec3 tp = new Vec3(Dot(p[0] - q[0], sn), Dot(p[0] - q[1], sn), Dot(p[0] - q[2], sn));
				int? index = null;
				if (tp.X > 0.0 && tp.Y > 0.0 && tp.Z > 0.0)
				{
					int idx = tp.X < tp.Y ? 0 : 1;
					if (tp.Z < tp[idx])
					{
						idx = 2;
					}

					index = idx;
				}
				else if (tp.X < 0.0 && tp.Y < 0.0 && tp.Z < 0.0)
				{
					int idx = tp.X > tp.Y ? 0 : 1;
					if (tp.Z > tp[idx])
					{
						idx = 2;
					}

					index = idx;
				}

				if (index.HasValue)
				{
					int indexValue = index.Value;
					shownDisjoint = true;
					Vec3 qIndex = q[indexValue];
					Vec3 v = qIndex - p[0];
					Vec3 z = Cross(sn, sv[0]);
					if (Dot(v, z) > 0.0)
					{
						v = qIndex - p[1];
						z = Cross(sn, sv[1]);
						if (Dot(v, z) > 0.0)
						{
							v = qIndex - p[2];
							z = Cross(sn, sv[2]);
							if (Dot(v, z) > 0.0)
							{
								Vec3 cp = qIndex + (sn * (tp[indexValue] / snl));
								Vec3 cq = qIndex;
								return Dot(cp - cq, cp - cq);
							}
						}
					}
				}
			}

			Vec3 tn = Cross(tv[0], tv[1]);
			double tnl = Dot(tn, tn);
			if (tnl > 1e-15)
			{
				Vec3 sp = new Vec3(Dot(q[0] - p[0], tn), Dot(q[0] - p[1], tn), Dot(q[0] - p[2], tn));
				int? index = null;
				if (sp.X > 0.0 && sp.Y > 0.0 && sp.Z > 0.0)
				{
					int idx = sp.X < sp.Y ? 0 : 1;
					if (sp.Z < sp[idx])
					{
						idx = 2;
					}

					index = idx;
				}
				else if (sp.X < 0.0 && sp.Y < 0.0 && sp.Z < 0.0)
				{
					int idx = sp.X > sp.Y ? 0 : 1;
					if (sp.Z > sp[idx])
					{
						idx = 2;
					}

					index = idx;
				}

				if (index.HasValue)
				{
					int indexValue = index.Value;
					shownDisjoint = true;
					Vec3 pIndex = p[indexValue];
					Vec3 v = pIndex - q[0];
					Vec3 z = Cross(tn, tv[0]);
					if (Dot(v, z) > 0.0)
					{
						v = pIndex - q[1];
						z = Cross(tn, tv[1]);
						if (Dot(v, z) > 0.0)
						{
							v = pIndex - q[2];
							z = Cross(tn, tv[2]);
							if (Dot(v, z) > 0.0)
							{
								Vec3 cp = pIndex;
								Vec3 cq = pIndex + (tn * (sp[indexValue] / tnl));
								return Dot(cp - cq, cp - cq);
							}
						}
					}
				}
			}

			return shownDisjoint ? mindd : 0.0;
		}

		/// <summary>
		/// Möller-Trumbore ray/triangle intersection.
		/// </summary>
		/// <param name="origin">The ray origin.</param>
		/// <param name="direction">The ray direction.</param>
		/// <param name="tri">The triangle's three vertices.</param>
		/// <returns>The parametric distance along the ray, or null when there is no hit.</returns>
		public static double? RayTriangleIntersection(Vec3 origin, Vec3 direction, Vec3[] tri)
		{
			const double Eps = 1e-9;
			Vec3 edge1 = tri[1] - tri[0];
			Vec3 edge2 = tri[2] - tri[0];
			Vec3 h = Cross(direction, edge2);
			double a = Dot(edge1, h);
			if (Math.Abs(a) < Eps)
			{
				return null;
			}

			double f = 1.0 / a;
			Vec3 s = origin - tri[0];
			double u = f * Dot(s, h);
			if (!(u >= 0.0 && u <= 1.0))
			{
				return null;
			}

			Vec3 q = Cross(s, edge1);
			double v = f * Dot(direction, q);
			if (v < 0.0 || u + v > 1.0)
			{
				return null;
			}

			double t = f * Dot(edge2, q);
			if (t > Eps)
			{
				return t;
			}

			return null;
		}

		/// <summary>
		/// Rust <c>f64::clamp</c>, transcribed: a pair of comparisons that leaves NaN as
		/// NaN. <see cref="LinalgFunctions.Clamp(double, double, double)"/> is the linalg
		/// <c>clamp_s</c> (max-then-min), which turns a NaN into <c>lo</c>, so it is
		/// <b>not</b> a substitute at these sites.
		/// </summary>
		private static double ClampF64(double value, double lo, double hi)
		{
			if (value < lo)
			{
				return lo;
			}

			if (value > hi)
			{
				return hi;
			}

			return value;
		}
	}

	/// <content>
	/// The <c>impl ManifoldImpl</c> block of collider.rs: the two queries that run against
	/// the impl's cached face BVH.
	/// </content>
	public sealed partial class ManifoldImpl
	{
		/// <summary>
		/// True when any two of this mesh's triangles genuinely intersect, rather than
		/// merely sharing an edge or vertex as every closed mesh does.
		/// </summary>
		/// <returns>True when the mesh self-intersects.</returns>
		public bool IsSelfIntersecting()
		{
			double ep = 2.0 * this.Epsilon;
			double epsilonSq = ep * ep;

			// Fresh face boxes for the queries; the tree itself is the cached
			// collider (C++ SelfIntersecting queries collider_ the same way).
			(Box[] faceBox, uint[] faceMorton) = Sort.GetFaceBoxMorton(this);
			_ = faceMorton;
			Collider collider = this.Collider;
			bool intersecting = false;

			collider.CollisionsWithBoxes(faceBox, true, (tri0, tri1) =>
			{
				if (intersecting)
				{
					return;
				}

				Vec3[] triVerts0 = this.FaceTriangleVertices(tri0);
				Vec3[] triVerts1 = this.FaceTriangleVertices(tri1);

				foreach (Vec3 a in triVerts0)
				{
					foreach (Vec3 b in triVerts1)
					{
						if (DistanceSquared(a, b) <= epsilonSq)
						{
							return;
						}
					}
				}

				if (ColliderFunctions.DistanceTriangleTriangleSquared(triVerts0, triVerts1) == 0.0)
				{
					// `let mut tmp0 = tri_verts0` copies the Rust's [Vec3; 3] value; the C#
					// array has to be cloned or the writes below would alias the source.
					Vec3[] tmp0 = (Vec3[])triVerts0.Clone();
					Vec3[] tmp1 = (Vec3[])triVerts1.Clone();
					for (int i = 0; i < 3; i++)
					{
						tmp0[i] = triVerts0[i] + (this.FaceNormal[tri1] * ep);
					}

					if (ColliderFunctions.DistanceTriangleTriangleSquared(tmp0, triVerts1) > 0.0)
					{
						return;
					}

					for (int i = 0; i < 3; i++)
					{
						tmp0[i] = triVerts0[i] - (this.FaceNormal[tri1] * ep);
					}

					if (ColliderFunctions.DistanceTriangleTriangleSquared(tmp0, triVerts1) > 0.0)
					{
						return;
					}

					for (int i = 0; i < 3; i++)
					{
						tmp1[i] = triVerts1[i] + (this.FaceNormal[tri0] * ep);
					}

					if (ColliderFunctions.DistanceTriangleTriangleSquared(triVerts0, tmp1) > 0.0)
					{
						return;
					}

					for (int i = 0; i < 3; i++)
					{
						tmp1[i] = triVerts1[i] - (this.FaceNormal[tri0] * ep);
					}

					if (ColliderFunctions.DistanceTriangleTriangleSquared(triVerts0, tmp1) > 0.0)
					{
						return;
					}

					intersecting = true;
				}
			});

			return intersecting;
		}

		/// <summary>
		/// The smallest distance between this mesh and <paramref name="other"/>, capped at
		/// <paramref name="searchLength"/>.
		/// </summary>
		/// <param name="other">The mesh to measure against.</param>
		/// <param name="searchLength">The largest gap worth reporting.</param>
		/// <returns>The gap, never larger than <paramref name="searchLength"/>.</returns>
		public double MinGap(ManifoldImpl other, double searchLength)
		{
			(Box[] otherBox, uint[] otherMorton) = Sort.GetFaceBoxMorton(other);
			_ = otherMorton;
			for (int i = 0; i < otherBox.Length; i++)
			{
				// An array element is a real variable, so these writes land — through a
				// List<Box> indexer they would update a temporary (see Bounds.cs).
				otherBox[i].Min = otherBox[i].Min - Vec3.Splat(searchLength);
				otherBox[i].Max = otherBox[i].Max + Vec3.Splat(searchLength);
			}

			// Query self's cached face BVH (C++ MinGap queries collider_).
			Collider collider = this.Collider;
			double minDistance = double.PositiveInfinity;
			collider.CollisionsWithBoxes(otherBox, false, (triOther, tri) =>
			{
				Vec3[] p = this.FaceTriangleVertices(tri);
				Vec3[] q = other.FaceTriangleVertices(triOther);
				minDistance = MinF64(
					minDistance,
					ColliderFunctions.DistanceTriangleTriangleSquared(p, q));
			});

			return Math.Sqrt(MinF64(minDistance, searchLength * searchLength));
		}

		private Vec3[] FaceTriangleVertices(int tri)
		{
			return new Vec3[]
			{
				this.VertPos[this.Halfedge[3 * tri].StartVert],
				this.VertPos[this.Halfedge[(3 * tri) + 1].StartVert],
				this.VertPos[this.Halfedge[(3 * tri) + 2].StartVert],
			};
		}
	}
}
