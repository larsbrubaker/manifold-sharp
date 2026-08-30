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

// InterpTri — smooth surface interpolation using rational Bezier patches.
//
// Port of the `InterpTri` struct from C++ `smoothing.cpp`.
// This applies tangent-based Bezier interpolation to subdivision vertices,
// producing smooth curved surfaces instead of flat linear interpolation.
//
// ── C# port notes ────────────────────────────────────────────────────────────
// The class is `InterpTriFunctions`, not `InterpTri`: C# forbids a member with
// the same name as its enclosing type, and the module's one public function is
// `interp_tri`. CLAUDE.md's naming rule already grants the `Functions` suffix
// for exactly this kind of collision (`LinalgFunctions`, `SvdFunctions`), and
// the ported function keeps its own name, which is the name that matters.
//
// ── Local math, deliberately not the shared math ─────────────────────────────
// interp_tri.rs re-declares four things linalg.rs also provides — `next3`,
// `lerp4`, `qmul` and a normalize — and they are ported as private members here
// rather than redirected at LinalgFunctions, because two of them are *not* the
// same function:
//
//   qmul            linalg's `qmul` sums the same four products in a different
//                   order (`a.x*b.w + a.w*b.x + ...` versus this file's
//                   `a.w*b.x + a.x*b.w + ...`). Floating-point addition is not
//                   associative, so the two disagree in the last bit. This
//                   file's order is the specification.
//   SafeNormalize   returns the zero vector below a length of 1e-30 instead of
//                   dividing; linalg's `normalize` does not guard.
//   Lerp4           `a + (b - a) * t`, the form docs/PORTING_PLAN.md's general
//                   lerp rule steers away from — but the rule is about linalg's
//                   `lerp`, and this module spells its own out the other way.
//   Next3           the same function as Types.Next3; kept local because the
//                   Rust keeps it local.
//
// f64::EPSILON at interp_tri.rs:122 is the literal 2.220446049250313E-16.
// C# `double.Epsilon` is the smallest subnormal and is a different number.
//
// ── Trig: System.Math here, not DeterministicMath ────────────────────────────
// Slerp calls System.Math.Acos and System.Math.Sin, because interp_tri.rs:126-129
// reaches for std's `f64::acos`/`f64::sin` rather than `crate::math`'s musl port
// — the same inconsistency CrossSection.Clipper.cs's arc tolerance reproduces,
// and for the same reason: DeterministicMath disagrees with the platform libm by
// a ULP or two, and using it here moved subdivided vertex positions off the
// Rust's. The differential harness is bit-for-bit identical with System.Math and
// is not with DeterministicMath, so the Rust's spelling is what ships. These are
// the only two transcendental calls in the file; Math.Sqrt is correctly rounded
// by IEEE-754 and is exact everywhere.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// Rational-Bezier surface interpolation for subdivision vertices — the free
	/// functions of interp_tri.rs.
	/// </summary>
	public static class InterpTriFunctions
	{
		private const double KPrecision = 1e-12;

		/// <summary>
		/// f64::EPSILON. C#'s <see cref="double.Epsilon"/> is the smallest subnormal and is
		/// a different number entirely.
		/// </summary>
		private const double F64Epsilon = 2.220446049250313E-16;

		/// <summary>
		/// Apply smooth Bezier interpolation to all subdivision vertices.
		/// </summary>
		/// <remarks>
		/// Port of C++ <c>InterpTri::operator()(int vert)</c>.
		/// This repositions each vertex based on its barycentric coordinates and the
		/// halfedge tangent vectors from the original mesh, producing a smooth surface.
		/// </remarks>
		/// <param name="vertPos">The subdivided mesh's vertex positions, rewritten in place.</param>
		/// <param name="vertBary">Each vertex's barycentric coordinate in the original mesh.</param>
		/// <param name="old">The original (pre-subdivision) mesh, with its halfedge tangents.</param>
		public static void InterpTri(IList<Vec3> vertPos, IReadOnlyList<Barycentric> vertBary, ManifoldImpl old)
		{
			ArgumentNullException.ThrowIfNull(vertPos);
			ArgumentNullException.ThrowIfNull(vertBary);
			ArgumentNullException.ThrowIfNull(old);

			int numVert = Math.Min(vertPos.Count, vertBary.Count);
			for (int vert = 0; vert < numVert; vert++)
			{
				int tri = vertBary[vert].Tri;
				if (tri < 0 || tri >= old.Halfedge.Count / 3)
				{
					continue;
				}

				Vec4 uvw = vertBary[vert].Uvw;

				IVec4 halfedgesIvec = old.GetHalfedgesQuad(tri);
				int[] halfedges = new int[]
				{
					halfedgesIvec[0],
					halfedgesIvec[1],
					halfedgesIvec[2],
					halfedgesIvec[3],
				};
				Vec3[] corners = new Vec3[]
				{
					old.VertPos[old.Halfedge[halfedges[0]].StartVert],
					old.VertPos[old.Halfedge[halfedges[1]].StartVert],
					old.VertPos[old.Halfedge[halfedges[2]].StartVert],
					halfedges[3] < 0
						? new Vec3(0.0, 0.0, 0.0)
						: old.VertPos[old.Halfedge[halfedges[3]].StartVert],
				};

				// If vertex is exactly at a corner, use the corner position
				bool isCorner = false;
				for (int i = 0; i < 4; i++)
				{
					if (uvw[i] == 1.0)
					{
						vertPos[vert] = corners[i];
						isCorner = true;
						break;
					}
				}

				if (isCorner)
				{
					continue;
				}

				Vec4 posH = new Vec4(0.0, 0.0, 0.0, 0.0);

				if (halfedges[3] < 0)
				{
					// Triangle case
					Vec4[] tangentR = new Vec4[]
					{
						old.HalfedgeTangent[halfedges[0]],
						old.HalfedgeTangent[halfedges[1]],
						old.HalfedgeTangent[halfedges[2]],
					};
					Vec4[] tangentL = new Vec4[]
					{
						old.HalfedgeTangent[old.Halfedge[halfedges[2]].PairedHalfedge],
						old.HalfedgeTangent[old.Halfedge[halfedges[0]].PairedHalfedge],
						old.HalfedgeTangent[old.Halfedge[halfedges[1]].PairedHalfedge],
					};
					Vec3 centroid = new Vec3(
						(corners[0].X + corners[1].X + corners[2].X) / 3.0,
						(corners[0].Y + corners[1].Y + corners[2].Y) / 3.0,
						(corners[0].Z + corners[1].Z + corners[2].Z) / 3.0);

					for (int i = 0; i < 3; i++)
					{
						int j = Next3(i);
						int k = (i + 2) % 3; // Prev3(i)
						double x = uvw[k] / (1.0 - uvw[i]);

						Vec4[] bez = Bezier2Bezier(
							new Vec3[] { corners[j], corners[k] },
							new Vec4[] { tangentR[j], tangentL[k] },
							new Vec4[] { tangentL[j], tangentR[k] },
							x,
							centroid);

						Vec4[] bez1 = CubicBezier2Linear(
							bez[0],
							Bezier(HNormalize(bez[0]), bez[1]),
							Bezier(corners[i], Lerp4(tangentR[i], tangentL[i], x)),
							Homogeneous3(corners[i]),
							uvw[i]);
						Vec3 p = BezierPoint(bez1, uvw[i]);
						double weight = uvw[j] * uvw[k];
						posH = posH + Homogeneous4(new Vec4(p.X, p.Y, p.Z, weight));
					}
				}
				else
				{
					// Quad case
					Vec4[] tangentsX = new Vec4[]
					{
						old.HalfedgeTangent[halfedges[0]],
						old.HalfedgeTangent[old.Halfedge[halfedges[0]].PairedHalfedge],
						old.HalfedgeTangent[halfedges[2]],
						old.HalfedgeTangent[old.Halfedge[halfedges[2]].PairedHalfedge],
					};
					Vec4[] tangentsY = new Vec4[]
					{
						old.HalfedgeTangent[old.Halfedge[halfedges[3]].PairedHalfedge],
						old.HalfedgeTangent[halfedges[1]],
						old.HalfedgeTangent[old.Halfedge[halfedges[1]].PairedHalfedge],
						old.HalfedgeTangent[halfedges[3]],
					};
					Vec3 centroid = new Vec3(
						(corners[0].X + corners[1].X + corners[2].X + corners[3].X) * 0.25,
						(corners[0].Y + corners[1].Y + corners[2].Y + corners[3].Y) * 0.25,
						(corners[0].Z + corners[1].Z + corners[2].Z + corners[3].Z) * 0.25);
					double x = uvw[1] + uvw[2];
					double y = uvw[2] + uvw[3];
					Vec3 pX = Bezier2D(corners, tangentsX, tangentsY, x, y, centroid);
					Vec3 pY = Bezier2D(
						new Vec3[] { corners[1], corners[2], corners[3], corners[0] },
						new Vec4[] { tangentsY[1], tangentsY[2], tangentsY[3], tangentsY[0] },
						new Vec4[] { tangentsX[1], tangentsX[2], tangentsX[3], tangentsX[0] },
						y,
						1.0 - x,
						centroid);
					posH = posH + Homogeneous4(new Vec4(pX.X, pX.Y, pX.Z, x * (1.0 - x)));
					posH = posH + Homogeneous4(new Vec4(pY.X, pY.Y, pY.Z, y * (1.0 - y)));
				}

				Vec3 pos = HNormalize(posH);

				// Guard against NaN/inf from degenerate Bezier weights (matches C++ la::isfinite check)
				vertPos[vert] = double.IsFinite(pos.X) && double.IsFinite(pos.Y) && double.IsFinite(pos.Z)
					? pos
					: corners[0];
			}
		}

		/// <summary>Next index in a triangle (0→1→2→0).</summary>
		private static int Next3(int i)
		{
			return i == 2 ? 0 : i + 1;
		}

		/// <summary>Homogeneous coordinate conversion for Vec4 (w already set).</summary>
		private static Vec4 Homogeneous4(Vec4 v)
		{
			return new Vec4(v.X * v.W, v.Y * v.W, v.Z * v.W, v.W);
		}

		/// <summary>Homogeneous coordinate conversion for Vec3 (w = 1).</summary>
		private static Vec4 Homogeneous3(Vec3 v)
		{
			return new Vec4(v.X, v.Y, v.Z, 1.0);
		}

		/// <summary>Normalize from homogeneous coordinates back to Vec3.</summary>
		private static Vec3 HNormalize(Vec4 v)
		{
			if (v.W == 0.0)
			{
				return new Vec3(v.X, v.Y, v.Z);
			}

			return new Vec3(v.X / v.W, v.Y / v.W, v.Z / v.W);
		}

		// NOTE: C++ `InterpTri::Scale` (smoothing.cpp) is not ported: it is defined but
		// never called in the upstream reference either.

		/// <summary>Bezier control point from a position and tangent.</summary>
		private static Vec4 Bezier(Vec3 point, Vec4 tangent)
		{
			return Homogeneous4(new Vec4(
				point.X + tangent.X,
				point.Y + tangent.Y,
				point.Z + tangent.Z,
				tangent.W));
		}

		/// <summary>Linear interpolation, written the way interp_tri.rs writes it.</summary>
		private static Vec4 Lerp4(Vec4 a, Vec4 b, double t)
		{
			return new Vec4(
				a.X + ((b.X - a.X) * t),
				a.Y + ((b.Y - a.Y) * t),
				a.Z + ((b.Z - a.Z) * t),
				a.W + ((b.W - a.W) * t));
		}

		/// <summary>
		/// Evaluate cubic Bezier at parameter x, returning two control points
		/// for the resulting linear segment (used for further Bezier evaluation).
		/// Returns (left, right) Vec4 pair.
		/// </summary>
		private static Vec4[] CubicBezier2Linear(Vec4 p0, Vec4 p1, Vec4 p2, Vec4 p3, double x)
		{
			Vec4 p12 = Lerp4(p1, p2, x);
			Vec4 left = Lerp4(Lerp4(p0, p1, x), p12, x);
			Vec4 right = Lerp4(p12, Lerp4(p2, p3, x), x);
			return new Vec4[] { left, right };
		}

		/// <summary>Get point on the Bezier segment defined by two control points.</summary>
		private static Vec3 BezierPoint(Vec4[] points, double x)
		{
			return HNormalize(Lerp4(points[0], points[1], x));
		}

		/// <summary>Get tangent direction at the Bezier segment.</summary>
		private static Vec3 BezierTangent(Vec4[] points)
		{
			return SafeNormalize(HNormalize(points[1]) - HNormalize(points[0]));
		}

		private static Vec3 SafeNormalize(Vec3 v)
		{
			double len = Math.Sqrt((v.X * v.X) + (v.Y * v.Y) + (v.Z * v.Z));
			if (len < 1e-30)
			{
				return new Vec3(0.0, 0.0, 0.0);
			}

			return new Vec3(v.X / len, v.Y / len, v.Z / len);
		}

		/// <summary>Rotate vector v from quaternion start's frame to end's frame.</summary>
		private static Vec3 RotateFromTo(Vec3 v, Vec4 start, Vec4 end)
		{
			return QRot(end, QRot(QConj(start), v));
		}

		/// <summary>Slerp between two unit quaternions. longWay reverses the short/long path.</summary>
		private static Vec4 Slerp(Vec4 x, Vec4 y, double a, bool longWay)
		{
			Vec4 z = y;
			double cosTheta = (x.X * y.X) + (x.Y * y.Y) + (x.Z * y.Z) + (x.W * y.W);

			// Take the long way around the sphere only when requested
			if ((cosTheta < 0.0) != longWay)
			{
				z = new Vec4(-y.X, -y.Y, -y.Z, -y.W);
				cosTheta = -cosTheta;
			}

			if (Math.Abs(cosTheta) > 1.0 - F64Epsilon)
			{
				// Nearly identical: use linear interpolation for stability
				return Lerp4(x, z, a);
			}
			else
			{
				// System.Math and not DeterministicMath: the Rust reaches for std's
				// f64::acos / f64::sin here, not crate::math's. See the file header.
				double angle = Math.Acos(cosTheta);
				double sinAngle = Math.Sin(angle);
				double s0 = Math.Sin((1.0 - a) * angle) / sinAngle;
				double s1 = Math.Sin(a * angle) / sinAngle;
				return new Vec4(
					(s0 * x.X) + (s1 * z.X),
					(s0 * x.Y) + (s1 * z.Y),
					(s0 * x.Z) + (s1 * z.Z),
					(s0 * x.W) + (s1 * z.W));
			}
		}

		/// <summary>
		/// Returns a normalized vector orthogonal to <paramref name="reference"/>, in the
		/// plane of <paramref name="reference"/> and <paramref name="input"/>. Falls back to
		/// <paramref name="altIn"/> if input is colinear with ref.
		/// </summary>
		private static Vec3 OrthogonalTo(Vec3 input, Vec3 altIn, Vec3 reference)
		{
			Vec3 outVec = input - (reference * Dot(input, reference));
			if (Dot(outVec, outVec) < KPrecision * Dot(input, input))
			{
				outVec = altIn - (reference * Dot(altIn, reference));
			}

			return SafeNormalize(outVec);
		}

		/// <summary>
		/// Quaternion multiplication.
		/// </summary>
		/// <remarks>
		/// Not <see cref="LinalgFunctions.QMul"/>: that one sums the same four products in a
		/// different order, and floating-point addition is not associative. See the file
		/// header.
		/// </remarks>
		private static Vec4 QMulLocal(Vec4 a, Vec4 b)
		{
			return new Vec4(
				(a.W * b.X) + (a.X * b.W) + (a.Y * b.Z) - (a.Z * b.Y),
				(a.W * b.Y) - (a.X * b.Z) + (a.Y * b.W) + (a.Z * b.X),
				(a.W * b.Z) + (a.X * b.Y) - (a.Y * b.X) + (a.Z * b.W),
				(a.W * b.W) - (a.X * b.X) - (a.Y * b.Y) - (a.Z * b.Z));
		}

		/// <summary>Build a rotation quaternion that maps the x-axis to the given direction.</summary>
		private static Vec4 RotationQuatXdirTo(Vec3 orig, Vec3 dest)
		{
			// Rotation from orig to dest (both should be unit vectors)
			double d = Dot(orig, dest);
			if (d >= 1.0 - 1e-12)
			{
				return new Vec4(0.0, 0.0, 0.0, 1.0); // identity
			}

			if (d <= -1.0 + 1e-12)
			{
				// 180 degree rotation — pick an orthogonal axis
				Vec3 axis = Math.Abs(orig.X) < 0.9
					? SafeNormalize(Cross(orig, new Vec3(1.0, 0.0, 0.0)))
					: SafeNormalize(Cross(orig, new Vec3(0.0, 1.0, 0.0)));
				return new Vec4(axis.X, axis.Y, axis.Z, 0.0);
			}

			Vec3 c = Cross(orig, dest);
			double w = 1.0 + d;
			double len = Math.Sqrt((c.X * c.X) + (c.Y * c.Y) + (c.Z * c.Z) + (w * w));
			return new Vec4(c.X / len, c.Y / len, c.Z / len, w / len);
		}

		/// <summary>Bezier2Bezier: compute a point and tangent along a cross-curve Bezier.</summary>
		private static Vec4[] Bezier2Bezier(
			Vec3[] corners,
			Vec4[] tangentsX,
			Vec4[] tangentsY,
			double x,
			Vec3 anchor)
		{
			Vec4[] bez = CubicBezier2Linear(
				Homogeneous3(corners[0]),
				Bezier(corners[0], tangentsX[0]),
				Bezier(corners[1], tangentsX[1]),
				Homogeneous3(corners[1]),
				x);
			Vec3 end = BezierPoint(bez, x);
			Vec3 tangent = BezierTangent(bez);

			Vec3[] nTangentsX = new Vec3[]
			{
				SafeNormalize(new Vec3(tangentsX[0].X, tangentsX[0].Y, tangentsX[0].Z)),
				SafeNormalize(new Vec3(
					-tangentsX[1].X,
					-tangentsX[1].Y,
					-tangentsX[1].Z)),
			};
			Vec3[] biTangents = new Vec3[]
			{
				OrthogonalTo(
					new Vec3(tangentsY[0].X, tangentsY[0].Y, tangentsY[0].Z),
					anchor - corners[0],
					nTangentsX[0]),
				OrthogonalTo(
					new Vec3(tangentsY[1].X, tangentsY[1].Y, tangentsY[1].Z),
					anchor - corners[1],
					nTangentsX[1]),
			};

			Vec4 q0 = RotationQuatMat(Mat3.FromCols(
				nTangentsX[0],
				biTangents[0],
				Cross(nTangentsX[0], biTangents[0])));
			Vec4 q1 = RotationQuatMat(Mat3.FromCols(
				nTangentsX[1],
				biTangents[1],
				Cross(nTangentsX[1], biTangents[1])));

			Vec3 edge = corners[1] - corners[0];
			bool longWay = Dot(nTangentsX[0], edge) + Dot(nTangentsX[1], edge) < 0.0;
			Vec4 qTmp = Slerp(q0, q1, x, longWay);
			Vec4 q = QMulLocal(RotationQuatXdirTo(QXDir(qTmp), tangent), qTmp);

			Vec3 delta;
			{
				Vec3 r0 = RotateFromTo(
					new Vec3(tangentsY[0].X, tangentsY[0].Y, tangentsY[0].Z),
					q0,
					q);
				Vec3 r1 = RotateFromTo(
					new Vec3(tangentsY[1].X, tangentsY[1].Y, tangentsY[1].Z),
					q1,
					q);
				delta = new Vec3(
					r0.X + ((r1.X - r0.X) * x),
					r0.Y + ((r1.Y - r0.Y) * x),
					r0.Z + ((r1.Z - r0.Z) * x));
			}

			double deltaW = tangentsY[0].W + ((tangentsY[1].W - tangentsY[0].W) * x);

			return new Vec4[]
			{
				Homogeneous3(end),
				new Vec4(delta.X, delta.Y, delta.Z, deltaW),
			};
		}

		/// <summary>Full 2D Bezier surface evaluation.</summary>
		private static Vec3 Bezier2D(
			Vec3[] corners,
			Vec4[] tangentsX,
			Vec4[] tangentsY,
			double x,
			double y,
			Vec3 centroid)
		{
			Vec4[] bez0 = Bezier2Bezier(
				new Vec3[] { corners[0], corners[1] },
				new Vec4[] { tangentsX[0], tangentsX[1] },
				new Vec4[] { tangentsY[0], tangentsY[1] },
				x,
				centroid);
			Vec4[] bez1 = Bezier2Bezier(
				new Vec3[] { corners[2], corners[3] },
				new Vec4[] { tangentsX[2], tangentsX[3] },
				new Vec4[] { tangentsY[2], tangentsY[3] },
				1.0 - x,
				centroid);

			Vec4[] bez = CubicBezier2Linear(
				bez0[0],
				Bezier(HNormalize(bez0[0]), bez0[1]),
				Bezier(HNormalize(bez1[0]), bez1[1]),
				bez1[0],
				y);
			return BezierPoint(bez, y);
		}
	}
}
