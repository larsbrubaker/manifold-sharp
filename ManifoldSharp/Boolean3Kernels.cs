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

// Boolean3Kernels.cs — the geometric kernels of the boolean intersection
// algorithm: shadow predicates (shadow01), the edge-edge / vertex-face /
// edge-face kernels (kernel11, kernel02, kernel12), and the two broadphase
// drivers built on them (intersect12, winding03).
//
// Ports the kernel portion of src/boolean3.cpp. Extracted from boolean3.rs,
// which owns the Boolean3 constructor pipeline and the public boolean()
// entry points; ray_cast (boolean3.rs) also reuses kernel12. These are the
// ONLY places where floating-point operations occur in the boolean
// algorithm. They are carefully designed to minimize rounding error and to
// eliminate it at edge cases. The branch structure must exactly match the
// C++ to produce identical results.
//
// ── C# port notes ────────────────────────────────────────────────────────────
// boolean3_kernels.rs is a private `mod` of boolean3.rs whose items are
// `pub(super)`; C# has no such scope, so the class is `internal` and the two
// entry points the sibling files need (Kernel12, Intersect12, Winding03) are
// internal members of it. Nothing outside this assembly can reach them, which
// is the same reachability the Rust grants.
//
// Every `f64::NAN` here is DeterministicMath.PositiveQuietNaN, never
// `double.NaN`: the C# constant has the sign bit set and would diverge from the
// Rust in any bit-based compare downstream. The NaNs written into `yz01`,
// `xyzz11`, `z02` and `v12` are read back only through `double.IsFinite`, but
// `v12` also reaches the output mesh through Boolean3's `Intersections`, so the
// payload matters.
//
// ── File split ───────────────────────────────────────────────────────────────
// boolean3_kernels.rs is 576 lines; its C# expansion does not fit the 800-line
// cap, so it lands as two files, both continuing one `static class`:
//   Boolean3Kernels.cs             this file — the four kernels and their
//                                  three floating-point helpers
//   Boolean3Kernels.Broadphase.cs  Intersect12 and Winding03, the two
//                                  collider-driven drivers built on them

using System.Runtime.CompilerServices;

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <summary>
	/// The floating-point kernels of the boolean intersection algorithm — the only
	/// place in the boolean pipeline where floating-point operations occur.
	/// </summary>
	internal static partial class Boolean3Kernels
	{
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static double WithSign(bool pos, double v)
		{
			return pos ? v : -v;
		}

		/// <summary>
		/// Interpolate along edge (aL, aR) at x-coordinate <paramref name="x"/>.
		/// Returns (y, z) at the interpolated point.
		/// Uses the closer endpoint as the base to minimize rounding error.
		/// </summary>
		private static Vec2 Interpolate(Vec3 aL, Vec3 aR, double x)
		{
			double dxL = x - aL.X;
			double dxR = x - aR.X;
			System.Diagnostics.Debug.Assert(dxL * dxR <= 0.0, "Boolean manifold error: not in domain");
			bool useL = Math.Abs(dxL) < Math.Abs(dxR);
			Vec3 dLr = aR - aL;
			double lambda = (useL ? dxL : dxR) / dLr.X;
			if (!double.IsFinite(lambda) || !double.IsFinite(dLr.Y) || !double.IsFinite(dLr.Z))
			{
				return new Vec2(aL.Y, aL.Z);
			}

			return new Vec2(
				(lambda * dLr.Y) + (useL ? aL.Y : aR.Y),
				(lambda * dLr.Z) + (useL ? aL.Z : aR.Z));
		}

		/// <summary>
		/// Find the intersection of two edges projected onto the yz-plane, parameterized
		/// by their y-coordinates. Returns (x, y, z_a, z_b) at the intersection.
		/// </summary>
		private static Vec4 IntersectEdges(Vec3 aL, Vec3 aR, Vec3 bL, Vec3 bR)
		{
			double dyL = bL.Y - aL.Y;
			double dyR = bR.Y - aR.Y;
			System.Diagnostics.Debug.Assert(dyL * dyR <= 0.0, "Boolean manifold error: no intersection");
			bool useL = Math.Abs(dyL) < Math.Abs(dyR);
			double dx = aR.X - aL.X;
			double lambda = (useL ? dyL : dyR) / (dyL - dyR);
			if (!double.IsFinite(lambda))
			{
				lambda = 0.0;
			}

			double x = (lambda * dx) + (useL ? aL.X : aR.X);
			double aDy = aR.Y - aL.Y;
			double bDy = bR.Y - bL.Y;
			bool useA = Math.Abs(aDy) < Math.Abs(bDy);
			double y = (lambda * (useA ? aDy : bDy))
				+ (useL
					? (useA ? aL.Y : bL.Y)
					: (useA ? aR.Y : bR.Y));
			double z = (lambda * (aR.Z - aL.Z)) + (useL ? aL.Z : aR.Z);
			double w = (lambda * (bR.Z - bL.Z)) + (useL ? bL.Z : bR.Z);
			return new Vec4(x, y, z, w);
		}

		/// <summary>
		/// Symbolic perturbation shadow predicate.
		/// When p == q, the tie is broken by the sign of dir.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static bool Shadows(double p, double q, double dir)
		{
			return p == q ? dir < 0.0 : p < q;
		}

		// -------------------------------------------------------------------
		// Shadow01 — vertex-edge shadow test
		// -------------------------------------------------------------------
		// Tests whether vertex a0 of mesh A shadows edge b1 of mesh B.
		// Returns (winding contribution, (y,z) interpolated position).

		private static (int S01, Vec2 Yz01) Shadow01(
			int a0,
			int b1,
			ManifoldImpl inA,
			ManifoldImpl inB,
			bool expandP,
			bool forward)
		{
			int b1s = inB.Halfedge[b1].StartVert;
			int b1e = inB.Halfedge[b1].EndVert;
			double a0x = inA.VertPos[a0].X;
			double b1sx = inB.VertPos[b1s].X;
			double b1ex = inB.VertPos[b1e].X;
			double a0xp = inA.VertNormal[a0].X;
			double b1sxp = inB.VertNormal[b1s].X;
			double b1exp = inB.VertNormal[b1e].X;

			int s01 = forward
				? (Shadows(a0x, b1ex, WithSign(expandP, a0xp) - b1exp) ? 1 : 0)
					- (Shadows(a0x, b1sx, WithSign(expandP, a0xp) - b1sxp) ? 1 : 0)
				: (Shadows(b1sx, a0x, WithSign(expandP, b1sxp) - a0xp) ? 1 : 0)
					- (Shadows(b1ex, a0x, WithSign(expandP, b1exp) - a0xp) ? 1 : 0);

			Vec2 yz01 = new Vec2(DeterministicMath.PositiveQuietNaN, DeterministicMath.PositiveQuietNaN);

			if (s01 != 0)
			{
				yz01 = Interpolate(inB.VertPos[b1s], inB.VertPos[b1e], inA.VertPos[a0].X);
				int b1pair = inB.Halfedge[b1].PairedHalfedge;
				double dir = inB.FaceNormal[b1 / 3].Y + inB.FaceNormal[b1pair / 3].Y;
				if (forward)
				{
					if (!Shadows(inA.VertPos[a0].Y, yz01.X, -dir))
					{
						s01 = 0;
					}
				}
				else if (!Shadows(yz01.X, inA.VertPos[a0].Y, WithSign(expandP, dir)))
				{
					s01 = 0;
				}
			}

			return (s01, yz01);
		}

		// -------------------------------------------------------------------
		// Kernel11 — edge-edge intersection
		// -------------------------------------------------------------------

		private static (int S11, Vec4 Xyzz11) Kernel11(
			int p1,
			int q1,
			ManifoldImpl inP,
			ManifoldImpl inQ,
			bool expandP)
		{
			Vec4 xyzz11;
			int s11 = 0;

			int k = 0;
			Vec3[] pRl = { Vec3.Splat(0.0), Vec3.Splat(0.0) };
			Vec3[] qRl = { Vec3.Splat(0.0), Vec3.Splat(0.0) };
			bool shadowState = false;

			int[] p0 = { inP.Halfedge[p1].StartVert, inP.Halfedge[p1].EndVert };
			for (int i = 0; i < 2; i++)
			{
				(int s01, Vec2 yz01) = Shadow01(p0[i], q1, inP, inQ, expandP, true);
				if (double.IsFinite(yz01.X))
				{
					s11 += s01 * (i == 0 ? -1 : 1);
					if (k < 2 && (k == 0 || (s01 != 0) != shadowState))
					{
						shadowState = s01 != 0;
						pRl[k] = inP.VertPos[p0[i]];
						qRl[k] = new Vec3(pRl[k].X, yz01.X, yz01.Y);
						k += 1;
					}
				}
			}

			int[] q0 = { inQ.Halfedge[q1].StartVert, inQ.Halfedge[q1].EndVert };
			for (int i = 0; i < 2; i++)
			{
				(int s10, Vec2 yz10) = Shadow01(q0[i], p1, inQ, inP, expandP, false);
				if (double.IsFinite(yz10.X))
				{
					s11 += s10 * (i == 0 ? -1 : 1);
					if (k < 2 && (k == 0 || (s10 != 0) != shadowState))
					{
						shadowState = s10 != 0;
						qRl[k] = inQ.VertPos[q0[i]];
						pRl[k] = new Vec3(qRl[k].X, yz10.X, yz10.Y);
						k += 1;
					}
				}
			}

			if (s11 == 0)
			{
				xyzz11 = Vec4.Splat(DeterministicMath.PositiveQuietNaN);
			}
			else
			{
				System.Diagnostics.Debug.Assert(k == 2, "Boolean manifold error: s11");
				xyzz11 = IntersectEdges(pRl[0], pRl[1], qRl[0], qRl[1]);

				int p1pair = inP.Halfedge[p1].PairedHalfedge;
				double dirP = inP.FaceNormal[p1 / 3].Z + inP.FaceNormal[p1pair / 3].Z;
				int q1pair = inQ.Halfedge[q1].PairedHalfedge;
				double dirQ = inQ.FaceNormal[q1 / 3].Z + inQ.FaceNormal[q1pair / 3].Z;
				if (!Shadows(xyzz11.Z, xyzz11.W, WithSign(expandP, dirP) - dirQ))
				{
					s11 = 0;
				}
			}

			return (s11, xyzz11);
		}

		// -------------------------------------------------------------------
		// Kernel02 — vertex-face intersection
		// -------------------------------------------------------------------

		private static (int S02, double Z02) Kernel02(
			int a0,
			int b2,
			ManifoldImpl inA,
			ManifoldImpl inB,
			bool expandP,
			bool forward)
		{
			int s02 = 0;
			double z02;

			int k = 0;
			Vec3[] yzzRl = { Vec3.Splat(0.0), Vec3.Splat(0.0) };
			bool shadowState = false;

			for (int i = 0; i < 3; i++)
			{
				int b1 = (3 * b2) + i;
				Halfedge edgeB = inB.Halfedge[b1];
				int b1f = edgeB.IsForward() ? b1 : edgeB.PairedHalfedge;

				(int s01, Vec2 yz01) = Shadow01(a0, b1f, inA, inB, expandP, forward);
				if (double.IsFinite(yz01.X))
				{
					s02 += s01 * (forward == edgeB.IsForward() ? -1 : 1);
					if (k < 2 && (k == 0 || (s01 != 0) != shadowState))
					{
						shadowState = s01 != 0;
						yzzRl[k] = new Vec3(yz01.X, yz01.Y, yz01.Y);
						k += 1;
					}
				}
			}

			if (s02 == 0)
			{
				z02 = DeterministicMath.PositiveQuietNaN;
			}
			else
			{
				System.Diagnostics.Debug.Assert(k == 2, "Boolean manifold error: s02");
				Vec3 vertPosA = inA.VertPos[a0];
				z02 = Interpolate(yzzRl[0], yzzRl[1], vertPosA.Y).Y;
				if (forward)
				{
					if (!Shadows(vertPosA.Z, z02, -inB.FaceNormal[b2].Z))
					{
						s02 = 0;
					}
				}
				else if (!Shadows(z02, vertPosA.Z, WithSign(expandP, inB.FaceNormal[b2].Z)))
				{
					s02 = 0;
				}
			}

			return (s02, z02);
		}

		// -------------------------------------------------------------------
		// Kernel12 — edge-face intersection
		// -------------------------------------------------------------------

		internal static (int X12, Vec3 V12) Kernel12(
			int a1,
			int b2,
			ManifoldImpl inA,
			ManifoldImpl inB,
			ManifoldImpl inP,
			ManifoldImpl inQ,
			bool expandP,
			bool forward)
		{
			int x12 = 0;
			Vec3 v12 = Vec3.Splat(DeterministicMath.PositiveQuietNaN);

			int k = 0;
			Vec3[] xzyLr0 = { Vec3.Splat(0.0), Vec3.Splat(0.0) };
			Vec3[] xzyLr1 = { Vec3.Splat(0.0), Vec3.Splat(0.0) };
			bool shadowState = false;

			Halfedge edgeA = inA.Halfedge[a1];

			foreach (int vertA in new[] { edgeA.StartVert, edgeA.EndVert })
			{
				(int s, double z) = Kernel02(vertA, b2, inA, inB, expandP, forward);
				if (double.IsFinite(z))
				{
					x12 += s * ((vertA == edgeA.StartVert) == forward ? 1 : -1);
					if (k < 2 && (k == 0 || (s != 0) != shadowState))
					{
						shadowState = s != 0;
						Vec3 pos = inA.VertPos[vertA];
						xzyLr0[k] = new Vec3(pos.X, pos.Z, pos.Y);
						xzyLr1[k] = xzyLr0[k];
						xzyLr1[k].Y = z;
						k += 1;
					}
				}
			}

			for (int i = 0; i < 3; i++)
			{
				int b1 = (3 * b2) + i;
				Halfedge edgeB = inB.Halfedge[b1];
				int b1f = edgeB.IsForward() ? b1 : edgeB.PairedHalfedge;
				(int s, Vec4 xyzz) = forward
					? Kernel11(a1, b1f, inP, inQ, expandP)
					: Kernel11(b1f, a1, inP, inQ, expandP);
				if (double.IsFinite(xyzz.X))
				{
					x12 -= s * (edgeB.IsForward() ? 1 : -1);
					if (k < 2 && (k == 0 || (s != 0) != shadowState))
					{
						shadowState = s != 0;
						xzyLr0[k] = new Vec3(xyzz.X, xyzz.Z, xyzz.Y);
						xzyLr1[k] = xzyLr0[k];
						xzyLr1[k].Y = xyzz.W;
						if (!forward)
						{
							double tmp = xzyLr0[k].Y;
							xzyLr0[k].Y = xzyLr1[k].Y;
							xzyLr1[k].Y = tmp;
						}

						k += 1;
					}
				}
			}

			if (x12 == 0)
			{
				v12 = Vec3.Splat(DeterministicMath.PositiveQuietNaN);
			}
			else
			{
				System.Diagnostics.Debug.Assert(k == 2, "Boolean manifold error: v12");
				Vec4 xzyy = IntersectEdges(xzyLr0[0], xzyLr0[1], xzyLr1[0], xzyLr1[1]);
				v12.X = xzyy.X;
				v12.Y = xzyy.Z;
				v12.Z = xzyy.Y;
			}

			return (x12, v12);
		}
	}
}
