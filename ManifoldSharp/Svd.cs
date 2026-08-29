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

// Phase 9: SVD — ported from cpp-reference/manifold/src/svd.h
//
// C# port of svd.rs. The Rust module's free functions become static members of
// SvdFunctions (the LinalgFunctions pattern), so `using static
// ManifoldSharp.SvdFunctions;` makes call sites read `Svd(a)` and
// `SpectralNorm(a)` exactly as the Rust does. Symmetric3x3, Givens and QR are
// module-private in the Rust and are private nested types here for the same
// reach. SVDSet keeps the Rust/C++ spelling so the three files line up.
//
// SVDSet's U/S/V must stay public mutable FIELDS, never properties. Mat3's
// indexer returns a ref, so `usv.S[0][0]` is only a variable when S is a field;
// as a property it returns a temporary copy, which either fails to compile on
// assignment or silently reads (and discards writes to) that copy — SpectralNorm
// reads exactly that way, so a "tidy-up" to properties breaks it.
//
// Every expression below is transcribed in the Rust's operation order: this is a
// fixed-step iteration (12 sweeps, no convergence test), so a changed low bit is
// not damped out, it is fed back in. No FMA, no reassociation, no hypot — see the
// two "Per C++ #1681" comments, ported verbatim from the Rust.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <summary>
	/// The result of <see cref="SvdFunctions.Svd"/>: <c>a == u * s * transpose(v)</c>,
	/// with the singular values on the diagonal of <see cref="S"/> in descending order.
	/// </summary>
	/// <remarks>
	/// <see cref="U"/>, <see cref="S"/> and <see cref="V"/> are public mutable fields on
	/// purpose and must stay fields. <see cref="Mat3"/>'s indexer is ref-returning, so
	/// <c>usv.S[0][0]</c> only reaches the real storage while <c>S</c> is a field; as a
	/// property it would hand back a temporary copy, so the chained indexer would fail to
	/// compile on assignment and silently operate on that copy on read. See the file
	/// header.
	/// </remarks>
	public struct SVDSet
	{
		/// <summary>The left singular vectors, as columns.</summary>
		public Mat3 U;

		/// <summary>The (near-)diagonal matrix of singular values, largest first.</summary>
		public Mat3 S;

		/// <summary>The right singular vectors, as columns.</summary>
		public Mat3 V;
	}

	/// <summary>
	/// The free functions of <c>svd.rs</c> — a fixed-iteration Jacobi SVD of a 3x3
	/// matrix, and the spectral norm built on it.
	/// </summary>
	public static class SvdFunctions
	{
		private const double Gamma = 5.82842712474619;
		private const double CStar = 0.9238795325112867;
		private const double SStar = 0.3826834323650898;
		private const double SvdEpsilon = 1e-6;
		private const int JacobiSteps = 12;

		/// <summary>
		/// Computes the singular value decomposition of <paramref name="a"/> so that
		/// <c>a == u * s * transpose(v)</c>.
		/// </summary>
		/// <param name="a">The matrix to decompose.</param>
		/// <returns>The decomposition, with singular values sorted descending.</returns>
		public static SVDSet Svd(Mat3 a)
		{
			Mat3 v = JacobiEigenAnalysis(Symmetric3x3.From(a.Transpose() * a));
			Mat3 b = a * v;
			SortSingularValues(ref b, ref v);
			QR qr = QrDecomposition(ref b);
			return new SVDSet
			{
				U = qr.Q,
				S = qr.R,
				V = v,
			};
		}

		/// <summary>
		/// The spectral norm of <paramref name="a"/> — its largest singular value, which
		/// <see cref="Svd"/> leaves at <c>s[0][0]</c>.
		/// </summary>
		/// <param name="a">The matrix to measure.</param>
		/// <returns>The largest singular value.</returns>
		public static double SpectralNorm(Mat3 a)
		{
			SVDSet usv = Svd(a);
			return usv.S[0][0];
		}

		private static void CondSwap(bool c, ref double x, ref double y)
		{
			if (c)
			{
				double t = x;
				x = y;
				y = t;
			}
		}

		private static void CondNegSwap(bool c, ref double x, ref double y)
		{
			if (c)
			{
				double z = -x;
				x = y;
				y = z;
			}
		}

		private static void CondNegSwapVec3(bool c, ref Vec3 x, ref Vec3 y)
		{
			CondNegSwap(c, ref x.X, ref y.X);
			CondNegSwap(c, ref x.Y, ref y.Y);
			CondNegSwap(c, ref x.Z, ref y.Z);
		}

		private static double Dist2(Vec3 v)
		{
			return LinalgFunctions.Dot(v, v);
		}

		private static Givens ApproximateGivensQuaternion(Symmetric3x3 a)
		{
			Givens g = new Givens
			{
				Ch = 2.0 * (a.M00 - a.M11),
				Sh = a.M10,
			};
			bool b = Gamma * g.Sh * g.Sh < g.Ch * g.Ch;

			// Per C++ #1681: use sqrt(a*a + b*b) instead of hypot. hypot is not
			// required to be correctly-rounded (IEEE 754 §9.2), so libms may round
			// it differently; *, +, and sqrt are correctly-rounded, giving bit-exact
			// cross-platform results inside SVD's iterative reductions.
			double w = 1.0 / Math.Sqrt(g.Ch * g.Ch + g.Sh * g.Sh);
			if (!double.IsFinite(w))
			{
				b = false;
			}

			return new Givens
			{
				Ch = b ? w * g.Ch : CStar,
				Sh = b ? w * g.Sh : SStar,
			};
		}

		private static void JacobiConjugation(int x, int y, int z, ref Symmetric3x3 s, ref Vec4 q)
		{
			Givens g = ApproximateGivensQuaternion(s);
			double scale = 1.0 / (g.Ch * g.Ch + g.Sh * g.Sh);
			double a = (g.Ch * g.Ch - g.Sh * g.Sh) * scale;
			double b = 2.0 * g.Sh * g.Ch * scale;
			Symmetric3x3 old = s;

			s.M00 = a * (a * old.M00 + b * old.M10) + b * (a * old.M10 + b * old.M11);
			s.M10 = a * (-b * old.M00 + a * old.M10) + b * (-b * old.M10 + a * old.M11);
			s.M11 = -b * (-b * old.M00 + a * old.M10) + a * (-b * old.M10 + a * old.M11);
			s.M20 = a * old.M20 + b * old.M21;
			s.M21 = -b * old.M20 + a * old.M21;
			s.M22 = old.M22;

			Vec3 tmp = new Vec3(g.Sh * q.X, g.Sh * q.Y, g.Sh * q.Z);
			g.Sh *= q.W;
			q[z] = q[z] * g.Ch + g.Sh;
			q.W = q.W * g.Ch - tmp[z];
			q[x] = q[x] * g.Ch + tmp[y];
			q[y] = q[y] * g.Ch - tmp[x];

			Symmetric3x3 permuted = new Symmetric3x3
			{
				M00 = s.M11,
				M10 = s.M21,
				M11 = s.M22,
				M20 = s.M10,
				M21 = s.M20,
				M22 = s.M00,
			};
			s = permuted;
		}

		private static Mat3 JacobiEigenAnalysis(Symmetric3x3 s)
		{
			Vec4 q = new Vec4(0.0, 0.0, 0.0, 1.0);
			for (int i = 0; i < JacobiSteps; i++)
			{
				JacobiConjugation(0, 1, 2, ref s, ref q);
				JacobiConjugation(1, 2, 0, ref s, ref q);
				JacobiConjugation(2, 0, 1, ref s, ref q);
			}

			return Mat3.FromCols(
				new Vec3(
					1.0 - 2.0 * (q.Y * q.Y + q.Z * q.Z),
					2.0 * (q.X * q.Y - q.W * q.Z),
					2.0 * (q.X * q.Z + q.W * q.Y)),
				new Vec3(
					2.0 * (q.X * q.Y + q.W * q.Z),
					1.0 - 2.0 * (q.X * q.X + q.Z * q.Z),
					2.0 * (q.Y * q.Z - q.W * q.X)),
				new Vec3(
					2.0 * (q.X * q.Z - q.W * q.Y),
					2.0 * (q.Y * q.Z + q.W * q.X),
					1.0 - 2.0 * (q.X * q.X + q.Y * q.Y)));
		}

		private static void SortSingularValues(ref Mat3 b, ref Mat3 v)
		{
			double rho1 = Dist2(b.X);
			double rho2 = Dist2(b.Y);
			double rho3 = Dist2(b.Z);

			bool c = rho1 < rho2;
			CondNegSwapVec3(c, ref b.X, ref b.Y);
			CondNegSwapVec3(c, ref v.X, ref v.Y);
			CondSwap(c, ref rho1, ref rho2);

			c = rho1 < rho3;
			CondNegSwapVec3(c, ref b.X, ref b.Z);
			CondNegSwapVec3(c, ref v.X, ref v.Z);
			CondSwap(c, ref rho1, ref rho3);

			c = rho2 < rho3;
			CondNegSwapVec3(c, ref b.Y, ref b.Z);
			CondNegSwapVec3(c, ref v.Y, ref v.Z);
			CondSwap(c, ref rho2, ref rho3);
		}

		private static Givens QrGivensQuaternion(double a1, double a2)
		{
			// Per C++ #1681: sqrt(a*a + b*b) instead of hypot for cross-platform
			// bit-exactness (see approximate_givens_quaternion).
			double rho = Math.Sqrt(a1 * a1 + a2 * a2);
			Givens g = new Givens
			{
				Ch = Math.Abs(a1) + LinalgFunctions.MaxF64(rho, SvdEpsilon),
				Sh = rho > SvdEpsilon ? a2 : 0.0,
			};
			bool b = a1 < 0.0;
			CondSwap(b, ref g.Sh, ref g.Ch);

			// Per C++ #1681: use sqrt(a*a + b*b) instead of hypot. hypot is not
			// required to be correctly-rounded (IEEE 754 §9.2), so libms may round
			// it differently; *, +, and sqrt are correctly-rounded, giving bit-exact
			// cross-platform results inside SVD's iterative reductions.
			double w = 1.0 / Math.Sqrt(g.Ch * g.Ch + g.Sh * g.Sh);
			g.Ch *= w;
			g.Sh *= w;
			return g;
		}

		private static QR QrDecomposition(ref Mat3 b)
		{
			Mat3 q = default(Mat3);
			Mat3 r = default(Mat3);

			Givens g1 = QrGivensQuaternion(b[0][0], b[0][1]);
			double a = -2.0 * g1.Sh * g1.Sh + 1.0;
			double bb = 2.0 * g1.Ch * g1.Sh;

			r[0][0] = a * b[0][0] + bb * b[0][1];
			r[1][0] = a * b[1][0] + bb * b[1][1];
			r[2][0] = a * b[2][0] + bb * b[2][1];
			r[0][1] = -bb * b[0][0] + a * b[0][1];
			r[1][1] = -bb * b[1][0] + a * b[1][1];
			r[2][1] = -bb * b[2][0] + a * b[2][1];
			r[0][2] = b[0][2];
			r[1][2] = b[1][2];
			r[2][2] = b[2][2];

			Givens g2 = QrGivensQuaternion(r[0][0], r[0][2]);
			a = -2.0 * g2.Sh * g2.Sh + 1.0;
			bb = 2.0 * g2.Ch * g2.Sh;

			b[0][0] = a * r[0][0] + bb * r[0][2];
			b[1][0] = a * r[1][0] + bb * r[1][2];
			b[2][0] = a * r[2][0] + bb * r[2][2];
			b[0][1] = r[0][1];
			b[1][1] = r[1][1];
			b[2][1] = r[2][1];
			b[0][2] = -bb * r[0][0] + a * r[0][2];
			b[1][2] = -bb * r[1][0] + a * r[1][2];
			b[2][2] = -bb * r[2][0] + a * r[2][2];

			Givens g3 = QrGivensQuaternion(b[1][1], b[1][2]);
			a = -2.0 * g3.Sh * g3.Sh + 1.0;
			bb = 2.0 * g3.Ch * g3.Sh;

			r[0][0] = b[0][0];
			r[1][0] = b[1][0];
			r[2][0] = b[2][0];
			r[0][1] = a * b[0][1] + bb * b[0][2];
			r[1][1] = a * b[1][1] + bb * b[1][2];
			r[2][1] = a * b[2][1] + bb * b[2][2];
			r[0][2] = -bb * b[0][1] + a * b[0][2];
			r[1][2] = -bb * b[1][1] + a * b[1][2];
			r[2][2] = -bb * b[2][1] + a * b[2][2];

			double sh12 = 2.0 * (g1.Sh * g1.Sh - 0.5);
			double sh22 = 2.0 * (g2.Sh * g2.Sh - 0.5);
			double sh32 = 2.0 * (g3.Sh * g3.Sh - 0.5);

			q[0][0] = sh12 * sh22;
			q[1][0] = 4.0 * g2.Ch * g3.Ch * sh12 * g2.Sh * g3.Sh + 2.0 * g1.Ch * g1.Sh * sh32;
			q[2][0] = 4.0 * g1.Ch * g3.Ch * g1.Sh * g3.Sh - 2.0 * g2.Ch * sh12 * g2.Sh * sh32;

			q[0][1] = -2.0 * g1.Ch * g1.Sh * sh22;
			q[1][1] = -8.0 * g1.Ch * g2.Ch * g3.Ch * g1.Sh * g2.Sh * g3.Sh + sh12 * sh32;
			q[2][1] =
				-2.0 * g3.Ch * g3.Sh + 4.0 * g1.Sh * (g3.Ch * g1.Sh * g3.Sh + g1.Ch * g2.Ch * g2.Sh * sh32);

			q[0][2] = 2.0 * g2.Ch * g2.Sh;
			q[1][2] = -2.0 * g3.Ch * sh22 * g3.Sh;
			q[2][2] = sh22 * sh32;

			return new QR { Q = q, R = r };
		}

		/// <summary>
		/// The lower triangle of a symmetric 3x3 matrix, the only part the Jacobi sweep
		/// reads or writes.
		/// </summary>
		private struct Symmetric3x3
		{
			public double M00;
			public double M10;
			public double M11;
			public double M20;
			public double M21;
			public double M22;

			/// <summary>
			/// Ports the Rust <c>From&lt;Mat3&gt;</c>: takes the <em>lower</em> triangle of
			/// <paramref name="m"/>, the same six elements the struct stores.
			/// </summary>
			/// <remarks>
			/// <see cref="Mat3"/> is column-major, so <c>m[c][r]</c> is column c, row r:
			/// <c>m[0][1]</c> is element (row 1, col 0) — below the diagonal, not above it.
			/// The input is symmetric in exact arithmetic (it is always
			/// <c>Aᵀ·A</c>), but the two triangles are computed by different expressions
			/// and can differ in the last bits, so the port has to read the same six
			/// elements the Rust reads.
			/// </remarks>
			public static Symmetric3x3 From(Mat3 m)
			{
				return new Symmetric3x3
				{
					M00 = m[0][0],
					M10 = m[0][1],
					M11 = m[1][1],
					M20 = m[0][2],
					M21 = m[1][2],
					M22 = m[2][2],
				};
			}
		}

		/// <summary>An unnormalized Givens rotation, held as the (cos-half, sin-half) quaternion pair.</summary>
		private struct Givens
		{
			public double Ch;
			public double Sh;
		}

		/// <summary>The output of the Givens QR sweep: an orthogonal Q and an upper-triangular R.</summary>
		private struct QR
		{
			public Mat3 Q;
			public Mat3 R;
		}
	}
}
