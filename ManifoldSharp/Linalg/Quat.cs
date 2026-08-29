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

// Ports the quaternion functions, the Mat3x4/Mat4 conversions and the matrix
// factory functions of linalg.rs. Continues the LinalgFunctions partial class
// begun in LinalgFunctions.cs; the module header for the whole Linalg folder
// lives in Vec3.cs.
//
// The Rust writes `pub type Quat = Vec4;` — a quaternion is a Vec4, not a
// distinct type, and every function below takes and returns the same struct the
// vector operators work on. C# type aliases are file-scoped, so the alias below
// buys readability inside this file only; from outside, these signatures read
// as Vec4. Any other file that wants the spelling declares its own
// `using Quat = ManifoldSharp.Linalg.Vec4;`.

using System.Runtime.CompilerServices;

using Quat = ManifoldSharp.Linalg.Vec4;

namespace ManifoldSharp.Linalg
{
	/// <content>
	/// Quaternion functions, matrix conversions and matrix factories.
	/// </content>
	public static partial class LinalgFunctions
	{
		// ─── Helper conversions ──────────────────────────────────────────────────

		/// <summary>Embed <see cref="Mat3x4"/> into <see cref="Mat4"/> by appending bottom row [0,0,0,1].</summary>
		public static Mat4 Mat3x4ToMat4(Mat3x4 m)
		{
			return Mat4.FromCols(
				new Vec4(m.X.X, m.X.Y, m.X.Z, 0.0),
				new Vec4(m.Y.X, m.Y.Y, m.Y.Z, 0.0),
				new Vec4(m.Z.X, m.Z.Y, m.Z.Z, 0.0),
				new Vec4(m.W.X, m.W.Y, m.W.Z, 1.0));
		}

		/// <summary>Extract upper-left 3 rows from a <see cref="Mat4"/> (drops the 4th row).</summary>
		public static Mat3x4 Mat4ToMat3x4(Mat4 m)
		{
			return Mat3x4.FromCols(m.X.Xyz(), m.Y.Xyz(), m.Z.Xyz(), m.W.Xyz());
		}

		// ─── Quaternion functions ────────────────────────────────────────────────

		/// <summary>Quaternion conjugate: <c>{-x, -y, -z, w}</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Quat QConj(Quat q)
		{
			return new Vec4(-q.X, -q.Y, -q.Z, q.W);
		}

		/// <summary>Quaternion inverse.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Quat QInv(Quat q)
		{
			return QConj(q) / LengthSquared(q);
		}

		/// <summary>Quaternion Hamilton product.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Quat QMul(Quat a, Quat b)
		{
			return new Quat(
				a.X * b.W + a.W * b.X + a.Y * b.Z - a.Z * b.Y,
				a.Y * b.W + a.W * b.Y + a.Z * b.X - a.X * b.Z,
				a.Z * b.W + a.W * b.Z + a.X * b.Y - a.Y * b.X,
				a.W * b.W - a.X * b.X - a.Y * b.Y - a.Z * b.Z);
		}

		/// <summary>X-axis direction from quaternion: <c>qrot(q, {1,0,0})</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 QXDir(Quat q)
		{
			return new Vec3(
				q.W * q.W + q.X * q.X - q.Y * q.Y - q.Z * q.Z,
				(q.X * q.Y + q.Z * q.W) * 2.0,
				(q.Z * q.X - q.Y * q.W) * 2.0);
		}

		/// <summary>Y-axis direction from quaternion: <c>qrot(q, {0,1,0})</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 QYDir(Quat q)
		{
			return new Vec3(
				(q.X * q.Y - q.Z * q.W) * 2.0,
				q.W * q.W - q.X * q.X + q.Y * q.Y - q.Z * q.Z,
				(q.Y * q.Z + q.X * q.W) * 2.0);
		}

		/// <summary>Z-axis direction from quaternion: <c>qrot(q, {0,0,1})</c>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 QZDir(Quat q)
		{
			return new Vec3(
				(q.Z * q.X + q.Y * q.W) * 2.0,
				(q.Y * q.Z - q.X * q.W) * 2.0,
				q.W * q.W - q.X * q.X - q.Y * q.Y + q.Z * q.Z);
		}

		/// <summary>Rotation matrix from quaternion.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Mat3 QMat(Quat q)
		{
			return Mat3.FromCols(QXDir(q), QYDir(q), QZDir(q));
		}

		/// <summary>Rotate vector <paramref name="v"/> by quaternion <paramref name="q"/>.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 QRot(Quat q, Vec3 v)
		{
			return QXDir(q) * v.X + QYDir(q) * v.Y + QZDir(q) * v.Z;
		}

		/// <summary>Rotation angle of a unit quaternion.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double QAngle(Quat q)
		{
			return DeterministicMath.Atan2(Length(q.Xyz()), q.W) * 2.0;
		}

		/// <summary>Rotation axis of a unit quaternion.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Vec3 QAxis(Quat q)
		{
			return Normalize(q.Xyz());
		}

		/// <summary>Quaternion nlerp — shortest path.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Quat QNLerp(Quat a, Quat b, double t)
		{
			Quat b2 = Dot(a, b) < 0.0 ? -b : b;
			return Normalize(Lerp(a, b2, t));
		}

		/// <summary>Quaternion slerp — shortest path.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Quat QSlerp(Quat a, Quat b, double t)
		{
			Quat b2 = Dot(a, b) < 0.0 ? -b : b;

			// slerp on Vec4 (unit quaternion treated as unit 4D vector)
			double d = MinF64(MaxF64(Dot(a, b2), -1.0), 1.0);
			double th = d > 1.0 ? 0.0 : DeterministicMath.Acos(d);
			if (th == 0.0)
			{
				return a;
			}

			return a * DeterministicMath.Sin(th * (1.0 - t)) / DeterministicMath.Sin(th)
				+ b2 * DeterministicMath.Sin(th * t) / DeterministicMath.Sin(th);
		}

		/// <summary>Unit quaternion from axis + angle.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static Quat RotationQuatAxisAngle(Vec3 axis, double angle)
		{
			return new Quat(
				axis.X * DeterministicMath.Sin(angle / 2.0),
				axis.Y * DeterministicMath.Sin(angle / 2.0),
				axis.Z * DeterministicMath.Sin(angle / 2.0),
				DeterministicMath.Cos(angle / 2.0));
		}

		/// <summary>Unit quaternion representing the shortest rotation from <paramref name="orig"/> to <paramref name="dest"/>.</summary>
		public static Quat RotationQuatVec(Vec3 orig, Vec3 dest)
		{
			double cosTheta = Dot(orig, dest);

			// f64::EPSILON. C#'s double.Epsilon is the smallest subnormal and is a
			// different number entirely.
			const double Eps = 2.220446049250313E-16;
			if (cosTheta >= 1.0 - Eps)
			{
				return new Quat(0.0, 0.0, 0.0, 1.0);
			}

			if (cosTheta < -1.0 + Eps)
			{
				Vec3 axis0 = Cross(new Vec3(0.0, 0.0, 1.0), orig);
				if (LengthSquared(axis0) < Eps)
				{
					axis0 = Cross(new Vec3(1.0, 0.0, 0.0), orig);
				}

				return RotationQuatAxisAngle(Normalize(axis0), Math.PI);
			}

			Vec3 axis = Cross(orig, dest);
			double s = Math.Sqrt((1.0 + cosTheta) * 2.0);
			return new Quat(axis.X / s, axis.Y / s, axis.Z / s, s * 0.5);
		}

		/// <summary>Unit quaternion from a rotation matrix.</summary>
		public static Quat RotationQuatMat(Mat3 m)
		{
			Vec4 q = new Vec4(
				m.X.X - m.Y.Y - m.Z.Z,
				m.Y.Y - m.X.X - m.Z.Z,
				m.Z.Z - m.X.X - m.Y.Y,
				m.X.X + m.Y.Y + m.Z.Z);

			// s[argmax(q)] gives the sign correction
			Vec4[] s = new Vec4[]
			{
				new Vec4(1.0, m.X.Y + m.Y.X, m.Z.X + m.X.Z, m.Y.Z - m.Z.Y),
				new Vec4(m.X.Y + m.Y.X, 1.0, m.Y.Z + m.Z.Y, m.Z.X - m.X.Z),
				new Vec4(m.X.Z + m.Z.X, m.Y.Z + m.Z.Y, 1.0, m.X.Y - m.Y.X),
				new Vec4(m.Y.Z - m.Z.Y, m.Z.X - m.X.Z, m.X.Y - m.Y.X, 1.0),
			};
			int idx = ArgMax(q);

			// copysign(normalize(sqrt(max(0, 1+q))), s[idx])
			Vec4 sq = new Vec4(
				Math.Sqrt(MaxF64(0.0, 1.0 + q.X)),
				Math.Sqrt(MaxF64(0.0, 1.0 + q.Y)),
				Math.Sqrt(MaxF64(0.0, 1.0 + q.Z)),
				Math.Sqrt(MaxF64(0.0, 1.0 + q.W)));
			Vec4 n = Normalize(sq);
			Vec4 si = s[idx];
			return new Vec4(
				Math.CopySign(n.X, si.X),
				Math.CopySign(n.Y, si.Y),
				Math.CopySign(n.Z, si.Z),
				Math.CopySign(n.W, si.W));
		}

		// ─── Matrix factory functions ────────────────────────────────────────────

		/// <summary>A 4x4 matrix translating by <paramref name="t"/>.</summary>
		public static Mat4 TranslationMatrix(Vec3 t)
		{
			return Mat4.FromCols(
				new Vec4(1.0, 0.0, 0.0, 0.0),
				new Vec4(0.0, 1.0, 0.0, 0.0),
				new Vec4(0.0, 0.0, 1.0, 0.0),
				new Vec4(t.X, t.Y, t.Z, 1.0));
		}

		/// <summary>A 4x4 matrix applying the rotation of <paramref name="q"/>.</summary>
		public static Mat4 RotationMatrix(Quat q)
		{
			return Mat4.FromCols(
				Vec4.From(QXDir(q), 0.0),
				Vec4.From(QYDir(q), 0.0),
				Vec4.From(QZDir(q), 0.0),
				new Vec4(0.0, 0.0, 0.0, 1.0));
		}

		/// <summary>A 4x4 matrix scaling by <paramref name="s"/> per axis.</summary>
		public static Mat4 ScalingMatrix(Vec3 s)
		{
			return Mat4.FromCols(
				new Vec4(s.X, 0.0, 0.0, 0.0),
				new Vec4(0.0, s.Y, 0.0, 0.0),
				new Vec4(0.0, 0.0, s.Z, 0.0),
				new Vec4(0.0, 0.0, 0.0, 1.0));
		}

		/// <summary>A 4x4 matrix combining rotation <paramref name="q"/> with position <paramref name="p"/>.</summary>
		public static Mat4 PoseMatrix(Quat q, Vec3 p)
		{
			return Mat4.FromCols(
				Vec4.From(QXDir(q), 0.0),
				Vec4.From(QYDir(q), 0.0),
				Vec4.From(QZDir(q), 0.0),
				new Vec4(p.X, p.Y, p.Z, 1.0));
		}

		/// <summary>Outer product: vec3 (x) vec3 -&gt; <see cref="Mat3"/>.</summary>
		public static Mat3 OuterProd(Vec3 a, Vec3 b)
		{
			return Mat3.FromCols(a * b.X, a * b.Y, a * b.Z);
		}
	}
}
