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

// Port of linalg_tests.rs — all 38 cases, same expected values, in the same
// order and with the same strictness. Where the Rust uses assert_eq! on doubles
// the port compares exactly; the EPS-based approx helpers are only used where
// the Rust used them.
//
// Two deviations inside the ported 38, both commented at the site: the
// reflexivity check goes through a copy (C# rejects a literal self-comparison,
// CS1718), and IVec3Ord sorts with LINQ OrderBy rather than List<T>.Sort,
// because that case is about stability and List<T>.Sort is an unstable
// introsort.
//
// After the 38 comes a clearly marked C#-only section of port regressions that
// have no Rust counterpart, guarding failure modes the Rust type system made
// impossible: ref-indexer mutation, and the signed-zero tie in MinF64/MaxF64.

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using ManifoldSharp.Linalg;

using Quat = ManifoldSharp.Linalg.Vec4;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp.Tests
{
	public class LinalgTests
	{
		private const double EPS = 1e-14;

		private static bool ApproxEq(double a, double b)
		{
			return Math.Abs(a - b) < EPS;
		}

		private static bool ApproxEq3(Vec3 a, Vec3 b)
		{
			return ApproxEq(a.X, b.X) && ApproxEq(a.Y, b.Y) && ApproxEq(a.Z, b.Z);
		}

		private static bool ApproxEq4(Vec4 a, Vec4 b)
		{
			return ApproxEq(a.X, b.X) && ApproxEq(a.Y, b.Y) && ApproxEq(a.Z, b.Z) && ApproxEq(a.W, b.W);
		}

		// ── Vec2 ──────────────────────────────────────────────────────────────

		[Test]
		public async Task Vec2New()
		{
			Vec2 v = new Vec2(1.0, 2.0);
			await Assert.That(v.X).IsEqualTo(1.0);
			await Assert.That(v.Y).IsEqualTo(2.0);
		}

		[Test]
		public async Task Vec2Splat()
		{
			Vec2 v = Vec2.Splat(3.0);
			await Assert.That(v).IsEqualTo(new Vec2(3.0, 3.0));
		}

		[Test]
		public async Task Vec2Ops()
		{
			Vec2 a = new Vec2(1.0, 2.0);
			Vec2 b = new Vec2(3.0, 4.0);
			await Assert.That(a + b).IsEqualTo(new Vec2(4.0, 6.0));
			await Assert.That(b - a).IsEqualTo(new Vec2(2.0, 2.0));
			await Assert.That(a * b).IsEqualTo(new Vec2(3.0, 8.0)); // element-wise
			await Assert.That(a * 2.0).IsEqualTo(new Vec2(2.0, 4.0));
			await Assert.That(2.0 * a).IsEqualTo(new Vec2(2.0, 4.0));
			await Assert.That(-a).IsEqualTo(new Vec2(-1.0, -2.0));
			await Assert.That(a / 2.0).IsEqualTo(new Vec2(0.5, 1.0));
		}

		[Test]
		public async Task Vec2Index()
		{
			Vec2 v = new Vec2(5.0, 6.0);
			await Assert.That(v[0]).IsEqualTo(5.0);
			await Assert.That(v[1]).IsEqualTo(6.0);
		}

		[Test]
		public async Task Vec2Ordering()
		{
			Vec2 a = new Vec2(1.0, 2.0);
			Vec2 b = new Vec2(1.0, 3.0);
			Vec2 c = new Vec2(2.0, 0.0);
			await Assert.That(a < b).IsTrue(); // same x, y differs
			await Assert.That(b < c).IsTrue(); // x differs
			// The Rust writes `assert!(a == a)`; C# rejects a literal self-comparison
			// (CS1718), so the reflexivity check goes through a copy of the same value.
			Vec2 aCopy = a;
			await Assert.That(a == aCopy).IsTrue();
		}

		[Test]
		public async Task Cross2Test()
		{
			// cross({1,0}, {0,1}) = 1
			await Assert.That(Cross2(new Vec2(1.0, 0.0), new Vec2(0.0, 1.0))).IsEqualTo(1.0);
			// cross({1,0}, {1,0}) = 0
			await Assert.That(Cross2(new Vec2(1.0, 0.0), new Vec2(1.0, 0.0))).IsEqualTo(0.0);
		}

		[Test]
		public async Task Dot2Test()
		{
			await Assert.That(Dot(new Vec2(1.0, 2.0), new Vec2(3.0, 4.0))).IsEqualTo(11.0);
		}

		// ── Vec3 ──────────────────────────────────────────────────────────────

		[Test]
		public async Task Vec3New()
		{
			Vec3 v = new Vec3(1.0, 2.0, 3.0);
			await Assert.That(v.X).IsEqualTo(1.0);
			await Assert.That(v.Y).IsEqualTo(2.0);
			await Assert.That(v.Z).IsEqualTo(3.0);
		}

		[Test]
		public async Task Vec3Ops()
		{
			Vec3 a = new Vec3(1.0, 2.0, 3.0);
			Vec3 b = new Vec3(4.0, 5.0, 6.0);
			await Assert.That(a + b).IsEqualTo(new Vec3(5.0, 7.0, 9.0));
			await Assert.That(b - a).IsEqualTo(new Vec3(3.0, 3.0, 3.0));
			await Assert.That(a * 2.0).IsEqualTo(new Vec3(2.0, 4.0, 6.0));
			await Assert.That(-a).IsEqualTo(new Vec3(-1.0, -2.0, -3.0));
		}

		[Test]
		public async Task Dot3Test()
		{
			await Assert.That(Dot(new Vec3(1.0, 2.0, 3.0), new Vec3(4.0, 5.0, 6.0))).IsEqualTo(32.0);
		}

		[Test]
		public async Task Cross3Test()
		{
			Vec3 a = new Vec3(1.0, 0.0, 0.0);
			Vec3 b = new Vec3(0.0, 1.0, 0.0);
			Vec3 c = Cross(a, b);
			await Assert.That(c).IsEqualTo(new Vec3(0.0, 0.0, 1.0));

			Vec3 x = Cross(new Vec3(0.0, 1.0, 0.0), new Vec3(0.0, 0.0, 1.0));
			await Assert.That(x).IsEqualTo(new Vec3(1.0, 0.0, 0.0));
		}

		[Test]
		public async Task Length3Test()
		{
			Vec3 v = new Vec3(3.0, 4.0, 0.0);
			await Assert.That(Length(v)).IsEqualTo(5.0);
			await Assert.That(LengthSquared(v)).IsEqualTo(25.0);
		}

		[Test]
		public async Task Normalize3Test()
		{
			Vec3 v = new Vec3(3.0, 0.0, 0.0);
			Vec3 n = Normalize(v);
			await Assert.That(ApproxEq3(n, new Vec3(1.0, 0.0, 0.0))).IsTrue();
			await Assert.That(ApproxEq(Length(n), 1.0)).IsTrue();
		}

		[Test]
		public async Task Vec3MinMax()
		{
			Vec3 a = new Vec3(1.0, 5.0, 3.0);
			Vec3 b = new Vec3(4.0, 2.0, 6.0);
			await Assert.That(Min(a, b)).IsEqualTo(new Vec3(1.0, 2.0, 3.0));
			await Assert.That(Max(a, b)).IsEqualTo(new Vec3(4.0, 5.0, 6.0));
		}

		[Test]
		public async Task Vec3OrderingLex()
		{
			Vec3 a = new Vec3(1.0, 2.0, 3.0);
			Vec3 b = new Vec3(1.0, 2.0, 4.0);
			Vec3 c = new Vec3(1.0, 3.0, 0.0);
			Vec3 d = new Vec3(2.0, 0.0, 0.0);
			await Assert.That(a < b).IsTrue();
			await Assert.That(b < c).IsTrue();
			await Assert.That(c < d).IsTrue();
		}

		// ── Vec4 / Quat ───────────────────────────────────────────────────────

		[Test]
		public async Task Vec4New()
		{
			Vec4 v = new Vec4(1.0, 2.0, 3.0, 4.0);
			await Assert.That(v.Xyz()).IsEqualTo(new Vec3(1.0, 2.0, 3.0));
			await Assert.That(v.W).IsEqualTo(4.0);
		}

		[Test]
		public async Task Dot4Test()
		{
			Vec4 a = new Vec4(1.0, 2.0, 3.0, 4.0);
			Vec4 b = new Vec4(1.0, 1.0, 1.0, 1.0);
			await Assert.That(Dot(a, b)).IsEqualTo(10.0);
		}

		[Test]
		public async Task QMulIdentity()
		{
			Quat id = new Quat(0.0, 0.0, 0.0, 1.0);
			Quat q = new Quat(0.1, 0.2, 0.3, 0.9274);
			Quat result = QMul(id, q);
			await Assert.That(ApproxEq4(result, q)).IsTrue();
		}

		[Test]
		public async Task QRotIdentity()
		{
			Quat id = new Quat(0.0, 0.0, 0.0, 1.0);
			Vec3 v = new Vec3(1.0, 2.0, 3.0);
			Vec3 result = QRot(id, v);
			await Assert.That(ApproxEq3(result, v)).IsTrue();
		}

		[Test]
		public async Task QRot90Z()
		{
			// 90 degree rotation around Z: maps X to Y
			Quat q = RotationQuatAxisAngle(new Vec3(0.0, 0.0, 1.0), Math.PI / 2.0);
			Vec3 v = QRot(q, new Vec3(1.0, 0.0, 0.0));
			await Assert.That(ApproxEq3(v, new Vec3(0.0, 1.0, 0.0))).IsTrue();
		}

		[Test]
		public async Task QMatIdentity()
		{
			Quat id = new Quat(0.0, 0.0, 0.0, 1.0);
			Mat3 m = QMat(id);
			await Assert.That(ApproxEq3(m.X, new Vec3(1.0, 0.0, 0.0))).IsTrue();
			await Assert.That(ApproxEq3(m.Y, new Vec3(0.0, 1.0, 0.0))).IsTrue();
			await Assert.That(ApproxEq3(m.Z, new Vec3(0.0, 0.0, 1.0))).IsTrue();
		}

		// ── Integer vectors ───────────────────────────────────────────────────

		[Test]
		public async Task IVec3Test()
		{
			IVec3 v = new IVec3(1, 2, 3);
			await Assert.That(v[0]).IsEqualTo(1);
			await Assert.That(v[1]).IsEqualTo(2);
			await Assert.That(v[2]).IsEqualTo(3);
		}

		[Test]
		public async Task IVec3Ord()
		{
			IVec3 a = new IVec3(1, 2, 3);
			IVec3 b = new IVec3(1, 2, 4);
			IVec3 c = new IVec3(1, 3, 0);
			await Assert.That(a < b).IsTrue();
			await Assert.That(b < c).IsTrue();

			// sort stability
			//
			// The Rust is `v.sort()`, which is stable, and this case is *about* sort
			// stability — so the port must not use List<T>.Sort, an unstable introsort.
			// Per the repo's stable-sort rule this goes through LINQ OrderBy, which is
			// documented stable.
			List<IVec3> v = new List<IVec3> { c, a, b }.OrderBy(x => x).ToList();
			// Asserted element by element rather than as a collection, so that the check
			// is unambiguously order-sensitive — the Rust compares the whole Vec.
			await Assert.That(v[0]).IsEqualTo(a);
			await Assert.That(v[1]).IsEqualTo(b);
			await Assert.That(v[2]).IsEqualTo(c);
		}

		// ── Mat3 ──────────────────────────────────────────────────────────────

		[Test]
		public async Task Mat3Identity()
		{
			Mat3 m = Mat3.Identity();
			Vec3 v = new Vec3(1.0, 2.0, 3.0);
			await Assert.That(m * v).IsEqualTo(v);
		}

		[Test]
		public async Task Mat3MulVec()
		{
			// Scale by 2
			Mat3 m = Mat3.FromCols(
				new Vec3(2.0, 0.0, 0.0),
				new Vec3(0.0, 2.0, 0.0),
				new Vec3(0.0, 0.0, 2.0));
			Vec3 v = new Vec3(1.0, 2.0, 3.0);
			await Assert.That(m * v).IsEqualTo(new Vec3(2.0, 4.0, 6.0));
		}

		[Test]
		public async Task Mat3Transpose()
		{
			Mat3 m = Mat3.FromCols(
				new Vec3(1.0, 2.0, 3.0),
				new Vec3(4.0, 5.0, 6.0),
				new Vec3(7.0, 8.0, 9.0));
			Mat3 t = m.Transpose();
			// row(0) of m = col(0) of t
			await Assert.That(t.X).IsEqualTo(new Vec3(1.0, 4.0, 7.0));
			await Assert.That(t.Y).IsEqualTo(new Vec3(2.0, 5.0, 8.0));
			await Assert.That(t.Z).IsEqualTo(new Vec3(3.0, 6.0, 9.0));
		}

		[Test]
		public async Task Mat3Determinant()
		{
			Mat3 m = Mat3.Identity();
			await Assert.That(m.Determinant()).IsEqualTo(1.0);

			// det of [[1,2,3],[0,1,4],[5,6,0]] (col-major: cols are [1,0,5],[2,1,6],[3,4,0])
			Mat3 m2 = Mat3.FromCols(
				new Vec3(1.0, 0.0, 5.0),
				new Vec3(2.0, 1.0, 6.0),
				new Vec3(3.0, 4.0, 0.0));
			// det([[1,2,3],[0,1,4],[5,6,0]]) expanding along row 0:
			// = 1*(1*0-4*6) - 2*(0*0-4*5) + 3*(0*6-1*5) = -24+40-15 = 1
			await Assert.That(m2.Determinant()).IsEqualTo(
				1.0 * (1.0 * 0.0 - 4.0 * 6.0) - 2.0 * (0.0 * 0.0 - 4.0 * 5.0)
					+ 3.0 * (0.0 * 6.0 - 1.0 * 5.0));
			await Assert.That(m2.Determinant()).IsEqualTo(1.0);
		}

		[Test]
		public async Task Mat3Inverse()
		{
			Mat3 m = Mat3.FromCols(
				new Vec3(2.0, 0.0, 0.0),
				new Vec3(0.0, 3.0, 0.0),
				new Vec3(0.0, 0.0, 4.0));
			Mat3 inv = m.Inverse();
			Mat3 product = m * inv;
			Mat3 id = Mat3.Identity();
			for (int j = 0; j < 3; j++)
			{
				for (int i = 0; i < 3; i++)
				{
					await Assert.That(Math.Abs(product[j][i] - id[j][i]) < EPS)
						.IsTrue()
						.Because($"product[{j}][{i}] = {product[j][i]}");
				}
			}
		}

		// ── Mat4 ──────────────────────────────────────────────────────────────

		[Test]
		public async Task Mat4Identity()
		{
			Mat4 m = Mat4.Identity();
			Vec4 v = new Vec4(1.0, 2.0, 3.0, 4.0);
			await Assert.That(m * v).IsEqualTo(v);
		}

		[Test]
		public async Task Mat4DeterminantIdentity()
		{
			await Assert.That(Math.Abs(Mat4.Identity().Determinant() - 1.0) < EPS).IsTrue();
		}

		[Test]
		public async Task Mat4Inverse()
		{
			Mat4 m = TranslationMatrix(new Vec3(1.0, 2.0, 3.0));
			Mat4 inv = m.Inverse();
			Mat4 product = m * inv;
			Mat4 id = Mat4.Identity();
			for (int j = 0; j < 4; j++)
			{
				for (int i = 0; i < 4; i++)
				{
					await Assert.That(Math.Abs(product[j][i] - id[j][i]) < EPS).IsTrue();
				}
			}
		}

		// ── Mat3x4 ────────────────────────────────────────────────────────────

		[Test]
		public async Task Mat3x4IdentityTransform()
		{
			Mat3x4 m = Mat3x4.Identity();
			// Apply to homogeneous point (x,y,z,1)
			Vec3 p = new Vec3(1.0, 2.0, 3.0);
			Vec4 hp = new Vec4(p.X, p.Y, p.Z, 1.0);
			Vec3 result = m * hp;
			await Assert.That(ApproxEq3(result, p)).IsTrue();
		}

		[Test]
		public async Task Mat3x4Translation()
		{
			// Translation by (1,2,3)
			Vec3 t = new Vec3(1.0, 2.0, 3.0);
			Mat3x4 m = Mat3x4.FromCols(
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 1.0, 0.0),
				new Vec3(0.0, 0.0, 1.0),
				t);
			Vec4 p = new Vec4(0.0, 0.0, 0.0, 1.0);
			Vec3 result = m * p;
			await Assert.That(ApproxEq3(result, t)).IsTrue();
		}

		// ── Factory functions ─────────────────────────────────────────────────

		[Test]
		public async Task TranslationMatrixTest()
		{
			Vec3 t = new Vec3(1.0, 2.0, 3.0);
			Mat4 m = TranslationMatrix(t);
			Vec4 p = new Vec4(0.0, 0.0, 0.0, 1.0);
			Vec4 result = m * p;
			await Assert.That(ApproxEq4(result, new Vec4(1.0, 2.0, 3.0, 1.0))).IsTrue();
		}

		[Test]
		public async Task ScalingMatrixTest()
		{
			Vec3 s = new Vec3(2.0, 3.0, 4.0);
			Mat4 m = ScalingMatrix(s);
			Vec4 v = new Vec4(1.0, 1.0, 1.0, 1.0);
			Vec4 result = m * v;
			await Assert.That(ApproxEq4(result, new Vec4(2.0, 3.0, 4.0, 1.0))).IsTrue();
		}

		// ── Reductions ────────────────────────────────────────────────────────

		[Test]
		public async Task MinMaxElem()
		{
			Vec3 v = new Vec3(3.0, 1.0, 2.0);
			await Assert.That(MinElem(v)).IsEqualTo(1.0);
			await Assert.That(MaxElem(v)).IsEqualTo(3.0);
			await Assert.That(ArgMin(v)).IsEqualTo(1);
			await Assert.That(ArgMax(v)).IsEqualTo(0);
		}

		// ── RotationQuatVec ───────────────────────────────────────────────────

		[Test]
		public async Task RotationQuatVecIdentity()
		{
			Quat q = RotationQuatVec(new Vec3(1.0, 0.0, 0.0), new Vec3(1.0, 0.0, 0.0));
			// should be identity quaternion
			await Assert.That(ApproxEq4(q, new Quat(0.0, 0.0, 0.0, 1.0))).IsTrue();
		}

		[Test]
		public async Task RotationQuatVecXToY()
		{
			Quat q = RotationQuatVec(new Vec3(1.0, 0.0, 0.0), new Vec3(0.0, 1.0, 0.0));
			Vec3 v = QRot(q, new Vec3(1.0, 0.0, 0.0));
			// Should rotate x to y
			await Assert.That(Math.Abs(v.X) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(v.Y - 1.0) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(v.Z) < 1e-10).IsTrue();
		}

		// ══ C#-only port regressions: ref-indexer mutation ═════════════════════
		//
		// Nothing below has a counterpart in linalg_tests.rs. The Rust gets `v[i] = x`
		// from IndexMut and cannot express the failure mode these guard: in C# a
		// ref-returning indexer and a by-value get/set indexer are interchangeable for
		// *reads*, so all 38 ported cases above pass either way. If the indexers ever
		// degrade — to a value-returning getter, or by having a caller route through a
		// defensive copy — writes silently land nowhere and only these cases notice.
		//
		// Each type is checked through both `x[i] = v` and a compound `x[i] += v`, and
		// verified against the named field rather than by reading the indexer back, so
		// that a get/set pair over a copy could not fake a pass.

		[Test]
		public async Task RefIndexerMutatesVec2()
		{
			Vec2 v = default;
			v[0] = 5.0;
			v[1] = 6.0;
			v[1] += 1.0;
			await Assert.That(v.X).IsEqualTo(5.0);
			await Assert.That(v.Y).IsEqualTo(7.0);
		}

		[Test]
		public async Task RefIndexerMutatesVec3()
		{
			Vec3 v = default;
			v[0] = 1.0;
			v[1] = 2.0;
			v[2] = 3.0;
			v[0] += 10.0;
			await Assert.That(v.X).IsEqualTo(11.0);
			await Assert.That(v.Y).IsEqualTo(2.0);
			await Assert.That(v.Z).IsEqualTo(3.0);
		}

		[Test]
		public async Task RefIndexerMutatesVec4()
		{
			Vec4 v = default;
			v[0] = 1.0;
			v[1] = 2.0;
			v[2] = 3.0;
			v[3] = 4.0;
			v[3] += 0.5;
			await Assert.That(v.X).IsEqualTo(1.0);
			await Assert.That(v.Y).IsEqualTo(2.0);
			await Assert.That(v.Z).IsEqualTo(3.0);
			await Assert.That(v.W).IsEqualTo(4.5);
		}

		[Test]
		public async Task RefIndexerMutatesIVec2()
		{
			IVec2 v = default;
			v[0] = 5;
			v[1] = 6;
			v[1] += 1;
			await Assert.That(v.X).IsEqualTo(5);
			await Assert.That(v.Y).IsEqualTo(7);
		}

		[Test]
		public async Task RefIndexerMutatesIVec3()
		{
			IVec3 v = default;
			v[0] = 1;
			v[1] = 2;
			v[2] = 3;
			v[0] += 10;
			await Assert.That(v.X).IsEqualTo(11);
			await Assert.That(v.Y).IsEqualTo(2);
			await Assert.That(v.Z).IsEqualTo(3);
		}

		[Test]
		public async Task RefIndexerMutatesIVec4()
		{
			IVec4 v = default;
			v[0] = 1;
			v[1] = 2;
			v[2] = 3;
			v[3] = 4;
			v[3] += 5;
			await Assert.That(v.X).IsEqualTo(1);
			await Assert.That(v.Y).IsEqualTo(2);
			await Assert.That(v.Z).IsEqualTo(3);
			await Assert.That(v.W).IsEqualTo(9);
		}

		[Test]
		public async Task RefIndexerMutatesBVec4()
		{
			BVec4 v = default;
			v[0] = true;
			v[2] = true;
			v[3] = true;
			v[3] &= false;
			await Assert.That(v.X).IsTrue();
			await Assert.That(v.Y).IsFalse();
			await Assert.That(v.Z).IsTrue();
			await Assert.That(v.W).IsFalse();
		}

		[Test]
		public async Task RefIndexerMutatesMat2Column()
		{
			Mat2 m = Mat2.Identity();
			m[0] = new Vec2(7.0, 8.0);
			m[1][0] = 9.0;
			m[1][1] += 4.0;
			await Assert.That(m.X).IsEqualTo(new Vec2(7.0, 8.0));
			await Assert.That(m.Y).IsEqualTo(new Vec2(9.0, 5.0));
		}

		[Test]
		public async Task RefIndexerMutatesMat3Column()
		{
			Mat3 m = Mat3.Identity();
			m[2][0] = 7.0;
			m[0][2] += 3.0;
			m[1] = new Vec3(4.0, 5.0, 6.0);
			await Assert.That(m.Z.X).IsEqualTo(7.0);
			await Assert.That(m.X.Z).IsEqualTo(3.0);
			await Assert.That(m.Y).IsEqualTo(new Vec3(4.0, 5.0, 6.0));
		}

		[Test]
		public async Task RefIndexerMutatesMat4Column()
		{
			Mat4 m = Mat4.Identity();
			m[3][0] = 1.0;
			m[3][1] = 2.0;
			m[3][2] = 3.0;
			m[0][0] += 4.0;
			await Assert.That(m.W).IsEqualTo(new Vec4(1.0, 2.0, 3.0, 1.0));
			await Assert.That(m.X.X).IsEqualTo(5.0);
		}

		[Test]
		public async Task RefIndexerMutatesMat3x4Column()
		{
			Mat3x4 m = Mat3x4.Identity();
			m[3][0] = 1.0;
			m[3][1] = 2.0;
			m[3][2] = 3.0;
			m[2][2] += 6.0;
			await Assert.That(m.W).IsEqualTo(new Vec3(1.0, 2.0, 3.0));
			await Assert.That(m.Z.Z).IsEqualTo(7.0);
		}

		[Test]
		public async Task RefIndexerMutatesThroughArrayElement()
		{
			// The struct-shape rule in Vec3.cs's header tells later phases to hold these
			// types in arrays rather than readonly fields, on the grounds that an array
			// element is real storage and a write through it lands. That claim is what
			// this pins.
			Vec3[] arena = new Vec3[2];
			arena[1][0] = 4.0;
			arena[1][2] += 9.0;
			await Assert.That(arena[1].X).IsEqualTo(4.0);
			await Assert.That(arena[1].Z).IsEqualTo(9.0);
			await Assert.That(arena[0].X).IsEqualTo(0.0);

			Mat3[] mats = new Mat3[1];
			mats[0][1][2] = 5.0;
			await Assert.That(mats[0].Y.Z).IsEqualTo(5.0);
		}

		[Test]
		public async Task IndexerRejectsOutOfRange()
		{
			await Assert.That(() =>
			{
				Vec3 v = default;
				return v[3];
			}).Throws<ArgumentOutOfRangeException>();

			await Assert.That(() =>
			{
				Mat3 m = default;
				return m[-1];
			}).Throws<ArgumentOutOfRangeException>();

			await Assert.That(() =>
			{
				UVec3 u = new UVec3(1, 2, 3);
				return u[3];
			}).Throws<ArgumentOutOfRangeException>();
		}

		// ══ C#-only port regressions: signed-zero min/max ties ═════════════════
		//
		// Rust documents the +0.0 / -0.0 tie in f64::min and f64::max as returning
		// "either input", which makes it target-dependent — x86 minsd returns the
		// second operand regardless of sign, arm64 FMIN is sign aware. This port pins
		// the tie to .NET's specified sign-aware behaviour, which equals the arm64 Rust
		// the oracle lane compares against. See docs/RUST_DIVERGENCES.md.

		[Test]
		public async Task MinMaxF64SignedZeroTiesAreSignAware()
		{
			await Assert.That(double.IsNegative(MinF64(-0.0, 0.0))).IsTrue();
			await Assert.That(double.IsNegative(MinF64(0.0, -0.0))).IsTrue();
			await Assert.That(double.IsNegative(MaxF64(-0.0, 0.0))).IsFalse();
			await Assert.That(double.IsNegative(MaxF64(0.0, -0.0))).IsFalse();
		}

		[Test]
		public async Task MinMaxF64NaNLoses()
		{
			// Rust's f64::min/max return the non-NaN operand; System.Math.Min/Max
			// propagate NaN, which is why MinF64/MaxF64 screen for it first.
			await Assert.That(MinF64(double.NaN, 1.0)).IsEqualTo(1.0);
			await Assert.That(MinF64(1.0, double.NaN)).IsEqualTo(1.0);
			await Assert.That(MaxF64(double.NaN, 1.0)).IsEqualTo(1.0);
			await Assert.That(MaxF64(1.0, double.NaN)).IsEqualTo(1.0);
			await Assert.That(double.IsNaN(MinF64(double.NaN, double.NaN))).IsTrue();
		}
	}
}
