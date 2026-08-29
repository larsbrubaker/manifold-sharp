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

// Port of the tests module in svd.rs — all 5 cases, same expected values, same
// 1e-8 tolerance, in the same order.

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using ManifoldSharp.Linalg;

using static ManifoldSharp.SvdFunctions;

namespace ManifoldSharp.Tests
{
	public class SvdTests
	{
		private static bool ApproxEq(double a, double b)
		{
			return Math.Abs(a - b) < 1e-8;
		}

		private static bool ApproxVec3(Vec3 a, Vec3 b)
		{
			return ApproxEq(a.X, b.X) && ApproxEq(a.Y, b.Y) && ApproxEq(a.Z, b.Z);
		}

		[Test]
		public async Task TestSvdIdentity()
		{
			SVDSet outSet = Svd(Mat3.Identity());
			await Assert.That(ApproxVec3(outSet.S.X, new Vec3(1.0, 0.0, 0.0))).IsTrue();
			await Assert.That(ApproxVec3(outSet.S.Y, new Vec3(0.0, 1.0, 0.0))).IsTrue();
			await Assert.That(ApproxVec3(outSet.S.Z, new Vec3(0.0, 0.0, 1.0))).IsTrue();
		}

		[Test]
		public async Task TestSvdDiagonal()
		{
			Mat3 a = Mat3.FromCols(
				new Vec3(3.0, 0.0, 0.0),
				new Vec3(0.0, 2.0, 0.0),
				new Vec3(0.0, 0.0, 1.0));
			SVDSet outSet = Svd(a);
			await Assert.That(ApproxEq(outSet.S[0][0], 3.0)).IsTrue();
			await Assert.That(ApproxEq(outSet.S[1][1], 2.0)).IsTrue();
			await Assert.That(ApproxEq(outSet.S[2][2], 1.0)).IsTrue();
		}

		[Test]
		public async Task TestSvdReconstruction()
		{
			Mat3 a = Mat3.FromCols(
				new Vec3(1.0, 2.0, 3.0),
				new Vec3(0.5, -1.0, 4.0),
				new Vec3(-2.0, 0.25, 1.5));
			SVDSet outSet = Svd(a);
			Mat3 reconstructed = outSet.U * outSet.S * outSet.V.Transpose();
			for (int col = 0; col < 3; col++)
			{
				for (int row = 0; row < 3; row++)
				{
					await Assert.That(ApproxEq(reconstructed[col][row], a[col][row]))
						.IsTrue()
						.Because($"mismatch at ({col}, {row}): {reconstructed[col][row]} vs {a[col][row]}");
				}
			}
		}

		[Test]
		public async Task TestSpectralNorm()
		{
			Mat3 a = Mat3.FromCols(
				new Vec3(4.0, 0.0, 0.0),
				new Vec3(0.0, 2.0, 0.0),
				new Vec3(0.0, 0.0, 1.0));
			await Assert.That(ApproxEq(SpectralNorm(a), 4.0)).IsTrue();
		}

		[Test]
		public async Task TestSingularValuesSorted()
		{
			Mat3 a = Mat3.FromCols(
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, 5.0, 0.0),
				new Vec3(0.0, 0.0, 3.0));
			SVDSet outSet = Svd(a);
			await Assert.That(Math.Abs(outSet.S[0][0]) >= Math.Abs(outSet.S[1][1])).IsTrue();
			await Assert.That(Math.Abs(outSet.S[1][1]) >= Math.Abs(outSet.S[2][2])).IsTrue();
		}
	}
}
