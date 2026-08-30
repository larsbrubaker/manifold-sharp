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

// Port of the tests module in minkowski.rs — all 3 cases, same inputs, same
// expected values, same tolerances, same order. Nothing deferred.
//
// All three exercise the convex×convex branch (plus the empty early-exit). The
// non-convex branches — the ones with the coplanarity filter and the
// Par.MaybeParMap batches — have no inline Rust test either; upstream covers
// them in manifold_tests/advanced.rs, and those six cases are now ported, in
// ManifoldAdvancedTests.cs (CppConvexConvexMinkowski, ...Difference,
// CppNonConvexConvexMinkowskiSum, ...Difference, and both
// CppNonConvexNonConvex... cases), with their analytical volumes, surface areas
// and pinned genus values. They are the real cover for this file's subject; the
// three below are the inline smoke tests minkowski.rs carries itself.
//
// Beyond those, parity is held by the differential harness the Minkowski step ran
// against the compiled manifold-rust, which includes non-convex×convex and
// non-convex×non-convex pairs.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class MinkowskiTests
	{
		[Test]
		public async Task ConvexConvexMinkowskiSum()
		{
			ManifoldImpl a = ManifoldImpl.Cube(Mat3x4.Identity());
			ManifoldImpl b = ManifoldImpl.Cube(Mat3x4.Identity());
			ManifoldImpl sum = Minkowski.Sum(a, b);
			await Assert.That(sum.NumTri())
				.IsGreaterThan(0)
				.Because("Minkowski sum should produce non-empty mesh");

			// Two unit cubes: Minkowski sum should be a 2x2x2 cube
			double vol = Math.Abs(sum.GetProperty(Property.Volume));
			await Assert.That(Math.Abs(vol - 8.0) < 0.5)
				.IsTrue()
				.Because($"Minkowski sum of two unit cubes should have volume ~8, got {vol}");
		}

		[Test]
		public async Task ConvexConvexMinkowskiDifference()
		{
			ManifoldImpl a = ManifoldImpl.Cube(
				LinalgFunctions.Mat4ToMat3x4(LinalgFunctions.ScalingMatrix(Vec3.Splat(2.0))));
			ManifoldImpl b = ManifoldImpl.Cube(LinalgFunctions.Mat4ToMat3x4(
				LinalgFunctions.TranslationMatrix(Vec3.Splat(-0.25))
				* LinalgFunctions.ScalingMatrix(Vec3.Splat(0.5))));
			ManifoldImpl diff = Minkowski.Difference(a, b);
			await Assert.That(diff.NumTri())
				.IsGreaterThan(0)
				.Because("Minkowski difference should produce non-empty mesh");
		}

		[Test]
		public async Task EmptyMinkowski()
		{
			ManifoldImpl a = ManifoldImpl.Cube(Mat3x4.Identity());
			ManifoldImpl b = new ManifoldImpl();
			ManifoldImpl sum = Minkowski.Sum(a, b);

			// If b is empty, result should be a
			await Assert.That(sum.NumTri()).IsEqualTo(a.NumTri());
		}
	}
}
