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

// Port of src/manifold_tests/error_propagation.rs — all 16 of its tests, whose
// own header reads:
//
//   Tests ported from C++ TEST(Manifold, ErrorPropagation*) in manifold_test.cpp
//   Verify that operations on an errored manifold propagate the error status.
//
// This file IS the specification CLAUDE.md points at for "errors are
// a status enum on the result, not exceptions": every case here takes a manifold
// that already failed and asserts the next operation hands the same status
// forward instead of throwing, computing on garbage, or quietly returning
// NoError on an empty result.
//
// Nothing here is deferred. The four cases that once were — CalculateNormals,
// SmoothByNormals, SmoothOut and the three-way Refine — landed with
// Manifold.Smooth.cs and are below in the Rust's order.
//
// They are worth more than their four lines suggest: manifold_smooth.rs guards
// every entry point with `require_paired()` and an `is_empty()` early return,
// and these are the only tests that walk an *errored* manifold into them. A
// missing guard would show up here as a NoError status on an empty result, not
// as a crash.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ManifoldErrorPropagationTests
	{
		/// <summary>C++ TEST(Manifold, ErrorPropagationDecompose).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationDecompose()
		{
			Manifold errored = await ErroredTet();
			List<Manifold> parts = errored.Decompose();
			await Assert.That(parts.Count).IsEqualTo(1).Because($"expected 1 part, got {parts.Count}");
			await Assert.That(parts[0].Status()).IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationHull).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationHull()
		{
			Manifold errored = await ErroredTet();
			await Assert.That(errored.ConvexHull().Status()).IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationHullMulti).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationHullMulti()
		{
			Manifold errored = await ErroredTet();
			Manifold good = Manifold.Cube(Vec3.Splat(1.0), false);
			await Assert.That(Manifold.HullManifolds(new List<Manifold> { good, errored }).Status())
				.IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationSetProperties).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationSetProperties()
		{
			Manifold errored = await ErroredTet();
			Manifold outManifold = errored.SetProperties(
				1,
				(Span<double> newProps, Vec3 pos, ReadOnlySpan<double> old) => { });
			await Assert.That(outManifold.Status()).IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationCalculateCurvature).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationCalculateCurvature()
		{
			Manifold errored = await ErroredTet();
			await Assert.That(errored.CalculateCurvature(0, 1).Status()).IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationCalculateNormals).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationCalculateNormals()
		{
			Manifold errored = await ErroredTet();
			await Assert.That(errored.CalculateNormals(0, 60.0).Status())
				.IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationSmoothByNormals).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationSmoothByNormals()
		{
			Manifold errored = await ErroredTet();
			await Assert.That(errored.SmoothByNormals(0).Status())
				.IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationSmoothOut).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationSmoothOut()
		{
			Manifold errored = await ErroredTet();
			await Assert.That(errored.SmoothOut(60.0, 0.0).Status())
				.IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationRefine).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationRefine()
		{
			Manifold errored = await ErroredTet();
			await Assert.That(errored.Refine(2).Status()).IsEqualTo(Error.NonFiniteVertex);
			await Assert.That(errored.RefineToLength(0.1).Status()).IsEqualTo(Error.NonFiniteVertex);
			await Assert.That(errored.RefineToTolerance(0.1).Status()).IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationSetTolerance).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationSetTolerance()
		{
			Manifold errored = await ErroredTet();
			await Assert.That(errored.SetTolerance(0.1).Status()).IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationAsOriginal).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationAsOriginal()
		{
			Manifold errored = await ErroredTet();
			await Assert.That(errored.AsOriginal().Status()).IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationWarp).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationWarp()
		{
			Manifold errored = await ErroredTet();
			await Assert.That(errored.Warp((ref Vec3 v) => { }).Status()).IsEqualTo(Error.NonFiniteVertex);
			await Assert.That(errored.WarpBatch(verts => { }).Status()).IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationSimplify).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationSimplify()
		{
			Manifold errored = await ErroredTet();
			await Assert.That(errored.Simplify(0.0).Status()).IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationMinkowski) — added in v3.5.0 (#1659).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationMinkowski()
		{
			Manifold errored = await ErroredTet();
			Manifold good = Manifold.Cube(Vec3.Splat(1.0), false);
			await Assert.That(errored.MinkowskiSum(good).Status()).IsEqualTo(Error.NonFiniteVertex);
			await Assert.That(good.MinkowskiSum(errored).Status()).IsEqualTo(Error.NonFiniteVertex);
			await Assert.That(errored.MinkowskiDifference(good).Status()).IsEqualTo(Error.NonFiniteVertex);
			await Assert.That(good.MinkowskiDifference(errored).Status()).IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationSplitByPlane) — added in v3.5.0 (#1659).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationSplitByPlane()
		{
			Manifold errored = await ErroredTet();
			(Manifold a, Manifold b) = errored.SplitByPlane(new Vec3(0.0, 0.0, 1.0), 0.0);
			await Assert.That(a.Status()).IsEqualTo(Error.NonFiniteVertex);
			await Assert.That(b.Status()).IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>C++ TEST(Manifold, ErrorPropagationMirror) — added in v3.5.0 (#1659).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppErrorPropagationMirror()
		{
			Manifold errored = await ErroredTet();
			await Assert.That(errored.Mirror(new Vec3(1.0, 0.0, 0.0)).Status())
				.IsEqualTo(Error.NonFiniteVertex);

			// Degenerate (zero-length) normal on errored input must still propagate.
			await Assert.That(errored.Mirror(new Vec3(0.0, 0.0, 0.0)).Status())
				.IsEqualTo(Error.NonFiniteVertex);
		}

		/// <summary>
		/// Build a tetrahedron MeshGL with a NaN vertex property, giving NonFiniteVertex.
		/// </summary>
		/// <returns>The errored manifold.</returns>
		private static async Task<Manifold> ErroredTet()
		{
			MeshGL gl = ManifoldApiTests.TetGl();

			// Set vertex 1's Z property to NaN (index 7 in 5-prop layout).
			gl.VertProperties[7] = float.NaN;
			Manifold m = Manifold.FromMeshGL(gl);
			await Assert.That(m.Status())
				.IsEqualTo(Error.NonFiniteVertex)
				.Because("Precondition: expected NonFiniteVertex");
			return m;
		}
	}
}
