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

// ParallelismTests.ConvexDilation.cs — the determinism tests for ConvexDilation.cs's
// two maps, dilating and eroding (leaves, each building its own hulls, and tree levels), which reach Par through
// Progress.MaybeParMapCtProgress. A separate file of the same class so it can share
// RunWith and AssertSameGeometry; ParallelismTests.cs's header covers the method.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ParallelismTests
	{
		/// <summary>
		/// The convex dilation's union tree gives the same geometry with the parallel
		/// switch on and off.
		/// </summary>
		/// <remarks>
		/// Carries the Minkowski hull map's one documented exception and no other: mesh-ID
		/// values are minted in worker order, so run labels are not compared. Every node's
		/// operands are fixed by index, so positions and triangles must match to the bit.
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		[NotInParallel(ParallelismGlobalStateKey)]
		public async Task ConvexDilationGeometryIsBitIdenticalInParallel()
		{
			Manifold solid = ConvexDilationTests.DrilledPart(16);
			Manifold ball = Manifold.Sphere(0.3, 8);

			// Anti-vacuity: enough triangles for several leaves, so the leaf and level maps
			// both have at least two elements and really go parallel.
			await Assert.That(solid.NumTri()).IsGreaterThanOrEqualTo(64);

			MeshGL64 sequential = RunWith(false, () => Dilate(solid, ball));
			MeshGL64 parallel = RunWith(true, () => Dilate(solid, ball));

			await Assert.That(sequential.NumTri()).IsGreaterThan(0);
			await AssertSameGeometry("convex dilation sequential vs parallel", sequential, parallel, compareRunLabels: false);
		}

		/// <summary>
		/// The convex erosion's union tree and closing subtraction give the same geometry
		/// with the parallel switch on and off.
		/// </summary>
		/// <remarks>The same maps as the dilation, so the same single exception.</remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		[NotInParallel(ParallelismGlobalStateKey)]
		public async Task ConvexErosionTreeGeometryIsBitIdenticalInParallel()
		{
			Manifold solid = ConvexDilationTests.DrilledPart(16);
			Manifold ball = Manifold.Sphere(0.3, 8);

			await Assert.That(solid.NumTri()).IsGreaterThanOrEqualTo(64);

			MeshGL64 sequential = RunWith(false, () => Erode(solid, ball));
			MeshGL64 parallel = RunWith(true, () => Erode(solid, ball));

			await Assert.That(sequential.NumTri()).IsGreaterThan(0);
			await AssertSameGeometry("convex erosion tree sequential vs parallel", sequential, parallel, compareRunLabels: false);
		}

		private static MeshGL64 Erode(Manifold solid, Manifold tool)
		{
			if (!solid.TryErodeByConvex(tool, null, null, out Manifold result))
			{
				throw new InvalidOperationException("the drilled part is eroded by a convex ball and must not be declined");
			}

			return result.GetMeshGL64(-1);
		}

		private static MeshGL64 Dilate(Manifold solid, Manifold tool)
		{
			if (!solid.TryDilateByConvex(tool, null, null, out Manifold result))
			{
				throw new InvalidOperationException("the drilled part is non-convex ⊕ convex and must not be declined");
			}

			return result.GetMeshGL64(-1);
		}
	}
}
