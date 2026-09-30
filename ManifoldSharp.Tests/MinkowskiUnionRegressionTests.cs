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

// MinkowskiUnionRegressionTests.cs — NOT A PORT; a C#-only adaptation test, SKIPPED as an
// open exact-engine item. It keeps an operand pair that shows the exact engine turning a
// folded but manifold, NoError operand into lost solid.
//
// Origin: Minkowski.Sum of Thingi10K 641145 (demo import, Sphere(0.05781898171099809,
// 12)). Before divergence ledger entry 7, triangle 109's QuickHull hull was non-convex
// (QuickHullContainmentTests.Thingi641145Triangle109SweptHullIsConvex), and unioning it
// in left a zero-thickness fin. The two fixtures are the operands of the 19th pairwise
// boolean in Minkowski.cs's reduction back then (287 and 286 verts), exported with
// GetMeshGL64. With the hulls fixed, that sum and 287448's both match ConvexDilation's
// tree (to 1e-15 and 1e-16, genus 1), so nothing in the port produces these operands any
// more; the case stays because the engine defect below is still there.
//
// The defect, shared bit for bit with manifold-rust 0.15.0 (ManifoldRust 0.5.1 returns
// the same 874-tri union, 3.9e-3 short) and with C++ SetNormalsAndCoplanar (impl.cpp
// v3.5.2): the boolean is right, and SimplifyTopology's SwapDegenerates pass loses the
// solid (skipping only that pass restores the exact volume). The coplanar flood fill
// gives a sound triangle 0.041 tall, next to the fin, the reversed normal of an
// opposite-facing coplanar seed. Projected through that normal it reads as inverted, so
// RecursiveEdgeSwap's normal path swaps its long edge. The triangle across that edge
// lies in a different plane (+z), so the swap cuts out a real wedge. Importing fixture b
// runs the same pass (FromMeshGL64), so b arrives 8.5e-4 short.
//
// Two narrow fixes were tried and rejected. Refusing opposite-facing, non-degenerate
// neighbours in the flood fill breaks CppSimplify (40 tris, not 12/20: its internal
// double wall must merge). Never swapping a triangle taller than tolerance breaks
// CppNonConvexConvexMinkowskiSum (genus 3, not 5). A fix has to tell a fin the swap
// should resolve from a sound triangle that only looks inverted through the normal it
// inherited.

using System.Globalization;

using ManifoldSharp;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class MinkowskiUnionRegressionTests
	{
		/// <summary>
		/// A union contains both operands: nothing of either may lie outside it.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[Skip("Open exact-engine item: SwapDegenerates cuts solid next to a zero-thickness fin (shared with manifold-rust and C++); see the file header")]
		public async Task ExactUnionOfThingi641145PartialUnionsContainsBothOperands()
		{
			Manifold a = LoadFixture("minkowski-641145-union-a.txt");
			Manifold b = LoadFixture("minkowski-641145-union-b.txt");

			Manifold union = a.UnionWithEngine(b, BooleanEngine.Exact);
			Manifold intersection = a.IntersectionWithEngine(b, BooleanEngine.Exact);
			double expected = a.Volume() + b.Volume() - intersection.Volume();

			await Assert.That(a.Status()).IsEqualTo(Error.NoError);
			await Assert.That(b.Status()).IsEqualTo(Error.NoError);
			await Assert.That(union.Status()).IsEqualTo(Error.NoError);

			// Measured: 3.9e-3 relative short, 4.2e-4 of B outside the union.
			await Assert.That(Math.Abs(union.Volume() - expected) / expected).IsLessThan(1e-9);
			await Assert.That((b - union).Volume()).IsLessThan(1e-12);
			await Assert.That((a - union).Volume()).IsLessThan(1e-12);
		}

		/// <summary>
		/// Reads a fixture of "numVert numTri", then one "x y z" line per vertex (round-trip
		/// doubles) and one "v0 v1 v2" line per triangle, as a MeshGL64.
		/// </summary>
		/// <param name="fileName">The fixture's file name under TestData/regressions.</param>
		/// <returns>The imported manifold.</returns>
		private static Manifold LoadFixture(string fileName)
		{
			string path = Path.Combine(AppContext.BaseDirectory, "TestData", "regressions", fileName);
			string[] lines = File.ReadAllLines(path);
			string[] header = lines[0].Split(' ');
			int numVert = int.Parse(header[0], CultureInfo.InvariantCulture);
			int numTri = int.Parse(header[1], CultureInfo.InvariantCulture);

			MeshGL64 mesh = new MeshGL64();
			mesh.NumProp = 3;
			mesh.VertProperties = new List<double>(3 * numVert);
			mesh.TriVerts = new List<ulong>(3 * numTri);
			for (int v = 0; v < numVert; v++)
			{
				foreach (string x in lines[1 + v].Split(' '))
				{
					mesh.VertProperties.Add(double.Parse(x, CultureInfo.InvariantCulture));
				}
			}

			for (int t = 0; t < numTri; t++)
			{
				foreach (string x in lines[1 + numVert + t].Split(' '))
				{
					mesh.TriVerts.Add(ulong.Parse(x, CultureInfo.InvariantCulture));
				}
			}

			return Manifold.FromMeshGL64(mesh);
		}
	}
}
