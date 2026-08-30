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

// LargeSceneTest.cs — port of manifold-rust examples/large_scene_test.rs. That file's
// header, which explains the one structural liberty both ports take:
//
//   Rust port of cpp-reference/manifold/extras/large_scene_test.cpp.
//
//   Unions an n x n x n grid of unit spheres (minus the origin one) and prints
//   the same "nTri = N, time = S sec" line as the C++ driver. C++ builds a lazy
//   CSG tree via repeated `scene.Boolean(sphere, Add)` and evaluates once at
//   NumTri(); on evaluation the nested Adds collapse into one n-ary union that
//   BatchUnion processes. We build that collapsed form directly with
//   CsgNode::op_n — same evaluation path (BatchUnion + Compose + heap-ordered
//   BatchBoolean), without recursing through ~n^3 nested tree levels (the C++
//   collapse uses an explicit stack; our collect_children recurses per level and
//   a left-deep 8000-node chain would risk stack overflow — flat n-ary is the
//   post-collapse equivalent).
//
//   Run with: cargo run --release --example large_scene_test [n]
//
// Here: `dotnet run -c Release --project ManifoldSharp.Benchmarks -- large-scene [n]`.
//
// The Rust shares one mesh across all n^3-1 leaves through an `Arc`, "as C++ shares the
// Impl via shared_ptr". A C# reference is that sharing already, so the Arc has no
// counterpart: `CsgLeafNode` takes the impl by reference and never copies it (see its
// remarks — the port relies on nothing mutating a mesh after intake, which is exactly
// what `Arc<ManifoldImpl>` enforces on the Rust side). Sharing rather than copying is
// not a micro-optimization here: at n=20 it is the difference between one sphere in
// memory and 7 999 of them, and it is what makes the measured number comparable.

using System.Globalization;

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Benchmarks
{
	/// <summary>Union of an n-cubed grid of unit spheres, minus the origin one.</summary>
	internal static class LargeSceneTest
	{
		/// <summary>Run the driver once.</summary>
		/// <param name="args">Optional single argument: the grid edge length n.</param>
		/// <returns>One timing sample.</returns>
		public static IReadOnlyList<Sample> Run(string[] args)
		{
			int n = DriverArgs.Int(args, 0, 20);

			Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"n = {n}"));

			long start = System.Diagnostics.Stopwatch.GetTimestamp();

			// One template sphere; each grid instance is a leaf carrying a lazy
			// translation, exactly like C++ Sphere(1).Translate(...) before evaluation.
			Manifold sphere = Manifold.Sphere(1.0, 0);

			// Share one mesh across all leaves, as C++ shares the Impl via shared_ptr.
			ManifoldImpl sphereImpl = sphere.AsImpl().Clone();

			List<CsgNode> leaves = new List<CsgNode>();
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
				{
					for (int k = 0; k < n; k++)
					{
						if (i == 0 && j == 0 && k == 0)
						{
							continue;
						}

						CsgLeafNode leaf = new CsgLeafNode(
							sphereImpl,
							LinalgFunctions.Mat4ToMat3x4(
								LinalgFunctions.TranslationMatrix(new Vec3(i, j, k))));
						leaves.Add(new CsgLeaf(leaf));
					}
				}
			}

			ManifoldImpl scene = new CsgOp(OpType.Add, leaves).Evaluate();
			int nTri = scene.NumTri();
			double elapsed = DriverArgs.SecondsSince(start);
			GC.KeepAlive(scene);
			Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"nTri = {nTri}, time = {elapsed} sec"));

			return new[] { new Sample($"large-scene/n={n}", elapsed, "sec") };
		}
	}
}
