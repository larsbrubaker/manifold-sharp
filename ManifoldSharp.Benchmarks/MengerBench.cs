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

// MengerBench.cs — the Menger-sponge hull as a driver.
//
// Unlike perf_test/large_scene_test/mem_profile, this has no Rust example behind it:
// manifold-rust measures it as a TEST (`test_cpp_hull_menger_sponge`, C++ TEST(Hull,
// MengerSponge)), which is where the "menger 19.9s sequential, 9.6s with --features
// parallel" figures in manifold-rust's PORTING_PLAN come from. The geometry and the assertions
// here are that test's, taken from ManifoldHullTests.CppHullMengerSponge and its
// MengerSponge/Fractal helpers — the same code the suite runs, so a number measured
// here is a number about the tested path.
//
// The assertions are kept, not dropped for speed. A benchmark that stops checking its
// result is a benchmark that will eventually be timing an empty manifold: depth-4
// sponge minus three rotated hole-unions is exactly the kind of CSG that fails by
// producing *less* geometry, which would look like a speed-up.
//
// Run: dotnet run -c Release --project ManifoldSharp.Benchmarks -- menger [depth]

using System.Globalization;

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Benchmarks
{
	/// <summary>Convex hull of a Menger sponge — heavy CSG followed by a hull.</summary>
	internal static class MengerBench
	{
		/// <summary>Run the driver once.</summary>
		/// <param name="args">Optional single argument: the sponge recursion depth.</param>
		/// <returns>Build, hull and total timing samples.</returns>
		public static IReadOnlyList<Sample> Run(string[] args)
		{
			int depth = DriverArgs.Int(args, 0, 4);

			long start = System.Diagnostics.Stopwatch.GetTimestamp();
			Manifold sponge = MengerSponge(depth).Rotate(10.0, 20.0, 30.0);
			int spongeTri = sponge.NumTri();
			double build = DriverArgs.SecondsSince(start);

			long hullStart = System.Diagnostics.Stopwatch.GetTimestamp();
			Manifold hull = sponge.ConvexHull();
			int hullTri = hull.NumTri();
			double hullTime = DriverArgs.SecondsSince(hullStart);
			double total = DriverArgs.SecondsSince(start);

			// C++ TEST(Hull, MengerSponge): the hull of a Menger sponge is the unit cube.
			if (hullTri != 12 || Math.Abs(hull.SurfaceArea() - 6.0) >= 1e-4 || Math.Abs(hull.Volume() - 1.0) >= 1e-4)
			{
				throw new InvalidOperationException(string.Create(
					CultureInfo.InvariantCulture,
					$"MengerSponge hull tris={hullTri} sa={hull.SurfaceArea()} vol={hull.Volume()} — expected 12, 6, 1"));
			}

			Console.WriteLine(string.Create(
				CultureInfo.InvariantCulture,
				$"depth = {depth}, spongeTri = {spongeTri}, build = {build} sec, hull = {hullTime} sec, total = {total} sec"));

			return new[]
			{
				new Sample($"menger/depth={depth} build", build, "sec"),
				new Sample($"menger/depth={depth} hull", hullTime, "sec"),
				new Sample($"menger/depth={depth} total", total, "sec"),
			};
		}

		/// <summary>C++ MengerSponge(n) — recursive cubic fractal via CSG subtraction.</summary>
		/// <param name="n">The recursion depth.</param>
		/// <returns>The sponge.</returns>
		private static Manifold MengerSponge(int n)
		{
			Manifold result = Manifold.Cube(Vec3.Splat(1.0), true);
			List<Manifold> holes = new List<Manifold>();
			Fractal(holes, result, 1.0, new Vec2(0.0, 0.0), 1, n);
			Manifold hole = Manifold.BatchBoolean(holes, OpType.Add);
			result = result.Difference(hole);
			hole = hole.Rotate(90.0, 0.0, 0.0);
			result = result.Difference(hole);
			hole = hole.Rotate(0.0, 0.0, 90.0);
			return result.Difference(hole);
		}

		private static void Fractal(
			List<Manifold> holes,
			Manifold hole,
			double w,
			Vec2 position,
			int depth,
			int maxDepth)
		{
			w /= 3.0;
			holes.Add(hole
				.Scale(new Vec3(w, w, 1.0))
				.Translate(new Vec3(position.X, position.Y, 0.0)));
			if (depth == maxDepth)
			{
				return;
			}

			Vec2[] offsets =
			{
				new Vec2(-w, -w),
				new Vec2(-w, 0.0),
				new Vec2(-w, w),
				new Vec2(0.0, w),
				new Vec2(w, w),
				new Vec2(w, 0.0),
				new Vec2(w, -w),
				new Vec2(0.0, -w),
			};
			foreach (Vec2 off in offsets)
			{
				Fractal(holes, hole, w, position + off, depth + 1, maxDepth);
			}
		}
	}
}
