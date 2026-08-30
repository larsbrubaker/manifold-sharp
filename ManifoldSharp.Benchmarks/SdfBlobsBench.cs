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

// SdfBlobsBench.cs — the metaball level set as a driver.
//
// manifold-rust measures this as the test `test_cpp_sdf_blobs` (C++ TEST(SDF, Blobs)),
// source of its "sdf_blobs 5.6s release, 3.5s with --features parallel" figures. The
// SDF, the bounds, the 0.05 edge length and the genus assertion are
// ManifoldSdfTests.CppSdfBlobs's, unchanged.
//
// This is the one benchmark in the set that is not a boolean: it is 10 metaballs
// evaluated over a 200^3 voxel grid, so it measures the SDF fill and marching-cube
// path instead. It is also the driver where the MANIFOLD_PARALLEL column has the most
// to say, since the voxel fill is one of the six sites the Rust `parallel` feature
// blesses by name.
//
// Run: dotnet run -c Release --project ManifoldSharp.Benchmarks -- sdf-blobs

using System.Globalization;

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Benchmarks
{
	/// <summary>Metaball level set at edge length 0.05.</summary>
	internal static class SdfBlobsBench
	{
		/// <summary>Run the driver once.</summary>
		/// <param name="args">Unused; the sample has no size parameter.</param>
		/// <returns>One timing sample.</returns>
		public static IReadOnlyList<Sample> Run(string[] args)
		{
			_ = args;

			double blend = 1.0;
			double[][] balls =
			{
				new[] { 0.0, 0.0, 0.0, 2.0 },
				new[] { 1.0, 2.0, 3.0, 2.0 },
				new[] { -2.0, 2.0, -2.0, 1.0 },
				new[] { -2.0, -3.0, -2.0, 2.0 },
				new[] { -3.0, -1.0, -3.0, 1.0 },
				new[] { 2.0, -3.0, -2.0, 2.0 },
				new[] { -2.0, 3.0, 2.0, 2.0 },
				new[] { -2.0, -3.0, 2.0, 2.0 },
				new[] { 1.0, -1.0, 1.0, -2.0 },
				new[] { -4.0, -3.0, -2.0, 1.0 },
			};

			long start = System.Diagnostics.Stopwatch.GetTimestamp();
			Manifold blobs = Manifold.LevelSetWithLevel(
				p =>
				{
					double d = 0.0;
					foreach (double[] ball in balls)
					{
						Vec3 center = new Vec3(ball[0], ball[1], ball[2]);
						double w = ball[3];
						double sign = w > 0.0 ? 1.0 : -1.0;
						Vec3 diff = p - center;
						double dist = Math.Sqrt((diff.X * diff.X) + (diff.Y * diff.Y) + (diff.Z * diff.Z));
						d += sign * Types.Smoothstep(-blend, blend, Math.Abs(w) - dist);
					}

					return d;
				},
				new Box(new Vec3(-5.0, -5.0, -5.0), new Vec3(5.0, 5.0, 5.0)),
				0.05,
				0.5);
			double elapsed = DriverArgs.SecondsSince(start);

			// C++ computes genus = 1 - chi/2 where chi = NumVert - NumTri/2
			int chi = blobs.NumVert() - (blobs.NumTri() / 2);
			int genus = 1 - (chi / 2);
			if (blobs.Status() != Error.NoError || blobs.IsEmpty() || genus != 0)
			{
				throw new InvalidOperationException(string.Create(
					CultureInfo.InvariantCulture,
					$"SdfBlobs: status={blobs.Status()}, empty={blobs.IsEmpty()}, genus={genus} — expected NoError, false, 0"));
			}

			Console.WriteLine(string.Create(
				CultureInfo.InvariantCulture,
				$"nTri = {blobs.NumTri()}, time = {elapsed} sec"));

			return new[] { new Sample("sdf-blobs", elapsed, "sec") };
		}
	}
}
