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

// KeyholeBench.cs — manifold-rust examples/keyhole_perf.rs, whose header reads:
//
//   Timing driver for keyholing (CutKeyhole / FindCloserBridge) in the ear
//   clipper. `holes`: one square with an N x N grid of 32-gon holes. `glyphs`:
//   R x C square outer rings, each with one or two octagonal holes. Prints the
//   triangle count and an FNV-1a hash of the triangle list next to the time.
//
// The Rust example takes its own repeat count and prints the best of them; here
// that count is the harness's `--reps`, so one run is one triangulation and the
// harness reports best-of-N. The printed line otherwise keeps the Rust format, so
// the hash can be diffed against `cargo run --release --example keyhole_perf`.
//
// Run: dotnet run -c Release --project ManifoldSharp.Benchmarks -- keyhole [holes N | glyphs R C] --reps 3

using System.Globalization;

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Benchmarks
{
	/// <summary>Times the ear clipper's keyholing on hole and glyph grids.</summary>
	internal static class KeyholeBench
	{
		/// <summary>Run the driver once.</summary>
		/// <param name="args">Optional: <c>holes N</c> (default 71) or <c>glyphs R C</c> (default 100 100).</param>
		/// <returns>One timing sample.</returns>
		public static IReadOnlyList<Sample> Run(string[] args)
		{
			string name;
			List<List<Vec2>> polys;
			if (args.Length > 0 && args[0] == "glyphs")
			{
				name = "glyphs";
				polys = Glyphs(DriverArgs.Int(args, 1, 100), DriverArgs.Int(args, 2, 100));
			}
			else
			{
				name = "holes";
				polys = Holes(DriverArgs.Int(args, 1, 71));
			}

			int verts = polys.Sum(p => p.Count);
			long start = System.Diagnostics.Stopwatch.GetTimestamp();
			List<IVec3> tris = Polygon.Triangulate(polys, -1.0, true);
			double elapsed = DriverArgs.SecondsSince(start);
			Console.WriteLine(string.Create(
				CultureInfo.InvariantCulture,
				$"{name}: {polys.Count} rings, {verts} verts -> {tris.Count} tris, hash 0x{Fnv(tris):x16}, {elapsed:F3} s"));
			return new[] { new Sample($"keyhole/{name}", elapsed, "sec") };
		}

		private static ulong Fnv(List<IVec3> tris)
		{
			ulong h = 0xcbf29ce484222325UL;
			foreach (IVec3 t in tris)
			{
				foreach (int c in new[] { t.X, t.Y, t.Z })
				{
					foreach (byte b in BitConverter.GetBytes(c))
					{
						h ^= b;
						h *= 0x100000001b3UL;
					}
				}
			}

			return h;
		}

		private static List<List<Vec2>> Holes(int n)
		{
			double size = (3.0 * n) + 1.0;
			List<List<Vec2>> polys = new List<List<Vec2>>
			{
				new List<Vec2> { new Vec2(0.0, 0.0), new Vec2(size, 0.0), new Vec2(size, size), new Vec2(0.0, size) },
			};
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
				{
					double cx = (3.0 * i) + 2.0;
					double cy = (3.0 * j) + 2.0;

					// Clockwise, so each ring is a hole.
					List<Vec2> ring = new List<Vec2>(32);
					for (int k = 0; k < 32; k++)
					{
						double a = -(double)k * (2.0 * Math.PI) / 32.0;
						ring.Add(new Vec2(cx + Math.Cos(a), cy + Math.Sin(a)));
					}

					polys.Add(ring);
				}
			}

			return polys;
		}

		private static List<Vec2> Octagon(double cx, double cy, double r)
		{
			(double X, double Y)[] corners =
			{
				(1.0, -0.5), (0.5, -1.0), (-0.5, -1.0), (-1.0, -0.5),
				(-1.0, 0.5), (-0.5, 1.0), (0.5, 1.0), (1.0, 0.5),
			};
			return corners.Select(c => new Vec2(cx + (r * c.X), cy + (r * c.Y))).ToList();
		}

		private static List<List<Vec2>> Glyphs(int rows, int cols)
		{
			List<List<Vec2>> polys = new List<List<Vec2>>();
			for (int row = 0; row < rows; row++)
			{
				for (int col = 0; col < cols; col++)
				{
					double x0 = 5.0 * col;
					double y0 = 7.0 * row;
					polys.Add(new List<Vec2>
					{
						new Vec2(x0, y0), new Vec2(x0 + 4.0, y0), new Vec2(x0 + 4.0, y0 + 6.0), new Vec2(x0, y0 + 6.0),
					});
					if ((row + col) % 2 == 0)
					{
						polys.Add(Octagon(x0 + 2.0, y0 + 3.0, 1.0));
					}
					else
					{
						polys.Add(Octagon(x0 + 2.0, y0 + 1.5, 0.75));
						polys.Add(Octagon(x0 + 2.0, y0 + 4.5, 0.75));
					}
				}
			}

			return polys;
		}
	}
}
