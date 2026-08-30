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

// RobustBench.cs — the whole-operation half of manifold-rust examples/robust_perf.rs,
// whose header reads:
//
//   Stage-by-stage timing driver for the robust boolean engine.
//
//   Runs the demo's spiky-dodecahedron-vs-spiky-dodecahedron boolean (the
//   Boolean Gallery default) through both engines and prints per-stage wall
//   times for the robust pipeline, so optimization work has a baseline and a
//   regression check. Usage:
//
//     cargo run --release --example robust_perf [-- <rot_x> <rot_y> <rot_z>]
//
//   The rotation (degrees, default 30/45/60) matches an animated gallery frame.
//
// Why this driver is in the set at all: every other benchmark here — perf, large
// scene, menger, bracelet, twins, sdf blobs — runs on the DEFAULT boolean engine,
// which is Exact, the float-plus-symbolic-perturbation engine. None of them execute a
// single BigInteger operation. So none of them can answer the question
// docs/PORTING_PLAN.md's Phase 11 poses: whether `System.Numerics.BigInteger` on the
// exact-predicate tier profiles differently from dashu's two-words-inline
// representation. This one does, by running the same boolean through both engines and
// reporting the pair.
//
// Deliberately NOT ported: the Rust example's per-stage breakdown (build_graph /
// build_cells / windings / extract / assemble) and its ROBUST_PERF_THINGI section. The
// stage functions are `internal` in this port — the Rust reaches them because its
// example is in-crate — and reproducing them would mean widening the public surface
// for a benchmark, which is the wrong trade. The whole-op numbers are what the
// comparison needs; a stage breakdown belongs to optimization work, and that work can
// use the test project, which does have access.
//
// Run: dotnet run -c Release --project ManifoldSharp.Benchmarks -- robust [rx ry rz]

using System.Globalization;

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Benchmarks
{
	/// <summary>The same boolean through both engines — the only driver that reaches
	/// the exact-arithmetic (BigInteger) tier.</summary>
	internal static class RobustBench
	{
		/// <summary>Run the driver once.</summary>
		/// <param name="args">Optional three arguments: the rotation of operand B in degrees.</param>
		/// <returns>One timing sample per engine per case.</returns>
		public static IReadOnlyList<Sample> Run(string[] args)
		{
			double rx = args.Length == 3 ? double.Parse(args[0], CultureInfo.InvariantCulture) : 30.0;
			double ry = args.Length == 3 ? double.Parse(args[1], CultureInfo.InvariantCulture) : 45.0;
			double rz = args.Length == 3 ? double.Parse(args[2], CultureInfo.InvariantCulture) : 60.0;

			Manifold a = MakeSpikyDodecahedron(0.4);
			Manifold b = MakeSpikyDodecahedron(0.4)
				.Rotate(rx, ry, rz)
				.Translate(new Vec3(0.3, 0.0, 0.0));
			Console.WriteLine(string.Create(
				CultureInfo.InvariantCulture,
				$"spiky vs spiky: {a.NumTri()} + {b.NumTri()} tris, rot=({rx},{ry},{rz})"));

			List<Sample> samples = new List<Sample>();
			samples.Add(TimeUnion("spiky", "exact", a, b, BooleanEngine.Exact));
			samples.Add(TimeUnion("spiky", "robust", a, b, BooleanEngine.Robust));

			// Dense-mesh case (sparse intersections relative to size): sphere pair.
			// This is where the BVH broad phase matters — brute-force box pruning is
			// O(|P|·|Q|) (e.g. 8k×8k tris = 68M box tests) while the BVH visits only
			// the overlapping region.
			Manifold s1 = Manifold.Sphere(1.0, 64);
			Manifold s2 = Manifold.Sphere(1.0, 64).Translate(new Vec3(1.7, 0.0, 0.0));
			samples.Add(TimeUnion("sphere64", "exact", s1, s2, BooleanEngine.Exact));
			samples.Add(TimeUnion("sphere64", "robust", s1, s2, BooleanEngine.Robust));

			return samples;
		}

		private static Sample TimeUnion(string caseName, string engineName, Manifold a, Manifold b, BooleanEngine engine)
		{
			long start = System.Diagnostics.Stopwatch.GetTimestamp();
			Manifold outMesh = a.UnionWithEngine(b, engine);
			double elapsed = DriverArgs.SecondsSince(start);
			int outTris = outMesh.NumTri();

			if (outMesh.Status() != Error.NoError || outTris == 0)
			{
				throw new InvalidOperationException(string.Create(
					CultureInfo.InvariantCulture,
					$"{caseName}/{engineName}: status={outMesh.Status()}, tris={outTris}"));
			}

			Console.WriteLine(string.Create(
				CultureInfo.InvariantCulture,
				$"{engineName,-6} {caseName} union: {elapsed * 1e3,8:F2} ms   ({outTris} tris out)"));
			return new Sample($"robust/{caseName} {engineName}", elapsed, "sec");
		}

		/// <summary>
		/// The Boolean Gallery's default operand: a dodecahedron with a spike raised
		/// over each of its twelve faces, transcribed from robust_perf.rs.
		/// </summary>
		/// <param name="spikeHeight">How far each spike stands off its face centre.</param>
		/// <returns>The spiky dodecahedron, imported through the plain (non-robust) path.</returns>
		private static Manifold MakeSpikyDodecahedron(double spikeHeight)
		{
			double phi = (1.0 + Math.Sqrt(5.0)) / 2.0;
			double invPhi = 1.0 / phi;
			double scale = 0.5;
			(double X, double Y, double Z)[] rawVerts =
			{
				(1.0, 1.0, 1.0),
				(1.0, 1.0, -1.0),
				(1.0, -1.0, 1.0),
				(1.0, -1.0, -1.0),
				(-1.0, 1.0, 1.0),
				(-1.0, 1.0, -1.0),
				(-1.0, -1.0, 1.0),
				(-1.0, -1.0, -1.0),
				(0.0, invPhi, phi),
				(0.0, invPhi, -phi),
				(0.0, -invPhi, phi),
				(0.0, -invPhi, -phi),
				(invPhi, phi, 0.0),
				(-invPhi, phi, 0.0),
				(invPhi, -phi, 0.0),
				(-invPhi, -phi, 0.0),
				(phi, 0.0, invPhi),
				(phi, 0.0, -invPhi),
				(-phi, 0.0, invPhi),
				(-phi, 0.0, -invPhi),
			};
			int[][] faces =
			{
				new[] { 0, 8, 10, 2, 16 },
				new[] { 0, 16, 17, 1, 12 },
				new[] { 0, 12, 13, 4, 8 },
				new[] { 1, 17, 3, 11, 9 },
				new[] { 1, 9, 5, 13, 12 },
				new[] { 2, 10, 6, 15, 14 },
				new[] { 2, 14, 3, 17, 16 },
				new[] { 4, 13, 5, 19, 18 },
				new[] { 4, 18, 6, 10, 8 },
				new[] { 5, 9, 11, 7, 19 },
				new[] { 6, 18, 19, 7, 15 },
				new[] { 3, 14, 15, 7, 11 },
			};

			(double X, double Y, double Z)[] verts = rawVerts
				.Select(v => (v.X * scale, v.Y * scale, v.Z * scale))
				.ToArray();

			List<float> positions = new List<float>(32 * 3);
			List<uint> triVerts = new List<uint>(60 * 3);
			foreach ((double x, double y, double z) in verts)
			{
				positions.Add((float)x);
				positions.Add((float)y);
				positions.Add((float)z);
			}

			foreach (int[] face in faces)
			{
				double cx = face.Sum(i => verts[i].X) / 5.0;
				double cy = face.Sum(i => verts[i].Y) / 5.0;
				double cz = face.Sum(i => verts[i].Z) / 5.0;
				double len = Math.Sqrt((cx * cx) + (cy * cy) + (cz * cz));
				double nx = cx / len;
				double ny = cy / len;
				double nz = cz / len;
				uint spikeIdx = (uint)(positions.Count / 3);
				positions.Add((float)(cx + (nx * spikeHeight)));
				positions.Add((float)(cy + (ny * spikeHeight)));
				positions.Add((float)(cz + (nz * spikeHeight)));
				for (int j = 0; j < 5; j++)
				{
					triVerts.Add(spikeIdx);
					triVerts.Add((uint)face[j]);
					triVerts.Add((uint)face[(j + 1) % 5]);
				}
			}

			MeshGL mesh = new MeshGL();
			mesh.NumProp = 3;
			mesh.VertProperties = positions;
			mesh.TriVerts = triVerts;
			return Manifold.FromMeshGL(mesh);
		}
	}
}
