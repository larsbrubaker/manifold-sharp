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

// TwinsBench.cs — the Generic_Twin_7081 union as a driver.
//
// manifold-rust measures this as the pair of tests `test_cpp_generic_twin_7081` and
// `test_cpp_complex_generic_twin_7081` (C++ TEST(BooleanComplex,
// GenericTwinBooleanTest7081)), source of its "twins ~18s each release, 13.7s for both
// with --features parallel" figures. Both Rust tests union the same two models; only
// the operator spelling differs (`m1 + m2` versus `m1.union(&m2)`), so ONE union is
// what this driver runs, and a Rust figure quoted "for both" is two of them.
//
// This is the hardest boolean in the benchmark set and the reason it is here: the two
// models are real-world CAD twins that meet on nearly-coincident faces, so it is the
// case where the shared-vertex and degeneracy paths dominate — the ones a port is most
// likely to get right but slowly.
//
// Loading is deliberately outside the timed region, as it is in the Rust test's
// `read_test_obj` (which is called before the operation, not measured). The parser is
// the Rust's, transcribed from ManifoldTestHelpers.ReadTestObj — quirks kept, since a
// different parse is a different mesh and therefore a different benchmark.
//
// Run: dotnet run -c Release --project ManifoldSharp.Benchmarks -- twins

using System.Globalization;

namespace ManifoldSharp.Benchmarks
{
	/// <summary>Union of the two Generic_Twin_7081 models.</summary>
	internal static class TwinsBench
	{
		/// <summary>Run the driver once.</summary>
		/// <param name="args">Unused; the sample has no size parameter.</param>
		/// <returns>Union, MeshGL export and total timing samples.</returns>
		public static IReadOnlyList<Sample> Run(string[] args)
		{
			_ = args;

			Manifold m1 = ReadTestObj("Generic_Twin_7081.1.t0_left.obj");
			Manifold m2 = ReadTestObj("Generic_Twin_7081.1.t0_right.obj");

			long start = System.Diagnostics.Stopwatch.GetTimestamp();
			Manifold res = m1.Union(m2);
			int nTri = res.NumTri();
			double union = DriverArgs.SecondsSince(start);

			long exportStart = System.Diagnostics.Stopwatch.GetTimestamp();
			MeshGL gl = res.GetMeshGL(0);
			double export = DriverArgs.SecondsSince(exportStart);
			double total = DriverArgs.SecondsSince(start);

			// The C++ test only checks that this does not crash, but an errored or empty
			// result would be a fast one, so it is worth one line to refuse to time it.
			if (res.Status() != Error.NoError || gl.TriVerts.Count == 0)
			{
				throw new InvalidOperationException(string.Create(
					CultureInfo.InvariantCulture,
					$"GenericTwin7081: status={res.Status()}, triVerts={gl.TriVerts.Count}"));
			}

			Console.WriteLine(string.Create(
				CultureInfo.InvariantCulture,
				$"nTri(in) = {m1.NumTri()} + {m2.NumTri()}, nTri(out) = {nTri}, union = {union} sec, mesh_gl = {export} sec, total = {total} sec"));

			return new[]
			{
				new Sample("twins/union", union, "sec"),
				new Sample("twins/mesh_gl", export, "sec"),
				new Sample("twins/total", total, "sec"),
			};
		}

		/// <summary>
		/// The Rust tests' <c>read_test_obj</c>, transcribed from
		/// <c>ManifoldTestHelpers.ReadTestObj</c>: a <c>v</c> line contributes only when at
		/// least three of its tokens parse as numbers, an <c>f</c> line is fan-triangulated
		/// from its first index with <c>v/vt/vn</c> groups reduced to the leading index, and
		/// positions narrow to f32 on the way into the MeshGL.
		/// </summary>
		/// <param name="fileName">The model's file name.</param>
		/// <returns>The imported manifold.</returns>
		private static Manifold ReadTestObj(string fileName)
		{
			string path = Path.Combine(AppContext.BaseDirectory, "TestData", "models", fileName);
			string contents = File.ReadAllText(path);

			List<float> verts = new List<float>();
			List<uint> triVerts = new List<uint>();
			foreach (string rawLine in contents.Split('\n'))
			{
				string line = rawLine.Trim();
				if (line.StartsWith("v ", StringComparison.Ordinal))
				{
					List<double> parts = new List<double>();
					foreach (string s in line.Substring(2).Split(
						(char[]?)null,
						StringSplitOptions.RemoveEmptyEntries))
					{
						if (double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double d))
						{
							parts.Add(d);
						}
					}

					if (parts.Count >= 3)
					{
						verts.Add((float)parts[0]);
						verts.Add((float)parts[1]);
						verts.Add((float)parts[2]);
					}
				}
				else if (line.StartsWith("f ", StringComparison.Ordinal))
				{
					List<uint> indices = new List<uint>();
					foreach (string s in line.Substring(2).Split(
						(char[]?)null,
						StringSplitOptions.RemoveEmptyEntries))
					{
						string head = s.Split('/')[0];
						if (uint.TryParse(head, NumberStyles.Integer, CultureInfo.InvariantCulture, out uint i))
						{
							indices.Add(i - 1);
						}
					}

					for (int i = 1; i + 1 < indices.Count; i++)
					{
						triVerts.Add(indices[0]);
						triVerts.Add(indices[i]);
						triVerts.Add(indices[i + 1]);
					}
				}
			}

			MeshGL mesh = new MeshGL();
			mesh.NumProp = 3;
			mesh.VertProperties = verts;
			mesh.TriVerts = triVerts;
			return Manifold.FromMeshGL(mesh);
		}
	}
}
