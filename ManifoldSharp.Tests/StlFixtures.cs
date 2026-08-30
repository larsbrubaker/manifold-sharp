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

// StlFixtures.cs — port of robust/stl_fixtures.rs, whose header reads:
//
//   Test-only STL fixture loading for the robust engine's test modules.
//
//   Reproduces the WASM demo's import pipeline exactly, because the fixtures in
//   `testdata/` are regressions found through that demo: parse an ASCII or
//   binary STL into an f32 position soup, normalize it (bbox center to origin,
//   longest side scaled to 2.0, f32 storage with f64 arithmetic as in JS),
//   weld with `MeshGL::merge`, then import through `from_mesh_gl_robust`.
//
//   Declared as a child module of the test modules that use it (see
//   `engine_tests.rs`), so it needs no entry in `robust/mod.rs`.
//
//   `thingi_tests.rs` still carries its own copy of this pipeline; folding it
//   onto this module is a follow-up, deliberately not done here to avoid
//   touching that file.
//
// ── The one thing C# has to add ──────────────────────────────────────────────
// The Rust reads its fixtures with `include_bytes!`, which has no C# equivalent;
// the 20 STL files are test *content* copied beside the test assembly instead
// (see ManifoldSharp.Tests.csproj), so this file also owns FixtureBytes, the
// loader the Rust does not need. Everything below it is the Rust's, sites and
// rounding included: coordinates are stored f32 and computed f64, exactly as the
// JS Float32Array pipeline does. The binary reads go through BinaryPrimitives'
// LittleEndian entry points because the Rust's are `from_le_bytes` — pinned, not
// host-dependent; the two agree on every target .NET runs on, so this is the
// literal form rather than a fix.

using System.Buffers.Binary;
using System.Text;

using ManifoldSharp;
using ManifoldSharp.Linalg;

namespace ManifoldSharp.Tests
{
	/// <summary>
	/// The WASM demo's STL import pipeline, reproduced for the robust engine's
	/// fixture-driven tests.
	/// </summary>
	public static class StlFixtures
	{
		/// <summary>
		/// Read one of the checked-in Thingi10K fixtures by file name (e.g.
		/// <c>"92068.stl"</c>) — the C# stand-in for the Rust's
		/// <c>include_bytes!("testdata/…")</c>.
		/// </summary>
		/// <param name="fileName">The fixture's file name.</param>
		/// <returns>The file's bytes.</returns>
		public static byte[] FixtureBytes(string fileName)
		{
			ArgumentNullException.ThrowIfNull(fileName);
			string path = Path.Combine(AppContext.BaseDirectory, "TestData", "thingi10k", fileName);
			return File.ReadAllBytes(path);
		}

		/// <summary>Import an STL fixture through the demo's pipeline.</summary>
		/// <param name="stl">The raw STL bytes.</param>
		/// <returns>The imported manifold (soup-backed when the input is non-manifold).</returns>
		public static Manifold ImportStlLikeDemo(byte[] stl)
		{
			ArgumentNullException.ThrowIfNull(stl);

			float[] positions = IsAscii(stl) ? ParseAscii(stl) : ParseBinary(stl);
			Normalize(positions);
			uint nVerts = (uint)(positions.Length / 3);
			MeshGL mesh = new MeshGL();
			mesh.NumProp = 3;
			mesh.VertProperties = new List<float>(positions);
			mesh.TriVerts = new List<uint>((int)nVerts);
			for (uint i = 0; i < nVerts; i++)
			{
				mesh.TriVerts.Add(i);
			}

			mesh.Merge();
			return Manifold.FromMeshGLRobust(mesh);
		}

		/// <summary>
		/// Parse a binary STL (80-byte header, u32 face count, 50-byte records of
		/// 12 f32 = normal + 3 vertices, then a u16 attribute count).
		/// </summary>
		/// <param name="data">The file bytes.</param>
		/// <returns>The corner positions, 9 floats per face.</returns>
		private static float[] ParseBinary(byte[] data)
		{
			if (data.Length < 84)
			{
				throw new InvalidDataException("STL too short");
			}

			// The Rust widens the header's u32 to usize and does the truncation check in
			// 64-bit; this keeps it in `long` for the same reason. In 32-bit arithmetic a
			// corrupt header declaring >= 42,949,673 faces overflows `nFaces * 50`
			// negative, sails past the guard, and surfaces as an OverflowException from
			// the allocation instead of the InvalidDataException this exists to raise.
			long nFaces = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(80, 4));
			if (84 + (nFaces * 50) > data.Length)
			{
				throw new InvalidDataException($"STL truncated: header declares {nFaces} faces");
			}

			// The guard above bounds nFaces by data.Length / 50, so the int arithmetic
			// below cannot overflow.
			int faceCount = (int)nFaces;
			float[] positions = new float[faceCount * 9];
			for (int f = 0; f < faceCount; f++)
			{
				int rec = 84 + (f * 50);

				// Skip the 12-byte facet normal; read the 3 corner vertices.
				for (int i = 0; i < 9; i++)
				{
					int off = rec + 12 + (i * 4);
					positions[(f * 9) + i] = BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(off, 4));
				}
			}

			return positions;
		}

		/// <summary>
		/// Parse an ASCII STL as the demo does: scan for <c>vertex x y z</c> lines, parse
		/// each coordinate as a double (JS <c>parseFloat</c>), store into f32.
		/// </summary>
		/// <param name="data">The file bytes.</param>
		/// <returns>The corner positions.</returns>
		private static float[] ParseAscii(byte[] data)
		{
			string text = new UTF8Encoding(false, true).GetString(data);
			List<float> positions = new List<float>();
			foreach (string line in text.Split('\n'))
			{
				string[] fields = line.Split(
					(char[]?)null,
					StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
				if (fields.Length != 0 && fields[0] == "vertex")
				{
					for (int i = 0; i < 3; i++)
					{
						// The Rust's `it.next().expect(...).parse::<f64>()`: a missing or
						// unparseable coordinate is a broken fixture, not a soft failure.
						double v = double.Parse(
							fields[i + 1],
							System.Globalization.CultureInfo.InvariantCulture);
						positions.Add((float)v);
					}
				}
			}

			return positions.ToArray();
		}

		/// <summary>
		/// The demo's format sniff: ASCII iff the head starts with "solid" and
		/// mentions "facet" early on.
		/// </summary>
		/// <param name="data">The file bytes.</param>
		/// <returns>True when the file should be parsed as ASCII.</returns>
		private static bool IsAscii(byte[] data)
		{
			// Rust's String::from_utf8_lossy over the first 512 bytes; the C# decoder
			// substitutes U+FFFD for invalid sequences the same way, which is all this
			// sniff needs (a binary header decodes to garbage either way).
			string head = Encoding.UTF8.GetString(data, 0, Math.Min(data.Length, 512));
			return head.TrimStart().StartsWith("solid", StringComparison.Ordinal)
				&& head.Contains("facet", StringComparison.Ordinal);
		}

		/// <summary>
		/// Normalize as the demo's <c>fetchMesh</c> does: bbox center to the origin,
		/// uniform scale so the longest bbox side is 2.0. JS holds positions in a
		/// Float32Array but computes in doubles: read f32 -&gt; widen to f64 -&gt;
		/// (x - c) * s in f64 -&gt; truncate to f32 on store.
		/// </summary>
		/// <param name="positions">The positions, normalized in place.</param>
		private static void Normalize(float[] positions)
		{
			int nVerts = positions.Length / 3;
			double[] min = new double[] { double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity };
			double[] max = new double[] { double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity };
			for (int i = 0; i < nVerts; i++)
			{
				for (int k = 0; k < 3; k++)
				{
					double x = positions[(i * 3) + k];

					// MinF64/MaxF64, not Math.Min/Math.Max: the Rust f64::min/max return
					// the non-NaN operand, C#'s propagate the NaN. A NaN coordinate would
					// therefore leave the Rust with a usable bbox and this port with an
					// all-NaN one — a silent divergence rather than a shared failure. No
					// fixture reaches it; the house convention exists for exactly this.
					min[k] = LinalgFunctions.MinF64(min[k], x);
					max[k] = LinalgFunctions.MaxF64(max[k], x);
				}
			}

			double[] center = new double[]
			{
				(min[0] + max[0]) / 2.0,
				(min[1] + max[1]) / 2.0,
				(min[2] + max[2]) / 2.0,
			};
			double maxSide = LinalgFunctions.MaxF64(
				LinalgFunctions.MaxF64(max[0] - min[0], max[1] - min[1]),
				max[2] - min[2]);
			double scale = maxSide > 0.0 ? 2.0 / maxSide : 1.0;
			for (int i = 0; i < nVerts; i++)
			{
				for (int k = 0; k < 3; k++)
				{
					positions[(i * 3) + k] = (float)((positions[(i * 3) + k] - center[k]) * scale);
				}
			}
		}
	}
}
