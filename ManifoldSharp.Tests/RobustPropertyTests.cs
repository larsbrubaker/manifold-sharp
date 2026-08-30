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

// RobustPropertyTests.cs — port of robust/property_tests.rs, whose header reads:
//
//   Vertex-property (color) pass-through for the robust boolean engine.
//
//   The robust pipeline must carry per-operand vertex properties to its output
//   the way the exact engine does: constant per-operand properties (the demo's
//   A/B colors) survive exactly, interpolated properties agree with the exact
//   engine to double precision, and the output vertex positions of a manifold
//   boolean match the exact engine's within f64 rounding. The reference
//   workload is the demo's spiky-dodecahedron pair.
//
// Same inputs, same expected values, same order as the Rust's three tests.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class RobustPropertyTests
	{
		private static readonly double[] Blue = { 0.27, 0.53, 0.80, 1.0 };

		private static readonly double[] Red = { 0.85, 0.25, 0.25, 0.6 };

		/// <summary>
		/// Constant per-operand colors must survive the robust engine exactly: every output
		/// vertex carries exactly the blue or the red RGBA.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ConstantColorsSurviveRobustUnion()
		{
			(Manifold A, Manifold B) pair = ColoredSpikyPair();
			Manifold outManifold = pair.A.UnionWithEngine(pair.B, BooleanEngine.Robust);
			await Assert.That(outManifold.Status()).IsEqualTo(Error.NoError);
			await Assert.That(outManifold.AsImpl().NumProp).IsEqualTo(4)
				.Because("robust output must keep the 4 color channels");
			MeshGL gl = outManifold.GetMeshGL(-1);
			await Assert.That(gl.NumProp).IsEqualTo(7u).Because("xyz + rgba");
			int n = gl.VertProperties.Count / 7;
			int blue = 0;
			int red = 0;
			for (int i = 0; i < n; i++)
			{
				double[] c = new double[4];
				for (int k = 3; k < 7; k++)
				{
					c[k - 3] = gl.VertProperties[(i * 7) + k];
				}

				bool Is(double[] rgba)
				{
					for (int k = 0; k < 4; k++)
					{
						if (Math.Abs(c[k] - rgba[k]) >= 1e-6)
						{
							return false;
						}
					}

					return true;
				}

				await Assert.That(Is(Blue) || Is(Red)).IsTrue()
					.Because($"vertex {i} color [{string.Join(", ", c)}] is neither operand color");
				if (Is(Blue))
				{
					blue++;
				}
				else
				{
					red++;
				}
			}

			await Assert.That(blue > 0 && red > 0).IsTrue()
				.Because("both operands must contribute vertices");
		}

		/// <summary>
		/// Robust and exact engines must produce the same set of output vertex positions
		/// (within double precision) for the colored spiky pair, and the same color at every
		/// shared position.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task RobustMatchesExactVerticesAndColors()
		{
			(Manifold A, Manifold B) pair = ColoredSpikyPair();
			Manifold exact = pair.A.UnionWithEngine(pair.B, BooleanEngine.Exact);
			Manifold robust = pair.A.UnionWithEngine(pair.B, BooleanEngine.Robust);
			await Assert.That(robust.Status()).IsEqualTo(Error.NoError);

			MeshGL ge = exact.GetMeshGL(-1);
			MeshGL gr = robust.GetMeshGL(-1);
			List<double[]> pe = PositionSet(ge);
			List<double[]> pr = PositionSet(gr);
			await Assert.That(pr.Count).IsEqualTo(pe.Count)
				.Because("engines must agree on the distinct output vertex count");
			for (int i = 0; i < pe.Count; i++)
			{
				double[] x = pe[i];
				double[] y = pr[i];
				bool close = true;
				for (int k = 0; k < 3; k++)
				{
					if (Math.Abs(x[k] - y[k]) > 1e-6 * Math.Max(Math.Abs(x[k]), 1.0))
					{
						close = false;
					}
				}

				await Assert.That(close).IsTrue()
					.Because($"position mismatch: [{string.Join(", ", x)}] vs [{string.Join(", ", y)}]");
			}

			// Per-position color agreement: build exact's position→color map and
			// check every robust vertex against it.
			const int Np = 7;
			static (uint X, uint Y, uint Z) Key(float[] p)
			{
				return (
					BitConverter.SingleToUInt32Bits(p[0]),
					BitConverter.SingleToUInt32Bits(p[1]),
					BitConverter.SingleToUInt32Bits(p[2]));
			}

			SortedDictionary<(uint X, uint Y, uint Z), List<float[]>> exactColors
				= new SortedDictionary<(uint X, uint Y, uint Z), List<float[]>>();
			for (int i = 0; i < ge.VertProperties.Count / Np; i++)
			{
				float[] p =
				{
					ge.VertProperties[i * Np],
					ge.VertProperties[(i * Np) + 1],
					ge.VertProperties[(i * Np) + 2],
				};
				float[] c =
				{
					ge.VertProperties[(i * Np) + 3],
					ge.VertProperties[(i * Np) + 4],
					ge.VertProperties[(i * Np) + 5],
					ge.VertProperties[(i * Np) + 6],
				};
				(uint X, uint Y, uint Z) k = Key(p);
				if (!exactColors.TryGetValue(k, out List<float[]>? list))
				{
					list = new List<float[]>();
					exactColors.Add(k, list);
				}

				list.Add(c);
			}

			for (int i = 0; i < gr.VertProperties.Count / Np; i++)
			{
				float[] p =
				{
					gr.VertProperties[i * Np],
					gr.VertProperties[(i * Np) + 1],
					gr.VertProperties[(i * Np) + 2],
				};
				float[] c =
				{
					gr.VertProperties[(i * Np) + 3],
					gr.VertProperties[(i * Np) + 4],
					gr.VertProperties[(i * Np) + 5],
					gr.VertProperties[(i * Np) + 6],
				};
				await Assert.That(exactColors.TryGetValue(Key(p), out List<float[]>? cands)).IsTrue()
					.Because($"robust vertex [{string.Join(", ", p)}] not among exact vertices");
				bool any = false;
				foreach (float[] e in cands!)
				{
					bool all = true;
					for (int k = 0; k < 4; k++)
					{
						if (Math.Abs(e[k] - c[k]) >= 1e-6)
						{
							all = false;
						}
					}

					if (all)
					{
						any = true;
					}
				}

				await Assert.That(any).IsTrue()
					.Because($"color mismatch at [{string.Join(", ", p)}]: robust [{string.Join(", ", c)}]");
			}
		}

		/// <summary>
		/// Position-dependent (non-constant) properties interpolate to the exact engine's
		/// values within double precision.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task InterpolatedPropertiesMatchExact()
		{
			static Manifold PropFn(Manifold m)
			{
				return m.SetProperties(1, (newProp, pos, _) =>
					newProp[0] = 0.25 + (pos.X * 0.5) + (pos.Y * 0.125) - (pos.Z * 0.0625));
			}

			Manifold a = PropFn(Manifold.Cube(new Vec3(2.0, 2.0, 2.0), true));
			Manifold b = PropFn(Manifold.Sphere(1.3, 8).Translate(new Vec3(0.4, 0.2, 0.1)));
			Manifold exact = a.DifferenceWithEngine(b, BooleanEngine.Exact);
			Manifold robust = a.DifferenceWithEngine(b, BooleanEngine.Robust);
			await Assert.That(robust.Status()).IsEqualTo(Error.NoError);
			await Assert.That(robust.AsImpl().NumProp).IsEqualTo(1);

			const int Np = 4; // xyz + 1
			MeshGL gr = robust.GetMeshGL(-1);
			await Assert.That((int)gr.NumProp).IsEqualTo(Np);

			// The property is an affine function of position on every input face, so
			// barycentric interpolation must reproduce it at every output vertex.
			for (int i = 0; i < gr.VertProperties.Count / Np; i++)
			{
				double x = gr.VertProperties[i * Np];
				double y = gr.VertProperties[(i * Np) + 1];
				double z = gr.VertProperties[(i * Np) + 2];
				double expect = 0.25 + (x * 0.5) + (y * 0.125) - (z * 0.0625);
				double got = gr.VertProperties[(i * Np) + 3];
				await Assert.That(Math.Abs(got - expect)).IsLessThan(1e-5)
					.Because($"vertex {i} ({x},{y},{z}): prop {got} vs affine {expect}");
			}

			_ = exact;
		}

		/// <summary>The demo's spiky dodecahedron: 12 pentagonal faces fanned to spike tips.</summary>
		/// <param name="spikeHeight">How far each face's spike stands off the surface.</param>
		/// <returns>The manifold.</returns>
		private static Manifold SpikyDodecahedron(double spikeHeight)
		{
			double phi = (1.0 + Math.Sqrt(5.0)) / 2.0;
			double invPhi = 1.0 / phi;
			const double Scale = 0.5;
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
			(double X, double Y, double Z)[] verts = new (double, double, double)[rawVerts.Length];
			for (int i = 0; i < rawVerts.Length; i++)
			{
				verts[i] = (rawVerts[i].X * Scale, rawVerts[i].Y * Scale, rawVerts[i].Z * Scale);
			}

			List<float> positions = new List<float>(32 * 3);
			List<uint> triVerts = new List<uint>(60 * 3);
			foreach ((double X, double Y, double Z) v in verts)
			{
				positions.Add((float)v.X);
				positions.Add((float)v.Y);
				positions.Add((float)v.Z);
			}

			foreach (int[] face in faces)
			{
				double cx = 0.0;
				double cy = 0.0;
				double cz = 0.0;
				foreach (int i in face)
				{
					cx += verts[i].X;
					cy += verts[i].Y;
					cz += verts[i].Z;
				}

				cx /= 5.0;
				cy /= 5.0;
				cz /= 5.0;
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

		/// <summary>Paints every vertex of <paramref name="m"/> one constant RGBA.</summary>
		/// <param name="m">The manifold.</param>
		/// <param name="rgba">The color.</param>
		/// <returns>The colored manifold.</returns>
		private static Manifold Color(Manifold m, double[] rgba)
		{
			return m.SetProperties(4, (newProp, _, _) => rgba.AsSpan().CopyTo(newProp));
		}

		/// <summary>The demo's colored spiky pair.</summary>
		/// <returns>The two operands.</returns>
		private static (Manifold A, Manifold B) ColoredSpikyPair()
		{
			Manifold a = Color(SpikyDodecahedron(0.4), Blue);
			Manifold b = Color(SpikyDodecahedron(0.4), Red)
				.Rotate(30.0, 45.0, 60.0)
				.Translate(new Vec3(0.3, 0.0, 0.0));
			return (a, b);
		}

		/// <summary>Distinct rounded positions of a MeshGL, sorted for set comparison.</summary>
		/// <param name="gl">The mesh.</param>
		/// <returns>The distinct positions, lexicographically ordered.</returns>
		private static List<double[]> PositionSet(MeshGL gl)
		{
			int np = (int)gl.NumProp;
			int n = gl.VertProperties.Count / np;
			List<double[]> outList = new List<double[]>(n);
			for (int i = 0; i < n; i++)
			{
				outList.Add(new[]
				{
					(double)gl.VertProperties[i * np],
					(double)gl.VertProperties[(i * np) + 1],
					(double)gl.VertProperties[(i * np) + 2],
				});
			}

			// Rust `sort_by(|a, b| a.partial_cmp(b).unwrap())` then `dedup` — lexicographic
			// order, then adjacent-duplicate removal (not a full distinct).
			//
			// NOT double.CompareTo: that is total order, which puts -0.0 strictly below
			// +0.0, while f64::partial_cmp calls them Equal (IEEE `==`). The two disagree
			// on any position carrying a signed zero, so the comparison is spelled with
			// `<` / `>` — IEEE operators, exactly what partial_cmp answers with. NaN makes
			// partial_cmp return None and the Rust's `.unwrap()` panic; the throw below is
			// that panic, not a softening of it.
			//
			// List.Sort is unstable and Rust's sort_by is stable, which is unobservable
			// here and only here: two elements compare Equal exactly when all three
			// coordinates are IEEE-equal, and the dedup below collapses precisely those
			// runs, so no ordering among them survives into the returned list.
			outList.Sort(static (a, b) =>
			{
				for (int k = 0; k < 3; k++)
				{
					if (a[k] < b[k])
					{
						return -1;
					}

					if (a[k] > b[k])
					{
						return 1;
					}

					if (double.IsNaN(a[k]) || double.IsNaN(b[k]))
					{
						throw new InvalidOperationException(
							"NaN output position: Rust's partial_cmp().unwrap() panics here");
					}
				}

				return 0;
			});
			List<double[]> deduped = new List<double[]>(outList.Count);
			foreach (double[] p in outList)
			{
				if (deduped.Count == 0
					|| deduped[deduped.Count - 1][0] != p[0]
					|| deduped[deduped.Count - 1][1] != p[1]
					|| deduped[deduped.Count - 1][2] != p[2])
				{
					deduped.Add(p);
				}
			}

			return deduped;
		}
	}
}
