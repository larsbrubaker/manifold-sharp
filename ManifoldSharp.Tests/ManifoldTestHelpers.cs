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

// Port of src/manifold_tests/mod.rs — the shared fixtures and checkers of the
// integration suite. The Rust file is a module *and* a helper library: the
// `mod advanced; mod api; ...` list at its bottom is the suite's table of
// contents, and everything above it is what those modules call through
// `use super::*`. Here the modules become one test class per file
// (ManifoldBasicTests, ManifoldApiTests, ...) and this file is the static
// `ManifoldTestHelpers` they all call.
//
// ── What is here ─────────────────────────────────────────────────────────────
//   SquareHole            square with a square hole, offset along x
//   Gyroid                C++ Gyroid() — gyroid surface via SDF
//   WithPositionColors    C++ WithPositionColors(m)
//   RelatedGl             C++ RelatedGL() — every output vertex traces back into
//   RelatedGlCheckNormals   its source triangle, optionally with normal checks
//   CubeStl               C++ CubeSTL() — unit cube, 6 props, no shared verts
//
// ── DEFERRED, and why (greppable) ────────────────────────────────────────────
// Three helpers of mod.rs are not here, and none of them is blocked by a missing
// module — they are blocked by having no caller yet:
//
//   read_cpp_test_source / cpp_inline_array / cpp_inline_array_u32
//     Parse mesh literals out of the pinned C++ test sources at test time. Per
//     docs/PORTING_PLAN.md's verification note this port does NOT keep a
//     cpp-reference dependency: those meshes get transcribed into checked-in
//     test data instead. Their only consumers are `InterpolatedNormals` and
//     `Ring`, both in manifold_tests/complex.rs, which is not ported yet, so
//     transcribing the data now would land a large fixture with nothing reading
//     it. Transcribe them with the first test that needs one. (An earlier
//     version of this note named normals.rs and mesh_ops.rs as the consumers;
//     neither ever called these, and mesh_ops.rs is now ported — its MergeRefine
//     mesh was a plain inline literal, transcribed into
//     ManifoldMeshOpsTests.MergeRefineData.cs, which is exactly the resolution
//     this paragraph prescribes.)
//   read_test_obj
//     Loads an OBJ from the C++ test models directory. Same story — its
//     consumers are in complex.rs / validation.rs, not ported here — and the
//     same resolution: the models become checked-in test data (the suite already
//     has a TestData/ folder) when a ported test first reads one.
//
// ── RelatedGL is an assertion helper, not a test ─────────────────────────────
// It is `async` here because TUnit assertions are, so its callers await it. The
// Rust panics with a formatted message on failure; the C# assertions carry the
// same message through `.Because(...)`, which is what a failure has to say to be
// as diagnosable as the Rust's.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;

namespace ManifoldSharp.Tests
{
	/// <summary>
	/// The shared fixtures and checkers of <c>manifold_tests/mod.rs</c>.
	/// </summary>
	public static class ManifoldTestHelpers
	{
		/// <summary>Helper: square with a square hole, offset along x.</summary>
		/// <param name="xOffset">How far to shift both loops along x.</param>
		/// <returns>The two loops, outer first.</returns>
		public static Polygons SquareHole(double xOffset)
		{
			return new Polygons
			{
				new SimplePolygon
				{
					new Vec2(2.0 + xOffset, 2.0),
					new Vec2(-2.0 + xOffset, 2.0),
					new Vec2(-2.0 + xOffset, -2.0),
					new Vec2(2.0 + xOffset, -2.0),
				},
				new SimplePolygon
				{
					new Vec2(-1.0 + xOffset, 1.0),
					new Vec2(1.0 + xOffset, 1.0),
					new Vec2(1.0 + xOffset, -1.0),
					new Vec2(-1.0 + xOffset, -1.0),
				},
			};
		}

		/// <summary>Port of C++ Gyroid() — gyroid surface via SDF.</summary>
		/// <returns>The gyroid manifold.</returns>
		public static Manifold Gyroid()
		{
			double twoPi = 2.0 * Types.KPi;
			return Manifold.LevelSet(
				p =>
				{
					double min3 = Math.Min(Math.Min(p.X, p.Y), p.Z);
					double max3 = Math.Min(twoPi - p.X, Math.Min(twoPi - p.Y, twoPi - p.Z));
					double bound = Math.Min(min3, max3);

					// System.Math.Cos/Sin and not DeterministicMath: the Rust helper calls
					// std `f64::cos`/`f64::sin` here, not `crate::math::cos`/`::sin`. Test
					// fixtures follow the Rust PER SITE, so the C# reads the same std libm
					// the Rust does — the standing exception to this port's
					// no-System.Math-transcendentals rule, alongside
					// CrossSection.Offset's arc_tol and the Slerp/Rot sites.
					//
					// What this costs, stated rather than hidden: .NET's libm and Rust's
					// can disagree by ~1 ulp, so the SDF this fixture feeds LevelSet is not
					// guaranteed bit-identical to the Rust's. That is the faithful choice
					// anyway — swapping in DeterministicMath would make the C# fixture a
					// *different function* from the one the Rust test defines, which is a
					// larger divergence than an ulp of libm. Any consumer of Gyroid()
					// therefore asserts on tolerances, never on exact counts or positions.
					double gyroid =
						(Math.Cos(p.X) * Math.Sin(p.Y))
						+ (Math.Cos(p.Y) * Math.Sin(p.Z))
						+ (Math.Cos(p.Z) * Math.Sin(p.X));
					return Math.Min(gyroid, bound);
				},
				new Box(new Vec3(0.0, 0.0, 0.0), new Vec3(twoPi, twoPi, twoPi)),
				0.5);
		}

		/// <summary>
		/// Port of C++ WithPositionColors(m) — adds normalized position as 3 extra
		/// properties.
		/// </summary>
		/// <param name="m">The mesh to colour.</param>
		/// <returns>The coloured mesh.</returns>
		public static Manifold WithPositionColors(Manifold m)
		{
			Box bb = m.BoundingBox();
			Vec3 size = bb.Size();
			return m.SetProperties(3, (Span<double> props, Vec3 pos, ReadOnlySpan<double> old) =>
			{
				props[0] = size.X > 0.0 ? (pos.X - bb.Min.X) / size.X : 0.0;
				props[1] = size.Y > 0.0 ? (pos.Y - bb.Min.Y) / size.Y : 0.0;
				props[2] = size.Z > 0.0 ? (pos.Z - bb.Min.Z) / size.Z : 0.0;
			});
		}

		/// <summary>
		/// Port of C++ RelatedGL() — verifies output triangles trace to valid input
		/// triangles.
		/// </summary>
		/// <remarks>
		/// For each run in the output MeshGL, finds the matching original mesh by
		/// <c>run_original_id</c>, then verifies that each output vertex position lies
		/// within the (transformed) input triangle identified by <c>face_id</c>. Matches
		/// the C++ test helper exactly.
		/// </remarks>
		/// <param name="outManifold">The manifold whose provenance to check.</param>
		/// <param name="originals">The source meshes it should trace back into.</param>
		/// <returns>A task that completes when every assertion has run.</returns>
		public static Task RelatedGl(Manifold outManifold, IReadOnlyList<MeshGL> originals)
		{
			return RelatedGlImpl(outManifold, originals, false, false);
		}

		/// <summary>
		/// <see cref="RelatedGl"/> with the normal checks on: vertex normals at property
		/// offset 3 must be unit vectors pointing in the same direction as the output face
		/// normal (C++ <c>checkNormals=true</c>), after the world-frame update C++
		/// <c>updateNormals=true</c> applies.
		/// </summary>
		/// <param name="outManifold">The manifold whose provenance to check.</param>
		/// <param name="originals">The source meshes it should trace back into.</param>
		/// <returns>A task that completes when every assertion has run.</returns>
		public static Task RelatedGlCheckNormals(Manifold outManifold, IReadOnlyList<MeshGL> originals)
		{
			return RelatedGlImpl(outManifold, originals, true, true);
		}

		/// <summary>
		/// Port of C++ CubeSTL() — a unit cube with 6 properties per vertex (xyz +
		/// face-normal xyz).
		/// </summary>
		/// <remarks>
		/// Every triangle has 3 unique vertices (no sharing), so each vertex carries the
		/// flat face normal. Matches C++ CubeSTL() exactly, including a reserved
		/// run_original_id.
		/// </remarks>
		/// <returns>The STL-style cube.</returns>
		public static MeshGL CubeStl()
		{
			Manifold cubeManifold = Manifold.Cube(Vec3.Splat(1.0), true);
			MeshGL cubeIn = cubeManifold.GetMeshGL(0);
			int inNp = (int)cubeIn.NumProp;
			int numTri = cubeIn.TriVerts.Count / 3;

			MeshGL cube = new MeshGL();
			cube.NumProp = 6;

			uint vert = 0;
			for (int tri = 0; tri < numTri; tri++)
			{
				// Collect per-tri positions
				double[][] triPos = new double[3][];
				for (int i = 0; i < 3; i++)
				{
					triPos[i] = new double[3];
					cube.TriVerts.Add(vert);
					vert += 1;
					int v = (int)cubeIn.TriVerts[(3 * tri) + i];
					for (int j = 0; j < 3; j++)
					{
						triPos[i][j] = cubeIn.VertProperties[(v * inNp) + j];
					}
				}

				// Compute flat face normal
				double[] e0 =
				{
					triPos[1][0] - triPos[0][0],
					triPos[1][1] - triPos[0][1],
					triPos[1][2] - triPos[0][2],
				};
				double[] e1 =
				{
					triPos[2][0] - triPos[0][0],
					triPos[2][1] - triPos[0][1],
					triPos[2][2] - triPos[0][2],
				};
				double[] cross =
				{
					(e0[1] * e1[2]) - (e0[2] * e1[1]),
					(e0[2] * e1[0]) - (e0[0] * e1[2]),
					(e0[0] * e1[1]) - (e0[1] * e1[0]),
				};
				double len = Math.Sqrt((cross[0] * cross[0]) + (cross[1] * cross[1]) + (cross[2] * cross[2]));
				double[] normal = len > 0.0
					? new[] { cross[0] / len, cross[1] / len, cross[2] / len }
					: new[] { 0.0, 0.0, 1.0 };

				for (int i = 0; i < 3; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						cube.VertProperties.Add((float)triPos[i][j]);
					}

					for (int j = 0; j < 3; j++)
					{
						cube.VertProperties.Add((float)normal[j]);
					}
				}
			}

			cube.RunOriginalId.Add(Manifold.ReserveIds(1));
			return cube;
		}

		private static async Task RelatedGlImpl(
			Manifold outManifold,
			IReadOnlyList<MeshGL> originals,
			bool checkNormals,
			bool updateNormals)
		{
			await Assert.That(outManifold.IsEmpty())
				.IsFalse()
				.Because("RelatedGL: output should not be empty");

			// Match C++ RelatedGL: GetMeshGL() (= -1) auto-substitutes slot 0 only when
			// every meshID recorded normals; mixed outputs keep raw slots for the
			// per-run normalization below.
			MeshGL output = outManifold.GetMeshGL(-1);
			int numRun = output.RunOriginalId.Count;

			// Base tolerance from the output manifold; per-run tolerance also includes
			// in_mesh.tolerance
			double baseTolerance = 3.0 * Math.Max(outManifold.GetTolerance(), output.Tolerance);

			// Save transforms before update_normals clears them
			float[]?[] runTransforms = new float[numRun][];
			for (int run = 0; run < numRun; run++)
			{
				int offset = 12 * run;
				if (offset + 12 <= output.RunTransform.Count)
				{
					float[] arr = new float[12];
					for (int i = 0; i < 12; i++)
					{
						arr[i] = output.RunTransform[offset + i];
					}

					runTransforms[run] = arr;
				}
				else
				{
					runTransforms[run] = null;
				}
			}

			if (updateNormals)
			{
				// Per C++ #1718 RelatedGL: bring slot 3..5 into world frame and
				// normalize. hasNormals runs (run_flags bit 1) are already world-frame
				// (identity); others need the inverse-transpose of the per-run linear
				// transform (with a backside sign flip). Replaces the removed
				// MeshGL::UpdateNormals.
				int outNpU = (int)output.NumProp;
				bool[] vertUpdated = new bool[output.NumVert()];
				for (int run = 0; run < numRun; run++)
				{
					Mat3? nt = null;
					if (!output.HasNormals(run) && runTransforms[run] != null)
					{
						float[] t = runTransforms[run]!;
						double sign = output.Backside(run) ? -1.0 : 1.0;
						Vec3 Col(int j) => new Vec3(t[3 * j], t[(3 * j) + 1], t[(3 * j) + 2]);
						nt = Mat3.FromCols(Col(0), Col(1), Col(2)).Transpose().Inverse() * sign;
					}

					int start = (int)output.RunIndex[run];
					int end = (int)output.RunIndex[run + 1];
					for (int k = start; k < end; k++)
					{
						int v = (int)output.TriVerts[k];
						if (vertUpdated[v])
						{
							continue;
						}

						vertUpdated[v] = true;
						int basePropIdx = (v * outNpU) + 3;
						Vec3 n = new Vec3(
							output.VertProperties[basePropIdx],
							output.VertProperties[basePropIdx + 1],
							output.VertProperties[basePropIdx + 2]);
						if (nt.HasValue)
						{
							n = nt.Value * n;
						}

						double len = Math.Sqrt((n.X * n.X) + (n.Y * n.Y) + (n.Z * n.Z));
						if (len > 0.0)
						{
							n = n / len;
						}

						output.VertProperties[basePropIdx] = (float)n.X;
						output.VertProperties[basePropIdx + 1] = (float)n.Y;
						output.VertProperties[basePropIdx + 2] = (float)n.Z;
					}
				}
			}

			for (int run = 0; run < numRun; run++)
			{
				uint outId = output.RunOriginalId[run];
				MeshGL? inMesh = null;
				foreach (MeshGL m in originals)
				{
					if (m.RunOriginalId.Count == 1 && m.RunOriginalId[0] == outId)
					{
						inMesh = m;
						break;
					}
				}

				await Assert.That(inMesh)
					.IsNotNull()
					.Because($"RelatedGL: no original with runOriginalID={outId}");

				// Per-run tolerance: max of output tolerance and this run's input mesh
				// tolerance
				double tolerance = Math.Max(baseTolerance, 3.0 * inMesh!.Tolerance);

				// Use saved transform (before update_normals may have cleared run_transform)
				bool hasTransform = runTransforms[run] != null;
				float[] t = runTransforms[run] ?? new float[12];

				int startTri = (int)output.RunIndex[run] / 3;
				int endTri = (int)output.RunIndex[run + 1] / 3;
				int inNp = (int)inMesh.NumProp;
				int outNp = (int)output.NumProp;

				for (int tri = startTri; tri < endTri; tri++)
				{
					int inTriIdx = output.FaceId.Count == 0 ? tri : (int)output.FaceId[tri];
					int inTriCount = inMesh.TriVerts.Count / 3;
					await Assert.That(inTriIdx)
						.IsLessThan(inTriCount)
						.Because($"RelatedGL: faceID {inTriIdx} out of range (original has {inTriCount} tris)");

					// Get original triangle vertices and apply transform
					double[][] inTriPos = new double[3][];
					for (int j = 0; j < 3; j++)
					{
						int vert = (int)inMesh.TriVerts[(3 * inTriIdx) + j];
						double x = inMesh.VertProperties[vert * inNp];
						double y = inMesh.VertProperties[(vert * inNp) + 1];
						double z = inMesh.VertProperties[(vert * inNp) + 2];
						inTriPos[j] = hasTransform
							? new[]
							{
								// Column-major mat3x4: col0=[t0,t1,t2], col1=[t3,t4,t5],
								// col2=[t6,t7,t8], col3=[t9,t10,t11]
								(t[0] * x) + (t[3] * y) + (t[6] * z) + t[9],
								(t[1] * x) + (t[4] * y) + (t[7] * z) + t[10],
								(t[2] * x) + (t[5] * y) + (t[8] * z) + t[11],
							}
							: new[] { x, y, z };
					}

					// Compute triangle normal and area
					double[] e0 =
					{
						inTriPos[1][0] - inTriPos[0][0],
						inTriPos[1][1] - inTriPos[0][1],
						inTriPos[1][2] - inTriPos[0][2],
					};
					double[] e1 =
					{
						inTriPos[2][0] - inTriPos[0][0],
						inTriPos[2][1] - inTriPos[0][1],
						inTriPos[2][2] - inTriPos[0][2],
					};
					double[] inCross =
					{
						(e0[1] * e1[2]) - (e0[2] * e1[1]),
						(e0[2] * e1[0]) - (e0[0] * e1[2]),
						(e0[0] * e1[1]) - (e0[1] * e1[0]),
					};
					double area = Math.Sqrt(
						(inCross[0] * inCross[0]) + (inCross[1] * inCross[1]) + (inCross[2] * inCross[2]));
					if (area == 0.0)
					{
						continue;
					}

					// Compute output triangle positions for normal check
					double[][] outTriPos = new double[3][];
					for (int j = 0; j < 3; j++)
					{
						int vert = (int)output.TriVerts[(3 * tri) + j];
						outTriPos[j] = new[]
						{
							(double)output.VertProperties[vert * outNp],
							(double)output.VertProperties[(vert * outNp) + 1],
							(double)output.VertProperties[(vert * outNp) + 2],
						};
					}

					double[] oe0 =
					{
						outTriPos[1][0] - outTriPos[0][0],
						outTriPos[1][1] - outTriPos[0][1],
						outTriPos[1][2] - outTriPos[0][2],
					};
					double[] oe1 =
					{
						outTriPos[2][0] - outTriPos[0][0],
						outTriPos[2][1] - outTriPos[0][1],
						outTriPos[2][2] - outTriPos[0][2],
					};
					double[] outNormalUnnorm =
					{
						(oe0[1] * oe1[2]) - (oe0[2] * oe1[1]),
						(oe0[2] * oe1[0]) - (oe0[0] * oe1[2]),
						(oe0[0] * oe1[1]) - (oe0[1] * oe1[0]),
					};

					// For each output vertex, check it's within the input triangle
					for (int j = 0; j < 3; j++)
					{
						int vert = (int)output.TriVerts[(3 * tri) + j];
						double[] outPos = outTriPos[j];

						// edges[k] = in_tri_pos[k] - out_pos
						double[][] edges = new double[3][];
						for (int k = 0; k < 3; k++)
						{
							edges[k] = new[]
							{
								inTriPos[k][0] - outPos[0],
								inTriPos[k][1] - outPos[1],
								inTriPos[k][2] - outPos[2],
							};
						}

						// Triple product = dot(edges[0], cross(edges[1], edges[2]))
						double[] c =
						{
							(edges[1][1] * edges[2][2]) - (edges[1][2] * edges[2][1]),
							(edges[1][2] * edges[2][0]) - (edges[1][0] * edges[2][2]),
							(edges[1][0] * edges[2][1]) - (edges[1][1] * edges[2][0]),
						};
						double volume = (edges[0][0] * c[0]) + (edges[0][1] * c[1]) + (edges[0][2] * c[2]);
						await Assert.That(volume)
							.IsLessThanOrEqualTo(area * tolerance)
							.Because(
								$"RelatedGL: run={run} tri={tri} vert={j}: volume={volume} > "
								+ $"area*tol={area * tolerance} (in_tri={inTriIdx})");

						if (checkNormals && outNp >= 6)
						{
							double nx = output.VertProperties[(vert * outNp) + 3];
							double ny = output.VertProperties[(vert * outNp) + 4];
							double nz = output.VertProperties[(vert * outNp) + 5];
							double len = Math.Sqrt((nx * nx) + (ny * ny) + (nz * nz));
							await Assert.That(Math.Abs(len - 1.0))
								.IsLessThan(0.0001)
								.Because($"RelatedGL: run={run} tri={tri} vert={j}: normal length={len} != 1");

							// Normal must point in same half-space as output face normal
							double dot =
								(nx * outNormalUnnorm[0])
								+ (ny * outNormalUnnorm[1])
								+ (nz * outNormalUnnorm[2]);
							await Assert.That(dot)
								.IsGreaterThan(0.0)
								.Because($"RelatedGL: run={run} tri={tri} vert={j}: normal dot face_normal={dot} <= 0");
						}
					}
				}
			}
		}
	}
}
