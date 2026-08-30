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

// Port of src/manifold_tests/smooth.rs — all 19 of its tests, same inputs, same
// expected values, same tolerances, in the same order. Two files, for the
// 800-line cap: this one runs from the top of the Rust module through
// TEST(Smooth, SDF) and carries the module's three helpers;
// ManifoldSmoothTests.Relations.cs continues with MissingNormals to the end.
//
// The Rust module has no `#[ignore]`d test, so neither does this class.
//
// ── The module's own helpers ─────────────────────────────────────────────────
//   CircularTangent   private in the Rust, private here. It is the *test's*
//                     transcription of C++ CircularTangent(), deliberately
//                     independent of the library's Smoothing.CircularTangent —
//                     the Torus test would prove nothing if it built its input
//                     tangents with the code under test.
//   CsaszarGl         `pub(super)` in the Rust because api.rs calls it as
//                     `super::smooth::csaszar_gl()`; `internal static` here for
//                     the same reason, and ManifoldApiTests calls it as
//                     ManifoldSmoothTests.CsaszarGl(). Same arrangement as
//                     ManifoldApiTests.TetGl.
//   ResizeTangents    no counterpart in the Rust: it is Rust's
//                     `Vec::resize(n, 0.0)` (truncate or zero-pad), which
//                     `List<T>` has no single call for.
//   WithPositionColors  smooth.rs redefines this at its bottom, shadowing the
//                     glob-imported `super::with_position_colors` for the
//                     unqualified call sites. The two bodies are identical, so
//                     this port calls ManifoldTestHelpers.WithPositionColors
//                     from every site — including the two that qualify with
//                     `super::` — rather than checking in a duplicate.
//
// Nothing here is deferred. TEST(Smooth, Fillet) was the one case that could not
// be written the day this module was ported — it opens with `cylinder.slice(0)`
// — and it went in as soon as Manifold.Slice landed.
//
// ── Why these numbers ────────────────────────────────────────────────────────
// The Rust comments that explain a surprising expected value (the v3.5.0
// #1724/#1671 smoothing fixes behind TruncatedCone's 1163.53, the removal of
// TEST(Smooth, SineSurface), Torus's GetMeshGL(-1) slot reasoning) are carried
// over verbatim — they are the reason the number is what it is.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldSmoothTests
	{
		/// <summary>
		/// C++ TEST(Smooth, Normals) — SmoothOut and SmoothByNormals produce same result.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothNormals()
		{
			// C++ uses SmoothOut() which defaults to (60, 0)
			Manifold cylinder = Manifold.Cylinder(10.0, 5.0, 5.0, 8);

			// C++ SmoothOut() defaults to (60, 0), CalculateNormals(0) defaults to (0, 60)
			Manifold outM = cylinder.Clone().SmoothOut(60.0, 0.0).RefineToLength(0.1);
			Manifold byNormals = cylinder
				.CalculateNormals(0, 60.0)
				.SmoothByNormals(0)
				.RefineToLength(0.1);
			await Assert.That(Math.Abs(outM.Volume() - byNormals.Volume()))
				.IsLessThan(1e-4)
				.Because($"Normals: vol {outM.Volume()} vs {byNormals.Volume()}");
			await Assert.That(Math.Abs(outM.SurfaceArea() - byNormals.SurfaceArea()))
				.IsLessThan(1e-4)
				.Because($"Normals: sa {outM.SurfaceArea()} vs {byNormals.SurfaceArea()}");
		}

		/// <summary>
		/// C++ TEST(Smooth, TruncatedCone) — smooth cylinder with different radii.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothTruncatedCone()
		{
			Manifold cone = Manifold.Cylinder(5.0, 10.0, 5.0, 12);

			// C++ v3.5.0: SmoothOut() defaults to (52.5, 0); CalculateNormals(0)
			// defaults to (0, 52.5). Values updated for the #1724/#1671 smoothing fixes.
			Manifold smooth = cone
				.Clone()
				.SmoothOut(52.5, 0.0)
				.RefineToLength(0.5)
				.CalculateNormals(0, 52.5);
			await Assert.That(Math.Abs(smooth.Volume() - 1163.53))
				.IsLessThan(0.01)
				.Because($"TruncatedCone vol={smooth.Volume()}");
			await Assert.That(Math.Abs(smooth.SurfaceArea() - 769.33))
				.IsLessThan(0.01)
				.Because($"TruncatedCone sa={smooth.SurfaceArea()}");

			Manifold smooth1 = cone.Clone().SmoothOut(180.0, 1.0).RefineToLength(0.5);
			Manifold smooth2 = cone.SmoothOut(180.0, 0.0).RefineToLength(0.5);
			await Assert.That(Math.Abs(smooth2.Volume() - smooth1.Volume())).IsLessThan(0.01);
			await Assert.That(Math.Abs(smooth2.SurfaceArea() - smooth1.SurfaceArea())).IsLessThan(0.01);
		}

		/// <summary>C++ TEST(Smooth, Mirrored) — mirrored smooth tetrahedron.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothMirrored()
		{
			MeshGL tetGl = Manifold.Tetrahedron()
				.Scale(new Vec3(1.0, 2.0, 3.0))
				.GetMeshGL(0);
			Manifold smooth = Manifold.Smooth(tetGl, Array.Empty<Smoothness>());
			Manifold mirror = smooth.Clone().Scale(new Vec3(-2.0, 2.0, 2.0)).Refine(10);
			Manifold scaled = smooth.Refine(10).Scale(new Vec3(2.0, 2.0, 2.0));
			await Assert.That(Math.Abs(scaled.Volume() - mirror.Volume()))
				.IsLessThan(0.1)
				.Because($"Mirrored vol: {scaled.Volume()} vs {mirror.Volume()}");
			await Assert.That(Math.Abs(scaled.SurfaceArea() - mirror.SurfaceArea()))
				.IsLessThan(0.1)
				.Because($"Mirrored sa: {scaled.SurfaceArea()} vs {mirror.SurfaceArea()}");
		}

		/// <summary>
		/// C++ TEST(Smooth, Tetrahedron) — smooth tetrahedron with curvature check.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothTetrahedron()
		{
			Manifold tet = Manifold.Tetrahedron();
			Manifold smooth = Manifold.Smooth(tet.GetMeshGL(0), Array.Empty<Smoothness>());
			int n = 100;
			Manifold refined = smooth.Refine(n);
			await Assert.That(refined.NumVert()).IsEqualTo((2 * n * n) + 2);
			await Assert.That(refined.NumTri()).IsEqualTo(4 * n * n);
			await Assert.That(Math.Abs(refined.Volume() - 17.0))
				.IsLessThan(0.1)
				.Because($"vol={refined.Volume()}");
			await Assert.That(Math.Abs(refined.SurfaceArea() - 32.9))
				.IsLessThan(0.1)
				.Because($"sa={refined.SurfaceArea()}");
		}

		/// <summary>C++ TEST(Smooth, Csaszar) — smooth Csaszar polyhedron.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothCsaszar()
		{
			MeshGL csaszar = CsaszarGl();
			Manifold smooth = Manifold.Smooth(csaszar, Array.Empty<Smoothness>());
			Manifold refined = smooth.Refine(100);
			await Assert.That(refined.NumVert()).IsEqualTo(70000);
			await Assert.That(refined.NumTri()).IsEqualTo(140000);
			await Assert.That(Math.Abs(refined.Volume() - 78760.0))
				.IsLessThan(10.0)
				.Because($"vol={refined.Volume()}");
			await Assert.That(Math.Abs(refined.SurfaceArea() - 11935.0))
				.IsLessThan(10.0)
				.Because($"sa={refined.SurfaceArea()}");
		}

		/// <summary>C++ TEST(Smooth, Manual) — manually adjusted tangent weights.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothManual()
		{
			MeshGL oct = Manifold.Sphere(1.0, 4).GetMeshGL(0);
			Manifold smoothM = Manifold.Smooth(oct, Array.Empty<Smoothness>());
			MeshGL smoothGl = smoothM.GetMeshGL(0);
			if (smoothGl.HalfedgeTangent.Count > (4 * 22) + 3)
			{
				smoothGl.HalfedgeTangent[(4 * 6) + 3] = 0.0f;
				smoothGl.HalfedgeTangent[(4 * 22) + 3] = 0.0f;
				smoothGl.HalfedgeTangent[(4 * 16) + 3] = 0.0f;
				smoothGl.HalfedgeTangent[(4 * 18) + 3] = 0.0f;
			}

			Manifold interp = Manifold.FromMeshGL(smoothGl).Refine(100);
			await Assert.That(interp.NumVert()).IsEqualTo(40002);
			await Assert.That(interp.NumTri()).IsEqualTo(80000);
			await Assert.That(Math.Abs(interp.Volume() - 3.74))
				.IsLessThan(0.01)
				.Because($"vol={interp.Volume()}");
			await Assert.That(Math.Abs(interp.SurfaceArea() - 11.78))
				.IsLessThan(0.01)
				.Because($"sa={interp.SurfaceArea()}");
		}

		/// <summary>
		/// C++ TEST(Smooth, RefineQuads) — smooth cylinder with position-color properties.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothRefineQuads()
		{
			// C++ uses SmoothOut() which defaults to (60, 0)
			Manifold cylinder = ManifoldTestHelpers
				.WithPositionColors(Manifold.Cylinder(2.0, 1.0, -1.0, 12))
				.SmoothOut(60.0, 0.0)
				.RefineToLength(0.05);
			await Assert.That(cylinder.NumTri())
				.IsEqualTo(17044)
				.Because($"RefineQuads tris={cylinder.NumTri()}");
			double pi = Math.PI;
			await Assert.That(Math.Abs(cylinder.Volume() - (2.0 * pi)))
				.IsLessThan(0.003)
				.Because($"RefineQuads vol={cylinder.Volume()}");
			await Assert.That(Math.Abs(cylinder.SurfaceArea() - (6.0 * pi)))
				.IsLessThan(0.004)
				.Because($"RefineQuads sa={cylinder.SurfaceArea()}");
		}

		/// <summary>C++ TEST(Smooth, Precision) — tolerance-based refinement precision.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothPrecision()
		{
			double tolerance = 0.001;
			double radius = 10.0;
			double height = 10.0;
			Manifold cylinder = Manifold.Cylinder(height, radius, radius, 8);

			// C++ uses SmoothOut() which defaults to (60, 0)
			Manifold smoothed = cylinder
				.SmoothOut(60.0, 0.0)
				.RefineToTolerance(tolerance);
			await Assert.That(smoothed.NumTri())
				.IsEqualTo(7984)
				.Because($"Precision tris={smoothed.NumTri()}");
		}

		/// <summary>
		/// C++ TEST(Smooth, ToLength) — smooth cone with RefineToLength and curvature check.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothToLength()
		{
			CrossSection circle = CrossSection.Circle(10.0, 10).Translate(new Vec2(10.0, 0.0));
			Polygons polygons = circle.ToPolygons();
			Manifold coneBase = Manifold.Extrude(polygons, 2.0, 0, 0.0, new Vec2(0.0, 0.0));
			Manifold cone = coneBase.Union(coneBase.Scale(new Vec3(1.0, 1.0, -5.0)));
			Manifold smooth = cone
				.AsOriginal()
				.Simplify(0.0)
				.SmoothOut(180.0, 0.0)
				.RefineToLength(0.1);

			int numTri = smooth.NumTri();
			int numVert = smooth.NumVert();
			await Assert.That(numTri).IsEqualTo(170496).Because($"ToLength tris={numTri}");
			await Assert.That(numVert).IsEqualTo(85250).Because($"ToLength verts={numVert}");
			await Assert.That(Math.Abs(smooth.Volume() - 4570.0))
				.IsLessThan(1.0)
				.Because($"ToLength vol={smooth.Volume()}");
			await Assert.That(Math.Abs(smooth.SurfaceArea() - 1348.0))
				.IsLessThan(1.0)
				.Because($"ToLength sa={smooth.SurfaceArea()}");

			MeshGL outGl = smooth.CalculateCurvature(-1, 0).GetMeshGL(0);
			int numProp = (int)outGl.NumProp;
			float maxMeanCurvature = 0.0f;
			int i = 3;
			while (i < outGl.VertProperties.Count)
			{
				maxMeanCurvature = Math.Max(maxMeanCurvature, Math.Abs(outGl.VertProperties[i]));
				i += numProp;
			}

			await Assert.That(Math.Abs(maxMeanCurvature - 1.63f))
				.IsLessThan(0.01f)
				.Because($"ToLength maxMeanCurvature={maxMeanCurvature}");
		}

		/// <summary>
		/// C++ TEST(Smooth, Torus) — manually-smoothed torus with CircularTangent.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothTorus()
		{
			CrossSection circle = CrossSection.Circle(1.0, 8).Translate(new Vec2(2.0, 0.0));
			Polygons polygons = circle.ToPolygons();
			MeshGL64 torusMesh = Manifold.Revolve(polygons, 6, 360.0).GetMeshGL64(0);
			int numTri = torusMesh.NumTri();

			// Set toroidal halfedge tangents (CircularTangent for each halfedge)
			ResizeTangents(torusMesh.HalfedgeTangent, 4 * 3 * numTri);
			for (int tri = 0; tri < numTri; tri++)
			{
				(ulong a, ulong b2, ulong c) = torusMesh.GetTriVerts(tri);
				ulong[] triVerts = { a, b2, c };
				for (int i = 0; i < 3; i++)
				{
					int vi = (int)triVerts[i];
					int vi1 = (int)triVerts[(i + 1) % 3];
					(double px, double py, double pz) = torusMesh.GetVertPos(vi);
					(double p1x, double p1y, double p1z) = torusMesh.GetVertPos(vi1);
					Vec3 v = new Vec3(px, py, pz);
					Vec3 v1 = new Vec3(p1x, p1y, p1z);
					Vec3 edge = v1 - v;
					double[] tangent;
					if (edge.Z == 0.0)
					{
						// Horizontal edge — tangent is circumferential
						Vec3 tan = new Vec3(v.Y, -v.X, 0.0);
						if (Dot(tan, edge) < 0.0)
						{
							tan = -tan;
						}

						tangent = CircularTangent(tan, edge);
					}
					else
					{
						double det = (v.X * edge.Y) - (v.Y * edge.X); // 2D determinant of xy parts
						if (Math.Abs(det) < 1e-5)
						{
							// Vertical edge — tangent is poloidal.
							// System.Math.Asin/Cos, not DeterministicMath: the Rust test
							// calls std `f64::asin`/`f64::cos` here. Test fixtures follow the
							// Rust per site — see ManifoldTestHelpers.Gyroid's policy note.
							double theta = Math.Asin(v.Z);
							Vec2 xy = new Vec2(v.X, v.Y);
							double r = Math.Sqrt((xy.X * xy.X) + (xy.Y * xy.Y));
							double scale = v.Z * (r > 2.0 ? -1.0 : 1.0);
							Vec2 xyTan = r > 0.0
								? new Vec2(xy.X / r * scale, xy.Y / r * scale)
								: new Vec2(0.0, 0.0);
							Vec3 tan = new Vec3(xyTan.X, xyTan.Y, Math.Cos(theta));
							if (Dot(tan, edge) < 0.0)
							{
								tan = -tan;
							}

							tangent = CircularTangent(tan, edge);
						}
						else
						{
							// Diagonal edge — no smooth tangent
							tangent = new[] { 0.0, 0.0, 0.0, -1.0 };
						}
					}

					int e = (3 * tri) + i;
					for (int j = 0; j < 4; j++)
					{
						torusMesh.HalfedgeTangent[(4 * e) + j] = tangent[j];
					}
				}
			}

			Manifold smooth = Manifold.FromMeshGL64(torusMesh)
				.RefineToLength(0.1)
				.CalculateCurvature(-1, 0)
				.CalculateNormals(1, 60.0);

			// C++ uses GetMeshGL() (= -1, no normal processing): slot 0 here is mean
			// curvature, and CalculateNormals(1) at a non-standard slot is not recorded,
			// so nothing is auto-substituted.
			MeshGL outGl = smooth.GetMeshGL(-1);
			int numProp = (int)outGl.NumProp;

			// Each vertex has 7 properties: xyz (pos), mean-curvature, normal (3)
			float maxMeanCurvature = 0.0f;
			int idx = 0;
			while (idx + numProp <= outGl.VertProperties.Count)
			{
				double x = outGl.VertProperties[idx];
				double y = outGl.VertProperties[idx + 1];
				double z = outGl.VertProperties[idx + 2];
				Vec3 v = new Vec3(x, y, z);

				// Project to nearest torus centerline (circle of radius 2 in xy-plane)
				Vec3 p = new Vec3(x, y, 0.0);
				double plen = Math.Sqrt((p.X * p.X) + (p.Y * p.Y));
				if (plen > 1e-10)
				{
					p = p * (2.0 / plen);
				}

				double r = Length(v - p);
				await Assert.That(Math.Abs(r - 1.0))
					.IsLessThan(0.006)
					.Because($"Torus vertex r={r} (expected 1.0)");
				maxMeanCurvature = Math.Max(maxMeanCurvature, Math.Abs(outGl.VertProperties[idx + 3]));
				idx += numProp;
			}

			await Assert.That(Math.Abs(maxMeanCurvature - 1.63f))
				.IsLessThan(0.01f)
				.Because($"Torus maxMeanCurvature={maxMeanCurvature}");
		}

		// C++ TEST(Smooth, SineSurface) was removed in v3.5.0 (#1724, "Fix
		// CalculateNormals"): SmoothOut became self-consistent and the test's
		// SmoothByNormals/SmoothOut equivalence assertions no longer applied, so
		// upstream deleted it. Removed here to match.

		/// <summary>C++ TEST(Smooth, SDF) — gyroid SDF with smooth normals.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSmoothSdf()
		{
			double r = 10.0;
			double extra = 2.0;

			Manifold gyroid = Manifold.LevelSet(
				p =>
				{
					// System.Math and not DeterministicMath: the Rust test calls std
					// `f64::cos`/`f64::sin`, not `crate::math::`. Test fixtures follow the
					// Rust per site — see the policy note in ManifoldTestHelpers.Gyroid.
					double g = (Math.Cos(p.X) * Math.Sin(p.Y))
						+ (Math.Cos(p.Y) * Math.Sin(p.Z))
						+ (Math.Cos(p.Z) * Math.Sin(p.X));
					double dist = Math.Sqrt((p.X * p.X) + (p.Y * p.Y) + (p.Z * p.Z));

					// Rust `f64::min` returns the non-NaN operand; Math.Min propagates NaN.
					double d = LinalgFunctions.MinF64(r - dist, 0.0);
					return g - (d * d / 2.0);
				},
				new Box(Vec3.Splat(-r - extra), Vec3.Splat(r + extra)),
				0.5);

			await Assert.That(gyroid.NumTri())
				.IsLessThan(76000)
				.Because($"SDF gyroid tris={gyroid.NumTri()}");
		}

		// ====================================================================
		// Helpers
		// ====================================================================

		/// <summary>C++ Csaszar() — Csaszar polyhedron MeshGL.</summary>
		/// <remarks>
		/// `pub(super)` in the Rust; <c>internal</c> here for the same reason —
		/// ManifoldApiTests reads it, exactly as api.rs calls
		/// <c>super::smooth::csaszar_gl()</c>.
		/// </remarks>
		/// <returns>The Csaszar polyhedron as a MeshGL.</returns>
		internal static MeshGL CsaszarGl()
		{
			MeshGL gl = new MeshGL();
			gl.NumProp = 3;
			gl.VertProperties = new List<float>
			{
				-20.0f, -20.0f, -10.0f, -20.0f, 20.0f, -15.0f, -5.0f, -8.0f, 8.0f, 0.0f, 0.0f, 30.0f,
				5.0f, 8.0f, 8.0f, 20.0f, -20.0f, -15.0f, 20.0f, 20.0f, -10.0f,
			};
			gl.TriVerts = new List<uint>
			{
				1, 3, 6, 1, 6, 5, 2, 5, 6, 0, 2, 6, 0, 6, 4, 3, 4, 6, 1, 2, 3, 1, 4, 2, 1, 0, 4,
				1, 5, 0, 3, 5, 4, 0, 5, 3, 0, 3, 2, 2, 4, 5,
			};
			gl.RunOriginalId = new List<uint> { ManifoldImpl.ReserveIds(1) };
			return gl;
		}

		/// <summary>
		/// C++ CircularTangent() — quadratic bezier tangent for circular interpolation.
		/// </summary>
		/// <remarks>
		/// The test's own transcription, kept independent of the library's
		/// <c>Smoothing.CircularTangent</c> on purpose: the Torus test builds its input
		/// tangents with this and then asserts on what the library did with them.
		/// </remarks>
		/// <param name="tangent">The desired tangent direction.</param>
		/// <param name="edgeVec">The halfedge vector this tangent belongs to.</param>
		/// <returns>The weighted cubic bezier tangent, in geometric form.</returns>
		private static double[] CircularTangent(Vec3 tangent, Vec3 edgeVec)
		{
			double tanLen = Length(tangent);
			Vec3 dir = tanLen > 0.0 ? tangent * (1.0 / tanLen) : tangent;
			double edgeLen = Length(edgeVec);
			double weight = Math.Abs(Dot(dir, edgeVec * (1.0 / edgeLen)));
			if (weight == 0.0)
			{
				weight = 1.0;
			}

			// Quadratic weighted bezier for circular interpolation
			Vec3 bz2Xyz = dir * (edgeLen / (2.0 * weight));
			double[] bz2 = { bz2Xyz.X * weight, bz2Xyz.Y * weight, bz2Xyz.Z * weight, weight };

			// Equivalent cubic weighted bezier: lerp(identity, bz2, 2/3)
			double t = 2.0 / 3.0;
			double[] bz3 =
			{
				((1.0 - t) * 0.0) + (t * bz2[0]),
				((1.0 - t) * 0.0) + (t * bz2[1]),
				((1.0 - t) * 0.0) + (t * bz2[2]),
				((1.0 - t) * 1.0) + (t * bz2[3]),
			};

			// Convert from homogeneous to geometric form
			double w = bz3[3];
			return new[] { bz3[0] / w, bz3[1] / w, bz3[2] / w, w };
		}

		/// <summary>
		/// Rust <c>Vec::resize(n, 0.0)</c>: truncate when longer, pad with zeros when
		/// shorter. <c>List&lt;T&gt;</c> has no single call with that meaning.
		/// </summary>
		/// <param name="values">The list to resize in place.</param>
		/// <param name="length">The target length.</param>
		private static void ResizeTangents(List<double> values, int length)
		{
			if (values.Count > length)
			{
				values.RemoveRange(length, values.Count - length);
			}
			else
			{
				while (values.Count < length)
				{
					values.Add(0.0);
				}
			}
		}
	}
}
