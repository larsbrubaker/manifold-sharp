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

// The second half of the basic.rs port, split from ManifoldBasicTests.cs for the
// 800-line file cap. That file carries the module header and the DEFERRED table;
// this one continues the same class and the same order, from
// `test_sphere_is_round` to the end of the Rust module.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldBasicTests
	{
		[Test]
		public async Task SphereIsRound()
		{
			Manifold m = Manifold.Sphere(1.0, 24);

			// Unit sphere: volume approaches 4π/3 ≈ 4.189 as resolution increases.
			// With exact n-way subdivision, Sphere(1, 24) uses n = (24+3)/4 = 6
			// divisions per octahedron edge — a genuinely coarse sphere (vol ≈ 4.02),
			// so this roundness check uses a 0.2 tolerance rather than assuming the
			// old over-refined (power-of-2) tessellation.
			double vol = m.Volume();
			double expected = 4.0 * Types.KPi / 3.0;
			await Assert.That(Math.Abs(vol - expected))
				.IsLessThan(0.2)
				.Because($"Sphere volume should be ~{expected:F3}, got {vol:F3}");

			// All vertices should be approximately at radius 1.0
			MeshGL mesh = m.GetMeshGL(0);
			int numProp = (int)mesh.NumProp;
			int vertCount = numProp > 0 ? mesh.VertProperties.Count / numProp : 0;
			for (int i = 0; i < vertCount; i++)
			{
				double x = mesh.VertProperties[i * numProp];
				double y = mesh.VertProperties[(i * numProp) + 1];
				double z = mesh.VertProperties[(i * numProp) + 2];
				double r = Math.Sqrt((x * x) + (y * y) + (z * z));
				await Assert.That(Math.Abs(r - 1.0))
					.IsLessThan(0.01)
					.Because($"Vertex {i} at ({x:F3},{y:F3},{z:F3}) has radius {r:F4}, expected ~1.0");
			}
		}

		[Test]
		public async Task SetPropertiesRoundtrip()
		{
			// Verify set_properties correctly assigns per-vertex properties
			Manifold cube = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), true);
			Manifold colored = cube.SetProperties(3, (Span<double> props, Vec3 pos, ReadOnlySpan<double> old) =>
			{
				props[0] = 1.0; // R
				props[1] = 0.0; // G
				props[2] = 0.0; // B
			});
			MeshGL gl = colored.GetMeshGL(0);
			int numProp = (int)gl.NumProp;
			await Assert.That(numProp).IsEqualTo(6); // 3 xyz + 3 RGB
			int vertCount = gl.VertProperties.Count / numProp;
			await Assert.That(vertCount).IsGreaterThan(0);

			// All vertices should have R=1, G=0, B=0
			for (int i = 0; i < vertCount; i++)
			{
				float r = gl.VertProperties[(i * numProp) + 3];
				float g = gl.VertProperties[(i * numProp) + 4];
				float b = gl.VertProperties[(i * numProp) + 5];
				await Assert.That(Math.Abs(r - 1.0f)).IsLessThan(1e-6f).Because($"Vertex {i} R={r}, expected 1.0");
				await Assert.That(Math.Abs(g)).IsLessThan(1e-6f).Because($"Vertex {i} G={g}, expected 0.0");
				await Assert.That(Math.Abs(b)).IsLessThan(1e-6f).Because($"Vertex {i} B={b}, expected 0.0");
			}
		}

		/// <summary>C++ TEST(Properties, Measurements) — basic volume/area.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesMeasurements()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			await Assert.That(Math.Abs(cube.Volume() - 1.0))
				.IsLessThan(1e-6)
				.Because($"cube volume: {cube.Volume()}");
			await Assert.That(Math.Abs(cube.SurfaceArea() - 6.0))
				.IsLessThan(1e-6)
				.Because($"cube area: {cube.SurfaceArea()}");

			// Scale by -1 should still have same volume/area (flips orientation but
			// absolute values same)
			Manifold flipped = cube.Scale(Vec3.Splat(-1.0));
			await Assert.That(Math.Abs(flipped.Volume() - 1.0))
				.IsLessThan(1e-6)
				.Because($"flipped cube volume: {flipped.Volume()}");
			await Assert.That(Math.Abs(flipped.SurfaceArea() - 6.0))
				.IsLessThan(1e-6)
				.Because($"flipped cube area: {flipped.SurfaceArea()}");
		}

		/// <summary>C++ TEST(Properties, Epsilon) — epsilon scales with geometry.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesEpsilon()
		{
			double kPrecision = Types.KPrecision;
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			await Assert.That(Math.Abs(cube.GetTolerance() - kPrecision))
				.IsLessThan(kPrecision * 0.1)
				.Because($"unit cube epsilon: {cube.GetTolerance()} expected ~{kPrecision}");

			Manifold scaled = cube.Scale(new Vec3(0.1, 1.0, 10.0));
			await Assert.That(Math.Abs(scaled.GetTolerance() - (10.0 * kPrecision)))
				.IsLessThan(kPrecision)
				.Because($"scaled cube epsilon: {scaled.GetTolerance()} expected ~{10.0 * kPrecision}");

			Manifold translated = scaled.Translate(new Vec3(-100.0, -10.0, -1.0));
			await Assert.That(Math.Abs(translated.GetTolerance() - (100.0 * kPrecision)))
				.IsLessThan(kPrecision * 10.0)
				.Because($"translated cube epsilon: {translated.GetTolerance()} expected ~{100.0 * kPrecision}");
		}

		/// <summary>C++ TEST(Properties, Epsilon2) — epsilon after translate+scale.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesEpsilon2()
		{
			double kPrecision = Types.KPrecision;
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false)
				.Translate(new Vec3(-0.5, 0.0, 0.0))
				.Scale(new Vec3(2.0, 1.0, 1.0));
			await Assert.That(Math.Abs(cube.GetTolerance() - (2.0 * kPrecision)))
				.IsLessThan(kPrecision)
				.Because($"epsilon2: {cube.GetTolerance()} expected ~{2.0 * kPrecision}");
		}

		/// <summary>C++ TEST(Properties, Coplanar) — coplanar check on primitives.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesCoplanar()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(1.0), false);
			await Assert.That(cube.MatchesTriNormals()).IsTrue().Because("Cube should match tri normals");
			await Assert.That(cube.NumDegenerateTris())
				.IsEqualTo(0)
				.Because("Cube should have no degenerate tris");

			Manifold tet = Manifold.Tetrahedron();
			await Assert.That(tet.MatchesTriNormals())
				.IsTrue()
				.Because("Tetrahedron should match tri normals");
		}

		/// <summary>C++ TEST(Manifold, MirrorUnion) — full version with Mirror API.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMirrorUnionFull()
		{
			Manifold a = Manifold.Cube(Vec3.Splat(5.0), true);
			Manifold b = a.Translate(new Vec3(2.5, 2.5, 2.5));
			Manifold bMirrored = b.Mirror(new Vec3(1.0, 1.0, 0.0));
			Manifold result = a.Union(b).Union(bMirrored);

			double volA = a.Volume();
			await Assert.That(Math.Abs(result.Volume() - (volA * 2.75)))
				.IsLessThan(1e-5)
				.Because($"volume: {result.Volume()} expected: {volA * 2.75}");

			// Mirror with zero normal should return empty
			await Assert.That(a.Mirror(new Vec3(0.0, 0.0, 0.0)).IsEmpty()).IsTrue();
		}

		/// <summary>C++ TEST(Manifold, Invalid).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppInvalidConstructors()
		{
			// Zero-size constructors should return InvalidConstruction
			await Assert.That(Manifold.Sphere(0.0, 16).Status()).IsEqualTo(Error.InvalidConstruction);
			await Assert.That(Manifold.Cylinder(0.0, 5.0, -1.0, 16).Status()).IsEqualTo(Error.InvalidConstruction);
			await Assert.That(Manifold.Cylinder(2.0, -5.0, -1.0, 16).Status()).IsEqualTo(Error.InvalidConstruction);
			await Assert.That(Manifold.Cylinder(2.0, 0.0, -1.0, 16).Status()).IsEqualTo(Error.InvalidConstruction);
			await Assert.That(Manifold.Cylinder(2.0, 0.0, 0.0, 16).Status()).IsEqualTo(Error.InvalidConstruction);
			await Assert.That(Manifold.Cube(new Vec3(0.0, 0.0, 0.0), false).Status())
				.IsEqualTo(Error.InvalidConstruction);
			await Assert.That(Manifold.Cube(new Vec3(-1.0, 1.0, 1.0), false).Status())
				.IsEqualTo(Error.InvalidConstruction);

			// Empty extrude
			Polygons emptyPoly = new Polygons();
			await Assert.That(Manifold.Extrude(emptyPoly, 0.0, 0, 0.0, new Vec2(1.0, 1.0)).Status())
				.IsEqualTo(Error.InvalidConstruction);
		}

		/// <summary>C++ TEST(Manifold, MeshDeterminism).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppMeshDeterminism()
		{
			Manifold cube1 = Manifold.Cube(new Vec3(2.0, 2.0, 2.0), true);
			Manifold cube2 = Manifold.Cube(new Vec3(2.0, 2.0, 2.0), true)
				.Translate(new Vec3(-1.1091, 0.88509, 1.3099));
			Manifold result = cube1 - cube2;
			MeshGL outMesh = result.GetMeshGL(3);

			// C++ expected triVerts and vertProperties — verify deterministic output
			List<uint> expectedTriVerts = new List<uint>
			{
				0, 2, 7, 0, 10, 1, 0, 6, 10, 0, 1, 2, 1, 3, 2, 1, 5, 3, 1, 11, 5, 0, 7, 6, 6, 7, 8, 6, 8,
				13, 10, 12, 11, 1, 10, 11, 11, 13, 5, 6, 12, 10, 6, 13, 12, 13, 9, 5, 13, 8, 9, 11, 12, 13,
				4, 2, 3, 4, 3, 5, 4, 7, 2, 4, 5, 8, 4, 8, 7, 9, 8, 5,
			};
			List<float> expectedVertProps = new List<float>
			{
				-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -0.11491f, 0.3099f, -1.0f, -0.11491f, 1.0f, -0.1091f,
				-0.11491f, 0.3099f, -0.1091f, -0.11491f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 0.3099f, -0.1091f, 1.0f,
				0.3099f, -0.1091f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
			};

			// CollectionOrdering.Matching: a ported assert_eq! on a Vec is a SEQUENCE
			// comparison, and TUnit's IsEquivalentTo defaults to order-insensitive.
			await Assert.That(outMesh.TriVerts)
				.IsEquivalentTo(expectedTriVerts, CollectionOrdering.Matching)
				.Because("MeshDeterminism: triVerts mismatch");

			// Check vertex properties with tolerance
			await Assert.That(outMesh.VertProperties.Count)
				.IsEqualTo(expectedVertProps.Count)
				.Because(
					"MeshDeterminism: vertProperties length mismatch: "
					+ $"{outMesh.VertProperties.Count} vs {expectedVertProps.Count}");
			for (int i = 0; i < outMesh.VertProperties.Count; i++)
			{
				float actual = outMesh.VertProperties[i];
				float expected = expectedVertProps[i];
				await Assert.That(Math.Abs(actual - expected))
					.IsLessThan(1e-4f)
					.Because($"MeshDeterminism: vertProperties[{i}] = {actual} expected {expected}");
			}
		}

		/// <summary>C++ TEST(Manifold, Slice) — slice a cube at z=0 and z=1.</summary>
		/// <returns>A task representing the test.</returns>
		/// <remarks>
		/// The Rust's assertions are on <c>area()</c>, which sums signed contour areas and
		/// is therefore invariant under both contour order and the rotation of a contour's
		/// vertex list — so the pinned slice seeding (docs/RUST_DIVERGENCES.md entry 3)
		/// does not reach this expected value, and it is ported exactly as written,
		/// <c>assert_eq!</c> and all.
		/// </remarks>
		[Test]
		public async Task CppManifoldSlice()
		{
			Manifold cube = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false);
			CrossSection bottom = cube.Slice(0.0);
			CrossSection top = cube.Slice(1.0);
			await Assert.That(bottom.Area())
				.IsEqualTo(1.0)
				.Because($"Slice at z=0 should have area 1, got {bottom.Area()}");
			await Assert.That(top.Area())
				.IsEqualTo(0.0)
				.Because($"Slice at z=1 should have area 0, got {top.Area()}");
		}

		/// <summary>C++ TEST(Manifold, SliceEmptyObject).</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppManifoldSliceEmptyObject()
		{
			Manifold empty = Manifold.Empty();
			await Assert.That(empty.IsEmpty()).IsTrue();
			CrossSection bottom = empty.Slice(0.0);
			await Assert.That(bottom.Area()).IsEqualTo(0.0).Because("Slice of empty should have area 0");
		}

		/// <summary>C++ TEST(Manifold, Project) — project a mesh onto XY plane.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppManifoldProject()
		{
			MeshGL input = new MeshGL();
			input.NumProp = 3;
			input.VertProperties = new List<float>
			{
				0.0f, 0.0f, 0.0f, -2.0f, -0.7f, -0.1f, -2.0f, -0.7f, 0.0f, -1.9f, -0.7f, -0.1f, -1.9f, -0.6901f,
				-0.1f, -1.9f, -0.7f, 0.0f, -1.9f, -0.6901f, 0.0f, -2.0f, -1.0f, 3.0f, -1.9f, -1.0f, 3.0f, -2.0f,
				-1.0f, 4.0f, -1.9f, -1.0f, 4.0f, -1.9f, -0.6901f, 3.0f, -1.9f, -0.6901f, 4.0f, -1.7f, -0.6901f, 3.0f,
				-1.7f, -0.6901f, 3.2f, -2.0f, 0.0f, -0.1f, -2.0f, 0.0f, 0.0f, -2.0f, 0.0f, 3.0f, -2.0f, 0.0f, 4.0f,
				-1.7f, 0.0f, 3.0f, -1.7f, 0.0f, 3.2f, -1.0f, -0.6901f, -0.1f, -1.0f, -0.6901f, 0.0f, -1.0f, -0.6901f,
				3.2f, -1.0f, -0.6901f, 4.0f, -1.0f, 0.0f, -0.1f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 3.2f, -1.0f, 0.0f,
				4.0f,
			};
			input.TriVerts = new List<uint>
			{
				1, 3, 2, 1, 4, 3, 2, 3, 5, 5, 6, 2, 3, 4, 6, 5, 3, 6, 6, 4, 21, 26, 22, 25, 21, 25, 22,
				25, 15, 26, 26, 6, 22, 21, 4, 25, 21, 22, 6, 16, 26, 15, 16, 6, 26, 4, 15, 25, 15, 1,
				16, 16, 2, 6, 4, 1, 15, 1, 2, 16, 12, 14, 23, 12, 13, 14, 12, 11, 13, 18, 9, 12, 11, 7,
				17, 7, 9, 18, 17, 7, 18, 13, 11, 19, 17, 18, 20, 19, 11, 17, 19, 17, 20, 14, 13, 20,
				18, 12, 24, 20, 13, 19, 20, 18, 27, 12, 10, 11, 24, 12, 23, 9, 10, 12, 9, 8, 10, 8, 11,
				10, 8, 7, 11, 8, 9, 7, 14, 20, 27, 24, 28, 18, 27, 18, 28, 23, 14, 27, 24, 23, 28, 28,
				23, 27,
			};
			Manifold m = Manifold.FromMeshGL(input);
			CrossSection projected = m.Project();
			double area = projected.Area();
			await Assert.That(Math.Abs(area - 0.72))
				.IsLessThan(0.01)
				.Because($"Project area: {area} expected ~0.72");
		}

		/// <summary>C++ TEST(Manifold, GetMeshGL) — sphere round-trip through MeshGL.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppManifoldGetMeshGl()
		{
			Manifold m1 = Manifold.Sphere(0.01, 0);
			MeshGL mesh1 = m1.GetMeshGL(3);
			Manifold m2 = Manifold.FromMeshGL(mesh1);
			MeshGL mesh2 = m2.GetMeshGL(3);

			// Check same number of vertices and triangles
			int nv1 = mesh1.VertProperties.Count / (int)mesh1.NumProp;
			int nv2 = mesh2.VertProperties.Count / (int)mesh2.NumProp;
			await Assert.That(nv1).IsEqualTo(nv2).Because($"GetMeshGL: vertex count mismatch {nv1} vs {nv2}");
			await Assert.That(mesh1.TriVerts.Count)
				.IsEqualTo(mesh2.TriVerts.Count)
				.Because("GetMeshGL: triVerts length mismatch");

			// Check vertex positions match
			for (int i = 0; i < nv1; i++)
			{
				(float p1X, float p1Y, float p1Z) = mesh1.GetVertPos(i);
				(float p2X, float p2Y, float p2Z) = mesh2.GetVertPos(i);
				float dx = p1X - p2X;
				float dy = p1Y - p2Y;
				float dz = p1Z - p2Z;
				double dist = Math.Sqrt((dx * dx) + (dy * dy) + (dz * dz));
				await Assert.That(dist).IsLessThanOrEqualTo(0.0001).Because($"GetMeshGL: vertex {i} distance {dist}");
			}
		}

		/// <summary>C++ TEST(Manifold, WarpBatch) — warp vs warp_batch produce same results.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppManifoldWarpBatch()
		{
			Manifold cube = Manifold.Cube(new Vec3(2.0, 3.0, 4.0), false);
			int id = cube.OriginalId();

			Manifold shape1 = cube.Warp((ref Vec3 v) => { v.X += v.Z * v.Z; });
			Manifold shape2 = cube.WarpBatch(vecs =>
			{
				for (int i = 0; i < vecs.Length; i++)
				{
					vecs[i].X += vecs[i].Z * vecs[i].Z;
				}
			});

			await Assert.That(id).IsGreaterThanOrEqualTo(0).Because("WarpBatch: original ID should be >= 0");
			await Assert.That(shape1.OriginalId()).IsEqualTo(-1).Because("WarpBatch: warped shape1 should have ID -1");
			await Assert.That(shape2.OriginalId()).IsEqualTo(-1).Because("WarpBatch: warped shape2 should have ID -1");

			// Check run_original_id
			MeshGL gl1 = shape1.GetMeshGL(3);
			await Assert.That(gl1.RunOriginalId.Count).IsEqualTo(1).Because("WarpBatch: shape1 should have 1 run");
			await Assert.That(gl1.RunOriginalId[0]).IsEqualTo((uint)id).Because("WarpBatch: shape1 run ID mismatch");

			MeshGL gl2 = shape2.GetMeshGL(3);
			await Assert.That(gl2.RunOriginalId.Count).IsEqualTo(1).Because("WarpBatch: shape2 should have 1 run");
			await Assert.That(gl2.RunOriginalId[0]).IsEqualTo((uint)id).Because("WarpBatch: shape2 run ID mismatch");

			await Assert.That(shape1.Volume()).IsEqualTo(shape2.Volume()).Because("WarpBatch: volumes differ");
			await Assert.That(shape1.SurfaceArea()).IsEqualTo(shape2.SurfaceArea()).Because("WarpBatch: areas differ");
		}

		/// <summary>C++ TEST(Manifold, Warp2) — extrude circle then warp into arc.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppManifoldWarp2()
		{
			CrossSection circle = CrossSection.Circle(5.0, 20).Translate(new Vec2(10.0, 10.0));
			Manifold shape = Manifold
				.Extrude(circle.ToPolygons(), 2.0, 10, 0.0, new Vec2(1.0, 1.0))
				.Warp((ref Vec3 v) =>
				{
					// The Rust test calls std `f64::sin`/`cos`; the port uses
					// DeterministicMath, which is the port's rule for reproducible math.
					// The two differ by at most an ULP and every assertion below is
					// orders of magnitude looser than that.
					int nSegments = 10;
					double angleStep = 2.0 / 3.0 * Types.KPi / nSegments;
					int zIndex = nSegments - 1 - (int)Math.Round(v.Z, MidpointRounding.AwayFromZero);
					double angle = zIndex * angleStep;
					double newZ = v.Y;
					double newY = v.X * DeterministicMath.Sin(angle);
					double newX = v.X * DeterministicMath.Cos(angle);
					v.X = newX;
					v.Y = newY;
					v.Z = newZ;
				});

			Manifold simplified = Manifold.BatchBoolean(new List<Manifold> { shape.Clone() }, OpType.Add);

			await Assert.That(Math.Abs(shape.Volume() - simplified.Volume()))
				.IsLessThan(0.0001)
				.Because($"Warp2: volume mismatch {shape.Volume()} vs {simplified.Volume()}");
			await Assert.That(Math.Abs(shape.SurfaceArea() - simplified.SurfaceArea()))
				.IsLessThan(0.0001)
				.Because($"Warp2: area mismatch {shape.SurfaceArea()} vs {simplified.SurfaceArea()}");
			await Assert.That(Math.Abs(shape.Volume() - 321.0))
				.IsLessThan(1.0)
				.Because($"Warp2: volume {shape.Volume()} expected ~321");
		}

		/// <summary>C++ TEST(Manifold, FaceIDRoundTrip) — faceID preserved through round-trip.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppManifoldFaceIdRoundTrip()
		{
			Manifold cube = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), false);
			await Assert.That(cube.OriginalId()).IsGreaterThanOrEqualTo(0);
			MeshGL inGl = cube.GetMeshGL(3);

			// Cube has 12 tris, 6 faces → 6 unique faceIDs
			HashSet<uint> uniqueIn = new HashSet<uint>(inGl.FaceId);
			await Assert.That(uniqueIn.Count)
				.IsEqualTo(6)
				.Because($"FaceIDRoundTrip: expected 6 unique faceIDs, got {uniqueIn.Count}");

			// Override with just 2 unique values
			inGl.FaceId = new List<uint> { 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5 };

			Manifold cube2 = Manifold.FromMeshGL(inGl);
			MeshGL outGl = cube2.GetMeshGL(3);
			HashSet<uint> uniqueOut = new HashSet<uint>(outGl.FaceId);
			await Assert.That(uniqueOut.Count)
				.IsEqualTo(2)
				.Because($"FaceIDRoundTrip: expected 2 unique faceIDs, got {uniqueOut.Count}");
		}
	}
}
