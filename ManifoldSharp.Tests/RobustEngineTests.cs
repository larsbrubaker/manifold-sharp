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

// RobustEngineTests.cs — port of robust/engine_tests.rs, whose header reads:
//
//   End-to-end tests for the robust boolean engine and its selection
//   plumbing: primitive booleans on the Robust engine checked against
//   exact-engine volumes/areas, BooleanConfig default handling, Auto
//   dispatch, fast paths, cancellation, and chained robust operations.
//
// Same inputs, same expected values, same order as the Rust's fifteen tests.
//
// The Rust's `import_stl_like_demo` comes from its `stl_fixtures` child module,
// which is StlFixtures.cs here (with FixtureBytes standing in for
// `include_bytes!`).

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class RobustEngineTests
	{
		/// <summary>
		/// The constraint key that serializes the tests exposed to the process-global
		/// <see cref="BooleanConfig"/> default engine, mirroring
		/// <see cref="TypesTests.QualityGlobalStateKey"/>. TUnit's <c>[NotInParallel]</c>
		/// only promises that tests sharing a key never run concurrently *with each other*,
		/// so the key has to be applied deliberately on both sides.
		/// </summary>
		/// <remarks>
		/// Every plain <c>Union</c> / <c>Difference</c> / <c>Intersection</c> / <c>Boolean</c>
		/// / <c>BatchBoolean</c> / CSG-tree call in the suite IS a read of that default —
		/// which is far too many tests to key. The key is required only where the *operands*
		/// make <see cref="BooleanEngine.Auto"/> resolve differently than
		/// <see cref="BooleanEngine.Exact"/>: on clean, non-self-intersecting manifolds Auto
		/// resolves to Exact and the result is byte-identical, so those readers are immune.
		/// A reader is exposed when its operands are soup or self-intersecting, because Auto
		/// then routes to Robust and answers where Exact refuses.
		/// <para>
		/// The only currently-reachable exposed reader is
		/// <see cref="RobustSoupTests.AllOpsChecklistOnSoup"/>, whose nine assertions expect
		/// <see cref="Error.NotManifold"/> from the plain entry points on a soup operand;
		/// under Auto it would get <see cref="Error.NoError"/> and real geometry. It carries
		/// this key. Any future test that drives a plain boolean on soup or self-intersecting
		/// operands must carry it too.
		/// </para>
		/// </remarks>
		public const string BooleanConfigGlobalStateKey = "BooleanConfigGlobalState";

		[Test]
		public async Task CubeCubeOverlapMatchesExact()
		{
			Manifold a = Manifold.Cube(V(2.0, 2.0, 2.0), false);
			Manifold b = a.Translate(V(1.0, 1.0, 1.0));
			await CheckOpsMatch(a, b, "cube/cube");
		}

		[Test]
		public async Task CubeSphereMatchesExact()
		{
			// Modest tessellation: debug-build exact-rational time grows quickly with
			// intersection count; battery_extended covers denser meshes in release.
			Manifold a = Manifold.Cube(V(2.0, 2.0, 2.0), true);
			Manifold b = Manifold.Sphere(1.3, 12);
			await CheckOpsMatch(a, b, "cube/sphere");
		}

		[Test]
		public async Task SphereCylinderMatchesExact()
		{
			Manifold a = Manifold.Sphere(1.0, 12);
			Manifold b = Manifold.Cylinder(3.0, 0.6, 0.6, 8).Translate(V(0.0, 0.0, -1.5));
			await CheckOpsMatch(a, b, "sphere/cylinder");
		}

		[Test]
		public async Task TetraTetraMatchesExact()
		{
			Manifold a = Manifold.Tetrahedron();
			Manifold b = Manifold.Tetrahedron()
				.Rotate(0.0, 0.0, 45.0)
				.Translate(V(0.2, 0.1, 0.3));
			await CheckOpsMatch(a, b, "tetra/tetra");
		}

		[Test]
		public async Task DisjointAndEmptyFastPaths()
		{
			Manifold a = Manifold.Cube(V(1.0, 1.0, 1.0), false);
			Manifold b = a.Translate(V(5.0, 0.0, 0.0));
			Manifold u = a.UnionWithEngine(b, BooleanEngine.Robust);
			await AssertClose(u.Volume(), 2.0, 1e-12, "disjoint union volume");
			await Assert.That(a.IntersectionWithEngine(b, BooleanEngine.Robust).IsEmpty()).IsTrue();
			await AssertClose(
				a.DifferenceWithEngine(b, BooleanEngine.Robust).Volume(),
				1.0,
				1e-12,
				"disjoint difference");
			Manifold empty = Manifold.Empty();
			await AssertClose(
				a.UnionWithEngine(empty, BooleanEngine.Robust).Volume(),
				1.0,
				1e-12,
				"union with empty");
			await Assert.That(a.IntersectionWithEngine(empty, BooleanEngine.Robust).IsEmpty()).IsTrue();
		}

		/// <summary>
		/// A bbox-disjoint union whose second operand is wound inside-out must still be
		/// classified by the winding rule: {w &gt;= 1} drops the inverted body, {w != 0} keeps
		/// it as positively wound material. The bbox-disjoint fast path used to concatenate
		/// the two soups untouched, passing the inversion straight through under either rule
		/// (found by an FFI test that measured a signed volume of -7 for such a union).
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task DisjointUnionClassifiesInvertedOperand()
		{
			Manifold a = MeshFromTris(CubeTris(0.0, 2.0));
			Manifold b = MeshFromTris(Flipped(CubeTris(5.0, 7.0)));
			await Assert.That(a.Status()).IsEqualTo(Error.NoError);
			await Assert.That(b.Status()).IsEqualTo(Error.NoError);
			await Assert.That(SignedVolume(b)).IsLessThan(0.0).Because("fixture B must import inverted");

			Manifold pos = a.BooleanWithEngineAndRule(
				b, OpType.Add, BooleanEngine.Robust, WindingRule.Positive);
			await Assert.That(pos.Status()).IsEqualTo(Error.NoError);
			await AssertClose(SignedVolume(pos), 8.0, 1e-12, "positive-rule disjoint union");

			Manifold nz = a.BooleanWithEngineAndRule(
				b, OpType.Add, BooleanEngine.Robust, WindingRule.Nonzero);
			await Assert.That(nz.Status()).IsEqualTo(Error.NoError);
			await AssertClose(SignedVolume(nz), 16.0, 1e-12, "nonzero-rule disjoint union");
		}

		/// <summary>
		/// The same gap on the disjoint <c>Subtract</c> fast path, which returned operand A
		/// untouched: an inverted A holds no material under {w &gt;= 1} (so the difference is
		/// empty) and positive material under {w != 0} (so it survives rewound). Only A
		/// matters here — B is discarded whatever it is wound like.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task DisjointSubtractClassifiesInvertedMinuend()
		{
			Manifold a = MeshFromTris(Flipped(CubeTris(0.0, 2.0)));
			Manifold b = MeshFromTris(CubeTris(5.0, 7.0));
			await Assert.That(a.Status()).IsEqualTo(Error.NoError);
			await Assert.That(b.Status()).IsEqualTo(Error.NoError);
			await Assert.That(SignedVolume(a)).IsLessThan(0.0).Because("fixture A must import inverted");

			Manifold pos = a.BooleanWithEngineAndRule(
				b, OpType.Subtract, BooleanEngine.Robust, WindingRule.Positive);
			await Assert.That(pos.Status()).IsEqualTo(Error.NoError);
			await Assert.That(pos.IsEmpty() || pos.Volume() == 0.0).IsTrue()
				.Because($"an inverted minuend bounds no material, got {pos.Volume()}");

			Manifold nz = a.BooleanWithEngineAndRule(
				b, OpType.Subtract, BooleanEngine.Robust, WindingRule.Nonzero);
			await Assert.That(nz.Status()).IsEqualTo(Error.NoError);
			await AssertClose(SignedVolume(nz), 8.0, 1e-12, "nonzero-rule disjoint subtract");
		}

		/// <summary>Clean minuend: the disjoint difference still returns operand A verbatim.</summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task CleanDisjointSubtractKeepsFastPathOutput()
		{
			Manifold a = MeshFromTris(CubeTris(0.0, 2.0));
			Manifold b = MeshFromTris(CubeTris(5.0, 7.0));
			Manifold d = a.DifferenceWithEngine(b, BooleanEngine.Robust);
			await Assert.That(d.Status()).IsEqualTo(Error.NoError);

			// Untouched clone of A, corner-per-vertex import and all (36 verts) —
			// pinned against the pre-gate behavior.
			await Assert.That(d.NumTri()).IsEqualTo(a.NumTri());
			await Assert.That(d.NumVert()).IsEqualTo(a.NumVert());
			await Assert.That(SignedVolume(d)).IsEqualTo(SignedVolume(a));
		}

		/// <summary>
		/// The narrowed gate must not perturb the fast path for well-wound operands: two clean
		/// disjoint cubes still come out of the concatenating path with exactly their input
		/// geometry (no retriangulation, no vertex churn).
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task CleanDisjointUnionKeepsFastPathOutput()
		{
			Manifold a = MeshFromTris(CubeTris(0.0, 2.0));
			Manifold b = MeshFromTris(CubeTris(5.0, 7.0));
			Manifold u = a.UnionWithEngine(b, BooleanEngine.Robust);
			await Assert.That(u.Status()).IsEqualTo(Error.NoError);

			// Pinned against the pre-gate behavior: the two soups concatenated and
			// re-imported, welding each cube's 36 corners back to 8 vertices.
			await Assert.That(u.NumTri()).IsEqualTo(a.NumTri() + b.NumTri());
			await Assert.That(u.NumTri()).IsEqualTo(24);
			await Assert.That(u.NumVert()).IsEqualTo(16);
			await Assert.That(SignedVolume(u)).IsEqualTo(16.0);
			await Assert.That(u.Volume()).IsEqualTo(16.0);
		}

		[Test]
		[NotInParallel(BooleanConfigGlobalStateKey)]
		public async Task GlobalDefaultEngineConfig()
		{
			await Assert.That(BooleanConfig.DefaultEngine()).IsEqualTo(BooleanEngine.Exact);
			BooleanConfig.SetDefaultEngine(BooleanEngine.Auto);
			await Assert.That(BooleanConfig.DefaultEngine()).IsEqualTo(BooleanEngine.Auto);

			// Plain boolean on manifold inputs under Auto → exact path → identical
			// to the explicit Exact call.
			Manifold a = Manifold.Cube(V(2.0, 2.0, 2.0), false);
			Manifold b = a.Translate(V(1.0, 0.0, 0.0));
			Manifold viaAuto = a.Union(b);
			Manifold viaExact = a.UnionWithEngine(b, BooleanEngine.Exact);
			await Assert.That(viaAuto.NumVert()).IsEqualTo(viaExact.NumVert());
			await Assert.That(viaAuto.NumTri()).IsEqualTo(viaExact.NumTri());
			await Assert.That(viaAuto.Volume()).IsEqualTo(viaExact.Volume());
			BooleanConfig.ResetToDefaults();
			await Assert.That(BooleanConfig.DefaultEngine()).IsEqualTo(BooleanEngine.Exact);
		}

		[Test]
		public async Task AutoDispatchesSoupOperandsToRobust()
		{
			// Non-manifold soup (edge-sharing cubes) + a manifold cutter.
			List<Vec3[]> tris = new List<Vec3[]>();
			static Manifold MakeCube(double[] lo, double[] hi)
			{
				return Manifold.Cube(V(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), false)
					.Translate(V(lo[0], lo[1], lo[2]));
			}

			Manifold c1 = MakeCube(new[] { 0.0, 0.0, 0.0 }, new[] { 2.0, 2.0, 2.0 });
			Manifold c2 = MakeCube(new[] { 2.0, 2.0, 0.0 }, new[] { 4.0, 4.0, 2.0 });
			foreach (Manifold m in new[] { c1, c2 })
			{
				MeshGL64 gl = m.GetMeshGL64(-1);
				for (int t = 0; t < gl.NumTri(); t++)
				{
					(ulong A, ulong B, ulong C) tv = gl.GetTriVerts(t);
					Vec3 P(ulong i)
					{
						int o = (int)i * 3;
						return V(gl.VertProperties[o], gl.VertProperties[o + 1], gl.VertProperties[o + 2]);
					}

					tris.Add(new[] { P(tv.A), P(tv.B), P(tv.C) });
				}
			}

			MeshGL64 mesh = new MeshGL64();
			mesh.NumProp = 3;
			foreach (Vec3[] t in tris)
			{
				foreach (Vec3 p in t)
				{
					mesh.VertProperties.Add(p.X);
					mesh.VertProperties.Add(p.Y);
					mesh.VertProperties.Add(p.Z);
				}
			}

			for (ulong i = 0; i < 3 * (ulong)tris.Count; i++)
			{
				mesh.TriVerts.Add(i);
			}

			Manifold soup = Manifold.FromMeshGL64Robust(mesh);
			await Assert.That(soup.AsImpl().IsSoup).IsTrue();
			await AssertClose(soup.Volume(), 16.0, 1e-12, "soup volume");

			Manifold cutter = Manifold.Cube(V(1.0, 1.0, 1.0), false).Translate(V(0.5, 0.5, 0.5));

			// Exact refuses.
			await Assert.That(
				soup.BooleanWithEngine(cutter, OpType.Subtract, BooleanEngine.Exact).Status())
				.IsEqualTo(Error.NotManifold);

			// Auto falls through to robust and produces the right volume.
			Manifold diff = soup.BooleanWithEngine(cutter, OpType.Subtract, BooleanEngine.Auto);
			await Assert.That(diff.Status()).IsEqualTo(Error.NoError);
			await AssertClose(diff.Volume(), 15.0, 1e-9, "auto soup difference volume");

			// The shared edge persists in the output; whether it re-imports as a
			// soup or as an index-paired halfedge mesh depends on how the strict
			// pairing resolves the 4-halfedge fan — both are valid. What matters is
			// that chained booleans keep working either way (below).

			// Chained robust op on the soup result.
			Manifold cutter2 = Manifold.Cube(V(1.0, 1.0, 1.0), false).Translate(V(2.5, 2.5, 0.5));
			Manifold diff2 = diff.BooleanWithEngine(cutter2, OpType.Subtract, BooleanEngine.Auto);
			await Assert.That(diff2.Status()).IsEqualTo(Error.NoError);
			await AssertClose(diff2.Volume(), 14.0, 1e-9, "chained difference volume");
		}

		/// <summary>
		/// Thingi10K #92068 welds into a topologically manifold mesh, but its shells are
		/// triple-wound duplicates of one another, so the exact engine mis-integrates the
		/// union: it reports 1.7699 where the Monte-Carlo ground truth is 1.5333 +/- 0.0111
		/// and the robust engine gives 1.5259. <c>Auto</c> must therefore pick Robust on the
		/// strength of the self-intersection test alone — neither operand is a soup.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task AutoDispatchesSelfIntersectingOperandsToRobust()
		{
			Manifold a = StlFixtures.ImportStlLikeDemo(StlFixtures.FixtureBytes("92068.stl"));
			Manifold b = StlFixtures.ImportStlLikeDemo(StlFixtures.FixtureBytes("39926.stl"))
				.Translate(V(0.3, 0.0, 0.0));
			await Assert.That(a.Status()).IsEqualTo(Error.NoError).Because("operand A import");
			await Assert.That(b.Status()).IsEqualTo(Error.NoError).Because("operand B import");
			await Assert.That(a.AsImpl().IsSoup).IsFalse().Because("operand A welds to a manifold");
			await Assert.That(b.AsImpl().IsSoup).IsFalse().Because("operand B welds to a manifold");
			await Assert.That(a.HasSelfIntersections()).IsTrue()
				.Because("92068's shells are coincident duplicates");

			Manifold auto = a.UnionWithEngine(b, BooleanEngine.Auto);
			Manifold robust = a.UnionWithEngine(b, BooleanEngine.Robust);
			await Assert.That(auto.Status()).IsEqualTo(Error.NoError).Because("auto union status");
			await Assert.That(auto.Volume()).IsEqualTo(robust.Volume())
				.Because("Auto must resolve to the robust engine");
		}

		[Test]
		public async Task RobustCancellation()
		{
			Manifold a = Manifold.Sphere(1.0, 16);
			Manifold b = Manifold.Sphere(1.0, 16).Translate(V(0.5, 0.0, 0.0));
			CancelToken token = new CancelToken();
			token.Cancel();
			Manifold r = a.BooleanWithEngineAndToken(b, OpType.Add, BooleanEngine.Robust, token);
			await Assert.That(r.Status()).IsEqualTo(Error.Cancelled);
		}

		[Test]
		public async Task CoincidentCubesUnionAndSubtract()
		{
			// Identical operands: union == either, difference == empty — classic
			// failure cases for tolerance-based engines, exact here.
			Manifold a = Manifold.Cube(V(2.0, 2.0, 2.0), false);
			Manifold u = a.UnionWithEngine(a.Clone(), BooleanEngine.Robust);
			await AssertClose(u.Volume(), 8.0, 1e-12, "self union volume");
			await AssertClose(u.SurfaceArea(), 24.0, 1e-12, "self union area");
			Manifold d = a.DifferenceWithEngine(a.Clone(), BooleanEngine.Robust);
			await Assert.That(d.IsEmpty() || Math.Abs(d.Volume()) < 1e-12).IsTrue()
				.Because("self difference must vanish");
			Manifold i = a.IntersectionWithEngine(a.Clone(), BooleanEngine.Robust);
			await AssertClose(i.Volume(), 8.0, 1e-12, "self intersection volume");
		}

		[Test]
		public async Task FaceTouchingUnionMergesCleanly()
		{
			// Stacked cubes sharing a full face: the union must dissolve the shared
			// wall (volume 16, area of a 2x2x4 box).
			Manifold a = Manifold.Cube(V(2.0, 2.0, 2.0), false);
			Manifold b = a.Translate(V(0.0, 0.0, 2.0));
			Manifold u = a.UnionWithEngine(b, BooleanEngine.Robust);
			await Assert.That(u.Status()).IsEqualTo(Error.NoError);
			await AssertClose(u.Volume(), 16.0, 1e-12, "stacked union volume");
			await AssertClose(u.SurfaceArea(), 40.0, 1e-12, "stacked union area");
		}

		/// <summary>Shorthand for the Rust's <c>v(x, y, z)</c>.</summary>
		/// <param name="x">The x coordinate.</param>
		/// <param name="y">The y coordinate.</param>
		/// <param name="z">The z coordinate.</param>
		/// <returns>The vector.</returns>
		internal static Vec3 V(double x, double y, double z)
		{
			return new Vec3(x, y, z);
		}

		/// <summary>The Rust's <c>assert_close</c>: relative tolerance with a floor of 1.</summary>
		/// <param name="a">The measured value.</param>
		/// <param name="b">The expected value.</param>
		/// <param name="tol">The relative tolerance.</param>
		/// <param name="what">What is being compared.</param>
		/// <returns>The assertion task.</returns>
		internal static async Task AssertClose(double a, double b, double tol, string what)
		{
			await Assert.That(Math.Abs(a - b) <= tol * Math.Max(Math.Abs(b), 1.0)).IsTrue()
				.Because($"{what}: {a} vs {b}");
		}

		/// <summary>
		/// Axis-aligned cube [lo,hi]³ as 12 outward-wound triangles (same fixture shape
		/// <see cref="RobustRepairTests"/> uses, in f64 so it imports without rounding).
		/// </summary>
		/// <param name="lo">The low coordinate on every axis.</param>
		/// <param name="hi">The high coordinate on every axis.</param>
		/// <returns>The cube's triangles.</returns>
		private static List<Vec3[]> CubeTris(double lo, double hi)
		{
			return RobustRepairTests.CubeTris(lo, hi);
		}

		/// <summary>Every triangle reversed.</summary>
		/// <param name="tris">The triangles.</param>
		/// <returns>The reversed triangles.</returns>
		private static List<Vec3[]> Flipped(IReadOnlyList<Vec3[]> tris)
		{
			return RobustRepairTests.Flipped(tris);
		}

		/// <summary>
		/// Signed (divergence-theorem) volume — <c>Volume()</c> takes the absolute value,
		/// which hides exactly the inversion these tests are about.
		/// </summary>
		/// <param name="m">The manifold.</param>
		/// <returns>Its signed volume.</returns>
		private static double SignedVolume(Manifold m)
		{
			return RobustRepairTests.SignedVolume(m);
		}

		/// <summary>The Rust's f64 <c>MeshGL64</c> fixture import, one corner per vertex.</summary>
		/// <param name="tris">The triangles.</param>
		/// <returns>The imported manifold.</returns>
		private static Manifold MeshFromTris(IReadOnlyList<Vec3[]> tris)
		{
			MeshGL64 mesh = new MeshGL64();
			mesh.NumProp = 3;
			foreach (Vec3[] t in tris)
			{
				foreach (Vec3 p in t)
				{
					mesh.VertProperties.Add(p.X);
					mesh.VertProperties.Add(p.Y);
					mesh.VertProperties.Add(p.Z);
				}
			}

			for (ulong i = 0; i < (ulong)(tris.Count * 3); i++)
			{
				mesh.TriVerts.Add(i);
			}

			return Manifold.FromMeshGL64Robust(mesh);
		}

		/// <summary>
		/// Both engines on the same manifold inputs must agree on volume and area to
		/// near-f64 precision, and the robust result must be a true manifold.
		/// </summary>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <param name="what">The pair's name, for failure messages.</param>
		/// <returns>The assertion task.</returns>
		private static async Task CheckOpsMatch(Manifold a, Manifold b, string what)
		{
			foreach (OpType op in new[] { OpType.Add, OpType.Subtract, OpType.Intersect })
			{
				Manifold exact = a.BooleanWithEngine(b, op, BooleanEngine.Exact);
				Manifold robust = a.BooleanWithEngine(b, op, BooleanEngine.Robust);
				await Assert.That(robust.Status()).IsEqualTo(Error.NoError).Because($"{what} {op}");
				await Assert.That(robust.AsImpl().IsSoup).IsFalse()
					.Because($"{what} {op}: output must be manifold");
				await AssertClose(robust.Volume(), exact.Volume(), 1e-9, $"{what} {op} volume");
				await AssertClose(robust.SurfaceArea(), exact.SurfaceArea(), 1e-9, $"{what} {op} area");
				await Assert.That(robust.Genus()).IsEqualTo(exact.Genus()).Because($"{what} {op} genus");
			}
		}
	}
}
