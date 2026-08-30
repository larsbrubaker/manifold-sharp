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

// RobustThingiTests.cs — port of robust/thingi_tests.rs, whose header reads:
//
//   Regression tests against real-world Thingi10K meshes (checked into
//   src/robust/testdata/). These reproduce failures found through the WASM
//   demo's Boolean Gallery: both operands import cleanly via
//   `Manifold::from_mesh_gl_robust`, yet a robust boolean of the pair returned
//   `Error::NotClosed`. The fixtures follow the demo's exact pipeline: binary
//   STL -> f32 triangle soup -> normalize (center at bbox center, max side =
//   2.0, f32 storage with f64 arithmetic, as in JS) -> MeshGL::merge ->
//   robust import.
//
// Same inputs, same expected values, same order as the Rust's thirteen tests.
//
// ── The Rust's duplicated import pipeline is folded here ─────────────────────
// thingi_tests.rs carries its own copy of `import_stl_like_demo`; stl_fixtures.rs
// says so and calls folding the two "a follow-up, deliberately not done here to
// avoid touching that file". This port takes that follow-up, because the copies
// differ only in spelling: `if x < min[k] { min[k] = x }` versus
// `min[k] = min[k].min(x)`, which agree on every non-NaN input and so on every
// fixture. StlFixtures.ImportStlLikeDemo is the one pipeline, and the Rust's
// `include_bytes!("testdata/…")` constants become FixtureBytes calls.

using ManifoldSharp;
using ManifoldSharp.Linalg;
using ManifoldSharp.Robust;
using ManifoldSharp.Robust.Exact;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class RobustThingiTests
	{
		private const string CastleStairs59082 = "59082.stl";
		private const string Ground1313535 = "1313535.stl";
		private const string Tentacle939888 = "939888.stl";
		private const string Pickaxe93557 = "93557.stl";
		private const string Model92068 = "92068.stl";
		private const string Model39926 = "39926.stl";
		private const string Frame1075458 = "1075458.stl";
		private const string Tower91115 = "91115.stl";
		private const string Model74660 = "74660.stl";
		private const string Model1147177 = "1147177.stl";
		private const string Recognizer68730 = "68730.stl";
		private const string Cuboctahedron61459 = "61459.stl";
		private const string Model36088 = "36088.stl";
		private const string Model301921 = "301921.stl";
		private const string Model36374 = "36374.stl";
		private const string TopWalls51360 = "51360.stl";
		private const string Model90225 = "90225.stl";

		[Test]
		public async Task Thingi59082ArrangementIsConsistent()
		{
			Manifold a = Import(CastleStairs59082);
			Manifold b = Import(Ground1313535)
				.Rotate(162.0, 156.0, 337.0)
				.Translate(new Vec3(0.3, 0.0, 0.0));
			await AssertArrangementConsistent(a, b, "59082 / 1313535");
		}

		[Test]
		public async Task Thingi74660ArrangementIsConsistent()
		{
			Manifold a = Import(Model74660);
			Manifold b = Import(Model1147177).Translate(new Vec3(0.3, 0.0, 0.0));
			await AssertArrangementConsistent(a, b, "74660 / 1147177");
		}

		/// <summary>
		/// Thingi10K #74660 union #1147177 (second demo repro of the same NotClosed family):
		/// the demo's default translate(0.3, 0, 0), no rotation.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task Thingi74660Union1147177IsClosed()
		{
			Manifold a = Import(Model74660);
			await Assert.That(a.Status()).IsEqualTo(Error.NoError).Because("operand A import");

			Manifold b = Import(Model1147177).Translate(new Vec3(0.3, 0.0, 0.0));
			await Assert.That(b.Status()).IsEqualTo(Error.NoError).Because("operand B import");

			Manifold result = a.UnionWithEngine(b, BooleanEngine.Robust);
			await Assert.That(result.Status()).IsEqualTo(Error.NoError).Because("robust union status");
			await Assert.That(result.IsEmpty()).IsFalse().Because("robust union should be non-empty");
			await Assert.That(result.Volume()).IsGreaterThan(0.0)
				.Because("union volume must be positive");
		}

		/// <summary>
		/// Thingi10K #92068 union #39926 (demo repro): both operands import as closed
		/// manifolds; the robust union returned NotClosed. Reproduces with no rotation at all
		/// — just the demo's translate(0.3, 0, 0) on operand B.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task Thingi92068Union39926IsClosed()
		{
			Manifold a = Import(Model92068);
			await Assert.That(a.Status()).IsEqualTo(Error.NoError).Because("operand A import");
			await Assert.That(a.AsImpl().IsSoup).IsFalse()
				.Because("operand A should weld to a manifold");

			Manifold b = Import(Model39926).Translate(new Vec3(0.3, 0.0, 0.0));
			await Assert.That(b.Status()).IsEqualTo(Error.NoError).Because("operand B import");
			await Assert.That(b.AsImpl().IsSoup).IsFalse()
				.Because("operand B should weld to a manifold");

			Manifold result = a.UnionWithEngine(b, BooleanEngine.Robust);
			await Assert.That(result.Status()).IsEqualTo(Error.NoError).Because("robust union status");
			await Assert.That(result.IsEmpty()).IsFalse().Because("robust union should be non-empty");
			await Assert.That(result.Volume()).IsGreaterThan(0.0)
				.Because("union volume must be positive");
		}

		/// <summary>
		/// Thingi10K #1075458 ("frame 1 n") minus #91115 ("castle corner tower"), demo repro:
		/// both operands import cleanly, yet the robust difference panicked (surfacing as
		/// <c>RuntimeError: unreachable</c> in WASM) with B rotated (311, 55, 345) and
		/// translated (0.7, -0.2, 0.4).
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task Thingi1075458Minus91115IsValid()
		{
			Manifold a = Import(Frame1075458);
			await Assert.That(a.Status()).IsEqualTo(Error.NoError).Because("operand A import");

			Manifold b = Import(Tower91115)
				.Rotate(311.0, 55.0, 345.0)
				.Translate(new Vec3(0.7, -0.2, 0.4));
			await Assert.That(b.Status()).IsEqualTo(Error.NoError).Because("operand B import");

			Manifold result = a.BooleanWithEngine(b, OpType.Subtract, BooleanEngine.Robust);
			await Assert.That(result.Status()).IsEqualTo(Error.NoError)
				.Because("robust difference status");
			double volume = result.Volume();
			await Assert.That(double.IsFinite(volume)).IsTrue()
				.Because("difference volume must be finite");
			await Assert.That(volume).IsGreaterThan(0.0)
				.Because("difference volume must be positive");
		}

		/// <summary>
		/// Thingi10K #59082 union #1313535 (demo repro): both operands import cleanly, yet the
		/// robust union returns NotClosed with an empty result. Surfaced in the browser as
		/// <c>RuntimeError: unreachable</c> — that part was the wasm-only <c>Instant::now()</c>
		/// panic (fixed separately); underneath it the boolean itself still fails, which is
		/// what this test pins.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task Thingi59082Union1313535IsClosed()
		{
			Manifold a = Import(CastleStairs59082);
			await Assert.That(a.Status()).IsEqualTo(Error.NoError).Because("operand A import");

			Manifold b = Import(Ground1313535)
				.Rotate(162.0, 156.0, 337.0)
				.Translate(new Vec3(0.3, 0.0, 0.0));
			await Assert.That(b.Status()).IsEqualTo(Error.NoError).Because("operand B import");

			Manifold result = a.UnionWithEngine(b, BooleanEngine.Robust);
			await Assert.That(result.Status()).IsEqualTo(Error.NoError).Because("robust union status");
			await Assert.That(result.IsEmpty()).IsFalse().Because("robust union should be non-empty");
			await Assert.That(result.Volume()).IsGreaterThan(0.0)
				.Because("union volume must be positive");
		}

		/// <summary>
		/// Thingi10K #939888 union #93557 (demo repro): both operands import as closed
		/// manifolds, so the robust union must produce a valid result — the engine returned
		/// NotClosed.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task Thingi939888Union93557IsClosed()
		{
			Manifold a = Import(Tentacle939888);
			await Assert.That(a.Status()).IsEqualTo(Error.NoError).Because("operand A import");
			await Assert.That(a.AsImpl().IsSoup).IsFalse()
				.Because("operand A should weld to a manifold");

			Manifold b = Import(Pickaxe93557)
				.Rotate(356.0, 140.0, 322.0)
				.Translate(new Vec3(0.3, 0.0, 0.0));
			await Assert.That(b.Status()).IsEqualTo(Error.NoError).Because("operand B import");
			await Assert.That(b.AsImpl().IsSoup).IsFalse()
				.Because("operand B should weld to a manifold");

			Manifold result = a.UnionWithEngine(b, BooleanEngine.Robust);
			await Assert.That(result.Status()).IsEqualTo(Error.NoError).Because("robust union status");
			await Assert.That(result.IsEmpty()).IsFalse().Because("robust union should be non-empty");

			// The union of two closed solids contains at least each operand's
			// volume-maximum and must exceed either operand alone minus overlap;
			// just sanity-check positivity here — status is the regression.
			await Assert.That(result.Volume()).IsGreaterThan(0.0)
				.Because("union volume must be positive");
		}

		/// <summary>
		/// Thingi10K #68730 ("recognizer cuerpo repaired"): edge/vertex manifold, but several
		/// of its bodies are wound inside-out, so under {w &gt;= 1} semantics most of its
		/// geometry is not material and vanishes from booleans (operand overlap ratio ~13x in
		/// the demo's repro frames against #486860). <c>RepairOrientation</c> must rewind those
		/// bodies so the mesh's enclosed volume matches what the independent winding sampler
		/// measures.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task Thingi68730RepairOrientationRestoresMaterial()
		{
			Manifold broken = Import(Recognizer68730);
			await Assert.That(broken.Status()).IsEqualTo(Error.NoError).Because("import");

			const int Samples = 20_000;
			(double Material, double Sigma) brokenSample = SampledMaterial(broken, Samples);
			double matBroken = brokenSample.Material;
			double divBroken = SignedVolume(broken);
			await Assert.That(Math.Abs(divBroken) > 5.0 * matBroken).IsTrue()
				.Because($"fixture regressed: divergence volume {divBroken} should dwarf sampled material {matBroken}");

			Manifold repaired = broken.RepairOrientation();
			await Assert.That(repaired.Status()).IsEqualTo(Error.NoError).Because("repair status");
			await Assert.That(repaired.NumTri()).IsEqualTo(broken.NumTri());
			await Assert.That(repaired.AsImpl().IsSoup).IsFalse()
				.Because("68730 imports as a manifold");
			await Assert.That(repaired.AsImpl().IsManifold()).IsTrue()
				.Because("pairing must survive repair");

			// Repaired, the divergence volume and the sampled {w >= 1} material must
			// agree: every body now bounds solid material exactly once.
			(double Material, double Sigma) fixedSample = SampledMaterial(repaired, Samples);
			double matFixed = fixedSample.Material;
			double sigma = fixedSample.Sigma;
			double divFixed = SignedVolume(repaired);

			// Rewinding turns each inverted body's negative contribution positive,
			// so the repaired total must exceed the broken total's magnitude.
			await Assert.That(divFixed >= Math.Abs(divBroken)).IsTrue()
				.Because($"repair must add enclosed volume ({divFixed} vs {divBroken})");
			await Assert.That(Math.Abs(divFixed - matFixed) < 5.0 * sigma).IsTrue()
				.Because($"repaired divergence volume {divFixed} vs sampled material {matFixed} (sigma {sigma})");

			// And the repaired bodies must survive a boolean: union with a probe box
			// through the model keeps at least the mesh's own material.
			Manifold probe = Manifold.Cube(new Vec3(0.5, 0.5, 0.5), true);
			Manifold union = repaired.UnionWithEngine(probe, BooleanEngine.Robust);
			await Assert.That(union.Status()).IsEqualTo(Error.NoError).Because("robust union status");
			await Assert.That(union.Volume() > divFixed - 1e-9).IsTrue()
				.Because($"union volume {union.Volume()} must contain the repaired material {divFixed}");
		}

		/// <summary>
		/// Thingi10K #61459 ("cuboctahedron less faces"): a valid, correctly wound solid that
		/// additionally carries a smaller **nested outward-wound shell** (20 of its 176
		/// triangles) strictly inside the 156-triangle body. Under {w &gt;= 1} that inner shell
		/// is harmless — its interior simply winds 2 — so the mesh's material is the outer
		/// body's 3.2 units³.
		/// </summary>
		/// <remarks>
		/// The depth-parity rule alone read the inner shell as a first-level cavity and
		/// rewound it, carving 2.58 units³ of real material out of the model (~80% loss,
		/// visible in the demo's Boolean Gallery). Repair must be a material no-op here:
		/// nothing is inside-out, so there is nothing to fix.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task Thingi61459RepairPreservesNestedSolid()
		{
			Manifold m = Import(Cuboctahedron61459);
			await Assert.That(m.Status()).IsEqualTo(Error.NoError).Because("import");
			await Assert.That(m.AsImpl().IsSoup).IsFalse()
				.Because("61459 welds to manifold connectivity");

			const int Samples = 40_000;
			double divBefore = SignedVolume(m);
			double matBefore = SampledMaterial(m, Samples).Material;
			await Assert.That(matBefore).IsGreaterThan(3.0)
				.Because($"fixture regressed: {matBefore} of material expected before repair");

			// Plan level: nothing here is inside-out, so no shell may be rewound.
			RepairPlan plan = Repair.PlanRepair(Robust.Soup.ImplToTris(m.AsImpl()));
			await Assert.That(plan.NumShells).IsEqualTo(2)
				.Because("156-tri body plus a 20-tri nested shell");
			await Assert.That(plan.FlippedShells).IsEqualTo(0)
				.Because("repair must be a no-op on 61459");

			Manifold repaired = m.RepairOrientation();
			await Assert.That(repaired.Status()).IsEqualTo(Error.NoError).Because("repair status");
			await Assert.That(repaired.NumTri()).IsEqualTo(m.NumTri());

			double divAfter = SignedVolume(repaired);
			double matAfter = SampledMaterial(repaired, Samples).Material;
			await Assert.That(Math.Abs(divAfter - divBefore) <= 0.01 * Math.Abs(divBefore)).IsTrue()
				.Because($"repair changed divergence volume {divBefore} -> {divAfter}");

			// The two estimates are paired — same seed, same bounding box (repair
			// never moves a vertex) — so they sample identical points and no
			// Monte-Carlo slack is needed on top of the 1% tolerance.
			await Assert.That(Math.Abs(matAfter - matBefore) <= 0.01 * matBefore).IsTrue()
				.Because($"repair destroyed material {matBefore} -> {matAfter}");
		}

		/// <summary>
		/// Thingi10K #301921 ∪ its rotated copy (the sweep's standard pass): both operands are
		/// clean manifolds (self-overlap ratio 1.00), the arrangement is consistent, yet the
		/// robust union lost ~7% of its volume (0.4033 vs exact 0.4381; the Monte-Carlo referee
		/// sides with exact at 6.5σ). Surface area was almost unchanged — an interior region
		/// flipped outside.
		/// </summary>
		/// <remarks>
		/// The tolerance is 1e-3 relative, not exact agreement: both engines run a topology
		/// cleanup over their own tessellation of the same boundary, and those cleanups
		/// legitimately move the volume at the ~1e-4 scale (here the robust result is the raw
		/// extraction volume 0.438010 and the exact engine's cleanup moved its own result to
		/// 0.438066). Anything larger is a real defect — this gate catches both the original
		/// 8% class and the 0.6% <c>swap_degenerates</c> class (docs/CPP_DIVERGENCES.md entry
		/// 1) — and the Monte-Carlo referee arbitrates disputes.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task Thingi301921UnionRotatedSelfMatchesExact()
		{
			Manifold a = Import(Model301921);
			await Assert.That(a.Status()).IsEqualTo(Error.NoError).Because("operand A import");
			await Assert.That(a.AsImpl().IsSoup).IsFalse()
				.Because("operand A should weld to a manifold");
			Manifold b = a.Rotate(30.0, 45.0, 60.0).Translate(new Vec3(0.3, 0.0, 0.0));
			Manifold robust = a.UnionWithEngine(b, BooleanEngine.Robust);
			Manifold exact = a.UnionWithEngine(b, BooleanEngine.Exact);
			await Assert.That(robust.Status()).IsEqualTo(Error.NoError).Because("robust union status");
			double rv = robust.Volume();
			double ev = exact.Volume();
			await Assert.That(Math.Abs(rv - ev) <= 1e-3 * Math.Abs(ev)).IsTrue()
				.Because($"robust volume {rv} != exact volume {ev}");
		}

		/// <summary>
		/// Thingi10K #36374 ∪ its rotated copy: returned NotClosed before the canonical
		/// cocircular tie-break — the downstream failure the understated walls of #36088 only
		/// threatened. BFS crossed a split coincident stack, seeded wrong windings, and
		/// extraction could not close.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task Thingi36374UnionRotatedSelfIsClosed()
		{
			Manifold a = Import(Model36374);
			await Assert.That(a.Status()).IsEqualTo(Error.NoError).Because("operand A import");
			Manifold b = a.Rotate(30.0, 45.0, 60.0).Translate(new Vec3(0.3, 0.0, 0.0));
			Manifold result = a.UnionWithEngine(b, BooleanEngine.Robust);
			await Assert.That(result.Status()).IsEqualTo(Error.NoError).Because("robust union status");
			await Assert.That(result.Volume()).IsGreaterThan(0.0)
				.Because("union volume must be positive");
			await AssertArrangementConsistent(a, b, "36374 ∪ rotated self");
		}

		/// <summary>
		/// Thingi10K #36088 ∪ its rotated copy (the sweep's standard pass): found by the
		/// full-corpus sweep as the one mesh in its window whose arrangement carried
		/// inconsistent walls — winding steps that contradict the resolved cell windings. The
		/// union's volume was still right, but an inconsistent complex means cells were merged
		/// that the geometry keeps apart.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task Thingi36088ArrangementIsConsistent()
		{
			Manifold a = Import(Model36088);
			await Assert.That(a.Status()).IsEqualTo(Error.NoError).Because("operand A import");
			Manifold b = a.Rotate(30.0, 45.0, 60.0).Translate(new Vec3(0.3, 0.0, 0.0));
			await AssertArrangementConsistent(a, b, "36088 ∪ rotated self");
		}

		/// <summary>
		/// Thingi10K #51360 ("top walls") ∪ #90225, the user's logged demo frame.
		/// </summary>
		/// <remarks>
		/// #51360 fuses a correctly wound region with an inside-out region of nearly equal
		/// volume into one connected shell — so its divergence volume is ~0 (the two halves
		/// cancel) and <c>RepairOrientation</c> correctly declines to rewind a shell that is
		/// inverted only in part. Under the default <c>{w &gt;= 1}</c> rule the inverted half
		/// is simply not material and the union drops it; <see cref="WindingRule.Nonzero"/>
		/// keeps it. The bands below are the frame's measured volumes, confirmed against the
		/// Monte-Carlo winding referee under each rule: 2.4981 for Positive and 2.8340 ±
		/// 0.0064 for Nonzero.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task Thingi51360NonzeroRuleKeepsTheInvertedHalf()
		{
			Manifold a = Import(TopWalls51360);
			await Assert.That(a.Status()).IsEqualTo(Error.NoError).Because("operand A import");
			Manifold b = Import(Model90225)
				.Rotate(340.0, 202.0, 341.0)
				.Translate(new Vec3(0.7, -0.2, 0.4));
			await Assert.That(b.Status()).IsEqualTo(Error.NoError).Because("operand B import");

			Manifold positive = a.BooleanWithEngineAndRule(
				b, OpType.Add, BooleanEngine.Robust, WindingRule.Positive);
			await Assert.That(positive.Status()).IsEqualTo(Error.NoError).Because("positive-rule status");
			Manifold nonzero = a.BooleanWithEngineAndRule(
				b, OpType.Add, BooleanEngine.Robust, WindingRule.Nonzero);
			await Assert.That(nonzero.Status()).IsEqualTo(Error.NoError).Because("nonzero-rule status");

			// The default rule must be untouched by the feature.
			await Assert.That(positive.Volume())
				.IsEqualTo(a.UnionWithEngine(b, BooleanEngine.Robust).Volume())
				.Because("default engine path must equal the explicit Positive rule");
			await Assert.That(Math.Abs(positive.Volume() - 2.4981)).IsLessThan(1e-3)
				.Because($"positive-rule volume {positive.Volume()}");
			await Assert.That(Math.Abs(nonzero.Volume() - 2.8340)).IsLessThan(5e-3)
				.Because($"nonzero-rule volume {nonzero.Volume()}");
			await Assert.That(nonzero.NumTri()).IsGreaterThan(positive.NumTri())
				.Because($"nonzero keeps more surface: {nonzero.NumTri()} vs {positive.NumTri()}");
		}

		/// <summary>Imports one checked-in fixture through the demo's pipeline.</summary>
		/// <param name="fileName">The fixture's file name.</param>
		/// <returns>The imported manifold.</returns>
		private static Manifold Import(string fileName)
		{
			return StlFixtures.ImportStlLikeDemo(StlFixtures.FixtureBytes(fileName));
		}

		/// <summary>
		/// Every wall's winding step must equal the difference between the cells it separates.
		/// A violation means the complex merged cells the geometry keeps apart — the condition
		/// that previously surfaced only as a hash-order dependent volume. Checked here on
		/// real, heavily self-intersecting scans rather than synthetic solids.
		/// </summary>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <param name="what">The pair's name, for failure messages.</param>
		/// <returns>The assertion task.</returns>
		private static async Task AssertArrangementConsistent(Manifold a, Manifold b, string what)
		{
			List<Vec3[]> p = Robust.Soup.ImplToTris(a.AsImpl());
			List<Vec3[]> q = Robust.Soup.ImplToTris(b.AsImpl());
			IntersectionGraph graph = IntersectionGraphFunctions.BuildGraph(p.ToArray(), q.ToArray());
			CellComplex complex = Cells.BuildCells(graph);
			Windings wind = Cells.Windings(graph, complex, new IReadOnlyList<Vec3[]>[] { p, q });
			List<(int Rep, (int P, int Q) Delta, (int P, int Q) Actual)> bad
				= Cells.InconsistentWalls(complex, wind);
			await Assert.That(bad.Count).IsEqualTo(0)
				.Because($"{what}: {bad.Count} inconsistent walls of {complex.NumCells} cells; "
					+ $"first (piece, step, actual) = {(bad.Count == 0 ? "None" : bad[0].ToString())}");
		}

		/// <summary>
		/// Sampled volume of <c>{winding &gt;= 1}</c> over the mesh's bounding box —
		/// stratified Monte Carlo with the exact winding query and a fixed SplitMix64 seed,
		/// the same independent arbiter the Rust's <c>examples/robust_repro.rs</c> uses.
		/// </summary>
		/// <param name="m">The mesh.</param>
		/// <param name="samples">How many points to draw.</param>
		/// <returns>The estimate and one standard deviation.</returns>
		private static (double Material, double Sigma) SampledMaterial(Manifold m, int samples)
		{
			List<Vec3[]> tris = Robust.Soup.ImplToTris(m.AsImpl());
			double[] min = { double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity };
			double[] max = { double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity };
			foreach (Vec3[] t in tris)
			{
				foreach (Vec3 v in t)
				{
					double[] c = { v.X, v.Y, v.Z };
					for (int k = 0; k < 3; k++)
					{
						min[k] = LinalgFunctions.MinF64(min[k], c[k]);
						max[k] = LinalgFunctions.MaxF64(max[k], c[k]);
					}
				}
			}

			double[] ext = { max[0] - min[0], max[1] - min[1], max[2] - min[2] };
			WindingIndex idx = new WindingIndex(tris);
			ulong state = 0x5EED_CAFE_F00D_D1CEUL;
			double Rand()
			{
				// SplitMix64. Rust's `wrapping_add`/`wrapping_mul`; C# `ulong` arithmetic
				// is unchecked by default and wraps identically.
				state += 0x9E3779B97F4A7C15UL;
				ulong z = state;
				z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
				z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
				return ((z ^ (z >> 31)) >> 11) / (double)(1UL << 53);
			}

			int hits = 0;
			for (int i = 0; i < samples; i++)
			{
				R3 pt = R3.FromVec3(new Vec3(
					min[0] + (ext[0] * Rand()),
					min[1] + (ext[1] * Rand()),
					min[2] + (ext[2] * Rand())));
				if (RayShoot.WindingNumberIndexed(pt, tris, idx) >= 1)
				{
					hits++;
				}
			}

			double volBox = ext[0] * ext[1] * ext[2];
			double frac = hits / (double)samples;
			return (volBox * frac, volBox * Math.Sqrt(frac * (1.0 - frac) / samples));
		}

		/// <summary>Signed divergence-theorem volume (<c>Volume()</c> hides the sign).</summary>
		/// <param name="m">The manifold.</param>
		/// <returns>Its signed volume.</returns>
		private static double SignedVolume(Manifold m)
		{
			return RobustRepairTests.SignedVolume(m);
		}
	}
}
