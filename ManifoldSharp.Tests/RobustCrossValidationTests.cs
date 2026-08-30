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

// RobustCrossValidationTests.cs — port of robust/cross_validation_tests.rs, whose
// header reads:
//
//   Phase 7 validation battery: on MANIFOLD inputs the robust engine must
//   near-exactly reproduce the exact engine (the user-specified acceptance
//   criterion for this feature):
//     - |Δvolume| and |Δarea| ≤ 1e-9 (relative),
//     - identical genus and manifoldness,
//     - vertex and triangle counts equal (triangulation itself may differ),
//     - bidirectional vertex-position match within 1e-9·bbox.scale().
//   Pairs are drawn from primitive shapes under seeded pseudo-random rigid
//   transforms (generic position — tangential configurations are covered by
//   dedicated tests elsewhere).
//
// Same inputs, same expected values, same order as the Rust's three tests
// (the third #[ignore]d, and skipped here for the same reason).
//
// ── `catch_unwind` becomes a try/catch ───────────────────────────────────────
// The Rust guards the *reference* call with `std::panic::catch_unwind`, because
// the exact engine can assert on rare rotated configurations (the `pair_up`
// "non-manifold edge" panic, a known C++-inherited failure class). This port's
// equivalent of that panic is an exception out of the same code, so the guard is
// a try/catch around the same call and the same fallback follows it. It is
// deliberately broad: the point is "the exact engine could not produce a
// reference", not any particular exception type.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class RobustCrossValidationTests
	{
		[Test]
		public async Task BatteryCubePairs()
		{
			Lcg rng = new Lcg(0xC0FFEE);
			Manifold cube = Manifold.Cube(V(2.0, 2.0, 2.0), true);
			for (int i = 0; i < 4; i++)
			{
				Manifold b = Jiggle(cube, rng);
				await CrossValidate(cube, b, $"cube/jiggled-cube #{i}");
			}
		}

		[Test]
		public async Task BatteryMixedPairs()
		{
			Lcg rng = new Lcg(0xBADC0DE);
			List<(string Name, Manifold Shape)> shapes = Shapes();

			// A deterministic selection of cross-shape pairs.
			(int A, int B)[] picks = { (0, 3), (1, 0), (2, 4), (3, 5), (4, 1), (5, 2) };
			for (int i = 0; i < picks.Length; i++)
			{
				(string Name, Manifold Shape) a = shapes[picks[i].A];
				(string Name, Manifold Shape) b = shapes[picks[i].B];
				Manifold jiggled = Jiggle(b.Shape, rng);
				await CrossValidate(a.Shape, jiggled, $"{a.Name}/{b.Name} #{i}");
			}
		}

		/// <summary>
		/// Extended battery — more pairs, bigger meshes. Slow with debug-build exact rational
		/// arithmetic; run explicitly in Release with the <c>--treenode-filter</c> that names it.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[Skip("extended battery: run in release, see comment")]
		public async Task BatteryExtended()
		{
			Lcg rng = new Lcg(0x5EED);
			List<(string Name, Manifold Shape)> shapes = Shapes();
			shapes.Add(("sphere20", Manifold.Sphere(1.1, 20)));
			shapes.Add(("cyl16", Manifold.CylinderCentered(2.4, 1.0, 1.0, 16, true)));
			int n = shapes.Count;
			for (int i = 0; i < 30; i++)
			{
				int sa = (int)(rng.Next() % (ulong)n);
				int sb = (int)(rng.Next() % (ulong)n);
				(string Name, Manifold Shape) a = shapes[sa];
				(string Name, Manifold Shape) b = shapes[sb];
				Manifold ja = Jiggle(a.Shape, rng);
				Manifold jb = Jiggle(b.Shape, rng);
				await CrossValidate(ja, jb, $"{a.Name}/{b.Name} ext#{i}");
			}
		}

		/// <summary>Shorthand for the Rust's <c>v(x, y, z)</c>.</summary>
		/// <param name="x">The x coordinate.</param>
		/// <param name="y">The y coordinate.</param>
		/// <param name="z">The z coordinate.</param>
		/// <returns>The vector.</returns>
		private static Vec3 V(double x, double y, double z)
		{
			return new Vec3(x, y, z);
		}

		/// <summary>The battery's primitive shapes.</summary>
		/// <returns>Six named primitives.</returns>
		private static List<(string Name, Manifold Shape)> Shapes()
		{
			return new List<(string, Manifold)>
			{
				("cube", Manifold.Cube(V(2.0, 2.0, 2.0), true)),
				("sphere8", Manifold.Sphere(1.2, 8)),
				("sphere12", Manifold.Sphere(1.0, 12)),
				("cylinder", Manifold.CylinderCentered(2.0, 0.9, 0.9, 10, true)),
				("cone", Manifold.CylinderCentered(2.0, 1.1, 0.3, 8, true)),
				("tetra", Manifold.Tetrahedron().Scale(V(1.5, 1.5, 1.5))),
			};
		}

		/// <summary>Random rigid transform in generic position.</summary>
		/// <param name="m">The shape to move.</param>
		/// <param name="rng">The generator.</param>
		/// <returns>The moved shape.</returns>
		private static Manifold Jiggle(Manifold m, Lcg rng)
		{
			return m.Rotate(
					rng.Uniform(5.0, 85.0),
					rng.Uniform(5.0, 85.0),
					rng.Uniform(5.0, 85.0))
				.Translate(V(
					rng.Uniform(-0.7, 0.7),
					rng.Uniform(-0.7, 0.7),
					rng.Uniform(-0.7, 0.7)));
		}

		/// <summary>
		/// Bidirectional vertex match: every vertex of <paramref name="a"/> has a vertex of
		/// <paramref name="b"/> within <paramref name="tol"/> and vice versa.
		/// </summary>
		/// <param name="a">The first mesh.</param>
		/// <param name="b">The second mesh.</param>
		/// <param name="tol">The per-axis tolerance.</param>
		/// <returns>The number of unmatched vertices.</returns>
		private static int UnmatchedVerts(Manifold a, Manifold b, double tol)
		{
			List<Vec3> av = a.AsImpl().VertPos;
			List<Vec3> bv = b.AsImpl().VertPos;
			bool Close(Vec3 p, Vec3 q)
			{
				return Math.Abs(p.X - q.X) <= tol
					&& Math.Abs(p.Y - q.Y) <= tol
					&& Math.Abs(p.Z - q.Z) <= tol;
			}

			int misses = 0;
			foreach (Vec3 p in av)
			{
				if (!bv.Any(q => Close(p, q)))
				{
					misses++;
				}
			}

			foreach (Vec3 q in bv)
			{
				if (!av.Any(p => Close(p, q)))
				{
					misses++;
				}
			}

			return misses;
		}

		/// <summary>Compares both engines on one operand pair, across all three operations.</summary>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <param name="what">The pair's name, for failure messages.</param>
		/// <returns>The assertion task.</returns>
		private static async Task CrossValidate(Manifold a, Manifold b, string what)
		{
			foreach (OpType op in new[] { OpType.Add, OpType.Subtract, OpType.Intersect })
			{
				string ctx = $"{what} {op}";

				// The exact engine can assert on rare rotated configurations (the
				// pair_up "non-manifold edge" panic, a known C++-inherited failure
				// class, inherited with the algorithm). When it cannot produce a reference,
				// sanity-check the robust result alone instead of comparing; the
				// robust engine existing is the fix for those inputs, not the bug.
				Manifold exact;
				try
				{
					exact = a.BooleanWithEngine(b, op, BooleanEngine.Exact);
				}
				catch (Exception ex)
				{
					// The Rust's `eprintln!` — a downgrade this quiet must say so, or a
					// battery that silently stopped comparing looks like a battery that
					// passed. The exception text is C#-only and additive: the Rust's
					// catch_unwind has no equivalent detail to print.
					TestContext.Current?.OutputWriter.WriteLine(
						$"{ctx}: exact engine panicked; robust-only sanity check ({ex.GetType().Name}: {ex.Message})");
					Manifold robustOnly = a.BooleanWithEngine(b, op, BooleanEngine.Robust);
					await Assert.That(robustOnly.Status()).IsEqualTo(Error.NoError)
						.Because($"{ctx}: robust status");
					await Assert.That(robustOnly.Volume()).IsGreaterThanOrEqualTo(0.0)
						.Because($"{ctx}: robust volume sane");
					continue;
				}

				Manifold robust = a.BooleanWithEngine(b, op, BooleanEngine.Robust);

				await Assert.That(exact.Status()).IsEqualTo(Error.NoError).Because($"{ctx}: exact status");
				await Assert.That(robust.Status()).IsEqualTo(Error.NoError).Because($"{ctx}: robust status");
				await Assert.That(robust.AsImpl().IsSoup).IsFalse()
					.Because($"{ctx}: robust output not manifold");

				double ve = exact.Volume();
				double vr = robust.Volume();
				await Assert.That(Math.Abs(vr - ve) <= 1e-9 * Math.Max(Math.Abs(ve), 1.0)).IsTrue()
					.Because($"{ctx}: volume {vr} vs {ve}");
				double ae = exact.SurfaceArea();
				double ar = robust.SurfaceArea();
				await Assert.That(Math.Abs(ar - ae) <= 1e-9 * Math.Max(Math.Abs(ae), 1.0)).IsTrue()
					.Because($"{ctx}: area {ar} vs {ae}");
				await Assert.That(robust.Genus()).IsEqualTo(exact.Genus()).Because($"{ctx}: genus");

				await Assert.That(robust.NumVert()).IsEqualTo(exact.NumVert())
					.Because($"{ctx}: vertex count {robust.NumVert()} vs {exact.NumVert()}");
				await Assert.That(robust.NumTri()).IsEqualTo(exact.NumTri())
					.Because($"{ctx}: triangle count {robust.NumTri()} vs {exact.NumTri()}");

				double tol = 1e-9 * Math.Max(exact.BoundingBox().Scale(), 1.0);
				int misses = UnmatchedVerts(robust, exact, tol);
				await Assert.That(misses).IsEqualTo(0).Because($"{ctx}: {misses} unmatched vertices");
			}
		}

		/// <summary>The Rust's seeded LCG.</summary>
		private sealed class Lcg
		{
			private ulong state;

			/// <summary>Seeds the generator.</summary>
			/// <param name="seed">The seed.</param>
			public Lcg(ulong seed)
			{
				this.state = seed;
			}

			/// <summary>
			/// The next raw value. Rust's `wrapping_mul`/`wrapping_add`; C#'s default
			/// unchecked arithmetic on <c>ulong</c> wraps identically.
			/// </summary>
			/// <returns>The next state.</returns>
			public ulong Next()
			{
				this.state = (this.state * 6364136223846793005UL) + 1442695040888963407UL;
				return this.state;
			}

			/// <summary>A uniform draw from [lo, hi).</summary>
			/// <param name="lo">The low bound.</param>
			/// <param name="hi">The high bound.</param>
			/// <returns>The draw.</returns>
			public double Uniform(double lo, double hi)
			{
				double u = (this.Next() >> 11) / (double)(1UL << 53);
				return lo + (u * (hi - lo));
			}
		}
	}
}
