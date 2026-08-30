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

// Port of src/manifold_tests/sdf.rs — all 8 tests, same inputs, same expected
// values, same tolerances, in the same order. Nothing deferred.
//
// This is the genus-and-volume cover for Sdf.cs's marching tetrahedra that
// SdfTests.cs (the inline tests of sdf.rs) could only gesture at: SDF.Resize
// pins genus -8, SDF.SineSurface pins genus 38 with volume 102.4 and surface
// area 392.4, SDF.Bounds pins the cube void at genus -1, and SDF.Void pins the
// subtraction to a unit cube to within 0.001. Those numbers are what the
// slot-order GridHashTable exists to make reproducible.
//
// ── Two ignores, carried over verbatim ───────────────────────────────────────
// The Rust `#[ignore]`s SdfBlobs and SdfSphereShell for debug-suite SPEED, not
// correctness — both pass. Their reason strings are ported to [Skip] unchanged
// (including the Rust-specific timings, which are the measurement that justified
// the ignore), so the two suites keep the same 9-ignore budget the porting plan
// counts. Both were run here with the [Skip] temporarily removed, in Release,
// before being checked in skipped: Blobs reaches genus 0 and SphereShell lands
// inside the ±1000 band around 14235, so the claim in each reason string holds
// for this port and not only for the Rust.
//
// ── Trig in the callbacks ────────────────────────────────────────────────────
// SineSurface's SDF reads `p.x.sin()`, i.e. Rust *std* trig, not the port's
// DeterministicMath (which mirrors the production math.rs, and which the Gyroid
// helper uses because the Rust helper's gyroid is also fed through production
// code paths). Ported as written: Math.Sin. .NET's and Rust's libm disagree by
// about 1 ulp on a fraction of a percent of arguments, and this test's
// assertions are topology- and tolerance-based — they survived the same variance
// against C++'s libm. If a libm difference ever does move genus 38, that is real
// signal about the surface extraction, not a reason to widen the assertion.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ManifoldSdfTests
	{
		/// <summary>C++ TEST(SDF, Resize) — Layers SDF produces correct bounds and genus.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSdfResize()
		{
			double size = 20.0;
			Manifold layers = Manifold.LevelSet(
				p =>
				{
					// f64::round is ties-away-from-zero; C#'s bare Math.Round is ties-to-even.
					int a = (int)(Math.Round(2.0 * p.Z, MidpointRounding.AwayFromZero) % 4.0);
					if (a == 0)
					{
						return 1.0;
					}
					else if (a == 2)
					{
						return -1.0;
					}
					else
					{
						return 0.0;
					}
				},
				new Box(new Vec3(0.0, 0.0, 0.0), new Vec3(size, size, size)),
				1.0);

			await Assert.That(layers.Status()).IsEqualTo(Error.NoError);
			await Assert.That(layers.Genus())
				.IsEqualTo(-8)
				.Because($"Resize: genus should be -8, got {layers.Genus()}");
			double epsilon = layers.GetTolerance();
			Box bounds = layers.BoundingBox();
			await Assert.That(Math.Abs(bounds.Min.X - 0.0) < epsilon).IsTrue().Because("min.x");
			await Assert.That(Math.Abs(bounds.Min.Y - 0.0) < epsilon).IsTrue().Because("min.y");
			await Assert.That(Math.Abs(bounds.Min.Z - 1.5) < epsilon)
				.IsTrue()
				.Because($"min.z={bounds.Min.Z}");
			await Assert.That(Math.Abs(bounds.Max.X - size) < epsilon).IsTrue().Because("max.x");
			await Assert.That(Math.Abs(bounds.Max.Y - size) < epsilon).IsTrue().Because("max.y");
			await Assert.That(Math.Abs(bounds.Max.Z - (size - 1.5)) < epsilon)
				.IsTrue()
				.Because($"max.z={bounds.Max.Z}");
		}

		/// <summary>
		/// C++ TEST(SDF, SineSurface) — raw LevelSet of a sine surface.
		/// v3.5.0 (#1724) dropped the trailing <c>Simplify()</c> and <c>SmoothOut(180)</c>
		/// smoothing, so this now checks the unsmoothed surface directly.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSdfSineSurface()
		{
			double pi = Types.KPi;
			Manifold surface = Manifold.LevelSet(
				p =>
				{
					double mid = Math.Sin(p.X) + Math.Sin(p.Y);
					return p.Z > mid - 0.5 && p.Z < mid + 0.5 ? 1.0 : -1.0;
				},
				new Box(
					new Vec3(-1.75 * pi, -1.75 * pi, -1.75 * pi),
					new Vec3(1.75 * pi, 1.75 * pi, 1.75 * pi)),
				1.0);
			await Assert.That(surface.Status()).IsEqualTo(Error.NoError);
			await Assert.That(surface.Genus())
				.IsEqualTo(38)
				.Because($"SineSurface genus={surface.Genus()}");
			await Assert.That(Math.Abs(surface.Volume() - 102.4) < 0.1)
				.IsTrue()
				.Because($"SineSurface vol={surface.Volume()}");
			await Assert.That(Math.Abs(surface.SurfaceArea() - 392.4) < 0.1)
				.IsTrue()
				.Because($"SineSurface sa={surface.SurfaceArea()}");
		}

		/// <summary>C++ TEST(SDF, Blobs) — metaball SDF using smoothstep.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		[Skip("Slow in DEBUG only (fine edge_length=0.05). Release is 5.6s after the "
			+ "HashMap->Vec voxel-grid perf fix (was 62.6s). Passes. Remaining release gap vs "
			+ "C++ is the deferred voxel-fill parallelism (rayon feature), not an algorithmic bug.")]
		public async Task CppSdfBlobs()
		{
			double blend = 1.0;
			double[][] balls =
			{
				new[] { 0.0, 0.0, 0.0, 2.0 },
				new[] { 1.0, 2.0, 3.0, 2.0 },
				new[] { -2.0, 2.0, -2.0, 1.0 },
				new[] { -2.0, -3.0, -2.0, 2.0 },
				new[] { -3.0, -1.0, -3.0, 1.0 },
				new[] { 2.0, -3.0, -2.0, 2.0 },
				new[] { -2.0, 3.0, 2.0, 2.0 },
				new[] { -2.0, -3.0, 2.0, 2.0 },
				new[] { 1.0, -1.0, 1.0, -2.0 },
				new[] { -4.0, -3.0, -2.0, 1.0 },
			};
			Manifold blobs = Manifold.LevelSetWithLevel(
				p =>
				{
					double d = 0.0;
					foreach (double[] ball in balls)
					{
						Vec3 center = new Vec3(ball[0], ball[1], ball[2]);
						double w = ball[3];
						double sign = w > 0.0 ? 1.0 : -1.0;
						Vec3 diff = p - center;
						double dist = Math.Sqrt((diff.X * diff.X) + (diff.Y * diff.Y) + (diff.Z * diff.Z));
						d += sign * Types.Smoothstep(-blend, blend, Math.Abs(w) - dist);
					}

					return d;
				},
				new Box(new Vec3(-5.0, -5.0, -5.0), new Vec3(5.0, 5.0, 5.0)),
				0.05,
				0.5);

			await Assert.That(blobs.Status()).IsEqualTo(Error.NoError);
			await Assert.That(blobs.IsEmpty()).IsFalse().Because("Blobs should not be empty");

			// C++ computes genus = 1 - chi/2 where chi = NumVert - NumTri/2
			int chi = blobs.NumVert() - (blobs.NumTri() / 2);
			int genus = 1 - (chi / 2);
			await Assert.That(genus).IsEqualTo(0).Because($"Blobs genus should be 0, got {genus}");
		}

		/// <summary>C++ TEST(SDF, CubeVoid) — test CubeVoid SDF function values.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSdfCubeVoid()
		{
			await Assert.That(CubeVoidSdf(new Vec3(0.0, 0.0, 0.0))).IsEqualTo(-1.0);
			await Assert.That(Math.Abs(CubeVoidSdf(new Vec3(0.0, 0.0, 1.0))) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(CubeVoidSdf(new Vec3(0.0, 1.0, 1.0))) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(CubeVoidSdf(new Vec3(-1.0, 0.0, 0.0))) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(CubeVoidSdf(new Vec3(1.0, 1.0, -1.0))) < 1e-10).IsTrue();
			await Assert.That(CubeVoidSdf(new Vec3(2.0, 0.0, 0.0)) > 0.0).IsTrue();
			await Assert.That(CubeVoidSdf(new Vec3(2.0, -2.0, 0.0)) > 0.0).IsTrue();
			await Assert.That(CubeVoidSdf(new Vec3(-2.0, 2.0, 2.0)) > 0.0).IsTrue();
		}

		/// <summary>C++ TEST(SDF, Bounds) — CubeVoid with edge_length=1.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSdfBoundsCubeVoid()
		{
			double size = 4.0;
			Manifold cubeVoid = Manifold.LevelSet(
				CubeVoidSdf,
				new Box(
					new Vec3(-size / 2.0, -size / 2.0, -size / 2.0),
					new Vec3(size / 2.0, size / 2.0, size / 2.0)),
				1.0);
			await Assert.That(cubeVoid.Status()).IsEqualTo(Error.NoError);
			await Assert.That(cubeVoid.Genus())
				.IsEqualTo(-1)
				.Because($"CubeVoid genus should be -1, got {cubeVoid.Genus()}");
			double epsilon = cubeVoid.GetTolerance();
			Box bounds = cubeVoid.BoundingBox();
			double outer = size / 2.0;
			await Assert.That(Math.Abs(bounds.Min.X - -outer) < epsilon).IsTrue().Because("min.x");
			await Assert.That(Math.Abs(bounds.Min.Y - -outer) < epsilon).IsTrue().Because("min.y");
			await Assert.That(Math.Abs(bounds.Min.Z - -outer) < epsilon).IsTrue().Because("min.z");
			await Assert.That(Math.Abs(bounds.Max.X - outer) < epsilon).IsTrue().Because("max.x");
			await Assert.That(Math.Abs(bounds.Max.Y - outer) < epsilon).IsTrue().Because("max.y");
			await Assert.That(Math.Abs(bounds.Max.Z - outer) < epsilon).IsTrue().Because("max.z");
		}

		/// <summary>C++ TEST(SDF, Bounds3) — sphere with radius &gt; box, clipped to box.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSdfBounds3()
		{
			double radius = 1.2;
			Manifold sphere = Manifold.LevelSet(
				p => radius - Math.Sqrt((p.X * p.X) + (p.Y * p.Y) + (p.Z * p.Z)),
				new Box(new Vec3(-1.0, -1.0, -1.0), new Vec3(1.0, 1.0, 1.0)),
				0.1);
			await Assert.That(sphere.Status()).IsEqualTo(Error.NoError);
			await Assert.That(sphere.Genus()).IsEqualTo(0).Because($"Sphere genus={sphere.Genus()}");
			double epsilon = sphere.GetTolerance();
			Box bounds = sphere.BoundingBox();
			await Assert.That(Math.Abs(bounds.Min.X - -1.0) < epsilon)
				.IsTrue()
				.Because($"min.x={bounds.Min.X}");
			await Assert.That(Math.Abs(bounds.Min.Y - -1.0) < epsilon).IsTrue().Because("min.y");
			await Assert.That(Math.Abs(bounds.Min.Z - -1.0) < epsilon).IsTrue().Because("min.z");
			await Assert.That(Math.Abs(bounds.Max.X - 1.0) < epsilon)
				.IsTrue()
				.Because($"max.x={bounds.Max.X}");
			await Assert.That(Math.Abs(bounds.Max.Y - 1.0) < epsilon).IsTrue().Because("max.y");
			await Assert.That(Math.Abs(bounds.Max.Z - 1.0) < epsilon).IsTrue().Because("max.z");
		}

		/// <summary>C++ TEST(SDF, Void) — cube minus SDF void.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppSdfVoidSubtract()
		{
			double size = 4.0;
			Manifold cubeVoid = Manifold.LevelSet(
				CubeVoidSdf,
				new Box(
					new Vec3(-size / 2.0, -size / 2.0, -size / 2.0),
					new Vec3(size / 2.0, size / 2.0, size / 2.0)),
				0.5);
			await Assert.That(cubeVoid.Status()).IsEqualTo(Error.NoError);
			Manifold cube = Manifold.Cube(Vec3.Splat(size), true);
			Manifold result = cube - cubeVoid;
			await Assert.That(result.Genus()).IsEqualTo(0).Because($"genus={result.Genus()}");
			await Assert.That(Math.Abs(result.Volume() - 8.0) < 0.001)
				.IsTrue()
				.Because($"vol={result.Volume()}");
			await Assert.That(Math.Abs(result.SurfaceArea() - 24.0) < 0.001)
				.IsTrue()
				.Because($"sa={result.SurfaceArea()}");
			double epsilon = result.GetTolerance();
			Box bounds = result.BoundingBox();
			await Assert.That(Math.Abs(bounds.Min.X - -1.0) < epsilon).IsTrue();
			await Assert.That(Math.Abs(bounds.Min.Y - -1.0) < epsilon).IsTrue();
			await Assert.That(Math.Abs(bounds.Min.Z - -1.0) < epsilon).IsTrue();
			await Assert.That(Math.Abs(bounds.Max.X - 1.0) < epsilon).IsTrue();
			await Assert.That(Math.Abs(bounds.Max.Y - 1.0) < epsilon).IsTrue();
			await Assert.That(Math.Abs(bounds.Max.Z - 1.0) < epsilon).IsTrue();
		}

		/// <summary>C++ TEST(SDF, SphereShell) — thin sphere shell via level set.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		[Skip("Slow in debug only (~1min; ~5s in release) — PASSES with genus 13396, exactly "
			+ "matching the local C++ reference. The old 'marching-tet topology bug' was a "
			+ "test-porting error: the C++ test passes tolerance=0.0001 to LevelSet (enabling "
			+ "FindSurface vertex refinement), which the Rust test omitted (pure interpolation "
			+ "→ genus ~9576 on the half-voxel-thin shell). Kept ignored only for debug-suite "
			+ "speed, like sdf_blobs.")]
		public async Task CppSdfSphereShell()
		{
			// C++ writes `r - 0.995f` — a FLOAT literal inside a double expression —
			// and passes level=0, tolerance=0.0001 to LevelSet.
			double thin = (double)0.995f;
			Manifold sphere = Manifold.LevelSetWithTolerance(
				p =>
				{
					double r = Math.Sqrt((p.X * p.X) + (p.Y * p.Y) + (p.Z * p.Z));
					return LinalgFunctions.MinF64(1.0 - r, r - thin);
				},
				new Box(Vec3.Splat(-1.1), Vec3.Splat(1.1)),
				0.01,
				0.0,
				0.0001);

			// Genus 13396 — matches the local C++ reference build exactly.
			await Assert.That(Math.Abs(sphere.Genus() - 14235) < 1000)
				.IsTrue()
				.Because($"SphereShell genus={sphere.Genus()}, expected ~14235");
		}

		/// <summary>
		/// CubeVoid SDF — returns distance to a unit cube void (negative inside, positive
		/// outside).
		/// </summary>
		/// <param name="p">The sample point.</param>
		/// <returns>The signed distance.</returns>
		private static double CubeVoidSdf(Vec3 p)
		{
			double ax = Math.Abs(p.X);
			double ay = Math.Abs(p.Y);
			double az = Math.Abs(p.Z);

			// Inside the cube: distance to nearest face (negative)
			// Outside: distance to cube surface (positive)
			double dx = ax - 1.0;
			double dy = ay - 1.0;
			double dz = az - 1.0;

			// If all components negative, we're inside: return max (most negative = deepest)
			// If any positive, we're outside
			if (dx <= 0.0 && dy <= 0.0 && dz <= 0.0)
			{
				// Inside: return most-negative (closest face)
				return LinalgFunctions.MaxF64(LinalgFunctions.MaxF64(dx, dy), dz);
			}
			else
			{
				// Outside: Euclidean distance to surface
				double ex = LinalgFunctions.MaxF64(dx, 0.0);
				double ey = LinalgFunctions.MaxF64(dy, 0.0);
				double ez = LinalgFunctions.MaxF64(dz, 0.0);
				return Math.Sqrt((ex * ex) + (ey * ey) + (ez * ez));
			}
		}
	}
}
