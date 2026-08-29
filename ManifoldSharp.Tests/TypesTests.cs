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

// Port of types_tests.rs — all 17 cases, same expected values, same order and
// same strictness. The Rust tolerances (1e-15, 1e-12) are transcribed, not
// widened; where the Rust used assert_eq! this compares exactly.
//
// Nothing here is deferred: types_tests.rs depends only on types.rs and
// types_bounds.rs, both of which land with this step. types_meshgl.rs is not
// part of it — it belongs to the Phase 3 mesh-core cycle, and its tests live in
// its own Rust test module.

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Tests
{
	public class TypesTests
	{
		/// <summary>
		/// The TUnit constraint key shared by every test that reads or writes the
		/// process-global <see cref="Quality"/> settings. Public so later phases can name
		/// it instead of retyping the string.
		/// </summary>
		public const string QualityGlobalStateKey = "QualityGlobalState";

		[Test]
		public async Task RadiansDegrees()
		{
			await Assert.That(Math.Abs(Types.Radians(180.0) - Types.KPi) < 1e-15).IsTrue();
			await Assert.That(Math.Abs(Types.Degrees(Types.KPi) - 180.0) < 1e-12).IsTrue();
			await Assert.That(Math.Abs(Types.Radians(90.0) - Types.KHalfPi) < 1e-15).IsTrue();
		}

		[Test]
		public async Task Smoothstep()
		{
			await Assert.That(Types.Smoothstep(0.0, 1.0, 0.0)).IsEqualTo(0.0);
			await Assert.That(Types.Smoothstep(0.0, 1.0, 1.0)).IsEqualTo(1.0);
			double mid = Types.Smoothstep(0.0, 1.0, 0.5);
			await Assert.That(Math.Abs(mid - 0.5) < 1e-15).IsTrue();
		}

		[Test]
		public async Task SindCosd()
		{
			await Assert.That(Math.Abs(Types.Sind(0.0)) < 1e-15).IsTrue();
			await Assert.That(Math.Abs(Types.Sind(90.0) - 1.0) < 1e-15).IsTrue();
			await Assert.That(Math.Abs(Types.Sind(180.0)) < 1e-15).IsTrue();
			await Assert.That(Math.Abs(Types.Sind(270.0) + 1.0) < 1e-15).IsTrue();
			await Assert.That(Math.Abs(Types.Sind(360.0)) < 1e-15).IsTrue();
			await Assert.That(Math.Abs(Types.Cosd(0.0) - 1.0) < 1e-15).IsTrue();
			await Assert.That(Math.Abs(Types.Cosd(90.0)) < 1e-15).IsTrue();
			await Assert.That(Math.Abs(Types.Cosd(180.0) + 1.0) < 1e-15).IsTrue();
		}

		[Test]
		public async Task BoxDefault()
		{
			// `new Box()` is the port of Box::new()/Default: the inverted, empty box.
			Box b = new Box();
			await Assert.That(double.IsInfinity(b.Min.X) && b.Min.X > 0.0).IsTrue();
			await Assert.That(double.IsInfinity(b.Max.X) && b.Max.X < 0.0).IsTrue();
		}

		[Test]
		public async Task BoxFromPoints()
		{
			Box b = Box.FromPoints(new Vec3(1.0, 2.0, 3.0), new Vec3(-1.0, -2.0, -3.0));
			await Assert.That(b.Min).IsEqualTo(new Vec3(-1.0, -2.0, -3.0));
			await Assert.That(b.Max).IsEqualTo(new Vec3(1.0, 2.0, 3.0));
		}

		[Test]
		public async Task BoxSizeCenter()
		{
			Box b = Box.FromPoints(new Vec3(0.0, 0.0, 0.0), new Vec3(2.0, 4.0, 6.0));
			await Assert.That(b.Size()).IsEqualTo(new Vec3(2.0, 4.0, 6.0));
			await Assert.That(b.Center()).IsEqualTo(new Vec3(1.0, 2.0, 3.0));
		}

		[Test]
		public async Task BoxScale()
		{
			Box b = Box.FromPoints(new Vec3(-3.0, -1.0, -1.0), new Vec3(2.0, 1.0, 1.0));
			await Assert.That(b.Scale()).IsEqualTo(3.0);
		}

		[Test]
		public async Task BoxContains()
		{
			Box b = Box.FromPoints(new Vec3(0.0, 0.0, 0.0), new Vec3(1.0, 1.0, 1.0));
			await Assert.That(b.ContainsPoint(new Vec3(0.5, 0.5, 0.5))).IsTrue();
			await Assert.That(b.ContainsPoint(new Vec3(0.0, 0.0, 0.0))).IsTrue();
			await Assert.That(b.ContainsPoint(new Vec3(1.5, 0.5, 0.5))).IsFalse();
		}

		[Test]
		public async Task BoxUnion()
		{
			Box b = Box.FromPoints(new Vec3(0.0, 0.0, 0.0), new Vec3(1.0, 1.0, 1.0));
			b.UnionPoint(new Vec3(2.0, 2.0, 2.0));
			await Assert.That(b.Max).IsEqualTo(new Vec3(2.0, 2.0, 2.0));
		}

		[Test]
		public async Task BoxOverlap()
		{
			Box a = Box.FromPoints(new Vec3(0.0, 0.0, 0.0), new Vec3(1.0, 1.0, 1.0));
			Box b = Box.FromPoints(new Vec3(0.5, 0.5, 0.5), new Vec3(1.5, 1.5, 1.5));
			Box c = Box.FromPoints(new Vec3(2.0, 2.0, 2.0), new Vec3(3.0, 3.0, 3.0));
			await Assert.That(a.DoesOverlapBox(b)).IsTrue();
			await Assert.That(a.DoesOverlapBox(c)).IsFalse();
		}

		[Test]
		public async Task RectBasic()
		{
			Rect r = Rect.FromPoints(new Vec2(0.0, 0.0), new Vec2(3.0, 4.0));
			await Assert.That(r.Size()).IsEqualTo(new Vec2(3.0, 4.0));
			await Assert.That(Math.Abs(r.Area() - 12.0) < 1e-15).IsTrue();
			await Assert.That(r.Center()).IsEqualTo(new Vec2(1.5, 2.0));
		}

		[Test]
		public async Task NextHalfedge()
		{
			await Assert.That(Types.NextHalfedge(0)).IsEqualTo(1);
			await Assert.That(Types.NextHalfedge(1)).IsEqualTo(2);
			await Assert.That(Types.NextHalfedge(2)).IsEqualTo(0);
			await Assert.That(Types.NextHalfedge(3)).IsEqualTo(4);
			await Assert.That(Types.NextHalfedge(5)).IsEqualTo(3);
		}

		[Test]
		public async Task HalfedgeForward()
		{
			Halfedge h = new Halfedge(1, 3, 0, 0);
			await Assert.That(h.IsForward()).IsTrue();
			Halfedge h2 = new Halfedge(3, 1, 0, 0);
			await Assert.That(h2.IsForward()).IsFalse();
		}

		[Test]
		public async Task TriRefSameFace()
		{
			TriRef a = new TriRef(1, 2, 3, 4);
			TriRef b = new TriRef(1, 99, 3, 4);
			await Assert.That(a.SameFace(b)).IsTrue();
			TriRef c = new TriRef(1, 2, 99, 4);
			await Assert.That(a.SameFace(c)).IsFalse();
		}

		[Test]
		public async Task TmpEdgeOrdering()
		{
			// Verify min/max normalization
			TmpEdge e1 = new TmpEdge(5, 2, 0);
			await Assert.That(e1.First).IsEqualTo(2);
			await Assert.That(e1.Second).IsEqualTo(5);

			// Ordering: only first/second matter for Ord, not halfedge_idx
			TmpEdge e2 = new TmpEdge(2, 5, 1);
			await Assert.That(e1.CompareTo(e2)).IsEqualTo(0);

			// e1 < e3
			TmpEdge e3 = new TmpEdge(3, 5, 0);
			await Assert.That(e1 < e3).IsTrue();
		}

		[Test]
		[NotInParallel(QualityGlobalStateKey)]
		public async Task QualitySegments()
		{
			// Quality is process-global. TUnit's [NotInParallel] does not grant exclusive
			// ownership of the run: it only promises that tests sharing a constraint key
			// never execute concurrently *with each other*, so the key has to be shared
			// deliberately. Every future test that touches Quality — Phase 4's sphere and
			// cylinder constructors are the first — must carry this same key, or it will
			// race this one through the settings.
			Quality.ResetToDefaults();

			// C++ formula: min(360/angle, 2πr/length) + 3, rounded down to multiple of 4
			// For radius=1.0: min(36, 6) + 3 = 9, rounded down = 8
			int n = Quality.GetCircularSegments(1.0);
			await Assert.That(n).IsEqualTo(8);
			await Assert.That(n % 4).IsEqualTo(0);

			// For radius=50: min(36, 314) + 3 = 39, rounded down = 36
			int n50 = Quality.GetCircularSegments(50.0);
			await Assert.That(n50).IsEqualTo(36);
		}

		[Test]
		public async Task ErrorDisplay()
		{
			await Assert.That(Error.NoError.ToStr()).IsEqualTo("No Error");
			await Assert.That(Error.NotManifold.ToStr()).IsEqualTo("Not Manifold");
		}

		// ── C#-only, no Rust counterpart ──────────────────────────────────────
		// Rust's `vec![T::default(); n]` fills with the real default; C#'s `new T[n]`
		// fills with zeros, which for these two types is a *different, non-empty* box.
		// Box.Filled / Rect.Filled exist to close that gap, so they get a case each.

		[Test]
		public async Task BoxFilledGivesRustDefaults()
		{
			await Assert.That(Box.Filled(3).All(b => b == new Box() && b.IsEmpty())).IsTrue();
		}

		[Test]
		public async Task RectFilledGivesRustDefaults()
		{
			await Assert.That(Rect.Filled(3).All(r => r == new Rect() && r.IsEmpty())).IsTrue();
		}
	}
}
