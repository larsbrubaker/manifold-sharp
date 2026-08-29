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

// Port of the tests module in constructors.rs — same 9 cases, same inputs, same
// assertions, in the same order.
//
// None of the nine carries TypesTests.QualityGlobalStateKey, and that is a
// property of the inputs, not an oversight: every Revolve/Cylinder call below
// passes an explicit circularSegments of 8, and both constructors consult
// Quality.GetCircularSegments only when that argument is <= 2. A later test that
// drops the explicit count (or exercises the auto path at all) is reading the
// process-global settings and must carry the key — see the note on
// TypesTests.QualitySegments.
//
// DEFERRED with their modules, because the callers do not exist yet:
//   - constructors.rs has no Sphere (it needs Subdivide), so there is nothing to
//     defer there; the Rust omits it too.
//   - The integration-level coverage of Extrude/Revolve/Cylinder lives in
//     manifold_tests/ and cross_section.rs, which arrive in Phases 6 and 8. Until
//     then the numeric parity of these three constructors is held by the
//     differential harness this step ran against the compiled Rust (see the step
//     report), not by a checked-in test.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ConstructorsTests
	{
		private static Polygons UnitSquare()
		{
			return new Polygons
			{
				new SimplePolygon
				{
					new Vec2(0.0, 0.0),
					new Vec2(1.0, 0.0),
					new Vec2(1.0, 1.0),
					new Vec2(0.0, 1.0),
				},
			};
		}

		[Test]
		public async Task ExtrudeBox()
		{
			ManifoldImpl m = Constructors.Extrude(UnitSquare(), 1.0, 0, 0.0, new Vec2(1.0, 1.0));

			// A unit cube extruded from a unit square: 8 verts, 12 triangles
			await Assert.That(m.Is2Manifold()).IsTrue().Because("extruded box is not 2-manifold");

			// bbox should span [0,1]^3
			await Assert.That(Math.Abs(m.Bbox.Min.Z) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(m.Bbox.Max.Z - 1.0) < 1e-10).IsTrue();
		}

		[Test]
		public async Task ExtrudeCone()
		{
			ManifoldImpl m = Constructors.Extrude(UnitSquare(), 1.0, 0, 0.0, new Vec2(0.0, 0.0));
			await Assert.That(m.Is2Manifold()).IsTrue().Because("extruded cone is not 2-manifold");
			await Assert.That(Math.Abs(m.Bbox.Max.Z - 1.0) < 1e-10).IsTrue();
		}

		[Test]
		public async Task ExtrudeTwist()
		{
			ManifoldImpl m = Constructors.Extrude(UnitSquare(), 1.0, 4, 90.0, new Vec2(1.0, 1.0));
			await Assert.That(m.Is2Manifold()).IsTrue().Because("twisted extrude is not 2-manifold");
		}

		[Test]
		public async Task RevolveFull()
		{
			// Revolve a square around Y axis → torus-ish solid ring
			SimplePolygon poly = new SimplePolygon
			{
				new Vec2(1.0, 0.0),
				new Vec2(2.0, 0.0),
				new Vec2(2.0, 1.0),
				new Vec2(1.0, 1.0),
			};
			ManifoldImpl m = Constructors.Revolve(new Polygons { poly }, 8, 360.0);
			await Assert.That(m.Is2Manifold()).IsTrue().Because("full revolve is not 2-manifold");
		}

		[Test]
		public async Task RevolvePartial()
		{
			SimplePolygon poly = new SimplePolygon
			{
				new Vec2(1.0, 0.0),
				new Vec2(2.0, 0.0),
				new Vec2(2.0, 1.0),
				new Vec2(1.0, 1.0),
			};
			ManifoldImpl m = Constructors.Revolve(new Polygons { poly }, 8, 180.0);
			await Assert.That(m.Is2Manifold()).IsTrue().Because("partial revolve is not 2-manifold");
		}

		[Test]
		public async Task CylinderBasic()
		{
			ManifoldImpl m = Constructors.Cylinder(1.0, 1.0, 1.0, 8, false);
			await Assert.That(m.Is2Manifold()).IsTrue().Because("cylinder is not 2-manifold");
			await Assert.That(Math.Abs(m.Bbox.Max.Z - 1.0) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(m.Bbox.Min.Z) < 1e-10).IsTrue();
		}

		[Test]
		public async Task CylinderCentered()
		{
			ManifoldImpl m = Constructors.Cylinder(2.0, 1.0, 1.0, 8, true);
			await Assert.That(m.Is2Manifold()).IsTrue().Because("centered cylinder is not 2-manifold");
			await Assert.That(Math.Abs(m.Bbox.Min.Z + 1.0) < 1e-10).IsTrue();
			await Assert.That(Math.Abs(m.Bbox.Max.Z - 1.0) < 1e-10).IsTrue();
		}

		[Test]
		public async Task CylinderCone()
		{
			// Cone: radius_high = 0
			ManifoldImpl m = Constructors.Cylinder(1.0, 1.0, 0.0, 8, false);
			await Assert.That(m.Is2Manifold()).IsTrue().Because("cone cylinder is not 2-manifold");
		}

		[Test]
		public async Task ExtrudeEmptyInvalid()
		{
			ManifoldImpl m = Constructors.Extrude(new Polygons(), 1.0, 0, 0.0, new Vec2(1.0, 1.0));
			await Assert.That(m.NumTri()).IsEqualTo(0);

			ManifoldImpl m2 = Constructors.Extrude(UnitSquare(), -1.0, 0, 0.0, new Vec2(1.0, 1.0));
			await Assert.That(m2.NumTri()).IsEqualTo(0);
		}
	}
}
