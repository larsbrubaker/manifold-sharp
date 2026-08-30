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
// Where the rest of these constructors' coverage lives:
//   - constructors.rs has no Sphere (it needs Subdivide), so there is nothing to
//     look for there; the Rust omits it too.
//   - The integration-level coverage of Extrude/Revolve/Cylinder lives in
//     manifold_tests/ and cross_section.rs. Both landed (Phases 6 and 8), so the
//     numeric parity of these three constructors is held by ManifoldBasicTests,
//     ManifoldComplexTests and CrossSectionTests as well as by the differential
//     harness constructors' own step ran against the compiled Rust.

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

		// -------------------------------------------------------------------
		// C#-ONLY adaptation tests. Not ports: manifold-rust has no counterpart,
		// because manifold-rust has the defect these pin. See
		// docs/RUST_DIVERGENCES.md entry 4 — `cylinder`'s `center` branch edited
		// vert_pos in place and refreshed only the bbox, leaving the cached face
		// BVH describing the pre-shift positions. Every boolean against such a
		// cylinder then queried a collider two units out in Z, missed
		// intersections, and tripped BooleanResult.PairUp's non-manifold assert.
		// The expected triangle counts below are the native library's, taken
		// through the ManifoldRust binding on the identical f64 arrays.
		// -------------------------------------------------------------------

		/// <summary>
		/// The centered cylinder must survive a boolean. Drilling an axis-aligned bore
		/// through a centered cube is about the most ordinary thing this API is asked for,
		/// and it threw.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task CenteredCylinderIsUsableInABoolean()
		{
			Manifold cube = Manifold.Cube(new Vec3(2.0, 2.0, 2.0), true);
			Manifold bore = Manifold.CylinderCentered(4.0, 0.4, 0.4, 64, true);

			Manifold drilled = cube.Difference(bore);

			await Assert.That(drilled.Status()).IsEqualTo(Error.NoError);
			await Assert.That(drilled.NumTri())
				.IsEqualTo(272)
				.Because("the native library produces 272 triangles for this subtraction");
		}

		/// <summary>
		/// The cone branch takes the same repair, because it is built on top of the
		/// centered cylinder: <c>Cylinder(h, radiusHigh, 0, n, center: true)</c> is its
		/// first step, so a stale collider there is mirrored into the cone.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task CenteredConeIsUsableInABoolean()
		{
			Manifold cube = Manifold.Cube(new Vec3(2.0, 2.0, 2.0), true);
			Manifold cone = Manifold.CylinderCentered(4.0, 0.0, 0.4, 64, true);

			Manifold drilled = cube.Difference(cone);

			await Assert.That(drilled.Status()).IsEqualTo(Error.NoError);
			await Assert.That(drilled.NumTri()).IsGreaterThan(0);
		}

		/// <summary>
		/// The root cause, pinned directly: a constructor must not hand back an impl whose
		/// cached collider disagrees with its own vertex positions.
		/// </summary>
		/// <remarks>
		/// Every face's true box is queried against the cached BVH; a leaf must at minimum
		/// find itself, since a box always overlaps itself. With the stale collider the
		/// bottom cap's true box sat at z = -2 while its leaf box sat at z = 0, so it found
		/// nothing. This is the assertion that says *why* the boolean above failed rather
		/// than merely that it did.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task CenteredCylinderColliderMatchesItsVertexPositions()
		{
			ManifoldImpl m = Constructors.Cylinder(4.0, 0.4, 0.4, 64, true);

			(Box[] faceBox, _) = Sort.GetFaceBoxMorton(m);
			bool[] foundItself = new bool[m.NumTri()];
			m.Collider.CollisionsWithBoxes(
				faceBox,
				false,
				(queryIdx, leafIdx) =>
				{
					if (queryIdx == leafIdx)
					{
						foundItself[queryIdx] = true;
					}
				});

			await Assert.That(foundItself.Count(found => !found))
				.IsEqualTo(0)
				.Because("every face must overlap its own leaf box in the cached collider");
		}
	}
}
