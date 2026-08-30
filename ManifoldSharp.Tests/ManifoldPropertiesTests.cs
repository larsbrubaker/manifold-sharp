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

// Port of src/manifold_tests/properties.rs — its one test and the two sample
// helpers it needs, same inputs, same tolerance. Nothing deferred.
//
// The Rust file's own header:
//
//   Tests for the Properties suite that rely on the StretchyBracelet sample.
//   The bracelet sample is a sizable exercise of extrude + CrossSection +
//   boolean composition, so it lives here rather than next to the primitive
//   min_gap tests in advanced.rs (which is close to the line-count cap).
//
// Not to be confused with PropertiesTests.cs, which ports the inline tests of
// the production module properties.rs (volume, area, curvature). This is the
// manifold_tests module of the same name.
//
// This is a deliberately heavy test — the Rust measures 3.8s release and ~41s
// debug and enables it anyway, because two bracelets built from ~50 unions and
// two intersections each are the only thing in the suite that exercises MinGap
// over a mesh that large. Kept enabled here for the same reason; it is the
// slowest test in the C# suite too.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ManifoldPropertiesTests
	{
		/// <summary>
		/// C++ TEST(Properties, MingapStretchyBracelet) — two stacked bracelets 20 apart.
		/// Width is 15, so gap along z is 20 - 15 = 5.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppPropertiesMingapStretchyBracelet()
		{
			Manifold a = StretchyBracelet();
			Manifold b = StretchyBracelet().Translate(new Vec3(0.0, 0.0, 20.0));
			double distance = a.MinGap(b, 10.0);
			await Assert.That(Math.Abs(distance - 5.0) < 0.001)
				.IsTrue()
				.Because($"MingapStretchyBracelet: distance={distance}, expected ~5");
		}

		/// <summary>Port of C++ samples/bracelet.cpp <c>Base</c> helper.</summary>
		/// <param name="width">Extrusion height of the band.</param>
		/// <param name="radius">Radius of the cylindrical core.</param>
		/// <param name="decorRadius">Radius of each twisted decoration.</param>
		/// <param name="twistRadius">Offset of the decoration from its own axis.</param>
		/// <param name="nDecor">How many decorations around the band.</param>
		/// <param name="innerRadius">Inner radius of the stretch cuts.</param>
		/// <param name="outerRadius">Outer radius of the stretch cuts.</param>
		/// <param name="cut">Half-width of each stretch cut.</param>
		/// <param name="nCut">How many stretch cuts around the band.</param>
		/// <param name="nDivision">Circular segments for the decoration.</param>
		/// <returns>The bracelet base.</returns>
		private static Manifold BraceletBase(
			double width,
			double radius,
			double decorRadius,
			double twistRadius,
			int nDecor,
			double innerRadius,
			double outerRadius,
			double cut,
			int nCut,
			int nDivision)
		{
			Manifold baseManifold = Manifold.Cylinder(width, radius + (twistRadius / 2.0), -1.0, 0);

			CrossSection circle =
				CrossSection.Circle(decorRadius, nDivision).Translate(new Vec2(twistRadius, 0.0));
			Manifold decor = Manifold.Extrude(
					circle.ToPolygons(),
					width,
					nDivision,
					180.0,
					new Vec2(1.0, 1.0))
				.Scale(new Vec3(1.0, 0.5, 1.0))
				.Translate(new Vec3(0.0, radius, 0.0));

			for (int i = 0; i < nDecor; i++)
			{
				double angleDeg = (360.0 / nDecor) * i;
				baseManifold = baseManifold.Union(decor.Rotate(0.0, 0.0, angleDeg));
			}

			double dPhiRad = 2.0 * Types.KPi / nCut;
			Vec2 p0 = new Vec2(outerRadius, 0.0);
			Vec2 p1 = new Vec2(innerRadius, -cut);
			Vec2 p2 = new Vec2(innerRadius, cut);
			static Vec2 Rot(double theta, Vec2 v)
			{
				double c = Math.Cos(theta);
				double s = Math.Sin(theta);
				return new Vec2((c * v.X) - (s * v.Y), (s * v.X) + (c * v.Y));
			}

			SimplePolygon ring = new SimplePolygon(nCut * 4);
			for (int i = 0; i < nCut; i++)
			{
				double t = dPhiRad * i;
				ring.Add(Rot(t, p0));
				ring.Add(Rot(t, p1));
				ring.Add(Rot(t, p2));
				ring.Add(Rot(t, p0));
			}

			Polygons stretch = new Polygons { ring };

			baseManifold = Manifold.Extrude(stretch, width, 0, 0.0, new Vec2(1.0, 1.0))
				.Intersection(baseManifold);
			return baseManifold.AsOriginal();
		}

		/// <summary>Port of C++ samples/bracelet.cpp <c>StretchyBracelet</c>.</summary>
		/// <returns>The bracelet.</returns>
		private static Manifold StretchyBracelet()
		{
			double radius = 30.0;
			double height = 8.0;
			double width = 15.0;
			double thickness = 0.4;
			int nDecor = 20;
			int nCut = 27;
			int nDivision = 30;

			double twistRadius = Types.KPi * radius / nDecor;
			double decorRadius = twistRadius * 1.5;
			double outerRadius = radius + ((decorRadius + twistRadius) * 0.5);
			double innerRadius = outerRadius - height;
			double cut = 0.5 * ((Types.KPi * 2.0 * innerRadius / nCut) - thickness);
			double adjThickness = 0.5 * thickness * height / cut;

			Manifold outer = BraceletBase(
				width,
				radius,
				decorRadius,
				twistRadius,
				nDecor,
				innerRadius + thickness,
				outerRadius + adjThickness,
				cut - adjThickness,
				nCut,
				nDivision);
			Manifold inner = BraceletBase(
				width,
				radius - thickness,
				decorRadius,
				twistRadius,
				nDecor,
				innerRadius,
				outerRadius + (3.0 * adjThickness),
				cut,
				nCut,
				nDivision);
			return outer.Difference(inner);
		}
	}
}
