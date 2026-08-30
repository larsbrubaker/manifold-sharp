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

// BraceletBench.cs — the stretchy-bracelet MinGap as a driver.
//
// No Rust example behind this one either: manifold-rust measures it as the test
// `test_cpp_properties_mingap_stretchy_bracelet` (C++ TEST(Properties,
// MingapStretchyBracelet)), which is where its "bracelet 3.83s sequential / 3.1s with
// --features parallel" figures come from. The geometry is
// ManifoldPropertiesTests.CppPropertiesMingapStretchyBracelet's, helpers included, so
// this measures the same code the suite runs.
//
// The split reported below matters when reading the numbers: two thirds of the work is
// *building* each bracelet (a cylinder unioned with 20 twisted extrusions, then
// intersected with a 27-tooth stretch ring, twice, then differenced), and only the
// tail is the MinGap collision query itself. A regression in either is visible
// separately.
//
// Run: dotnet run -c Release --project ManifoldSharp.Benchmarks -- bracelet

using System.Globalization;

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Benchmarks
{
	/// <summary>Two stacked stretchy bracelets and the minimum gap between them.</summary>
	internal static class BraceletBench
	{
		/// <summary>Run the driver once.</summary>
		/// <param name="args">Unused; the sample has no size parameter.</param>
		/// <returns>Build, MinGap and total timing samples.</returns>
		public static IReadOnlyList<Sample> Run(string[] args)
		{
			_ = args;

			long start = System.Diagnostics.Stopwatch.GetTimestamp();
			Manifold a = StretchyBracelet();
			Manifold b = StretchyBracelet().Translate(new Vec3(0.0, 0.0, 20.0));
			int nTri = a.NumTri();
			double build = DriverArgs.SecondsSince(start);

			long gapStart = System.Diagnostics.Stopwatch.GetTimestamp();
			double distance = a.MinGap(b, 10.0);
			double gap = DriverArgs.SecondsSince(gapStart);
			double total = DriverArgs.SecondsSince(start);

			// C++ TEST(Properties, MingapStretchyBracelet): width is 15 and the offset is
			// 20, so the gap along z is 5.
			if (Math.Abs(distance - 5.0) >= 0.001)
			{
				throw new InvalidOperationException(string.Create(
					CultureInfo.InvariantCulture,
					$"MingapStretchyBracelet: distance={distance}, expected ~5"));
			}

			Console.WriteLine(string.Create(
				CultureInfo.InvariantCulture,
				$"nTri = {nTri}, build = {build} sec, mingap = {gap} sec, total = {total} sec"));

			return new[]
			{
				new Sample("bracelet/build", build, "sec"),
				new Sample("bracelet/mingap", gap, "sec"),
				new Sample("bracelet/total", total, "sec"),
			};
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

			List<Vec2> ring = new List<Vec2>(nCut * 4);
			for (int i = 0; i < nCut; i++)
			{
				double t = dPhiRad * i;
				ring.Add(Rot(t, p0));
				ring.Add(Rot(t, p1));
				ring.Add(Rot(t, p2));
				ring.Add(Rot(t, p0));
			}

			List<List<Vec2>> stretch = new List<List<Vec2>> { ring };

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
