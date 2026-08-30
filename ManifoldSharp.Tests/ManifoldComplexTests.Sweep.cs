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

// Test 22 of src/manifold_tests/complex.rs — BooleanComplex.Sweep — split out of
// ManifoldComplexTests.cs for the 800-line cap because of the 90 path points it
// carries. The Rust's array is `[(f64, f64); 90]`, transcribed here digit for
// digit as a flat f64 array of 90 (x, y) pairs; every point is scaled by 0.9 at
// use, exactly as the Rust does.
//
// ── Trig sites ───────────────────────────────────────────────────────────────
// Every cos/sin/atan2/sqrt/floor/ceil below is System.Math, because the Rust
// test calls std `f64::cos`/`sin`/`atan2`/… — not `crate::math::*`. Per the
// port's per-site rule the C# follows the Rust site for site, so a fixture that
// uses std libm gets .NET's libm. The assertion is a ±1 volume band around
// 3757, which is why that is safe here.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ManifoldComplexTests
	{
		/// <summary>
		/// The Rust test's <c>path_points_raw</c> — 90 (x, y) pairs, interleaved.
		/// </summary>
		private static readonly double[] SweepPathPointsRaw =
		{
			-21.707751473606564, 10.04202769267855,
			-21.840846948218307, 9.535474475521578,
			-21.940954413815387, 9.048287386171369,
			-22.005569458385835, 8.587741145234093,
			-22.032187669917704, 8.16111047331591,
			-22.022356960178296, 7.755456475810721,
			-21.9823319178086, 7.356408291345673,
			-21.91208498286602, 6.964505631629036,
			-21.811437268778267, 6.579251589515578,
			-21.68020988897306, 6.200149257860059,
			-21.51822395687812, 5.82670172951726,
			-21.254086890521585, 5.336709200579579,
			-21.01963533308061, 4.974523796623895,
			-20.658228140926262, 4.497743844638198,
			-20.350337020134603, 4.144115181723373,
			-19.9542029967, 3.7276501717684054,
			-20.6969129296381, 3.110639833377638,
			-21.026318197401537, 2.793796378245609,
			-21.454710558515973, 2.3418076758544806,
			-21.735944543382722, 2.014266362004704,
			-21.958999535447845, 1.7205197644485681,
			-22.170169612837164, 1.3912359628761894,
			-22.376940405634056, 1.0213515348242117,
			-22.62545385249271, 0.507889651991388,
			-22.77620002102207, 0.13973666928102288,
			-22.8689989640578, -0.135962138067232,
			-22.974385239894364, -0.5322784681448909,
			-23.05966775687304, -0.9551466941218276,
			-23.102914137841445, -1.2774406685179822,
			-23.14134824916783, -1.8152432718003662,
			-23.152085124298473, -2.241104719188421,
			-23.121576743285054, -2.976332948223073,
			-23.020491352156856, -3.6736813934577914,
			-22.843552165110886, -4.364810769710428,
			-22.60334013490563, -5.033012850282157,
			-22.305015243491663, -5.67461444847819,
			-21.942709324216615, -6.330962778427178,
			-21.648491707764062, -6.799117771996025,
			-21.15330508818782, -7.496539096945377,
			-21.10687739725184, -7.656798276710632,
			-21.01253055778545, -8.364144493707382,
			-20.923211927856293, -8.782280691344269,
			-20.771325204062215, -9.258087073404687,
			-20.554404009259198, -9.72613360625344,
			-20.384050989017144, -9.985885743112847,
			-20.134404839253612, -10.263023004626703,
			-19.756998832033442, -10.613109670467736,
			-18.83161393127597, -15.68768837402245,
			-19.155593463785983, -17.65410871259763,
			-17.930304365744544, -19.005810988385562,
			-16.893408103100064, -19.50558228186199,
			-16.27514960757635, -19.8288501942628,
			-15.183033464853374, -20.47781203017123,
			-14.906850387751492, -20.693472553142833,
			-14.585198957236713, -21.015257964547136,
			-11.013839210807205, -34.70394287828328,
			-8.79778020674896, -36.17434400175442,
			-7.850491148257242, -36.48835987119041,
			-6.982497182376991, -36.74546968896842,
			-6.6361688522576, -36.81653354539242,
			-6.0701080598244035, -36.964332993204,
			-5.472439187922815, -37.08824838436714,
			-4.802871164820756, -37.20127157090685,
			-3.6605994233344745, -37.34427653957914,
			-1.7314396363710867, -37.46415201430501,
			-0.7021130485987349, -37.5,
			0.01918509410483974, -37.49359541901704,
			1.2107837650065625, -37.45093992812552,
			3.375529069920302, 32.21823383780513,
			1.9041980552754056, 32.89839543047101,
			1.4107184651094313, 33.16556804736585,
			1.1315552947605065, 33.34344755450097,
			0.8882931135353977, 33.52377699790175,
			0.6775397019893341, 33.708817857198056,
			0.49590284067753837, 33.900831612019715,
			0.2291596803839543, 34.27380625039597,
			0.03901816126171688, 34.66402375075138,
			-0.02952797094655369, 34.8933309389416,
			-0.0561772851849209, 35.044928843125824,
			-0.067490756643705, 35.27129875796868,
			-0.05587453990569748, 35.42204271802184,
			0.013497378362074697, 35.72471438137191,
			0.07132375113026912, 35.877348797053145,
			0.18708820875448923, 36.108917464873215,
			0.39580614140195136, 36.424415957998825,
			0.8433687814267005, 36.964365016108914,
			0.7078417131710703, 37.172455373435916,
			0.5992848016685662, 37.27482757003058,
			0.40594743344375905, 37.36664006036318,
			0.1397973410299913, 37.434752779117005,
		};

		/// <summary>
		/// C++ TEST(BooleanComplex, Sweep) — sweep a fillet profile along a closed 2D
		/// path, building an <c>Extrude</c>+<c>Warp</c> primitive per segment and
		/// batch-unioning. Expects final volume ≈ 3757.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppComplexSweep()
		{
			double pi = Math.PI;
			double kTwoPi = 2.0 * pi;

			// profile: (filletWidth-filletRadius, 0) → arc of 10 pts → (0, filletWidth) → (0,0)
			double filletRadius = 2.5;
			double filletWidth = 5.0;
			int numArcPoints = 10;
			Vec2 arcCp = new Vec2(filletWidth - filletRadius, filletRadius);

			SimplePolygon profile = new SimplePolygon
			{
				new Vec2(0.0, 0.0),
				new Vec2(filletWidth - filletRadius, 0.0),
			};
			for (int i = 0; i < numArcPoints; i++)
			{
				double angle = i * pi / numArcPoints;
				double y = arcCp.Y - (Math.Cos(angle) * filletRadius);
				double x = arcCp.X + (Math.Sin(angle) * filletRadius);
				profile.Add(new Vec2(x, y));
			}

			profile.Add(new Vec2(0.0, filletWidth));
			Polygons profilePolys = new Polygons { profile };

			double MinPosAngle(double angle)
			{
				double div = angle / kTwoPi;
				double whole = Math.Floor(div);
				return angle - (whole * kTwoPi);
			}

			Manifold PartialRevolve(double startAngle, double endAngle, int nSegmentsPerRotation)
			{
				double posEnd = MinPosAngle(endAngle);
				double total = startAngle < 0.0 && endAngle < 0.0 && startAngle < endAngle
					? endAngle - startAngle
					: posEnd - startAngle;
				int nSegments = (int)Math.Ceiling((total / kTwoPi * nSegmentsPerRotation) + 1.0);
				if (nSegments < 2)
				{
					nSegments = 2;
				}

				double angleStep = total / (nSegments - 1);
				double nSegmentsF = nSegments - 1;
				return Manifold.Extrude(
					profilePolys,
					nSegmentsF,
					nSegments - 2,
					0.0,
					new Vec2(1.0, 1.0))
					.Warp((ref Vec3 v) =>
					{
						double zIndex = nSegmentsF - v.Z;
						double angle = (zIndex * angleStep) + startAngle;
						double oldX = v.X;
						double oldY = v.Y;
						v.Z = oldY;
						v.Y = oldX * Math.Sin(angle);
						v.X = oldX * Math.Cos(angle);
					});
			}

			static double Det(Vec2 a, Vec2 b) => (a.X * b.Y) - (a.Y * b.X);

			List<Manifold> CutterPrimitives(Vec2 p1, Vec2 p2, Vec2 p3)
			{
				Vec2 diff = p2 - p1;
				Vec2 v1 = p1 - p2;
				Vec2 v2 = p3 - p2;
				double determinant = Det(v1, v2);
				double startAngle = Math.Atan2(v1.X, -v1.Y);
				double endAngle = Math.Atan2(-v2.X, v2.Y);
				Manifold round = PartialRevolve(startAngle, endAngle, 20)
					.Translate(new Vec3(p2.X, p2.Y, 0.0));
				double distance = Math.Sqrt((diff.X * diff.X) + (diff.Y * diff.Y));
				double angle = Math.Atan2(diff.Y, diff.X);
				Manifold extrusion =
					Manifold.Extrude(profilePolys, distance, 0, 0.0, new Vec2(1.0, 1.0))
						.Rotate(90.0, 0.0, -90.0)
						.Translate(new Vec3(distance, 0.0, 0.0))
						.Rotate(0.0, 0.0, angle * 180.0 / pi)
						.Translate(new Vec3(p1.X, p1.Y, 0.0));
				return determinant < 0.0
					? new List<Manifold> { round, extrusion }
					: new List<Manifold> { extrusion };
			}

			// Exact C++ path_points, scaled by 0.9
			int n = SweepPathPointsRaw.Length / 2;
			Vec2[] pathPoints = new Vec2[n];
			for (int i = 0; i < n; i++)
			{
				pathPoints[i] = new Vec2(SweepPathPointsRaw[2 * i], SweepPathPointsRaw[(2 * i) + 1]) * 0.9;
			}

			List<Manifold> primitives = new List<Manifold>();
			for (int i = 0; i < n; i++)
			{
				primitives.AddRange(
					CutterPrimitives(pathPoints[i], pathPoints[(i + 1) % n], pathPoints[(i + 2) % n]));
			}

			Manifold shape = Manifold.BatchBoolean(primitives, OpType.Add);
			await Assert.That(Math.Abs(shape.Volume() - 3757.0) < 1.0)
				.IsTrue()
				.Because($"Sweep: vol={shape.Volume()}, expected ~3757");
		}
	}
}
