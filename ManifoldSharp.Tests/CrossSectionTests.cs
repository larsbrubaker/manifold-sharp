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

// Port of the tests module in cross_section.rs — all 5 cases, same inputs, same
// tolerances, same order — plus two C#-only regression tests in their own labeled
// region at the bottom, pinning the coordinate grid the boolean layer must
// produce. Nothing deferred.
//
// The interim gap this file used to carry is closed. Its two deferrals were
// test_cpp_cross_section_square (needs Manifold::cube, Manifold::extrude and the
// boolean engine to express `a.difference(&b).volume()`) and the 14 cases of
// manifold_tests/cross_section2.rs; the Phase 6 façade landed both. The
// fourteen are now CrossSection2Tests.cs, which is where the real coverage of
// Simplify, Decompose, Minkowski, Hull, Warp, BatchBoolean, Compose, Mirror and
// the fill-rule constructors lives — until they landed, that coverage was the
// differential harness against the compiled Rust (101 cases, bit-for-bit) rather
// than a checked-in test.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class CrossSectionTests
	{
		[Test]
		public async Task CrossSectionAreaBounds()
		{
			CrossSection cs = CrossSection.Square(2.0);
			await Assert.That(Math.Abs(cs.Area() - 4.0) < 1e-10).IsTrue();
			Rect b = cs.Bounds();
			await Assert.That(Math.Abs(b.Max.X - 2.0) < 1e-10).IsTrue();
		}

		[Test]
		public async Task CrossSectionBoolean()
		{
			CrossSection a = CrossSection.Square(2.0);
			CrossSection b = CrossSection.Square(2.0).Translate(new Vec2(1.0, 0.0));
			await Assert.That(a.Intersection(b).Area() > 0.9).IsTrue();
			await Assert.That(a.Union(b).Area() > a.Area()).IsTrue();
			await Assert.That(a.Difference(b).Area() < a.Area()).IsTrue();
		}

		[Test]
		public async Task CrossSectionOffset()
		{
			CrossSection a = CrossSection.Square(1.0);
			CrossSection b = a.Offset(0.25);
			await Assert.That(b.Area() > a.Area()).IsTrue();
		}

		/// <summary>C++ TEST(CrossSection, Square) — cube from extrusion matches cube.</summary>
		/// <remarks>
		/// Not the same test as <c>CrossSection2Tests.CppCrossSectionSquare</c>, which ports
		/// the C++ case of the same name out of manifold_tests/cross_section2.rs: that one
		/// builds its square with <c>SquareVec2((5,5), false)</c> and allows 1e-4, this one
		/// uses <c>Square(5)</c> and allows 1e-6. Both are in the Rust; both are ported.
		/// </remarks>
		[Test]
		public async Task CppCrossSectionSquare()
		{
			CrossSection cs = CrossSection.Square(5.0);
			Manifold a = Manifold.Cube(new Vec3(5.0, 5.0, 5.0), false);
			Manifold b = Manifold.Extrude(cs.ToPolygons(), 5.0, 0, 0.0, new Vec2(1.0, 1.0));
			Manifold diff = a.Difference(b);
			await Assert.That(Math.Abs(diff.Volume()) < 1e-6)
				.IsTrue()
				.Because($"CrossSection square extrusion should match cube, diff volume: {diff.Volume()}");
		}

		/// <summary>C++ TEST(CrossSection, Empty) — empty cross section from empty polygons.</summary>
		[Test]
		public async Task CppCrossSectionEmpty()
		{
			Polygons polys = new Polygons { new SimplePolygon(), new SimplePolygon() };
			CrossSection cs = new CrossSection(polys);
			await Assert.That(Math.Abs(cs.Area()) < 1e-10).IsTrue()
				.Because("CrossSection from empty polygons should have zero area");
		}

		#region C#-only regression tests (no Rust counterpart)

		/// <summary>
		/// Pins the boolean layer to clipper2-rust's power-of-two coordinate grid.
		/// </summary>
		/// <remarks>
		/// This has no counterpart in cross_section.rs — it exists because the obvious
		/// C# transcription of <c>union_d</c> is wrong in a way nothing else in the
		/// suite notices. Clipper2Lib's <c>ClipperD</c> scales by <c>10^precision</c>;
		/// upstream C++ Clipper2 (issue #25) and therefore clipper2-rust scale by
		/// <c>2^(ilogb(10^precision)+1)</c>. Both return three well-formed points in the
		/// same order, so counts, ordering, winding and every tolerance-based assertion
		/// in this file pass either way — only the low mantissa bits differ, and the
		/// port's whole premise is that those match.
		/// <para>
		/// <see cref="CrossSection.FromPolygonWithFillRule"/> with NonZero is the
		/// shortest production path to a bare union, so this drives the real code rather
		/// than a copy of it. If someone "simplifies" CrossSection.Clipper.cs back to
		/// <c>Clipper.Union(PathsD, ...)</c>, this is the assertion that fails.
		/// </para>
		/// </remarks>
		[Test]
		public async Task ClipperDScaleIsPowerOfTwo()
		{
			// The x that is not representable on either grid, so the two disagree.
			SimplePolygon triangle = new SimplePolygon
			{
				new Vec2(0.0, 0.0),
				new Vec2(1.0, 0.0),
				new Vec2(0.1234567890123, 1.0),
			};

			// fillRule 1 = NonZero, matching the Rust's union_d call.
			Polygons result = CrossSection.FromPolygonWithFillRule(triangle, 1).ToPolygons();

			await Assert.That(result.Count).IsEqualTo(1);
			await Assert.That(result[0].Count).IsEqualTo(3);

			// manifold-rust returns 0.12345695495605469 here: 0.1234567890123 * 2^20 is
			// 129453.826, which rounds away from zero to 129454, and 129454 / 2^20 is
			// that value exactly.
			const ulong RustBits = 0x3fbf9ae000000000;

			// What Clipper2Lib's own ClipperD would have returned: 0.123457, on the
			// 10^-6 grid. Named so the failure message says which side you landed on.
			const ulong ClipperDBits = 0x3fbf9ae0c1765775;

			ulong actual = BitConverter.DoubleToUInt64Bits(result[0][0].X);
			await Assert.That(actual).IsNotEqualTo(ClipperDBits)
				.Because("the booleans went back through Clipper2Lib's ClipperD, which grids to 10^-precision");
			await Assert.That(actual).IsEqualTo(RustBits)
				.Because($"expected manifold-rust's 0x{RustBits:x16}, got 0x{actual:x16}");
		}

		/// <summary>
		/// Pins <c>Offset(0)</c> to returning the input verbatim, not a re-gridded copy.
		/// </summary>
		/// <remarks>
		/// No Rust counterpart: clipper2-rust's <c>inflate_paths_d</c> returns
		/// <c>paths.clone()</c> before it scales anything when delta is zero
		/// (clipper.rs:256), and Clipper2Lib's <c>InflatePaths</c> has no such guard — it
		/// scales to the 10^-6 grid, offsets by nothing, and scales back, so
		/// <c>0.1234567890123</c> comes out as <c>0.123457</c>.
		/// <para>
		/// The coordinate is off-grid on purpose. The differential harness's
		/// <c>Offset(0.0)</c> case ran on a square with coordinates 0 and 2, which are
		/// exact on every grid in play, so it matched the Rust for the whole time the
		/// guard was missing. Any replacement for this test has to keep a coordinate that
		/// is not representable at 10^-6.
		/// </para>
		/// </remarks>
		[Test]
		public async Task OffsetByZeroIsIdentity()
		{
			Vec2 offGrid = new Vec2(0.1234567890123, 1.0);
			CrossSection cs = new CrossSection(new Polygons
			{
				new SimplePolygon { new Vec2(0.0, 0.0), new Vec2(1.0, 0.0), offGrid },
			});

			// Both entry points carry the guard, so both are pinned.
			Polygons viaOffset = cs.Offset(0.0).ToPolygons();
			Polygons viaParams = cs.OffsetWithParams(0.0, 1, 2.0, 16).ToPolygons();

			// What Clipper2Lib's ungated InflatePaths would have produced for that x.
			const ulong ReGriddedBits = 0x3fbf9ae0c1765775;
			ulong expected = BitConverter.DoubleToUInt64Bits(offGrid.X);

			foreach (Polygons result in new[] { viaOffset, viaParams })
			{
				await Assert.That(result.Count).IsEqualTo(1);
				await Assert.That(result[0].Count).IsEqualTo(3);

				ulong actual = BitConverter.DoubleToUInt64Bits(result[0][2].X);
				await Assert.That(actual).IsNotEqualTo(ReGriddedBits)
					.Because("a zero offset went through InflatePaths and got snapped to the 10^-6 grid");
				await Assert.That(actual).IsEqualTo(expected)
					.Because($"expected the input's 0x{expected:x16} back unchanged, got 0x{actual:x16}");
			}
		}

		#endregion
	}
}
