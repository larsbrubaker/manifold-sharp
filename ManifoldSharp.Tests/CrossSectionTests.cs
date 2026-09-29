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

// Port of cross_section_tests.rs, the tests module of cross_section.rs and
// cross_section_ops.rs — all 20 cases, same inputs, same tolerances, same order —
// plus two C#-only regression tests in their own labeled region at the bottom,
// pinning the coordinate grid the boolean layer must produce. Nothing deferred.
// The constructor tests the Rust keeps in cross_section_ctor_tests.rs are
// CrossSectionCtorTests.cs. Where the Rust reaches the pub(crate) from_raw, this
// file calls the internal CrossSection.FromRaw (InternalsVisibleTo), and where it
// calls new — which unions, as the C++ Polygons constructor does — so does this.
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
using TUnit.Assertions.Enums;
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

		/// <summary>
		/// <see cref="CrossSection.Offset"/> reads the process-global Quality settings,
		/// so this carries <see cref="TypesTests.QualityGlobalStateKey"/>.
		/// </summary>
		[Test]
		[NotInParallel(TypesTests.QualityGlobalStateKey)]
		public async Task CrossSectionOffset()
		{
			CrossSection a = CrossSection.Square(1.0);
			CrossSection b = a.Offset(0.25);
			await Assert.That(b.Area() > a.Area()).IsTrue();
		}

		/// <summary>
		/// A 10x10 square minus an inner 4x4 square yields an outer contour and a
		/// hole; the hole's area is subtracted, giving 100 - 16 = 84.
		/// </summary>
		[Test]
		public async Task CrossSectionAreaSubtractsHoles()
		{
			CrossSection outer = CrossSection.Square(10.0);
			CrossSection hole = CrossSection.Square(4.0).Translate(new Vec2(3.0, 3.0));
			CrossSection ring = outer.Difference(hole);
			await Assert.That(ring.NumContour()).IsEqualTo(2)
				.Because("difference should yield outer + hole");
			await Assert.That(ring.Area()).IsEqualTo(84.0);
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

		/// <summary>
		/// C++ <c>CrossSection::Area</c> is <c>C2::Area(paths)</c>, which starts from
		/// <c>a = 0.0</c> and adds each contour, so a section with no contours reports
		/// +0.0 (an iterator <c>.sum()</c> of no f64s yields -0.0).
		/// </summary>
		[Test]
		public async Task CrossSectionAreaEmptyIsPositiveZero()
		{
			await Assert.That(BitConverter.DoubleToUInt64Bits(new CrossSection().Area()))
				.IsEqualTo(BitConverter.DoubleToUInt64Bits(0.0));
		}

		/// <summary>
		/// Off-origin polygons separate Clipper2's trapezoid Area from a shoelace
		/// sum in the last bits. Expected bits come from compiling Clipper2 commit
		/// 46f6391's <c>clipper.core.h</c> <c>Area</c> (MSVC /O2) on these exact
		/// coordinates: odd count (33) and even count (first 32 points).
		/// </summary>
		[Test]
		public async Task CrossSectionAreaMatchesClipper2Bits()
		{
			CrossSection cs = CrossSection.Circle(1.0, 33).Translate(new Vec2(100.0, -50.0));
			await Assert.That(BitConverter.DoubleToUInt64Bits(cs.Area())).IsEqualTo(0x4008fb2d94a5b1f1UL);
			Polygons even = cs.ToPolygons();
			even[0].RemoveAt(even[0].Count - 1);
			await Assert.That(BitConverter.DoubleToUInt64Bits(CrossSection.FromRaw(even).Area()))
				.IsEqualTo(0x4008f42c81cc8074UL);
		}

		/// <summary>
		/// C++ runs every Clipper2 op at <c>precision_ = 8</c> decimal places. ClipperD
		/// scales by the power of two above 10^precision (2^27 at 8, 2^20 at 6),
		/// so x = 1.00000012 snaps to 1 + 16 * 2^-27 = 1 + 2^-23, where precision
		/// 6 would round the 1.2e-7 feature away to 1.0.
		/// </summary>
		[Test]
		public async Task CrossSectionUnionKeepsEighthDecimal()
		{
			double x = 1.000_000_12;
			double snapped = 1.0 + Math.Pow(2.0, -23);
			CrossSection a = CrossSection.FromRaw(new Polygons
			{
				new SimplePolygon
				{
					new Vec2(0.0, 0.0),
					new Vec2(x, 0.0),
					new Vec2(x, 1.0),
					new Vec2(0.0, 1.0),
				},
			});
			CrossSection u = a.Union(new CrossSection());
			await Assert.That(u.Bounds().Max.X).IsEqualTo(snapped);
			CrossSection f = CrossSection.FromPolygonsFill(a.ToPolygons());
			await Assert.That(f.Bounds().Max.X).IsEqualTo(snapped);
		}

		/// <summary>
		/// C++ booleans use FillRule::Positive and the Polygons constructor
		/// defaults to Positive, so a clockwise (negative) contour fills nothing.
		/// </summary>
		[Test]
		public async Task CrossSectionBooleansUsePositiveFill()
		{
			CrossSection cw = CrossSection.FromRaw(new Polygons
			{
				new SimplePolygon
				{
					new Vec2(0.0, 0.0),
					new Vec2(0.0, 1.0),
					new Vec2(1.0, 1.0),
					new Vec2(1.0, 0.0),
				},
			});
			await Assert.That(cw.Union(new CrossSection()).IsEmpty()).IsTrue();
			await Assert.That(CrossSection.FromPolygonsFill(cw.ToPolygons()).IsEmpty()).IsTrue();
			CrossSection batch = CrossSection.BatchBoolean(new[] { cw.Clone(), cw.Clone() }, OpType.Add);
			await Assert.That(batch.IsEmpty()).IsTrue();
			CrossSection sq = CrossSection.Square(1.0);
			await Assert.That(sq.Intersection(cw).IsEmpty()).IsTrue();
			await Assert.That(sq.Difference(cw).Area()).IsEqualTo(1.0);
		}

		/// <summary>
		/// C++ <c>fr()</c> starts from EvenOdd and only overrides it for the three
		/// other enumerators, and <c>jt()</c> likewise starts from Square; unknown
		/// integer codes fall through to those initial values.
		/// </summary>
		[Test]
		public async Task CrossSectionUnknownCodesMatchCppDefaults()
		{
			SimplePolygon star = new SimplePolygon(5);
			for (int i = 0; i < 5; i++)
			{
				double a = ((double)i) * 4.0 * Math.PI / 5.0;
				star.Add(new Vec2(10.0 * DeterministicMath.Cos(a), 10.0 * DeterministicMath.Sin(a)));
			}

			CrossSection evenOdd = CrossSection.FromPolygonWithFillRule(new SimplePolygon(star), 0);
			CrossSection positive = CrossSection.FromPolygonWithFillRule(new SimplePolygon(star), 2);
			CrossSection unknown = CrossSection.FromPolygonWithFillRule(star, 99);
			await Assert.That(evenOdd.Area() < positive.Area()).IsTrue();
			await Assert.That(PolygonsEqual(unknown.ToPolygons(), evenOdd.ToPolygons())).IsTrue();
			CrossSection sq = CrossSection.Square(1.0);
			await Assert.That(PolygonsEqual(
				sq.OffsetWithParams(0.5, 99, 2.0, 0).ToPolygons(),
				sq.OffsetWithParams(0.5, 0, 2.0, 0).ToPolygons())).IsTrue();
		}

		/// <summary>
		/// C++ <c>Offset</c> defaults to Round joins with <c>circularSegments = 0</c>, which
		/// derives the arc tolerance from <c>Quality::GetCircularSegments(delta)</c>.
		/// </summary>
		/// <remarks>
		/// Reads the process-global Quality settings, so it carries
		/// <see cref="TypesTests.QualityGlobalStateKey"/>.
		/// </remarks>
		[Test]
		[NotInParallel(TypesTests.QualityGlobalStateKey)]
		public async Task CrossSectionOffsetDefaultSegmentsMatchQuality()
		{
			CrossSection sq = CrossSection.Square(1.0);
			int n = Quality.GetCircularSegments(3.0);
			Polygons expected = sq.OffsetWithParams(3.0, 1, 2.0, n).ToPolygons();
			await Assert.That(PolygonsEqual(sq.Offset(3.0).ToPolygons(), expected)).IsTrue();
			await Assert.That(PolygonsEqual(sq.OffsetWithParams(3.0, 1, 2.0, 0).ToPolygons(), expected))
				.IsTrue();
		}

		/// <summary>
		/// C++ <c>Decompose</c> groups holes by Clipper2's PolyTree containment, so the
		/// bar keeps its hole even though the U's bounding box also covers it. Expected
		/// contours and order from the C++ reference compiled against Clipper2 46f6391.
		/// </summary>
		[Test]
		public async Task DecomposeKeepsHoleWithItsOutline()
		{
			CrossSection cs = BarAndU();
			(double, double)[] bar = { (10.0, 2.0), (0.0, 2.0), (0.0, 0.0), (10.0, 0.0) };
			(double, double)[] hole = { (8.0, 1.5), (9.0, 1.5), (9.0, 0.5), (8.0, 0.5) };
			await Assert.That(PolygonsEqual(cs.ToPolygons(), Polys(UOutline, bar, hole))).IsTrue();
			List<Polygons> comps = cs.Decompose().Select(c => c.ToPolygons()).ToList();
			await Assert.That(ComponentsEqual(comps, Polys(bar, hole), Polys(UOutline))).IsTrue();
		}

		/// <summary>
		/// C++ emits the reversed stack of its outline/hole recursion: an island inside a
		/// hole is pushed before its enclosing outline, later siblings after. Expected
		/// order from the compiled C++ reference.
		/// </summary>
		[Test]
		public async Task DecomposeOrderMatchesCpp()
		{
			List<Polygons> comps = NestedRings().Decompose().Select(c => c.ToPolygons()).ToList();
			await Assert.That(ComponentsEqual(
				comps,
				Polys(new[] { (21.0, 1.0), (20.0, 1.0), (20.0, 0.0), (21.0, 0.0) }),
				Polys(Sq(5.0), Hole(4.0)),
				Polys(Sq(2.0), Hole(1.0)))).IsTrue();
		}

		/// <summary>
		/// C++ returns <c>*this</c> unchanged when <c>NumContour() &lt; 2</c>: an empty
		/// section decomposes to one empty section, and a single contour is not pushed
		/// through Clipper2 (which would snap it to the 2^-27 grid).
		/// </summary>
		[Test]
		public async Task DecomposeShortCircuitsBelowTwoContours()
		{
			List<CrossSection> empty = new CrossSection().Decompose();
			await Assert.That(empty.Count).IsEqualTo(1);
			await Assert.That(empty[0].IsEmpty()).IsTrue();
			CrossSection circ = CrossSection.Circle(1.0, 8).Translate(new Vec2(0.1, 0.2));
			List<CrossSection> comps = circ.Decompose();
			await Assert.That(comps.Count).IsEqualTo(1);
			await Assert.That(FlatBits(comps[0].ToPolygons()))
				.IsEquivalentTo(FlatBits(circ.ToPolygons()), CollectionOrdering.Matching);
		}

		/// <summary>
		/// C++ <c>Simplify</c> unions into a PolyTree and <c>flatten</c>s it, pushing each
		/// node's descendants before the node itself, so holes precede their outline.
		/// Expected contours from the compiled C++ reference.
		/// </summary>
		[Test]
		public async Task SimplifyFlattensPolytreeLikeCpp()
		{
			CrossSection ring = CrossSection.Square(10.0)
				.Difference(CrossSection.Square(4.0).Translate(new Vec2(3.0, 3.0)));
			await Assert.That(PolygonsEqual(
				ring.Simplify(1e-6).ToPolygons(),
				Polys(
					new[] { (3.0, 7.0), (7.0, 7.0), (7.0, 3.0), (3.0, 3.0) },
					new[] { (10.0, 10.0), (0.0, 10.0), (0.0, 0.0), (10.0, 0.0) }))).IsTrue();
			await Assert.That(PolygonsEqual(
				NestedRings().Simplify(1e-6).ToPolygons(),
				Polys(
					Hole(1.0),
					Sq(2.0),
					Hole(4.0),
					Sq(5.0),
					new[] { (21.0, 1.0), (20.0, 1.0), (20.0, 0.0), (21.0, 0.0) }))).IsTrue();
		}

		/// <summary>
		/// C++ <c>BatchBoolean</c> Add/Subtract run one <c>BooleanOp</c> with the first
		/// section as subject and the rest as clips; Intersect folds pairwise, and
		/// <c>Compose</c> is BatchBoolean Add. Expected contours from the compiled C++.
		/// </summary>
		[Test]
		public async Task BatchBooleanMatchesCpp()
		{
			List<CrossSection> secs = ThreeSquares();
			Polygons add = Polys(new[]
			{
				(2.0, 1.0),
				(3.0, 1.0),
				(3.0, 3.0),
				(1.0, 3.0),
				(1.0, 3.5),
				(-1.0, 3.5),
				(-1.0, 1.5),
				(0.0, 1.5),
				(0.0, 0.0),
				(2.0, 0.0),
			});
			await Assert.That(PolygonsEqual(CrossSection.BatchBoolean(secs, OpType.Add).ToPolygons(), add))
				.IsTrue();
			await Assert.That(PolygonsEqual(CrossSection.Compose(secs).ToPolygons(), add)).IsTrue();
			await Assert.That(PolygonsEqual(
				CrossSection.BatchBoolean(secs, OpType.Subtract).ToPolygons(),
				Polys(new[]
				{
					(2.0, 1.0),
					(1.0, 1.0),
					(1.0, 1.5),
					(0.0, 1.5),
					(0.0, 0.0),
					(2.0, 0.0),
				}))).IsTrue();
			await Assert.That(CrossSection.BatchBoolean(secs, OpType.Intersect).IsEmpty()).IsTrue();
		}

		/// <summary>
		/// C++ <c>BatchBoolean</c> returns <c>crossSections[0]</c> itself for a single
		/// input, so neither it nor <c>Compose</c> snaps the contours through Clipper2.
		/// </summary>
		[Test]
		public async Task BatchBooleanSingleSectionIsUnchanged()
		{
			CrossSection circ = CrossSection.Circle(1.0, 8).Translate(new Vec2(0.1, 0.2));
			CrossSection[] one = { circ.Clone() };
			List<(ulong, ulong)> want = FlatBits(circ.ToPolygons());
			foreach (OpType op in new[] { OpType.Add, OpType.Subtract, OpType.Intersect })
			{
				await Assert.That(FlatBits(CrossSection.BatchBoolean(one, op).ToPolygons()))
					.IsEquivalentTo(want, CollectionOrdering.Matching);
			}

			await Assert.That(FlatBits(CrossSection.Compose(one).ToPolygons()))
				.IsEquivalentTo(want, CollectionOrdering.Matching);
			await Assert.That(CrossSection.Compose(Array.Empty<CrossSection>()).IsEmpty()).IsTrue();
		}

		/// <summary>
		/// Subtract runs one <c>BooleanOp</c> with every tail contour as a clip; a pairwise
		/// fold reaches the same region with its contours in another order. Clip triangles
		/// as C++ <c>Hull</c> emits them; expected contours from the compiled C++ reference
		/// (its pairwise fold gives c1/c2 swapped).
		/// </summary>
		[Test]
		public async Task BatchSubtractIsOneBooleanOp()
		{
			static CrossSection Tri((double, double)[] p) => CrossSection.FromRaw(Polys(p));
			CrossSection[] secs =
			{
				CrossSection.Square(8.0).Translate(new Vec2(1.0, 1.0)),
				Tri(new[]
				{
					(5.505859375, 9.8291015625),
					(6.05078125, 0.2421875),
					(9.4619140625, 1.42578125),
				}),
				Tri(new[]
				{
					(2.4287109375, 5.0869140625),
					(5.408203125, 0.8125),
					(4.029296875, 9.94140625),
				}),
			};
			await Assert.That(PolygonsEqual(
				CrossSection.BatchBoolean(secs, OpType.Subtract).ToPolygons(),
				Polys(
					new[]
					{
						(2.4287109375, 5.0869140625),
						(3.718903623521328, 9.0),
						(1.0, 9.0),
						(1.0, 1.0),
						(5.2775057330727577, 1.0),
					},
					new[]
					{
						(5.5529856532812119, 9.0),
						(4.1714947372674942, 9.0),
						(5.3798815608024597, 1.0),
						(6.0077070519328117, 1.0),
					},
					new[]
					{
						(9.0, 9.0),
						(5.8961778432130814, 9.0),
						(9.0, 2.4069637954235077),
					},
					new[]
					{
						(9.0, 1.2655064538121223),
						(8.2348068803548813, 1.0),
						(9.0, 1.0),
					}))).IsTrue();
		}

		/// <summary>
		/// C++ <c>Warp</c> goes through <c>WarpBatch</c>, which re-unions the moved
		/// contours with FillRule::Positive at <c>precision_</c>: a warp that twists a
		/// square into a bowtie keeps only the positively wound lobe (expected contour
		/// from the compiled C++ reference), and moved vertices land on Clipper2's grid.
		/// </summary>
		[Test]
		public async Task WarpUnionsLikeCpp()
		{
			CrossSection bowtie = CrossSection.Square(2.0).Warp((ref Vec2 v) =>
			{
				if (v.Y > 1.0)
				{
					v.X = 2.0 - v.X;
				}
			});
			await Assert.That(PolygonsEqual(
				bowtie.ToPolygons(),
				Polys(new[] { (1.0, 1.0), (0.0, 0.0), (2.0, 0.0) }))).IsTrue();
			await Assert.That(bowtie.Area()).IsEqualTo(1.0);
			CrossSection stretched = CrossSection.Square(1.0).Warp((ref Vec2 v) => { v.X *= 1.000_000_12; });
			await Assert.That(stretched.Bounds().Max.X).IsEqualTo(1.0 + Math.Pow(2.0, -23));
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

			// fillRule 1 = NonZero: any fill rule reaches union_d; this one keeps the
			// counter-clockwise triangle whichever way the booleans' own rule points.
			Polygons result = CrossSection.FromPolygonWithFillRule(triangle, 1).ToPolygons();

			await Assert.That(result.Count).IsEqualTo(1);
			await Assert.That(result[0].Count).IsEqualTo(3);

			// manifold-rust returns 0.12345679104328156 here at precision 8:
			// 0.1234567890123 * 2^27 is 16570089.727, which rounds away from zero to
			// 16570090, and 16570090 / 2^27 is that value exactly.
			const ulong RustBits = 0x3fbf9add40000000;

			// What Clipper2Lib's own ClipperD would have returned: 0.12345679, on the
			// 10^-8 grid. Named so the failure message says which side you landed on.
			const ulong ClipperDBits = 0x3fbf9add3b84e659;

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
		/// scales to the 10^-8 grid (precision 8), offsets by nothing, and scales back,
		/// so <c>0.1234567890123</c> comes out as <c>0.12345679</c>.
		/// <para>
		/// The coordinate is off-grid on purpose. The differential harness's
		/// <c>Offset(0.0)</c> case ran on a square with coordinates 0 and 2, which are
		/// exact on every grid in play, so it matched the Rust for the whole time the
		/// guard was missing. Any replacement for this test has to keep a coordinate that
		/// is not representable at 10^-8.
		/// </para>
		/// </remarks>
		[Test]
		public async Task OffsetByZeroIsIdentity()
		{
			Vec2 offGrid = new Vec2(0.1234567890123, 1.0);
			// FromRaw, not the unioning constructor, which would snap offGrid to the
			// 2^-27 grid before Offset ever saw it.
			CrossSection cs = CrossSection.FromRaw(new Polygons
			{
				new SimplePolygon { new Vec2(0.0, 0.0), new Vec2(1.0, 0.0), offGrid },
			});

			// Both entry points carry the guard, so both are pinned.
			Polygons viaOffset = cs.Offset(0.0).ToPolygons();
			Polygons viaParams = cs.OffsetWithParams(0.0, 1, 2.0, 16).ToPolygons();

			// What Clipper2Lib's ungated InflatePaths would have produced for that x.
			const ulong ReGriddedBits = 0x3fbf9add3b84e659;
			ulong expected = BitConverter.DoubleToUInt64Bits(offGrid.X);

			foreach (Polygons result in new[] { viaOffset, viaParams })
			{
				await Assert.That(result.Count).IsEqualTo(1);
				await Assert.That(result[0].Count).IsEqualTo(3);

				ulong actual = BitConverter.DoubleToUInt64Bits(result[0][2].X);
				await Assert.That(actual).IsNotEqualTo(ReGriddedBits)
					.Because("a zero offset went through InflatePaths and got snapped to the 10^-8 grid");
				await Assert.That(actual).IsEqualTo(expected)
					.Because($"expected the input's 0x{expected:x16} back unchanged, got 0x{actual:x16}");
			}
		}

		#endregion

		/// <summary>The Rust <c>U_OUTLINE</c>: the U's single outline as C++ emits it.</summary>
		private static readonly (double, double)[] UOutline =
		{
			(11.0, 2.5),
			(7.0, 2.5),
			(7.0, 2.25),
			(10.5, 2.25),
			(10.5, -0.25),
			(7.0, -0.25),
			(7.0, -0.5),
			(11.0, -0.5),
		};

		/// <summary>The Rust <c>polys</c>: Polygons from coordinate-pair literals.</summary>
		private static Polygons Polys(params (double X, double Y)[][] contours)
		{
			Polygons result = new Polygons(contours.Length);
			foreach ((double X, double Y)[] c in contours)
			{
				SimplePolygon poly = new SimplePolygon(c.Length);
				foreach ((double x, double y) in c)
				{
					poly.Add(new Vec2(x, y));
				}

				result.Add(poly);
			}

			return result;
		}

		/// <summary>
		/// The Rust <c>bar_and_u</c>: the bar [0,10]x[0,2] with hole [8,9]x[0.5,1.5],
		/// unioned with a U whose bbox [7,11]x[-0.5,2.5] covers the hole's vertices while
		/// its opening embraces the bar's right end.
		/// </summary>
		private static CrossSection BarAndU()
		{
			CrossSection bar = CrossSection.SquareVec2(new Vec2(10.0, 2.0), false).Difference(
				CrossSection.SquareVec2(new Vec2(1.0, 1.0), false).Translate(new Vec2(8.0, 0.5)));
			CrossSection u = CrossSection.SquareVec2(new Vec2(4.0, 3.0), false)
				.Translate(new Vec2(7.0, -0.5))
				.Difference(
					CrossSection.SquareVec2(new Vec2(3.5, 2.5), false).Translate(new Vec2(7.0, -0.25)));
			return bar.Union(u);
		}

		/// <summary>
		/// The Rust tests' <c>nest</c>, built from their <c>ring</c> closure: a 10/8 ring
		/// around a 4/2 ring, both centered, plus a unit square off at x = 20.
		/// </summary>
		private static CrossSection NestedRings()
		{
			static CrossSection Ring(double outer, double inner) =>
				CrossSection.SquareVec2(new Vec2(outer, outer), true)
					.Difference(CrossSection.SquareVec2(new Vec2(inner, inner), true));
			return Ring(10.0, 8.0)
				.Union(Ring(4.0, 2.0))
				.Union(CrossSection.Square(1.0).Translate(new Vec2(20.0, 0.0)));
		}

		/// <summary>The Rust tests' <c>sq</c> closure: a centered outline of half-width h.</summary>
		private static (double, double)[] Sq(double h)
		{
			return new[] { (h, h), (-h, h), (-h, -h), (h, -h) };
		}

		/// <summary>The Rust tests' <c>hole</c> closure: a centered hole of half-width h.</summary>
		private static (double, double)[] Hole(double h)
		{
			return new[] { (-h, h), (h, h), (h, -h), (-h, -h) };
		}

		/// <summary>The Rust <c>three_squares</c>.</summary>
		private static List<CrossSection> ThreeSquares()
		{
			return new List<CrossSection>
			{
				CrossSection.Square(2.0),
				CrossSection.Square(2.0).Translate(new Vec2(1.0, 1.0)),
				CrossSection.Square(2.0).Translate(new Vec2(-1.0, 1.5)),
			};
		}

		/// <summary>
		/// The Rust tests' <c>bits</c> closure: every vertex's coordinate bit patterns,
		/// contours flattened in order.
		/// </summary>
		private static List<(ulong, ulong)> FlatBits(Polygons p)
		{
			List<(ulong, ulong)> bits = new List<(ulong, ulong)>();
			foreach (SimplePolygon c in p)
			{
				foreach (Vec2 v in c)
				{
					bits.Add((BitConverter.DoubleToUInt64Bits(v.X), BitConverter.DoubleToUInt64Bits(v.Y)));
				}
			}

			return bits;
		}

		/// <summary>
		/// The Rust <c>assert_eq!</c> on a <c>Vec&lt;Polygons&gt;</c>: same component count,
		/// and each component equal under <see cref="PolygonsEqual"/>, in order.
		/// </summary>
		private static bool ComponentsEqual(List<Polygons> actual, params Polygons[] expected)
		{
			if (actual.Count != expected.Length)
			{
				return false;
			}

			for (int i = 0; i < actual.Count; i++)
			{
				if (!PolygonsEqual(actual[i], expected[i]))
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// The Rust <c>assert_eq!</c> on two <c>Polygons</c>: the derived <c>PartialEq</c>,
		/// so contour order, vertex order and every coordinate must agree under IEEE
		/// <c>==</c> (<see cref="Vec2"/>'s <c>operator==</c>, not its bitwise
		/// <c>Equals</c>). TUnit's collection equivalence is order-insensitive by default,
		/// which is why this is spelled out.
		/// </summary>
		private static bool PolygonsEqual(Polygons a, Polygons b)
		{
			if (a.Count != b.Count)
			{
				return false;
			}

			for (int i = 0; i < a.Count; i++)
			{
				if (a[i].Count != b[i].Count)
				{
					return false;
				}

				for (int j = 0; j < a[i].Count; j++)
				{
					if (a[i][j] != b[i][j])
					{
						return false;
					}
				}
			}

			return true;
		}
	}
}
