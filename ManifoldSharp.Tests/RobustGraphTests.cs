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

// RobustGraphTests.cs — NOT A TEST PORT. robust/graph_types.rs,
// robust/graph_geom.rs and robust/graph_self_cut.rs carry no `#[cfg(test)]`
// module between them: they are exercised by the Rust suite only through
// intersection_graph and the engine tests, which arrive later in Phase 10. So
// there was nothing to transcribe, and this file is an *adaptation* test in the
// sense of RobustExactAdaptationTests: it pins the three decisions where the C#
// port could silently diverge from the Rust without any ported test noticing.
//
// Every expected value below was produced by the differential harness this wave
// was written against — an in-crate `#[test]` calling the same Rust functions in
// the same order — and matched the C# line for line (584/584). They are the
// Rust's values, not the port's.
//
// What is pinned, and why each one is a real hazard:
//
//   1. The interner's two key spaces. `VertInterner::intern` routes exactly
//      representable rationals into the f64 key space and everything else into
//      the rational map, so three *distinct* rationals that all round to 1.0 get
//      three ids while their `verts_f64` entries are all 1.0. Get the routing
//      backwards and vertex identity quietly changes.
//   2. `[u64; 3]` ordering. Rust's array `Ord` on raw coordinate bits puts +1.0
//      before -1.0 (0x3ff0… < 0xbff0…), which is not the numeric order. The C#
//      value tuple must agree; a hand-rolled comparison that "fixed" the sign
//      would relabel edges.
//   3. Shared-vertex identity is IEEE `==`, not bit equality. Vec3.Equals is bit
//      equality here, so -0.0 and 0.0 would stop being the same shared vertex
//      and a mesh carrying a -0.0 would start cutting its own edge-neighbours.
//
// The whole-mesh test at the end is the coarse net over all of it: 12 triangles
// of a cube must classify as pure adjacency, and the same cube plus an offset
// copy must find exactly six real self-contacts.

using System.Numerics;

using TUnit.Assertions;
using TUnit.Assertions.Enums;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Tests
{
	public class RobustGraphTests
	{
		/// <summary>Rust's <c>f64::EPSILON</c>.</summary>
		private const double Eps = 2.220446049250313E-16;

		/// <summary>
		/// The adversarial point sequence the harness interns, in order: exact zeros
		/// including a negative one, a representable rational written three ways, a
		/// non-representable rational written two ways, a near-duplicate of it, and three
		/// distinct rationals that all round to 1.0.
		/// </summary>
		private static List<R3> AdversarialPoints()
		{
			return new List<R3>
			{
				R3.FromVec3(new Vec3(0.0, 0.0, 0.0)),
				R3.FromVec3(new Vec3(-0.0, -0.0, -0.0)),
				R3.FromVec3(new Vec3(0.0, -0.0, 0.0)),
				new R3(Ri(1, 2), Ri(1, 4), Ri(1, 8)),
				R3.FromVec3(new Vec3(0.5, 0.25, 0.125)),
				new R3(Ri(2, 4), Ri(4, 16), Ri(8, 64)),
				new R3(Ri(1, 3), Ri(0, 1), Ri(0, 1)),
				new R3(Ri(7, 21), Ri(0, 1), Ri(0, 1)),
				new R3(Ri(1000000, 3000001), Ri(0, 1), Ri(0, 1)),
				R3.FromVec3(new Vec3(1.0, 0.0, 0.0)),
				R3.FromVec3(new Vec3(1.0 + Eps, 0.0, 0.0)),
				new R3(Rational.Rat(1.0) + Ri(1, 1L << 62), Rational.Rat(0.0), Rational.Rat(0.0)),
				new R3(Rational.Rat(1.0) + Ri(1, 1L << 61), Rational.Rat(0.0), Rational.Rat(0.0)),
				new R3(Rational.Rat(1.0) - Ri(1, 1L << 62), Rational.Rat(0.0), Rational.Rat(0.0)),
				new R3(Ri(-1, 3), Ri(0, 1), Ri(0, 1)),
				new R3(Ri(-5, 15), Ri(0, 1), Ri(0, 1)),
				R3.FromVec3(new Vec3(0.0, 0.0, 0.0)),
				new R3(Ri(1, 3), Ri(0, 1), Ri(0, 1)),
			};
		}

		private static BigRational Ri(long n, long d)
		{
			return Backend.RatNew(new BigInteger(n), new BigInteger(d));
		}

		private static Vec3[] Tri(Vec3 a, Vec3 b, Vec3 c)
		{
			return new Vec3[] { a, b, c };
		}

		/// <summary>A closed 12-triangle cube of edge <paramref name="s"/> at the given origin.</summary>
		private static List<Vec3[]> Cube(double ox, double oy, double oz, double s)
		{
			Vec3 Pt(double x, double y, double z)
			{
				return new Vec3(ox + (x * s), oy + (y * s), oz + (z * s));
			}

			Vec3[] c =
			{
				Pt(0, 0, 0), Pt(1, 0, 0), Pt(1, 1, 0), Pt(0, 1, 0),
				Pt(0, 0, 1), Pt(1, 0, 1), Pt(1, 1, 1), Pt(0, 1, 1),
			};
			int[][] quads =
			{
				new[] { 0, 3, 2, 1 },
				new[] { 4, 5, 6, 7 },
				new[] { 0, 1, 5, 4 },
				new[] { 1, 2, 6, 5 },
				new[] { 2, 3, 7, 6 },
				new[] { 3, 0, 4, 7 },
			};
			List<Vec3[]> tris = new List<Vec3[]>();
			foreach (int[] q in quads)
			{
				tris.Add(Tri(c[q[0]], c[q[1]], c[q[2]]));
				tris.Add(Tri(c[q[0]], c[q[2]], c[q[3]]));
			}

			return tris;
		}

		[Test]
		public async Task VertInternerRoutesRepresentablePointsToTheF64KeySpace()
		{
			List<R3> points = AdversarialPoints();
			VertInterner interner = new VertInterner();
			List<uint> ids = new List<uint>();
			foreach (R3 p in points)
			{
				ids.Add(interner.Intern(p));
			}

			// Equal rationals built different ways share an id; the three rationals that
			// merely *round* to 1.0 (ids 6, 7, 8) do not share id 4 with 1.0 itself.
			await Assert.That(ids).IsEquivalentTo(
				new List<uint> { 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 9, 0, 2 },
				CollectionOrdering.Matching);
			await Assert.That(interner.Verts.Count).IsEqualTo(10);

			// …and they all carry 1.0 as their rounded approximation, which is the whole
			// reason the interner caches VertsF64 instead of re-rounding downstream.
			foreach (uint id in new uint[] { 4, 6, 7, 8 })
			{
				await Assert.That(BitConverter.DoubleToUInt64Bits(interner.VertsF64[(int)id].X))
					.IsEqualTo(0x3ff0000000000000UL);
			}

			// The same points fed to the f64 door instead collapse onto their rounded
			// values: 1.0's id for all three near-one rationals.
			VertInterner viaF64 = new VertInterner();
			List<uint> roundedIds = new List<uint>();
			foreach (R3 p in points)
			{
				roundedIds.Add(viaF64.InternF64(p.ToVec3Rounded()));
			}

			await Assert.That(roundedIds).IsEquivalentTo(
				new List<uint> { 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 4, 4, 4, 6, 6, 0, 2 },
				CollectionOrdering.Matching);
		}

		[Test]
		public async Task PointTableIdsAreDenseAndResolveInIdOrder()
		{
			List<R3> points = AdversarialPoints();
			PointTable table = new PointTable();
			List<uint> ids = new List<uint>();
			foreach (R3 p in points)
			{
				ids.Add(table.Intern(p));
			}

			await Assert.That(ids).IsEquivalentTo(
				new List<uint> { 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 9, 0, 2 },
				CollectionOrdering.Matching);
			await Assert.That(table.Count).IsEqualTo(10);

			// Resolve reconstructs the id-indexed order out of a Dictionary whose own
			// enumeration order is unspecified — the reason PointTable is allowed to be a
			// Dictionary at all.
			R3[] resolved = table.Resolve();
			await Assert.That(resolved.Length).IsEqualTo(10);
			for (int i = 0; i < resolved.Length; i++)
			{
				await Assert.That(table.Intern(resolved[i])).IsEqualTo((uint)i);
			}
		}

		[Test]
		public async Task BitEdgeKeysSortOnRawBitsNotOnValue()
		{
			// -0.0 folds onto +0.0, so both endpoints key identically.
			((ulong X, ulong Y, ulong Z) A, (ulong X, ulong Y, ulong Z) B) zeros =
				GraphTypes.BitEdgeKey(new Vec3(0.0, 0.0, 0.0), new Vec3(-0.0, -0.0, -0.0));
			await Assert.That(zeros.A).IsEqualTo((0UL, 0UL, 0UL));
			await Assert.That(zeros.B).IsEqualTo((0UL, 0UL, 0UL));

			// Canonical: the two orders of the same edge give the same key.
			Vec3 x1 = new Vec3(1.0, 0.0, 0.0);
			Vec3 y1 = new Vec3(0.0, 1.0, 0.0);
			await Assert.That(GraphTypes.BitEdgeKey(x1, y1)).IsEqualTo(GraphTypes.BitEdgeKey(y1, x1));

			// The ordering is Rust's `[u64; 3]` Ord on the *bit patterns*: +1.0 (0x3ff0…)
			// sorts before -1.0 (0xbff0…), the opposite of the numeric order.
			((ulong X, ulong Y, ulong Z) A, (ulong X, ulong Y, ulong Z) B) signed =
				GraphTypes.BitEdgeKey(new Vec3(-1.0, 0.0, 0.0), new Vec3(1.0, 0.0, 0.0));
			await Assert.That(signed.A.X).IsEqualTo(0x3ff0000000000000UL);
			await Assert.That(signed.B.X).IsEqualTo(0xbff0000000000000UL);

			await Assert.That(GraphTypes.EdgeKey(2, 1)).IsEqualTo(((uint)1, (uint)2));
			await Assert.That(GraphTypes.GeoEdgeKey(2, 1)).IsEqualTo(((uint)1, (uint)2));
		}

		[Test]
		public async Task PointOnSegmentEscalatesOnlyWhenTheFilterCannotDecide()
		{
			// (name, p, a, b, expected on-segment, expected to reach the exact tier)
			(R3 P, R3 A, R3 B, bool OnSegment, bool NeedsExact)[] cases =
			{
				(R3.FromVec3(new Vec3(0.5, 0.0, 0.0)), R3.FromVec3(new Vec3(0.0, 0.0, 0.0)), R3.FromVec3(new Vec3(1.0, 0.0, 0.0)), true, true),
				(R3.FromVec3(new Vec3(0.5, 1.0, 0.0)), R3.FromVec3(new Vec3(0.0, 0.0, 0.0)), R3.FromVec3(new Vec3(1.0, 0.0, 0.0)), false, false),
				(R3.FromVec3(new Vec3(2.0, 0.0, 0.0)), R3.FromVec3(new Vec3(0.0, 0.0, 0.0)), R3.FromVec3(new Vec3(1.0, 0.0, 0.0)), false, false),
				(new R3(Ri(1, 3), Ri(1, 3), Ri(1, 3)), R3.FromVec3(new Vec3(0.0, 0.0, 0.0)), R3.FromVec3(new Vec3(1.0, 1.0, 1.0)), true, true),
				(new R3(Ri(1, 3), Ri(1, 3), Ri(1000000, 3000001)), R3.FromVec3(new Vec3(0.0, 0.0, 0.0)), R3.FromVec3(new Vec3(1.0, 1.0, 1.0)), false, false),

				// Off the line by 2^-62 — and the FILTER rejects it, which is the
				// counter-intuitive half of the prefilter. Its bound is a product of
				// *per-axis* magnitude sums, so the y bound is 16·EPS·px·py with
				// py = 2^-62 itself: the bound collapses to ~5.8e-34 while the deviation
				// |cz| stays at 2^-62 ≈ 2.2e-19, fifteen orders above it. A deviation that
				// is tiny in absolute terms is not near-incident when every coordinate on
				// that axis is equally tiny — the bound shrinks with it and then some.
				(new R3(Rational.Rat(0.5), Ri(1, 1L << 62), Rational.Rat(0.0)), R3.FromVec3(new Vec3(0.0, 0.0, 0.0)), R3.FromVec3(new Vec3(1.0, 0.0, 0.0)), false, false),

				// Past the far endpoint by 2^-62: collinear, so only the parameter range
				// can reject it, and only exactly.
				(new R3(Rational.Rat(1.0) + Ri(1, 1L << 62), Rational.Rat(0.0), Rational.Rat(0.0)), R3.FromVec3(new Vec3(0.0, 0.0, 0.0)), R3.FromVec3(new Vec3(1.0, 0.0, 0.0)), false, true),
			};

			foreach ((R3 p, R3 a, R3 b, bool onSegment, bool needsExact) in cases)
			{
				Vec3 pa = GraphGeom.Approx3(p);
				Vec3 aa = GraphGeom.Approx3(a);
				Vec3 ba = GraphGeom.Approx3(b);

				// null from the prefilter means "the exact test must decide"; the filter
				// never certifies a hit.
				await Assert.That(Approx.NotOnSegmentA(pa, aa, ba) == null).IsEqualTo(needsExact);
				await Assert.That(GraphGeom.PointOnSegmentF(pa, p, aa, a, ba, b)).IsEqualTo(onSegment);
				await Assert.That(GraphGeom.PointOnSegment(p, a, b)).IsEqualTo(onSegment);

				// The inflated box never rejects a point that is on the segment.
				if (onSegment)
				{
					await Assert.That(GraphGeom.Box3Contains(GraphGeom.SegBox3(aa, ba), pa)).IsTrue();
				}
			}
		}

		[Test]
		public async Task CoplanarClipIsIndependentOfThePolygonsWinding()
		{
			List<R3> ccw = new List<R3>
			{
				R3.FromVec3(new Vec3(0.0, 0.0, 0.0)),
				R3.FromVec3(new Vec3(2.0, 0.0, 0.0)),
				R3.FromVec3(new Vec3(2.0, 2.0, 0.0)),
				R3.FromVec3(new Vec3(0.0, 2.0, 0.0)),
			};
			List<R3> cw = new List<R3>(ccw);
			cw.Reverse();

			R3 a = R3.FromVec3(new Vec3(-1.0, 1.0, 0.0));
			R3 b = R3.FromVec3(new Vec3(3.0, 1.0, 0.0));
			foreach (List<R3> poly in new[] { ccw, cw })
			{
				(R3 A, R3 B)? clipped = GraphGeom.ClipSegmentToPolygon(a, b, poly);
				await Assert.That(clipped != null).IsTrue();
				await Assert.That(Rational.R3Eq(clipped!.Value.A, R3.FromVec3(new Vec3(0.0, 1.0, 0.0)))).IsTrue();
				await Assert.That(Rational.R3Eq(clipped.Value.B, R3.FromVec3(new Vec3(2.0, 1.0, 0.0)))).IsTrue();

				// A segment whose endpoints are non-representable thirds still clips to
				// exact corner values.
				(R3 A, R3 B)? diagonal = GraphGeom.ClipSegmentToPolygon(
					new R3(Ri(-1, 3), Ri(-1, 3), Rational.Rat(0.0)),
					new R3(Ri(7, 3), Ri(7, 3), Rational.Rat(0.0)),
					poly);
				await Assert.That(diagonal != null).IsTrue();
				await Assert.That(Rational.R3Eq(diagonal!.Value.A, R3.FromVec3(new Vec3(0.0, 0.0, 0.0)))).IsTrue();
				await Assert.That(Rational.R3Eq(diagonal.Value.B, R3.FromVec3(new Vec3(2.0, 2.0, 0.0)))).IsTrue();

				await Assert.That(GraphGeom.ClipSegmentToPolygon(
					R3.FromVec3(new Vec3(-2.0, 3.0, 0.0)),
					R3.FromVec3(new Vec3(-1.0, 3.0, 0.0)),
					poly) == null).IsTrue();

				// Containment projects away the dominant axis, so the z of the first point
				// is dropped and it lands inside; the second is on the boundary (Zero, not
				// Neg, so still "in"); the third is outside in x.
				await Assert.That(GraphGeom.PointInPolygonCoplanar(R3.FromVec3(new Vec3(0.5, 0.25, 0.125)), poly)).IsTrue();
				await Assert.That(GraphGeom.PointInPolygonCoplanar(new R3(Ri(1, 3), Ri(0, 1), Ri(0, 1)), poly)).IsTrue();
				await Assert.That(GraphGeom.PointInPolygonCoplanar(new R3(Ri(-1, 3), Ri(0, 1), Ri(0, 1)), poly)).IsFalse();
			}
		}

		[Test]
		public async Task SharedVertexIdentityIsIeeeEqualityNotBitEquality()
		{
			// The two triangles share the origin and (1,0,0) — but one spells the origin's
			// x as -0.0. Under bit equality that is a *different* vertex, the pair would
			// look like a one-vertex contact, and a coplanar-fan shortcut would not apply;
			// under IEEE == it is the ordinary non-coplanar edge-neighbour it really is.
			SelfCutStats stats = new SelfCutStats();
			List<(R3 A, R3 B)>? cut = GraphSelfCut.RealSelfContact(
				Tri(new Vec3(-0.0, 0.0, 0.0), new Vec3(1.0, 0.0, 0.0), new Vec3(0.0, 1.0, 0.0)),
				Tri(new Vec3(0.0, -0.0, 0.0), new Vec3(1.0, 0.0, 0.0), new Vec3(0.0, 0.0, 1.0)),
				stats);
			await Assert.That(cut == null).IsTrue();
			await Assert.That(stats.EdgeBenign).IsEqualTo(1);
			await Assert.That(stats.VertBenign).IsEqualTo(0);
			await Assert.That(stats.Full).IsEqualTo(0);
		}

		[Test]
		public async Task RealSelfContactSeparatesAdjacencyFromSelfIntersection()
		{
			// A closed cube: every overlapping pair is an edge- or vertex-neighbour or a
			// plain miss, so nothing may be cut.
			SelfCutStats adjacency = new SelfCutStats();
			int adjacencyCuts = CountCuts(Cube(0.0, 0.0, 0.0, 1.0), adjacency, out int adjacencySegs);
			await Assert.That(adjacencyCuts).IsEqualTo(0);
			await Assert.That(adjacencySegs).IsEqualTo(0);
			await Assert.That(adjacency.Identical).IsEqualTo(0);
			await Assert.That(adjacency.EdgeBenign).IsEqualTo(18);
			await Assert.That(adjacency.VertBenign).IsEqualTo(16);
			await Assert.That(adjacency.Full).IsEqualTo(32);
			await Assert.That(adjacency.FullNone).IsEqualTo(20);
			await Assert.That(adjacency.FullPoint).IsEqualTo(12);
			await Assert.That(adjacency.FullSegBenign).IsEqualTo(0);

			// The same cube plus a copy offset by half an edge: the two surfaces really do
			// pass through each other, and exactly six triangle pairs must be cut.
			List<Vec3[]> selfContact = Cube(0.0, 0.0, 0.0, 1.0);
			selfContact.AddRange(Cube(0.5, 0.5, 0.5, 1.0));
			SelfCutStats stats = new SelfCutStats();
			int cuts = CountCuts(selfContact, stats, out int segs);
			await Assert.That(cuts).IsEqualTo(6);
			await Assert.That(segs).IsEqualTo(6);
			await Assert.That(stats.EdgeBenign).IsEqualTo(36);
			await Assert.That(stats.VertBenign).IsEqualTo(32);
			await Assert.That(stats.Full).IsEqualTo(208);
			await Assert.That(stats.FullNone).IsEqualTo(166);
			await Assert.That(stats.FullPoint).IsEqualTo(36);
			await Assert.That(stats.FullSegBenign).IsEqualTo(0);

			// Add merges per-worker counters field for field.
			SelfCutStats merged = new SelfCutStats();
			merged.Add(adjacency);
			merged.Add(stats);
			await Assert.That(merged.Full).IsEqualTo(240);
			await Assert.That(merged.EdgeBenign).IsEqualTo(54);
		}

		[Test]
		public async Task TriBoxAndDegeneracyAgreeWithTheExactNormal()
		{
			Vec3[] unit = Tri(new Vec3(0.0, 0.0, 0.0), new Vec3(1.0, 0.0, 0.0), new Vec3(0.0, 1.0, 0.0));
			Box box = GraphGeom.TriBox(unit);
			await Assert.That(box.Min).IsEqualTo(new Vec3(0.0, 0.0, 0.0));
			await Assert.That(box.Max).IsEqualTo(new Vec3(1.0, 1.0, 0.0));
			await Assert.That(GraphGeom.IsDegenerate(unit)).IsFalse();

			// Both of these have an f64 cross of exactly (0,0,0), which no bound can
			// certify away, so they are the cases that actually reach the rational cross —
			// and it agrees they are degenerate.
			await Assert.That(GraphGeom.IsDegenerate(
				Tri(new Vec3(0.0, 0.0, 0.0), new Vec3(1.0, 1.0, 1.0), new Vec3(2.0, 2.0, 2.0)))).IsTrue();
			await Assert.That(GraphGeom.IsDegenerate(
				Tri(new Vec3(1.0, 2.0, 3.0), new Vec3(1.0, 2.0, 3.0), new Vec3(4.0, 5.0, 6.0)))).IsTrue();

			// These two slivers, by contrast, exit at the float filter: the bound is a
			// product of per-axis magnitude sums, so the vanishing y magnitude drags it
			// below the cross it is meant to bound (1e-300 against ~2.7e-315). "Nearly
			// zero area" is not the same as "near-degenerate to the filter".
			await Assert.That(GraphGeom.IsDegenerate(
				Tri(new Vec3(0.0, 0.0, 0.0), new Vec3(1.0, 0.0, 0.0), new Vec3(0.5, 1e-300, 0.0)))).IsFalse();
			await Assert.That(GraphGeom.IsDegenerate(
				Tri(new Vec3(0.0, 0.0, 0.0), new Vec3(1.0, 0.0, 0.0), new Vec3(2.0, Eps, 0.0)))).IsFalse();
		}

		/// <summary>
		/// Runs <see cref="GraphSelfCut.RealSelfContact"/> over every unordered pair of a
		/// mesh's triangles.
		/// </summary>
		/// <param name="mesh">The triangles.</param>
		/// <param name="stats">Counters, accumulated across the whole sweep.</param>
		/// <param name="segments">The total number of cutting segments produced.</param>
		/// <returns>The number of pairs that produced a cut.</returns>
		private static int CountCuts(List<Vec3[]> mesh, SelfCutStats stats, out int segments)
		{
			int cuts = 0;
			segments = 0;
			for (int i = 0; i < mesh.Count; i++)
			{
				for (int j = i + 1; j < mesh.Count; j++)
				{
					List<(R3 A, R3 B)>? segs = GraphSelfCut.RealSelfContact(mesh[i], mesh[j], stats);
					if (segs != null)
					{
						cuts++;
						segments += segs.Count;
					}
				}
			}

			return cuts;
		}
	}
}
