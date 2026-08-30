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

// Arrangement.Sweeps.cs — the acceleration structures of robust/arrangement.rs:
// the conservative 2D box, the correctly-rounded translated approximations the
// filters run on, the interval sweep that enumerates overlapping box pairs, and
// the x-sorted point index. The module header is on Arrangement.cs.
//
// ── `[f64; 4]` is a Box2 ─────────────────────────────────────────────────────
// The Rust's box is a bare `[f64; 4]` = [min_x, min_y, max_x, max_y], indexed
// with computed indices (`b[k]` / `b[k + 2]`, `boxes[a][shi]`) all through the
// sweep. A named 4-tuple could not be indexed and an array would allocate one
// object per segment, so it lands as a struct with the same four slots and a
// by-index `ref` accessor, keeping every one of those expressions transcribable.
//
// ── total_cmp ────────────────────────────────────────────────────────────────
// Rust's `f64::total_cmp` is the IEEE-754 totalOrder predicate, which
// double.CompareTo is NOT: CompareTo says -0.0 == +0.0, totalOrder says
// -0.0 < +0.0. The sweep's sort and the point index's binary search are both
// documented to run on the total order (it is what makes them well defined in
// the presence of NaN), so TotalCmp below is the bit-twiddling definition from
// the Rust standard library, not an approximation of it.

using System.Diagnostics.CodeAnalysis;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp.Robust
{
	public static partial class ArrangementFunctions
	{
		/// <summary>
		/// Rust's <c>[f64; 4]</c> conservative 2D box: <c>[min_x, min_y, max_x, max_y]</c>.
		/// </summary>
		internal struct Box2
		{
			/// <summary>Slot 0: the low x bound.</summary>
			public double MinX;

			/// <summary>Slot 1: the low y bound.</summary>
			public double MinY;

			/// <summary>Slot 2: the high x bound.</summary>
			public double MaxX;

			/// <summary>Slot 3: the high y bound.</summary>
			public double MaxY;

			/// <summary>Creates a box from its four slots, in the Rust's order.</summary>
			/// <param name="minX">The low x bound.</param>
			/// <param name="minY">The low y bound.</param>
			/// <param name="maxX">The high x bound.</param>
			/// <param name="maxY">The high y bound.</param>
			public Box2(double minX, double minY, double maxX, double maxY)
			{
				this.MinX = minX;
				this.MinY = minY;
				this.MaxX = maxX;
				this.MaxY = maxY;
			}

			/// <summary>Slot by index, 0..3, as the Rust indexes its array.</summary>
			/// <param name="i">The slot index.</param>
			/// <returns>A reference to that slot.</returns>
			/// <exception cref="ArgumentOutOfRangeException">The index is not 0..3.</exception>
			[UnscopedRef]
			public ref double this[int i]
			{
				get
				{
					switch (i)
					{
						case 0:
							return ref this.MinX;
						case 1:
							return ref this.MinY;
						case 2:
							return ref this.MaxX;
						case 3:
							return ref this.MaxY;
						default:
							throw new ArgumentOutOfRangeException(nameof(i), $"Box2 index out of range: {i}");
					}
				}
			}
		}

		/// <summary>
		/// Conservative 2D box <c>[min_x, min_y, max_x, max_y]</c> around exact points
		/// from their correctly rounded f64 approximations: inflated past rounding
		/// error, so geometry lying exactly on/inside the exact hull is never
		/// rejected by a box test on the approximations. Used to prefilter the
		/// quadratic sweeps — Thingi10K's scan-like inputs put hundreds of
		/// short segments in one triangle, and paying an (often escalating) exact
		/// predicate per pair made 4k-triangle meshes run for minutes.
		/// </summary>
		/// <param name="pts">The approximations to bound, at least one.</param>
		/// <returns>The inflated box.</returns>
		internal static Box2 ApproxBox(ReadOnlySpan<Vec2> pts)
		{
			Box2 b = new Box2(pts[0][0], pts[0][1], pts[0][0], pts[0][1]);
			ReadOnlySpan<Vec2> rest = pts[1..];
			foreach (Vec2 p in rest)
			{
				// f64::min / f64::max, not Math.Min / Math.Max: a NaN operand loses
				// rather than poisoning the bound (the same call GraphGeom.SegBox3 makes,
				// docs/RUST_DIVERGENCES.md entry 2).
				b[0] = MinF64(b[0], p[0]);
				b[1] = MinF64(b[1], p[1]);
				b[2] = MaxF64(b[2], p[0]);
				b[3] = MaxF64(b[3], p[1]);
			}

			// ~4.5 ulp of slack (rounding error is ≤ 0.5 ulp) plus a subnormal floor.
			return new Box2(b[0] - Pad(b[0]), b[1] - Pad(b[1]), b[2] + Pad(b[2]), b[3] + Pad(b[3]));

			static double Pad(double x)
			{
				return (Math.Abs(x) * 1e-15) + MinPositive;
			}
		}

		/// <summary>
		/// One coordinate of <c>p − o</c>, correctly rounded to f64, plus "this f64 is
		/// EXACTLY that difference".
		/// </summary>
		/// <remarks>
		/// <c>pn/pd − on/od = (pn·od − on·pd) / (pd·od)</c>, handed to
		/// <see cref="Rational.IntRatioToF64"/> as a raw — deliberately unreduced —
		/// big-integer ratio. Correct rounding is a function of the VALUE, not of the
		/// representation, so this is bit-identical to subtracting in
		/// <see cref="BigRational"/> and rounding the result (pinned by
		/// <c>RobustExactTests.HomogeneousTranslationRoundsIdenticallyToRationalSubtraction</c>
		/// in RobustExactTests.Conversion.cs, the port of the Rust's
		/// <c>exact::tests::homogeneous_translation_rounds_identically_to_rational_subtraction</c>),
		/// but it skips the gcd that constructing the canonical difference would cost —
		/// one reduction per coordinate per point, on arrangements that reach over a
		/// million points.
		/// <para>
		/// Deliberately per-coordinate rather than over the <see cref="Homog2"/> values
		/// already on hand: homogenizing puts BOTH coordinates' denominators into every
		/// <c>W</c>, so the homogeneous form of this same difference carries a four-way
		/// denominator product and its wider operands cost more in the final division
		/// than the gcd ever did.
		/// </para>
		/// </remarks>
		/// <param name="pc">The point's coordinate.</param>
		/// <param name="oc">The origin's coordinate.</param>
		/// <returns>The rounded difference and whether it is exact.</returns>
		internal static (double Value, bool Exact) TranslatedCoord(in BigRational pc, in BigRational oc)
		{
			System.Numerics.BigInteger pn = Backend.Numer(pc);
			System.Numerics.BigInteger pd = Backend.Denom(pc);
			System.Numerics.BigInteger on = Backend.Numer(oc);
			System.Numerics.BigInteger od = Backend.Denom(oc);
			System.Numerics.BigInteger num = Backend.MulIntUint(pn, od) - Backend.MulIntUint(on, pd);
			System.Numerics.BigInteger den = Backend.IntFromUint(Backend.MulUint(pd, od));
			return Rational.IntRatioToF64(num, den);
		}

		/// <summary>
		/// <c>p − o</c> in the projection plane: both coordinates via
		/// <see cref="TranslatedCoord"/>, with the exactness flags combined (the tight
		/// filters need the whole point to be exact).
		/// </summary>
		/// <remarks>
		/// The flag here is the conversion's own "no rounding occurred", which is the
		/// exact-input filters' actual precondition. It is strictly more permissive than
		/// cdt.rs's <see cref="CdtFunctions.IsExact"/> — that one is a cheap syntactic test
		/// on a reduced rational and gives up on representable-but-large dyadics — so some
		/// points gain the tight filter that the standalone
		/// <see cref="CdtFunctions.Triangulate"/> path would not grant them. Sound either
		/// way: both filters certify true signs, and the flag is true only when the f64
		/// really is the translated value.
		/// </remarks>
		/// <param name="o">The origin.</param>
		/// <param name="p">The point.</param>
		/// <returns>The rounded translated point and whether it is exact.</returns>
		internal static (Vec2 Point, bool Exact) TranslatedApprox(R2 o, R2 p)
		{
			(double x, bool xExact) = TranslatedCoord(p.X, o.X);
			(double y, bool yExact) = TranslatedCoord(p.Y, o.Y);
			return (new Vec2(x, y), xExact && yExact);
		}

		/// <summary>Closed overlap of two conservative boxes.</summary>
		/// <param name="a">One box.</param>
		/// <param name="b">The other box.</param>
		/// <returns>True when they overlap.</returns>
		internal static bool BoxesOverlap(in Box2 a, in Box2 b)
		{
			return a.MinX <= b.MaxX && b.MinX <= a.MaxX && a.MinY <= b.MaxY && b.MinY <= a.MaxY;
		}

		/// <summary>Closed containment of a point in a conservative box.</summary>
		/// <param name="b">The box.</param>
		/// <param name="p">The point.</param>
		/// <returns>True when the point is inside or on the box.</returns>
		internal static bool BoxContains(in Box2 b, Vec2 p)
		{
			return p[0] >= b.MinX && p[0] <= b.MaxX && p[1] >= b.MinY && p[1] <= b.MaxY;
		}

		/// <summary>
		/// Rust's <c>f64::total_cmp</c>: the IEEE-754 totalOrder predicate, transcribed
		/// from the standard library's implementation.
		/// </summary>
		/// <param name="a">The left operand.</param>
		/// <param name="b">The right operand.</param>
		/// <returns>Negative, zero or positive as <c>a</c> sorts before, with, or after <c>b</c>.</returns>
		internal static int TotalCmp(double a, double b)
		{
			long left = BitConverter.DoubleToInt64Bits(a);
			long right = BitConverter.DoubleToInt64Bits(b);

			// Flip the whole word for negatives so the sign-magnitude bit pattern becomes
			// a two's-complement-ordered integer; the shifts are Rust's
			// `(x >> 63) as u64 >> 1`.
			left ^= (long)((ulong)(left >> 63) >> 1);
			right ^= (long)((ulong)(right >> 63) >> 1);
			return left.CompareTo(right);
		}

		/// <summary>
		/// Interval sweep producing <see cref="BoxPairs"/>. Returns null only when
		/// <paramref name="token"/> is cancelled: a monster triangle can hold thousands of
		/// segments and the sweep itself must stay interruptible.
		/// </summary>
		/// <param name="boxes">The conservative boxes to pair up.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The overlapping pairs, or null when cancelled.</returns>
		internal static BoxPairs? OverlappingBoxPairs(Box2[] boxes, CancelToken? token)
		{
			int n = boxes.Length;
			uint[] starts = new uint[n + 1];

			// Below this size the sweep's sorts cost more than the scan they save.
			if (n <= 64)
			{
				List<uint> smallPartners = new List<uint>();
				for (int i = 0; i < n; i++)
				{
					starts[i] = (uint)smallPartners.Count;
					for (int j = i + 1; j < n; j++)
					{
						if (BoxesOverlap(boxes[i], boxes[j]))
						{
							smallPartners.Add((uint)j);
						}
					}
				}

				starts[n] = (uint)smallPartners.Count;
				return new BoxPairs(starts, smallPartners.ToArray());
			}

			// Sweep along whichever axis packs the boxes more thinly: the expected
			// active-list length is (sum of box extents) / (total span).
			double[] lo = new double[] { double.PositiveInfinity, double.PositiveInfinity };
			double[] hi = new double[] { double.NegativeInfinity, double.NegativeInfinity };
			double[] width = new double[] { 0.0, 0.0 };
			foreach (Box2 b in boxes)
			{
				for (int k = 0; k < 2; k++)
				{
					lo[k] = MinF64(lo[k], b[k]);
					hi[k] = MaxF64(hi[k], b[k + 2]);
					width[k] += b[k + 2] - b[k];
				}
			}

			int axis = Density(1) < Density(0) ? 1 : 0;
			(int slo, int shi) = (axis, axis + 2);
			(int olo, int ohi) = (1 - axis, 3 - axis);

			// Ties broken by index so the sweep is a deterministic function of the
			// boxes (NaN coordinates, which the old scan rejected outright, sort to
			// one end under total_cmp and never satisfy an overlap test).
			uint[] order = new uint[n];
			for (uint i = 0; i < n; i++)
			{
				order[i] = i;
			}

			// sort_unstable_by is sound here (and is one of the audited unstable sites):
			// the index tie-break makes the comparator a total order, so no two elements
			// compare equal and stability cannot matter.
			Array.Sort(order, (a, b) =>
			{
				int c = TotalCmp(boxes[a][slo], boxes[b][slo]);
				return c != 0 ? c : a.CompareTo(b);
			});

			// The sweep is run twice — once to size each CSR row, once to fill it —
			// so the pair list is never materialized as (i, j) tuples.
			List<uint> active = new List<uint>();

			if (!Sweep((i, _) => starts[i + 1]++))
			{
				return null;
			}

			for (int i = 0; i < n; i++)
			{
				starts[i + 1] += starts[i];
			}

			// NARROWING AUDIT (uint -> int). Rust holds `total` in usize and allocates
			// `vec![0u32; total]`; here it narrows to int to size a C# array, whose
			// length is int-typed. C#'s default unchecked conversion truncates to the low
			// 32 bits exactly as Rust `as` does, so the two agree up to 2^31 pairs and a
			// larger count surfaces as an OverflowException from `new uint[total]` rather
			// than silently allocating short. That is not a parity gap worth closing: the
			// CSR row offsets themselves are `u32` on BOTH sides, so the Rust's own
			// `starts` overflow at 2^32 pairs in one triangle and its release build wraps
			// there silently. Neither implementation is defined past that point, and an
			// arrangement with two billion overlapping segment-box pairs on a single
			// triangle has exhausted memory long before either limit.
			int total = (int)starts[n];
			uint[] partners = new uint[total];
			uint[] cursor = new uint[n];
			Array.Copy(starts, cursor, n);
			if (!Sweep((i, j) =>
			{
				partners[cursor[i]] = j;
				cursor[i]++;
			}))
			{
				return null;
			}

			// The sweep visits each row's partners in sweep order, not index order.
			for (int i = 0; i < n; i++)
			{
				// Rust `sort_unstable` on a row of partner indices: a row holds each
				// partner at most once, so there are no ties to break and the unstable
				// introsort Array.Sort uses is exactly it.
				Array.Sort(partners, (int)starts[i], (int)(starts[i + 1] - starts[i]));
			}

			return new BoxPairs(starts, partners);

			double Density(int k)
			{
				double span = hi[k] - lo[k];
				if (span > 0.0 && double.IsFinite(span) && double.IsFinite(width[k]))
				{
					return width[k] / span;
				}
				else
				{
					return double.PositiveInfinity;
				}
			}

			bool Sweep(Action<uint, uint> emit)
			{
				active.Clear();
				for (int step = 0; step < order.Length; step++)
				{
					if (step % 1024 == 0 && Cancel.IsCancelled(token))
					{
						return false;
					}

					uint c = order[step];
					Box2 cb = boxes[c];

					// Retire boxes that ended before this one starts; what survives
					// already overlaps on the sweep axis (their low ends precede
					// cb[slo]), so only the other axis is left to test.
					// RemoveAll keeps the relative order of what remains, which is what
					// Rust's Vec::retain guarantees and what the emit order depends on.
					active.RemoveAll(a => !(boxes[a][shi] >= cb[slo]));
					foreach (uint a in active)
					{
						Box2 ab = boxes[a];
						if (ab[olo] <= cb[ohi] && cb[olo] <= ab[ohi])
						{
							emit(Math.Min(a, c), Math.Max(a, c));
						}
					}

					active.Add(c);
				}

				return true;
			}
		}

		/// <summary>
		/// Every index pair <c>i &lt; j</c> whose conservative boxes overlap, stored as CSR
		/// rows: <c>partners[starts[i]..starts[i + 1]]</c> holds <c>i</c>'s partners in
		/// ascending order.
		/// </summary>
		/// <remarks>
		/// Iterating rows in index order therefore yields exactly <c>(i, ascending j)</c>
		/// lexicographic order — the enumeration order of the O(S²) nested loops this
		/// replaces. The exact predicate stack downstream sees the identical sequence of
		/// surviving pairs and constructs identical points in identical order, so the
		/// acceleration is invisible to the output by construction.
		/// <para>
		/// CSR rather than a list of <c>(uint, uint)</c>: a monster triangle can produce tens
		/// of millions of overlapping pairs, and storing only the partner index halves the
		/// peak footprint and removes a global sort of that whole list.
		/// </para>
		/// </remarks>
		internal sealed class BoxPairs
		{
			private readonly uint[] starts;
			private readonly uint[] partners;

			/// <summary>Wraps the CSR arrays the sweep built.</summary>
			/// <param name="starts">Row offsets, length n + 1.</param>
			/// <param name="partners">Partner indices, row by row.</param>
			internal BoxPairs(uint[] starts, uint[] partners)
			{
				this.starts = starts;
				this.partners = partners;
			}

			/// <summary>The pairs, in <c>(i, ascending j)</c> lexicographic order.</summary>
			/// <returns>Each overlapping pair once, with the low index first.</returns>
			internal IEnumerable<(int I, int J)> Iter()
			{
				for (int i = 0; i + 1 < this.starts.Length; i++)
				{
					for (uint k = this.starts[i]; k < this.starts[i + 1]; k++)
					{
						yield return (i, (int)this.partners[k]);
					}
				}
			}
		}

		/// <summary>
		/// Static x-sorted index over the registered points, for the point-on-segment
		/// sweep: replaces the O(P) rescan of every point per segment with a range
		/// query over the segment's conservative box. Query results are re-sorted
		/// into ascending point index, which is the order the old inner loop visited
		/// them in, so the filtered predicates run in an unchanged sequence.
		/// </summary>
		internal sealed class PointIndex
		{
			private readonly int[] byX;

			private PointIndex(int[] byX)
			{
				this.byX = byX;
			}

			/// <summary>Builds the index over the approximations.</summary>
			/// <param name="apts">The rounded translated points.</param>
			/// <returns>The index.</returns>
			internal static PointIndex Build(List<Vec2> apts)
			{
				int[] byX = new int[apts.Count];
				for (int i = 0; i < byX.Length; i++)
				{
					byX[i] = i;
				}

				// Unstable is sound: the index tie-break makes the comparator a total order.
				Array.Sort(byX, (a, b) =>
				{
					int c = TotalCmp(apts[a][0], apts[b][0]);
					return c != 0 ? c : a.CompareTo(b);
				});
				return new PointIndex(byX);
			}

			/// <summary>
			/// Every point whose approximation lies in <paramref name="b"/>, in ascending
			/// point index.
			/// </summary>
			/// <param name="apts">The rounded translated points.</param>
			/// <param name="b">The query box.</param>
			/// <param name="output">Reused output buffer, cleared first.</param>
			internal void Query(List<Vec2> apts, in Box2 b, List<int> output)
			{
				output.Clear();

				// `byX` is ordered by `total_cmp`, so partitioning on that same total
				// order is well defined whatever the coordinates are; the rewind then
				// restores plain numeric semantics at the low end (−0.0 and +0.0 are
				// equal to `BoxContains` but not to `total_cmp`). The result is a
				// superset of what the old full rescan's `BoxContains` accepted,
				// which is all conservativeness requires.
				double bLo = b.MinX;
				double bHi = b.MaxX;
				int start = PartitionPoint(apts, this.byX, bLo);
				while (start > 0 && apts[this.byX[start - 1]][0] >= bLo)
				{
					start--;
				}

				for (int s = start; s < this.byX.Length; s++)
				{
					int i = this.byX[s];
					Vec2 p = apts[i];

					// Numeric test at the high end: the slice is numerically
					// non-decreasing (`total_cmp` only reorders values that compare
					// equal), so this never stops early, and a hypothetical NaN
					// simply fails the test and costs a wasted `BoxContains`.
					if (p[0] > bHi)
					{
						break;
					}

					if (BoxContains(b, p))
					{
						output.Add(i);
					}
				}

				// Rust `out.sort_unstable()`: the query visits each point index at most
				// once, so the indices are distinct, the ordering is fully determined and
				// stability cannot be observed.
				output.Sort();
			}

			/// <summary>
			/// Rust's <c>slice::partition_point</c> for the predicate
			/// <c>total_cmp(apts[i].x, bound).is_lt()</c>: the first index where it turns
			/// false.
			/// </summary>
			/// <param name="apts">The rounded translated points.</param>
			/// <param name="byX">The x-sorted point indices.</param>
			/// <param name="bound">The low bound to partition on.</param>
			/// <returns>The partition point.</returns>
			private static int PartitionPoint(List<Vec2> apts, int[] byX, double bound)
			{
				int lo = 0;
				int hi = byX.Length;
				while (lo < hi)
				{
					int mid = lo + ((hi - lo) / 2);
					if (TotalCmp(apts[byX[mid]][0], bound) < 0)
					{
						lo = mid + 1;
					}
					else
					{
						hi = mid;
					}
				}

				return lo;
			}
		}
	}
}
