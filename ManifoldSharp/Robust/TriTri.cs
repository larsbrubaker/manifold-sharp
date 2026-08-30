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

// TriTri.cs — port of robust/tri_tri.rs — Exact triangle-triangle intersection
// for the robust boolean engine.
//
// Narrow phase behind the Collider broad phase: given one triangle from each
// operand mesh, classify their intersection exactly as nothing, a single
// point, a segment, or (for coplanar pairs) a convex overlap polygon. All
// vertex-vs-plane tests go through the filtered predicates in
// Robust/Exact/Filtered.cs; every constructed point is exact rational
// (Robust/Exact/Predicates.Constructions.cs), so downstream arrangements
// (Robust/Arrangement.cs) never see rounded coordinates.
//
// Degenerate (zero-area) input triangles are the caller's responsibility to
// drop beforehand (paper §5 pre-processing); this file debug-asserts that.
//
// The one Rust file lands as three, to stay under the 800-line cap:
//   TriTri.cs           — the narrow phase proper and the interval overlap
//   TriTri.Coplanar.cs  — CoplanarSeparated2d, CoplanarOverlap, the
//                         Sutherland–Hodgman clip and its canonicalization
//   TriTriIsect.cs      — the result type
//
// ── Rust's [Vec3; 3] triangle parameters ─────────────────────────────────────
// The Rust passes triangles as `[Vec3; 3]` by value on the stack. C# has no
// fixed-length array type, and a `ReadOnlySpan<Vec3>` cannot be captured by the
// local functions that carry this file's closure-heavy structure, so triangles
// arrive as `Vec3[]` of length three (checked, as Approx.SatEdgeAxesDisjoint
// checks). The caller therefore allocates one 3-element array per triangle per
// candidate pair where the Rust allocates nothing; that is a Phase 11
// measurement item, not a correctness one.

using System.Diagnostics;
using System.Globalization;
using System.Numerics;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

// `type Frac = (Int, Int);` — an unreduced fraction with positive denominator.
// A C# using-alias over a named tuple is the literal reading of the Rust alias:
// same two BigIntegers, no wrapper type, no allocation.
using Frac = (System.Numerics.BigInteger Num, System.Numerics.BigInteger Den);

namespace ManifoldSharp.Robust
{
	/// <summary>
	/// The exact triangle-triangle narrow phase (Rust's <c>robust::tri_tri</c>).
	/// </summary>
	public static partial class TriTri
	{
		/// <summary>
		/// Dominant-axis choice for the paper's bijective drop-one-coordinate
		/// projection: the axis of the exactly-largest |normal| component (ties
		/// broken toward z, then y). The chosen component is guaranteed nonzero for
		/// a non-degenerate triangle.
		/// </summary>
		public static int DominantAxis(R3 n)
		{
			BigRational ax = Backend.RatAbs(n.X);
			BigRational ay = Backend.RatAbs(n.Y);
			BigRational az = Backend.RatAbs(n.Z);
			if (az >= ax && az >= ay)
			{
				return 2;
			}
			else if (ay >= ax)
			{
				return 1;
			}
			else
			{
				return 0;
			}
		}

		/// <summary>
		/// Forwarder for the existing call sites; the implementation lives with the
		/// other integer-only constructions in Robust/Exact/Predicates.Constructions.cs.
		/// </summary>
		/// <remarks>
		/// The Rust writes <c>pub use super::exact::predicates::lift_to_plane;</c>. C#
		/// has no re-export, so this one-line delegation stands in for it and keeps
		/// <c>tri_tri::lift_to_plane</c> call sites (arrangement.rs, the tri_tri tests)
		/// transcribable.
		/// </remarks>
		public static R3 LiftToPlane(R2 p, int axis, R3 a, R3 n)
		{
			return Predicates.LiftToPlane(p, axis, a, n);
		}

		/// <summary>
		/// Exit-path counters for perf analysis, printed under MANIFOLD_TIMING by
		/// the self-cut loop. Relaxed atomics in the Rust; here plain
		/// <see cref="Interlocked"/> adds, negligible cost on the hot path.
		/// </summary>
		/// <remarks>
		/// The Rust exposes the <c>AtomicU64</c> statics directly and every mutation is
		/// a <c>fetch_add</c> from inside this module. Public mutable statics have no
		/// good C# spelling, so the counters are private and the only public operation
		/// is <see cref="SnapshotAndReset"/> — which is all any caller
		/// (intersection_graph.rs) ever does with them.
		/// </remarks>
		public static class Stats
		{
			private static long planeReject;
			private static long coplanar;
			private static long coplanarSat;
			private static long satReject;
			private static long interval;
			private static long coplanarNs;

			// Time spent in the exact coplanar clip proper, i.e. after the f64
			// separating-edge pre-reject has failed. Broken out because the
			// pre-reject and the clip differ by two orders of magnitude per pair.
			private static long coplanarClipNs;
			private static long planeNs;
			private static long intervalNs;

			/// <summary>
			/// Reads and zeroes every counter, formatting them as the Rust's one-line
			/// summary.
			/// </summary>
			/// <remarks>
			/// The format string is transcribed specifier for specifier from
			/// <c>stats::snapshot_and_reset</c> so that trace diffs against the Rust show
			/// only real differences (see the same argument on Timing.FormatStageLine).
			/// Rust's <c>{:.3}</c> is C#'s <c>F3</c>; the two disagree only on exact
			/// midpoints (Rust rounds half to even, .NET half away from zero), which no
			/// wall-clock reading ever hits meaningfully.
			/// </remarks>
			/// <returns>The formatted counter line.</returns>
			public static string SnapshotAndReset()
			{
				return string.Create(
					CultureInfo.InvariantCulture,
					$"plane-reject {Take(ref planeReject)} ({Take(ref planeNs) * 1e-9:F3}s signs), coplanar {Take(ref coplanar)} (sat {Take(ref coplanarSat)}, {Take(ref coplanarNs) * 1e-9:F3}s of which clip {Take(ref coplanarClipNs) * 1e-9:F3}s), sat-reject {Take(ref satReject)}, interval {Take(ref interval)} ({Take(ref intervalNs) * 1e-9:F3}s)");
			}

			internal static void AddPlaneReject()
			{
				Interlocked.Increment(ref planeReject);
			}

			internal static void AddCoplanar()
			{
				Interlocked.Increment(ref coplanar);
			}

			internal static void AddCoplanarSat()
			{
				Interlocked.Increment(ref coplanarSat);
			}

			internal static void AddSatReject()
			{
				Interlocked.Increment(ref satReject);
			}

			internal static void AddInterval()
			{
				Interlocked.Increment(ref interval);
			}

			internal static void AddCoplanarNs(long ns)
			{
				Interlocked.Add(ref coplanarNs, ns);
			}

			internal static void AddCoplanarClipNs(long ns)
			{
				Interlocked.Add(ref coplanarClipNs, ns);
			}

			internal static void AddPlaneNs(long ns)
			{
				Interlocked.Add(ref planeNs, ns);
			}

			internal static void AddIntervalNs(long ns)
			{
				Interlocked.Add(ref intervalNs, ns);
			}

			/// <summary>Rust's <c>|a: &amp;AtomicU64| a.swap(0, Relaxed)</c>.</summary>
			private static long Take(ref long counter)
			{
				return Interlocked.Exchange(ref counter, 0L);
			}
		}

		/// <summary>
		/// Exact intersection of triangles t1 and t2 (each three finite f64
		/// vertices). Symmetric: swapping the arguments yields the same set.
		/// </summary>
		/// <exception cref="ArgumentException">
		/// Either array is not exactly three vertices; see the file header on why the
		/// Rust's <c>[Vec3; 3]</c> lands as a length-checked array.
		/// </exception>
		public static TriTriIsect TriTriIntersect(Vec3[] t1, Vec3[] t2)
		{
			if (t1.Length != 3 || t2.Length != 3)
			{
				throw new ArgumentException(
					$"TriTriIntersect takes two triangles of three vertices, got {t1.Length} and {t2.Length}");
			}

			Timing.Stopwatch tSigns = Timing.Stopwatch.Start();

			// Signs of t2's vertices against t1's plane.
			Sign[] s2 =
			{
				Filtered.Orient3d(t1[0], t1[1], t1[2], t2[0]),
				Filtered.Orient3d(t1[0], t1[1], t1[2], t2[1]),
				Filtered.Orient3d(t1[0], t1[1], t1[2], t2[2]),
			};
			if (AllSameStrict(s2))
			{
				Stats.AddPlaneReject();
				Stats.AddPlaneNs(tSigns.ElapsedNs());
				return TriTriIsect.None;
			}

			if (AllZero(s2))
			{
				Stats.AddCoplanar();
				TriTriIsect coplanarOut = CoplanarOverlap(t1, t2);
				Stats.AddCoplanarNs(tSigns.ElapsedNs());
				return coplanarOut;
			}

			// Signs of t1's vertices against t2's plane.
			Sign[] s1 =
			{
				Filtered.Orient3d(t2[0], t2[1], t2[2], t1[0]),
				Filtered.Orient3d(t2[0], t2[1], t2[2], t1[1]),
				Filtered.Orient3d(t2[0], t2[1], t2[2], t1[2]),
			};
			if (AllSameStrict(s1))
			{
				Stats.AddPlaneReject();
				Stats.AddPlaneNs(tSigns.ElapsedNs());
				return TriTriIsect.None;
			}

			Stats.AddPlaneNs(tSigns.ElapsedNs());
			Debug.Assert(
				!AllZero(s1),
				"t1 coplanar with t2's plane implies t2 coplanar with t1's — handled above");

			// Both triangles straddle each other's plane, but most such box-pair
			// candidates still miss along the common line. A certified
			// separating-axis check on the raw f64 vertices skips the entire
			// rational interval construction for them.
			//
			// (The Rust repacks both triangles into [[f64; 3]; 3] here; the C#
			// SatEdgeAxesDisjoint reads Vec3 directly, so the repack is not needed.)
			if (Approx.SatEdgeAxesDisjoint(t1, t2))
			{
				Stats.AddSatReject();
				return TriTriIsect.None;
			}

			Stats.AddInterval();
			Timing.Stopwatch tInterval = Timing.Stopwatch.Start();

			// Both triangles meet the common line L of the two planes. Overlap the
			// two 1- or 2-point intervals along L entirely in scaled integer
			// arithmetic (no rational constructions, no gcds): per-axis power-of-two
			// scaling maps every vertex to an exact integer, and for points on L the
			// scaled-space parameter dir_s·x_s is a positive multiple of the true
			// parameter dir·x (x−y ∥ dir makes the difference d·|A·dir|²·λ for
			// x−y = λ·dir), so ordering — including its orientation — matches the
			// rational computation exactly. Endpoints stay symbolic; only the 1–2
			// points of the final answer are constructed rationally.
			TriTriIsect intervalOut = IntervalOverlap(t1, t2, s1, s2);
			Stats.AddIntervalNs(tInterval.ElapsedNs());
			return intervalOut;
		}

		/// <summary>
		/// Symbolic interval endpoint on the common line L: an original vertex
		/// exactly on the other plane, or a strictly straddling edge's crossing.
		/// </summary>
		/// <remarks>
		/// The Rust enum has two cases with identical payloads —
		/// <c>Vert(which_tri, vertex index)</c> and <c>Cross(which_tri, edge start
		/// index i)</c> — so the tag collapses to one bool with no loss.
		/// </remarks>
		private readonly struct EndPt
		{
			/// <summary>False for the Rust's <c>Vert</c>, true for its <c>Cross</c>.</summary>
			public readonly bool IsCross;

			/// <summary>Which triangle the endpoint belongs to: 0 or 1.</summary>
			public readonly byte Which;

			/// <summary>
			/// The vertex index, or the start index of the edge running i → (i+1)%3.
			/// </summary>
			public readonly byte Index;

			private EndPt(bool isCross, byte which, byte index)
			{
				this.IsCross = isCross;
				this.Which = which;
				this.Index = index;
			}

			/// <summary>An original vertex lying exactly on the other triangle's plane.</summary>
			public static EndPt Vert(byte which, byte index)
			{
				return new EndPt(false, which, index);
			}

			/// <summary>A strictly straddling edge's crossing of the other plane.</summary>
			public static EndPt Cross(byte which, byte index)
			{
				return new EndPt(true, which, index);
			}
		}

		private static TriTriIsect IntervalOverlap(Vec3[] t1, Vec3[] t2, Sign[] s1, Sign[] s2)
		{
			// Fast path: when a triangle has exactly one vertex ON the other's plane
			// and its remaining vertices strictly on one side, its interval on the
			// common line L is that single vertex. Two degenerate intervals overlap
			// iff the vertices coincide — original f64 vertices are equal as
			// rationals iff equal as f64, so no arithmetic at all. This is the
			// dominant configuration on touching sheets (vertex-to-vertex contacts).
			//
			// Rust's Option<usize> becomes a -1 sentinel; nothing indexes with it.
			static int DegenerateAt(Sign[] s)
			{
				for (int i = 0; i < 3; i++)
				{
					if (s[i] == Sign.Zero && s[(i + 1) % 3] != Sign.Zero && s[(i + 1) % 3] == s[(i + 2) % 3])
					{
						return i;
					}
				}

				return -1;
			}

			int di = DegenerateAt(s1);
			int dj = DegenerateAt(s2);
			if (di >= 0 && dj >= 0)
			{
				// Ties prefer t1's endpoint, matching the general path's lo pick.
				return t1[di] == t2[dj] ? TriTriIsect.Point(R3.FromVec3(t1[di])) : TriTriIsect.None;
			}

			// Scaled integer coordinates; one common scale per axis across BOTH
			// triangles so cross-triangle parameter comparisons share a basis.
			BigInteger[] sx = IntPred.ScaledBig(
				stackalloc double[] { t1[0].X, t1[1].X, t1[2].X, t2[0].X, t2[1].X, t2[2].X });
			BigInteger[] sy = IntPred.ScaledBig(
				stackalloc double[] { t1[0].Y, t1[1].Y, t1[2].Y, t2[0].Y, t2[1].Y, t2[2].Y });
			BigInteger[] sz = IntPred.ScaledBig(
				stackalloc double[] { t1[0].Z, t1[1].Z, t1[2].Z, t2[0].Z, t2[1].Z, t2[2].Z });
			BigInteger[] V(int k)
			{
				return new BigInteger[] { sx[k], sy[k], sz[k] };
			}

			static BigInteger[] Sub(BigInteger[] a, BigInteger[] b)
			{
				return new BigInteger[] { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
			}

			static BigInteger[] Cross(BigInteger[] a, BigInteger[] b)
			{
				return new BigInteger[]
				{
					(a[1] * b[2]) - (a[2] * b[1]),
					(a[2] * b[0]) - (a[0] * b[2]),
					(a[0] * b[1]) - (a[1] * b[0]),
				};
			}

			static BigInteger Dot(BigInteger[] a, BigInteger[] b)
			{
				return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
			}

			BigInteger[] n1 = Cross(Sub(V(1), V(0)), Sub(V(2), V(0)));
			BigInteger[] n2 = Cross(Sub(V(4), V(3)), Sub(V(5), V(3)));
			BigInteger[] dir = Cross(n1, n2);
			Debug.Assert(
				!dir[0].IsZero || !dir[1].IsZero || !dir[2].IsZero,
				"non-coplanar intersecting planes");

			// Parameters dir·v and signed heights against the other triangle's
			// plane, computed lazily: a typical call touches 2–4 of the six
			// vertices, and every skipped dot product is three skipped BigInteger
			// multiplications. Height signs replicate s1/s2 exactly.
			BigInteger Du(int k)
			{
				return Dot(dir, V(k));
			}

			BigInteger H(int k, BigInteger[] n, int origin)
			{
				return Dot(n, V(k)) - Dot(n, V(origin));
			}

#if DEBUG
			for (int i = 0; i < 3; i++)
			{
				Debug.Assert(IntSign(H(i, n2, 3)) == s1[i], "scaled height disagrees with s1");
				Debug.Assert(IntSign(H(3 + i, n1, 0)) == s2[i], "scaled height disagrees with s2");
			}
#endif

			// The ≤2 endpoints of one triangle's crossing with the other's plane, as
			// (unreduced parameter fraction, symbolic point), in the same
			// enumeration order as the rational implementation used (vertices in
			// index order, then edges (0,1), (1,2), (2,0)) so ties break alike.
			List<(Frac T, EndPt Pt)> Endpoints(byte which, Sign[] s)
			{
				int baseIndex = which == 0 ? 0 : 3;
				BigInteger[] n = which == 0 ? n2 : n1;
				int origin = which == 0 ? 3 : 0;
				List<(Frac T, EndPt Pt)> pts = new List<(Frac, EndPt)>(2);
				for (int i = 0; i < 3; i++)
				{
					if (s[i] == Sign.Zero)
					{
						pts.Add(((Du(baseIndex + i), BigInteger.One), EndPt.Vert(which, (byte)i)));
					}
				}

				for (int i = 0; i < 3; i++)
				{
					int j = (i + 1) % 3;
					if (s[i] != Sign.Zero && s[j] != Sign.Zero && s[i] != s[j])
					{
						// x = u + h_u/(h_u−h_v)·(v−u) ⇒
						// dir·x = [(h_u−h_v)·du_u + h_u·(du_v−du_u)] / (h_u−h_v).
						BigInteger hu = H(baseIndex + i, n, origin);
						BigInteger hv = H(baseIndex + j, n, origin);
						BigInteger duU = Du(baseIndex + i);
						BigInteger duV = Du(baseIndex + j);
						BigInteger den = hu - hv;
						BigInteger num = (den * duU) + (hu * (duV - duU));
						if (den.Sign < 0)
						{
							den = -den;
							num = -num;
						}

						pts.Add(((num, den), EndPt.Cross(which, (byte)i)));
					}
				}

				Debug.Assert(pts.Count > 0 && pts.Count <= 2, "an endpoint list holds one or two points");
				return pts;
			}

			List<(Frac T, EndPt Pt)> pts1 = Endpoints(0, s1);
			List<(Frac T, EndPt Pt)> pts2 = Endpoints(1, s2);

			// Per-triangle interval, first-encountered point winning ties (matching
			// the old interval_along).
			static ((Frac T, EndPt Pt) Lo, (Frac T, EndPt Pt) Hi) MinMax(List<(Frac T, EndPt Pt)> pts)
			{
				(Frac T, EndPt Pt) lo = pts[0];
				(Frac T, EndPt Pt) hi = pts[0];
				for (int k = 1; k < pts.Count; k++)
				{
					(Frac T, EndPt Pt) p = pts[k];
					if (CmpFrac(p.T, lo.T) < 0)
					{
						lo = p;
					}

					if (CmpFrac(p.T, hi.T) > 0)
					{
						hi = p;
					}
				}

				return (lo, hi);
			}

			((Frac T, EndPt Pt) Lo, (Frac T, EndPt Pt) Hi) i1 = MinMax(pts1);
			((Frac T, EndPt Pt) Lo, (Frac T, EndPt Pt) Hi) i2 = MinMax(pts2);
			(Frac lo, EndPt loPt) = CmpFrac(i1.Lo.T, i2.Lo.T) >= 0 ? i1.Lo : i2.Lo;
			(Frac hi, EndPt hiPt) = CmpFrac(i1.Hi.T, i2.Hi.T) <= 0 ? i1.Hi : i2.Hi;

			int order = CmpFrac(lo, hi);
			if (order > 0)
			{
				return TriTriIsect.None;
			}
			else if (order == 0)
			{
				return TriTriIsect.Point(BuildEndpoint(loPt, t1, t2));
			}
			else
			{
				return TriTriIsect.Segment(
					BuildEndpoint(loPt, t1, t2),
					BuildEndpoint(hiPt, t1, t2));
			}
		}

#if DEBUG
		private static Sign IntSign(BigInteger v)
		{
			if (v.IsZero)
			{
				return Sign.Zero;
			}
			else if (v.Sign < 0)
			{
				return Sign.Neg;
			}
			else
			{
				return Sign.Pos;
			}
		}
#endif

		/// <summary>
		/// Materialize a symbolic interval endpoint as the exact rational point the
		/// fully rational implementation would have produced.
		/// </summary>
		private static R3 BuildEndpoint(EndPt e, Vec3[] t1, Vec3[] t2)
		{
			Vec3[] Tri(int which)
			{
				return which == 0 ? t1 : t2;
			}

			if (!e.IsCross)
			{
				return R3.FromVec3(Tri(e.Which)[e.Index]);
			}

			Vec3[] own = Tri(e.Which);
			Vec3[] other = Tri(1 - e.Which);
			R3 a = R3.FromVec3(own[e.Index]);
			R3 b = R3.FromVec3(own[(e.Index + 1) % 3]);
			R3[] p =
			{
				R3.FromVec3(other[0]),
				R3.FromVec3(other[1]),
				R3.FromVec3(other[2]),
			};
			R3? x = Predicates.LinePlaneIntersect(a, b, p[0], p[1], p[2]);
			if (x is null)
			{
				throw new InvalidOperationException(
					"strictly straddling edge cannot be parallel to the plane");
			}

			return x;
		}

		private static int CmpFrac(Frac a, Frac b)
		{
			// Denominators positive → cross-multiplication preserves order.
			return (a.Num * b.Den).CompareTo(b.Num * a.Den);
		}

		private static bool AllSameStrict(Sign[] s)
		{
			return s[0] != Sign.Zero && s[0] == s[1] && s[1] == s[2];
		}

		/// <summary>Rust's <c>s.iter().all(|s| *s == Sign::Zero)</c>.</summary>
		private static bool AllZero(Sign[] s)
		{
			return s[0] == Sign.Zero && s[1] == Sign.Zero && s[2] == Sign.Zero;
		}
	}
}
