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

// Arrangement.cs — port of robust/arrangement.rs, whose header reads:
//
//   Per-triangle 2D arrangement (paper §6.3–6.4).
//
//   Each input triangle that intersects triangles of the other operand gets an
//   arrangement: the exact intersection primitives that landed on it (points,
//   segments, coplanar-overlap polygon boundaries) are projected into the
//   triangle's dominant-axis plane, split at their mutual crossings, and handed
//   to robust/cdt.rs as constraints. The result subdivides the triangle into
//   sub-triangles whose edges preserve every intersection segment, with a
//   provenance map from constraint edges back to the primitives that created
//   them — the edge→triangles incidence the classification stage
//   (robust/classify.rs) walks.
//
//   Everything here is exact: projection is coordinate dropping (bijective on
//   the triangle's plane), crossings are rational constructions, and identity
//   is rational equality — no tolerances anywhere.
//
// The Rust file splits across three C# files, one type each side of the seam:
// this one (the vocabulary types, the phase timers, the exact-point sweeps'
// scalar helpers and `Build`), Arrangement.Sweeps.cs (the conservative box
// type, the interval sweep that enumerates overlapping box pairs, and the
// x-sorted point index) and Arrangement.Candidates.cs (`CandidatePoints`).
//
// ── Naming ───────────────────────────────────────────────────────────────────
// The module's free functions land in `ArrangementFunctions`, with the
// `Functions` suffix the porting plan prescribes when the bare module name is
// already the primary type's — exactly the `CdtFunctions` / `Cdt` situation.
//
// ── `usize` provenance ids ───────────────────────────────────────────────────
// Provenance is a caller-defined id (the robust pipeline uses a pair index), so
// it ports as `int` like every other index in this port. The one place the Rust
// puts a *sentinel* in that slot is intersection_graph.rs's registry-supplied
// split points, which carry `usize::MAX`; here that is `int.MaxValue`. Nothing
// reads a point's provenance — only segment provenance reaches
// `Arrangement.Constraints` — so the sentinel is never compared against a real
// pair index and the narrower type cannot collide.
//
// ── `[f64; 2]` is a Vec2 ─────────────────────────────────────────────────────
// The Rust passes filter inputs around as bare `[f64; 2]`; this port's exact
// tier already settled on Vec2 for that shape (Approx.Orient2dA), and
// robust/cdt.rs's `apts` is a `Vec2[]` for the same reason.

using System.Diagnostics;
using System.Globalization;
using System.Numerics;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Robust
{
	/// <summary>
	/// Intersection primitives to arrange on one triangle, each tagged with a
	/// caller-defined provenance id (the robust pipeline uses the index of the
	/// opposing triangle pair that produced it).
	/// </summary>
	/// <remarks>
	/// A class, not a struct, because the Rust derives <c>Default</c> and the pipeline
	/// builds one per triangle and hands out a reference; <c>new ArrangementInput()</c>
	/// is Rust's <c>ArrangementInput::default()</c>.
	/// </remarks>
	public sealed class ArrangementInput
	{
		/// <summary>Isolated contact points (vertex touches, edge-through-edge points).</summary>
		public List<(R3 Point, int Prov)> Points = new List<(R3 Point, int Prov)>();

		/// <summary>
		/// Intersection segments, including coplanar-overlap polygon edges.
		/// Endpoints must be distinct and lie inside or on the triangle.
		/// </summary>
		public List<(R3 A, R3 B, int Prov)> Segments = new List<(R3 A, R3 B, int Prov)>();
	}

	/// <summary>The subdivided triangle.</summary>
	public sealed class Arrangement
	{
		/// <summary>Dropped coordinate of the projection (0=x, 1=y, 2=z).</summary>
		public int Axis;

		/// <summary>Exact 3D points; indices 0..3 are the triangle corners in input order.</summary>
		public List<R3> Points3 = new List<R3>();

		/// <summary>Their exact 2D projections (same indexing).</summary>
		public List<R2> Points2 = new List<R2>();

		/// <summary>Sub-triangles (indices into points*), CCW in projection space.</summary>
		public List<IVec3> Tris = new List<IVec3>();

		/// <summary>
		/// Constraint edges as (min,max) index pairs → provenance ids of every
		/// primitive that generated the edge.
		/// </summary>
		/// <remarks>
		/// Rust's <c>BTreeMap</c>: a <see cref="SortedDictionary{TKey, TValue}"/>, never a
		/// <c>Dictionary</c>. The build pipeline iterates these keys to seed
		/// <c>IntersectionGraph.IsectEdges</c>, and the ported tests read
		/// <c>constraints.values()</c> in key order, so the ordering is load-bearing.
		/// </remarks>
		public SortedDictionary<(int A, int B), List<int>> Constraints
			= new SortedDictionary<(int A, int B), List<int>>();

		/// <summary>
		/// True when the 2D projection mirrors the triangle's 3D winding (the
		/// dropped normal component is negative): consumers must swap two
		/// indices of each sub-triangle to recover outward orientation.
		/// </summary>
		public bool Flipped;
	}

	/// <summary>
	/// The free functions of <c>robust/arrangement.rs</c>: building the per-triangle
	/// exact 2D arrangement, and the candidate-point pre-pass the intersection-graph
	/// build runs before it. Named with the <c>Functions</c> suffix because the
	/// module's own primary type, <see cref="Arrangement"/>, already owns the bare name.
	/// </summary>
	public static partial class ArrangementFunctions
	{
		/// <summary>The smallest positive normal f64 (Rust's <c>f64::MIN_POSITIVE</c>).</summary>
		private const double MinPositive = 2.2250738585072014E-308;

		/// <summary>
		/// Aggregate phase timers for <see cref="Build"/>, printed under MANIFOLD_TIMING by
		/// the pipeline after the arrangements stage. Relaxed atomics, nanoseconds.
		/// </summary>
		/// <remarks>
		/// The Rust exposes the <c>AtomicU64</c> statics directly and every mutation is a
		/// <c>fetch_add</c> from inside this module. Public mutable statics have no good C#
		/// spelling, so — exactly as <see cref="TriTri.Stats"/> does — the counters are
		/// private and the only public operation is <see cref="SnapshotAndReset"/>, which is
		/// all any caller (intersection_graph.rs) ever does with them.
		/// </remarks>
		public static class Stats
		{
			private static long setupNs;
			private static long normNs;
			private static long crossNs;
			private static long onSegNs;
			private static long cdtNs;
			private static long calls;
			private static long segs;
			private static long pts;

			/// <summary>
			/// Reads and zeroes every counter, formatting them as the Rust's one-line summary.
			/// </summary>
			/// <remarks>
			/// The format string is transcribed specifier for specifier from
			/// <c>stats::snapshot_and_reset</c> so that trace diffs against the Rust show only
			/// real differences. Rust's <c>{:.3}</c> is C#'s <c>F3</c>; the arguments are
			/// evaluated left to right on both sides, which matters because reading a counter
			/// also zeroes it.
			/// <para>
			/// The two <c>.3</c> formatters disagree on exact midpoints — Rust rounds half to
			/// even, .NET half away from zero — so a nanosecond count landing exactly on a
			/// half-millisecond would print one ulp apart. This is a DISPLAY line only (the
			/// counters themselves never reach a result), and no wall-clock reading hits an
			/// exact midpoint meaningfully; the same note is on TriTri.Stats.
			/// </para>
			/// </remarks>
			/// <returns>The formatted counter line.</returns>
			public static string SnapshotAndReset()
			{
				return string.Create(
					CultureInfo.InvariantCulture,
					$"setup {Take(ref setupNs) * 1e-9:F3}s (norm {Take(ref normNs) * 1e-9:F3}s), crossings {Take(ref crossNs) * 1e-9:F3}s, on-seg {Take(ref onSegNs) * 1e-9:F3}s, cdt {Take(ref cdtNs) * 1e-9:F3}s ({Take(ref calls)} calls, {Take(ref segs)} segs, {Take(ref pts)} pts)");
			}

			internal static void AddSetupNs(long ns)
			{
				Interlocked.Add(ref setupNs, ns);
			}

			internal static void AddNormNs(long ns)
			{
				Interlocked.Add(ref normNs, ns);
			}

			internal static void AddCrossNs(long ns)
			{
				Interlocked.Add(ref crossNs, ns);
			}

			internal static void AddOnSegNs(long ns)
			{
				Interlocked.Add(ref onSegNs, ns);
			}

			internal static void AddCdtNs(long ns)
			{
				Interlocked.Add(ref cdtNs, ns);
			}

			internal static void AddCalls(long n)
			{
				Interlocked.Add(ref calls, n);
			}

			internal static void AddSegs(long n)
			{
				Interlocked.Add(ref segs, n);
			}

			internal static void AddPts(long n)
			{
				Interlocked.Add(ref pts, n);
			}

			/// <summary>Rust's <c>|a: &amp;AtomicU64| a.swap(0, Relaxed)</c>.</summary>
			private static long Take(ref long counter)
			{
				return Interlocked.Exchange(ref counter, 0L);
			}
		}

		/// <summary>
		/// Build the arrangement of <paramref name="input"/> on triangle
		/// <paramref name="tri"/>. Primitives must already be clipped to the triangle
		/// (tri_tri output is), with rational coordinates on its plane.
		/// </summary>
		/// <remarks>
		/// Returns null only when <paramref name="token"/> is cancelled: a single
		/// segment-heavy triangle can spend minutes in the quadratic sweeps below, so the
		/// per-phase checks in build_graph alone leave a cancel unanswered for however long
		/// the worst triangle takes.
		/// </remarks>
		/// <param name="tri">The triangle to subdivide, three vertices.</param>
		/// <param name="input">The primitives that landed on it.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The arrangement, or null when cancelled.</returns>
		/// <exception cref="ArgumentException">The array is not exactly three vertices.</exception>
		public static Arrangement? Build(Vec3[] tri, ArrangementInput input, CancelToken? token)
		{
			ArgumentNullException.ThrowIfNull(tri);
			ArgumentNullException.ThrowIfNull(input);
			if (tri.Length != 3)
			{
				throw new ArgumentException($"a triangle is three vertices, got {tri.Length}", nameof(tri));
			}

			Timing.Stopwatch t0 = Timing.Stopwatch.Start();
			Stats.AddCalls(1);
			Stats.AddSegs(input.Segments.Count);
			Stats.AddPts(input.Points.Count);
			R3[] corners = new R3[]
			{
				R3.FromVec3(tri[0]),
				R3.FromVec3(tri[1]),
				R3.FromVec3(tri[2]),
			};
			R3 normal = Predicates.TriNormalR(corners[0], corners[1], corners[2]);
			Debug.Assert(!normal.IsZero(), "degenerate triangle in arrangement");
			int axis = TriTri.DominantAxis(normal);
			Stats.AddNormNs(t0.ElapsedNs());

			List<R3> points3 = new List<R3>();
			List<R2> points2 = new List<R2>();

			// Hash-keyed dedup via R2Key: division-free structural hashing of the
			// canonical rationals (a general rational Hash must tolerate unreduced
			// values, and its cross-multiplication dominated this function). Indices
			// are assigned in insertion order, so output stays deterministic.
			// Fx hashing (unseeded, deterministic): the index is probe-only and never
			// iterated, so only insertion order — which is the caller's order — can
			// reach `points3`/`points2`.
			// (Plain Dictionary here per docs/PORTING_PLAN.md's dependency table; the
			// probe-only invariant above is exactly what makes that swap sound.)
			Dictionary<R2Key, int> index = new Dictionary<R2Key, int>();

			foreach (R3 c in corners)
			{
				AddPoint(c);
			}

			Debug.Assert(points3.Count == 3, "corner points must be distinct");

			// Register primitive endpoints / isolated points. (Rust declares a local
			// `struct Seg`; C# has no local struct declaration, and the three fields
			// transcribe exactly as a named value tuple.)
			List<(int A, int B, int Prov)> segs = new List<(int A, int B, int Prov)>(input.Segments.Count);
			foreach ((R3 a3, R3 b3, int prov) in input.Segments)
			{
				Debug.Assert(!a3.Equals(b3), "zero-length segment primitive");
				int a = AddPoint(a3);
				int b = AddPoint(b3);

				// Rust's second check is a bare `debug_assert_ne!(a, b)` with no message:
				// distinct endpoints must also dedup to distinct indices, which is a
				// statement about R2Key rather than about the caller's input.
				Debug.Assert(a != b, "distinct endpoints collided in the point index");
				segs.Add((a, b, prov));
			}

			foreach ((R3 p3, int _) in input.Points)
			{
				AddPoint(p3);
			}

			Stats.AddSetupNs(t0.ElapsedNs());
			t0 = Timing.Stopwatch.Start();

			// Mutual proper crossings between segments become new points. Points are
			// homogenized once (Homog2) for the exact fallback, and approximated
			// once (correctly rounded f64) for the semi-static filters that certify
			// generic-position signs without any Int work.
			List<Homog2> homogs = new List<Homog2>(points2.Count);
			foreach (R2 p in points2)
			{
				homogs.Add(Predicates.Homog2Of(p));
			}

			// Approximations are taken of the points TRANSLATED by points2[0] (the
			// triangle's first corner, so the origin is a deterministic function of
			// the arrangement's own data), with the subtraction done EXACTLY in
			// rationals and only the result rounded — the same lever robust/cdt.rs
			// applies to its filter inputs, and sound here for the same two reasons:
			//
			//  (a) Predicates. Every filtered call below is orient2d_a, whose sign is
			//      translation invariant, with the exact fallback (orient2d_h) still
			//      running on the UNTRANSLATED homogenized rationals. A sign certified
			//      for the translated configuration is therefore the sign of the
			//      original one, while the filter's error bound — which is built from
			//      the coordinate magnitudes |a|+|b|+|c|, not from the determinant —
			//      shrinks from "distance to the world origin" to "extent of this one
			//      triangle". Far-from-origin arrangements stop escalating on every
			//      call.
			//
			//  (b) Box tests. `ApproxBox`/`BoxesOverlap`/`BoxContains` are NOT
			//      translation invariant: translating changes the rounding, so the
			//      sweep and the point-index query may admit a DIFFERENT superset of
			//      candidates. That is sound here because both prefilters remain
			//      conservative about the translated exact points (the pad is ~4.5 ulp
			//      of the translated magnitude, and rounding error is ≤ 0.5 ulp of it),
			//      and because every admitted candidate is then fully decided by exact
			//      predicates: a spurious pair fails the strict-crossing test and
			//      constructs nothing, a spurious point fails orient2d_h/the exact
			//      range test and never enters `onSeg`. Enumeration order is unchanged
			//      too: pairs still arrive in (i, ascending j) order, so the surviving
			//      true crossings are constructed in the same sequence regardless of
			//      which spurious pairs sit between them, and `onSeg` is ordered by an
			//      exact rational parameter comparison over a set of distinct points,
			//      which is independent of insertion order. Nothing that DEFINES output
			//      order — point insertion indices, the constraints SortedDictionary
			//      keys, the `Flipped` rational sign, or CdtFunctions' untranslated
			//      `points2` input — reads `apts` at all.
			//
			// The translation is computed from the homogeneous coordinates
			// (`TranslatedApprox`), which is the same exact value an R2 subtraction
			// would produce but skips the gcd that constructing the difference as a
			// canonical Rational costs — on million-point arrangements that gcd cost
			// more than the escalations the lever saves.
			List<Vec2> apts = new List<Vec2>(points2.Count);

			// Per-point "this f64 pair is EXACTLY the translated point", the
			// precondition of cdt's tight exact-input filters. Handed to the CDT below
			// so it does not repeat the translation.
			List<bool> aptsExact = new List<bool>(points2.Count);

			// A monster arrangement translates over a million points here, each one a
			// pair of exact big-integer divisions, so this loop is long enough that a
			// cancel arriving inside it must not wait it out. Strictly an early
			// return: a run that completes builds the identical vectors.
			for (int i = 0; i < points2.Count; i++)
			{
				if (i % 1024 == 0 && Cancel.IsCancelled(token))
				{
					return null;
				}

				(Vec2 a, bool e) = TranslatedApprox(points2[0], points2[i]);
				apts.Add(a);
				aptsExact.Add(e);
			}

			Box2[] segBoxes = new Box2[segs.Count];
			for (int i = 0; i < segs.Count; i++)
			{
				segBoxes[i] = ApproxBox([apts[segs[i].A], apts[segs[i].B]]);
			}

			// A strict crossing lies in both exact boxes, which sit inside these
			// inflated ones, so the box prefilter is conservative; the sweep emits
			// the surviving pairs in the same (i, ascending j) order the old nested
			// loops used, keeping point construction order identical.
			BoxPairs? pairs = OverlappingBoxPairs(segBoxes, token);
			if (pairs is null)
			{
				return null;
			}

			int k = 0;
			foreach ((int i, int j) in pairs.Iter())
			{
				if (k % 1024 == 0 && Cancel.IsCancelled(token))
				{
					return null;
				}

				k++;
				(int ia, int ib) = (segs[i].A, segs[i].B);
				(int ic, int id) = (segs[j].A, segs[j].B);
				Sign sc = O2(ia, ib, ic);
				Sign sd = O2(ia, ib, id);
				Sign sa = O2(ic, id, ia);
				Sign sb = O2(ic, id, ib);

				// Strict crossing only — endpoint contacts and collinear overlap
				// are handled by the point-on-segment sweep below.
				if (sc != Sign.Zero
					&& sd != Sign.Zero
					&& sc != sd
					&& sa != Sign.Zero
					&& sb != Sign.Zero
					&& sa != sb)
				{
					R2 x2 = Predicates.LineLineIntersect2d(
						points2[segs[i].A],
						points2[segs[i].B],
						points2[segs[j].A],
						points2[segs[j].B])
						?? throw new InvalidOperationException("properly crossing segments are not parallel");
					R3 x3 = TriTri.LiftToPlane(x2, axis, corners[0], normal);
					AddPoint(x3);
				}
			}

			// Points added by crossings need homogs and approximations too.
			for (int i = homogs.Count; i < points2.Count; i++)
			{
				homogs.Add(Predicates.Homog2Of(points2[i]));
			}

			for (int i = apts.Count; i < points2.Count; i++)
			{
				if (i % 1024 == 0 && Cancel.IsCancelled(token))
				{
					return null;
				}

				(Vec2 a, bool e) = TranslatedApprox(points2[0], points2[i]);
				apts.Add(a);
				aptsExact.Add(e);
			}

			Debug.Assert(
				points2.TrueForAll(p =>
					Predicates.PointInTri2d(p, points2[0], points2[1], points2[2]) != TriLoc.Outside),
				"arrangement primitive escapes its triangle");

			Stats.AddCrossNs(t0.ElapsedNs());
			t0 = Timing.Stopwatch.Start();

			// Subdivide each segment at every registered point lying exactly on it;
			// consecutive point pairs become constraint edges carrying provenance.
			SortedDictionary<(int A, int B), List<int>> constraints
				= new SortedDictionary<(int A, int B), List<int>>();
			PointIndex pindex = PointIndex.Build(apts);
			List<int> cand = new List<int>();
			for (int si = 0; si < segs.Count; si++)
			{
				if (Cancel.IsCancelled(token))
				{
					return null;
				}

				(int A, int B, int Prov) seg = segs[si];
				Box2 sbox = segBoxes[si];
				Homog2 ha = homogs[seg.A];
				Homog2 hb = homogs[seg.B];

				// Segment direction cleared of denominators: v = (pb−pa)·(Bw·Aw).
				BigInteger vx = (hb.X * ha.W) - (ha.X * hb.W);
				BigInteger vy = (hb.Y * ha.W) - (ha.Y * hb.W);
				BigInteger vv = (vx * vx) + (vy * vy);

				// Parameters as unreduced fractions: for point P (homog (Px,Py,Pw)),
				// u = (p−pa)·(Pw·Aw) and t ∝ (u·v)/Pw — ordered and range-checked by
				// cross-multiplication with the positive denominators, no canonical
				// rationals anywhere.
				List<(BigInteger Uv, BigInteger Pw, int Idx)> onSeg
					= new List<(BigInteger Uv, BigInteger Pw, int Idx)>();

				// A point on the segment lies in its exact box, so the inflated box
				// test cannot reject a true hit; the range query returns exactly the
				// points that pass it, re-sorted into the ascending index order the
				// old full rescan visited.
				pindex.Query(apts, sbox, cand);
				foreach (int idx in cand)
				{
					Homog2 hp = homogs[idx];

					// Approx filter next: almost every remaining point is certifiably
					// off the segment's line; only near-collinear candidates pay for
					// Int.
					if (Approx.Orient2dA(apts[seg.A], apts[seg.B], apts[idx]).HasValue)
					{
						continue; // certified nonzero → not collinear
					}

					if (Predicates.Orient2dH(ha, hb, hp) != Sign.Zero)
					{
						continue;
					}

					BigInteger ux = (hp.X * ha.W) - (ha.X * hp.W);
					BigInteger uy = (hp.Y * ha.W) - (ha.Y * hp.W);
					BigInteger uv = (ux * vx) + (uy * vy);

					// 0 ≤ t  ⇔  0 ≤ u·v;   t ≤ |dir|²  ⇔  (u·v)·Bw ≤ (v·v)·Pw.
					if (uv.Sign >= 0 && uv * hb.W <= vv * hp.W)
					{
						onSeg.Add((uv, hp.W, idx));
					}
				}

				// t_i < t_j  ⇔  uv_i·Pw_j < uv_j·Pw_i (denominators positive).
				// OrderBy, not List.Sort: Rust's `sort_by` is stable and this comparator
				// is not a total order (equal parameters compare equal), so an introsort
				// could permute coincident points and change the constraint edges.
				onSeg = onSeg
					.OrderBy(
						x => x,
						Comparer<(BigInteger Uv, BigInteger Pw, int Idx)>.Create(
							(a, b) => (a.Uv * b.Pw).CompareTo(b.Uv * a.Pw)))
					.ToList();
				Debug.Assert(onSeg.Count >= 2, "segment lost its own endpoints");
				for (int w = 0; w + 1 < onSeg.Count; w++)
				{
					(int u, int v) = (onSeg[w].Idx, onSeg[w + 1].Idx);
					Debug.Assert(u != v, "duplicate points on segment");
					(int A, int B) key = (Math.Min(u, v), Math.Max(u, v));
					if (!constraints.TryGetValue(key, out List<int>? provs))
					{
						provs = new List<int>();
						constraints.Add(key, provs);
					}

					if (!provs.Contains(seg.Prov))
					{
						provs.Add(seg.Prov);
					}
				}
			}

			Stats.AddOnSegNs(t0.ElapsedNs());
			t0 = Timing.Stopwatch.Start();

			List<(int A, int B)> constraintPairs = new List<(int A, int B)>(constraints.Keys);

			// The CDT translates by points2[0] for its own filters, which is exactly
			// what `apts`/`aptsExact` already hold — hand them over rather than let it
			// redo one exact subtraction per point. The token rides along on the same
			// call: a monster triangulation must stay interruptible.
			//
			// Seam note: CdtFunctions.TriangulateWithApts takes arrays where the Rust
			// moves its Vecs, so the two ToArray() calls are this port's cost for the
			// same hand-over. They copy doubles and bools, never rationals, and they
			// replace the copy the Rust's `Vec -> &[..]` reborrow gets for free.
			List<IVec3>? tris = CdtFunctions.TriangulateWithApts(
				points2,
				constraintPairs,
				apts.ToArray(),
				aptsExact.ToArray(),
				token);
			if (tris is null)
			{
				return null;
			}

			Stats.AddCdtNs(t0.ElapsedNs());

			BigRational axisComp = axis switch
			{
				0 => normal.X,
				1 => normal.Y,
				_ => normal.Z,
			};
			bool flipped = SignFunctions.OfRat(axisComp) == Sign.Neg;

			return new Arrangement
			{
				Axis = axis,
				Points3 = points3,
				Points2 = points2,
				Tris = tris,
				Constraints = constraints,
				Flipped = flipped,
			};

			// Rust's `add_point` closure; C# captures `points3`/`points2` directly where
			// the Rust has to pass them in to satisfy the borrow checker.
			int AddPoint(R3 p3)
			{
				R2 p2 = p3.ProjectDrop(axis);
				int next = points3.Count;
				R2Key key = new R2Key(p2);
				if (index.TryGetValue(key, out int hit))
				{
					return hit;
				}

				index.Add(key, next);
				points3.Add(p3);
				points2.Add(p2);
				return next;
			}

			Sign O2(int i, int j, int k2)
			{
				return Approx.Orient2dA(apts[i], apts[j], apts[k2])
					?? Predicates.Orient2dH(homogs[i], homogs[j], homogs[k2]);
			}
		}
	}
}
