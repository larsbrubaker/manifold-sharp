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

// Cdt.cs — port of robust/cdt.rs, whose header reads:
//
//   Constrained Delaunay triangulation on exact 2D points.
//
//   Triangulates the interior of one input triangle after robust/arrangement.rs
//   has collected every intersection point and constraint segment that lands on
//   it (paper §6.4). All predicates are exact (robust/exact), so there are no
//   epsilon decisions anywhere: point location, edge flips, and constraint
//   recovery all reason on true signs. Delaunay-ness itself is not load-bearing
//   for the boolean — validity and constraint preservation are — but Delaunay
//   flips avoid the near-degenerate sub-triangles that would otherwise stress
//   downstream float code.
//
//   Scale note: an arrangement usually holds one mesh triangle's intersections
//   and its point count is small (tens), but heavily self-intersecting
//   Thingi10K meshes routinely push thousands — and occasionally >10⁴ — points
//   through a single call, so nothing here may be O(n) per insertion. Point
//   location is a visibility walk seeded from the newest triangle,
//   `MarkIfPresent`/`CollectCorridor` rotate around a vertex via `vertTri`,
//   and `FixSplitPairs` looks only at the halves it just created. What
//   remains superlinear is per-operation and bounded by local complexity
//   (corridor length, pseudo-polygon chain length), not by the arena size.
//
// The Rust file splits across three C# files, all one type: this one (the
// module's free functions, the arena's declaration and the filtered
// predicates), Cdt.Arena.cs (navigating and editing the arena: vertex fans,
// point location, splits and Lawson legalization) and Cdt.Constraints.cs (port
// of robust/cdt_constraints.rs, itself a second `impl Cdt` block in the Rust).
//
// ── Arena representation ─────────────────────────────────────────────────────
// The Rust's `Vec<Tri>` is a growable *array* of structs here, not a
// `List<Tri>`: the code mutates single fields of `tris[i]` in a dozen places,
// which `List<T>` of a struct cannot do (CS1612), and `CollectionsMarshal.AsSpan`
// is unusable because every operation appends while walking. An array element is
// a variable, so `this.tris[t].Adj[e] = n` writes in place with no copy and no
// per-triangle allocation — the "arrays + int indices, no object graphs" rule
// from the porting plan, which is also what makes >10⁴-point arrangements
// affordable. `triCount` is the Rust's `tris.len()`.
//
// Rust's `[usize; 3]` vertex/adjacency triples become `IVec3` (int triple with a
// by-index accessor); every index here is an arena or point index, far below
// 2³¹. The `[bool; 3]` constraint flags become `Bool3`, the same shape for bools.

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Robust
{
	/// <summary>
	/// The free functions of <c>robust/cdt.rs</c>: constrained Delaunay triangulation
	/// of the exact points of one arrangement. Named with the <c>Functions</c> suffix
	/// because the module's own primary type, <see cref="Cdt"/>, already owns the bare
	/// name.
	/// </summary>
	public static class CdtFunctions
	{
		/// <summary>
		/// 2⁵³, hoisted so <see cref="IsExact"/> stays allocation-free as its ported
		/// comment claims — <c>BigInteger.One &lt;&lt; 53</c> builds a new value on every
		/// call.
		/// </summary>
		private static readonly BigInteger TwoPow53 = BigInteger.One << 53;

		/// <summary>
		/// Constrained Delaunay triangulation of <paramref name="points"/> inside the
		/// triangle formed by <c>points[0..3]</c> (any winding). Every other point must
		/// lie inside that triangle or on its boundary, all points must be pairwise
		/// distinct, and every constraint (index pair) must be free of interior points
		/// and proper crossings with other constraints — robust/arrangement.rs
		/// guarantees all three. Returns CCW index triangles exactly covering the input
		/// triangle.
		/// </summary>
		/// <param name="points">The exact points; the first three are the base triangle.</param>
		/// <param name="constraints">Index pairs that must appear as edges.</param>
		/// <returns>CCW index triples covering the base triangle.</returns>
		public static List<IVec3> Triangulate(IReadOnlyList<R2> points, IReadOnlyList<(int A, int B)> constraints)
		{
			return TriangulateWithToken(points, constraints, null)
				?? throw new InvalidOperationException("uncancellable triangulate cannot cancel");
		}

		/// <summary>
		/// <see cref="Triangulate"/> with cooperative cancellation. Returns null when the
		/// token fires.
		/// </summary>
		/// <remarks>
		/// A single monster arrangement pushes &gt;10⁶ points and constraints through one
		/// call and can run for minutes, so the per-arrangement check in
		/// robust/intersection_graph.rs is not enough on its own: a cancel that only the
		/// enclosing map notices waits out the worst triangulation in the mesh. The checks
		/// are strictly early returns — they never touch the insertion order, the arena, or
		/// any predicate — so a run that completes produces exactly the triangles
		/// <see cref="Triangulate"/> produces.
		/// </remarks>
		/// <param name="points">The exact points; the first three are the base triangle.</param>
		/// <param name="constraints">Index pairs that must appear as edges.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>CCW index triples, or null when cancelled.</returns>
		public static List<IVec3>? TriangulateWithToken(
			IReadOnlyList<R2> points,
			IReadOnlyList<(int A, int B)> constraints,
			CancelToken? token)
		{
			(Vec2[] apts, bool[] exact) = TranslatedFilterInputs(points);
			return TriangulateWithApts(points, constraints, apts, exact, token);
		}

		/// <summary>
		/// The translated filter inputs <see cref="TriangulateWithToken"/> would build for
		/// <paramref name="points"/>: each point minus <c>points[0]</c>, the subtraction
		/// exact in rationals and only the result rounded, plus a flag per point saying
		/// whether that f64 is EXACTLY the translated value (see
		/// <see cref="TriangulateWithApts"/> for why translating is sound and why it
		/// matters).
		/// </summary>
		/// <param name="points">The exact points.</param>
		/// <returns>The rounded translated points and their exactness flags.</returns>
		internal static (Vec2[] Apts, bool[] Exact) TranslatedFilterInputs(IReadOnlyList<R2> points)
		{
			ArgumentNullException.ThrowIfNull(points);
			R2 origin = points[0];
			Vec2[] apts = new Vec2[points.Count];
			bool[] exact = new bool[points.Count];
			for (int i = 0; i < points.Count; i++)
			{
				R2 t = points[i].Sub(origin);
				apts[i] = new Vec2(Rational.RatToF64(t.X), Rational.RatToF64(t.Y));
				exact[i] = IsExact(t.X) && IsExact(t.Y);
			}

			return (apts, exact);
		}

		/// <summary>
		/// <see cref="TriangulateWithToken"/> with the translated filter inputs supplied by
		/// the caller, so the per-point exact translation is paid once per arrangement
		/// instead of twice — robust/arrangement.rs derives exactly these values, from
		/// exactly this origin, for its own filtered sweeps.
		/// </summary>
		/// <remarks>
		/// <c>apts[i]</c> must be a correctly rounded f64 pair of
		/// <c>points[i] − points[0]</c> and <c>exact[i]</c> must be true only when that
		/// pair is EXACTLY that difference. Nothing else may be passed: the semi-static
		/// filters are sound for any inputs meeting those two conditions and unsound for
		/// any others. The exact fallbacks are unaffected either way — they run on
		/// <paramref name="points"/>' own homogenized rationals, untranslated.
		/// </remarks>
		/// <param name="points">The exact points; the first three are the base triangle.</param>
		/// <param name="constraints">Index pairs that must appear as edges.</param>
		/// <param name="apts">The rounded translated points.</param>
		/// <param name="exact">Per-point exactness of <paramref name="apts"/>.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>CCW index triples, or null when cancelled.</returns>
		internal static List<IVec3>? TriangulateWithApts(
			IReadOnlyList<R2> points,
			IReadOnlyList<(int A, int B)> constraints,
			Vec2[] apts,
			bool[] exact,
			CancelToken? token)
		{
			ArgumentNullException.ThrowIfNull(points);
			ArgumentNullException.ThrowIfNull(constraints);
			ArgumentNullException.ThrowIfNull(apts);
			ArgumentNullException.ThrowIfNull(exact);
			if (points.Count < 3)
			{
				throw new ArgumentException("need the three corner points", nameof(points));
			}

			Debug.Assert(apts.Length == points.Count, "one filter input per point");
			Debug.Assert(exact.Length == points.Count, "one exactness flag per point");
			Homog2[] hom = new Homog2[points.Count];
			for (int i = 0; i < points.Count; i++)
			{
				hom[i] = Predicates.Homog2Of(points[i]);
			}

			IVec3 corners = new IVec3(0, 1, 2);
			Sign orient = Predicates.Orient2dH(hom[0], hom[1], hom[2]);
			if (orient == Sign.Zero)
			{
				throw new ArgumentException("degenerate base triangle in CDT input", nameof(points));
			}

			if (orient == Sign.Neg)
			{
				(corners.Y, corners.Z) = (corners.Z, corners.Y);
			}

			// Approximations are taken of the points TRANSLATED by points[0], with the
			// subtraction done exactly in rationals and only the result rounded.
			//
			// Both filters (orient2d, incircle) are translation invariant, so the sign
			// they certify for the translated configuration is the sign of the
			// original one — but their error bounds are not: a rounded coordinate
			// carries a perturbation proportional to its own magnitude, so for an
			// arrangement sitting far from the world origin the untranslated bound is
			// built from |p| while the determinant is built from the (tiny) spacing
			// within the cluster, and every call escalates to the bignum tier.
			// Translating first replaces |p| by the cluster extent: a third of the
			// exact-tier incircle escalations disappear on the #42322 family, and
			// #1716279 (1.44M arrangement points) spends 127s in this stage instead
			// of 491s.
			// The exact-input path stays sound under the same argument: what it
			// requires is that the f64 it sees be exactly the value whose determinant
			// it is bounding, which after translation is the translated rational.
			//
			// `apts`/`exact` arrive from the caller (see `TranslatedFilterInputs`
			// for the values they must hold) because robust/arrangement.rs derives the
			// identical pair, from the identical origin, for its own sweeps.
			//
			// OrderBy, not Array.Sort: the Rust's `sort_by` is stable, and R2's
			// IComparable is the same lexicographic (x, then y) comparison the Rust
			// comparator spells out.
			int[] byCoord = Enumerable.Range(0, points.Count).OrderBy(i => points[i]).ToArray();
			int[] rank = new int[points.Count];
			for (int r = 0; r < byCoord.Length; r++)
			{
				rank[byCoord[r]] = r;
			}

			Cdt cdt = new Cdt(hom, apts, exact, corners, rank);

			// One point insertion is a located split plus a bounded flip cascade, so
			// batching the check keeps it off the hot path; one constraint recovery is
			// a whole corridor walk plus two pseudo-polygon retriangulations, which is
			// expensive enough to check every time.
			for (int p = 3; p < cdt.PointCount; p++)
			{
				if (p % 64 == 0 && Cancel.IsCancelled(token))
				{
					return null;
				}

				int firstNew = cdt.TriCount;
				cdt.InsertPoint(p);
				cdt.SeedSuspects(firstNew);
				cdt.LegalizeSuspects();
			}

			foreach ((int a, int b) in constraints)
			{
				if (Cancel.IsCancelled(token))
				{
					return null;
				}

				Debug.Assert(a != b, "zero-length constraint");
				int firstNew = cdt.TriCount;
				cdt.InsertConstraint(a, b);
				cdt.SeedSuspects(firstNew);
				cdt.LegalizeSuspects();
			}

			return cdt.LiveTriangles();
		}

		/// <summary>
		/// Is <paramref name="r"/> exactly representable in f64 — i.e. is
		/// <c>RatToF64(r)</c> equal to <paramref name="r"/>? Only then may the
		/// exact-input filters run: they assume zero input perturbation.
		/// </summary>
		/// <remarks>
		/// Deliberately CONSERVATIVE and allocation-free: rationals are stored fully
		/// reduced, so <c>n/d</c> is exactly representable whenever <c>|n| &lt; 2⁵³</c>
		/// and <c>d</c> is a power of two (the value then has ≤ 53 significant bits and,
		/// with both parts inside 64 bits, an exponent nowhere near the f64 range limits).
		/// Any value that fails this test — including the representable-but-huge ones like
		/// 2⁶⁰ — is simply reported inexact, which costs an optimization, never
		/// correctness. The two width tests reject on size alone, so constructed
		/// intersection points (huge numerators) bail in a few instructions with no bignum
		/// work at all; this precompute runs once per point per triangulation, against
		/// O(n log n) predicate calls.
		/// </remarks>
		/// <param name="r">The canonical rational to test.</param>
		/// <returns>True when the exact-input filters may be used for this coordinate.</returns>
		internal static bool IsExact(in BigRational r)
		{
			// The Rust asks dashu for `numer.to_i64()` and `denom.to_u64()` BEFORE it
			// looks at either value, and answers false when either does not fit; the two
			// bit-length gates below are that rejection, in that order.
			//
			// The numerator's gate is deliberately TIGHTER than the Rust's 63 bits,
			// because the very next conjunct the Rust evaluates is |n| < 2⁵³: 53 is the
			// strongest bound that still rejects nothing the Rust accepts, so the
			// composition stays extensionally identical while bailing sooner.
			// GetBitLength() excludes the sign bit and measures the two's-complement
			// form, so `> 53` implies |n| > 2⁵³ for either sign, while every value the
			// Rust accepts measures ≤ 53. The one boundary case the gate deliberately
			// lets through, n = −2⁵³ (length exactly 53), is rejected by the exact
			// |n| < 2⁵³ test at the bottom.
			BigInteger n = Backend.Numer(r);
			if (n.GetBitLength() > 53)
			{
				return false;
			}

			// A denominator wider than 64 bits has to fail here rather than at the
			// power-of-two test below, which would otherwise ACCEPT values like 2⁶⁴ that
			// `to_u64` rejects.
			BigInteger d = Backend.Denom(r);
			if (d.GetBitLength() > 64)
			{
				return false;
			}

			// Canonical rationals keep the denominator strictly positive, so this is
			// exactly u64::is_power_of_two (which is false for zero).
			bool denomIsPowerOfTwo = !d.IsZero && (d & (d - BigInteger.One)).IsZero;
			return denomIsPowerOfTwo && BigInteger.Abs(n) < TwoPow53;
		}
	}

	/// <summary>
	/// The private triangle arena of the constrained Delaunay triangulator: exact points,
	/// their filtered approximations, the triangles and the incremental-insertion
	/// bookkeeping. One instance lives for one <see cref="CdtFunctions.Triangulate"/> call.
	/// </summary>
	internal sealed partial class Cdt
	{
		/// <summary>Homogenized once per triangulation; exact predicates reuse these.</summary>
		private readonly Homog2[] pts;

		/// <summary>
		/// Correctly rounded f64 approximations (relative error ≤ ε) of the points
		/// TRANSLATED by <c>points[0]</c>, computed once; the semi-static filters in
		/// exact/Approx.cs certify most predicate signs from these alone, escalating to
		/// <see cref="pts"/> only on near-degeneracies. See
		/// <see cref="CdtFunctions.TriangulateWithApts"/> for why translating first is both
		/// sound and the difference between a usable and a useless filter on arrangements
		/// far from the origin.
		/// </summary>
		private readonly Vec2[] apts;

		/// <summary>
		/// True where <c>apts[i]</c> is EXACTLY the translated point (both coordinates
		/// round-trip), which lets the predicates use the far tighter exact-input filters
		/// in exact/Approx.cs — those assume zero input perturbation, so it is the
		/// TRANSLATED value that must be exact, since that is what the filters see.
		/// </summary>
		private readonly bool[] exact;

		/// <summary>The triangle arena; only the first <see cref="triCount"/> slots exist.</summary>
		private Tri[] tris;

		/// <summary>The Rust's <c>tris.len()</c>: how many arena slots have been pushed.</summary>
		private int triCount;

		/// <summary>Suspect triangles for queue-based Lawson legalization.</summary>
		private readonly List<int> suspects;

		/// <summary>
		/// For each vertex, the most recently created triangle containing it.
		/// Invariant: always alive — every operation that kills a triangle containing v
		/// pushes (and records) a replacement containing v first. Constraint recovery uses
		/// this to rotate around a vertex instead of scanning the whole triangulation.
		/// </summary>
		private readonly int[] vertTri;

		/// <summary>
		/// Rank of each point in exact lexicographic (x, y) coordinate order.
		/// Cocircular-tie flips key on this so the triangulation is a function of the point
		/// coordinates, not of construction history — coincident coplanar triangles must
		/// tile their shared region identically or the cell complex sees one physical sheet
		/// crossing as several vi-distinct walls with understated winding steps.
		/// </summary>
		private readonly int[] rank;

		/// <summary>
		/// Builds the arena holding the single base triangle <paramref name="corners"/>.
		/// </summary>
		/// <param name="pts">Homogenized points.</param>
		/// <param name="apts">Rounded translated points.</param>
		/// <param name="exact">Per-point exactness of <paramref name="apts"/>.</param>
		/// <param name="corners">The base triangle's corner indices, already CCW.</param>
		/// <param name="rank">Lexicographic coordinate rank per point.</param>
		internal Cdt(Homog2[] pts, Vec2[] apts, bool[] exact, IVec3 corners, int[] rank)
		{
			this.pts = pts;
			this.apts = apts;
			this.exact = exact;
			this.tris = new Tri[8];
			this.tris[0] = new Tri
			{
				V = corners,
				Adj = new IVec3(-1, -1, -1),
				Con = default(Bool3),
				Alive = true,
			};
			this.triCount = 1;
			this.suspects = new List<int>();
			this.vertTri = new int[pts.Length];
			this.rank = rank;
		}

		/// <summary>How many points the triangulation covers.</summary>
		internal int PointCount
		{
			get { return this.pts.Length; }
		}

		/// <summary>The Rust's <c>tris.len()</c>.</summary>
		internal int TriCount
		{
			get { return this.triCount; }
		}

		/// <summary>The vertex triples of every triangle still alive, in arena order.</summary>
		/// <returns>The output triangulation.</returns>
		internal List<IVec3> LiveTriangles()
		{
			List<IVec3> outTris = new List<IVec3>();
			for (int i = 0; i < this.triCount; i++)
			{
				if (this.tris[i].Alive)
				{
					outTris.Add(this.tris[i].V);
				}
			}

			return outTris;
		}

		/// <summary>Appends a triangle to the arena, growing it when full.</summary>
		/// <param name="t">The triangle to push.</param>
		private void PushTri(in Tri t)
		{
			if (this.triCount == this.tris.Length)
			{
				Array.Resize(ref this.tris, this.tris.Length * 2);
			}

			this.tris[this.triCount] = t;
			this.triCount++;
		}

		/// <summary>Rust's <c>[usize; 3]::contains</c> over an index triple.</summary>
		/// <param name="v">The triple.</param>
		/// <param name="a">The index to look for.</param>
		/// <returns>True when <paramref name="a"/> is one of the three.</returns>
		private static bool Contains(in IVec3 v, int a)
		{
			return v.X == a || v.Y == a || v.Z == a;
		}

		/// <summary>
		/// orient2d with the approx filter first, exact fallback. All CDT sign tests funnel
		/// through here (and <see cref="NonDelaunay"/>) so generic-position queries never
		/// touch the bignum tier.
		/// </summary>
		/// <param name="i">First point index.</param>
		/// <param name="j">Second point index.</param>
		/// <param name="k">Third point index.</param>
		/// <returns>The exact orientation sign.</returns>
		private Sign O2(int i, int j, int k)
		{
			if (this.exact[i] && this.exact[j] && this.exact[k])
			{
				Sign? s = Approx.Orient2dAExact(this.apts[i], this.apts[j], this.apts[k]);
				if (s.HasValue)
				{
					return s.Value;
				}
			}

			return Approx.Orient2dA(this.apts[i], this.apts[j], this.apts[k])
				?? Predicates.Orient2dH(this.pts[i], this.pts[j], this.pts[k]);
		}

		/// <summary>
		/// Incircle sign with the approx filter first, exact fallback. <c>Pos</c> means
		/// <paramref name="d"/> is strictly inside the circumcircle of the (CCW) triangle;
		/// <c>Zero</c> is an exact cocircular tie.
		/// </summary>
		/// <param name="tri">The CCW triangle's vertex indices.</param>
		/// <param name="d">The query point index.</param>
		/// <returns>The exact incircle sign.</returns>
		private Sign Incircle(in IVec3 tri, int d)
		{
			if (this.exact[tri.X] && this.exact[tri.Y] && this.exact[tri.Z] && this.exact[d])
			{
				// Tight arithmetic-only bound; neither filter dominates the other
				// (their permanents differ), so fall through to the general one.
				Sign? s = Approx.IncircleAExact(
					this.apts[tri.X],
					this.apts[tri.Y],
					this.apts[tri.Z],
					this.apts[d]);
				if (s.HasValue)
				{
					return s.Value;
				}
			}

			Sign? general = Approx.IncircleA(
				this.apts[tri.X],
				this.apts[tri.Y],
				this.apts[tri.Z],
				this.apts[d]);
			return general ?? Predicates.IncircleH(
				this.pts[tri.X],
				this.pts[tri.Y],
				this.pts[tri.Z],
				this.pts[d]);
		}

		/// <summary>Strict Delaunay violation (see <see cref="Incircle"/>).</summary>
		/// <param name="tri">The CCW triangle's vertex indices.</param>
		/// <param name="d">The query point index.</param>
		/// <returns>True when <paramref name="d"/> is strictly inside the circumcircle.</returns>
		private bool NonDelaunay(in IVec3 tri, int d)
		{
			return this.Incircle(tri, d) == Sign.Pos;
		}

		/// <summary>
		/// Canonical identity of the diagonal {u, v}: the pair of coordinate ranks, low
		/// first. Ranks order points by exact coordinates, so two coincident coplanar
		/// triangles — which project into the same dominant-axis plane and therefore hand
		/// this CDT the same coordinates for shared points — agree on this key even though
		/// their local point indices differ.
		/// </summary>
		/// <param name="u">One endpoint.</param>
		/// <param name="v">The other endpoint.</param>
		/// <returns>The ordered rank pair.</returns>
		private (int Low, int High) DiagKey(int u, int v)
		{
			int ru = this.rank[u];
			int rv = this.rank[v];
			return (Math.Min(ru, rv), Math.Max(ru, rv));
		}

		/// <summary>
		/// <c>point_in_tri_2d</c> over the filtered predicate (same TriLoc semantics as
		/// <see cref="Predicates.PointInTri2dH"/>; triangle is CCW by construction).
		/// </summary>
		/// <param name="p">The query point index.</param>
		/// <param name="v">The CCW triangle's vertex indices.</param>
		/// <returns>Where <paramref name="p"/> lies relative to the triangle.</returns>
		private TriLoc LocInTri(int p, in IVec3 v)
		{
			Sign s0 = this.O2(v.X, v.Y, p);
			Sign s1 = this.O2(v.Y, v.Z, p);
			Sign s2 = this.O2(v.Z, v.X, p);
			if (s0 == Sign.Neg || s1 == Sign.Neg || s2 == Sign.Neg)
			{
				return TriLoc.Outside;
			}

			bool z0 = s0 == Sign.Zero;
			bool z1 = s1 == Sign.Zero;
			bool z2 = s2 == Sign.Zero;
			if (!z0 && !z1 && !z2)
			{
				return TriLoc.Inside;
			}
			else if (z0 && !z1 && !z2)
			{
				return TriLoc.OnEdge(0);
			}
			else if (!z0 && z1 && !z2)
			{
				return TriLoc.OnEdge(1);
			}
			else if (!z0 && !z1 && z2)
			{
				return TriLoc.OnEdge(2);
			}
			else if (z0 && !z1 && z2)
			{
				return TriLoc.OnVertex(0);
			}
			else if (z0 && z1 && !z2)
			{
				return TriLoc.OnVertex(1);
			}
			else if (!z0 && z1 && z2)
			{
				return TriLoc.OnVertex(2);
			}
			else
			{
				return TriLoc.Outside;
			}
		}

		/// <summary>
		/// One triangle of the CDT. Vertices are CCW; edge i runs v[i] → v[i+1],
		/// adj[i] is the triangle index across that edge (-1 on the hull), con[i]
		/// marks a constrained edge that flips must never cross.
		/// </summary>
		private struct Tri
		{
			/// <summary>The three CCW vertex indices.</summary>
			public IVec3 V;

			/// <summary>Neighbor triangle across each edge, or -1 on the hull.</summary>
			public IVec3 Adj;

			/// <summary>Whether each edge is a constraint flips must not cross.</summary>
			public Bool3 Con;

			/// <summary>Whether this arena slot is still part of the triangulation.</summary>
			public bool Alive;
		}

		/// <summary>
		/// Rust's <c>[bool; 3]</c>: three flags with a by-index <c>ref</c> accessor, so an
		/// arena element's flag can be written in place exactly as the Rust writes it.
		/// </summary>
		private struct Bool3
		{
			/// <summary>The flag for edge 0.</summary>
			public bool E0;

			/// <summary>The flag for edge 1.</summary>
			public bool E1;

			/// <summary>The flag for edge 2.</summary>
			public bool E2;

			/// <summary>Creates the triple from its three flags.</summary>
			/// <param name="e0">The flag for edge 0.</param>
			/// <param name="e1">The flag for edge 1.</param>
			/// <param name="e2">The flag for edge 2.</param>
			public Bool3(bool e0, bool e1, bool e2)
			{
				this.E0 = e0;
				this.E1 = e1;
				this.E2 = e2;
			}

			/// <summary>Flag by edge index.</summary>
			/// <param name="i">The edge index, 0..2.</param>
			/// <returns>A reference to that flag.</returns>
			/// <exception cref="ArgumentOutOfRangeException">The index is not 0, 1 or 2.</exception>
			[UnscopedRef]
			public ref bool this[int i]
			{
				get
				{
					switch (i)
					{
						case 0:
							return ref this.E0;
						case 1:
							return ref this.E1;
						case 2:
							return ref this.E2;
						default:
							throw new ArgumentOutOfRangeException(nameof(i), $"Bool3 index out of range: {i}");
					}
				}
			}
		}
	}
}
