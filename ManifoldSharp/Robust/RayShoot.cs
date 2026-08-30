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

// RayShoot.cs — port of robust/ray_shoot.rs, whose header reads:
//
//   Exact point-in-solid classification for surface components no intersection
//   ring reaches (paper §7.4).
//
//   A component of mesh P that never meets Q lies entirely inside or entirely
//   outside Q; one exact winding-number query at any of its points decides the
//   tag (inside → intersection, outside → union). The winding number is the
//   signed count of ray crossings against Q's original triangles, evaluated
//   with rational Plücker side tests. Rays that graze any edge, vertex, or
//   containing plane are detected exactly and retried with the next direction
//   from a fixed list — no perturbation, no tolerances.
//
//   Complement operands (the flipped Q of a subtraction): a flipped closed
//   mesh has winding -1 inside the original and 0 outside, so "inside the
//   complement solid" is winding == 0 — the `complement` flag selects that
//   interpretation.
//
// ── `[f64; 3]` is a Vec3 here ────────────────────────────────────────────────
// Same call GraphGeom.cs made: the exact tier already settled on Vec3 for the
// Rust's bare `[f64; 3]` approximations (Approx.Orient3dA takes Vec3), so the
// `ap`/`ao2`/`fa`/`fb`/`fc` locals are Vec3 — the same three doubles in the
// same order, computed in the same order, so `ap[0]` becomes `ap.X` with no
// change of meaning. Vec3's indexer keeps the `for k in 0..3` slab loop
// transcribable.
//
// ── `&[[Vec3; 3]]` is an IReadOnlyList<Vec3[]> ───────────────────────────────
// Same call TriTri.cs and GraphGeom.cs made: a C# array carries its length at
// run time instead of in its type, and Soup.ImplToTris hands triangles out as
// Vec3[] already.
//
// ── `matches!(…, (Pos, Neg) | (Neg, Pos))` is StrictlyOpposite ───────────────
// C# has no `matches!`, and the Rust spells the same two-arm pattern six times
// across two functions plus three more in `could_graze`. One helper taking
// `Sign?` covers both the filtered (`Option<Sign>`) and exact (`Sign`) sites —
// a bare `Sign` converts implicitly — so every call site keeps the Rust's
// three-term `||` shape verbatim.
//
// ── `continue 'dirs` is a flag ───────────────────────────────────────────────
// C# has no labelled continue. WindingOffSurface's inner loop sets `grazed` and
// breaks; the outer loop then continues. `winding_one_dir` needs no such thing
// — it is already a function whose `None` return *is* the retry signal, and the
// C# keeps that exactly (`int?`).

using System.Diagnostics;
using System.Numerics;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp.Robust
{
	/// <summary>
	/// The free functions of <c>robust/ray_shoot.rs</c>: exact winding numbers by
	/// rational Plücker ray shooting, with graze detection and a fixed retry list.
	/// </summary>
	public static class RayShoot
	{
		/// <summary>
		/// Fixed retry directions; pairwise non-parallel, chosen to make consecutive
		/// degeneracies essentially impossible. All integer, so exact.
		/// </summary>
		/// <remarks>
		/// Never handed to a caller directly — <see cref="Dir"/> copies. Rust's
		/// <c>for d in DIRS</c> iterates <c>[i32; 3]</c> <b>by value</b>, so every
		/// direction the algorithm sees there is already a copy; a C# <c>foreach</c> over
		/// the jagged table would instead hand out references into the shared static, where
		/// one future in-place write would corrupt the retry list process-wide. Copying is
		/// both the faithful translation and the safe one, and the twelve small allocations
		/// per query are nothing beside the per-triangle bignum work they guard.
		/// </remarks>
		private static readonly int[][] Dirs =
		{
			new[] { 1, 0, 0 },
			new[] { 0, 1, 0 },
			new[] { 0, 0, 1 },
			new[] { 1, 1, 1 },
			new[] { 1, 2, 3 },
			new[] { 3, 1, 7 },
			new[] { 5, 11, 2 },
			new[] { 7, 3, 13 },
			new[] { 2, 9, 5 },
			new[] { 11, 4, 1 },
			new[] { 3, 17, 8 },
			new[] { 13, 6, 5 },
		};

		/// <summary>
		/// Exact winding number of <paramref name="point"/> with respect to the closed
		/// oriented triangle soup <paramref name="tris"/>. The point must not lie on the
		/// surface.
		/// </summary>
		/// <param name="point">The exact query point.</param>
		/// <param name="tris">The closed oriented triangle soup.</param>
		/// <returns>The signed crossing count.</returns>
		public static int WindingNumber(R3 point, IReadOnlyList<Vec3[]> tris)
		{
			ArgumentNullException.ThrowIfNull(tris);
			Box[] boxes = new Box[tris.Count];
			for (int i = 0; i < tris.Count; i++)
			{
				boxes[i] = TriBoxF64(tris[i]);
			}

			return WindingNumberBoxed(point, tris, boxes);
		}

		/// <summary>
		/// <see cref="WindingNumber"/> against a prebuilt <see cref="WindingIndex"/>.
		/// </summary>
		/// <param name="point">The exact query point.</param>
		/// <param name="tris">The closed oriented triangle soup the index was built over.</param>
		/// <param name="index">The prebuilt acceleration structure.</param>
		/// <returns>The signed crossing count.</returns>
		/// <exception cref="InvalidOperationException">Every candidate direction grazed.</exception>
		public static int WindingNumberIndexed(R3 point, IReadOnlyList<Vec3[]> tris, WindingIndex index)
		{
			ArgumentNullException.ThrowIfNull(tris);
			ArgumentNullException.ThrowIfNull(index);
			RayPrefilter prefilter = new RayPrefilter(point);
			Vec3 ap = new Vec3(
				Rational.RatToF64(point.X),
				Rational.RatToF64(point.Y),
				Rational.RatToF64(point.Z));
			for (int di = 0; di < Dirs.Length; di++)
			{
				int[] d = Dir(di);
				List<int> cand = index.Candidates(prefilter, d);

				// Rust `sort_unstable`: the candidate indices are distinct, so the
				// ordering is fully determined and stability cannot be observed.
				cand.Sort(); // deterministic evaluation order
				int? w = WindingOneDir(point, ap, d, prefilter, Pairs());
				if (w != null)
				{
					return w.Value;
				}

				IEnumerable<(Vec3[] Tri, Box Bbox)> Pairs()
				{
					foreach (int i in cand)
					{
						yield return (tris[i], index.BoxAt(i));
					}
				}
			}

			throw new InvalidOperationException("all candidate ray directions degenerate — malformed input");
		}

		/// <summary>
		/// <see cref="WindingNumber"/> with caller-provided per-triangle boxes — batch
		/// query sites build them once per operand instead of once per query.
		/// </summary>
		/// <param name="point">The exact query point.</param>
		/// <param name="tris">The closed oriented triangle soup.</param>
		/// <param name="boxes">One f64 bounding box per triangle.</param>
		/// <returns>The signed crossing count.</returns>
		/// <exception cref="InvalidOperationException">Every candidate direction grazed.</exception>
		public static int WindingNumberBoxed(R3 point, IReadOnlyList<Vec3[]> tris, IReadOnlyList<Box> boxes)
		{
			ArgumentNullException.ThrowIfNull(tris);
			ArgumentNullException.ThrowIfNull(boxes);
			RayPrefilter prefilter = new RayPrefilter(point);
			Vec3 ap = new Vec3(
				Rational.RatToF64(point.X),
				Rational.RatToF64(point.Y),
				Rational.RatToF64(point.Z));
			for (int di = 0; di < Dirs.Length; di++)
			{
				int[] d = Dir(di);
				int? w = WindingOneDir(point, ap, d, prefilter, Zip());
				if (w != null)
				{
					return w.Value;
				}

				IEnumerable<(Vec3[] Tri, Box Bbox)> Zip()
				{
					// `Math.Min` reproduces Rust `zip`'s truncation to the shorter of the
					// two: a length mismatch violates the one-box-per-triangle precondition,
					// and this site under-counts silently exactly as the Rust does.
					// WindingOffSurface's three-way zip deliberately does not — see there.
					int n = Math.Min(tris.Count, boxes.Count);
					for (int i = 0; i < n; i++)
					{
						yield return (tris[i], boxes[i]);
					}
				}
			}

			throw new InvalidOperationException("all candidate ray directions degenerate — malformed input");
		}

		/// <summary>
		/// Exact winding number of <c>point + ε·outward</c> for an infinitesimal ε &gt; 0:
		/// the winding just off a surface piece through <paramref name="point"/>, on the
		/// piece's outward side. Used to detect pieces that are interior walls of their own
		/// mesh (self-overlapping or nested sheets): a piece lies on the boundary of the
		/// solid <c>{w ≠ 0}</c> only when this value is 0 (or −1 for an
		/// orientation-flipped complement operand).
		/// </summary>
		/// <remarks>
		/// <paramref name="boxes"/> are per-triangle f64 bounding boxes (rounded vertices
		/// are fine — the prefilter inflation covers the rounding) for conservative pruning.
		/// <para>
		/// <paramref name="tris"/> may pass through <paramref name="point"/> (the piece's
		/// own triangle always does). Triangles whose plane contains <paramref name="point"/>
		/// are resolved by a second-order perturbation argument: the query point is
		/// x + ε·outward + ε²·dir, so such a triangle is crossed by the forward ray exactly
		/// when the plane's normal separates <paramref name="outward"/> from <c>dir</c>
		/// (sides differ); when <paramref name="outward"/> lies in the plane, the ε² term
		/// decides and the ray always leaves on the <c>dir</c> side — no crossing. Ray
		/// directions are restricted to the outward hemisphere so the piece's own plane
		/// never counts.
		/// </para>
		/// </remarks>
		/// <param name="point">The exact point on the surface piece.</param>
		/// <param name="outward">The piece's outward direction at that point.</param>
		/// <param name="tris">The exact triangle soup.</param>
		/// <param name="trisF64">The same triangles, rounded to f64.</param>
		/// <param name="boxes">One f64 bounding box per triangle.</param>
		/// <returns>The signed crossing count just off the surface.</returns>
		/// <exception cref="InvalidOperationException">Every candidate direction grazed.</exception>
		public static int WindingOffSurface(
			R3 point,
			R3 outward,
			IReadOnlyList<R3[]> tris,
			IReadOnlyList<Vec3[]> trisF64,
			IReadOnlyList<Box> boxes)
		{
			ArgumentNullException.ThrowIfNull(tris);
			ArgumentNullException.ThrowIfNull(trisF64);
			ArgumentNullException.ThrowIfNull(boxes);
			Debug.Assert(tris.Count == boxes.Count, "one box per triangle");
			Debug.Assert(tris.Count == trisF64.Count, "one f64 triangle per exact triangle");
			RayPrefilter prefilter = new RayPrefilter(point);
			Vec3 ap = new Vec3(
				Rational.RatToF64(point.X),
				Rational.RatToF64(point.Y),
				Rational.RatToF64(point.Z));
			foreach (int[] d in DirsWithOpposites())
			{
				R3 dir = DirR3(d);

				// Outward hemisphere only: the ε·outward offset must stay on the
				// near side of the piece's own plane relative to the ray.
				if (SignFunctions.OfRat(dir.Dot(outward)) != Sign.Pos)
				{
					continue;
				}

				R3 o2 = point.Add(dir);
				Vec3 ao2 = new Vec3(ap.X + d[0], ap.Y + d[1], ap.Z + d[2]);
				int winding = 0;
				bool grazed = false;

				// Rust's `tris.zip(tris_f64).zip(boxes)` truncates to the shortest, so a
				// caller that broke the one-entry-per-triangle precondition would there
				// silently get a winding number computed over a prefix. Indexing all three
				// by `tris.Count` instead throws — a deliberate divergence, and the better
				// failure mode for a precondition the debug assertions above already state.
				for (int i = 0; i < tris.Count; i++)
				{
					R3[] t = tris[i];
					Vec3[] tf = trisF64[i];
					Box bbox = boxes[i];
					if (!prefilter.MayHit(d, bbox))
					{
						continue;
					}

					Vec3 fa = tf[0];
					Vec3 fb = tf[1];
					Vec3 fc = tf[2];

					// Approx-first (certified misses skip all rational work). Three locals
					// rather than the Rust's `sides` array: this runs once per surviving
					// triangle per direction, and a heap array there is the hottest
					// allocation in the file on operands with millions of points.
					Sign? side0 = Approx.Orient3dA(ap, ao2, fa, fb);
					Sign? side1 = Approx.Orient3dA(ap, ao2, fb, fc);
					Sign? side2 = Approx.Orient3dA(ap, ao2, fc, fa);
					if (StrictlyOpposite(side0, side1)
						|| StrictlyOpposite(side1, side2)
						|| StrictlyOpposite(side0, side2))
					{
						continue;
					}

					R3 a = t[0];
					R3 b = t[1];
					R3 c = t[2];
					Sign sAb = side0 ?? Predicates.Orient3dR(point, o2, a, b);
					Sign sBc = side1 ?? Predicates.Orient3dR(point, o2, b, c);
					Sign sCa = side2 ?? Predicates.Orient3dR(point, o2, c, a);
					if (sAb == Sign.Zero || sBc == Sign.Zero || sCa == Sign.Zero)
					{
						if (CouldGraze(point, o2, a, b, c))
						{
							grazed = true; // Rust `continue 'dirs`
							break;
						}

						continue;
					}

					if (sAb != sBc || sBc != sCa)
					{
						continue; // line misses the triangle
					}

					Sign nDotDir = sAb; // sign(n·dir), n the CCW normal
					Sign h = Approx.Orient3dA(fa, fb, fc, ap) ?? Predicates.Orient3dR(a, b, c, point);
					if (h == Sign.Zero)
					{
						// Plane through the query point; the pierce point is `point`
						// itself (the transversal line meets the plane only there).
						// Perturbed crossing exists iff outward and dir are on
						// opposite sides of the plane.
						R3 n = Predicates.TriNormalR(a, b, c);
						Sign sOut = SignFunctions.OfRat(n.Dot(outward));
						if (sOut == nDotDir.Flip())
						{
							winding += nDotDir == Sign.Pos ? 1 : -1;
						}

						continue;
					}

					if (h != nDotDir.Flip())
					{
						continue; // intersection lies behind the ray origin
					}

					winding += nDotDir == Sign.Pos ? 1 : -1;
				}

				if (grazed)
				{
					continue;
				}

				return winding;
			}

			throw new InvalidOperationException("all candidate ray directions degenerate — malformed input");
		}

		/// <summary>Representative interior point of a piece: its centroid (exact).</summary>
		/// <param name="v">The piece's three exact corners.</param>
		/// <returns>Their centroid.</returns>
		/// <exception cref="ArgumentException">The array is not exactly three points.</exception>
		public static R3 PieceCentroid(IReadOnlyList<R3> v)
		{
			ArgumentNullException.ThrowIfNull(v);
			if (v.Count != 3)
			{
				throw new ArgumentException($"a piece is three corners, got {v.Count}", nameof(v));
			}

			BigRational third = Backend.RatNew(BigInteger.One, new BigInteger(3));
			return v[0].Add(v[1]).Add(v[2]).Scale(third);
		}

		/// <summary>
		/// Is <paramref name="point"/> inside the solid bounded by <paramref name="tris"/>?
		/// <paramref name="complement"/> flips the interpretation for orientation-reversed
		/// (subtraction) operands.
		/// </summary>
		/// <param name="point">The exact query point.</param>
		/// <param name="tris">The closed oriented triangle soup.</param>
		/// <param name="complement">True for an orientation-reversed operand.</param>
		/// <returns>True when the point is inside.</returns>
		public static bool PointInside(R3 point, IReadOnlyList<Vec3[]> tris, bool complement)
		{
			int w = WindingNumber(point, tris);
			if (complement)
			{
				return w == 0;
			}

			return w != 0;
		}

		/// <summary>
		/// The axis-aligned bounding box of a triangle. Duplicates
		/// <see cref="GraphGeom.TriBox"/>, as the Rust duplicates
		/// <c>graph_geom::tri_box</c> here.
		/// </summary>
		internal static Box TriBoxF64(Vec3[] t)
		{
			Box b = Box.FromPoints(t[0], t[1]);
			b.UnionPoint(t[2]);
			return b;
		}

		/// <summary>
		/// A private copy of retry direction <paramref name="i"/> — see <see cref="Dirs"/>
		/// for why the table itself is never handed out.
		/// </summary>
		/// <param name="i">The index into <see cref="Dirs"/>.</param>
		/// <returns>A fresh three-element direction.</returns>
		private static int[] Dir(int i)
		{
			int[] d = Dirs[i];
			return new[] { d[0], d[1], d[2] };
		}

		/// <summary>The integer retry direction as an exact rational vector.</summary>
		private static R3 DirR3(int[] d)
		{
			return new R3(Rational.Rat(d[0]), Rational.Rat(d[1]), Rational.Rat(d[2]));
		}

		/// <summary>
		/// Rust's <c>DIRS.iter().flat_map(|d| [*d, [-d[0], -d[1], -d[2]]])</c>: each fixed
		/// direction immediately followed by its opposite.
		/// </summary>
		private static IEnumerable<int[]> DirsWithOpposites()
		{
			for (int i = 0; i < Dirs.Length; i++)
			{
				int[] d = Dirs[i];
				yield return Dir(i);
				yield return new[] { -d[0], -d[1], -d[2] };
			}
		}

		/// <summary>
		/// The Rust's <c>matches!((x, y), (Some(Pos), Some(Neg)) | (Some(Neg), Some(Pos)))</c>:
		/// two strictly opposite side signs. Uncertain (null) sides are never opposite.
		/// </summary>
		private static bool StrictlyOpposite(Sign? a, Sign? b)
		{
			return (a == Sign.Pos && b == Sign.Neg) || (a == Sign.Neg && b == Sign.Pos);
		}

		/// <summary>
		/// One direction's signed crossing count over the given (triangle, box) pairs;
		/// null means the ray grazed something and the caller must retry with the next
		/// direction. Approx-filtered first, exact on demand.
		/// </summary>
		private static int? WindingOneDir(
			R3 point,
			Vec3 ap,
			int[] d,
			RayPrefilter prefilter,
			IEnumerable<(Vec3[] Tri, Box Bbox)> trisBoxes)
		{
			R3 dir = DirR3(d);
			R3 o2 = point.Add(dir);
			Vec3 ao2 = new Vec3(ap.X + d[0], ap.Y + d[1], ap.Z + d[2]);
			int winding = 0;
			foreach ((Vec3[] t, Box bbox) in trisBoxes)
			{
				if (!prefilter.MayHit(d, bbox))
				{
					continue;
				}

				Vec3 fa = t[0];
				Vec3 fb = t[1];
				Vec3 fc = t[2];

				// Plücker side tests of the ray line against the three edges —
				// approx first, exact only when the filter cannot certify. Three locals
				// rather than the Rust's `sides` array: this runs once per surviving
				// triangle per direction, and a heap array there is the hottest allocation
				// in the file on operands with millions of points.
				Sign? side0 = Approx.Orient3dA(ap, ao2, fa, fb);
				Sign? side1 = Approx.Orient3dA(ap, ao2, fb, fc);
				Sign? side2 = Approx.Orient3dA(ap, ao2, fc, fa);

				// Certified miss without touching the bignum tier: two strict opposite
				// signs mean the line cannot pierce the closed triangle.
				if (StrictlyOpposite(side0, side1)
					|| StrictlyOpposite(side1, side2)
					|| StrictlyOpposite(side0, side2))
				{
					continue;
				}

				// Resolve any uncertain side exactly (rational triangle built
				// only when a filter actually missed).
				Sign sAb;
				Sign sBc;
				Sign sCa;
				if (side0 is Sign x && side1 is Sign y && side2 is Sign z)
				{
					// Rust `(Some(x), Some(y), Some(z)) => (x, y, z)`.
					sAb = x;
					sBc = y;
					sCa = z;
				}
				else
				{
					R3 a = R3.FromVec3(t[0]);
					R3 b = R3.FromVec3(t[1]);
					R3 c = R3.FromVec3(t[2]);
					sAb = side0 ?? Predicates.Orient3dR(point, o2, a, b);
					sBc = side1 ?? Predicates.Orient3dR(point, o2, b, c);
					sCa = side2 ?? Predicates.Orient3dR(point, o2, c, a);
					if (sAb == Sign.Zero || sBc == Sign.Zero || sCa == Sign.Zero)
					{
						// Might graze an edge or vertex of this triangle —
						// only a problem if the grazing happens on the
						// forward ray within the triangle's neighborhood;
						// retrying is always safe.
						if (CouldGraze(point, o2, a, b, c))
						{
							return null;
						}

						continue;
					}
				}

				if (sAb != sBc || sBc != sCa)
				{
					continue; // line misses the triangle
				}

				// Line pierces the triangle interior. Forward (t > 0)?
				Sign h = Approx.Orient3dA(fa, fb, fc, ap)
					?? Predicates.Orient3dR(
						R3.FromVec3(t[0]),
						R3.FromVec3(t[1]),
						R3.FromVec3(t[2]),
						point);
				if (h == Sign.Zero)
				{
					// Point on the triangle's plane while the line pierces the
					// interior ⇒ the point is on the surface — caller violated
					// the precondition, or the ray grazes; retry.
					return null;
				}

				// n·dir sign == common side-sign (all three Pos ⇔ dir on the
				// CCW-normal side).
				Sign nDotDir = sAb; // sAb == sBc == sCa == sign(n·dir)
				if (h != nDotDir.Flip())
				{
					continue; // intersection lies behind the ray origin
				}

				// Pos exits through a front face.
				winding += nDotDir == Sign.Pos ? 1 : -1;
			}

			return winding;
		}

		/// <summary>
		/// Conservative check whether a zero Plücker sign can affect the crossing count:
		/// true when the ray's line meets the triangle's plane inside or on the triangle
		/// boundary, or runs inside its plane. False positives only cost a retry.
		/// </summary>
		private static bool CouldGraze(R3 o, R3 o2, R3 a, R3 b, R3 c)
		{
			Sign sAb = Predicates.Orient3dR(o, o2, a, b);
			Sign sBc = Predicates.Orient3dR(o, o2, b, c);
			Sign sCa = Predicates.Orient3dR(o, o2, c, a);

			// The line misses the closed triangle only if two side tests have
			// strictly opposite signs.
			return !(StrictlyOpposite(sAb, sBc)
				|| StrictlyOpposite(sBc, sCa)
				|| StrictlyOpposite(sAb, sCa));
		}
	}

	/// <summary>
	/// Conservative f64 prefilter for one winding query: triangles whose (inflated)
	/// bounding box cannot meet the forward ray are skipped before any exact arithmetic.
	/// Sound because a pruned triangle can contribute neither a forward crossing nor a
	/// forward grazing hazard: the inflation margin (1e-6, magnitude-scaled) exceeds every
	/// f64 rounding error involved by many orders of magnitude, so only provably-clear
	/// triangles are pruned. False keeps merely cost an exact test.
	/// </summary>
	internal readonly struct RayPrefilter
	{
		/// <summary>The query point rounded to f64.</summary>
		internal readonly Vec3 Origin;

		/// <summary>The magnitude-scaled inflation margin.</summary>
		internal readonly double Eps;

		/// <summary>Builds the prefilter for one exact query point.</summary>
		/// <param name="point">The exact query point.</param>
		internal RayPrefilter(R3 point)
		{
			ArgumentNullException.ThrowIfNull(point);
			Vec3 origin = point.ToVec3Rounded();
			double mag = MaxF64(MaxF64(Math.Abs(origin.X), Math.Abs(origin.Y)), Math.Abs(origin.Z));
			this.Origin = origin;
			this.Eps = 1e-6 * (1.0 + mag);
		}

		/// <summary>
		/// Could the forward ray from <see cref="Origin"/> along integer direction
		/// <paramref name="d"/> pass within <see cref="Eps"/> of <paramref name="bbox"/>?
		/// Slab test; conservative on every comparison.
		/// </summary>
		/// <param name="d">The integer ray direction.</param>
		/// <param name="bbox">The triangle's bounding box.</param>
		/// <returns>False only when the ray provably clears the inflated box.</returns>
		internal bool MayHit(int[] d, Box bbox)
		{
			double t0 = double.NegativeInfinity;
			double t1 = double.PositiveInfinity;
			for (int k = 0; k < 3; k++)
			{
				double lo = bbox.Min[k] - this.Eps;
				double hi = bbox.Max[k] + this.Eps;
				double dk = d[k];
				double pk = this.Origin[k];
				if (dk == 0.0)
				{
					if (pk < lo || pk > hi)
					{
						return false;
					}

					continue;
				}

				double ta = (lo - pk) / dk;
				double tb = (hi - pk) / dk;
				if (ta > tb)
				{
					(ta, tb) = (tb, ta);
				}

				t0 = MaxF64(t0, ta);
				t1 = MinF64(t1, tb);
			}

			return t0 <= t1 + this.Eps && t1 >= -this.Eps;
		}
	}

	/// <summary>
	/// Per-operand acceleration structure for batches of winding queries: a BVH over the
	/// triangle boxes, queried with a conservative semi-infinite box around each candidate
	/// ray. Build once per operand; thousands of per-component queries then touch only the
	/// triangles near their ray instead of scanning the whole soup.
	/// </summary>
	public sealed class WindingIndex
	{
		private readonly Box[] boxes;
		private readonly Collider collider;
		private readonly int[] order;

		/// <summary>Builds the index over a triangle soup.</summary>
		/// <param name="tris">The triangles the winding queries will be evaluated against.</param>
		public WindingIndex(IReadOnlyList<Vec3[]> tris)
		{
			ArgumentNullException.ThrowIfNull(tris);
			Box[] boxes = new Box[tris.Count];
			for (int i = 0; i < tris.Count; i++)
			{
				boxes[i] = RayShoot.TriBoxF64(tris[i]);
			}

			Box scene = new Box();
			foreach (Box b in boxes)
			{
				scene = scene.UnionBox(b);
			}

			// Rust `sort_by_key` is stable; OrderBy is the documented-stable C# equivalent.
			int[] order = Enumerable.Range(0, tris.Count)
				.OrderBy(i => Sort.MortonCode(boxes[i].Center(), scene))
				.ToArray();
			Box[] leafBox = new Box[order.Length];
			uint[] leafMorton = new uint[order.Length];
			for (int k = 0; k < order.Length; k++)
			{
				leafBox[k] = boxes[order[k]];
				leafMorton[k] = Sort.MortonCode(boxes[order[k]].Center(), scene);
			}

			this.boxes = boxes;
			this.collider = new Collider(leafBox, leafMorton);
			this.order = order;
		}

		/// <summary>
		/// The f64 bounding box of triangle <paramref name="index"/>, in input order.
		/// </summary>
		/// <remarks>
		/// An accessor rather than the backing array: the Rust field is module-private and
		/// never escapes, and handing out a <c>Box[]</c> (or an <c>IReadOnlyList</c> a caller
		/// can cast back to one) would let a consumer mutate boxes the collider was already
		/// built from, silently desynchronizing the BVH from the leaves it indexes.
		/// </remarks>
		/// <param name="index">The triangle index.</param>
		/// <returns>That triangle's bounding box.</returns>
		internal Box BoxAt(int index)
		{
			return this.boxes[index];
		}

		/// <summary>
		/// Conservative superset of every triangle <see cref="RayPrefilter.MayHit"/> would
		/// keep for this (origin, direction): a box stretching to infinity along the ray,
		/// inflated past the prefilter's epsilon and backward reach.
		/// </summary>
		/// <param name="prefilter">The query's prefilter.</param>
		/// <param name="d">The integer ray direction.</param>
		/// <returns>The candidate triangle indices, in BVH traversal order.</returns>
		internal List<int> Candidates(RayPrefilter prefilter, int[] d)
		{
			// Rust's `d.iter().map(|v| v.abs()).max().unwrap_or(1)`; the fallback is
			// unreachable for a fixed three-element direction.
			double maxd = Math.Max(Math.Max(Math.Abs(d[0]), Math.Abs(d[1])), Math.Abs(d[2]));
			double slack = prefilter.Eps * (2.0 + maxd);
			Vec3 lo = new Vec3(0.0, 0.0, 0.0);
			Vec3 hi = new Vec3(0.0, 0.0, 0.0);
			for (int k = 0; k < 3; k++)
			{
				double o = prefilter.Origin[k];
				if (d[k] > 0)
				{
					lo[k] = o - slack;
					hi[k] = double.PositiveInfinity;
				}
				else if (d[k] < 0)
				{
					lo[k] = double.NegativeInfinity;
					hi[k] = o + slack;
				}
				else
				{
					lo[k] = o - slack;
					hi[k] = o + slack;
				}
			}

			Box query = new Box(lo, hi);
			List<int> outIndices = new List<int>();
			this.collider.CollisionsOne(query, int.MaxValue, (_, leaf) => outIndices.Add(this.order[leaf]));
			return outIndices;
		}
	}
}
