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

// GraphSelfCut.cs — port of robust/graph_self_cut.rs, whose header reads:
//
//   Same-mesh narrow phase: decides whether a triangle pair from one operand is
//   ordinary adjacency or a genuine self-intersection that must cut the surface.
//
//   Split out of robust/intersection_graph.rs, whose self-intersection phase
//   calls `real_self_contact` per box pair and prints the `SelfCutStats`
//   counters; robust/soup.rs reuses the same predicate to decide engine
//   dispatch (both reach it through the `intersection_graph` re-exports). The
//   exact tests come from robust/tri_tri.rs and robust/exact/*, and the plain
//   segment/degeneracy helpers from robust/graph_geom.rs.
//
// ── Vec3 equality: `==`, never Equals ────────────────────────────────────────
// Rust's `[Vec3; 3]::contains(&v)` compares with Vec3's derived PartialEq, which
// is IEEE `==` per component — so -0.0 counts as the same shared vertex as 0.0.
// This port's Vec3.Equals is *bit* equality (it backs hashing and welding), and
// -0.0 and 0.0 have different bits. Span.Contains/List.Contains route through
// IEquatable and would therefore be a *different* predicate: a mesh with a -0.0
// coordinate would stop recognizing its own edge-neighbours and start cutting
// them. Contains below spells the `==` out; do not "simplify" it.

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp.Robust
{
	/// <summary>
	/// Per-path counters for the self-cut narrow phase, printed under MANIFOLD_TIMING to
	/// show where box-pair time goes (shortcut hits vs full tri_tri calls and their
	/// outcomes).
	/// </summary>
	/// <remarks>
	/// A class, not a struct: the Rust passes it as <c>&amp;mut SelfCutStats</c> to every
	/// call, and the per-worker merge in <see cref="Add"/> depends on the callee's
	/// increments being visible to the caller.
	/// </remarks>
	internal sealed class SelfCutStats
	{
		/// <summary>Pairs that were exactly the same triangle.</summary>
		public int Identical;

		/// <summary>Edge-neighbour pairs rejected by a shortcut.</summary>
		public int EdgeBenign;

		/// <summary>Vertex-neighbour pairs rejected by a shortcut.</summary>
		public int VertBenign;

		/// <summary>Pairs that reached the full tri_tri narrow phase.</summary>
		public int Full;

		/// <summary>Of those, the ones that did not intersect.</summary>
		public int FullNone;

		/// <summary>Of those, the ones whose contact was a single point.</summary>
		public int FullPoint;

		/// <summary>Of those, the segments that lay on the pair's shared edge.</summary>
		public int FullSegBenign;

		/// <summary>Seconds spent inside the full tri_tri narrow phase.</summary>
		public double FullSecs;

		/// <summary>
		/// Merge a worker's counters. Only diagnostics — <see cref="FullSecs"/> summed
		/// across workers reports aggregate CPU seconds, not wall time.
		/// </summary>
		/// <param name="o">The worker's counters.</param>
		public void Add(SelfCutStats o)
		{
			ArgumentNullException.ThrowIfNull(o);
			this.Identical += o.Identical;
			this.EdgeBenign += o.EdgeBenign;
			this.VertBenign += o.VertBenign;
			this.Full += o.Full;
			this.FullNone += o.FullNone;
			this.FullPoint += o.FullPoint;
			this.FullSegBenign += o.FullSegBenign;
			this.FullSecs += o.FullSecs;
		}
	}

	/// <summary>
	/// The free functions of <c>robust/graph_self_cut.rs</c>: the same-mesh narrow phase
	/// that separates ordinary adjacency from a genuine self-intersection.
	/// </summary>
	internal static class GraphSelfCut
	{
		/// <summary>
		/// Real self-intersection of one triangle pair from the same mesh: the contact of
		/// <paramref name="t1"/> and <paramref name="t2"/> reduced by ordinary mesh
		/// adjacency. Shared-vertex point contacts and (sub-)segments of a shared edge are
		/// the normal way neighboring triangles of a closed mesh touch and yield null;
		/// anything else is a genuine self-intersection whose segments must cut the
		/// surface, so that every emitted piece lies on a single sheet level of its own
		/// mesh (robust/mod.rs classifies own-mesh winding per component).
		/// </summary>
		/// <remarks>
		/// The self-cut narrow phase: a segment list when <paramref name="t1"/> and
		/// <paramref name="t2"/> genuinely intersect (a crossing or a positive-area
		/// coplanar overlap), null when their only contact is ordinary adjacency — a
		/// shared edge, a shared vertex, an isolated point, or an exactly duplicated
		/// triangle. Also used by <c>soup::has_self_intersections</c> to decide engine
		/// dispatch.
		/// </remarks>
		/// <param name="t1">The first triangle's three vertices.</param>
		/// <param name="t2">The second triangle's three vertices.</param>
		/// <param name="stats">Counters, incremented on whichever path is taken.</param>
		/// <returns>The cutting segments, or null for ordinary adjacency.</returns>
		/// <exception cref="ArgumentException">Either array is not exactly three vertices.</exception>
		internal static List<(R3 A, R3 B)>? RealSelfContact(Vec3[] t1, Vec3[] t2, SelfCutStats stats)
		{
			ArgumentNullException.ThrowIfNull(stats);
			if (t1.Length != 3 || t2.Length != 3)
			{
				throw new ArgumentException(
					$"RealSelfContact takes two triangles of three vertices, got {t1.Length} and {t2.Length}");
			}

			// Shared vertex positions (exact f64 identity) between the pair. Kept in
			// f64: hundreds of thousands of benign pairs pass through here, and the
			// rational form is only needed by the final Segment-benign check.
			//
			// Exactly identical triangles (all three vertices coincide — doubled
			// surfaces, which some scans apply to their whole mesh) need no cut:
			// both emit whole pieces with identical interned ids, so they land on
			// one wall in robust/cells.rs and their winding steps add (a doubled
			// sheet steps by two, a fold cancels to zero). Cutting them instead
			// would drag every such triangle through the full arrangement pipeline
			// along its own boundary, for nothing.

			// Adjacency fast paths — the overwhelming bulk of same-mesh box-overlap
			// pairs are edge- or vertex-neighbors whose only contact is that shared
			// simplex, which never needs a cut. All shortcuts are exact (filtered
			// predicates escalate to rationals when uncertain); flat triangulated
			// regions make the *coplanar* neighbor cases as common as the generic
			// ones, and without their 2D shortcuts every such pair pays for a full
			// rational coplanar-overlap clip.
			// Stack-allocated shared-vertex list: this runs per box pair (hundreds
			// of thousands on dense meshes) and a heap Vec here is measurable.
			Span<Vec3> sharedBuf = stackalloc Vec3[3];
			int nShared = 0;
			foreach (Vec3 v in t1)
			{
				if (Contains(t2, v))
				{
					sharedBuf[nShared] = v;
					nShared++;
				}
			}

			ReadOnlySpan<Vec3> sharedF = sharedBuf[..nShared];
			if (sharedF.Length == 3)
			{
				stats.Identical++;
				return null;
			}

			if (sharedF.Length == 2)
			{
				if (TryFindNotIn(t2, t1, out Vec3 opp))
				{
					// Non-coplanar edge-neighbors only meet along the shared edge.
					if (Orient3dPlane(t1, opp) != Sign.Zero)
					{
						stats.EdgeBenign++;
						return null;
					}

					// Coplanar edge-neighbors: benign exactly when the two opposite
					// corners strictly straddle the shared edge's line within the
					// plane (the flat-plate case) — then the closed half-plane
					// intersection is the shared edge itself.
					if (TryFindNotIn(t1, t2, out Vec3 own))
					{
						int axis = DominantAxisF64(t1);
						Sign sOwn = Filtered.Orient2d(
							ProjectF64(sharedF[0], axis),
							ProjectF64(sharedF[1], axis),
							ProjectF64(own, axis));
						Sign sOpp = Filtered.Orient2d(
							ProjectF64(sharedF[0], axis),
							ProjectF64(sharedF[1], axis),
							ProjectF64(opp, axis));
						if (sOwn != Sign.Zero && sOpp != Sign.Zero && sOwn != sOpp)
						{
							stats.EdgeBenign++;
							return null;
						}
					}
				}
			}
			else if (sharedF.Length == 1)
			{
				// Vertex-adjacent: if t2's two non-shared corners lie strictly on
				// one side of t1's plane, the contact is exactly the shared vertex —
				// an isolated point, no cut.
				//
				// The Rust's `[(Vec3, Sign); 3]` is two parallel spans here: a stackalloc
				// of a tuple type would work, but the pair is only ever read field-wise.
				Span<Vec3> otherVerts = stackalloc Vec3[3];
				Span<Sign> otherSigns = stackalloc Sign[3];
				int nOthers = 0;
				foreach (Vec3 v in t2)
				{
					if (!Contains(t1, v))
					{
						otherVerts[nOthers] = v;
						otherSigns[nOthers] = Orient3dPlane(t1, v);
						nOthers++;
					}
				}

				if (nOthers == 2 && otherSigns[0] != Sign.Zero && otherSigns[0] == otherSigns[1])
				{
					stats.VertBenign++;
					return null;
				}

				// Fully coplanar vertex-neighbors (triangle fans on flat regions):
				// an edge through the shared vertex that strictly separates the two
				// triangles certifies the contact is exactly that vertex.
				if (nOthers == 2 && otherSigns[0] == Sign.Zero && otherSigns[1] == Sign.Zero)
				{
					int axis = DominantAxisF64(t1);
					Vec3 v0 = sharedF[0];
					Span<Vec3> ownBuf = stackalloc Vec3[3];
					int nOwn = 0;
					foreach (Vec3 v in t1)
					{
						if (!Contains(t2, v))
						{
							ownBuf[nOwn] = v;
							nOwn++;
						}
					}

					ReadOnlySpan<Vec3> own = ownBuf[..nOwn];
					Vec3 other0 = otherVerts[0];
					Vec3 other1 = otherVerts[1];

					// Candidate separators: each triangle's two edges through v0,
					// tested against its own third corner vs the other triangle's
					// two corners.
					bool Separated(Vec3 ea, Vec3 third, Vec3 far0, Vec3 far1)
					{
						Sign st = Filtered.Orient2d(
							ProjectF64(v0, axis),
							ProjectF64(ea, axis),
							ProjectF64(third, axis));
						if (st == Sign.Zero)
						{
							return false;
						}

						return Opposes(far0) && Opposes(far1);

						bool Opposes(Vec3 f)
						{
							Sign s = Filtered.Orient2d(
								ProjectF64(v0, axis),
								ProjectF64(ea, axis),
								ProjectF64(f, axis));
							return s != Sign.Zero && s != st;
						}
					}

					if (own.Length == 2
						&& (Separated(own[0], own[1], other0, other1)
							|| Separated(own[1], own[0], other0, other1)
							|| Separated(other0, other1, own[0], own[1])
							|| Separated(other1, other0, own[0], own[1])))
					{
						stats.VertBenign++;
						return null;
					}
				}
			}

			stats.Full++;
			Timing.Stopwatch tFull = Timing.Stopwatch.Start();
			TriTriIsect isect = TriTri.TriTriIntersect(t1, t2);
			stats.FullSecs += tFull.ElapsedSecs();
			switch (isect.Kind)
			{
				case TriTriIsectKind.None:
					stats.FullNone++;
					return null;

				// Isolated point contacts (vertex-on-face, edge-through-edge) have
				// zero area on both sides: they never change which sheet a region
				// is on, so they need no cut.
				case TriTriIsectKind.Point:
					stats.FullPoint++;
					return null;

				case TriTriIsectKind.Segment:
				{
					R3 x = isect.P0!;
					R3 y = isect.P1!;
					bool benign = false;
					if (sharedF.Length >= 2)
					{
						R3 s0 = R3.FromVec3(sharedF[0]);
						R3 s1 = R3.FromVec3(sharedF[1]);
						benign = GraphGeom.PointOnSegment(x, s0, s1) && GraphGeom.PointOnSegment(y, s0, s1);
					}

					if (benign)
					{
						stats.FullSegBenign++;
					}

					return benign ? null : new List<(R3 A, R3 B)> { (x, y) };
				}

				// Positive-area coplanar overlap (a fold or doubled patch): cut both
				// triangles along the overlap region's boundary.
				default:
				{
					IReadOnlyList<R3> polygon = isect.Polygon!;
					List<(R3 A, R3 B)> segments = new List<(R3 A, R3 B)>(polygon.Count);
					for (int i = 0; i < polygon.Count; i++)
					{
						segments.Add((polygon[i], polygon[(i + 1) % polygon.Count]));
					}

					return segments;
				}
			}
		}

		/// <summary>
		/// orient3d(t[0], t[1], t[2], v): float filter first, exact integer determinant on
		/// escalation. This replaced a cached exact-plane structure (TriPlane) — with
		/// intpred's division-free fallback, building planes eagerly per triangle cost more
		/// than it ever saved.
		/// </summary>
		/// <param name="t">The plane's triangle.</param>
		/// <param name="v">The query point.</param>
		/// <returns>The exact orientation sign.</returns>
		private static Sign Orient3dPlane(Vec3[] t, Vec3 v)
		{
			Sign? filtered = Approx.Orient3dA(t[0], t[1], t[2], v);
			if (filtered != null)
			{
				return filtered.Value;
			}

			return IntPred.Orient3dI(t[0], t[1], t[2], v);
		}

		/// <summary>
		/// Axis of the largest |component| of the (f64) triangle normal. Only a projection
		/// *choice*: when the exact normal's chosen component happens to be zero, the
		/// projected points go collinear, the exact 2D signs come back Zero, and every
		/// shortcut below falls through — sound, just unoptimized.
		/// </summary>
		/// <param name="t">The triangle.</param>
		/// <returns>0, 1 or 2.</returns>
		private static int DominantAxisF64(Vec3[] t)
		{
			Vec3 n = Cross(t[1] - t[0], t[2] - t[0]);
			double ax = Math.Abs(n.X);
			double ay = Math.Abs(n.Y);
			double az = Math.Abs(n.Z);
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
		/// <see cref="R3.ProjectDrop"/> for raw f64 points (same cyclic axis convention).
		/// </summary>
		/// <param name="v">The point.</param>
		/// <param name="axis">The dropped axis: 0, 1 or 2.</param>
		/// <returns>The projected point.</returns>
		private static Vec2 ProjectF64(Vec3 v, int axis)
		{
			switch (axis)
			{
				case 0:
					return new Vec2(v.Y, v.Z);
				case 1:
					return new Vec2(v.Z, v.X);
				default:
					return new Vec2(v.X, v.Y);
			}
		}

		/// <summary>
		/// Rust's <c>[Vec3; 3]::contains(&amp;v)</c> — IEEE <c>==</c> per component, so
		/// -0.0 matches 0.0. See the file header: <see cref="Vec3.Equals(Vec3)"/> is bit
		/// equality and would answer differently.
		/// </summary>
		/// <param name="t">The vertices to search.</param>
		/// <param name="v">The vertex to find.</param>
		/// <returns>True when some element compares equal.</returns>
		private static bool Contains(ReadOnlySpan<Vec3> t, Vec3 v)
		{
			foreach (Vec3 x in t)
			{
				if (x == v)
				{
					return true;
				}
			}

			return false;
		}

		/// <summary>
		/// Rust's <c>source.iter().find(|v| !exclude.contains(v))</c>: the first vertex of
		/// <paramref name="source"/> that is not shared with <paramref name="exclude"/>.
		/// </summary>
		/// <param name="source">The vertices to scan, in order.</param>
		/// <param name="exclude">The vertices that disqualify a match.</param>
		/// <param name="found">The first unshared vertex, or default.</param>
		/// <returns>True when one was found.</returns>
		private static bool TryFindNotIn(ReadOnlySpan<Vec3> source, ReadOnlySpan<Vec3> exclude, out Vec3 found)
		{
			foreach (Vec3 v in source)
			{
				if (!Contains(exclude, v))
				{
					found = v;
					return true;
				}
			}

			found = default;
			return false;
		}
	}
}
