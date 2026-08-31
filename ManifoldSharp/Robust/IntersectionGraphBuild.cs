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

// IntersectionGraphBuild.cs — port of robust/intersection_graph.rs, whose header
// reads:
//
//   From two triangle soups to classified-ready pieces (paper §6).
//
//   Pipeline stage between the narrow phase (robust/tri_tri.rs) and
//   classification (robust/cells.rs):
//     1. AABB broad phase over the P×Q triangle pairs, exact narrow phase.
//     2. Distribute each pair's intersection primitives to both triangles.
//     3. For coplanar overlaps, cross-copy each side's other primitives
//        (clipped to the overlap region) so both sides subdivide the shared
//        region identically.
//     4. Global registries force a common subdivision of (a) every original
//        mesh edge and (b) every intersection segment, by feeding all split
//        points to every arrangement that sees the same geometry — exact-key
//        edge matching downstream depends on this.
//     5. Build the per-triangle arrangements (robust/arrangement.rs +
//        robust/cdt.rs) and emit `Piece`s: outward-oriented sub-triangles (or
//        whole untouched triangles) tagged with their origin.
//
//   Everything is exact; broad-phase boxes are conservative f64.
//
//   This file holds the build pipeline only. Its vocabulary types live in
//   robust/graph_types.rs (edge keys, `VertInterner`, `Piece`,
//   `IntersectionGraph`), its geometric helpers in robust/graph_geom.rs, and
//   the same-mesh narrow phase in robust/graph_self_cut.rs — all re-exported
//   here so callers keep using `intersection_graph::` paths.
//
// (Here those are GraphTypes.cs, GraphGeom.cs and GraphSelfCut.cs. C# has no
// re-export and does not need one: all four files share the ManifoldSharp.Robust
// namespace, so every name the Rust re-exports is already in scope for callers.)
//
// ── Maps: FxHashMap vs Dictionary ────────────────────────────────────────────
// The Rust imports rustc_hash for every map and set in this file, with the
// standing justification:
//
//   Fx hashing instead of SipHash. Every map/set below is probe-only or has an
//   order-invariant consumer (documented per site); the hasher is unseeded, so
//   even iteration order is stable across runs — output cannot depend on it.
//
// This port swaps them for plain Dictionary/HashSet per the dependency table in
// CLAUDE.md, sound for exactly that reason: every site below either is
// probe-only or feeds an ordered consumer, so .NET leaving enumeration order
// unspecified cannot reach the output. The per-site invariant comments are the
// proof and are kept verbatim — do not delete one while leaving its Dictionary
// behind. The one map this file *does* enumerate is `edgeRegistry`/`segSplits` in
// the memory-release sweep, which only rewrites values.
//
// ── Naming ───────────────────────────────────────────────────────────────────
// The module's free functions land in `IntersectionGraphFunctions`, with the
// `Functions` suffix the porting plan prescribes when the bare module name is
// already a primary type's (GraphTypes.cs's `IntersectionGraph`).
//
// ── `&[[Vec3; 3]]` is a length-checked Vec3[][] ──────────────────────────────
// Same call TriTri.cs and GraphGeom.cs made for the same reason: a C# array
// carries its length at run time instead of in its type, and Soup.ImplToTris
// hands soups out as Vec3[][] already, so the check is at the boundary and the
// inner loops pass through.

using System.Globalization;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

// The Rust type aliases, spelled out for this file as GraphTypes.cs spells them
// out for its own (C# aliases are file-scoped, so they cannot be shared).
using BitEdgeKey = ((ulong X, ulong Y, ulong Z) A, (ulong X, ulong Y, ulong Z) B);
using EdgeKey = (uint A, uint B);
using GeoEdgeKey = (uint A, uint B);

namespace ManifoldSharp.Robust
{
	/// <summary>
	/// The free functions of <c>robust/intersection_graph.rs</c>: the build pipeline that
	/// turns two triangle soups into the classified-ready
	/// <see cref="IntersectionGraph"/>. Named with the <c>Functions</c> suffix because
	/// the module's output type already owns the bare name.
	/// </summary>
	public static partial class IntersectionGraphFunctions
	{
		/// <summary>
		/// Build the intersection graph for soups <paramref name="p"/> and
		/// <paramref name="q"/> (each triangle wound outward; degenerate triangles are
		/// dropped here, paper §5).
		/// </summary>
		/// <param name="p">The first operand's triangles.</param>
		/// <param name="q">The second operand's triangles.</param>
		/// <returns>The intersection graph.</returns>
		public static IntersectionGraph BuildGraph(Vec3[][] p, Vec3[][] q)
		{
			return BuildGraphWithToken(p, q, null)
				?? throw new InvalidOperationException("uncancellable build_graph cannot cancel");
		}

		/// <summary>
		/// <see cref="BuildGraph"/> with cooperative cancellation. Returns null when the
		/// token fires.
		/// </summary>
		/// <remarks>
		/// Checks run per triangle in every phase and inside the arrangement sweeps —
		/// heavily self-intersecting inputs spend minutes in per-triangle quadratic loops,
		/// and a cancel that only top-level phases notice can overshoot its deadline by
		/// that much (Thingi10K #42211 ran 565 s past a 60 s cancel before this plumbing).
		/// </remarks>
		/// <param name="p">The first operand's triangles.</param>
		/// <param name="q">The second operand's triangles.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The intersection graph, or null when cancelled.</returns>
		public static IntersectionGraph? BuildGraphWithToken(Vec3[][] p, Vec3[][] q, CancelToken? token)
		{
			return BuildGraphWithProgress(p, q, token, null);
		}

		/// <summary>
		/// <see cref="BuildGraphWithToken"/> that also reports its five phases to
		/// <paramref name="progress"/> (see <see cref="ProgressReporter"/>). Null is exactly
		/// <see cref="BuildGraphWithToken"/>: no counter is touched and no branch is taken
		/// inside any inner loop.
		/// </summary>
		/// <param name="p">The first operand's triangles.</param>
		/// <param name="q">The second operand's triangles.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>The intersection graph, or null when cancelled.</returns>
		/// <exception cref="ArgumentException">Some triangle is not exactly three vertices.</exception>
		public static IntersectionGraph? BuildGraphWithProgress(
			Vec3[][] p,
			Vec3[][] q,
			CancelToken? token,
			ProgressReporter? progress)
		{
			RequireTriangles(p, nameof(p));
			RequireTriangles(q, nameof(q));
			long? tAll = Timing.Start();
			Vec3[][][] meshes = new Vec3[][][] { p, q };
			bool[][] live = new bool[2][];
			live[0] = new bool[p.Length];
			live[1] = new bool[q.Length];
			for (int i = 0; i < p.Length; i++)
			{
				live[0][i] = !GraphGeom.IsDegenerate(p[i]);
			}

			for (int i = 0; i < q.Length; i++)
			{
				live[1][i] = !GraphGeom.IsDegenerate(q[i]);
			}

			// 1. Broad + narrow phase. The broad phase is a BVH (the same Collider
			// the exact engine uses) over Q's triangle boxes, queried with each P
			// triangle's box — O((|P|+|Q|)·log|Q|) instead of the all-pairs box
			// sweep. Candidates are re-sorted to ascending qi per pi, so the pair
			// provenance ids match the exhaustive loop exactly (only genuinely
			// intersecting pairs consume an id, and the exact narrow phase decides
			// those identically regardless of broad-phase method).
			Box[] pBoxes = new Box[p.Length];
			for (int i = 0; i < p.Length; i++)
			{
				pBoxes[i] = GraphGeom.TriBox(p[i]);
			}

			Box[] qBoxes = new Box[q.Length];
			for (int i = 0; i < q.Length; i++)
			{
				qBoxes[i] = GraphGeom.TriBox(q[i]);
			}

			Box sceneBox = new Box();
			for (int qi = 0; qi < qBoxes.Length; qi++)
			{
				if (live[1][qi])
				{
					sceneBox = sceneBox.UnionBox(qBoxes[qi]);
				}
			}

			// OrderBy, not List.Sort: Rust's `sort_by_key` is a stable sort and morton
			// codes collide freely, so an introsort could permute equal-code triangles
			// and change every downstream provenance id.
			Box qSceneBox = sceneBox;
			int[] qOrder = Enumerable.Range(0, q.Length)
				.Where(qi => live[1][qi])
				.OrderBy(qi => Sort.MortonCode(qBoxes[qi].Center(), qSceneBox))
				.ToArray();
			Box[] leafBoxes = new Box[qOrder.Length];
			uint[] leafMorton = new uint[qOrder.Length];
			for (int i = 0; i < qOrder.Length; i++)
			{
				leafBoxes[i] = qBoxes[qOrder[i]];
				leafMorton[i] = Sort.MortonCode(qBoxes[qOrder[i]].Center(), qSceneBox);
			}

			Collider collider = new Collider(leafBoxes, leafMorton);

			// Per-(mesh, tri) primitive lists; provenance = pair index.
			TriPrims[][] prims = new TriPrims[2][];
			prims[0] = new TriPrims[p.Length];
			prims[1] = new TriPrims[q.Length];
			for (int m = 0; m < 2; m++)
			{
				for (int i = 0; i < prims[m].Length; i++)
				{
					prims[m][i] = new TriPrims();
				}
			}

			// Coplanar overlap regions per pair, for the cross-copy step:
			// (p_tri, q_tri, polygon).
			List<(int Pi, int Qi, IReadOnlyList<R3> Poly)> coplanarRegions
				= new List<(int Pi, int Qi, IReadOnlyList<R3> Poly)>();
			bool anyIntersections = false;
			int pairCount = 0;

			List<int> candidatesQ = new List<int>();
			Progress.BeginPhase(progress, Phase.NarrowPhase, (ulong)p.Length);
			for (int pi = 0; pi < p.Length; pi++)
			{
				if (Cancel.IsCancelled(token))
				{
					return null;
				}

				progress?.Advance(1);
				if (!live[0][pi])
				{
					continue;
				}

				candidatesQ.Clear();
				collider.CollisionsOne(pBoxes[pi], pi, (_, leaf) => candidatesQ.Add(qOrder[leaf]));

				// Rust `sort_unstable` on plain triangle indices: the BVH reports each
				// leaf at most once per query, so there are no ties to break and the
				// unstable introsort List.Sort uses is exactly it.
				candidatesQ.Sort();
				foreach (int qi in candidatesQ)
				{
					Vec3[] qt = q[qi];
					if (!pBoxes[pi].DoesOverlapBox(qBoxes[qi]))
					{
						continue;
					}

					TriTriIsect isect = TriTri.TriTriIntersect(p[pi], qt);
					int pair = pairCount;
					switch (isect.Kind)
					{
						case TriTriIsectKind.None:
							continue;
						case TriTriIsectKind.Point:
							prims[0][pi].Points.Add((isect.P0!, pair));
							prims[1][qi].Points.Add((isect.P0!, pair));
							break;
						case TriTriIsectKind.Segment:
							prims[0][pi].Segments.Add((isect.P0!, isect.P1!, pair));
							prims[1][qi].Segments.Add((isect.P0!, isect.P1!, pair));
							break;
						case TriTriIsectKind.Coplanar:
							IReadOnlyList<R3> polygon = isect.Polygon!;
							for (int i = 0; i < polygon.Count; i++)
							{
								R3 a = polygon[i];
								R3 b = polygon[(i + 1) % polygon.Count];
								prims[0][pi].Segments.Add((a, b, pair));
								prims[1][qi].Segments.Add((a, b, pair));
							}

							coplanarRegions.Add((pi, qi, polygon));
							break;

						// The Rust's `match` is exhaustive over TriTriIsect's four
						// variants; C# cannot prove a TriTriIsectKind holds a declared
						// value, so the unreachable arm says so loudly rather than
						// routing a hypothetical fifth variant into the coplanar path
						// (same call CsgTree.cs makes on OpType).
						default:
							throw new ArgumentOutOfRangeException(
								nameof(isect),
								$"unhandled TriTriIsectKind: {isect.Kind}");
					}

					anyIntersections = true;
					pairCount++;
				}
			}

			// All |P| units spent, cancel-free: close the bar at 1.0, which the throttle
			// alone cannot — ProgressReporter.CompletePhase's remarks say why, for all five.
			Progress.CompletePhase(progress);
			Timing.Print("robust: pair narrow phase", tAll);
			long? tSelf = Timing.Start();

			// 2b. Self-intersections: cut each mesh along its own P×P / Q×Q contact
			// segments (beyond ordinary adjacency). Without these cuts a piece could
			// straddle a fold of a self-overlapping operand, making "is this piece
			// an interior wall of its own solid" ill-defined; with them, both
			// winding numbers the classification needs are constant per flood-fill
			// component (robust/propagate.rs never crosses constraint edges).
			// Broad phase: per-mesh BVH, same approach as the cross-mesh loop above
			// (candidates re-sorted so provenance ids stay deterministic).
			// Widen first, then add: Rust's `(p.len() + q.len()) as u64` sums in usize,
			// which is 64-bit on every target this runs on, so summing in `int` here
			// would be the narrower arithmetic, not the same one.
			Progress.BeginPhase(progress, Phase.SelfIntersections, (ulong)p.Length + (ulong)q.Length);
			for (int m = 0; m < 2; m++)
			{
				if (!CutSelfIntersections(
					m,
					m == 0 ? p : q,
					m == 0 ? pBoxes : qBoxes,
					live[m],
					prims[m],
					ref pairCount,
					token,
					progress))
				{
					return null;
				}
			}

			// Both meshes cut, so all |P| + |Q| units are spent.
			Progress.CompletePhase(progress);

			Timing.Print("robust: self-intersection cuts", tSelf);
			long? tCross = Timing.Start();

			// 3. Cross-copy primitives through coplanar overlap regions so both
			// sides see identical geometry inside the shared area. Clip against the
			// region to avoid dragging unrelated geometry across.
			foreach ((int pi, int qi, IReadOnlyList<R3> poly) in coplanarRegions)
			{
				if (Cancel.IsCancelled(token))
				{
					return null;
				}

				// Both snapshots are taken BEFORE either copy runs: the Rust clones for
				// the borrow checker, but the pre-copy state is also the semantics —
				// the second copy must not see what the first one just added.
				TriPrims fromP = prims[0][pi].Clone();
				TriPrims fromQ = prims[1][qi].Clone();
				CopyThroughRegion(fromP, prims[1][qi], poly);
				CopyThroughRegion(fromQ, prims[0][pi], poly);
			}

			Timing.Print("robust: coplanar cross-copy", tCross);
			long? tCand = Timing.Start();

			// 4a. Candidate points per intersected triangle. Pure per triangle, so
			// the map parallelizes under the bit-identical rule: results land in
			// worklist order regardless of schedule.
			//
			// Candidates are interned into a build-local PointTable as they land
			// and kept as `uint` ids from here on. Everything downstream of this point
			// (the two registry sweeps, their hit lists, their dedup sets and their
			// keys) then moves ids instead of rational triples, which is what makes
			// million-split meshes fit in memory — see PointTable's own comment.
			List<uint>?[][] candidates = new List<uint>?[2][];
			candidates[0] = new List<uint>?[p.Length];
			candidates[1] = new List<uint>?[q.Length];
			PointTable ptab = new PointTable();
			List<(int M, int Ti)> candWork = new List<(int M, int Ti)>();
			for (int m = 0; m < 2; m++)
			{
				for (int ti = 0; ti < meshes[m].Length; ti++)
				{
					TriPrims pr = prims[m][ti];
					if (pr.Points.Count != 0 || pr.Segments.Count != 0)
					{
						candWork.Add((m, ti));
					}
				}
			}

			Progress.BeginPhase(progress, Phase.CandidatePoints, (ulong)candWork.Count);

			// Swept in chunks: a parallel map over the whole worklist would hold
			// every triangle's rational candidate list alive at once (12.6 M points
			// on Thingi10K #252784), whereas interning after each chunk keeps only
			// one copy per distinct point. Chunk boundaries are pure batching — the
			// closure, the result order and the interning order are unchanged, so
			// the ids (and everything downstream) are identical to the one-shot map.
			const int CandChunk = 1 << 16;
			int nCandTotal = 0;
			int baseIndex = 0;
			while (baseIndex < candWork.Count)
			{
				int len = Math.Min(CandChunk, candWork.Count - baseIndex);
				int chunkBase = baseIndex;
				List<R3>?[]? candResults = Progress.MaybeParMapCtProgress(len, 16, token, progress, i =>
				{
					(int m, int ti) = candWork[chunkBase + i];
					TriPrims pr = prims[m][ti];
					ArrangementInput input = new ArrangementInput
					{
						Points = new List<(R3 Point, int Prov)>(pr.Points),
						Segments = new List<(R3 A, R3 B, int Prov)>(pr.Segments),
					};
					return ArrangementFunctions.CandidatePoints(meshes[m][ti], input, token);
				});
				if (candResults is null)
				{
					return null;
				}

				for (int k = 0; k < len; k++)
				{
					if (k % 1024 == 0 && Cancel.IsCancelled(token))
					{
						return null;
					}

					(int m, int ti) = candWork[chunkBase + k];
					List<R3>? cands = candResults[k];
					if (cands is null)
					{
						return null;
					}

					nCandTotal += cands.Count;
					List<uint> ids = new List<uint>(cands.Count);
					foreach (R3 pt in cands)
					{
						ids.Add(ptab.Intern(pt));
					}

					candidates[m][ti] = ids;
				}

				baseIndex += len;
			}

			// Every chunk mapped and interned; the endpoint sweep below is not this phase.
			Progress.CompletePhase(progress);

			// Intersection-segment endpoints share the id space, so the segment
			// registry keys on `(uint, uint)` too. Flat per-mesh arrays (offsets +
			// endpoint-id pairs) mirror `prims[m][ti].Segments` exactly; phases 4c
			// and 5 read the ids instead of rebuilding rational keys per probe.
			List<int>[] segOff = new List<int>[] { new List<int>(), new List<int>() };
			List<(uint A, uint B)>[] segEnds
				= new List<(uint A, uint B)>[] { new List<(uint A, uint B)>(), new List<(uint A, uint B)>() };
			for (int m = 0; m < 2; m++)
			{
				segOff[m].Capacity = meshes[m].Length + 1;
				segOff[m].Add(0);
				for (int ti = 0; ti < meshes[m].Length; ti++)
				{
					if (ti % 1024 == 0 && Cancel.IsCancelled(token))
					{
						return null;
					}

					foreach ((R3 a, R3 b, int _) in prims[m][ti].Segments)
					{
						uint ia = ptab.Intern(a);
						uint ib = ptab.Intern(b);
						segEnds[m].Add((ia, ib));
					}

					segOff[m].Add(segEnds[m].Count);
				}
			}

			// One exact point and one rounded approximation per id, shared by every
			// triangle that sees it (both registry sweeps used to re-round every
			// candidate per triangle).
			R3[] pts = ptab.Resolve();
			Vec3[] ptsA = new Vec3[pts.Length];
			for (int i = 0; i < pts.Length; i++)
			{
				ptsA[i] = GraphGeom.Approx3(pts[i]);
			}

			Timing.Print("robust: candidate points", tCand);
			Timing.PrintCount(string.Create(
				CultureInfo.InvariantCulture,
				$"robust: candidate points: {nCandTotal} total, {ptab.Count} interned (incl. segment endpoints), {segEnds[0].Count + segEnds[1].Count} segment instances"));

			long? tReg = Timing.Start();

			// 4b. Original-edge registry: split points on each mesh edge (geometric
			// identity — soups have no reliable connectivity). Bit-keyed: original
			// edges join exact f64 vertices.
			// Both registry sweeps are pure per triangle; workers collect local
			// (key, point) hits and the single-threaded merge inserts them in
			// worklist order. Registry values are sets, so content is
			// order-independent and the merge order only preserves determinism of
			// allocation, not meaning.
			List<(int M, int Ti)> regWork = new List<(int M, int Ti)>();
			for (int m = 0; m < 2; m++)
			{
				for (int ti = 0; ti < meshes[m].Length; ti++)
				{
					if (candidates[m][ti] is not null)
					{
						regWork.Add((m, ti));
					}
				}
			}

			// Two sweeps (original edges, then intersection segments) over the same
			// worklist, so the phase total counts it twice. Widen first, then double:
			// Rust's `2 * reg_work.len() as u64` doubles in u64, so doubling in `int`
			// here would be the narrower arithmetic, not the same one.
			Progress.BeginPhase(progress, Phase.Registries, 2 * (ulong)regWork.Count);

			// Order invariance: the registry is only ever probed, and its points land
			// in a SortedSet<R3> (`extra`) below, which sorts.
			Dictionary<BitEdgeKey, SplitIds>[] edgeRegistry = new Dictionary<BitEdgeKey, SplitIds>[]
			{
				new Dictionary<BitEdgeKey, SplitIds>(),
				new Dictionary<BitEdgeKey, SplitIds>(),
			};
			List<(BitEdgeKey Key, uint Id)>[]? edgeHits = Progress.MaybeParMapCtProgress(
				regWork.Count,
				16,
				token,
				progress,
				i =>
				{
					(int m, int ti) = regWork[i];
					List<uint> cands = candidates[m][ti] ?? throw new InvalidOperationException("filtered to Some");
					return EdgeHits(cands, meshes[m][ti], pts, ptsA);
				});
			if (edgeHits is null)
			{
				return null;
			}

			for (int k = 0; k < regWork.Count; k++)
			{
				if (k % 1024 == 0 && Cancel.IsCancelled(token))
				{
					return null;
				}

				int m = regWork[k].M;
				foreach ((BitEdgeKey key, uint id) in edgeHits[k])
				{
					if (!edgeRegistry[m].TryGetValue(key, out SplitIds? e))
					{
						e = new SplitIds();
						edgeRegistry[m].Add(key, e);
					}

					if (e.Seen.Add(id))
					{
						e.Ids.Add(id);
					}
				}
			}

			int nEdgeHits = 0;
			foreach (List<(BitEdgeKey Key, uint Id)> h in edgeHits)
			{
				nEdgeHits += h.Count;
			}

			edgeHits = null;

			// 4c. Intersection-segment registry: for every pair segment, gather the
			// split points both sides know about.
			// Keyed on the segment's two endpoint ids: the map is only ever probed,
			// never iterated, and rational keys cost both the compare (ordered map) or
			// hash (R3Key) per probe and a cloned rational triple per segment instance.
			// Same order-invariance argument as `edgeRegistry`: probe-only, and the
			// points it hands out are re-sorted through `extra: SortedSet<R3>`.
			Dictionary<GeoEdgeKey, SplitIds> segSplits = new Dictionary<GeoEdgeKey, SplitIds>();
			List<(GeoEdgeKey Key, uint Id)>[]? splitHits = Progress.MaybeParMapCtProgress(
				regWork.Count,
				16,
				token,
				progress,
				i =>
				{
					(int m, int ti) = regWork[i];
					List<uint> cands = candidates[m][ti] ?? throw new InvalidOperationException("filtered to Some");
					return SplitHits(cands, segEnds[m], segOff[m][ti], segOff[m][ti + 1], pts, ptsA);
				});
			if (splitHits is null)
			{
				return null;
			}

			for (int k = 0; k < splitHits.Length; k++)
			{
				if (k % 1024 == 0 && Cancel.IsCancelled(token))
				{
					return null;
				}

				foreach ((GeoEdgeKey key, uint id) in splitHits[k])
				{
					if (!segSplits.TryGetValue(key, out SplitIds? e))
					{
						e = new SplitIds();
						segSplits.Add(key, e);
					}

					if (e.Seen.Add(id))
					{
						e.Ids.Add(id);
					}
				}
			}

			int nSplitHits = 0;
			foreach (List<(GeoEdgeKey Key, uint Id)> h in splitHits)
			{
				nSplitHits += h.Count;
			}

			splitHits = null;

			// Both sweeps mapped and merged: the whole 2 × |regWork| total, cancel-free.
			Progress.CompletePhase(progress);

			// The dedup sets have done their job; the arrangement phase below reads
			// only the id lists. Releasing them (and the candidate lists, which no
			// later phase touches) before phase 5 keeps the two peaks from stacking.
			// Each step frees a set and reallocates a vector; millions of registry
			// entries make even that a measurable stretch of uninterruptible work.
			for (int m = 0; m < 2; m++)
			{
				int k = 0;
				foreach (SplitIds v in edgeRegistry[m].Values)
				{
					if (k % 4096 == 0 && Cancel.IsCancelled(token))
					{
						return null;
					}

					k++;
					v.Seen = new HashSet<uint>();
					v.Ids.TrimExcess();
				}
			}

			{
				int k = 0;
				foreach (SplitIds v in segSplits.Values)
				{
					if (k % 4096 == 0 && Cancel.IsCancelled(token))
					{
						return null;
					}

					k++;
					v.Seen = new HashSet<uint>();
					v.Ids.TrimExcess();
				}
			}

			// Rust's `drop(candidates); drop(pts_a);`. C# locals captured by the
			// closures above live in one shared display class, so clearing the
			// references here is what actually frees what the Rust's `drop` frees.
			candidates = null!;
			ptsA = null!;

			Timing.Print("robust: split registries", tReg);
			Timing.PrintCount(string.Create(
				CultureInfo.InvariantCulture,
				$"robust: split registries: {nEdgeHits} edge hits over {edgeRegistry[0].Count + edgeRegistry[1].Count} edges, {nSplitHits} segment hits over {segSplits.Count} segments"));
			long? tArr = Timing.Start();

			// 5. Build arrangements and emit pieces. The per-triangle arrangement
			// (registry probes, CDT, crossings) is pure and runs in parallel; the
			// interner is order-sensitive, so interning and piece emission replay
			// the results strictly in worklist order — outputs are bit-identical to
			// the sequential build.
			List<(int M, int Ti)> arrWork = new List<(int M, int Ti)>();
			for (int m = 0; m < 2; m++)
			{
				for (int ti = 0; ti < meshes[m].Length; ti++)
				{
					if (live[m][ti])
					{
						arrWork.Add((m, ti));
					}
				}
			}

			Progress.BeginPhase(progress, Phase.Arrangements, (ulong)arrWork.Count);
			TriResult?[]? arrResults = Progress.MaybeParMapCtProgress(
				arrWork.Count,
				16,
				token,
				progress,
				i =>
				{
					(int m, int ti) = arrWork[i];
					return ArrangeTriangle(
						meshes[m][ti],
						prims[m][ti],
						edgeRegistry[m],
						segSplits,
						segEnds[m],
						segOff[m][ti],
						segOff[m][ti + 1],
						pts,
						token);
				});
			if (arrResults is null)
			{
				return null;
			}

			List<Piece> pieces = new List<Piece>();

			// Membership-only set (never iterated by this assembly); order-invariant.
			HashSet<EdgeKey> isectEdges = new HashSet<EdgeKey>();
			VertInterner interner = new VertInterner();
			for (int k = 0; k < arrWork.Count; k++)
			{
				if (k % 1024 == 0 && Cancel.IsCancelled(token))
				{
					return null;
				}

				(int m, int ti) = arrWork[k];
				Vec3[] t = meshes[m][ti];
				TriResult? result = arrResults[k];
				if (result is null)
				{
					return null;
				}

				Arrangement? arr = result.Value.Arranged;
				if (arr is null)
				{
					// Untouched triangle → whole piece, interned by f64 bits.
					pieces.Add(new Piece(
						(byte)m,
						ti,
						new UVec3(
							interner.InternF64(t[0]),
							interner.InternF64(t[1]),
							interner.InternF64(t[2]))));
				}
				else
				{
					// Intern each arrangement point once; sub-triangles and
					// constraint edges then only shuffle ids.
					uint[] ids = new uint[arr.Points3.Count];
					for (int i = 0; i < ids.Length; i++)
					{
						ids[i] = interner.Intern(arr.Points3[i]);
					}

					foreach ((int u, int w) in arr.Constraints.Keys)
					{
						isectEdges.Add(GraphTypes.EdgeKey(ids[u], ids[w]));
					}

					foreach (IVec3 st in arr.Tris)
					{
						(int a, int b, int c) = (st.X, st.Y, st.Z);
						UVec3 vi = arr.Flipped
							? new UVec3(ids[a], ids[c], ids[b])
							: new UVec3(ids[a], ids[b], ids[c]);
						pieces.Add(new Piece((byte)m, ti, vi));
					}
				}
			}

			// The map spent every unit and the interning replay finished it, cancel-free.
			Progress.CompletePhase(progress);

			Timing.Print("robust: arrangements", tArr);
			Timing.PrintCount(string.Create(
				CultureInfo.InvariantCulture,
				$"robust: arrangement phases: {ArrangementFunctions.Stats.SnapshotAndReset()}"));

			return new IntersectionGraph
			{
				Pieces = pieces,
				Verts = interner.Verts,
				VertsF64 = interner.VertsF64,
				IsectEdges = isectEdges,
				AnyIntersections = anyIntersections,
			};
		}
	}
}
