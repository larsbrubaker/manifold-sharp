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

// IntersectionGraphBuild.Types.cs — the build pipeline's private vocabulary, its
// self-cut phase, and its three pure per-triangle kernels. The module header is on
// IntersectionGraphBuild.cs.
//
// ── Why the kernels are methods and not lambdas ──────────────────────────────
// The Rust writes phases 4b, 4c and 5 as closures handed to
// `maybe_par_map_ct_progress`. Each closure body is already a pure function of
// one triangle — that purity is exactly what the Rust's own comments cite as the
// licence to parallelize them — so lifting each one to a static method changes
// nothing about what runs, and it keeps the single 600-line
// `BuildGraphWithProgress` inside the repo's 800-line file cap. The call sites
// pass what the closures captured; nothing else moved.
//
// `CutSelfIntersections` is one `for m in 0..2` iteration of phase 2b, lifted
// for the same reason and by the same rule: its body is unchanged, and the one
// piece of build-wide state it touches (the provenance counter) is threaded
// through as a `ref` parameter rather than copied.

using System.Globalization;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

// The Rust type aliases, spelled out for this file as GraphTypes.cs spells them
// out for its own (C# aliases are file-scoped, so they cannot be shared).
using BitEdgeKey = ((ulong X, ulong Y, ulong Z) A, (ulong X, ulong Y, ulong Z) B);
using GeoEdgeKey = (uint A, uint B);

namespace ManifoldSharp.Robust
{
	public static partial class IntersectionGraphFunctions
	{
		/// <summary>
		/// A pair's primitives after distribution: segments (including coplanar
		/// boundary edges) and isolated points.
		/// </summary>
		private sealed class TriPrims
		{
			public List<(R3 Point, int Prov)> Points = new List<(R3 Point, int Prov)>();

			public List<(R3 A, R3 B, int Prov)> Segments = new List<(R3 A, R3 B, int Prov)>();

			/// <summary>Rust's derived <c>Clone</c>: fresh lists over the same points.</summary>
			public TriPrims Clone()
			{
				return new TriPrims
				{
					Points = new List<(R3 Point, int Prov)>(this.Points),
					Segments = new List<(R3 A, R3 B, int Prov)>(this.Segments),
				};
			}
		}

		/// <summary>
		/// A split registry's value: Rust's <c>(Vec&lt;u32&gt;, HashSet&lt;u32&gt;)</c> — the
		/// id list in discovery order, plus the probe set that dedups it.
		/// </summary>
		/// <remarks>
		/// Registry values are (ids, seen) rather than an ordered set: dedup is a uint hash
		/// probe, so the sequential merge never pays ordered rational comparisons, and the
		/// consuming <c>extra</c> sets see each point exactly once — the same content a
		/// BTreeSet gave. Dedup by id is dedup by exact value: <see cref="PointTable"/> is
		/// injective on the same equality (<c>R3Eq</c> on canonical rationals) the
		/// <c>R3Key</c> sets used.
		/// </remarks>
		private sealed class SplitIds
		{
			public List<uint> Ids = new List<uint>();

			public HashSet<uint> Seen = new HashSet<uint>();
		}

		/// <summary>
		/// Rust's <c>enum TriResult { Untouched, Arranged(Arrangement) }</c>. A struct with a
		/// null <see cref="Arranged"/> for the untouched case, so the map's
		/// <c>Option&lt;TriResult&gt;</c> element costs no allocation on the common path.
		/// </summary>
		private readonly struct TriResult
		{
			/// <summary>The arrangement, or null for Rust's <c>Untouched</c>.</summary>
			public readonly Arrangement? Arranged;

			public TriResult(Arrangement? arranged)
			{
				this.Arranged = arranged;
			}
		}

		/// <summary>
		/// Phase 2b for one mesh: cut it along its own contact segments (beyond ordinary
		/// adjacency), appending them to <paramref name="prims"/> with fresh provenance ids.
		/// </summary>
		/// <remarks>
		/// The exact narrow phase per triangle is pure; workers return each triangle's
		/// (j, segments) contacts and per-worker stats, and the sequential merge assigns
		/// provenance pair ids in (i, j) order — identical to the sequential sweep.
		/// <para>
		/// One <c>for m in 0..2</c> iteration of the Rust's phase 2b, lifted to a method
		/// only so the pipeline file stays inside the 800-line cap; the body is unchanged
		/// and <paramref name="pairCount"/> keeps flowing through the whole build.
		/// </para>
		/// </remarks>
		/// <param name="mesh">0 for P, 1 for Q — the id the timing lines report.</param>
		/// <param name="tris">This mesh's triangles.</param>
		/// <param name="boxes">Their bounding boxes.</param>
		/// <param name="live">Which triangles survived the degeneracy filter.</param>
		/// <param name="prims">This mesh's per-triangle primitive lists, appended to.</param>
		/// <param name="pairCount">The build-wide provenance counter.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>False when the token fired.</returns>
		private static bool CutSelfIntersections(
			int mesh,
			Vec3[][] tris,
			Box[] boxes,
			bool[] live,
			TriPrims[] prims,
			ref int pairCount,
			CancelToken? token,
			ProgressReporter? progress)
		{
			Box selfScene = new Box();
			for (int i = 0; i < boxes.Length; i++)
			{
				if (live[i])
				{
					selfScene = selfScene.UnionBox(boxes[i]);
				}
			}

			// OrderBy, not Array.Sort: Rust's `sort_by_key` is stable and morton codes
			// collide freely (see the same call in the cross-mesh broad phase).
			Box selfSceneBox = selfScene;
			int[] order = Enumerable.Range(0, tris.Length)
				.Where(i => live[i])
				.OrderBy(i => Sort.MortonCode(boxes[i].Center(), selfSceneBox))
				.ToArray();
			Box[] selfLeafBoxes = new Box[order.Length];
			uint[] selfLeafMorton = new uint[order.Length];
			for (int i = 0; i < order.Length; i++)
			{
				selfLeafBoxes[i] = boxes[order[i]];
				selfLeafMorton[i] = Sort.MortonCode(boxes[order[i]].Center(), selfSceneBox);
			}

			Collider selfCollider = new Collider(selfLeafBoxes, selfLeafMorton);
			int nPairs = 0;
			int nCut = 0;
			SelfCutStats stats = new SelfCutStats();
			(List<(int J, List<(R3 A, R3 B)> Segs)> Contacts, SelfCutStats Local, int LocalPairs)[]? contactResults
				= Progress.MaybeParMapCtProgress(tris.Length, 64, token, progress, i =>
				{
					SelfCutStats local = new SelfCutStats();
					List<(int J, List<(R3 A, R3 B)> Segs)> contacts
						= new List<(int J, List<(R3 A, R3 B)> Segs)>();
					int localPairs = 0;
					if (live[i])
					{
						List<int> cands = new List<int>();
						selfCollider.CollisionsOne(boxes[i], i, (_, leaf) => cands.Add(order[leaf]));

						// Rust `sort_unstable` on plain triangle indices: the BVH reports
						// each leaf at most once per query, so there are no ties to break
						// and the unstable introsort List.Sort uses is exactly it.
						cands.Sort();
						foreach (int j in cands)
						{
							if (j <= i || !boxes[i].DoesOverlapBox(boxes[j]))
							{
								continue;
							}

							localPairs++;
							List<(R3 A, R3 B)>? segs = GraphSelfCut.RealSelfContact(tris[i], tris[j], local);
							if (segs is not null)
							{
								contacts.Add((j, segs));
							}
						}
					}

					return (contacts, local, localPairs);
				});
			if (contactResults is null)
			{
				return false;
			}

			for (int i = 0; i < contactResults.Length; i++)
			{
				(List<(int J, List<(R3 A, R3 B)> Segs)> contacts, SelfCutStats local, int localPairs)
					= contactResults[i];
				nPairs += localPairs;
				stats.Add(local);
				foreach ((int j, List<(R3 A, R3 B)> segs) in contacts)
				{
					nCut++;
					foreach ((R3 x, R3 y) in segs)
					{
						int pair = pairCount;
						prims[i].Segments.Add((x, y, pair));
						prims[j].Segments.Add((x, y, pair));
						pairCount++;
					}
				}
			}

			Timing.PrintCount(string.Create(
				CultureInfo.InvariantCulture,
				$"robust: self-cut mesh {mesh}: {nPairs} box pairs, {nCut} cutting"));
			Timing.PrintCount(string.Create(
				CultureInfo.InvariantCulture,
				$"robust: self-cut mesh {mesh} tri_tri exits: {TriTri.Stats.SnapshotAndReset()}"));
			Timing.PrintCount(string.Create(
				CultureInfo.InvariantCulture,
				$"robust: self-cut mesh {mesh} paths: identical {stats.Identical}, edge-benign {stats.EdgeBenign}, vert-benign {stats.VertBenign}, full {stats.Full} ({stats.FullSecs:F3}s: none {stats.FullNone}, point {stats.FullPoint}, seg-benign {stats.FullSegBenign})"));
			return true;
		}

		/// <summary>
		/// Phase 4b's kernel: the split points of one triangle that land on one of its own
		/// original mesh edges (geometric identity — soups have no reliable connectivity).
		/// Bit-keyed: original edges join exact f64 vertices.
		/// </summary>
		/// <param name="cands">This triangle's candidate point ids.</param>
		/// <param name="t">The triangle.</param>
		/// <param name="pts">Exact point per id.</param>
		/// <param name="ptsA">Rounded approximation per id.</param>
		/// <returns>The (edge key, point id) hits, in edge then candidate order.</returns>
		private static List<(BitEdgeKey Key, uint Id)> EdgeHits(
			List<uint> cands,
			Vec3[] t,
			R3[] pts,
			Vec3[] ptsA)
		{
			R3[] corners = new R3[]
			{
				R3.FromVec3(t[0]),
				R3.FromVec3(t[1]),
				R3.FromVec3(t[2]),
			};
			Vec3[] ca = new Vec3[] { t[0], t[1], t[2] };
			List<(BitEdgeKey Key, uint Id)> hits = new List<(BitEdgeKey Key, uint Id)>();
			for (int e = 0; e < 3; e++)
			{
				R3 a = corners[e];
				R3 b = corners[(e + 1) % 3];
				BitEdgeKey key = GraphTypes.BitEdgeKey(t[e], t[(e + 1) % 3]);
				(Vec3 Lo, Vec3 Hi) sbox = GraphGeom.SegBox3(ca[e], ca[(e + 1) % 3]);
				foreach (uint id in cands)
				{
					R3 pt = pts[id];
					Vec3 ptA = ptsA[id];

					// A point on the edge lies inside its inflated box; the
					// reject skips the exact comparisons for everything else.
					if (!GraphGeom.Box3Contains(sbox, ptA))
					{
						continue;
					}

					if (!Rational.R3Eq(pt, a)
						&& !Rational.R3Eq(pt, b)
						&& GraphGeom.PointOnSegmentF(ptA, pt, ca[e], a, ca[(e + 1) % 3], b))
					{
						hits.Add((key, id));
					}
				}
			}

			return hits;
		}

		/// <summary>
		/// Phase 4c's kernel: for every intersection segment on one triangle, the split
		/// points this side knows about.
		/// </summary>
		/// <param name="cands">This triangle's candidate point ids.</param>
		/// <param name="segEnds">The mesh's flat endpoint-id array.</param>
		/// <param name="from">First segment slot of this triangle.</param>
		/// <param name="to">One past its last segment slot.</param>
		/// <param name="pts">Exact point per id.</param>
		/// <param name="ptsA">Rounded approximation per id.</param>
		/// <returns>The (segment key, point id) hits, in segment then candidate order.</returns>
		private static List<(GeoEdgeKey Key, uint Id)> SplitHits(
			List<uint> cands,
			List<(uint A, uint B)> segEnds,
			int from,
			int to,
			R3[] pts,
			Vec3[] ptsA)
		{
			List<(GeoEdgeKey Key, uint Id)> hits = new List<(GeoEdgeKey Key, uint Id)>();
			for (int s = from; s < to; s++)
			{
				(uint ia, uint ib) = segEnds[s];
				GeoEdgeKey key = GraphTypes.GeoEdgeKey(ia, ib);
				R3 a = pts[ia];
				R3 b = pts[ib];
				Vec3 aa = ptsA[ia];
				Vec3 ba = ptsA[ib];
				(Vec3 Lo, Vec3 Hi) sbox = GraphGeom.SegBox3(aa, ba);
				foreach (uint id in cands)
				{
					R3 pt = pts[id];
					Vec3 ptA = ptsA[id];
					if (!GraphGeom.Box3Contains(sbox, ptA))
					{
						continue;
					}

					// Endpoint rejection by id: ids are injective on exact
					// value, so this is exactly the `R3Eq` test it replaces.
					if (id != ia && id != ib && GraphGeom.PointOnSegmentF(ptA, pt, aa, a, ba, b))
					{
						hits.Add((key, id));
					}
				}
			}

			return hits;
		}

		/// <summary>
		/// Phase 5's kernel: one triangle's arrangement, or the untouched case. Pure — the
		/// registry probes, the CDT and the crossings all read shared immutable state.
		/// </summary>
		/// <param name="t">The triangle.</param>
		/// <param name="pr">Its distributed primitives.</param>
		/// <param name="edgeRegistry">This mesh's original-edge split registry.</param>
		/// <param name="segSplits">The shared intersection-segment split registry.</param>
		/// <param name="segEnds">The mesh's flat endpoint-id array.</param>
		/// <param name="from">First segment slot of this triangle.</param>
		/// <param name="to">One past its last segment slot.</param>
		/// <param name="pts">Exact point per id.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The result, or null when cancelled.</returns>
		private static TriResult? ArrangeTriangle(
			Vec3[] t,
			TriPrims pr,
			Dictionary<BitEdgeKey, SplitIds> edgeRegistry,
			Dictionary<GeoEdgeKey, SplitIds> segSplits,
			List<(uint A, uint B)> segEnds,
			int from,
			int to,
			R3[] pts,
			CancelToken? token)
		{
			// Boundary split points for this triangle (bit-keyed: uncut
			// triangles probe with zero rational work).
			// Registry ids resolve back to points only here, one clone per point
			// that actually reaches this triangle's arrangement — the set itself
			// stays an ordered set of R3 so the arrangement's input order is untouched.
			SortedSet<R3> extra = new SortedSet<R3>();
			for (int e = 0; e < 3; e++)
			{
				if (edgeRegistry.TryGetValue(GraphTypes.BitEdgeKey(t[e], t[(e + 1) % 3]), out SplitIds? set))
				{
					foreach (uint id in set.Ids)
					{
						extra.Add(pts[id]);
					}
				}
			}

			// Split points along this triangle's intersection segments
			// discovered by the other side.
			for (int s = from; s < to; s++)
			{
				(uint ia, uint ib) = segEnds[s];
				if (segSplits.TryGetValue(GraphTypes.GeoEdgeKey(ia, ib), out SplitIds? set))
				{
					foreach (uint id in set.Ids)
					{
						extra.Add(pts[id]);
					}
				}
			}

			if (pr.Points.Count == 0 && pr.Segments.Count == 0 && extra.Count == 0)
			{
				return new TriResult(null);
			}

			ArrangementInput input = new ArrangementInput
			{
				Points = new List<(R3 Point, int Prov)>(pr.Points),
				Segments = new List<(R3 A, R3 B, int Prov)>(pr.Segments),
			};
			foreach (R3 pt in extra)
			{
				// Rust's `usize::MAX` sentinel; see the note on ArrangementInput's
				// provenance type in Arrangement.cs. Point provenance is never read.
				input.Points.Add((pt, int.MaxValue));
			}

			Arrangement? arr = ArrangementFunctions.Build(t, input, token);
			if (arr is null)
			{
				return null;
			}

			return new TriResult(arr);
		}

		/// <summary>
		/// Rust's <c>copy</c> closure in phase 3: cross-copy one side's primitives into the
		/// other's list, clipped to the shared coplanar overlap region.
		/// </summary>
		/// <param name="src">The snapshot to copy from.</param>
		/// <param name="dst">The list to copy into.</param>
		/// <param name="poly">The overlap region.</param>
		private static void CopyThroughRegion(TriPrims src, TriPrims dst, IReadOnlyList<R3> poly)
		{
			foreach ((R3 a, R3 b, int prov) in src.Segments)
			{
				(R3 A, R3 B)? clipped = GraphGeom.ClipSegmentToPolygon(a, b, poly);
				if (clipped is null)
				{
					continue;
				}

				(R3 ca, R3 cb) = clipped.Value;
				bool present = false;
				foreach ((R3 x, R3 y, int pv) in dst.Segments)
				{
					if (pv == prov && ((x.Equals(ca) && y.Equals(cb)) || (x.Equals(cb) && y.Equals(ca))))
					{
						present = true;
						break;
					}
				}

				if (!present)
				{
					dst.Segments.Add((ca, cb, prov));
				}
			}

			foreach ((R3 pt, int prov) in src.Points)
			{
				if (GraphGeom.ClipSegmentToPolygon(pt, pt, poly) is not null
					|| GraphGeom.PointInPolygonCoplanar(pt, poly))
				{
					bool present = false;
					foreach ((R3 x, int pv) in dst.Points)
					{
						if (pv == prov && x.Equals(pt))
						{
							present = true;
							break;
						}
					}

					if (!present)
					{
						dst.Points.Add((pt, prov));
					}
				}
			}
		}

		/// <summary>
		/// The length check that stands in for Rust's <c>&amp;[[Vec3; 3]]</c> element type.
		/// </summary>
		/// <param name="soup">The triangle soup.</param>
		/// <param name="name">The parameter name to blame.</param>
		/// <exception cref="ArgumentException">Some triangle is not exactly three vertices.</exception>
		private static void RequireTriangles(Vec3[][] soup, string name)
		{
			ArgumentNullException.ThrowIfNull(soup, name);
			foreach (Vec3[] t in soup)
			{
				if (t is null || t.Length != 3)
				{
					throw new ArgumentException("a triangle is three vertices", name);
				}
			}
		}
	}
}
