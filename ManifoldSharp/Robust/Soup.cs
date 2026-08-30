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

// Soup.cs — port of robust/soup.rs, whose header reads:
//
//   Triangle-soup import support for the robust boolean engine.
//
//   When the strict halfedge pairing of `Manifold::from_mesh_gl` fails
//   (non-manifold connectivity), `from_mesh_gl_robust` falls through to
//   `soupify`: the geometry is kept as an unpaired-halfedge triangle soup
//   inside `ManifoldImpl` (is_soup = true), provided it passes the one check
//   the robust engine genuinely needs — the soup must be geometrically
//   **closed and orientable**: after welding vertices by exact position and
//   dropping exactly-degenerate triangles (paper §5), every directed edge
//   must be balanced by its reverse. That is precisely the condition for the
//   soup to bound a solid via winding numbers.
//
// ── Maps: BTreeMap vs Dictionary ─────────────────────────────────────────────
// The Rust uses `BTreeMap` for all three maps below. Two of them (`weld`,
// `balance`) are probe-only — never iterated in a way whose order is
// observable: `weld` is read by key, and `balance` is only asked whether *any*
// value is non-zero — so a `Dictionary` is exact, per the dependency-replacement
// table in docs/PORTING_PLAN.md. The third (`open`) *is* iterated, so it stays a
// `SortedDictionary` even though the pairing inside each group depends only on
// that group's own halfedge indices; see the comment at the loop.
//
// ── Where the Rust's `intersection_graph::` re-exports land ──────────────────
// The detector below reaches `real_self_contact` / `tri_box` / `is_degenerate` /
// `SelfCutStats` through `super::intersection_graph::…` in the Rust, but those are
// re-exports: the definitions live in robust/graph_self_cut.rs and
// robust/graph_geom.rs, which this port spells directly as
// `GraphSelfCut` / `GraphGeom`. Same functions, one hop shorter.
//
// `SelfIntersectCache` is in SelfIntersectCache.cs rather than here; it was lifted
// out ahead of its phase because ManifoldImpl holds it as a field.

using System.Diagnostics;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp.Robust
{
	/// <summary>
	/// The free functions of <c>robust/soup.rs</c>: triangle-soup import validation and
	/// the triangle/property views of a <see cref="ManifoldImpl"/> the robust engine
	/// works from.
	/// </summary>
	public static class Soup
	{
		/// <summary>
		/// Convert <paramref name="imp"/> (with <c>VertPos</c> and
		/// <c>MeshRelation.TriRef</c> already populated, and <c>Halfedge</c> full of
		/// whatever strict pairing produced) into a validated triangle soup:
		/// <list type="bullet">
		/// <item><description>drop exactly-degenerate triangles (and their tri_refs),</description></item>
		/// <item><description>verify the remainder is closed and orientable (else
		/// <see cref="Error.NotClosed"/>),</description></item>
		/// <item><description>rebuild halfedges with best-effort pairing
		/// (<c>PairedHalfedge == -1</c> where no partner exists),</description></item>
		/// <item><description>recompute per-face normals (no pairing required),</description></item>
		/// <item><description>set <c>IsSoup</c>.</description></item>
		/// </list>
		/// </summary>
		/// <remarks>
		/// <paramref name="triProp"/> / <paramref name="triVert"/> mirror the
		/// <c>CreateHalfedges</c> inputs: <paramref name="triVert"/> holds the position
		/// indices when properties are mapped separately, else <paramref name="triProp"/>
		/// does double duty.
		/// <para>
		/// The Rust returns <c>Result&lt;(), Error&gt;</c>; the C# port returns the status
		/// enum instead, per the "errors are a status, not exceptions" rule in
		/// docs/PORTING_PLAN.md. <see cref="Error.NoError"/> is the Rust's <c>Ok(())</c>.
		/// </para>
		/// </remarks>
		/// <param name="imp">The impl to convert, edited in place on success.</param>
		/// <param name="triProp">Per-triangle property-vertex indices.</param>
		/// <param name="triVert">Per-triangle position indices, or empty.</param>
		/// <returns><see cref="Error.NoError"/>, or <see cref="Error.NotClosed"/>.</returns>
		public static Error Soupify(
			ManifoldImpl imp,
			IReadOnlyList<IVec3> triProp,
			IReadOnlyList<IVec3> triVert)
		{
			ArgumentNullException.ThrowIfNull(imp);
			ArgumentNullException.ThrowIfNull(triProp);
			ArgumentNullException.ThrowIfNull(triVert);

			IReadOnlyList<IVec3> positionTris = triVert.Count == 0 ? triProp : triVert;
			Debug.Assert(positionTris.Count == triProp.Count, "tri_vert/tri_prop length mismatch");

			// Weld vertex ids by exact position for the closedness bookkeeping only
			// (vert_pos itself is left as imported).
			Dictionary<(ulong, ulong, ulong), int> weld = new Dictionary<(ulong, ulong, ulong), int>();
			int[] weldedId = new int[imp.VertPos.Count];
			for (int i = 0; i < imp.VertPos.Count; i++)
			{
				(ulong, ulong, ulong) key = PosKey(imp.VertPos[i]);
				if (!weld.TryGetValue(key, out int id))
				{
					id = i;
					weld[key] = id;
				}

				weldedId[i] = id;
			}

			// Keep non-degenerate triangles; balance directed edges on welded ids.
			List<int> keep = new List<int>(positionTris.Count);
			Dictionary<(int, int), long> balance = new Dictionary<(int, int), long>();
			for (int t = 0; t < positionTris.Count; t++)
			{
				IVec3 tv = positionTris[t];
				int a = tv.X;
				int b = tv.Y;
				int c = tv.Z;
				int wa = weldedId[a];
				int wb = weldedId[b];
				int wc = weldedId[c];
				if (wa == wb
					|| wb == wc
					|| wc == wa
					|| IsDegenerate(imp.VertPos[a], imp.VertPos[b], imp.VertPos[c]))
				{
					continue; // paper §5: degenerate triangles are safe to drop
				}

				keep.Add(t);
				AddBalance(balance, wa, wb);
				AddBalance(balance, wb, wc);
				AddBalance(balance, wc, wa);
			}

			if (keep.Count < 4)
			{
				return Error.NotClosed;
			}

			foreach (long n in balance.Values)
			{
				if (n != 0)
				{
					return Error.NotClosed;
				}
			}

			// Rebuild halfedges for the kept triangles with best-effort pairing:
			// LIFO multimap on welded undirected edges, forward (start < end) pairs
			// with reverse.
			bool hasProps = triVert.Count != 0;
			List<Halfedge> halfedges = new List<Halfedge>(3 * keep.Count);
			List<TriRef> triRef = new List<TriRef>(keep.Count);
			foreach (int oldT in keep)
			{
				IVec3 tv = positionTris[oldT];
				IVec3 tp = triProp[oldT];
				for (int i = 0; i < 3; i++)
				{
					int j = (i + 1) % 3;
					halfedges.Add(new Halfedge(
						tv[i],
						tv[j],
						-1,
						hasProps ? tp[i] : tv[i]));
				}

				if (oldT < imp.MeshRelation.TriRef.Count)
				{
					triRef.Add(imp.MeshRelation.TriRef[oldT]);
				}
			}

			SortedDictionary<(int, int), List<int>> open = new SortedDictionary<(int, int), List<int>>();
			for (int idx = 0; idx < halfedges.Count; idx++)
			{
				Halfedge he = halfedges[idx];
				int u = weldedId[he.StartVert];
				int v = weldedId[he.EndVert];
				(int, int) key = (Math.Min(u, v), Math.Max(u, v));
				if (!open.TryGetValue(key, out List<int>? group))
				{
					group = new List<int>();
					open[key] = group;
				}

				group.Add(idx);
			}

			// The Rust iterates the BTreeMap in key order. Each halfedge belongs to
			// exactly one group and the pairing below reads and writes only that
			// group's own indices, so the assignment is independent of the order the
			// groups are visited in — but the map stays sorted anyway, because a
			// deliberate order is cheaper to keep than to re-argue.
			foreach (KeyValuePair<(int, int), List<int>> entry in open)
			{
				// Pair forwards with reverses greedily; leftovers stay -1.
				List<int> fwd = new List<int>();
				List<int> bwd = new List<int>();
				foreach (int idx in entry.Value)
				{
					Halfedge he = halfedges[idx];
					if (weldedId[he.StartVert] < weldedId[he.EndVert])
					{
						fwd.Add(idx);
					}
					else
					{
						bwd.Add(idx);
					}
				}

				while (fwd.Count > 0 && bwd.Count > 0)
				{
					int f = fwd[fwd.Count - 1];
					fwd.RemoveAt(fwd.Count - 1);
					int b = bwd[bwd.Count - 1];
					bwd.RemoveAt(bwd.Count - 1);
					Halfedge hf = halfedges[f];
					hf.PairedHalfedge = b;
					halfedges[f] = hf;
					Halfedge hb = halfedges[b];
					hb.PairedHalfedge = f;
					halfedges[b] = hb;
				}
			}

			imp.Halfedge = halfedges;

			// The Rust assigns the whole vec (`imp.mesh_relation.tri_ref = tri_ref`);
			// MeshRelationD.TriRef is get-only here, so refill it in place — same
			// contents, same order, and the alternative would be widening that property
			// for one caller.
			imp.MeshRelation.TriRef.Clear();
			if (triRef.Count == keep.Count)
			{
				imp.MeshRelation.TriRef.AddRange(triRef);
			}

			// Per-face normals need only the triangle itself.
			int numTri = imp.NumTri();
			List<Vec3> faceNormal = new List<Vec3>(numTri);
			for (int t = 0; t < numTri; t++)
			{
				Vec3 a = imp.VertPos[imp.Halfedge[3 * t].StartVert];
				Vec3 b = imp.VertPos[imp.Halfedge[(3 * t) + 1].StartVert];
				Vec3 c = imp.VertPos[imp.Halfedge[(3 * t) + 2].StartVert];
				Vec3 n = Cross(b - a, c - a);
				double len = Length(n);
				faceNormal.Add(len > 0.0 ? n / len : new Vec3(0.0, 0.0, 0.0));
			}

			imp.FaceNormal = faceNormal;
			imp.VertNormal.Clear();
			imp.IsSoup = true;
			return Error.NoError;
		}

		// -------------------------------------------------------------------
		// Geometric self-intersection detection
		// -------------------------------------------------------------------

		/// <summary>
		/// True when two of <paramref name="imp"/>'s own triangles genuinely intersect —
		/// they cross, they overlap, or they are coincident surface — as opposed to merely
		/// sharing an edge or a vertex as every closed mesh does.
		/// </summary>
		/// <remarks>
		/// Answers from the cache after the first call. The narrow phase is
		/// <see cref="GraphSelfCut.RealSelfContact"/>, the same predicate the robust engine
		/// uses to decide which self-cuts it must make, plus the exact duplicate-triangle
		/// case that predicate deliberately passes over (see <see cref="GenuineContact"/>).
		/// Unlike the engine's phase 2b this stops at the first genuine contact and builds
		/// no graph.
		/// </remarks>
		/// <param name="imp">The impl to scan.</param>
		/// <returns>True when the mesh self-intersects.</returns>
		public static bool HasSelfIntersections(ManifoldImpl imp)
		{
			return HasSelfIntersectionsWithToken(imp, null);
		}

		/// <summary>
		/// <see cref="HasSelfIntersections"/> with cancellation, for the boolean dispatcher.
		/// </summary>
		/// <remarks>
		/// A cancelled scan answers <c>true</c> (route to the robust engine, which then
		/// reports <see cref="Error.Cancelled"/> itself) and caches nothing, so the verdict
		/// is recomputed properly if the impl is used again.
		/// </remarks>
		/// <param name="imp">The impl to scan.</param>
		/// <param name="token">The cancellation token, or null for an uncancellable scan.</param>
		/// <returns>True when the mesh self-intersects.</returns>
		public static bool HasSelfIntersectionsWithToken(ManifoldImpl imp, CancelToken? token)
		{
			ArgumentNullException.ThrowIfNull(imp);

			bool? cached = imp.SelfIntersects.Get();
			if (cached.HasValue)
			{
				return cached.Value;
			}

			bool? verdict = ComputeSelfIntersections(imp, token);
			if (verdict.HasValue)
			{
				imp.SelfIntersects.Set(verdict.Value);
				return verdict.Value;
			}

			return true;
		}

		/// <summary>
		/// Per-corner vertex properties of an impl, flattened as
		/// <c>props[(3*tri + corner) * numProp + channel]</c>, aligned with
		/// <see cref="ImplToTris"/> ordering. Empty when the impl carries no properties.
		/// </summary>
		/// <param name="imp">The impl to read.</param>
		/// <returns>The flattened corner properties.</returns>
		public static List<double> ImplToCornerProps(ManifoldImpl imp)
		{
			ArgumentNullException.ThrowIfNull(imp);

			int np = imp.NumProp;
			if (np == 0)
			{
				return new List<double>();
			}

			int numTri = imp.NumTri();
			List<double> outProps = new List<double>(3 * numTri * np);
			for (int t = 0; t < numTri; t++)
			{
				for (int i = 0; i < 3; i++)
				{
					int pv = imp.Halfedge[(3 * t) + i].PropVert;
					for (int channel = 0; channel < np; channel++)
					{
						outProps.Add(imp.Properties[(pv * np) + channel]);
					}
				}
			}

			return outProps;
		}

		/// <summary>
		/// The triangle list of any impl (soup or manifold) as position triples — the
		/// robust engine's working form.
		/// </summary>
		/// <remarks>
		/// The Rust element type is <c>[Vec3; 3]</c>, a fixed-size array indexed
		/// <c>t[0..2]</c> and iterated as a whole by the engine's predicates. The C#
		/// element is a three-element <c>Vec3[]</c> — the construct that keeps both of
		/// those spellings — and every array this returns has exactly three entries.
		/// </remarks>
		/// <param name="imp">The impl to read.</param>
		/// <returns>One three-element array of positions per triangle.</returns>
		public static List<Vec3[]> ImplToTris(ManifoldImpl imp)
		{
			ArgumentNullException.ThrowIfNull(imp);

			int numTri = imp.NumTri();
			List<Vec3[]> tris = new List<Vec3[]>(numTri);
			for (int t = 0; t < numTri; t++)
			{
				tris.Add(new Vec3[]
				{
					imp.VertPos[imp.Halfedge[3 * t].StartVert],
					imp.VertPos[imp.Halfedge[(3 * t) + 1].StartVert],
					imp.VertPos[imp.Halfedge[(3 * t) + 2].StartVert],
				});
			}

			return tris;
		}

		/// <summary>
		/// Do these two triangles of one mesh meet in anything beyond ordinary adjacency?
		/// </summary>
		/// <remarks>
		/// <see cref="GraphSelfCut.RealSelfContact"/> answers that for every case but one:
		/// it reports exactly duplicated triangles (all three vertices coincide, either
		/// winding) as benign, because the robust arrangement needs no cut there — both
		/// copies emit identical pieces and the winding arithmetic resolves them. They are
		/// still coincident surface, which is precisely what the exact engine cannot
		/// integrate (Thingi10K #92068's shells are triple-wound duplicates and nothing
		/// else), so the dispatch detector counts them.
		/// </remarks>
		/// <param name="t1">The first triangle's three vertices.</param>
		/// <param name="t2">The second triangle's three vertices.</param>
		/// <param name="stats">Narrow-phase counters, incremented on whichever path is taken.</param>
		/// <returns>True when the contact is more than ordinary adjacency.</returns>
		private static bool GenuineContact(Vec3[] t1, Vec3[] t2, SelfCutStats stats)
		{
			foreach (Vec3 v in t1)
			{
				if (!ContainsExact(t2, v))
				{
					return GraphSelfCut.RealSelfContact(t1, t2, stats) is not null;
				}
			}

			return true;
		}

		/// <summary>
		/// Uncached detector: BVH broad phase over the impl's own triangles, exact narrow
		/// phase, early exit on the first genuine contact. Null means the scan was cancelled
		/// before it could reach a verdict.
		/// </summary>
		/// <remarks>
		/// The broad phase reuses <c>imp.Collider</c> — the face BVH <c>SortGeometry</c>
		/// already built, whose leaves are in face order — and only builds a private
		/// morton-ordered tree (as the intersection graph's self-cut phase does) when the
		/// impl carries no matching collider, which is the case for soup impls.
		/// </remarks>
		/// <param name="imp">The impl to scan.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The verdict, or null when cancelled.</returns>
		private static bool? ComputeSelfIntersections(ManifoldImpl imp, CancelToken? token)
		{
			List<Vec3[]> tris = ImplToTris(imp);
			if (tris.Count < 2)
			{
				return false;
			}

			// Non-finite positions (a warp to NaN/inf, which no import check rejects)
			// have no exact rational form, so the narrow phase cannot run on them.
			// "Self-intersecting" is the safe verdict: it routes the operand to the
			// robust engine rather than letting the exact kernels integrate garbage.
			foreach (Vec3[] t in tris)
			{
				foreach (Vec3 v in t)
				{
					if (!double.IsFinite(v.X) || !double.IsFinite(v.Y) || !double.IsFinite(v.Z))
					{
						return true;
					}
				}
			}

			Box[] boxes = new Box[tris.Count];
			bool[] live = new bool[tris.Count];
			for (int i = 0; i < tris.Count; i++)
			{
				boxes[i] = GraphGeom.TriBox(tris[i]);
				live[i] = !GraphGeom.IsDegenerate(tris[i]);
			}

			// Leaf index -> triangle index; empty when the cached face collider is used,
			// whose leaves already are triangle indices.
			int[] leafTri = Array.Empty<int>();
			Collider collider;
			if (imp.Collider.NumLeaves() == tris.Count)
			{
				collider = imp.Collider;
			}
			else
			{
				List<int> order = new List<int>();
				for (int i = 0; i < tris.Count; i++)
				{
					if (live[i])
					{
						order.Add(i);
					}
				}

				if (order.Count < 2)
				{
					return false;
				}

				Box scene = new Box();
				for (int i = 0; i < tris.Count; i++)
				{
					if (live[i])
					{
						scene = scene.UnionBox(boxes[i]);
					}
				}

				// `sort_by_key` is stable in Rust; OrderBy is the documented-stable C#
				// equivalent (docs/PORTING_PLAN.md, "Stable sorts").
				leafTri = order.OrderBy(i => Sort.MortonCode(boxes[i].Center(), scene)).ToArray();
				Box[] leafBoxes = new Box[leafTri.Length];
				uint[] leafMorton = new uint[leafTri.Length];
				for (int k = 0; k < leafTri.Length; k++)
				{
					leafBoxes[k] = boxes[leafTri[k]];
					leafMorton[k] = Sort.MortonCode(boxes[leafTri[k]].Center(), scene);
				}

				collider = new Collider(leafBoxes, leafMorton);
			}

			bool mapped = leafTri.Length != 0;

			SelfCutStats stats = new SelfCutStats();
			List<int> cands = new List<int>();
			for (int i = 0; i < tris.Count; i++)
			{
				if (!live[i])
				{
					continue;
				}

				if (Cancel.IsCancelled(token))
				{
					return null;
				}

				cands.Clear();
				collider.CollisionsOne(boxes[i], i, (_, leaf) => cands.Add(mapped ? leafTri[leaf] : leaf));

				// Rust `sort_unstable` on plain indices: no ties to break, so the
				// unstable introsort List.Sort uses is exactly it.
				cands.Sort();
				foreach (int j in cands)
				{
					if (j <= i || !live[j] || !boxes[i].DoesOverlapBox(boxes[j]))
					{
						continue;
					}

					if (GenuineContact(tris[i], tris[j], stats))
					{
						return true;
					}
				}
			}

			return false;
		}

		/// <summary>
		/// Rust's <c>[Vec3; 3]::contains(&amp;v)</c> — IEEE <c>==</c> per component, so -0.0
		/// matches 0.0. <see cref="Vec3.Equals(Vec3)"/> is *bit* equality (it backs hashing
		/// and welding) and would answer differently; see the file header of
		/// <c>GraphSelfCut.cs</c>, which carries the same predicate for the same reason.
		/// </summary>
		/// <param name="t">The vertices to search.</param>
		/// <param name="v">The vertex to find.</param>
		/// <returns>True when some element compares equal.</returns>
		private static bool ContainsExact(Vec3[] t, Vec3 v)
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

		/// <summary>Weld key: exact position identity (with -0.0 normalized so it equals 0.0).</summary>
		/// <param name="v">The position to key.</param>
		/// <returns>The three normalized bit patterns.</returns>
		private static (ulong X, ulong Y, ulong Z) PosKey(Vec3 v)
		{
			return (NormalizedBits(v.X), NormalizedBits(v.Y), NormalizedBits(v.Z));
		}

		/// <summary>
		/// The Rust's <c>|x: f64| if x == 0.0 { 0.0f64 } else { x }.to_bits()</c>. The
		/// normalization is load-bearing: <c>-0.0</c> and <c>0.0</c> compare equal but have
		/// different bit patterns, so welding on raw bits would split a vertex that every
		/// other part of the engine considers one point.
		/// </summary>
		/// <param name="x">The coordinate.</param>
		/// <returns>Its bit pattern, with negative zero folded onto positive zero.</returns>
		private static ulong NormalizedBits(double x)
		{
			return BitConverter.DoubleToUInt64Bits(x == 0.0 ? 0.0 : x);
		}

		/// <summary>Exact zero-area test on f64 positions.</summary>
		/// <param name="a">First corner.</param>
		/// <param name="b">Second corner.</param>
		/// <param name="c">Third corner.</param>
		/// <returns>True when the triangle has exactly zero area.</returns>
		private static bool IsDegenerate(Vec3 a, Vec3 b, Vec3 c)
		{
			R3 ra = R3.FromVec3(a);
			return R3.FromVec3(b).Sub(ra).Cross(R3.FromVec3(c).Sub(ra)).IsZero();
		}

		/// <summary>
		/// The Rust's <c>*balance.entry(key).or_insert(0) += if u &lt; v { 1 } else { -1 }</c>
		/// on the undirected key: +1 for a directed edge running low→high, -1 for high→low,
		/// so a closed orientable soup sums to zero on every edge.
		/// </summary>
		/// <param name="balance">The running per-edge balance.</param>
		/// <param name="u">Welded start vertex.</param>
		/// <param name="v">Welded end vertex.</param>
		private static void AddBalance(Dictionary<(int, int), long> balance, int u, int v)
		{
			(int, int) key = (Math.Min(u, v), Math.Max(u, v));
			balance.TryGetValue(key, out long n);
			balance[key] = n + (u < v ? 1 : -1);
		}
	}
}
