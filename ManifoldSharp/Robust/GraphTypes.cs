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

// GraphTypes.cs — port of robust/graph_types.rs, whose header reads:
//
//   Shared vocabulary of the intersection graph: the edge key spaces, the
//   exact-point interner, and the `Piece` / `IntersectionGraph` output types.
//
//   Split out of robust/intersection_graph.rs, which builds these values;
//   robust/cells.rs, robust/pairing.rs, robust/propagate-style flood fills and
//   robust/assemble.rs consume them (all through the `intersection_graph`
//   re-exports, so the public paths are unchanged). The exact rational point
//   type lives in robust/exact/rational.rs.
//
// (Here that last one is Robust/Exact/Rational.Points.cs.)
//
// ── Maps: FxHashMap vs Dictionary ────────────────────────────────────────────
// The Rust imports rustc_hash for every map and set in this file, with the
// standing justification:
//
//   Fx hashing instead of SipHash. Every map/set here is probe-only (documented
//   per site); the hasher is unseeded, so even iteration order is stable across
//   runs — output cannot depend on it.
//
// This port swaps them for plain Dictionary per the dependency-replacement table
// in CLAUDE.md, and it is sound for exactly the same reason, not by
// coincidence: every site below is probe-only, so .NET leaving Dictionary's
// enumeration order unspecified cannot reach the output either. The per-site
// invariant comments are the proof and are kept verbatim — do not delete one
// while leaving its Dictionary behind.
//
// ── Rust type aliases spelled out ────────────────────────────────────────────
// `EdgeKey`/`GeoEdgeKey` are `(u32, u32)` and `BitEdgeKey` is `([u64; 3],
// [u64; 3])`. C# aliases are file-scoped, so a downstream file could not see
// them; the tuple types are written out instead, and the alias names survive as
// the method names that build them. `[u64; 3]` becomes a
// `(ulong X, ulong Y, ulong Z)` value tuple — never a `ulong[]`, whose equality
// and hashing are by reference — matching Soup.cs's PosKey, the sibling port of
// the same "raw coordinate bits" idea.

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Robust
{
	/// <summary>
	/// The free functions of <c>robust/graph_types.rs</c>: the canonical edge keys of
	/// the intersection graph's three key spaces.
	/// </summary>
	public static class GraphTypes
	{
		/// <summary>
		/// Canonical (sorted) edge between two interned vertex ids. Downstream stages
		/// (classify rings, propagate flood fill) key their maps on these integers
		/// instead of exact rational point pairs — vertex interning at piece-emission
		/// time makes id equality coincide with exact geometric identity.
		/// </summary>
		/// <param name="a">One endpoint's interned id.</param>
		/// <param name="b">The other endpoint's interned id.</param>
		/// <returns>The two ids in nondecreasing order (Rust's <c>EdgeKey</c>).</returns>
		public static (uint A, uint B) EdgeKey(uint a, uint b)
		{
			return a <= b ? (a, b) : (b, a);
		}

		/// <summary>
		/// Canonical (sorted) exact edge between two <see cref="PointTable"/> ids — local
		/// key for the split-point registries, which run before output interning exists.
		/// Ids stand in for the points themselves: the table is injective on exact value,
		/// so id equality *is* exact geometric identity, and a registry key costs 8 bytes
		/// instead of two cloned rational triples.
		/// </summary>
		/// <param name="a">One endpoint's table id.</param>
		/// <param name="b">The other endpoint's table id.</param>
		/// <returns>The two ids in nondecreasing order (Rust's <c>GeoEdgeKey</c>).</returns>
		internal static (uint A, uint B) GeoEdgeKey(uint a, uint b)
		{
			return a <= b ? (a, b) : (b, a);
		}

		/// <summary>
		/// Canonical original-mesh edge keyed by raw coordinate bits — original edges
		/// always join exact f64 vertices, so the boundary-split registry never needs
		/// rational keys (and untouched triangles probe it for free).
		/// </summary>
		/// <remarks>
		/// The ordering is Rust's <c>[u64; 3]</c> array <c>Ord</c>: lexicographic on the
		/// x, y, z bit patterns, which is what the value tuple's <c>CompareTo</c> does.
		/// </remarks>
		/// <param name="a">One endpoint.</param>
		/// <param name="b">The other endpoint.</param>
		/// <returns>The two bit keys in nondecreasing order (Rust's <c>BitEdgeKey</c>).</returns>
		internal static ((ulong X, ulong Y, ulong Z) A, (ulong X, ulong Y, ulong Z) B) BitEdgeKey(Vec3 a, Vec3 b)
		{
			(ulong X, ulong Y, ulong Z) ka = F64Key(a);
			(ulong X, ulong Y, ulong Z) kb = F64Key(b);
			return ka.CompareTo(kb) <= 0 ? (ka, kb) : (kb, ka);
		}

		/// <summary>
		/// The raw coordinate bits of an exact-f64 point, the key of
		/// <see cref="VertInterner"/>'s f64 key space.
		/// </summary>
		/// <param name="v">The point.</param>
		/// <returns>Its three coordinate bit patterns.</returns>
		internal static (ulong X, ulong Y, ulong Z) F64Key(Vec3 v)
		{
			// Normalize -0.0 so it shares an id with +0.0 (they are the same
			// rational point).
			return (Norm(v.X), Norm(v.Y), Norm(v.Z));

			static ulong Norm(double x)
			{
				return BitConverter.DoubleToUInt64Bits(x == 0.0 ? 0.0 : x);
			}
		}
	}

	/// <summary>
	/// Registry-local point interner: one dense <c>uint</c> id per distinct exact point,
	/// used by the split registries (intersection_graph.rs phases 4b/4c) so they can
	/// store ids instead of cloning a rational triple per (edge, point) incidence. A
	/// giant self-intersecting mesh produces millions of split-point incidences, and the
	/// clone multiplicity — hit lists, dedup sets and edge keys each holding their own
	/// copy — was the memory wall, not any single structure.
	/// </summary>
	/// <remarks>
	/// Deliberately NOT <see cref="VertInterner"/>: that one's insertion order defines
	/// output vertex ids, so it must keep seeing points in piece-emission order. This
	/// table is internal to the registries; its ids never reach the output (they only
	/// group and dedup, and the points they hand back are re-sorted through a
	/// <c>BTreeSet&lt;R3&gt;</c>), so assigning them earlier is invisible.
	/// <para>
	/// Order invariance: probe-only (<c>entry</c>, never iterated except by
	/// <see cref="Resolve"/>, which reconstructs the id-indexed order).
	/// </para>
	/// </remarks>
	internal sealed class PointTable
	{
		private readonly Dictionary<R3Key, uint> map = new Dictionary<R3Key, uint>();

		/// <summary>The number of distinct points interned so far (Rust's <c>len</c>).</summary>
		public int Count
		{
			get { return this.map.Count; }
		}

		/// <summary>
		/// Id of <paramref name="p"/>, assigning the next one on first sight. The clone is
		/// paid once per *distinct* point (the probe key on a hit is temporary) — in this
		/// port R3 is a reference type, so even that clone is only a reference copy.
		/// </summary>
		/// <param name="p">The exact point.</param>
		/// <returns>Its dense id.</returns>
		public uint Intern(R3 p)
		{
			uint next = (uint)this.map.Count;
			R3Key key = new R3Key(p);
			if (this.map.TryGetValue(key, out uint id))
			{
				return id;
			}

			this.map.Add(key, next);
			return next;
		}

		/// <summary>
		/// Borrowed point per id. Returning references (not clones) is the whole point:
		/// the table then holds exactly one copy of each distinct point for the rest of
		/// the build.
		/// </summary>
		/// <returns>The interned points, indexed by their ids.</returns>
		/// <exception cref="InvalidOperationException">
		/// The ids are not dense in <c>0..Count</c>, which only a bug in
		/// <see cref="Intern"/> could produce (Rust's <c>expect</c>).
		/// </exception>
		public R3[] Resolve()
		{
			R3?[] pts = new R3?[this.map.Count];
			foreach (KeyValuePair<R3Key, uint> kv in this.map)
			{
				pts[kv.Value] = kv.Key.Point;
			}

			R3[] result = new R3[pts.Length];
			for (int i = 0; i < pts.Length; i++)
			{
				result[i] = pts[i] ?? throw new InvalidOperationException("ids are dense in 0..len");
			}

			return result;
		}
	}

	/// <summary>
	/// One output fragment: a sub-triangle of an arranged input triangle, or an untouched
	/// whole triangle. <c>v</c> is wound to match the input mesh's outward orientation;
	/// <see cref="Vi"/> are the interned ids of the same three vertices.
	/// </summary>
	/// <remarks>
	/// A readonly struct because the Rust derives <c>Clone, Copy</c>: pieces are held in
	/// big arenas and copied out by value, never aliased.
	/// </remarks>
	public readonly struct Piece
	{
		/// <summary>0 = first operand (P), 1 = second operand (Q).</summary>
		public readonly byte Mesh;

		/// <summary>Index of the originating triangle in its soup.</summary>
		public readonly int Tri;

		/// <summary>
		/// Interned vertex ids (indices into <see cref="IntersectionGraph.Verts"/>), wound
		/// to the input mesh's outward orientation. Pieces carry no coordinates of their
		/// own — the shared tables keep untouched triangles free of rational clones
		/// entirely.
		/// </summary>
		public readonly UVec3 Vi;

		/// <summary>Creates a piece.</summary>
		/// <param name="mesh">0 for P, 1 for Q.</param>
		/// <param name="tri">Index of the originating triangle in its soup.</param>
		/// <param name="vi">The three interned vertex ids, in the input winding.</param>
		public Piece(byte mesh, int tri, UVec3 vi)
		{
			this.Mesh = mesh;
			this.Tri = tri;
			this.Vi = vi;
		}

		/// <summary>The Rust's derived <c>Debug</c>, for trace diffing.</summary>
		/// <returns>The three fields in declaration order.</returns>
		public override string ToString()
		{
			return $"Piece {{ mesh: {this.Mesh}, tri: {this.Tri}, vi: [{this.Vi.X}, {this.Vi.Y}, {this.Vi.Z}] }}";
		}
	}

	/// <summary>Everything classification and assembly need.</summary>
	public sealed class IntersectionGraph
	{
		/// <summary>The output fragments.</summary>
		public List<Piece> Pieces = new List<Piece>();

		/// <summary>
		/// Interned unique vertices; <see cref="Piece.Vi"/> and the edge keys index into
		/// this.
		/// </summary>
		public List<R3> Verts = new List<R3>();

		/// <summary>
		/// Correctly rounded f64 approximation per interned vertex (exact for input
		/// vertices) — float filters and output assembly read these instead of re-rounding
		/// rationals.
		/// </summary>
		public List<Vec3> VertsF64 = new List<Vec3>();

		/// <summary>
		/// Canonical keys of every arrangement constraint edge — the exact intersection
		/// sub-segments the classification rings live on.
		/// </summary>
		/// <remarks>
		/// Order invariance: trivially held — the Rust writes this set (intersection_graph.rs)
		/// and never enumerates it, so a HashSet's unspecified enumeration order has nothing
		/// to reach. If a later stage ever does iterate it, that is the moment to re-derive
		/// the invariant, not to assume it.
		/// </remarks>
		public HashSet<(uint A, uint B)> IsectEdges = new HashSet<(uint A, uint B)>();

		/// <summary>True when any P×Q pair intersected at all.</summary>
		public bool AnyIntersections;

		/// <summary>The three exact vertices of a piece.</summary>
		/// <param name="pi">Index into <see cref="Pieces"/>.</param>
		/// <returns>The piece's three vertices, in its own winding.</returns>
		public R3[] PieceVerts(int pi)
		{
			UVec3 vi = this.Pieces[pi].Vi;
			return new R3[]
			{
				this.Verts[(int)vi[0]],
				this.Verts[(int)vi[1]],
				this.Verts[(int)vi[2]],
			};
		}
	}

	/// <summary>
	/// Exact-point interner: one id per distinct point, with two disjoint key spaces.
	/// f64-representable points (all input vertices, and any constructed point that rounds
	/// exactly) key on their coordinate bits — no rational hashing, so untouched input
	/// triangles intern for the cost of a HashMap probe. Only genuinely non-representable
	/// constructed points use the rational map. <see cref="VertsF64"/> caches the
	/// correctly rounded approximation of every id (exact for bit-keyed points), which
	/// downstream float filters and output assembly reuse instead of re-rounding.
	/// </summary>
	/// <remarks>
	/// Order invariance: both maps are probe-only (<c>get</c>/<c>entry</c>, never
	/// iterated); ids come from <c>verts.len()</c> at insertion time, so they depend only
	/// on the sequential call order, not on the hasher.
	/// </remarks>
	public sealed class VertInterner
	{
		private readonly Dictionary<R3Key, uint> map = new Dictionary<R3Key, uint>();

		private readonly Dictionary<(ulong X, ulong Y, ulong Z), uint> fmap
			= new Dictionary<(ulong X, ulong Y, ulong Z), uint>();

		/// <summary>The interned points, indexed by id.</summary>
		public List<R3> Verts = new List<R3>();

		/// <summary>Their correctly rounded f64 approximations, indexed by the same id.</summary>
		public List<Vec3> VertsF64 = new List<Vec3>();

		/// <summary>
		/// Intern an exact-f64 point (input mesh vertices): zero rational work on hits; one
		/// <see cref="R3.FromVec3"/> on first sight, for the exact table.
		/// </summary>
		/// <param name="v">The point.</param>
		/// <returns>Its interned id.</returns>
		public uint InternF64(Vec3 v)
		{
			(ulong X, ulong Y, ulong Z) key = GraphTypes.F64Key(v);
			if (this.fmap.TryGetValue(key, out uint hit))
			{
				return hit;
			}

			uint id = (uint)this.Verts.Count;
			this.fmap.Add(key, id);
			this.Verts.Add(R3.FromVec3(v));
			this.VertsF64.Add(v);
			return id;
		}

		/// <summary>
		/// Intern an exact rational point. Representable points route to the f64 key space
		/// so both paths agree on ids.
		/// </summary>
		/// <param name="p">The exact point.</param>
		/// <returns>Its interned id.</returns>
		public uint Intern(R3 p)
		{
			Vec3 rounded = p.ToVec3Rounded();
			if (Rational.R3Eq(R3.FromVec3(rounded), p))
			{
				return this.InternF64(rounded);
			}

			uint next = (uint)this.Verts.Count;
			R3Key key = new R3Key(p);
			if (this.map.TryGetValue(key, out uint hit))
			{
				return hit;
			}

			this.map.Add(key, next);
			this.Verts.Add(p);
			this.VertsF64.Add(rounded);
			return next;
		}
	}
}
