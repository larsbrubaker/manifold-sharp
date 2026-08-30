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

// Cdt.Arena.cs — the arena half of robust/cdt.rs (see Cdt.cs for the file's
// header): navigating the triangle arena (vertex fans, adjacency, point
// location) and editing it (the vertex splits that grow the triangulation and
// the queue-based Lawson legalization that restores the Delaunay property after
// them). Split out of Cdt.cs for the 800-line file cap only; in the Rust these
// are methods of the same `impl Cdt` block, and they are the same partial class
// here.

using System.Diagnostics;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Robust
{
	/// <content>Arena navigation, the vertex splits and the Lawson flips.</content>
	internal sealed partial class Cdt
	{
		/// <summary>Record <paramref name="t"/> as the latest triangle containing each of its vertices.</summary>
		/// <param name="t">The triangle index.</param>
		private void Record(int t)
		{
			for (int k = 0; k < 3; k++)
			{
				this.vertTri[this.tris[t].V[k]] = t;
			}
		}

		/// <summary>
		/// A live triangle containing vertex <paramref name="a"/>. The
		/// <see cref="vertTri"/> invariant makes the recorded entry always alive; the scan
		/// is a defensive fallback.
		/// </summary>
		/// <param name="a">The vertex index.</param>
		/// <returns>The index of a live triangle with that vertex.</returns>
		private int LiveTriWith(int a)
		{
			int cand = this.vertTri[a];
			if (cand < this.triCount && this.tris[cand].Alive && Contains(this.tris[cand].V, a))
			{
				return cand;
			}

			Debug.Assert(false, $"vertTri invariant broken for vertex {a}");
			for (int i = this.triCount - 1; i >= 0; i--)
			{
				if (this.tris[i].Alive && Contains(this.tris[i].V, a))
				{
					return i;
				}
			}

			throw new InvalidOperationException("vertex not in any live triangle");
		}

		/// <summary>
		/// Rotate around vertex <paramref name="a"/> (both directions from its recorded
		/// triangle, so boundary fans are fully covered) applying <paramref name="f"/> to
		/// each incident triangle until it returns a value.
		/// </summary>
		/// <typeparam name="T">The payload the caller extracts.</typeparam>
		/// <param name="a">The vertex to rotate around.</param>
		/// <param name="f">The per-triangle probe.</param>
		/// <returns>The first payload produced, or null.</returns>
		private T? RotateAround<T>(int a, Func<int, T?> f)
			where T : struct
		{
			int seed = this.LiveTriWith(a);
			for (int dirn = 0; dirn < 2; dirn++)
			{
				int cur = seed;
				while (true)
				{
					if (dirn == 0 || cur != seed)
					{
						T? outValue = f(cur);
						if (outValue.HasValue)
						{
							return outValue;
						}
					}

					int ia = -1;
					for (int k = 0; k < 3; k++)
					{
						if (this.tris[cur].V[k] == a)
						{
							ia = k;
							break;
						}
					}

					if (ia < 0)
					{
						throw new InvalidOperationException("rotation left the vertex fan");
					}

					// dir 0 crosses the edge leaving `a`; dir 1 the edge entering.
					int eStep = dirn == 0 ? ia : (ia + 2) % 3;
					int n = this.tris[cur].Adj[eStep];
					if (n < 0 || n == seed)
					{
						break;
					}

					cur = n;
				}
			}

			return null;
		}

		/// <summary>
		/// Index of <paramref name="t"/>'s edge shared with triangle <paramref name="n"/>
		/// (throws if not adjacent — an internal-consistency violation, not an input
		/// condition).
		/// </summary>
		/// <param name="t">The triangle whose edge is wanted.</param>
		/// <param name="n">The neighbor across it.</param>
		/// <returns>The edge index in <paramref name="t"/>.</returns>
		private int SharedEdge(int t, int n)
		{
			for (int e = 0; e < 3; e++)
			{
				if (this.tris[t].Adj[e] == n)
				{
					return e;
				}
			}

			throw new InvalidOperationException("adjacency tables out of sync");
		}

		/// <summary>Point back the neighbor across (<paramref name="t"/>, <paramref name="e"/>) (if any) to <paramref name="newT"/>.</summary>
		/// <param name="t">The triangle being replaced.</param>
		/// <param name="e">Its edge.</param>
		/// <param name="newT">The replacement triangle.</param>
		private void Rewire(int t, int e, int newT)
		{
			int n = this.tris[t].Adj[e];
			if (n >= 0)
			{
				int ne = this.SharedEdge(n, t);
				this.tris[n].Adj[ne] = newT;
			}
		}

		/// <summary>Insert point <paramref name="p"/>, splitting whichever triangle contains it.</summary>
		/// <param name="p">The point index.</param>
		internal void InsertPoint(int p)
		{
			(int t, TriLoc loc) = this.Locate(p);
			switch (loc.Kind)
			{
				case TriLocKind.Inside:
					this.SplitInterior(t, p);
					break;
				case TriLocKind.OnEdge:
					this.SplitEdge(t, loc.Index, p);
					break;
				case TriLocKind.OnVertex:
					Debug.Assert(false, $"duplicate point reached CDT: {p}");
					break;
				default:
					throw new InvalidOperationException("point outside base triangle");
			}
		}

		/// <summary>Locate the live triangle containing <paramref name="p"/>.</summary>
		/// <param name="p">The point index.</param>
		/// <returns>The triangle and where in it the point lies.</returns>
		private (int Tri, TriLoc Loc) Locate(int p)
		{
			// Visibility walk from the most recent live triangle: step across any
			// edge whose line strictly separates the point (points are inserted
			// before constraints, so the triangulation is Delaunay here and the
			// walk terminates — Edelsbrunner). The step cap is a safety net; on
			// overrun the exhaustive scan below still answers.
			int start = -1;
			for (int i = this.triCount - 1; i >= 0; i--)
			{
				if (this.tris[i].Alive)
				{
					start = i;
					break;
				}
			}

			if (start >= 0)
			{
				int cur = start;
				int steps = 0;
				int cap = (4 * this.triCount) + 16;
				while (steps < cap)
				{
					steps++;
					TriLoc loc = this.LocInTri(p, this.tris[cur].V);
					if (loc != TriLoc.Outside)
					{
						return (cur, loc);
					}

					bool stepped = false;
					for (int e = 0; e < 3; e++)
					{
						// CCW triangle: strictly outside edge e ⇔ orient2d neg.
						if (this.O2(this.tris[cur].V[e], this.tris[cur].V[(e + 1) % 3], p) == Sign.Neg)
						{
							int n = this.tris[cur].Adj[e];
							if (n >= 0)
							{
								cur = n;
								stepped = true;
								break;
							}
						}
					}

					if (!stepped)
					{
						break; // no separating edge with a neighbor — fall through
					}
				}
			}

			for (int i = 0; i < this.triCount; i++)
			{
				if (!this.tris[i].Alive)
				{
					continue;
				}

				if (this.LocInTri(p, this.tris[i].V) != TriLoc.Outside)
				{
					return (i, this.LocInTri(p, this.tris[i].V));
				}
			}

			throw new InvalidOperationException($"point {p} not located in any live triangle");
		}

		/// <summary>
		/// Split triangle <paramref name="t"/> = (a,b,c) at interior point
		/// <paramref name="p"/> into (a,b,p), (b,c,p), (c,a,p).
		/// </summary>
		/// <param name="t">The triangle to split.</param>
		/// <param name="p">The interior point index.</param>
		private void SplitInterior(int t, int p)
		{
			Tri old = this.tris[t];
			int firstNew = this.triCount;

			// New triangle k (k=0,1,2) covers edge k of the old triangle; its
			// inner edges connect to the cyclic neighbors.
			for (int k = 0; k < 3; k++)
			{
				int next = firstNew + ((k + 1) % 3);
				int prev = firstNew + ((k + 2) % 3);
				this.PushTri(new Tri
				{
					V = new IVec3(old.V[k], old.V[(k + 1) % 3], p),
					Adj = new IVec3(old.Adj[k], next, prev),
					Con = new Bool3(old.Con[k], false, false),
					Alive = true,
				});
			}

			for (int k = 0; k < 3; k++)
			{
				this.Rewire(t, k, firstNew + k);
				this.Record(firstNew + k);
			}

			this.tris[t].Alive = false;
		}

		/// <summary>
		/// Split edge <paramref name="e"/> of triangle <paramref name="t"/> at point
		/// <paramref name="p"/> lying exactly on it — two new triangles per side of the
		/// edge.
		/// </summary>
		/// <param name="t">The triangle owning the edge.</param>
		/// <param name="e">The edge index.</param>
		/// <param name="p">The point index, exactly on that edge.</param>
		private void SplitEdge(int t, int e, int p)
		{
			int n = this.tris[t].Adj[e];
			this.SplitEdgeOneSide(t, e, p);
			if (n >= 0)
			{
				int ne = this.SharedEdge(n, t);

				// SplitEdgeOneSide left dangling adjacency into the dead
				// triangle t; splitting the neighbor rewires everything below.
				this.SplitEdgeOneSide(n, ne, p);

				// The four halves now flank the split point: wire them to each
				// other. The two t-halves still point at n and vice versa.
				this.FixSplitPairs(p);
			}
		}

		/// <summary>
		/// Replace <paramref name="t"/> = (a,b,c) with (a,p,c) and (p,b,c), where p lies on
		/// edge e=(a,b). The two halves keep the old adjacency indices toward the (possibly
		/// split) neighbor across e; <see cref="FixSplitPairs"/> repairs those when both
		/// sides exist.
		/// </summary>
		/// <param name="t">The triangle to halve.</param>
		/// <param name="e">The split edge's index in it.</param>
		/// <param name="p">The point index, exactly on that edge.</param>
		private void SplitEdgeOneSide(int t, int e, int p)
		{
			Tri old = this.tris[t];
			int a = old.V[e];
			int b = old.V[(e + 1) % 3];
			int c = old.V[(e + 2) % 3];
			int adjAb = old.Adj[e];
			int adjBc = old.Adj[(e + 1) % 3];
			int adjCa = old.Adj[(e + 2) % 3];
			bool conAb = old.Con[e];
			bool conBc = old.Con[(e + 1) % 3];
			bool conCa = old.Con[(e + 2) % 3];
			int t1 = this.triCount;
			int t2 = t1 + 1;

			// (a, p, c): edges (a,p) toward old neighbor, (p,c) inner, (c,a) old.
			this.PushTri(new Tri
			{
				V = new IVec3(a, p, c),
				Adj = new IVec3(adjAb, t2, adjCa),
				Con = new Bool3(conAb, false, conCa),
				Alive = true,
			});

			// (p, b, c): edges (p,b) toward old neighbor, (b,c) old, (c,p) inner.
			this.PushTri(new Tri
			{
				V = new IVec3(p, b, c),
				Adj = new IVec3(adjAb, adjBc, t1),
				Con = new Bool3(conAb, conBc, false),
				Alive = true,
			});
			this.Rewire(t, (e + 1) % 3, t2);
			this.Rewire(t, (e + 2) % 3, t1);
			this.Record(t1);
			this.Record(t2);
			this.tris[t].Alive = false;
		}

		/// <summary>
		/// After both sides of a split edge have been divided, adjacency across the split
		/// still names the two dead triangles. Match the four live halves incident to
		/// <paramref name="p"/> whose cross-split edges contain <paramref name="p"/>,
		/// pairing edges with identical endpoint sets.
		/// </summary>
		/// <remarks>
		/// Only the four halves the two <see cref="SplitEdgeOneSide"/> calls just pushed can
		/// still name a dead triangle — every other neighbor of the two dead triangles was
		/// repaired by <see cref="Rewire"/> — so the search is confined to the tail of the
		/// arena. The old full-arena scan made every on-edge insertion O(arena) and the
		/// whole triangulation quadratic on monster arrangements. The tail is visited in
		/// ascending index order, the same relative order the full scan used, so the pairing
		/// decisions are unchanged.
		/// </remarks>
		/// <param name="p">The split point index.</param>
		private void FixSplitPairs(int p)
		{
			// Collect (tri, edge) pairs whose adjacency points at a dead triangle.
			int first = Math.Max(this.triCount - 4, 0);
			List<(int Tri, int Edge)> dangling = new List<(int, int)>();
			for (int i = first; i < this.triCount; i++)
			{
				if (!this.tris[i].Alive)
				{
					continue;
				}

				for (int e = 0; e < 3; e++)
				{
					int n = this.tris[i].Adj[e];
					if (n >= 0 && !this.tris[n].Alive)
					{
						Debug.Assert(
							this.tris[i].V[e] == p || this.tris[i].V[(e + 1) % 3] == p,
							"dangling edge must touch the split point");
						dangling.Add((i, e));
					}
				}
			}

			Debug.Assert(
				dangling.Count == this.CountDanglingBefore(first) + dangling.Count,
				"dangling adjacency outside the four freshly split halves");
			for (int i = 0; i < dangling.Count; i++)
			{
				for (int j = i + 1; j < dangling.Count; j++)
				{
					(int ti, int ei) = dangling[i];
					(int tj, int ej) = dangling[j];
					int vi0 = this.tris[ti].V[ei];
					int vi1 = this.tris[ti].V[(ei + 1) % 3];
					int vj0 = this.tris[tj].V[ej];
					int vj1 = this.tris[tj].V[(ej + 1) % 3];
					if (vi0 == vj1 && vi1 == vj0)
					{
						this.tris[ti].Adj[ei] = tj;
						this.tris[tj].Adj[ej] = ti;
					}
				}
			}
		}

		/// <summary>
		/// How many edges below <paramref name="first"/> still point at a dead triangle —
		/// the Rust's inline <c>debug_assert_eq!</c> comprehension in
		/// <see cref="FixSplitPairs"/>, which must always find none.
		/// </summary>
		/// <param name="first">The start of the freshly split tail.</param>
		/// <returns>The count of dangling edges outside that tail.</returns>
		private int CountDanglingBefore(int first)
		{
			int count = 0;
			for (int i = 0; i < first; i++)
			{
				if (!this.tris[i].Alive)
				{
					continue;
				}

				for (int e = 0; e < 3; e++)
				{
					int n = this.tris[i].Adj[e];
					if (n >= 0 && !this.tris[n].Alive)
					{
						count++;
					}
				}
			}

			return count;
		}

		/// <summary>
		/// Enqueue every triangle from <paramref name="firstNew"/> onward (the ones an
		/// insert or constraint pass just created) as legalization suspects.
		/// </summary>
		/// <param name="firstNew">The arena length before the pass.</param>
		internal void SeedSuspects(int firstNew)
		{
			for (int t = firstNew; t < this.triCount; t++)
			{
				this.suspects.Add(t);
			}
		}

		/// <summary>
		/// Queue-based Lawson legalization: flip every non-constrained, strictly
		/// non-Delaunay, flippable edge reachable from the suspect set. A flip enqueues the
		/// two replacement triangles, and their neighbors get re-tested through the shared
		/// edges when those triangles are examined, so the worklist reaches everything the
		/// old global fixpoint rescan reached. Termination is argued at
		/// <see cref="TryFlip"/>.
		/// </summary>
		internal void LegalizeSuspects()
		{
			while (this.suspects.Count > 0)
			{
				int t = this.suspects[this.suspects.Count - 1];
				this.suspects.RemoveAt(this.suspects.Count - 1);
				if (t >= this.triCount || !this.tris[t].Alive)
				{
					continue;
				}

				for (int e = 0; e < 3; e++)
				{
					if (this.TryFlip(t, e))
					{
						// t is dead; its replacements are the last two triangles,
						// and the flipped neighbor's replacement edges are on them.
						int n = this.triCount;
						this.suspects.Add(n - 2);
						this.suspects.Add(n - 1);
						break;
					}
				}
			}
		}

		/// <summary>
		/// Flip edge <paramref name="e"/> of <paramref name="t"/> if it is internal,
		/// unconstrained, its quad is strictly convex, and the flip either repairs a strict
		/// Delaunay violation or resolves an exact cocircular tie toward the canonical
		/// (lowest <see cref="DiagKey"/>) diagonal. Returns whether a flip happened.
		/// </summary>
		/// <remarks>
		/// <para>
		/// Tie flips make the triangulation of a cocircular quad a function of its point
		/// coordinates instead of its construction history, so coincident coplanar triangles
		/// tile shared regions identically (found on Thingi10K #36088, where a doubled
		/// sheet's two copies picked opposite diagonals of an exact square and the cell
		/// complex saw four [1,0]-step walls where the geometry has [2,0]-step stacks).
		/// </para>
		/// <para>
		/// Termination: a strict flip strictly lowers the lifted-paraboloid volume; a tie
		/// flip keeps it equal (that is what cocircular means for the lift) while strictly
		/// lowering the multiset of diagonal keys. The pair (lift volume, key multiset)
		/// therefore strictly decreases lexicographically on every flip.
		/// </para>
		/// </remarks>
		/// <param name="t">The triangle owning the candidate edge.</param>
		/// <param name="e">The edge index.</param>
		/// <returns>True when the edge was flipped.</returns>
		private bool TryFlip(int t, int e)
		{
			if (this.tris[t].Con[e])
			{
				return false;
			}

			int n = this.tris[t].Adj[e];
			if (n < 0)
			{
				return false;
			}

			int ne = this.SharedEdge(n, t);
			int a = this.tris[t].V[e];
			int b = this.tris[t].V[(e + 1) % 3];
			int c = this.tris[t].V[(e + 2) % 3];
			int d = this.tris[n].V[(ne + 2) % 3];
			switch (this.Incircle(this.tris[t].V, d))
			{
				case Sign.Neg:
					return false;
				case Sign.Zero:
					// Cocircular: flip only toward the canonical diagonal.
					if (Compare(this.DiagKey(c, d), this.DiagKey(a, b)) >= 0)
					{
						return false;
					}

					break;
				default:
					break;
			}

			// Strictly convex quad a-d-b-c (new triangles must be CCW)?
			if (this.O2(a, d, c) != Sign.Pos || this.O2(d, b, c) != Sign.Pos)
			{
				return false;
			}

			this.Flip(t, e, n, ne);
			return true;
		}

		/// <summary>
		/// Lexicographic comparison of two diagonal keys — Rust compares the <c>(u32, u32)</c>
		/// tuples with <c>&gt;=</c>, which C# value tuples do not define as an operator.
		/// </summary>
		/// <param name="x">The left key.</param>
		/// <param name="y">The right key.</param>
		/// <returns>Negative, zero or positive as <paramref name="x"/> orders before, with or after <paramref name="y"/>.</returns>
		private static int Compare((int Low, int High) x, (int Low, int High) y)
		{
			int c = x.Low.CompareTo(y.Low);
			return c != 0 ? c : x.High.CompareTo(y.High);
		}

		/// <summary>
		/// Replace triangles t=(a,b,c) and n=(b,a,d) with (a,d,c) and (d,b,c).
		/// </summary>
		/// <param name="t">The first triangle.</param>
		/// <param name="e">The shared edge's index in <paramref name="t"/>.</param>
		/// <param name="n">The second triangle.</param>
		/// <param name="ne">The shared edge's index in <paramref name="n"/>.</param>
		private void Flip(int t, int e, int n, int ne)
		{
			int a = this.tris[t].V[e];
			int b = this.tris[t].V[(e + 1) % 3];
			int c = this.tris[t].V[(e + 2) % 3];
			int d = this.tris[n].V[(ne + 2) % 3];
			Debug.Assert(this.tris[n].V[ne] == b, "the neighbor's shared edge runs b → a");
			Debug.Assert(this.tris[n].V[(ne + 1) % 3] == a, "the neighbor's shared edge runs b → a");

			int adjBc = this.tris[t].Adj[(e + 1) % 3];
			bool conBc = this.tris[t].Con[(e + 1) % 3];
			int adjCa = this.tris[t].Adj[(e + 2) % 3];
			bool conCa = this.tris[t].Con[(e + 2) % 3];
			int adjAd = this.tris[n].Adj[(ne + 1) % 3];
			bool conAd = this.tris[n].Con[(ne + 1) % 3];
			int adjDb = this.tris[n].Adj[(ne + 2) % 3];
			bool conDb = this.tris[n].Con[(ne + 2) % 3];

			int t1 = this.triCount;
			int t2 = t1 + 1;

			// (a, d, c): edges (a,d) old, (d,c) diagonal, (c,a) old.
			this.PushTri(new Tri
			{
				V = new IVec3(a, d, c),
				Adj = new IVec3(adjAd, t2, adjCa),
				Con = new Bool3(conAd, false, conCa),
				Alive = true,
			});

			// (d, b, c): edges (d,b) old, (b,c) old, (c,d) diagonal.
			this.PushTri(new Tri
			{
				V = new IVec3(d, b, c),
				Adj = new IVec3(adjDb, adjBc, t1),
				Con = new Bool3(conDb, conBc, false),
				Alive = true,
			});
			this.Rewire(t, (e + 1) % 3, t2);
			this.Rewire(t, (e + 2) % 3, t1);
			this.Rewire(n, (ne + 1) % 3, t1);
			this.Rewire(n, (ne + 2) % 3, t2);
			this.Record(t1);
			this.Record(t2);
			this.tris[t].Alive = false;
			this.tris[n].Alive = false;
		}
	}
}
