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

// Cdt.Constraints.cs — port of robust/cdt_constraints.rs, whose header reads:
//
//   Constraint recovery for robust/cdt.rs.
//
//   The second half of the constrained Delaunay triangulator: forcing a given
//   index pair to appear as an edge of the triangulation via Anglada's cavity
//   retriangulation. Split out of cdt.rs purely for file size; it is a second
//   `impl Cdt` block on the same private arena and shares every helper
//   (`RotateAround`, `SharedEdge`, `Record`, the filtered predicates) with its
//   parent module.
//
// Two Rust files, one type — so two C# files, one partial class.

using System.Runtime.InteropServices;

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Robust
{
	/// <content>Anglada cavity retriangulation: constraint recovery.</content>
	internal sealed partial class Cdt
	{
		/// <summary>
		/// Recover constraint edge (a,b) by Anglada's cavity retriangulation: walk the
		/// corridor of triangles the open segment pierces, delete them, and re-triangulate
		/// the two pseudo-polygon halves flanking the segment (each necessarily gains edge
		/// (a,b)). Edge-flip recovery — flipping away crossing edges one at a time — is NOT
		/// used because it can cycle: when a flipped diagonal still crosses the segment, the
		/// next search can select it and flip it straight back (observed on Thingi10K
		/// 1075458 − 91115, where two edges alternated forever and the triangle arena grew
		/// without bound). The corridor walk is bounded by the live triangle count and the
		/// retriangulation recursion by the chain length, so this terminates
		/// unconditionally.
		/// Precondition (from robust/arrangement.rs): no vertex lies strictly inside segment
		/// (a,b) and no other constraint properly crosses it.
		/// </summary>
		/// <param name="a">One endpoint's point index.</param>
		/// <param name="b">The other endpoint's point index.</param>
		internal void InsertConstraint(int a, int b)
		{
			if (this.MarkIfPresent(a, b))
			{
				return;
			}

			(List<int> crossed, List<int> left, List<int> right) = this.CollectCorridor(a, b);
			this.RetriangulateCorridor(a, b, crossed, left, right);
			if (!this.MarkIfPresent(a, b))
			{
				throw new InvalidOperationException(
					$"cavity retriangulation did not produce constraint edge ({a},{b})");
			}
		}

		/// <summary>
		/// The corridor of triangles pierced by open segment (a,b): the triangle list in
		/// walk order plus the chains of corridor vertices strictly left and strictly right
		/// of the directed line a→b, each ordered from a to b. Chains may repeat a vertex
		/// (pseudo-polygon pinch); consecutive duplicates cannot occur.
		/// </summary>
		/// <param name="a">The segment's start point index.</param>
		/// <param name="b">The segment's end point index.</param>
		/// <returns>The crossed triangles and the two vertex chains.</returns>
		private (List<int> Crossed, List<int> Left, List<int> Right) CollectCorridor(int a, int b)
		{
			// Entry: the incident triangle of `a` whose opposite edge the segment
			// leaves through. No vertex lies strictly inside (a,b) and (a,b) is
			// not an edge (MarkIfPresent said so), so the exit is a strict edge
			// crossing.
			(int T, int E)? entry = this.RotateAround<(int T, int E)>(a, t =>
			{
				int ia = this.VertexSlot(t, a);
				int eOpp = (ia + 1) % 3;
				int u0 = this.tris[t].V[eOpp];
				int v0 = this.tris[t].V[(eOpp + 1) % 3];
				if (u0 != b && v0 != b && this.StrictlyCrosses(a, b, u0, v0))
				{
					return (t, eOpp);
				}

				return null;
			});
			if (!entry.HasValue)
			{
				throw new InvalidOperationException("segment leaves its start vertex through a crossed edge");
			}

			int t0 = entry.Value.T;
			int e0 = entry.Value.E;

			// CCW triangle (a, u, v): u is strictly right of a→b, v strictly left.
			int u = this.tris[t0].V[e0];
			int v = this.tris[t0].V[(e0 + 1) % 3];
			List<int> crossed = new List<int> { t0 };
			List<int> left = new List<int> { v };
			List<int> right = new List<int> { u };
			int tCur = t0;
			int eCur = e0;
			while (true)
			{
				if (crossed.Count > this.triCount)
				{
					throw new InvalidOperationException("corridor walk revisited a triangle — triangulation corrupt");
				}

				int n = this.tris[tCur].Adj[eCur];
				if (n < 0)
				{
					throw new InvalidOperationException("segment crossed the hull — point outside base triangle");
				}

				int ne = this.SharedEdge(n, tCur);
				crossed.Add(n);
				int w = this.tris[n].V[(ne + 2) % 3];
				if (w == b)
				{
					return (crossed, left, right);
				}

				switch (this.O2(a, b, w))
				{
					case Sign.Pos:
						left.Add(w);
						break;
					case Sign.Neg:
						right.Add(w);
						break;
					default:
						throw new InvalidOperationException(
							$"vertex {w} lies on constraint segment ({a},{b}) — precondition violated");
				}

				// The segment exits `n` through whichever remaining edge it
				// strictly crosses (exactly one, since w is strictly off-line).
				int exit = -1;
				for (int k = 1; k < 3; k++)
				{
					int e2 = (ne + k) % 3;
					int p = this.tris[n].V[e2];
					int q = this.tris[n].V[(e2 + 1) % 3];
					if (this.StrictlyCrosses(a, b, p, q))
					{
						exit = e2;
						break;
					}
				}

				if (exit < 0)
				{
					throw new InvalidOperationException("segment must exit the corridor triangle it entered");
				}

				eCur = exit;
				tCur = n;
			}
		}

		/// <summary>Which corner of triangle <paramref name="t"/> is vertex <paramref name="a"/>.</summary>
		/// <param name="t">The triangle index.</param>
		/// <param name="a">The vertex index.</param>
		/// <returns>The corner index, 0..2.</returns>
		private int VertexSlot(int t, int a)
		{
			for (int k = 0; k < 3; k++)
			{
				if (this.tris[t].V[k] == a)
				{
					return k;
				}
			}

			throw new InvalidOperationException($"vertex {a} is not a corner of triangle {t}");
		}

		/// <summary>
		/// Is <paramref name="v"/> strictly inside the circumcircle of triangle (a,b,c),
		/// whatever its orientation? Collinear (a,b,c) has no circumcircle: false.
		/// </summary>
		/// <param name="a">First triangle vertex.</param>
		/// <param name="b">Second triangle vertex.</param>
		/// <param name="c">Third triangle vertex.</param>
		/// <param name="v">The query vertex.</param>
		/// <returns>True when strictly inside.</returns>
		private bool InCircumcircle(int a, int b, int c, int v)
		{
			switch (this.O2(a, b, c))
			{
				case Sign.Pos:
					return this.NonDelaunay(new IVec3(a, b, c), v);
				case Sign.Neg:
					return this.NonDelaunay(new IVec3(a, c, b), v);
				default:
					return false;
			}
		}

		/// <summary>
		/// Anglada's pseudo-polygon triangulation: the region bounded by directed base edge
		/// (a,b) and <paramref name="chain"/> (the boundary path from a to b, region on the
		/// left of a→b). Picks the Delaunay apex c — the chain vertex whose circumcircle
		/// with the base is empty of chain vertices — emits CCW triangle (a,b,c), and
		/// recurses on the two sub-chains.
		/// </summary>
		/// <param name="a">The base edge's start.</param>
		/// <param name="b">The base edge's end.</param>
		/// <param name="chain">The boundary path from a to b.</param>
		/// <param name="outTris">Where the emitted triangles accumulate.</param>
		private void TriangulatePseudo(int a, int b, ReadOnlySpan<int> chain, List<IVec3> outTris)
		{
			if (chain.IsEmpty)
			{
				return;
			}

			int ci = 0;
			for (int i = 1; i < chain.Length; i++)
			{
				// A collinear "apex" has no circumcircle and can never be the
				// Delaunay apex; any candidate supersedes it.
				bool replace = this.O2(a, b, chain[ci]) == Sign.Zero
					|| this.InCircumcircle(a, b, chain[ci], chain[i]);
				if (replace)
				{
					ci = i;
				}
			}

			int c = chain[ci];
			if (this.O2(a, b, c) != Sign.Pos)
			{
				throw new InvalidOperationException("pseudo-polygon apex must lie strictly left of its base");
			}

			this.TriangulatePseudo(a, c, chain.Slice(0, ci), outTris);
			this.TriangulatePseudo(c, b, chain.Slice(ci + 1), outTris);
			outTris.Add(new IVec3(a, b, c));
		}

		/// <summary>
		/// Replace the corridor triangles with the CDTs of the two pseudo-polygon halves and
		/// wire the new patch into the surrounding triangulation.
		/// </summary>
		/// <param name="a">The constraint's start point index.</param>
		/// <param name="b">The constraint's end point index.</param>
		/// <param name="crossed">The corridor triangles, in walk order.</param>
		/// <param name="left">The corridor chain strictly left of a→b.</param>
		/// <param name="right">The corridor chain strictly right of a→b.</param>
		private void RetriangulateCorridor(int a, int b, List<int> crossed, List<int> left, List<int> right)
		{
			// Fx hashing (unseeded) in the Rust; plain Dictionary/HashSet here for the
			// same reason the porting plan sanctions everywhere else: both maps below
			// are probe-only scratch — `boundary` is looked up per patch edge and
			// `open` is a pairing buffer drained by key, neither is ever iterated.
			// Cavity boundary: edges of corridor triangles whose neighbor is
			// outside the corridor. Keyed by undirected endpoints (each boundary
			// edge has exactly one inside face, so keys are unique). Interior
			// edges shared by two corridor triangles vanish; a constrained one
			// (possible only at a pseudo-polygon pinch) must be re-marked after
			// the rebuild.
			Dictionary<(int Low, int High), (int Adj, bool Con)> boundary =
				new Dictionary<(int, int), (int, bool)>();
			List<(int Low, int High)> interiorCon = new List<(int, int)>();

			// Corridor membership: the linear `crossed.Contains` this replaces made
			// the loop below O(|corridor|²), which long corridors on monster
			// arrangements do reach. The set is only worth its allocation past a
			// handful of triangles, and — being probe-only, never iterated — the
			// hasher cannot affect determinism.
			HashSet<int>? corridorSet = crossed.Count > 16 ? new HashSet<int>(crossed) : null;
			foreach (int t in crossed)
			{
				for (int e = 0; e < 3; e++)
				{
					int n = this.tris[t].Adj[e];
					int u = this.tris[t].V[e];
					int v = this.tris[t].V[(e + 1) % 3];
					(int, int) key = (Math.Min(u, v), Math.Max(u, v));
					bool inCorridor = n >= 0
						&& (corridorSet != null ? corridorSet.Contains(n) : crossed.Contains(n));
					if (n < 0 || !inCorridor)
					{
						boundary[key] = (n, this.tris[t].Con[e]);
					}
					else if (this.tris[t].Con[e])
					{
						interiorCon.Add(key);
					}
				}
			}

			// Left half: region left of a→b, chain ordered a to b. Right half:
			// region left of b→a, so its chain must run b to a.
			List<IVec3> newTris = new List<IVec3>();
			this.TriangulatePseudo(a, b, CollectionsMarshal.AsSpan(left), newTris);
			List<int> rightRev = new List<int>(right);
			rightRev.Reverse();
			this.TriangulatePseudo(b, a, CollectionsMarshal.AsSpan(rightRev), newTris);

			// Push the patch, then wire adjacency: boundary edges reconnect to
			// the outside; the rest pair up patch-internally (including (a,b)
			// between the two halves).
			int patchBase = this.triCount;
			foreach (IVec3 tv in newTris)
			{
				this.PushTri(new Tri
				{
					V = tv,
					Adj = new IVec3(-1, -1, -1),
					Con = default(Bool3),
					Alive = true,
				});
			}

			Dictionary<(int Low, int High), (int T, int E)> open =
				new Dictionary<(int, int), (int, int)>();
			for (int t = patchBase; t < this.triCount; t++)
			{
				for (int e = 0; e < 3; e++)
				{
					int u = this.tris[t].V[e];
					int v = this.tris[t].V[(e + 1) % 3];
					(int, int) key = (Math.Min(u, v), Math.Max(u, v));
					if (boundary.TryGetValue(key, out (int Adj, bool Con) bound))
					{
						this.tris[t].Adj[e] = bound.Adj;
						this.tris[t].Con[e] = bound.Con;
						if (bound.Adj >= 0)
						{
							int n = bound.Adj;
							int ne = -1;
							for (int k = 0; k < 3; k++)
							{
								int nu = this.tris[n].V[k];
								int nv = this.tris[n].V[(k + 1) % 3];
								if ((Math.Min(nu, nv), Math.Max(nu, nv)) == key)
								{
									ne = k;
									break;
								}
							}

							if (ne < 0)
							{
								throw new InvalidOperationException("cavity boundary neighbor lost its edge");
							}

							this.tris[n].Adj[ne] = t;
						}
					}
					else if (open.TryGetValue(key, out (int T, int E) mate))
					{
						open.Remove(key);
						this.tris[t].Adj[e] = mate.T;
						this.tris[mate.T].Adj[mate.E] = t;
					}
					else
					{
						open[key] = (t, e);
					}
				}
			}

			if (open.Count != 0)
			{
				throw new InvalidOperationException("cavity retriangulation left unmatched patch edges");
			}

			// vertTri invariant: record replacements before killing the corridor.
			for (int t = patchBase; t < this.triCount; t++)
			{
				this.Record(t);
			}

			foreach (int t in crossed)
			{
				this.tris[t].Alive = false;
			}

			foreach ((int u, int v) in interiorCon)
			{
				if (!this.MarkIfPresent(u, v))
				{
					throw new InvalidOperationException(
						$"constrained pinch edge ({u},{v}) was not reproduced by the rebuild");
				}
			}
		}

		/// <summary>
		/// If edge (a,b) exists, set its constrained flag on both sides and return true.
		/// Rotation around <paramref name="a"/> visits every incident triangle, so absence
		/// there is definitive — no global scan.
		/// </summary>
		/// <param name="a">One endpoint's point index.</param>
		/// <param name="b">The other endpoint's point index.</param>
		/// <returns>True when the edge was present and is now constrained.</returns>
		private bool MarkIfPresent(int a, int b)
		{
			(int T, int E)? found = this.RotateAround<(int T, int E)>(a, t =>
			{
				for (int e = 0; e < 3; e++)
				{
					int u = this.tris[t].V[e];
					int v = this.tris[t].V[(e + 1) % 3];
					if ((u == a && v == b) || (u == b && v == a))
					{
						return (t, e);
					}
				}

				return null;
			});

			if (!found.HasValue)
			{
				return false;
			}

			int tf = found.Value.T;
			int ef = found.Value.E;
			this.tris[tf].Con[ef] = true;
			int adj = this.tris[tf].Adj[ef];
			if (adj >= 0)
			{
				int ne = this.SharedEdge(adj, tf);
				this.tris[adj].Con[ne] = true;
			}

			return true;
		}

		/// <summary>Strict proper-crossing test of edge (u,v) against segment (a,b).</summary>
		/// <param name="a">The segment's start point index.</param>
		/// <param name="b">The segment's end point index.</param>
		/// <param name="u">The edge's first endpoint.</param>
		/// <param name="v">The edge's second endpoint.</param>
		/// <returns>True when the two cross properly.</returns>
		private bool StrictlyCrosses(int a, int b, int u, int v)
		{
			Sign su = this.O2(a, b, u);
			Sign sv = this.O2(a, b, v);
			if (su == Sign.Zero || sv == Sign.Zero || su == sv)
			{
				return false;
			}

			Sign sa = this.O2(u, v, a);
			Sign sb = this.O2(u, v, b);
			return sa != Sign.Zero && sb != Sign.Zero && sa != sb;
		}
	}
}
