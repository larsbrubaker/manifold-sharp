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

// EarClip triangulator — extracted from polygon.rs
// Port of C++ ear-clipping algorithm with 2D KD-tree acceleration
//
// ── C# port of polygon_earclip.rs (file 1 of 2) ──────────────────────────────
// The Rust file is 833 lines and its C# expansion does not fit the 800-line cap,
// so it lands as two partial-class files, both `internal sealed partial class
// EarClip`:
//   PolygonEarclip.cs           the supporting types, construction, the
//                               linked-list helpers and the Vert predicates
//   PolygonEarclip.Algorithm.cs clipping, keyholing and the ear-queue loop
//
// `Vert`, `EarEntry` and `IdxCollider` are nested types: they are module-private
// in the Rust (the module itself being private inside polygon.rs), and nesting is
// the C# reach that matches. `EarClip` is `internal`, which is the Rust
// `pub(super)`.
//
// The Rust names the vert arena `polygon`; here it is `PolygonVerts`, because
// `Polygon` is the static class this file calls into and a member of that name
// would shadow it. Members the Rust keeps private are `internal` here for the one
// reason InternalsVisibleTo exists in this port: the Rust tests are a child
// module and see their parent's privates (polygon_earclip_tests.rs reads
// `ear_clip.holes` and `ear_clip.polygon[..]`).
//
// The arena is a `Vert[]` plus a count, not a `List<Vert>`: `polygon[v].cost = x`
// is on nearly every line of this algorithm, and a `List<T>` of structs hands
// back a *copy* from its indexer, so that assignment would not compile — and the
// obvious "fix" of making Vert a class trades the arena for an object graph.
// Array elements are real variables, so the Rust text transcribes directly.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;
using static ManifoldSharp.Polygon;

namespace ManifoldSharp
{
	/// <summary>
	/// The ear-clipping triangulator: Rust <c>EarClip</c> in polygon_earclip.rs.
	/// </summary>
	internal sealed partial class EarClip
	{
		/// <summary>A vertex of the doubly-linked polygon ring being clipped.</summary>
		internal struct Vert
		{
			/// <summary>The original input index this vert carries into the output triangles.</summary>
			internal int MeshIdx;

			/// <summary>The ear cost last computed for this vert; 1.0 marks a reflex (non-ear) vert.</summary>
			internal double Cost;

			/// <summary>Bumped on every requeue so stale queue entries can be lazily discarded.</summary>
			internal uint EarVersion;

			/// <summary>The vertex position.</summary>
			internal Vec2 Pos;

			/// <summary>Unit vector from this vert to <see cref="Right"/>, maintained by <c>Link</c>.</summary>
			internal Vec2 RightDir;

			/// <summary>Index of the previous vert in the ring.</summary>
			internal int Left;

			/// <summary>Index of the next vert in the ring.</summary>
			internal int Right;
		}

		/// <summary>Entry in the min-heap ear queue.</summary>
		private readonly struct EarEntry
		{
			/// <summary>The ear cost; lower is clipped sooner.</summary>
			internal readonly double Cost;

			/// <summary>The vert this entry refers to.</summary>
			internal readonly int Idx;

			/// <summary>The vert's <see cref="Vert.EarVersion"/> when this entry was queued.</summary>
			internal readonly uint Version;

			/// <summary>
			/// Monotonic insertion sequence. C++ uses a std::multiset, which keeps
			/// equal-cost entries in INSERTION order and pops the oldest first —
			/// equal costs are common (kBest = -inf for short ears), so this FIFO
			/// tie-break decides which degenerate ear clips first and must match.
			/// </summary>
			internal readonly ulong Seq;

			internal EarEntry(double cost, int idx, uint version, ulong seq)
			{
				this.Cost = cost;
				this.Idx = idx;
				this.Version = version;
				this.Seq = seq;
			}
		}

		/// <summary>
		/// The ear queue's order: lowest cost first, oldest insertion first among equal
		/// costs.
		/// </summary>
		/// <remarks>
		/// The <c>Seq</c> tie-break is not decoration and must not be simplified away.
		/// C#'s <see cref="PriorityQueue{TElement, TPriority}"/> is explicitly *not* a
		/// stable queue: elements with equal priority are dequeued in an unspecified
		/// order that depends on the heap's internal array. Rust's
		/// <c>BinaryHeap</c> is not stable either — which is exactly why the Rust port
		/// put <c>seq</c> in the ordering rather than relying on the container. Keeping
		/// it here makes the order total, so both ports pop the identical entry.
		/// </remarks>
		private sealed class EarEntryComparer : IComparer<EarEntry>
		{
			internal static readonly EarEntryComparer Instance = new EarEntryComparer();

			public int Compare(EarEntry x, EarEntry y)
			{
				// Rust orders for a max-heap ("greater" = popped first): `other.cost
				// .partial_cmp(&self.cost).then(other.seq.cmp(&self.seq))`. C#'s
				// PriorityQueue dequeues the *minimum*, so the same order is spelled the
				// other way up: ascending cost, then ascending seq.
				if (x.Cost < y.Cost)
				{
					return -1;
				}

				if (x.Cost > y.Cost)
				{
					return 1;
				}

				// Equal costs (or a NaN cost, Rust's unwrap_or(Equal)) fall through to the
				// FIFO tie-break.
				return x.Seq.CompareTo(y.Seq);
			}
		}

		/// <summary>The kd-tree of ring verts used to accelerate the ear-cost query.</summary>
		private sealed class IdxCollider
		{
			/// <summary>The kd-tree points; <c>Idx</c> is an index into <see cref="Itr"/>.</summary>
			internal List<PolyVert> Points = new List<PolyVert>();

			/// <summary><c>Itr[i]</c> = polygon vert index for <c>Points[..].Idx == i</c>.</summary>
			internal List<int> Itr = new List<int>();
		}

		/// <summary>
		/// The vert arena — Rust <c>polygon</c>. Grown through <see cref="PushVert"/>; only
		/// the first <see cref="PolygonCount"/> entries are live.
		/// </summary>
		internal Vert[] PolygonVerts;

		/// <summary>The number of live verts in <see cref="PolygonVerts"/>.</summary>
		internal int PolygonCount;

		/// <summary>
		/// Start verts of the hole contours, sorted by descending x by the constructor.
		/// </summary>
		internal List<int> Holes = new List<int>();

		/// <summary>Start verts of the contours with positive area.</summary>
		internal List<int> Outers = new List<int>();

		/// <summary>Start verts of the contours to ear-clip directly.</summary>
		internal List<int> Simples = new List<int>();

		/// <summary>
		/// Bounding box per hole start vert. Probe-only (insert and lookup, never
		/// iterated), so a plain Dictionary is safe here.
		/// </summary>
		private readonly Dictionary<int, Rect> hole2bbox = new Dictionary<int, Rect>();

		/// <summary>The lazily-invalidated min-heap of candidate ears.</summary>
		private readonly PriorityQueue<EarEntry, EarEntry> earsQueue
			= new PriorityQueue<EarEntry, EarEntry>(EarEntryComparer.Instance);

		/// <summary>Monotonic counter for EarEntry.Seq (FIFO tie-break among equal costs).</summary>
		private ulong earSeq;

		/// <summary>The triangles emitted so far.</summary>
		internal List<IVec3> Triangles;

		/// <summary>Bounding box of every input vert, used to derive epsilon when unset.</summary>
		private Rect bbox = new Rect();

		/// <summary>The tolerance in use; resolved from <see cref="bbox"/> when the input was negative.</summary>
		internal double Epsilon;

		/// <summary>
		/// Builds the clipping state for the given polygons: one linked ring per contour,
		/// degenerate verts already clipped, contours classified into holes and simples.
		/// </summary>
		/// <param name="polys">The indexed polygons.</param>
		/// <param name="epsilon">Tolerance; negative means "derive from the bounding box".</param>
		internal EarClip(PolygonsIdx polys, double epsilon)
		{
			int numVert = 0;
			foreach (SimplePolygonIdx p in polys)
			{
				numVert += p.Count;
			}

			this.PolygonVerts = new Vert[numVert + (2 * polys.Count)];
			this.Triangles = new List<IVec3>(numVert + (2 * polys.Count));
			this.Epsilon = epsilon;

			List<int> starts = this.Initialize(polys);

			// Clip degenerate verts
			for (int i = 0; i < this.PolygonCount; i++)
			{
				this.ClipIfDegenerate(i);
			}

			// Find start/classify each polygon
			foreach (int start in starts)
			{
				this.FindStart(start);
			}

			// C++ stores holes_ in a multiset ordered by MaxX. Keyholing from
			// right to left prevents a later bridge from crossing an earlier one.
			// Rust `sort_by` is stable, so this is OrderBy with the same reversed
			// comparator (see Polygon.PartialCmpDescending).
			this.Holes = this.Holes
				.OrderBy(hole => this.PolygonVerts[hole].Pos.X, PartialCmpDescending)
				.ToList();
		}

		/// <summary>Set by <see cref="Triangulate"/>; see the guard there.</summary>
		private bool consumed;

		/// <summary>
		/// Keyholes every hole into an outer contour, then ear-clips every simple
		/// contour. Callable once per instance.
		/// </summary>
		/// <returns>The triangles and the epsilon that was actually used.</returns>
		/// <exception cref="InvalidOperationException">
		/// This instance has already been triangulated.
		/// </exception>
		internal (List<IVec3> Triangles, double Epsilon) Triangulate()
		{
			// The Rust signature is `pub fn triangulate(mut self)`: it CONSUMES the
			// EarClip, so calling it twice is a compile error there. C# cannot express
			// that, and a second call is not harmlessly idempotent — the state is
			// single-use. CutKeyhole would run again on holes that are already bridged,
			// appending garbage triangles to the ones already in Triangles. The guard
			// turns that silent corruption into a loud misuse report.
			if (this.consumed)
			{
				throw new InvalidOperationException(
					"EarClip.Triangulate has already run on this instance. The Rust it ports, "
					+ "`triangulate(mut self)`, consumes the value; construct a new EarClip instead.");
			}

			this.consumed = true;

			// Key-hole all holes into outer polygons
			List<int> holes = new List<int>(this.Holes);
			foreach (int start in holes)
			{
				this.CutKeyhole(start);
			}

			// Ear-clip each simple polygon. The copy is taken *after* keyholing, so it
			// includes the holes CutKeyhole failed to bridge and pushed onto Simples.
			List<int> simples = new List<int>(this.Simples);
			foreach (int start in simples)
			{
				this.TriangulatePoly(start);
			}

			double eps = this.Epsilon;
			return (this.Triangles, eps);
		}

		/// <summary>Appends a vert to the arena, growing it if the initial estimate was short.</summary>
		/// <param name="vert">The vert to append.</param>
		/// <returns>The index of the appended vert.</returns>
		private int PushVert(Vert vert)
		{
			if (this.PolygonCount == this.PolygonVerts.Length)
			{
				Array.Resize(ref this.PolygonVerts, Math.Max(4, this.PolygonVerts.Length * 2));
			}

			int idx = this.PolygonCount;
			this.PolygonVerts[idx] = vert;
			this.PolygonCount++;
			return idx;
		}

		// -----------------------------------------------------------------------
		// Linked-list helpers
		// -----------------------------------------------------------------------

		/// <summary>Whether vert <paramref name="v"/> has been unlinked from its ring.</summary>
		/// <param name="v">The vert to test.</param>
		/// <returns>True when the vert is no longer part of the ring.</returns>
		private bool Clipped(int v)
		{
			return this.PolygonVerts[this.PolygonVerts[v].Right].Left != v;
		}

		/// <summary>Links <paramref name="left"/> to <paramref name="right"/> and refreshes the edge direction.</summary>
		/// <param name="left">The vert on the left of the new edge.</param>
		/// <param name="right">The vert on the right of the new edge.</param>
		private void Link(int left, int right)
		{
			this.PolygonVerts[left].Right = right;
			this.PolygonVerts[right].Left = left;
			Vec2 dir = this.PolygonVerts[right].Pos - this.PolygonVerts[left].Pos;
			this.PolygonVerts[left].RightDir = SafeNormalize2d(dir);
		}

		/// <summary>
		/// Collects each unclipped vert in the polygon ring starting from
		/// <paramref name="first"/>. Returns the verts on success, or null if degenerate.
		/// </summary>
		/// <param name="first">The vert to start walking from.</param>
		/// <returns>The ring's verts, or null when the ring has collapsed.</returns>
		private List<int>? LoopVerts(int first)
		{
			List<int> result = new List<int>();
			int v = first;
			int curFirst = first;
			while (true)
			{
				if (this.Clipped(v))
				{
					curFirst = this.PolygonVerts[this.PolygonVerts[v].Right].Left;
					if (!this.Clipped(curFirst))
					{
						v = curFirst;
						if (this.PolygonVerts[v].Right == this.PolygonVerts[v].Left)
						{
							return null;
						}

						result.Add(v);
					}
				}
				else
				{
					if (this.PolygonVerts[v].Right == this.PolygonVerts[v].Left)
					{
						return null;
					}

					result.Add(v);
				}

				v = this.PolygonVerts[v].Right;
				if (v == curFirst)
				{
					break;
				}
			}

			return result;
		}

		// -----------------------------------------------------------------------
		// Vert predicate methods (take vert index `v`)
		// -----------------------------------------------------------------------

		/// <summary>Whether the edge leaving <paramref name="v"/> is shorter than the tolerance.</summary>
		/// <param name="v">The vert to test.</param>
		/// <returns>True when the edge is degenerate-short.</returns>
		private bool VertIsShort(int v)
		{
			Vec2 edge = this.PolygonVerts[this.PolygonVerts[v].Right].Pos - this.PolygonVerts[v].Pos;
			return Dot2d(edge, edge) * 4.0 < this.Epsilon * this.Epsilon;
		}

		// NOTE: C++ `Vert::Interior` (polygon.cpp) is not ported: it is defined but
		// never called in the upstream reference either.

		/// <summary>
		/// Returns true if vert <paramref name="v"/> is on the inside of the edge
		/// tail -&gt; tail.right.
		/// </summary>
		/// <param name="v">The vert being tested.</param>
		/// <param name="tail">The vert at the tail of the edge.</param>
		/// <param name="toLeft">Walk v's polygon edges to the left (vs right).</param>
		/// <returns>True when v is inside the edge.</returns>
		private bool VertInsideEdge(int v, int tail, bool toLeft)
		{
			double p2 = this.Epsilon * this.Epsilon;
			int nextL = this.PolygonVerts[this.PolygonVerts[v].Left].Right;
			int nextR = this.PolygonVerts[tail].Right;
			int center = tail;
			int last = center;

			int vStop = toLeft
				? this.PolygonVerts[v].Right
				: this.PolygonVerts[v].Left;

			while (true)
			{
				if (nextL == nextR || tail == nextR || nextL == vStop)
				{
					break;
				}

				Vec2 edgeL = this.PolygonVerts[nextL].Pos - this.PolygonVerts[center].Pos;
				double l2 = Dot2d(edgeL, edgeL);
				if (l2 <= p2)
				{
					nextL = toLeft
						? this.PolygonVerts[nextL].Left
						: this.PolygonVerts[nextL].Right;
					continue;
				}

				Vec2 edgeR = this.PolygonVerts[nextR].Pos - this.PolygonVerts[center].Pos;
				double r2 = Dot2d(edgeR, edgeR);
				if (r2 <= p2)
				{
					nextR = this.PolygonVerts[nextR].Right;
					continue;
				}

				Vec2 vecLr = this.PolygonVerts[nextR].Pos - this.PolygonVerts[nextL].Pos;
				double lr2 = Dot2d(vecLr, vecLr);
				if (lr2 <= p2)
				{
					last = center;
					center = nextL;
					nextL = toLeft
						? this.PolygonVerts[nextL].Left
						: this.PolygonVerts[nextL].Right;
					if (nextL == nextR)
					{
						break;
					}

					nextR = this.PolygonVerts[nextR].Right;
					continue;
				}

				int convexity = Ccw(
					this.PolygonVerts[nextL].Pos,
					this.PolygonVerts[center].Pos,
					this.PolygonVerts[nextR].Pos,
					this.Epsilon);
				if (center != last)
				{
					convexity += Ccw(
						this.PolygonVerts[last].Pos,
						this.PolygonVerts[center].Pos,
						this.PolygonVerts[nextL].Pos,
						this.Epsilon) + Ccw(
						this.PolygonVerts[nextR].Pos,
						this.PolygonVerts[center].Pos,
						this.PolygonVerts[last].Pos,
						this.Epsilon);
				}

				if (convexity != 0)
				{
					return convexity > 0;
				}

				if (l2 < r2)
				{
					center = nextL;
					nextL = toLeft
						? this.PolygonVerts[nextL].Left
						: this.PolygonVerts[nextL].Right;
				}
				else
				{
					center = nextR;
					nextR = this.PolygonVerts[nextR].Right;
				}

				last = center;
			}

			return true;
		}

		/// <summary>Whether the ring turns left (or straight) at <paramref name="v"/>.</summary>
		/// <param name="v">The vert to test.</param>
		/// <param name="epsilon">The colinearity tolerance to use.</param>
		/// <returns>True when the vert is convex.</returns>
		private bool VertIsConvex(int v, double epsilon)
		{
			int left = this.PolygonVerts[v].Left;
			int right = this.PolygonVerts[v].Right;
			return Ccw(
				this.PolygonVerts[left].Pos,
				this.PolygonVerts[v].Pos,
				this.PolygonVerts[right].Pos,
				epsilon) >= 0;
		}

		/// <summary>Whether the ring turns right at <paramref name="v"/>.</summary>
		/// <param name="v">The vert to test.</param>
		/// <returns>True when the vert is reflex.</returns>
		private bool VertIsReflex(int v)
		{
			int left = this.PolygonVerts[v].Left;
			return !this.VertInsideEdge(left, this.PolygonVerts[left].Right, true);
		}

		/// <summary>
		/// x-value on this vert's right edge at <c>start.y</c>, or NaN if no crossing.
		/// </summary>
		/// <param name="v">The vert whose right edge is tested.</param>
		/// <param name="start">The point the horizontal ray starts from.</param>
		/// <param name="onTop">1 when start is on the top of its bbox, -1 on the bottom, else 0.</param>
		/// <returns>The crossing x, or NaN.</returns>
		private double VertInterpY2X(int v, Vec2 start, int onTop)
		{
			Vec2 pos = this.PolygonVerts[v].Pos;
			Vec2 rpos = this.PolygonVerts[this.PolygonVerts[v].Right].Pos;
			double eps = this.Epsilon;

			// Rust's f64::NAN is the *positive* quiet NaN; C#'s double.NaN has the sign
			// bit set. Only IsFinite reads these today, but the port's rule is the
			// constant, not double.NaN.
			if (Math.Abs(pos.Y - start.Y) <= eps)
			{
				if (rpos.Y <= start.Y + eps || onTop == 1)
				{
					return DeterministicMath.PositiveQuietNaN;
				}
				else
				{
					return pos.X;
				}
			}
			else if (pos.Y < start.Y - eps)
			{
				if (rpos.Y > start.Y + eps)
				{
					return pos.X + ((start.Y - pos.Y) * (rpos.X - pos.X) / (rpos.Y - pos.Y));
				}
				else if (rpos.Y < start.Y - eps || onTop == -1)
				{
					return DeterministicMath.PositiveQuietNaN;
				}
				else
				{
					return rpos.X;
				}
			}
			else
			{
				return DeterministicMath.PositiveQuietNaN;
			}
		}

		/// <summary>
		/// Signed distance of vert <paramref name="other"/> relative to the edge at
		/// <paramref name="v"/> in direction <paramref name="unit"/>.
		/// </summary>
		/// <param name="v">The vert the edge starts at.</param>
		/// <param name="other">The vert being measured.</param>
		/// <param name="unit">The edge direction.</param>
		/// <returns>The signed distance.</returns>
		private double VertSignedDist(int v, int other, Vec2 unit)
		{
			double eps = this.Epsilon;
			double d = Determinant2x2(unit, this.PolygonVerts[other].Pos - this.PolygonVerts[v].Pos);
			if (Math.Abs(d) < eps)
			{
				double dR = Determinant2x2(
					unit,
					this.PolygonVerts[this.PolygonVerts[other].Right].Pos - this.PolygonVerts[v].Pos);
				if (Math.Abs(dR) > eps)
				{
					return dR;
				}

				double dL = Determinant2x2(
					unit,
					this.PolygonVerts[this.PolygonVerts[other].Left].Pos - this.PolygonVerts[v].Pos);
				if (Math.Abs(dL) > eps)
				{
					return dL;
				}
			}

			return d;
		}

		/// <summary>
		/// Cost of vert <paramref name="other"/> within ear <paramref name="v"/>, where
		/// <paramref name="openSide"/> is the unit vector left-&gt;right.
		/// </summary>
		/// <param name="v">The candidate ear vert.</param>
		/// <param name="other">The vert being tested against the ear.</param>
		/// <param name="openSide">Unit vector along the ear's open side.</param>
		/// <returns>The cost contribution; negative means the vert intrudes.</returns>
		private double VertCost(int v, int other, Vec2 openSide)
		{
			int left = this.PolygonVerts[v].Left;
			int right = this.PolygonVerts[v].Right;
			double cost = MinF64(
				this.VertSignedDist(v, other, this.PolygonVerts[v].RightDir),
				this.VertSignedDist(left, other, this.PolygonVerts[left].RightDir));
			double openCost = Determinant2x2(
				openSide,
				this.PolygonVerts[other].Pos - this.PolygonVerts[right].Pos);
			return MinF64(cost, openCost);
		}

		/// <summary>The Delaunay-flavoured cost that replaces a strongly intruding vert's cost.</summary>
		/// <param name="diff">Offset of the intruding vert from the ear's open-side centre.</param>
		/// <param name="scale">4 / |openSide|^2, or 0 for a degenerate open side.</param>
		/// <param name="epsilon">The tolerance in use.</param>
		/// <returns>The cost.</returns>
		private static double DelaunayCost(Vec2 diff, double scale, double epsilon)
		{
			return -epsilon - (scale * Dot2d(diff, diff));
		}

		/// <summary>
		/// The cost of clipping <paramref name="v"/> as an ear: the worst intrusion by any
		/// other ring vert inside the ear's bounding box.
		/// </summary>
		/// <param name="v">The candidate ear vert.</param>
		/// <param name="collider">The kd-tree of ring verts.</param>
		/// <returns>The ear cost; negative is a valid ear, lower is better.</returns>
		private double VertEarCost(int v, IdxCollider collider)
		{
			int left = this.PolygonVerts[v].Left;
			int right = this.PolygonVerts[v].Right;
			Vec2 openSideVec = this.PolygonVerts[left].Pos - this.PolygonVerts[right].Pos;
			Vec2 center = (this.PolygonVerts[left].Pos + this.PolygonVerts[right].Pos) * 0.5;
			double denom = Dot2d(openSideVec, openSideVec);
			double scale = denom > 0.0 ? 4.0 / denom : 0.0;
			double radius = Math.Sqrt(denom) * 0.5;
			Vec2 openSide = SafeNormalize2d(openSideVec);

			double totalCost = Dot2d(this.PolygonVerts[left].RightDir, this.PolygonVerts[v].RightDir)
				- 1.0
				- this.Epsilon;

			// Folded ears: clip first
			if (Ccw(
				this.PolygonVerts[v].Pos,
				this.PolygonVerts[left].Pos,
				this.PolygonVerts[right].Pos,
				this.Epsilon) == 0)
			{
				return totalCost;
			}

			// Build ear bounding box, expanded to include pos, then by epsilon
			double cx = center.X;
			double cy = center.Y;
			Rect earBox = Rect.FromPoints(
				new Vec2(cx - radius, cy - radius),
				new Vec2(cx + radius, cy + radius));
			earBox.UnionPoint(this.PolygonVerts[v].Pos);
			earBox.Min = earBox.Min - Vec2.Splat(this.Epsilon);
			earBox.Max = earBox.Max + Vec2.Splat(this.Epsilon);

			int lid = this.PolygonVerts[left].MeshIdx;
			int rid = this.PolygonVerts[right].MeshIdx;
			int midId = this.PolygonVerts[v].MeshIdx;

			double tc = totalCost;
			QueryTwoDTree(collider.Points, earBox, point =>
			{
				int test = collider.Itr[point.Idx];
				if (!this.Clipped(test)
					&& this.PolygonVerts[test].MeshIdx != midId
					&& this.PolygonVerts[test].MeshIdx != lid
					&& this.PolygonVerts[test].MeshIdx != rid)
				{
					double cost = this.VertCost(v, test, openSide);
					if (cost < -this.Epsilon)
					{
						cost = DelaunayCost(this.PolygonVerts[test].Pos - center, scale, this.Epsilon);
					}

					if (cost > tc)
					{
						tc = cost;
					}
				}
			});
			return tc;
		}
	}
}
