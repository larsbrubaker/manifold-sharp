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
// ── C# port of polygon_earclip.rs (file 2 of 2) ──────────────────────────────
// The "Core algorithm methods" half of the Rust file: clipping, contour
// classification and the ear-queue loop (keyholing is PolygonEarclip.Keyhole.cs,
// as the Rust's is polygon_earclip_keyhole.rs). See PolygonEarclip.cs for the
// split rationale and the type layout.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;
using static ManifoldSharp.Polygon;

namespace ManifoldSharp
{
	/// <content>The core algorithm methods of <c>EarClip</c>.</content>
	internal sealed partial class EarClip
	{
		// -----------------------------------------------------------------------
		// Core algorithm methods
		// -----------------------------------------------------------------------

		/// <summary>Remove ear vert from polygon and emit a triangle.</summary>
		/// <param name="ear">The vert to clip.</param>
		private void ClipEar(int ear)
		{
			int left = this.PolygonVerts[ear].Left;
			int right = this.PolygonVerts[ear].Right;
			this.Link(left, right);
			if (this.PolygonVerts[left].MeshIdx != this.PolygonVerts[ear].MeshIdx
				&& this.PolygonVerts[ear].MeshIdx != this.PolygonVerts[right].MeshIdx
				&& this.PolygonVerts[right].MeshIdx != this.PolygonVerts[left].MeshIdx)
			{
				this.Triangles.Add(new IVec3(
					this.PolygonVerts[left].MeshIdx,
					this.PolygonVerts[ear].MeshIdx,
					this.PolygonVerts[right].MeshIdx));
			}
		}

		/// <summary>Clip degenerate ears (zero-area or very short edges).</summary>
		/// <param name="ear">The vert to test and, if degenerate, clip.</param>
		private void ClipIfDegenerate(int ear)
		{
			if (this.Clipped(ear))
			{
				return;
			}

			if (this.PolygonVerts[ear].Left == this.PolygonVerts[ear].Right)
			{
				return;
			}

			int right = this.PolygonVerts[ear].Right;
			int left = this.PolygonVerts[ear].Left;
			bool isShort = this.VertIsShort(ear);
			Vec2 lp = this.PolygonVerts[left].Pos;
			Vec2 ep = this.PolygonVerts[ear].Pos;
			Vec2 rp = this.PolygonVerts[right].Pos;
			Vec2 diffL = lp - ep;
			Vec2 diffR = rp - ep;
			bool isColinearFold = Ccw(lp, ep, rp, this.Epsilon) == 0 && Dot2d(diffL, diffR) > 0.0;
			if (isShort || isColinearFold)
			{
				this.ClipEar(ear);

				// Re-read the neighbours: ClipEar relinked them, and the recursion has to
				// follow the *new* ring, not the `left`/`right` captured above.
				int l = this.PolygonVerts[ear].Left;
				int r = this.PolygonVerts[ear].Right;
				this.ClipIfDegenerate(l);
				this.ClipIfDegenerate(r);
			}
		}

		/// <summary>Build the circular polygon list from input polygons. Returns start indices.</summary>
		/// <param name="polys">The indexed polygons.</param>
		/// <returns>The first vert index of each non-empty contour.</returns>
		private List<int> Initialize(PolygonsIdx polys)
		{
			List<int> starts = new List<int>();
			foreach (SimplePolygonIdx poly in polys)
			{
				if (poly.Count == 0)
				{
					continue;
				}

				PolyVert vert = poly[0];
				int firstIdx = this.PushVert(new Vert
				{
					MeshIdx = vert.Idx,
					Cost = 0.0,
					EarVersion = 0,
					Pos = vert.Pos,
					RightDir = new Vec2(0.0, 0.0),
					Left = Invalid,
					Right = Invalid,
				});
				this.bbox.UnionPoint(vert.Pos);
				int last = firstIdx;
				starts.Add(firstIdx);

				for (int i = 1; i < poly.Count; i++)
				{
					PolyVert next = poly[i];
					int nextIdx = this.PushVert(new Vert
					{
						MeshIdx = next.Idx,
						Cost = 0.0,
						EarVersion = 0,
						Pos = next.Pos,
						RightDir = new Vec2(0.0, 0.0),
						Left = Invalid,
						Right = Invalid,
					});
					this.bbox.UnionPoint(next.Pos);
					this.Link(last, nextIdx);
					last = nextIdx;
				}

				this.Link(last, firstIdx);
			}

			if (this.Epsilon < 0.0)
			{
				this.Epsilon = this.bbox.Scale() * Types.KPrecision;
			}

			return starts;
		}

		/// <summary>
		/// Classify a polygon as hole or outer. For holes, find the rightmost reflex vert.
		/// </summary>
		/// <param name="first">A vert of the contour to classify.</param>
		private void FindStart(int first)
		{
			Vec2 origin = this.PolygonVerts[first].Pos;
			int start = first;
			double maxX = double.NegativeInfinity;
			Rect bbox = new Rect();

			// Kahan summation
			double area = 0.0;
			double areaComp = 0.0;

			List<int>? verts = this.LoopVerts(first);
			if (verts == null)
			{
				return;
			}

			foreach (int v in verts)
			{
				bbox.UnionPoint(this.PolygonVerts[v].Pos);
				double a1 = Determinant2x2(
					this.PolygonVerts[v].Pos - origin,
					this.PolygonVerts[this.PolygonVerts[v].Right].Pos - origin);
				double t1 = area + a1;
				areaComp += (area - t1) + a1;
				area = t1;

				if (this.PolygonVerts[v].Pos.X > maxX && this.VertIsReflex(v))
				{
					maxX = this.PolygonVerts[v].Pos.X;
					start = v;
				}
			}

			area += areaComp;
			Vec2 size = bbox.Size();
			double minArea = this.Epsilon * MaxF64(size.X, size.Y);

			if (double.IsFinite(maxX) && area < -minArea)
			{
				// Hole (negative area)
				// Collected here and sorted by descending max X after classification.
				this.Holes.Add(start);
				this.hole2bbox[start] = bbox;
			}
			else
			{
				this.Simples.Add(start);
				if (area > minArea)
				{
					this.Outers.Add(start);
					this.outerBbox.Add(bbox);
				}
			}
		}

		/// <summary>Update ear queue entry for vert <paramref name="v"/>.</summary>
		/// <param name="v">The vert to (re)queue.</param>
		/// <param name="collider">The kd-tree of ring verts.</param>
		private void ProcessEar(int v, IdxCollider collider)
		{
			// Lazy-delete existing queue entry. Rust `wrapping_add(1)`: the counter is
			// allowed to wrap, and C#'s default unchecked u32 arithmetic is that wrap.
			this.PolygonVerts[v].EarVersion = unchecked(this.PolygonVerts[v].EarVersion + 1);

			if (this.VertIsShort(v))
			{
				this.PolygonVerts[v].Cost = KBest;
				uint version = this.PolygonVerts[v].EarVersion;
				ulong seq = this.earSeq;
				this.earSeq++;
				EarEntry entry = new EarEntry(KBest, v, version, seq);
				this.earsQueue.Enqueue(entry, entry);
			}
			else if (this.VertIsConvex(v, 2.0 * this.Epsilon))
			{
				double cost = this.VertEarCost(v, collider);
				this.PolygonVerts[v].Cost = cost;
				uint version = this.PolygonVerts[v].EarVersion;
				ulong seq = this.earSeq;
				this.earSeq++;
				EarEntry entry = new EarEntry(cost, v, version, seq);
				this.earsQueue.Enqueue(entry, entry);
			}
			else
			{
				this.PolygonVerts[v].Cost = 1.0; // reflex, not an ear
			}
		}

		/// <summary>Build a 2D KD-tree collider of all polygon verts for ear cost queries.</summary>
		/// <param name="start">A vert of the ring to collide against.</param>
		/// <returns>The collider; empty when the ring has collapsed.</returns>
		private IdxCollider VertCollider(int start)
		{
			List<int>? verts = this.LoopVerts(start);
			if (verts == null)
			{
				return new IdxCollider
				{
					Points = new List<PolyVert>(),
					Itr = new List<int>(),
				};
			}

			List<int> itr = new List<int>(verts.Count);
			List<PolyVert> points = new List<PolyVert>(verts.Count);
			for (int k = 0; k < verts.Count; k++)
			{
				int v = verts[k];
				points.Add(new PolyVert(this.PolygonVerts[v].Pos, k));
				itr.Add(v);
			}

			BuildTwoDTree(points);
			return new IdxCollider { Points = points, Itr = itr };
		}

		/// <summary>Main ear-clipping loop for a simple polygon.</summary>
		/// <param name="start">A vert of the ring to triangulate.</param>
		private void TriangulatePoly(int start)
		{
			IdxCollider collider = this.VertCollider(start);
			if (collider.Itr.Count == 0)
			{
				return;
			}

			this.earsQueue.Clear();
			int numTri = -2;

			// Queue all verts and count
			List<int>? verts = this.LoopVerts(start);
			if (verts == null)
			{
				return;
			}

			foreach (int v in verts)
			{
				this.ProcessEar(v, collider);
				numTri++;
			}

			int lastV = verts[verts.Count - 1];

			while (numTri > 0)
			{
				// Pop the cheapest valid ear
				int ear;
				while (true)
				{
					if (!this.earsQueue.TryDequeue(out EarEntry entry, out _))
					{
						// Fallback: use last_v
						ear = lastV;
						break;
					}

					if (entry.Version == this.PolygonVerts[entry.Idx].EarVersion)
					{
						ear = entry.Idx;
						break;
					}

					// Stale entry, try next
				}

				this.ClipEar(ear);
				numTri--;

				int earLeft = this.PolygonVerts[ear].Left;
				int earRight = this.PolygonVerts[ear].Right;
				this.ProcessEar(earLeft, collider);
				this.ProcessEar(earRight, collider);
				lastV = earRight;
			}
		}
	}
}
