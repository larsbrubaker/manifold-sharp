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

// EarClip keyholing — the hole-to-outer bridge searches of the ear clipper
//
// Ports C++ `EarClip::CutKeyhole`, `FindCloserBridge` and `JoinPolygons`
// (src/polygon.cpp). Before ear clipping, `EarClip::triangulate` (in
// polygon_earclip.rs, which defines the struct, the vert predicates and the
// ear-clipping loop) joins each hole, rightmost first, to an outer ring by a
// zero-width bridge so every polygon left is simple. This file adds a second
// `impl EarClip` block holding just those three steps.
//
// ── C# port of polygon_earclip_keyhole.rs ───────────────────────────────────
// A third file of the `internal sealed partial class EarClip`, mirroring the
// Rust's child module so the keyhole code diffs against one Rust file. See
// PolygonEarclip.cs for the type layout.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;
using static ManifoldSharp.Polygon;

namespace ManifoldSharp
{
	/// <content>The keyholing steps of <c>EarClip</c>.</content>
	internal sealed partial class EarClip
	{
		/// <summary>Attach a hole to an outer polygon via a keyhole.</summary>
		/// <param name="start">The hole's start vert (its rightmost reflex vert).</param>
		private void CutKeyhole(int start)
		{
			Rect bbox = this.hole2bbox[start];
			Vec2 startPos = this.PolygonVerts[start].Pos;
			int onTop;
			if (startPos.Y >= bbox.Max.Y - this.Epsilon)
			{
				onTop = 1;
			}
			else if (startPos.Y <= bbox.Min.Y + this.Epsilon)
			{
				onTop = -1;
			}
			else
			{
				onTop = 0;
			}

			// Speed only, not in C++: a ring wholly above or below start.y -+ eps
			// has no edge with a finite `VertInterpY2X`, which needs one end at
			// or below start.y + eps and the other at or above start.y - eps, so
			// it cannot take the connector and is not walked. The margin only
			// widens the band: twice eps, plus 1e-9 relative, far above the
			// rounding of the comparisons. A NaN or overflowed slack culls nothing.
			double slack = (2.0 * Math.Abs(this.Epsilon)) + (1e-9 * (1.0 + Math.Abs(startPos.Y)));

			// Port of the C++ CheckEdge lambda (CheckEdgeVisitor below): take `edge`
			// as the new connector when the horizontal ray from `start` crosses it
			// (finite x), `start` lies inside THAT edge's wedge, and it beats the
			// current connector — either the crossing point is CCW of the connector
			// edge, or (for any non-CCW result) the vertical-ordering InsideEdge
			// tie-break holds.
			// A degenerate ring is skipped whole, as `LoopVerts` returning null
			// skipped it, so the connector is restored if the walk stops part-way.
			// C++ `Loop` keeps what it saw before the degenerate vert; see
			// manifold-rust docs/CPP_DIVERGENCES.md entry 12, which this port shares
			// (it matches the Rust, so it is no RUST_DIVERGENCES.md entry).
			CheckEdgeVisitor visitor = new CheckEdgeVisitor(this, start, startPos, onTop);
			for (int k = 0; k < this.Outers.Count; k++)
			{
				Rect rb = this.outerBbox[k];
				if (rb.Min.Y > startPos.Y + slack || rb.Max.Y < startPos.Y - slack)
				{
					continue;
				}

				int beforeConnector = visitor.Connector;
				int beforeRing = visitor.Ring;
				visitor.K = k;
				bool complete = this.ForEachLoopVert(this.Outers[k], ref visitor);
				if (!complete)
				{
					visitor.Connector = beforeConnector;
					visitor.Ring = beforeRing;
				}
			}

			if (visitor.Connector == Invalid)
			{
				this.Simples.Add(start);
				return;
			}

			(int connector, int ring) = this.FindCloserBridge(start, visitor.Connector, visitor.Ring);
			this.JoinPolygons(start, connector);

			// The hole's verts are now part of that ring. The joined verts are
			// copies of verts already in one box or the other, and clipping only
			// removes verts, so the box still holds every vert the ring walks.
			Rect grown = this.outerBbox[ring];
			grown.UnionPoint(bbox.Min);
			grown.UnionPoint(bbox.Max);
			this.outerBbox[ring] = grown;
		}

		/// <summary>
		/// Refine keyhole connector: find any reflex vert closer to start. Also returns
		/// the <see cref="Outers"/> index of the connector's ring, which starts as
		/// <paramref name="edgeRing"/>, the ring <paramref name="edge"/> came from.
		/// </summary>
		/// <param name="start">The hole's start vert.</param>
		/// <param name="edge">The connector edge found by <see cref="CutKeyhole"/>.</param>
		/// <param name="edgeRing">The <see cref="Outers"/> index of <paramref name="edge"/>'s ring.</param>
		/// <returns>The vert to bridge to, and its ring's index.</returns>
		private (int Connector, int Ring) FindCloserBridge(int start, int edge, int edgeRing)
		{
			Vec2 startPos = this.PolygonVerts[start].Pos;
			int edgeRight = this.PolygonVerts[edge].Right;
			int connector;
			if (this.PolygonVerts[edge].Pos.X < startPos.X)
			{
				connector = edgeRight;
			}
			else if (this.PolygonVerts[edgeRight].Pos.X < startPos.X)
			{
				connector = edge;
			}
			else if (this.PolygonVerts[edgeRight].Pos.Y - startPos.Y
				> startPos.Y - this.PolygonVerts[edge].Pos.Y)
			{
				connector = edge;
			}
			else
			{
				connector = edgeRight;
			}

			if (Math.Abs(this.PolygonVerts[connector].Pos.Y - startPos.Y) <= this.Epsilon)
			{
				return (connector, edgeRing);
			}

			double above = this.PolygonVerts[connector].Pos.Y > startPos.Y ? 1.0 : -1.0;

			// Degenerate rings are skipped whole, as in `CutKeyhole`.
			//
			// Speed only, not in C++: a ring whose bounding box shows that no
			// vert in it can pass the test below is not walked, so the bridge,
			// and the triangles, are unchanged. A vert must lie right of
			// start.x - eps, on the `above` side of start.y -+ eps, and not
			// clearly outside start -> connector. The coordinate tests are bounded
			// by the box's edges, widened by twice eps plus 1e-9 relative, far
			// above the rounding of the comparisons; they need no guard, as a
			// NaN or overflowed slack culls nothing.
			double eps = Math.Abs(this.Epsilon);
			double slack = (2.0 * eps) + (1e-9 * (1.0 + Math.Abs(startPos.X) + Math.Abs(startPos.Y)));
			CloserBridgeVisitor visitor = new CloserBridgeVisitor(this, edge, startPos, above, connector, edgeRing);
			for (int k = 0; k < this.Outers.Count; k++)
			{
				Rect rb = this.outerBbox[k];
				if (rb.Max.X < startPos.X - slack
					|| (above > 0.0 && rb.Max.Y < startPos.Y - slack)
					|| (above < 0.0 && rb.Min.Y > startPos.Y + slack))
				{
					continue;
				}

				// The `Ccw` test rejects a vert only when Ccw(start, vert,
				// connector) is nonzero with sign -above, that is when its area
				// a = v1 x v2 (v1 = vert - start, v2 = connector - start) has
				// above * a < 0 and 4a^2 > max(|v1|^2, |v2|^2) eps^2. above * a
				// is linear in the vert, so its maximum over the box is at a
				// corner (`best`); `dist` bounds |v1| and |v2| over the box. In
				// exact arithmetic best < -(dist eps) rejects every vert in the
				// box. The 1e-9 dist |v2| margin covers rounding: each difference
				// and product is rounded to ~1e-16 relative, so the computed a of
				// a culled vert keeps its sign and |a| > dist eps.
				//
				// That argument needs the squares in `Ccw` to be finite normal
				// numbers, so the cull applies only inside a magnitude window.
				// There a culled vert has |a| > 1e-9 dist |v2| >= 1e-9 |v2|^2 >=
				// 1e-129, so 4a^2 >= 4e-258 does not underflow, and |a| <= dist
				// |v2| <= 1e120 and max(|v1|^2, |v2|^2) eps^2 <= 1e240 do not
				// overflow. Outside it, a^2 can underflow to 0 (or both sides
				// overflow to inf) and `Ccw` returns 0 for a vert the bound
				// calls outside, which the `inside == 0` tie-break can then take;
				// see `KeyholeCullKeepsABridgeWhoseCcwUnderflows`.
				//
				// Rust `f64::max` returns the other operand when one is NaN;
				// `MaxF64` is that (Math.Max would propagate the NaN). The
				// signed-zero tie it pins (RUST_DIVERGENCES entry 2) cannot matter
				// here: `best` and `dist` are only compared, and -0.0 == +0.0.
				Vec2 v2 = this.PolygonVerts[visitor.Connector].Pos - startPos;
				double v2Len = Math.Sqrt((v2.X * v2.X) + (v2.Y * v2.Y));
				double best = double.NegativeInfinity;
				double dist = v2Len;
				for (int corner = 0; corner < 4; corner++)
				{
					Vec2 c = new Vec2(
						(corner & 1) == 0 ? rb.Min.X : rb.Max.X,
						(corner & 2) == 0 ? rb.Min.Y : rb.Max.Y);
					Vec2 v1 = c - startPos;
					best = MaxF64(best, above * ((v1.X * v2.Y) - (v1.Y * v2.X)));
					dist = MaxF64(dist, Math.Sqrt((v1.X * v1.X) + (v1.Y * v1.Y)));
				}

				// False on NaN.
				bool inWindow = v2Len >= 1e-60 && dist <= 1e60 && eps <= 1e60;
				if (inWindow && best < -((dist * eps) + (1e-9 * dist * v2Len)))
				{
					continue;
				}

				int beforeConnector = visitor.Connector;
				int beforeRing = visitor.Ring;
				visitor.K = k;
				bool complete = this.ForEachLoopVert(this.Outers[k], ref visitor);
				if (!complete)
				{
					visitor.Connector = beforeConnector;
					visitor.Ring = beforeRing;
				}
			}

			return (visitor.Connector, visitor.Ring);
		}

		/// <summary>
		/// The body of <see cref="CutKeyhole"/>'s ring walk, the Rust closure handed to
		/// <c>for_each_loop_vert</c>: a struct so the walk dispatches statically and
		/// allocates nothing per hole.
		/// </summary>
		private struct CheckEdgeVisitor : ILoopVertVisitor
		{
			private readonly EarClip clip;
			private readonly int start;
			private readonly Vec2 startPos;
			private readonly int onTop;

			/// <summary>The best connector edge so far, or <see cref="Invalid"/>.</summary>
			internal int Connector;

			/// <summary>The <see cref="Outers"/> index of <see cref="Connector"/>'s ring.</summary>
			internal int Ring;

			/// <summary>The <see cref="Outers"/> index of the ring being walked.</summary>
			internal int K;

			internal CheckEdgeVisitor(EarClip clip, int start, Vec2 startPos, int onTop)
			{
				this.clip = clip;
				this.start = start;
				this.startPos = startPos;
				this.onTop = onTop;
				this.Connector = Invalid;
				this.Ring = Invalid;
				this.K = Invalid;
			}

			public void Visit(int edge)
			{
				EarClip c = this.clip;
				int connector = this.Connector;
				double x = c.VertInterpY2X(edge, this.startPos, this.onTop);
				if (double.IsFinite(x)
					&& c.VertInsideEdge(this.start, edge, true)
					&& (connector == Invalid
						|| Ccw(
							new Vec2(x, this.startPos.Y),
							c.PolygonVerts[connector].Pos,
							c.PolygonVerts[c.PolygonVerts[connector].Right].Pos,
							c.Epsilon) == 1
						|| (c.PolygonVerts[connector].Pos.Y < c.PolygonVerts[edge].Pos.Y
							? c.VertInsideEdge(edge, connector, false)
							: !c.VertInsideEdge(connector, edge, false))))
				{
					this.Connector = edge;
					this.Ring = this.K;
				}
			}
		}

		/// <summary>
		/// The body of <see cref="FindCloserBridge"/>'s ring walk, the Rust closure
		/// handed to <c>for_each_loop_vert</c>.
		/// </summary>
		private struct CloserBridgeVisitor : ILoopVertVisitor
		{
			private readonly EarClip clip;
			private readonly int edge;
			private readonly Vec2 startPos;
			private readonly double above;

			/// <summary>The closest reflex vert so far.</summary>
			internal int Connector;

			/// <summary>The <see cref="Outers"/> index of <see cref="Connector"/>'s ring.</summary>
			internal int Ring;

			/// <summary>The <see cref="Outers"/> index of the ring being walked.</summary>
			internal int K;

			internal CloserBridgeVisitor(EarClip clip, int edge, Vec2 startPos, double above, int connector, int ring)
			{
				this.clip = clip;
				this.edge = edge;
				this.startPos = startPos;
				this.above = above;
				this.Connector = connector;
				this.Ring = ring;
				this.K = Invalid;
			}

			public void Visit(int vert)
			{
				EarClip c = this.clip;
				double above = this.above;
				Vec2 startPos = this.startPos;
				int connector = this.Connector;
				double inside = above * (double)Ccw(
					startPos,
					c.PolygonVerts[vert].Pos,
					c.PolygonVerts[connector].Pos,
					c.Epsilon);
				Vec2 vp = c.PolygonVerts[vert].Pos;
				Vec2 cp = c.PolygonVerts[connector].Pos;
				if (vp.X > startPos.X - c.Epsilon
					&& vp.Y * above > startPos.Y * above - c.Epsilon
					&& (inside > 0.0
						|| (inside == 0.0 && vp.X < cp.X && vp.Y * above < cp.Y * above))
					&& c.VertInsideEdge(vert, this.edge, true)
					&& c.VertIsReflex(vert))
				{
					this.Connector = vert;
					this.Ring = this.K;
				}
			}
		}

		/// <summary>Create a keyhole between hole <paramref name="start"/> and outer polygon <paramref name="connector"/>.</summary>
		/// <param name="start">The hole's start vert.</param>
		/// <param name="connector">The outer vert to bridge to.</param>
		private void JoinPolygons(int start, int connector)
		{
			int newStart = this.PushVert(this.PolygonVerts[start]);
			int newConnector = this.PushVert(this.PolygonVerts[connector]);

			int startRight = this.PolygonVerts[start].Right;
			this.PolygonVerts[startRight].Left = newStart;
			int connectorLeft = this.PolygonVerts[connector].Left;
			this.PolygonVerts[connectorLeft].Right = newConnector;

			this.Link(start, connector);
			this.Link(newConnector, newStart);

			this.ClipIfDegenerate(start);
			this.ClipIfDegenerate(newStart);
			this.ClipIfDegenerate(connector);
			this.ClipIfDegenerate(newConnector);
		}
	}
}
