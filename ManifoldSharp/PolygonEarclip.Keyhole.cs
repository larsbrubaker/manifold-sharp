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
			foreach (int outerStart in this.Outers)
			{
				int before = visitor.Connector;
				bool complete = this.ForEachLoopVert(outerStart, ref visitor);
				if (!complete)
				{
					visitor.Connector = before;
				}
			}

			int connector = visitor.Connector;
			if (connector == Invalid)
			{
				this.Simples.Add(start);
				return;
			}

			connector = this.FindCloserBridge(start, connector);
			this.JoinPolygons(start, connector);
		}

		/// <summary>Refine keyhole connector: find any reflex vert closer to start.</summary>
		/// <param name="start">The hole's start vert.</param>
		/// <param name="edge">The connector edge found by <see cref="CutKeyhole"/>.</param>
		/// <returns>The vert to bridge to.</returns>
		private int FindCloserBridge(int start, int edge)
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
				return connector;
			}

			double above = this.PolygonVerts[connector].Pos.Y > startPos.Y ? 1.0 : -1.0;

			// Degenerate rings are skipped whole, as in `CutKeyhole`.
			CloserBridgeVisitor visitor = new CloserBridgeVisitor(this, edge, startPos, above, connector);
			foreach (int outerStart in this.Outers)
			{
				int before = visitor.Connector;
				bool complete = this.ForEachLoopVert(outerStart, ref visitor);
				if (!complete)
				{
					visitor.Connector = before;
				}
			}

			return visitor.Connector;
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

			internal CheckEdgeVisitor(EarClip clip, int start, Vec2 startPos, int onTop)
			{
				this.clip = clip;
				this.start = start;
				this.startPos = startPos;
				this.onTop = onTop;
				this.Connector = Invalid;
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

			internal CloserBridgeVisitor(EarClip clip, int edge, Vec2 startPos, double above, int connector)
			{
				this.clip = clip;
				this.edge = edge;
				this.startPos = startPos;
				this.above = above;
				this.Connector = connector;
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
