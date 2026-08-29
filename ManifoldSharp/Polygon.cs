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

// Phase 3: Polygon Triangulation — ported from src/polygon.cpp, src/tree2d.h/cpp, src/utils.h
//
// Key algorithm: ear-clipping with 2D KD-tree acceleration, convex fast-path,
// hole key-holing. Must produce identical triangulations to the C++ version.
//
// ── C# port of polygon.rs ────────────────────────────────────────────────────
// The Rust module's free items land on the static class named for the module
// (`Polygon`), per the naming rule in CLAUDE.md. The ear clipper the module
// pulls in with `#[path = "polygon_earclip.rs"] mod polygon_earclip` lives in
// PolygonEarclip*.cs; its `pub(super)` reach becomes `internal`, and the helpers
// it imports back out of this file (`ccw`, `determinant2x2`, `dot2d`,
// `safe_normalize_2d`, `build_two_d_tree`, `query_two_d_tree`, `INVALID`,
// `K_BEST`) are `internal` here for the same reason.
//
// `IVec3Out` is a bare Rust type alias for `IVec3`; C# uses `IVec3` directly.
//
// `INVALID` is Rust `usize::MAX` and is only ever compared for equality — it is
// never used to index — so the C# sentinel is the port's conventional -1.
//
// ── This file's kd-tree is deliberately not Tree2d.cs ────────────────────────
// polygon.rs carries its own copy of `build_two_d_tree` / `query_two_d_tree`,
// and polygon_earclip.rs imports *those* (`use super::{build_two_d_tree, ...}`),
// not the ones in tree2d.rs. The two copies are NOT interchangeable: Tree2d
// transcribes the C++ iterative descent (walk into the left child, stack the
// right sibling), while the query below is a plain pop-loop that pushes left
// then right onto a LIFO stack and therefore visits the RIGHT child first. Same
// point set, different callback order. Routing the ear clipper through Tree2d
// would be a silent behaviour change, so the duplicate stays a duplicate, exactly
// as it is in the Rust — Tree2d.cs's header records the same from its side.
// (Measured, so the risk is stated honestly: flipping this file's push order to
// Tree2d's changed nothing across a 45-polygon differential corpus, because the
// only caller folds the visited points into a maximum. The divergence is latent,
// not observed — which is why it is not worth taking on, and why nothing here
// may start depending on the visit order.)
//
// Every Rust `.min()` / `.max()` on an f64 is `LinalgFunctions.MinF64` /
// `MaxF64` here (NaN loses), never `Math.Min` / `Math.Max`. Every Rust
// `sort_by` is a *stable* sort, so it becomes LINQ `OrderBy`.

using System.Diagnostics;

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// Triangulation of 2D polygons: the free functions of <c>polygon.rs</c>.
	/// </summary>
	public static class Polygon
	{
		/// <summary>
		/// Rust <c>INVALID</c> (<c>usize::MAX</c>) — the "no such vert" sentinel. Only ever
		/// compared for equality, never used as an index, so -1 carries the same meaning.
		/// </summary>
		internal const int Invalid = -1;

		/// <summary>
		/// Rust <c>K_BEST</c> — the cost assigned to an ear that must be clipped first.
		/// </summary>
		internal const double KBest = double.NegativeInfinity;

		/// <summary>
		/// Rust's <c>a.partial_cmp(&amp;b).unwrap_or(Ordering::Equal)</c> on doubles: a NaN
		/// operand compares Equal to everything rather than sorting to one end, which is
		/// what <see cref="Comparer{T}.Default"/> would do.
		/// </summary>
		internal static readonly IComparer<double> PartialCmp = new PartialCmpComparer(false);

		/// <summary>
		/// <see cref="PartialCmp"/> with the operands swapped — Rust
		/// <c>b.partial_cmp(&amp;a).unwrap_or(Ordering::Equal)</c>, i.e. descending.
		/// </summary>
		internal static readonly IComparer<double> PartialCmpDescending = new PartialCmpComparer(true);

		// ---------------------------------------------------------------------------
		// CCW (from utils.h)
		// ---------------------------------------------------------------------------

		/// <summary>
		/// Determines if p0, p1, p2 are wound CCW, CW, or colinear within tolerance.
		/// Returns 1 (CCW), -1 (CW), or 0 (colinear within tol).
		/// </summary>
		/// <param name="p0">First point.</param>
		/// <param name="p1">Second point.</param>
		/// <param name="p2">Third point.</param>
		/// <param name="tol">Colinearity tolerance.</param>
		/// <returns>1, -1 or 0.</returns>
		public static int Ccw(Vec2 p0, Vec2 p1, Vec2 p2, double tol)
		{
			// The expression order is literal, down to the `* 4.0` on the left of the
			// comparison: this is the C++ utils.h predicate that every downstream
			// classification agrees with, and reassociating it changes which
			// near-degenerate triangles report colinear.
			Vec2 v1 = p1 - p0;
			Vec2 v2 = p2 - p0;
			double area = v1.X * v2.Y - v1.Y * v2.X;
			double base2 = MaxF64(v1.X * v1.X + v1.Y * v1.Y, v2.X * v2.X + v2.Y * v2.Y);
			if (area * area * 4.0 <= base2 * tol * tol)
			{
				return 0;
			}
			else if (area > 0.0)
			{
				return 1;
			}
			else
			{
				return -1;
			}
		}

		/// <summary>The 2x2 determinant of the two column vectors.</summary>
		/// <param name="a">First column.</param>
		/// <param name="b">Second column.</param>
		/// <returns><c>a.x * b.y - a.y * b.x</c>.</returns>
		internal static double Determinant2x2(Vec2 a, Vec2 b)
		{
			return (a.X * b.Y) - (a.Y * b.X);
		}

		/// <summary>Unit vector along <paramref name="v"/>, or zero if that is not defined.</summary>
		/// <param name="v">The vector to normalize.</param>
		/// <returns>The normalized vector, or (0, 0).</returns>
		internal static Vec2 SafeNormalize2d(Vec2 v)
		{
			double len = Math.Sqrt((v.X * v.X) + (v.Y * v.Y));
			if (len == 0.0 || !double.IsFinite(len))
			{
				return new Vec2(0.0, 0.0);
			}
			else
			{
				return new Vec2(v.X / len, v.Y / len);
			}
		}

		/// <summary>The 2D dot product.</summary>
		/// <param name="a">First vector.</param>
		/// <param name="b">Second vector.</param>
		/// <returns><c>a.x * b.x + a.y * b.y</c>.</returns>
		internal static double Dot2d(Vec2 a, Vec2 b)
		{
			return (a.X * b.X) + (a.Y * b.Y);
		}

		// ---------------------------------------------------------------------------
		// 2D KD-tree (from tree2d.h/cpp)
		// ---------------------------------------------------------------------------

		/// <summary>
		/// Recursive in-place kd-tree construction on a PolyVert slice.
		/// Alternates between sorting by x and y.
		/// </summary>
		/// <param name="points">The point arena.</param>
		/// <param name="start">First index of the slice.</param>
		/// <param name="len">Length of the slice.</param>
		/// <param name="sortX">Sort this level by x rather than y.</param>
		private static void BuildTwoDTreeImpl(List<PolyVert> points, int start, int len, bool sortX)
		{
			// Rust `sort_by` is stable, so this is OrderBy over the slice, copied back.
			SortRange(points, start, len, sortX);
			if (len < 2)
			{
				return;
			}

			int mid = len / 2;
			BuildTwoDTreeImpl(points, start, mid, !sortX);
			BuildTwoDTreeImpl(points, start + mid + 1, len - mid - 1, !sortX);
		}

		/// <summary>Stable sort of <c>points[start..start + len]</c> by x or by y.</summary>
		/// <param name="points">The point arena.</param>
		/// <param name="start">First index of the slice.</param>
		/// <param name="len">Length of the slice.</param>
		/// <param name="sortX">Sort by x rather than y.</param>
		private static void SortRange(List<PolyVert> points, int start, int len, bool sortX)
		{
			if (len < 2)
			{
				return;
			}

			PolyVert[] slice = new PolyVert[len];
			points.CopyTo(start, slice, 0, len);
			IOrderedEnumerable<PolyVert> ordered = sortX
				? slice.OrderBy(p => p.Pos.X, PartialCmp)
				: slice.OrderBy(p => p.Pos.Y, PartialCmp);
			int i = start;
			foreach (PolyVert p in ordered)
			{
				points[i] = p;
				i++;
			}
		}

		/// <summary>
		/// Reorders <paramref name="points"/> in place into a 2D kd-tree, ready for
		/// <see cref="QueryTwoDTree"/>. Inputs of 8 points or fewer are left alone; the
		/// query scans those linearly.
		/// </summary>
		/// <param name="points">The points to arrange.</param>
		internal static void BuildTwoDTree(List<PolyVert> points)
		{
			if (points.Count <= 8)
			{
				return;
			}

			BuildTwoDTreeImpl(points, 0, points.Count, true);
		}

		/// <summary>Query the 2D kd-tree, calling <paramref name="f"/> for every point inside rect <paramref name="r"/>.</summary>
		/// <param name="points">A point list already arranged by <see cref="BuildTwoDTree"/>.</param>
		/// <param name="r">The query rect.</param>
		/// <param name="f">Called once per contained point.</param>
		internal static void QueryTwoDTree(List<PolyVert> points, Rect r, Action<PolyVert> f)
		{
			if (points.Count <= 8)
			{
				foreach (PolyVert p in points)
				{
					if (r.ContainsPoint(p.Pos))
					{
						f(p);
					}
				}

				return;
			}

			// Stack-based traversal. (current_rect, start, len, level)
			Stack<(Rect Rect, int Start, int Len, int Level)> stack = new Stack<(Rect, int, int, int)>(64);

			// Initial rect: infinite
			Rect infRect = new Rect(
				new Vec2(double.NegativeInfinity, double.NegativeInfinity),
				new Vec2(double.PositiveInfinity, double.PositiveInfinity));
			stack.Push((infRect, 0, points.Count, 0));

			while (stack.Count > 0)
			{
				(Rect current, int start, int len, int level) = stack.Pop();
				if (len <= 8)
				{
					for (int i = start; i < start + len; i++)
					{
						PolyVert p = points[i];
						if (r.ContainsPoint(p.Pos))
						{
							f(p);
						}
					}

					continue;
				}

				int mid = len / 2;
				PolyVert middle = points[start + mid];

				Rect left = current;
				Rect right = current;
				if (level % 2 == 0)
				{
					left.Max.X = middle.Pos.X;
					right.Min.X = middle.Pos.X;
				}
				else
				{
					left.Max.Y = middle.Pos.Y;
					right.Min.Y = middle.Pos.Y;
				}

				if (r.ContainsPoint(middle.Pos))
				{
					f(middle);
				}

				if (left.DoesOverlap(r))
				{
					stack.Push((left, start, mid, level + 1));
				}

				if (right.DoesOverlap(r))
				{
					stack.Push((right, start + mid + 1, len - mid - 1, level + 1));
				}
			}
		}

		// ---------------------------------------------------------------------------
		// IsConvex fast-path check
		// ---------------------------------------------------------------------------

		/// <summary>Whether every contour turns consistently left, so the fast path applies.</summary>
		/// <param name="polys">The indexed polygons.</param>
		/// <param name="epsilon">The tolerance below which a turn counts as straight.</param>
		/// <returns>True when all contours are convex.</returns>
		internal static bool IsConvex(PolygonsIdx polys, double epsilon)
		{
			foreach (SimplePolygonIdx poly in polys)
			{
				if (poly.Count == 0)
				{
					continue;
				}

				Vec2 firstEdge = poly[0].Pos - poly[poly.Count - 1].Pos;
				Vec2 lastEdge = SafeNormalize2d(firstEdge);
				for (int v = 0; v < poly.Count; v++)
				{
					Vec2 edge = v + 1 < poly.Count
						? poly[v + 1].Pos - poly[v].Pos
						: firstEdge;
					double det = Determinant2x2(lastEdge, edge);
					if (det <= 0.0 || (Math.Abs(det) < epsilon && Dot2d(lastEdge, edge) < 0.0))
					{
						return false;
					}

					lastEdge = SafeNormalize2d(edge);
				}
			}

			return true;
		}

		// ---------------------------------------------------------------------------
		// TriangulateConvex — fast alternating triangulation for convex polygons
		// ---------------------------------------------------------------------------

		/// <summary>Alternating fan triangulation, valid only for convex contours.</summary>
		/// <param name="polys">The indexed polygons.</param>
		/// <returns>The triangles, in original vertex indices.</returns>
		private static List<IVec3> TriangulateConvex(PolygonsIdx polys)
		{
			// `saturating_sub` on usize: a contour of 0 or 1 verts contributes 0, it does
			// not wrap. C#'s int would go negative instead, hence the explicit clamps.
			int numTri = 0;
			foreach (SimplePolygonIdx p in polys)
			{
				numTri += Math.Max(0, p.Count - 2);
			}

			List<IVec3> triangles = new List<IVec3>(numTri);
			foreach (SimplePolygonIdx poly in polys)
			{
				int i = 0;
				int k = Math.Max(0, poly.Count - 1);
				bool right = true;
				while (i + 1 < k)
				{
					int j = right ? i + 1 : k - 1;
					triangles.Add(new IVec3(poly[i].Idx, poly[j].Idx, poly[k].Idx));
					if (right)
					{
						i = j;
					}
					else
					{
						k = j;
					}

					right = !right;
				}
			}

			return triangles;
		}

		// ---------------------------------------------------------------------------
		// HalfedgeTriangulation — triangulation result as paired halfedges
		// ---------------------------------------------------------------------------

		/// <summary>
		/// Mirrors <c>HalfedgeTriangulation::AddHalfedge</c>: pair against the most
		/// recently added unpaired opposite edge, else record as unpaired.
		/// </summary>
		/// <param name="halfedges">The halfedge arena being appended to.</param>
		/// <param name="edge2halfedge">The unpaired-halfedge multimap, keyed on (start, end).</param>
		/// <param name="start">Start vertex of the new halfedge.</param>
		/// <param name="end">End vertex of the new halfedge.</param>
		private static void AddHalfedge(
			List<Halfedge> halfedges,
			Dictionary<(int Start, int End), List<int>> edge2halfedge,
			int start,
			int end)
		{
			int idx = halfedges.Count;
			Halfedge data = new Halfedge(start, end, -1, -1);
			if (edge2halfedge.TryGetValue((end, start), out List<int>? reverse))
			{
				// Rust `Vec::pop` takes the LAST element: the multimap is LIFO, and that is
				// the property the pairing of duplicate vertex pairs depends on. A C#
				// `List<int>` popped from the end is the same container; `Queue` or
				// `RemoveAt(0)` would be a different (and wrong) triangulation.
				if (reverse.Count > 0)
				{
					int pair = reverse[reverse.Count - 1];
					reverse.RemoveAt(reverse.Count - 1);
					data.PairedHalfedge = pair;
					Halfedge other = halfedges[pair];
					other.PairedHalfedge = idx;
					halfedges[pair] = other;
					if (reverse.Count == 0)
					{
						edge2halfedge.Remove((end, start));
					}

					halfedges.Add(data);
					return;
				}
			}

			// Probe-only map: nothing iterates it, so a plain Dictionary is safe here (the
			// `rustc-hash` → `Dictionary` rule in PORTING_PLAN.md).
			if (!edge2halfedge.TryGetValue((start, end), out List<int>? forward))
			{
				forward = new List<int>();
				edge2halfedge.Add((start, end), forward);
			}

			forward.Add(idx);
			halfedges.Add(data);
		}

		/// <summary>
		/// Triangulates indexed polygons, returning both the triangle halfedges and
		/// the contour halfedges with their pairing. Port of C++
		/// <c>TriangulateIdxHalfedges</c> (polygon.cpp): the C++ triangulators build the
		/// <see cref="HalfedgeTriangulation"/> incrementally via <c>AddTriangle</c>; here we
		/// replay the triangle list (which <see cref="TriangulateIdx"/> emits in exactly
		/// that call order) through the same <c>AddHalfedge</c> sequence, which yields
		/// identical pairing.
		/// </summary>
		/// <param name="polys">The indexed polygons to triangulate.</param>
		/// <param name="epsilon">Tolerance; negative means "derive from the bounding box".</param>
		/// <param name="allowConvex">Permit the convex fast path.</param>
		/// <returns>The paired halfedges of the triangulation.</returns>
		public static HalfedgeTriangulation TriangulateIdxHalfedges(
			PolygonsIdx polys,
			double epsilon,
			bool allowConvex)
		{
			List<IVec3> triangles = TriangulateIdx(polys, epsilon, allowConvex);

			HalfedgeTriangulation result = new HalfedgeTriangulation
			{
				Halfedges = new List<Halfedge>(),
				ContourEnd = 0,
			};
			Dictionary<(int Start, int End), List<int>> edge2halfedge
				= new Dictionary<(int Start, int End), List<int>>();

			// AddContours: store the exterior contour halfedge, opposite the filled
			// contour, so each triangle edge on the boundary pairs against it.
			foreach (SimplePolygonIdx poly in polys)
			{
				for (int i = 0; i < poly.Count; i++)
				{
					int start = poly[i].Idx;
					int end = poly[i + 1 < poly.Count ? i + 1 : 0].Idx;
					AddHalfedge(result.Halfedges, edge2halfedge, end, start);
				}
			}

			result.ContourEnd = result.Halfedges.Count;

			foreach (IVec3 tri in triangles)
			{
				AddHalfedge(result.Halfedges, edge2halfedge, tri.X, tri.Y);
				AddHalfedge(result.Halfedges, edge2halfedge, tri.Y, tri.Z);
				AddHalfedge(result.Halfedges, edge2halfedge, tri.Z, tri.X);
			}

#if DEBUG
			// Mirror of C++ Finalize() MANIFOLD_DEBUG checks. Rust guards these with
			// #[cfg(debug_assertions)], which is this #if.
			Debug.Assert(edge2halfedge.Count == 0, "triangulation has unpaired halfedges");
			for (int i = 0; i < result.Halfedges.Count; i++)
			{
				Halfedge he = result.Halfedges[i];
				int pair = he.PairedHalfedge;
				Debug.Assert(pair >= 0 && pair < result.Halfedges.Count, "invalid paired halfedge");
				Halfedge other = result.Halfedges[pair];
				Debug.Assert(other.PairedHalfedge == i, "halfedge pair is not reciprocal");
				Debug.Assert(
					he.StartVert == other.EndVert && he.EndVert == other.StartVert,
					"halfedge pair endpoints do not match");
			}
#endif

			return result;
		}

		// ---------------------------------------------------------------------------
		// Public API
		// ---------------------------------------------------------------------------

		/// <summary>
		/// Triangulates indexed polygons. Returns triangle indices referencing original
		/// vertex indices.
		/// </summary>
		/// <param name="polys">The indexed polygons; holes wound opposite the outers.</param>
		/// <param name="epsilon">Tolerance; negative means "derive from the bounding box".</param>
		/// <param name="allowConvex">Permit the convex fast path.</param>
		/// <returns>The triangles, in original vertex indices.</returns>
		public static List<IVec3> TriangulateIdx(PolygonsIdx polys, double epsilon, bool allowConvex)
		{
			if (polys.Count == 0 || polys.All(p => p.Count == 0))
			{
				return new List<IVec3>();
			}

			if (allowConvex && IsConvex(polys, epsilon))
			{
				return TriangulateConvex(polys);
			}

			(List<IVec3> triangles, double _) = new EarClip(polys, epsilon).Triangulate();
			return triangles;
		}

		/// <summary>
		/// Triangulates unindexed polygons. Vertices are indexed sequentially across all
		/// contours.
		/// </summary>
		/// <param name="polygons">The polygons; holes wound opposite the outers.</param>
		/// <param name="epsilon">Tolerance; negative means "derive from the bounding box".</param>
		/// <param name="allowConvex">Permit the convex fast path.</param>
		/// <returns>The triangles, indexed sequentially across the contours.</returns>
		public static List<IVec3> Triangulate(Polygons polygons, double epsilon, bool allowConvex)
		{
			int idx = 0;
			PolygonsIdx polygonsIndexed = new PolygonsIdx();
			foreach (SimplePolygon poly in polygons)
			{
				SimplePolygonIdx simple = new SimplePolygonIdx(poly.Count);
				foreach (Vec2 pos in poly)
				{
					simple.Add(new PolyVert(pos, idx));
					idx++;
				}

				polygonsIndexed.Add(simple);
			}

			return TriangulateIdx(polygonsIndexed, epsilon, allowConvex);
		}

		/// <summary>
		/// Rust's <c>partial_cmp(...).unwrap_or(Ordering::Equal)</c>, optionally with the
		/// operands swapped for a descending order.
		/// </summary>
		private sealed class PartialCmpComparer : IComparer<double>
		{
			private readonly bool descending;

			internal PartialCmpComparer(bool descending)
			{
				this.descending = descending;
			}

			public int Compare(double x, double y)
			{
				double a = this.descending ? y : x;
				double b = this.descending ? x : y;
				if (a < b)
				{
					return -1;
				}

				if (a > b)
				{
					return 1;
				}

				// Equal, or one of them is NaN — Rust's unwrap_or(Equal). Comparer<double>
				// .Default instead sorts NaN below everything, which is a different order.
				return 0;
			}
		}
	}

	/// <summary>
	/// Triangulation result represented directly as halfedges, ported from
	/// <c>HalfedgeTriangulation</c> in C++ <c>polygon_internal.h</c>.
	/// </summary>
	/// <remarks>
	/// <c>Halfedges[0..ContourEnd]</c> are the *exterior* contour halfedges (the
	/// input edges reversed, so their <c>EndVert</c> holds the original <c>Idx</c> of the
	/// edge's start point). <c>Halfedges[ContourEnd..]</c> hold three halfedges per
	/// output triangle. All vertex fields are original <c>Idx</c> values.
	/// <para>
	/// Pairing happens incrementally with a LIFO multimap keyed on <c>(start, end)</c>,
	/// so duplicate vertex pairs (from degenerate/overlapping polygons) still pair
	/// one-to-one within the triangulation. This is what <c>Face2Tri</c> relies on to
	/// keep boolean results manifold on exactly-coplanar faces — re-deriving pairs
	/// from vertex positions after the fact cannot distinguish duplicates.
	/// </para>
	/// </remarks>
	public sealed class HalfedgeTriangulation
	{
		/// <summary>The contour halfedges followed by three halfedges per triangle.</summary>
		public List<Halfedge> Halfedges = new List<Halfedge>();

		/// <summary>One past the last contour halfedge; the first triangle halfedge.</summary>
		public int ContourEnd;

		/// <summary>The number of triangles in the triangulation.</summary>
		/// <returns>The triangle count.</returns>
		public int NumTri()
		{
			return (this.Halfedges.Count - this.ContourEnd) / 3;
		}
	}
}
