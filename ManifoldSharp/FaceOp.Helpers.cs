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

// FaceOp.Helpers.cs — the second half of face_op.rs: GetBarycentric, the
// polygon-assembly helpers Face2Tri runs on (AssembleHalfedges,
// ProjectPolygons) and ReorderHalfedges. The module header, and the
// normals/coplanarity half it describes, live in FaceOp.cs.

using System.Runtime.InteropServices;

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <content>
	/// Barycentric coordinates, polygon-loop assembly and projection, and the
	/// canonical halfedge reordering.
	/// </content>
	public static partial class FaceOp
	{
		// -------------------------------------------------------------------
		// GetBarycentric — barycentric coordinates of point in triangle
		// -------------------------------------------------------------------

		/// <summary>
		/// Compute barycentric coordinates of <paramref name="v"/> with respect to triangle
		/// <paramref name="triPos"/>. Returns [u, v, w] where vertex i has weight uvw[i].
		/// Returns exact 1.0 for vertices within <paramref name="tolerance"/> of a triangle
		/// vertex, and exact 0.0 for points within tolerance of an edge.
		/// </summary>
		/// <remarks>Mirrors <c>GetBarycentric</c> in <c>src/shared.h</c>.</remarks>
		/// <param name="v">The point to locate.</param>
		/// <param name="triPos">The triangle's three corner positions.</param>
		/// <param name="tolerance">The snap-to-vertex / snap-to-edge tolerance.</param>
		/// <returns>The barycentric weights.</returns>
		public static Vec3 GetBarycentric(Vec3 v, Vec3[] triPos, double tolerance)
		{
			Vec3[] edges = new Vec3[]
			{
				triPos[2] - triPos[1],
				triPos[0] - triPos[2],
				triPos[1] - triPos[0],
			};
			double[] d2 = new double[]
			{
				Dot(edges[0], edges[0]),
				Dot(edges[1], edges[1]),
				Dot(edges[2], edges[2]),
			};
			int longSide = d2[0] > d2[1] && d2[0] > d2[2]
				? 0
				: (d2[1] > d2[2] ? 1 : 2);
			Vec3 crossP = Cross(edges[0], edges[1]);
			double area2 = Dot(crossP, crossP);
			double tol2 = tolerance * tolerance;

			Vec3 uvw = Vec3.Splat(0.0);
			for (int i = 0; i < 3; i++)
			{
				Vec3 dv = v - triPos[i];
				if (Dot(dv, dv) < tol2)
				{
					uvw = Vec3.Splat(0.0);
					switch (i)
					{
						case 0:
							uvw.X = 1.0;
							break;
						case 1:
							uvw.Y = 1.0;
							break;
						default:
							uvw.Z = 1.0;
							break;
					}

					return uvw;
				}
			}

			if (d2[longSide] < tol2)
			{
				// Degenerate point
				return new Vec3(1.0, 0.0, 0.0);
			}
			else if (area2 > d2[longSide] * tol2)
			{
				// Triangle case
				for (int i = 0; i < 3; i++)
				{
					int j = (i + 1) % 3;
					Vec3 crossPv = Cross(edges[i], v - triPos[j]);
					double area2v = Dot(crossPv, crossPv);
					double val = area2v < d2[i] * tol2 ? 0.0 : Dot(crossPv, crossP);
					switch (i)
					{
						case 0:
							uvw.X = val;
							break;
						case 1:
							uvw.Y = val;
							break;
						default:
							uvw.Z = val;
							break;
					}
				}

				double sum = uvw.X + uvw.Y + uvw.Z;
				uvw = uvw / sum;
				return uvw;
			}
			else
			{
				// Line case
				int nextV = (longSide + 1) % 3;
				double alpha = Dot(v - triPos[nextV], edges[longSide]) / d2[longSide];
				uvw = Vec3.Splat(0.0);
				int lastV = (nextV + 1) % 3;
				switch (nextV)
				{
					case 0:
						uvw.X = 1.0 - alpha;
						break;
					case 1:
						uvw.Y = 1.0 - alpha;
						break;
					default:
						uvw.Z = 1.0 - alpha;
						break;
				}

				switch (lastV)
				{
					case 0:
						uvw.X = alpha;
						break;
					case 1:
						uvw.Y = alpha;
						break;
					default:
						uvw.Z = alpha;
						break;
				}

				return uvw;
			}
		}

		// -------------------------------------------------------------------
		// AssembleHalfedges — group halfedges into polygon loops
		// -------------------------------------------------------------------

		/// <summary>
		/// Given a slice of halfedges (from a polygonal face), group them into polygon loops
		/// by following start_vert → end_vert chains. Returns a list of polygon loops, where
		/// each loop is a list of halfedge indices (offset by
		/// <paramref name="startHalfedgeIdx"/>).
		/// </summary>
		/// <remarks>
		/// Mirrors <c>AssembleHalfedges</c> in <c>src/face_op.cpp</c>. The vert→edge multimap
		/// is a <see cref="SortedDictionary{TKey, TValue}"/> and not a
		/// <see cref="Dictionary{TKey, TValue}"/>: the Rust is a <c>BTreeMap</c> and the loop
		/// picks each new contour's seed with <c>iter().next()</c>, so the *smallest
		/// remaining start vertex* decides both which loop is emitted next and where it
		/// starts. An unordered map would permute the returned contours.
		/// </remarks>
		/// <param name="halfedges">The face's halfedges, in face-local order.</param>
		/// <param name="startHalfedgeIdx">The index the first of them has in the caller's arena.</param>
		/// <returns>One list of arena halfedge indices per polygon loop.</returns>
		public static List<List<int>> AssembleHalfedges(
			IReadOnlyList<Halfedge> halfedges,
			int startHalfedgeIdx)
		{
			// Build multimap: start_vert → local edge index
			SortedDictionary<int, List<int>> vertEdge = new SortedDictionary<int, List<int>>();
			for (int i = 0; i < halfedges.Count; i++)
			{
				int key = halfedges[i].StartVert;
				if (!vertEdge.TryGetValue(key, out List<int>? bucket))
				{
					bucket = new List<int>();
					vertEdge.Add(key, bucket);
				}

				bucket.Add(i);
			}

			List<List<int>> polys = new List<List<int>>();
			int startEdge = 0;
			int thisEdge = startEdge;

			while (true)
			{
				if (thisEdge == startEdge)
				{
					// Find next unvisited edge
					if (vertEdge.Count == 0)
					{
						break;
					}

					List<int> seed = FirstValue(vertEdge);
					startEdge = seed[0];
					thisEdge = startEdge;
					polys.Add(new List<int>());
				}

				polys[polys.Count - 1].Add(startHalfedgeIdx + thisEdge);
				int endVert = halfedges[thisEdge].EndVert;

				// Rust `.expect("non-manifold edge")` — a missing bucket is a broken input,
				// not a recoverable state, so the indexer's KeyNotFoundException is the
				// faithful behaviour.
				List<int> edges = vertEdge[endVert];

				// Remove the first occurrence
				thisEdge = edges[0];
				edges.RemoveAt(0);
				if (edges.Count == 0)
				{
					vertEdge.Remove(endVert);
				}
			}

			return polys;
		}

		/// <summary>
		/// Project polygon loops into 2D using projection matrix and vertex positions.
		/// </summary>
		/// <remarks>Mirrors <c>ProjectPolygons</c> in <c>src/face_op.cpp</c>.</remarks>
		/// <param name="polys">The polygon loops, as arena halfedge indices.</param>
		/// <param name="halfedge">The halfedge arena the loops index.</param>
		/// <param name="vertPos">The vertex positions.</param>
		/// <param name="projection">The plane projection to apply.</param>
		/// <returns>The 2D polygons, each vertex tagged with its halfedge index.</returns>
		public static PolygonsIdx ProjectPolygons(
			IReadOnlyList<List<int>> polys,
			IReadOnlyList<Halfedge> halfedge,
			IReadOnlyList<Vec3> vertPos,
			Proj2x3 projection)
		{
			PolygonsIdx polygons = new PolygonsIdx();
			foreach (List<int> poly in polys)
			{
				SimplePolygonIdx simplePoly = new SimplePolygonIdx();
				foreach (int edge in poly)
				{
					int vert = halfedge[edge].StartVert;
					simplePoly.Add(new PolyVert(projection.Apply(vertPos[vert]), edge));
				}

				polygons.Add(simplePoly);
			}

			return polygons;
		}

		// -------------------------------------------------------------------
		// ReorderHalfedges — canonical ordering within each triangle
		// -------------------------------------------------------------------

		/// <summary>
		/// Reorders halfedges within each face so the one with the smallest start_vert is
		/// first, then fixes paired_halfedge references.
		/// </summary>
		/// <remarks>
		/// Mirrors <c>Manifold::Impl::ReorderHalfedges</c> in <c>src/sort.cpp</c> — it lives
		/// in face_op.rs there, so it lives here.
		/// </remarks>
		/// <param name="mesh">The mesh whose halfedges are rotated in place.</param>
		public static void ReorderHalfedges(ManifoldImpl mesh)
		{
			int numTri = mesh.Halfedge.Count / 3;

			// Rust's `self.halfedge[i] = ...` on an arena; CollectionsMarshal.AsSpan is the
			// port of that, and the list is not resized while the span is alive.
			Span<Halfedge> halfedge = CollectionsMarshal.AsSpan(mesh.Halfedge);

			// Step 1: rotate each triangle so smallest start_vert is first
			for (int tri = 0; tri < numTri; tri++)
			{
				int baseIdx = tri * 3;
				Halfedge[] face = new Halfedge[]
				{
					halfedge[baseIdx],
					halfedge[baseIdx + 1],
					halfedge[baseIdx + 2],
				};
				if (face[0].StartVert < 0)
				{
					continue;
				}

				int index = 0;
				for (int i = 1; i < 3; i++)
				{
					if (face[i].StartVert < face[index].StartVert)
					{
						index = i;
					}
				}

				for (int i = 0; i < 3; i++)
				{
					halfedge[baseIdx + i] = face[(index + i) % 3];
				}
			}

			// Step 2: fix paired_halfedge references
			for (int tri = 0; tri < numTri; tri++)
			{
				for (int i = 0; i < 3; i++)
				{
					int baseIdx = (tri * 3) + i;
					ref Halfedge curr = ref halfedge[baseIdx];
					if (curr.StartVert < 0)
					{
						break; // skip collapsed triangle
					}

					if (curr.PairedHalfedge < 0)
					{
						continue; // unpaired halfedge
					}

					int oppFace = curr.PairedHalfedge / 3;

					// `index` stays -1 when no halfedge of the opposite face ends at this
					// halfedge's start vertex, which writes `oppFace * 3 - 1`. That is the
					// Rust's (and C++'s) behaviour on an inconsistent pairing and is
					// transcribed rather than repaired — the inputs this runs on have
					// already been through CreateHalfedges.
					int index = -1;
					for (int j = 0; j < 3; j++)
					{
						if (curr.StartVert == halfedge[(oppFace * 3) + j].EndVert)
						{
							index = j;
						}
					}

					curr.PairedHalfedge = (oppFace * 3) + index;
				}
			}
		}

		/// <summary>The smallest-keyed value of an ordered map — Rust <c>iter().next()</c>.</summary>
		private static List<int> FirstValue(SortedDictionary<int, List<int>> map)
		{
			foreach (KeyValuePair<int, List<int>> entry in map)
			{
				return entry.Value;
			}

			throw new InvalidOperationException("FirstValue called on an empty map");
		}
	}
}
