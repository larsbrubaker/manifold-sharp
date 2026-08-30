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

// BooleanResult.Edges.cs — the pairing half of boolean_result.rs: CppPartition,
// PairUp, and the three Append*Edges that drive them. See BooleanResult.cs for
// the module header.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <content>
	/// PairUp and the edge-append passes.
	/// </content>
	internal static partial class BooleanResult
	{
		// -------------------------------------------------------------------
		// PairUp — pair start/end vertices to form halfedges
		// -------------------------------------------------------------------

		/// <summary>
		/// Port of C++ <c>std::partition</c> (Hoare two-pointer, bidirectional form used by
		/// MSVC and libstdc++): elements satisfying <paramref name="pred"/> end up first.
		/// UNSTABLE — the swap pattern determines the relative order of fully-tied
		/// <see cref="EdgePos"/> entries (identical edgePos AND collisionId, e.g. duplicated
		/// retained verts on coincident geometry), which the subsequent stable sorts
		/// preserve. That order decides how degenerate edges pair up, so it must match C++
		/// exactly.
		/// </summary>
		private static int CppPartition<T>(List<T> v, Func<T, bool> pred)
		{
			int first = 0;
			int last = v.Count;
			while (true)
			{
				while (true)
				{
					if (first == last)
					{
						return first;
					}

					if (!pred(v[first]))
					{
						break;
					}

					first += 1;
				}

				while (true)
				{
					last -= 1;
					if (first == last)
					{
						return first;
					}

					if (pred(v[last]))
					{
						break;
					}
				}

				(v[first], v[last]) = (v[last], v[first]);
				first += 1;
			}
		}

		private static void PairUp(List<EdgePos> edgePos, Action<Halfedge> f)
		{
			if (edgePos.Count % 2 != 0)
			{
				// The Rust's `assert!`, which panics rather than debug-asserts: this is a
				// real (if rare) input condition, not an internal invariant.
				int starts = edgePos.Count(e => e.IsStart);
				int ends = edgePos.Count(e => !e.IsStart);
				throw new InvalidOperationException(
					"Non-manifold edge! Not an even number of points. "
					+ $"Got {edgePos.Count} points: starts={starts}, ends={ends}");
			}

			int nEdges = edgePos.Count / 2;

			// C++ PairUp: unstable partition (starts first), then stable_sort each
			// half, then pair edgePos[i] with edgePos[i + nEdges].
			int middle = CppPartition(edgePos, e => e.IsStart);
			System.Diagnostics.Debug.Assert(middle == nEdges, "Non-manifold edge!");
			SortRangeByKey(edgePos, 0, nEdges);
			SortRangeByKey(edgePos, nEdges, edgePos.Count - nEdges);

			for (int i = 0; i < nEdges; i++)
			{
				f(new Halfedge(edgePos[i].Vert, edgePos[i + nEdges].Vert, -1, -1));
			}
		}

		/// <summary>
		/// The Rust's <c>sort_by_key(|e| e.sort_key())</c> over one range of the list.
		/// </summary>
		/// <remarks>
		/// SORT AUDIT (boolean_result.rs:330/331/360/463): Rust <c>sort_by_key</c>, which is
		/// STABLE, on the key <c>(OrderedF64(edge_pos), collision_id)</c>. Full ties are the
		/// interesting case, not the exception — duplicated retained verts on coincident
		/// geometry share both key components — and their relative order is the output of
		/// <see cref="CppPartition"/>, which this sort must carry through unchanged, because
		/// it decides how degenerate edges pair up. LINQ OrderBy/ThenBy is the
		/// documented-stable C# sort; <c>List&lt;T&gt;.Sort</c> is introsort and is not
		/// usable here.
		/// </remarks>
		private static void SortRangeByKey(List<EdgePos> list, int start, int count)
		{
			List<EdgePos> sorted = list
				.GetRange(start, count)
				.OrderBy(e => e.EdgePosition, Polygon.PartialCmp)
				.ThenBy(e => e.CollisionId)
				.ToList();
			for (int i = 0; i < count; i++)
			{
				list[start + i] = sorted[i];
			}
		}

		// -------------------------------------------------------------------
		// AppendPartialEdges — edges of P/Q that have intersections
		// -------------------------------------------------------------------

		internal static void AppendPartialEdges(
			ManifoldImpl outR,
			bool[] wholeHalfedgeP,
			List<int> facePtrR,
			SortedDictionary<int, List<EdgePos>> edgesP,
			List<TriRef> halfedgeRef,
			ManifoldImpl inP,
			IReadOnlyList<int> i03,
			IReadOnlyList<int> vp2r,
			IReadOnlyList<int> faceP2r,
			bool forward)
		{
			foreach (KeyValuePair<int, List<EdgePos>> entry in edgesP)
			{
				int edgeP = entry.Key;
				List<EdgePos> edgePosP = entry.Value;
				SortRangeByKey(edgePosP, 0, edgePosP.Count);

				Halfedge halfedge = inP.Halfedge[edgeP];
				wholeHalfedgeP[edgeP] = false;
				wholeHalfedgeP[halfedge.PairedHalfedge] = false;

				int vStart = halfedge.StartVert;
				int vEnd = halfedge.EndVert;

				// C++ computes edgeVec from the INPUT mesh positions — the output
				// slots at vp2r[v] hold this vertex's position only when it is
				// retained (i03 != 0), so projecting along an output-derived vector
				// would use garbage for non-retained endpoints.
				Vec3 edgeVec = inP.VertPos[vEnd] - inP.VertPos[vStart];

				// Fill in edge positions of existing intersection verts
				for (int i = 0; i < edgePosP.Count; i++)
				{
					EdgePos edge = edgePosP[i];
					edge.EdgePosition = Dot(outR.VertPos[edge.Vert], edgeVec);
					edgePosP[i] = edge;
				}

				// Add start vertex (only retained verts — inclusion != 0 — have
				// valid output positions at vp2r)
				int inclusion = i03[vStart];
				if (inclusion != 0)
				{
					double startPos = Dot(outR.VertPos[vp2r[vStart]], edgeVec);
					for (int j = 0; j < Math.Abs(inclusion); j++)
					{
						edgePosP.Add(new EdgePos(startPos, vp2r[vStart] + j, int.MaxValue, inclusion > 0));
					}
				}

				// Add end vertex
				inclusion = i03[vEnd];
				if (inclusion != 0)
				{
					double endPos = Dot(outR.VertPos[vp2r[vEnd]], edgeVec);
					for (int j = 0; j < Math.Abs(inclusion); j++)
					{
						edgePosP.Add(new EdgePos(endPos, vp2r[vEnd] + j, int.MaxValue, inclusion < 0));
					}
				}

				// Pair up and add halfedges to result
				int faceLeftP = edgeP / 3;
				int faceLeft = faceP2r[faceLeftP];
				int faceRightP = halfedge.PairedHalfedge / 3;
				int faceRight = faceP2r[faceRightP];

				TriRef forwardRef = new TriRef(forward ? 0 : 1, -1, faceLeftP, -1);
				TriRef backwardRef = new TriRef(forward ? 0 : 1, -1, faceRightP, -1);

				PairUp(edgePosP, e =>
				{
					int forwardEdge = facePtrR[faceLeft];
					facePtrR[faceLeft] += 1;
					int backwardEdge = facePtrR[faceRight];
					facePtrR[faceRight] += 1;

					e.PairedHalfedge = backwardEdge;
					outR.Halfedge[forwardEdge] = e;
					halfedgeRef[forwardEdge] = forwardRef;

					Halfedge rev = new Halfedge(e.EndVert, e.StartVert, forwardEdge, -1);
					outR.Halfedge[backwardEdge] = rev;
					halfedgeRef[backwardEdge] = backwardRef;
				});
			}
		}

		// -------------------------------------------------------------------
		// AppendNewEdges — edges formed at face-face intersections
		// -------------------------------------------------------------------

		internal static void AppendNewEdges(
			ManifoldImpl outR,
			List<int> facePtrR,
			SortedDictionary<(int FaceP, int FaceQ), List<EdgePos>> edgesNew,
			List<TriRef> halfedgeRef,
			IReadOnlyList<int> facePq2r,
			int numFaceP)
		{
			foreach (KeyValuePair<(int FaceP, int FaceQ), List<EdgePos>> entry in edgesNew)
			{
				int faceP = entry.Key.FaceP;
				int faceQ = entry.Key.FaceQ;
				List<EdgePos> edgePos = entry.Value;
				SortRangeByKey(edgePos, 0, edgePos.Count);

				// Compute bounding box to find longest dimension
				Vec3 min = Vec3.Splat(double.PositiveInfinity);
				Vec3 max = Vec3.Splat(double.NegativeInfinity);
				foreach (EdgePos edge in edgePos)
				{
					Vec3 p = outR.VertPos[edge.Vert];
					min.X = MinF64(min.X, p.X);
					min.Y = MinF64(min.Y, p.Y);
					min.Z = MinF64(min.Z, p.Z);
					max.X = MaxF64(max.X, p.X);
					max.Y = MaxF64(max.Y, p.Y);
					max.Z = MaxF64(max.Z, p.Z);
				}

				Vec3 size = max - min;

				// Order points along longest dimension
				int dim = size.X > size.Y && size.X > size.Z
					? 0
					: size.Y > size.Z ? 1 : 2;

				for (int i = 0; i < edgePos.Count; i++)
				{
					EdgePos edge = edgePos[i];
					Vec3 p = outR.VertPos[edge.Vert];
					edge.EdgePosition = dim switch
					{
						0 => p.X,
						1 => p.Y,
						_ => p.Z,
					};
					edgePos[i] = edge;
				}

				int faceLeft = facePq2r[faceP];
				int faceRight = facePq2r[numFaceP + faceQ];
				TriRef forwardRef = new TriRef(0, -1, faceP, -1);
				TriRef backwardRef = new TriRef(1, -1, faceQ, -1);

				PairUp(edgePos, e =>
				{
					int forwardEdge = facePtrR[faceLeft];
					facePtrR[faceLeft] += 1;
					int backwardEdge = facePtrR[faceRight];
					facePtrR[faceRight] += 1;

					e.PairedHalfedge = backwardEdge;
					outR.Halfedge[forwardEdge] = e;
					halfedgeRef[forwardEdge] = forwardRef;

					Halfedge rev = new Halfedge(e.EndVert, e.StartVert, forwardEdge, -1);
					outR.Halfedge[backwardEdge] = rev;
					halfedgeRef[backwardEdge] = backwardRef;
				});
			}
		}

		// -------------------------------------------------------------------
		// AppendWholeEdges — edges with no intersections (fully retained)
		// -------------------------------------------------------------------

		internal static void AppendWholeEdges(
			ManifoldImpl outR,
			List<int> facePtrR,
			List<TriRef> halfedgeRef,
			ManifoldImpl inP,
			bool[] wholeHalfedgeP,
			IReadOnlyList<int> i03,
			IReadOnlyList<int> vp2r,
			IReadOnlyList<int> faceP2r,
			bool forward)
		{
			for (int idx = 0; idx < inP.Halfedge.Count; idx++)
			{
				if (!wholeHalfedgeP[idx])
				{
					continue;
				}

				Halfedge halfedge = inP.Halfedge[idx];
				if (!halfedge.IsForward())
				{
					continue;
				}

				int inclusion = i03[halfedge.StartVert];
				if (inclusion == 0)
				{
					continue;
				}

				if (inclusion < 0)
				{
					// Reverse the halfedge
					(halfedge.StartVert, halfedge.EndVert) = (halfedge.EndVert, halfedge.StartVert);
				}

				halfedge.StartVert = vp2r[halfedge.StartVert];
				halfedge.EndVert = vp2r[halfedge.EndVert];

				int faceLeftP = idx / 3;
				int newFace = faceP2r[faceLeftP];
				int faceRightP = halfedge.PairedHalfedge / 3;
				int faceRight = faceP2r[faceRightP];

				TriRef forwardRef = new TriRef(forward ? 0 : 1, -1, faceLeftP, -1);
				TriRef backwardRef = new TriRef(forward ? 0 : 1, -1, faceRightP, -1);

				for (int i = 0; i < Math.Abs(inclusion); i++)
				{
					int forwardEdge = facePtrR[newFace];
					facePtrR[newFace] += 1;
					int backwardEdge = facePtrR[faceRight];
					facePtrR[faceRight] += 1;

					Halfedge he = new Halfedge(
						halfedge.StartVert + i,
						halfedge.EndVert + i,
						backwardEdge,
						-1);
					outR.Halfedge[forwardEdge] = he;
					halfedgeRef[forwardEdge] = forwardRef;

					Halfedge rev = new Halfedge(
						halfedge.EndVert + i,
						halfedge.StartVert + i,
						forwardEdge,
						-1);
					outR.Halfedge[backwardEdge] = rev;
					halfedgeRef[backwardEdge] = backwardRef;
				}
			}
		}
	}
}
