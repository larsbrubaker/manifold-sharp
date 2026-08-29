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

// QuickHull algorithm internals — extracted from quickhull.rs
// Contains MeshBuilder, Pool, FaceData, and QuickHull struct/impl
//
// (That is quickhull_algo.rs's header verbatim. This file carries the first
// three of those four — Face, QHEdge, MeshBuilder, Pool and FaceData; the
// QuickHull driver is in QuickHull.Algo.cs. See the QuickHull.cs header for the
// split and for why the coupling between the two survives it.)
//
// ── Arena discipline ─────────────────────────────────────────────────────────
// Everything here is an index arena, exactly as the Rust: faces and halfedges
// are `List` entries addressed by `int`, never by reference, and the index
// encodings are preserved unchanged —
//   * a face is disabled iff `He == -1`, and disabled faces are recycled
//     through `DisabledFaces` (LIFO);
//   * a halfedge is disabled iff `PairedHalfedge == -1`, recycled through
//     `DisabledHalfedges` (LIFO);
//   * `HalfedgeNext` is the per-face 3-cycle; `HalfedgeToFace` is its inverse.
// Recycling order is part of the result, not an optimization: which index a
// recycled face or halfedge gets decides the order faces are visited in, which
// decides the output triangle order. Both pop from the back, and must keep
// doing so.
//
// ── Why Face is a class and QHEdge is a struct ───────────────────────────────
// The Rust mutates single fields of `faces[i]` in a dozen places. In C# a
// `List<T>` of a *struct* cannot do that (CS1612), and the two escapes the port
// uses elsewhere are both unavailable here: `CollectionsMarshal.AsSpan` is
// invalid because the driver appends to `Faces` and `Halfedges` mid-walk (see
// the EdgeOp.cs header for the same hazard), and read-modify-store helpers for
// nine fields would bury the transcription. So `Face` is a sealed class whose
// list slot is a reference — which also matches the Rust's recycling, where
// `add_face` reuses the existing record and deliberately leaves
// `VisibilityCheckedOnIteration` and `MostDistantPoint` at their stale values.
// `QHEdge` has two fields and stays a struct, mutated through the two
// read-modify-store helpers at the bottom of MeshBuilder.

using System.Diagnostics;

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <summary>A triangular face of the working hull.</summary>
	internal sealed class Face
	{
		/// <summary>One halfedge of this face, or -1 when the face is disabled.</summary>
		public int He;

		/// <summary>The supporting plane, valid once the driver computes it.</summary>
		public Plane Plane;

		/// <summary>The distance to <see cref="MostDistantPoint"/>.</summary>
		public double MostDistantPointDist;

		/// <summary>The point of <see cref="PointsOnPositiveSide"/> farthest from the plane.</summary>
		public int MostDistantPoint;

		/// <summary>
		/// The driver iteration this face's visibility was last classified on.
		/// </summary>
		/// <remarks>
		/// The Rust counter is a <c>usize</c> that wraps, so this is a <see cref="ulong"/>
		/// and not an <c>int</c>: on the 64-bit targets the Rust port supports, the wrap is
		/// unreachable, and narrowing it here would move that unreachable point.
		/// </remarks>
		public ulong VisibilityCheckedOnIteration;

		/// <summary>Whether the last classification found this face visible.</summary>
		public bool IsVisibleFaceOnCurrentIteration;

		/// <summary>Whether this face is queued in the driver's face list.</summary>
		public bool InFaceStack;

		/// <summary>Bit i is set when this face's i-th halfedge is a horizon edge.</summary>
		public byte HorizonEdgesOnCurrentIteration;

		/// <summary>
		/// Indices of the input points on this face's positive side, or null for the Rust
		/// <c>None</c>. Owned lists come from, and go back to, the driver's <see cref="Pool"/>.
		/// </summary>
		public List<int>? PointsOnPositiveSide;

		/// <summary>Creates a face anchored on the halfedge <paramref name="he"/>.</summary>
		/// <param name="he">The face's first halfedge, or -1 for a disabled face.</param>
		public Face(int he)
		{
			this.He = he;
			this.Plane = Plane.Default();
			this.MostDistantPointDist = 0.0;
			this.MostDistantPoint = 0;
			this.VisibilityCheckedOnIteration = 0;
			this.IsVisibleFaceOnCurrentIteration = false;
			this.InFaceStack = false;
			this.HorizonEdgesOnCurrentIteration = 0;
			this.PointsOnPositiveSide = null;
		}

		/// <summary>Creates a face in the disabled state.</summary>
		/// <returns>The disabled face.</returns>
		public static Face Disabled()
		{
			return new Face(-1);
		}

		/// <summary>Marks this face disabled.</summary>
		public void Disable()
		{
			this.He = -1;
		}

		/// <summary>True when this face is disabled.</summary>
		/// <returns>True when <see cref="He"/> is -1.</returns>
		public bool IsDisabled()
		{
			return this.He == -1;
		}
	}

	/// <summary>
	/// The internal half-edge used during quickhull construction.
	/// Only uses <c>EndVert</c> and <c>PairedHalfedge</c> -- <c>StartVert</c> is set at the end.
	/// </summary>
	internal struct QHEdge
	{
		/// <summary>The vertex this halfedge ends at.</summary>
		public int EndVert;

		/// <summary>The opposed halfedge, or -1 when this halfedge is disabled.</summary>
		public int PairedHalfedge;
	}

	/// <summary>The half-edge mesh the hull is grown in.</summary>
	internal sealed class MeshBuilder
	{
		/// <summary>The face arena.</summary>
		public List<Face> Faces;

		/// <summary>The halfedge arena.</summary>
		public List<QHEdge> Halfedges;

		/// <summary>The face each halfedge belongs to, parallel to <see cref="Halfedges"/>.</summary>
		public List<int> HalfedgeToFace;

		/// <summary>The next halfedge around the face, parallel to <see cref="Halfedges"/>.</summary>
		public List<int> HalfedgeNext;

		/// <summary>Free list of disabled face indices, popped from the back.</summary>
		public List<int> DisabledFaces;

		/// <summary>Free list of disabled halfedge indices, popped from the back.</summary>
		public List<int> DisabledHalfedges;

		/// <summary>Creates an empty mesh.</summary>
		public MeshBuilder()
		{
			this.Faces = new List<Face>();
			this.Halfedges = new List<QHEdge>();
			this.HalfedgeToFace = new List<int>();
			this.HalfedgeNext = new List<int>();
			this.DisabledFaces = new List<int>();
			this.DisabledHalfedges = new List<int>();
		}

		/// <summary>
		/// Allocates a face, recycling the most recently disabled one when there is one.
		/// </summary>
		/// <returns>The face index; the face is left disabled for the caller to anchor.</returns>
		public int AddFace()
		{
			if (this.DisabledFaces.Count > 0)
			{
				int index = this.DisabledFaces[this.DisabledFaces.Count - 1];
				this.DisabledFaces.RemoveAt(this.DisabledFaces.Count - 1);
				Face f = this.Faces[index];
				Debug.Assert(f.IsDisabled(), "recycled face was not disabled");
				f.MostDistantPointDist = 0.0;
				f.He = -1; // will be set by caller
				f.IsVisibleFaceOnCurrentIteration = false;
				f.InFaceStack = false;
				f.HorizonEdgesOnCurrentIteration = 0;
				f.PointsOnPositiveSide = null;
				return index;
			}

			this.Faces.Add(Face.Disabled());
			return this.Faces.Count - 1;
		}

		/// <summary>
		/// Allocates a halfedge, recycling the most recently disabled one when there is one.
		/// </summary>
		/// <returns>The halfedge index; a recycled slot keeps its stale field values.</returns>
		public int AddHalfedge()
		{
			if (this.DisabledHalfedges.Count > 0)
			{
				int index = this.DisabledHalfedges[this.DisabledHalfedges.Count - 1];
				this.DisabledHalfedges.RemoveAt(this.DisabledHalfedges.Count - 1);
				return index;
			}

			this.Halfedges.Add(default(QHEdge));
			this.HalfedgeToFace.Add(0);
			this.HalfedgeNext.Add(0);
			return this.Halfedges.Count - 1;
		}

		/// <summary>Resets the mesh to the tetrahedron on the four given vertices.</summary>
		/// <param name="a">The first vertex.</param>
		/// <param name="b">The second vertex.</param>
		/// <param name="c">The third vertex.</param>
		/// <param name="d">The fourth vertex.</param>
		public void Setup(int a, int b, int c, int d)
		{
			this.Faces.Clear();
			this.Halfedges.Clear();
			this.HalfedgeToFace.Clear();
			this.HalfedgeNext.Clear();
			this.DisabledFaces.Clear();
			this.DisabledHalfedges.Clear();

			// Create 12 halfedges for 4 faces (tetrahedron)
			// Face 0: AB, BC, CA
			this.PushHe(0, b, 6, 0, 1); // 0: AB
			this.PushHe(0, c, 9, 0, 2); // 1: BC
			this.PushHe(0, a, 3, 0, 0); // 2: CA

			// Face 1: AC, CD, DA
			this.PushHe(0, c, 2, 1, 4); // 3: AC
			this.PushHe(0, d, 11, 1, 5); // 4: CD
			this.PushHe(0, a, 7, 1, 3); // 5: DA

			// Face 2: BA, AD, DB
			this.PushHe(0, a, 0, 2, 7); // 6: BA
			this.PushHe(0, d, 5, 2, 8); // 7: AD
			this.PushHe(0, b, 10, 2, 6); // 8: DB

			// Face 3: CB, BD, DC
			this.PushHe(0, b, 1, 3, 10); // 9: CB
			this.PushHe(0, d, 8, 3, 11); // 10: BD
			this.PushHe(0, c, 4, 3, 9); // 11: DC

			this.Faces.Add(new Face(0));
			this.Faces.Add(new Face(3));
			this.Faces.Add(new Face(6));
			this.Faces.Add(new Face(9));
		}

		/// <summary>Appends one halfedge and its parallel-array entries.</summary>
		/// <param name="start">
		/// Unused, as in the Rust <c>_start</c>: QHEdge has no start vertex, which is set
		/// only when the mesh is exported. Kept so the twelve calls above read as a table.
		/// </param>
		/// <param name="end">The end vertex.</param>
		/// <param name="paired">The opposed halfedge.</param>
		/// <param name="face">The owning face.</param>
		/// <param name="next">The next halfedge around the face.</param>
		public void PushHe(int start, int end, int paired, int face, int next)
		{
			_ = start;
			this.Halfedges.Add(new QHEdge
			{
				EndVert = end,
				PairedHalfedge = paired,
			});
			this.HalfedgeToFace.Add(face);
			this.HalfedgeNext.Add(next);
		}

		/// <summary>Returns vertex indices of a face given its index. Does not borrow &amp;Face.</summary>
		/// <param name="faceIndex">The face.</param>
		/// <returns>The face's three vertices, in ring order.</returns>
		public IVec3 GetVertexIndicesOfFaceByIndex(int faceIndex)
		{
			int i0 = this.Faces[faceIndex].He;
			int i1 = this.HalfedgeNext[i0];
			int i2 = this.HalfedgeNext[i1];
			return new IVec3(
				this.Halfedges[i0].EndVert,
				this.Halfedges[i1].EndVert,
				this.Halfedges[i2].EndVert);
		}

		/// <summary>Returns the two vertices a halfedge runs between.</summary>
		/// <param name="heIdx">The halfedge.</param>
		/// <returns>The start vertex (its pair's end) then the end vertex.</returns>
		public IVec2 GetVertexIndicesOfHalfedge(int heIdx)
		{
			int paired = this.Halfedges[heIdx].PairedHalfedge;
			return new IVec2(
				this.Halfedges[paired].EndVert,
				this.Halfedges[heIdx].EndVert);
		}

		/// <summary>Returns halfedge indices of a face given its index. Does not borrow &amp;Face.</summary>
		/// <param name="faceIndex">The face.</param>
		/// <returns>The face's three halfedges, in ring order.</returns>
		public IVec3 GetHalfedgeIndicesOfFaceByIndex(int faceIndex)
		{
			int i0 = this.Faces[faceIndex].He;
			int i1 = this.HalfedgeNext[i0];
			int i2 = this.HalfedgeNext[i1];
			return new IVec3(i0, i1, i2);
		}

		/// <summary>Disables a face and hands its point list back to the caller.</summary>
		/// <param name="faceIndex">The face to disable.</param>
		/// <returns>The face's points-on-positive-side list, or null when it had none.</returns>
		public List<int>? DisableFace(int faceIndex)
		{
			Face f = this.Faces[faceIndex];
			f.Disable();
			this.DisabledFaces.Add(faceIndex);

			// Rust `Option::take` — read the list out and leave None behind.
			List<int>? points = f.PointsOnPositiveSide;
			f.PointsOnPositiveSide = null;
			return points;
		}

		/// <summary>Disables a halfedge, adding it to the free list.</summary>
		/// <param name="heIndex">The halfedge to disable.</param>
		public void DisableHalfedge(int heIndex)
		{
			this.SetPairedHalfedge(heIndex, -1);
			this.DisabledHalfedges.Add(heIndex);
		}

		// ---------------------------------------------------------------------
		// Field writers — Rust's `self.halfedges[i].field = x` has no direct C#
		// spelling on a List<QHEdge> (CS1612), and AsSpan is not safe here because
		// AddHalfedge appends to the same list between writes.
		// ---------------------------------------------------------------------

		/// <summary>Sets a halfedge's end vertex.</summary>
		/// <param name="heIndex">The halfedge.</param>
		/// <param name="value">The new end vertex.</param>
		public void SetEndVert(int heIndex, int value)
		{
			QHEdge e = this.Halfedges[heIndex];
			e.EndVert = value;
			this.Halfedges[heIndex] = e;
		}

		/// <summary>Sets a halfedge's paired halfedge.</summary>
		/// <param name="heIndex">The halfedge.</param>
		/// <param name="value">The new paired halfedge, or -1 to disable it.</param>
		public void SetPairedHalfedge(int heIndex, int value)
		{
			QHEdge e = this.Halfedges[heIndex];
			e.PairedHalfedge = value;
			this.Halfedges[heIndex] = e;
		}
	}

	/// <summary>
	/// Pool for recycling point index vectors.
	/// </summary>
	/// <remarks>
	/// Recycling order reaches the output only through allocation counts, but the pool is
	/// LIFO in the Rust and stays LIFO here so the two allocate identically under a
	/// profiler.
	/// </remarks>
	internal sealed class Pool
	{
		private readonly List<List<int>> data;

		/// <summary>Creates an empty pool.</summary>
		public Pool()
		{
			this.data = new List<List<int>>();
		}

		/// <summary>Takes a cleared list from the pool, or a fresh one when it is empty.</summary>
		/// <returns>An empty list.</returns>
		public List<int> Get()
		{
			if (this.data.Count > 0)
			{
				List<int> v = this.data[this.data.Count - 1];
				this.data.RemoveAt(this.data.Count - 1);
				v.Clear();
				return v;
			}

			return new List<int>();
		}

		/// <summary>Returns a list to the pool.</summary>
		/// <param name="v">The list to reclaim; the caller must drop its reference.</param>
		public void Reclaim(List<int> v)
		{
			this.data.Add(v);
		}

		/// <summary>Drops every pooled list.</summary>
		public void Clear()
		{
			this.data.Clear();
		}
	}

	/// <summary>
	/// FaceData for traversal — a face reached through a halfedge during the
	/// visible-face flood fill.
	/// </summary>
	internal struct FaceData
	{
		/// <summary>The face reached.</summary>
		public int FaceIndex;

		/// <summary>The halfedge it was entered from, -1 for the face the fill started at.</summary>
		public int EnteredFromHalfedge;
	}
}
