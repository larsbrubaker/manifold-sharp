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

// QuickHull.Exact.cs — NOT A PORT; divergence ledger entry 7. The one test
// quickhull_algo.rs makes with the float plane distance that this port makes exactly,
// kept out of QuickHull.Algo.cs (which sits near its exemption ceiling).
//
// The flood fill calls a face visible when the apex is above its plane, and every
// edge between a visible and a hidden face becomes a horizon edge coned to the apex.
// C++ quickhull.cpp and the Rust read "above" off the float distance to the face's
// stored plane. When the apex is collinear with an edge, it lies in the plane of both
// faces on that edge, and rounding can put one face at +5.6e-17 and the other at 0:
// the edge becomes a horizon edge and the new face has zero area, with a normal that
// is noise. Later visibility tests against that plane are noise too, and the hull
// ends non-convex (Thingi10K 63451 triangle 163 swept by Sphere(0.3, 8) left a vertex
// 0.58 outside a face, which the convex-dilation union tree turned into lost volume;
// 22 of the 368 hulls CppNonConvexConvexMinkowskiSum builds were non-convex).
//
// The exact orientation of the apex against the face's three corners gives both faces
// the same answer, zero, so neither is visible and no zero-area face is built. A point
// is also queued on a face only when it is exactly above it, so the face the fill
// starts from is always visible; the epsilon that decides whether a point is outside
// at all is unchanged.

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp
{
	/// <content>The exact above-the-face test.</content>
	internal sealed partial class QuickHull
	{
		/// <summary>
		/// Whether a point is strictly above a face: on the side its counterclockwise
		/// normal points to, decided exactly from the face's corners.
		/// </summary>
		/// <param name="face">The face.</param>
		/// <param name="point">The point.</param>
		/// <returns>True when the point is strictly above the face's plane.</returns>
		private bool IsAbove(int face, Vec3 point)
		{
			IVec3 corners = this.mesh.GetVertexIndicesOfFaceByIndex(face);
			return Filtered.Orient3d(this.verts[corners[0]], this.verts[corners[1]], this.verts[corners[2]], point)
				== Sign.Pos;
		}
	}
}
