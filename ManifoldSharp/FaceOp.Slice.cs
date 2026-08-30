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

// -----------------------------------------------------------------------
// Slice & Project — cross-section operations on ManifoldImpl
// -----------------------------------------------------------------------
//
// The tail of face_op.rs: its `impl ManifoldImpl` block, holding `slice` and
// `project` (Manifold::Impl::Slice / ::Project in src/face_op.cpp). It lands in
// its own file because the rest of face_op.rs already fills three (see the
// FaceOp.cs header), and because these two are members of ManifoldImpl rather
// than of the FaceOp static class — the Rust file's one `impl` block.
//
// ── DIVERGENCE: the polygon seeding order is pinned ──────────────────────────
// `slice` is the one place in this port where the Rust is not a function of its
// input. It collects the plane-straddling triangles into a
// `std::collections::HashSet<usize>` and seeds each output loop with
// `tris.iter().next()`, whose iteration order `RandomState` randomizes per
// process — so the Rust's own polygon order, and each loop's starting vertex,
// differ between two runs of the same Rust binary. Under the exactness bar's
// "genuinely unspecified Rust behavior" clause this port pins it:
//
//   SortedSet<int> replaces the HashSet, and each loop is seeded from the
//   SMALLEST triangle index still remaining in it.
//
// The consequences, which are the observable part of the pin: polygons come out
// ordered by the smallest triangle index they contain, ascending; and each
// polygon's point sequence starts at the crossing contributed by that seed
// triangle. Nothing else changes on a manifold mesh — the straddling-triangle
// SET, the loop membership, the walk, and every coordinate are the Rust's, so
// each polygon is bit-identical to a Rust polygon up to cyclic rotation. The one
// further difference is off that path: where the Rust panics on an unpaired
// halfedge, this port throws (see the guard in the walk, which C# integer
// division would otherwise turn into a silent restart at triangle 0). See
// docs/RUST_DIVERGENCES.md entry 3.
//
// `project` has no such problem — `assemble_halfedges` seeds from a `BTreeMap`,
// which is ordered — and is ported literally.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <content>
	/// Slicing and silhouette projection — the <c>impl ManifoldImpl</c> block at the end
	/// of <c>face_op.rs</c>.
	/// </content>
	public sealed partial class ManifoldImpl
	{
		/// <summary>
		/// Slice the mesh at the given Z height, returning 2D polygon loops.
		/// </summary>
		/// <remarks>
		/// Mirrors <c>Manifold::Impl::Slice</c> in <c>src/face_op.cpp</c>. The order of the
		/// returned loops, and the vertex each loop starts at, are pinned here and are a
		/// documented divergence from the Rust — see the file header and
		/// docs/RUST_DIVERGENCES.md entry 3.
		/// </remarks>
		/// <param name="height">The Z height to slice at.</param>
		/// <returns>One closed loop of 2D points per cross-section contour.</returns>
		/// <exception cref="InvalidOperationException">
		/// The walk reached an unpaired halfedge — a soup impl, which
		/// <see cref="Manifold.Slice"/> screens out before calling this.
		/// </exception>
		public Polygons Slice(double height)
		{
			int numTri = this.NumTri();
			if (numTri == 0)
			{
				return new Polygons();
			}

			// Build plane query box spanning the full XY extent at the given Z.
			// Box is a struct, so this is the Rust's `let mut plane = self.bbox` copy;
			// mutating it does not touch the cached bbox.
			Box plane = this.Bbox;
			plane.Min.Z = height;
			plane.Max.Z = height;

			// Query the cached face BVH (C++ Slice uses collider_).
			Collider collider = this.Collider;

			// Find all triangles that straddle the slice plane.
			//
			// DIVERGENCE (file header): the Rust holds these in a `HashSet<usize>` and
			// seeds each loop below with `tris.iter().next()`. A SortedSet holds the same
			// set with an order that is a function of the input, so `Min` is a
			// reproducible seed where `iter().next()` is not.
			SortedSet<int> tris = new SortedSet<int>();
			Box[] query = new Box[] { Box.FromPoints(plane.Min, plane.Max) };
			collider.CollisionsWithBoxes(query, false, (_, tri) =>
			{
				double minZ = double.PositiveInfinity;
				double maxZ = double.NegativeInfinity;
				for (int j = 0; j < 3; j++)
				{
					double z = this.VertPos[this.Halfedge[(3 * tri) + j].StartVert].Z;
					// Rust `f64::min`/`f64::max`, which return the non-NaN operand where
					// Math.Min/Math.Max propagate NaN — MinF64/MaxF64 are the wrappers
					// that screen for that (docs/RUST_DIVERGENCES.md entry 2).
					minZ = MinF64(minZ, z);
					maxZ = MaxF64(maxZ, z);
				}

				if (minZ <= height && maxZ > height)
				{
					tris.Add(tri);
				}
			});

			// Trace polygon loops through intersected triangles
			Polygons polys = new Polygons();
			while (tris.Count > 0)
			{
				// The pinned seed: smallest remaining triangle index (Rust:
				// `*tris.iter().next().unwrap()`, i.e. an arbitrary member).
				int startTri = tris.Min;
				SimplePolygon poly = new SimplePolygon();

				// Find the edge where the slice enters (above→below transition)
				int k = 0;
				for (int j = 0; j < 3; j++)
				{
					int nextJ = (j + 1) % 3;
					if (this.VertPos[this.Halfedge[(3 * startTri) + j].StartVert].Z > height
						&& this.VertPos[this.Halfedge[(3 * startTri) + nextJ].StartVert].Z <= height)
					{
						k = nextJ;
						break;
					}
				}

				int tri = startTri;
				while (true)
				{
					tris.Remove(tri);
					if (this.VertPos[this.Halfedge[(3 * tri) + k].EndVert].Z <= height)
					{
						k = (k + 1) % 3;
					}

					Halfedge up = this.Halfedge[(3 * tri) + k];
					Vec3 below = this.VertPos[up.StartVert];
					Vec3 above = this.VertPos[up.EndVert];
					double a = (height - below.Z) / (above.Z - below.Z);

					// The Rust writes this crossing out longhand as
					// `below + a * (above - below)`, NOT as the `a*(1-t) + b*t` form the
					// porting rules mandate for `linalg::lerp` — those are different
					// roundings, and this one is what C++ Slice computes. Transcribed as
					// written; do not "fix" it to the lerp form.
					Vec2 pt = new Vec2(
						below.X + (a * (above.X - below.X)),
						below.Y + (a * (above.Y - below.Y)));
					poly.Add(pt);

					int pair = up.PairedHalfedge;
					if (pair < 0)
					{
						// The Rust-panic analog. Rust writes `pair as usize / 3`, so an
						// unpaired -1 widens to a huge index and panics on the next
						// halfedge access. C# int division gives -1/3 == 0 instead, which
						// would silently restart the walk at triangle 0 and can spin
						// forever — a wrong answer where the Rust stops. Unreachable from
						// a manifold mesh (Manifold.Slice screens IsSoup first), but a
						// direct ManifoldImpl.Slice call on a soup impl reaches it.
						throw new InvalidOperationException(
							"Non-manifold edge! Slice walked onto an unpaired halfedge "
							+ $"{(3 * tri) + k} (triangle {tri}).");
					}

					tri = pair / 3;
					k = ((pair % 3) + 1) % 3;

					if (tri == startTri)
					{
						break;
					}
				}

				polys.Add(poly);
			}

			return polys;
		}

		/// <summary>
		/// Project the mesh silhouette onto the XY plane, returning 2D polygon loops.
		/// </summary>
		/// <remarks>
		/// Mirrors <c>Manifold::Impl::Project</c> in <c>src/face_op.cpp</c>. Unlike
		/// <see cref="Slice"/> this is deterministic in the Rust —
		/// <see cref="FaceOp.AssembleHalfedges"/> seeds from an ordered map — and is ported
		/// literally.
		/// </remarks>
		/// <returns>One closed loop of 2D points per silhouette contour.</returns>
		public Polygons Project()
		{
			if (this.NumTri() == 0 || this.FaceNormal.Count == 0)
			{
				return new Polygons();
			}

			Proj2x3 projection = FaceOp.GetAxisAlignedProjection(new Vec3(0.0, 0.0, 1.0));

			// Find cusp edges: silhouette edges where one adjacent face points up
			// and the other points down (z-component of normals).
			//
			// The two locals are named as the Rust names them, and the Rust's names are
			// the two faces swapped: pairing is an involution, so
			// `halfedge[edge.paired_halfedge].paired_halfedge` is `edge`'s own index and
			// `paired_face` is therefore edge's OWN face, while `this_face` is the paired
			// one. The test is "own face points up, neighbour points down"; only the
			// spelling is backwards, and it is kept so the two sources diff line for line.
			List<Halfedge> cusps = new List<Halfedge>();
			foreach (Halfedge edge in this.Halfedge)
			{
				int pairedFace = this.Halfedge[edge.PairedHalfedge].PairedHalfedge / 3;
				int thisFace = edge.PairedHalfedge / 3;
				if (this.FaceNormal[pairedFace].Z >= 0.0 && this.FaceNormal[thisFace].Z < 0.0)
				{
					cusps.Add(edge);
				}
			}

			if (cusps.Count == 0)
			{
				return new Polygons();
			}

			List<List<int>> loops = FaceOp.AssembleHalfedges(cusps, 0);
			PolygonsIdx polysIndexed = FaceOp.ProjectPolygons(loops, cusps, this.VertPos, projection);

			// Convert PolygonsIdx to Polygons
			Polygons polys = new Polygons();
			foreach (SimplePolygonIdx poly in polysIndexed)
			{
				SimplePolygon simple = new SimplePolygon(poly.Count);
				foreach (PolyVert pv in poly)
				{
					simple.Add(pv.Pos);
				}

				polys.Add(simple);
			}

			return polys;
		}
	}
}
