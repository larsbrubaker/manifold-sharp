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

// Manifold.Smooth.cs — port of manifold_smooth.rs, whose header reads:
//
//   manifold_smooth.rs — smoothing and refinement methods on Manifold:
//   normal calculation, tangent creation (Smooth/SmoothOut/SmoothByNormals)
//   and the Refine family that subdivides toward the smooth surface.
//
//   Extracted from manifold.rs for file size management; a child module of
//   `manifold` so these methods keep access to the private `imp` field and
//   callers keep the same `m.refine(...)` paths. Ports the corresponding
//   methods of C++ Manifold (smoothing.cpp, subdivision.cpp via
//   Impl::Subdivide, interp_tri for tangent interpolation).
//
// The "child module of `manifold`" arrangement is a `partial class Manifold`
// continuation here, which keeps the same access to the private `imp` field.
//
// ── Nothing here is deferred ─────────────────────────────────────────────────
// Every machine this file drives — SetNormals, SharpenEdges,
// UpdateSharpenedEdges, CreateTangents, CreateTangentsFromNormals,
// ValidTangents, Subdivide, InterpTri, CalculateVertNormals — landed with
// Phase 7. This file is the last of Phase 6.
//
// ── The CreateTangents seam ──────────────────────────────────────────────────
// Rust's `create_tangents` takes `Vec<Smoothness>` **by value** and pushes the
// flat-face edges onto it as it works; the C# takes `IReadOnlyList<Smoothness>`
// and copies. That is observationally identical at both call sites below,
// because in the Rust the vector is *moved* in and never read again:
// `smooth` moves the result of `update_sharpened_edges`, and `smooth_out` moves
// the result of `sharpen_edges`. No caller here needs the mutated list back, so
// the copy costs one allocation and nothing else. A future caller that DOES want
// the appended flat-face edges must change CreateTangents, not work around it.
//
// ── FinishRefine ─────────────────────────────────────────────────────────────
// The three Refine entry points end with a tail the Rust repeats *character for
// character* (interp_tri, clear tangents, calculate_bbox, set_epsilon,
// sort_geometry, normals, original_id = -1). It is one private helper here so
// there is one body to audit against the Rust instead of three copies to keep in
// step. Note refine_to_tolerance early-returns when there are no tangents, so its
// two `had_tangents` tests are statically true — the shared helper preserves that
// rather than relying on it.
//
// ── No degree/radian conversions ─────────────────────────────────────────────
// manifold_smooth.rs has no `.to_degrees()`/`.to_radians()` call of its own: the
// `min_sharp_angle` arguments are passed through in degrees to SetNormals and
// SharpenEdges, which do their own conversion (via Smoothing.ToDegrees).

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	public sealed partial class Manifold
	{
		/// <summary>
		/// Port of C++ <c>Manifold::CalculateNormals()</c>.
		/// Fills in vertex properties for normals. Edges sharper than
		/// <paramref name="minSharpAngle"/> (degrees) get separate normals on each side.
		/// </summary>
		/// <param name="normalIdx">
		/// The property slot the three normal components start at. The Rust types this
		/// <c>usize</c>, so a negative value is unrepresentable there; <c>int</c> here can
		/// carry one and this method does not guard it, which keeps the behaviour identical
		/// to the Rust for every value the Rust can express.
		/// </param>
		/// <param name="minSharpAngle">The dihedral angle, in degrees, above which an edge is sharp.</param>
		/// <returns>A manifold carrying vertex normals in the requested property slots.</returns>
		public Manifold CalculateNormals(int normalIdx, double minSharpAngle)
		{
			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			if (this.IsEmpty())
			{
				return this.Clone();
			}

			ManifoldImpl outR = this.imp.Clone();
			outR.SetNormals(normalIdx, minSharpAngle);

			// Per #1718: record per-meshID hasNormals so get_mesh_gl(-1) can
			// auto-substitute slot 0 on export. Restricted to the standard slot —
			// a non-zero slot would be ambiguous when round-tripping through MeshGL.
			if (normalIdx == 0)
			{
				// Rust iterates `values_mut()`; Relation is a struct here, so the value
				// has to be read out, edited and written back (see its remarks in
				// Types.Shared.cs). Materializing the keys first is not required —
				// re-assigning an EXISTING key during enumeration has been legal since
				// .NET Core 3.0 — it just makes the read-edit-write-back explicit, and
				// keeps this loop obviously safe if it ever grows a branch that adds a
				// key, which would not be.
				List<int> meshIds = new List<int>(outR.MeshRelation.MeshIdTransform.Keys);
				foreach (int meshId in meshIds)
				{
					Relation rel = outR.MeshRelation.MeshIdTransform[meshId];
					rel.HasNormals = true;
					outR.MeshRelation.MeshIdTransform[meshId] = rel;
				}
			}

			return FromImpl(outR);
		}

		/// <summary>
		/// Port of C++ <c>Manifold::Smooth(MeshGL, sharpenedEdges)</c>.
		/// Constructs a smooth version of the input mesh by creating tangents.
		/// The actual triangle resolution is unchanged; use <see cref="Refine"/> to
		/// interpolate to a higher-resolution curve.
		/// </summary>
		/// <param name="meshGl">The mesh to smooth.</param>
		/// <param name="sharpenedEdges">Per-halfedge smoothness overrides; may be empty.</param>
		/// <returns>The smoothed manifold.</returns>
		public static Manifold Smooth(MeshGL meshGl, IReadOnlyList<Smoothness> sharpenedEdges)
		{
			ArgumentNullException.ThrowIfNull(meshGl);
			ArgumentNullException.ThrowIfNull(sharpenedEdges);

			// Assign sequential faceIDs if not present
			MeshGL meshTmp = meshGl.Clone();
			int numTri = meshTmp.NumTri();
			ResizeFaceId(meshTmp.FaceId, numTri);
			for (int i = 0; i < numTri; i++)
			{
				meshTmp.FaceId[i] = (uint)i;
			}

			Manifold m = FromMeshGL(meshTmp);
			if (m.IsEmpty())
			{
				return m;
			}

			// UpdateSharpenedEdges + CreateTangents
			List<Smoothness> sharpened = new List<Smoothness>(sharpenedEdges);
			List<Smoothness> updated = m.imp.UpdateSharpenedEdges(sharpened);
			m.imp.CreateTangents(updated);

			// Restore original faceIDs
			int numTriImpl = m.imp.NumTri();
			for (int i = 0; i < numTriImpl; i++)
			{
				if (i < m.imp.MeshRelation.TriRef.Count)
				{
					int faceId = m.imp.MeshRelation.TriRef[i].FaceId;
					if (meshGl.FaceId.Count == numTri && faceId >= 0 && faceId < numTri)
					{
						TriRef triRef = m.imp.MeshRelation.TriRef[i];
						triRef.FaceId = (int)meshGl.FaceId[faceId];
						m.imp.MeshRelation.TriRef[i] = triRef;
					}
					else
					{
						TriRef triRef = m.imp.MeshRelation.TriRef[i];
						triRef.FaceId = -1;
						m.imp.MeshRelation.TriRef[i] = triRef;
					}
				}
			}

			return m;
		}

		/// <summary>Port of C++ <c>Manifold::SmoothOut()</c>.</summary>
		/// <param name="minSharpAngle">The dihedral angle, in degrees, above which an edge stays sharp.</param>
		/// <param name="minSmoothness">The smoothness floor applied to sharpened edges.</param>
		/// <returns>The manifold with smoothing tangents attached.</returns>
		public Manifold SmoothOut(double minSharpAngle, double minSmoothness)
		{
			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			if (this.IsEmpty())
			{
				return this.Clone();
			}

			ManifoldImpl outR = this.imp.Clone();

			// Per C++ #1724 (Fix CalculateNormals): SmoothOut is now self-consistent —
			// it always derives tangents from SharpenEdges, regardless of
			// min_smoothness. The old min_smoothness==0 path (SetNormals +
			// CreateTangentsFromNormals + property restore) was removed.
			List<Smoothness> sharpened = outR.SharpenEdges(minSharpAngle, minSmoothness);
			outR.CreateTangents(sharpened);
			return FromImpl(outR);
		}

		/// <summary>Port of C++ <c>Manifold::SmoothByNormals()</c>.</summary>
		/// <param name="normalIdx">
		/// The property slot the three normal components start at. Unguarded against
		/// negatives for the same reason as <see cref="CalculateNormals"/>: the Rust's
		/// <c>usize</c> cannot express one, so there is no Rust behaviour to match.
		/// </param>
		/// <returns>The manifold with smoothing tangents derived from its stored normals.</returns>
		public Manifold SmoothByNormals(int normalIdx)
		{
			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			if (this.IsEmpty())
			{
				return this.Clone();
			}

			ManifoldImpl outR = this.imp.Clone();
			outR.CreateTangentsFromNormals(normalIdx);
			return FromImpl(outR);
		}

		/// <summary>
		/// Port of C++ <c>Manifold::Refine(int n)</c>.
		/// Splits every edge into n pieces, sub-triangulating each face.
		/// </summary>
		/// <param name="n">How many pieces to split each edge into; 1 or less is a no-op.</param>
		/// <returns>The refined manifold.</returns>
		public Manifold Refine(int n)
		{
			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			if (n <= 1 || this.imp.IsEmpty())
			{
				return this.Clone();
			}

			ManifoldImpl outR = this.imp.Clone();
			if (!outR.ValidTangents())
			{
				outR.MakeEmpty(Error.InvalidTangents);
				return FromImpl(outR);
			}

			ManifoldImpl old = outR.Clone();
			bool hadTangents = outR.HalfedgeTangent.Count == outR.Halfedge.Count;
			List<Barycentric> vertBary = outR.Subdivide((_, _, _) => n - 1, false);
			FinishRefine(outR, old, vertBary, hadTangents);
			return FromImpl(outR);
		}

		/// <summary>Port of C++ <c>Manifold::RefineToLength(double length)</c>.</summary>
		/// <param name="length">The target edge length; the absolute value is used, 0 is a no-op.</param>
		/// <returns>The refined manifold.</returns>
		public Manifold RefineToLength(double length)
		{
			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			double len = Math.Abs(length);
			if (len == 0.0 || this.imp.IsEmpty())
			{
				return this.Clone();
			}

			ManifoldImpl outR = this.imp.Clone();
			if (!outR.ValidTangents())
			{
				outR.MakeEmpty(Error.InvalidTangents);
				return FromImpl(outR);
			}

			ManifoldImpl old = outR.Clone();
			bool hadTangents = outR.HalfedgeTangent.Count == outR.Halfedge.Count;
			List<Barycentric> vertBary = outR.Subdivide(
				(edgeVec, _, _) =>
				{
					double edgeLen = Math.Sqrt(
						(edgeVec.X * edgeVec.X) + (edgeVec.Y * edgeVec.Y) + (edgeVec.Z * edgeVec.Z));

					// C++: static_cast<int>(la::length(edge) / length) — truncation
					return (int)(edgeLen / len);
				},
				false);
			FinishRefine(outR, old, vertBary, hadTangents);
			return FromImpl(outR);
		}

		/// <summary>Port of C++ <c>Manifold::RefineToTolerance(double tolerance)</c>.</summary>
		/// <param name="tolerance">The target deviation from the smooth surface; 0 is a no-op.</param>
		/// <returns>The refined manifold.</returns>
		public Manifold RefineToTolerance(double tolerance)
		{
			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			double tol = Math.Abs(tolerance);
			if (tol == 0.0 || this.imp.IsEmpty())
			{
				return this.Clone();
			}

			// C++ only refines when tangents are present
			ManifoldImpl outR = this.imp.Clone();
			bool hadTangents = outR.HalfedgeTangent.Count == outR.Halfedge.Count;
			if (!hadTangents)
			{
				return this.Clone();
			}

			if (!outR.ValidTangents())
			{
				outR.MakeEmpty(Error.InvalidTangents);
				return FromImpl(outR);
			}

			ManifoldImpl old = outR.Clone();
			List<Barycentric> vertBary = outR.Subdivide(
				(edgeVec, tangent0, tangent1) =>
				{
					double edgeLen = Math.Sqrt(
						(edgeVec.X * edgeVec.X) + (edgeVec.Y * edgeVec.Y) + (edgeVec.Z * edgeVec.Z));
					if (edgeLen == 0.0)
					{
						return 0;
					}

					Vec3 edgeNorm = new Vec3(
						edgeVec.X / edgeLen,
						edgeVec.Y / edgeLen,
						edgeVec.Z / edgeLen);
					Vec3 tStart = new Vec3(tangent0.X, tangent0.Y, tangent0.Z);
					Vec3 tEnd = new Vec3(tangent1.X, tangent1.Y, tangent1.Z);

					// Perpendicular to edge
					double dotS = (edgeNorm.X * tStart.X) + (edgeNorm.Y * tStart.Y) + (edgeNorm.Z * tStart.Z);
					Vec3 start = new Vec3(
						tStart.X - (edgeNorm.X * dotS),
						tStart.Y - (edgeNorm.Y * dotS),
						tStart.Z - (edgeNorm.Z * dotS));
					double dotE = (edgeNorm.X * tEnd.X) + (edgeNorm.Y * tEnd.Y) + (edgeNorm.Z * tEnd.Z);
					Vec3 end = new Vec3(
						tEnd.X - (edgeNorm.X * dotE),
						tEnd.Y - (edgeNorm.Y * dotE),
						tEnd.Z - (edgeNorm.Z * dotE));

					// Circular arc result plus heuristic term for non-circular curves
					double lenStart = Math.Sqrt((start.X * start.X) + (start.Y * start.Y) + (start.Z * start.Z));
					double lenEnd = Math.Sqrt((end.X * end.X) + (end.Y * end.Y) + (end.Z * end.Z));
					Vec3 diff = new Vec3(start.X - end.X, start.Y - end.Y, start.Z - end.Z);
					double lenDiff = Math.Sqrt((diff.X * diff.X) + (diff.Y * diff.Y) + (diff.Z * diff.Z));
					double d = (0.5 * (lenStart + lenEnd)) + lenDiff;
					return (int)Math.Sqrt(3.0 * d / (4.0 * tol));
				},
				true);
			FinishRefine(outR, old, vertBary, hadTangents);
			return FromImpl(outR);
		}

		/// <summary>
		/// The tail the three Refine entry points share verbatim in the Rust: interpolate
		/// the new vertices onto the smooth surface, drop the now-stale tangents, and
		/// rebuild the derived state.
		/// </summary>
		/// <param name="outR">The subdivided impl, edited in place.</param>
		/// <param name="old">The pre-subdivision impl the tangents belong to.</param>
		/// <param name="vertBary">Subdivide's per-vertex barycentric coordinates.</param>
		/// <param name="hadTangents">Whether the input carried a full set of halfedge tangents.</param>
		private static void FinishRefine(
			ManifoldImpl outR,
			ManifoldImpl old,
			List<Barycentric> vertBary,
			bool hadTangents)
		{
			if (hadTangents && vertBary.Count != 0)
			{
				InterpTriFunctions.InterpTri(outR.VertPos, vertBary, old);
			}

			outR.HalfedgeTangent.Clear();
			outR.CalculateBBox();
			outR.SetEpsilon(-1.0, false);
			outR.SortGeometry();
			if (hadTangents)
			{
				outR.SetNormalsAndCoplanar();
			}
			else
			{
				FaceOp.CalculateVertNormals(outR);
			}

			outR.MeshRelation.OriginalId = -1;
		}

		/// <summary>
		/// Rust <c>Vec::resize(n, 0)</c> on the faceID list: truncate when longer, pad with
		/// zeros when shorter. <c>List&lt;T&gt;</c> has no single call with that meaning.
		/// </summary>
		/// <param name="faceId">The list to resize in place.</param>
		/// <param name="numTri">The target length.</param>
		private static void ResizeFaceId(List<uint> faceId, int numTri)
		{
			if (faceId.Count > numTri)
			{
				faceId.RemoveRange(numTri, faceId.Count - numTri);
			}
			else
			{
				while (faceId.Count < numTri)
				{
					faceId.Add(0);
				}
			}
		}
	}
}
