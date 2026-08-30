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

// Phase 9: smoothing/tangent generation — ported from
// cpp-reference/manifold/src/smoothing.cpp
//
// (Rust's phase label is kept; this lands in the C# port's Phase 7.)
//
// ── Module shape ─────────────────────────────────────────────────────────────
// smoothing.rs has both module-level free functions and an `impl ManifoldImpl`
// block, so it lands the way collider.rs did: the free functions on a static
// class named for the module (`Smoothing`), the impl block as a partial of
// `ManifoldImpl`. Both live in this file and its companions:
//
//   Smoothing.cs             this file — the free functions (vec3_from_vec4,
//                            safe_normalize, wrap, angle_between, equal_normals,
//                            circular_tangent, collect_vertex_cycle, plus
//                            ToDegrees, which is a translation helper and not in
//                            the Rust) and the small impl methods: GetNormal,
//                            TangentFromNormal, IsInsideQuad,
//                            IsMarkedInsideQuad, UpdateSharpenedEdges,
//                            FlatFaces, VertFlatFace
//   Smoothing.SetNormals.cs  SetNormals alone — 270 lines of Rust with four
//                            distinct orbit walks in it
//   SmoothingTangents.cs     smoothing_tangents.rs (its own module in the Rust,
//                            `#[path]`-included by this one)
//
// ── `.to_degrees()` is not `Types.Degrees` ───────────────────────────────────
// Rust's `f64::to_degrees` is one multiply by the correctly-rounded 180/pi;
// types.rs's `degrees()` is `a * 180.0 / K_PI`, two operations. They differ in
// the last ulp for some inputs, and every use here feeds a `> min_sharp_angle`
// threshold test, so the difference is observable as a flipped sharp-edge
// decision. `ToDegrees` below is the former; do not "simplify" it to the latter.
//
// ── Orbit walks in the smoothing port, and why none of them is ForVert ───────
// Eight sites across smoothing.rs and smoothing_tangents.rs walk the halfedges
// around a vertex, and no two have the same guards. As in EdgeOp.cs, each is
// transcribed inline from its own Rust rather than routed through
// ManifoldImpl.ForVert. The five in this module:
//
//   CollectVertexCycle   pushes *then* steps, so the start halfedge is FIRST in
//                        the returned cycle; carries the `paired < 0` guard that
//                        ForVert lacks, and so is the only walk here safe on an
//                        unpaired mesh.
//   SetNormals, single-normal branch
//                        visits start_edge FIRST and then steps — the C++
//                        ForVert<...>(halfedge, ...) shape where the body runs
//                        before the advance. The `last_prop` dedup below it is
//                        order-sensitive, so this is not interchangeable with
//                        the steps-first form.
//   SetNormals, sharp-edge search
//                        steps first and breaks on the *dihedral*, not on the
//                        cycle: it can exit early at a sharp edge, or wrap all
//                        the way back to start_edge having found none.
//   SetNormals, pseudo-normal accumulation
//                        seeded from end_edge before the loop and stepping first
//                        inside it, so end_edge is both seed and terminator: it
//                        supplies the `here` side of the first pair and the
//                        `next` side of the last. Each iteration works on an
//                        adjacent *pair* of halfedges, which is why the seed
//                        exists at all, and the order it emits `group` in is the
//                        order the assignment walk consumes.
//   SetNormals, property-vertex assignment
//                        steps first (the advance happens once before the loop
//                        too), so end_edge is processed LAST and takes the last
//                        group. This is the invariant docs/PORTING_PLAN.md names
//                        smoothing for; getting it backwards silently assigns
//                        group[0]'s normal to the wrong property vertex.
//
// The three in SmoothingTangents.cs are documented in that file's header.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// The free functions of <c>smoothing.rs</c> — the tangent and normal primitives the
	/// <see cref="ManifoldImpl"/> smoothing methods are built from.
	/// </summary>
	public static class Smoothing
	{
		/// <summary>
		/// Minimum sharp angle in degrees, below which edges are considered coplanar.
		/// Floating point noise in the dihedral angle computation can reach ~1e-6
		/// degrees for nearly-parallel face normals; this threshold must exceed that.
		/// </summary>
		public const double KMinSharpAngle = 1e-4;

		/// <summary>Drops the w component of a <see cref="Vec4"/>.</summary>
		/// <param name="v">The vector to truncate.</param>
		/// <returns>The xyz part.</returns>
		public static Vec3 Vec3FromVec4(Vec4 v)
		{
			return new Vec3(v.X, v.Y, v.Z);
		}

		/// <summary>
		/// Normalize, returning the zero vector for a zero-length or non-finite input.
		/// </summary>
		/// <remarks>
		/// Deliberately not <see cref="ManifoldImpl"/>'s private helper of the same name:
		/// that one normalizes first and tests only the x component of the result, this
		/// one tests the squared length first. The Rust has both, one per module. A
		/// squared length that merely overflows to infinity is not where they part —
		/// (1e200, 1e200, 1e200) gives the zero vector either way. They disagree when a
		/// component other than x is itself infinite: (1, inf, 0) is zero here, and
		/// (0, NaN, 0) there, because the finiteness test passes on the x it computed.
		/// </remarks>
		/// <param name="v">The vector to normalize.</param>
		/// <returns>The unit vector, or zero.</returns>
		public static Vec3 SafeNormalize(Vec3 v)
		{
			double len2 = Dot(v, v);
			if (len2 <= 0.0 || !double.IsFinite(len2))
			{
				return new Vec3(0.0, 0.0, 0.0);
			}

			return v / Math.Sqrt(len2);
		}

		/// <summary>
		/// Shifts an angle by a single turn toward [-pi, pi] — one step, not a modulo.
		/// </summary>
		/// <remarks>
		/// The interval is closed at both ends: the tests are strict, so exactly +/-pi
		/// passes through untouched. Because only one turn is ever added or subtracted,
		/// an input beyond 3pi in magnitude comes back still out of range. Both are the
		/// Rust's behaviour, and the caller (DistributeTangents' offset accumulation)
		/// only ever feeds it differences of angles already within a turn.
		/// </remarks>
		/// <param name="radians">The angle in radians.</param>
		/// <returns>The angle, shifted by at most one turn.</returns>
		public static double Wrap(double radians)
		{
			if (radians < -Types.KPi)
			{
				return radians + Types.KTwoPi;
			}
			else if (radians > Types.KPi)
			{
				return radians - Types.KTwoPi;
			}
			else
			{
				return radians;
			}
		}

		/// <summary>
		/// The angle between two vectors, matching C++'s AngleBetween helper (acos with
		/// +/-1 clamping), the #1634 form.
		/// </summary>
		/// <param name="a">The first vector; expected normalized.</param>
		/// <param name="b">The second vector; expected normalized.</param>
		/// <returns>The angle in radians.</returns>
		public static double AngleBetween(Vec3 a, Vec3 b)
		{
			double d = Dot(a, b);
			if (d >= 1.0)
			{
				return 0.0;
			}
			else if (d <= -1.0)
			{
				return Types.KPi;
			}
			else
			{
				return DeterministicMath.Acos(d);
			}
		}

		/// <summary>
		/// Two normals are considered equal when their normalized dot product exceeds
		/// 0.9999 (~0.81 degrees). Per C++ #1671 this tolerance replaces the old exact /
		/// <c>kPrecision</c>-squared comparisons in the tangent-generation code.
		/// </summary>
		/// <param name="a">The first normal.</param>
		/// <param name="b">The second normal.</param>
		/// <returns>True when the two point the same way within tolerance.</returns>
		public static bool EqualNormals(Vec3 a, Vec3 b)
		{
			return Dot(SafeNormalize(a), SafeNormalize(b)) > 0.9999;
		}

		/// <summary>
		/// The cubic Bezier tangent that traces a circular arc along
		/// <paramref name="edgeVec"/> starting in direction <paramref name="tangent"/>.
		/// </summary>
		/// <param name="tangent">The desired starting direction; need not be normalized.</param>
		/// <param name="edgeVec">The edge the arc spans.</param>
		/// <returns>The tangent as xyz plus a rational-Bezier weight in w.</returns>
		public static Vec4 CircularTangent(Vec3 tangent, Vec3 edgeVec)
		{
			Vec3 dir = SafeNormalize(tangent);
			double weight = MaxF64(0.5, Dot(dir, SafeNormalize(edgeVec)));
			Vec4 bz2 = new Vec4(
				dir.X * 0.5 * Length(edgeVec),
				dir.Y * 0.5 * Length(edgeVec),
				dir.Z * 0.5 * Length(edgeVec),
				weight);
			Vec4 bz3 = new Vec4(
				bz2.X * (2.0 / 3.0),
				bz2.Y * (2.0 / 3.0),
				bz2.Z * (2.0 / 3.0),
				1.0 + (bz2.W - 1.0) * (2.0 / 3.0));
			return new Vec4(bz3.X / bz3.W, bz3.Y / bz3.W, bz3.Z / bz3.W, bz3.W);
		}

		/// <summary>
		/// The halfedges around the vertex <paramref name="start"/> begins at, in orbit
		/// order and with <paramref name="start"/> first.
		/// </summary>
		/// <remarks>
		/// ORBIT WALK: pushes then steps, so the start halfedge is element 0. Unlike
		/// <see cref="ManifoldImpl.ForVert"/> this carries the <c>paired &lt; 0</c> guard,
		/// so it terminates on a boundary instead of indexing off the arena.
		/// </remarks>
		/// <param name="mesh">The mesh to walk.</param>
		/// <param name="start">The halfedge to start from.</param>
		/// <returns>The cycle of halfedge indices.</returns>
		public static List<int> CollectVertexCycle(ManifoldImpl mesh, int start)
		{
			List<int> cycle = new List<int>();
			int current = start;
			while (true)
			{
				cycle.Add(current);
				int paired = mesh.Halfedge[current].PairedHalfedge;
				if (paired < 0)
				{
					break;
				}

				current = ManifoldImpl.NextHalfedge(paired);
				if (current == start)
				{
					break;
				}
			}

			return cycle;
		}

		/// <summary>
		/// Rust's <c>f64::to_degrees</c>: a single multiply by the correctly-rounded
		/// 180/pi. See this file's header for why <see cref="Types.Degrees"/> is not a
		/// substitute at the threshold tests below.
		/// </summary>
		/// <param name="radians">The angle in radians.</param>
		/// <returns>The angle in degrees.</returns>
		internal static double ToDegrees(double radians)
		{
			return radians * (180.0 / Types.KPi);
		}
	}

	/// <content>
	/// The <c>impl ManifoldImpl</c> block of <c>smoothing.rs</c>: normal lookup, tangent
	/// construction from a normal, the quad-interior tests, and the flat-face analysis
	/// the tangent generator runs on.
	/// </content>
	public sealed partial class ManifoldImpl
	{
		/// <summary>
		/// The stored normal for a halfedge's property vertex, in world frame.
		/// </summary>
		/// <param name="halfedge">The halfedge whose property vertex to read.</param>
		/// <param name="normalIdx">The first of the three property slots holding the normal.</param>
		/// <returns>The world-frame normal.</returns>
		public Vec3 GetNormal(int halfedge, int normalIdx)
		{
			int prop = this.Halfedge[halfedge].PropVert;
			int baseIdx = (prop * this.NumProp) + normalIdx;
			Vec3 normal = new Vec3(
				this.Properties[baseIdx],
				this.Properties[baseIdx + 1],
				this.Properties[baseIdx + 2]);

			// Per #1718: hasNormals=true means CalculateNormals (or a flagged
			// round-trip) wrote world-frame values, kept world-frame through
			// Transform/Compose — return them directly. Without the flag, treat
			// the slot as per-mesh frame and re-rotate to world (legacy contract
			// for hand-built MeshGL inputs that don't set the bit).
			int meshId = this.MeshRelation.TriRef[halfedge / 3].MeshId;
			if (this.MeshRelation.MeshIdTransform.TryGetValue(meshId, out Relation rel) && !rel.HasNormals)
			{
				return rel.GetNormalTransform() * normal;
			}

			return normal;
		}

		/// <summary>
		/// The tangent for <paramref name="halfedge"/> implied by a vertex
		/// <paramref name="normal"/>: circular in the plane perpendicular to the normal.
		/// </summary>
		/// <param name="normal">The vertex normal at the halfedge's start.</param>
		/// <param name="halfedge">The halfedge to build a tangent for.</param>
		/// <returns>The tangent as xyz plus weight.</returns>
		public Vec4 TangentFromNormal(Vec3 normal, int halfedge)
		{
			Halfedge edge = this.Halfedge[halfedge];
			Vec3 edgeVec = this.VertPos[edge.EndVert] - this.VertPos[edge.StartVert];
			Vec3 edgeNormal = this.FaceNormal[halfedge / 3] + this.FaceNormal[edge.PairedHalfedge / 3];

			// Per C++ #1671 (More smoothing fixes): pick the bi-tangent from the
			// edge pseudo-normal or the supplied normal depending on their
			// relative orientation, then cross with the normal. This is more
			// numerically robust than the old single cross-of-cross form.
			Vec3 biTangent = Dot(normal, edgeNormal) < 0.0
				? Cross(edgeNormal, edgeVec)
				: Cross(normal, edgeVec);
			return Smoothing.CircularTangent(Cross(biTangent, normal), edgeVec);
		}

		/// <summary>
		/// Whether this halfedge is the diagonal inside a quad — either because the
		/// tangents already say so, or because its two triangles and their other
		/// neighbours say so topologically.
		/// </summary>
		/// <param name="halfedge">The halfedge to test.</param>
		/// <returns>True when the halfedge is a quad's interior diagonal.</returns>
		public bool IsInsideQuad(int halfedge)
		{
			if (this.HalfedgeTangent.Count != 0)
			{
				return this.HalfedgeTangent[halfedge].W < 0.0;
			}

			int tri = halfedge / 3;
			TriRef refTri = this.MeshRelation.TriRef[tri];
			int pair = this.Halfedge[halfedge].PairedHalfedge;
			int pairTri = pair / 3;
			TriRef pairRef = this.MeshRelation.TriRef[pairTri];
			if (!refTri.SameFace(pairRef))
			{
				return false;
			}

			bool SameFace(int edgeIdx, TriRef reference)
			{
				int p = this.Halfedge[edgeIdx].PairedHalfedge / 3;
				return reference.SameFace(this.MeshRelation.TriRef[p]);
			}

			int neighbor = NextHalfedge(halfedge);
			if (SameFace(neighbor, refTri))
			{
				return false;
			}

			neighbor = NextHalfedge(neighbor);
			if (SameFace(neighbor, refTri))
			{
				return false;
			}

			neighbor = NextHalfedge(pair);
			if (SameFace(neighbor, pairRef))
			{
				return false;
			}

			neighbor = NextHalfedge(neighbor);
			if (SameFace(neighbor, pairRef))
			{
				return false;
			}

			return true;
		}

		/// <summary>
		/// Whether the tangent array already marks this halfedge as a quad interior.
		/// </summary>
		/// <param name="halfedge">The halfedge to test.</param>
		/// <returns>True when the tangent's w is exactly the kInsideQuad marker.</returns>
		public bool IsMarkedInsideQuad(int halfedge)
		{
			// Check for kInsideQuad == -1.0 exactly (distinct from kMissingNormal == -3.0)
			return this.HalfedgeTangent.Count != 0 && this.HalfedgeTangent[halfedge].W == -1.0;
		}

		/// <summary>
		/// Remaps sharpened-edge halfedge indices from the original faces onto this
		/// impl's current triangulation.
		/// </summary>
		/// <param name="sharpenedEdges">The smoothness entries to remap.</param>
		/// <returns>A remapped copy; entries with no mapping are returned unchanged.</returns>
		public List<Smoothness> UpdateSharpenedEdges(IReadOnlyList<Smoothness> sharpenedEdges)
		{
			// Probe-only map: built once, then only looked up, so a Dictionary is safe
			// (iteration order never reaches the output). A negative face_id would key
			// this at a negative int here and at a huge usize in the Rust; both are
			// unreachable by the non-negative halfedge indices looked up below, so the
			// two ports agree without needing the cast.
			Dictionary<int, int> oldHalfedgeToNew = new Dictionary<int, int>();
			for (int tri = 0; tri < this.NumTri(); tri++)
			{
				int oldTri = this.MeshRelation.TriRef[tri].FaceId;
				for (int i = 0; i < 3; i++)
				{
					oldHalfedgeToNew[(3 * oldTri) + i] = (3 * tri) + i;
				}
			}

			List<Smoothness> outList = new List<Smoothness>(sharpenedEdges);
			for (int i = 0; i < outList.Count; i++)
			{
				Smoothness edge = outList[i];
				if (oldHalfedgeToNew.TryGetValue(edge.Halfedge, out int newEdge))
				{
					edge.Halfedge = newEdge;
					outList[i] = edge;
				}
			}

			return outList;
		}

		/// <summary>
		/// Which triangles belong to a flat (multi-triangle coplanar) face — those with
		/// more than one same-face neighbour, plus those neighbours.
		/// </summary>
		/// <returns>One flag per triangle.</returns>
		public List<bool> FlatFaces()
		{
			int numTri = this.NumTri();
			List<bool> triIsFlatFace = new List<bool>(numTri);
			for (int i = 0; i < numTri; i++)
			{
				triIsFlatFace.Add(false);
			}

			for (int tri = 0; tri < numTri; tri++)
			{
				TriRef reference = this.MeshRelation.TriRef[tri];
				int faceNeighbors = 0;
				int[] faceTris = new int[] { -1, -1, -1 };
				for (int j = 0; j < 3; j++)
				{
					int neighborTri = this.Halfedge[(3 * tri) + j].PairedHalfedge / 3;
					TriRef jRef = this.MeshRelation.TriRef[neighborTri];
					if (jRef.SameFace(reference))
					{
						faceNeighbors += 1;
						faceTris[j] = neighborTri;
					}
				}

				if (faceNeighbors > 1)
				{
					triIsFlatFace[tri] = true;
					for (int j = 0; j < 3; j++)
					{
						if (faceTris[j] >= 0)
						{
							triIsFlatFace[faceTris[j]] = true;
						}
					}
				}
			}

			return triIsFlatFace;
		}

		/// <summary>
		/// For each vertex, the single flat-face triangle it belongs to: -1 for none,
		/// -2 for more than one.
		/// </summary>
		/// <param name="flatFaces">The per-triangle flags from <see cref="FlatFaces"/>.</param>
		/// <returns>One triangle index (or sentinel) per vertex.</returns>
		public List<int> VertFlatFace(IReadOnlyList<bool> flatFaces)
		{
			int numVert = this.NumVert();
			List<int> vertFlatFace = new List<int>(numVert);
			TriRef[] vertRef = new TriRef[numVert];
			for (int i = 0; i < numVert; i++)
			{
				vertFlatFace.Add(-1);
				vertRef[i] = default;
			}

			for (int tri = 0; tri < this.NumTri(); tri++)
			{
				if (flatFaces[tri])
				{
					for (int j = 0; j < 3; j++)
					{
						int vert = this.Halfedge[(3 * tri) + j].StartVert;
						if (vertRef[vert].SameFace(this.MeshRelation.TriRef[tri]))
						{
							continue;
						}

						vertRef[vert] = this.MeshRelation.TriRef[tri];
						vertFlatFace[vert] = vertFlatFace[vert] == -1 ? tri : -2;
					}
				}
			}

			return vertFlatFace;
		}
	}
}
