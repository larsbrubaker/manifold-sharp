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

// Smoothing.SetNormals.cs — SetNormals, split out of Smoothing.cs for the
// 800-line cap. The module header for the smoothing.rs port, including the
// audit of the three orbit walks in this one method, is in Smoothing.cs.
//
// ── Property vertices are rewritten under two strides at once ────────────────
// The one thing to hold in mind reading this: `orig_props` is the *old* buffer
// at the *old* stride (old_num_prop values per property vertex) and
// `new_properties` is the new buffer at the new stride (num_prop). Every copy
// below therefore computes two different starts for the same property vertex,
// and the guard `src_start + p < orig_props.len()` is what makes a mesh whose
// properties buffer is shorter than num_prop_vert * old_num_prop survive
// instead of indexing off the end.

using System.Runtime.InteropServices;

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <content>
	/// <c>SetNormals</c> — the second half of the <c>impl ManifoldImpl</c> block in
	/// <c>smoothing.rs</c>.
	/// </content>
	public sealed partial class ManifoldImpl
	{
		/// <summary>
		/// Port of C++ Manifold::Impl::SetNormals()
		/// Fills in vertex properties with unshared normals across edges bent
		/// more than minSharpAngle (in degrees).
		/// </summary>
		/// <param name="normalIdx">The first of three property slots to write the normal into.</param>
		/// <param name="minSharpAngle">The dihedral, in degrees, above which an edge splits normals.</param>
		public void SetNormals(int normalIdx, double minSharpAngle)
		{
			if (this.IsEmpty() || normalIdx < 0)
			{
				return;
			}

			// Clamp to avoid treating nearly-coplanar faces as sharp due to
			// floating point noise in the dihedral computation (~1e-6 degrees).
			minSharpAngle = MaxF64(minSharpAngle, Smoothing.KMinSharpAngle);

			int oldNumProp = this.NumProp;

			// Count sharp edges per vertex. Per C++ #1724 (Fix CalculateNormals),
			// SetNormals no longer special-cases flat faces — an edge is sharp iff
			// its dihedral exceeds min_sharp_angle. `AngleBetween` matches C++'s
			// AngleBetween helper (acos with +/-1 clamping), the #1634 form.
			int[] vertNumSharp = new int[this.NumVert()];
			for (int e = 0; e < this.Halfedge.Count; e++)
			{
				if (!this.Halfedge[e].IsForward())
				{
					continue;
				}

				int pair = this.Halfedge[e].PairedHalfedge;
				int tri1 = e / 3;
				int tri2 = pair / 3;
				double dihedral = Smoothing.ToDegrees(
					Smoothing.AngleBetween(this.FaceNormal[tri1], this.FaceNormal[tri2]));
				if (dihedral > minSharpAngle)
				{
					vertNumSharp[this.Halfedge[e].StartVert] += 1;
					vertNumSharp[this.Halfedge[e].EndVert] += 1;
				}
			}

			// Expand properties to accommodate normals.
			// orig_props: old data at old stride (old_num_prop per vert)
			// new_properties: output buffer at new stride (num_prop per vert)
			int numProp = Math.Max(oldNumProp, normalIdx + 3);
			int numPropVert = this.NumPropVert();

			// Rust `std::mem::take`: origProps takes the old buffer and the field is left
			// empty. NumPropVert above had to run first — it reads both.
			List<double> origProps = this.Properties;
			this.Properties = new List<double>();
			List<double> newProperties = new List<double>(numProp * numPropVert);
			for (int i = 0; i < numProp * numPropVert; i++)
			{
				newProperties.Add(0.0);
			}

			this.NumProp = numProp;

			// Save old prop assignments and reset
			int[] oldHalfedgeProp = new int[this.Halfedge.Count];
			for (int i = 0; i < this.Halfedge.Count; i++)
			{
				oldHalfedgeProp[i] = this.Halfedge[i].PropVert;
			}

			// Nothing below appends to Halfedge, so one span for the whole method is
			// safe — this is the `iter_mut()` of the Rust, not EdgeOp's append-while-
			// walking case.
			Span<Halfedge> halfedges = CollectionsMarshal.AsSpan(this.Halfedge);
			for (int i = 0; i < halfedges.Length; i++)
			{
				halfedges[i].PropVert = -1;
			}

			// Build per-mesh-id inverse normal transform cache.
			// Matches C++ SetNormals which applies GetInverseNormalTransform() to each vertex's normal
			// before writing to properties, so that GetNormal (which applies GetNormalTransform()) can
			// correctly reconstruct the world-space normal.
			//
			// Probe-only map: filled and looked up, never iterated, so a Dictionary is
			// safe here (CLAUDE.md's rustc-hash replacement rule).
			Dictionary<int, Mat3> meshIdToInvTransform = new Dictionary<int, Mat3>();

			int numEdge = this.Halfedge.Count;
			for (int startEdge = 0; startEdge < numEdge; startEdge++)
			{
				if (halfedges[startEdge].PropVert >= 0)
				{
					continue;
				}

				int vert = halfedges[startEdge].StartVert;

				// Look up the inverse normal transform for this vertex's mesh.
				int meshId = this.MeshRelation.TriRef[startEdge / 3].MeshId;
				if (!meshIdToInvTransform.TryGetValue(meshId, out Mat3 invTransform))
				{
					invTransform = this.MeshRelation.MeshIdTransform.TryGetValue(meshId, out Relation rel)
						? rel.GetInverseNormalTransform()
						: Mat3.Identity();
					meshIdToInvTransform[meshId] = invTransform;
				}

				if (vertNumSharp[vert] < 2)
				{
					// Vertex has single normal. Per #1724 this is always the vertex
					// pseudo-normal (no flat-face substitution).
					Vec3 worldNormal = this.VertNormal[vert];

					// Per #1718: slot 0 (standard) stores world-frame directly — the
					// eager-transform contract keeps it in sync with vert_pos /
					// face_normal. Legacy non-zero idx stores per-mesh frame, which
					// GetMeshGL's run transform recovers to world on export.
					Vec3 normal = normalIdx == 0 ? worldNormal : invTransform * worldNormal;

					int lastProp = -1;

					// ORBIT WALK: ForVert traversal, visiting start_edge FIRST and then
					// stepping — the C++ ForVert(halfedge, func) shape. The last_prop
					// dedup makes the order observable, so this is not the steps-first
					// walk the assignment loop below uses.
					int current = startEdge;
					while (true)
					{
						int prop = oldHalfedgeProp[current];
						halfedges[current].PropVert = prop;
						if (prop != lastProp)
						{
							lastProp = prop;

							// Copy old properties using old stride, write into new stride
							int dstStart = prop * numProp;
							int srcStart = prop * oldNumProp;
							for (int p = 0; p < Math.Min(oldNumProp, numProp); p++)
							{
								if (srcStart + p < origProps.Count)
								{
									newProperties[dstStart + p] = origProps[srcStart + p];
								}
							}

							// Set normal
							newProperties[(prop * numProp) + normalIdx] = normal.X;
							newProperties[(prop * numProp) + normalIdx + 1] = normal.Y;
							newProperties[(prop * numProp) + normalIdx + 2] = normal.Z;
						}

						current = Types.NextHalfedge(halfedges[current].PairedHalfedge);
						if (current == startEdge)
						{
							break;
						}
					}
				}
				else
				{
					// Vertex has multiple normals
					Vec3 centerPos = this.VertPos[vert];
					List<int> group = new List<int>();
					List<Vec3> normals = new List<Vec3>();

					// ORBIT WALK: find a sharp edge to start on. Steps first and breaks on
					// the dihedral, so it either stops at a sharp edge or wraps back to
					// start_edge having found none.
					int current = startEdge;
					int prevFace = current / 3;
					while (true)
					{
						int next = Types.NextHalfedge(halfedges[current].PairedHalfedge);
						int face = next / 3;
						double dihedral = Smoothing.ToDegrees(
							Smoothing.AngleBetween(this.FaceNormal[face], this.FaceNormal[prevFace]));
						if (dihedral > minSharpAngle)
						{
							break;
						}

						current = next;
						prevFace = face;
						if (current == startEdge)
						{
							break;
						}
					}

					int endEdge = current;

					// Calculate pseudo-normals between each sharp edge.
					// Mirrors C++ ForVert<FaceEdge>(endEdge, transform, binaryOp):
					//   here = transform(endEdge); current = endEdge;
					//   do { next = transform(NextHalfedge(current.paired));
					//        binaryOp(current, here, next); here = next; current = next_halfedge;
					//   } while (current != endEdge);
					// normals starts EMPTY — first sharp edge creates group 0.
					// Per #1724 the edge vector is simply the normalized direction
					// from the center vertex to the far end of the halfedge; the
					// old quad/flair-out tangent logic was removed.
					// Reads through the List rather than the span above: a local function
					// may not close over a ref struct, and the two see the same storage.
					Vec3 GetEdgeVec(int he)
					{
						int endV = this.Halfedge[he].EndVert;
						return Smoothing.SafeNormalize(this.VertPos[endV] - centerPos);
					}

					// ORBIT WALK: seeded from end_edge *before* the loop (hereFace/hereEv
					// are end_edge's) and stepping first inside it, so end_edge is both
					// the seed and the terminator — it contributes the `here` side of the
					// first pair and the `next` side of the last. That is what makes
					// `group` come out in the same order the assignment walk below
					// consumes it.
					int hereFace = endEdge / 3;
					Vec3 hereEv = GetEdgeVec(endEdge);
					current = endEdge;
					while (true)
					{
						int nextHe = Types.NextHalfedge(halfedges[current].PairedHalfedge);
						int nextFace = nextHe / 3;
						Vec3 nextEv = GetEdgeVec(nextHe);

						// Check for sharp edge between here and next.
						double dihedral = Smoothing.ToDegrees(
							Smoothing.AngleBetween(this.FaceNormal[hereFace], this.FaceNormal[nextFace]));
						if (dihedral > minSharpAngle)
						{
							normals.Add(Vec3.Splat(0.0));
						}

						// The Rust is `group.push(normals.len() - 1)`, and on an empty
						// `normals` that subtraction wraps to usize::MAX in a release build
						// (a debug build panics). The C# int goes to -1 instead. Both are
						// then rejected by the `g < normals.len()` guards in the assignment
						// loop below — but only because those are transcribed unsigned;
						// see the note there. Unreachable in practice: this branch runs
						// only when vert_num_sharp[vert] >= 2, which means at least one
						// sharp edge is incident to the vertex, so the sharp-edge search
						// above always lands on one and the first iteration here pushes
						// group 0.
						group.Add(normals.Count - 1);

						// Accumulate angle-weighted normal (C++: cross(next.edgeVec, here.edgeVec)).
						if (double.IsFinite(nextEv.X))
						{
							Vec3 c = Cross(nextEv, hereEv);
							double angle = Smoothing.AngleBetween(hereEv, nextEv);
							normals[normals.Count - 1] = normals[normals.Count - 1] + (Smoothing.SafeNormalize(c) * angle);
						}
						else
						{
							nextEv = hereEv;
						}

						hereFace = nextFace;
						hereEv = nextEv;
						current = nextHe;
						if (current == endEdge)
						{
							break;
						}
					}

					// Per #1724: transform into the storage frame first, then
					// normalize. Per #1718: only the legacy non-zero idx applies the
					// inv_transform; slot 0 stores world-frame directly.
					for (int n = 0; n < normals.Count; n++)
					{
						normals[n] = normalIdx == 0
							? Smoothing.SafeNormalize(normals[n])
							: Smoothing.SafeNormalize(invTransform * normals[n]);
					}

					// Assign property vertices.
					// ORBIT WALK. Mirrors C++ ForVert(endEdge, func) which advances BEFORE visiting:
					//   do { current = NextHalfedge(paired); func(current); } while (current != endEdge)
					// So endEdge itself is visited LAST (gets group[N-1]), not first.
					int lastGroup = 0;
					int lastProp = -1;
					int newProp = -1;
					int idx = 0;
					current = Types.NextHalfedge(halfedges[endEdge].PairedHalfedge);
					while (true)
					{
						int prop = oldHalfedgeProp[current];
						int g = idx < group.Count ? group[idx] : 0;

						if (g != lastGroup && g != 0 && prop == lastProp)
						{
							// Split property vertex
							lastGroup = g;
							newProp = newProperties.Count / numProp;
							for (int z = 0; z < numProp; z++)
							{
								newProperties.Add(0.0);
							}

							int srcStart = prop * oldNumProp;
							for (int p = 0; p < Math.Min(oldNumProp, numProp); p++)
							{
								if (srcStart + p < origProps.Count)
								{
									newProperties[(newProp * numProp) + p] = origProps[srcStart + p];
								}
							}

							// The Rust guard is `g < normals.len()` on a usize, which also
							// rejects the wrapped usize::MAX the empty-`normals` case above
							// can produce; a signed `<` would accept its -1 counterpart and
							// index out of bounds, so it is transcribed unsigned (the
							// pattern the FaceOp.cs header names).
							if ((uint)g < (uint)normals.Count)
							{
								newProperties[(newProp * numProp) + normalIdx] = normals[g].X;
								newProperties[(newProp * numProp) + normalIdx + 1] = normals[g].Y;
								newProperties[(newProp * numProp) + normalIdx + 2] = normals[g].Z;
							}
						}
						else if (prop != lastProp)
						{
							// Update property vertex
							lastProp = prop;
							newProp = prop;
							int dstStart = prop * numProp;
							int srcStart = prop * oldNumProp;
							for (int p = 0; p < Math.Min(oldNumProp, numProp); p++)
							{
								if (srcStart + p < origProps.Count)
								{
									newProperties[dstStart + p] = origProps[srcStart + p];
								}
							}

							// Unsigned for the same reason as the guard in the split branch.
							if ((uint)g < (uint)normals.Count)
							{
								newProperties[(prop * numProp) + normalIdx] = normals[g].X;
								newProperties[(prop * numProp) + normalIdx + 1] = normals[g].Y;
								newProperties[(prop * numProp) + normalIdx + 2] = normals[g].Z;
							}
						}

						halfedges[current].PropVert = newProp;
						idx += 1;

						int nextCurrent = Types.NextHalfedge(halfedges[current].PairedHalfedge);

						// Stop after visiting end_edge (C++ stops when current == halfedge)
						if (current == endEdge)
						{
							break;
						}

						current = nextCurrent;
					}
				}
			}

			this.Properties = newProperties;
		}
	}
}
