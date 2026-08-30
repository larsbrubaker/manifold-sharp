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

// Manifold.MeshGL.Export.cs — `get_mesh_gl_impl` and `normal_transform_for_run`
// from manifold_meshgl.rs, split from the import half for the 800-line cap. The
// module header and the "one body, two adapters" argument are in
// Manifold.MeshGL.cs and MeshGLAccess.cs.
//
// Two things here are load-bearing and easy to lose in translation:
//
//   * The run sort is STABLE. `tri_new2old.sort_by(...)` in the Rust is
//     `slice::sort_by`, which is documented stable, and triangles that share an
//     (originalID, meshID) key keep their kernel order — which is the order the
//     rest of the port's determinism rests on. `Array.Sort` is an unstable
//     introsort and is not a substitute; this uses LINQ OrderBy/ThenBy, per the
//     stable-sort rule in CLAUDE.md.
//   * The normal slot is READ BACK after being written. In the f32
//     instantiation that is a real f32 round-trip before normalizing, exactly as
//     the C++ float template does, so the read goes through the sink and not
//     through a retained f64 local.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <content>
	/// The MeshGL export body.
	/// </content>
	public sealed partial class Manifold
	{
		/// <summary>How the import treats non-manifold connectivity.</summary>
		private enum ImportMode
		{
			/// <summary>Reject with <see cref="Error.NotManifold"/>.</summary>
			Strict,

			/// <summary>
			/// Keep geometrically closed + orientable geometry as a triangle soup
			/// (<c>ManifoldImpl.IsSoup</c>) for the robust boolean engine; reject only with
			/// <see cref="Error.NotClosed"/> when even that fails.
			/// </summary>
			AllowSoup,
		}

		/// <summary>
		/// Per-run normal transform for the legacy export path (slot interpreted as
		/// per-mesh-frame normals on a run without the hasNormals bit).
		/// </summary>
		/// <remarks>
		/// Reconstructs <c>NormalTransform(runTransform) * (backside ? -1 : 1)</c> from the
		/// flat column-major run_transform buffer. Matches C++ GetMeshGLImpl. (#1718)
		/// </remarks>
		/// <param name="mesh">The output mesh being written, read back for its transforms.</param>
		/// <param name="run">The run index.</param>
		/// <param name="flags">That run's flags.</param>
		/// <returns>The normal transform.</returns>
		private static Mat3 NormalTransformForRun(IMeshGLAccess mesh, int run, byte flags)
		{
			int b = 12 * run;

			// run_transform[b + 3*col + row] = transform[col][row]; cols 0..2 = 3x3 part.
			Vec3 Col(int j) => new Vec3(
				mesh.RunTransform(b + (3 * j)),
				mesh.RunTransform(b + (3 * j) + 1),
				mesh.RunTransform(b + (3 * j) + 2));

			Mat3 m3 = Mat3.FromCols(Col(0), Col(1), Col(2));
			double sign = (flags & 1) != 0 ? -1.0 : 1.0;

			// NormalTransform(M) = inverse(transpose(M)).
			return m3.Transpose().Inverse() * sign;
		}

		private void GetMeshGLImpl(IMeshGLAccess outMesh, int normalIdx)
		{
			// Per #1718: GetMeshGL(-1) auto-substitutes slot 0 when CalculateNormals
			// recorded world-frame normals on every meshID. A non-negative idx is the
			// legacy interface; >= 0 means "interpret this extra-prop slot as normals
			// and normalize it on export (transforming legacy per-mesh-frame runs)".
			if (normalIdx < 0 && this.imp.AllHaveNormals())
			{
				normalIdx = 0;
			}

			int extraProp = this.imp.NumProp;
			bool updateNormals = normalIdx >= 0 && normalIdx + 3 <= extraProp;
			int numTri = this.imp.NumTri();
			outMesh.SetNumPropFromUSize(3 + extraProp);

			// C++: for float output only, floor tolerance at float_epsilon *
			// bBox.Scale() to avoid catastrophic cancellation in RelatedGL checks.
			// The f64 output exports the kernel tolerance unchanged.
			double tolerance = this.imp.Tolerance;
			if (outMesh.IsSinglePrecision)
			{
				// Rust `f32::EPSILON as f64` — 2^-23, the gap between 1.0f and the next
				// float. C# `float.Epsilon` is the smallest subnormal and is wrong here.
				const double F32Epsilon = 1.1920928955078125E-07;
				tolerance = Math.Max(tolerance, F32Epsilon * this.imp.Bbox.Scale());
			}

			outMesh.SetTolerance(tolerance);

			if (this.imp.HalfedgeTangent.Count != 0)
			{
				foreach (Vec4 t in this.imp.HalfedgeTangent)
				{
					outMesh.AddHalfedgeTangent(t.X);
					outMesh.AddHalfedgeTangent(t.Y);
					outMesh.AddHalfedgeTangent(t.Z);
					outMesh.AddHalfedgeTangent(t.W);
				}
			}

			// Sort triangles by (originalID, meshID) for run grouping
			bool isOriginal = this.imp.MeshRelation.OriginalId >= 0;
			List<TriRef> triRef = this.imp.MeshRelation.TriRef;
			int[] triNew2Old;
			if (!isOriginal && triRef.Count != 0)
			{
				// Stable, per the file header: OrderBy/ThenBy, never Array.Sort.
				triNew2Old = Enumerable.Range(0, numTri)
					.OrderBy(a => triRef[a].OriginalId)
					.ThenBy(a => triRef[a].MeshId)
					.ToArray();
			}
			else
			{
				triNew2Old = new int[numTri];
				for (int i = 0; i < numTri; i++)
				{
					triNew2Old[i] = i;
				}
			}

			outMesh.ResizeTriVerts(3 * numTri);
			outMesh.ResizeFaceId(numTri);
			SortedDictionary<int, Relation> meshIdTransform =
				new SortedDictionary<int, Relation>(this.imp.MeshRelation.MeshIdTransform);
			int lastMeshId = -1;

			// run index that each output triangle belongs to (for per-vertex normal
			// export below). -1 until the first run is pushed.
			int[] triRun = new int[numTri];
			int currentRun = -1;

			for (int newTri = 0; newTri < numTri; newTri++)
			{
				int oldTri = triNew2Old[newTri];
				if (oldTri < triRef.Count)
				{
					TriRef r = triRef[oldTri];
					outMesh.SetFaceIdFromI32(newTri, r.FaceId >= 0 ? r.FaceId : r.CoplanarId);
				}

				for (int i = 0; i < 3; i++)
				{
					Halfedge he = this.imp.Halfedge[(3 * oldTri) + i];

					// First pass: set to geometric vertex (startVert). When numProp > 0,
					// a second pass below will replace these with property-vertex indices.
					outMesh.SetTriVertFromI32((3 * newTri) + i, he.StartVert);
				}

				int meshId = oldTri < triRef.Count ? triRef[oldTri].MeshId : 0;
				if (meshId != lastMeshId)
				{
					Relation rel;
					if (meshIdTransform.TryGetValue(meshId, out Relation found))
					{
						rel = found;
						meshIdTransform.Remove(meshId);
					}
					else
					{
						rel = new Relation();
					}

					outMesh.AddRunIndexFromUSize(3 * newTri);
					outMesh.RunOriginalId.Add((uint)Math.Max(rel.OriginalId, 0));
					outMesh.RunFlags.Add(RunFlagsFor(rel));

					// C++: runTransform only emitted for non-original manifolds
					if (!isOriginal)
					{
						for (int col = 0; col < 4; col++)
						{
							for (int row = 0; row < 3; row++)
							{
								outMesh.AddRunTransform(rel.Transform[col][row]);
							}
						}
					}

					lastMeshId = meshId;
					currentRun++;
				}

				triRun[newTri] = Math.Max(currentRun, 0);
			}

			// Add runs for originals that contributed no tris
			foreach (KeyValuePair<int, Relation> entry in meshIdTransform)
			{
				Relation rel = entry.Value;
				outMesh.AddRunIndexFromUSize(3 * numTri);
				outMesh.RunOriginalId.Add((uint)Math.Max(rel.OriginalId, 0));
				outMesh.RunFlags.Add(RunFlagsFor(rel));
				if (!isOriginal)
				{
					for (int col = 0; col < 4; col++)
					{
						for (int row = 0; row < 3; row++)
						{
							outMesh.AddRunTransform(rel.Transform[col][row]);
						}
					}
				}
			}

			outMesh.AddRunIndexFromUSize(3 * numTri);

			int numGeomVert = this.imp.NumVert();
			int numProp = this.imp.NumProp;
			int totalProp = 3 + numProp; // xyz + extra

			if (numProp == 0)
			{
				// No extra properties: positions only, indexed by geometric vertex
				outMesh.ResizeVertProperties(3 * numGeomVert);
				for (int i = 0; i < numGeomVert; i++)
				{
					Vec3 v = this.imp.VertPos[i];
					outMesh.SetVertProperty(3 * i, v.X);
					outMesh.SetVertProperty((3 * i) + 1, v.Y);
					outMesh.SetVertProperty((3 * i) + 2, v.Z);
				}

				return;
			}

			// When properties exist: deduplicate (start_vert, prop_vert) pairs, matching
			// C++ GetMeshGLImpl. Each unique (vert, prop) pair gets its own slot in
			// vert_properties. Merge vectors record which slots share the same geometry.
			//
			// C++: vertPropPair[vert] = list of {prop, idx}; vert2idx[vert] = first idx
			int[] vert2Idx = new int[numGeomVert];
			for (int i = 0; i < numGeomVert; i++)
			{
				vert2Idx[i] = -1;
			}

			List<(int Prop, int Idx)>[] vertPropPairs = new List<(int, int)>[numGeomVert];
			for (int i = 0; i < numGeomVert; i++)
			{
				vertPropPairs[i] = new List<(int, int)>();
			}

			for (int newTri = 0; newTri < numTri; newTri++)
			{
				int oldTri = triNew2Old[newTri];
				for (int i = 0; i < 3; i++)
				{
					Halfedge he = this.imp.Halfedge[(3 * oldTri) + i];
					if (he.StartVert < 0)
					{
						continue;
					}

					int vert = he.StartVert;
					int prop = he.PropVert;

					// Look for existing (vert, prop) pair
					List<(int Prop, int Idx)> pairs = vertPropPairs[vert];
					int foundIdx = -1;
					foreach ((int p, int existing) in pairs)
					{
						if (p == prop)
						{
							foundIdx = existing;
							break;
						}
					}

					int idx;
					if (foundIdx >= 0)
					{
						idx = foundIdx;
					}
					else
					{
						idx = outMesh.VertPropertiesCount / totalProp;

						// Write position
						Vec3 pos = this.imp.VertPos[vert];
						outMesh.AddVertProperty(pos.X);
						outMesh.AddVertProperty(pos.Y);
						outMesh.AddVertProperty(pos.Z);

						// Write extra properties (zeros if prop_vert is invalid)
						if (prop >= 0)
						{
							int basePropIdx = prop * numProp;
							for (int p = 0; p < numProp; p++)
							{
								outMesh.AddVertProperty(this.imp.Properties[basePropIdx + p]);
							}
						}
						else
						{
							for (int p = 0; p < numProp; p++)
							{
								outMesh.AddVertPropertyDefault();
							}
						}

						// Per #1718: normalize the requested normal slot on export. Runs
						// that already carry world-frame normals (hasNormals bit) just
						// get normalized; legacy runs without the bit are interpreted as
						// per-mesh-frame and rotated to world via the inverse-frame
						// transform first.
						if (updateNormals)
						{
							int ni = normalIdx;
							int off = outMesh.VertPropertiesCount - numProp + ni;

							// Read back the just-written values: in the f32
							// instantiation this round-trips through f32 exactly as
							// the C++ float template does before normalizing.
							Vec3 n = new Vec3(
								outMesh.VertProperty(off),
								outMesh.VertProperty(off + 1),
								outMesh.VertProperty(off + 2));
							int run = triRun[newTri];
							bool runHasN = !isOriginal && (outMesh.RunFlags[run] & 2) != 0;
							if (!isOriginal && !runHasN)
							{
								n = NormalTransformForRun(outMesh, run, outMesh.RunFlags[run]) * n;
							}

							n = Smoothing.SafeNormalize(n);
							outMesh.SetVertProperty(off, n.X);
							outMesh.SetVertProperty(off + 1, n.Y);
							outMesh.SetVertProperty(off + 2, n.Z);
						}

						vertPropPairs[vert].Add((prop, idx));

						// First slot for this geometric vertex is the canonical merge target.
						// Additional slots get merge entries so from_mesh_gl knows they
						// are coincident with the first slot.
						if (vert2Idx[vert] == -1)
						{
							vert2Idx[vert] = idx;
						}
						else
						{
							outMesh.AddMergeFromVertFromUSize(idx);
							outMesh.AddMergeToVertFromI32(vert2Idx[vert]);
						}
					}

					outMesh.SetTriVertFromUSize((3 * newTri) + i, idx);
				}
			}
		}

		/// <summary>
		/// run_flags layout (#1718): bit 0 = backside, bit 1 = hasNormals (slot 0..2 of the
		/// extra properties is world-frame normals; consumers skip re-applying
		/// run_transform to it).
		/// </summary>
		/// <param name="rel">The run's relation.</param>
		/// <returns>The flag byte.</returns>
		private static byte RunFlagsFor(Relation rel)
		{
			return (byte)((rel.BackSide ? 1 : 0) | (rel.HasNormals ? 2 : 0));
		}
	}
}
