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

// BooleanResultAssemble.Properties.cs — CreateProperties, the barycentric
// interpolation pass of boolean_result_assemble.rs. See BooleanResultAssemble.cs
// for the module header.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <content>
	/// CreateProperties — barycentric interpolation of properties.
	/// </content>
	public static partial class BooleanResultAssemble
	{
		// -------------------------------------------------------------------
		// CreateProperties -- barycentric interpolation of properties
		// -------------------------------------------------------------------

		internal static void CreateProperties(
			ManifoldImpl outR,
			ManifoldImpl inP,
			ManifoldImpl inQ,
			bool invertQ)
		{
			int numPropP = inP.NumProp;
			int numPropQ = inQ.NumProp;
			int numProp = Math.Max(numPropP, numPropQ);
			outR.NumProp = numProp;
			if (numProp == 0)
			{
				return;
			}

			int numTri = outR.NumTri();

			// Compute barycentric coordinates for each output halfedge
			Vec3[] bary = new Vec3[outR.Halfedge.Count];
			for (int i = 0; i < bary.Length; i++)
			{
				bary[i] = Vec3.Splat(0.0);
			}

			for (int tri = 0; tri < numTri; tri++)
			{
				TriRef refPq = outR.MeshRelation.TriRef[tri];
				if (outR.Halfedge[3 * tri].StartVert < 0)
				{
					continue;
				}

				int triPq = refPq.FaceId;
				bool pq = refPq.MeshId == 0;
				List<Vec3> vertPos = pq ? inP.VertPos : inQ.VertPos;
				List<Halfedge> halfedge = pq ? inP.Halfedge : inQ.Halfedge;

				// Rust compares `3 * tri_pq + 2 >= halfedge.len()` on a usize, which also
				// rejects a negative face_id (it wraps to a huge value); the unsigned
				// comparison here reproduces that.
				if ((uint)((3 * triPq) + 2) >= (uint)halfedge.Count)
				{
					continue;
				}

				Vec3[] triPos =
				{
					vertPos[halfedge[3 * triPq].StartVert],
					vertPos[halfedge[(3 * triPq) + 1].StartVert],
					vertPos[halfedge[(3 * triPq) + 2].StartVert],
				};

				for (int i = 0; i < 3; i++)
				{
					int vert = outR.Halfedge[(3 * tri) + i].StartVert;
					if (vert >= 0 && vert < outR.VertPos.Count)
					{
						bary[(3 * tri) + i] =
							FaceOp.GetBarycentric(outR.VertPos[vert], triPos, outR.Epsilon);
					}
				}
			}

			// Build properties with deduplication (matches C++ CreateProperties)
			outR.Properties.Clear();
			outR.Properties.Capacity = Math.Max(outR.Properties.Capacity, outR.NumVert() * numProp);
			int idx = 0;

			// Property vertex deduplication structures
			int idMissProp = outR.NumVert();

			// propIdx: indexed by output vertex; bins hold ([pq, key_z, key_w], prop_idx)
			List<(IVec3 Key, int Idx)>[] propIdx = new List<(IVec3, int)>[outR.NumVert() + 1];
			for (int i = 0; i < propIdx.Length; i++)
			{
				propIdx[i] = new List<(IVec3, int)>();
			}

			// propMissIdx: [0] for mesh Q, [1] for mesh P -- indexed by source propVert
			int[][] propMissIdx =
			{
				CreateFilled(inQ.NumPropVert(), -1),
				CreateFilled(inP.NumPropVert(), -1),
			};

			for (int tri = 0; tri < numTri; tri++)
			{
				if (outR.Halfedge[3 * tri].StartVert < 0)
				{
					continue;
				}

				TriRef refPq = outR.MeshRelation.TriRef[tri];
				bool pq = refPq.MeshId == 0;
				int pqFlag = pq ? 0 : 1;
				int oldNumProp = pq ? numPropP : numPropQ;
				List<double> properties = pq ? inP.Properties : inQ.Properties;
				List<Halfedge> halfedge = pq ? inP.Halfedge : inQ.Halfedge;

				// Per #1718: for Subtract, Q's triangles are flipped in the result, so
				// Q's world-frame vertex normals (slot 0..2 when hasNormals) need a
				// sign flip to point outward from the result's solid (into the cavity).
				// Check is per-source-triangle — in_q may itself be a mixed Boolean
				// result with only some meshIDs carrying normals.
				bool negateNormals =
					!pq && invertQ && oldNumProp >= 3 && inQ.TriHasNormals(refPq.FaceId);

				for (int i = 0; i < 3; i++)
				{
					int vert = outR.Halfedge[(3 * tri) + i].StartVert;
					Vec3 uvw = bary[(3 * tri) + i];

					// Build dedup key: [pq_flag, vert_key, key_z, key_w]
					int[] key = { pqFlag, idMissProp, -1, -1 };
					if (oldNumProp > 0 && (uint)((3 * refPq.FaceId) + 2) < (uint)halfedge.Count)
					{
						int edge = -2;
						for (int j = 0; j < 3; j++)
						{
							if (uvw[j] == 1.0)
							{
								// On a retained vertex
								key[2] = halfedge[(3 * refPq.FaceId) + j].PropVert;
								edge = -1;
								break;
							}

							if (uvw[j] == 0.0)
							{
								edge = j;
							}
						}

						if (edge >= 0)
						{
							// On an edge: both prop verts must match
							int p0 = halfedge[(3 * refPq.FaceId) + Next3(edge)].PropVert;
							int p1 = halfedge[(3 * refPq.FaceId) + Prev3(edge)].PropVert;
							key[1] = vert;
							key[2] = Math.Min(p0, p1);
							key[3] = Math.Max(p0, p1);
						}
						else if (edge == -2)
						{
							// Interior point
							key[1] = vert;
						}
					}

					// Attempt dedup lookup
					bool found = false;
					if (key[1] == idMissProp && key[2] >= 0)
					{
						// Vertex case: use propMissIdx
						int pqIdx = key[0];
						int propKey = key[2];
						if ((uint)pqIdx < 2 && (uint)propKey < (uint)propMissIdx[pqIdx].Length)
						{
							int entry = propMissIdx[pqIdx][propKey];
							if (entry >= 0)
							{
								SetPropVert(outR, (3 * tri) + i, entry);
								found = true;
							}
							else
							{
								propMissIdx[pqIdx][propKey] = idx;
							}
						}
					}
					else
					{
						// Edge/interior case: use propIdx
						int binIdx = key[1];
						if ((uint)binIdx < (uint)propIdx.Length)
						{
							IVec3 searchKey = new IVec3(key[0], key[2], key[3]);
							int hit = -1;
							foreach ((IVec3 k, int v) in propIdx[binIdx])
							{
								if (k == searchKey)
								{
									hit = v;
									break;
								}
							}

							if (hit >= 0)
							{
								SetPropVert(outR, (3 * tri) + i, hit);
								found = true;
							}
							else
							{
								propIdx[binIdx].Add((searchKey, idx));
							}
						}
					}

					if (found)
					{
						continue;
					}

					// No dedup match -- assign new property vertex and interpolate
					SetPropVert(outR, (3 * tri) + i, idx);
					idx += 1;

					for (int p = 0; p < numProp; p++)
					{
						if (p < oldNumProp && (uint)((3 * refPq.FaceId) + 2) < (uint)halfedge.Count)
						{
							double[] oldProps = new double[3];
							for (int j = 0; j < 3; j++)
							{
								int propVert = halfedge[(3 * refPq.FaceId) + j].PropVert;
								if (propVert >= 0)
								{
									int propIdxVal = (oldNumProp * propVert) + p;
									if (propIdxVal < properties.Count)
									{
										oldProps[j] = properties[propIdxVal];
									}
								}
							}

							double val =
								(uvw.X * oldProps[0]) + (uvw.Y * oldProps[1]) + (uvw.Z * oldProps[2]);
							if (negateNormals && p < 3)
							{
								val = -val;
							}

							outR.Properties.Add(val);
						}
						else
						{
							outR.Properties.Add(0.0);
						}
					}
				}
			}
		}

		/// <summary>
		/// The Rust's file-local <c>next3</c>. Kept local rather than routed to
		/// <see cref="Types.Next3"/>, which is the same function written differently — the
		/// Rust has both and this pass uses its own.
		/// </summary>
		private static int Next3(int i)
		{
			return (i + 1) % 3;
		}

		/// <summary>The Rust's file-local <c>prev3</c>.</summary>
		private static int Prev3(int i)
		{
			return (i + 2) % 3;
		}

		private static int[] CreateFilled(int length, int value)
		{
			int[] result = new int[length];
			Array.Fill(result, value);
			return result;
		}

		/// <summary>
		/// <c>out_r.halfedge[h].prop_vert = value</c>. A method because a
		/// <c>List&lt;Halfedge&gt;</c> indexer returns a copy, so the field write has to be
		/// read-modify-written back.
		/// </summary>
		private static void SetPropVert(ManifoldImpl outR, int halfedgeIdx, int value)
		{
			Halfedge he = outR.Halfedge[halfedgeIdx];
			he.PropVert = value;
			outR.Halfedge[halfedgeIdx] = he;
		}
	}
}
