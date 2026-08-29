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

// ManifoldImpl.Shapes.cs — the primitive constructors and Transform half of
// impl_mesh.rs, plus the Mat3x4 composition helper the module ends with. The
// module header lives in ManifoldImpl.cs.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <content>
	/// The three Platonic primitives, the affine transform, and the field-by-field copy.
	/// </content>
	public sealed partial class ManifoldImpl
	{
		/// <summary>Builds a regular tetrahedron, transformed by <paramref name="transform"/>.</summary>
		/// <param name="transform">The transform to apply to the canonical shape.</param>
		/// <returns>The tetrahedron impl.</returns>
		public static ManifoldImpl Tetrahedron(Mat3x4 transform)
		{
			double[][] vertPosRaw = new double[][]
			{
				new double[] { -1.0, -1.0, 1.0 },
				new double[] { -1.0, 1.0, -1.0 },
				new double[] { 1.0, -1.0, -1.0 },
				new double[] { 1.0, 1.0, 1.0 },
			};
			IVec3[] triVerts = new IVec3[]
			{
				new IVec3(2, 0, 1),
				new IVec3(0, 3, 1),
				new IVec3(2, 3, 0),
				new IVec3(3, 2, 1),
			};
			return FromShape(vertPosRaw, triVerts, transform);
		}

		/// <summary>Builds the unit cube, transformed by <paramref name="transform"/>.</summary>
		/// <param name="transform">The transform to apply to the canonical shape.</param>
		/// <returns>The cube impl.</returns>
		public static ManifoldImpl Cube(Mat3x4 transform)
		{
			double[][] vertPosRaw = new double[][]
			{
				new double[] { 0.0, 0.0, 0.0 },
				new double[] { 0.0, 0.0, 1.0 },
				new double[] { 0.0, 1.0, 0.0 },
				new double[] { 0.0, 1.0, 1.0 },
				new double[] { 1.0, 0.0, 0.0 },
				new double[] { 1.0, 0.0, 1.0 },
				new double[] { 1.0, 1.0, 0.0 },
				new double[] { 1.0, 1.0, 1.0 },
			};
			IVec3[] triVerts = new IVec3[]
			{
				new IVec3(1, 0, 4),
				new IVec3(2, 4, 0),
				new IVec3(1, 3, 0),
				new IVec3(3, 1, 5),
				new IVec3(3, 2, 0),
				new IVec3(3, 7, 2),
				new IVec3(5, 4, 6),
				new IVec3(5, 1, 4),
				new IVec3(6, 4, 2),
				new IVec3(7, 6, 2),
				new IVec3(7, 3, 5),
				new IVec3(7, 5, 6),
			};
			return FromShape(vertPosRaw, triVerts, transform);
		}

		/// <summary>Builds a regular octahedron, transformed by <paramref name="transform"/>.</summary>
		/// <param name="transform">The transform to apply to the canonical shape.</param>
		/// <returns>The octahedron impl.</returns>
		public static ManifoldImpl Octahedron(Mat3x4 transform)
		{
			double[][] vertPosRaw = new double[][]
			{
				new double[] { 1.0, 0.0, 0.0 },
				new double[] { -1.0, 0.0, 0.0 },
				new double[] { 0.0, 1.0, 0.0 },
				new double[] { 0.0, -1.0, 0.0 },
				new double[] { 0.0, 0.0, 1.0 },
				new double[] { 0.0, 0.0, -1.0 },
			};
			IVec3[] triVerts = new IVec3[]
			{
				new IVec3(0, 2, 4),
				new IVec3(1, 5, 3),
				new IVec3(2, 1, 4),
				new IVec3(3, 5, 0),
				new IVec3(1, 3, 4),
				new IVec3(0, 5, 2),
				new IVec3(3, 0, 4),
				new IVec3(2, 5, 1),
			};
			return FromShape(vertPosRaw, triVerts, transform);
		}

		// ---------------------------------------------------------------------
		// Transform
		// ---------------------------------------------------------------------

		/// <summary>Apply affine transform, returning a new ManifoldImpl.</summary>
		/// <param name="t">The transform to apply.</param>
		/// <returns>The transformed impl.</returns>
		public ManifoldImpl Transform(Mat3x4 t)
		{
			Mat3x4 identity = Mat3x4.Identity();
			if (t == identity)
			{
				// Clone self — this is a simplified version (full version uses Collider)
				return this.ShallowClone();
			}

			ManifoldImpl result = new ManifoldImpl();
			if (this.Status != Error.NoError)
			{
				result.Status = this.Status;
				return result;
			}

			result.MeshRelation = this.MeshRelation.Clone();

			// Scale epsilon by spectral norm of transform, matching C++:
			// result.epsilon_ *= SpectralNorm(mat3(transform_));
			Mat3 m3ForNorm = Mat3.FromCols(
				new Vec3(t.X.X, t.X.Y, t.X.Z),
				new Vec3(t.Y.X, t.Y.Y, t.Y.Z),
				new Vec3(t.Z.X, t.Z.Y, t.Z.Z));
			result.Epsilon = this.Epsilon * SvdFunctions.SpectralNorm(m3ForNorm);
			result.Tolerance = this.Tolerance;
			result.NumProp = this.NumProp;
			result.Properties = new List<double>(this.Properties);
			result.Bbox = this.Bbox;
			result.Halfedge = new List<Halfedge>(this.Halfedge);
			result.MeshRelation.OriginalId = -1;

			// Soup impls stay soups across transforms; every step below already
			// guards paired_halfedge < 0.
			result.IsSoup = this.IsSoup;

			// The self-intersection cache is deliberately *not* carried across:
			// transformed positions are rounded to f64, so an extreme scale can
			// collapse distinct vertices onto each other and create coincident
			// surface that the source mesh did not have. Re-running the detector
			// costs microseconds; propagating a stale `false` costs correctness.

			// Update mesh transforms. Relation is a struct, so this is read-modify-store —
			// the Rust's `iter_mut()` has no C# equivalent through a dictionary, and
			// writing to `MeshIdTransform[key].Transform` directly is a compile error.
			List<int> relationKeys = new List<int>(result.MeshRelation.MeshIdTransform.Keys);
			foreach (int key in relationKeys)
			{
				Relation rel = result.MeshRelation.MeshIdTransform[key];

				// rel.transform = t * Mat4(rel.transform) — combine transforms
				rel.Transform = Mat3x4MulMat3x4(t, rel.Transform);
				result.MeshRelation.MeshIdTransform[key] = rel;
			}

			// Transform vertex positions
			result.VertPos = new List<Vec3>(this.VertPos.Count);
			foreach (Vec3 v in this.VertPos)
			{
				result.VertPos.Add(t * new Vec4(v.X, v.Y, v.Z, 1.0));
			}

			// Transform normals (using inverse-transpose of 3x3 part)
			Mat3 m3 = Mat3.FromCols(
				new Vec3(t.X.X, t.X.Y, t.X.Z),
				new Vec3(t.Y.X, t.Y.Y, t.Y.Z),
				new Vec3(t.Z.X, t.Z.Y, t.Z.Z));
			Mat3 normalT = m3.Inverse().Transpose();

			result.FaceNormal = new List<Vec3>(this.FaceNormal.Count);
			foreach (Vec3 n in this.FaceNormal)
			{
				result.FaceNormal.Add(SafeNormalize(normalT * n));
			}

			result.VertNormal = new List<Vec3>(this.VertNormal.Count);
			foreach (Vec3 n in this.VertNormal)
			{
				result.VertNormal.Add(SafeNormalize(normalT * n));
			}

			// Per #1718: the properties clone above doesn't go through the vertPos /
			// faceNormal transform, so eager-transform slot 0..2 per-meshID to keep
			// recorded world-frame normals in sync. tri_ref / hasNormals flags are
			// identical in self and result; iterate by prop vert (winding flip
			// below only reorders halfedges, not prop assignments).
			if (this.NumProp >= 3)
			{
				EagerTransformPropNormals(
					this.Halfedge,
					this.MeshRelation,
					normalT,
					result.Properties,
					this.NumPropVert(),
					this.NumProp,
					0);
			}

			bool invert = m3.Determinant() < 0.0;

			// Transform tangents — C++ TransformTangents
			// Must happen BEFORE FlipTris (matches C++ order)
			if (this.HalfedgeTangent.Count != 0)
			{
				result.HalfedgeTangent = new List<Vec4>();
				result.HalfedgeTangent.Resize(this.HalfedgeTangent.Count, new Vec4(0.0, 0.0, 0.0, 0.0));
				for (int edgeOut = 0; edgeOut < this.HalfedgeTangent.Count; edgeOut++)
				{
					int edgeIn;
					if (invert)
					{
						int tri = edgeOut / 3;
						int vert = 2 - (edgeOut - (3 * tri));
						int flipped = (3 * tri) + vert;
						edgeIn = this.Halfedge[flipped].PairedHalfedge;
					}
					else
					{
						edgeIn = edgeOut;
					}

					Vec4 oldT = this.HalfedgeTangent[edgeIn];
					Vec3 xyz = m3 * new Vec3(oldT.X, oldT.Y, oldT.Z);
					result.HalfedgeTangent[edgeOut] = new Vec4(xyz.X, xyz.Y, xyz.Z, oldT.W);
				}
			}

			if (invert)
			{
				// Flip triangle winding — matches C++ FlipTris
				Span<Halfedge> resultHalfedge =
					System.Runtime.InteropServices.CollectionsMarshal.AsSpan(result.Halfedge);
				for (int tri = 0; tri < result.NumTri(); tri++)
				{
					// Swap first and third halfedge within tri
					(resultHalfedge[3 * tri], resultHalfedge[(3 * tri) + 2]) =
						(resultHalfedge[(3 * tri) + 2], resultHalfedge[3 * tri]);

					// For each halfedge: swap startVert/endVert and remap pairedHalfedge
					for (int i = 0; i < 3; i++)
					{
						int idx = (3 * tri) + i;
						ref Halfedge h = ref resultHalfedge[idx];
						(h.StartVert, h.EndVert) = (h.EndVert, h.StartVert);

						// FlipHalfedge: within the paired tri, mirror the edge index
						int paired = h.PairedHalfedge;
						if (paired >= 0)
						{
							int p = paired;
							int pTri = p / 3;
							int pVert = 2 - (p - (3 * pTri));
							h.PairedHalfedge = (3 * pTri) + pVert;
						}
					}
				}
			}

			result.CalculateBBox();
			result.SetEpsilon(result.Epsilon, false);

			// Keep the cached collider valid without a full rebuild, mirroring C++
			// Impl::Transform: an axis-aligned transform maps the existing tree's
			// boxes directly; otherwise recompute leaf boxes on the transformed
			// mesh and refit the same tree topology.
			// Soup impls never went through sort_geometry, which is where the
			// collider is built, so they carry a zero-leaf tree. There is no
			// topology to refit in that case — cloning the empty tree and
			// refitting it against real face boxes indexes out of bounds, and
			// update_boxes' debug assert is compiled out in release. Leave the
			// default collider, exactly as the untransformed soup carried.
			if (!result.IsEmpty() && this.Collider.NumLeaves() == this.NumTri())
			{
				if (ManifoldSharp.Collider.IsAxisAligned(t))
				{
					result.Collider = this.Collider.Clone();
					result.Collider.Transform(t);
				}
				else
				{
					result.Collider = this.Collider.Clone();
					(Box[] faceBox, uint[] faceMorton) = Sort.GetFaceBoxMorton(result);
					_ = faceMorton;
					result.Collider.UpdateBoxes(faceBox);
				}
			}

			return result;
		}

		/// <summary>
		/// The Rust's derived <c>Clone</c> — field for field, with every vector copied. The
		/// self-intersection cache carries its settled value across, so see the warning on
		/// <see cref="InvalidateSelfIntersects"/> before editing a clone's geometry.
		/// </summary>
		/// <returns>A deep copy of this impl.</returns>
		public ManifoldImpl Clone()
		{
			// The Rust has both `#[derive(Clone)]` and a hand-written `shallow_clone`, and
			// the two are field-for-field identical; one body serves both here.
			return this.ShallowClone();
		}

		/// <summary>
		/// Multiply two Mat3x4 transforms as affine matrices (t1 * t2 = (t1 * to_mat4(t2)).
		/// Result is t1 applied after t2.
		/// </summary>
		/// <param name="t1">The outer transform.</param>
		/// <param name="t2">The inner transform.</param>
		/// <returns>The composed transform.</returns>
		internal static Mat3x4 Mat3x4MulMat3x4(Mat3x4 t1, Mat3x4 t2)
		{
			// Column vectors of t2 (as Vec4 with w=0 for rotation cols, w=1 for translation)
			Vec3 c0 = t1 * new Vec4(t2.X.X, t2.X.Y, t2.X.Z, 0.0);
			Vec3 c1 = t1 * new Vec4(t2.Y.X, t2.Y.Y, t2.Y.Z, 0.0);
			Vec3 c2 = t1 * new Vec4(t2.Z.X, t2.Z.Y, t2.Z.Z, 0.0);
			Vec3 c3 = t1 * new Vec4(t2.W.X, t2.W.Y, t2.W.Z, 1.0);
			return Mat3x4.FromCols(c0, c1, c2, c3);
		}

		private static ManifoldImpl FromShape(double[][] vertPosRaw, IVec3[] triVerts, Mat3x4 transform)
		{
			ManifoldImpl m = new ManifoldImpl();
			m.VertPos = new List<Vec3>(vertPosRaw.Length);
			foreach (double[] v in vertPosRaw)
			{
				Vec3 p = new Vec3(v[0], v[1], v[2]);

				// Apply transform: m * vec4(p, 1)
				m.VertPos.Add(transform * new Vec4(p.X, p.Y, p.Z, 1.0));
			}

			m.CreateHalfedges(triVerts, Array.Empty<IVec3>());
			m.InitializeOriginal();
			m.CalculateBBox();
			m.SetEpsilon(-1.0, false);
			m.SortGeometry();

			// The Rust ends from_shape with `m.set_normals_and_coplanar()`, impl_mesh.rs's
			// one-line delegator onto `face_op::set_normals_and_coplanar`. Both are ported
			// now, so this is the same call in the same form.
			m.SetNormalsAndCoplanar();
			return m;
		}

		/// <summary>
		/// Field-by-field copy used by the identity-transform fast path; the collider is
		/// copied as-is (C++ copies collider_ with the Impl).
		/// </summary>
		private ManifoldImpl ShallowClone()
		{
			ManifoldImpl copy = new ManifoldImpl();
			copy.Bbox = this.Bbox;
			copy.Epsilon = this.Epsilon;
			copy.Tolerance = this.Tolerance;
			copy.NumProp = this.NumProp;
			copy.Status = this.Status;
			copy.VertPos = new List<Vec3>(this.VertPos);
			copy.Halfedge = new List<Halfedge>(this.Halfedge);
			copy.Properties = new List<double>(this.Properties);
			copy.VertNormal = new List<Vec3>(this.VertNormal);
			copy.FaceNormal = new List<Vec3>(this.FaceNormal);
			copy.HalfedgeTangent = new List<Vec4>(this.HalfedgeTangent);
			copy.MeshRelation = this.MeshRelation.Clone();
			copy.Collider = this.Collider.Clone();
			copy.IsSoup = this.IsSoup;
			copy.SelfIntersects = this.SelfIntersects.Clone();
			return copy;
		}
	}
}
