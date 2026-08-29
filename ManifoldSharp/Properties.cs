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

// Phase 8: Properties — ported from src/properties.cpp
//
// Mesh property calculations: volume, surface area, curvature, convexity,
// triangle validation, and degenerate detection.
//
// ── The whole file is one `impl ManifoldImpl` block ──────────────────────────
// properties.rs declares one type (the `Property` enum) and hangs everything
// else off `ManifoldImpl`. Those methods land here as a partial of that class,
// the arrangement Collider.Geometry.cs established: the `impl ManifoldImpl`
// block travels with its own module, not with ManifoldImpl.cs.
//
// ── Reduction order is the result ────────────────────────────────────────────
// GetProperty is a Kahan sum and CalculateCurvature accumulates per-vertex
// through a triangle loop. Both are transcribed statement for statement,
// including which operand each `+=` lands on, because in float arithmetic the
// order of a reduction *is* the value it produces. Reassociating a line here —
// or hoisting an invariant out of one of these loops — is a numerical change,
// not a refactor.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// Which scalar property to compute over the mesh.
	/// </summary>
	public enum Property
	{
		/// <summary>The total area of every triangle.</summary>
		SurfaceArea,

		/// <summary>The signed volume enclosed by the triangles.</summary>
		Volume,
	}

	/// <content>
	/// The <c>impl ManifoldImpl</c> block of properties.rs: the global scalar properties,
	/// the two validity scans, convexity, per-vertex curvature, and the index-bounds check.
	/// </content>
	public sealed partial class ManifoldImpl
	{
		/// <summary>
		/// Compute a global scalar property (volume or surface area) using Kahan
		/// summation for numerical stability.
		/// </summary>
		/// <param name="prop">Which property to accumulate.</param>
		/// <returns>The property value, or 0 for an empty mesh.</returns>
		public double GetProperty(Property prop)
		{
			if (this.IsEmpty())
			{
				return 0.0;
			}

			double value = 0.0;
			double compensation = 0.0;

			for (int tri = 0; tri < this.NumTri(); tri++)
			{
				int sv = this.Halfedge[3 * tri].StartVert;
				if (sv < 0)
				{
					continue;
				}

				Vec3 v0 = this.VertPos[sv];
				Vec3 v1 = this.VertPos[this.Halfedge[(3 * tri) + 1].StartVert];
				Vec3 v2 = this.VertPos[this.Halfedge[(3 * tri) + 2].StartVert];

				// The Rust `match` has exactly the two arms the enum has; an if/else keeps
				// both bodies statement-for-statement without inventing an unreachable
				// default the exhaustive match does not have.
				double value1;
				if (prop == Property.Volume)
				{
					Vec3 crossP = Cross(v1 - v0, v2 - v0);
					value1 = Dot(crossP, v0) / 6.0;
				}
				else
				{
					value1 = Length(Cross(v1 - v0, v2 - v0)) / 2.0;
				}

				double t = value + value1;
				compensation += (value - t) + value1;
				value = t;
			}

			return value + compensation;
		}

		/// <summary>
		/// Returns true if all triangles are CCW relative to their face normals.
		/// </summary>
		/// <returns>True when every triangle agrees with its face normal.</returns>
		public bool MatchesTriNormals()
		{
			if (this.Halfedge.Count == 0 || this.FaceNormal.Count != this.NumTri())
			{
				return true;
			}

			for (int face = 0; face < this.NumTri(); face++)
			{
				if (this.Halfedge[3 * face].PairedHalfedge < 0)
				{
					continue;
				}

				Proj2x3 projection = FaceOp.GetAxisAlignedProjection(this.FaceNormal[face]);
				Vec2[] v = new Vec2[3] { new Vec2(0.0, 0.0), new Vec2(0.0, 0.0), new Vec2(0.0, 0.0) };
				double maxD = double.NegativeInfinity;
				double minD = double.PositiveInfinity;
				bool anyNonFinite = false;

				for (int i = 0; i < 3; i++)
				{
					Vec3 p = this.VertPos[this.Halfedge[(3 * face) + i].StartVert];
					v[i] = projection.Apply(p);
					double d = Dot(p, this.FaceNormal[face]);
					if (!double.IsFinite(d))
					{
						anyNonFinite = true;
						break;
					}

					maxD = MaxF64(maxD, d);
					minD = MinF64(minD, d);
				}

				if (anyNonFinite)
				{
					continue;
				}

				if (maxD - minD > 2.0 * this.Tolerance)
				{
					return false;
				}

				int winding = Polygon.Ccw(v[0], v[1], v[2], this.Epsilon * 2.0);
				if (winding < 0)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// Returns the number of triangles that are colinear within epsilon.
		/// </summary>
		/// <returns>The degenerate triangle count.</returns>
		public int NumDegenerateTris()
		{
			if (this.Halfedge.Count == 0 || this.FaceNormal.Count != this.NumTri())
			{
				return 0;
			}

			int count = 0;
			for (int face = 0; face < this.NumTri(); face++)
			{
				if (this.Halfedge[3 * face].PairedHalfedge < 0)
				{
					count += 1;
					continue;
				}

				Proj2x3 projection = FaceOp.GetAxisAlignedProjection(this.FaceNormal[face]);
				Vec2[] v = new Vec2[3] { new Vec2(0.0, 0.0), new Vec2(0.0, 0.0), new Vec2(0.0, 0.0) };
				for (int i = 0; i < 3; i++)
				{
					v[i] = projection.Apply(this.VertPos[this.Halfedge[(3 * face) + i].StartVert]);
				}

				// Per #1671: degeneracy is judged within tolerance_, not epsilon_.
				int winding = Polygon.Ccw(v[0], v[1], v[2], this.Tolerance / 2.0);
				if (winding == 0)
				{
					count += 1;
				}
			}

			return count;
		}

		/// <summary>
		/// Returns true if the manifold is genus 0 and contains no concave edges.
		/// </summary>
		/// <returns>True when the mesh is convex.</returns>
		public bool IsConvex()
		{
			long chi = (long)this.NumVert() - (long)this.NumEdge() + (long)this.NumTri();
			long genus = 1 - (chi / 2);
			if (genus != 0)
			{
				return false;
			}

			int nbEdges = this.Halfedge.Count;
			for (int idx = 0; idx < nbEdges; idx++)
			{
				Halfedge edge = this.Halfedge[idx];
				if (!edge.IsForward())
				{
					continue;
				}

				Vec3 normal0 = this.FaceNormal[idx / 3];

				// The Rust writes `paired_halfedge as usize / 3`, which on an unpaired
				// halfedge (-1) wraps to usize::MAX and panics on the index below. The
				// C# int division yields 0 instead and would silently read triangle 0's
				// normal. Neither is reachable from a manifold mesh — the only kind this
				// method is called on — and both are wrong answers for a soup impl. The
				// twin of this site is in CalculateCurvature.
				Vec3 normal1 = this.FaceNormal[edge.PairedHalfedge / 3];

				if (normal0 == normal1)
				{
					continue;
				}

				Vec3 edgeVec = this.VertPos[edge.EndVert] - this.VertPos[edge.StartVert];
				bool convex = Dot(edgeVec, Cross(normal0, normal1)) > 0.0;
				if (!convex)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// Compute Gaussian and/or mean curvature per vertex, storing results
		/// into the property channels at the given indices. Pass -1 to skip.
		/// </summary>
		/// <param name="gaussianIdx">The property channel for Gaussian curvature, or -1 to skip.</param>
		/// <param name="meanIdx">The property channel for mean curvature, or -1 to skip.</param>
		public void CalculateCurvature(int gaussianIdx, int meanIdx)
		{
			if (this.IsEmpty())
			{
				return;
			}

			if (gaussianIdx < 0 && meanIdx < 0)
			{
				return;
			}

			int numVert = this.NumVert();
			double[] meanCurvature = new double[numVert];
			double[] gaussianCurvature = new double[numVert];
			for (int i = 0; i < numVert; i++)
			{
				gaussianCurvature[i] = Types.KTwoPi;
			}

			double[] area = new double[numVert];
			double[] degree = new double[numVert];

			for (int tri = 0; tri < this.NumTri(); tri++)
			{
				Vec3[] edgeDirs = new Vec3[3]
				{
					new Vec3(0.0, 0.0, 0.0),
					new Vec3(0.0, 0.0, 0.0),
					new Vec3(0.0, 0.0, 0.0),
				};
				double[] edgeLength = new double[3];

				for (int i = 0; i < 3; i++)
				{
					int startVert = this.Halfedge[(3 * tri) + i].StartVert;
					int endVert = this.Halfedge[(3 * tri) + i].EndVert;
					edgeDirs[i] = this.VertPos[endVert] - this.VertPos[startVert];
					edgeLength[i] = Length(edgeDirs[i]);
					if (edgeLength[i] > 0.0)
					{
						edgeDirs[i] = edgeDirs[i] / edgeLength[i];
					}

					// The Rust writes `paired_halfedge as usize / 3`, which on an unpaired
					// halfedge (-1) wraps to usize::MAX and panics on the index below. The
					// C# int division yields 0 instead and would silently read triangle 0's
					// normal. Neither is reachable from a manifold mesh — the only kind this
					// method is called on — and both are wrong answers for a soup impl. The
					// twin of this site is in IsConvex.
					int neighborTri = this.Halfedge[(3 * tri) + i].PairedHalfedge / 3;
					double dihedral = 0.25
						* edgeLength[i]
						* DeterministicMath.Asin(Dot(
							Cross(this.FaceNormal[tri], this.FaceNormal[neighborTri]),
							edgeDirs[i]));
					meanCurvature[startVert] += dihedral;
					meanCurvature[endVert] += dihedral;
					degree[startVert] += 1.0;
				}

				double phi0 = DeterministicMath.Acos(-Dot(edgeDirs[2], edgeDirs[0]));
				double phi1 = DeterministicMath.Acos(-Dot(edgeDirs[0], edgeDirs[1]));
				double phi2 = Math.PI - phi0 - phi1;
				double area3 =
					edgeLength[0] * edgeLength[1] * Length(Cross(edgeDirs[0], edgeDirs[1])) / 6.0;

				double[] phi = new double[3] { phi0, phi1, phi2 };
				for (int i = 0; i < 3; i++)
				{
					int vert = this.Halfedge[(3 * tri) + i].StartVert;
					gaussianCurvature[vert] -= phi[i];
					area[vert] += area3;
				}
			}

			for (int vert = 0; vert < numVert; vert++)
			{
				double factor = degree[vert] / (6.0 * area[vert]);
				meanCurvature[vert] *= factor;
				gaussianCurvature[vert] *= factor;
			}

			int oldNumProp = this.NumProp;
			int numProp = Math.Max(
				Math.Max(oldNumProp, gaussianIdx >= 0 ? gaussianIdx + 1 : 0),
				meanIdx >= 0 ? meanIdx + 1 : 0);

			List<double> oldProperties = new List<double>(this.Properties);

			// NumPropVert reads NumProp, so it has to be taken before the assignment
			// below moves NumProp to its new value — the Rust orders these four
			// statements the same way for the same reason.
			int numPropVert = this.NumPropVert();

			// `vec![0.0; num_prop * num_prop_vert]` — the constructor argument is capacity,
			// so the Resize is what gives the list its length.
			this.Properties = new List<double>(numProp * numPropVert);
			this.Properties.Resize(numProp * numPropVert, 0.0);
			this.NumProp = numProp;

			bool[] visited = new bool[numPropVert];

			for (int tri = 0; tri < this.NumTri(); tri++)
			{
				for (int i = 0; i < 3; i++)
				{
					Halfedge edge = this.Halfedge[(3 * tri) + i];
					int vert = edge.StartVert;
					int propVert = edge.PropVert;

					if (visited[propVert])
					{
						continue;
					}

					visited[propVert] = true;

					for (int p = 0; p < oldNumProp; p++)
					{
						this.Properties[(numProp * propVert) + p] =
							oldProperties[(oldNumProp * propVert) + p];
					}

					if (gaussianIdx >= 0)
					{
						this.Properties[(numProp * propVert) + gaussianIdx] =
							gaussianCurvature[vert];
					}

					if (meanIdx >= 0)
					{
						this.Properties[(numProp * propVert) + meanIdx] = meanCurvature[vert];
					}
				}
			}
		}

		/// <summary>
		/// Checks that all indices in the given triVerts array are within the
		/// bounds of <see cref="VertPos"/>.
		/// </summary>
		/// <param name="triVerts">The triangles to check.</param>
		/// <returns>True when every index is in range.</returns>
		/// <remarks>
		/// The Rust takes <c>&amp;[IVec3]</c>, which callers satisfy with either a slice or a
		/// <c>Vec</c>; <see cref="IReadOnlyList{T}"/> is the parameter type that keeps both
		/// an array and a <see cref="List{T}"/> callable without a copy.
		/// </remarks>
		public bool IsIndexInBounds(IReadOnlyList<IVec3> triVerts)
		{
			if (triVerts.Count == 0)
			{
				return true;
			}

			int numVert = this.NumVert();
			foreach (IVec3 tri in triVerts)
			{
				int minV = Math.Min(Math.Min(tri.X, tri.Y), tri.Z);
				int maxV = Math.Max(Math.Max(tri.X, tri.Y), tri.Z);
				if (minV < 0 || maxV >= numVert)
				{
					return false;
				}
			}

			return true;
		}
	}
}
