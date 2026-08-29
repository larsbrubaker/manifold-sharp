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

// MeshGL64.cs — the `MeshGLP<f64, u64>` instantiation of types_meshgl.rs. The
// module header, and the reasoning behind spelling the two instantiations out by
// hand instead of using a generic, live in MeshGL.cs.
//
// This half is smaller than its sibling on purpose: the Rust implements Merge,
// Backside, HasNormals and UpdateNormals *only* on `MeshGLP<f32, u32>`, so there
// is nothing here for them to port to. What remains is the generic
// `impl<P: MeshPrecision, I: MeshIndex> MeshGLP<P, I>` block — the field set and
// the five accessors — at double precision with 64-bit indices.
//
// Precision is lossless end to end on this path: coordinates that went in
// through the f64 import and were left untouched by an operation come back
// bit-identical, with no float round-trip anywhere. The tolerance floor of
// `f32::EPSILON * bbox.scale()` that the f32 instantiation applies (the
// `IS_SINGLE` half of the MeshPrecision trait) does *not* apply here.

namespace ManifoldSharp
{
	/// <summary>
	/// GL-style mesh representation, double precision with 64-bit indices — the Rust
	/// <c>MeshGLP&lt;f64, u64&gt;</c>, aliased <c>MeshGL64</c>. For huge meshes and for
	/// callers that need the kernel's full precision.
	/// </summary>
	public sealed class MeshGL64
	{
		/// <summary>Creates an empty mesh — the Rust derived <c>Default</c>.</summary>
		public MeshGL64()
		{
			this.NumProp = 0;
			this.VertProperties = new List<double>();
			this.TriVerts = new List<ulong>();
			this.MergeFromVert = new List<ulong>();
			this.MergeToVert = new List<ulong>();
			this.RunIndex = new List<ulong>();
			this.RunOriginalId = new List<uint>();
			this.RunTransform = new List<double>();
			this.FaceId = new List<ulong>();
			this.HalfedgeTangent = new List<double>();
			this.RunFlags = new List<byte>();
			this.Tolerance = 0.0;
		}

		/// <summary>Number of properties per vertex, always &gt;= 3.</summary>
		public ulong NumProp { get; set; }

		/// <summary>Flat interleaved vertex properties: [x, y, z, ...] x num_verts.</summary>
		public List<double> VertProperties { get; set; }

		/// <summary>Triangle vertex indices, 3 per triangle (CCW from outside).</summary>
		public List<ulong> TriVerts { get; set; }

		/// <summary>Optional: merge-from vertex indices.</summary>
		public List<ulong> MergeFromVert { get; set; }

		/// <summary>Optional: merge-to vertex indices.</summary>
		public List<ulong> MergeToVert { get; set; }

		/// <summary>Optional: run start indices into triVerts.</summary>
		public List<ulong> RunIndex { get; set; }

		/// <summary>
		/// Optional: original mesh ID per run. Always 32-bit, on both instantiations — a
		/// mesh ID is not an index into the mesh, so the Rust field is <c>Vec&lt;u32&gt;</c>
		/// regardless of the index parameter.
		/// </summary>
		public List<uint> RunOriginalId { get; set; }

		/// <summary>Optional: 3x4 column-major transform per run (12 elements each).</summary>
		public List<double> RunTransform { get; set; }

		/// <summary>Optional: source face ID per triangle.</summary>
		public List<ulong> FaceId { get; set; }

		/// <summary>Optional: halfedge tangent vectors (4 per halfedge).</summary>
		public List<double> HalfedgeTangent { get; set; }

		/// <summary>Optional: per-run flags; 1 = backside (normals need flipping).</summary>
		public List<byte> RunFlags { get; set; }

		/// <summary>Tolerance for mesh simplification.</summary>
		public double Tolerance { get; set; }

		// NARROWING AUDIT (ulong -> int, this method and GetVertPos below). The Rust
		// computes in `usize` via `self.num_prop.to_u64() as usize`; here NumProp is the
		// u64 the type parameter names, narrowed to int to index a List. C#'s default
		// unchecked conversion truncates to the low 32 bits exactly as Rust `as` does, so
		// the two agree — and they can only disagree with the mathematical value for
		// NumProp > int.MaxValue, which VertProperties (a List<double>) could not hold one
		// vertex of. This is the wider of the two instantiations, so it is the one where
		// an out-of-range stride is at least expressible; it still cannot be stored.
		/// <summary>The number of vertices, i.e. property groups.</summary>
		/// <returns>The vertex count, or 0 when there are no properties per vertex.</returns>
		public int NumVert()
		{
			if (this.NumProp == 0)
			{
				return 0;
			}

			return this.VertProperties.Count / (int)this.NumProp;
		}

		/// <summary>The number of triangles.</summary>
		/// <returns>The triangle count.</returns>
		public int NumTri()
		{
			return this.TriVerts.Count / 3;
		}

		/// <summary>The position (property slots 0..2) of one vertex.</summary>
		/// <param name="v">The vertex index.</param>
		/// <returns>The x, y and z property values.</returns>
		public (double X, double Y, double Z) GetVertPos(int v)
		{
			int offset = v * (int)this.NumProp;
			return (
				this.VertProperties[offset],
				this.VertProperties[offset + 1],
				this.VertProperties[offset + 2]);
		}

		/// <summary>The three vertex indices of one triangle.</summary>
		/// <param name="t">The triangle index.</param>
		/// <returns>The triangle's three vertex indices.</returns>
		public (ulong A, ulong B, ulong C) GetTriVerts(int t)
		{
			int offset = 3 * t;
			return (this.TriVerts[offset], this.TriVerts[offset + 1], this.TriVerts[offset + 2]);
		}

		/// <summary>The tangent vector of one halfedge.</summary>
		/// <param name="h">The halfedge index.</param>
		/// <returns>The tangent's four components.</returns>
		public (double X, double Y, double Z, double W) GetTangent(int h)
		{
			int offset = 4 * h;
			return (
				this.HalfedgeTangent[offset],
				this.HalfedgeTangent[offset + 1],
				this.HalfedgeTangent[offset + 2],
				this.HalfedgeTangent[offset + 3]);
		}

		/// <summary>The Rust derived <c>Clone</c>: an independent copy of every vector.</summary>
		/// <returns>A deep copy of this mesh.</returns>
		public MeshGL64 Clone()
		{
			MeshGL64 copy = new MeshGL64();
			copy.NumProp = this.NumProp;
			copy.VertProperties = new List<double>(this.VertProperties);
			copy.TriVerts = new List<ulong>(this.TriVerts);
			copy.MergeFromVert = new List<ulong>(this.MergeFromVert);
			copy.MergeToVert = new List<ulong>(this.MergeToVert);
			copy.RunIndex = new List<ulong>(this.RunIndex);
			copy.RunOriginalId = new List<uint>(this.RunOriginalId);
			copy.RunTransform = new List<double>(this.RunTransform);
			copy.FaceId = new List<ulong>(this.FaceId);
			copy.HalfedgeTangent = new List<double>(this.HalfedgeTangent);
			copy.RunFlags = new List<byte>(this.RunFlags);
			copy.Tolerance = this.Tolerance;
			return copy;
		}
	}
}
