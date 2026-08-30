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

// MeshGLAccess.cs — the narrow conversion interface MeshGL.cs's header called
// for, and the Phase 6 decision it deferred to this phase.
//
// ── The debt this pays ───────────────────────────────────────────────────────
// MeshGL.cs spells out `MeshGLP<f32, u32>` and `MeshGLP<f64, u64>` as two
// concrete classes and records the cost: "the two genuinely generic functions in
// manifold_meshgl.rs (`from_mesh_impl<P, I>` and `get_mesh_gl_impl<P, I>`, Phase
// 6) have no single C# body to land in. When that phase arrives it either
// duplicates them or introduces a narrow conversion interface over these two
// classes — a decision for Phase 6."
//
// This is the interface. The bodies stay single (Manifold.MeshGL.cs,
// Manifold.MeshGL.Export.cs), and every place the Rust writes `P::from_f64(x)`,
// `p.to_f64()`, `I::from_usize(n)`, `I::from_i32(n)` or `i.to_u64()` becomes a
// call on this interface, so the f32 instantiation narrows at exactly the sites
// the C++ float template narrows at and the f64 instantiation stays lossless.
// Duplication was the alternative and was rejected: two ~230-line bodies that
// must be kept identical *except* at the conversion sites is precisely the shape
// of change that goes wrong silently, and the exactness bar cannot afford it.
//
// ── Why the conversions are the interface, and nothing else is ───────────────
// `IMeshGLAccess` is deliberately not "a mesh". It exposes the fields as flat
// indexed accessors in the widest type (f64 / u64), because that is what the two
// bodies actually do with them — read one element, convert, use — and because a
// richer abstraction would start making decisions the Rust makes inline.
//
// Read-back matters and is why the *writing* half is on the same interface:
// `get_mesh_gl_impl` re-reads `out.vert_properties` and `out.run_transform`
// after writing them (the #1718 normal normalization, and
// `normal_transform_for_run`). In the f32 instantiation those values have been
// through an `as f32` and read back widened, and the Rust comment says so
// explicitly — "in the f32 instantiation this round-trips through f32 exactly as
// the C++ float template does before normalizing". Staging the output in f64 and
// narrowing at the end would quietly lose that round-trip, so the sink writes
// into the real list and reads back out of it.
//
// The Rust trait constants and free conversions map as:
//   P::IS_SINGLE        -> IsSinglePrecision
//   P::from_f64 / to_f64 -> ToPrecision / FromPrecision (on the sink's storage)
//   I::from_usize        -> unchecked (uint) / (ulong)   — truncating for u32
//   I::from_i32          -> unchecked (uint) / (ulong)   — bit-pattern for u32,
//                           sign-extension for u64, per types_meshgl.rs
//   I::to_u64            -> widening, no narrowing anywhere

namespace ManifoldSharp
{
	/// <summary>
	/// The conversion contract of <see cref="MeshGL"/> and <see cref="MeshGL64"/>, so the
	/// import and export bodies of manifold_meshgl.rs have one C# body each.
	/// </summary>
	/// <remarks>
	/// Values come out widened (f64 / u64) and go in as f64 / int, narrowing inside the
	/// implementation exactly where the Rust's <c>P::from_f64</c> / <c>I::from_*</c> do.
	/// See the file header for why this exists.
	/// </remarks>
	internal interface IMeshGLAccess
	{
		/// <summary>The Rust <c>P::IS_SINGLE</c>: true only for the f32 instantiation.</summary>
		bool IsSinglePrecision { get; }

		/// <summary>The raw <c>num_prop</c>, widened; the import range-checks it against 3.</summary>
		ulong NumPropRaw { get; }

		/// <summary>The mesh tolerance, widened.</summary>
		double Tolerance { get; }

		/// <summary>Entry count of <c>vert_properties</c>.</summary>
		int VertPropertiesCount { get; }

		/// <summary>Entry count of <c>tri_verts</c>.</summary>
		int TriVertsCount { get; }

		/// <summary>Entry count of <c>merge_from_vert</c>.</summary>
		int MergeFromVertCount { get; }

		/// <summary>Entry count of <c>merge_to_vert</c>.</summary>
		int MergeToVertCount { get; }

		/// <summary>Entry count of <c>run_index</c>.</summary>
		int RunIndexCount { get; }

		/// <summary>Entry count of <c>run_transform</c>.</summary>
		int RunTransformCount { get; }

		/// <summary>Entry count of <c>face_id</c>.</summary>
		int FaceIdCount { get; }

		/// <summary>Entry count of <c>halfedge_tangent</c>.</summary>
		int HalfedgeTangentCount { get; }

		/// <summary><c>run_original_id</c>, which is <c>Vec&lt;u32&gt;</c> on both instantiations.</summary>
		List<uint> RunOriginalId { get; }

		/// <summary><c>run_flags</c>, which is <c>Vec&lt;u8&gt;</c> on both instantiations.</summary>
		List<byte> RunFlags { get; }

		/// <summary>The number of vertices, i.e. property groups.</summary>
		/// <returns>The vertex count.</returns>
		int NumVert();

		/// <summary>The number of triangles.</summary>
		/// <returns>The triangle count.</returns>
		int NumTri();

		/// <summary>One entry of <c>vert_properties</c>, widened.</summary>
		/// <param name="i">The flat index.</param>
		/// <returns>The value as f64.</returns>
		double VertProperty(int i);

		/// <summary>One entry of <c>tri_verts</c>, widened.</summary>
		/// <param name="i">The flat index.</param>
		/// <returns>The value as u64.</returns>
		ulong TriVert(int i);

		/// <summary>One entry of <c>merge_from_vert</c>, widened.</summary>
		/// <param name="i">The entry index.</param>
		/// <returns>The value as u64.</returns>
		ulong MergeFromVert(int i);

		/// <summary>One entry of <c>merge_to_vert</c>, widened.</summary>
		/// <param name="i">The entry index.</param>
		/// <returns>The value as u64.</returns>
		ulong MergeToVert(int i);

		/// <summary>One entry of <c>run_index</c>, widened.</summary>
		/// <param name="i">The entry index.</param>
		/// <returns>The value as u64.</returns>
		ulong RunIndex(int i);

		/// <summary>One entry of <c>run_transform</c>, widened.</summary>
		/// <param name="i">The flat index.</param>
		/// <returns>The value as f64.</returns>
		double RunTransform(int i);

		/// <summary>One entry of <c>face_id</c>, widened.</summary>
		/// <param name="i">The entry index.</param>
		/// <returns>The value as u64.</returns>
		ulong FaceId(int i);

		/// <summary>One entry of <c>halfedge_tangent</c>, widened.</summary>
		/// <param name="i">The flat index.</param>
		/// <returns>The value as f64.</returns>
		double HalfedgeTangent(int i);

		/// <summary>The position (property slots 0..2) of one vertex, widened.</summary>
		/// <param name="v">The vertex index.</param>
		/// <returns>The x, y and z property values.</returns>
		(double X, double Y, double Z) GetVertPos(int v);

		/// <summary>The three vertex indices of one triangle, widened.</summary>
		/// <param name="t">The triangle index.</param>
		/// <returns>The triangle's three vertex indices.</returns>
		(ulong A, ulong B, ulong C) GetTriVerts(int t);

		// ─── Write side (export only) ────────────────────────────────────────

		/// <summary>Sets <c>num_prop</c> from a <c>usize</c>, narrowing like <c>I::from_usize</c>.</summary>
		/// <param name="value">The new property count.</param>
		void SetNumPropFromUSize(int value);

		/// <summary>Sets <c>tolerance</c>, narrowing like <c>P::from_f64</c>.</summary>
		/// <param name="value">The new tolerance.</param>
		void SetTolerance(double value);

		/// <summary>Appends to <c>halfedge_tangent</c>, narrowing like <c>P::from_f64</c>.</summary>
		/// <param name="value">The value to append.</param>
		void AddHalfedgeTangent(double value);

		/// <summary>Grows <c>tri_verts</c> to <paramref name="count"/> zeros.</summary>
		/// <param name="count">The new length.</param>
		void ResizeTriVerts(int count);

		/// <summary>Grows <c>face_id</c> to <paramref name="count"/> zeros.</summary>
		/// <param name="count">The new length.</param>
		void ResizeFaceId(int count);

		/// <summary>Grows <c>vert_properties</c> to <paramref name="count"/> zeros.</summary>
		/// <param name="count">The new length.</param>
		void ResizeVertProperties(int count);

		/// <summary>Writes <c>face_id[i]</c> from an <c>i32</c>, like <c>I::from_i32</c>.</summary>
		/// <param name="i">The entry index.</param>
		/// <param name="value">The value to store.</param>
		void SetFaceIdFromI32(int i, int value);

		/// <summary>Writes <c>tri_verts[i]</c> from an <c>i32</c>, like <c>I::from_i32</c>.</summary>
		/// <param name="i">The flat index.</param>
		/// <param name="value">The value to store.</param>
		void SetTriVertFromI32(int i, int value);

		/// <summary>Writes <c>tri_verts[i]</c> from a <c>usize</c>, like <c>I::from_usize</c>.</summary>
		/// <param name="i">The flat index.</param>
		/// <param name="value">The value to store.</param>
		void SetTriVertFromUSize(int i, int value);

		/// <summary>Appends to <c>run_index</c> from a <c>usize</c>, like <c>I::from_usize</c>.</summary>
		/// <param name="value">The value to append.</param>
		void AddRunIndexFromUSize(int value);

		/// <summary>Appends to <c>run_transform</c>, narrowing like <c>P::from_f64</c>.</summary>
		/// <param name="value">The value to append.</param>
		void AddRunTransform(double value);

		/// <summary>Writes <c>vert_properties[i]</c>, narrowing like <c>P::from_f64</c>.</summary>
		/// <param name="i">The flat index.</param>
		/// <param name="value">The value to store.</param>
		void SetVertProperty(int i, double value);

		/// <summary>Appends to <c>vert_properties</c>, narrowing like <c>P::from_f64</c>.</summary>
		/// <param name="value">The value to append.</param>
		void AddVertProperty(double value);

		/// <summary>Appends <c>P::default()</c> (zero) to <c>vert_properties</c>.</summary>
		void AddVertPropertyDefault();

		/// <summary>Appends to <c>merge_from_vert</c> from a <c>usize</c>.</summary>
		/// <param name="value">The value to append.</param>
		void AddMergeFromVertFromUSize(int value);

		/// <summary>Appends to <c>merge_to_vert</c> from an <c>i32</c>.</summary>
		/// <param name="value">The value to append.</param>
		void AddMergeToVertFromI32(int value);
	}

	/// <summary>The <see cref="IMeshGLAccess"/> view of a single-precision <see cref="MeshGL"/>.</summary>
	internal sealed class MeshGLAccess : IMeshGLAccess
	{
		private readonly MeshGL mesh;

		/// <summary>Wraps a mesh.</summary>
		/// <param name="mesh">The mesh to view.</param>
		public MeshGLAccess(MeshGL mesh)
		{
			this.mesh = mesh;
		}

		/// <inheritdoc/>
		public bool IsSinglePrecision => true;

		/// <inheritdoc/>
		public ulong NumPropRaw => this.mesh.NumProp;

		/// <inheritdoc/>
		public double Tolerance => this.mesh.Tolerance;

		/// <inheritdoc/>
		public int VertPropertiesCount => this.mesh.VertProperties.Count;

		/// <inheritdoc/>
		public int TriVertsCount => this.mesh.TriVerts.Count;

		/// <inheritdoc/>
		public int MergeFromVertCount => this.mesh.MergeFromVert.Count;

		/// <inheritdoc/>
		public int MergeToVertCount => this.mesh.MergeToVert.Count;

		/// <inheritdoc/>
		public int RunIndexCount => this.mesh.RunIndex.Count;

		/// <inheritdoc/>
		public int RunTransformCount => this.mesh.RunTransform.Count;

		/// <inheritdoc/>
		public int FaceIdCount => this.mesh.FaceId.Count;

		/// <inheritdoc/>
		public int HalfedgeTangentCount => this.mesh.HalfedgeTangent.Count;

		/// <inheritdoc/>
		public List<uint> RunOriginalId => this.mesh.RunOriginalId;

		/// <inheritdoc/>
		public List<byte> RunFlags => this.mesh.RunFlags;

		/// <inheritdoc/>
		public int NumVert() => this.mesh.NumVert();

		/// <inheritdoc/>
		public int NumTri() => this.mesh.NumTri();

		/// <inheritdoc/>
		public double VertProperty(int i) => this.mesh.VertProperties[i];

		/// <inheritdoc/>
		public ulong TriVert(int i) => this.mesh.TriVerts[i];

		/// <inheritdoc/>
		public ulong MergeFromVert(int i) => this.mesh.MergeFromVert[i];

		/// <inheritdoc/>
		public ulong MergeToVert(int i) => this.mesh.MergeToVert[i];

		/// <inheritdoc/>
		public ulong RunIndex(int i) => this.mesh.RunIndex[i];

		/// <inheritdoc/>
		public double RunTransform(int i) => this.mesh.RunTransform[i];

		/// <inheritdoc/>
		public ulong FaceId(int i) => this.mesh.FaceId[i];

		/// <inheritdoc/>
		public double HalfedgeTangent(int i) => this.mesh.HalfedgeTangent[i];

		/// <inheritdoc/>
		public (double X, double Y, double Z) GetVertPos(int v)
		{
			(float x, float y, float z) = this.mesh.GetVertPos(v);
			return (x, y, z);
		}

		/// <inheritdoc/>
		public (ulong A, ulong B, ulong C) GetTriVerts(int t)
		{
			(uint a, uint b, uint c) = this.mesh.GetTriVerts(t);
			return (a, b, c);
		}

		/// <inheritdoc/>
		public void SetNumPropFromUSize(int value) => this.mesh.NumProp = (uint)value;

		/// <inheritdoc/>
		public void SetTolerance(double value) => this.mesh.Tolerance = (float)value;

		/// <inheritdoc/>
		public void AddHalfedgeTangent(double value) => this.mesh.HalfedgeTangent.Add((float)value);

		/// <inheritdoc/>
		public void ResizeTriVerts(int count) => Resize(this.mesh.TriVerts, count, 0u);

		/// <inheritdoc/>
		public void ResizeFaceId(int count) => Resize(this.mesh.FaceId, count, 0u);

		/// <inheritdoc/>
		public void ResizeVertProperties(int count) => Resize(this.mesh.VertProperties, count, 0f);

		/// <inheritdoc/>
		public void SetFaceIdFromI32(int i, int value) => this.mesh.FaceId[i] = (uint)value;

		/// <inheritdoc/>
		public void SetTriVertFromI32(int i, int value) => this.mesh.TriVerts[i] = (uint)value;

		/// <inheritdoc/>
		public void SetTriVertFromUSize(int i, int value) => this.mesh.TriVerts[i] = (uint)value;

		/// <inheritdoc/>
		public void AddRunIndexFromUSize(int value) => this.mesh.RunIndex.Add((uint)value);

		/// <inheritdoc/>
		public void AddRunTransform(double value) => this.mesh.RunTransform.Add((float)value);

		/// <inheritdoc/>
		public void SetVertProperty(int i, double value) => this.mesh.VertProperties[i] = (float)value;

		/// <inheritdoc/>
		public void AddVertProperty(double value) => this.mesh.VertProperties.Add((float)value);

		/// <inheritdoc/>
		public void AddVertPropertyDefault() => this.mesh.VertProperties.Add(0f);

		/// <inheritdoc/>
		public void AddMergeFromVertFromUSize(int value) => this.mesh.MergeFromVert.Add((uint)value);

		/// <inheritdoc/>
		public void AddMergeToVertFromI32(int value) => this.mesh.MergeToVert.Add((uint)value);

		/// <summary>The Rust <c>Vec::resize</c>: grow with a fill value, or truncate.</summary>
		/// <typeparam name="T">The element type.</typeparam>
		/// <param name="list">The list to resize.</param>
		/// <param name="count">The new length.</param>
		/// <param name="fill">The value new elements take.</param>
		private static void Resize<T>(List<T> list, int count, T fill)
		{
			while (list.Count > count)
			{
				list.RemoveAt(list.Count - 1);
			}

			while (list.Count < count)
			{
				list.Add(fill);
			}
		}
	}

	/// <summary>The <see cref="IMeshGLAccess"/> view of a double-precision <see cref="MeshGL64"/>.</summary>
	internal sealed class MeshGL64Access : IMeshGLAccess
	{
		private readonly MeshGL64 mesh;

		/// <summary>Wraps a mesh.</summary>
		/// <param name="mesh">The mesh to view.</param>
		public MeshGL64Access(MeshGL64 mesh)
		{
			this.mesh = mesh;
		}

		/// <inheritdoc/>
		public bool IsSinglePrecision => false;

		/// <inheritdoc/>
		public ulong NumPropRaw => this.mesh.NumProp;

		/// <inheritdoc/>
		public double Tolerance => this.mesh.Tolerance;

		/// <inheritdoc/>
		public int VertPropertiesCount => this.mesh.VertProperties.Count;

		/// <inheritdoc/>
		public int TriVertsCount => this.mesh.TriVerts.Count;

		/// <inheritdoc/>
		public int MergeFromVertCount => this.mesh.MergeFromVert.Count;

		/// <inheritdoc/>
		public int MergeToVertCount => this.mesh.MergeToVert.Count;

		/// <inheritdoc/>
		public int RunIndexCount => this.mesh.RunIndex.Count;

		/// <inheritdoc/>
		public int RunTransformCount => this.mesh.RunTransform.Count;

		/// <inheritdoc/>
		public int FaceIdCount => this.mesh.FaceId.Count;

		/// <inheritdoc/>
		public int HalfedgeTangentCount => this.mesh.HalfedgeTangent.Count;

		/// <inheritdoc/>
		public List<uint> RunOriginalId => this.mesh.RunOriginalId;

		/// <inheritdoc/>
		public List<byte> RunFlags => this.mesh.RunFlags;

		/// <inheritdoc/>
		public int NumVert() => this.mesh.NumVert();

		/// <inheritdoc/>
		public int NumTri() => this.mesh.NumTri();

		/// <inheritdoc/>
		public double VertProperty(int i) => this.mesh.VertProperties[i];

		/// <inheritdoc/>
		public ulong TriVert(int i) => this.mesh.TriVerts[i];

		/// <inheritdoc/>
		public ulong MergeFromVert(int i) => this.mesh.MergeFromVert[i];

		/// <inheritdoc/>
		public ulong MergeToVert(int i) => this.mesh.MergeToVert[i];

		/// <inheritdoc/>
		public ulong RunIndex(int i) => this.mesh.RunIndex[i];

		/// <inheritdoc/>
		public double RunTransform(int i) => this.mesh.RunTransform[i];

		/// <inheritdoc/>
		public ulong FaceId(int i) => this.mesh.FaceId[i];

		/// <inheritdoc/>
		public double HalfedgeTangent(int i) => this.mesh.HalfedgeTangent[i];

		/// <inheritdoc/>
		public (double X, double Y, double Z) GetVertPos(int v) => this.mesh.GetVertPos(v);

		/// <inheritdoc/>
		public (ulong A, ulong B, ulong C) GetTriVerts(int t) => this.mesh.GetTriVerts(t);

		/// <inheritdoc/>
		public void SetNumPropFromUSize(int value) => this.mesh.NumProp = (ulong)value;

		/// <inheritdoc/>
		public void SetTolerance(double value) => this.mesh.Tolerance = value;

		/// <inheritdoc/>
		public void AddHalfedgeTangent(double value) => this.mesh.HalfedgeTangent.Add(value);

		/// <inheritdoc/>
		public void ResizeTriVerts(int count) => Resize(this.mesh.TriVerts, count, 0ul);

		/// <inheritdoc/>
		public void ResizeFaceId(int count) => Resize(this.mesh.FaceId, count, 0ul);

		/// <inheritdoc/>
		public void ResizeVertProperties(int count) => Resize(this.mesh.VertProperties, count, 0.0);

		// NARROWING AUDIT (i32 -> u64, the two SetFrom*FromI32 members below and
		// AddMergeToVertFromI32). Rust `I::from_i32` for u64 is `v as i64 as u64`, i.e.
		// SIGN-EXTEND then reinterpret, so -1 becomes 0xFFFF_FFFF_FFFF_FFFF and not
		// 0x0000_0000_FFFF_FFFF. C#'s `(ulong)someInt` widens through long the same way,
		// so the cast below is that conversion and not a coincidence. The u32 sibling
		// takes the bit pattern instead, which is why the two adapters differ here.
		/// <inheritdoc/>
		public void SetFaceIdFromI32(int i, int value) => this.mesh.FaceId[i] = (ulong)value;

		/// <inheritdoc/>
		public void SetTriVertFromI32(int i, int value) => this.mesh.TriVerts[i] = (ulong)value;

		/// <inheritdoc/>
		public void SetTriVertFromUSize(int i, int value) => this.mesh.TriVerts[i] = (ulong)value;

		/// <inheritdoc/>
		public void AddRunIndexFromUSize(int value) => this.mesh.RunIndex.Add((ulong)value);

		/// <inheritdoc/>
		public void AddRunTransform(double value) => this.mesh.RunTransform.Add(value);

		/// <inheritdoc/>
		public void SetVertProperty(int i, double value) => this.mesh.VertProperties[i] = value;

		/// <inheritdoc/>
		public void AddVertProperty(double value) => this.mesh.VertProperties.Add(value);

		/// <inheritdoc/>
		public void AddVertPropertyDefault() => this.mesh.VertProperties.Add(0.0);

		/// <inheritdoc/>
		public void AddMergeFromVertFromUSize(int value) => this.mesh.MergeFromVert.Add((ulong)value);

		/// <inheritdoc/>
		public void AddMergeToVertFromI32(int value) => this.mesh.MergeToVert.Add((ulong)value);

		/// <summary>The Rust <c>Vec::resize</c>: grow with a fill value, or truncate.</summary>
		/// <typeparam name="T">The element type.</typeparam>
		/// <param name="list">The list to resize.</param>
		/// <param name="count">The new length.</param>
		/// <param name="fill">The value new elements take.</param>
		private static void Resize<T>(List<T> list, int count, T fill)
		{
			while (list.Count > count)
			{
				list.RemoveAt(list.Count - 1);
			}

			while (list.Count < count)
			{
				list.Add(fill);
			}
		}
	}
}
