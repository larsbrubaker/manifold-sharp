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

// Types.Shared.cs — the src/shared.h and src/impl.h half of types.rs, split out
// of Types.cs for the 800-line cap: Halfedge, Barycentric, TriRef, TmpEdge, the
// Relation / MeshRelationD tables, and the halfedge index helpers (which
// continue the `Types` static class). The module header for the whole types.rs
// port lives in Types.cs.
//
// Field names, types and order are transcribed exactly — these structs are the
// mesh arena, read and written by every phase from 3 onward, and the orderings
// below decide sort results downstream (TmpEdge in particular is sorted in
// edge_op / face_op, so its comparison is a numerical-parity surface, not a
// convenience).

using System.Runtime.CompilerServices;

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <summary>
	/// One directed half of an edge, from src/shared.h.
	/// </summary>
	/// <remarks>
	/// Ordering is by <see cref="StartVert"/>, then <see cref="EndVert"/> — never by
	/// <see cref="PairedHalfedge"/> or <see cref="PropVert"/> — matching the Rust
	/// <c>Ord</c>. Equality does look at all four fields, so
	/// <see cref="CompareTo"/> returning 0 does not imply <c>Equals</c>: that split is
	/// the Rust's and is what the sort sites want, but it means an ordered *set* keyed on
	/// this type would collapse halfedges the port considers distinct. Sort with it; do
	/// not deduplicate with it.
	/// </remarks>
	public struct Halfedge : IEquatable<Halfedge>, IComparable<Halfedge>
	{
		/// <summary>Vertex this halfedge starts at.</summary>
		public int StartVert;

		/// <summary>Vertex this halfedge ends at.</summary>
		public int EndVert;

		/// <summary>Index of the halfedge running the other way along the same edge.</summary>
		public int PairedHalfedge;

		/// <summary>Property vertex index for the start of this halfedge.</summary>
		public int PropVert;

		/// <summary>Creates a halfedge from its four indices.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Halfedge(int startVert, int endVert, int pairedHalfedge, int propVert)
		{
			this.StartVert = startVert;
			this.EndVert = endVert;
			this.PairedHalfedge = pairedHalfedge;
			this.PropVert = propVert;
		}

		/// <summary>True for the halfedge of the pair whose start vertex is the lower index.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public bool IsForward()
		{
			return this.StartVert < this.EndVert;
		}

		/// <summary>Equality over all four fields, the Rust's derived <c>PartialEq</c>/<c>Eq</c>.</summary>
		public static bool operator ==(Halfedge a, Halfedge b)
		{
			return a.StartVert == b.StartVert
				&& a.EndVert == b.EndVert
				&& a.PairedHalfedge == b.PairedHalfedge
				&& a.PropVert == b.PropVert;
		}

		/// <summary>Inequality over all four fields.</summary>
		public static bool operator !=(Halfedge a, Halfedge b)
		{
			return !(a == b);
		}

		/// <summary>Ordering by start vertex, then end vertex — the Rust <c>Ord</c>.</summary>
		public static bool operator <(Halfedge a, Halfedge b)
		{
			return a.CompareTo(b) < 0;
		}

		/// <summary>Ordering by start vertex, then end vertex — the Rust <c>Ord</c>.</summary>
		public static bool operator >(Halfedge a, Halfedge b)
		{
			return a.CompareTo(b) > 0;
		}

		/// <summary>Ordering by start vertex, then end vertex — the Rust <c>Ord</c>.</summary>
		public static bool operator <=(Halfedge a, Halfedge b)
		{
			return a.CompareTo(b) <= 0;
		}

		/// <summary>Ordering by start vertex, then end vertex — the Rust <c>Ord</c>.</summary>
		public static bool operator >=(Halfedge a, Halfedge b)
		{
			return a.CompareTo(b) >= 0;
		}

		/// <summary>
		/// The Rust <c>Ord</c>: start vertex first, end vertex as the tie-break. Two
		/// halfedges that share both ends compare equal here even when their paired
		/// halfedge or property vertex differ, which is what the sort sites rely on.
		/// </summary>
		/// <param name="other">The halfedge to compare against.</param>
		/// <returns>Negative, zero or positive per <see cref="IComparable{T}"/>.</returns>
		public int CompareTo(Halfedge other)
		{
			if (this.StartVert == other.StartVert)
			{
				return this.EndVert.CompareTo(other.EndVert);
			}

			return this.StartVert.CompareTo(other.StartVert);
		}

		/// <summary>Equality over all four fields.</summary>
		public bool Equals(Halfedge other)
		{
			return this == other;
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Halfedge other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.StartVert, this.EndVert, this.PairedHalfedge, this.PropVert);
		}
	}

	/// <summary>
	/// Barycentric coordinates of a point within a triangle, from src/shared.h.
	/// </summary>
	public struct Barycentric : IEquatable<Barycentric>
	{
		/// <summary>Triangle index the coordinates are relative to.</summary>
		public int Tri;

		/// <summary>The barycentric weights.</summary>
		public Vec4 Uvw;

		/// <summary>Creates a barycentric coordinate.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Barycentric(int tri, Vec4 uvw)
		{
			this.Tri = tri;
			this.Uvw = uvw;
		}

		/// <summary>IEEE equality, the Rust's derived <c>PartialEq</c>.</summary>
		public static bool operator ==(Barycentric a, Barycentric b)
		{
			return a.Tri == b.Tri && a.Uvw == b.Uvw;
		}

		/// <summary>IEEE inequality, the Rust's derived <c>PartialEq</c>.</summary>
		public static bool operator !=(Barycentric a, Barycentric b)
		{
			return !(a == b);
		}

		/// <summary>Bit-exact equality; the partner of <see cref="GetHashCode"/>.</summary>
		public bool Equals(Barycentric other)
		{
			return this.Tri == other.Tri && this.Uvw.Equals(other.Uvw);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Barycentric other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.Tri, this.Uvw);
		}
	}

	/// <summary>
	/// Provenance of a triangle: which mesh instance, original mesh, source face and
	/// coplanar group it came from. From src/shared.h.
	/// </summary>
	public struct TriRef : IEquatable<TriRef>
	{
		/// <summary>Unique ID of the mesh instance of this triangle.</summary>
		public int MeshId;

		/// <summary>OriginalID of the mesh this triangle came from.</summary>
		public int OriginalId;

		/// <summary>Source face ID.</summary>
		public int FaceId;

		/// <summary>Triangles with same coplanar_id are coplanar.</summary>
		public int CoplanarId;

		/// <summary>Creates a triangle reference from its four IDs.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public TriRef(int meshId, int originalId, int faceId, int coplanarId)
		{
			this.MeshId = meshId;
			this.OriginalId = originalId;
			this.FaceId = faceId;
			this.CoplanarId = coplanarId;
		}

		/// <summary>Equality over all four fields, the Rust's derived <c>PartialEq</c>/<c>Eq</c>.</summary>
		public static bool operator ==(TriRef a, TriRef b)
		{
			return a.MeshId == b.MeshId
				&& a.OriginalId == b.OriginalId
				&& a.FaceId == b.FaceId
				&& a.CoplanarId == b.CoplanarId;
		}

		/// <summary>Inequality over all four fields.</summary>
		public static bool operator !=(TriRef a, TriRef b)
		{
			return !(a == b);
		}

		/// <summary>
		/// Whether two triangles belong to the same source face: mesh ID, coplanar ID and
		/// face ID must all agree. Deliberately ignores <see cref="OriginalId"/>, matching
		/// the Rust.
		/// </summary>
		/// <param name="other">The triangle reference to compare against.</param>
		/// <returns>True when both triangles came from one face.</returns>
		public bool SameFace(TriRef other)
		{
			return this.MeshId == other.MeshId
				&& this.CoplanarId == other.CoplanarId
				&& this.FaceId == other.FaceId;
		}

		/// <summary>Equality over all four fields.</summary>
		public bool Equals(TriRef other)
		{
			return this == other;
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is TriRef other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.MeshId, this.OriginalId, this.FaceId, this.CoplanarId);
		}
	}

	/// <summary>
	/// An undirected edge in normalized (low, high) form plus the halfedge it came from,
	/// from src/shared.h.
	/// </summary>
	/// <remarks>
	/// Ordering is by <see cref="First"/> then <see cref="Second"/> and never by
	/// <see cref="HalfedgeIdx"/>, so two TmpEdges over the same vertex pair compare equal
	/// however they were built. Downstream sorts depend on that: the tie-break they get
	/// comes from the sort's stability, not from this comparison. Equality, on the other
	/// hand, does look at <see cref="HalfedgeIdx"/> — the same deliberate split the Rust
	/// has between <c>Ord</c> and <c>Eq</c> — so this type sorts correctly but must not be
	/// used as the key of an ordered set.
	/// </remarks>
	public struct TmpEdge : IEquatable<TmpEdge>, IComparable<TmpEdge>
	{
		/// <summary>The lower of the two vertex indices.</summary>
		public int First;

		/// <summary>The higher of the two vertex indices.</summary>
		public int Second;

		/// <summary>Index of the halfedge this edge was built from.</summary>
		public int HalfedgeIdx;

		/// <summary>
		/// Builds an edge from a directed halfedge, normalizing so that
		/// <see cref="First"/> &lt;= <see cref="Second"/> — the Rust <c>TmpEdge::new</c>.
		/// </summary>
		/// <param name="start">Start vertex of the halfedge.</param>
		/// <param name="end">End vertex of the halfedge.</param>
		/// <param name="idx">Index of the halfedge.</param>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public TmpEdge(int start, int end, int idx)
		{
			this.First = Math.Min(start, end);
			this.Second = Math.Max(start, end);
			this.HalfedgeIdx = idx;
		}

		/// <summary>Equality over all three fields, the Rust's derived <c>PartialEq</c>/<c>Eq</c>.</summary>
		public static bool operator ==(TmpEdge a, TmpEdge b)
		{
			return a.First == b.First && a.Second == b.Second && a.HalfedgeIdx == b.HalfedgeIdx;
		}

		/// <summary>Inequality over all three fields.</summary>
		public static bool operator !=(TmpEdge a, TmpEdge b)
		{
			return !(a == b);
		}

		/// <summary>Ordering by first vertex, then second — the Rust <c>Ord</c>.</summary>
		public static bool operator <(TmpEdge a, TmpEdge b)
		{
			return a.CompareTo(b) < 0;
		}

		/// <summary>Ordering by first vertex, then second — the Rust <c>Ord</c>.</summary>
		public static bool operator >(TmpEdge a, TmpEdge b)
		{
			return a.CompareTo(b) > 0;
		}

		/// <summary>Ordering by first vertex, then second — the Rust <c>Ord</c>.</summary>
		public static bool operator <=(TmpEdge a, TmpEdge b)
		{
			return a.CompareTo(b) <= 0;
		}

		/// <summary>Ordering by first vertex, then second — the Rust <c>Ord</c>.</summary>
		public static bool operator >=(TmpEdge a, TmpEdge b)
		{
			return a.CompareTo(b) >= 0;
		}

		/// <summary>
		/// The Rust <c>Ord</c>: first vertex, then second vertex.
		/// <see cref="HalfedgeIdx"/> never participates.
		/// </summary>
		/// <param name="other">The edge to compare against.</param>
		/// <returns>Negative, zero or positive per <see cref="IComparable{T}"/>.</returns>
		public int CompareTo(TmpEdge other)
		{
			if (this.First == other.First)
			{
				return this.Second.CompareTo(other.Second);
			}

			return this.First.CompareTo(other.First);
		}

		/// <summary>Equality over all three fields.</summary>
		public bool Equals(TmpEdge other)
		{
			return this == other;
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is TmpEdge other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.First, this.Second, this.HalfedgeIdx);
		}
	}

	/// <summary>
	/// Transform relation between meshes, from src/impl.h.
	/// </summary>
	/// <remarks>
	/// <c>new Relation()</c> reproduces the Rust <c>Default</c> (original ID -1, identity
	/// transform); <c>default(Relation)</c> and a zeroed array element do <em>not</em> —
	/// they carry original ID 0 and an all-zero matrix. Construct these, never default
	/// them.
	/// </remarks>
	public struct Relation
	{
		/// <summary>OriginalID of the mesh this relation describes; -1 when unset.</summary>
		public int OriginalId;

		/// <summary>Transform from the original mesh's frame into this instance's frame.</summary>
		public Mat3x4 Transform;

		/// <summary>Whether the transform flips orientation, so the mesh is seen back-side.</summary>
		public bool BackSide;

		/// <summary>
		/// True when this meshID's contribution to <c>properties_</c> slots 0..2 holds
		/// world-frame vertex normals (set by <c>CalculateNormals</c> at slot 0). Carries
		/// through Transforms and Booleans; exported as run_flags bit 1. Per C++ #1718.
		/// </summary>
		public bool HasNormals;

		/// <summary>Creates a relation with the Rust <c>Default</c> field values.</summary>
		public Relation()
		{
			this.OriginalId = -1;
			this.Transform = Mat3x4.Identity();
			this.BackSide = false;
			this.HasNormals = false;
		}

		/// <summary>
		/// Normal transform: inverse-transpose of the 3x3 linear part. Multiply
		/// stored-property normals by this to get world-space normals. Matches C++
		/// <c>Relation::GetNormalTransform()</c>.
		/// </summary>
		/// <returns>The normal transform.</returns>
		public Mat3 GetNormalTransform()
		{
			double sign = this.BackSide ? -1.0 : 1.0;

			// NormalTransform(M) = inverse(transpose(M)) = (M^T)^{-1}
			return this.Transform.Rotation().Transpose().Inverse() * sign;
		}

		/// <summary>
		/// Inverse normal transform: transpose of the 3x3 linear part. Multiply
		/// world-space normals by this before storing in properties. Matches C++
		/// <c>Relation::GetInverseNormalTransform()</c>.
		/// </summary>
		/// <returns>The inverse normal transform.</returns>
		public Mat3 GetInverseNormalTransform()
		{
			double sign = this.BackSide ? -1.0 : 1.0;

			// InverseNormalTransform(M) = M^T
			return this.Transform.Rotation().Transpose() * sign;
		}
	}

	/// <summary>
	/// Mesh relation table stored on ManifoldImpl.
	/// </summary>
	/// <remarks>
	/// The Rust is a <c>Clone</c> struct; here it is a class, because it owns two growable
	/// collections and every use site holds it by reference as part of the impl. Copies
	/// therefore have to be explicit — see <see cref="Clone"/>.
	/// <para>
	/// <b><see cref="Relation"/> stays a struct</b>, as it is in the Rust, so the values in
	/// <see cref="MeshIdTransform"/> are copies and not shared objects. The two Rust sites
	/// that mutate a relation in place through <c>iter_mut</c> / <c>values_mut</c> —
	/// impl_mesh.rs:889 (composing a transform into every relation) and
	/// manifold_smooth.rs:32 (setting <c>has_normals</c> on every relation) — must
	/// therefore port as read-modify-store: read the value out, change the copy, assign it
	/// back with <c>MeshIdTransform[key] = modified</c>. Forgetting that is not a silent
	/// bug: writing to <c>MeshIdTransform[key].Transform</c> directly is CS1612 ("cannot
	/// modify the return value ... because it is not a variable"), a compile error. Making
	/// this a class would trade that loud failure for a quiet aliasing change, which is
	/// why it stays a struct.
	/// </para>
	/// </remarks>
	public sealed class MeshRelationD
	{
		/// <summary>Creates an empty relation table, the Rust <c>MeshRelationD::new</c>.</summary>
		/// <remarks>
		/// The Rust type also derives <c>Default</c>, which would leave
		/// <see cref="OriginalId"/> at 0; nothing calls it, and <c>new()</c>'s -1 is the
		/// meaningful value ("not an original"), so the constructor is <c>new()</c>.
		/// </remarks>
		public MeshRelationD()
		{
			this.OriginalId = -1;
			this.MeshIdTransform = new SortedDictionary<int, Relation>();
			this.TriRef = new List<TriRef>();
		}

		/// <summary>originalID of this Manifold if it is an original; -1 otherwise.</summary>
		public int OriginalId { get; set; }

		/// <summary>
		/// Per-meshID transform relations, ordered by meshID.
		/// </summary>
		/// <remarks>
		/// C++ uses std::map (ordered by meshID); several sites iterate this map and feed
		/// the order into output runs and fresh-ID assignment, so an unordered map here
		/// breaks determinism and C++ parity. <see cref="SortedDictionary{TKey, TValue}"/>
		/// is the ordered-iteration equivalent of the Rust <c>BTreeMap</c>;
		/// <see cref="Dictionary{TKey, TValue}"/> is not a substitute.
		/// </remarks>
		public SortedDictionary<int, Relation> MeshIdTransform { get; }

		/// <summary>Provenance of every triangle, indexed by triangle.</summary>
		public List<TriRef> TriRef { get; }

		/// <summary>
		/// Deep copy, the Rust's derived <c>Clone</c>: the map and the list are copied, so
		/// the clone shares nothing with the original.
		/// </summary>
		/// <returns>An independent copy of this table.</returns>
		public MeshRelationD Clone()
		{
			MeshRelationD copy = new MeshRelationD();
			copy.OriginalId = this.OriginalId;
			foreach (KeyValuePair<int, Relation> entry in this.MeshIdTransform)
			{
				copy.MeshIdTransform.Add(entry.Key, entry.Value);
			}

			copy.TriRef.AddRange(this.TriRef);
			return copy;
		}
	}

	/// <content>
	/// The inline halfedge index utilities of src/shared.h.
	/// </content>
	public static partial class Types
	{
		/// <summary>
		/// Return next halfedge index within the same triangle (wraps 0-&gt;1-&gt;2-&gt;0).
		/// </summary>
		/// <param name="current">The halfedge index.</param>
		/// <returns>The next halfedge index in the same triangle.</returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static int NextHalfedge(int current)
		{
			int n = current + 1;
			return n % 3 == 0 ? n - 3 : n;
		}

		/// <summary>
		/// Returns the previous halfedge index within the same triangle.
		/// For triangle t: PrevHalfedge(3t+i) = 3t + (i+2)%3.
		/// </summary>
		/// <param name="current">The halfedge index.</param>
		/// <returns>The previous halfedge index in the same triangle.</returns>
		public static int PrevHalfedge(int current)
		{
			int baseIdx = current - (current % 3);
			int pos = ((current % 3) + 2) % 3;
			return baseIdx + pos;
		}

		/// <summary>Return next index within 0..3 (wraps 0-&gt;1-&gt;2-&gt;0).</summary>
		/// <param name="i">The index.</param>
		/// <returns>The next index.</returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static int Next3(int i)
		{
			int n = i + 1;
			return n == 3 ? 0 : n;
		}
	}
}
