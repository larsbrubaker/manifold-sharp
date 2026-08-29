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

// Phase 4: Mesh Data Structure — ported from src/impl.h, src/impl.cpp,
// src/properties.cpp
//
// This module implements the core ManifoldImpl struct: halfedge mesh
// representation, bounding box, epsilon, manifold checks, and shape
// constructors.
//
// ── File split ───────────────────────────────────────────────────────────────
// impl_mesh.rs is 1,061 lines; its C# expansion does not fit the 800-line cap,
// so it lands as three partial files (plus the `impl ManifoldImpl` block that
// collider.rs carries, which stays with its own module in Collider.Geometry.cs):
//   ManifoldImpl.cs           this file — the mesh-ID counter, the helpers, the
//                             fields, the accessors, MakeEmpty, ForVert,
//                             CalculateBBox, SetEpsilon and the manifold checks
//   ManifoldImpl.Topology.cs  CreateHalfedges, InitializeOriginal, the #1718
//                             normal bookkeeping, IncrementMeshIds,
//                             DedupePropVerts, RemoveUnreferencedVerts
//   ManifoldImpl.Shapes.cs    the primitive constructors and Transform
//
// ── Module-level Rust items land as statics on this type ─────────────────────
// The naming convention in CLAUDE.md puts a Rust module's free functions on a
// static class named for the module — `ImplMesh` here. That name sits one letter
// from `ManifoldImpl` and would be misread at every call site, so `reserve_ids`,
// `K_REMOVED_HALFEDGE`, `next_halfedge` and `next3` are static members of
// `ManifoldImpl` instead. This is the port's one deviation from that convention,
// and it is confined to this module.
//
// Note that impl_mesh.rs defines its own `next_halfedge` / `next3` even though
// types.rs already has both; the duplication is the Rust's, and both copies are
// kept so a later port of either module transcribes against the one its Rust
// source imports. `ManifoldImpl.Next3` takes and returns the Rust's `usize`
// (an int here), while `Types.Next3` takes the i32 — that difference is real.
//
// ── Fields, not properties ───────────────────────────────────────────────────
// Every piece of state below is a public field, matching the Rust's `pub`
// fields. That is not a style preference: `Bbox` is a `Box`, whose `UnionPoint`
// mutates in place, and through a property that call would compile and silently
// grow a temporary (the trap the Bounds.cs header names). A field access is a
// variable; a property access is not.
//
// The `Vec<T>` fields are `List<T>` — they are pushed into and resized all over
// the later phases — and mutation-in-place of a `List<Halfedge>` element goes
// through `CollectionsMarshal.AsSpan`, which is the literal port of Rust's
// `iter_mut()`. Writing `list[i].Field = x` instead is a compile error (CS1612),
// which is the loud failure this arrangement is chosen for.

using System.Threading;

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// Internal halfedge mesh representation, mirroring <c>Manifold::Impl</c> in C++.
	/// </summary>
	public sealed partial class ManifoldImpl
	{
		/// <summary>
		/// The sentinel <see cref="Halfedge.PairedHalfedge"/> value marking a halfedge that
		/// <see cref="CreateHalfedges"/> found to belong to a pair of opposed triangles and
		/// will delete.
		/// </summary>
		public const int KRemovedHalfedge = -2;

		// Rust `f32::EPSILON as f64` — 2^-23. C# `float.Epsilon` is the smallest
		// *subnormal* float and is wrong here; see the f64 counterpart of this rule in
		// docs/PORTING_PLAN.md.
		private const double F32Epsilon = 1.1920928955078125E-07;

		// ---------------------------------------------------------------------
		// Global mesh ID counter (mirrors Manifold::Impl::meshIDCounter_)
		// ---------------------------------------------------------------------
		private static uint meshIdCounter = 1;

		/// <summary>The bounding box of <see cref="VertPos"/>.</summary>
		public Box Bbox;

		/// <summary>The geometric tolerance below which vertices are considered coincident.</summary>
		public double Epsilon;

		/// <summary>The simplification tolerance carried into and out of MeshGL.</summary>
		public double Tolerance;

		/// <summary>Number of property channels per property vertex; 0 when there are none.</summary>
		public int NumProp;

		/// <summary>The error status this impl carries instead of throwing.</summary>
		public Error Status;

		/// <summary>Vertex positions, indexed by vertex.</summary>
		public List<Vec3> VertPos;

		/// <summary>The halfedge arena: three consecutive entries per triangle.</summary>
		public List<Halfedge> Halfedge;

		/// <summary>Flat interleaved property values, <see cref="NumProp"/> per property vertex.</summary>
		public List<double> Properties;

		/// <summary>Per-vertex normals, empty when not computed.</summary>
		public List<Vec3> VertNormal;

		/// <summary>Per-face normals, empty when not computed.</summary>
		public List<Vec3> FaceNormal;

		/// <summary>Per-halfedge tangents for smoothing, empty when not computed.</summary>
		public List<Vec4> HalfedgeTangent;

		/// <summary>Triangle provenance and the per-meshID transform table.</summary>
		public MeshRelationD MeshRelation;

		/// <summary>
		/// Cached face BVH, built by <see cref="Sort.SortGeometry"/> and updated (not
		/// rebuilt) on transform — mirrors C++ <c>Impl::collider_</c>. Query sites (boolean
		/// kernels, ray cast, face merging, self-intersection) use this instead of
		/// rebuilding the tree per query.
		/// </summary>
		public Collider Collider;

		/// <summary>
		/// True when this impl carries geometrically closed but topologically non-manifold
		/// "triangle soup" imported via <c>FromMeshGLRobust</c>: halfedge pairing is
		/// incomplete (<c>PairedHalfedge == -1</c> permitted).
		/// </summary>
		/// <remarks>
		/// Only the robust boolean engine, transforms, bbox, and MeshGL export accept soup
		/// impls; pairing-dependent operations return an empty result with
		/// <see cref="Error.NotManifold"/>. Always false on the strict import path, so
		/// existing behavior is unchanged.
		/// </remarks>
		public bool IsSoup;

		/// <summary>
		/// Lazily-resolved verdict of <c>robust::soup::has_self_intersections</c> — whether
		/// this impl's own triangles genuinely intersect rather than merely sharing
		/// edges/vertices. Computed at most once per impl (the scan is a full BVH
		/// self-query) and consulted by <c>Auto</c> boolean dispatch, which must route
		/// geometrically self-intersecting operands to the robust engine even when their
		/// connectivity is manifold.
		/// </summary>
		/// <remarks>
		/// Write-once and thread-safe so it can be filled through a shared impl from
		/// parallel workers. Assembly-private deliberately (the Rust field is
		/// <c>pub(crate)</c>): <see cref="SelfIntersectCache.Clone"/> copies the settled
		/// value, so <b>any</b> code that clones an impl and then edits its geometry in
		/// place must call <see cref="InvalidateSelfIntersects"/>. Rebuilds that go through
		/// <see cref="CreateHalfedges"/> or <see cref="MakeEmpty"/> are covered
		/// automatically.
		/// </remarks>
		internal SelfIntersectCache SelfIntersects;

		/// <summary>Creates an empty impl — the Rust <c>Default</c> / <c>ManifoldImpl::new</c>.</summary>
		public ManifoldImpl()
		{
			this.Bbox = new Box();
			this.Epsilon = -1.0;
			this.Tolerance = -1.0;
			this.NumProp = 0;
			this.Status = Error.NoError;
			this.VertPos = new List<Vec3>();
			this.Halfedge = new List<Halfedge>();
			this.Properties = new List<double>();
			this.VertNormal = new List<Vec3>();
			this.FaceNormal = new List<Vec3>();
			this.HalfedgeTangent = new List<Vec4>();
			this.MeshRelation = new MeshRelationD();
			this.Collider = new Collider();
			this.IsSoup = false;
			this.SelfIntersects = new SelfIntersectCache();
		}

		/// <summary>
		/// Reserve <paramref name="n"/> fresh mesh IDs from the process-global counter and
		/// return the first of them — the Rust <c>reserve_ids</c>, mirroring C++
		/// <c>Manifold::Impl::meshIDCounter_</c>.
		/// </summary>
		/// <param name="n">How many consecutive IDs to reserve.</param>
		/// <returns>The first reserved ID.</returns>
		public static uint ReserveIds(uint n)
		{
			// Rust `fetch_add` returns the value *before* the add; Interlocked.Add returns
			// the value after, so subtract the increment back off.
			return unchecked(Interlocked.Add(ref meshIdCounter, n) - n);
		}

		/// <summary>Next halfedge within the same triangle: 0-&gt;1-&gt;2-&gt;0.</summary>
		/// <param name="current">The halfedge index.</param>
		/// <returns>The next halfedge index in the same triangle.</returns>
		public static int NextHalfedge(int current)
		{
			int n = current + 1;
			return n % 3 == 0 ? n - 3 : n;
		}

		/// <summary>Next index within 0..3 (wraps 0-&gt;1-&gt;2-&gt;0).</summary>
		/// <param name="i">The index.</param>
		/// <returns>The next index.</returns>
		public static int Next3(int i)
		{
			return i == 2 ? 0 : i + 1;
		}

		// ---------------------------------------------------------------------
		// Basic accessors
		// ---------------------------------------------------------------------

		/// <summary>The number of vertices.</summary>
		/// <returns>The vertex count.</returns>
		public int NumVert()
		{
			return this.VertPos.Count;
		}

		/// <summary>The number of halfedges.</summary>
		/// <returns>The halfedge count.</returns>
		public int NumHalfedge()
		{
			return this.Halfedge.Count;
		}

		/// <summary>The number of undirected edges.</summary>
		/// <returns>Half the halfedge count.</returns>
		public int NumEdge()
		{
			return this.Halfedge.Count / 2;
		}

		/// <summary>The number of triangles.</summary>
		/// <returns>A third of the halfedge count.</returns>
		public int NumTri()
		{
			return this.Halfedge.Count / 3;
		}

		/// <summary>The number of property vertices.</summary>
		/// <returns>The vertex count when there are no properties, else the property groups.</returns>
		public int NumPropVert()
		{
			if (this.NumProp == 0)
			{
				return this.NumVert();
			}

			return this.Properties.Count / this.NumProp;
		}

		/// <summary>True when the mesh has no triangles.</summary>
		/// <returns>True when empty.</returns>
		public bool IsEmpty()
		{
			return this.NumTri() == 0;
		}

		// ---------------------------------------------------------------------
		// MakeEmpty
		// ---------------------------------------------------------------------

		/// <summary>Discards all geometry and records <paramref name="status"/>.</summary>
		/// <param name="status">The status the emptied impl carries.</param>
		public void MakeEmpty(Error status)
		{
			this.Bbox = new Box();
			this.VertPos.Clear();
			this.Halfedge.Clear();
			this.VertNormal.Clear();
			this.FaceNormal.Clear();
			this.HalfedgeTangent.Clear();
			this.MeshRelation = new MeshRelationD();
			this.Collider = new Collider();
			this.Status = status;
			this.IsSoup = false;

			// Geometry is gone; any cached self-intersection verdict is stale.
			this.InvalidateSelfIntersects();
		}

		/// <summary>
		/// Drop any cached self-intersection verdict. Must be called by every operation that
		/// edits <see cref="VertPos"/> or <see cref="Halfedge"/> in place on an impl it
		/// cloned (the cache is copied by <see cref="Clone"/>); rebuilds through
		/// <see cref="CreateHalfedges"/> and <see cref="MakeEmpty"/> do it themselves.
		/// </summary>
		public void InvalidateSelfIntersects()
		{
			this.SelfIntersects = new SelfIntersectCache();
		}

		// ---------------------------------------------------------------------
		// ForVert — iterate halfedges around a vertex
		// ---------------------------------------------------------------------

		/// <summary>
		/// Apply <paramref name="func"/> to each halfedge index around the vertex starting
		/// from <paramref name="halfedgeIdx"/>.
		/// </summary>
		/// <param name="halfedgeIdx">The halfedge to start (and finish) at.</param>
		/// <param name="func">Called once per halfedge around the vertex.</param>
		public void ForVert(int halfedgeIdx, Action<int> func)
		{
			int current = halfedgeIdx;
			while (true)
			{
				current = NextHalfedge(this.Halfedge[current].PairedHalfedge);
				func(current);
				if (current == halfedgeIdx)
				{
					break;
				}
			}
		}

		// ---------------------------------------------------------------------
		// CalculateBBox
		// ---------------------------------------------------------------------

		/// <summary>Recomputes <see cref="Bbox"/> from the vertex positions.</summary>
		public void CalculateBBox()
		{
			Box bbox = new Box();
			foreach (Vec3 v in this.VertPos)
			{
				if (!double.IsNaN(v.X))
				{
					bbox.UnionPoint(v);
				}
			}

			this.Bbox = bbox;
			if (!this.Bbox.IsFinite())
			{
				this.MakeEmpty(Error.NoError);
			}
		}

		// ---------------------------------------------------------------------
		// SetEpsilon
		// ---------------------------------------------------------------------

		/// <summary>
		/// Sets <see cref="Epsilon"/> from a floor and the bounding-box scale, and raises
		/// <see cref="Tolerance"/> to match.
		/// </summary>
		/// <param name="minEpsilon">The smallest epsilon to accept; -1 for "no floor".</param>
		/// <param name="useSingle">
		/// True when the mesh will round-trip through single precision, which floors the
		/// tolerance at <c>f32::EPSILON * bbox.scale()</c>.
		/// </param>
		public void SetEpsilon(double minEpsilon, bool useSingle)
		{
			this.Epsilon = MaxEpsilon(minEpsilon, this.Bbox);
			double minTol = this.Epsilon;
			if (useSingle)
			{
				double floatEps = F32Epsilon * this.Bbox.Scale();
				minTol = MaxF64(minTol, floatEps);
			}

			this.Tolerance = MaxF64(this.Tolerance, minTol);
		}

		// ---------------------------------------------------------------------
		// IsFinite
		// ---------------------------------------------------------------------

		/// <summary>True when every vertex position is finite.</summary>
		/// <returns>True when no coordinate is infinite or NaN.</returns>
		public bool IsFinite()
		{
			foreach (Vec3 v in this.VertPos)
			{
				if (!(double.IsFinite(v.X) && double.IsFinite(v.Y) && double.IsFinite(v.Z)))
				{
					return false;
				}
			}

			return true;
		}

		// ---------------------------------------------------------------------
		// IsManifold / Is2Manifold
		// ---------------------------------------------------------------------

		/// <summary>
		/// Check that the halfedge data structure is consistent (oriented even manifold).
		/// </summary>
		/// <returns>True when the halfedge pairing is consistent.</returns>
		public bool IsManifold()
		{
			if (this.Halfedge.Count == 0)
			{
				return true;
			}

			if (this.Halfedge.Count % 3 != 0)
			{
				return false;
			}

			for (int edge = 0; edge < this.Halfedge.Count; edge++)
			{
				Halfedge h = this.Halfedge[edge];

				// Valid removed halfedge
				if (h.StartVert == -1 && h.EndVert == -1 && h.PairedHalfedge == -1)
				{
					continue;
				}

				// Neighbors in same triangle must not be removed
				int n1 = NextHalfedge(edge);
				int n2 = NextHalfedge(n1);
				if (this.Halfedge[n1].StartVert == -1 || this.Halfedge[n2].StartVert == -1)
				{
					return false;
				}

				if (h.PairedHalfedge == -1)
				{
					return false;
				}

				int pairedIdx = h.PairedHalfedge;
				Halfedge paired = this.Halfedge[pairedIdx];
				if (paired.PairedHalfedge != edge)
				{
					return false;
				}

				if (h.StartVert == h.EndVert)
				{
					return false;
				}

				if (h.StartVert != paired.EndVert)
				{
					return false;
				}

				if (h.EndVert != paired.StartVert)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>Check that the mesh is a 2-manifold (no duplicate edges).</summary>
		/// <returns>True when no undirected edge appears twice.</returns>
		public bool Is2Manifold()
		{
			if (this.Halfedge.Count == 0)
			{
				return true;
			}

			if (!this.IsManifold())
			{
				return false;
			}

			// Sort halfedges and check for duplicates.
			// SORT AUDIT (impl_mesh.rs:312): Rust `sort_unstable`, and unstable is correct
			// here. `Halfedge`'s ordering is (StartVert, EndVert) only, and the scan below
			// compares nothing else — so the permutation chosen among entries with equal
			// (start, end) cannot change the answer. Removed halfedges all carry
			// (-1, -1) and are skipped by the same key, so they cannot be split across a
			// tie either. List<T>.Sort (introsort, unstable) is therefore admissible, and
			// this is one of the audited sites docs/PORTING_PLAN.md allows it at.
			List<Halfedge> sorted = new List<Halfedge>(this.Halfedge);
			sorted.Sort();
			for (int i = 0; i + 1 < sorted.Count; i++)
			{
				Halfedge h = sorted[i];
				Halfedge h1 = sorted[i + 1];

				// Skip removed halfedges
				if (h.StartVert == -1 && h.EndVert == -1 && h.PairedHalfedge == -1)
				{
					continue;
				}

				if (h.StartVert == h1.StartVert && h.EndVert == h1.EndVert)
				{
					return false; // Duplicate edge
				}
			}

			return true;
		}

		// ---------------------------------------------------------------------
		// SetNormalsAndCoplanar
		// ---------------------------------------------------------------------

		/// <summary>Compute face normals, assign coplanar IDs, and calculate vertex normals.</summary>
		/// <remarks>
		/// impl_mesh.rs pairs this one-line delegator with the <c>sort_geometry</c> one; that
		/// twin landed in ManifoldImpl.Topology.cs with the rest of the topology half.
		/// </remarks>
		public void SetNormalsAndCoplanar()
		{
			FaceOp.SetNormalsAndCoplanar(this);
		}

		// ---------------------------------------------------------------------
		// Helpers
		// ---------------------------------------------------------------------

		/// <summary>Safe normalize: returns zero vector if input is zero or non-finite.</summary>
		private static Vec3 SafeNormalize(Vec3 v)
		{
			Vec3 n = Normalize(v);
			if (double.IsFinite(n.X))
			{
				return n;
			}

			return new Vec3(0.0, 0.0, 0.0);
		}

		private static double MaxEpsilon(double minEpsilon, Box bbox)
		{
			double epsilon = MaxF64(minEpsilon, Types.KPrecision * bbox.Scale());
			if (double.IsFinite(epsilon))
			{
				return epsilon;
			}

			return -1.0;
		}
	}
}
