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

// Manifold.cs — port of manifold.rs: the public façade over ManifoldImpl.
//
// manifold.rs itself carries no header beyond its licence; what it *does* carry
// is the module wiring at the bottom, which is the map of this port:
//
//   #[path = "manifold_meshgl.rs"] mod meshgl;   -> Manifold.MeshGL*.cs
//   #[path = "manifold_shape.rs"]  mod shape;    -> Manifold.Shapes.cs
//   #[path = "manifold_smooth.rs"] mod smooth;   -> Manifold.Smooth.cs
//
// Those are child modules of `manifold` so their methods keep access to the
// private `imp` field; here they are `partial class Manifold` continuations,
// which is the same arrangement.
//
// ── Files ────────────────────────────────────────────────────────────────────
//   Manifold.cs             this file — construction, queries, transforms
//   Manifold.Regions.cs     split/trim/slice/project, the robust repairs,
//                           decompose, ray cast, and the halfspace helper
//   Manifold.Booleans.cs    the boolean family (every entry point folds into
//                           `BooleanWithEngineRuleAndProgress`), Compose,
//                           SetProperties, MinGap, and the operator overloads
//   Manifold.Shapes.cs      manifold_shape.rs — primitives, hulls, Minkowski
//   Manifold.Smooth.cs      manifold_smooth.rs — CalculateNormals, Smooth,
//                           SmoothOut, SmoothByNormals and the Refine family
//   Manifold.MeshGL.cs      manifold_meshgl.rs import half + OBJ read/write
//   Manifold.MeshGL.Export.cs   manifold_meshgl.rs export half
//   MeshGLAccess.cs         the narrow conversion interface that lets the two
//                           generic bodies above have one C# body each
//
// ── Value semantics, and why Clone() returns `this` ──────────────────────────
// The Rust `Manifold` is `#[derive(Clone)]` over an owned `ManifoldImpl`, so
// `self.clone()` deep-copies the mesh. Here `Manifold` is a sealed class holding
// one private impl reference, and `Clone()` returns `this`. That is
// observationally identical *because the façade never mutates an impl it did not
// just create*: every method that changes anything opens with
// `ManifoldImpl outR = this.imp.Clone();` exactly as the Rust does with
// `let mut out = self.imp.clone();`. Sharing an immutable value is the same as
// copying it, and this is the same bargain CsgLeafNode already struck for the
// Rust's `Arc<ManifoldImpl>` (see its remarks) — with the same caveat: a caller
// that reaches through <see cref="AsImpl"/> and edits the mesh changes every
// Manifold sharing it. Treat an impl handed to a Manifold as given away.
//
// ── Errors are status, not exceptions ────────────────────────────────────────
// Per docs/PORTING_PLAN.md, a failed operation returns an *empty* Manifold whose
// Status() names the failure; nothing here throws for bad geometry. The only
// exceptions in this file are `NotSupportedException` on the paths whose Rust
// body calls into a module this phase does not have yet, and those are marked
// DEFERRED with the phase that lands them — never silently wrong answers.
//
// ── DEFERRED across the façade (greppable) ───────────────────────────────────
// All four of manifold.rs's files are now ported. What is left is not a file but
// four methods, and every one of them is waiting on Phase 10's robust engine:
//
//   Manifold.Regions.cs  HasSelfIntersections, RepairOrientation, RebuildSolid,
//                        RebuildSolidWithToken
//                        (robust::soup / robust::repair / robust::rebuild_with_rule)
//   Manifold.MeshGL.cs   the robust import's non-manifold leg (robust::soup::soupify)
//
// Each file carries the same list in its own header; this is the union, and it is
// the whole of it. Slice and Project are NOT on it — they had their own blocker,
// a decision rather than a phase (FaceOp.cs, the seed-triangle HashSet order),
// and that decision has since been made, so both are real here.
//
// Nothing else in the façade is deferred: manifold_shape.rs runs Sphere and the
// LevelSet family straight into Phase 7's ManifoldImpl.Subdivide and
// Sdf.LevelSet, and manifold_smooth.rs is Manifold.Smooth.cs.
//
// The thirteen tests those four files used to defer on manifold_smooth.rs are
// written and green; smooth.rs's own module is ManifoldSmoothTests.cs. Test
// deferrals that remain are a separate list and live where they belong, in the
// DEFERRED table of the test file that owns them.

using System.Runtime.InteropServices;

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// A watertight, oriented triangle mesh, and the operations on it — the library's
	/// public entry point.
	/// </summary>
	/// <remarks>
	/// Every operation returns a new <see cref="Manifold"/>; nothing mutates in place.
	/// A failed operation returns an empty manifold whose <see cref="Status"/> names the
	/// error, matching the Rust and C++ contract.
	/// </remarks>
	public sealed partial class Manifold
	{
		private readonly ManifoldImpl imp;

		/// <summary>Creates the empty manifold — the Rust <c>Manifold::new</c>.</summary>
		public Manifold()
		{
			this.imp = new ManifoldImpl();
		}

		private Manifold(ManifoldImpl impl)
		{
			this.imp = impl;
		}

		/// <summary>
		/// The vertex-position mutator handed to <see cref="Warp"/>.
		/// </summary>
		/// <remarks>
		/// The Rust closure takes <c>&amp;mut Vec3</c> and edits in place rather than
		/// returning a new one, so the C# delegate takes <c>ref Vec3</c> and returns void
		/// — the same shape CrossSection.WarpFunc uses for its 2D counterpart.
		/// </remarks>
		/// <param name="v">The vertex position to modify in place.</param>
		public delegate void WarpFunc(ref Vec3 v);

		/// <summary>
		/// The whole-array vertex mutator handed to <see cref="WarpBatch"/>.
		/// </summary>
		/// <remarks>
		/// The Rust closure takes <c>&amp;mut [Vec3]</c> — one call for the entire vertex
		/// buffer, which is the point of the batch form. <see cref="Span{T}"/> is the C#
		/// type with that meaning; the span is the live backing store of the result's
		/// vertex list, so writes through it land.
		/// </remarks>
		/// <param name="verts">Every vertex position, writable in place.</param>
		public delegate void WarpBatchFunc(Span<Vec3> verts);

		/// <summary>
		/// The per-vertex property writer handed to <see cref="SetProperties"/>.
		/// </summary>
		/// <remarks>
		/// The Rust closure is <c>Fn(&amp;mut [f64], Vec3, &amp;[f64])</c>: write the new
		/// property values, given the vertex position and the old property values. The old
		/// slice is empty when the mesh had no properties.
		/// </remarks>
		/// <param name="newProps">The new property slots for this vertex, to be written.</param>
		/// <param name="position">The vertex position.</param>
		/// <param name="oldProps">The vertex's previous property values, possibly empty.</param>
		public delegate void SetPropertiesFunc(
			Span<double> newProps,
			Vec3 position,
			ReadOnlySpan<double> oldProps);

		/// <summary>The empty manifold — the Rust <c>Manifold::empty</c>.</summary>
		/// <returns>An empty manifold with <see cref="Error.NoError"/>.</returns>
		public static Manifold Empty()
		{
			return new Manifold();
		}

		/// <summary>Create an empty manifold with a specific error status.</summary>
		/// <param name="status">The status to report.</param>
		/// <returns>The empty, errored manifold.</returns>
		public static Manifold MakeEmpty(Error status)
		{
			ManifoldImpl impl = new ManifoldImpl();
			impl.MakeEmpty(status);
			return new Manifold(impl);
		}

		/// <summary>Wraps an impl as a Manifold — the Rust <c>Manifold::from_impl</c>.</summary>
		/// <param name="impl">The impl to wrap; taken by reference, never copied.</param>
		/// <returns>The wrapping manifold.</returns>
		public static Manifold FromImpl(ManifoldImpl impl)
		{
			ArgumentNullException.ThrowIfNull(impl);
			return new Manifold(impl);
		}

		/// <summary>Port of C++ <c>Manifold::ReserveIDs()</c>.</summary>
		/// <param name="n">How many consecutive IDs to reserve.</param>
		/// <returns>The first reserved ID.</returns>
		public static uint ReserveIds(uint n)
		{
			return ManifoldImpl.ReserveIds(n);
		}

		/// <summary>
		/// The wrapped impl, borrowed — the Rust <c>as_impl</c>, which hands out
		/// <c>&amp;ManifoldImpl</c>. C# cannot express the immutable borrow, so the
		/// contract is by convention: read it, never write it. See the file header.
		/// </summary>
		/// <returns>The wrapped impl.</returns>
		public ManifoldImpl AsImpl()
		{
			return this.imp;
		}

		/// <summary>
		/// The wrapped impl, owned — the Rust <c>into_impl</c>, which *moves* the impl out
		/// and consumes the Manifold. C# has no move, so this returns an independent copy;
		/// that is the only way the caller can own it without also aliasing this Manifold.
		/// </summary>
		/// <returns>A copy of the wrapped impl.</returns>
		public ManifoldImpl IntoImpl()
		{
			return this.imp.Clone();
		}

		/// <summary>
		/// The Rust's derived <c>Clone</c>. Returns <c>this</c>: a Manifold is immutable,
		/// so sharing it is copying it. See the file header for the one caveat.
		/// </summary>
		/// <returns>This manifold.</returns>
		public Manifold Clone()
		{
			return this;
		}

		/// <summary>The number of vertices.</summary>
		/// <returns>The vertex count.</returns>
		public int NumVert()
		{
			return this.imp.NumVert();
		}

		/// <summary>The number of triangles.</summary>
		/// <returns>The triangle count.</returns>
		public int NumTri()
		{
			return this.imp.NumTri();
		}

		/// <summary>The number of edges.</summary>
		/// <returns>The edge count.</returns>
		public int NumEdge()
		{
			return this.imp.NumEdge();
		}

		/// <summary>The number of extra properties per vertex, beyond position.</summary>
		/// <returns>The extra-property count.</returns>
		public int NumProp()
		{
			return this.imp.NumProp;
		}

		/// <summary>The number of property vertices.</summary>
		/// <returns>The property-vertex count.</returns>
		public int NumPropVert()
		{
			return this.imp.NumPropVert();
		}

		/// <summary>Whether this manifold has no geometry.</summary>
		/// <returns>True when empty.</returns>
		public bool IsEmpty()
		{
			return this.imp.IsEmpty();
		}

		/// <summary>The error status of the operation that produced this manifold.</summary>
		/// <returns>The status.</returns>
		public Error Status()
		{
			return this.imp.Status;
		}

		/// <summary>The enclosed volume.</summary>
		/// <returns>The absolute volume.</returns>
		public double Volume()
		{
			return Math.Abs(this.imp.GetProperty(Property.Volume));
		}

		/// <summary>The total surface area.</summary>
		/// <returns>The surface area.</returns>
		public double SurfaceArea()
		{
			return this.imp.GetProperty(Property.SurfaceArea);
		}

		/// <summary>Whether every triangle's winding agrees with its stored face normal.</summary>
		/// <returns>True when they all agree.</returns>
		public bool MatchesTriNormals()
		{
			return this.imp.MatchesTriNormals();
		}

		/// <summary>The number of degenerate (zero-area) triangles.</summary>
		/// <returns>The degenerate-triangle count.</returns>
		public int NumDegenerateTris()
		{
			return this.imp.NumDegenerateTris();
		}

		/// <summary>The mesh tolerance — the scale below which geometry is merged.</summary>
		/// <returns>The tolerance.</returns>
		public double GetTolerance()
		{
			return this.imp.Tolerance;
		}

		/// <summary>The mesh epsilon — the floating-point resolution of its bounding box.</summary>
		/// <returns>The epsilon.</returns>
		public double GetEpsilon()
		{
			return this.imp.Epsilon;
		}

		/// <summary>Port of C++ <c>Manifold::Genus()</c>.</summary>
		/// <returns>The topological genus.</returns>
		public int Genus()
		{
			int chi = this.NumVert() - this.imp.NumEdge() + this.NumTri();
			return 1 - (chi / 2);
		}

		/// <summary>Port of C++ <c>Manifold::OriginalID()</c>.</summary>
		/// <returns>The original mesh ID, or -1 when this is not an original.</returns>
		public int OriginalId()
		{
			return this.imp.MeshRelation.OriginalId;
		}

		/// <summary>
		/// Port of C++ <c>Manifold::AsOriginal()</c>.
		/// Removes all mesh relations and recreates as an original mesh.
		/// </summary>
		/// <returns>The manifold as a fresh original.</returns>
		public Manifold AsOriginal()
		{
			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			if (this.IsEmpty())
			{
				return this.Clone();
			}

			ManifoldImpl outR = this.imp.Clone();
			outR.InitializeOriginal();
			outR.SetNormalsAndCoplanar();
			return FromImpl(outR);
		}

		/// <summary>Port of C++ <c>Manifold::SetTolerance()</c>.</summary>
		/// <param name="tolerance">The new tolerance.</param>
		/// <returns>A manifold with the tolerance applied.</returns>
		public Manifold SetTolerance(double tolerance)
		{
			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			if (this.IsEmpty())
			{
				return this.Clone();
			}

			ManifoldImpl outR = this.imp.Clone();

			// Matches C++ SetTolerance: operate on the `tolerance` field (which
			// drives coplanar grouping in mark_coplanar), not `epsilon`. When
			// raising tolerance, recompute coplanar groups *first* so the
			// colinear-edge collapse in simplify_topology can see the newly
			// co-planar regions, then simplify, then re-sort.
			if (tolerance > outR.Tolerance)
			{
				outR.Tolerance = tolerance;
				outR.SetNormalsAndCoplanar();
				EdgeOp.SimplifyTopology(outR, 0);
				outR.SortGeometry();

				// Collapsed edges move geometry; the cloned verdict is stale.
				outR.InvalidateSelfIntersects();
			}
			else
			{
				// Reducing tolerance: keep it at least epsilon.
				outR.Tolerance = Math.Max(outR.Epsilon, tolerance);
			}

			return FromImpl(outR);
		}

		/// <summary>Port of C++ <c>Manifold::Simplify()</c>.</summary>
		/// <param name="tolerance">The simplification tolerance; 0 means "use the mesh's own".</param>
		/// <returns>The simplified manifold.</returns>
		public Manifold Simplify(double tolerance)
		{
			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			if (this.IsEmpty())
			{
				return this.Clone();
			}

			ManifoldImpl outR = this.imp.Clone();

			// C++ uses tolerance_ (not epsilon_) throughout Simplify()
			double oldTolerance = outR.Tolerance;
			double tol = tolerance;
			if (tol == 0.0)
			{
				tol = oldTolerance;
			}

			if (tol > oldTolerance)
			{
				outR.Tolerance = tol;
				outR.SetNormalsAndCoplanar();
			}

			EdgeOp.SimplifyTopology(outR, 0);
			outR.SortGeometry();

			// Collapsed edges move geometry; the cloned verdict is stale.
			outR.InvalidateSelfIntersects();
			outR.Tolerance = oldTolerance;
			return FromImpl(outR);
		}

		/// <summary>Port of C++ <c>Manifold::WarpBatch()</c>.</summary>
		/// <param name="warpFn">Called once with the whole vertex buffer.</param>
		/// <returns>The warped manifold.</returns>
		public Manifold WarpBatch(WarpBatchFunc warpFn)
		{
			ArgumentNullException.ThrowIfNull(warpFn);

			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			if (this.IsEmpty())
			{
				return this.Clone();
			}

			ManifoldImpl outR = this.imp.Clone();
			warpFn(CollectionsMarshal.AsSpan(outR.VertPos));

			// Arbitrary deformation: nothing about the old verdict survives.
			outR.InvalidateSelfIntersects();
			outR.CalculateBBox();
			outR.SortGeometry();
			outR.SetNormalsAndCoplanar();
			outR.MeshRelation.OriginalId = -1;
			return FromImpl(outR);
		}

		/// <summary>Translate every vertex by <paramref name="v"/>.</summary>
		/// <param name="v">The translation.</param>
		/// <returns>The translated manifold.</returns>
		public Manifold Translate(Vec3 v)
		{
			Mat3x4 t = Mat4ToMat3x4(TranslationMatrix(v));
			return FromImpl(this.imp.Transform(t));
		}

		/// <summary>
		/// Rotate by Euler angles in degrees: first about X, then Y, then Z.
		/// </summary>
		/// <remarks>
		/// Port of C++ <c>CsgNode::Rotate</c> (csg_tree.cpp): degree-based
		/// <c>sind</c>/<c>cosd</c> axis matrices composed <c>rZ*rY*rX</c>. Quaternions
		/// (sin/cos of half-angles in radians) differ from this by ~1 ULP, which is enough
		/// to flip symbolic-perturbation ties in the boolean kernel on almost-coplanar
		/// inputs (see PORTING_PLAN.md, <c>almost_coplanar</c>).
		/// </remarks>
		/// <param name="xDegrees">Rotation about X, in degrees.</param>
		/// <param name="yDegrees">Rotation about Y, in degrees.</param>
		/// <param name="zDegrees">Rotation about Z, in degrees.</param>
		/// <returns>The rotated manifold.</returns>
		public Manifold Rotate(double xDegrees, double yDegrees, double zDegrees)
		{
			double sx = Types.Sind(xDegrees);
			double cx = Types.Cosd(xDegrees);
			double sy = Types.Sind(yDegrees);
			double cy = Types.Cosd(yDegrees);
			double sz = Types.Sind(zDegrees);
			double cz = Types.Cosd(zDegrees);
			Mat3 rx = Mat3.FromCols(
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(0.0, cx, sx),
				new Vec3(0.0, -sx, cx));
			Mat3 ry = Mat3.FromCols(
				new Vec3(cy, 0.0, -sy),
				new Vec3(0.0, 1.0, 0.0),
				new Vec3(sy, 0.0, cy));
			Mat3 rz = Mat3.FromCols(
				new Vec3(cz, sz, 0.0),
				new Vec3(-sz, cz, 0.0),
				new Vec3(0.0, 0.0, 1.0));
			Mat3 m = rz * ry * rx;
			Mat3x4 t = Mat3x4.FromCols(m.X, m.Y, m.Z, new Vec3(0.0, 0.0, 0.0));
			return FromImpl(this.imp.Transform(t));
		}

		/// <summary>Scale every vertex by <paramref name="v"/>, componentwise.</summary>
		/// <param name="v">The per-axis scale factors.</param>
		/// <returns>The scaled manifold.</returns>
		public Manifold Scale(Vec3 v)
		{
			Mat3x4 t = Mat4ToMat3x4(ScalingMatrix(v));
			return FromImpl(this.imp.Transform(t));
		}

		/// <summary>Apply an arbitrary affine transform.</summary>
		/// <param name="m">The 3x4 column-major transform.</param>
		/// <returns>The transformed manifold.</returns>
		public Manifold Transform(Mat3x4 m)
		{
			return FromImpl(this.imp.Transform(m));
		}

		/// <summary>
		/// Mirror this manifold over the plane defined by the given normal vector.
		/// If the normal is zero-length, returns an empty manifold.
		/// </summary>
		/// <param name="normal">The plane normal.</param>
		/// <returns>The mirrored manifold.</returns>
		public Manifold Mirror(Vec3 normal)
		{
			// Per C++ #1659: propagate an errored input even on the degenerate
			// (zero-length normal) path, which otherwise returns an empty manifold.
			if (this.imp.Status != Error.NoError)
			{
				return this.Clone();
			}

			double len = Math.Sqrt((normal.X * normal.X) + (normal.Y * normal.Y) + (normal.Z * normal.Z));
			if (len == 0.0)
			{
				return Empty();
			}

			Vec3 n = new Vec3(normal.X / len, normal.Y / len, normal.Z / len);

			// Householder reflection: M = I - 2*n*n^T
			Mat3x4 m = Mat3x4.FromCols(
				new Vec3(1.0 - (2.0 * n.X * n.X), -2.0 * n.X * n.Y, -2.0 * n.X * n.Z),
				new Vec3(-2.0 * n.Y * n.X, 1.0 - (2.0 * n.Y * n.Y), -2.0 * n.Y * n.Z),
				new Vec3(-2.0 * n.Z * n.X, -2.0 * n.Z * n.Y, 1.0 - (2.0 * n.Z * n.Z)),
				new Vec3(0.0, 0.0, 0.0));
			return FromImpl(this.imp.Transform(m));
		}

		/// <summary>
		/// Warp the mesh by applying a function to each vertex position.
		/// Does not check for self-intersection.
		/// </summary>
		/// <param name="warpFn">Called once per vertex, editing it in place.</param>
		/// <returns>The warped manifold.</returns>
		public Manifold Warp(WarpFunc warpFn)
		{
			ArgumentNullException.ThrowIfNull(warpFn);

			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			if (this.IsEmpty())
			{
				return this.Clone();
			}

			ManifoldImpl outR = this.imp.Clone();

			// The Rust iterates `out.vert_pos.iter_mut()`, so each callback sees the live
			// element. A `foreach` over List<Vec3> would hand out copies; the span's
			// elements are the list's own storage, so `ref` writes land.
			Span<Vec3> verts = CollectionsMarshal.AsSpan(outR.VertPos);
			for (int i = 0; i < verts.Length; i++)
			{
				warpFn(ref verts[i]);
			}

			// Arbitrary deformation: nothing about the old verdict survives.
			outR.InvalidateSelfIntersects();
			outR.CalculateBBox();
			outR.SortGeometry();
			outR.SetNormalsAndCoplanar();
			outR.MeshRelation.OriginalId = -1;
			return FromImpl(outR);
		}

		/// <summary>Get the bounding box of this manifold.</summary>
		/// <returns>The bounding box.</returns>
		public Box BoundingBox()
		{
			return new Box(this.imp.Bbox.Min, this.imp.Bbox.Max);
		}


		/// <summary>
		/// Soup impls (non-manifold geometry imported via <see cref="FromMeshGLRobust"/>)
		/// support only robust-engine booleans, transforms, bounding-box queries, hulls,
		/// and mesh export. Pairing-dependent operations call this and return an empty
		/// manifold with <see cref="Error.NotManifold"/> instead of walking incomplete
		/// halfedges.
		/// </summary>
		/// <returns>The errored replacement, or null when this mesh is properly paired.</returns>
		internal Manifold? RequirePaired()
		{
			if (this.imp.IsSoup)
			{
				return MakeEmpty(Error.NotManifold);
			}

			return null;
		}
	}
}
