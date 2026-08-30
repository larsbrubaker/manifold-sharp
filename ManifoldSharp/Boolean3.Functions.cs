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

// Boolean3.Functions.cs — the free functions of boolean3.rs: ComposeMeshes, the
// public Boolean entry points, the engine dispatch, and RayCast. See
// Boolean3.cs for the module header and for why they live on
// `Boolean3Functions` rather than on `Boolean3`.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// The free functions of <c>boolean3.rs</c>.
	/// </summary>
	public static class Boolean3Functions
	{
		// -------------------------------------------------------------------
		// compose_meshes — concatenate disjoint meshes (unchanged from before)
		// -------------------------------------------------------------------

		/// <summary>
		/// Concatenate multiple disjoint meshes into one. This is a genuine utility
		/// used by both boolean operations and CSG compose. It does NOT perform any
		/// boolean intersection — the meshes must be non-overlapping for correct results.
		/// </summary>
		/// <param name="meshes">The meshes to concatenate.</param>
		/// <returns>The concatenated impl.</returns>
		public static ManifoldImpl ComposeMeshes(IReadOnlyList<ManifoldImpl> meshes)
		{
			ArgumentNullException.ThrowIfNull(meshes);

			if (meshes.Count == 0)
			{
				return new ManifoldImpl();
			}

			if (meshes.Count == 1)
			{
				return meshes[0].Clone();
			}

			// Soup inputs (robust non-manifold import) cannot go through
			// create_halfedges' strict pairing below; concatenate them geometrically
			// instead. Mesh relations are not preserved on this path — soups carry
			// none that survive a boolean anyway.
			if (meshes.Any(m => m.IsSoup))
			{
				// DEFERRED(Phase 10, robust): this arm is `robust::soup::impl_to_tris` +
				// `robust::assemble_all`, both of which land with the robust engine. It is
				// unreachable today — `IsSoup` is only ever set by the robust MeshGL import,
				// which does not exist yet — so it throws rather than silently composing
				// soup through a path that assumes strict halfedge pairing.
				throw new NotSupportedException(
					"ComposeMeshes on soup operands needs the robust engine (DEFERRED: Phase 10, robust).");
			}

			int numProp = meshes.Max(m => m.NumProp);
			List<Vec3> vertPos = new List<Vec3>();
			List<double> properties = new List<double>();
			List<IVec3> triVert = new List<IVec3>();
			List<IVec3> triProp = new List<IVec3>();
			int vertOffset = 0;
			int propOffset = 0;

			foreach (ManifoldImpl mesh in meshes)
			{
				vertPos.AddRange(mesh.VertPos);

				foreach (IVec3 t in ExtractTriVert(mesh))
				{
					triVert.Add(new IVec3(t.X + vertOffset, t.Y + vertOffset, t.Z + vertOffset));
				}

				foreach (IVec3 t in ExtractTriProp(mesh))
				{
					triProp.Add(new IVec3(t.X + propOffset, t.Y + propOffset, t.Z + propOffset));
				}

				if (numProp > 0)
				{
					int propRows = mesh.NumPropVert();
					for (int row = 0; row < propRows; row++)
					{
						properties.AddRange(PropertyRow(mesh, row, numProp));
					}

					propOffset += propRows;
				}
				else
				{
					propOffset += mesh.NumPropVert();
				}

				vertOffset += mesh.NumVert();
			}

			// Concatenate tri_refs and merge mesh_id_transforms from all input meshes.
			// Each mesh's coplanar_id is a triangle-local group index, so offset by tri_offset.
			List<TriRef> allTriRefs = new List<TriRef>();
			SortedDictionary<int, Relation> mergedTransforms = new SortedDictionary<int, Relation>();
			int triOffset = 0;
			foreach (ManifoldImpl mesh in meshes)
			{
				int meshTriCount = mesh.NumTri();
				foreach (TriRef triRef in mesh.MeshRelation.TriRef)
				{
					allTriRefs.Add(new TriRef(
						triRef.MeshId,
						triRef.OriginalId,
						triRef.FaceId,
						triRef.CoplanarId + triOffset));
				}

				foreach (KeyValuePair<int, Relation> entry in mesh.MeshRelation.MeshIdTransform)
				{
					mergedTransforms[entry.Key] = entry.Value;
				}

				triOffset += meshTriCount;
			}

			ManifoldImpl outR = new ManifoldImpl();
			outR.VertPos = vertPos;
			outR.NumProp = numProp;
			outR.Properties = properties;
			outR.CreateHalfedges(triProp, triVert);

			// Preserve tri_refs and transforms from input meshes instead of
			// calling initialize_original(), which would lose mesh transform data.
			outR.MeshRelation.TriRef.Clear();
			outR.MeshRelation.TriRef.AddRange(allTriRefs);
			outR.MeshRelation.MeshIdTransform.Clear();
			foreach (KeyValuePair<int, Relation> entry in mergedTransforms)
			{
				outR.MeshRelation.MeshIdTransform[entry.Key] = entry.Value;
			}

			outR.MeshRelation.OriginalId = -1;
			outR.CalculateBBox();
			outR.SetEpsilon(-1.0, false);

			// required to remove parts that are smaller than the tolerance (matches C++)
			EdgeOp.RemoveDegenerates(outR, 0);
			outR.SortGeometry();
			outR.IncrementMeshIds();
			outR.SetNormalsAndCoplanar();
			return outR;
		}

		// -------------------------------------------------------------------
		// boolean — public entry point
		// -------------------------------------------------------------------

		/// <summary>
		/// Perform a 3D boolean operation on two manifold meshes.
		/// </summary>
		/// <remarks>
		/// For overlapping meshes, uses the full Boolean3 intersection algorithm.
		/// For disjoint meshes, uses fast-path shortcuts.
		/// </remarks>
		/// <param name="meshA">The first operand.</param>
		/// <param name="meshB">The second operand.</param>
		/// <param name="op">The operation to perform.</param>
		/// <returns>The result impl.</returns>
		public static ManifoldImpl Boolean(ManifoldImpl meshA, ManifoldImpl meshB, OpType op)
		{
			return BooleanWithToken(meshA, meshB, op, null);
		}

		/// <summary>
		/// <see cref="Boolean"/> with cooperative cancellation.
		/// </summary>
		/// <remarks>
		/// A cancelled operation yields an empty mesh whose <c>Status</c> is
		/// <see cref="Error.Cancelled"/>, matching what C++ produces via
		/// <c>MakeEmpty(Cancelled)</c> at every checkpoint (execution_impl.h:150-160,
		/// boolean_result.cpp:758-770).
		/// </remarks>
		/// <param name="meshA">The first operand.</param>
		/// <param name="meshB">The second operand.</param>
		/// <param name="op">The operation to perform.</param>
		/// <param name="token">The cancellation token, or null for an uncancellable run.</param>
		/// <returns>The result impl.</returns>
		public static ManifoldImpl BooleanWithToken(
			ManifoldImpl meshA,
			ManifoldImpl meshB,
			OpType op,
			CancelToken? token)
		{
			ArgumentNullException.ThrowIfNull(meshA);
			ArgumentNullException.ThrowIfNull(meshB);

			// Entry gate: a token cancelled before the call wins over every fast path
			// below, including the empty-input ones. C++ does the same at its outermost
			// gates (csg_tree.cpp:172, execution_impl.cpp's static factories), so an
			// already-cancelled context never reports NoError.
			if (Cancel.IsCancelled(token))
			{
				return CancelledImpl();
			}

			// The exact engine's kernels assume complete halfedge pairing; soup
			// impls (robust import of non-manifold geometry) must use the robust
			// engine instead. Unreachable for all pre-existing callers: is_soup is
			// false everywhere outside the from_mesh_gl_robust path.
			if (meshA.IsSoup || meshB.IsSoup)
			{
				ManifoldImpl notManifold = new ManifoldImpl();
				notManifold.MakeEmpty(Error.NotManifold);
				return notManifold;
			}

			if (meshA.IsEmpty())
			{
				switch (op)
				{
					case OpType.Add:
						return meshB.Clone();
					default:
						// Intersect and Subtract both yield nothing when A is empty.
						return new ManifoldImpl();
				}
			}

			if (meshB.IsEmpty())
			{
				switch (op)
				{
					case OpType.Intersect:
						return new ManifoldImpl();
					default:
						// Add and Subtract both yield A when B is empty.
						return meshA.Clone();
				}
			}

			if (!meshA.Bbox.DoesOverlapBox(meshB.Bbox))
			{
				// Non-overlapping fast paths. For Subtract, we still run through the full
				// boolean_result to preserve both meshes' run metadata (C++ behavior).
				switch (op)
				{
					case OpType.Add:
						return ComposeMeshes(new[] { meshA.Clone(), meshB.Clone() });
					case OpType.Intersect:
						return new ManifoldImpl();
					default:
						// Subtract falls through to the full boolean.
						break;
				}
			}

			// Full boolean — compute intersections
			Boolean3? bool3 = Boolean3.NewWithToken(meshA, meshB, op, token);
			if (bool3 is null)
			{
				return CancelledImpl();
			}

			if (!bool3.Valid)
			{
				return new ManifoldImpl();
			}

			return BooleanResultAssemble.BooleanResultWithToken(meshA, meshB, op, bool3, token);
		}

		/// <summary>
		/// Route a boolean to the requested engine (<see cref="BooleanEngine"/>).
		/// </summary>
		/// <remarks>
		/// <c>Auto</c> is clean-by-default: it picks the faster <c>Exact</c> engine only when
		/// correctness is not at risk — i.e. when <b>both</b> operands are topologically
		/// manifold (not soup) <b>and</b> free of self-intersection — no two of an
		/// operand's own triangles crossing, overlapping, or coinciding. Either
		/// condition failing routes the pair to <c>Robust</c>, because the exact engine's
		/// kernels assume complete halfedge pairing and a non-self-intersecting
		/// surface; on self-intersecting-but-manifold inputs it silently
		/// mis-integrates the result (e.g. Thingi10K #92068's triple-wound
		/// concentric shells).
		/// <para>
		/// The self-intersection test is cached per impl (see
		/// <see cref="SelfIntersectCache"/>), so an operand pays for the scan at most
		/// once. <c>Exact</c> with a soup operand yields an empty result with
		/// <see cref="Error.NotManifold"/> (the guard inside
		/// <see cref="BooleanWithToken"/>); no panic-catching is involved anywhere —
		/// dispatch is input-based only.
		/// </para>
		/// </remarks>
		/// <param name="meshA">The first operand.</param>
		/// <param name="meshB">The second operand.</param>
		/// <param name="op">The operation to perform.</param>
		/// <param name="engine">The engine to route to.</param>
		/// <param name="token">The cancellation token, or null for an uncancellable run.</param>
		/// <returns>The result impl.</returns>
		public static ManifoldImpl BooleanDispatch(
			ManifoldImpl meshA,
			ManifoldImpl meshB,
			OpType op,
			BooleanEngine engine,
			CancelToken? token)
		{
			return BooleanDispatchWithProgress(meshA, meshB, op, engine, token, null);
		}

		/// <summary>
		/// <see cref="BooleanDispatch"/> with optional progress reporting (see
		/// <see cref="ProgressReporter"/>).
		/// </summary>
		/// <remarks>
		/// The robust engine reports its pipeline phases; the exact engine reports a
		/// single indeterminate <c>ExactBoolean</c> phase and is otherwise untouched, so
		/// its timing and results are exactly what they were. Null is byte-for-byte
		/// <see cref="BooleanDispatch"/>.
		/// </remarks>
		/// <param name="meshA">The first operand.</param>
		/// <param name="meshB">The second operand.</param>
		/// <param name="op">The operation to perform.</param>
		/// <param name="engine">The engine to route to.</param>
		/// <param name="token">The cancellation token, or null for an uncancellable run.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>The result impl.</returns>
		public static ManifoldImpl BooleanDispatchWithProgress(
			ManifoldImpl meshA,
			ManifoldImpl meshB,
			OpType op,
			BooleanEngine engine,
			CancelToken? token,
			ProgressReporter? progress)
		{
			return BooleanDispatchFull(
				meshA,
				meshB,
				op,
				engine,
				WindingRule.Positive,
				token,
				progress);
		}

		/// <summary>
		/// <see cref="BooleanDispatchWithProgress"/> with an explicit winding rule.
		/// </summary>
		/// <remarks>
		/// Winding rules are a robust-engine semantic: the exact engine has no cell
		/// labels to reinterpret and <b>ignores</b> <paramref name="rule"/> entirely.
		/// Because of that, <c>Auto</c> with <see cref="WindingRule.Nonzero"/> resolves to
		/// <c>Robust</c> even for two clean manifold operands — nonzero semantics can only
		/// be honored there, and silently answering with positive-rule geometry would be
		/// worse than paying for the robust pipeline. An explicit <c>Exact</c> still runs
		/// the exact engine, rule and all, so a caller who pinned the engine gets exactly
		/// what it asked for.
		/// <para>
		/// <see cref="WindingRule.Positive"/> is byte-for-byte
		/// <see cref="BooleanDispatchWithProgress"/>, including <c>Auto</c>'s resolution.
		/// </para>
		/// </remarks>
		/// <param name="meshA">The first operand.</param>
		/// <param name="meshB">The second operand.</param>
		/// <param name="op">The operation to perform.</param>
		/// <param name="engine">The engine to route to.</param>
		/// <param name="rule">Which winding numbers count as solid material.</param>
		/// <param name="token">The cancellation token, or null for an uncancellable run.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>The result impl.</returns>
		/// <exception cref="NotSupportedException">
		/// The routing needs the robust engine, which is deferred to Phase 10.
		/// </exception>
		public static ManifoldImpl BooleanDispatchFull(
			ManifoldImpl meshA,
			ManifoldImpl meshB,
			OpType op,
			BooleanEngine engine,
			WindingRule rule,
			CancelToken? token,
			ProgressReporter? progress)
		{
			ArgumentNullException.ThrowIfNull(meshA);
			ArgumentNullException.ThrowIfNull(meshB);

			// `rule` is read only by Auto's resolution (and by the robust engine it can
			// route to); the exact engine ignores it, exactly as the Rust does. Both of
			// those readers are deferred below, so the parameter is deliberately unused
			// on the one path that currently runs.
			_ = rule;

			// For the Phase 6 façade wiring: an `Auto` that throws would be a trap if
			// `Auto` were this enum's zero value, because every `default(BooleanEngine)`,
			// every zero-initialized options struct and every `new T[]` of engines would
			// land on the throwing arm. It is not — Quality.cs pins Exact = 0, Robust = 1,
			// Auto = 2, matching the Rust's FFI-meaningful order — so a defaulted engine
			// resolves to Exact and runs. Two consequences worth stating rather than
			// rediscovering: the façade must pass `Auto` deliberately to hit this, and
			// `BooleanConfig.DefaultEngine()` (also Exact at zero) stays usable while the
			// robust engine is deferred. If the enum's values are ever renumbered, this
			// arm becomes reachable by accident and this comment is the thing that was
			// wrong.
			if (engine == BooleanEngine.Auto)
			{
				// DEFERRED(Phase 10, robust): the Rust resolves Auto with
				// `robust::soup::has_self_intersections_with_token` on both operands, and
				// that detector is part of the robust engine. Resolving Auto to Exact
				// without it would be a *silent* behavioural divergence — exactly the
				// mis-integration this branch exists to prevent — so Auto is refused until
				// the detector lands.
				throw new NotSupportedException(
					"BooleanEngine.Auto needs robust::soup::has_self_intersections to resolve "
					+ "(DEFERRED: Phase 10, robust). Pass BooleanEngine.Exact explicitly.");
			}

			BooleanEngine resolved = engine;

			switch (resolved)
			{
				case BooleanEngine.Exact:
					Progress.BeginPhase(progress, Phase.ExactBoolean, 0);
					return BooleanWithToken(meshA, meshB, op, token);
				default:
					// DEFERRED(Phase 10, robust): `robust::boolean_with_rule`.
					throw new NotSupportedException(
						"BooleanEngine.Robust needs robust::boolean_with_rule "
						+ "(DEFERRED: Phase 10, robust).");
			}
		}

		/// <summary>
		/// The observable result of an interrupted operation: an empty mesh carrying
		/// <see cref="Error.Cancelled"/>. Mirrors C++ <c>MakeEmpty(Manifold::Error::Cancelled)</c>.
		/// </summary>
		/// <returns>The empty, cancelled impl.</returns>
		internal static ManifoldImpl CancelledImpl()
		{
			ManifoldImpl outR = new ManifoldImpl();
			outR.MakeEmpty(Error.Cancelled);
			return outR;
		}

		/// <summary>
		/// Cast a ray segment from <paramref name="origin"/> to <paramref name="endpoint"/>
		/// against <paramref name="mesh"/>, returning all triangle intersections sorted by
		/// parametric distance.
		/// </summary>
		/// <remarks>
		/// Mirrors C++ <c>Manifold::Impl::RayCast(vec3, vec3)</c> in boolean3.cpp.
		/// Builds a degenerate single-edge Impl representing the ray, then uses
		/// Kernel12 (edge-face intersection) with the mesh BVH to find hits.
		/// </remarks>
		/// <param name="mesh">The mesh to cast against.</param>
		/// <param name="origin">The ray segment's start.</param>
		/// <param name="endpoint">The ray segment's end.</param>
		/// <returns>The hits, ascending by parametric distance.</returns>
		public static List<RayHit> RayCast(ManifoldImpl mesh, Vec3 origin, Vec3 endpoint)
		{
			ArgumentNullException.ThrowIfNull(mesh);

			if (mesh.IsEmpty())
			{
				return new List<RayHit>();
			}

			Vec3 dir = endpoint - origin;
			if (Dot(dir, dir) == 0.0)
			{
				return new List<RayHit>();
			}

			// Build a minimal single-edge Impl representing the ray segment.
			// halfedge[0]: forward (0→1), halfedge[1]: backward (1→0).
			ManifoldImpl rayImpl = new ManifoldImpl();
			rayImpl.VertPos = new List<Vec3> { origin, endpoint };
			rayImpl.VertNormal = new List<Vec3> { Vec3.Splat(0.0), Vec3.Splat(0.0) };
			rayImpl.Halfedge = new List<Halfedge>
			{
				new Halfedge(0, 1, 1, 0),
				new Halfedge(1, 0, 0, 0),
			};
			rayImpl.FaceNormal = new List<Vec3> { Vec3.Splat(0.0) };

			// Query the mesh's cached face BVH (C++ RayCast uses collider_).
			Collider collider = mesh.Collider;

			// Ray AABB for BVH query.
			Box rayBox = Box.FromPoints(
				new Vec3(
					MinF64(origin.X, endpoint.X),
					MinF64(origin.Y, endpoint.Y),
					MinF64(origin.Z, endpoint.Z)),
				new Vec3(
					MaxF64(origin.X, endpoint.X),
					MaxF64(origin.Y, endpoint.Y),
					MaxF64(origin.Z, endpoint.Z)));

			// Determine which component axis is largest for stable t computation.
			Vec3 absDir = new Vec3(Math.Abs(dir.X), Math.Abs(dir.Y), Math.Abs(dir.Z));
			int tAxis = absDir.X > absDir.Y && absDir.X > absDir.Z
				? 0
				: absDir.Y > absDir.Z ? 1 : 2;

			List<RayHit> hits = new List<RayHit>();

			// Query BVH with ray AABB and test each candidate triangle.
			collider.CollisionsWithBoxes(new[] { rayBox }, false, (_, tri) =>
			{
				// halfedge 0 (forward) vs triangle tri; expand_p=false, forward=true.
				(int s, Vec3 v) = Boolean3Kernels.Kernel12(0, tri, rayImpl, mesh, rayImpl, mesh, false, true);
				if (s != 0 && double.IsFinite(v.X))
				{
					// Compute parametric t along the ray.
					double originT = origin[tAxis];
					double dirT = dir[tAxis];
					double vT = v[tAxis];
					double t = (vT - originT) / dirT;
					if (t >= 0.0 && t <= 1.0)
					{
						hits.Add(new RayHit
						{
							FaceId = (ulong)tri,
							Distance = t,
							Position = v,
							Normal = mesh.FaceNormal[tri],
						});
					}
				}
			});

			// SORT AUDIT (boolean3.rs:593): Rust `sort_by`, which is STABLE, comparing
			// `a.distance.partial_cmp(&b.distance).unwrap_or(Equal)`. Two hits at the same
			// parametric distance — a ray grazing a shared edge, or passing through
			// coincident faces — must stay in collider-emission order, which is what the
			// caller sees as "the order the surfaces were met". `Polygon.PartialCmp` is
			// exactly that comparator (NaN compares Equal rather than sorting to one end),
			// and LINQ OrderBy is the stable sort. `List<T>.Sort` is introsort and is not
			// usable here.
			return hits.OrderBy(h => h.Distance, Polygon.PartialCmp).ToList();
		}

		private static List<IVec3> ExtractTriVert(ManifoldImpl mesh)
		{
			List<IVec3> result = new List<IVec3>(mesh.NumTri());
			for (int tri = 0; tri < mesh.NumTri(); tri++)
			{
				result.Add(new IVec3(
					mesh.Halfedge[3 * tri].StartVert,
					mesh.Halfedge[(3 * tri) + 1].StartVert,
					mesh.Halfedge[(3 * tri) + 2].StartVert));
			}

			return result;
		}

		private static List<IVec3> ExtractTriProp(ManifoldImpl mesh)
		{
			List<IVec3> result = new List<IVec3>(mesh.NumTri());
			for (int tri = 0; tri < mesh.NumTri(); tri++)
			{
				result.Add(new IVec3(
					mesh.Halfedge[3 * tri].PropVert,
					mesh.Halfedge[(3 * tri) + 1].PropVert,
					mesh.Halfedge[(3 * tri) + 2].PropVert));
			}

			return result;
		}

		private static List<double> PropertyRow(ManifoldImpl mesh, int row, int width)
		{
			List<double> outRow = new List<double>(width);
			outRow.Resize(width, 0.0);
			if (mesh.NumProp == 0)
			{
				return outRow;
			}

			for (int i = 0; i < mesh.NumProp; i++)
			{
				outRow[i] = mesh.Properties[(row * mesh.NumProp) + i];
			}

			return outRow;
		}
	}
}
