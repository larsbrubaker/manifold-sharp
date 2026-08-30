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

// Manifold.Regions.cs — the "carve this mesh up and ask where things are" half
// of manifold.rs, split from Manifold.cs for the 800-line file cap. Same class,
// same order as the Rust: Split, SplitByPlane, TrimByPlane, Slice, Project, the
// three robust repairs, Decompose, RayCast, and the private `halfspace` helper
// the plane splits share.
//
// ── DEFERRED in this file (greppable) ────────────────────────────────────────
//   HasSelfIntersections, RepairOrientation, RebuildSolid,
//   RebuildSolidWithToken     Phase 10 (robust::soup / robust::repair /
//                             robust::rebuild_with_rule)

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <content>
	/// Splitting, slicing, decomposition and ray casting.
	/// </content>
	public sealed partial class Manifold
	{
		/// <summary>
		/// Split this manifold into two using a cutter manifold.
		/// </summary>
		/// <param name="cutter">The cutting manifold.</param>
		/// <returns>The intersection and the difference, in that order.</returns>
		public (Manifold Intersection, Manifold Difference) Split(Manifold cutter)
		{
			ArgumentNullException.ThrowIfNull(cutter);
			Manifold intersection = this.Intersection(cutter);
			Manifold difference = this.Difference(cutter);
			return (intersection, difference);
		}

		/// <summary>
		/// Split this manifold by a plane defined by a normal and offset from origin.
		/// </summary>
		/// <param name="normal">The plane normal.</param>
		/// <param name="originOffset">The plane's signed offset from the origin.</param>
		/// <returns>The half in the direction of the normal, then the opposite half.</returns>
		public (Manifold Positive, Manifold Negative) SplitByPlane(Vec3 normal, double originOffset)
		{
			// Per C++ #1659: errored manifolds are empty, so the IsEmpty()
			// early-return below would silently drop their status — guard first.
			if (this.imp.Status != Error.NoError)
			{
				return (this.Clone(), this.Clone());
			}

			if (this.IsEmpty())
			{
				return (Empty(), Empty());
			}

			Manifold halfspace = Halfspace(this.imp.Bbox, normal, originOffset);
			return this.Split(halfspace);
		}

		/// <summary>
		/// Trim this manifold by a half-space, keeping only the part in the direction of
		/// the normal vector.
		/// </summary>
		/// <param name="normal">The half-space normal.</param>
		/// <param name="originOffset">The plane's signed offset from the origin.</param>
		/// <returns>The trimmed manifold.</returns>
		public Manifold TrimByPlane(Vec3 normal, double originOffset)
		{
			if (this.IsEmpty())
			{
				return Empty();
			}

			Manifold halfspace = Halfspace(this.imp.Bbox, normal, originOffset);
			return this.Intersection(halfspace);
		}

		/// <summary>
		/// Slice this manifold at the given Z height, returning the cross-section as a
		/// CrossSection. Mirrors C++ <c>Manifold::Slice</c>.
		/// </summary>
		/// <param name="height">The Z height to slice at.</param>
		/// <returns>The cross-section.</returns>
		/// <remarks>
		/// The contour ORDER, and the vertex each contour starts at, are pinned by this
		/// port — the Rust seeds each loop from a <c>HashSet</c> and so is not reproducible
		/// against itself. See docs/RUST_DIVERGENCES.md entry 3. Contour content is the
		/// Rust's exactly, and nothing a <see cref="CrossSection"/> reports depends on the
		/// order anyway.
		/// </remarks>
		public CrossSection Slice(double height)
		{
			if (this.imp.IsSoup || this.IsEmpty())
			{
				return new CrossSection(new Polygons());
			}

			Polygons polys = this.imp.Slice(height);
			return new CrossSection(polys);
		}

		/// <summary>
		/// Project this manifold onto the XY plane, returning the silhouette as a
		/// CrossSection. Mirrors C++ <c>Manifold::Project</c>.
		/// </summary>
		/// <returns>The silhouette.</returns>
		public CrossSection Project()
		{
			if (this.imp.IsSoup || this.IsEmpty())
			{
				return new CrossSection(new Polygons());
			}

			Polygons polys = this.imp.Project();
			return CrossSection.FromPolygonsFill(polys);
		}

		/// <summary>
		/// True when two of this mesh's own triangles genuinely intersect — they cross,
		/// they overlap, or they are coincident surface — rather than merely sharing edges
		/// and vertices as every closed mesh does.
		/// </summary>
		/// <returns>True when the mesh self-intersects.</returns>
		/// <exception cref="NotSupportedException">
		/// The detector is part of the robust engine, deferred to Phase 10.
		/// </exception>
		/// <remarks>
		/// DEFERRED(Phase 10, robust): the Rust body is one call to
		/// <c>robust::soup::has_self_intersections</c>. Answering <c>false</c> without the
		/// detector would be a silent divergence — the same trap
		/// <c>Boolean3Functions.BooleanDispatchFull</c> refuses to fall into for
		/// <c>BooleanEngine.Auto</c>, which consults exactly this predicate.
		/// </remarks>
		public bool HasSelfIntersections()
		{
			throw new NotSupportedException(
				"Manifold.HasSelfIntersections needs robust::soup::has_self_intersections "
				+ "(DEFERRED: Phase 10, robust).");
		}

		/// <summary>
		/// Repair the winding of inside-out shells so every body reads as solid material
		/// under the robust engine's {winding &gt;= 1} semantics.
		/// </summary>
		/// <returns>The rewound manifold.</returns>
		/// <exception cref="NotSupportedException">
		/// The repair planner is part of the robust engine, deferred to Phase 10.
		/// </exception>
		/// <remarks>
		/// DEFERRED(Phase 10, robust): needs <c>robust::soup::impl_to_tris</c> and
		/// <c>robust::repair::plan_repair</c>/<c>apply_flips</c>.
		/// </remarks>
		public Manifold RepairOrientation()
		{
			throw new NotSupportedException(
				"Manifold.RepairOrientation needs robust::repair (DEFERRED: Phase 10, robust).");
		}

		/// <summary>
		/// Rebuild this mesh into a fresh, properly paired 2-manifold enclosing the same
		/// solid region under <paramref name="rule"/>.
		/// </summary>
		/// <param name="rule">Which winding numbers count as solid material.</param>
		/// <returns>The rebuilt manifold.</returns>
		/// <exception cref="NotSupportedException">
		/// The whole robust pipeline is deferred to Phase 10.
		/// </exception>
		public Manifold RebuildSolid(WindingRule rule)
		{
			return this.RebuildSolidWithToken(rule, null);
		}

		/// <summary>
		/// <see cref="RebuildSolid"/> with cooperative cancellation. Soup rebuilds are as
		/// expensive as a boolean against a partner, so anything interactive wants this
		/// form.
		/// </summary>
		/// <param name="rule">Which winding numbers count as solid material.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The rebuilt manifold.</returns>
		/// <exception cref="NotSupportedException">
		/// The whole robust pipeline is deferred to Phase 10.
		/// </exception>
		/// <remarks>
		/// DEFERRED(Phase 10, robust): the body is one call to
		/// <c>robust::rebuild_with_rule</c>. The empty-input fast path is kept above it so
		/// the deferral cannot be reached by an empty mesh, which the Rust also short-circuits.
		/// </remarks>
		public Manifold RebuildSolidWithToken(WindingRule rule, CancelToken? token)
		{
			if (this.IsEmpty())
			{
				return this.Clone();
			}

			_ = rule;
			_ = token;
			throw new NotSupportedException(
				"Manifold.RebuildSolid needs robust::rebuild_with_rule (DEFERRED: Phase 10, robust).");
		}

		/// <summary>
		/// Split this manifold into its connected components.
		/// </summary>
		/// <returns>
		/// One manifold per connected component; a single-component mesh returns itself,
		/// and an errored mesh returns itself so the status propagates.
		/// </returns>
		public List<Manifold> Decompose()
		{
			Manifold? paired = this.RequirePaired();
			if (paired != null)
			{
				return new List<Manifold> { paired };
			}

			int numVert = this.imp.NumVert();
			if (numVert == 0)
			{
				// Propagate error status: errored manifolds decompose to [self]
				if (this.imp.Status != Error.NoError)
				{
					return new List<Manifold> { this.Clone() };
				}

				return new List<Manifold>();
			}

			DisjointSets uf = new DisjointSets((uint)numVert);
			foreach (Halfedge he in this.imp.Halfedge)
			{
				if (he.IsForward())
				{
					uf.Unite((uint)he.StartVert, (uint)he.EndVert);
				}
			}

			List<int> componentIndices = new List<int>(numVert);
			for (int i = 0; i < numVert; i++)
			{
				componentIndices.Add(0);
			}

			int numComponents = uf.ConnectedComponents(componentIndices);

			if (numComponents <= 1)
			{
				return new List<Manifold> { this.Clone() };
			}

			int numTri = this.imp.NumTri();
			List<Manifold> meshes = new List<Manifold>();

			for (int comp = 0; comp < numComponents; comp++)
			{
				ManifoldImpl impl = new ManifoldImpl();
				impl.Tolerance = this.imp.Tolerance;

				// Collect vertices belonging to this component
				List<int> vertNew2OldList = new List<int>();
				for (int v = 0; v < numVert; v++)
				{
					if (componentIndices[v] == comp)
					{
						vertNew2OldList.Add(v);
					}
				}

				int nVert = vertNew2OldList.Count;
				if (nVert == 0)
				{
					continue;
				}

				int[] vertNew2Old = vertNew2OldList.ToArray();

				foreach (int v in vertNew2Old)
				{
					impl.VertPos.Add(this.imp.VertPos[v]);
				}

				if (this.imp.VertNormal.Count != 0)
				{
					foreach (int v in vertNew2Old)
					{
						impl.VertNormal.Add(this.imp.VertNormal[v]);
					}
				}

				// Collect faces belonging to this component
				List<int> faceNew2OldList = new List<int>();
				for (int f = 0; f < numTri; f++)
				{
					int sv = this.imp.Halfedge[3 * f].StartVert;
					if (sv >= 0 && componentIndices[sv] == comp)
					{
						faceNew2OldList.Add(f);
					}
				}

				if (faceNew2OldList.Count == 0)
				{
					continue;
				}

				// Copy full data from original, then gather_faces will filter
				impl.Halfedge.AddRange(this.imp.Halfedge);
				impl.FaceNormal.AddRange(this.imp.FaceNormal);
				impl.HalfedgeTangent.AddRange(this.imp.HalfedgeTangent);
				impl.NumProp = this.imp.NumProp;
				impl.Properties.AddRange(this.imp.Properties);
				impl.MeshRelation = this.imp.MeshRelation.Clone();

				Sort.GatherFaces(impl, faceNew2OldList.ToArray());
				Sort.ReindexVerts(impl, vertNew2Old, this.imp.NumVert());
				impl.CalculateBBox();
				impl.SortGeometry();

				meshes.Add(FromImpl(impl));
			}

			return meshes;
		}

		/// <summary>
		/// Cast a ray segment from <paramref name="origin"/> to <paramref name="endpoint"/>,
		/// returning all triangle intersections sorted by parametric distance along the
		/// segment. Mirrors C++ <c>Manifold::RayCast(vec3, vec3)</c>.
		/// </summary>
		/// <param name="origin">The segment start.</param>
		/// <param name="endpoint">The segment end.</param>
		/// <returns>Every hit, nearest first.</returns>
		public List<RayHit> RayCast(Vec3 origin, Vec3 endpoint)
		{
			return Boolean3Functions.RayCast(this.imp, origin, endpoint);
		}
		/// <summary>Internal helper: create a halfspace (large cube) for plane splitting.</summary>
		private static Manifold Halfspace(Box bbox, Vec3 normal, double originOffset)
		{
			Vec3 n = Normalize(normal);
			Manifold cutter = Cube(Vec3.Splat(2.0), true).Translate(new Vec3(1.0, 0.0, 0.0));
			Vec3 center = bbox.Center();
			Vec3 size = bbox.Size();
			double sizeLen = Math.Sqrt((size.X * size.X) + (size.Y * size.Y) + (size.Z * size.Z));
			double cx = center.X - (n.X * originOffset);
			double cy = center.Y - (n.Y * originOffset);
			double cz = center.Z - (n.Z * originOffset);
			double dist = Math.Sqrt((cx * cx) + (cy * cy) + (cz * cz)) + (0.5 * sizeLen);
			cutter = cutter
				.Scale(Vec3.Splat(dist))
				.Translate(new Vec3(originOffset, 0.0, 0.0));
			// Rust `.to_degrees()`, which is ONE multiply by the correctly-rounded
			// 180/pi — never Types.Degrees (`a * 180.0 / K_PI`), whose extra rounding
			// differs by an ULP on about a quarter of all inputs. Smoothing.ToDegrees is
			// the faithful transcription; its file header carries the measurement.
			//
			// Rust `-math::asin(n.z).to_degrees()`: the method call binds tighter than the
			// unary minus, so the negation is of the *degrees*, not of the arcsine.
			double yDeg = -Smoothing.ToDegrees(DeterministicMath.Asin(n.Z));
			double zDeg = Smoothing.ToDegrees(DeterministicMath.Atan2(n.Y, n.X));
			return cutter.Rotate(0.0, yDeg, zDeg);
		}
	}
}
