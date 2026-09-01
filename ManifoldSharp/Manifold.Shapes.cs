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

// Manifold.Shapes.cs — port of manifold_shape.rs, whose header reads:
//
//   manifold_shape.rs — shape-producing constructors on Manifold: primitives
//   (tetrahedron, cube, cylinder, sphere, extrude, revolve), convex hulls,
//   SDF level sets, Minkowski sums and 2D cross-section helpers.
//
//   Extracted from manifold.rs for file size management; a child module of
//   `manifold` so these methods keep access to the private `imp` field and
//   callers keep the same `Manifold::cube(...)` paths. Ports the corresponding
//   static constructors of C++ Manifold (constructors.cpp, sdf.cpp,
//   quickhull.cpp, minkowski.cpp).
//
// The C# counterpart of "a child module of `manifold`" is a `partial class`
// continuation, which is how the private impl field stays reachable.
//
// Nothing here is deferred. Sphere and the LevelSet family reach into Phase 7
// (ManifoldImpl.Subdivide, Sdf.LevelSet), which landed alongside this phase.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <content>
	/// The shape-producing constructors of manifold_shape.rs.
	/// </content>
	public sealed partial class Manifold
	{
		/// <summary>The regular tetrahedron.</summary>
		/// <returns>A tetrahedron.</returns>
		public static Manifold Tetrahedron()
		{
			return FromImpl(ManifoldImpl.Tetrahedron(Mat3x4.Identity()));
		}

		/// <summary>An axis-aligned box.</summary>
		/// <param name="size">The extent along each axis; every component must be &gt;= 0.</param>
		/// <param name="center">When true, centre it on the origin instead of the first octant.</param>
		/// <returns>The cube, or an empty manifold with
		/// <see cref="Error.InvalidConstruction"/> for a degenerate size.</returns>
		public static Manifold Cube(Vec3 size, bool center)
		{
			if (size.X < 0.0 || size.Y < 0.0 || size.Z < 0.0 || Length(size) == 0.0)
			{
				return MakeEmpty(Error.InvalidConstruction);
			}

			Mat4 translation = center
				? TranslationMatrix(-size * 0.5)
				: TranslationMatrix(new Vec3(0.0, 0.0, 0.0));
			Mat3x4 transform = Mat4ToMat3x4(translation * ScalingMatrix(size));
			return FromImpl(ManifoldImpl.Cube(transform));
		}

		/// <summary>A cylinder, frustum or cone, sitting on the XY plane.</summary>
		/// <param name="height">Z extent; must be &gt; 0.</param>
		/// <param name="radiusLow">Radius at the bottom; must be &gt;= 0.</param>
		/// <param name="radiusHigh">Radius at the top; negative means "same as low".</param>
		/// <param name="circularSegments">Sides around the circle; 0 means "ask Quality".</param>
		/// <returns>The cylinder, or an empty manifold with
		/// <see cref="Error.InvalidConstruction"/>.</returns>
		public static Manifold Cylinder(
			double height,
			double radiusLow,
			double radiusHigh,
			int circularSegments)
		{
			return CylinderCentered(height, radiusLow, radiusHigh, circularSegments, false);
		}

		/// <summary><see cref="Cylinder"/> with the option to centre it vertically.</summary>
		/// <param name="height">Z extent; must be &gt; 0.</param>
		/// <param name="radiusLow">Radius at the bottom; must be &gt;= 0.</param>
		/// <param name="radiusHigh">Radius at the top; negative means "same as low".</param>
		/// <param name="circularSegments">Sides around the circle; 0 means "ask Quality".</param>
		/// <param name="center">When true, centre it on the origin in Z.</param>
		/// <returns>The cylinder, or an empty manifold with
		/// <see cref="Error.InvalidConstruction"/>.</returns>
		public static Manifold CylinderCentered(
			double height,
			double radiusLow,
			double radiusHigh,
			int circularSegments,
			bool center)
		{
			if (height <= 0.0 || radiusLow < 0.0)
			{
				return MakeEmpty(Error.InvalidConstruction);
			}

			if (radiusLow == 0.0 && radiusHigh <= 0.0)
			{
				return MakeEmpty(Error.InvalidConstruction);
			}

			return FromImpl(Constructors.Cylinder(
				height,
				radiusLow,
				radiusHigh,
				circularSegments,
				center));
		}

		/// <summary>A sphere centred on the origin.</summary>
		/// <param name="radius">The radius; must be &gt; 0.</param>
		/// <param name="circularSegments">
		/// Segments around a great circle; 0 means "ask Quality".
		/// </param>
		/// <returns>The sphere, or an empty manifold with
		/// <see cref="Error.InvalidConstruction"/> for a non-positive radius.</returns>
		/// <remarks>
		/// Build a unit octahedron and subdivide each edge into exactly n parts in one pass
		/// (n^2 tris per original face), matching C++:
		/// <code>
		///   n = (circularSegments + 3) / 4   (or Quality-based when unspecified)
		///   Subdivide([n](...) { return n - 1; })
		/// </code>
		/// n-1 is the number of *added* verts per edge. Using the n-way Subdivide (not
		/// binary midpoint splits) gives exact non-power-of-2 counts, e.g.
		/// <c>Sphere(_, 25)</c> -&gt; 8 * 25^2 = 5000 tris rather than 8 * 32^2 = 8192.
		/// </remarks>
		public static Manifold Sphere(double radius, int circularSegments)
		{
			if (radius <= 0.0)
			{
				return MakeEmpty(Error.InvalidConstruction);
			}

			Mat3x4 identity = Mat4ToMat3x4(ScalingMatrix(Vec3.Splat(1.0)));
			ManifoldImpl mesh = ManifoldImpl.Octahedron(identity);
			int n = circularSegments > 0
				? (circularSegments + 3) / 4
				: Quality.GetCircularSegments(radius) / 4;
			if (n > 1)
			{
				// The Rust discards the returned barycentric table here; only the
				// subdivided mesh matters.
				mesh.Subdivide((edgeVec, tangent0, tangent1) => n - 1, false);
			}

			// Map subdivided octahedron vertices onto the sphere surface
			// (matches C++: v = cos(π/2 * (1 - v)); v = radius * normalize(v))
			for (int i = 0; i < mesh.VertPos.Count; i++)
			{
				Vec3 v = mesh.VertPos[i];
				Vec3 mapped = new Vec3(
					DeterministicMath.Cos(Types.KHalfPi * (1.0 - v.X)),
					DeterministicMath.Cos(Types.KHalfPi * (1.0 - v.Y)),
					DeterministicMath.Cos(Types.KHalfPi * (1.0 - v.Z)));
				Vec3 n2 = Normalize(mapped);
				mesh.VertPos[i] = double.IsNaN(n2.X)
					? Vec3.Splat(0.0)
					: new Vec3(n2.X * radius, n2.Y * radius, n2.Z * radius);
			}

			// Rebuild mesh metadata after vertex positions changed
			mesh.CalculateBBox();
			mesh.SetEpsilon(-1.0, false);
			mesh.SortGeometry();
			mesh.SetNormalsAndCoplanar();

			return FromImpl(mesh);
		}

		/// <summary>Extrude polygons along Z.</summary>
		/// <param name="crossSection">The polygons to extrude.</param>
		/// <param name="height">Z extent; must be &gt; 0.</param>
		/// <param name="nDivisions">Extra vertical divisions.</param>
		/// <param name="twistDegrees">Rotation applied to the top.</param>
		/// <param name="scaleTop">X/Y scaling applied to the top; (0,0) makes a cone.</param>
		/// <returns>The extrusion, or an empty manifold with
		/// <see cref="Error.InvalidConstruction"/>.</returns>
		public static Manifold Extrude(
			Polygons crossSection,
			double height,
			int nDivisions,
			double twistDegrees,
			Vec2 scaleTop)
		{
			ArgumentNullException.ThrowIfNull(crossSection);

			if (crossSection.Count == 0 || height <= 0.0)
			{
				return MakeEmpty(Error.InvalidConstruction);
			}

			return FromImpl(Constructors.Extrude(
				crossSection,
				height,
				nDivisions,
				twistDegrees,
				scaleTop));
		}

		/// <summary>Revolve polygons about the Y axis (which becomes Z).</summary>
		/// <param name="crossSection">The polygons to revolve.</param>
		/// <param name="circularSegments">Divisions around the circle; 0 means "ask Quality".</param>
		/// <param name="revolveDegrees">How far to revolve.</param>
		/// <returns>The solid of revolution, or an empty manifold with
		/// <see cref="Error.InvalidConstruction"/>.</returns>
		public static Manifold Revolve(
			Polygons crossSection,
			int circularSegments,
			double revolveDegrees)
		{
			ArgumentNullException.ThrowIfNull(crossSection);

			if (crossSection.Count == 0)
			{
				return MakeEmpty(Error.InvalidConstruction);
			}

			return FromImpl(Constructors.Revolve(crossSection, circularSegments, revolveDegrees));
		}

		/// <summary>The zero level set of a signed distance function.</summary>
		/// <remarks>
		/// <see cref="Sdf.LevelSet"/>'s remarks carry the two obligations this inherits:
		/// <paramref name="sdfFn"/> must be pure and thread-safe, and if it throws, a
		/// single fault propagates unwrapped while several concurrent faults surface as an
		/// <see cref="AggregateException"/> — reachable only with
		/// <see cref="ManifoldParallel.Enabled"/> set.
		/// </remarks>
		/// <param name="sdfFn">
		/// The signed distance function. Must be pure and thread-safe; see
		/// <see cref="Sdf.LevelSet"/>'s remarks.
		/// </param>
		/// <param name="bounds">The region to sample.</param>
		/// <param name="edgeLength">The voxel edge length.</param>
		/// <returns>The extracted surface.</returns>
		public static Manifold LevelSet(Func<Vec3, double> sdfFn, Box bounds, double edgeLength)
		{
			return LevelSetWithTolerance(sdfFn, bounds, edgeLength, 0.0, -1.0);
		}

		/// <summary><see cref="LevelSet"/> at a non-zero level.</summary>
		/// <remarks>
		/// <see cref="Sdf.LevelSet"/>'s remarks carry the two obligations this inherits:
		/// <paramref name="sdfFn"/> must be pure and thread-safe, and if it throws, a
		/// single fault propagates unwrapped while several concurrent faults surface as an
		/// <see cref="AggregateException"/> — reachable only with
		/// <see cref="ManifoldParallel.Enabled"/> set.
		/// </remarks>
		/// <param name="sdfFn">
		/// The signed distance function. Must be pure and thread-safe; see
		/// <see cref="Sdf.LevelSet"/>'s remarks.
		/// </param>
		/// <param name="bounds">The region to sample.</param>
		/// <param name="edgeLength">The voxel edge length.</param>
		/// <param name="level">The iso-level to extract.</param>
		/// <returns>The extracted surface.</returns>
		public static Manifold LevelSetWithLevel(
			Func<Vec3, double> sdfFn,
			Box bounds,
			double edgeLength,
			double level)
		{
			return LevelSetWithTolerance(sdfFn, bounds, edgeLength, level, -1.0);
		}

		/// <summary>
		/// Full-parameter LevelSet matching C++ <c>Manifold::LevelSet(sdf, bounds,
		/// edgeLength, level, tolerance)</c>. Positive <paramref name="tolerance"/> refines
		/// each crossing vertex to within that distance of the true surface.
		/// </summary>
		/// <remarks>
		/// <see cref="Sdf.LevelSet"/>'s remarks carry the two obligations this inherits:
		/// <paramref name="sdfFn"/> must be pure and thread-safe, and if it throws, a
		/// single fault propagates unwrapped while several concurrent faults surface as an
		/// <see cref="AggregateException"/> — reachable only with
		/// <see cref="ManifoldParallel.Enabled"/> set.
		/// </remarks>
		/// <param name="sdfFn">
		/// The signed distance function. Must be pure and thread-safe; see
		/// <see cref="Sdf.LevelSet"/>'s remarks.
		/// </param>
		/// <param name="bounds">The region to sample.</param>
		/// <param name="edgeLength">The voxel edge length.</param>
		/// <param name="level">The iso-level to extract.</param>
		/// <param name="tolerance">The crossing-refinement tolerance, or negative for none.</param>
		/// <returns>The extracted surface.</returns>
		public static Manifold LevelSetWithTolerance(
			Func<Vec3, double> sdfFn,
			Box bounds,
			double edgeLength,
			double level,
			double tolerance)
		{
			ArgumentNullException.ThrowIfNull(sdfFn);
			return FromImpl(Sdf.LevelSet(sdfFn, bounds, edgeLength, level, tolerance));
		}

		/// <summary>The convex hull of a point cloud.</summary>
		/// <param name="points">The points to hull.</param>
		/// <returns>The hull.</returns>
		public static Manifold Hull(IReadOnlyList<Vec3> points)
		{
			ArgumentNullException.ThrowIfNull(points);
			return FromImpl(QuickHullFunctions.ConvexHull(points));
		}

		/// <summary>
		/// Compute the convex hull of multiple manifolds' combined vertices.
		/// If any manifold is errored, propagates its error status.
		/// </summary>
		/// <param name="manifolds">The manifolds whose vertices to hull.</param>
		/// <returns>The hull, or the first errored input.</returns>
		public static Manifold HullManifolds(IReadOnlyList<Manifold> manifolds)
		{
			ArgumentNullException.ThrowIfNull(manifolds);

			// Propagate any error from inputs
			foreach (Manifold m in manifolds)
			{
				if (m.imp.Status != Error.NoError)
				{
					return m.Clone();
				}
			}

			List<Vec3> allVerts = new List<Vec3>();
			foreach (Manifold m in manifolds)
			{
				allVerts.AddRange(m.imp.VertPos);
			}

			return FromImpl(QuickHullFunctions.ConvexHull(allVerts));
		}

		/// <summary>A square cross-section — the 2D helper C++ hangs off Manifold.</summary>
		/// <param name="size">The side length.</param>
		/// <returns>The square.</returns>
		public static CrossSection CrossSectionSquare(double size)
		{
			return CrossSection.Square(size);
		}

		/// <summary>A circular cross-section — the 2D helper C++ hangs off Manifold.</summary>
		/// <param name="radius">The radius.</param>
		/// <param name="segments">The number of sides; 0 means "ask Quality".</param>
		/// <returns>The circle.</returns>
		public static CrossSection CrossSectionCircle(double radius, int segments)
		{
			return CrossSection.Circle(radius, segments);
		}

		/// <summary>Compute the convex hull of this manifold's vertices.</summary>
		/// <returns>The hull.</returns>
		public Manifold ConvexHull()
		{
			if (this.IsEmpty())
			{
				return this.Clone();
			}

			return FromImpl(QuickHullFunctions.ConvexHull(this.imp.VertPos));
		}

		/// <summary>The Minkowski sum of this manifold with <paramref name="other"/>.</summary>
		/// <remarks>
		/// Cancellation and progress travel together for the same reason they do on the
		/// booleans (see <see cref="BooleanWithEngineAndProgress"/>); pass null for either
		/// independently, and both null is byte-for-byte the uninstrumented path. A cancelled
		/// run comes back empty with <see cref="Error.Cancelled"/>, not as an exception.
		/// </remarks>
		/// <param name="other">The structuring manifold.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>The sum, or a propagated error.</returns>
		public Manifold MinkowskiSum(
			Manifold other,
			CancelToken? token = null,
			ProgressReporter? progress = null)
		{
			ArgumentNullException.ThrowIfNull(other);

			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			Manifold? e2 = other.RequirePaired();
			if (e2 != null)
			{
				return e2;
			}

			// Per C++ #1659: propagate errored input status before computing.
			if (this.imp.Status != Error.NoError)
			{
				return this.Clone();
			}

			if (other.imp.Status != Error.NoError)
			{
				return other.Clone();
			}

			return FromImpl(Minkowski.Sum(this.imp, other.imp, token, progress));
		}

		/// <summary>The Minkowski difference of this manifold with <paramref name="other"/>.</summary>
		/// <inheritdoc cref="MinkowskiSum"/>
		/// <param name="other">The structuring manifold.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>The difference, or a propagated error.</returns>
		public Manifold MinkowskiDifference(
			Manifold other,
			CancelToken? token = null,
			ProgressReporter? progress = null)
		{
			ArgumentNullException.ThrowIfNull(other);

			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			Manifold? e2 = other.RequirePaired();
			if (e2 != null)
			{
				return e2;
			}

			if (this.imp.Status != Error.NoError)
			{
				return this.Clone();
			}

			if (other.imp.Status != Error.NoError)
			{
				return other.Clone();
			}

			return FromImpl(Minkowski.Difference(this.imp, other.imp, token, progress));
		}

		/// <summary>
		/// <see cref="MinkowskiDifference"/> in closed form, for the convex solids that have
		/// one — the halfspace intersection of <see cref="ConvexErosion"/>.
		/// </summary>
		/// <remarks>
		/// A fast path a caller opts into, not a reroute: <see cref="MinkowskiDifference"/>
		/// still runs the ported sweep for every input, so nothing that was bit-identical to
		/// manifold-rust has moved. This is the entry point that has no Rust counterpart at
		/// all (divergence ledger entry 5), and the contract is deliberately blunt — it
		/// answers false for anything it cannot prove itself on, and the answer to a false is
		/// to call <see cref="MinkowskiDifference"/>, which is the specification it is
		/// measured against.
		/// <para>
		/// Worth taking because erosion is the one operation with no cheap branch: a sweep
		/// costs a convex hull and a boolean per triangle of the solid, and the closed form
		/// costs two hulls total, whatever the triangle count.
		/// </para>
		/// </remarks>
		/// <param name="other">The structuring manifold.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <param name="result">
		/// The eroded solid when this returns true — including a cancelled run's empty
		/// result, which comes back as true so a cancelled caller does not go on to run the
		/// sweep. Empty when it returns false.
		/// </param>
		/// <returns>True when the closed form applied.</returns>
		public bool TryConvexErosion(
			Manifold other,
			CancelToken? token,
			ProgressReporter? progress,
			out Manifold result)
		{
			ArgumentNullException.ThrowIfNull(other);

			result = Empty();

			// Unpaired halfedges make IsConvex read a neighbour that is not there, and an
			// errored operand has a status the sweep is the one that knows how to propagate.
			// Both are handed straight back rather than answered here.
			if (this.RequirePaired() != null || other.RequirePaired() != null)
			{
				return false;
			}

			if (!ConvexErosion.TryCompute(this.imp, other.imp, token, progress, out ManifoldImpl eroded))
			{
				return false;
			}

			result = FromImpl(eroded);
			return true;
		}
	}
}
