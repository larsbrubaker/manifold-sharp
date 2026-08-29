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

// Port of quickhull.cpp + quickhull.h -- 3D convex hull using QuickHull algorithm.
//
// Derived from the public domain work of Antti Kuukka at
// https://github.com/akuukka/quickhull
//
// ── File split ───────────────────────────────────────────────────────────────
// The Rust is two files — quickhull.rs (the public entry plus the geometry
// helpers, this file) and quickhull_algo.rs (the machinery, which carries a
// documented exemption from the 800-line cap because its pieces are one tightly
// coupled unit). The C# expansion of that second file does not fit one file, so
// it lands as two more, both continuing the same unit:
//   QuickHull.cs        this file — DefaultEps, the geometry helpers, Plane,
//                       and ConvexHull
//   QuickHull.Mesh.cs   Face, QHEdge, MeshBuilder, Pool, FaceData — the arena
//   QuickHull.Algo.cs   the QuickHull driver
// The coupling the Rust exemption is about survives the split: MeshBuilder's
// invariants (a face's three halfedges are a `HalfedgeNext` ring, a disabled
// face is `He == -1`, a disabled halfedge is `PairedHalfedge == -1`) are
// established and relied on by the driver, and neither file can be read alone.
//
// ── Naming ───────────────────────────────────────────────────────────────────
// quickhull.rs's free functions would land on a static class named `QuickHull`
// per CLAUDE.md, but quickhull_algo.rs's primary type already has that name, so
// the functions class takes the `Functions` suffix (same rule that produced
// `LinalgFunctions`). The five module-private Rust types keep their Rust
// spellings — `Plane`, `Face`, `QHEdge`, `MeshBuilder`, `Pool`, `FaceData` — as
// assembly-`internal` types, because they are the names the three-way diff
// against the Rust and the C++ is read through. They are quickhull-private in
// spirit; nothing outside these three files may use them.
//
// ── Float comparisons are load-bearing ───────────────────────────────────────
// Hull robustness rests on the exact comparison order and on several *equality*
// tests against an epsilon that was just used as a running maximum
// (`max_d == self.epsilon`, `max_d == self.epsilon_squared` in
// SetupInitialTetrahedron): those read as "the scan found nothing strictly
// better", and rewriting them as tolerance compares changes which degenerate
// branch is taken. Every comparison here is transcribed literally.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// A plane through a point with a (not necessarily unit) normal, as used by the
	/// QuickHull face records.
	/// </summary>
	/// <remarks>
	/// <see cref="SqrNLength"/> is cached because <c>AddPointToFace</c> compares a squared
	/// signed distance against <c>epsilonSquared * SqrNLength</c> rather than normalizing.
	/// </remarks>
	internal struct Plane
	{
		/// <summary>The plane normal, not normalized in general.</summary>
		public Vec3 N;

		/// <summary>The plane offset, <c>dot(-N, point)</c>.</summary>
		public double D;

		/// <summary>The cached <c>dot(N, N)</c>.</summary>
		public double SqrNLength;

		/// <summary>Creates the plane with normal <paramref name="n"/> through <paramref name="point"/>.</summary>
		/// <param name="n">The plane normal.</param>
		/// <param name="point">A point on the plane.</param>
		public Plane(Vec3 n, Vec3 point)
		{
			this.D = Dot(-n, point);
			this.SqrNLength = Dot(n, n);
			this.N = n;
		}

		/// <summary>
		/// The Rust <c>Plane::default()</c> — a zero normal at zero offset, which is what a
		/// freshly created <see cref="Face"/> carries until its plane is computed.
		/// </summary>
		/// <returns>The all-zero plane.</returns>
		public static Plane Default()
		{
			// Every field is zero, so `default(Plane)` is the Rust default exactly; it is
			// spelled out here so the Face constructor can read like its Rust.
			return new Plane
			{
				N = new Vec3(0.0, 0.0, 0.0),
				D = 0.0,
				SqrNLength = 0.0,
			};
		}

		/// <summary>True when <paramref name="q"/> is on the positive side of the plane, boundary included.</summary>
		/// <param name="q">The point to classify.</param>
		/// <returns>True when the signed distance is not negative.</returns>
		public bool IsPointOnPositiveSide(Vec3 q)
		{
			return Dot(this.N, q) + this.D >= 0.0;
		}
	}

	/// <summary>
	/// The <c>quickhull.rs</c> module: the convex hull entry point and its geometry
	/// helpers.
	/// </summary>
	public static class QuickHullFunctions
	{
		/// <summary>
		/// The relative epsilon the hull is built with; scaled by the point cloud's extent
		/// in <see cref="QuickHull.BuildMesh"/>.
		/// </summary>
		internal const double DefaultEps = 0.0000001;

		// ---------------------------------------------------------------------
		// Geometry helpers
		// ---------------------------------------------------------------------

		/// <summary>The squared distance between two points.</summary>
		/// <param name="p1">The first point.</param>
		/// <param name="p2">The second point.</param>
		/// <returns>The squared distance.</returns>
		internal static double SquaredDistance(Vec3 p1, Vec3 p2)
		{
			return Dot(p1 - p2, p1 - p2);
		}

		/// <summary>
		/// The squared distance from a point to an infinite ray, given the ray direction's
		/// precomputed inverse squared length.
		/// </summary>
		/// <param name="p">The point.</param>
		/// <param name="rayS">A point on the ray.</param>
		/// <param name="rayV">The ray direction, not normalized.</param>
		/// <param name="vInvLenSq"><c>1 / dot(rayV, rayV)</c>.</param>
		/// <returns>The squared perpendicular distance.</returns>
		internal static double SquaredDistancePointRay(Vec3 p, Vec3 rayS, Vec3 rayV, double vInvLenSq)
		{
			Vec3 s = p - rayS;
			double t = Dot(s, rayV);
			return Dot(s, s) - (t * t * vInvLenSq);
		}

		/// <summary>
		/// The unit normal of the triangle <paramref name="a"/>, <paramref name="b"/>,
		/// <paramref name="c"/>.
		/// </summary>
		/// <remarks>
		/// The cross product is written out rather than delegated to <c>Cross</c>: the
		/// operand order below is the one the C++ and Rust use, and it decides the last
		/// bits of the normal, which decide face visibility at the epsilon boundary.
		/// </remarks>
		/// <param name="a">The first corner.</param>
		/// <param name="b">The second corner.</param>
		/// <param name="c">The third corner.</param>
		/// <returns>The normalized triangle normal.</returns>
		internal static Vec3 TriangleNormal(Vec3 a, Vec3 b, Vec3 c)
		{
			double x = a.X - c.X;
			double y = a.Y - c.Y;
			double z = a.Z - c.Z;
			double rhsx = b.X - c.X;
			double rhsy = b.Y - c.Y;
			double rhsz = b.Z - c.Z;
			double px = (y * rhsz) - (z * rhsy);
			double py = (z * rhsx) - (x * rhsz);
			double pz = (x * rhsy) - (y * rhsx);
			return Normalize(new Vec3(px, py, pz));
		}

		/// <summary>The signed distance from <paramref name="v"/> to the plane <paramref name="p"/>.</summary>
		/// <param name="v">The point.</param>
		/// <param name="p">The plane.</param>
		/// <returns>The signed distance, scaled by the normal's length.</returns>
		internal static double SignedDistanceToPlane(Vec3 v, in Plane p)
		{
			return Dot(p.N, v) + p.D;
		}

		// ---------------------------------------------------------------------
		// Public API
		// ---------------------------------------------------------------------

		/// <summary>
		/// Compute the convex hull of a set of 3D points, returning a ManifoldImpl.
		/// </summary>
		/// <param name="points">The point cloud to hull.</param>
		/// <returns>
		/// The hull as a manifold impl, or an empty impl when there are no points or the
		/// hull is degenerate enough to produce no halfedges.
		/// </returns>
		public static ManifoldImpl ConvexHull(IReadOnlyList<Vec3> points)
		{
			if (points.Count == 0)
			{
				return new ManifoldImpl();
			}

			QuickHull qh = new QuickHull(points);
			(List<Halfedge> halfedges, List<Vec3> vertices) = qh.BuildMesh(DefaultEps);

			if (halfedges.Count == 0)
			{
				return new ManifoldImpl();
			}

			ManifoldImpl imp = new ManifoldImpl();
			imp.Halfedge = halfedges;
			imp.VertPos = vertices;
			imp.CalculateBBox();
			imp.SetEpsilon(-1.0, false);
			imp.InitializeOriginal();
			imp.SortGeometry();
			imp.SetNormalsAndCoplanar();
			return imp;
		}
	}
}
