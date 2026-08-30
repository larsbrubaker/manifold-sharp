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

// RobustSoupTests.cs — port of robust/soup_tests.rs, whose header reads:
//
//   Tests for the robust (triangle-soup) import path: manifold input matches
//   the strict import, non-manifold-but-closed input is retained as a soup,
//   open/imbalanced input is rejected with NotClosed, the strict path is
//   unchanged, and every public Manifold operation either works or degrades
//   gracefully (no panics) on a soup — the "all-ops checklist".
//
// Same inputs, same expected values, same order as the Rust's 20 tests. Nothing
// here is deferred.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class RobustSoupTests
	{
		private static Vec3 V(double x, double y, double z)
		{
			return new Vec3(x, y, z);
		}

		/// <summary>Cube [lo,hi]³ as 12 outward triangles.</summary>
		private static List<Vec3[]> CubeTris(double[] lo, double[] hi)
		{
			double x0 = lo[0];
			double y0 = lo[1];
			double z0 = lo[2];
			double x1 = hi[0];
			double y1 = hi[1];
			double z1 = hi[2];
			Vec3[][] quads = new Vec3[][]
			{
				new Vec3[] { V(x0, y0, z0), V(x0, y1, z0), V(x1, y1, z0), V(x1, y0, z0) },
				new Vec3[] { V(x0, y0, z1), V(x1, y0, z1), V(x1, y1, z1), V(x0, y1, z1) },
				new Vec3[] { V(x0, y0, z0), V(x1, y0, z0), V(x1, y0, z1), V(x0, y0, z1) },
				new Vec3[] { V(x0, y1, z0), V(x0, y1, z1), V(x1, y1, z1), V(x1, y1, z0) },
				new Vec3[] { V(x0, y0, z0), V(x0, y0, z1), V(x0, y1, z1), V(x0, y1, z0) },
				new Vec3[] { V(x1, y0, z0), V(x1, y1, z0), V(x1, y1, z1), V(x1, y0, z1) },
			};
			List<Vec3[]> outTris = new List<Vec3[]>();
			foreach (Vec3[] q in quads)
			{
				outTris.Add(new Vec3[] { q[0], q[1], q[2] });
				outTris.Add(new Vec3[] { q[0], q[2], q[3] });
			}

			return outTris;
		}

		/// <summary>
		/// Fully disconnected triangle-soup MeshGL (3 duplicated verts per tri) —
		/// guaranteed to fail strict halfedge pairing while staying geometrically
		/// identical to the source.
		/// </summary>
		private static MeshGL SoupMeshGl(IReadOnlyList<Vec3[]> tris)
		{
			MeshGL mesh = new MeshGL();
			mesh.NumProp = 3;
			foreach (Vec3[] t in tris)
			{
				foreach (Vec3 p in t)
				{
					mesh.VertProperties.Add((float)p.X);
					mesh.VertProperties.Add((float)p.Y);
					mesh.VertProperties.Add((float)p.Z);
				}
			}

			for (uint i = 0; i < 3 * (uint)tris.Count; i++)
			{
				mesh.TriVerts.Add(i);
			}

			return mesh;
		}

		private static Manifold SoupCube()
		{
			return Manifold.FromMeshGLRobust(SoupMeshGl(CubeTris(Fill(0.0), Fill(2.0))));
		}

		/// <summary>The Rust's <c>[x; 3]</c> array literal.</summary>
		private static double[] Fill(double x)
		{
			return new double[] { x, x, x };
		}

		[Test]
		public async Task RobustImportOfManifoldMeshMatchesStrict()
		{
			Manifold strict = Manifold.Cube(V(2.0, 2.0, 2.0), false);
			MeshGL gl = strict.GetMeshGL(-1);
			Manifold a = Manifold.FromMeshGL(gl);
			Manifold b = Manifold.FromMeshGLRobust(gl);
			await Assert.That(a.Status()).IsEqualTo(Error.NoError);
			await Assert.That(b.Status()).IsEqualTo(Error.NoError);
			await Assert.That(a.NumVert()).IsEqualTo(b.NumVert());
			await Assert.That(a.NumTri()).IsEqualTo(b.NumTri());
			await Assert.That(a.Volume()).IsEqualTo(b.Volume());
			await Assert.That(b.AsImpl().IsSoup).IsFalse()
				.Because("manifold input must not become a soup");
		}

		[Test]
		public async Task DuplicatedVertSoupIsRetained()
		{
			Manifold m = SoupCube();
			await Assert.That(m.Status()).IsEqualTo(Error.NoError);
			await Assert.That(m.AsImpl().IsSoup).IsTrue();
			await Assert.That(m.NumTri()).IsEqualTo(12);

			// Volume/area work from triangles alone.
			await Assert.That(m.Volume()).IsEqualTo(8.0);
			await Assert.That(m.SurfaceArea()).IsEqualTo(24.0);

			// Strict import of the same mesh still refuses.
			Manifold strict = Manifold.FromMeshGL(SoupMeshGl(CubeTris(Fill(0.0), Fill(2.0))));
			await Assert.That(strict.Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(strict.IsEmpty()).IsTrue();
		}

		[Test]
		public async Task EdgeSharingCubesImportAsSoup()
		{
			// Two cubes sharing exactly one edge — a genuinely non-manifold closed
			// solid no strict import can represent.
			List<Vec3[]> tris = CubeTris(Fill(0.0), Fill(2.0));
			tris.AddRange(CubeTris(new double[] { 2.0, 2.0, 0.0 }, new double[] { 4.0, 4.0, 2.0 }));
			Manifold m = Manifold.FromMeshGLRobust(SoupMeshGl(tris));
			await Assert.That(m.Status()).IsEqualTo(Error.NoError);
			await Assert.That(m.AsImpl().IsSoup).IsTrue();
			await Assert.That(m.NumTri()).IsEqualTo(24);
			await Assert.That(m.Volume()).IsEqualTo(16.0);
		}

		[Test]
		public async Task InternalVoidImportsAsSoup()
		{
			List<Vec3[]> tris = CubeTris(Fill(0.0), Fill(6.0));

			// Inner cube flipped inward = void.
			foreach (Vec3[] t in CubeTris(Fill(2.0), Fill(4.0)))
			{
				tris.Add(new Vec3[] { t[0], t[2], t[1] });
			}

			Manifold m = Manifold.FromMeshGLRobust(SoupMeshGl(tris));
			await Assert.That(m.Status()).IsEqualTo(Error.NoError);
			await Assert.That(m.Volume()).IsEqualTo((6.0 * 6.0 * 6.0) - 8.0);
		}

		[Test]
		public async Task OpenMeshIsRejectedNotClosed()
		{
			// Cube missing one triangle → open.
			List<Vec3[]> tris = CubeTris(Fill(0.0), Fill(2.0));
			tris.RemoveAt(tris.Count - 1);
			Manifold m = Manifold.FromMeshGLRobust(SoupMeshGl(tris));
			await Assert.That(m.Status()).IsEqualTo(Error.NotClosed);
			await Assert.That(m.IsEmpty()).IsTrue();
		}

		[Test]
		public async Task UnbalancedOrientationIsRejectedNotClosed()
		{
			// Cube with one triangle flipped → directed edges unbalanced.
			List<Vec3[]> tris = CubeTris(Fill(0.0), Fill(2.0));
			Vec3[] t = tris[0];
			tris[0] = new Vec3[] { t[0], t[2], t[1] };
			Manifold m = Manifold.FromMeshGLRobust(SoupMeshGl(tris));
			await Assert.That(m.Status()).IsEqualTo(Error.NotClosed);
		}

		[Test]
		public async Task DegenerateTrianglesAreDroppedOnSoupImport()
		{
			List<Vec3[]> tris = CubeTris(Fill(0.0), Fill(2.0));

			// Add an exactly-degenerate triangle (collinear); must be ignored.
			tris.Add(new Vec3[] { V(5.0, 5.0, 5.0), V(6.0, 6.0, 6.0), V(7.0, 7.0, 7.0) });
			Manifold m = Manifold.FromMeshGLRobust(SoupMeshGl(tris));
			await Assert.That(m.Status()).IsEqualTo(Error.NoError);
			await Assert.That(m.NumTri()).IsEqualTo(12);
			await Assert.That(m.Volume()).IsEqualTo(8.0);
		}

		[Test]
		public async Task TinyOrEmptyMeshes()
		{
			MeshGL empty = new MeshGL();
			await Assert.That(Manifold.FromMeshGLRobust(empty).Status()).IsEqualTo(Error.NoError);
			MeshGL twoTris = SoupMeshGl(CubeTris(Fill(0.0), Fill(1.0)).GetRange(0, 2));
			await Assert.That(Manifold.FromMeshGLRobust(twoTris).Status()).IsEqualTo(Error.NotClosed);
		}

		// -------------------------------------------------------------------
		// Geometric self-intersection detection (drives Auto engine dispatch)
		// -------------------------------------------------------------------

		[Test]
		public async Task CleanShapesHaveNoSelfIntersections()
		{
			// A strict cube: every same-mesh triangle contact is an ordinary shared
			// edge or shared vertex, which must not count as a self-intersection.
			Manifold cube = Manifold.Cube(V(2.0, 2.0, 2.0), false);
			await Assert.That(cube.HasSelfIntersections()).IsFalse();

			// A denser mesh with fans of coplanar and non-coplanar neighbors.
			Manifold sphere = Manifold.Sphere(1.0, 16);
			await Assert.That(sphere.HasSelfIntersections()).IsFalse();

			// Same geometry imported as an unpaired soup: detection works off
			// positions, not halfedge pairing, so the answer is unchanged.
			await Assert.That(SoupCube().HasSelfIntersections()).IsFalse();
		}

		[Test]
		public async Task OverlappingShellsAreSelfIntersecting()
		{
			// Two cubes whose interiors overlap, concatenated into one closed soup:
			// triangles of the second shell cross faces of the first.
			List<Vec3[]> tris = CubeTris(Fill(0.0), Fill(2.0));
			tris.AddRange(CubeTris(Fill(1.0), Fill(3.0)));
			Manifold m = Manifold.FromMeshGLRobust(SoupMeshGl(tris));
			await Assert.That(m.Status()).IsEqualTo(Error.NoError);
			await Assert.That(m.HasSelfIntersections()).IsTrue();

			// Repeat queries hit the cache and stay consistent.
			await Assert.That(m.HasSelfIntersections()).IsTrue();
		}

		[Test]
		public async Task EdgeOnlyContactsAreNotSelfIntersections()
		{
			// Two cubes sharing exactly one edge: non-manifold connectivity, but the
			// only same-mesh contacts are that shared edge and ordinary adjacency —
			// no geometric self-intersection.
			List<Vec3[]> tris = CubeTris(Fill(0.0), Fill(2.0));
			tris.AddRange(CubeTris(new double[] { 2.0, 2.0, 0.0 }, new double[] { 4.0, 4.0, 2.0 }));
			Manifold m = Manifold.FromMeshGLRobust(SoupMeshGl(tris));
			await Assert.That(m.Status()).IsEqualTo(Error.NoError);
			await Assert.That(m.HasSelfIntersections()).IsFalse();
		}

		[Test]
		public async Task CoincidentSheetsCountAsSelfIntersecting()
		{
			// Doubled surface: every triangle has an exact duplicate. The robust
			// engine needs no cut there (both copies emit the same pieces and the
			// winding arithmetic resolves them), but the surface is coincident, and
			// that is exactly what the exact engine mis-integrates — so the dispatch
			// detector must report it.
			List<Vec3[]> tris = CubeTris(Fill(0.0), Fill(2.0));
			List<Vec3[]> doubled = new List<Vec3[]>(tris);
			foreach (Vec3[] t in doubled)
			{
				tris.Add(new Vec3[] { t[0], t[2], t[1] });
			}

			Manifold m = Manifold.FromMeshGLRobust(SoupMeshGl(tris));
			await Assert.That(m.Status()).IsEqualTo(Error.NoError);
			await Assert.That(m.HasSelfIntersections()).IsTrue();
		}

		[Test]
		public async Task SelfIntersectionIsRecomputedAfterTransform()
		{
			// The verdict is deliberately not carried across a transform (rounded
			// positions can create coincidences the source lacked), so each result
			// is a fresh scan — and each must still be right.
			List<Vec3[]> tris = CubeTris(Fill(0.0), Fill(2.0));
			tris.AddRange(CubeTris(Fill(1.0), Fill(3.0)));
			Manifold m = Manifold.FromMeshGLRobust(SoupMeshGl(tris));
			await Assert.That(m.HasSelfIntersections()).IsTrue();
			await Assert.That(
				m.Translate(V(3.0, -1.0, 0.5)).Scale(V(2.0, 2.0, 2.0)).HasSelfIntersections())
				.IsTrue();
			Manifold clean = Manifold.Cube(V(2.0, 2.0, 2.0), false);
			await Assert.That(clean.HasSelfIntersections()).IsFalse();
			await Assert.That(clean.Rotate(10.0, 20.0, 30.0).HasSelfIntersections()).IsFalse();
		}

		[Test]
		public async Task WarpInvalidatesTheCachedVerdict()
		{
			// Settle a `false` verdict, then fold the cube onto itself: the warped
			// copy must be scanned afresh, not inherit the clone's answer.
			Manifold cube = Manifold.Cube(V(2.0, 2.0, 2.0), false);
			await Assert.That(cube.HasSelfIntersections()).IsFalse();

			// Mirror the top half down through the bottom half — every triangle
			// above z = 1 crosses the material below it.
			Manifold folded = cube.Warp((ref Vec3 p) =>
			{
				if (p.Z > 1.0)
				{
					p.Z = 2.0 - p.Z;
				}
			});
			await Assert.That(folded.HasSelfIntersections()).IsTrue()
				.Because("warp must re-run the scan");

			// simplify() also clones-then-mutates; its verdict must be re-derived.
			await Assert.That(cube.Simplify(0.0).HasSelfIntersections()).IsFalse();
		}

		[Test]
		public async Task NonFinitePositionsReportSelfIntersectingWithoutPanicking()
		{
			// A warp to NaN survives every import check but has no exact rational
			// form, so the detector cannot evaluate it. "Self-intersecting" is the
			// safe answer (route to the robust engine) and, above all, not a panic.
			Manifold cube = Manifold.Cube(V(2.0, 2.0, 2.0), false);
			Manifold nanY = cube.Warp((ref Vec3 p) =>
			{
				if (p.Y > 1.0)
				{
					// Rust `f64::NAN`; C# `double.NaN` has the sign bit set and is a
					// different bit pattern (CLAUDE.md, C# translation rules).
					p.Y = DeterministicMath.PositiveQuietNaN;
				}
			});
			await Assert.That(nanY.IsEmpty()).IsFalse().Because("NaN survives the warp pipeline");
			await Assert.That(nanY.HasSelfIntersections()).IsTrue();

			// Infinities are a different story: the bbox becomes non-finite and the
			// warp pipeline empties the mesh, so the detector never sees them.
			Manifold infX = cube.Warp((ref Vec3 p) =>
			{
				if (p.X > 1.0)
				{
					p.X = double.PositiveInfinity;
				}
			});
			await Assert.That(infX.IsEmpty()).IsTrue();
			await Assert.That(infX.HasSelfIntersections()).IsFalse();
		}

		[Test]
		public async Task SameWindingDuplicatesCannotReachTheDetector()
		{
			// The detector's duplicate-triangle test is winding-agnostic (it
			// compares vertex sets), so a same-winding duplicate would report true
			// exactly like the reversed one in the test above. It cannot arise from
			// any import, though: duplicating a triangle without flipping it leaves
			// its three directed edges doubled in the same direction, which is
			// precisely what soupify's closed-and-orientable check rejects.
			List<Vec3[]> tris = CubeTris(Fill(0.0), Fill(2.0));
			Vec3[] dup = tris[0];
			tris.Add(dup);
			Manifold m = Manifold.FromMeshGLRobust(SoupMeshGl(tris));
			await Assert.That(m.Status()).IsEqualTo(Error.NotClosed);
		}

		[Test]
		public async Task HalfOffsetFaceContactReportsSelfIntersecting()
		{
			// Pin test for the boundary case: two shells whose touching faces
			// overlap in a rectangle (positive area) but do not coincide. That is a
			// coplanar overlap, so the detector reports true — the same verdict it
			// gives fully coincident sheets, and for the same reason (the exact
			// engine cannot integrate shared surface). Contrast with the
			// edge-and-vertex-only contacts above, which report false.
			List<Vec3[]> tris = CubeTris(Fill(0.0), Fill(2.0));
			tris.AddRange(CubeTris(new double[] { 1.0, 0.0, 2.0 }, new double[] { 3.0, 2.0, 4.0 }));
			Manifold m = Manifold.FromMeshGLRobust(SoupMeshGl(tris));
			await Assert.That(m.Status()).IsEqualTo(Error.NoError);
			await Assert.That(m.HasSelfIntersections()).IsTrue();
		}

		[Test]
		public async Task SoupExportRoundTrips()
		{
			Manifold m = SoupCube();
			MeshGL gl = m.GetMeshGL(-1);
			await Assert.That(gl.NumTri()).IsEqualTo(12);
			Manifold re = Manifold.FromMeshGLRobust(gl);
			await Assert.That(re.Status()).IsEqualTo(Error.NoError);
			await Assert.That(re.Volume()).IsEqualTo(8.0);

			// 64-bit export too.
			MeshGL64 gl64 = m.GetMeshGL64(-1);
			Manifold re64 = Manifold.FromMeshGL64Robust(gl64);
			await Assert.That(re64.Status()).IsEqualTo(Error.NoError);
			await Assert.That(re64.Volume()).IsEqualTo(8.0);
		}

		[Test]
		public async Task SoupTransformsWorkAndStaySoup()
		{
			Manifold m = SoupCube();
			Manifold t = m.Translate(V(10.0, 0.0, 0.0));
			await Assert.That(t.Status()).IsEqualTo(Error.NoError);

			// Transformed coordinates change the f64 summation order in the volume
			// accumulation — allow last-ulp wobble on all transform volumes.
			await Assert.That(Math.Abs(t.Volume() - 8.0) < 1e-12).IsTrue();
			await Assert.That(t.BoundingBox().Min.X).IsEqualTo(10.0);
			await Assert.That(t.AsImpl().IsSoup).IsTrue().Because("transform must preserve soup-ness");
			Manifold s = m.Scale(V(2.0, 1.0, 1.0));
			await Assert.That(Math.Abs(s.Volume() - 16.0) < 1e-12).IsTrue();
			Manifold r = m.Rotate(0.0, 0.0, 90.0);
			await Assert.That(r.Status()).IsEqualTo(Error.NoError);
			Manifold mi = m.Mirror(V(1.0, 0.0, 0.0));
			await Assert.That(mi.Status()).IsEqualTo(Error.NoError);

			// Mirrored coordinates change the f64 summation order in the volume
			// accumulation — allow the last-ulp wobble.
			await Assert.That(Math.Abs(mi.Volume() - 8.0) < 1e-12).IsTrue();
		}

		/// <summary>
		/// The all-ops checklist: every pairing-dependent public operation on a soup
		/// must return a graceful empty result (status NotManifold), never panic;
		/// the always-safe queries must keep working.
		/// </summary>
		/// <remarks>
		/// Carries <see cref="RobustEngineTests.BooleanConfigGlobalStateKey"/> because the
		/// NotManifold expectations below go through the PLAIN boolean entry points, which
		/// read the process-global default engine. Under
		/// <see cref="BooleanEngine.Auto"/> a soup operand routes to the robust engine and
		/// answers NoError with real geometry instead of refusing — so this test must never
		/// run concurrently with the one that flips that default
		/// (<see cref="RobustEngineTests.GlobalDefaultEngineConfig"/>). See the key's own
		/// doc for why most readers of the default need no key and this one does.
		/// </remarks>
		[Test]
		[NotInParallel(RobustEngineTests.BooleanConfigGlobalStateKey)]
		public async Task AllOpsChecklistOnSoup()
		{
			Manifold m = SoupCube();
			Manifold other = Manifold.Cube(V(1.0, 1.0, 1.0), false);

			// Safe queries.
			await Assert.That(m.Status()).IsEqualTo(Error.NoError);
			await Assert.That(m.IsEmpty()).IsFalse();
			await Assert.That(m.NumTri()).IsEqualTo(12);
			_ = m.NumVert();
			_ = m.NumEdge();
			_ = m.BoundingBox();
			_ = m.Volume();
			_ = m.SurfaceArea();
			_ = m.GetTolerance();
			_ = m.GetEpsilon();
			_ = m.OriginalId();
			_ = m.MatchesTriNormals();
			_ = m.NumDegenerateTris();

			// Hulls work from vertices alone.
			Manifold hull = m.ConvexHull();
			await Assert.That(hull.Status()).IsEqualTo(Error.NoError);
			await Assert.That(hull.Volume()).IsEqualTo(8.0);

			// Exact-engine booleans refuse soups.
			foreach (OpType op in new OpType[] { OpType.Add, OpType.Subtract, OpType.Intersect })
			{
				Manifold r = m.Boolean(other, op);
				await Assert.That(r.Status()).IsEqualTo(Error.NotManifold).Because($"op {op}");
				await Assert.That(r.IsEmpty()).IsTrue();
				Manifold r2 = other.Boolean(m, op);
				await Assert.That(r2.Status()).IsEqualTo(Error.NotManifold);
			}

			await Assert.That(m.Union(other).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.Difference(other).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.Intersection(other).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(
				Manifold.BatchBoolean(new List<Manifold> { m.Clone(), other.Clone() }, OpType.Add).Status())
				.IsEqualTo(Error.NotManifold);

			// Split family goes through booleans → empty NotManifold results.
			(Manifold a, Manifold b) = m.Split(other);
			await Assert.That(a.Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(b.Status()).IsEqualTo(Error.NotManifold);
			(Manifold pa, Manifold pb) = m.SplitByPlane(V(0.0, 0.0, 1.0), 1.0);
			await Assert.That(pa.IsEmpty() && pb.IsEmpty()).IsTrue();
			await Assert.That(m.TrimByPlane(V(0.0, 0.0, 1.0), 1.0).IsEmpty()).IsTrue();

			// Pairing-dependent unary ops refuse gracefully.
			await Assert.That(m.AsOriginal().Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.SetTolerance(0.1).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.Simplify(0.1).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.Warp((ref Vec3 p) => { }).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.WarpBatch((Span<Vec3> verts) => { }).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.Refine(2).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.RefineToLength(0.5).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.RefineToTolerance(0.5).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.SmoothOut(60.0, 0.0).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.SmoothByNormals(0).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.CalculateNormals(0, 60.0).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.CalculateCurvature(-1, -1).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(
				m.SetProperties(1, (Span<double> p, Vec3 _, ReadOnlySpan<double> _) => p[0] = 1.0).Status())
				.IsEqualTo(Error.NotManifold);
			List<Manifold> parts = m.Decompose();
			await Assert.That(parts.Count).IsEqualTo(1);
			await Assert.That(parts[0].Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.MinkowskiSum(other).Status()).IsEqualTo(Error.NotManifold);
			await Assert.That(m.MinkowskiDifference(other).Status()).IsEqualTo(Error.NotManifold);

			// Cross-section ops return empty sections.
			await Assert.That(m.Slice(1.0).IsEmpty()).IsTrue();
			await Assert.That(m.Project().IsEmpty()).IsTrue();
		}
	}
}
