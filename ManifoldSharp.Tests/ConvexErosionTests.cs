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

// ConvexErosionTests.cs — NOT A PORT, and counted as a C#-only adaptation test the
// way CLAUDE.md requires: its subject is ConvexErosion.cs, which manifold-rust has
// no counterpart for (divergence ledger entry 5). No Rust expected value appears
// here, because there is no Rust function to have produced one.
//
// TWO ORACLES, deliberately, because the closed form is a second answer to a
// question the library already answers and the whole risk is that the two disagree:
//
//  1. An INDEPENDENT one, HalfspaceIntersectionByTriples below. It enumerates every
//     triple of offset face planes, keeps the candidate points that satisfy the
//     DEFINITION of erosion — x - b lies inside the solid for every vertex b of the
//     tool — and hulls what survives. Different algorithm, different failure modes:
//     it shares no code with the dual-hull construction it checks, and its filter is
//     the definition rather than the plane-offset shortcut the production path is
//     built on, so it would catch that shortcut being wrong. It is O(faces^3) and
//     could only ever be a test.
//  2. The GENERAL SWEEP, Minkowski.Difference — the ported algorithm, which is the
//     specification ConvexErosion promises to agree with and the thing a caller gets
//     when the closed form declines. This is the equivalence oracle the opening test
//     uses, since Round All Edges is an erosion followed by a dilation and what
//     matters there is that swapping the erosion out changes nothing a user sees.
//
// Volumes are compared relatively at 1e-6. They are not bit-identical and cannot be:
// the sweep reaches the same solid through a union of per-triangle hulls and a
// boolean, so its vertices carry that path's rounding, while the closed form solves
// each vertex from three planes. On a box both land exactly; on a tessellated sphere
// they differ in the twelfth digit. The one exception is
// ADenseSolidAgreesWithTheSweepOnlyToATolerance, which needs 5e-6 and says why —
// past about a thousand faces the dual hull starts discarding near-coplanar points
// and the two answers genuinely part company.
//
// The other half of this class is ConvexErosionTests.Contract.cs: the routing pin,
// every decline, and the progress/cancellation contract. Shared helpers live here.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp.Tests
{
	public partial class ConvexErosionTests
	{
		/// <summary>
		/// Relative slop between the closed form and either oracle. See the file header
		/// for why it is not zero.
		/// </summary>
		private const double VolumeTolerance = 1e-6;

		/// <summary>
		/// A cube eroded by the kernel's ball is a smaller cube, and the arithmetic is
		/// exact — the one case where an analytic answer can be asserted to the bit.
		/// </summary>
		/// <remarks>
		/// The ball is a subdivided octahedron, so its support along an axis is exactly the
		/// radius (a vertex sits on each pole). A 20-cube eroded by radius 1 is therefore
		/// an 18-cube, volume 5832, and every coordinate is a whole number that a double
		/// holds exactly. Nothing else in this file can assert equality; this one can, and
		/// so it does — a closed form that is only ever checked against a tolerance has no
		/// test that would notice it drifting by an ulp per release.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ACubeErodesToAnExactlySmallerCube()
		{
			Manifold cube = Manifold.Cube(Vec3.Splat(20.0), true);
			Manifold ball = Manifold.Sphere(1.0, 12);

			await Assert.That(cube.TryConvexErosion(ball, null, null, out Manifold eroded)).IsTrue()
				.Because("a cube and a ball are both convex, which is the whole gate");

			await Assert.That(eroded.Volume()).IsEqualTo(5832.0)
				.Because("18^3 exactly: the plane offsets and the corner solves are all whole numbers here");
			await Assert.That(eroded.NumTri()).IsEqualTo(12);

			Box bounds = eroded.BoundingBox();
			await Assert.That(bounds.Min.X).IsEqualTo(-9.0);
			await Assert.That(bounds.Min.Y).IsEqualTo(-9.0);
			await Assert.That(bounds.Min.Z).IsEqualTo(-9.0);
			await Assert.That(bounds.Max.X).IsEqualTo(9.0);
			await Assert.That(bounds.Max.Y).IsEqualTo(9.0);
			await Assert.That(bounds.Max.Z).IsEqualTo(9.0);
		}

		/// <summary>
		/// The closed form is the halfspace intersection, on each of the three convex
		/// polyhedra — measured against a brute-force enumeration that shares none of its
		/// code.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[Arguments("cube", 0)]
		[Arguments("tetrahedron", 1)]
		[Arguments("icosahedron", 2)]
		public async Task TheClosedFormIsTheHalfspaceIntersection(string label, int shape)
		{
			Manifold solid = Polyhedron(shape);
			Manifold ball = Manifold.Sphere(1.0, 12);

			await Assert.That(solid.TryConvexErosion(ball, null, null, out Manifold eroded)).IsTrue()
				.Because($"{label} is convex");

			Manifold expected = HalfspaceIntersectionByTriples(solid, ball);

			await Assert.That(Relative(eroded.Volume(), expected.Volume())).IsLessThan(VolumeTolerance)
				.Because($"{label}: closed form {eroded.Volume()} vs brute-force halfspace intersection {expected.Volume()}");
			await Assert.That(Relative(eroded.SurfaceArea(), expected.SurfaceArea())).IsLessThan(VolumeTolerance)
				.Because($"{label}: the surfaces have to agree too, or the volumes agreeing was luck");
		}

		/// <summary>
		/// Every vertex the closed form produces is a place the tool actually fits — the
		/// definition of erosion, checked directly on the answer rather than on the
		/// construction that built it.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		[Arguments("cube", 0)]
		[Arguments("tetrahedron", 1)]
		[Arguments("icosahedron", 2)]
		public async Task EveryErodedVertexIsSomewhereTheToolStillFits(string label, int shape)
		{
			Manifold solid = Polyhedron(shape);
			Manifold ball = Manifold.Sphere(1.0, 12);
			await Assert.That(solid.TryConvexErosion(ball, null, null, out Manifold eroded)).IsTrue();

			(List<Vec3> normals, List<double> offsets) = FacePlanes(solid);
			double slop = Length(solid.BoundingBox().Size()) * 1e-9;

			List<Vec3> verts = eroded.AsImpl().VertPos;
			await Assert.That(verts.Count).IsGreaterThan(3)
				.Because($"{label}: a solid needs four vertices before this proves anything");

			foreach (Vec3 vert in verts)
			{
				await Assert.That(ToolFitsAt(vert, normals, offsets, ball, slop)).IsTrue()
					.Because($"{label}: the ball does not fit inside the solid when centred on {vert.X},{vert.Y},{vert.Z}");
			}
		}

		/// <summary>
		/// The equivalence oracle Round All Edges cares about: an opening built on the
		/// closed form is the same rounded solid as an opening built on the general sweep.
		/// </summary>
		/// <remarks>
		/// Erode by r then dilate by r is what a uniform fillet is, so this is the whole
		/// user-visible claim of the fast path in one assertion. Both halves matter — the
		/// erosion is swapped, the dilation is not — and a closed form that was subtly the
		/// wrong size would show up here as a rounded solid of the wrong volume even if the
		/// erosion alone had passed on a coarse tolerance.
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		[Arguments("cube", 0)]
		[Arguments("tetrahedron", 1)]
		[Arguments("icosahedron", 2)]
		public async Task AnOpeningOnTheClosedFormMatchesAnOpeningOnTheSweep(string label, int shape)
		{
			Manifold solid = Polyhedron(shape);
			Manifold ball = Manifold.Sphere(1.0, 12);

			await Assert.That(solid.TryConvexErosion(ball, null, null, out Manifold fastEroded)).IsTrue();
			Manifold sweptEroded = solid.MinkowskiDifference(ball);

			await Assert.That(Relative(fastEroded.Volume(), sweptEroded.Volume())).IsLessThan(VolumeTolerance)
				.Because($"{label}: erosions differ — closed form {fastEroded.Volume()}, sweep {sweptEroded.Volume()}");

			Manifold fastOpening = fastEroded.MinkowskiSum(ball);
			Manifold sweptOpening = sweptEroded.MinkowskiSum(ball);

			await Assert.That(Relative(fastOpening.Volume(), sweptOpening.Volume())).IsLessThan(VolumeTolerance)
				.Because($"{label}: openings differ — closed form {fastOpening.Volume()}, sweep {sweptOpening.Volume()}");
			await Assert.That(Relative(fastOpening.SurfaceArea(), sweptOpening.SurfaceArea())).IsLessThan(VolumeTolerance)
				.Because($"{label}: the rounded surfaces have to agree, not just the volume they enclose");
		}

		/// <summary>
		/// An asymmetric tool erodes the way the sweep does — the test that the support is
		/// taken against <c>-n</c> and not <c>+n</c>.
		/// </summary>
		/// <remarks>
		/// Every other fixture in this file uses a centred ball, where <c>B = -B</c> and the
		/// reflection is invisible: flipping the sign in <c>TrySupportPlanes</c> would not have
		/// failed one of them, which made the header, the oracle and the divergence entry all
		/// assert something nothing measured. This tool reaches 1.0 in +x and only 0.3 in -x,
		/// and 0.8 in +z against 0.4 in -z, so the flipped sign gives the mirrored solid and is
		/// caught by a whole unit.
		/// <para>
		/// Asserted bit-exactly, not within a tolerance, and against the sweep rather than
		/// against a hand-computed number: the point is agreement, and on a box the two paths
		/// reach the same plane offsets by whole-number arithmetic, so there is nothing here
		/// for a tolerance to absorb.
		/// </para>
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task AnAsymmetricToolErodesTheWayTheSweepDoes()
		{
			Manifold tool = AsymmetricTool();
			Box toolBounds = tool.BoundingBox();
			await Assert.That(toolBounds.Min.X).IsEqualTo(-0.3);
			await Assert.That(toolBounds.Max.X).IsEqualTo(1.0);
			await Assert.That(toolBounds.Min.Z).IsEqualTo(-0.4);
			await Assert.That(toolBounds.Max.Z).IsEqualTo(0.8);

			Manifold box = Manifold.Cube(Vec3.Splat(40.0), true);
			await Assert.That(box.TryConvexErosion(tool, null, null, out Manifold eroded)).IsTrue();

			Box bounds = eroded.BoundingBox();

			// The erosion is { x : x - B subset A }, so the face at +x moves in by the tool's
			// reach in -x (0.3) and the face at -x by its reach in +x (1.0). Flip the sign and
			// these two swap, which is the mutation this test exists to catch.
			await Assert.That(bounds.Max.X).IsEqualTo(19.7)
				.Because("+x face moves in by the tool's -x reach, 0.3");
			await Assert.That(bounds.Min.X).IsEqualTo(-19.0)
				.Because("-x face moves in by the tool's +x reach, 1.0");
			await Assert.That(bounds.Max.Z).IsEqualTo(19.6)
				.Because("+z face moves in by the tool's -z reach, 0.4");
			await Assert.That(bounds.Min.Z).IsEqualTo(-19.2)
				.Because("-z face moves in by the tool's +z reach, 0.8");

			Box swept = box.MinkowskiDifference(tool).BoundingBox();
			await Assert.That(bounds.Min.X).IsEqualTo(swept.Min.X);
			await Assert.That(bounds.Max.X).IsEqualTo(swept.Max.X);
			await Assert.That(bounds.Min.Y).IsEqualTo(swept.Min.Y);
			await Assert.That(bounds.Max.Y).IsEqualTo(swept.Max.Y);
			await Assert.That(bounds.Min.Z).IsEqualTo(swept.Min.Z);
			await Assert.That(bounds.Max.Z).IsEqualTo(swept.Max.Z);
		}

		/// <summary>
		/// On a densely tessellated solid the two paths agree only to a tolerance, and this
		/// pins how big it is so it cannot drift unnoticed.
		/// </summary>
		/// <remarks>
		/// The cause is the dual hull, not the arithmetic. QuickHull discards points within
		/// its relative epsilon of an existing facet, and on a 2048-triangle sphere many dual
		/// points sit that close together — so a handful of halfspaces are dropped as if
		/// redundant when they are very slightly not. Measured at 4.9e-7 relative on volume
		/// and 4016 triangles against the sweep's 4024. The bound below is 5e-6, an order of
		/// magnitude of headroom, because the exact figure depends on where QuickHull's
		/// epsilon falls and pinning it tighter would make this a test of that epsilon.
		/// <para>
		/// A 1152-triangle sphere already differs in triangle count (2544 against 2550) while
		/// its volumes still agree to 4e-15, so the triangulation parts company first and the
		/// volume follows. Everything at or below a thousand faces — which is every other
		/// fixture here — agrees at 1e-15, and this is the only test in the file that needs a
		/// tolerance looser than 1e-6.
		/// </para>
		/// <para>
		/// Both numbers matter to the caller: 4.9e-7 of a rounding radius is far inside the
		/// error the tessellated ball itself introduces, so the fast path is still the right
		/// answer for a fillet — but it is not the sweep's answer, and the divergence entry
		/// and the file header say so because of this measurement.
		/// </para>
		/// </remarks>
		/// <returns>The test task.</returns>
		[Test]
		public async Task ADenseSolidAgreesWithTheSweepOnlyToATolerance()
		{
			Manifold sphere = Manifold.Sphere(10.0, 64);
			Manifold ball = Manifold.Sphere(1.0, 12);

			await Assert.That(sphere.NumTri()).IsEqualTo(2048)
				.Because("the measured figures below are for this tessellation");

			await Assert.That(sphere.TryConvexErosion(ball, null, null, out Manifold eroded)).IsTrue();
			Manifold swept = sphere.MinkowskiDifference(ball);

			double relative = Relative(eroded.Volume(), swept.Volume());
			await Assert.That(relative).IsLessThan(5e-6)
				.Because($"measured at 4.9e-7; a jump past 5e-6 means dual points are being dropped that should not be ({eroded.Volume()} against {swept.Volume()})");
			await Assert.That(relative).IsGreaterThan(1e-9)
				.Because("and if this ever became exact the tolerance above, the header and the divergence entry are all describing a problem that no longer exists");
		}

		/// <summary>
		/// The three convex polyhedra the correctness tests run over.
		/// </summary>
		/// <param name="shape">0 cube, 1 tetrahedron, 2 icosahedron.</param>
		/// <returns>The solid, sized so a unit ball fits well inside it.</returns>
		private static Manifold Polyhedron(int shape)
		{
			if (shape == 0)
			{
				return Manifold.Cube(Vec3.Splat(20.0), true);
			}

			if (shape == 1)
			{
				return Manifold.Tetrahedron().Scale(Vec3.Splat(8.0));
			}

			// The regular icosahedron's twelve vertices are the cyclic permutations of
			// (0, +-1, +-phi); hulling them is the shortest honest way to get one, since the
			// library has no icosahedron primitive.
			double phi = (1.0 + Math.Sqrt(5.0)) / 2.0;
			List<Vec3> points = new List<Vec3>(12);
			foreach (double a in new[] { -1.0, 1.0 })
			{
				foreach (double b in new[] { -phi, phi })
				{
					points.Add(new Vec3(0.0, a, b) * 6.0);
					points.Add(new Vec3(a, b, 0.0) * 6.0);
					points.Add(new Vec3(b, 0.0, a) * 6.0);
				}
			}

			return Manifold.FromImpl(QuickHullFunctions.ConvexHull(points));
		}

		/// <summary>
		/// A convex tool that reaches further one way than the other on every axis, and still
		/// contains the origin strictly.
		/// </summary>
		/// <remarks>
		/// A skewed octahedron: the hull of six points, one on each axis direction at a
		/// different distance. Both properties are load-bearing — the asymmetry is what makes
		/// the support's sign observable, and containing the origin is what keeps it past the
		/// gate so that it can be observed at all.
		/// </remarks>
		/// <returns>The tool, x in [-0.3, 1], y in [-0.5, 0.6], z in [-0.4, 0.8].</returns>
		private static Manifold AsymmetricTool()
		{
			List<Vec3> points = new List<Vec3>
			{
				new Vec3(1.0, 0.0, 0.0),
				new Vec3(-0.3, 0.0, 0.0),
				new Vec3(0.0, 0.6, 0.0),
				new Vec3(0.0, -0.5, 0.0),
				new Vec3(0.0, 0.0, 0.8),
				new Vec3(0.0, 0.0, -0.4),
			};

			return Manifold.FromImpl(QuickHullFunctions.ConvexHull(points));
		}

		/// <summary>
		/// The solid's outward unit face normals and their plane offsets.
		/// </summary>
		/// <param name="solid">The convex solid.</param>
		/// <returns>The normals and the matching offsets.</returns>
		private static (List<Vec3> Normals, List<double> Offsets) FacePlanes(Manifold solid)
		{
			ManifoldImpl impl = solid.AsImpl();
			List<Vec3> normals = new List<Vec3>();
			List<double> offsets = new List<double>();

			for (int tri = 0; tri < impl.NumTri(); tri++)
			{
				Vec3 normal = impl.FaceNormal[tri];
				double length = Length(normal);
				if (!(length > 0.0))
				{
					continue;
				}

				normal = normal / length;
				normals.Add(normal);
				offsets.Add(Dot(normal, impl.VertPos[impl.Halfedge[tri * 3].StartVert]));
			}

			return (normals, offsets);
		}

		/// <summary>
		/// The erosion, by brute force: every triple of offset planes is a candidate vertex,
		/// and a candidate survives when the tool, centred there, is still inside the solid.
		/// </summary>
		/// <remarks>
		/// Independent of the production construction on purpose — see the file header. It
		/// keeps candidates by the DEFINITION of erosion rather than by feasibility against
		/// the offset planes, so it is also a check on the offset formula itself and not only
		/// on the dual hull that consumes it. Cubic in the face count; a test only.
		/// </remarks>
		/// <param name="solid">The convex solid.</param>
		/// <param name="tool">The structuring element.</param>
		/// <returns>The eroded solid.</returns>
		private static Manifold HalfspaceIntersectionByTriples(Manifold solid, Manifold tool)
		{
			(List<Vec3> normals, List<double> offsets) = FacePlanes(solid);
			List<Vec3> toolVerts = tool.AsImpl().VertPos;
			double slop = Length(solid.BoundingBox().Size()) * 1e-9;

			// Every vertex of the erosion lies on three planes pushed in by the tool's
			// support, so this generates a superset of them; the filter does the rest.
			List<double> pushed = new List<double>(normals.Count);
			for (int i = 0; i < normals.Count; i++)
			{
				double support = double.NegativeInfinity;
				foreach (Vec3 b in toolVerts)
				{
					support = Math.Max(support, -Dot(normals[i], b));
				}

				pushed.Add(offsets[i] - support);
			}

			List<Vec3> survivors = new List<Vec3>();
			for (int a = 0; a < normals.Count; a++)
			{
				for (int b = a + 1; b < normals.Count; b++)
				{
					for (int c = b + 1; c < normals.Count; c++)
					{
						Vec3 cross = Cross(normals[b], normals[c]);
						double det = Dot(normals[a], cross);
						if (Math.Abs(det) < 1e-9)
						{
							continue;
						}

						Vec3 point = ((cross * pushed[a])
							+ (Cross(normals[c], normals[a]) * pushed[b])
							+ (Cross(normals[a], normals[b]) * pushed[c])) / det;

						if (ToolFitsAt(point, normals, offsets, tool, slop))
						{
							survivors.Add(point);
						}
					}
				}
			}

			return Manifold.FromImpl(QuickHullFunctions.ConvexHull(survivors));
		}

		/// <summary>
		/// Whether the tool, centred at <paramref name="centre"/>, lies inside the solid the
		/// planes describe — the definition of a point of the erosion.
		/// </summary>
		/// <remarks>
		/// The tool is placed as the sweep places it, <c>centre - b</c>: Minkowski.cs erodes
		/// by sweeping B rather than -B over the boundary, so what it computes is the set of
		/// centres where <c>x - B</c> fits. For the centred ball the distinction is invisible;
		/// it is written the matching way so this oracle cannot pass a fast path that got the
		/// reflection wrong.
		/// </remarks>
		/// <param name="centre">The candidate centre.</param>
		/// <param name="normals">The solid's outward unit face normals.</param>
		/// <param name="offsets">The matching plane offsets.</param>
		/// <param name="tool">The structuring element.</param>
		/// <param name="slop">How far outside a plane still counts as on it.</param>
		/// <returns>True when the tool fits.</returns>
		private static bool ToolFitsAt(Vec3 centre, List<Vec3> normals, List<double> offsets, Manifold tool, double slop)
		{
			foreach (Vec3 b in tool.AsImpl().VertPos)
			{
				Vec3 placed = centre - b;
				for (int i = 0; i < normals.Count; i++)
				{
					if (Dot(normals[i], placed) > offsets[i] + slop)
					{
						return false;
					}
				}
			}

			return true;
		}

		/// <summary>The relative difference between two positive quantities.</summary>
		/// <param name="actual">The measured value.</param>
		/// <param name="expected">The value it is measured against.</param>
		/// <returns>The relative difference.</returns>
		private static double Relative(double actual, double expected)
		{
			return Math.Abs(actual - expected) / Math.Max(Math.Abs(expected), 1e-30);
		}

		/// <summary>
		/// FNV-1a over the raw bits of every vertex coordinate and halfedge index, the same
		/// fingerprint <c>MinkowskiTests</c> uses to prove an operand was not touched.
		/// </summary>
		/// <param name="manifold">The mesh to fingerprint.</param>
		/// <returns>The fingerprint.</returns>
		private static long GeometryHash(Manifold manifold)
		{
			unchecked
			{
				ManifoldImpl mesh = manifold.AsImpl();
				ulong hash = 14695981039346656037UL;

				void Mix(ulong value)
				{
					for (int shift = 0; shift < 64; shift += 8)
					{
						hash ^= (value >> shift) & 0xFF;
						hash *= 1099511628211UL;
					}
				}

				foreach (Vec3 vert in mesh.VertPos)
				{
					Mix((ulong)BitConverter.DoubleToInt64Bits(vert.X));
					Mix((ulong)BitConverter.DoubleToInt64Bits(vert.Y));
					Mix((ulong)BitConverter.DoubleToInt64Bits(vert.Z));
				}

				foreach (Halfedge edge in mesh.Halfedge)
				{
					Mix((ulong)(uint)edge.StartVert);
					Mix((ulong)(uint)edge.EndVert);
					Mix((ulong)(uint)edge.PairedHalfedge);
				}

				return (long)hash;
			}
		}
	}
}
