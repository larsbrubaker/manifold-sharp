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

// The per-site half of ParallelismTests: one bit-identity test per
// determinism-preserving site. Continues ParallelismTests.cs, which carries the
// file header, the site-to-test table, the anti-vacuity rule, the reason
// mesh-ID labels are excluded from the comparison, and every helper these
// tests call — read that first.
//
// Split off purely for the 800-line cap; the two files are one class and one
// subject.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class ParallelismTests
	{
		// ---------------------------------------------------------------------
		// The six sites
		// ---------------------------------------------------------------------

		/// <summary>
		/// Intersect12, Winding03 and Face2Tri at once: all three live inside one boolean,
		/// and a boolean is the only way the last of them ever sees a face with more than
		/// four edges.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		[NotInParallel(ParallelismGlobalStateKey)]
		public async Task BooleanPipelineIsBitIdenticalInParallel()
		{
			Manifold a = Manifold.Sphere(1.0, 96);
			Manifold b = Manifold.Sphere(1.0, 96).Translate(new Vec3(0.5, 0.0, 0.0));

			// Anti-vacuity, directly asserted: Intersect12 maps over a.Halfedge.Count with
			// threshold 10_000, and Winding03 maps over one query box per union-find
			// representative — at most one per vertex — with threshold 1_000.
			await Assert.That(a.AsImpl().Halfedge.Count)
				.IsGreaterThanOrEqualTo(10_000)
				.Because("Intersect12's threshold is 10_000 halfedges");
			await Assert.That(a.NumVert())
				.IsGreaterThanOrEqualTo(1_000)
				.Because("Winding03's threshold is 1_000 query boxes");

			(MeshGL64 sequential, MeshGL64 parallel) = BothWays(() => a.Union(b).GetMeshGL64(-1));

			await Assert.That(sequential.NumTri())
				.IsGreaterThan(0)
				.Because("the union must leave geometry to compare");

			// Face2Tri's threshold is 512 *faces*, and the face count is internal to
			// boolean assembly — not observable from here. The bound is indirect: assembly
			// emits one face per output polygon and nearly every one triangulates to a
			// single triangle, so an output this size cannot have come from under 512
			// faces. Face2TriIsBitIdenticalOverItsOwnThreshold pins the threshold crossing
			// where it *can* be asserted.
			await Assert.That(sequential.NumTri()).IsGreaterThan(512);

			await AssertSameGeometry("sphere union", sequential, parallel, compareRunLabels: true);
		}

		/// <summary>
		/// Face2Tri over a face count that is asserted rather than inferred, by calling it
		/// directly with one face per triangle.
		/// </summary>
		/// <remarks>
		/// The identity retriangulation: every face has exactly three edges, so the map
		/// returns null for each and the general triangulator never runs. That is the
		/// point — it isolates the parallel map and the face-ordered merge that consumes
		/// it, with a face count (= triangle count) the test can state outright.
		/// BooleanPipelineIsBitIdenticalInParallel covers the general-face path with real
		/// data.
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		[NotInParallel(ParallelismGlobalStateKey)]
		public async Task Face2TriIsBitIdenticalOverItsOwnThreshold()
		{
			// Built once and cloned per leg: a second Manifold.Sphere would reserve a
			// fresh mesh ID and the TriRefs would differ for a reason that has nothing to
			// do with parallelism.
			ManifoldImpl source = Manifold.Sphere(1.0, 48).IntoImpl();
			int numFaces = source.NumTri();
			await Assert.That(numFaces)
				.IsGreaterThanOrEqualTo(512)
				.Because("Face2Tri's threshold is 512 faces");

			int[] faceEdge = new int[numFaces + 1];
			for (int face = 0; face <= numFaces; face++)
			{
				faceEdge[face] = 3 * face;
			}

			TriRef[] halfedgeRef = new TriRef[3 * numFaces];
			for (int halfedge = 0; halfedge < halfedgeRef.Length; halfedge++)
			{
				halfedgeRef[halfedge] = source.MeshRelation.TriRef[halfedge / 3];
			}

			// The completion flag is carried out and asserted rather than checked with
			// Debug.Assert, which compiles out in Release — the configuration CI runs, and
			// the one where a silently-false return would leave both legs holding the same
			// half-built mesh and the comparison passing on it.
			((bool Ok, ManifoldImpl Mesh) Sequential, (bool Ok, ManifoldImpl Mesh) Parallel) runs =
				BothWays(() =>
				{
					ManifoldImpl mesh = source.Clone();
					bool completed = FaceOp.Face2TriCt(mesh, faceEdge, halfedgeRef, false, null);
					return (completed, mesh);
				});

			await Assert.That(runs.Sequential.Ok)
				.IsTrue()
				.Because("an uncancelled Face2Tri must complete");
			await Assert.That(runs.Parallel.Ok)
				.IsTrue()
				.Because("an uncancelled Face2Tri must complete");

			ManifoldImpl sequential = runs.Sequential.Mesh;
			ManifoldImpl parallel = runs.Parallel.Mesh;

			await AssertSameTopology("face2tri", sequential, parallel);
			await AssertSameTriRefs("face2tri", sequential, parallel);
			await AssertSameGeometry(
				"face2tri",
				Manifold.FromImpl(sequential).GetMeshGL64(-1),
				Manifold.FromImpl(parallel).GetMeshGL64(-1),
				compareRunLabels: true);
		}

		/// <summary>The SDF voxel fill, the one site whose map calls user code.</summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		[NotInParallel(ParallelismGlobalStateKey)]
		public async Task SdfVoxelFillIsBitIdenticalInParallel()
		{
			// The voxel map's threshold is 10_000 *encoded* indices, and the encoding gives
			// each axis ceil(log2(cells + 3)) bits — so 20 cells per axis costs 5 bits per
			// axis and the index space is at least 32^3 = 32_768, comfortably over. The
			// exact count is internal to LevelSet; this bound on it is not.
			Box bounds = new Box(new Vec3(-1.2, -1.2, -1.2), new Vec3(1.2, 1.2, 1.2));
			const double EdgeLength = 0.1;
			int cellsPerAxis = (int)((bounds.Max.X - bounds.Min.X) / EdgeLength);
			await Assert.That(cellsPerAxis)
				.IsGreaterThanOrEqualTo(20)
				.Because("the voxel fill's threshold is 10_000 encoded indices");

			// A pure, thread-safe sdf — the obligation Sdf.LevelSet's remarks state, which
			// C# cannot express as a type bound the way Rust's `+ Sync` does.
			static double UnitSphere(Vec3 p)
			{
				return 1.0 - Math.Sqrt((p.X * p.X) + (p.Y * p.Y) + (p.Z * p.Z));
			}

			(ManifoldImpl sequential, ManifoldImpl parallel) = BothWays(
				() => Sdf.LevelSetSimple(UnitSphere, bounds, EdgeLength));

			await Assert.That(sequential.NumTri())
				.IsGreaterThan(0)
				.Because("the level set must extract a surface to compare");
			await AssertSameTopology("levelset", sequential, parallel);
			await AssertSameGeometry(
				"levelset",
				Manifold.FromImpl(sequential).GetMeshGL64(-1),
				Manifold.FromImpl(parallel).GetMeshGL64(-1),
				compareRunLabels: false);
		}

		/// <summary>
		/// CalculateVertNormals, whose per-vertex ForVert walk fixes a float accumulation
		/// order that the Boolean3 SOS tie-breaks then depend on.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		[NotInParallel(ParallelismGlobalStateKey)]
		public async Task VertNormalsAreBitIdenticalInParallel()
		{
			ManifoldImpl source = Manifold.Sphere(1.0, 256).IntoImpl();
			await Assert.That(source.NumVert())
				.IsGreaterThanOrEqualTo(10_000)
				.Because("CalculateVertNormals' threshold is 10_000 vertices");

			(ManifoldImpl sequential, ManifoldImpl parallel) = BothWays(() =>
			{
				ManifoldImpl mesh = source.Clone();
				FaceOp.CalculateVertNormals(mesh);
				return mesh;
			});

			await Assert.That(parallel.VertNormal.Count).IsEqualTo(sequential.VertNormal.Count);
			for (int vert = 0; vert < sequential.VertNormal.Count; vert++)
			{
				Vec3 s = sequential.VertNormal[vert];
				Vec3 p = parallel.VertNormal[vert];
				await AssertSameBits($"normal[{vert}].x", s.X, p.X);
				await AssertSameBits($"normal[{vert}].y", s.Y, p.Y);
				await AssertSameBits($"normal[{vert}].z", s.Z, p.Z);
			}
		}

		/// <summary>
		/// The Minkowski hulls — the one site carrying a documented determinism exception
		/// (Minkowski.cs's header): geometry and topology are bit-identical, but mesh-ID
		/// <em>values</em> may be consumed in a different order.
		/// </summary>
		/// <remarks>
		/// The test measures that exception rather than merely tolerating it. It runs the
		/// same sum three times — sequential, sequential, parallel — and shows the ID
		/// labels already differ between the two <em>sequential</em> runs, because each
		/// hull mints fresh IDs from the process-global counter. Parallelism therefore
		/// changes nothing an unchanged caller could observe: it reorders values that were
		/// never stable in the first place.
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		[NotInParallel(ParallelismGlobalStateKey)]
		public async Task MinkowskiGeometryIsBitIdenticalInParallel()
		{
			Manifold nonConvex = Manifold.Cube(Vec3.Splat(2.0), true)
				.Difference(Manifold.Sphere(1.2, 8));
			Manifold ball = Manifold.Sphere(0.1, 8);

			// The convex+non-convex branch maps over min(numTri, 1000) hulls per batch
			// with threshold 8.
			await Assert.That(nonConvex.NumTri())
				.IsGreaterThanOrEqualTo(8)
				.Because("the Minkowski hull map's threshold is 8 faces");

			MeshGL64 firstSequential = RunWith(false, () => nonConvex.MinkowskiSum(ball).GetMeshGL64(-1));
			MeshGL64 secondSequential = RunWith(false, () => nonConvex.MinkowskiSum(ball).GetMeshGL64(-1));
			MeshGL64 parallel = RunWith(true, () => nonConvex.MinkowskiSum(ball).GetMeshGL64(-1));

			await Assert.That(firstSequential.NumTri())
				.IsGreaterThan(0)
				.Because("the Minkowski sum must leave geometry to compare");

			// Geometry and topology: no exception, no slack.
			await AssertSameGeometry(
				"minkowski sequential twice",
				firstSequential,
				secondSequential,
				compareRunLabels: false);
			await AssertSameGeometry(
				"minkowski sequential vs parallel",
				firstSequential,
				parallel,
				compareRunLabels: false);

			// The exception, bounded. "Reordered" is the whole permission the header grants,
			// and reordering is a BIJECTION: the same number of runs, each still carrying a
			// distinct label. A parallel run that handed two hulls the same id would be a
			// real bug — two provenance groups collapsed into one — and it would slip past
			// a count-only check, because the count would still match. Comparing the
			// distinct count is what closes that.
			await Assert.That(secondSequential.RunOriginalId.Count)
				.IsEqualTo(firstSequential.RunOriginalId.Count);
			await Assert.That(parallel.RunOriginalId.Count)
				.IsEqualTo(firstSequential.RunOriginalId.Count);
			await Assert.That(parallel.RunOriginalId.Distinct().Count())
				.IsEqualTo(firstSequential.RunOriginalId.Distinct().Count())
				.Because("a parallel run must relabel the hulls, never merge two of them");
			await Assert.That(secondSequential.RunOriginalId.Distinct().Count())
				.IsEqualTo(firstSequential.RunOriginalId.Distinct().Count())
				.Because("the sequential baseline for the same property");

			// Brittleness note, deliberate: this asserts the labels DO move, which is a
			// statement about the global counter's monotonicity and not about parallelism.
			// It would start failing if mesh IDs ever became content-derived or per-call
			// reset — and that is the point of writing it down: on that day the right fix
			// is to re-derive what the exception permits, not to delete the line. Nothing
			// above it depends on this holding.
			await Assert.That(secondSequential.RunOriginalId.SequenceEqual(firstSequential.RunOriginalId))
				.IsFalse()
				.Because(
					"mesh IDs come from a process-global counter, so even two sequential "
					+ "Minkowski runs relabel — which is what makes the parallel relabel a "
					+ "non-event");
		}

		/// <summary>
		/// The robust engine's five per-triangle maps, all at once — narrow phase,
		/// candidate points, the two registries and the arrangements.
		/// </summary>
		/// <remarks>
		/// <para>
		/// They are not among the six sites blessed by name, and they are parallel anyway
		/// for the same reason they are in the Rust: they reach <see cref="Par"/> through
		/// <see cref="Progress.MaybeParMapCtProgress"/>, which the <c>parallel</c> feature
		/// covers without naming them. One robust boolean drives all five, so one test
		/// covers all five — and it is the only test here whose failure would implicate a
		/// file this step did not write.
		/// </para>
		/// <para>
		/// The engine is named explicitly rather than reached through <c>Auto</c> on
		/// self-intersecting operands: that would make the test depend on the soup
		/// checklist's verdict as well as on parallelism, and would need
		/// <see cref="RobustEngineTests.BooleanConfigGlobalStateKey"/> on top of this
		/// file's key. <c>Robust</c> by name is the same pipeline with one fewer thing that
		/// can go wrong for an unrelated reason.
		/// </para>
		/// </remarks>
		/// <returns>A task representing the test.</returns>
		[Test]
		[NotInParallel(ParallelismGlobalStateKey)]
		public async Task RobustEnginePipelineIsBitIdenticalInParallel()
		{
			Manifold a = Manifold.Sphere(1.0, 32);
			Manifold b = Manifold.Sphere(1.0, 32).Translate(new Vec3(0.5, 0.0, 0.0));

			// Anti-vacuity, as far as it can be asserted: the narrow phase maps over the
			// combined triangle soup with threshold 64. The other four map over candidate,
			// registry and arrangement work lists with threshold 16 — internal sizes, but
			// two deeply overlapping 512-triangle spheres cannot produce fewer than 16
			// arrangement faces and still yield the 700+ triangle result asserted below.
			await Assert.That(a.NumTri() + b.NumTri())
				.IsGreaterThanOrEqualTo(64)
				.Because("the narrow-phase map's threshold is 64 triangles");

			MeshGL64 firstSequential = RunWith(
				false,
				() => a.UnionWithEngine(b, BooleanEngine.Robust).GetMeshGL64(-1));
			MeshGL64 secondSequential = RunWith(
				false,
				() => a.UnionWithEngine(b, BooleanEngine.Robust).GetMeshGL64(-1));
			MeshGL64 parallel = RunWith(
				true,
				() => a.UnionWithEngine(b, BooleanEngine.Robust).GetMeshGL64(-1));

			await Assert.That(firstSequential.NumTri())
				.IsGreaterThan(512)
				.Because("the robust union must leave geometry to compare");

			await AssertSameGeometry(
				"robust union sequential vs parallel",
				firstSequential,
				parallel,
				compareRunLabels: false);

			// Same shape as the Minkowski test, and for the same reason: robust assembly
			// mints one fresh original ID per result from the process-global counter, so
			// the label moves between two SEQUENTIAL runs too. Measuring that here is what
			// justifies excluding it above, instead of excluding it on trust.
			await Assert.That(secondSequential.RunOriginalId.SequenceEqual(firstSequential.RunOriginalId))
				.IsFalse()
				.Because(
					"robust assembly relabels on every run, sequential included, so the "
					+ "label is not evidence about parallelism either way");
			await AssertSameGeometry(
				"robust union sequential twice",
				firstSequential,
				secondSequential,
				compareRunLabels: false);
		}
	}
}
