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

// Port of the tests module in edge_op.rs (edge_op_tests.rs) — same cases, same
// inputs, same assertions, in the same order, with one exception:
// test_dedupe_edges_never_moves_a_triangle_corner originated in this port and lives
// in DedupeEdgesRegressionTests.cs, which the Rust copied with its fixture. Its
// end-to-end companion, test_union_keeps_concave_corner (manifold-rust 9989b92), is
// ported here as UnionKeepsConcaveCorner, its fixtures copied verbatim into
// TestData/regressions.
//
// The three unit tests are thin cover for 1,144 lines of Rust, and that is the Rust's
// state too: edge_op's real coverage lives downstream, in the suites that drive
// SimplifyTopology / RemoveDegenerates / CleanupTopology on meshes that
// actually have degenerates. Every such caller is listed here so a later phase
// can grep the list and know which of its tests it is inheriting edge_op's
// coverage from. ALL OF THEM HAVE NOW LANDED — nothing in this table is
// deferred any more:
//
//   Phase 5   boolean3.rs:316               remove_degenerates — LANDED, the
//                                           caller is Boolean3.Functions.cs:157.
//             boolean_result_assemble.rs:8  simplify_topology — LANDED, the
//                                           caller is
//                                           BooleanResultAssemble.cs:480. Every
//                                           exact boolean runs both, so
//                                           Boolean3Tests, ManifoldBooleanTests
//                                           and ManifoldValidationTests carry
//                                           this share.
//   Phase 6   manifold.rs:169, :199         simplify_topology — the
//                                           tolerance-raising path and the
//                                           unconditional simplify after it.
//                                           LANDED, the callers are
//                                           Manifold.cs:367 and :414.
//             manifold_meshgl.rs:349-353    cleanup_topology,
//                                           collapse_short_edges,
//                                           remove_degenerates — LANDED, the
//                                           callers are
//                                           Manifold.MeshGL.cs:679-685, so every
//                                           MeshGL import in MeshGLAdaptationTests
//                                           and ManifoldMeshOpsTests exercises them.
//   Phase 7   sdf.rs:907                    cleanup_topology, on the freshly
//                                           marched voxel surface — LANDED, the
//                                           caller is Sdf.cs:483. SdfTests and
//                                           the sdf differential harness (15
//                                           level-set cases, byte-identical to
//                                           the Rust, genus 1 and genus 27
//                                           included) now carry this share of
//                                           edge_op's coverage.
//   Phase 10  robust/assemble.rs:218-220    cleanup_topology,
//                                           collapse_short_edges,
//                                           collapse_colinear_edges — LANDED,
//                                           the callers are Assemble.cs:252-254.
//                                           Every robust boolean runs all three
//                                           over its own extraction, so
//                                           RobustEngineTests, RobustRebuildTests
//                                           and RobustThingiTests (13 real scans)
//                                           now carry this share of edge_op's
//                                           coverage, with the Phase 10 finale
//                                           harness pinning it bit-for-bit.
//
// So the parity of CollapseEdge, RecursiveEdgeSwap, DedupeEdge and the rest is
// no longer held only by the differential harness edge_op's own step ran against
// the compiled Rust — every caller above is now a checked-in test driving them on
// meshes that really do carry degenerates.

using System.Globalization;

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class EdgeOpTests
	{
		[Test]
		public async Task PairUp()
		{
			List<Halfedge> halfedges = new List<Halfedge>
			{
				new Halfedge(0, 1, -1, 0),
				new Halfedge(1, 0, -1, 1),
			};
			EdgeOp.PairUp(halfedges, 0, 1);
			await Assert.That(halfedges[0].PairedHalfedge).IsEqualTo(1);
			await Assert.That(halfedges[1].PairedHalfedge).IsEqualTo(0);
		}

		[Test]
		public async Task CleanupTopologyNoopOnCleanMesh()
		{
			ManifoldImpl m = ManifoldImpl.Tetrahedron(Mat3x4.Identity());
			FaceOp.SetNormalsAndCoplanar(m);
			int beforeVerts = m.VertPos.Count;
			int beforeHalfedges = m.Halfedge.Count;
			EdgeOp.CleanupTopology(m);

			// Tetrahedron is already 2-manifold; cleanup should not add verts/halfedges
			await Assert.That(m.VertPos.Count).IsEqualTo(beforeVerts);
			await Assert.That(m.Halfedge.Count).IsEqualTo(beforeHalfedges);
		}

		[Test]
		public async Task SimplifyTopologyNoopOnCleanMesh()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			FaceOp.SetNormalsAndCoplanar(m);
			EdgeOp.SimplifyTopology(m, 0);

			// After simplify, cube should still be 2-manifold (no degenerate edges)
			// (Some verts/edges may be removed, but topology must be valid)
			// Just check it's still 2-manifold where halfedges are valid
			bool valid = m.Halfedge
				.Where(h => h.PairedHalfedge >= 0)
				.All(h => h.PairedHalfedge < m.Halfedge.Count);
			await Assert.That(valid).IsTrue().Because("invalid paired halfedge after simplify");
		}

		/// <summary>
		/// The union's volume must match inclusion-exclusion. Before the stale-entry
		/// check in DedupeEdges (manifold-rust CPP_DIVERGENCES entry 3), the cleanup
		/// filled the concave corner, as it does in C++ Manifold 3.5.2.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task UnionKeepsConcaveCorner()
		{
			Manifold a = LoadTriFixture("union-concave-corner-a.txt");
			Manifold b = LoadTriFixture("union-concave-corner-b.txt");
			double expected = a.Volume() + b.Volume() - a.Boolean(b, OpType.Intersect).Volume();
			foreach ((Manifold x, Manifold y) in new[] { (a, b), (b, a) })
			{
				Manifold union = x.Boolean(y, OpType.Add);
				await Assert.That(Math.Abs(union.Volume() - expected) < 1e-9 * expected).IsTrue()
					.Because($"union volume {union.Volume()}, expected {expected}");
			}
		}

		/// <summary>
		/// Two operands cut from a union in a BOSL2 <c>cubetruss</c> model, around a
		/// concave corner. Format: "3 numVert numTri tolerance 0", then one "x y z" per
		/// vertex, then one "v0 v1 v2" per triangle. Positions are round-trip formatted,
		/// so bit-exact. Neither mesh carries face IDs. The Rust's load_tri_fixture.
		/// </summary>
		/// <param name="fileName">The fixture's file name under TestData/regressions.</param>
		/// <returns>The imported manifold.</returns>
		private static Manifold LoadTriFixture(string fileName)
		{
			string path = Path.Combine(AppContext.BaseDirectory, "TestData", "regressions", fileName);
			string[] lines = File.ReadAllLines(path);
			string[] header = lines[0].Split(' ', StringSplitOptions.RemoveEmptyEntries);
			int numVert = int.Parse(header[1], CultureInfo.InvariantCulture);
			int numTri = int.Parse(header[2], CultureInfo.InvariantCulture);
			MeshGL64 mesh = new MeshGL64();
			mesh.NumProp = 3;
			mesh.Tolerance = double.Parse(header[3], CultureInfo.InvariantCulture);
			for (int v = 0; v < numVert; v++)
			{
				foreach (string x in lines[1 + v].Split(' ', StringSplitOptions.RemoveEmptyEntries))
				{
					mesh.VertProperties.Add(double.Parse(x, CultureInfo.InvariantCulture));
				}
			}

			for (int t = 0; t < numTri; t++)
			{
				foreach (string x in lines[1 + numVert + t].Split(' ', StringSplitOptions.RemoveEmptyEntries))
				{
					mesh.TriVerts.Add(ulong.Parse(x, CultureInfo.InvariantCulture));
				}
			}

			return Manifold.FromMeshGL64(mesh);
		}
	}
}
