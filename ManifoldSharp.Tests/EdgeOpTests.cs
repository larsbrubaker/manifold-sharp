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

// Port of the tests module in edge_op.rs — same 3 cases, same inputs, same
// assertions, in the same order.
//
// Three tests is thin cover for 1,144 lines of Rust, and that is the Rust's
// state too: edge_op's real coverage lives downstream, in the suites that drive
// SimplifyTopology / RemoveDegenerates / CleanupTopology on meshes that
// actually have degenerates. Those are DEFERRED here with their modules,
// because the callers do not exist yet. Every deferred caller, so that a later
// phase can grep this list and know which of its tests it is inheriting
// edge_op's coverage from:
//
//   Phase 5   boolean3.rs:316               remove_degenerates
//             boolean_result_assemble.rs:8  simplify_topology
//   Phase 6   manifold.rs:169, :199         simplify_topology — the
//                                           tolerance-raising path and the
//                                           unconditional simplify after it
//             manifold_meshgl.rs:349-353    cleanup_topology,
//                                           collapse_short_edges,
//                                           remove_degenerates
//   Phase 7   sdf.rs:907                    cleanup_topology, on the freshly
//                                           marched voxel surface
//   Phase 10  robust/assemble.rs:218-220    cleanup_topology,
//                                           collapse_short_edges,
//                                           collapse_colinear_edges
//
// Until those land, the parity of CollapseEdge, RecursiveEdgeSwap, DedupeEdge
// and the rest is held by the differential harness this step ran against the
// compiled Rust (see the step report), not by a checked-in test.

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
	}
}
