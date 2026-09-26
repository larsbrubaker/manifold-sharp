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

// DedupeEdgesRegressionTests.cs — NOT A PORT; a C#-only adaptation test. It pins a
// defect found by the convex-dilation union tree: an exact-engine union of two
// NoError operands that did not contain one of them (B − (A ∪ B) = 2.3e-5).
//
// The boolean itself was right — with SimplifyTopology skipped the union's volume
// matched the robust engine's to 1e-15 — and the volume was lost in DedupeEdges, the
// first step of the post-boolean cleanup. It collects every duplicated (4-manifold)
// edge in one pass and then repairs them one by one; an earlier repair in the same
// pass can resolve or re-wire a later edge in the list, and repairing that stale
// entry copies the position of the wrong vertex into the new vertex it relabels a
// whole orbit to, so triangle corners jump ~0.13 and a sliver of solid vanishes.
// The input satisfies every halfedge invariant (endVert = next.startVert, pairs
// mutual) and the damage is done by DedupeEdges alone.
//
// The fixture is the 852 triangles within three vertex rings of the sixteen duplicate
// edges, cut from the 28060-triangle union right before DedupeEdges (the boolean of
// the level-7 nodes 0 and 1 of DrilledPart(400) ⊕ Sphere(0.3, 16)). Halfedges are
// stored as start, end and pair (−1 where the pair fell outside the cut); positions
// round-trip through "R" formatting, so they are bit-exact. Exporting the two boolean
// operands to MeshGL does NOT reproduce it — the re-import welds and re-orders enough
// that the duplicates come out differently — which is why the fixture is this
// intermediate state rather than the operands.

using System.Globalization;

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class DedupeEdgesRegressionTests
	{
		/// <summary>
		/// Splitting duplicated edges only relabels vertices and adds zero-area triangles,
		/// so no pre-existing triangle corner may change position.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		// Fixed the way manifold-rust 4a99dc4 fixes it (its CPP_DIVERGENCES entry 3); the
		// Rust carries the same fixture and test (edge_op_tests.rs).
		public async Task DedupeEdgesNeverMovesATriangleCorner()
		{
			ManifoldImpl mesh = LoadFixture("dedupe-stale-duplicate.txt");
			int halfedges = mesh.Halfedge.Count;
			Vec3[] before = new Vec3[halfedges];
			for (int h = 0; h < halfedges; h++)
			{
				before[h] = mesh.VertPos[mesh.Halfedge[h].StartVert];
			}

			EdgeOp.DedupeEdges(mesh);

			int moved = 0;
			for (int h = 0; h < halfedges; h++)
			{
				Vec3 after = mesh.VertPos[mesh.Halfedge[h].StartVert];
				if (after.X != before[h].X || after.Y != before[h].Y || after.Z != before[h].Z)
				{
					moved++;
				}
			}

			await Assert.That(moved).IsEqualTo(0)
				.Because("a corner that moves changes the solid; this one lost 2.3e-5 of volume in a union");

			// And skipping stale entries must not leave a real duplicate behind: every directed
			// edge is unique once the passes are done.
			HashSet<(int Start, int End)> edges = new HashSet<(int Start, int End)>();
			int duplicates = 0;
			foreach (Halfedge halfedge in mesh.Halfedge)
			{
				if (halfedge.StartVert >= 0 && !edges.Add((halfedge.StartVert, halfedge.EndVert)))
				{
					duplicates++;
				}
			}

			await Assert.That(duplicates).IsEqualTo(0);
		}

		private static ManifoldImpl LoadFixture(string fileName)
		{
			string path = Path.Combine(AppContext.BaseDirectory, "TestData", "regressions", fileName);
			string[] lines = File.ReadAllLines(path);
			string[] header = lines[0].Split(' ');
			int numVert = int.Parse(header[0], CultureInfo.InvariantCulture);
			int numTri = int.Parse(header[1], CultureInfo.InvariantCulture);

			ManifoldImpl mesh = new ManifoldImpl();
			for (int v = 0; v < numVert; v++)
			{
				string[] p = lines[1 + v].Split(' ');
				mesh.VertPos.Add(new Vec3(
					double.Parse(p[0], CultureInfo.InvariantCulture),
					double.Parse(p[1], CultureInfo.InvariantCulture),
					double.Parse(p[2], CultureInfo.InvariantCulture)));
			}

			for (int t = 0; t < numTri; t++)
			{
				string[] p = lines[1 + numVert + t].Split(' ');
				for (int k = 0; k < 3; k++)
				{
					int start = int.Parse(p[3 * k], CultureInfo.InvariantCulture);
					int end = int.Parse(p[(3 * k) + 1], CultureInfo.InvariantCulture);
					int pair = int.Parse(p[(3 * k) + 2], CultureInfo.InvariantCulture);
					mesh.Halfedge.Add(new Halfedge(start, end, pair, start));
				}
			}

			return mesh;
		}
	}
}
