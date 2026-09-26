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

// ConvexDilation.cs — NOT A PORT. manifold-rust dilates a non-convex solid by a convex
// tool in Minkowski.cs's middle branch: one hull per triangle, then CsgTree.BatchUnion
// 1000 hulls at a time. This file answers the same question with the same hulls but a
// different reduction, and it is divergence ledger entry 6 (docs/RUST_DIVERGENCES.md)
// — an ADDED entry point, reached only when a caller asks for it by name.
// `Minkowski.Compute` is not routed through it and does not know it exists, so every
// ported path still produces the Rust's bits.
//
// ── Why a different reduction ───────────────────────────────────────────────────
// BatchBoolean pops the largest-vertex mesh first, so within a batch the growing union
// is always one operand: the batch is a serial chain absorbing one group after
// another, and ~85% of a dilation's time is spent there with every core but one idle.
// The hulls are the cheap part. Measured on a 2642-triangle part: 13.3 s through the
// ported path against 2.96 s here, identical volume. A single-threaded balanced tree
// was NOT faster — the gain is the parallelism a balanced tree exposes, not the shape.
//
// ── The tree ────────────────────────────────────────────────────────────────────
//   1. Leaves: the solid on its own, then runs of LeafSize triangles in face order,
//      each leaf building its triangles' hulls exactly as Minkowski.cs does (same
//      vertex-sum order) and unioning them through the CSG tree (the same BatchUnion
//      the ported path uses, on a batch small enough that its serial chain is short).
//      Hulls live only inside their leaf, so memory is bounded by the leaves in
//      flight, not by the triangle count.
//   2. Pairwise levels: node k of a level is the union of nodes 2k and 2k+1 of the
//      level below; an odd last node is carried up unchanged. Each level is one map.
//
// ── Determinism: the proof every parallel site carries ──────────────────────────
// The leaf map and every level map go through Progress.MaybeParMapCtProgress →
// Par.MaybeParMapCt, so
// the MANIFOLD_PARALLEL switch (ManifoldParallel.Enabled) governs them as it governs
// the other sites, and with it off this file runs sequentially. With it on, each
// worker writes result[i] for its own i and reads only the two input meshes (leaves)
// or the previous level's array, which is complete and never written again once its
// map has returned. The boolean engine is read once, before the tree, and every union
// in it - leaf and level alike - runs on that engine. The tree's
// shape is a function of the triangle count alone, and every node's operands are
// fixed by index, so the booleans a parallel run performs are the booleans a
// sequential run performs, on the same inputs in the same operand order — the answer
// is the same bits whatever the threads do.
//
// The one exception is the one Minkowski.cs's header already documents for its hull
// map: each hull and each boolean mints mesh IDs from the process-global counter, and
// in parallel they are taken in worker order rather than index order. Those IDs are
// opaque handles that reach no geometry and no topology, and the closing
// InitializeOriginal replaces them with one fresh ID anyway, exactly as Minkowski.cs
// finishes. ParallelismTests.ConvexDilationGeometryIsBitIdenticalInParallel measures it.
//
// ── What it is not ──────────────────────────────────────────────────────────────
// Not bit-identical to Minkowski.Sum. Union is associative on the solid, not on the
// mesh: reducing the same hulls in a different order rounds intersection vertices
// differently and triangulates differently. Volume and genus agree to roundoff —
// ConvexDilationTests pins it. (An earlier measurement showed a 2.6e-7 relative
// drift on an 11916-triangle part; that was a DedupeEdges defect, shared with the
// C++ and fixed in both ports (EdgeOp.Dedupe.cs, manifold-rust 4a99dc4), dropping
// solid out of one union node — not the reduction order.)
// Triangle lists do not agree, which is why this is a separate entry point rather
// than a reroute.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <summary>
	/// The dilation of a non-convex solid by a convex tool, reduced through a balanced,
	/// parallel union tree instead of <see cref="Minkowski.Sum"/>'s serial batches.
	/// </summary>
	/// <remarks>
	/// A fast path and nothing more: <see cref="TryCompute"/> answers false for every
	/// input outside the non-convex ⊕ convex case or with a shell nested in another, and the caller then runs
	/// <see cref="Minkowski.Sum"/>. See the file header for the tree and its determinism
	/// argument.
	/// </remarks>
	public static class ConvexDilation
	{
		/// <summary>
		/// Hulls per leaf. Small enough that a leaf's BatchUnion chain is short, large
		/// enough that the leaves outnumber the cores on any part worth speeding up.
		/// </summary>
		private const int LeafSize = 16;

		/// <summary>
		/// The leaf and tree-level maps' parallel threshold. Each element is a whole
		/// boolean, so two of them already pay for the dispatch.
		/// </summary>
		private const int UnionParThreshold = 2;

		/// <summary>
		/// The Minkowski sum of a non-convex <paramref name="solid"/> and a convex
		/// <paramref name="tool"/>, as a union of the solid and one hull per triangle.
		/// </summary>
		/// <param name="solid">The shape being grown. Must be non-convex.</param>
		/// <param name="tool">The structuring element. Must be convex.</param>
		/// <param name="token">
		/// The cancellation token, or null. A cancelled run answers <c>true</c> with an
		/// empty result carrying <see cref="Error.Cancelled"/>, so a cancelled caller does
		/// not go on to run the general path.
		/// </param>
		/// <param name="progress">
		/// The progress reporter, or null. One <see cref="Phase.Minkowski"/> unit per hull,
		/// per leaf and per tree node, plus one for the closing normals pass that
		/// <see cref="Progress.CompletePhase"/> spends, so a finished run lands on 1.0.
		/// Declines happen before the phase opens and report nothing. Under the parallel
		/// switch workers advance the shared counter concurrently, so two reports can cross
		/// (see <see cref="ProgressReporter.Advance"/>).
		/// </param>
		/// <param name="result">The dilated solid when this returns true; empty otherwise.</param>
		/// <returns>True when this path applied; false when the caller must run
		/// <see cref="Minkowski.Sum"/>.</returns>
		public static bool TryCompute(
			ManifoldImpl solid,
			ManifoldImpl tool,
			CancelToken? token,
			ProgressReporter? progress,
			out ManifoldImpl result)
		{
			ArgumentNullException.ThrowIfNull(solid);
			ArgumentNullException.ThrowIfNull(tool);

			result = new ManifoldImpl();

			if (Cancel.IsCancelled(token))
			{
				result = Boolean3Functions.CancelledImpl();
				return true;
			}

			// Empty, soup and errored operands are the general path's business; it has a
			// considered answer for each (a clone, a propagated status).
			if (solid.IsEmpty()
				|| tool.IsEmpty()
				|| solid.IsSoup
				|| tool.IsSoup
				|| solid.Status != Error.NoError
				|| tool.Status != Error.NoError)
			{
				return false;
			}

			// Exactly Minkowski.Compute's middle branch without the operand swap: convex ⊕
			// convex is one hull there and needs no help, and non-convex ⊕ non-convex is a
			// different algorithm.
			if (solid.IsConvex() || !tool.IsConvex())
			{
				return false;
			}

			// A shell nested inside another makes winding number 2 inside it, which the exact
			// engine's unions are not defined for (on Thingi10K 54229 the part unioned with itself
			// keeps 0.29 of its volume). The tree unions the raw solid as a leaf and lost up to
			// 1.5% of the dilation on such parts; the ported sum reaches it in a different order
			// and answered them right, so it keeps them.
			if (HasNestedComponents(solid))
			{
				return false;
			}

			int numTri = solid.NumTri();
			int numHullLeaves = (numTri + LeafSize - 1) / LeafSize;
			int numLeaves = numHullLeaves + 1;

			// A binary reduction of L leaves performs exactly L - 1 unions, whatever the
			// carries, so the node count is known before the tree is built.
			ulong total = (ulong)numTri + (ulong)numLeaves + (ulong)(numLeaves - 1) + 1;
			Progress.BeginPhase(progress, Phase.Minkowski, total);

			// Read once so every node of the tree runs on the same engine even if a host
			// flips the process default while this is in flight.
			BooleanEngine engine = BooleanConfig.DefaultEngine();

			// Leaf 0 is the solid; leaf k > 0 is the union of the hulls of triangles
			// [(k-1)*LeafSize, k*LeafSize), built inside the leaf so only the leaves in
			// flight hold hulls, never all numTri of them. Each worker reads only the two
			// input meshes and writes its own slot.
			ManifoldImpl[]? level = Progress.MaybeParMapCtProgress(numLeaves, UnionParThreshold, token, progress, leaf =>
			{
				if (leaf == 0)
				{
					return solid.Clone();
				}

				int start = (leaf - 1) * LeafSize;
				int count = Math.Min(LeafSize, numTri - start);
				return LeafUnion(solid, tool, start, count, engine, token, progress);
			});

			while (level is not null && level.Length > 1)
			{
				// Per-level gate: a cancel that landed inside the last level's booleans comes
				// back as Cancelled leaves, which must not be unioned into a result.
				if (Cancel.IsCancelled(token))
				{
					break;
				}

				ManifoldImpl[] below = level;
				int pairs = below.Length / 2;
				ManifoldImpl[]? merged = Progress.MaybeParMapCtProgress(pairs, UnionParThreshold, token, progress, node =>
					Boolean3Functions.BooleanDispatch(
						below[2 * node],
						below[(2 * node) + 1],
						OpType.Add,
						engine,
						token));

				if (merged is null)
				{
					level = null;
					break;
				}

				if ((below.Length & 1) == 1)
				{
					// The odd node rides up unchanged; it costs no boolean and no unit.
					Array.Resize(ref merged, pairs + 1);
					merged[pairs] = below[below.Length - 1];
				}

				level = merged;
			}

			// The closing check CancelToken.cs's invariant requires: a cancelled token can
			// never produce a NoError result, whichever stage the cancel landed in.
			if (level is null || Cancel.IsCancelled(token))
			{
				result = Boolean3Functions.CancelledImpl();
				return true;
			}

			ManifoldImpl outR = level[0];

			// Minkowski.Compute's closing AsOriginal, so the two entry points hand back the
			// same kind of mesh: one fresh original ID, normals and coplanar faces set.
			outR.InitializeOriginal();
			outR.SetNormalsAndCoplanar();

			Progress.CompletePhase(progress);
			result = outR;
			return true;
		}

		/// <summary>
		/// Whether one connected component's bounding box contains another's - the cheap,
		/// conservative test for a shell nested inside another. Declining on a false positive
		/// (interlocked parts whose boxes nest) only costs speed.
		/// </summary>
		/// <param name="solid">The solid to test.</param>
		/// <returns>True when two components' boxes nest.</returns>
		private static bool HasNestedComponents(ManifoldImpl solid)
		{
			int numVert = solid.NumVert();
			DisjointSets sets = new DisjointSets((uint)numVert);
			foreach (Halfedge halfedge in solid.Halfedge)
			{
				if (halfedge.IsForward())
				{
					sets.Unite((uint)halfedge.StartVert, (uint)halfedge.EndVert);
				}
			}

			List<int> component = new List<int>(numVert);
			for (int i = 0; i < numVert; i++)
			{
				component.Add(0);
			}

			int numComponents = sets.ConnectedComponents(component);
			if (numComponents <= 1)
			{
				return false;
			}

			Vec3[] min = new Vec3[numComponents];
			Vec3[] max = new Vec3[numComponents];
			bool[] seen = new bool[numComponents];
			for (int v = 0; v < numVert; v++)
			{
				int c = component[v];
				Vec3 p = solid.VertPos[v];
				if (!seen[c])
				{
					min[c] = p;
					max[c] = p;
					seen[c] = true;
					continue;
				}

				min[c] = new Vec3(Math.Min(min[c].X, p.X), Math.Min(min[c].Y, p.Y), Math.Min(min[c].Z, p.Z));
				max[c] = new Vec3(Math.Max(max[c].X, p.X), Math.Max(max[c].Y, p.Y), Math.Max(max[c].Z, p.Z));
			}

			for (int a = 0; a < numComponents; a++)
			{
				for (int b = 0; b < numComponents; b++)
				{
					if (a != b
						&& seen[a]
						&& seen[b]
						&& min[a].X <= min[b].X && min[a].Y <= min[b].Y && min[a].Z <= min[b].Z
						&& max[a].X >= max[b].X && max[a].Y >= max[b].Y && max[a].Z >= max[b].Z)
					{
						return true;
					}
				}
			}

			return false;
		}

		/// <summary>
		/// Builds the hulls of <paramref name="count"/> triangles starting at
		/// <paramref name="start"/> and unions them through the CSG tree — the same
		/// reduction Minkowski.cs applies to a whole batch, on a batch of one leaf.
		/// </summary>
		/// <param name="solid">The solid whose triangles are swept.</param>
		/// <param name="tool">The convex tool.</param>
		/// <param name="start">The first triangle of the leaf.</param>
		/// <param name="count">How many triangles the leaf holds, at least one.</param>
		/// <param name="engine">The engine every union in the tree runs on, read once up front.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null; one unit per hull.</param>
		/// <returns>The union of the leaf's hulls.</returns>
		private static ManifoldImpl LeafUnion(
			ManifoldImpl solid,
			ManifoldImpl tool,
			int start,
			int count,
			BooleanEngine engine,
			CancelToken? token,
			ProgressReporter? progress)
		{
			List<CsgLeafNode> children = new List<CsgLeafNode>(count);
			for (int i = 0; i < count; i++)
			{
				if (Cancel.IsCancelled(token))
				{
					return Boolean3Functions.CancelledImpl();
				}

				// The vertex sums in Minkowski.cs's order, so each hull is the same hull.
				int tri = start + i;
				List<Vec3> simpleHull = new List<Vec3>(3 * tool.VertPos.Count);
				for (int k = 0; k < 3; k++)
				{
					Vec3 aVert = solid.VertPos[solid.Halfedge[(tri * 3) + k].StartVert];
					foreach (Vec3 bVert in tool.VertPos)
					{
						simpleHull.Add(aVert + bVert);
					}
				}

				ManifoldImpl hull = QuickHullFunctions.ConvexHull(simpleHull);
				progress?.Advance(1);
				if (count == 1)
				{
					return hull;
				}

				children.Add(new CsgLeafNode(hull));
			}

			// The same BatchUnion a CsgOp of these leaves would reach (identity transforms, so
			// flattening them changes nothing), called directly so it can take the engine.
			return CsgTree.BatchUnion(children, token, engine).GetImpl();
		}
	}
}
