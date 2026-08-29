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

// Collider — BVH-based broadphase collision detection.
// Uses a radix tree built from Morton codes for O(n log n) queries,
// matching the C++ Manifold implementation.
//
// The free functions of collider.rs (edge_edge_dist,
// distance_triangle_triangle_squared, ray_triangle_intersection) and the
// `impl ManifoldImpl` block it carries (is_self_intersecting, min_gap) live in
// Collider.Geometry.cs; the two files together are the port of the one Rust
// file, split for the 800-line cap.
//
// ── The node layout is load-bearing ──────────────────────────────────────────
// One interleaved node array, leaves at even indices and internal nodes at odd,
// root at index 1. That is the C++ storage layout, kept exactly: it is a
// documented memory-parity win (leaf boxes live inside node_bbox, so there is no
// separate leaf copy, and the Morton codes are consumed at construction rather
// than stored — which matters because a collider is cached on every
// ManifoldImpl). Do not "simplify" it into separate leaf/internal arrays.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <summary>
	/// Face BVH matching the C++ Collider's storage layout exactly: one interleaved node
	/// array (leaves at even indices, internals at odd), the radix-tree topology, and
	/// parent links for bottom-up box refits.
	/// </summary>
	/// <remarks>
	/// Leaf boxes live inside <c>nodeBBox</c> — no separate leaf copy, and the morton
	/// codes are consumed at construction rather than stored, which matters because a
	/// collider is cached on every <see cref="ManifoldImpl"/>.
	/// <para>
	/// The Rust type is a <c>Clone</c> struct; here it is a sealed class, because it owns
	/// three arrays and every use site holds it as a field of an impl. Copies are
	/// therefore explicit — see <see cref="Clone"/>, which every ported site that relied
	/// on Rust's implicit clone has to call.
	/// </para>
	/// </remarks>
	public sealed class Collider
	{
		// Node encoding (matches C++):
		// - Even indices are leaf nodes: leaf i -> node 2*i
		// - Odd indices are internal nodes: internal i -> node 2*i + 1
		// - Root is always node index 1 (internal node 0)
		private const int KRoot = 1;

		// AABBs for all nodes (2*num_leaves - 1)
		private Box[] nodeBBox;

		// parent node per node (-1 at root)
		private int[] nodeParent;

		// child pairs for internal nodes (num_leaves - 1)
		private ChildPair[] internalChildren;

		/// <summary>
		/// Creates an empty collider — the Rust derived <c>Default</c>: no nodes, so every
		/// query returns nothing.
		/// </summary>
		public Collider()
		{
			this.nodeBBox = Array.Empty<Box>();
			this.nodeParent = Array.Empty<int>();
			this.internalChildren = Array.Empty<ChildPair>();
		}

		/// <summary>
		/// Create a new Collider from leaf bounding boxes and morton codes.
		/// </summary>
		/// <remarks>
		/// Like the C++ constructor, leaves must already be sorted by morton code — every
		/// production caller builds them from a morton-sorted source (SortGeometry sorts
		/// faces, Merge sorts open verts).
		/// </remarks>
		/// <param name="leafBbox">The leaf boxes, in morton order.</param>
		/// <param name="leafMorton">The matching morton codes, ascending.</param>
		public Collider(Box[] leafBbox, uint[] leafMorton)
			: this()
		{
			System.Diagnostics.Debug.Assert(
				leafBbox.Length == leafMorton.Length,
				"Collider leaf boxes and morton codes must be the same length");
			System.Diagnostics.Debug.Assert(
				IsSortedAscending(leafMorton),
				"Collider leaves must be pre-sorted by morton code");

			int n = leafBbox.Length;
			if (n == 0)
			{
				return;
			}

			int numNodes = (2 * n) - 1;

			// Box.Filled, never `new Box[numNodes]` — collider.rs:79 is
			// `vec![BBox::default(); num_nodes]`, and the Rust default is the *inverted*
			// empty box. A zeroed array would seed every unwritten node with a box at the
			// origin, which reports IsEmpty() == false and drags any refit into it.
			this.nodeBBox = Box.Filled(numNodes);
			this.nodeParent = new int[numNodes];
			for (int i = 0; i < numNodes; i++)
			{
				this.nodeParent[i] = -1;
			}

			// Rust `n.saturating_sub(1)`; n >= 1 here because n == 0 returned above.
			this.internalChildren = new ChildPair[n - 1];
			for (int i = 0; i < this.internalChildren.Length; i++)
			{
				this.internalChildren[i] = new ChildPair(-1, -1);
			}

			this.CreateRadixTree(leafMorton);
			this.UpdateBoxes(leafBbox);
		}

		/// <summary>The number of leaves the tree was built over.</summary>
		/// <returns>The leaf count, or 0 for an empty collider.</returns>
		internal int NumLeaves()
		{
			if (this.nodeBBox.Length == 0)
			{
				return 0;
			}

			return this.internalChildren.Length + 1;
		}

		/// <summary>The Rust <c>Collider::morton_code</c> — the sort.rs function, re-exported.</summary>
		/// <param name="position">The position to encode.</param>
		/// <param name="bbox">The box the position is normalized against.</param>
		/// <returns>The Morton code.</returns>
		public static uint MortonCode(Vec3 position, Box bbox)
		{
			return Sort.MortonCode(position, bbox);
		}

		/// <summary>
		/// Whether the transform maps every axis to an axis, so it carries AABBs to exact
		/// AABBs.
		/// </summary>
		/// <param name="transform">The transform to test.</param>
		/// <returns>True when the transform is axis-aligned.</returns>
		public static bool IsAxisAligned(Mat3x4 transform)
		{
			for (int row = 0; row < 3; row++)
			{
				int zeroCount = 0;
				for (int col = 0; col < 3; col++)
				{
					if (transform[col][row] == 0.0)
					{
						zeroCount++;
					}
				}

				if (zeroCount != 2)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>The Rust derived <c>Clone</c>: an independent copy of the whole tree.</summary>
		/// <returns>A deep copy of this collider.</returns>
		public Collider Clone()
		{
			Collider copy = new Collider();
			copy.nodeBBox = (Box[])this.nodeBBox.Clone();
			copy.nodeParent = (int[])this.nodeParent.Clone();
			copy.internalChildren = (ChildPair[])this.internalChildren.Clone();
			return copy;
		}

		/// <summary>
		/// Run a single query box against the BVH, invoking <c>record(queryIdx, leafIdx)</c>
		/// per overlap.
		/// </summary>
		/// <remarks>
		/// Same traversal as <see cref="CollisionsFn"/> for one index — the per-query entry
		/// point for parallel callers (read-only, so queries can run concurrently with
		/// thread-local recording).
		/// </remarks>
		/// <param name="query">The query box.</param>
		/// <param name="queryIdx">The index reported back for this query.</param>
		/// <param name="record">Called once per overlapping leaf.</param>
		public void CollisionsOne(Box query, int queryIdx, Action<int, int> record)
		{
			if (query.IsEmpty())
			{
				return;
			}

			// No internal nodes (0-1 leaves): no collisions, matching C++
			// Collisions' early return on internalChildren_.empty().
			if (this.internalChildren.Length == 0)
			{
				return;
			}

			this.TraverseBvh(query, queryIdx, false, record);
		}

		/// <summary>
		/// BVH-accelerated collision query with function-generated query boxes. For each
		/// query index 0..n, calls <paramref name="queryBoxFn"/> to get the query AABB, then
		/// traverses the BVH to find overlapping leaves.
		/// </summary>
		/// <param name="queryBoxFn">Produces the query box for an index.</param>
		/// <param name="n">The number of queries.</param>
		/// <param name="record">Called once per overlapping (query, leaf) pair.</param>
		public void CollisionsFn(Func<int, Box> queryBoxFn, int n, Action<int, int> record)
		{
			if (this.internalChildren.Length == 0)
			{
				return;
			}

			for (int queryIdx = 0; queryIdx < n; queryIdx++)
			{
				Box query = queryBoxFn(queryIdx);
				if (query.IsEmpty())
				{
					continue;
				}

				this.TraverseBvh(query, queryIdx, false, record);
			}
		}

		/// <summary>BVH-accelerated collision query with pre-computed query boxes.</summary>
		/// <param name="queries">The query boxes.</param>
		/// <param name="selfCollision">When true, a leaf never collides with its own index.</param>
		/// <param name="record">Called once per overlapping (query, leaf) pair.</param>
		public void CollisionsWithBoxes(Box[] queries, bool selfCollision, Action<int, int> record)
		{
			if (this.internalChildren.Length == 0)
			{
				return;
			}

			for (int queryIdx = 0; queryIdx < queries.Length; queryIdx++)
			{
				Box query = queries[queryIdx];
				if (query.IsEmpty())
				{
					continue;
				}

				this.TraverseBvh(query, queryIdx, selfCollision, record);
			}
		}

		/// <summary>Point-based collision query using BVH.</summary>
		/// <param name="pointFn">Produces the query point for an index.</param>
		/// <param name="n">The number of queries.</param>
		/// <param name="record">Called once per overlapping (query, leaf) pair.</param>
		public void CollisionsPoint(Func<int, Vec3> pointFn, int n, Action<int, int> record)
		{
			if (this.internalChildren.Length == 0)
			{
				return;
			}

			for (int queryIdx = 0; queryIdx < n; queryIdx++)
			{
				Vec3 pt = pointFn(queryIdx);
				Box query = Box.FromPoint(pt);
				this.TraverseBvh(query, queryIdx, false, record);
			}
		}

		/// <summary>
		/// Replace the leaf boxes (given in leaf/tree order — for a face collider that is
		/// face order, since faces are morton-sorted) and refit the internal boxes. Tree
		/// topology is untouched (C++ UpdateBoxes).
		/// </summary>
		/// <param name="leafBbox">The new leaf boxes, in leaf order.</param>
		public void UpdateBoxes(Box[] leafBbox)
		{
			System.Diagnostics.Debug.Assert(
				leafBbox.Length == this.NumLeaves(),
				"UpdateBoxes needs exactly one box per leaf");

			for (int i = 0; i < leafBbox.Length; i++)
			{
				this.nodeBBox[LeafToNode(i)] = leafBbox[i];
			}

			this.BuildInternalBoxes();
		}

		/// <summary>
		/// Map every node box through an axis-aligned transform (C++ Collider::Transform) —
		/// no refit needed since axis-aligned transforms map AABBs to exact AABBs.
		/// </summary>
		/// <param name="transform">The axis-aligned transform to apply.</param>
		public void Transform(Mat3x4 transform)
		{
			System.Diagnostics.Debug.Assert(
				IsAxisAligned(transform),
				"Collider.Transform requires an axis-aligned transform");

			for (int i = 0; i < this.nodeBBox.Length; i++)
			{
				this.nodeBBox[i] = this.nodeBBox[i].Transform(transform);
			}
		}

		private static int LeafToNode(int leaf)
		{
			return leaf * 2;
		}

		private static int NodeToLeaf(int node)
		{
			return node / 2;
		}

		private static int InternalToNode(int internalIdx)
		{
			return (internalIdx * 2) + 1;
		}

		private static int NodeToInternal(int node)
		{
			return node / 2;
		}

		private static bool IsLeaf(int node)
		{
			return node % 2 == 0;
		}

		private static bool IsInternal(int node)
		{
			return node % 2 == 1;
		}

		private static bool IsSortedAscending(uint[] codes)
		{
			for (int i = 0; i + 1 < codes.Length; i++)
			{
				if (codes[i] > codes[i + 1])
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// Build radix tree from sorted Morton codes (matches C++ CreateRadixTree). Only
		/// fills in topology (internalChildren, nodeParent); boxes are populated afterwards
		/// by UpdateBoxes.
		/// </summary>
		private void CreateRadixTree(uint[] leafMorton)
		{
			int numLeaves = leafMorton.Length;
			if (numLeaves <= 1)
			{
				return;
			}

			int numInternal = numLeaves - 1;

			const int KInitialLength = 128;
			const int KLengthMultiple = 4;

			// Helper: count leading zeros of XOR of two morton codes
			int PrefixLength(int i, int j)
			{
				if (j < 0 || j >= numLeaves)
				{
					return -1;
				}

				uint mi = leafMorton[i];
				uint mj = leafMorton[j];
				uint xor = mi ^ mj;
				if (xor == 0)
				{
					// Same morton code, use index as tiebreaker (matches C++ clz)
					return 32 + System.Numerics.BitOperations.LeadingZeroCount((uint)i ^ (uint)j);
				}

				return System.Numerics.BitOperations.LeadingZeroCount(xor);
			}

			// RangeEnd: find the other end of the range for internal node i
			int RangeEnd(int i)
			{
				int dirVal = PrefixLength(i, i + 1) - PrefixLength(i, i - 1);
				int dir = dirVal > 0 ? 1 : (dirVal < 0 ? -1 : -1);
				int commonPrefix = PrefixLength(i, i - dir);
				int maxLength = KInitialLength;
				while (PrefixLength(i, i + (dir * maxLength)) > commonPrefix)
				{
					maxLength *= KLengthMultiple;
				}

				int length = 0;
				int step = maxLength / 2;
				while (step > 0)
				{
					if (PrefixLength(i, i + (dir * (length + step))) > commonPrefix)
					{
						length += step;
					}

					step /= 2;
				}

				return i + (dir * length);
			}

			// FindSplit: find where the split occurs within [first, last]
			int FindSplit(int first, int last)
			{
				int commonPrefix = PrefixLength(first, last);
				int split = first;
				int step = last - first;
				while (true)
				{
					step = (step + 1) >> 1; // divide by 2, rounding up
					int newSplit = split + step;
					if (newSplit < last)
					{
						int splitPrefix = PrefixLength(first, newSplit);
						if (splitPrefix > commonPrefix)
						{
							split = newSplit;
						}
					}

					if (step <= 1)
					{
						break;
					}
				}

				return split;
			}

			// For each internal node, find its range and split point
			for (int internalIdx = 0; internalIdx < numInternal; internalIdx++)
			{
				int i = internalIdx;
				int first = i;
				int last = RangeEnd(i);
				if (first > last)
				{
					(first, last) = (last, first);
				}

				int split = FindSplit(first, last);

				// Assign children (matches C++ exactly)
				int child1 = split == first ? LeafToNode(split) : InternalToNode(split);

				// C++ increments split before computing child2
				int split2 = split + 1;
				int child2;
				if (split2 == last)
				{
					child2 = LeafToNode(split2);
				}
				else if (split2 < numInternal)
				{
					child2 = InternalToNode(split2);
				}
				else
				{
					// Degenerate case: split2 exceeds internal node range.
					// This mirrors C++ UB when dir=0 in RangeEnd (clz(0) is UB).
					// Make child2 = child1 so traversal still works.
					child2 = child1;
				}

				this.internalChildren[internalIdx] = new ChildPair(child1, child2);
				int node = InternalToNode(i);
				this.nodeParent[child1] = node;
				this.nodeParent[child2] = node;
			}
		}

		/// <summary>
		/// Refit internal boxes bottom-up from the leaf boxes already present in nodeBBox,
		/// using the parent links (matches C++ BuildInternalBoxes).
		/// </summary>
		private void BuildInternalBoxes()
		{
			int numLeaves = this.NumLeaves();
			int numInternal = this.internalChildren.Length;
			if (numInternal == 0)
			{
				return;
			}

			uint[] counter = new uint[numInternal];

			for (int leaf = 0; leaf < numLeaves; leaf++)
			{
				int node = LeafToNode(leaf);
				while (true)
				{
					int parent = this.nodeParent[node];
					if (parent < 0)
					{
						break; // at root
					}

					int internalIdx = NodeToInternal(parent);
					if (internalIdx < 0 || internalIdx >= numInternal)
					{
						break;
					}

					counter[internalIdx]++;
					if (counter[internalIdx] < 2)
					{
						break; // wait for second child
					}

					// Both children ready, compute union
					ChildPair children = this.internalChildren[internalIdx];
					Box b1 = this.nodeBBox[children.Child1];
					Box b2 = this.nodeBBox[children.Child2];
					this.nodeBBox[parent] = b1.UnionBox(b2);
					node = parent;
				}
			}
		}

		/// <summary>Stack-based depth-first BVH traversal (matches C++ FindCollision).</summary>
		private void TraverseBvh(Box query, int queryIdx, bool selfCollision, Action<int, int> record)
		{
			Span<int> stack = stackalloc int[64];
			int top = -1;
			int node = KRoot;

			while (true)
			{
				int internalIdx = NodeToInternal(node);
				if (internalIdx < 0 || internalIdx >= this.internalChildren.Length)
				{
					if (top < 0)
					{
						break;
					}

					node = stack[top];
					top -= 1;
					continue;
				}

				ChildPair children = this.internalChildren[internalIdx];
				int child1 = children.Child1;
				int child2 = children.Child2;

				bool traverse1 = this.CheckNode(query, child1, queryIdx, selfCollision, record);
				bool traverse2 = this.CheckNode(query, child2, queryIdx, selfCollision, record);

				if (!traverse1 && !traverse2)
				{
					if (top < 0)
					{
						break;
					}

					node = stack[top];
					top -= 1;
				}
				else
				{
					node = traverse1 ? child1 : child2;
					if (traverse1 && traverse2)
					{
						top += 1;
						System.Diagnostics.Debug.Assert(top < 64, "BVH stack overflow");
						stack[top] = child2;
					}
				}
			}
		}

		/// <summary>
		/// Check if a node's AABB overlaps the query. If it's a leaf, record the hit.
		/// Returns true if the node is internal and overlaps (should traverse deeper).
		/// </summary>
		private bool CheckNode(Box query, int node, int queryIdx, bool selfCollision, Action<int, int> record)
		{
			if (node < 0 || node >= this.nodeBBox.Length)
			{
				return false;
			}

			Box nodeBox = this.nodeBBox[node];
			bool overlaps = query.DoesOverlapBox(nodeBox);
			if (overlaps && IsLeaf(node))
			{
				// Leaves are stored in tree order == input order (pre-sorted by
				// morton), so the leaf index IS the caller's index — no mapping.
				int leafIdx = NodeToLeaf(node);
				if (!selfCollision || leafIdx != queryIdx)
				{
					record(queryIdx, leafIdx);
				}
			}

			return overlaps && IsInternal(node);
		}

		/// <summary>
		/// The two children of an internal node — the Rust <c>[i32; 2]</c>, named so the
		/// destructuring at the two use sites reads like the Rust's.
		/// </summary>
		private readonly struct ChildPair
		{
			/// <summary>Creates a child pair.</summary>
			public ChildPair(int child1, int child2)
			{
				this.Child1 = child1;
				this.Child2 = child2;
			}

			/// <summary>The first child's node index.</summary>
			public int Child1 { get; }

			/// <summary>The second child's node index.</summary>
			public int Child2 { get; }
		}
	}
}
