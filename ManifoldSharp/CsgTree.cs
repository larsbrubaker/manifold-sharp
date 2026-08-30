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

// CsgTree.cs — port of csg_tree.rs, whose own header reads:
//
//   Phase 12: CSG Tree — ported from C++ csg_tree.cpp (764 lines)
//
//   Implements the full CSG tree evaluation system with:
//   - CsgLeafNode: lazy transform propagation, Arvo's AABB transform
//   - CsgOpNode: N-ary children with caching
//   - SimpleBoolean: wrapper invoking Boolean3
//   - BatchBoolean: min-heap approach for commutative ops
//   - BatchUnion: bounding-box partitioning + Compose + BatchBoolean
//   - Explicit-stack DFS evaluation (no recursion)
//
// Two notes on that header, which is transcribed above exactly as the Rust
// carries it:
//   - "min-heap" describes the C++ container, not the pop order. Both the C++
//     and the Rust pop the MOST-vert node; see CsgTree.Batch.cs.
//   - "no recursion" describes the intent, not the Rust body: `to_leaf_node`
//     calls `collect_children` calls `to_leaf_node_inner` calls `to_leaf_node`,
//     so the traversal recurses at tree depth in the Rust exactly as it does
//     here. Ported as written — the evaluation ORDER is the specification, and
//     converting it to an explicit stack is a change to that order's proof, not
//     a transcription. Depth is bounded by CSG nesting (collapsing flattens
//     associative runs into one batch), not by mesh size.
//
// This file carries the node types; the module's free functions (SimpleBoolean,
// BatchBoolean, BatchUnion — Rust module-level `fn`s, which C# has no place for)
// are on the `CsgTree` static class in CsgTree.Batch.cs.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	// -----------------------------------------------------------------------
	// CsgLeafNode — wraps an immutable mesh plus a lazy transform
	// -----------------------------------------------------------------------

	/// <summary>
	/// A mesh plus a not-yet-applied transform: the leaf of a CSG tree.
	/// </summary>
	/// <remarks>
	/// The Rust holds the mesh in an <c>Arc&lt;ManifoldImpl&gt;</c> so that composing a
	/// transform is a refcount bump rather than a mesh copy. C# reference semantics give
	/// that for free — <see cref="PImpl"/> is a shared reference, and the sharing is safe
	/// because this type is immutable and <see cref="GetImpl"/> never hands back the
	/// shared instance. The Rust's <c>#[derive(Clone)]</c> therefore has no counterpart:
	/// copying an immutable node is the same as sharing it.
	/// <para>
	/// One thing C# cannot reproduce: the Rust MOVES the mesh into the <c>Arc</c>, so the
	/// caller keeps no way to mutate it afterwards. Here the constructor only borrows the
	/// reference, and a caller that holds onto its own and edits the mesh after handing it
	/// over changes every leaf sharing it — including entries already ordered inside
	/// BatchBoolean's heap by their vertex count. Treat a mesh given to a leaf as given
	/// away; nothing in the port mutates one after intake.
	/// </para>
	/// </remarks>
	public sealed class CsgLeafNode
	{
		/// <summary>The mesh, shared by reference with every leaf derived from it.</summary>
		public readonly ManifoldImpl PImpl;

		/// <summary>The transform still owed to <see cref="PImpl"/>.</summary>
		public readonly Mat3x4 Transform;

		/// <summary>Create a leaf from a mesh with identity transform.</summary>
		/// <param name="mesh">The mesh to wrap; taken by reference, never copied.</param>
		public CsgLeafNode(ManifoldImpl mesh)
			: this(mesh, Mat3x4.Identity())
		{
		}

		/// <summary>Create a leaf from a mesh with a specific transform.</summary>
		/// <param name="mesh">The mesh to wrap; taken by reference, never copied.</param>
		/// <param name="transform">The transform to owe it.</param>
		public CsgLeafNode(ManifoldImpl mesh, Mat3x4 transform)
		{
			ArgumentNullException.ThrowIfNull(mesh);
			this.PImpl = mesh;
			this.Transform = transform;
		}

		/// <summary>Create an empty leaf.</summary>
		/// <returns>A leaf over an empty mesh with identity transform.</returns>
		public static CsgLeafNode Empty()
		{
			return new CsgLeafNode(new ManifoldImpl(), Mat3x4.Identity());
		}

		/// <summary>
		/// Empty leaf carrying <see cref="Error.Cancelled"/>, the value every cancelled
		/// branch of the tree evaluates to. Port of C++
		/// <c>ErrorLeaf(Manifold::Error::Cancelled)</c> (csg_tree.cpp:172, 460, 511, 759).
		/// </summary>
		/// <returns>The cancelled leaf.</returns>
		internal static CsgLeafNode Cancelled()
		{
			ManifoldImpl imp = new ManifoldImpl();
			imp.MakeEmpty(Error.Cancelled);
			return new CsgLeafNode(imp, Mat3x4.Identity());
		}

		/// <summary>
		/// Get the mesh, applying the lazy transform if needed.
		/// Port of C++ <c>CsgLeafNode::GetImpl()</c>.
		/// </summary>
		/// <remarks>
		/// Clone-don't-mutate, and that is load-bearing rather than tidy. C++ v3.5.2 had to
		/// add mutexes around this very function (#1740/#1741, "originalID races on shared
		/// CSG nodes") because it realizes the lazy transform by mutating through a const
		/// accessor, so two threads evaluating a shared leaf race on its originalID. That
		/// race is structurally impossible here for the same reason it is in the Rust:
		/// <c>GetImpl</c> clones and transforms into a FRESH impl instead of mutating in
		/// place, and there is no op-node cache and no interior mutability anywhere in this
		/// file. The identity branch clones too — callers do mutate what they get back
		/// (Minkowski finishes with <c>InitializeOriginal</c>), and handing them
		/// <see cref="PImpl"/> itself would edit every other leaf sharing it.
		/// </remarks>
		/// <returns>A fresh impl with the transform applied.</returns>
		public ManifoldImpl GetImpl()
		{
			if (this.Transform == Mat3x4.Identity())
			{
				return this.PImpl.Clone();
			}

			// ManifoldImpl.Transform returns the transformed mesh (it is not in-place) —
			// discarding its return value silently drops the lazy transform.
			return this.PImpl.Transform(this.Transform);
		}

		/// <summary>
		/// Return a new leaf with composed transform.
		/// Port of C++ <c>CsgLeafNode::Transform()</c>.
		/// </summary>
		/// <param name="m">The transform to apply on the outside of the existing one.</param>
		/// <returns>A leaf over the same mesh with the composed transform.</returns>
		public CsgLeafNode ApplyTransform(Mat3x4 m)
		{
			Mat3x4 newTransform = Mat4ToMat3x4(Mat3x4ToMat4(m) * Mat3x4ToMat4(this.Transform));
			return new CsgLeafNode(this.PImpl, newTransform);
		}

		/// <summary>
		/// Get bounding box without materializing the full mesh.
		/// Uses Arvo's algorithm for AABB transform.
		/// Port of C++ <c>CsgLeafNode::GetBoundingBox()</c>.
		/// </summary>
		/// <returns>The transformed bounding box.</returns>
		public Box GetBoundingBox()
		{
			Box implBbox = this.PImpl.Bbox;
			if (this.Transform == Mat3x4.Identity())
			{
				return implBbox;
			}

			// Arvo's AABB transform: transform center and half-extents
			Vec3 center = (implBbox.Min + implBbox.Max) * 0.5;
			Vec3 half = (implBbox.Max - implBbox.Min) * 0.5;

			// Transform center point
			Mat3x4 mat = this.Transform;
			Vec3 newCenter = new Vec3(
				(mat[0].X * center.X) + (mat[1].X * center.Y) + (mat[2].X * center.Z) + mat[3].X,
				(mat[0].Y * center.X) + (mat[1].Y * center.Y) + (mat[2].Y * center.Z) + mat[3].Y,
				(mat[0].Z * center.X) + (mat[1].Z * center.Y) + (mat[2].Z * center.Z) + mat[3].Z);

			// Transform half-extents using absolute values of matrix entries
			Vec3 newHalf = new Vec3(
				(Math.Abs(mat[0].X) * half.X) + (Math.Abs(mat[1].X) * half.Y) + (Math.Abs(mat[2].X) * half.Z),
				(Math.Abs(mat[0].Y) * half.X) + (Math.Abs(mat[1].Y) * half.Y) + (Math.Abs(mat[2].Y) * half.Z),
				(Math.Abs(mat[0].Z) * half.X) + (Math.Abs(mat[1].Z) * half.Y) + (Math.Abs(mat[2].Z) * half.Z));

			return new Box { Min = newCenter - newHalf, Max = newCenter + newHalf };
		}

		/// <summary>Vertex count without triggering transform.</summary>
		/// <returns>The vertex count of the untransformed mesh.</returns>
		public int NumVert()
		{
			return this.PImpl.NumVert();
		}
	}

	// -----------------------------------------------------------------------
	// CsgNode — the main CSG tree node (leaf or N-ary operation)
	// -----------------------------------------------------------------------

	/// <summary>
	/// A node of a CSG tree: either a <see cref="CsgLeaf"/> or a <see cref="CsgOp"/>.
	/// </summary>
	/// <remarks>
	/// The Rust <c>enum CsgNode { Leaf(CsgLeafNode), Op { .. } }</c> becomes an abstract
	/// base with two sealed subclasses, per docs/PORTING_PLAN.md's enum-with-data rule.
	/// The traversal methods stay HERE, on the base, matching on the concrete type rather
	/// than dispatching virtually. That is deliberate: <see cref="CollectChildren"/>
	/// matches on the *child* while needing the parent's op and child index, so it could
	/// not become an override at all, and splitting only its siblings into overrides would
	/// scatter one collapsing rule across three files. Pattern matching here reads as the
	/// Rust's <c>match</c> does, which is what a three-way diff needs.
	/// </remarks>
	public abstract class CsgNode
	{
		// Only the two variants in this file may extend the enum.
		private protected CsgNode()
		{
		}

		/// <summary>
		/// Evaluate the CSG tree to produce a single mesh.
		/// Port of C++ <c>CsgOpNode::ToLeafNode()</c>.
		/// </summary>
		/// <returns>The evaluated mesh.</returns>
		public ManifoldImpl Evaluate()
		{
			return this.EvaluateWithToken(null);
		}

		/// <summary>
		/// <see cref="Evaluate"/> with cooperative cancellation.
		/// </summary>
		/// <remarks>
		/// A cancelled evaluation yields an empty mesh whose status is
		/// <see cref="Error.Cancelled"/>. Mirrors C++ <c>CsgOpNode::ToLeafNode(ctx)</c>
		/// (csg_tree.cpp:644-800), which checks the flag once per stack step and
		/// substitutes an <c>ErrorLeaf(Cancelled)</c> for the pending work.
		/// </remarks>
		/// <param name="token">The cancellation token, or null for an uncancellable run.</param>
		/// <returns>The evaluated mesh, or an empty Cancelled mesh.</returns>
		public ManifoldImpl EvaluateWithToken(CancelToken? token)
		{
			CsgLeafNode leaf = this.ToLeafNode(Mat3x4.Identity(), token);
			return leaf.GetImpl();
		}

		/// <summary>
		/// Internal: convert this node to a CsgLeafNode, applying the given parent transform.
		/// </summary>
		/// <param name="parentTransform">The transform inherited from the parent.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The resolved leaf.</returns>
		internal CsgLeafNode ToLeafNode(Mat3x4 parentTransform, CancelToken? token)
		{
			// One check per stack step, as C++ does at csg_tree.cpp:752. Cancel is
			// sticky, so every enclosing step short-circuits here too and the
			// Cancelled leaf propagates to the root without further work.
			if (Cancel.IsCancelled(token))
			{
				return CsgLeafNode.Cancelled();
			}

			if (this is CsgLeaf leaf)
			{
				return leaf.Node.ApplyTransform(parentTransform);
			}

			CsgOp opNode = (CsgOp)this;

			// Compose local transform with parent
			Mat3x4 combined = Mat4ToMat3x4(
				Mat3x4ToMat4(parentTransform) * Mat3x4ToMat4(opNode.Transform));

			// Flatten: recursively resolve all children to leaves
			List<CsgLeafNode> positive = new List<CsgLeafNode>();
			List<CsgLeafNode> negative = new List<CsgLeafNode>();

			CollectChildren(opNode.Op, combined, opNode.Children, positive, negative, token);

			// Perform the operation
			switch (opNode.Op)
			{
				case OpType.Add:
					// Union of all positive children
					return CsgTree.BatchUnion(positive, token);

				case OpType.Intersect:
					// Intersection of all positive children
					return CsgTree.BatchBoolean(OpType.Intersect, positive, token);

				case OpType.Subtract:
					// Subtract: first child is positive, rest are negative
					if (positive.Count == 0)
					{
						// `CollectChildren` may have produced a Cancelled leaf and no
						// positive one; returning the plain empty leaf here would launder
						// that into NoError.
						return Cancel.IsCancelled(token) ? CsgLeafNode.Cancelled() : CsgLeafNode.Empty();
					}

					CsgLeafNode posResult = CsgTree.BatchUnion(positive, token);
					if (negative.Count == 0)
					{
						return posResult;
					}

					CsgLeafNode negResult = CsgTree.BatchUnion(negative, token);
					return CsgTree.SimpleBoolean(posResult, negResult, OpType.Subtract, token);

				default:
					// The Rust's match is exhaustive over three variants and has no fallback
					// arm. C# enums are not closed — `(OpType)7` is a legal value — so the
					// case the Rust cannot reach gets an explicit throw rather than silently
					// falling into Subtract, which is what listing Subtract under `default`
					// would have done.
					throw new ArgumentOutOfRangeException(
						nameof(opNode),
						$"Unknown OpType value: {(int)opNode.Op}");
			}
		}

		/// <summary>
		/// Recursively collect children, flattening compatible operations.
		/// Port of the collapsing logic in C++ <c>CsgOpNode::ToLeafNode</c>.
		/// </summary>
		/// <remarks>
		/// Static because the Rust takes <c>&amp;self</c> and never reads it — every operand
		/// the body needs is already a parameter.
		/// </remarks>
		/// <param name="parentOp">The operation of the node owning these children.</param>
		/// <param name="transform">The transform accumulated down to this node.</param>
		/// <param name="children">The children to collect.</param>
		/// <param name="positive">Receives the additive operands.</param>
		/// <param name="negative">Receives the subtractive operands.</param>
		/// <param name="token">The cancellation token, or null.</param>
		private static void CollectChildren(
			OpType parentOp,
			Mat3x4 transform,
			IReadOnlyList<CsgNode> children,
			List<CsgLeafNode> positive,
			List<CsgLeafNode> negative,
			CancelToken? token)
		{
			for (int i = 0; i < children.Count; i++)
			{
				CsgNode child = children[i];
				if (child is CsgLeaf childLeaf)
				{
					CsgLeafNode transformed = childLeaf.Node.ApplyTransform(transform);
					if (parentOp == OpType.Subtract && i > 0)
					{
						negative.Add(transformed);
					}
					else
					{
						positive.Add(transformed);
					}

					continue;
				}

				CsgOp childOpNode = (CsgOp)child;
				OpType childOp = childOpNode.Op;
				IReadOnlyList<CsgNode> grandchildren = childOpNode.Children;

				Mat3x4 combined = Mat4ToMat3x4(
					Mat3x4ToMat4(transform) * Mat3x4ToMat4(childOpNode.Transform));

				// Collapsing: flatten compatible ops
				bool canCollapse;
				if (parentOp == OpType.Add && childOp == OpType.Add)
				{
					// Union is associative: (A ∪ B) ∪ C = A ∪ B ∪ C
					canCollapse = true;
				}
				else if (parentOp == OpType.Intersect && childOp == OpType.Intersect)
				{
					// Intersection is associative: (A ∩ B) ∩ C = A ∩ B ∩ C
					canCollapse = true;
				}
				else if (parentOp == OpType.Subtract && childOp == OpType.Subtract && i == 0)
				{
					// (A - B) - C = A - (B ∪ C): first child's subtraction collapses
					canCollapse = true;
				}
				else
				{
					canCollapse = false;
				}

				if (canCollapse)
				{
					// Flatten: merge grandchildren directly
					if (parentOp == OpType.Subtract && childOp == OpType.Subtract && i == 0)
					{
						// (A - B) is first child of Subtract: A goes to positive, B goes to negative
						for (int gi = 0; gi < grandchildren.Count; gi++)
						{
							CsgLeafNode leaf = grandchildren[gi].ToLeafNodeInner(combined, token);
							if (gi == 0)
							{
								positive.Add(leaf);
							}
							else
							{
								negative.Add(leaf);
							}
						}
					}
					else
					{
						foreach (CsgNode gc in grandchildren)
						{
							CsgLeafNode leaf = gc.ToLeafNodeInner(combined, token);
							if (parentOp == OpType.Subtract && i > 0)
							{
								negative.Add(leaf);
							}
							else
							{
								positive.Add(leaf);
							}
						}
					}
				}
				else
				{
					// Cannot collapse: evaluate child subtree fully
					CsgLeafNode result = child.ToLeafNode(combined, token);
					if (parentOp == OpType.Subtract && i > 0)
					{
						negative.Add(result);
					}
					else
					{
						positive.Add(result);
					}
				}
			}
		}

		/// <summary>
		/// Helper: convert a single node to leaf with given transform (non-flattening).
		/// </summary>
		/// <param name="transform">The transform to apply.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The resolved leaf.</returns>
		private CsgLeafNode ToLeafNodeInner(Mat3x4 transform, CancelToken? token)
		{
			if (this is CsgLeaf leaf)
			{
				return leaf.Node.ApplyTransform(transform);
			}

			return this.ToLeafNode(transform, token);
		}
	}

	/// <summary>
	/// The Rust's <c>CsgNode::Leaf(CsgLeafNode)</c> variant: a tree node holding one mesh.
	/// </summary>
	public sealed class CsgLeaf : CsgNode
	{
		/// <summary>The mesh-plus-lazy-transform this node wraps.</summary>
		public readonly CsgLeafNode Node;

		/// <summary>The Rust <c>CsgNode::leaf</c> — wrap a mesh with identity transform.</summary>
		/// <param name="mesh">The mesh.</param>
		public CsgLeaf(ManifoldImpl mesh)
			: this(new CsgLeafNode(mesh))
		{
		}

		/// <summary>The Rust <c>CsgNode::leaf_node</c> — wrap an existing leaf.</summary>
		/// <param name="node">The leaf.</param>
		public CsgLeaf(CsgLeafNode node)
		{
			ArgumentNullException.ThrowIfNull(node);
			this.Node = node;
		}
	}

	/// <summary>
	/// The Rust's <c>CsgNode::Op { op, children, transform }</c> variant: an N-ary
	/// operation over child nodes, C++'s <c>CsgOpNode</c>.
	/// </summary>
	public sealed class CsgOp : CsgNode
	{
		/// <summary>The operation to apply across <see cref="Children"/>.</summary>
		public readonly OpType Op;

		/// <summary>The operands, in order; for Subtract the first is the minuend.</summary>
		public readonly IReadOnlyList<CsgNode> Children;

		/// <summary>The transform this node contributes to the whole subtree.</summary>
		public readonly Mat3x4 Transform;

		/// <summary>The Rust <c>CsgNode::op</c> — a binary operation with identity transform.</summary>
		/// <param name="op">The operation.</param>
		/// <param name="left">The first operand.</param>
		/// <param name="right">The second operand.</param>
		public CsgOp(OpType op, CsgNode left, CsgNode right)
			: this(op, new CsgNode[] { left, right }, Mat3x4.Identity())
		{
		}

		/// <summary>The Rust <c>CsgNode::op_n</c> — an N-ary operation with identity transform.</summary>
		/// <param name="op">The operation.</param>
		/// <param name="children">The operands, in order.</param>
		public CsgOp(OpType op, IReadOnlyList<CsgNode> children)
			: this(op, children, Mat3x4.Identity())
		{
		}

		/// <summary>An N-ary operation carrying a transform of its own.</summary>
		/// <param name="op">The operation.</param>
		/// <param name="children">The operands, in order.</param>
		/// <param name="transform">The subtree transform.</param>
		public CsgOp(OpType op, IReadOnlyList<CsgNode> children, Mat3x4 transform)
		{
			ArgumentNullException.ThrowIfNull(children);

			// Copied, not aliased: the Rust variant owns its Vec, and a caller mutating the
			// list it passed would otherwise reach inside an already-built tree.
			CsgNode[] copy = new CsgNode[children.Count];
			for (int i = 0; i < children.Count; i++)
			{
				copy[i] = children[i] ?? throw new ArgumentNullException(nameof(children));
			}

			this.Op = op;
			this.Children = copy;
			this.Transform = transform;
		}
	}
}
