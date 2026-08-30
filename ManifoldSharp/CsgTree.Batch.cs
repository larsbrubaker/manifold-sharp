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

// CsgTree.Batch.cs — the free functions of csg_tree.rs: SimpleBoolean,
// BatchBoolean and BatchUnion. Rust module-level `fn`s land on a static class
// named for the module (CLAUDE.md naming rule); the node types they
// operate on, and the module header, are in CsgTree.cs.
//
// The class AND its members are `internal`, matching the Rust's module-private
// `fn`s: a public class holding only internal members would advertise a surface
// that is not there. The Rust's tests are in-module and call these directly, so
// CsgTreeTests does too, through InternalsVisibleTo. Nothing outside the
// assembly needs them — a consumer reduces a list of meshes by building a
// CsgOp and calling Evaluate, which is the public route and the one Minkowski
// takes.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <summary>
	/// The free functions of <c>csg_tree.rs</c> — the reductions that turn a list of
	/// leaves into one leaf.
	/// </summary>
	internal static class CsgTree
	{
		// -------------------------------------------------------------------
		// BatchUnion — bounding-box partitioning + Compose + BatchBoolean
		// Port of C++ BatchUnion() (lines 434-491)
		// -------------------------------------------------------------------
		private const int KMaxUnionSize = 1000;

		// -------------------------------------------------------------------
		// SimpleBoolean — wrapper invoking Boolean3
		// Port of C++ SimpleBoolean() (lines 142-184)
		// -------------------------------------------------------------------

		/// <summary>
		/// The two-operand boolean every reduction below bottoms out in.
		/// </summary>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <param name="op">The operation.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The result as a fresh identity-transform leaf.</returns>
		internal static CsgLeafNode SimpleBoolean(CsgLeafNode a, CsgLeafNode b, OpType op, CancelToken? token)
		{
			// Entry gate before the (expensive) transform materialisation, matching
			// C++ SimpleBoolean's first line (csg_tree.cpp:172).
			if (Cancel.IsCancelled(token))
			{
				return CsgLeafNode.Cancelled();
			}

			ManifoldImpl implA = a.GetImpl();
			ManifoldImpl implB = b.GetImpl();

			// Engine selection: CSG evaluation honors the process-global default
			// (BooleanConfig). With the default (Exact) this call resolves to
			// Boolean3Functions.BooleanWithToken — behavior byte-identical to before the
			// robust engine existed. A process that has set the default to Robust or Auto
			// gets that engine here too, on exactly the routing rule
			// BooleanDispatch applies to every other caller: there is no CSG-specific
			// engine policy, and no CSG-specific failure mode.
			ManifoldImpl result = Boolean3Functions.BooleanDispatch(
				implA,
				implB,
				op,
				BooleanConfig.DefaultEngine(),
				token);
			return new CsgLeafNode(result);
		}

		// -------------------------------------------------------------------
		// BatchBoolean — heap-ordered reduction for commutative ops
		// Port of C++ BatchBoolean() in csg_tree.cpp (v3.5.0)
		// -------------------------------------------------------------------

		/// <summary>
		/// Reduce <paramref name="children"/> to one leaf with <paramref name="op"/>,
		/// in the C++ heap order.
		/// </summary>
		/// <remarks>
		/// Consumes <paramref name="children"/>: the Rust takes <c>&amp;mut Vec</c> and
		/// drains it, and callers below rely on the list being emptied.
		/// </remarks>
		/// <param name="op">The (commutative) operation to reduce with.</param>
		/// <param name="children">The operands; emptied by this call.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The reduced leaf.</returns>
		internal static CsgLeafNode BatchBoolean(OpType op, List<CsgLeafNode> children, CancelToken? token)
		{
			if (children.Count == 0)
			{
				return CsgLeafNode.Empty();
			}

			if (children.Count == 1)
			{
				CsgLeafNode only = children[0];
				children.RemoveAt(0);
				return only;
			}

			if (children.Count == 2)
			{
				CsgLeafNode second = children[children.Count - 1];
				children.RemoveAt(children.Count - 1);
				CsgLeafNode first = children[children.Count - 1];
				children.RemoveAt(children.Count - 1);
				return SimpleBoolean(first, second, op, token);
			}

			PriorityQueue<CsgLeafNode, MeshEntryKey> heap =
				new PriorityQueue<CsgLeafNode, MeshEntryKey>(MeshEntryKey.PopMostVertsFirst);
			ulong nextSerial = (ulong)children.Count;
			for (int i = 0; i < children.Count; i++)
			{
				heap.Enqueue(children[i], new MeshEntryKey(children[i].NumVert(), (ulong)i));
			}

			children.Clear();

			// C++ processes up to 4 pairs per round (for its parallel lane), pushing
			// the results back only at the end of the round — even in sequential
			// builds. The round structure changes which meshes pair up, so mirror it.
			List<(CsgLeafNode Node, MeshEntryKey Key)> tmp = new List<(CsgLeafNode, MeshEntryKey)>();
			while (heap.Count > 1)
			{
				// Once-per-round check, matching C++ BatchBoolean's per-iteration gate
				// (csg_tree.cpp:460).
				if (Cancel.IsCancelled(token))
				{
					return CsgLeafNode.Cancelled();
				}

				for (int round = 0; round < 4; round++)
				{
					if (heap.Count <= 1)
					{
						break;
					}

					CsgLeafNode a = heap.Dequeue();
					CsgLeafNode b = heap.Dequeue();
					CsgLeafNode result = SimpleBoolean(a, b, op, token);
					tmp.Add((result, new MeshEntryKey(result.NumVert(), nextSerial)));
					nextSerial += 1;
				}

				foreach ((CsgLeafNode Node, MeshEntryKey Key) entry in tmp)
				{
					heap.Enqueue(entry.Node, entry.Key);
				}

				tmp.Clear();
			}

			return heap.Dequeue();
		}

		/// <summary>
		/// Union <paramref name="children"/>, composing disjoint groups instead of
		/// intersecting them.
		/// </summary>
		/// <remarks>
		/// Consumes <paramref name="children"/>, as the Rust's <c>&amp;mut Vec</c> does.
		/// </remarks>
		/// <param name="children">The operands; emptied by this call.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The union as a single leaf.</returns>
		internal static CsgLeafNode BatchUnion(List<CsgLeafNode> children, CancelToken? token)
		{
			if (children.Count == 0)
			{
				return CsgLeafNode.Empty();
			}

			if (children.Count == 1)
			{
				CsgLeafNode only = children[0];
				children.RemoveAt(0);
				return only;
			}

			// Process in chunks to avoid O(n^2) overlap checks
			while (children.Count > 1)
			{
				// Once-per-chunk check, matching C++ BatchUnion (csg_tree.cpp:511).
				if (Cancel.IsCancelled(token))
				{
					return CsgLeafNode.Cancelled();
				}

				int chunkSize = Math.Min(children.Count, KMaxUnionSize);
				int chunkStart = children.Count - chunkSize;

				// Get bounding boxes for the chunk
				Box[] boxes = new Box[chunkSize];
				for (int i = 0; i < chunkSize; i++)
				{
					boxes[i] = children[chunkStart + i].GetBoundingBox();
				}

				// Greedy partition into disjoint sets
				List<List<int>> sets = new List<List<int>>(); // each set is indices into chunk
				for (int i = 0; i < chunkSize; i++)
				{
					bool foundSet = false;
					foreach (List<int> set in sets)
					{
						bool overlaps = false;
						foreach (int j in set)
						{
							if (boxes[i].DoesOverlapBox(boxes[j]))
							{
								overlaps = true;
								break;
							}
						}

						if (!overlaps)
						{
							set.Add(i);
							foundSet = true;
							break;
						}
					}

					if (!foundSet)
					{
						sets.Add(new List<int> { i });
					}
				}

				// Process each disjoint set
				List<CsgLeafNode> chunk = children.GetRange(chunkStart, chunkSize);
				children.RemoveRange(chunkStart, chunkSize);
				List<CsgLeafNode> results = new List<CsgLeafNode>();

				foreach (List<int> set in sets)
				{
					if (set.Count == 1)
					{
						results.Add(chunk[set[0]]);
					}
					else
					{
						// Compose disjoint meshes without boolean
						ManifoldImpl[] meshes = new ManifoldImpl[set.Count];
						for (int i = 0; i < set.Count; i++)
						{
							meshes[i] = chunk[set[i]].GetImpl();
						}

						ManifoldImpl composed = Boolean3Functions.ComposeMeshes(meshes);
						results.Add(new CsgLeafNode(composed));
					}
				}

				// BatchBoolean the composed results, then move the (complicated) new
				// child to the front: C++ push_backs and swaps front↔back, which also
				// moves the old front to the back when chunking (>kMaxUnionSize).
				CsgLeafNode result = BatchBoolean(OpType.Add, results, token);
				children.Add(result);
				int last = children.Count - 1;
				(children[0], children[last]) = (children[last], children[0]);
			}

			CsgLeafNode reduced = children[0];
			children.RemoveAt(0);
			return reduced;
		}

		/// <summary>
		/// Heap key ordered like C++ <c>MeshCompare</c> on <c>(CsgLeafNode, serial)</c>
		/// pairs: by vertex count, tie-broken by insertion serial.
		/// </summary>
		/// <remarks>
		/// The serial makes the order total, so the pop sequence is deterministic and
		/// heap-implementation independent — required for exact match with the C++
		/// reduction order. It is not decoration: <see cref="PriorityQueue{TElement,TPriority}"/>
		/// is explicitly NOT stable for equal priorities, so without the serial two
		/// same-vertex-count meshes would pair up in an unspecified order.
		/// <para>
		/// Vertex count is captured at enqueue rather than recomputed in the comparer (as
		/// the Rust's <c>Ord</c> does), because a priority that changed while its element sat
		/// in the heap would corrupt the heap invariant rather than merely reorder it. That
		/// is safe, not impossible: <see cref="CsgLeafNode"/>'s own fields are readonly, but
		/// <see cref="CsgLeafNode.PImpl"/> is a shared, mutable <see cref="ManifoldImpl"/> —
		/// C# cannot reproduce the Rust's move-into-<c>Arc</c>, which makes the mesh
		/// genuinely unreachable for mutation. The guarantee is therefore behavioural: no
		/// path in the port mutates a leaf's mesh after it is enqueued. The one way in is a
		/// caller that keeps its own reference to a mesh it handed to a leaf and edits it
		/// mid-evaluation — the hazard the <see cref="CsgLeafNode"/> intake doc already
		/// names, whose blast radius includes this heap.
		/// </para>
		/// </remarks>
		private readonly struct MeshEntryKey
		{
			/// <summary>
			/// The comparer the heap runs with — <see cref="Compare"/> INVERTED.
			/// </summary>
			/// <remarks>
			/// C++ <c>std::pop_heap</c> with <c>MeshCompare</c> (a less-than) pops the MAX:
			/// the node with the most verts, ties going to the largest serial. Rust's
			/// <c>BinaryHeap</c> is a max-heap and uses that less-than directly. C#'s
			/// <see cref="PriorityQueue{TElement,TPriority}"/> is a MIN-queue, so the same
			/// pop order requires feeding it the reversed comparison — hence the swapped
			/// operands below, and nothing else about the order changes.
			/// </remarks>
			internal static readonly IComparer<MeshEntryKey> PopMostVertsFirst =
				Comparer<MeshEntryKey>.Create((x, y) => Compare(y, x));

			private readonly int vertCount;
			private readonly ulong serial;

			internal MeshEntryKey(int vertCount, ulong serial)
			{
				this.vertCount = vertCount;
				this.serial = serial;
			}

			/// <summary>The C++ <c>MeshCompare</c> less-than, as a three-way compare.</summary>
			private static int Compare(MeshEntryKey x, MeshEntryKey y)
			{
				int byVerts = x.vertCount.CompareTo(y.vertCount);
				if (byVerts != 0)
				{
					return byVerts;
				}

				return x.serial.CompareTo(y.serial);
			}
		}
	}
}
