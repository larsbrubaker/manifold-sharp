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

// Port of disjoint_sets.rs (itself a port of disjoint_sets.h) — Union-Find data
// structure with path compression and union-by-rank, plus connected component
// computation.
//
// The C++ version uses atomic operations for thread-safety. This sequential
// port uses the same bit-packing layout: the upper 32 bits of each ulong
// store the rank, and the lower 32 bits store the parent index.

namespace ManifoldSharp
{
	/// <summary>
	/// Union-Find over <c>[0, size)</c> with path compression and union by rank.
	/// </summary>
	/// <remarks>
	/// Not thread-safe, and unlike the Rust source nothing here says so: its
	/// <c>Cell&lt;u64&gt;</c> made the type <c>!Sync</c>, so sharing an instance across
	/// threads was a compile error, whereas in C# it merely races. <see cref="Find"/>
	/// mutates during what reads like a read (path splitting), and the packed layout
	/// is inherited from the C++ <em>atomic</em> variant - which makes "it was built
	/// for concurrency" exactly the wrong conclusion to draw from the shape of it. A
	/// future parallel phase must not share an instance across threads without
	/// revisiting this.
	/// </remarks>
	public sealed class DisjointSets
	{
		// One entry per element: rank in the high 32 bits, parent index in the low
		// 32. The Rust port wraps each entry in a Cell<u64> only so that find can
		// compress paths through a &self borrow; C# needs no such wrapper, and the
		// packed layout is what actually matters here.
		private readonly ulong[] data;

		/// <summary>
		/// Creates <paramref name="size"/> singleton sets, each its own parent at rank 0.
		/// </summary>
		public DisjointSets(uint size)
		{
			this.data = new ulong[size];
			for (uint i = 0; i < size; i++)
			{
				this.data[i] = i;
			}
		}

		/// <summary>
		/// Returns the root of the set containing <paramref name="id"/>, compressing
		/// the path walked on the way up.
		/// </summary>
		public uint Find(uint id)
		{
			while (true)
			{
				uint p = this.Parent(id);
				if (p == id)
				{
					return id;
				}

				// Path splitting: point to grandparent (matches C++ path compression)
				uint gp = this.Parent(p);
				if (gp != p)
				{
					ulong value = this.data[id];
					ulong newValue = (value & 0xFFFFFFFF00000000ul) | gp;
					this.data[id] = newValue;
				}

				id = gp;
			}
		}

		/// <summary>
		/// Whether the two elements are in the same set.
		/// </summary>
		public bool Same(uint id1, uint id2)
		{
			while (true)
			{
				id1 = this.Find(id1);
				id2 = this.Find(id2);
				if (id1 == id2)
				{
					return true;
				}

				if (this.Parent(id1) == id1)
				{
					return false;
				}
			}
		}

		/// <summary>
		/// Merges the two sets and returns the root of the merged set.
		/// </summary>
		public uint Unite(uint id1, uint id2)
		{
			while (true)
			{
				id1 = this.Find(id1);
				id2 = this.Find(id2);

				if (id1 == id2)
				{
					return id1;
				}

				uint r1 = this.Rank(id1);
				uint r2 = this.Rank(id2);

				// Ensure id1 has lower rank (or lower index if equal)
				if (r1 > r2 || (r1 == r2 && id1 < id2))
				{
					(r1, r2) = (r2, r1);
					(id1, id2) = (id2, id1);
				}

				// Point id1 to id2
				ulong oldEntry = ((ulong)r1 << 32) | id1;
				ulong newEntry = ((ulong)r1 << 32) | id2;

				if (this.data[id1] != oldEntry)
				{
					continue;
				}

				this.data[id1] = newEntry;

				if (r1 == r2)
				{
					ulong oldEntry2 = ((ulong)r2 << 32) | id2;
					ulong newEntry2 = ((ulong)(r2 + 1) << 32) | id2;
					if (this.data[id2] == oldEntry2)
					{
						this.data[id2] = newEntry2;
					}
					else if (r2 == 0)
					{
						continue;
					}
				}

				break;
			}

			return id2;
		}

		/// <summary>
		/// The number of elements the structure was created with.
		/// </summary>
		public uint Size()
		{
			return (uint)this.data.Length;
		}

		/// <summary>
		/// The rank stored for <paramref name="id"/> — the high 32 bits of its entry,
		/// with the sign bit masked off.
		/// </summary>
		public uint Rank(uint id)
		{
			return (uint)(this.data[id] >> 32) & 0x7FFFFFFFu;
		}

		/// <summary>
		/// The parent stored for <paramref name="id"/> — the low 32 bits of its entry.
		/// This is one step up the tree, not necessarily the root.
		/// </summary>
		public uint Parent(uint id)
		{
			return (uint)this.data[id];
		}

		/// <summary>
		/// Labels every element with its component index, writing one label per
		/// element into <paramref name="components"/> (resized to fit), and returns
		/// the number of components.
		/// </summary>
		public int ConnectedComponents(List<int> components)
		{
			Resize(components, this.data.Length);
			int lonelyNodes = 0;
			Dictionary<uint, int> toLabel = new Dictionary<uint, int>();

			for (int i = 0; i < this.data.Length; i++)
			{
				uint iParent = this.Find((uint)i);

				// Optimize for connected components of size 1
				if (this.Rank(iParent) == 0)
				{
					components[i] = toLabel.Count + lonelyNodes;
					lonelyNodes += 1;
					continue;
				}

				if (toLabel.TryGetValue(iParent, out int label))
				{
					components[i] = label;
				}
				else
				{
					int s = toLabel.Count + lonelyNodes;
					toLabel.Add(iParent, s);
					components[i] = s;
				}
			}

			return toLabel.Count + lonelyNodes;
		}

		// Rust's Vec::resize, which the caller's buffer goes through before labelling:
		// grow with zeros, truncate when too long. Every slot is overwritten by the
		// loop above, so the fill value never reaches the output.
		private static void Resize(List<int> values, int length)
		{
			if (values.Count > length)
			{
				values.RemoveRange(length, values.Count - length);
			}

			while (values.Count < length)
			{
				values.Add(0);
			}
		}
	}
}
