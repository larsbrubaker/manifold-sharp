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

// Sdf.HashTable.cs — the GridVert record and the port of C++ hashtable.h that
// sdf.rs carries inline. Split out of Sdf.cs only to stay under the 800-line
// cap; in the Rust this is one file.
//
// This is the flagship invariant of the SDF module. From sdf.rs's header:
//
//   A faithful port of the C++ open-addressing HashTable (hashtable.h):
//   ComputeVerts/BuildTris iterate the table in SLOT order, so the hash
//   function, probing, sizing and resize protocol all determine the output
//   vertex numbering and triangle order. A std HashMap iterates in a random
//   order per process, which made the output nondeterministic.
//
// The C# reading of that sentence, spelled out because the substitution is so
// tempting: `Dictionary<ulong, GridVert>` is NOT a replacement here, and
// neither is any other associative container. Phases 2 and 3 of LevelSet walk
// `slot = 0 .. Size` and skip the open slots, so which slot a key lands in *is*
// the output vertex numbering. That slot is decided by three things, all of
// which are ported here exactly: the splitmix64 finalizer in Hash64Bit, linear
// probing with step 1, and the power-of-two capacity. Change any one and the
// mesh changes — the Rust discovered this as three different genus values from
// three runs of the same input, because std's HashMap seeds itself per process.
//
// Two further C++ behaviors that reach the output and are therefore ported
// rather than "cleaned up":
//   - Full() is `used * 2 > size`, checked *inside* the insert loop, so a table
//     that fills mid-pass abandons the insert and the caller reruns the whole
//     NearSurface pass against a bigger table.
//   - A lookup of an absent key returns the default-initialized GridVert that
//     lives in the first open slot on its probe chain, rather than signalling
//     "missing" — that is C++ `operator[]` semantics, and ComputeVerts/BuildTris
//     both rely on getting a NaN-distance, no-edge-verts GridVert back.

using System.Numerics;
using System.Runtime.CompilerServices;

namespace ManifoldSharp
{
	/// <summary>
	/// Seven <see cref="int"/>s stored inline in <see cref="GridVert"/>, the port of
	/// the Rust <c>[i32; 7]</c> field. An inline array (not <c>int[]</c>) because the
	/// Rust type has value semantics: the table stores GridVerts by value and the
	/// marching phases copy them freely, so a reference-typed array field would
	/// silently alias table entries that the Rust keeps independent.
	/// </summary>
	[InlineArray(7)]
	internal struct EdgeVertArray
	{
		private int element0;
	}

	/// <summary>
	/// A grid vertex near the surface: its signed distance, the output vertex it
	/// snapped to (or <see cref="Sdf.KNone"/>), and the output vertex on each of its
	/// seven forward BCC edges.
	/// </summary>
	internal struct GridVert
	{
		/// <summary>Signed distance at this grid point; NaN in the default value.</summary>
		public double Distance;

		/// <summary>Output vertex this grid vert snapped to, or <see cref="Sdf.KNone"/>.</summary>
		public int MovedVert;

		/// <summary>Output vertex on each of the seven forward edges, or <see cref="Sdf.KNone"/>.</summary>
		public EdgeVertArray EdgeVerts;

		/// <summary>
		/// The Rust <c>impl Default for GridVert</c>. NaN distance is what marks an entry
		/// as "not a real grid vert" for ComputeVerts, which tests
		/// <c>nv.distance.is_finite()</c>; it must be the positive quiet NaN Rust's
		/// <c>f64::NAN</c> is, per the porting rules.
		/// </summary>
		/// <returns>A default GridVert.</returns>
		public static GridVert Default()
		{
			GridVert v = default;
			v.Distance = DeterministicMath.PositiveQuietNaN;
			v.MovedVert = Sdf.KNone;
			for (int i = 0; i < 7; i++)
			{
				v.EdgeVerts[i] = Sdf.KNone;
			}

			return v;
		}

		/// <summary>Whether this grid vert was snapped onto the surface.</summary>
		/// <returns>True when a moved vertex was allocated.</returns>
		public readonly bool HasMoved()
		{
			return this.MovedVert >= 0;
		}

		/// <summary>Whether <paramref name="dist"/> has the same sign as this vert's distance.</summary>
		/// <param name="dist">The distance to compare against.</param>
		/// <returns>True when both are positive or both are not.</returns>
		public readonly bool SameSide(double dist)
		{
			return (dist > 0.0) == (this.Distance > 0.0);
		}

		/// <summary>+1 when this grid vert is inside the surface, -1 when outside.</summary>
		/// <returns>The inside sign.</returns>
		public readonly int Inside()
		{
			return this.Distance > 0.0 ? 1 : -1;
		}

		/// <summary>The inside sign of the neighbor across forward edge <paramref name="i"/>.</summary>
		/// <param name="i">The forward edge index, 0..6.</param>
		/// <returns>The neighbor's inside sign, inferred from whether the edge is crossed.</returns>
		public readonly int NeighborInside(int i)
		{
			return this.Inside() * (this.EdgeVerts[i] == Sdf.KNone ? 1 : -1);
		}
	}

	/// <summary>
	/// Open-addressing hash table matching C++ <c>HashTable&lt;GridVert&gt;</c>: capacity is
	/// the next power of two, probing is linear (step 1), <c>Full()</c> is used*2 &gt;
	/// size, and lookups of absent keys return the default value sitting in the
	/// first open slot — all of which the marching phases depend on for their
	/// SLOT-ORDER iteration.
	/// </summary>
	internal sealed class GridHashTable
	{
		/// <summary>The sentinel stored in an unoccupied slot (C++ <c>kOpen</c>).</summary>
		public const ulong KOpen = ulong.MaxValue;

		/// <summary>The key of each slot, <see cref="KOpen"/> when unoccupied.</summary>
		public readonly ulong[] Keys;

		/// <summary>The value of each slot; open slots hold the default GridVert.</summary>
		public readonly GridVert[] Values;

		private int used;

		/// <summary>
		/// Allocates a table of the next power of two at or above <paramref name="size"/>.
		/// </summary>
		/// <param name="size">The requested minimum capacity.</param>
		public GridHashTable(ulong size)
		{
			ulong n = size == 0 ? 0 : 1UL << Sdf.CeilLog2(size);

			// The Rust indexes with usize, C# arrays with int. The value is bounded well
			// under int range by LevelSet's own sizing (the sparse heuristic caps it at
			// 10*sqrt(maxIndex), and maxIndex must already fit a double[] there), so
			// `checked` here is the C# stand-in for the Rust's allocation failure rather
			// than a reachable path.
			int count = checked((int)n);
			this.Keys = new ulong[count];
			this.Values = new GridVert[count];
			for (int i = 0; i < count; i++)
			{
				this.Keys[i] = KOpen;
				this.Values[i] = GridVert.Default();
			}

			this.used = 0;
		}

		/// <summary>The slot count, always a power of two.</summary>
		public int Size
		{
			[MethodImpl(MethodImplOptions.AggressiveInlining)]
			get { return this.Keys.Length; }
		}

		/// <summary>
		/// C++ <c>Full()</c>: more than half the slots used. The load factor is part of the
		/// output contract, not a tuning knob — it decides when LevelSet abandons a pass
		/// and reruns it against a bigger table, which renumbers every output vertex.
		/// </summary>
		/// <returns>True when the table is over half full.</returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public bool Full()
		{
			// Rust does this in usize; `used * 2` in int could overflow on a table larger
			// than 2^30 slots, so the comparison widens.
			return (long)this.used * 2 > this.Size;
		}

		/// <summary>
		/// Inserts <paramref name="val"/> under <paramref name="key"/>, doing nothing if the
		/// key is present or the table has become full mid-probe (the caller reruns).
		/// </summary>
		/// <param name="key">The encoded grid index.</param>
		/// <param name="val">The value to store.</param>
		public void Insert(ulong key, GridVert val)
		{
			ulong mask = (ulong)(this.Size - 1);
			int idx = (int)(Sdf.Hash64Bit(key) & mask);
			while (true)
			{
				if (this.Full())
				{
					return;
				}

				ulong k = this.Keys[idx];
				if (k == KOpen)
				{
					this.Keys[idx] = key;
					this.used++;
					this.Values[idx] = val;
					return;
				}

				if (k == key)
				{
					return;
				}

				idx = (int)(((ulong)idx + 1) & mask);
			}
		}

		/// <summary>
		/// Probe for <paramref name="key"/>; stops at the key's slot or the first open slot
		/// (whose default-initialized value serves as the "missing" GridVert, matching
		/// C++ operator[] semantics).
		/// </summary>
		/// <param name="key">The encoded grid index.</param>
		/// <returns>The slot the probe stopped on.</returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public int SlotOf(ulong key)
		{
			ulong mask = (ulong)(this.Size - 1);
			int idx = (int)(Sdf.Hash64Bit(key) & mask);
			while (true)
			{
				ulong k = this.Keys[idx];
				if (k == key || k == KOpen)
				{
					return idx;
				}

				idx = (int)(((ulong)idx + 1) & mask);
			}
		}

		/// <summary>The value stored under <paramref name="key"/>, or the default GridVert.</summary>
		/// <param name="key">The encoded grid index.</param>
		/// <returns>The stored or default value.</returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public GridVert Get(ulong key)
		{
			return this.Values[this.SlotOf(key)];
		}
	}

	public static partial class Sdf
	{
		/// <summary>
		/// C++ <c>hash64bit</c> (utils.h) — splitmix64-style finalizer. The multiplies wrap
		/// (Rust <c>wrapping_mul</c>); C#'s default unchecked context is that behavior.
		/// </summary>
		/// <param name="x">The key to hash.</param>
		/// <returns>The hashed key.</returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		internal static ulong Hash64Bit(ulong x)
		{
			x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9UL;
			x = (x ^ (x >> 27)) * 0x94d049bb133111ebUL;
			return x ^ (x >> 31);
		}

		/// <summary>
		/// Rust <c>ceil_log2_usize</c>: the smallest <c>k</c> with <c>2^k &gt;= v</c>. Used both for
		/// table capacity and — via ComputeGridPow — for the index bit widths, so it must
		/// be exact rather than merely monotonic.
		/// </summary>
		/// <param name="v">The value to take the ceiling log2 of.</param>
		/// <returns>The exponent, 0 for <paramref name="v"/> of 0 or 1.</returns>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		internal static int CeilLog2(ulong v)
		{
			if (v <= 1)
			{
				return 0;
			}

			// Rust: usize::BITS - (v - 1).leading_zeros(), with usize 64-bit.
			return 64 - BitOperations.LeadingZeroCount(v - 1);
		}
	}
}
