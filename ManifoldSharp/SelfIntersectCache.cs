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

// SelfIntersectCache.cs — the one type lifted out of robust/soup.rs ahead of its
// phase, because ManifoldImpl holds it as a field and impl_mesh.rs could not be
// ported without it. The rest of robust/soup.rs has since landed: the detector
// that fills this cell is Robust.Soup.HasSelfIntersections, and it is what
// BooleanEngine.Auto consults to pick an engine.
//
// The Rust is `pub struct SelfIntersectCache(std::sync::OnceLock<bool>)` with a
// hand-written Clone. .NET has no OnceLock, so the cell is a single reference
// field settled with Interlocked.CompareExchange: the first writer wins, later
// writers are ignored, and readers never tear. That is exactly OnceLock's
// contract for a Copy payload.

using System.Threading;

namespace ManifoldSharp
{
	/// <summary>
	/// A write-once, thread-safe cell holding the verdict of
	/// <c>robust::soup::has_self_intersections</c> for one
	/// <see cref="ManifoldImpl"/>.
	/// </summary>
	/// <remarks>
	/// The detector's scan is a full BVH self-query, so it is computed at most once
	/// per impl and consulted by <c>Auto</c> boolean dispatch, which must route
	/// geometrically self-intersecting operands to the robust engine even when their
	/// connectivity is manifold.
	/// <para>
	/// <b>The trap this type sets.</b> Settled through a shared reference (in the Rust,
	/// from rayon workers under the <c>parallel</c> feature), and
	/// <see cref="Clone"/> copies the settled value across — so <b>any</b> code that
	/// clones an impl and then edits its geometry in place must call
	/// <see cref="ManifoldImpl.InvalidateSelfIntersects"/>. Rebuilds that go through
	/// <see cref="ManifoldImpl.CreateHalfedges"/> or <see cref="ManifoldImpl.MakeEmpty"/>
	/// are covered automatically. The type is <c>internal</c> deliberately, matching the
	/// Rust field's <c>pub(crate)</c>, so the only way to reach it is from inside this
	/// assembly where that rule is reviewable.
	/// </para>
	/// </remarks>
	internal sealed class SelfIntersectCache
	{
		// null = never settled. Otherwise a boxed bool, written exactly once. A
		// reference field is the smallest thing CompareExchange can settle
		// atomically while still distinguishing "unset" from "set to false".
		private object? settled;

		/// <summary>The settled verdict, or null if the detector has not run yet.</summary>
		/// <returns>The verdict, or null when the cell is empty.</returns>
		public bool? Get()
		{
			object? value = Volatile.Read(ref this.settled);
			return value is null ? null : (bool)value;
		}

		/// <summary>
		/// Seed an already-known verdict (used when a transform carries the answer
		/// forward). No-op once the cell is settled.
		/// </summary>
		/// <param name="value">The verdict to record.</param>
		public void Set(bool value)
		{
			Interlocked.CompareExchange(ref this.settled, value, null);
		}

		/// <summary>
		/// The Rust's hand-written <c>Clone</c>: a fresh cell carrying this one's settled
		/// value, if it has one.
		/// </summary>
		/// <returns>An independent cell with the same verdict.</returns>
		public SelfIntersectCache Clone()
		{
			SelfIntersectCache copy = new SelfIntersectCache();
			bool? value = this.Get();
			if (value.HasValue)
			{
				copy.Set(value.Value);
			}

			return copy;
		}
	}
}
