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

// RustVec.cs — NOT PORTED FROM RUST. The one `Vec<T>` operation the BCL has no
// equivalent for: `Vec::resize(new_len, value)`, which truncates when shrinking
// and pads with `value` when growing.
//
// `List<T>` has no Resize at all, and `Array.Resize` pads with `default(T)` —
// which is the wrong value at every Box site in this port (see the Bounds.cs
// header: the Rust default Box is inverted, `default(Box)` is a box at the
// origin). Both halves live here so the mesh-core files can transcribe
// `x.resize(n, v)` as `x.Resize(n, v)` and nothing has to remember which of the
// two behaviours the BCL happens to offer.

namespace ManifoldSharp
{
	/// <summary>
	/// The <c>Vec::resize</c> shim the port transcribes Rust resizes through.
	/// </summary>
	internal static class RustVec
	{
		/// <summary>
		/// Rust <c>Vec::resize</c>: truncate to <paramref name="newLength"/>, or grow to it
		/// by appending copies of <paramref name="value"/>.
		/// </summary>
		/// <typeparam name="T">The element type.</typeparam>
		/// <param name="list">The list to resize in place.</param>
		/// <param name="newLength">The length to resize to.</param>
		/// <param name="value">The value new elements take.</param>
		public static void Resize<T>(this List<T> list, int newLength, T value)
		{
			if (newLength < list.Count)
			{
				list.RemoveRange(newLength, list.Count - newLength);
				return;
			}

			// One allocation instead of a doubling chain. This matters more than it looks:
			// CreateHalfedges clears the halfedge list and resizes it to 3 * numTri on
			// every call, so without this each mesh build re-grows the arena from zero.
			list.EnsureCapacity(newLength);
			while (list.Count < newLength)
			{
				list.Add(value);
			}
		}

		/// <summary>
		/// Rust <c>Vec::resize</c> for an array held by reference — the arena form, used
		/// where the Bounds.cs header requires an array rather than a
		/// <see cref="List{T}"/> (a mutating method called through a
		/// <see cref="List{T}"/> indexer silently updates a copy).
		/// </summary>
		/// <typeparam name="T">The element type.</typeparam>
		/// <param name="array">The array to replace with the resized one.</param>
		/// <param name="newLength">The length to resize to.</param>
		/// <param name="value">The value new elements take.</param>
		public static void Resize<T>(ref T[] array, int newLength, T value)
		{
			int oldLength = array.Length;
			Array.Resize(ref array, newLength);

			// Array.Resize zero-fills; the Rust fills with `value`, which for Box is the
			// inverted (empty) box and not the zeroed one.
			for (int i = oldLength; i < newLength; i++)
			{
				array[i] = value;
			}
		}
	}
}
