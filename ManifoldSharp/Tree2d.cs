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

// Tree2d.cs — port of tree2d.rs.
//
// Phase 10: tree2d — ported from cpp-reference/manifold/src/tree2d.h/.cpp
//
// ── Two kd-trees, not one ────────────────────────────────────────────────────
// polygon.rs carries its own private copy of this pair of functions, and the
// ear-clip triangulator calls *that* one, not this module. The two build the
// same array but traverse it differently: this one is the literal transcription
// of the C++ iterative descent (walk left, stack the right sibling), while
// polygon.rs's is a plain pop-from-the-stack loop that pushes left then right
// and therefore visits right first. The set of points handed to the callback is
// the same; the *order* is not, so the two are not interchangeable anywhere the
// callback's order can reach the output.
//
// ── Slices ───────────────────────────────────────────────────────────────────
// The Rust builds in place on `&mut [PolyVert]` sub-slices, so the C# takes
// Span/ReadOnlySpan: an array converts implicitly, and a List<PolyVert> goes
// through CollectionsMarshal.AsSpan. No copy of the point array is made, which
// is the whole point — the build *is* the reordering.
//
// ── Stable sort ──────────────────────────────────────────────────────────────
// Rust's `sort_by` is stable and the tie order reaches the built tree (equal
// coordinates are the common case in a polygon, not the exotic one), so the
// sort here is LINQ OrderBy — documented stable — and never Span.Sort, which is
// an unstable introsort. The comparator is `partial_cmp(...).unwrap_or(Equal)`
// transcribed by hand rather than Comparer<double>.Default, which is a *total*
// order and disagrees with Rust twice: it separates -0.0 from 0.0, and it sorts
// NaN below everything instead of calling it equal.

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <summary>
	/// A 2D kd-tree stored as a permutation of the points themselves — no nodes, no
	/// links. <see cref="BuildTwoDTree"/> reorders the array so that each sub-range's
	/// middle element is its splitting plane, alternating x and y by depth, and
	/// <see cref="QueryTwoDTree"/> walks that implicit structure.
	/// </summary>
	public static class Tree2d
	{
		/// <summary>
		/// Reorders <paramref name="points"/> in place into a 2D kd-tree, splitting on x
		/// at the root. Ranges of 8 or fewer points are left alone: both the build and
		/// the query treat them as one leaf to scan linearly.
		/// </summary>
		/// <param name="points">The points to reorder; modified in place.</param>
		public static void BuildTwoDTree(Span<PolyVert> points)
		{
			if (points.Length <= 8)
			{
				return;
			}

			BuildTwoDTreeImpl(points, true);
		}

		/// <summary>
		/// Calls <paramref name="f"/> once for every point of the tree that lies inside
		/// <paramref name="rect"/>, skipping the sub-ranges whose bounds cannot overlap it.
		/// </summary>
		/// <param name="points">A range previously passed to <see cref="BuildTwoDTree"/>.</param>
		/// <param name="rect">The query rectangle; containment is inclusive of the edges.</param>
		/// <param name="f">The callback, invoked in traversal order.</param>
		public static void QueryTwoDTree(ReadOnlySpan<PolyVert> points, Rect rect, Action<PolyVert> f)
		{
			if (points.Length <= 8)
			{
				foreach (PolyVert p in points)
				{
					if (rect.ContainsPoint(p.Pos))
					{
						f(p);
					}
				}

				return;
			}

			Rect current = new Rect();
			current.Min = Vec2.Splat(double.NegativeInfinity);
			current.Max = Vec2.Splat(double.PositiveInfinity);

			int level = 0;
			(int Start, int End) currentRange = (0, points.Length);
			Stack<Rect> rectStack = new Stack<Rect>();
			Stack<(int Start, int End)> rangeStack = new Stack<(int Start, int End)>();
			Stack<int> levelStack = new Stack<int>();

			while (true)
			{
				ReadOnlySpan<PolyVert> currentView = points[currentRange.Start..currentRange.End];
				if (currentView.Length <= 8)
				{
					foreach (PolyVert p in currentView)
					{
						if (rect.ContainsPoint(p.Pos))
						{
							f(p);
						}
					}

					if (levelStack.TryPop(out int nextLevel))
					{
						level = nextLevel;
						currentRange = rangeStack.Pop();
						current = rectStack.Pop();
						continue;
					}

					break;
				}

				Rect left = current;
				Rect right = current;
				int mid = currentView.Length / 2;
				PolyVert middle = currentView[mid];
				if (level % 2 == 0)
				{
					left.Max.X = middle.Pos.X;
					right.Min.X = middle.Pos.X;
				}
				else
				{
					left.Max.Y = middle.Pos.Y;
					right.Min.Y = middle.Pos.Y;
				}

				if (rect.ContainsPoint(middle.Pos))
				{
					f(middle);
				}

				bool leftOverlaps = left.DoesOverlap(rect);
				bool rightOverlaps = right.DoesOverlap(rect);
				if (leftOverlaps)
				{
					if (rightOverlaps)
					{
						rectStack.Push(right);
						rangeStack.Push((currentRange.Start + mid + 1, currentRange.End));
						levelStack.Push(level + 1);
					}

					current = left;
					currentRange = (currentRange.Start, currentRange.Start + mid);
					level += 1;
				}
				else if (rightOverlaps)
				{
					current = right;
					currentRange = (currentRange.Start + mid + 1, currentRange.End);
					level += 1;
				}
				else if (levelStack.TryPop(out int nextLevel))
				{
					level = nextLevel;
					currentRange = rangeStack.Pop();
					current = rectStack.Pop();
				}
				else
				{
					break;
				}
			}
		}

		// Sorts the range on one axis, then recurses on the halves either side of its
		// middle element with the axis flipped. The sort comes before the length check,
		// exactly as in the Rust: a one-element range is already sorted, so the order is
		// only load-bearing for readers, not for the result.
		private static void BuildTwoDTreeImpl(Span<PolyVert> points, bool sortX)
		{
			if (sortX)
			{
				StableSortByKey(points, static p => p.Pos.X);
			}
			else
			{
				StableSortByKey(points, static p => p.Pos.Y);
			}

			if (points.Length < 2)
			{
				return;
			}

			int mid = points.Length / 2;
			BuildTwoDTreeImpl(points[..mid], !sortX);
			if (mid + 1 < points.Length)
			{
				BuildTwoDTreeImpl(points[(mid + 1)..], !sortX);
			}
		}

		// Rust `sort_by` on a slice: stable, in place. The copy out and back is what buys
		// the stability guarantee - Span.Sort would be an unstable introsort.
		private static void StableSortByKey(Span<PolyVert> points, Func<PolyVert, double> key)
		{
			if (points.Length < 2)
			{
				return;
			}

			PolyVert[] ordered = points.ToArray().OrderBy(key, PartialOrdComparer.Instance).ToArray();
			ordered.CopyTo(points);
		}

		/// <summary>
		/// f64's <c>partial_cmp(...).unwrap_or(Ordering::Equal)</c>: IEEE comparison, with
		/// every unordered pair (either operand NaN) reported equal.
		/// </summary>
		/// <remarks>
		/// Reporting NaN equal to everything is not a consistent ordering, in C# or in
		/// Rust; both leave the resulting permutation unspecified, so a NaN coordinate
		/// here is a caller bug rather than a divergence between the two ports.
		/// </remarks>
		private sealed class PartialOrdComparer : IComparer<double>
		{
			public static readonly PartialOrdComparer Instance = new PartialOrdComparer();

			public int Compare(double a, double b)
			{
				if (a < b)
				{
					return -1;
				}

				if (a > b)
				{
					return 1;
				}

				return 0;
			}
		}
	}
}
