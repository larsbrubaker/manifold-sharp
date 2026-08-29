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

// Port of the tests module in tree2d.rs — both cases, same points in the same
// order, same expected results.

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

using ManifoldSharp.Linalg;

namespace ManifoldSharp.Tests
{
	public class Tree2dTests
	{
		[Test]
		public async Task BuildTwoDTreeAndQuery()
		{
			PolyVert[] points = new PolyVert[]
			{
				new PolyVert(new Vec2(0.0, 0.0), 0),
				new PolyVert(new Vec2(1.0, 1.0), 1),
				new PolyVert(new Vec2(2.0, 2.0), 2),
				new PolyVert(new Vec2(3.0, 3.0), 3),
				new PolyVert(new Vec2(4.0, 4.0), 4),
				new PolyVert(new Vec2(5.0, 5.0), 5),
				new PolyVert(new Vec2(6.0, 6.0), 6),
				new PolyVert(new Vec2(7.0, 7.0), 7),
				new PolyVert(new Vec2(8.0, 8.0), 8),
			};

			Tree2d.BuildTwoDTree(points);
			Rect rect = Rect.FromPoints(new Vec2(1.5, 1.5), new Vec2(6.5, 6.5));
			List<int> outIdx = new List<int>();
			Tree2d.QueryTwoDTree(points, rect, p => outIdx.Add(p.Idx));

			// The Rust sorts here with sort_unstable, one of the audited sites: the
			// query's visit order is not what is under test, and the indices are
			// distinct so no tie can expose the instability. List<T>.Sort is the same
			// unstable introsort.
			outIdx.Sort();
			await Assert.That(outIdx).IsEquivalentTo(new List<int> { 2, 3, 4, 5, 6 });
		}

		[Test]
		public async Task QueryTwoDTreeSmallInput()
		{
			PolyVert[] points = new PolyVert[]
			{
				new PolyVert(new Vec2(0.0, 0.0), 0),
				new PolyVert(new Vec2(2.0, 2.0), 1),
			};

			Rect rect = Rect.FromPoints(new Vec2(-1.0, -1.0), new Vec2(1.0, 1.0));
			List<int> outIdx = new List<int>();
			Tree2d.QueryTwoDTree(points, rect, p => outIdx.Add(p.Idx));

			await Assert.That(outIdx).IsEquivalentTo(new List<int> { 0 });
		}
	}
}
