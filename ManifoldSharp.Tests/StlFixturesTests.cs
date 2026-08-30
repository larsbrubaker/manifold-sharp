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

// StlFixturesTests.cs — one adaptation test for StlFixtures.cs, which has no
// counterpart in the Rust suite because the thing it guards does not exist
// there: the Rust reads its fixtures with `include_bytes!` (a compile error if
// a file is missing), while C# copies them beside the test assembly at build
// time and reads them at run time. A mis-globbed <Content> item, an
// ASCII/binary sniff that went the wrong way, or a normalization that lost its
// f32 storage would all surface here as a wrong triangle count rather than as
// twenty confusing failures in a later wave.
//
// The expected counts are the manifold-rust values, taken from the differential
// harness this port was written against (soup import + the demo pipeline over
// all 20 fixtures, compared bit-for-bit including welded positions, halfedges
// and face normals — zero diffs). Fifteen of the twenty parse as binary and
// five as ASCII, so both parsers are exercised.

using ManifoldSharp;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class StlFixturesTests
	{
		[Test]
		public async Task Thingi10kFixturesImportWithTheRustsCounts()
		{
			(string File, int NumVert, int NumTri)[] expected = new[]
			{
				("1075458.stl", 6396, 12820),
				("1147177.stl", 3201, 6418),
				("1313535.stl", 658, 1332),
				("1663774.stl", 6551, 13122),
				("301921.stl", 584, 1160),
				("36088.stl", 228, 440),
				("36374.stl", 760, 1544),
				("39926.stl", 414, 836),
				("51334.stl", 7200, 15012),
				("51360.stl", 3006, 5988),
				("59082.stl", 4869, 9722),
				("61459.stl", 92, 176),
				("68730.stl", 578, 1156),
				("74660.stl", 1798, 3636),
				("90225.stl", 86, 168),
				("91115.stl", 5589, 10954),
				("91946.stl", 2232, 5364),
				("92068.stl", 2046, 4068),
				("93557.stl", 500, 772),
				("939888.stl", 860, 1716),
			};

			foreach ((string file, int numVert, int numTri) in expected)
			{
				Manifold m = StlFixtures.ImportStlLikeDemo(StlFixtures.FixtureBytes(file));
				await Assert.That(m.Status()).IsEqualTo(Error.NoError).Because(file);
				await Assert.That(m.NumVert()).IsEqualTo(numVert).Because(file);
				await Assert.That(m.NumTri()).IsEqualTo(numTri).Because(file);
			}
		}
	}
}
