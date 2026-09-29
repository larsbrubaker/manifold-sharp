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

// The tail of cross_section_tests.rs, split from CrossSectionTests.cs to keep
// both files under the 800-line cap: the cases that pin C++ Hull's degenerate
// output and the counts that see it. C++ Hull returns one contour even when it
// is degenerate (empty for fewer than three points, two vertices for collinear
// ones), and IsEmpty / NumContour count such contours, as C++ paths_.empty() /
// paths_.size() do. Same inputs, same expected values, same order as the Rust.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public partial class CrossSectionTests
	{
		/// <summary>
		/// C++ <c>IsEmpty</c> is <c>paths_.empty()</c> and <c>NumContour</c> is
		/// <c>paths_.size()</c>: contours with fewer than three vertices still count. C++
		/// <c>Hull</c> produces exactly such sections — one empty contour for fewer than
		/// three points (<c>h_2pts</c>: contours=1 nvert=0 empty=0) and a two-vertex
		/// contour for collinear points (<c>h_collinear</c>: contours=1 nvert=2 empty=0) —
		/// through the private no-union constructor that <c>FromRaw</c> mirrors.
		/// </summary>
		[Test]
		public async Task IsEmptyAndNumContourCountEveryPathLikeCpp()
		{
			CrossSection oneEmpty = CrossSection.FromRaw(new Polygons { new SimplePolygon() });
			await Assert.That(oneEmpty.IsEmpty()).IsFalse();
			await Assert.That(oneEmpty.NumContour()).IsEqualTo(1);
			await Assert.That(oneEmpty.NumVert()).IsEqualTo(0);
			CrossSection degenerate = CrossSection.FromRaw(new Polygons
			{
				new SimplePolygon(),
				new SimplePolygon { new Vec2(0.0, 0.0), new Vec2(3.0, 0.0) },
			});
			await Assert.That(degenerate.IsEmpty()).IsFalse();
			await Assert.That(degenerate.NumContour()).IsEqualTo(2);
			await Assert.That(degenerate.NumVert()).IsEqualTo(2);
			CrossSection none = new CrossSection();
			await Assert.That(none.IsEmpty()).IsTrue();
			await Assert.That(none.NumContour()).IsEqualTo(0);
			await Assert.That(none.NumVert()).IsEqualTo(0);
		}
	}
}
