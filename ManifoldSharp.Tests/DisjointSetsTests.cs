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

// Port of the tests module in disjoint_sets.rs — same cases, same expected
// values, in the same order.

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class DisjointSetsTests
	{
		[Test]
		public async Task DisjointSetsBasic()
		{
			DisjointSets ds = new DisjointSets(10);
			await Assert.That(ds.Size()).IsEqualTo(10u);
			await Assert.That(ds.Same(0, 1)).IsFalse();
			ds.Unite(0, 1);
			await Assert.That(ds.Same(0, 1)).IsTrue();
			await Assert.That(ds.Same(0, 2)).IsFalse();
		}

		[Test]
		public async Task DisjointSetsChain()
		{
			DisjointSets ds = new DisjointSets(5);
			ds.Unite(0, 1);
			ds.Unite(1, 2);
			ds.Unite(2, 3);
			await Assert.That(ds.Same(0, 3)).IsTrue();
			await Assert.That(ds.Same(0, 4)).IsFalse();
		}

		[Test]
		public async Task ConnectedComponents()
		{
			DisjointSets ds = new DisjointSets(6);
			ds.Unite(0, 1);
			ds.Unite(1, 2);
			ds.Unite(3, 4);

			// Groups: {0,1,2}, {3,4}, {5}
			List<int> components = new List<int>();
			int num = ds.ConnectedComponents(components);
			await Assert.That(num).IsEqualTo(3);

			// Members of the same group should have the same component id
			await Assert.That(components[0]).IsEqualTo(components[1]);
			await Assert.That(components[1]).IsEqualTo(components[2]);
			await Assert.That(components[3]).IsEqualTo(components[4]);
			await Assert.That(components[0]).IsNotEqualTo(components[3]);
			await Assert.That(components[0]).IsNotEqualTo(components[5]);
			await Assert.That(components[3]).IsNotEqualTo(components[5]);
		}

		[Test]
		public async Task UniteReturnsRoot()
		{
			DisjointSets ds = new DisjointSets(4);
			uint root = ds.Unite(0, 1);

			// The root should be one of the two
			await Assert.That(root == 0 || root == 1).IsTrue();

			// Find should return the same root
			await Assert.That(ds.Find(0)).IsEqualTo(ds.Find(1));
		}

		[Test]
		public async Task SingleElement()
		{
			DisjointSets ds = new DisjointSets(1);
			await Assert.That(ds.Find(0)).IsEqualTo(0u);

			List<int> components = new List<int>();
			int num = ds.ConnectedComponents(components);
			await Assert.That(num).IsEqualTo(1);
		}
	}
}
