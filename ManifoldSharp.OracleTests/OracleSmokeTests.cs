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

// The oracle lane's plumbing check. This assembly exists to compare ManifoldSharp
// against the native kernel through the ManifoldRust P/Invoke binding; this one
// test proves the package resolves, its native loads, and geometry actually comes
// back out - so a failure here is never mistaken for a real comparison failure in
// BooleanOracleTests.

using ManifoldRust;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.OracleTests
{
	public class OracleSmokeTests
	{
		[Test]
		public async Task OracleImportsACubeAndExportsItBack()
		{
			// Transliterated from the binding's own test meshes rather than generated,
			// for the reason that file gives: an oracle must not be fed by the thing it
			// is meant to check.
			float[] verts =
			{
				0, 0, 0,
				1, 0, 0,
				1, 1, 0,
				0, 1, 0,
				0, 0, 1,
				1, 0, 1,
				1, 1, 1,
				0, 1, 1,
			};

			uint[] tris =
			{
				0, 2, 1, 0, 3, 2, // -Z
				4, 5, 6, 4, 6, 7, // +Z
				0, 1, 5, 0, 5, 4, // -Y
				1, 2, 6, 1, 6, 5, // +X
				2, 3, 7, 2, 7, 6, // +Y
				3, 0, 4, 3, 4, 7, // -X
			};

			using ManifoldRust.Manifold cube = ManifoldRust.Manifold.FromMesh(verts, tris);

			await Assert.That(cube.Status).IsEqualTo(ManifoldStatus.NoError);

			// Fully qualified: this file lives in namespace ManifoldSharp.OracleTests, so
			// the port's own ManifoldSharp.MeshGL (Phase 3) resolves ahead of the
			// `using ManifoldRust`. Phase 6 added ManifoldSharp.Manifold and the same
			// shadowing now applies to the manifold type itself, hence the qualification
			// on `cube` above — which is exactly the pairing this lane exists to compare,
			// so the ambiguity is expected, not a naming mistake.
			ManifoldRust.MeshGL mesh = cube.GetMeshGL();
			await Assert.That(mesh.NumProp).IsEqualTo(3u);
			await Assert.That(mesh.TriangleCount).IsEqualTo(12);
			await Assert.That(mesh.VertexCount).IsEqualTo(8);
		}
	}
}
