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

// Port of the tests module in smoothing.rs (smoothing_tests.rs) — the same 4
// cases, same inputs, same assertions, in the same order. Nothing here is
// deferred: all four run against ManifoldImpl directly, as the Rust's do.
//
// Four tests is thin cover for 1,100 lines of Rust, and that is the Rust's
// state too — smoothing's real coverage lives in the manifold_tests suite that
// drives Manifold::Smooth / SmoothByNormals / SmoothOut / Refine. Those façade
// methods are manifold_smooth.rs, which has since landed as Manifold.Smooth.cs,
// so that coverage is now checked in: ManifoldSmoothTests(.Relations).cs ports
// manifold_tests/smooth.rs and ManifoldNormalsTests.cs ports
// manifold_tests/normals.rs. The methods this port adds that no test *below*
// reaches, and which of those files reaches them:
//
//   SetNormals             manifold_smooth.rs's calculate_normals /
//   UpdateSharpenedEdges   set_properties path, and manifold.rs's smooth entry
//   ValidTangents          points — ManifoldNormalsTests.cs and
//   FlatFaces              ManifoldSmoothTests.cs respectively.
//   VertFlatFace           (FlatFaces/VertFlatFace are also exercised
//                          indirectly here: CreateTangents calls FlatFaces,
//                          VertFlatFace and LinearizeFlatTangents on the cube,
//                          whose six faces are all flat.)

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class SmoothingTests
	{
		[Test]
		public async Task CircularTangentIsFinite()
		{
			Vec4 t = Smoothing.CircularTangent(new Vec3(0.0, 1.0, 0.0), new Vec3(1.0, 0.0, 0.0));
			await Assert.That(double.IsFinite(t.X)).IsTrue();
			await Assert.That(double.IsFinite(t.Y)).IsTrue();
			await Assert.That(double.IsFinite(t.Z)).IsTrue();
			await Assert.That(double.IsFinite(t.W)).IsTrue();
			await Assert.That(t.W > 0.0).IsTrue();
		}

		[Test]
		public async Task SharpenEdgesCube()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			List<Smoothness> edges = m.SharpenEdges(45.0, 0.0);
			await Assert.That(edges.Count).IsEqualTo(24);
		}

		[Test]
		public async Task CreateTangentsCube()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			List<Smoothness> sharp = m.SharpenEdges(45.0, 0.0);
			m.CreateTangents(sharp);
			await Assert.That(m.HalfedgeTangent.Count).IsEqualTo(m.NumHalfedge());
			await Assert.That(m.HalfedgeTangent.All(t =>
				double.IsFinite(t.X) && double.IsFinite(t.Y) && double.IsFinite(t.Z) && double.IsFinite(t.W)))
				.IsTrue();
			await Assert.That(m.HalfedgeTangent.Any(t => t.W < 0.0)).IsTrue();
		}

		[Test]
		public async Task CreateTangentsFromNormalsCube()
		{
			ManifoldImpl m = ManifoldImpl.Cube(Mat3x4.Identity());
			m.NumProp = 3;
			m.Properties = m.VertNormal.SelectMany(n => new double[] { n.X, n.Y, n.Z }).ToList();
			m.CreateTangentsFromNormals(0);
			await Assert.That(m.HalfedgeTangent.Count).IsEqualTo(m.NumHalfedge());
			await Assert.That(m.HalfedgeTangent.All(t => double.IsFinite(t.W))).IsTrue();
		}
	}
}
