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

// Port of src/manifold_tests/normals.rs — all 8 tests and the module's one
// helper, same inputs, same expected values, same tolerances, in the same order.
// Nothing deferred: every test here opens with Manifold::calculate_normals, so
// the whole module was blocked until manifold_smooth.rs landed as Manifold.Smooth.cs.
//
// The Rust file's own header:
//
//   Tests ported from C++ TEST(Manifold, Normals*) in manifold_test.cpp (#1718).
//   Verify that CalculateNormals records world-frame normals on the Manifold and
//   that GetMeshGL() auto-substitutes / round-trips them, including across
//   transforms and Boolean subtraction (cavity inversion).
//
// CountSphereNormalAlignment stays private to this class rather than joining
// ManifoldTestHelpers: normals.rs is its only caller in the whole Rust suite, and
// it is `fn` (module-private) there for exactly that reason.
//
// RotateBeforeCalc and RotateAfterCalc are the pair worth reading together —
// they assert the same thing (bad == 0) through two different mechanisms, which
// is the point of #1718. Before: SetNormals computes from already-rotated face
// normals and stores the result in the world frame. After: Transform
// eager-transforms the stored slot 0..2 so it tracks the new orientation. A port
// that got either half wrong passes one and fails the other.

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class ManifoldNormalsTests
	{
		/// <summary>
		/// C++ CalculateNormals() / CalculateNormals(0) default minSharpAngle is 52.5.
		/// </summary>
		private const double DefaultAngle = 52.5;

		/// <summary>
		/// C++ TEST(Manifold, NormalsCavity) — inner-sphere normals from a Boolean diff
		/// should point toward the origin (outward from the surrounding solid).
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNormalsCavity()
		{
			MeshGL mesh = Manifold.Sphere(10.0, 32)
				.Difference(Manifold.Sphere(3.0, 32))
				.CalculateNormals(0, DefaultAngle)
				.GetMeshGL(-1);
			await Assert.That(mesh.NumProp >= 6).IsTrue().Because($"numProp={mesh.NumProp}");
			(int good, int bad) = CountSphereNormalAlignment(
				mesh,
				3.0,
				pos => LinalgFunctions.Normalize(pos * -1.0));
			await Assert.That(good > 0).IsTrue().Because($"NormalsCavity: good={good}");
			await Assert.That(bad).IsEqualTo(0).Because($"NormalsCavity: bad={bad}");
		}

		/// <summary>
		/// C++ TEST(Manifold, NormalsRotateBeforeCalc) — rotation before CalculateNormals:
		/// SetNormals computes from already-rotated face normals and stores world-frame.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNormalsRotateBeforeCalc()
		{
			MeshGL mesh = Manifold.Sphere(10.0, 32)
				.Rotate(45.0, 0.0, 0.0)
				.CalculateNormals(0, DefaultAngle)
				.GetMeshGL(-1);
			(int _, int bad) = CountSphereNormalAlignment(mesh, 10.0, LinalgFunctions.Normalize);
			await Assert.That(bad).IsEqualTo(0).Because($"NormalsRotateBeforeCalc: bad={bad}");
		}

		/// <summary>
		/// C++ TEST(Manifold, NormalsRotateAfterCalc) — rotation *after* CalculateNormals:
		/// Transform eager-transforms the stored slot 0..2 so it tracks the new orientation.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNormalsRotateAfterCalc()
		{
			MeshGL mesh = Manifold.Sphere(10.0, 32)
				.CalculateNormals(0, DefaultAngle)
				.Rotate(45.0, 0.0, 0.0)
				.GetMeshGL(-1);
			(int _, int bad) = CountSphereNormalAlignment(mesh, 10.0, LinalgFunctions.Normalize);
			await Assert.That(bad).IsEqualTo(0).Because($"NormalsRotateAfterCalc: bad={bad}");
		}

		/// <summary>
		/// C++ TEST(Manifold, NormalsAutoSubstitute) — no-arg invocation defaults to
		/// slot 0 and sets the per-run hasNormals bit on every output run.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNormalsAutoSubstitute()
		{
			MeshGL mesh = Manifold.Sphere(10.0, 32)
				.CalculateNormals(0, DefaultAngle)
				.GetMeshGL(-1);
			await Assert.That(mesh.NumProp >= 6).IsTrue().Because($"numProp={mesh.NumProp}");
			await Assert.That(mesh.RunOriginalId.Count == 0).IsFalse().Because("no runs");
			await Assert.That(mesh.HasNormals(0)).IsTrue().Because("run 0 should have hasNormals bit");
		}

		/// <summary>
		/// C++ TEST(Manifold, NormalsRoundTrip) — getMesh -&gt; ofMesh -&gt; getMesh preserves
		/// the per-run flag, so the second getMesh still emits world-frame normals.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNormalsRoundTrip()
		{
			Manifold round = Manifold.Sphere(10.0, 32)
				.Difference(Manifold.Sphere(3.0, 32))
				.CalculateNormals(0, DefaultAngle);
			MeshGL out1 = round.GetMeshGL(-1);
			await Assert.That(out1.HasNormals(0)).IsTrue().Because("out1 run 0 hasNormals");
			MeshGL out2 = Manifold.FromMeshGL(out1).GetMeshGL(-1);
			await Assert.That(out2.HasNormals(0)).IsTrue().Because("out2 run 0 hasNormals");
			(int good, int bad) = CountSphereNormalAlignment(
				out2,
				3.0,
				pos => LinalgFunctions.Normalize(pos * -1.0));
			await Assert.That(good > 0).IsTrue().Because($"NormalsRoundTrip: good={good}");
			await Assert.That(bad).IsEqualTo(0).Because($"NormalsRoundTrip: bad={bad}");
		}

		/// <summary>
		/// C++ TEST(Manifold, NormalsRefinePreserved) — Refine keeps the recording
		/// (linearly-interpolated normals at new verts, but the flag survives).
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNormalsRefinePreserved()
		{
			MeshGL mesh = Manifold.Sphere(10.0, 32)
				.CalculateNormals(0, DefaultAngle)
				.Refine(2)
				.GetMeshGL(-1);
			await Assert.That(mesh.RunOriginalId.Count == 0).IsFalse().Because("no runs");
			await Assert.That(mesh.HasNormals(0)).IsTrue().Because("Refine should preserve hasNormals");
		}

		/// <summary>
		/// C++ TEST(Manifold, NormalsSmoothByNormalsNoArg) — no-arg SmoothByNormals reads
		/// the recorded slot 0 and produces a valid manifold.
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNormalsSmoothByNormalsNoArg()
		{
			Manifold smoothed = Manifold.Sphere(10.0, 32)
				.CalculateNormals(0, DefaultAngle)
				.SmoothByNormals(0);
			await Assert.That(smoothed.Status()).IsEqualTo(Error.NoError);
		}

		/// <summary>
		/// C++ TEST(Manifold, NormalsNonStandardSlotNotRecorded) — CalculateNormals(3)
		/// does NOT set the recording, since a non-standard slot can't be safely
		/// auto-substituted on GetMeshGL(-1).
		/// </summary>
		/// <returns>A task representing the test.</returns>
		[Test]
		public async Task CppNormalsNonStandardSlotNotRecorded()
		{
			MeshGL mesh = Manifold.Sphere(10.0, 32)
				.CalculateNormals(3, DefaultAngle)
				.GetMeshGL(-1);
			await Assert.That(mesh.RunOriginalId.Count == 0).IsFalse().Because("no runs");
			await Assert.That(mesh.HasNormals(0))
				.IsFalse()
				.Because("non-standard slot must not record hasNormals");
		}

		/// <summary>
		/// Count verts on the sphere surface (|pos| ~ radius) whose stored normal at
		/// channel 3..5 aligns with <paramref name="expected"/>(pos) (dot &gt; 0.9).
		/// </summary>
		/// <param name="gl">The exported mesh to scan.</param>
		/// <param name="radius">The sphere radius the verts should sit on, within 0.1.</param>
		/// <param name="expected">The normal each qualifying position should carry.</param>
		/// <returns>(good, bad).</returns>
		private static (int Good, int Bad) CountSphereNormalAlignment(
			MeshGL gl,
			double radius,
			Func<Vec3, Vec3> expected)
		{
			int np = (int)gl.NumProp;
			(int good, int bad) = (0, 0);
			int numVert = gl.VertProperties.Count / np;
			for (int v = 0; v < numVert; v++)
			{
				Vec3 pos = new Vec3(
					gl.VertProperties[v * np],
					gl.VertProperties[(v * np) + 1],
					gl.VertProperties[(v * np) + 2]);
				if (Math.Abs(LinalgFunctions.Length(pos) - radius) > 0.1)
				{
					continue;
				}

				Vec3 n = new Vec3(
					gl.VertProperties[(v * np) + 3],
					gl.VertProperties[(v * np) + 4],
					gl.VertProperties[(v * np) + 5]);
				if (LinalgFunctions.Dot(n, expected(pos)) > 0.9)
				{
					good += 1;
				}
				else
				{
					bad += 1;
				}
			}

			return (good, bad);
		}
	}
}
