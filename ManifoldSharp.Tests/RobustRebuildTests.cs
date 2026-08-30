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

// RobustRebuildTests.cs — port of robust/rebuild_tests.rs, whose header reads:
//
//   Unit tests for the single-operand "rebuild solid" entry
//   (`robust::rebuild_with_rule` / `Manifold::rebuild_solid`).
//
//   Where robust/repair_tests.rs covers the cheap winding-only repair, these
//   cover the full arrangement pipeline run on one mesh: the fixtures are
//   deliberately *not* fixable by rewinding shells (doubled faces, mutually
//   overlapping bodies, redundant interior shells) and can only come out
//   2-manifold if the mesh is genuinely re-derived from the winding numbers.
//
// Same inputs, same expected values, same order as the Rust's nine tests.

using ManifoldSharp;
using ManifoldSharp.Linalg;
using ManifoldSharp.Robust;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.Tests
{
	public class RobustRebuildTests
	{
		[Test]
		public async Task DuplicatedFacesCollapseToOneShell()
		{
			// Every triangle present twice: >2 faces per edge everywhere, so nothing
			// pairs. The winding jumps 0 → 2 across the surface, which {w >= 1} still
			// reads as one solid boundary.
			List<Vec3[]> tris = CubeTris(0.0, 2.0);
			tris.AddRange(CubeTris(0.0, 2.0));
			Manifold m = MeshFromTris(tris);

			// The import keeps both copies (it never dedups faces), so the fixture
			// really does carry the doubled sheet the rebuild has to collapse — its
			// divergence-theorem volume double-counts at 16.
			await Assert.That(m.NumTri()).IsEqualTo(24).Because("fixture must keep both copies");
			await Assert.That(Math.Abs(SignedVolume(m) - 16.0)).IsLessThan(1e-9);

			Manifold outManifold = m.RebuildSolid(WindingRule.Positive);
			await AssertCleanManifold(outManifold);
			await Assert.That(Math.Abs(SignedVolume(outManifold) - 8.0)).IsLessThan(1e-9)
				.Because($"expected 8, got {SignedVolume(outManifold)}");
		}

		[Test]
		public async Task OverlappingBodiesInOneSoupBecomeTheirUnion()
		{
			// Two mutually penetrating cubes concatenated into a single mesh: the
			// interior walls have material on both sides and must dissolve.
			List<Vec3[]> tris = CubeTris(0.0, 2.0);
			tris.AddRange(BoxTris(V(1.0, 1.0, 1.0), V(3.0, 3.0, 3.0)));
			Manifold soup = MeshFromTris(tris);
			await Assert.That(soup.HasSelfIntersections()).IsTrue()
				.Because("fixture must genuinely self-intersect");
			Manifold outManifold = soup.RebuildSolid(WindingRule.Positive);
			await AssertCleanManifold(outManifold);

			Manifold a = Manifold.Cube(V(2.0, 2.0, 2.0), false);
			Manifold b = Manifold.Cube(V(2.0, 2.0, 2.0), false).Translate(V(1.0, 1.0, 1.0));
			Manifold reference = a.Union(b);

			// Signed, not `Volume()`: an inside-out union would have the same absolute
			// volume and would sail through.
			double expected = SignedVolume(reference);
			await Assert.That(expected).IsGreaterThan(0.0)
				.Because("reference union must be outward-wound");
			await Assert.That(Math.Abs(SignedVolume(outManifold) - expected)).IsLessThan(1e-9)
				.Because($"expected {expected}, got {SignedVolume(outManifold)}");
		}

		[Test]
		public async Task GenuineCavitySurvivesRebuild()
		{
			// Outer [0,6] outward + inner [2,4] inward: a solid with a void. The
			// inner shell is a real boundary (material outside it, none inside), so
			// the rebuild must keep it.
			List<Vec3[]> tris = CubeTris(0.0, 6.0);
			tris.AddRange(Flipped(CubeTris(2.0, 4.0)));
			Manifold outManifold = MeshFromTris(tris).RebuildSolid(WindingRule.Positive);
			await AssertCleanManifold(outManifold);
			await Assert.That(Math.Abs(SignedVolume(outManifold) - (216.0 - 8.0))).IsLessThan(1e-9)
				.Because($"expected 208, got {SignedVolume(outManifold)}");
		}

		[Test]
		public async Task RedundantInteriorShellIsRemoved()
		{
			// Outer [0,6] and inner [2,4] both wound outward: the inner shell is
			// material on both sides (winding 2 inside, 1 outside), so it is not a
			// boundary at all and must vanish.
			List<Vec3[]> tris = CubeTris(0.0, 6.0);
			tris.AddRange(CubeTris(2.0, 4.0));
			Manifold m = MeshFromTris(tris);
			await Assert.That(m.NumTri()).IsEqualTo(24).Because("fixture must carry both shells");
			Manifold outManifold = m.RebuildSolid(WindingRule.Positive);
			await AssertCleanManifold(outManifold);
			await Assert.That(Math.Abs(SignedVolume(outManifold) - 216.0)).IsLessThan(1e-9)
				.Because($"expected 216, got {SignedVolume(outManifold)}");
			await Assert.That(outManifold.NumTri()).IsEqualTo(12)
				.Because("only the outer cube's faces should remain");
		}

		[Test]
		public async Task InsideOutCubeDependsOnTheWindingRule()
		{
			// Winding -1 inside. {w != 0} calls that solid and rewinds it outward;
			// {w >= 1} calls it nothing at all.
			Manifold m = MeshFromTris(Flipped(CubeTris(0.0, 2.0)));
			await Assert.That(SignedVolume(m)).IsLessThan(0.0).Because("fixture must import inverted");

			Manifold nonzero = m.RebuildSolid(WindingRule.Nonzero);
			await AssertCleanManifold(nonzero);
			await Assert.That(Math.Abs(SignedVolume(nonzero) - 8.0)).IsLessThan(1e-9)
				.Because($"expected +8, got {SignedVolume(nonzero)}");

			Manifold positive = m.RebuildSolid(WindingRule.Positive);
			await Assert.That(positive.IsEmpty()).IsTrue()
				.Because($"no positive material, got {positive.NumTri()} tris");
			await Assert.That(positive.Status()).IsEqualTo(Error.NoError);
		}

		/// <summary>
		/// The operand occupies mesh slot 0, so <c>PropCtx</c> interpolates its corner
		/// properties onto the rebuilt triangles exactly as a two-operand boolean would. An
		/// affine-in-position property must therefore reproduce itself at every output vertex,
		/// including the ones the arrangement invents.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task CornerPropertiesSurviveTheRebuild()
		{
			static double Prop(Vec3 pos)
			{
				return 0.25 + (pos.X * 0.5) + (pos.Y * 0.125) - (pos.Z * 0.0625);
			}

			// Two overlapping cubes in one body, so the rebuild really does
			// re-triangulate and must interpolate rather than copy.
			Manifold a = Manifold.Cube(V(2.0, 2.0, 2.0), false);
			Manifold b = Manifold.Cube(V(2.0, 2.0, 2.0), false).Translate(V(1.0, 1.0, 1.0));
			Manifold joined = a.Union(b)
				.SetProperties(1, (newProp, pos, _) => newProp[0] = Prop(pos));
			Manifold outManifold = joined.RebuildSolid(WindingRule.Positive);
			await Assert.That(outManifold.Status()).IsEqualTo(Error.NoError);
			await Assert.That(outManifold.AsImpl().NumProp).IsEqualTo(1)
				.Because("the property must survive");

			const int Np = 4; // xyz + 1
			MeshGL gl = outManifold.GetMeshGL(-1);
			await Assert.That((int)gl.NumProp).IsEqualTo(Np);
			for (int i = 0; i < gl.VertProperties.Count / Np; i++)
			{
				Vec3 pos = V(
					gl.VertProperties[i * Np],
					gl.VertProperties[(i * Np) + 1],
					gl.VertProperties[(i * Np) + 2]);
				double got = gl.VertProperties[(i * Np) + 3];
				await Assert.That(Math.Abs(got - Prop(pos))).IsLessThan(1e-6)
					.Because($"property at {pos}: expected {Prop(pos)}, got {got}");
			}
		}

		/// <summary>
		/// The no-op case has to actually be a no-op. Nothing in a strictly manifold,
		/// self-intersection-free mesh needs cutting, so every original triangle must survive
		/// whole: same count, and a volume that matches bit-for-bit rather than merely to a
		/// tolerance. A rebuild that quietly perturbed clean input would be invisible to every
		/// other test here, which all work in tolerances.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task AlreadyManifoldInputRoundTripsUnchanged()
		{
			foreach (Manifold input in new[]
			{
				Manifold.Cube(V(2.0, 3.0, 5.0), false),
				Manifold.Sphere(1.5, 32),
			})
			{
				Manifold outManifold = input.RebuildSolid(WindingRule.Positive);
				await AssertCleanManifold(outManifold);
				await Assert.That(outManifold.NumTri()).IsEqualTo(input.NumTri())
					.Because("clean input must not be re-triangulated");
				await Assert.That(BitConverter.DoubleToUInt64Bits(outManifold.Volume()))
					.IsEqualTo(BitConverter.DoubleToUInt64Bits(input.Volume()))
					.Because($"expected {input.Volume()}, got {outManifold.Volume()}");
			}
		}

		[Test]
		public async Task EmptyInputRebuildsToEmpty()
		{
			Manifold empty = new Manifold();
			Manifold outManifold = empty.RebuildSolid(WindingRule.Positive);
			await Assert.That(outManifold.IsEmpty()).IsTrue();
			await Assert.That(outManifold.Status()).IsEqualTo(Error.NoError);
		}

		[Test]
		public async Task RebuildRespectsACancelToken()
		{
			CancelToken token = new CancelToken();
			token.Cancel();
			List<Vec3[]> tris = CubeTris(0.0, 2.0);
			tris.AddRange(CubeTris(0.0, 2.0));
			Manifold m = MeshFromTris(tris);
			ManifoldImpl outImpl = RobustFunctions.RebuildWithRule(
				m.AsImpl(), WindingRule.Positive, token, null);
			await Assert.That(outImpl.IsEmpty()).IsTrue();
			await Assert.That(outImpl.Status).IsEqualTo(Error.Cancelled);
		}

		/// <summary>Shorthand for the Rust's <c>v(x, y, z)</c>.</summary>
		/// <param name="x">The x coordinate.</param>
		/// <param name="y">The y coordinate.</param>
		/// <param name="z">The z coordinate.</param>
		/// <returns>The vector.</returns>
		private static Vec3 V(double x, double y, double z)
		{
			return new Vec3(x, y, z);
		}

		/// <summary>Axis-aligned cube [lo,hi]³ as 12 outward-wound triangles.</summary>
		/// <param name="lo">The low coordinate on every axis.</param>
		/// <param name="hi">The high coordinate on every axis.</param>
		/// <returns>The cube's triangles.</returns>
		private static List<Vec3[]> CubeTris(double lo, double hi)
		{
			return RobustRepairTests.CubeTris(lo, hi);
		}

		/// <summary>Box [lo,hi] with independent per-axis extents, outward-wound.</summary>
		/// <param name="lo">The low corner.</param>
		/// <param name="hi">The high corner.</param>
		/// <returns>The box's triangles.</returns>
		private static List<Vec3[]> BoxTris(Vec3 lo, Vec3 hi)
		{
			List<Vec3[]> unit = CubeTris(0.0, 1.0);
			Vec3 M(Vec3 p) => V(
				lo.X + (p.X * (hi.X - lo.X)),
				lo.Y + (p.Y * (hi.Y - lo.Y)),
				lo.Z + (p.Z * (hi.Z - lo.Z)));
			List<Vec3[]> outTris = new List<Vec3[]>(unit.Count);
			foreach (Vec3[] t in unit)
			{
				outTris.Add(new[] { M(t[0]), M(t[1]), M(t[2]) });
			}

			return outTris;
		}

		/// <summary>Every triangle reversed.</summary>
		/// <param name="tris">The triangles.</param>
		/// <returns>The reversed triangles.</returns>
		private static List<Vec3[]> Flipped(IReadOnlyList<Vec3[]> tris)
		{
			return RobustRepairTests.Flipped(tris);
		}

		/// <summary>
		/// Import a raw soup exactly as given — no welding away of duplicate faces, no
		/// orientation repair — so the rebuild is what has to clean it up.
		/// </summary>
		/// <param name="tris">The triangles.</param>
		/// <returns>The imported manifold.</returns>
		private static Manifold MeshFromTris(IReadOnlyList<Vec3[]> tris)
		{
			MeshGL mesh = new MeshGL();
			mesh.NumProp = 3;
			foreach (Vec3[] t in tris)
			{
				foreach (Vec3 p in t)
				{
					mesh.VertProperties.Add((float)p.X);
					mesh.VertProperties.Add((float)p.Y);
					mesh.VertProperties.Add((float)p.Z);
				}
			}

			for (uint i = 0; i < (uint)(tris.Count * 3); i++)
			{
				mesh.TriVerts.Add(i);
			}

			mesh.Merge();
			return Manifold.FromMeshGLRobust(mesh);
		}

		/// <summary>
		/// Signed (divergence-theorem) volume — <c>Volume()</c> takes the absolute value,
		/// which would hide an inside-out result.
		/// </summary>
		/// <param name="m">The manifold.</param>
		/// <returns>Its signed volume.</returns>
		private static double SignedVolume(Manifold m)
		{
			return RobustRepairTests.SignedVolume(m);
		}

		/// <summary>
		/// Every assertion a "properly paired 2-manifold" has to satisfy: real halfedge
		/// pairing (not a soup fallback), a valid import status, and no duplicated face.
		/// </summary>
		/// <param name="m">The rebuilt manifold.</param>
		/// <returns>The assertion task.</returns>
		private static async Task AssertCleanManifold(Manifold m)
		{
			await Assert.That(m.Status()).IsEqualTo(Error.NoError)
				.Because("rebuilt mesh must import clean");
			await Assert.That(m.AsImpl().IsSoup).IsFalse()
				.Because("rebuilt mesh must not fall back to soup");
			await Assert.That(m.AsImpl().IsManifold()).IsTrue()
				.Because("rebuilt mesh must be 2-manifold");

			List<Vec3[]> tris = Robust.Soup.ImplToTris(m.AsImpl());
			List<ulong[]> keys = new List<ulong[]>(tris.Count);
			foreach (Vec3[] t in tris)
			{
				// The Rust sorts the three corner keys inside each triangle, then sorts and
				// dedups the whole list — so a triangle is identified by its unordered
				// vertex set. Flattened to nine ulongs here, with a lexicographic comparer,
				// because C# arrays have reference equality.
				ulong[][] corners =
				{
					Bits(t[0]),
					Bits(t[1]),
					Bits(t[2]),
				};
				Array.Sort(corners, CompareTriple);
				keys.Add(new[]
				{
					corners[0][0], corners[0][1], corners[0][2],
					corners[1][0], corners[1][1], corners[1][2],
					corners[2][0], corners[2][1], corners[2][2],
				});
			}

			int before = keys.Count;
			keys.Sort(CompareNine);
			int distinct = 0;
			for (int i = 0; i < keys.Count; i++)
			{
				if (i == 0 || CompareNine(keys[i - 1], keys[i]) != 0)
				{
					distinct++;
				}
			}

			await Assert.That(before).IsEqualTo(distinct)
				.Because("rebuilt mesh must have no doubled faces");
		}

		/// <summary>The three coordinate bit patterns of one corner.</summary>
		/// <param name="p">The corner.</param>
		/// <returns>Its bits.</returns>
		private static ulong[] Bits(Vec3 p)
		{
			return new[]
			{
				BitConverter.DoubleToUInt64Bits(p.X),
				BitConverter.DoubleToUInt64Bits(p.Y),
				BitConverter.DoubleToUInt64Bits(p.Z),
			};
		}

		/// <summary>Lexicographic order on a three-element key.</summary>
		/// <param name="a">The left key.</param>
		/// <param name="b">The right key.</param>
		/// <returns>The comparison.</returns>
		private static int CompareTriple(ulong[] a, ulong[] b)
		{
			for (int i = 0; i < 3; i++)
			{
				int c = a[i].CompareTo(b[i]);
				if (c != 0)
				{
					return c;
				}
			}

			return 0;
		}

		/// <summary>Lexicographic order on a nine-element key.</summary>
		/// <param name="a">The left key.</param>
		/// <param name="b">The right key.</param>
		/// <returns>The comparison.</returns>
		private static int CompareNine(ulong[] a, ulong[] b)
		{
			for (int i = 0; i < 9; i++)
			{
				int c = a[i].CompareTo(b[i]);
				if (c != 0)
				{
					return c;
				}
			}

			return 0;
		}
	}
}
