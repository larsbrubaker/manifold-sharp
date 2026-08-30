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

// The oracle lane proper: the same boolean, on the same geometry, through
// ManifoldSharp's managed exact engine and through the ManifoldRust P/Invoke
// binding (the native manifold-rust cdylib), compared bit-for-bit. This is
// verification net #2 from docs/PORTING_PLAN.md, and the reason the binding is a
// dependency of this repo at all.
//
// NOT A PORT: manifold-rust has no such test. It is C#-only, and its expected
// values are whatever the oracle says — never a number typed in by hand.
//
// ── How the two sides are made comparable, and what that costs ───────────────
// Both engines start from one array of f64 positions and one array of triangle
// indices, so neither side's geometry is derived from the other's. The port has
// no MeshGL importer or exporter yet (manifold_meshgl.rs is Phase 6), so:
//
//   * Input.  The port's side is built with ManifoldImpl.CreateHalfedges and the
//     same finishing sequence Phase 4's FromShape uses; the binding's side is
//     Manifold.FromMesh64 on the identical arrays. AssertSameGeometry runs on
//     the *inputs* first, so a mismatch there is reported as an input-construction
//     problem and never mistaken for a boolean bug.
//   * Output. Both sides now go through the same exporter contract: the port's
//     Manifold.GetMeshGL64(-1) (Phase 6) applies the identical (originalID,
//     meshID) run sort the binding's GetMeshGL64 does, so the two triangle lists
//     are compared ROW FOR ROW, in order, with no canonicalization. The triple
//     is compared as written — (v0, v1, v2) in corner order, because neither
//     exporter rotates corners (manifold_meshgl.rs:451), so the starting corner
//     is comparable content and not an artifact. Vertex *positions* are compared
//     bit-for-bit in index order, because both engines end in the same
//     sort_geometry and must agree on it exactly.
//
// There is now NO slack in this lane. The earlier note here recorded one — an
// ordinal sort of the triangle list, standing in for a run sort the port could
// not yet reproduce — and Phase 6 retired it: with the exporter in place the raw
// lists match, so the sort came out rather than being kept as a safety net that
// would hide a real reordering bug.

using ManifoldRust;

using ManifoldSharp;
using ManifoldSharp.Linalg;

using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace ManifoldSharp.OracleTests
{
	public class BooleanOracleTests
	{
		/// <summary>
		/// The unit cube as raw MeshGL data: eight positions, twelve triangles wound
		/// counter-clockwise seen from outside. Transliterated from the binding's own test
		/// meshes rather than generated from either engine, for the reason OracleSmokeTests
		/// gives — an oracle must not be fed by the thing it is meant to check.
		/// </summary>
		private static readonly double[] CubeVerts =
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

		private static readonly uint[] CubeTris =
		{
			0, 2, 1, 0, 3, 2, // -Z
			4, 5, 6, 4, 6, 7, // +Z
			0, 1, 5, 0, 5, 4, // -Y
			1, 2, 6, 1, 6, 5, // +X
			2, 3, 7, 2, 7, 6, // +Y
			3, 0, 4, 3, 4, 7, // -X
		};

		[Test]
		[Arguments(0.5, 0.3, 0.2, ManifoldOpType.Add)]
		[Arguments(0.5, 0.3, 0.2, ManifoldOpType.Subtract)]
		[Arguments(0.5, 0.3, 0.2, ManifoldOpType.Intersect)]
		[Arguments(0.9, 0.0, 0.0, ManifoldOpType.Add)]
		[Arguments(0.9, 0.0, 0.0, ManifoldOpType.Subtract)]
		[Arguments(0.9, 0.0, 0.0, ManifoldOpType.Intersect)]
		[Arguments(1.0, 0.0, 0.0, ManifoldOpType.Add)]
		[Arguments(1.0, 0.0, 0.0, ManifoldOpType.Subtract)]
		[Arguments(0.0, 0.0, 0.0, ManifoldOpType.Add)]
		[Arguments(0.0, 0.0, 0.0, ManifoldOpType.Intersect)]
		[Arguments(0.25, 0.25, 0.25, ManifoldOpType.Subtract)]
		[Arguments(3.0, 0.0, 0.0, ManifoldOpType.Add)]
		public async Task CubeBooleanMatchesTheNativeOracle(
			double dx,
			double dy,
			double dz,
			ManifoldOpType op)
		{
			double[] shifted = Translate(CubeVerts, dx, dy, dz);

			// Inputs first: if these disagree the boolean comparison below is meaningless,
			// and the failure is in construction, not in the engine under test.
			ManifoldSharp.ManifoldImpl portA = BuildImpl(CubeVerts, CubeTris);
			ManifoldSharp.ManifoldImpl portB = BuildImpl(shifted, CubeTris);
			using ManifoldRust.Manifold oracleA = ManifoldRust.Manifold.FromMesh64(CubeVerts, CubeTris);
			using ManifoldRust.Manifold oracleB = ManifoldRust.Manifold.FromMesh64(shifted, CubeTris);
			await Assert.That(oracleA.Status).IsEqualTo(ManifoldStatus.NoError);
			await Assert.That(oracleB.Status).IsEqualTo(ManifoldStatus.NoError);

			await AssertSameGeometry("input A", portA, oracleA.GetMeshGL64());
			await AssertSameGeometry("input B", portB, oracleB.GetMeshGL64());

			ManifoldSharp.ManifoldImpl portResult =
				Boolean3Functions.Boolean(portA, portB, ToPortOp(op));

			// Exact on both sides: the port has only the exact engine (the robust one is
			// Phase 10), so pinning the oracle to Exact is what makes this a like-for-like
			// comparison rather than an engine bake-off.
			using ManifoldRust.Manifold oracleResult =
				ManifoldRust.Manifold.Boolean(oracleA, oracleB, op, ManifoldRust.BooleanEngine.Exact);
			await Assert.That(oracleResult.Status).IsEqualTo(ManifoldStatus.NoError);

			// Anti-vacuity. Every row of the table above is chosen to leave material
			// behind, so "the two agreed" can never mean "both produced nothing".
			await Assert.That(portResult.NumTri())
				.IsGreaterThan(0)
				.Because($"{op} at ({dx}, {dy}, {dz}) should leave geometry to compare");

			await AssertSameGeometry($"{op} at ({dx}, {dy}, {dz})", portResult, oracleResult.GetMeshGL64());
		}

		private static ManifoldSharp.OpType ToPortOp(ManifoldOpType op)
		{
			switch (op)
			{
				case ManifoldOpType.Add:
					return ManifoldSharp.OpType.Add;
				case ManifoldOpType.Subtract:
					return ManifoldSharp.OpType.Subtract;
				case ManifoldOpType.Intersect:
					return ManifoldSharp.OpType.Intersect;
				default:
					throw new ArgumentOutOfRangeException(nameof(op), $"Unknown op: {(int)op}");
			}
		}

		private static double[] Translate(double[] verts, double dx, double dy, double dz)
		{
			double[] outVerts = new double[verts.Length];
			for (int i = 0; i < verts.Length; i += 3)
			{
				outVerts[i] = verts[i] + dx;
				outVerts[i + 1] = verts[i + 1] + dy;
				outVerts[i + 2] = verts[i + 2] + dz;
			}

			return outVerts;
		}

		/// <summary>
		/// The port's counterpart of <c>Manifold.FromMesh64</c> for clean manifold input
		/// with no extra properties: the finishing sequence Phase 4's
		/// <c>ManifoldImpl.FromShape</c> runs, which is the part of the Rust's
		/// <c>from_mesh_impl</c> that a validated, merge-free, property-free mesh reaches.
		/// The <c>AssertSameGeometry</c> on the inputs is what keeps that claim honest.
		/// </summary>
		private static ManifoldSharp.ManifoldImpl BuildImpl(double[] verts, uint[] tris)
		{
			ManifoldSharp.ManifoldImpl m = new ManifoldSharp.ManifoldImpl();
			for (int i = 0; i < verts.Length; i += 3)
			{
				m.VertPos.Add(new Vec3(verts[i], verts[i + 1], verts[i + 2]));
			}

			List<IVec3> triVerts = new List<IVec3>(tris.Length / 3);
			for (int i = 0; i < tris.Length; i += 3)
			{
				triVerts.Add(new IVec3((int)tris[i], (int)tris[i + 1], (int)tris[i + 2]));
			}

			m.CreateHalfedges(triVerts, Array.Empty<IVec3>());
			m.InitializeOriginal();
			m.CalculateBBox();
			m.SetEpsilon(-1.0, false);
			m.SortGeometry();
			m.SetNormalsAndCoplanar();
			return m;
		}

		private static async Task AssertSameGeometry(
			string what,
			ManifoldSharp.ManifoldImpl port,
			ManifoldRust.MeshGL64 oracle)
		{
			await Assert.That(port.NumVert())
				.IsEqualTo(oracle.VertexCount)
				.Because($"{what}: vertex count");
			await Assert.That(port.NumTri())
				.IsEqualTo(oracle.TriangleCount)
				.Because($"{what}: triangle count");

			// The oracle's numProp is 3 + extraProps; with no extra properties the vertex
			// rows are exactly the positions, in the kernel's own vertex order.
			await Assert.That((int)oracle.NumProp).IsEqualTo(3).Because($"{what}: numProp");

			for (int v = 0; v < port.NumVert(); v++)
			{
				Vec3 p = port.VertPos[v];
				await AssertSameBits($"{what}: vert[{v}].x", p.X, oracle.VertProperties[(3 * v) + 0]);
				await AssertSameBits($"{what}: vert[{v}].y", p.Y, oracle.VertProperties[(3 * v) + 1]);
				await AssertSameBits($"{what}: vert[{v}].z", p.Z, oracle.VertProperties[(3 * v) + 2]);
			}

			ulong[] portTris = PortTriangles(port);
			uint[] oracleTris = oracle.TriVerts;
			await Assert.That(portTris.Length)
				.IsEqualTo(oracleTris.Length)
				.Because($"{what}: triangle index count");
			for (int t = 0; t + 2 < portTris.Length; t += 3)
			{
				await Assert.That($"{portTris[t]},{portTris[t + 1]},{portTris[t + 2]}")
					.IsEqualTo($"{oracleTris[t]},{oracleTris[t + 1]},{oracleTris[t + 2]}")
					.Because($"{what}: triangle {t / 3}");
			}
		}

		/// <summary>
		/// Bit-for-bit, not approximately: the exactness bar is identical doubles, and
		/// <c>IsEqualTo</c> on a double would let a signed zero or a NaN payload through.
		/// </summary>
		private static async Task AssertSameBits(string what, double port, double oracle)
		{
			await Assert.That(BitConverter.DoubleToUInt64Bits(port).ToString("x16"))
				.IsEqualTo(BitConverter.DoubleToUInt64Bits(oracle).ToString("x16"))
				.Because($"{what}: {port} vs {oracle}");
		}

		/// <summary>
		/// The port's triangle list as its own exporter writes it. Phase 6's
		/// <c>Manifold.GetMeshGL64</c> applies the same (originalID, meshID) run sort the
		/// binding's exporter does, which is what lets the caller compare row for row
		/// instead of canonicalizing; see the file header. With no extra properties the
		/// exporter indexes by geometric vertex, so these indices are directly comparable
		/// to the positions checked above.
		/// </summary>
		private static ulong[] PortTriangles(ManifoldSharp.ManifoldImpl m)
		{
			return ManifoldSharp.Manifold.FromImpl(m).GetMeshGL64(-1).TriVerts.ToArray();
		}
	}
}
