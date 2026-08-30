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

			// Exact on both sides. The port has both engines now, which is precisely why
			// the pin stays: naming the engine explicitly on each side keeps every row a
			// like-for-like comparison of one engine against its own counterpart, rather
			// than an engine bake-off that would pass or fail for the wrong reason. The
			// robust lane is the separate table below, pinned to Robust the same way.
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

		/// <summary>
		/// The same table again, on the ROBUST engine both sides — Phase 10's engine, pinned
		/// against the native one row for row with the same zero slack. Both operands are
		/// clean manifolds, so the two engines are being asked for the identical arrangement,
		/// the identical cell labels and the identical assembly.
		/// </summary>
		/// <param name="dx">Operand B's x offset.</param>
		/// <param name="dy">Operand B's y offset.</param>
		/// <param name="dz">Operand B's z offset.</param>
		/// <param name="op">The boolean operation.</param>
		/// <returns>The test task.</returns>
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
		public async Task CubeBooleanOnTheRobustEngineMatchesTheNativeOracle(
			double dx,
			double dy,
			double dz,
			ManifoldOpType op)
		{
			await RobustRow(dx, dy, dz, op, ManifoldRust.WindingRule.Positive);
		}

		/// <summary>
		/// The robust engine again under <c>{w != 0}</c>. On outward-wound operands the rule
		/// cannot change the answer, which is exactly why it is worth pinning: any divergence
		/// here is the rule leaking into the arrangement rather than staying in
		/// <c>in_result</c>.
		/// </summary>
		/// <param name="dx">Operand B's x offset.</param>
		/// <param name="dy">Operand B's y offset.</param>
		/// <param name="dz">Operand B's z offset.</param>
		/// <param name="op">The boolean operation.</param>
		/// <returns>The test task.</returns>
		[Test]
		[Arguments(0.5, 0.3, 0.2, ManifoldOpType.Add)]
		[Arguments(0.5, 0.3, 0.2, ManifoldOpType.Subtract)]
		[Arguments(0.5, 0.3, 0.2, ManifoldOpType.Intersect)]
		[Arguments(0.9, 0.0, 0.0, ManifoldOpType.Add)]
		public async Task CubeBooleanOnTheRobustEngineUnderNonzeroMatchesTheNativeOracle(
			double dx,
			double dy,
			double dz,
			ManifoldOpType op)
		{
			await RobustRow(dx, dy, dz, op, ManifoldRust.WindingRule.Nonzero);
		}

		/// <summary>
		/// <c>RepairOrientation</c> on an inside-out cube: the shell classification, the
		/// containment depths and the in-place rewind must produce the identical mesh on both
		/// sides, triangle order included.
		/// </summary>
		/// <returns>The test task.</returns>
		[Test]
		public async Task RepairOrientationMatchesTheNativeOracle()
		{
			uint[] inverted = Reversed(CubeTris);
			ManifoldSharp.Manifold port = ImportRobust(CubeVerts, inverted);
			using ManifoldRust.Manifold oracle =
				ManifoldRust.Manifold.FromMesh64Robust(CubeVerts, Widen(inverted), 3);
			await Assert.That(oracle.Status).IsEqualTo(ManifoldStatus.NoError);
			await AssertSameGeometry("inverted input", port.AsImpl(), oracle.GetMeshGL64());

			ManifoldSharp.Manifold portFixed = port.RepairOrientation();
			using ManifoldRust.Manifold oracleFixed = oracle.RepairOrientation();
			await Assert.That(oracleFixed.Status).IsEqualTo(ManifoldStatus.NoError);
			await Assert.That(portFixed.NumTri()).IsGreaterThan(0)
				.Because("the repaired cube should still have its twelve triangles");
			await AssertSameGeometry("repair_orientation", portFixed.AsImpl(), oracleFixed.GetMeshGL64());
		}

		/// <summary>
		/// <c>RebuildSolid</c> on a doubled cover — every facet present twice, so nothing
		/// pairs and only a genuine re-derivation from the winding numbers can produce a
		/// 2-manifold. Run under both rules, both compared bit-for-bit.
		/// </summary>
		/// <param name="rule">The winding rule.</param>
		/// <returns>The test task.</returns>
		[Test]
		[Arguments(ManifoldRust.WindingRule.Positive)]
		[Arguments(ManifoldRust.WindingRule.Nonzero)]
		public async Task RebuildSolidMatchesTheNativeOracle(ManifoldRust.WindingRule rule)
		{
			// The doubled cover needs distinct vertex rows per copy or the importer would
			// weld the two sheets into one; corner-per-vertex is what the demo pipeline
			// hands the robust import anyway.
			(double[] Verts, uint[] Tris) doubled = DoubledCover();
			ManifoldSharp.Manifold port = ImportRobust(doubled.Verts, doubled.Tris);
			using ManifoldRust.Manifold oracle =
				ManifoldRust.Manifold.FromMesh64Robust(doubled.Verts, Widen(doubled.Tris), 3);
			await Assert.That(oracle.Status).IsEqualTo(ManifoldStatus.NoError);

			ManifoldSharp.Manifold portOut = port.RebuildSolid(ToPortRule(rule));
			using ManifoldRust.Manifold oracleOut = oracle.RebuildSolid(rule);
			await Assert.That(oracleOut.Status).IsEqualTo(ManifoldStatus.NoError);
			await Assert.That(portOut.NumTri()).IsGreaterThan(0)
				.Because($"rebuild under {rule} should leave geometry to compare");
			await AssertSameGeometry($"rebuild_solid {rule}", portOut.AsImpl(), oracleOut.GetMeshGL64());
		}

		/// <summary>
		/// <see cref="Subdivision.SubdivideImpl"/>'s repair, proved against the native
		/// library — see docs/RUST_DIVERGENCES.md entry 5.
		/// </summary>
		/// <remarks>
		/// The repair reorders the output (SortGeometry must, because Collider's radix tree
		/// requires ascending Morton codes), so "the port did not change" is not available as
		/// evidence and the reorder has to be shown to be the *right* one. This does that
		/// directly: the port's subdivided impl is handed to the native importer as a plain
		/// triangle soup, the native runs its own sort_geometry on it, and the two vertex
		/// orders and triangle lists are compared row for row with no canonicalization. The
		/// pre-repair output fails this — it is not in Morton order at all — and so does any
		/// repair that sorts differently. The boolean afterwards is the symptom the entry
		/// opens with, checked the same way.
		/// </remarks>
		/// <param name="levels">How many times to subdivide.</param>
		/// <returns>The test task.</returns>
		[Test]
		[Arguments(1)]
		[Arguments(2)]
		public async Task SubdividedCubeMatchesTheNativeOracle(int levels)
		{
			ManifoldSharp.ManifoldImpl port = Subdivision.SubdivideImpl(
				ManifoldSharp.ManifoldImpl.Cube(Mat3x4.Identity()),
				levels);

			// The port's own output as a soup; the oracle re-derives everything from it, so
			// the two orders agree only if the port's finishing is the native's finishing.
			ManifoldSharp.MeshGL64 soup = ManifoldSharp.Manifold.FromImpl(port).GetMeshGL64(-1);
			double[] verts = soup.VertProperties.ToArray();
			uint[] tris = soup.TriVerts.Select(v => (uint)v).ToArray();

			using ManifoldRust.Manifold oracle = ManifoldRust.Manifold.FromMesh64(verts, tris);
			await Assert.That(oracle.Status).IsEqualTo(ManifoldStatus.NoError);
			await AssertSameGeometry($"subdivide {levels}", port, oracle.GetMeshGL64());

			// And the boolean that could not run before the repair, on both engines.
			double[] boreVerts = Translate(Scale(CubeVerts, 0.5, 0.5, 4.0), 0.25, 0.25, -1.5);
			ManifoldSharp.ManifoldImpl portBore = BuildImpl(boreVerts, CubeTris);
			using ManifoldRust.Manifold oracleBore = ManifoldRust.Manifold.FromMesh64(boreVerts, CubeTris);
			await Assert.That(oracleBore.Status).IsEqualTo(ManifoldStatus.NoError);
			await AssertSameGeometry("bore", portBore, oracleBore.GetMeshGL64());

			ManifoldSharp.ManifoldImpl portDrilled =
				Boolean3Functions.Boolean(port, portBore, ManifoldSharp.OpType.Subtract);
			using ManifoldRust.Manifold oracleDrilled = ManifoldRust.Manifold.Boolean(
				oracle,
				oracleBore,
				ManifoldOpType.Subtract,
				ManifoldRust.BooleanEngine.Exact);
			await Assert.That(oracleDrilled.Status).IsEqualTo(ManifoldStatus.NoError);
			await Assert.That(portDrilled.NumTri())
				.IsGreaterThan(0)
				.Because("drilling the subdivided cube should leave geometry to compare");
			await AssertSameGeometry($"subdivide {levels} minus bore", portDrilled, oracleDrilled.GetMeshGL64());
		}

		/// <summary>One row of the robust boolean table, under one winding rule.</summary>
		/// <param name="dx">Operand B's x offset.</param>
		/// <param name="dy">Operand B's y offset.</param>
		/// <param name="dz">Operand B's z offset.</param>
		/// <param name="op">The boolean operation.</param>
		/// <param name="rule">The winding rule.</param>
		/// <returns>The assertion task.</returns>
		private static async Task RobustRow(
			double dx,
			double dy,
			double dz,
			ManifoldOpType op,
			ManifoldRust.WindingRule rule)
		{
			double[] shifted = Translate(CubeVerts, dx, dy, dz);

			ManifoldSharp.ManifoldImpl portA = BuildImpl(CubeVerts, CubeTris);
			ManifoldSharp.ManifoldImpl portB = BuildImpl(shifted, CubeTris);
			using ManifoldRust.Manifold oracleA = ManifoldRust.Manifold.FromMesh64(CubeVerts, CubeTris);
			using ManifoldRust.Manifold oracleB = ManifoldRust.Manifold.FromMesh64(shifted, CubeTris);
			await Assert.That(oracleA.Status).IsEqualTo(ManifoldStatus.NoError);
			await Assert.That(oracleB.Status).IsEqualTo(ManifoldStatus.NoError);

			await AssertSameGeometry("input A", portA, oracleA.GetMeshGL64());
			await AssertSameGeometry("input B", portB, oracleB.GetMeshGL64());

			ManifoldSharp.ManifoldImpl portResult = ManifoldSharp.Robust.RobustFunctions.BooleanWithRule(
				portA, portB, ToPortOp(op), ToPortRule(rule), null, null);

			using ManifoldRust.Manifold oracleResult = ManifoldRust.Manifold.Boolean(
				oracleA, oracleB, op, ManifoldRust.BooleanEngine.Robust, rule);
			await Assert.That(oracleResult.Status).IsEqualTo(ManifoldStatus.NoError);

			// Anti-vacuity. Every row of the tables above is chosen to leave material
			// behind, so "the two agreed" can never mean "both produced nothing".
			await Assert.That(portResult.NumTri())
				.IsGreaterThan(0)
				.Because($"robust {op}/{rule} at ({dx}, {dy}, {dz}) should leave geometry to compare");

			await AssertSameGeometry(
				$"robust {op}/{rule} at ({dx}, {dy}, {dz})",
				portResult,
				oracleResult.GetMeshGL64());
		}

		/// <summary>
		/// The binding's robust importer indexes in <c>ulong</c> (its MeshGL64 shape), while
		/// the exact one takes <c>uint</c>; widening keeps both sides fed from one array.
		/// </summary>
		/// <param name="tris">The triangle indices.</param>
		/// <returns>The same indices as <c>ulong</c>.</returns>
		private static ulong[] Widen(uint[] tris)
		{
			ulong[] outTris = new ulong[tris.Length];
			for (int i = 0; i < tris.Length; i++)
			{
				outTris[i] = tris[i];
			}

			return outTris;
		}

		/// <summary>Every triangle reversed — a cube wound inside-out.</summary>
		/// <param name="tris">The triangle indices.</param>
		/// <returns>The reversed indices.</returns>
		private static uint[] Reversed(uint[] tris)
		{
			uint[] outTris = new uint[tris.Length];
			for (int i = 0; i < tris.Length; i += 3)
			{
				outTris[i] = tris[i];
				outTris[i + 1] = tris[i + 2];
				outTris[i + 2] = tris[i + 1];
			}

			return outTris;
		}

		/// <summary>
		/// The unit cube with every facet present twice, one vertex row per corner so the
		/// importer cannot weld the two sheets into one.
		/// </summary>
		/// <returns>The positions and indices.</returns>
		private static (double[] Verts, uint[] Tris) DoubledCover()
		{
			int corners = CubeTris.Length;
			double[] verts = new double[2 * corners * 3];
			uint[] tris = new uint[2 * corners];
			for (int copy = 0; copy < 2; copy++)
			{
				for (int i = 0; i < corners; i++)
				{
					uint v = CubeTris[i];
					int slot = (copy * corners) + i;
					verts[(slot * 3) + 0] = CubeVerts[(3 * v) + 0];
					verts[(slot * 3) + 1] = CubeVerts[(3 * v) + 1];
					verts[(slot * 3) + 2] = CubeVerts[(3 * v) + 2];
					tris[slot] = (uint)slot;
				}
			}

			return (verts, tris);
		}

		/// <summary>The port's counterpart of <c>Manifold.FromMesh64Robust</c>.</summary>
		/// <param name="verts">The flat positions.</param>
		/// <param name="tris">The triangle indices.</param>
		/// <returns>The imported manifold.</returns>
		private static ManifoldSharp.Manifold ImportRobust(double[] verts, uint[] tris)
		{
			MeshGL64 mesh = new MeshGL64();
			mesh.NumProp = 3;
			mesh.VertProperties = new List<double>(verts);
			mesh.TriVerts = new List<ulong>(tris.Length);
			foreach (uint t in tris)
			{
				mesh.TriVerts.Add(t);
			}

			return ManifoldSharp.Manifold.FromMeshGL64Robust(mesh);
		}

		private static ManifoldSharp.WindingRule ToPortRule(ManifoldRust.WindingRule rule)
		{
			switch (rule)
			{
				case ManifoldRust.WindingRule.Positive:
					return ManifoldSharp.WindingRule.Positive;
				case ManifoldRust.WindingRule.Nonzero:
					return ManifoldSharp.WindingRule.Nonzero;
				default:
					throw new ArgumentOutOfRangeException(nameof(rule), $"Unknown rule: {(int)rule}");
			}
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

		private static double[] Scale(double[] verts, double sx, double sy, double sz)
		{
			double[] outVerts = new double[verts.Length];
			for (int i = 0; i < verts.Length; i += 3)
			{
				outVerts[i] = verts[i] * sx;
				outVerts[i + 1] = verts[i + 1] * sy;
				outVerts[i + 2] = verts[i + 2] * sz;
			}

			return outVerts;
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
