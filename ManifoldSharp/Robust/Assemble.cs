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

// Assemble.cs — port of robust/assemble.rs, whose header reads:
//
//   From tagged pieces to a Manifold (paper §7.5, output browse).
//
//   The selected pieces are welded on their exact rational coordinates first
//   (so identical points are identical regardless of construction path), then
//   each unique vertex rounds once to the nearest f64
//   (robust/exact/rational.rs) and the result re-enters the library through
//   the robust MeshGL64 import: manifold results get the full strict pipeline
//   (normals, degenerate removal, sorting — the same post-processing the
//   exact engine's outputs receive), while legitimately non-manifold results
//   (booleans of non-manifold inputs) are retained as soup impls, ready for
//   chained robust operations.
//
//   When either operand carries vertex properties (colors, UVs, …), each
//   output vertex's properties are barycentrically interpolated from its
//   originating input triangle — exact rational barycentrics, one f64
//   rounding — so constant per-operand properties survive exactly and
//   interpolated ones agree with the exact engine to double precision.
//   Coincident vertices with different properties stay separate property
//   vertices linked by merge vectors, mirroring the exact engine's MeshGL
//   output shape.
//
// ── The one rounding path ────────────────────────────────────────────────────
// Nothing here re-rounds a rational position: `vertsF64` already holds the
// correctly rounded image of every interned vertex (built once by
// `VertInterner`, through the exact tier's RatToF64), and this file reads that
// table. The only RatToF64 calls below are on the three barycentric *weights*,
// which have no cached form.
//
// ── Named `AssembleFunctions` ────────────────────────────────────────────────
// The module's own function is `assemble`, so a class named `Assemble` could not
// carry it (CS0542). Same call CdtFunctions and ArrangementFunctions made.

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Robust
{
	/// <summary>
	/// Per-operand property data for interpolation. <c>Props[m]</c> is flattened as
	/// <c>Props[m][((3 * tri) + corner) * NumProp[m] + channel]</c>, aligned with the
	/// operand's soup triangle order (the <see cref="Piece.Tri"/> indexing).
	/// </summary>
	public sealed class PropCtx
	{
		/// <summary>Property channel count per operand.</summary>
		public IReadOnlyList<int> NumProp;

		/// <summary>Each operand's soup triangles.</summary>
		public IReadOnlyList<IReadOnlyList<Vec3[]>> Tris;

		/// <summary>Each operand's flattened corner properties.</summary>
		public IReadOnlyList<IReadOnlyList<double>> Props;

		/// <summary>Creates a property context (the Rust's struct literal).</summary>
		/// <param name="numProp">Property channel count per operand.</param>
		/// <param name="tris">Each operand's soup triangles.</param>
		/// <param name="props">Each operand's flattened corner properties.</param>
		public PropCtx(
			IReadOnlyList<int> numProp,
			IReadOnlyList<IReadOnlyList<Vec3[]>> tris,
			IReadOnlyList<IReadOnlyList<double>> props)
		{
			this.NumProp = numProp;
			this.Tris = tris;
			this.Props = props;
		}

		/// <summary>The output's property channel count.</summary>
		/// <returns>The larger of the two operands' channel counts.</returns>
		public int OutNumProp()
		{
			return Math.Max(this.NumProp[0], this.NumProp[1]);
		}
	}

	/// <summary>
	/// The free functions of <c>robust/assemble.rs</c>: welding the selected pieces on
	/// exact coordinates and re-importing them as a Manifold.
	/// </summary>
	public static class AssembleFunctions
	{
		/// <summary>
		/// Build the output manifold from every piece whose index passes
		/// <paramref name="select"/>.
		/// </summary>
		/// <remarks>
		/// <paramref name="verts"/> / <paramref name="vertsF64"/> are the graph's interned
		/// tables: exact coordinates for property interpolation, cached correctly rounded
		/// positions for the output — no per-vertex rational rounding here. With a
		/// <see cref="PropCtx"/> whose operands carry properties, output vertices get
		/// interpolated properties; otherwise the output is positions-only and
		/// byte-identical to the pre-property behavior.
		/// </remarks>
		/// <param name="pieces">Every candidate piece.</param>
		/// <param name="verts">The interned exact vertices.</param>
		/// <param name="vertsF64">Their correctly rounded f64 images.</param>
		/// <param name="select">Which piece indices to keep.</param>
		/// <param name="props">The property context, or null for positions only.</param>
		/// <returns>The assembled manifold.</returns>
		public static Manifold Assemble(
			IReadOnlyList<Piece> pieces,
			IReadOnlyList<R3> verts,
			IReadOnlyList<Vec3> vertsF64,
			Func<int, bool> select,
			PropCtx? props)
		{
			ArgumentNullException.ThrowIfNull(pieces);
			ArgumentNullException.ThrowIfNull(verts);
			ArgumentNullException.ThrowIfNull(vertsF64);
			ArgumentNullException.ThrowIfNull(select);
			int outProp = props == null ? 0 : props.OutNumProp();

			List<Piece> selected = new List<Piece>();
			for (int pi = 0; pi < pieces.Count; pi++)
			{
				if (select(pi))
				{
					selected.Add(pieces[pi]);
				}
			}

			if (selected.Count == 0)
			{
				return Manifold.Empty();
			}

			// A boundary that touches itself along an edge carries more than two
			// half-edges on that vertex-id edge, which the import's id-based pairing
			// can only guess at. Splitting the pinched vertices into one copy per
			// geometric fan makes that pairing reproduce the geometry. The plan is
			// `null` — and everything below unchanged — for every mesh without such
			// an edge.
			UVec3[] tris = new UVec3[selected.Count];
			for (int i = 0; i < selected.Count; i++)
			{
				tris[i] = selected[i].Vi;
			}

			uint[]? plan = Pairing.PlanVertexSplits(tris, new VertTables(verts, vertsF64));

			// Property-vertex identity: interned position id + fan copy + property
			// bit pattern (id equality is exact geometric identity — see
			// VertInterner).
			//
			// Fx hashing (unseeded): probe-only map — output vertex ids come from
			// `vert_order.len()` at first sight, i.e. from triangle/corner order.
			Dictionary<VertKey, ulong> vertIndex = new Dictionary<VertKey, ulong>();
			List<(uint Vid, uint Split, double[] Pvals)> vertOrder
				= new List<(uint Vid, uint Split, double[] Pvals)>();
			List<ulong> triVerts = new List<ulong>();

			for (int t = 0; t < selected.Count; t++)
			{
				Piece piece = selected[t];
				for (int c = 0; c < 3; c++)
				{
					uint vid = piece.Vi[c];
					uint split = plan == null ? 0u : plan[(3 * t) + c];
					double[] pvals = props != null && outProp > 0
						? InterpolateProps(props, piece, verts[(int)vid], outProp)
						: Array.Empty<double>();
					VertKey key = new VertKey(vid, split, pvals);
					ulong next = (ulong)vertOrder.Count;
					if (!vertIndex.TryGetValue(key, out ulong id))
					{
						vertOrder.Add((vid, split, pvals));
						id = next;
						vertIndex.Add(key, id);
					}

					triVerts.Add(id);
				}
			}

			int stride = 3 + outProp;
			MeshGL64 mesh = new MeshGL64();
			mesh.NumProp = (ulong)stride;
			mesh.VertProperties = new List<double>(stride * vertOrder.Count);
			foreach ((uint Vid, uint Split, double[] Pvals) v in vertOrder)
			{
				Vec3 p = vertsF64[(int)v.Vid];
				mesh.VertProperties.Add(p.X);
				mesh.VertProperties.Add(p.Y);
				mesh.VertProperties.Add(p.Z);
				mesh.VertProperties.AddRange(v.Pvals);
			}

			mesh.TriVerts = triVerts;

			// Coincident positions with different properties are distinct property
			// vertices; merge vectors tell the import they are topologically one.
			// Keyed on the fan copy too, so split copies of a pinched vertex stay
			// separate geometric vertices.
			if (outProp > 0)
			{
				// Probe-only; merge pairs are emitted in `vertOrder` index order.
				Dictionary<(uint Vid, uint Split), ulong> byPos
					= new Dictionary<(uint Vid, uint Split), ulong>();
				for (int i = 0; i < vertOrder.Count; i++)
				{
					(uint Vid, uint Split) key = (vertOrder[i].Vid, vertOrder[i].Split);
					if (byPos.TryGetValue(key, out ulong first))
					{
						mesh.MergeFromVert.Add((ulong)i);
						mesh.MergeToVert.Add(first);
					}
					else
					{
						byPos.Add(key, (ulong)i);
					}
				}
			}

			// The robust import handles everything rounding can produce: verts that
			// collapsed to identical f64 positions, exactly-degenerate triangles,
			// and non-manifold connectivity (kept as a soup impl).
			Manifold outManifold = Manifold.FromMeshGL64RobustAssembled(mesh);

			// Manifold results get the same topology simplification the exact
			// engine's boolean_result applies: without it the CDT's coplanar
			// subdivision vertices survive and the output carries more (redundant)
			// vertices than the exact engine produces for the same inputs.
			//
			// The one stage held back is `swap_degenerates` — the pieces of
			// `simplify_topology` are composed here without it, matching the import
			// above. See docs/CPP_DIVERGENCES.md entry 1: a boolean result
			// legitimately contains coplanar antiparallel adjacencies, and the
			// flood-filled face normals those produce make the swap misclassify
			// large valid triangles and physically move the surface (−2.5e-3 of the
			// volume on Thingi10K #301921 ∪ rotated-self).
			if (outManifold.Status() == Error.NoError
				&& !outManifold.AsImpl().IsSoup
				&& !outManifold.IsEmpty())
			{
				ManifoldImpl imp = outManifold.IntoImpl();
				EdgeOp.CleanupTopology(imp);
				EdgeOp.CollapseShortEdges(imp, 0);
				EdgeOp.CollapseColinearEdges(imp, 0);
				FaceOp.CalculateVertNormals(imp);
				imp.RemoveUnreferencedVerts();
				imp.CalculateBBox();
				imp.SortGeometry();
				return Manifold.FromImpl(imp);
			}

			return outManifold;
		}

		/// <summary>
		/// Exact barycentric coordinates of <paramref name="p"/> on triangle
		/// <paramref name="tri"/> (p must lie on the triangle's plane), computed in the
		/// dominant-axis projection. The three weights sum to exactly 1.
		/// </summary>
		/// <param name="p">The point.</param>
		/// <param name="tri">The triangle's three exact corners.</param>
		/// <returns>The three exact weights.</returns>
		private static BigRational[] BarycentricR(R3 p, R3[] tri)
		{
			R3 n = Predicates.TriNormalR(tri[0], tri[1], tri[2]);
			int axis = TriTri.DominantAxis(n);
			R2 p2 = p.ProjectDrop(axis);
			R2 a = tri[0].ProjectDrop(axis);
			R2 b = tri[1].ProjectDrop(axis);
			R2 c = tri[2].ProjectDrop(axis);
			BigRational total = b.Sub(a).Cross(c.Sub(a));
			BigRational w0 = b.Sub(p2).Cross(c.Sub(p2)) / total;
			BigRational w1 = c.Sub(p2).Cross(a.Sub(p2)) / total;
			BigRational w2 = Backend.RatOne() - w0 - w1;
			return new[] { w0, w1, w2 };
		}

		/// <summary>
		/// Interpolated properties (padded to <paramref name="outCount"/> channels) for
		/// piece vertex <paramref name="v"/>.
		/// </summary>
		/// <param name="ctx">The property context.</param>
		/// <param name="piece">The piece the vertex belongs to.</param>
		/// <param name="v">The vertex's exact position.</param>
		/// <param name="outCount">The output channel count.</param>
		/// <returns>The interpolated property row.</returns>
		private static double[] InterpolateProps(PropCtx ctx, Piece piece, R3 v, int outCount)
		{
			int m = piece.Mesh;
			int np = ctx.NumProp[m];
			double[] result = new double[outCount];
			if (np == 0)
			{
				return result;
			}

			IReadOnlyList<double> flat = ctx.Props[m];

			// NARROWING (usize -> int): the Rust's `3 * piece.tri * np` is usize. Here it
			// is an int product that wraps unchecked, but it indexes `flat`, a
			// List<double> whose own Count is an int — so any input big enough to wrap has
			// already failed to allocate the property table this reads.
			int baseOffset = 3 * piece.Tri * np;
			double Corner(int i, int k) => flat[baseOffset + (i * np) + k];

			// Constant-per-face channels pass through exactly, no arithmetic.
			bool allConst = true;
			for (int k = 0; k < np; k++)
			{
				if (Corner(0, k) != Corner(1, k) || Corner(0, k) != Corner(2, k))
				{
					allConst = false;
					break;
				}
			}

			if (allConst)
			{
				for (int k = 0; k < np; k++)
				{
					result[k] = Corner(0, k);
				}

				return result;
			}

			Vec3[] t = ctx.Tris[m][piece.Tri];
			R3[] corners = { R3.FromVec3(t[0]), R3.FromVec3(t[1]), R3.FromVec3(t[2]) };
			BigRational[] w = BarycentricR(v, corners);
			double[] wf =
			{
				Rational.RatToF64(w[0]),
				Rational.RatToF64(w[1]),
				Rational.RatToF64(w[2]),
			};
			for (int k = 0; k < np; k++)
			{
				result[k] = Corner(0, k) == Corner(1, k) && Corner(0, k) == Corner(2, k)
					? Corner(0, k)
					: (wf[0] * Corner(0, k)) + (wf[1] * Corner(1, k)) + (wf[2] * Corner(2, k));
			}

			return result;
		}

		/// <summary>
		/// The Rust's <c>type Key = (u32, u32, Vec&lt;u64&gt;)</c>: interned position id, fan
		/// copy, and the property row's bit patterns.
		/// </summary>
		/// <remarks>
		/// A dedicated key type rather than a value tuple because C# tuple equality on a
		/// <c>double[]</c>/<c>ulong[]</c> member is by reference, which would make every
		/// vertex unique and defeat the weld. The bits are captured at construction so the
		/// key stays immutable even though the caller keeps the property row.
		/// </remarks>
		private readonly struct VertKey : IEquatable<VertKey>
		{
			private readonly uint vid;

			private readonly uint split;

			private readonly ulong[] bits;

			private readonly int hash;

			/// <summary>Builds a key from an interned id, a fan copy and a property row.</summary>
			/// <param name="vid">The interned position id.</param>
			/// <param name="split">The fan copy index.</param>
			/// <param name="pvals">The property row.</param>
			public VertKey(uint vid, uint split, double[] pvals)
			{
				this.vid = vid;
				this.split = split;
				this.bits = new ulong[pvals.Length];
				HashCode h = default;
				h.Add(vid);
				h.Add(split);
				for (int i = 0; i < pvals.Length; i++)
				{
					this.bits[i] = BitConverter.DoubleToUInt64Bits(pvals[i]);
					h.Add(this.bits[i]);
				}

				this.hash = h.ToHashCode();
			}

			/// <inheritdoc/>
			public bool Equals(VertKey other)
			{
				if (this.vid != other.vid || this.split != other.split
					|| this.bits.Length != other.bits.Length)
				{
					return false;
				}

				for (int i = 0; i < this.bits.Length; i++)
				{
					if (this.bits[i] != other.bits[i])
					{
						return false;
					}
				}

				return true;
			}

			/// <inheritdoc/>
			public override bool Equals(object? obj)
			{
				return obj is VertKey other && this.Equals(other);
			}

			/// <inheritdoc/>
			public override int GetHashCode()
			{
				return this.hash;
			}
		}
	}
}
