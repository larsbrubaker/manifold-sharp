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

// Sdf.cs — port of sdf.rs, whose header reads:
//
//   Phase 16: SDF Mesh Generation — ported from C++ sdf.cpp (538 lines)
//
//   Implements Marching Tetrahedra on a body-centered cubic (BCC) grid with:
//   - ITP root-finding for precise surface location
//   - GridVert snapping to avoid short edges
//   - A faithful port of the C++ open-addressing HashTable (hashtable.h):
//     ComputeVerts/BuildTris iterate the table in SLOT order, so the hash
//     function, probing, sizing and resize protocol all determine the output
//     vertex numbering and triangle order. A std HashMap iterates in a random
//     order per process, which made the output nondeterministic.
//   - TetTri lookup tables for tetrahedra triangulation
//
// The Rust file is one module; the C# splits it three ways to stay under the
// 800-line cap, with no reordering inside each part:
//   Sdf.cs           — constants, LevelSet (the three marching phases), CreateTris
//   Sdf.Grid.cs      — TetTri tables, neighbors, encode/decode, FindSurface
//   Sdf.HashTable.cs — GridVert and the hashtable.h port
//
// The voxel fill below is the "SDF voxel fill" entry in the six sites
// docs/PORTING_PLAN.md blesses for parallelism: each voxel is an independent SDF
// evaluation written to its own index, so the parallel body is bit-identical to
// the sequential one (SdfVoxelFillIsBitIdenticalInParallel asserts it). It goes
// through Par.MaybeParMap with the Rust's own threshold (10_000), so going
// parallel was a change inside Par and not here.
//
// The obligation that buys: the caller's `sdf` delegate must be pure and
// thread-safe, since this is the only blessed site whose map calls user code.
// The Rust says so in the type (`F: Fn(Vec3) -> f64 + Sync`); C# has no such
// bound, so LevelSet's doc comment says it instead.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// The free functions of <c>sdf.rs</c> — level-set meshing of a signed distance
	/// function by Marching Tetrahedra on a body-centered cubic grid.
	/// </summary>
	public static partial class Sdf
	{
		// -------------------------------------------------------------------
		// Constants
		// -------------------------------------------------------------------

		/// <summary>Edge-vert marker for "this edge is crossed but has no vertex yet".</summary>
		internal const int KCrossing = -2;

		/// <summary>Edge-vert / moved-vert marker for "none".</summary>
		internal const int KNone = -1;

		/// <summary>Maximum fraction of spacing that a vert can move.</summary>
		private const double KS = 0.25;

		/// <summary>Corresponding approximate distance ratio bound.</summary>
		private const double KD = (1.0 / KS) - 1.0;

		/// <summary>Maximum number of opposed verts (of 7) to allow collapse.</summary>
		private const int KMaxOpposed = 3;

		/// <summary>
		/// The shift applied before indexing the dense voxel array, so that the -1 ring
		/// of neighbors lands at non-negative encoded indices.
		/// </summary>
		private static readonly IVec4 KVoxelOffset = new IVec4(1, 1, 1, 0);

		// -------------------------------------------------------------------
		// Main level_set function — port of C++ Manifold::LevelSet()
		// -------------------------------------------------------------------

		/// <summary>
		/// Generate a mesh from a signed distance function using Marching Tetrahedra
		/// on a body-centered cubic (BCC) grid.
		/// </summary>
		/// <remarks>
		/// <para>
		/// <paramref name="sdf"/> must be pure and thread-safe. The Rust spells that in the
		/// type — <c>F: Fn(Vec3) -&gt; f64 + Sync</c>, required whether or not the parallel
		/// feature is on — and C# has no such bound, so it is a documented obligation
		/// instead: with <see cref="ManifoldParallel.Enabled"/> set, the voxel fill
		/// evaluates it concurrently from thread-pool workers.
		/// </para>
		/// <para>
		/// <b>If <paramref name="sdf"/> throws, what a caller catches depends on the
		/// switch.</b> This is the only entry point in the port whose parallel map runs
		/// caller-supplied code, so it is the only place the distinction is reachable. One
		/// faulting evaluation propagates unwrapped — the same exception the sequential
		/// loop would have thrown, with its original stack, which is what
		/// <c>Par.RunParallel</c> exists to preserve. <em>Several</em> evaluations faulting
		/// concurrently has no sequential counterpart (the sequential loop stops at the
		/// first), so that case surfaces as an <see cref="AggregateException"/> over them
		/// rather than the port picking a winner. A caller that catches a specific type
		/// from its own sdf and enables parallelism must therefore also handle
		/// <see cref="AggregateException"/>; a caller whose sdf is total — which is the
		/// normal case, and the one every test here uses — can ignore all of this.
		/// </para>
		/// </remarks>
		/// <param name="sdf">
		/// Signed-distance function. Positive values are inside, negative outside. Must be
		/// pure and thread-safe; see the remarks.
		/// </param>
		/// <param name="bounds">Axis-aligned box defining the grid extent.</param>
		/// <param name="edgeLength">Approximate maximum edge length of output triangles.</param>
		/// <param name="level">Extract surface at this SDF value (default 0).</param>
		/// <param name="tolerance">
		/// Max distance from true surface per vertex. Negative = use interpolation only.
		/// </param>
		/// <returns>The extracted mesh, empty when nothing crosses the level.</returns>
		public static ManifoldImpl LevelSet(
			Func<Vec3, double> sdf,
			Box bounds,
			double edgeLength,
			double level,
			double tolerance)
		{
			ArgumentNullException.ThrowIfNull(sdf);

			if (edgeLength <= 0.0)
			{
				return new ManifoldImpl();
			}

			double tol = tolerance <= 0.0 ? double.PositiveInfinity : tolerance;

			Vec3 dim = bounds.Size();
			IVec3 gridSize = new IVec3(
				(int)((dim.X / edgeLength) + 1.0),
				(int)((dim.Y / edgeLength) + 1.0),
				(int)((dim.Z / edgeLength) + 1.0));
			Vec3 spacing = new Vec3(
				dim.X / MaxF64((double)gridSize.X - 1.0, 1.0),
				dim.Y / MaxF64((double)gridSize.Y - 1.0, 1.0),
				dim.Z / MaxF64((double)gridSize.Z - 1.0, 1.0));

			// C++ ComputeGridPow: axisPow(n) = CeilLog2(n + 2 + 1) — the bit widths
			// feed the index ENCODING, and encoded indices are the hash-table keys, so
			// this must match C++ exactly for slot order to match.
			IVec3 gridPow = new IVec3(
				AxisPow(gridSize.X),
				AxisPow(gridSize.Y),
				AxisPow(gridSize.Z));
			ulong maxIndex = EncodeIndex(
				new IVec4(gridSize.X + 2, gridSize.Y + 2, gridSize.Z + 2, 1),
				gridPow);

			Vec3 origin = bounds.Min;

			// Evaluate SDF at all grid points into a flat dense array indexed directly
			// by the encoded index, matching C++ `Vec<double> voxels(maxIndex)`. The
			// encoded indices are contiguous in [0, max_index), so a Vec gives O(1)
			// direct access; the previous HashMap<u64,f64> paid SipHash on millions of
			// keys (both inserts and the 14-neighbor lookups per grid vert), which
			// dominated runtime on fine grids (sdf_blobs/sdf_sphere_shell).
			// Each voxel is an independent SDF evaluation with an indexed write, so
			// the parallel path is bit-identical to sequential.
			//
			// The Rust's `max_index as usize` is the identity on a 64-bit target; C#
			// arrays are int-indexed, so the same value has to fit an int. `checked`
			// stands in for the Rust's allocation failure — a maxIndex past int range
			// is a 16 GB double[] that neither port can allocate.
			double[] voxels = Par.MaybeParMap(
				checked((int)maxIndex),
				10_000,
				idx =>
				{
					IVec4 gi = DecodeIndex((ulong)idx, gridPow);
					IVec4 giShifted = IVec4Add(
						gi,
						new IVec4(-KVoxelOffset.X, -KVoxelOffset.Y, -KVoxelOffset.Z, -KVoxelOffset.W));
					return BoundedSdf(giShifted, origin, spacing, gridSize, level, sdf);
				});

			double GetVoxel(IVec4 gi)
			{
				ulong key = EncodeIndex(IVec4Add(gi, KVoxelOffset), gridPow);
				return key < (ulong)voxels.Length ? voxels[(int)key] : 0.0;
			}

			// Table sizing per C++ LevelSet: dense cap of 2*maxIndex, sparse heuristic
			// of 10*sqrt(maxIndex); on overflow the whole NearSurface pass reruns with
			// a bigger table (sizing from the last vert's grid position).
			ulong tableSizeCap = ulong.MaxValue;
			ulong denseTableSize = maxIndex > tableSizeCap / 2 ? tableSizeCap : 2 * maxIndex;
			ulong sparseTableSize = Math.Min(tableSizeCap, (ulong)(10.0 * Math.Sqrt((double)maxIndex)));
			ulong tableSize = Math.Max(Math.Min(denseTableSize, sparseTableSize), 1);
			GridHashTable gridVerts = new GridHashTable(tableSize);
			List<Vec3> vertPos = new List<Vec3>();

			ulong surfaceMax = EncodeIndex(
				new IVec4(gridSize.X, gridSize.Y, gridSize.Z, 1),
				gridPow);
			while (true)
			{
				// Phase 1: NearSurface — identify grid verts near the surface
				for (ulong index = 0; index < surfaceMax; index++)
				{
					if (gridVerts.Full())
					{
						break;
					}

					IVec4 gridIndex = DecodeIndex(index, gridPow);
					if (gridIndex.X > gridSize.X
						|| gridIndex.Y > gridSize.Y
						|| gridIndex.Z > gridSize.Z)
					{
						continue;
					}

					GridVert gridVert = GridVert.Default();
					gridVert.Distance = GetVoxel(gridIndex);

					bool keep = false;
					double vMax = 0.0;
					int closestNeighbor = -1;
					int opposedVerts = 0;

					for (int i = 0; i < 7; i++)
					{
						double val = GetVoxel(Neighbor(gridIndex, i));
						double valOp = GetVoxel(Neighbor(gridIndex, i + 7));

						if (!gridVert.SameSide(val))
						{
							gridVert.EdgeVerts[i] = KCrossing;
							keep = true;
							if (!gridVert.SameSide(valOp))
							{
								opposedVerts += 1;
							}

							if (Math.Abs(val) > KD * Math.Abs(gridVert.Distance)
								&& Math.Abs(val) > Math.Abs(vMax))
							{
								vMax = val;
								closestNeighbor = i;
							}
						}
						else if (!gridVert.SameSide(valOp)
							&& Math.Abs(valOp) > KD * Math.Abs(gridVert.Distance)
							&& Math.Abs(valOp) > Math.Abs(vMax))
						{
							vMax = valOp;
							closestNeighbor = i + 7;
						}
					}

					// Snap to surface if possible
					if (closestNeighbor >= 0 && opposedVerts <= KMaxOpposed)
					{
						Vec3 gridPos = Position(gridIndex, origin, spacing);
						IVec4 neighborIndex = Neighbor(gridIndex, closestNeighbor);
						Vec3 pos = FindSurface(
							gridPos,
							gridVert.Distance,
							Position(neighborIndex, origin, spacing),
							vMax,
							tol,
							level,
							sdf);
						Vec3 delta = new Vec3(
							Math.Abs(pos.X - gridPos.X),
							Math.Abs(pos.Y - gridPos.Y),
							Math.Abs(pos.Z - gridPos.Z));
						if (delta.X < KS * spacing.X
							&& delta.Y < KS * spacing.Y
							&& delta.Z < KS * spacing.Z)
						{
							int idx = vertPos.Count;
							vertPos.Add(Bound(pos, origin, spacing, gridSize));
							gridVert.MovedVert = idx;
							for (int j = 0; j < 7; j++)
							{
								if (gridVert.EdgeVerts[j] == KCrossing)
								{
									gridVert.EdgeVerts[j] = idx;
								}
							}

							keep = true;
						}
					}
					else
					{
						for (int j = 0; j < 7; j++)
						{
							gridVert.EdgeVerts[j] = KNone;
						}
					}

					if (keep)
					{
						gridVerts.Insert(index, gridVert);
					}
				}

				if (gridVerts.Full())
				{
					// Resize per C++: estimate the fill ratio from how far through the
					// grid the last allocated vert got, then rerun from scratch.
					if (vertPos.Count == 0)
					{
						// The Rust's `.expect("table full before any vert")` at sdf.rs:748,
						// which matches the Rust panic; reachable when a table fills purely
						// from crossings that never snap. A step SDF does exactly that —
						// every |val| equals |distance|, so the `> KD * |distance|` test
						// never fires and closestNeighbor stays -1, while the crossings
						// themselves keep setting `keep` and filling the table.
						throw new InvalidOperationException("table full before any vert");
					}

					Vec3 lastVert = vertPos[vertPos.Count - 1];
					ulong lastIndex = EncodeIndex(
						new IVec4(
							(int)((lastVert.X - origin.X) / spacing.X),
							(int)((lastVert.Y - origin.Y) / spacing.Y),
							(int)((lastVert.Z - origin.Z) / spacing.Z),
							1),
						gridPow);
					double ratio = (double)maxIndex / (double)lastIndex;
					if (ratio > 1000.0)
					{
						tableSize *= 2;
					}
					else
					{
						tableSize = (ulong)((double)tableSize * ratio);
					}

					gridVerts = new GridHashTable(tableSize);
					vertPos.Clear();
				}
				else
				{
					break;
				}
			}

			// Phase 2: ComputeVerts — create edge-crossing vertices, iterating the
			// hash table in SLOT order (this fixes the output vert numbering).
			for (int slot = 0; slot < gridVerts.Size; slot++)
			{
				if (gridVerts.Keys[slot] == GridHashTable.KOpen)
				{
					continue;
				}

				ulong baseKey = gridVerts.Keys[slot];
				GridVert gridVertClone = gridVerts.Values[slot];
				if (gridVertClone.HasMoved())
				{
					continue;
				}

				IVec4 gridIndex = DecodeIndex(baseKey, gridPow);
				Vec3 pos = Position(gridIndex, origin, spacing);

				// The Rust collects the edge assignments and applies them after the loop
				// because the borrow checker forbids mutating the table while `nv` borrows
				// it. C# would allow the interleaved write, and the output would not change:
				// no forward neighbor offset is (0,0,0,0), so no iteration ever reads the
				// slot this loop writes. The deferred list is kept for structural fidelity
				// to the Rust, not because interleaving is observable.
				List<(int Edge, int Vert)> updates = new List<(int, int)>();
				for (int i = 0; i < 7; i++)
				{
					IVec4 neighborIndex = Neighbor(gridIndex, i);
					ulong neighborKey = EncodeIndex(neighborIndex, gridPow);
					GridVert nv = gridVerts.Get(neighborKey);

					double val = double.IsFinite(nv.Distance)
						? nv.Distance
						: GetVoxel(neighborIndex);

					if (gridVertClone.SameSide(val))
					{
						continue;
					}

					if (nv.HasMoved())
					{
						updates.Add((i, nv.MovedVert));
						continue;
					}

					Vec3 newPos = FindSurface(
						pos,
						gridVertClone.Distance,
						Position(neighborIndex, origin, spacing),
						val,
						tol,
						level,
						sdf);
					int idx = vertPos.Count;
					vertPos.Add(Bound(newPos, origin, spacing, gridSize));
					updates.Add((i, idx));
				}

				foreach ((int i, int idx) in updates)
				{
					gridVerts.Values[slot].EdgeVerts[i] = idx;
				}
			}

			// Phase 3: BuildTris — generate triangles from tetrahedra, in slot order.
			List<IVec3> triVerts = new List<IVec3>();

			for (int slot = 0; slot < gridVerts.Size; slot++)
			{
				ulong baseKey = gridVerts.Keys[slot];
				if (baseKey == GridHashTable.KOpen)
				{
					continue;
				}

				GridVert baseVert = gridVerts.Values[slot];
				IVec4 baseIndex = DecodeIndex(baseKey, gridPow);

				IVec4 leadIndex = baseIndex;
				if (leadIndex.W == 0)
				{
					leadIndex.W = 1;
				}
				else
				{
					leadIndex.X += 1;
					leadIndex.Y += 1;
					leadIndex.Z += 1;
					leadIndex.W = 0;
				}

				// 6 tetrahedra around the (1,1,1) edge
				IVec4 tet = new IVec4(baseVert.NeighborInside(0), baseVert.Inside(), -2, -2);
				IVec4 thisIndex = baseIndex;
				thisIndex.X += 1;
				GridVert thisVert = gridVerts.Get(EncodeIndex(thisIndex, gridPow));

				tet[2] = baseVert.NeighborInside(1);
				for (int i = 0; i < 3; i++)
				{
					thisIndex = leadIndex;
					thisIndex[Prev3(i)] -= 1;

					GridVert nextVert = thisIndex[Prev3(i)] < 0
						? GridVert.Default()
						: gridVerts.Get(EncodeIndex(thisIndex, gridPow));
					tet[3] = baseVert.NeighborInside(Prev3(i) + 4);

					int[] edges1 = new int[6]
					{
						baseVert.EdgeVerts[0],
						baseVert.EdgeVerts[i + 1],
						nextVert.EdgeVerts[Next3(i) + 4],
						nextVert.EdgeVerts[Prev3(i) + 1],
						thisVert.EdgeVerts[i + 4],
						baseVert.EdgeVerts[Prev3(i) + 4],
					};
					thisVert = nextVert;
					CreateTris(triVerts, tet, edges1);

					thisIndex = baseIndex;
					thisIndex[Next3(i)] += 1;
					nextVert = gridVerts.Get(EncodeIndex(thisIndex, gridPow));
					tet[2] = tet[3];
					tet[3] = baseVert.NeighborInside(Next3(i) + 1);

					int[] edges2 = new int[6]
					{
						baseVert.EdgeVerts[0],
						edges1[5],
						thisVert.EdgeVerts[i + 4],
						nextVert.EdgeVerts[Next3(i) + 4],
						edges1[3],
						baseVert.EdgeVerts[Next3(i) + 1],
					};
					thisVert = nextVert;
					CreateTris(triVerts, tet, edges2);

					tet[2] = tet[3];
				}
			}

			// Build the mesh
			if (triVerts.Count == 0 || vertPos.Count == 0)
			{
				return new ManifoldImpl();
			}

			ManifoldImpl result = new ManifoldImpl();
			result.VertPos = vertPos;
			result.CreateHalfedges(triVerts, Array.Empty<IVec3>());

			// edge_op's cleanup_topology, on the freshly marched voxel surface — the
			// Phase 7 entry in EdgeOpTests' deferred-callers table.
			EdgeOp.CleanupTopology(result);
			result.RemoveUnreferencedVerts();
			result.InitializeOriginal();
			result.CalculateBBox();
			result.SetEpsilon(-1.0, false);
			result.SortGeometry();
			result.SetNormalsAndCoplanar();
			return result;
		}

		/// <summary>
		/// Simple wrapper that calls <see cref="LevelSet"/> with default level=0 and
		/// tolerance=-1.
		/// </summary>
		/// <remarks>
		/// <see cref="LevelSet"/>'s remarks apply unchanged, including what a caller
		/// catches when <paramref name="sdf"/> throws under parallelism.
		/// </remarks>
		/// <param name="sdf">
		/// Signed-distance function. Positive values are inside. Must be pure and
		/// thread-safe, for the reason <see cref="LevelSet"/>'s remarks give.
		/// </param>
		/// <param name="bounds">Axis-aligned box defining the grid extent.</param>
		/// <param name="edgeLength">Approximate maximum edge length of output triangles.</param>
		/// <returns>The extracted mesh.</returns>
		public static ManifoldImpl LevelSetSimple(Func<Vec3, double> sdf, Box bounds, double edgeLength)
		{
			return LevelSet(sdf, bounds, edgeLength, 0.0, -1.0);
		}

		/// <summary>
		/// C++ ComputeGridPow's per-axis width, <c>CeilLog2(n + 3)</c>. Kept as its own
		/// function because the "+ 3" is the documented exactness item: it is C++'s
		/// <c>n + 2 + 1</c>, and one off here renumbers every output vertex.
		/// </summary>
		/// <param name="n">The grid point count along one axis.</param>
		/// <returns>The bit width for that axis.</returns>
		private static int AxisPow(int n)
		{
			// Rust `n as usize` on an i32 wraps; grid sizes are positive here, and the
			// widening is written out so the intent survives a negative n.
			return CeilLog2(unchecked((ulong)(long)n + 3));
		}

		/// <summary>
		/// Emit one triangle from a tetrahedron's edge crossings, dropping degenerate
		/// and absent cases.
		/// </summary>
		/// <param name="triVerts">The triangle list to append to.</param>
		/// <param name="tri">The TetTri entry: three edge indices, or -1 for none.</param>
		/// <param name="edges">The six tetrahedron edge vertices.</param>
		private static void CreateTri(List<IVec3> triVerts, IVec3 tri, int[] edges)
		{
			if (tri[0] < 0)
			{
				return;
			}

			IVec3 verts = new IVec3(edges[tri[0]], edges[tri[1]], edges[tri[2]]);
			if (verts[0] == verts[1] || verts[1] == verts[2] || verts[2] == verts[0])
			{
				return;
			}

			triVerts.Add(verts);
		}

		/// <summary>
		/// Triangulate one tetrahedron: pack its four corner signs into the TetTri index
		/// and emit the zero, one or two triangles the tables give.
		/// </summary>
		/// <param name="triVerts">The triangle list to append to.</param>
		/// <param name="tet">The four corner inside-signs.</param>
		/// <param name="edges">The six tetrahedron edge vertices.</param>
		private static void CreateTris(List<IVec3> triVerts, IVec4 tet, int[] edges)
		{
			int i = (tet[0] > 0 ? 1 : 0)
				+ (tet[1] > 0 ? 2 : 0)
				+ (tet[2] > 0 ? 4 : 0)
				+ (tet[3] > 0 ? 8 : 0);
			CreateTri(triVerts, TetTri0(i), edges);
			CreateTri(triVerts, TetTri1(i), edges);
		}
	}
}
