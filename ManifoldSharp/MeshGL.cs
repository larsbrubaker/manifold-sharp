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

// MeshGL.cs — port of types_meshgl.rs — MeshGLP: the GL-style interchange mesh
// representation.
//
// Ported from include/manifold/manifold.h (MeshGLP) and src/manifold.cpp
// (MeshGL::Merge, MeshGL::UpdateNormals). Extracted from types.rs, which
// re-exports MeshGLP / MeshGL / MeshGL64 so external paths
// (`crate::types::MeshGL`, ...) are unchanged. Meshes enter and leave the
// library in this form; ManifoldImpl.cs / ManifoldMeshGL.cs convert between it
// and the internal halfedge representation.
//
// ── Why two concrete classes and not one generic type ────────────────────────
// `MeshGLP<Precision, Index>` is the crate's only type generic over traits:
// `MeshPrecision` (f32/f64) and `MeshIndex` (u32/u64). It has exactly two
// instantiations, aliased `MeshGL` and `MeshGL64`, and this port spells both of
// them out by hand instead of reaching for `INumber<T>` generic math. Three
// reasons, in order of weight:
//
//   1. The two methods that carry real behaviour — Merge and UpdateNormals —
//      are implemented in the Rust *only* on `MeshGLP<f32, u32>`. There is no
//      generic version of them to transcribe. Writing them against a concrete
//      `float` / `uint` class is the literal transcription; writing them
//      against a type parameter would be a translation.
//   2. The exactness bar is per instantiation. `vert_properties` is stored f32
//      here, and every narrowing in UpdateNormals is a real `(float)` cast at
//      the same place the Rust has `as f32`. Under generic math those casts
//      become `T.CreateTruncating`, whose rounding and out-of-range behaviour
//      would have to be re-argued at every site.
//   3. C# cannot implement an interface on `float`, so a faithful generic port
//      needs witness structs and four type parameters
//      (`MeshGLP<float, F32Precision, uint, U32Index>`) — further from the Rust
//      than the duplication it avoids, and unreadable at every use site.
//
// What was lost: the two genuinely generic functions in manifold_meshgl.rs
// (`from_mesh_impl<P, I>` and `get_mesh_gl_impl<P, I>`) have no single C# body to
// land in, so the choice was "duplicate them" or "introduce a narrow conversion
// interface over these two classes". Phase 6 SETTLED IT: the interface, which is
// `IMeshGLAccess` in MeshGLAccess.cs. Each of the two bodies stays single, and
// every `P::from_f64` / `I::from_usize` / `to_u64` in the Rust becomes a call on
// that interface, so the f32 instantiation narrows at exactly the sites the C++
// float template narrows at. MeshGLAccess.cs's header argues the choice; the
// deciding point is that two near-identical 230-line bodies differing only at the
// conversion sites is the shape of change that goes wrong silently, which the
// exactness bar cannot afford.
//
// The trait doc comments are kept as the *conversion contract* of the two
// classes, since that is where the knowledge lived:
//
//   MeshPrecision — conversions to and from the kernel's internal f64 happen
//     through this trait, so the f32 instantiation narrows exactly where the
//     C++ float instantiation does and the f64 instantiation is lossless end to
//     end. `IS_SINGLE` is true for f32 only: the exported tolerance is floored
//     at `f32::EPSILON * bbox.scale()` only for single precision, matching the
//     `std::is_same<Precision, float>` checks in the C++ template.
//   MeshIndex — the kernel itself indexes with 32 bits for both instantiations
//     (as the C++ does — its import casts every index to `uint32_t`), so u64
//     indices wider than 32 bits truncate on import exactly like the C++
//     `static_cast`. Conversion from the kernel's `int` indices on export
//     matches C++ integral conversion: bit-pattern for u32, sign-extension for
//     u64.
//
// ── Vec<T> as List<T> ────────────────────────────────────────────────────────
// Every field here is a Rust `Vec` that is *built by pushing* — the importer
// and exporter in manifold_meshgl.rs append to them, and Merge clears and
// refills the two merge vectors — so they are `List<T>`, per the same reasoning
// the Types.cs header gives for the polygon aliases. They are lists of
// primitives, so the List-of-struct indexer trap does not apply.
//
// The `[P; 3]` / `[I; 3]` / `[P; 4]` accessor returns become C# value tuples
// rather than arrays: a Rust fixed-size array is a value on the stack, and a
// tuple is the C# construct with that property. `float[3]` would allocate once
// per vertex inside Merge's loops.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// GL-style mesh representation, single precision with 32-bit indices — the Rust
	/// <c>MeshGLP&lt;f32, u32&gt;</c>, aliased <c>MeshGL</c>. The standard form for
	/// graphics.
	/// </summary>
	public sealed class MeshGL
	{
		// Rust `f32::EPSILON as f64` — 2^-23, the gap between 1.0f and the next float.
		// C# `float.Epsilon` is the smallest *subnormal* float and is wrong here; see
		// the f64 counterpart of this rule in CLAUDE.md.
		private const double F32Epsilon = 1.1920928955078125E-07;

		/// <summary>Creates an empty mesh — the Rust derived <c>Default</c>.</summary>
		public MeshGL()
		{
			this.NumProp = 0;
			this.VertProperties = new List<float>();
			this.TriVerts = new List<uint>();
			this.MergeFromVert = new List<uint>();
			this.MergeToVert = new List<uint>();
			this.RunIndex = new List<uint>();
			this.RunOriginalId = new List<uint>();
			this.RunTransform = new List<float>();
			this.FaceId = new List<uint>();
			this.HalfedgeTangent = new List<float>();
			this.RunFlags = new List<byte>();
			this.Tolerance = 0.0f;
		}

		/// <summary>Number of properties per vertex, always &gt;= 3.</summary>
		public uint NumProp { get; set; }

		/// <summary>Flat interleaved vertex properties: [x, y, z, ...] x num_verts.</summary>
		public List<float> VertProperties { get; set; }

		/// <summary>Triangle vertex indices, 3 per triangle (CCW from outside).</summary>
		public List<uint> TriVerts { get; set; }

		/// <summary>Optional: merge-from vertex indices.</summary>
		public List<uint> MergeFromVert { get; set; }

		/// <summary>Optional: merge-to vertex indices.</summary>
		public List<uint> MergeToVert { get; set; }

		/// <summary>Optional: run start indices into triVerts.</summary>
		public List<uint> RunIndex { get; set; }

		/// <summary>Optional: original mesh ID per run.</summary>
		public List<uint> RunOriginalId { get; set; }

		/// <summary>Optional: 3x4 column-major transform per run (12 elements each).</summary>
		public List<float> RunTransform { get; set; }

		/// <summary>Optional: source face ID per triangle.</summary>
		public List<uint> FaceId { get; set; }

		/// <summary>Optional: halfedge tangent vectors (4 per halfedge).</summary>
		public List<float> HalfedgeTangent { get; set; }

		/// <summary>Optional: per-run flags; 1 = backside (normals need flipping).</summary>
		public List<byte> RunFlags { get; set; }

		/// <summary>Tolerance for mesh simplification.</summary>
		public float Tolerance { get; set; }

		// NARROWING AUDIT (uint -> int, this method and GetVertPos below). The Rust
		// computes these in `usize` via `self.num_prop.to_u64() as usize`; here NumProp is
		// the u32 the type parameter names, narrowed to int to index a List. C#'s default
		// unchecked conversion wraps, matching Rust `as`, so the two agree — but they
		// agree on a wrapped value only for NumProp > int.MaxValue, which cannot arise:
		// the property arrays are List<T>, so a mesh whose stride exceeded int.MaxValue
		// could not have been built in the first place. A divergence here needs
		// NumProp > 2^31, i.e. an input this type cannot represent.
		/// <summary>The number of vertices, i.e. property groups.</summary>
		/// <returns>The vertex count, or 0 when there are no properties per vertex.</returns>
		public int NumVert()
		{
			if (this.NumProp == 0)
			{
				return 0;
			}

			return this.VertProperties.Count / (int)this.NumProp;
		}

		/// <summary>The number of triangles.</summary>
		/// <returns>The triangle count.</returns>
		public int NumTri()
		{
			return this.TriVerts.Count / 3;
		}

		/// <summary>The position (property slots 0..2) of one vertex.</summary>
		/// <param name="v">The vertex index.</param>
		/// <returns>The x, y and z property values.</returns>
		public (float X, float Y, float Z) GetVertPos(int v)
		{
			int offset = v * (int)this.NumProp;
			return (
				this.VertProperties[offset],
				this.VertProperties[offset + 1],
				this.VertProperties[offset + 2]);
		}

		/// <summary>The three vertex indices of one triangle.</summary>
		/// <param name="t">The triangle index.</param>
		/// <returns>The triangle's three vertex indices.</returns>
		public (uint A, uint B, uint C) GetTriVerts(int t)
		{
			int offset = 3 * t;
			return (this.TriVerts[offset], this.TriVerts[offset + 1], this.TriVerts[offset + 2]);
		}

		/// <summary>The tangent vector of one halfedge.</summary>
		/// <param name="h">The halfedge index.</param>
		/// <returns>The tangent's four components.</returns>
		public (float X, float Y, float Z, float W) GetTangent(int h)
		{
			int offset = 4 * h;
			return (
				this.HalfedgeTangent[offset],
				this.HalfedgeTangent[offset + 1],
				this.HalfedgeTangent[offset + 2],
				this.HalfedgeTangent[offset + 3]);
		}

		/// <summary>The Rust derived <c>Clone</c>: an independent copy of every vector.</summary>
		/// <returns>A deep copy of this mesh.</returns>
		public MeshGL Clone()
		{
			MeshGL copy = new MeshGL();
			copy.NumProp = this.NumProp;
			copy.VertProperties = new List<float>(this.VertProperties);
			copy.TriVerts = new List<uint>(this.TriVerts);
			copy.MergeFromVert = new List<uint>(this.MergeFromVert);
			copy.MergeToVert = new List<uint>(this.MergeToVert);
			copy.RunIndex = new List<uint>(this.RunIndex);
			copy.RunOriginalId = new List<uint>(this.RunOriginalId);
			copy.RunTransform = new List<float>(this.RunTransform);
			copy.FaceId = new List<uint>(this.FaceId);
			copy.HalfedgeTangent = new List<float>(this.HalfedgeTangent);
			copy.RunFlags = new List<byte>(this.RunFlags);
			copy.Tolerance = this.Tolerance;
			return copy;
		}

		/// <summary>
		/// Merges coincident vertices based on position within tolerance. Uses BVH
		/// collision detection to find open edges, then groups coincident vertices via
		/// union-find. Returns true if new merges were found, false if the mesh was
		/// already fully merged.
		/// </summary>
		/// <returns>True when the merge vectors changed.</returns>
		public bool Merge()
		{
			int numVert = this.NumVert();
			int numTri = this.NumTri();

			// Build initial merge map from existing merge vectors
			int[] mergeMap = new int[numVert];
			for (int i = 0; i < numVert; i++)
			{
				mergeMap[i] = i;
			}

			for (int i = 0; i < this.MergeFromVert.Count; i++)
			{
				mergeMap[(int)this.MergeFromVert[i]] = (int)this.MergeToVert[i];
			}

			// Find open (non-manifold) edges
			int[] next = new int[] { 1, 2, 0 };

			// Rust `BTreeSet<(usize, usize)>`: ordered, and the order reaches the output
			// through open_verts below. SortedSet over a ValueTuple compares
			// lexicographically, which is the Rust tuple `Ord`.
			SortedSet<(int, int)> openEdges = new SortedSet<(int, int)>();
			for (int tri = 0; tri < numTri; tri++)
			{
				for (int i = 0; i < 3; i++)
				{
					int a = mergeMap[(int)this.TriVerts[(3 * tri) + next[i]]];
					int b = mergeMap[(int)this.TriVerts[(3 * tri) + i]];
					(int, int) edge = (a, b);

					// Look for the reverse edge
					(int, int) rev = (b, a);
					if (openEdges.Contains(rev))
					{
						openEdges.Remove(rev);
					}
					else
					{
						openEdges.Add(edge);
					}
				}
			}

			if (openEdges.Count == 0)
			{
				return false;
			}

			// Collect unique open vertices — only the START vertex of each open
			// halfedge, matching C++ which stores (start,end) and takes edge.first=start.
			// Our BTreeSet stores (end,start) so we take edge.1 (= b = start vertex).
			List<int> openVerts;
			{
				SortedSet<int> vset = new SortedSet<int>();
				foreach ((int _, int b) in openEdges)
				{
					vset.Add(b);
				}

				openVerts = new List<int>(vset);
			}

			int numOpen = openVerts.Count;

			// Compute bounding box
			Box bbox = new Box();
			for (int v = 0; v < numVert; v++)
			{
				(float X, float Y, float Z) pos = this.GetVertPos(v);
				Vec3 p = new Vec3(pos.X, pos.Y, pos.Z);
				bbox.UnionPoint(p);
			}

			double tolerance = MaxF64((double)this.Tolerance, F32Epsilon * bbox.Scale());

			// Build BVH boxes and morton codes for open vertices
			Box[] vertBox = new Box[numOpen];
			uint[] vertMorton = new uint[numOpen];
			for (int k = 0; k < numOpen; k++)
			{
				int v = openVerts[k];
				(float X, float Y, float Z) pos = this.GetVertPos(v);
				Vec3 center = new Vec3(pos.X, pos.Y, pos.Z);
				double halfTol = tolerance / 2.0;
				Box bx = Box.FromPoints(
					center - new Vec3(halfTol, halfTol, halfTol),
					center + new Vec3(halfTol, halfTol, halfTol));
				vertBox[k] = bx;
				vertMorton[k] = Sort.MortonCode(center, bbox);
			}

			// Sort by morton code.
			// SORT AUDIT (types_meshgl.rs:236): Rust `sort_by_key`, which is STABLE, so
			// open verts sharing a morton code keep ascending vertex order. That order
			// reaches the collider's leaf order and hence the (a, b) argument order of
			// every union below, so it is a numerical-parity surface. LINQ OrderBy is the
			// documented-stable C# sort; Array.Sort is not usable here.
			int[] order = Enumerable.Range(0, numOpen).OrderBy(i => vertMorton[i]).ToArray();

			Box[] sortedBox = new Box[numOpen];
			uint[] sortedMorton = new uint[numOpen];
			int[] sortedVerts = new int[numOpen];
			for (int k = 0; k < numOpen; k++)
			{
				sortedBox[k] = vertBox[order[k]];
				sortedMorton[k] = vertMorton[order[k]];
				sortedVerts[k] = openVerts[order[k]];
			}

			// Build collider and find coincident vertex pairs. The Rust passes
			// `sorted_box.clone()` only because `Collider::new` takes the Vec by value and
			// the original is queried below; the C# constructor copies the boxes it needs
			// into its node array and keeps no reference, so the array is shared safely.
			Collider collider = new Collider(sortedBox, sortedMorton);
			DisjointSets uf = new DisjointSets((uint)numVert);

			collider.CollisionsWithBoxes(
				sortedBox,
				false,
				(a, b) => uf.Unite((uint)sortedVerts[a], (uint)sortedVerts[b]));

			// Also merge from existing merge vectors
			for (int i = 0; i < this.MergeFromVert.Count; i++)
			{
				uf.Unite(this.MergeFromVert[i], this.MergeToVert[i]);
			}

			// Rebuild merge vectors
			this.MergeFromVert.Clear();
			this.MergeToVert.Clear();
			for (int v = 0; v < numVert; v++)
			{
				int mergeTo = (int)uf.Find((uint)v);
				if (mergeTo != v)
				{
					this.MergeFromVert.Add((uint)v);
					this.MergeToVert.Add((uint)mergeTo);
				}
			}

			return true;
		}

		/// <summary>
		/// True if triangle run <paramref name="run"/> is on the backside (e.g. from a
		/// subtraction). run_flags is a bitmask (#1718): bit 0 = backside. Informational
		/// only — the framework already orients stored normals on the standard flow.
		/// </summary>
		/// <param name="run">The run index.</param>
		/// <returns>True when the run is backside.</returns>
		public bool Backside(int run)
		{
			return run < this.RunFlags.Count && (this.RunFlags[run] & 1) != 0;
		}

		/// <summary>
		/// True if the first three extra-property channels (slots 3, 4, 5) of run
		/// <paramref name="run"/> carry world-frame vertex normals (set by
		/// <c>CalculateNormals(0)</c>, round-tripped via run_flags bit 1, #1718).
		/// Consumers should treat the slot as normals and skip re-applying run_transform
		/// to it.
		/// </summary>
		/// <param name="run">The run index.</param>
		/// <returns>True when the run carries normals.</returns>
		public bool HasNormals(int run)
		{
			return run < this.RunFlags.Count && (this.RunFlags[run] & 2) != 0;
		}

		/// <summary>
		/// Applies run transforms to normals stored at <paramref name="normalIdx"/> in each
		/// vertex's properties, then clears run_transform and run_flags. Matches C++
		/// <c>MeshGL::UpdateNormals(normalIdx)</c>.
		/// </summary>
		/// <remarks>
		/// The normal transform is the inverse-transpose of the 3x3 rotation part of the
		/// run transform. For backside runs (run_flags bit 0 set), normals are additionally
		/// negated.
		/// </remarks>
		/// <param name="normalIdx">The property slot the normal starts at.</param>
		public void UpdateNormals(int normalIdx)
		{
			// NARROWING AUDIT (uint -> int, both casts in this method). The Rust compares
			// and strides in `usize` (`self.num_prop as usize`); this narrows the u32 to
			// int. The unchecked conversion wraps exactly as Rust `as` does, and a wrapped
			// result needs NumProp > int.MaxValue — unreachable, since VertProperties is a
			// List<float> and could not hold one vertex at that stride.
			if (normalIdx < 3 || normalIdx + 3 > (int)this.NumProp)
			{
				return;
			}

			int numVert = this.NumVert();
			int numRun = this.RunOriginalId.Count;
			int np = (int)this.NumProp;
			bool[] vertUpdated = new bool[numVert];

			for (int run = 0; run < numRun; run++)
			{
				// Build the 3x3 normal transform from the column-major 3x4 run transform
				int offset = 12 * run;
				bool hasTransform = offset + 12 <= this.RunTransform.Count;

				// Extract mat3 (upper-left 3x3 of the 3x4 transform)
				double m00;
				double m01;
				double m02;
				double m10;
				double m11;
				double m12;
				double m20;
				double m21;
				double m22;
				if (hasTransform)
				{
					// Column-major: col0=[t0,t1,t2], col1=[t3,t4,t5], col2=[t6,t7,t8]
					m00 = this.RunTransform[offset + 0];
					m01 = this.RunTransform[offset + 3];
					m02 = this.RunTransform[offset + 6];
					m10 = this.RunTransform[offset + 1];
					m11 = this.RunTransform[offset + 4];
					m12 = this.RunTransform[offset + 7];
					m20 = this.RunTransform[offset + 2];
					m21 = this.RunTransform[offset + 5];
					m22 = this.RunTransform[offset + 8];
				}
				else
				{
					m00 = 1.0;
					m01 = 0.0;
					m02 = 0.0;
					m10 = 0.0;
					m11 = 1.0;
					m12 = 0.0;
					m20 = 0.0;
					m21 = 0.0;
					m22 = 1.0;
				}

				// Normal transform = inverse(transpose(M)) = (M^T)^{-1}
				// For a rotation matrix R: (R^T)^{-1} = R itself.
				// For a general transform with scale s: det = s^3, inv_trans = M / s^2.
				// We compute full adjugate/determinant to match C++ la::inverse(la::transpose(M)).
				double det = (m00 * ((m11 * m22) - (m12 * m21)))
					- (m01 * ((m10 * m22) - (m12 * m20)))
					+ (m02 * ((m10 * m21) - (m11 * m20)));

				double n00;
				double n01;
				double n02;
				double n10;
				double n11;
				double n12;
				double n20;
				double n21;
				double n22;
				if (Math.Abs(det) < 1e-30)
				{
					n00 = 1.0;
					n01 = 0.0;
					n02 = 0.0;
					n10 = 0.0;
					n11 = 1.0;
					n12 = 0.0;
					n20 = 0.0;
					n21 = 0.0;
					n22 = 1.0;
				}
				else
				{
					double inv = 1.0 / det;

					// Adjugate of transpose(M) = transpose of adjugate(M)
					n00 = ((m11 * m22) - (m12 * m21)) * inv;
					n01 = ((m02 * m21) - (m01 * m22)) * inv;
					n02 = ((m01 * m12) - (m02 * m11)) * inv;
					n10 = ((m12 * m20) - (m10 * m22)) * inv;
					n11 = ((m00 * m22) - (m02 * m20)) * inv;
					n12 = ((m02 * m10) - (m00 * m12)) * inv;
					n20 = ((m10 * m21) - (m11 * m20)) * inv;
					n21 = ((m01 * m20) - (m00 * m21)) * inv;
					n22 = ((m00 * m11) - (m01 * m10)) * inv;
				}

				double sign = this.Backside(run) ? -1.0 : 1.0;

				// Determine run's vertex range
				int start = run < this.RunIndex.Count ? (int)this.RunIndex[run] : 0;
				int end = run + 1 < this.RunIndex.Count
					? (int)this.RunIndex[run + 1]
					: this.TriVerts.Count;

				for (int idx = start; idx < end; idx++)
				{
					int vert = (int)this.TriVerts[idx];
					if (vert >= numVert || vertUpdated[vert])
					{
						continue;
					}

					vertUpdated[vert] = true;
					int propStart = (vert * np) + normalIdx;
					double nx = this.VertProperties[propStart];
					double ny = this.VertProperties[propStart + 1];
					double nz = this.VertProperties[propStart + 2];

					// Apply normal transform
					double tx = (n00 * nx) + (n01 * ny) + (n02 * nz);
					double ty = (n10 * nx) + (n11 * ny) + (n12 * nz);
					double tz = (n20 * nx) + (n21 * ny) + (n22 * nz);

					// SafeNormalize
					double len = Math.Sqrt((tx * tx) + (ty * ty) + (tz * tz));
					if (len > 0.0)
					{
						tx = sign * tx / len;
						ty = sign * ty / len;
						tz = sign * tz / len;
					}
					else
					{
						tx = 0.0;
						ty = 0.0;
						tz = 0.0;
					}

					this.VertProperties[propStart] = (float)tx;
					this.VertProperties[propStart + 1] = (float)ty;
					this.VertProperties[propStart + 2] = (float)tz;
				}
			}

			this.RunTransform.Clear();
			this.RunFlags.Clear();
		}
	}
}
