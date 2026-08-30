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

// Manifold.MeshGL.cs — the import half of manifold_meshgl.rs, whose header reads:
//
//   MeshGL conversion methods for Manifold
//   Extracted from manifold.rs for file size management
//
//   The import/export bodies are generic over the mesh precision and index
//   types, mirroring the C++ `Impl::Impl(const MeshGLP<Precision, I>&)` and
//   `GetMeshGLImpl<Precision, I>` templates: the f32/u32 instantiation narrows
//   exactly where the C++ float instantiation does, and the f64/u64
//   instantiation is lossless into and out of the f64 kernel.
//
// C# has no such generic (see the MeshGL.cs header), so "generic over the mesh
// precision and index types" is realized by IMeshGLAccess — one body, two
// adapters, narrowing at the same sites. MeshGLAccess.cs argues that choice.
//
// The export half is in Manifold.MeshGL.Export.cs; the OBJ text round-trip,
// which the Rust keeps in this same file, is at the bottom of this one.
//
// Nothing in this file is deferred any more: ImportMode.AllowSoup's non-manifold
// leg was the last one, and Robust/Soup.cs's Soupify landed it.

using System.Globalization;

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <content>
	/// Conversion to and from <see cref="MeshGL"/> / <see cref="MeshGL64"/>, and the OBJ
	/// text round-trip.
	/// </content>
	public sealed partial class Manifold
	{
		/// <summary>Import a single-precision mesh.</summary>
		/// <param name="mesh">The mesh to import.</param>
		/// <returns>The manifold, or an empty one carrying the validation error.</returns>
		public static Manifold FromMeshGL(MeshGL mesh)
		{
			ArgumentNullException.ThrowIfNull(mesh);
			return FromMeshImpl(new MeshGLAccess(mesh), ImportMode.Strict, false);
		}

		/// <summary>
		/// Import a double-precision mesh directly into the f64 kernel.
		/// </summary>
		/// <remarks>
		/// Unlike <see cref="FromMeshGL"/>, no narrowing occurs anywhere on this path:
		/// coordinates, run transforms, tangents, and tolerance are consumed at full f64
		/// precision, matching C++ <c>Manifold(const MeshGL64&amp;)</c>. Indices still
		/// truncate to 32 bits exactly as the C++ import's <c>uint32_t</c> casts do — the
		/// kernel indexes vertices with 32 bits.
		/// </remarks>
		/// <param name="mesh">The mesh to import.</param>
		/// <returns>The manifold, or an empty one carrying the validation error.</returns>
		public static Manifold FromMeshGL64(MeshGL64 mesh)
		{
			ArgumentNullException.ThrowIfNull(mesh);
			return FromMeshImpl(new MeshGL64Access(mesh), ImportMode.Strict, false);
		}

		/// <summary>
		/// Import that accepts non-manifold geometry for the robust boolean engine.
		/// </summary>
		/// <remarks>
		/// Manifold input behaves exactly like <see cref="FromMeshGL"/>; non-manifold input
		/// is kept as a triangle soup (<see cref="Status"/> stays
		/// <see cref="Error.NoError"/>) as long as it is geometrically closed and
		/// orientable, otherwise the result is empty with <see cref="Error.NotClosed"/>.
		/// Soup-backed manifolds support booleans via
		/// <see cref="BooleanEngine.Robust"/>/<see cref="BooleanEngine.Auto"/>, transforms,
		/// and mesh export; pairing-dependent operations return empty results with
		/// <see cref="Error.NotManifold"/>.
		/// </remarks>
		/// <param name="mesh">The mesh to import.</param>
		/// <returns>The manifold, or an empty one carrying the validation error.</returns>
		public static Manifold FromMeshGLRobust(MeshGL mesh)
		{
			ArgumentNullException.ThrowIfNull(mesh);
			return FromMeshImpl(new MeshGLAccess(mesh), ImportMode.AllowSoup, false);
		}

		/// <summary>Double-precision variant of <see cref="FromMeshGLRobust"/>.</summary>
		/// <param name="mesh">The mesh to import.</param>
		/// <returns>The manifold, or an empty one carrying the validation error.</returns>
		public static Manifold FromMeshGL64Robust(MeshGL64 mesh)
		{
			ArgumentNullException.ThrowIfNull(mesh);
			return FromMeshImpl(new MeshGL64Access(mesh), ImportMode.AllowSoup, false);
		}

		/// <summary>
		/// Read a manifold from a Wavefront OBJ-format string. Recognizes the
		/// <c># tolerance</c> and <c># epsilon</c> comment metadata emitted by
		/// <see cref="WriteObj"/>.
		/// </summary>
		/// <param name="source">The OBJ text.</param>
		/// <returns>The parsed manifold.</returns>
		public static Manifold ReadObj(string source)
		{
			ArgumentNullException.ThrowIfNull(source);

			MeshGL64 mesh = new MeshGL64();
			mesh.NumProp = 3;
			double? epsilon = null;
			foreach (string line in SplitLines(source))
			{
				string trimmed = line.TrimEnd('\r', '\n');
				if (trimmed.StartsWith("# tolerance = ", StringComparison.Ordinal))
				{
					string rest = trimmed.Substring("# tolerance = ".Length);
					if (double.TryParse(
						rest.Trim(),
						NumberStyles.Float,
						CultureInfo.InvariantCulture,
						out double v))
					{
						mesh.Tolerance = v;
					}
				}
				else if (trimmed.StartsWith("# epsilon = ", StringComparison.Ordinal))
				{
					string rest = trimmed.Substring("# epsilon = ".Length);
					if (double.TryParse(
						rest.Trim(),
						NumberStyles.Float,
						CultureInfo.InvariantCulture,
						out double v))
					{
						epsilon = v;
					}
				}
				else if (trimmed.StartsWith("v ", StringComparison.Ordinal))
				{
					string[] parts = trimmed
						.Substring(2)
						.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
					if (parts.Length >= 3
						&& TryParseF64(parts[0], out double x)
						&& TryParseF64(parts[1], out double y)
						&& TryParseF64(parts[2], out double z))
					{
						mesh.VertProperties.Add(x);
						mesh.VertProperties.Add(y);
						mesh.VertProperties.Add(z);
					}
				}
				else if (trimmed.StartsWith("f ", StringComparison.Ordinal))
				{
					string[] parts = trimmed
						.Substring(2)
						.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
					if (parts.Length >= 3
						&& TryParseObjVert(parts[0], out ulong a)
						&& TryParseObjVert(parts[1], out ulong b)
						&& TryParseObjVert(parts[2], out ulong c))
					{
						mesh.TriVerts.Add(a);
						mesh.TriVerts.Add(b);
						mesh.TriVerts.Add(c);
					}
				}
			}

			Manifold m = FromMeshGL64(mesh);
			if (epsilon.HasValue)
			{
				m.ApplyEpsilon(epsilon.Value);
			}

			return m;
		}

		/// <summary>
		/// <see cref="FromMeshGL64Robust"/> for meshes the robust boolean engine assembled
		/// itself (<c>robust::assemble</c>), which skip <c>edge_op::swap_degenerates</c>.
		/// </summary>
		/// <remarks>
		/// See docs/CPP_DIVERGENCES.md entry 1: a boolean result legitimately contains
		/// coplanar antiparallel adjacencies, whose flood-filled face normals make the swap
		/// misclassify large valid triangles and physically move the surface. Every other
		/// import path — including user meshes coming in through
		/// <see cref="FromMeshGLRobust"/> — is unaffected.
		/// </remarks>
		/// <param name="mesh">The mesh to import.</param>
		/// <returns>The manifold, or an empty one carrying the validation error.</returns>
		internal static Manifold FromMeshGL64RobustAssembled(MeshGL64 mesh)
		{
			return FromMeshImpl(new MeshGL64Access(mesh), ImportMode.AllowSoup, true);
		}

		/// <summary>Export a single-precision mesh.</summary>
		/// <param name="normalIdx">
		/// The extra-property slot to interpret as normals and normalize on export, or -1
		/// for the legacy behaviour (which still auto-substitutes slot 0 when every meshID
		/// recorded world-frame normals).
		/// </param>
		/// <returns>The exported mesh.</returns>
		public MeshGL GetMeshGL(int normalIdx)
		{
			MeshGL outMesh = new MeshGL();
			this.GetMeshGLImpl(new MeshGLAccess(outMesh), normalIdx);
			return outMesh;
		}

		/// <summary>
		/// Export a double-precision mesh straight from the f64 kernel.
		/// </summary>
		/// <remarks>
		/// Unlike <see cref="GetMeshGL"/>, nothing narrows on this path — coordinates,
		/// tangents, transforms, and tolerance come out at full f64 precision — and the
		/// tolerance is *not* floored at <c>f32::EPSILON * bbox.scale()</c>, matching C++
		/// <c>GetMeshGL64</c> vs <c>GetMeshGL</c>.
		/// </remarks>
		/// <param name="normalIdx">The extra-property slot to treat as normals, or -1.</param>
		/// <returns>The exported mesh.</returns>
		public MeshGL64 GetMeshGL64(int normalIdx)
		{
			MeshGL64 outMesh = new MeshGL64();
			this.GetMeshGLImpl(new MeshGL64Access(outMesh), normalIdx);
			return outMesh;
		}

		/// <summary>
		/// Write the manifold to a Wavefront OBJ-format string.
		/// </summary>
		/// <remarks>
		/// Mirrors C++ <c>Manifold::WriteOBJ</c>, using 19-digit fixed precision and
		/// sorting faces for deterministic output. Also emits <c># tolerance</c> and
		/// <c># epsilon</c> comment metadata.
		/// </remarks>
		/// <returns>The OBJ text.</returns>
		public string WriteObj()
		{
			MeshGL64 mesh = this.GetMeshGL64(-1);
			double epsilon = this.imp.Epsilon;
			System.Text.StringBuilder outText = new System.Text.StringBuilder();
			outText.Append("# ======= begin mesh ======\n");
			outText.Append(CultureInfo.InvariantCulture, $"# tolerance = {Fixed19(mesh.Tolerance)}\n");
			outText.Append(CultureInfo.InvariantCulture, $"# epsilon = {Fixed19(epsilon)}\n");
			int numProp = (int)mesh.NumProp;
			for (int i = 0; i < mesh.NumVert(); i++)
			{
				int offset = i * numProp;
				outText.Append(CultureInfo.InvariantCulture, $"v {Fixed19(mesh.VertProperties[offset])} ");
				outText.Append(CultureInfo.InvariantCulture, $"{Fixed19(mesh.VertProperties[offset + 1])} ");
				outText.Append(CultureInfo.InvariantCulture, $"{Fixed19(mesh.VertProperties[offset + 2])}\n");
			}

			List<(ulong A, ulong B, ulong C)> tris = new List<(ulong, ulong, ulong)>(mesh.NumTri());
			for (int i = 0; i < mesh.NumTri(); i++)
			{
				tris.Add((
					mesh.TriVerts[3 * i] + 1,
					mesh.TriVerts[(3 * i) + 1] + 1,
					mesh.TriVerts[(3 * i) + 2] + 1));
			}

			// Rust `Vec<[u64; 3]>::sort()` is lexicographic on the array; a ValueTuple
			// compares its fields in order, which is that same ordering.
			tris.Sort();
			foreach ((ulong a, ulong b, ulong c) in tris)
			{
				outText.Append(CultureInfo.InvariantCulture, $"f {a} {b} {c}\n");
			}

			outText.Append("# ======== end mesh =======\n");
			return outText.ToString();
		}

		/// <summary>Rust `{:.19}`: fixed notation with exactly nineteen fractional digits.</summary>
		/// <param name="v">The value to format.</param>
		/// <returns>The formatted value.</returns>
		private static string Fixed19(double v)
		{
			return v.ToString("F19", CultureInfo.InvariantCulture);
		}

		/// <summary>Rust <c>str::lines()</c>: split on \n, dropping a trailing \r.</summary>
		/// <param name="source">The text to split.</param>
		/// <returns>The lines.</returns>
		private static IEnumerable<string> SplitLines(string source)
		{
			return source.Split('\n');
		}

		/// <summary>Rust <c>str::parse::&lt;f64&gt;()</c>, culture-independent.</summary>
		/// <param name="s">The text to parse.</param>
		/// <param name="value">The parsed value.</param>
		/// <returns>True when the text parsed.</returns>
		private static bool TryParseF64(string s, out double value)
		{
			return double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
		}

		/// <summary>
		/// One OBJ face entry, which may be "v", "v/vt", or "v/vt/vn", converted to a
		/// zero-based index.
		/// </summary>
		/// <param name="s">The face entry.</param>
		/// <param name="value">The zero-based vertex index.</param>
		/// <returns>True when the entry parsed.</returns>
		private static bool TryParseObjVert(string s, out ulong value)
		{
			string first = s.Split('/')[0];
			if (ulong.TryParse(first, NumberStyles.Integer, CultureInfo.InvariantCulture, out ulong n))
			{
				// NARROWING AUDIT (wrapping): Rust `n - 1` on a u64 in release mode wraps,
				// so "f 0 0 0" yields u64::MAX and is rejected downstream by the
				// vertex-bounds check. C#'s default unchecked subtraction wraps the same way.
				value = unchecked(n - 1);
				return true;
			}

			value = 0;
			return false;
		}

		private static Manifold FromMeshImpl(IMeshGLAccess mesh, ImportMode mode, bool skipSwap)
		{
			// NARROWING GUARD (num_prop). The Rust keeps num_prop at full width — on a
			// 64-bit target `to_u64() as usize` is a no-op — and its `num_vert()` divides
			// the property array by that full-width stride. For a num_prop past 2^31 the
			// array cannot be that long, so the Rust's num_vert() is 0 and the import
			// falls out through the empty / `num_vert < 4` checks below with a status.
			//
			// C# cannot follow it there: NumVert() and GetVertPos() index List<T>, so they
			// truncate num_prop to int first (the documented audits in MeshGL.cs and
			// MeshGL64.cs). num_prop = 0x1_0000_0003 truncates to 3, NumVert() becomes
			// Count/3 instead of 0, and the import proceeds at a stride the mesh does not
			// have — a silent garbage read where the Rust returns empty. So the raw value
			// is checked at full width, before anything narrows it, and refused loudly.
			//
			// This throws rather than returning a status because it is not a geometry
			// error with a Rust counterpart to report: it is an argument this type can
			// express but cannot hold (a single vertex of that stride exceeds any List),
			// which is the same class of failure as the ArgumentNullException guards on
			// the public entry points above.
			if (mesh.NumPropRaw > int.MaxValue)
			{
				throw new ArgumentOutOfRangeException(
					nameof(mesh),
					mesh.NumPropRaw,
					"MeshGL.NumProp exceeds int.MaxValue, so the vertex stride cannot be "
					+ "represented; no mesh with that many properties per vertex is storable.");
			}

			uint numVert = (uint)mesh.NumVert();
			int numTri = mesh.NumTri();

			// Validation checks matching C++ Impl::Impl(const MeshGLP&)
			if (numVert == 0 && numTri == 0)
			{
				return MakeEmpty(Error.NoError);
			}

			if (numVert < 4 || numTri < 4)
			{
				return MakeEmpty(
					mode == ImportMode.Strict ? Error.NotManifold : Error.NotClosed);
			}

			if (mesh.NumPropRaw < 3)
			{
				return MakeEmpty(Error.MissingPositionProperties);
			}

			if (mesh.MergeFromVertCount != mesh.MergeToVertCount)
			{
				return MakeEmpty(Error.MergeVectorsDifferentLengths);
			}

			if (mesh.RunTransformCount != 0
				&& 12 * mesh.RunOriginalId.Count != mesh.RunTransformCount)
			{
				return MakeEmpty(Error.TransformWrongLength);
			}

			if (mesh.RunOriginalId.Count != 0
				&& mesh.RunIndexCount != 0
				&& mesh.RunOriginalId.Count + 1 != mesh.RunIndexCount
				&& mesh.RunOriginalId.Count != mesh.RunIndexCount)
			{
				return MakeEmpty(Error.RunIndexWrongLength);
			}

			if (mesh.FaceIdCount != 0 && mesh.FaceIdCount != numTri)
			{
				return MakeEmpty(Error.FaceIdWrongLength);
			}

			for (int i = 0; i < mesh.VertPropertiesCount; i++)
			{
				if (!double.IsFinite(mesh.VertProperty(i)))
				{
					return MakeEmpty(Error.NonFiniteVertex);
				}
			}

			for (int i = 0; i < mesh.RunTransformCount; i++)
			{
				if (!double.IsFinite(mesh.RunTransform(i)))
				{
					return MakeEmpty(Error.InvalidConstruction);
				}
			}

			for (int i = 0; i < mesh.HalfedgeTangentCount; i++)
			{
				if (!double.IsFinite(mesh.HalfedgeTangent(i)))
				{
					return MakeEmpty(Error.InvalidConstruction);
				}
			}

			// Check merge indices are in bounds. Like the C++ import, u64 indices are
			// first truncated to u32 (the kernel's index width).
			for (int i = 0; i < mesh.MergeFromVertCount; i++)
			{
				if ((uint)mesh.MergeFromVert(i) >= numVert || (uint)mesh.MergeToVert(i) >= numVert)
				{
					return MakeEmpty(Error.MergeIndexOutOfBounds);
				}
			}

			// Check tri_verts are in bounds
			for (int i = 0; i < mesh.TriVertsCount; i++)
			{
				if ((uint)mesh.TriVert(i) >= numVert)
				{
					return MakeEmpty(Error.VertexOutOfBounds);
				}
			}

			ManifoldImpl imp = new ManifoldImpl();

			// NARROWING AUDIT (u32/u64 -> int). The Rust is
			// `num_prop.to_u64() as usize`, which does not narrow on a 64-bit target; this
			// cast does. It is safe only because of the full-width guard at the top of
			// this method — without it a num_prop past 2^31 would wrap to a small stride
			// and be read as valid. The `>= 3` check above then makes the subtraction
			// non-negative, matching the Rust's `saturating_sub(3)`.
			int numProp = (int)mesh.NumPropRaw;
			imp.NumProp = numProp - 3;

			for (int i = 0; i < mesh.NumVert(); i++)
			{
				(double px, double py, double pz) = mesh.GetVertPos(i);
				imp.VertPos.Add(new Vec3(px, py, pz));
			}

			if (imp.NumProp > 0)
			{
				for (int i = 0; i < mesh.NumVert(); i++)
				{
					int offset = i * numProp;
					for (int p = offset + 3; p < offset + numProp; p++)
					{
						imp.Properties.Add(mesh.VertProperty(p));
					}
				}
			}

			// Build prop2vert mapping from merge vectors
			bool hasMerges = mesh.MergeFromVertCount != 0;
			bool needsPropMap = imp.NumProp > 0 && hasMerges;
			int[] prop2Vert;
			if (hasMerges)
			{
				prop2Vert = new int[numVert];
				for (int i = 0; i < numVert; i++)
				{
					prop2Vert[i] = i;
				}

				for (int i = 0; i < mesh.MergeFromVertCount; i++)
				{
					prop2Vert[(uint)mesh.MergeFromVert(i)] = (int)(uint)mesh.MergeToVert(i);
				}
			}
			else
			{
				prop2Vert = Array.Empty<int>();
			}

			// Set up mesh relations from runOriginalID (matches C++ MeshGL constructor)
			List<int> runIndex = new List<int>();
			if (mesh.RunIndexCount == 0)
			{
				runIndex.Add(0);
				runIndex.Add(3 * numTri);
			}
			else
			{
				for (int i = 0; i < mesh.RunIndexCount; i++)
				{
					runIndex.Add((int)mesh.RunIndex(i));
				}

				if (runIndex.Count == mesh.RunOriginalId.Count)
				{
					runIndex.Add(3 * numTri);
				}
				else if (runIndex.Count == 1)
				{
					runIndex.Add(3 * numTri);
				}
			}

			List<uint> runOriginalId = new List<uint>(mesh.RunOriginalId);
			int numRuns = Math.Max(runOriginalId.Count, 1);
			int startId = (int)ManifoldImpl.ReserveIds((uint)numRuns);
			if (runOriginalId.Count == 0)
			{
				runOriginalId.Add((uint)startId);
			}

			// Build tri_ref for all input tris. `TriRef::default()` is the derived
			// all-zero Default, which is what `new TriRef[n]` gives.
			TriRef[] allTriRef = new TriRef[numTri];

			for (int i = 0; i < runOriginalId.Count; i++)
			{
				uint origId = runOriginalId[i];
				int meshId = startId + i;
				int runStart = i < runIndex.Count ? runIndex[i] / 3 : numTri;
				int runEnd = i + 1 < runIndex.Count ? runIndex[i + 1] / 3 : numTri;
				for (int tri = runStart; tri < runEnd; tri++)
				{
					if (tri < allTriRef.Length)
					{
						allTriRef[tri].MeshId = meshId;
						allTriRef[tri].OriginalId = (int)origId;
						allTriRef[tri].FaceId = mesh.FaceIdCount != 0 && tri < mesh.FaceIdCount
							? (int)mesh.FaceId(tri)
							: -1;
						allTriRef[tri].CoplanarId = tri;
					}
				}

				Mat3x4 transform;
				if (mesh.RunTransformCount >= (i + 1) * 12)
				{
					int b = i * 12;
					transform = Mat3x4.FromCols(
						new Vec3(mesh.RunTransform(b), mesh.RunTransform(b + 1), mesh.RunTransform(b + 2)),
						new Vec3(mesh.RunTransform(b + 3), mesh.RunTransform(b + 4), mesh.RunTransform(b + 5)),
						new Vec3(mesh.RunTransform(b + 6), mesh.RunTransform(b + 7), mesh.RunTransform(b + 8)),
						new Vec3(mesh.RunTransform(b + 9), mesh.RunTransform(b + 10), mesh.RunTransform(b + 11)));
				}
				else
				{
					transform = Mat3x4.Identity();
				}

				// run_flags is a bitmask (#1718): bit 0 = backside, bit 1 = hasNormals.
				byte flags = i < mesh.RunFlags.Count ? mesh.RunFlags[i] : (byte)0;
				bool backSide = (flags & 1) != 0;

				// Defensively require >= 3 extra props so a caller setting the bit on a
				// too-small MeshGL doesn't make us read past the slot 0..2 bounds.
				bool runHasNormals = (flags & 2) != 0 && imp.NumProp >= 3;
				Relation relation = new Relation();
				relation.OriginalId = (int)origId;
				relation.Transform = transform;
				relation.BackSide = backSide;
				relation.HasNormals = runHasNormals;
				imp.MeshRelation.MeshIdTransform[meshId] = relation;
			}

			// Build triangles, filtering out degenerates from merges
			List<IVec3> triProp = new List<IVec3>(numTri);
			List<IVec3> triVert = new List<IVec3>();
			if (needsPropMap)
			{
				triVert.Capacity = numTri;
			}

			imp.MeshRelation.TriRef.Clear();
			imp.MeshRelation.TriRef.Capacity = numTri;

			for (int i = 0; i < numTri; i++)
			{
				(ulong t0, ulong t1, ulong t2) = mesh.GetTriVerts(i);

				// Indices truncate to u32 like the C++ import's uint32_t casts; the
				// bounds check above already ran on the truncated values.
				uint a = (uint)t0;
				uint b = (uint)t1;
				uint c = (uint)t2;
				IVec3 triP = new IVec3((int)a, (int)b, (int)c);
				IVec3 triV = prop2Vert.Length == 0
					? triP
					: new IVec3(prop2Vert[a], prop2Vert[b], prop2Vert[c]);

				// Skip degenerate triangles (where merged verts collapse)
				if (triV.X != triV.Y && triV.Y != triV.Z && triV.Z != triV.X)
				{
					if (needsPropMap)
					{
						triProp.Add(triP);
						triVert.Add(triV);
					}
					else
					{
						triProp.Add(triV);
					}

					imp.MeshRelation.TriRef.Add(allTriRef[i]);
				}
			}

			imp.CreateHalfedges(triProp, triVert);

			// Import halfedge tangents from MeshGL (flat array, 4 per halfedge)
			if (mesh.HalfedgeTangentCount != 0)
			{
				int nTangents = mesh.HalfedgeTangentCount / 4;
				imp.HalfedgeTangent.Clear();
				for (int i = 0; i < nTangents; i++)
				{
					imp.HalfedgeTangent.Add(new Vec4(
						mesh.HalfedgeTangent(4 * i),
						mesh.HalfedgeTangent((4 * i) + 1),
						mesh.HalfedgeTangent((4 * i) + 2),
						mesh.HalfedgeTangent((4 * i) + 3)));
				}
			}

			if (!imp.IsManifold())
			{
				if (mode == ImportMode.Strict)
				{
					return MakeEmpty(Error.NotManifold);
				}

				// Keep the geometry as a validated triangle soup: unpaired
				// halfedges allowed, no pairing-dependent pipeline steps.
				Error soupStatus = Robust.Soup.Soupify(imp, triProp, triVert);
				if (soupStatus != Error.NoError)
				{
					return MakeEmpty(soupStatus);
				}

				imp.MeshRelation.OriginalId = -1;
				imp.CalculateBBox();
				imp.SetEpsilon(mesh.Tolerance, false);
				return new Manifold(imp);
			}

			// A Manifold created from input mesh is never an original
			imp.MeshRelation.OriginalId = -1;

			// C++ pipeline: DedupePropVerts, SetNormalsAndCoplanar,
			// RemoveDegenerates, RemoveUnreferencedVerts, SortGeometry
			// Note: CleanupTopology omitted — it would fix opposite-face meshes but
			// conflicts with is_manifold check ordering.
			imp.DedupePropVerts();
			imp.CalculateBBox();
			imp.SetEpsilon(mesh.Tolerance, false);
			imp.SetNormalsAndCoplanar();
			if (skipSwap)
			{
				// remove_degenerates without its swap_degenerates stage; see
				// docs/CPP_DIVERGENCES.md entry 1.
				EdgeOp.CleanupTopology(imp);
				EdgeOp.CollapseShortEdges(imp, 0);
				FaceOp.CalculateVertNormals(imp);
			}
			else
			{
				EdgeOp.RemoveDegenerates(imp, 0);
			}

			imp.RemoveUnreferencedVerts();
			imp.SortGeometry();
			return new Manifold(imp);
		}

		private void ApplyEpsilon(double epsilon)
		{
			this.imp.SetEpsilon(epsilon, false);
		}
	}
}
