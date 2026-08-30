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

// Sdf.Grid.cs — the BCC grid itself: the TetTri lookup tables, the 14 neighbor
// offsets, index encode/decode, grid-point positions, the bounded SDF sampler,
// and the ITP root finder. Split out of Sdf.cs to stay under the 800-line cap;
// in the Rust all of this is sdf.rs.
//
// Both TetTri tables are transcribed digit-for-digit from the Rust, which took
// them digit-for-digit from C++ sdf.cpp. They are not derivable from anything
// here: entry i is indexed by the four tetrahedron corner signs packed as
// 1|2|4|8 (see CreateTris), and its three values index the six tetrahedron
// edges in the order CreateTris' `edges` array builds them. Nothing in this
// file may "simplify" them.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	public static partial class Sdf
	{
		// -------------------------------------------------------------------
		// Lookup tables — TetTri0, TetTri1
		// -------------------------------------------------------------------

		private static readonly IVec3[] TetTri0Table = new IVec3[16]
		{
			new IVec3(-1, -1, -1),
			new IVec3(0, 3, 4),
			new IVec3(0, 1, 5),
			new IVec3(1, 5, 3),
			new IVec3(1, 4, 2),
			new IVec3(1, 0, 3),
			new IVec3(2, 5, 0),
			new IVec3(5, 3, 2),
			new IVec3(2, 3, 5),
			new IVec3(0, 5, 2),
			new IVec3(3, 0, 1),
			new IVec3(2, 4, 1),
			new IVec3(3, 5, 1),
			new IVec3(5, 1, 0),
			new IVec3(4, 3, 0),
			new IVec3(-1, -1, -1),
		};

		private static readonly IVec3[] TetTri1Table = new IVec3[16]
		{
			new IVec3(-1, -1, -1),
			new IVec3(-1, -1, -1),
			new IVec3(-1, -1, -1),
			new IVec3(3, 4, 1),
			new IVec3(-1, -1, -1),
			new IVec3(3, 2, 1),
			new IVec3(0, 4, 2),
			new IVec3(-1, -1, -1),
			new IVec3(-1, -1, -1),
			new IVec3(2, 4, 0),
			new IVec3(1, 2, 3),
			new IVec3(-1, -1, -1),
			new IVec3(1, 4, 3),
			new IVec3(-1, -1, -1),
			new IVec3(-1, -1, -1),
			new IVec3(-1, -1, -1),
		};

		// -------------------------------------------------------------------
		// BCC grid neighbor offsets
		// -------------------------------------------------------------------

		private static readonly IVec4[] NeighborsTable = new IVec4[14]
		{
			new IVec4(0, 0, 0, 1),
			new IVec4(1, 0, 0, 0),
			new IVec4(0, 1, 0, 0),
			new IVec4(0, 0, 1, 0),
			new IVec4(-1, 0, 0, 1),
			new IVec4(0, -1, 0, 1),
			new IVec4(0, 0, -1, 1),
			new IVec4(-1, -1, -1, 1),
			new IVec4(-1, 0, 0, 0),
			new IVec4(0, -1, 0, 0),
			new IVec4(0, 0, -1, 0),
			new IVec4(0, -1, -1, 1),
			new IVec4(-1, 0, -1, 1),
			new IVec4(-1, -1, 0, 1),
		};

		/// <summary>The first TetTri lookup table, indexed by the packed corner signs.</summary>
		/// <param name="i">The packed corner signs, 0..15.</param>
		/// <returns>Three edge indices, or (-1,-1,-1) for no triangle.</returns>
		internal static IVec3 TetTri0(int i)
		{
			return TetTri0Table[i];
		}

		/// <summary>The second TetTri lookup table, for the cases that emit two triangles.</summary>
		/// <param name="i">The packed corner signs, 0..15.</param>
		/// <returns>Three edge indices, or (-1,-1,-1) for no triangle.</returns>
		internal static IVec3 TetTri1(int i)
		{
			return TetTri1Table[i];
		}

		/// <summary>
		/// The <paramref name="i"/>th BCC neighbor of <paramref name="base"/>. Offsets 0..6 are
		/// the forward edges, 7..13 their opposites; a w of 2 folds back onto the next
		/// cell's corner lattice.
		/// </summary>
		/// <param name="base">The grid index to offset from.</param>
		/// <param name="i">The neighbor index, 0..13.</param>
		/// <returns>The neighbor's grid index.</returns>
		internal static IVec4 Neighbor(IVec4 @base, int i)
		{
			IVec4 neighborIndex = IVec4Add(@base, NeighborsTable[i]);
			if (neighborIndex.W == 2)
			{
				neighborIndex.X += 1;
				neighborIndex.Y += 1;
				neighborIndex.Z += 1;
				neighborIndex.W = 0;
			}

			return neighborIndex;
		}

		// -------------------------------------------------------------------
		// Grid encoding/decoding
		// -------------------------------------------------------------------

		/// <summary>
		/// Pack a grid index into the hash key. The per-axis bit widths come from
		/// ComputeGridPow, and the packed value is the hash-table key, so the exact
		/// layout reaches the output vertex numbering.
		/// </summary>
		/// <param name="gridPos">The grid index to encode.</param>
		/// <param name="gridPow">The per-axis bit widths.</param>
		/// <returns>The encoded index.</returns>
		internal static ulong EncodeIndex(IVec4 gridPos, IVec3 gridPow)
		{
			// Rust `as u64` on an i32 SIGN-EXTENDS, and so does the C++ static_cast this
			// ports; ComputeVerts encodes neighbors with a -1 component, whose key is
			// therefore huge rather than small. Kept deliberately: it is a key, not an
			// address, and both references produce the same one. Each term is widened
			// into its own local so the `|` never sees a signed operand (CS0675, which
			// this project treats as an error).
			ulong w = unchecked((ulong)(long)gridPos.W);
			ulong z = unchecked((ulong)(long)gridPos.Z);
			ulong y = unchecked((ulong)(long)gridPos.Y);
			ulong x = unchecked((ulong)(long)gridPos.X);
			return w | (z << 1) | (y << (1 + gridPow.Z)) | (x << (1 + gridPow.Z + gridPow.Y));
		}

		/// <summary>The inverse of <see cref="EncodeIndex"/>.</summary>
		/// <param name="idx">The encoded index.</param>
		/// <param name="gridPow">The per-axis bit widths.</param>
		/// <returns>The decoded grid index.</returns>
		internal static IVec4 DecodeIndex(ulong idx, IVec3 gridPow)
		{
			// Every `as i32` here truncates the low 32 bits, which is what C#'s unchecked
			// ulong-to-int conversion does; the fields are small in practice.
			int w = unchecked((int)(idx & 1));
			idx >>= 1;
			int z = unchecked((int)(idx & ((1UL << gridPow.Z) - 1)));
			idx >>= gridPow.Z;
			int y = unchecked((int)(idx & ((1UL << gridPow.Y) - 1)));
			idx >>= gridPow.Y;
			int x = unchecked((int)(idx & ((1UL << gridPow.X) - 1)));
			return new IVec4(x, y, z, w);
		}

		/// <summary>The world position of a grid index.</summary>
		/// <param name="gridIndex">The grid index; w selects the corner (1) or body-center (0) lattice.</param>
		/// <param name="origin">The grid origin.</param>
		/// <param name="spacing">The per-axis grid spacing.</param>
		/// <returns>The world position.</returns>
		internal static Vec3 Position(IVec4 gridIndex, Vec3 origin, Vec3 spacing)
		{
			double offset = gridIndex.W == 1 ? 0.0 : -0.5;
			return origin
				+ new Vec3(
					spacing.X * ((double)gridIndex.X + offset),
					spacing.Y * ((double)gridIndex.Y + offset),
					spacing.Z * ((double)gridIndex.Z + offset));
		}

		/// <summary>Clamp a position into the grid's world extent.</summary>
		/// <param name="pos">The position to clamp.</param>
		/// <param name="origin">The grid origin.</param>
		/// <param name="spacing">The per-axis grid spacing.</param>
		/// <param name="gridSize">The per-axis grid point count.</param>
		/// <returns>The clamped position.</returns>
		internal static Vec3 Bound(Vec3 pos, Vec3 origin, Vec3 spacing, IVec3 gridSize)
		{
			Vec3 maxBound = new Vec3(
				origin.X + spacing.X * ((double)gridSize.X - 1.0),
				origin.Y + spacing.Y * ((double)gridSize.Y - 1.0),
				origin.Z + spacing.Z * ((double)gridSize.Z - 1.0));
			return new Vec3(
				MinF64(MaxF64(pos.X, origin.X), maxBound.X),
				MinF64(MaxF64(pos.Y, origin.Y), maxBound.Y),
				MinF64(MaxF64(pos.Z, origin.Z), maxBound.Z));
		}

		/// <summary>
		/// Sample the SDF at a grid index, forcing the boundary shell negative so the
		/// extracted surface closes inside the bounds.
		/// </summary>
		/// <param name="gridIndex">The grid index to sample.</param>
		/// <param name="origin">The grid origin.</param>
		/// <param name="spacing">The per-axis grid spacing.</param>
		/// <param name="gridSize">The per-axis grid point count.</param>
		/// <param name="level">The level being extracted.</param>
		/// <param name="sdf">The signed distance function; positive is inside.</param>
		/// <returns>The level-relative distance, clamped at the boundary.</returns>
		internal static double BoundedSdf(
			IVec4 gridIndex,
			Vec3 origin,
			Vec3 spacing,
			IVec3 gridSize,
			double level,
			Func<Vec3, double> sdf)
		{
			IVec3 xyz = new IVec3(gridIndex.X, gridIndex.Y, gridIndex.Z);
			int lowerBoundDist = Math.Min(Math.Min(xyz.X, xyz.Y), xyz.Z);
			int upperBoundDist = Math.Min(
				Math.Min(gridSize.X - xyz.X, gridSize.Y - xyz.Y),
				gridSize.Z - xyz.Z);
			int boundDist = Math.Min(lowerBoundDist, upperBoundDist - gridIndex.W);

			if (boundDist < 0)
			{
				return 0.0;
			}

			double d = sdf(Position(gridIndex, origin, spacing)) - level;
			if (boundDist == 0)
			{
				return MinF64(d, 0.0);
			}

			return d;
		}

		// -------------------------------------------------------------------
		// ITP root-finding
		// -------------------------------------------------------------------

		/// <summary>
		/// Locate the level crossing on the segment pos0..pos1 by ITP (interpolate,
		/// truncate, project) bisection, stopping once the bracket is under
		/// <paramref name="tol"/>.
		/// </summary>
		/// <param name="pos0">One endpoint.</param>
		/// <param name="d0">Level-relative distance at <paramref name="pos0"/>.</param>
		/// <param name="pos1">The other endpoint.</param>
		/// <param name="d1">Level-relative distance at <paramref name="pos1"/>.</param>
		/// <param name="tol">Distance tolerance; infinity means interpolate only.</param>
		/// <param name="level">The level being extracted.</param>
		/// <param name="sdf">The signed distance function.</param>
		/// <returns>The refined crossing position.</returns>
		internal static Vec3 FindSurface(
			Vec3 pos0,
			double d0,
			Vec3 pos1,
			double d1,
			double tol,
			double level,
			Func<Vec3, double> sdf)
		{
			if (d0 == 0.0)
			{
				return pos0;
			}
			else if (d1 == 0.0)
			{
				return pos1;
			}

			double k = 0.1;
			Vec3 diff = pos0 - pos1;
			double len = Math.Sqrt((diff.X * diff.X) + (diff.Y * diff.Y) + (diff.Z * diff.Z));
			double check = 2.0 * tol / len;
			double frac = 1.0;
			double biFrac = 1.0;
			while (frac > check)
			{
				double tRaw = d0 / (d0 - d1);
				double t = (tRaw * (1.0 - k)) + (0.5 * k); // la::lerp(t_raw, 0.5, k) = a*(1-t)+b*t
				double r = (biFrac / frac) - 0.5;
				double x = Math.Abs(t - 0.5) < r
					? t
					: 0.5 - (r * (t < 0.5 ? 1.0 : -1.0));

				Vec3 mid = LerpVec3(pos0, pos1, x);
				double d = sdf(mid) - level;

				if ((d > 0.0) == (d0 > 0.0))
				{
					d0 = d;
					pos0 = mid;
					frac *= 1.0 - x;
				}
				else
				{
					d1 = d;
					pos1 = mid;
					frac *= x;
				}

				biFrac /= 2.0;
			}

			return LerpVec3(pos0, pos1, d0 / (d0 - d1));
		}

		// -------------------------------------------------------------------
		// Helper math
		// -------------------------------------------------------------------

		/// <summary>Componentwise <see cref="IVec4"/> addition.</summary>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <returns>The sum.</returns>
		internal static IVec4 IVec4Add(IVec4 a, IVec4 b)
		{
			return new IVec4(a.X + b.X, a.Y + b.Y, a.Z + b.Z, a.W + b.W);
		}

		/// <summary>
		/// C++ <c>la::lerp(a, b, t)</c> is <c>a*(1-t) + b*t</c> — NOT <c>a + (b-a)*t</c>. The two
		/// differ in float association, which shifts the ITP midpoints and hence the
		/// refined surface verts; keep the C++ form bit-for-bit.
		/// </summary>
		/// <param name="a">The start point.</param>
		/// <param name="b">The end point.</param>
		/// <param name="t">The interpolation parameter.</param>
		/// <returns>The interpolated point.</returns>
		internal static Vec3 LerpVec3(Vec3 a, Vec3 b, double t)
		{
			return new Vec3(
				(a.X * (1.0 - t)) + (b.X * t),
				(a.Y * (1.0 - t)) + (b.Y * t),
				(a.Z * (1.0 - t)) + (b.Z * t));
		}

		/// <summary>The next index in a 3-cycle.</summary>
		/// <param name="i">The index, 0..2.</param>
		/// <returns>The successor, wrapping 2 to 0.</returns>
		internal static int Next3(int i)
		{
			return i == 2 ? 0 : i + 1;
		}

		/// <summary>The previous index in a 3-cycle.</summary>
		/// <param name="i">The index, 0..2.</param>
		/// <returns>The predecessor, wrapping 0 to 2.</returns>
		internal static int Prev3(int i)
		{
			return i == 0 ? 2 : i - 1;
		}
	}
}
