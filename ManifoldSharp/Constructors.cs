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

// Constructors.cs — Phase 6: Primitive and polygon constructors
//
// Ports src/constructors.cpp from the Manifold C++ library.
// Sphere() requires Subdivide() (Phase 15) and is omitted here.
// Cube, Tetrahedron, Octahedron are in impl_mesh.rs.
//
// (The header above is constructors.rs's, verbatim; its phase numbers are the
// Rust port's own, not docs/PORTING_PLAN.md's — this is that plan's Phase 4, and the
// three Platonic primitives live in ManifoldImpl.Shapes.cs here.)

using System.Runtime.InteropServices;

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// The free functions of <c>constructors.rs</c>: extrude, revolve and the cylinder
	/// built on top of them.
	/// </summary>
	public static class Constructors
	{
		// -----------------------------------------------------------------------
		// Extrude
		// -----------------------------------------------------------------------

		/// <summary>
		/// Extrudes a set of polygons along the Z axis.
		/// </summary>
		/// <param name="crossSection">Non-overlapping polygons (each a <c>List&lt;Vec2&gt;</c>).</param>
		/// <param name="height">Z extent (must be &gt; 0).</param>
		/// <param name="nDivisions">Extra copies inserted vertically (≥ 0).</param>
		/// <param name="twistDegrees">Rotation applied to top cross-section.</param>
		/// <param name="scaleTop">X/Y scaling applied to top cross-section; (0,0) = cone.</param>
		/// <returns>The extruded impl, or an empty one for degenerate input.</returns>
		public static ManifoldImpl Extrude(
			Polygons crossSection,
			double height,
			int nDivisions,
			double twistDegrees,
			Vec2 scaleTop)
		{
			if (crossSection.Count == 0 || height <= 0.0)
			{
				return new ManifoldImpl();
			}

			scaleTop = new Vec2(MaxF64(scaleTop.X, 0.0), MaxF64(scaleTop.Y, 0.0));
			int nDiv = nDivisions + 1; // total levels above bottom

			List<Vec3> vertPos = new List<Vec3>();
			List<IVec3> triVerts = new List<IVec3>();
			bool isCone = scaleTop.X == 0.0 && scaleTop.Y == 0.0;

			// Count total cross-section vertices
			int nCross = 0;
			foreach (SimplePolygon poly in crossSection)
			{
				nCross += poly.Count;
			}

			// Build indexed form for bottom triangulation
			PolygonsIdx polygonsIndexed = new PolygonsIdx();
			int idx = 0;
			foreach (SimplePolygon poly in crossSection)
			{
				SimplePolygonIdx simpleIndexed = new SimplePolygonIdx();
				foreach (Vec2 pv in poly)
				{
					vertPos.Add(new Vec3(pv.X, pv.Y, 0.0));
					simpleIndexed.Add(new PolyVert(pv, idx));
					idx++;
				}

				polygonsIndexed.Add(simpleIndexed);
			}

			// Build side walls: levels 1..=n_div
			for (int i = 1; i <= nDiv; i++)
			{
				double alpha = (double)i / (double)nDiv;
				double phi = alpha * twistDegrees;
				Vec2 scale = Lerp2(new Vec2(1.0, 1.0), scaleTop, alpha);
				double cosPhi = Types.Cosd(phi);
				double sinPhi = Types.Sind(phi);

				// C++ builds transform = mat2(scale) * mat2(rotation) FIRST (entries
				// are products like sx*cos), then multiplies the vector — match that
				// association exactly.
				double t00 = scale.X * cosPhi; // col0.x
				double t01 = scale.Y * sinPhi; // col0.y
				double t10 = scale.X * -sinPhi; // col1.x
				double t11 = scale.Y * cosPhi; // col1.y

				int j = 0; // apex vertex index for cone top
				int polyOffset = 0; // offset within cross-section for this level

				// (The Rust enumerates here and discards the index with `let _ = pi;`.)
				foreach (SimplePolygon poly in crossSection)
				{
					int polyLen = poly.Count;
					for (int vert = 0; vert < polyLen; vert++)
					{
						int offset = polyOffset + (nCross * i);
						int thisVert = vert + offset;
						int lastVert = (vert == 0 ? polyLen : vert) - 1 + offset;
						if (i == nDiv && isCone)
						{
							// Connect to apex; apex index = n_cross * n_div + j
							int apex = (nCross * nDiv) + j;
							triVerts.Add(new IVec3(apex, lastVert - nCross, thisVert - nCross));
						}
						else
						{
							Vec2 pos2 = poly[vert];
							double px = (t00 * pos2.X) + (t10 * pos2.Y);
							double py = (t01 * pos2.X) + (t11 * pos2.Y);
							vertPos.Add(new Vec3(px, py, height * alpha));
							triVerts.Add(new IVec3(thisVert, lastVert, thisVert - nCross));
							triVerts.Add(new IVec3(
								lastVert,
								lastVert - nCross,
								thisVert - nCross));
						}
					}

					j++;
					polyOffset += polyLen;
				}
			}

			// Add cone apex vertices (one per polygon)
			if (isCone)
			{
				for (int k = 0; k < crossSection.Count; k++)
				{
					vertPos.Add(new Vec3(0.0, 0.0, height));
				}
			}

			// Triangulate bottom (winding reversed for outward normal) and top.
			// C++ calls TriangulateIdx with its DEFAULTS: epsilon=-1, allowConvex=true
			// (the convex fast path picks the alternating fan for e.g. circle caps).
			List<IVec3> topTris = Polygon.TriangulateIdx(polygonsIndexed, -1.0, true);
			foreach (IVec3 tri in topTris)
			{
				// Bottom: reverse winding for correct outward normal (points -Z)
				triVerts.Add(new IVec3(tri.X, tri.Z, tri.Y));

				// Top: forward winding
				if (!isCone)
				{
					triVerts.Add(new IVec3(
						tri.X + (nCross * nDiv),
						tri.Y + (nCross * nDiv),
						tri.Z + (nCross * nDiv)));
				}
			}

			ManifoldImpl m = new ManifoldImpl();
			m.VertPos = vertPos;
			m.CreateHalfedges(triVerts, Array.Empty<IVec3>());
			m.InitializeOriginal();
			m.CalculateBBox();
			m.SetEpsilon(-1.0, false);
			m.SortGeometry();
			m.SetNormalsAndCoplanar();
			return m;
		}

		// -----------------------------------------------------------------------
		// Revolve
		// -----------------------------------------------------------------------

		/// <summary>
		/// Constructs a manifold by revolving polygons around the Y axis (becomes Z).
		/// </summary>
		/// <param name="crossSection">Non-overlapping polygons (each a <c>List&lt;Vec2&gt;</c>).</param>
		/// <param name="circularSegments">Number of divisions around the circle (0 = auto).</param>
		/// <param name="revolveDegrees">How many degrees to revolve (clamped to 360).</param>
		/// <returns>The revolved impl, or an empty one when nothing survives the axis clip.</returns>
		public static ManifoldImpl Revolve(
			Polygons crossSection,
			int circularSegments,
			double revolveDegrees)
		{
			// Filter to positive-x portion only, clipping at axis
			Polygons polygons = new Polygons();
			double radius = 0.0;
			foreach (SimplePolygon poly in crossSection)
			{
				int i = 0;
				while (i < poly.Count && poly[i].X < 0.0)
				{
					i++;
				}

				if (i == poly.Count)
				{
					continue;
				}

				SimplePolygon clipped = new SimplePolygon();
				int start = i;
				int polyLen = poly.Count;
				while (true)
				{
					if (poly[i].X >= 0.0)
					{
						clipped.Add(poly[i]);
						radius = MaxF64(radius, poly[i].X);
					}

					int next = (i + 1) == polyLen ? 0 : i + 1;

					// Add axis-crossing interpolated point
					if ((poly[next].X < 0.0) != (poly[i].X < 0.0))
					{
						double y = poly[next].Y
							- (poly[next].X * (poly[i].Y - poly[next].Y) / (poly[i].X - poly[next].X));
						clipped.Add(new Vec2(0.0, y));
					}

					i = next;
					if (i == start)
					{
						break;
					}
				}

				if (clipped.Count != 0)
				{
					polygons.Add(clipped);
				}
			}

			if (polygons.Count == 0)
			{
				return new ManifoldImpl();
			}

			revolveDegrees = MinF64(revolveDegrees, 360.0);
			bool isFullRevolution = revolveDegrees == 360.0;

			int nDivisions;
			if (circularSegments > 2)
			{
				nDivisions = circularSegments;
			}
			else
			{
				int segs = Quality.GetCircularSegments(radius);
				nDivisions = (int)((double)segs * revolveDegrees / 360.0);
			}

			nDivisions = Math.Max(nDivisions, 3);

			List<Vec3> vertPos = new List<Vec3>();
			List<IVec3> triVerts = new List<IVec3>();

			List<int> startPoses = new List<int>();
			List<int> endPoses = new List<int>();

			double dPhi = revolveDegrees / (double)nDivisions;

			// First and last slice are distinct if not a full revolution
			int nSlices = isFullRevolution ? nDivisions : nDivisions + 1;

			foreach (SimplePolygon poly in polygons)
			{
				int nPosVerts = 0;
				foreach (Vec2 p in poly)
				{
					if (p.X > 0.0)
					{
						nPosVerts++;
					}
				}

				// (The Rust also computes an `n_axis_verts` count here and immediately
				// discards it with `let _ =`; the live count is the identical
				// `n_revolve_axis_verts` loop below, so only that one is ported.)
				int nRevolveAxisVerts = 0;
				foreach (Vec2 pt in poly)
				{
					if (pt.X == 0.0)
					{
						nRevolveAxisVerts++;
					}
				}

				for (int polyVert = 0; polyVert < poly.Count; polyVert++)
				{
					int startPosIndex = vertPos.Count;

					if (!isFullRevolution)
					{
						startPoses.Add(startPosIndex);
					}

					Vec2 curr = poly[polyVert];
					Vec2 prev = poly[polyVert == 0 ? poly.Count - 1 : polyVert - 1];

					// Index of the previous poly_vert's first position
					int prevStartPosIndex = startPosIndex
						+ (polyVert == 0 ? nRevolveAxisVerts + (nSlices * nPosVerts) : 0)
						+ (prev.X == 0.0 ? -1 : -nSlices);

					for (int slice = 0; slice < nSlices; slice++)
					{
						double phi = (double)slice * dPhi;

						// Only push a vertex when it's the first slice OR the vert is not on axis
						if (slice == 0 || curr.X > 0.0)
						{
							vertPos.Add(new Vec3(
								curr.X * Types.Cosd(phi),
								curr.X * Types.Sind(phi),
								curr.Y));
						}

						if (isFullRevolution || slice > 0)
						{
							int lastSlice = (slice == 0 ? nDivisions : slice) - 1;
							if (curr.X > 0.0)
							{
								triVerts.Add(new IVec3(
									startPosIndex + slice,
									startPosIndex + lastSlice,
									prev.X == 0.0
										? prevStartPosIndex
										: prevStartPosIndex + lastSlice));
							}

							if (prev.X > 0.0)
							{
								triVerts.Add(new IVec3(
									prevStartPosIndex + lastSlice,
									prevStartPosIndex + slice,
									curr.X == 0.0
										? startPosIndex
										: startPosIndex + slice));
							}
						}
					}

					if (!isFullRevolution)
					{
						endPoses.Add(vertPos.Count - 1);
					}
				}
			}

			// Cap front and back for partial revolution
			if (!isFullRevolution)
			{
				List<IVec3> frontTris = Polygon.Triangulate(polygons, -1.0, false);
				foreach (IVec3 t in frontTris)
				{
					triVerts.Add(new IVec3(
						startPoses[t.X],
						startPoses[t.Y],
						startPoses[t.Z]));
				}

				foreach (IVec3 t in frontTris)
				{
					triVerts.Add(new IVec3(
						endPoses[t.Z],
						endPoses[t.Y],
						endPoses[t.X]));
				}
			}

			ManifoldImpl m = new ManifoldImpl();
			m.VertPos = vertPos;
			m.CreateHalfedges(triVerts, Array.Empty<IVec3>());
			m.InitializeOriginal();
			m.CalculateBBox();
			m.SetEpsilon(-1.0, false);
			m.SortGeometry();
			m.SetNormalsAndCoplanar();
			return m;
		}

		// -----------------------------------------------------------------------
		// Cylinder
		// -----------------------------------------------------------------------

		/// <summary>
		/// Constructs a cylinder (or frustum/cone) by extruding a circle polygon.
		/// </summary>
		/// <param name="height">Z extent (must be &gt; 0).</param>
		/// <param name="radiusLow">Radius at bottom (must be ≥ 0).</param>
		/// <param name="radiusHigh">
		/// Radius at top (&lt; 0 means same as low). If both radii are 0 the result is empty.
		/// </param>
		/// <param name="circularSegments">Number of sides (0 = auto from Quality).</param>
		/// <param name="center">If true, center vertically on the origin.</param>
		/// <returns>The cylinder impl, or an empty one for degenerate input.</returns>
		public static ManifoldImpl Cylinder(
			double height,
			double radiusLow,
			double radiusHigh,
			int circularSegments,
			bool center)
		{
			if (height <= 0.0 || radiusLow < 0.0)
			{
				return new ManifoldImpl();
			}

			if (radiusLow == 0.0)
			{
				if (radiusHigh <= 0.0)
				{
					return new ManifoldImpl();
				}

				// Cone with apex at bottom: C++ builds the centered apex-at-top cone,
				// Mirrors over z, Translates, and finishes with AsOriginal. Mirror and
				// Translate are lazy CSG transforms that compose into ONE
				// Impl::Transform application (which also flips triangle winding for
				// the negative-determinant mirror) — replicate that exactly.
				ManifoldImpl coneSource = Cylinder(height, radiusHigh, 0.0, circularSegments, true);
				double translateZ = center ? 0.0 : height / 2.0;
				Mat3x4 mirror = Mat3x4.FromCols(
					new Vec3(1.0, 0.0, 0.0),
					new Vec3(0.0, 1.0, 0.0),
					new Vec3(0.0, 0.0, -1.0),
					new Vec3(0.0, 0.0, translateZ));
				ManifoldImpl cone = coneSource.Transform(mirror);

				// AsOriginal
				cone.InitializeOriginal();
				FaceOp.SetNormalsAndCoplanar(cone);
				return cone;
			}

			double scale = radiusHigh >= 0.0 ? radiusHigh / radiusLow : 1.0;
			double radius = MaxF64(radiusLow, radiusHigh >= 0.0 ? radiusHigh : 0.0);
			int n = circularSegments > 2
				? circularSegments
				: Quality.GetCircularSegments(radius);

			double dPhi = 360.0 / (double)n;
			SimplePolygon circle = new SimplePolygon(n);
			for (int i = 0; i < n; i++)
			{
				circle.Add(new Vec2(
					radiusLow * Types.Cosd(dPhi * (double)i),
					radiusLow * Types.Sind(dPhi * (double)i)));
			}

			ManifoldImpl m = Extrude(
				new Polygons { circle },
				height,
				0,
				0.0,
				new Vec2(scale, scale));

			if (center)
			{
				// Rust `for v in m.vert_pos.iter_mut()`: an in-place edit of each element.
				// Vec3 is a mutable struct in a List, so a foreach would mutate a copy —
				// the span gives the same by-reference iteration the Rust has.
				Span<Vec3> verts = CollectionsMarshal.AsSpan(m.VertPos);
				for (int i = 0; i < verts.Length; i++)
				{
					verts[i].Z -= height / 2.0;
				}

				m.CalculateBBox();
			}

			return m;
		}

		// -----------------------------------------------------------------------
		// Helpers
		// -----------------------------------------------------------------------

		// There are two `lerp2`s in the Rust, and this is the other one. `linalg::lerp2`
		// is `a*(1-t) + b*t` — the form docs/PORTING_PLAN.md's lerp rule names, ported here
		// as LinalgFunctions.Lerp(Vec2, Vec2, double). constructors.rs declares its own
		// private `lerp2` as `a + (b - a) * t` and does not import the linalg one, so the
		// local definition shadows it and `extrude` calls *this* function. So the rule is
		// not being overridden; a different function is being ported. C# needs no shadowing
		// to say that — the two keep distinct names (Lerp2 here, Lerp there).
		//
		// The distinction is not cosmetic: the two forms differ by an ULP, and that ULP
		// reaches extrude's side-wall vertices through `scale`.
		private static Vec2 Lerp2(Vec2 a, Vec2 b, double t)
		{
			return new Vec2(a.X + ((b.X - a.X) * t), a.Y + ((b.Y - a.Y) * t));
		}
	}
}
