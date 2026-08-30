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

// Smoothing tangent methods — extracted from smoothing.rs
// Contains: vert_halfedge, sharpen_edges, sharpen_tangent, linearize_flat_tangents,
//           distribute_tangents, create_tangents_from_normals, create_tangents
//
// ── File split ───────────────────────────────────────────────────────────────
// smoothing_tangents.rs is one `impl ManifoldImpl` block of 594 lines, past the
// 800-line cap once expanded, so it lands as two partials:
//
//   SmoothingTangents.cs         this file — VertHalfedge, SharpenEdges,
//                                SharpenTangent, LinearizeFlatTangents,
//                                DistributeTangents
//   SmoothingTangents.Create.cs  CreateTangentsFromNormals, ValidTangents,
//                                CreateTangents — the three entry points, each
//                                of which ends by calling DistributeTangents
//
// The smoothing module header is in Smoothing.cs, which is where the Rust puts
// it too (this file is `#[path]`-included by smoothing.rs).
//
// ── Orbit walks here ─────────────────────────────────────────────────────────
// Continuing the audit in Smoothing.cs's header — three of the port's eight
// walks are here, none of which is ManifoldImpl.ForVert:
//
//   DistributeTangents, measuring pass
//                     steps FIRST, so the seed halfedge is measured LAST, when
//                     the walk comes back around to it; that closing visit is
//                     the one that pushes K_TWO_PI instead of a measured angle,
//                     which is what makes the full-orbit case distinguishable
//                     from the fixed-edge-to-fixed-edge case. Skips inside-quad
//                     halfedges with `continue` (but still breaks if the skipped
//                     one is the seed), breaks on any other fixed halfedge, and
//                     carries a step guard of halfedge.len() + 1 against a
//                     malformed orbit.
//   DistributeTangents, rotating pass
//                     the same walk again, and it must visit the same halfedges
//                     in the same order — the angle arrays are indexed by an `i`
//                     that only advances on a non-skipped halfedge. Its extra
//                     guard (`current != start && fixed`) is the #1671 change
//                     that stops the terminating fixed edge from being rotated;
//                     the measuring pass deliberately does NOT have it, because
//                     that edge's angle still had to be measured.
//   CreateTangentsFromNormals, missing-normal fill
//                     runs BACKWARD — paired(PrevHalfedge(current)) — and
//                     processes the start halfedge first, because it is filling
//                     in from the last good normal it walked past.
//
// CreateTangents itself does not walk; it calls Smoothing.CollectVertexCycle,
// which materializes the orbit first and so is safe to mutate tangents inside.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <content>
	/// The tangent half of the smoothing port: seeding, sharpening, flattening and
	/// redistribution of halfedge tangents.
	/// </content>
	public sealed partial class ManifoldImpl
	{
		/// <summary>
		/// For each vertex, the lowest-indexed halfedge that starts there; -1 for an
		/// unreferenced vertex.
		/// </summary>
		/// <returns>One halfedge index per vertex.</returns>
		public List<int> VertHalfedge()
		{
			int numVert = this.NumVert();
			List<int> vertHalfedge = new List<int>(numVert);
			for (int i = 0; i < numVert; i++)
			{
				vertHalfedge.Add(-1);
			}

			for (int idx = 0; idx < this.Halfedge.Count; idx++)
			{
				int start = this.Halfedge[idx].StartVert;
				if (vertHalfedge[start] < 0)
				{
					vertHalfedge[start] = idx;
				}
			}

			return vertHalfedge;
		}

		/// <summary>
		/// The halfedge pairs whose dihedral exceeds <paramref name="minSharpAngle"/>,
		/// each returned twice (once per direction) at <paramref name="minSmoothness"/>.
		/// </summary>
		/// <param name="minSharpAngle">The dihedral threshold, in degrees.</param>
		/// <param name="minSmoothness">The smoothness to assign the sharp edges.</param>
		/// <returns>Two entries per sharp edge, forward then paired.</returns>
		public List<Smoothness> SharpenEdges(double minSharpAngle, double minSmoothness)
		{
			List<Smoothness> outList = new List<Smoothness>();

			// Clamp to avoid float-noise false positives (matches C++ kMinSharpAngle)
			double minRadians = Types.Radians(MaxF64(minSharpAngle, 1e-4));
			for (int e = 0; e < this.Halfedge.Count; e++)
			{
				if (!this.Halfedge[e].IsForward())
				{
					continue;
				}

				int pair = this.Halfedge[e].PairedHalfedge;
				double d = ClampF64(Dot(this.FaceNormal[e / 3], this.FaceNormal[pair / 3]), -1.0, 1.0);
				double dihedral = DeterministicMath.Acos(d);
				if (dihedral > minRadians)
				{
					outList.Add(new Smoothness(e, minSmoothness));
					outList.Add(new Smoothness(pair, minSmoothness));
				}
			}

			return outList;
		}

		/// <summary>
		/// Scales a halfedge's tangent toward its vertex, zeroing the Bezier weight when
		/// the smoothness is exactly zero.
		/// </summary>
		/// <param name="halfedge">The halfedge whose tangent to sharpen.</param>
		/// <param name="smoothness">0 = fully sharp, 1 = unchanged.</param>
		public void SharpenTangent(int halfedge, double smoothness)
		{
			this.HalfedgeTangent[halfedge] = new Vec4(
				smoothness * this.HalfedgeTangent[halfedge].X,
				smoothness * this.HalfedgeTangent[halfedge].Y,
				smoothness * this.HalfedgeTangent[halfedge].Z,
				smoothness == 0.0 ? 0.0 : this.HalfedgeTangent[halfedge].W);
		}

		/// <summary>
		/// Replaces zero-weight (fully sharpened) tangents with the linear interpolation
		/// of the edge, so a sharpened edge renders as a straight line rather than a
		/// collapsed curve.
		/// </summary>
		public void LinearizeFlatTangents()
		{
			for (int halfedge = 0; halfedge < this.HalfedgeTangent.Count; halfedge++)
			{
				if (!this.Halfedge[halfedge].IsForward())
				{
					continue;
				}

				int pair = this.Halfedge[halfedge].PairedHalfedge;
				Vec4 tangent = this.HalfedgeTangent[halfedge];
				Vec4 other = this.HalfedgeTangent[pair];
				bool[] flat = new bool[] { tangent.W == 0.0, other.W == 0.0 };
				if (!flat[0] && !flat[1])
				{
					continue;
				}

				Vec3 edgeVec = this.VertPos[this.Halfedge[halfedge].EndVert]
					- this.VertPos[this.Halfedge[halfedge].StartVert];

				if (flat[0] && flat[1])
				{
					this.HalfedgeTangent[halfedge] =
						new Vec4(edgeVec.X / 3.0, edgeVec.Y / 3.0, edgeVec.Z / 3.0, 1.0);
					this.HalfedgeTangent[pair] =
						new Vec4(-edgeVec.X / 3.0, -edgeVec.Y / 3.0, -edgeVec.Z / 3.0, 1.0);
				}
				else if (flat[0])
				{
					Vec3 otherV = Smoothing.Vec3FromVec4(other);
					Vec3 v = (edgeVec + otherV) / 2.0;
					this.HalfedgeTangent[halfedge] = new Vec4(v.X, v.Y, v.Z, 1.0);
				}
				else
				{
					Vec3 tanV = Smoothing.Vec3FromVec4(tangent);
					Vec3 v = (-edgeVec + tanV) / 2.0;
					this.HalfedgeTangent[pair] = new Vec4(v.X, v.Y, v.Z, 1.0);
				}
			}
		}

		/// <summary>
		/// Rotates the free tangents around each vertex so their angular spacing matches
		/// the angular spacing of the edges, holding the fixed ones in place.
		/// </summary>
		/// <param name="fixedHalfedges">Per-halfedge flags marking the tangents that may not move.</param>
		public void DistributeTangents(IReadOnlyList<bool> fixedHalfedges)
		{
			for (int halfedge = 0; halfedge < fixedHalfedges.Count; halfedge++)
			{
				// Per #1671: skip non-fixed and inside-quad seeds outright (the old
				// code re-seeded inside-quad halfedges to their neighbor).
				if (!fixedHalfedges[halfedge] || this.IsMarkedInsideQuad(halfedge))
				{
					continue;
				}

				int start = halfedge;

				Vec3 normal = new Vec3(0.0, 0.0, 0.0);
				List<double> currentAngle = new List<double>();
				List<double> desiredAngle = new List<double>();

				Vec3 approxNormal = this.VertNormal[this.Halfedge[start].StartVert];
				Vec3 center = this.VertPos[this.Halfedge[start].StartVert];
				Vec3 lastEdgeVec = Smoothing.SafeNormalize(this.VertPos[this.Halfedge[start].EndVert] - center);
				Vec3 firstTangent = Smoothing.SafeNormalize(Smoothing.Vec3FromVec4(this.HalfedgeTangent[start]));
				Vec3 lastTangent = firstTangent;
				int current = start;
				int guard = 0;

				// ORBIT WALK (measuring pass) — see this file's header.
				while (true)
				{
					guard += 1;
					if (guard > this.Halfedge.Count + 1)
					{
						break;
					}

					current = NextHalfedge(this.Halfedge[current].PairedHalfedge);
					if (this.IsMarkedInsideQuad(current))
					{
						if (current == start)
						{
							break;
						}

						continue;
					}

					Vec3 thisEdgeVec = Smoothing.SafeNormalize(this.VertPos[this.Halfedge[current].EndVert] - center);
					Vec3 thisTangent = Smoothing.SafeNormalize(Smoothing.Vec3FromVec4(this.HalfedgeTangent[current]));
					normal = normal + Cross(thisTangent, lastTangent);

					double cumulative = Smoothing.AngleBetween(thisEdgeVec, lastEdgeVec)
						+ (desiredAngle.Count > 0 ? desiredAngle[desiredAngle.Count - 1] : 0.0);
					desiredAngle.Add(cumulative);

					if (current == start)
					{
						currentAngle.Add(Types.KTwoPi);
					}
					else
					{
						double angle = Smoothing.AngleBetween(thisTangent, firstTangent);
						if (Dot(approxNormal, Cross(thisTangent, firstTangent)) < 0.0)
						{
							angle = Types.KTwoPi - angle;
						}

						currentAngle.Add(angle);
					}

					lastEdgeVec = thisEdgeVec;
					lastTangent = thisTangent;
					if (fixedHalfedges[current])
					{
						break;
					}
				}

				if (currentAngle.Count == 1 || Dot(normal, normal) == 0.0)
				{
					continue;
				}

				double scale = (currentAngle.Count > 0 ? currentAngle[currentAngle.Count - 1] : Types.KTwoPi)
					/ (desiredAngle.Count > 0 ? desiredAngle[desiredAngle.Count - 1] : Types.KTwoPi);
				double offset = 0.0;
				if (current == start)
				{
					for (int i = 0; i < currentAngle.Count; i++)
					{
						offset += Smoothing.Wrap(currentAngle[i] - (scale * desiredAngle[i]));
					}

					offset /= (double)currentAngle.Count;
				}

				current = start;
				Vec3 axis = Smoothing.SafeNormalize(normal);
				int idx = 0;
				guard = 0;

				// ORBIT WALK (rotating pass) — same order as the measuring pass, with the
				// one extra guard #1671 added; see this file's header.
				while (true)
				{
					guard += 1;
					if (guard > this.Halfedge.Count + 1)
					{
						break;
					}

					current = NextHalfedge(this.Halfedge[current].PairedHalfedge);

					// Per #1671: stop before processing a *different* fixed halfedge
					// (the terminating fixed edge is no longer rotated here).
					if (current != start && fixedHalfedges[current])
					{
						break;
					}

					if (this.IsMarkedInsideQuad(current))
					{
						if (current == start)
						{
							break;
						}

						continue;
					}

					desiredAngle[idx] *= scale;
					double lastAngle = idx > 0 ? desiredAngle[idx - 1] : 0.0;
					if (desiredAngle[idx] - lastAngle > Types.KPi)
					{
						desiredAngle[idx] = lastAngle + Types.KPi;
					}
					else if (idx + 1 < desiredAngle.Count
						&& (scale * desiredAngle[idx + 1]) - desiredAngle[idx] > Types.KPi)
					{
						desiredAngle[idx] = (scale * desiredAngle[idx + 1]) - Types.KPi;
					}

					double angle = currentAngle[idx] - desiredAngle[idx] - offset;
					Vec3 tangent = Smoothing.Vec3FromVec4(this.HalfedgeTangent[current]);
					Vec4 q = RotationQuatAxisAngle(axis, angle);
					Vec3 rotated = QRot(q, tangent);
					this.HalfedgeTangent[current] = new Vec4(
						rotated.X,
						rotated.Y,
						rotated.Z,
						this.HalfedgeTangent[current].W);
					idx += 1;
					if (fixedHalfedges[current])
					{
						break;
					}
				}
			}
		}

		/// <summary>
		/// Rust's <c>f64::clamp</c>: NaN propagates (every comparison is false), and the
		/// bounds are assumed already ordered. The twin of the private helper of the same
		/// name in Collider.Geometry.cs — neither is public, and this port keeps the
		/// spelling local rather than growing the LinalgFunctions surface for it.
		/// </summary>
		private static double ClampF64(double value, double lo, double hi)
		{
			if (value < lo)
			{
				return lo;
			}

			if (value > hi)
			{
				return hi;
			}

			return value;
		}
	}
}
