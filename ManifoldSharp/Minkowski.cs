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

// Minkowski.cs — port of minkowski.rs, whose header reads:
//
//   Phase 17: Minkowski Sum/Difference — ported from C++ minkowski.cpp (175 lines)
//
//   Implements the Minkowski sum/difference using:
//   - Convex+Convex: pairwise vertex sums → Hull
//   - NonConvex+Convex: per-triangle vertex sums → Hull → BatchBoolean (in batches)
//   - NonConvex+NonConvex: per-face-pair sums with coplanarity filtering → BatchBoolean
//
// Both per-face hull loops go through Par.MaybeParMap: they are the "minkowski
// hulls" entry in the six sites docs/PORTING_PLAN.md blesses for Phase 11
// parallelism, and the ONLY place in the port carrying a documented determinism
// exception. Verbatim from the Rust port's plan, where it is the single caveat
// on "parallel is bit-identical to sequential":
//
//   One documented caveat: mesh-ID *values* from the global atomic counter can
//   be consumed in a different order under parallel minkowski hulls — they are
//   opaque process-global handles (already test-order dependent) and do not
//   affect geometry, topology, or any assertion.
//
// Concretely: each ConvexHull below calls InitializeOriginal, which takes an ID
// from ManifoldImpl.ReserveIds. Sequentially the hulls take them in face order;
// in parallel they take the same SET of IDs in whatever order the workers get
// there. Geometry, topology, counts and every TriRef *relationship* are
// unaffected — only which opaque integer labels which hull. Nothing may be
// tightened here that turns those handles into observable values.
//
// NAME COLLISION, deliberate and load-bearing: this class makes `Minkowski` an
// unqualified name inside namespace ManifoldSharp, and Clipper2Lib has a
// `Minkowski` class of its own. A same-namespace type beats a using-directive
// one, so inside this assembly the bare spelling `Minkowski` now binds HERE.
// CrossSection.Clipper.cs therefore writes `Clipper2Lib.Minkowski.Sum(...)`
// fully qualified — that qualification is what keeps it compiling against the
// 2D Clipper sum, and removing it as "redundant" would silently rebind it to
// this class (or, more likely, fail to compile in a confusing way).
//
// Rust module-level functions land on a static class named for the module, so
// `minkowski::minkowski` would be `Minkowski.Minkowski` — which C# forbids (a
// member may not share its type's name). The three entry points are spelled
// Compute / Sum / Difference; each names the Rust function it ports.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// The free functions of <c>minkowski.rs</c> — Minkowski sum and difference of two
	/// meshes.
	/// </summary>
	public static class Minkowski
	{
		private const int BatchSize = 1000;
		private const int ReduceThreshold = 200;
		private const double KCoplanarTol = 1e-12;

		/// <summary>
		/// Compute the Minkowski sum or difference of two meshes.
		/// Port of C++ <c>Manifold::Impl::Minkowski()</c> (Rust <c>minkowski</c>).
		/// </summary>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <param name="inset">
		/// If true, computes the Minkowski difference (erosion); if false, computes the
		/// Minkowski sum (dilation).
		/// </param>
		/// <returns>The resulting mesh.</returns>
		public static ManifoldImpl Compute(ManifoldImpl a, ManifoldImpl b, bool inset)
		{
			ArgumentNullException.ThrowIfNull(a);
			ArgumentNullException.ThrowIfNull(b);

			ManifoldImpl aImpl = a;
			ManifoldImpl bImpl = b;

			bool aConvex = aImpl.IsConvex();
			bool bConvex = bImpl.IsConvex();

			// If the convex manifold was supplied first, swap them
			if (aConvex && !bConvex)
			{
				aImpl = b;
				bImpl = a;
				(aConvex, bConvex) = (bConvex, aConvex);
			}

			// Early-exit if either input is empty
			if (bImpl.IsEmpty())
			{
				return aImpl.Clone();
			}

			if (aImpl.IsEmpty())
			{
				return bImpl.Clone();
			}

			List<ManifoldImpl> composedHulls = new List<ManifoldImpl>();
			composedHulls.Add(aImpl.Clone());

			// Convex-Convex Minkowski: Very Fast
			if (!inset && aConvex && bConvex)
			{
				// Rust reserves `b.len * a.len` in usize (64-bit); the same product in C#
				// `int` wraps negative past 2^31, and List's constructor would then answer a
				// 46,341-vert operand with a bare "capacity out of range". Both ports are
				// doomed on an input this size — the pairwise sum alone is >2 billion Vec3 —
				// so widen only to fail honestly about which limit was hit.
				long pairCount = (long)bImpl.VertPos.Count * aImpl.VertPos.Count;
				if (pairCount > int.MaxValue)
				{
					throw new ArgumentException(
						$"Convex Minkowski sum needs {pairCount} vertex pairs "
						+ $"({aImpl.VertPos.Count} x {bImpl.VertPos.Count}), past the "
						+ $"{int.MaxValue} a single hull input can hold.");
				}

				List<Vec3> simpleHull = new List<Vec3>((int)pairCount);
				foreach (Vec3 aVert in aImpl.VertPos)
				{
					foreach (Vec3 bVert in bImpl.VertPos)
					{
						simpleHull.Add(aVert + bVert);
					}
				}

				composedHulls.Add(QuickHullFunctions.ConvexHull(simpleHull));
			}

			// Convex + Non-Convex (or inset): Slower
			else if ((inset || !aConvex) && bConvex)
			{
				int numTri = aImpl.NumTri();

				// Process in batches. Each per-triangle hull is independent (C++ runs
				// this loop via for_each_n); results are collected in index order so
				// the batch content matches sequential. C++ pushes every hull
				// unconditionally — no empty filter here (unlike the
				// non-convex×non-convex branch); filtering would shift BatchBoolean
				// serials and change the reduction order.
				int offset = 0;
				while (offset < numTri)
				{
					int numIter = Math.Min(numTri - offset, BatchSize);
					int batchOffset = offset;
					ManifoldImpl[] newHulls = Par.MaybeParMap(numIter, 8, iter =>
					{
						int tri = batchOffset + iter;
						List<Vec3> simpleHull = new List<Vec3>(3 * bImpl.VertPos.Count);
						for (int i = 0; i < 3; i++)
						{
							Vec3 aVert = aImpl.VertPos[aImpl.Halfedge[(tri * 3) + i].StartVert];
							foreach (Vec3 bVert in bImpl.VertPos)
							{
								simpleHull.Add(aVert + bVert);
							}
						}

						return QuickHullFunctions.ConvexHull(simpleHull);
					});

					composedHulls.Add(BatchBooleanImpls(newHulls, OpType.Add));
					offset += BatchSize;
				}
			}

			// Non-Convex + Non-Convex: Very Slow
			else if (!aConvex && !bConvex)
			{
				int numTriA = aImpl.NumTri();
				int numTriB = bImpl.NumTri();

				List<ManifoldImpl> accumulated = new List<ManifoldImpl>();

				for (int aFace = 0; aFace < numTriA; aFace++)
				{
					Vec3 a1 = aImpl.VertPos[aImpl.Halfedge[aFace * 3].StartVert];
					Vec3 a2 = aImpl.VertPos[aImpl.Halfedge[(aFace * 3) + 1].StartVert];
					Vec3 a3 = aImpl.VertPos[aImpl.Halfedge[(aFace * 3) + 2].StartVert];
					Vec3 nA = aImpl.FaceNormal[aFace];

					// Per-B-face hulls are independent (C++ parallel for_each_n over
					// bFace); collect in index order, then filter like C++'s
					// validFaceHulls pass so batch content and order match sequential.
					ManifoldImpl?[] hulls = Par.MaybeParMap<ManifoldImpl?>(numTriB, 8, bFace =>
					{
						Vec3 nB = bImpl.FaceNormal[bFace];
						double dotSame = Dot(nA, nB);
						double dotOpp = Dot(nA, new Vec3(-nB.X, -nB.Y, -nB.Z));
						bool coplanar = Math.Abs(dotSame - 1.0) < KCoplanarTol
							|| Math.Abs(dotOpp - 1.0) < KCoplanarTol;
						if (coplanar)
						{
							return null;
						}

						Vec3 b1 = bImpl.VertPos[bImpl.Halfedge[bFace * 3].StartVert];
						Vec3 b2 = bImpl.VertPos[bImpl.Halfedge[(bFace * 3) + 1].StartVert];
						Vec3 b3 = bImpl.VertPos[bImpl.Halfedge[(bFace * 3) + 2].StartVert];

						return QuickHullFunctions.ConvexHull(new Vec3[]
						{
							a1 + b1,
							a1 + b2,
							a1 + b3,
							a2 + b1,
							a2 + b2,
							a2 + b3,
							a3 + b1,
							a3 + b2,
							a3 + b3,
						});
					});

					List<ManifoldImpl> faceHulls = new List<ManifoldImpl>();
					foreach (ManifoldImpl? hull in hulls)
					{
						if (hull is not null && !hull.IsEmpty())
						{
							faceHulls.Add(hull);
						}
					}

					if (faceHulls.Count > 0)
					{
						accumulated.Add(BatchBooleanImpls(faceHulls, OpType.Add));
					}

					// Periodically reduce to limit memory
					if (accumulated.Count >= ReduceThreshold)
					{
						ManifoldImpl reduced = BatchBooleanImpls(accumulated, OpType.Add);
						accumulated.Clear();
						accumulated.Add(reduced);
					}
				}

				if (accumulated.Count > 0)
				{
					composedHulls.Add(BatchBooleanImpls(accumulated, OpType.Add));
				}
			}

			// Final merge; C++ finishes with AsOriginal() = InitializeOriginal +
			// SetNormalsAndCoplanar.
			OpType op = inset ? OpType.Subtract : OpType.Add;
			ManifoldImpl outR = BatchBooleanImpls(composedHulls, op);
			outR.InitializeOriginal();
			outR.SetNormalsAndCoplanar();
			return outR;
		}

		/// <summary>
		/// Convenience wrapper: Minkowski sum (dilation). The Rust <c>minkowski_sum</c>.
		/// </summary>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <returns>The dilated mesh.</returns>
		public static ManifoldImpl Sum(ManifoldImpl a, ManifoldImpl b)
		{
			return Compute(a, b, false);
		}

		/// <summary>
		/// Convenience wrapper: Minkowski difference (erosion). The Rust
		/// <c>minkowski_difference</c>.
		/// </summary>
		/// <param name="a">The first operand.</param>
		/// <param name="b">The second operand.</param>
		/// <returns>The eroded mesh.</returns>
		public static ManifoldImpl Difference(ManifoldImpl a, ManifoldImpl b)
		{
			return Compute(a, b, true);
		}

		/// <summary>Helper: BatchBoolean on ManifoldImpl directly via the CSG tree.</summary>
		/// <param name="meshes">The operands.</param>
		/// <param name="op">The operation.</param>
		/// <returns>The reduced mesh.</returns>
		private static ManifoldImpl BatchBooleanImpls(IReadOnlyList<ManifoldImpl> meshes, OpType op)
		{
			if (meshes.Count == 0)
			{
				return new ManifoldImpl();
			}

			if (meshes.Count == 1)
			{
				return meshes[0].Clone();
			}

			CsgNode[] children = new CsgNode[meshes.Count];
			for (int i = 0; i < meshes.Count; i++)
			{
				children[i] = new CsgLeaf(new CsgLeafNode(meshes[i].Clone()));
			}

			CsgNode tree = new CsgOp(op, children);
			return tree.Evaluate();
		}
	}
}
