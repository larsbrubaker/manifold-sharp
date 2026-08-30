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

// Arrangement.Candidates.cs — robust/arrangement.rs's `candidate_points`, the
// registry pre-pass the intersection-graph build runs before the real
// arrangements. The module header is on Arrangement.cs.

using ManifoldSharp.Linalg;
using ManifoldSharp.Robust.Exact;

namespace ManifoldSharp.Robust
{
	public static partial class ArrangementFunctions
	{
		/// <summary>
		/// The exact points this input would register on <paramref name="tri"/>, without
		/// running the CDT: primitive endpoints, isolated points, and pairwise proper
		/// segment crossings. robust/intersection_graph.rs uses this to build the global
		/// registries that force a common subdivision on both sides of every shared
		/// intersection segment and mesh edge before the real arrangements run.
		/// </summary>
		/// <remarks>
		/// Returns null only when <paramref name="token"/> is cancelled (see
		/// <see cref="Build"/>).
		/// </remarks>
		/// <param name="tri">The triangle, three vertices.</param>
		/// <param name="input">The primitives that landed on it.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The candidate points in registration order, or null when cancelled.</returns>
		/// <exception cref="ArgumentException">The array is not exactly three vertices.</exception>
		public static List<R3>? CandidatePoints(Vec3[] tri, ArrangementInput input, CancelToken? token)
		{
			ArgumentNullException.ThrowIfNull(tri);
			ArgumentNullException.ThrowIfNull(input);
			if (tri.Length != 3)
			{
				throw new ArgumentException($"a triangle is three vertices, got {tri.Length}", nameof(tri));
			}

			R3[] corners = new R3[]
			{
				R3.FromVec3(tri[0]),
				R3.FromVec3(tri[1]),
				R3.FromVec3(tri[2]),
			};
			R3 normal = Predicates.TriNormalR(corners[0], corners[1], corners[2]);
			int axis = TriTri.DominantAxis(normal);

			List<R3> output = new List<R3>();

			// Membership-only set with division-free R2Key hashing; `output` keeps
			// insertion order so determinism is unaffected (the set itself is never
			// iterated, which is why the Fx hasher is safe here — and why the plain
			// HashSet this port swaps in is safe for exactly the same reason).
			HashSet<R2Key> seen = new HashSet<R2Key>();

			foreach ((R3 p, int _) in input.Points)
			{
				Add(p);
			}

			foreach ((R3 a, R3 b, int _) in input.Segments)
			{
				Add(a);
				Add(b);
			}

			// Same filtered predicate stack as Build(): correctly rounded f64
			// approximations certify almost every non-crossing pair, homogenized
			// exact signs only on near-degeneracies — the all-rational version made
			// this sweep a dominant pipeline stage on segment-heavy meshes.
			(R2 A, R2 B)[] segs2 = new (R2 A, R2 B)[input.Segments.Count];
			for (int i = 0; i < segs2.Length; i++)
			{
				(R3 a, R3 b, int _) = input.Segments[i];
				segs2[i] = (a.ProjectDrop(axis), b.ProjectDrop(axis));
			}

			(Homog2 A, Homog2 B)[] homogs = new (Homog2 A, Homog2 B)[segs2.Length];
			for (int i = 0; i < segs2.Length; i++)
			{
				homogs[i] = (Predicates.Homog2Of(segs2[i].A), Predicates.Homog2Of(segs2[i].B));
			}

			// Same translated-filter-input lever as `Build`, with the same origin
			// (the triangle's first corner, projected) and the same soundness
			// argument: orient2d_a's certified sign is translation invariant and its
			// exact fallback still runs on the untranslated homogs, while the box
			// sweep stays conservative about the translated points and only ever
			// proposes candidates that the exact strict-crossing test then decides.
			// `output` is keyed and ordered by untranslated rationals, so the set and
			// sequence of registered points are unchanged.
			// No CDT runs on this path, so only the filter inputs are needed — no
			// exactness flags.
			R2 origin = corners[0].ProjectDrop(axis);
			(Vec2 A, Vec2 B)[] apts = new (Vec2 A, Vec2 B)[segs2.Length];
			for (int i = 0; i < segs2.Length; i++)
			{
				apts[i] = (
					TranslatedApprox(origin, segs2[i].A).Point,
					TranslatedApprox(origin, segs2[i].B).Point);
			}

			Box2[] segBoxes = new Box2[apts.Length];
			for (int i = 0; i < apts.Length; i++)
			{
				segBoxes[i] = ApproxBox([apts[i].A, apts[i].B]);
			}

			// Same order-restoring sweep as Build(): pairs come back in (i, ascending
			// j) order, so `output` receives the same crossings in the same sequence.
			BoxPairs? pairs = OverlappingBoxPairs(segBoxes, token);
			if (pairs is null)
			{
				return null;
			}

			int k = 0;
			foreach ((int i, int j) in pairs.Iter())
			{
				if (k % 1024 == 0 && Cancel.IsCancelled(token))
				{
					return null;
				}

				k++;
				(Vec2 Pt, Homog2 H) a = (apts[i].A, homogs[i].A);
				(Vec2 Pt, Homog2 H) b = (apts[i].B, homogs[i].B);
				(Vec2 Pt, Homog2 H) c = (apts[j].A, homogs[j].A);
				(Vec2 Pt, Homog2 H) d = (apts[j].B, homogs[j].B);
				Sign sc = O2(a, b, c);
				Sign sd = O2(a, b, d);
				Sign sa = O2(c, d, a);
				Sign sb = O2(c, d, b);
				if (sc != Sign.Zero
					&& sd != Sign.Zero
					&& sc != sd
					&& sa != Sign.Zero
					&& sb != Sign.Zero
					&& sa != sb)
				{
					R2 x2 = Predicates.LineLineIntersect2d(segs2[i].A, segs2[i].B, segs2[j].A, segs2[j].B)
						?? throw new InvalidOperationException("properly crossing segments are not parallel");
					Add(TriTri.LiftToPlane(x2, axis, corners[0], normal));
				}
			}

			return output;

			void Add(R3 p3)
			{
				if (seen.Add(new R2Key(p3.ProjectDrop(axis))))
				{
					output.Add(p3);
				}
			}

			static Sign O2((Vec2 Pt, Homog2 H) a, (Vec2 Pt, Homog2 H) b, (Vec2 Pt, Homog2 H) c)
			{
				return Approx.Orient2dA(a.Pt, b.Pt, c.Pt) ?? Predicates.Orient2dH(a.H, b.H, c.H);
			}
		}
	}
}
