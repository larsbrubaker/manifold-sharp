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

// ConvexErosion.cs — NOT A PORT. There is no `convex_erosion.rs`: manifold-rust's
// minkowski.rs has one erosion algorithm and it is the per-triangle sweep in
// Minkowski.cs. This file adds a second, closed-form one that only applies when the
// eroded solid is convex, and it is divergence ledger entry 5
// (docs/RUST_DIVERGENCES.md) — an ADDED entry point, reached only when a caller asks
// for it by name. `Minkowski.Compute` is not routed through it and does not know it
// exists, so every ported path still produces the Rust's bits.
//
// ── The closed form ─────────────────────────────────────────────────────────────
// A convex solid IS the intersection of its face halfspaces:
//
//     A = { x : n_i . x <= d_i }   over A's faces i, n_i outward and unit
//
// and the erosion of a convex A by any B is that same intersection with each plane
// pushed inward by B's support in that direction:
//
//     A (-) B = { x : n_i . x <= d_i - h_B(-n_i) },   h_B(u) = max over b in B of u.b
//
// which is EXACT — no tessellation, no boolean, no per-triangle hull. Two facts make
// it work: erosion distributes over an intersection of halfspaces, and eroding a
// halfspace by B just slides its plane by B's support. Neither survives A being
// non-convex, which is why the applicability gate below is the whole design.
//
// The sign is `h_B(-n_i)`, not `h_B(n_i)`, because Minkowski.cs's erosion is
// A \ (boundary(A) (+) B) — it sweeps B, not -B, over the boundary — so what it
// computes is { x : x - B subset A }. For the centred ball every caller actually
// erodes with, B = -B and the distinction is invisible; it is written the matching
// way anyway so an asymmetric tool does not silently answer a different question.
//
// ── Why a dual hull rather than a plane-by-plane clip ───────────────────────────
// Intersecting n halfspaces by clipping is n booleans, which for anything past a
// prism costs more than the sweep it replaces. The polar dual turns it into two
// convex hulls: with p strictly inside the eroded body, the dual point q_i = n_i / c_i
// (c_i the plane's slack at p) has the property that a FACET of hull{q_i} names the
// three planes meeting at a VERTEX of the result. So the dual hull is used only to
// enumerate those triples; each vertex is then solved in the primal, from the
// original n and d, which keeps a box's corners exactly on their planes instead of
// round-tripping them through 1/c.
//
// ── Everything that makes it decline ───────────────────────────────────────────
// This is a fast path, so every case it is not sure of returns false and lets the
// caller run the general sweep. It declines on: a non-convex solid or tool, a tool
// that does not contain the origin (where the general path's formula and this one
// genuinely part company — see ToolContainsOrigin), a centroid that is not strictly
// inside the eroded body (so the dual is undefined; near-total erosion of a skew
// solid does this), a degenerate plane triple, and — the backstop — any output
// vertex that does not satisfy every constraint it was built from. Declining is
// always safe: the general path is the specification this one is measured against.
//
// ── Where it is exact, and where it is not ─────────────────────────────────────
// The arithmetic is exact and beats the sweep at it: a 20-cube eroded by a unit ball
// comes out at volume 5832.0 with no rounding at all, because every plane offset and
// every corner solve is whole-number. Agreement with the sweep is typically at 1e-15
// relative and holds there up to about a thousand faces.
//
// It is NOT exact on a dense solid, and the reason is the dual hull rather than the
// arithmetic. QuickHull discards points within its relative epsilon of an existing
// facet, and on a finely tessellated solid many dual points sit that close to one
// another — so a few halfspaces are dropped as if redundant when they are not quite.
// Measured: a 2048-triangle sphere (Sphere(10, 64)) eroded by a unit ball comes out
// with 4016 triangles against the sweep's 4024 and a relative volume difference of
// 4.9e-7. A 1152-triangle sphere already differs in triangle count (2544 against
// 2550) while the volumes still agree to 4e-15. Both are far inside the error the
// tessellated ball itself introduces, and ConvexErosionTests pins the 2048-triangle
// case so the size of it cannot drift unnoticed — but a caller that needs a dense
// convex erosion right to the last bit wants the sweep.

using ManifoldSharp.Linalg;

using static ManifoldSharp.Linalg.LinalgFunctions;

namespace ManifoldSharp
{
	/// <summary>
	/// The closed-form erosion of a <em>convex</em> solid — the halfspace-intersection
	/// answer to <see cref="Minkowski.Difference"/>, for the inputs that have one.
	/// </summary>
	/// <remarks>
	/// A fast path and nothing more: <see cref="TryCompute"/> answers false for every
	/// input it cannot prove itself on, and the caller then runs
	/// <see cref="Minkowski.Difference"/>, which is the specification. See the file
	/// header for the derivation and for the full list of declines.
	/// </remarks>
	public static class ConvexErosion
	{
		/// <summary>
		/// How often the support pass polls the cancel flag, as a power-of-two mask on the
		/// face index. The pass is one dot product per face per tool vertex, so a poll per
		/// face would cost more than the work it guards on a coarse tool.
		/// </summary>
		private const int CancelPollMask = 63;

		/// <summary>
		/// Relative slack a candidate interior point must clear, and the relative slop the
		/// output vertices are verified within, both scaled by the solid's bounding-box
		/// diagonal so they mean the same thing in millimetres and in metres.
		/// </summary>
		private const double RelativeTolerance = 1e-9;

		/// <summary>
		/// The erosion of <paramref name="solid"/> by <paramref name="tool"/> in closed
		/// form, when <paramref name="solid"/> is convex.
		/// </summary>
		/// <param name="solid">The shape being shrunk. Must be a manifold mesh.</param>
		/// <param name="tool">The structuring element that has to fit inside it.</param>
		/// <param name="token">
		/// The cancellation token, or null. A cancelled run answers <c>true</c> with an
		/// empty result carrying <see cref="Error.Cancelled"/> — <c>true</c>, not
		/// <c>false</c>, because a cancelled token must never send the caller off to run
		/// the general path it was cancelled out of.
		/// </param>
		/// <param name="progress">
		/// The progress reporter, or null. One <see cref="Phase.Minkowski"/> unit per face of
		/// <paramref name="solid"/> — the support pass, which is the only part of this with
		/// duration — plus one for everything after it, closed by
		/// <see cref="Progress.CompletePhase"/> on success. The cheap declines (a non-convex
		/// operand, a tool missing the origin) happen before the phase opens and report
		/// nothing at all, which is what leaves the general path's own <c>BeginPhase</c> the
		/// first thing the caller sees in the case that actually happens. The rare numeric
		/// declines after it leave the phase open and short; that is deliberate, because the
		/// alternative is a run with no seam at which a watcher could cancel it.
		/// </param>
		/// <param name="result">
		/// The eroded solid when this returns true; an empty impl when it returns false.
		/// </param>
		/// <returns>True when the closed form applied and <paramref name="result"/> holds
		/// the answer; false when the caller must run the general erosion.</returns>
		public static bool TryCompute(
			ManifoldImpl solid,
			ManifoldImpl tool,
			CancelToken? token,
			ProgressReporter? progress,
			out ManifoldImpl result)
		{
			ArgumentNullException.ThrowIfNull(solid);
			ArgumentNullException.ThrowIfNull(tool);

			result = new ManifoldImpl();

			// The entry gate every cancellable entry point in the port opens with. It
			// answers true so the caller stops here rather than falling through to the
			// general path, which would then have to discover the cancel all over again.
			if (Cancel.IsCancelled(token))
			{
				result = Boolean3Functions.CancelledImpl();
				return true;
			}

			// An empty or errored operand is the general path's business: Minkowski.Compute
			// answers those with the other operand's clone or a propagated status, and this
			// file is not the place to have a second opinion about what they mean.
			if (solid.IsEmpty()
				|| tool.IsEmpty()
				|| solid.IsSoup
				|| tool.IsSoup
				|| solid.Status != Error.NoError
				|| tool.Status != Error.NoError)
			{
				return false;
			}

			// The whole premise. IsConvex on the tool is not needed by the mathematics —
			// erosion by B and by hull(B) are the same thing when A is convex — but it is
			// needed for the promise this path makes to its caller, which is that it agrees
			// with the general erosion, and the general erosion of a convex solid by a
			// non-convex tool takes a branch that swaps its operands.
			if (!solid.IsConvex() || !tool.IsConvex())
			{
				return false;
			}

			if (!ToolContainsOrigin(tool))
			{
				return false;
			}

			// Everything above is O(faces) at worst and rejects without touching the
			// reporter, which matters: the common decline is a non-convex solid, and the
			// caller's next move is the sweep, which opens this same phase itself. From here
			// on the work is real - one dot product per face per tool vertex - so the phase
			// opens, reports per face, and can be cancelled somewhere a watcher can see.
			Progress.BeginPhase(progress, Phase.Minkowski, (ulong)solid.NumTri() + 1);

			if (!TrySupportPlanes(solid, tool, token, progress, out List<Vec3> normals, out List<double> offsets, out bool cancelled))
			{
				if (cancelled)
				{
					result = Boolean3Functions.CancelledImpl();
					return true;
				}

				return false;
			}

			Vec3 diagonal = solid.Bbox.Size();
			double tolerance = Length(diagonal) * RelativeTolerance;

			if (!TryInteriorPoint(solid, normals, offsets, tolerance, out Vec3 interior, out List<double> slacks))
			{
				return false;
			}

			if (!TryVertices(normals, offsets, slacks, interior, tolerance, out List<Vec3> vertices))
			{
				return false;
			}

			if (Cancel.IsCancelled(token))
			{
				result = Boolean3Functions.CancelledImpl();
				return true;
			}

			ManifoldImpl eroded = QuickHullFunctions.ConvexHull(vertices);

			// The closing check CancelToken.cs's invariant requires - "a cancelled token can
			// never produce a NoError result" - and the twin of the one Minkowski.Compute
			// makes after its last BatchBoolean. Without it a cancel that landed during the
			// final hull would come back dressed up as a finished erosion.
			if (Cancel.IsCancelled(token))
			{
				result = Boolean3Functions.CancelledImpl();
				return true;
			}

			if (eroded.IsEmpty())
			{
				// A result that collapsed to a point, a segment or a sheet. The general
				// path has a considered answer for those (usually an empty mesh, which is
				// what "the ball fits nowhere" is spelled as); this one only has a hull
				// that gave up, so hand the question back.
				return false;
			}

			// Reached only on success, so a full bar is never a claim about work that was
			// abandoned. The declines above leave the phase open and short; the sweep the
			// caller then runs opens it again, and a consumer that keeps a high-water mark
			// (agg-sharp's BooleanProgressAdapter does) sees a bar that pauses rather than
			// one that goes backwards.
			Progress.CompletePhase(progress);
			result = eroded;
			return true;
		}

		/// <summary>
		/// Whether the origin lies inside <paramref name="tool"/> — the condition under which
		/// the closed form and the general sweep are the same function.
		/// </summary>
		/// <remarks>
		/// This is the gate, and it has to be this one. The sweep computes
		/// <c>A \ (boundary(A) ⊕ B)</c>, which drops a point of A exactly when its swept copy
		/// <c>x - B</c> MEETS the boundary. When the origin is in B that is the erosion:
		/// <c>x</c> is in <c>x - B</c>, so a swept copy that leaves A has to cross out of it,
		/// and one that stays inside is kept. When the origin is NOT in B the swept copy can
		/// sail clear over the boundary and land wholly outside, and the sweep keeps that
		/// point too — so its answer stops being an erosion of anything.
		/// <para>
		/// Measured, both halves: a 20-cube swept by a unit ball centred at (2,0,0) comes back
		/// 5908 with the whole cube's bounding box, and by one centred at (0.8,0.8,0) comes
		/// back 5832.547 where the erosion is 5832. The second is the one that matters here —
		/// it is only 1.13 out, so a gate that merely asked whether the tool reaches past each
		/// of A's own face planes passed it and let the two paths disagree silently. Sampling
		/// A's normals cannot see a tool that escapes in a direction A has no face in. The
		/// tool's own planes can, exactly.
		/// </para>
		/// <para>
		/// Exact because the tool is already known convex: a convex solid contains the origin
		/// precisely when every outward face plane has a non-negative offset. The epsilon is
		/// the same relative one the rest of this file uses, scaled by the tool's own extent,
		/// so a tool with a face passing through the origin is admitted rather than rejected
		/// on a rounding bit — it contains the origin on its boundary, which is all the
		/// argument above needs.
		/// </para>
		/// </remarks>
		/// <param name="tool">The structuring element, already known convex and non-empty.</param>
		/// <returns>True when the origin is inside or on the tool.</returns>
		private static bool ToolContainsOrigin(ManifoldImpl tool)
		{
			double tolerance = Length(tool.Bbox.Size()) * RelativeTolerance;

			for (int tri = 0; tri < tool.NumTri(); tri++)
			{
				Vec3 normal = tool.FaceNormal[tri];
				double length = Length(normal);
				if (!(length > 0.0))
				{
					// Degenerate face, no plane to test against; the tool's faces with area
					// still bound it.
					continue;
				}

				normal = normal / length;
				double planeOffset = Dot(normal, tool.VertPos[tool.Halfedge[tri * 3].StartVert]);
				if (planeOffset < -tolerance)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>
		/// Collects the solid's face planes with each one already pushed inward by the
		/// tool's support in that direction — the <c>d_i - h_B(-n_i)</c> of the file header.
		/// </summary>
		/// <remarks>
		/// The support is taken against <c>-n_i</c>, which is the reflection
		/// <see cref="Minkowski"/>'s erosion carries: it sweeps B rather than -B over the
		/// boundary, so it answers <c>{ x : x - B subset A }</c>. Getting the sign backwards
		/// would mirror the result — invisible on the centred ball every real caller uses,
		/// and pinned by <c>ConvexErosionTests.AnAsymmetricToolErodesTheWayTheSweepDoes</c>,
		/// which uses a tool with different reach in +x and -x for exactly that reason.
		/// <para>
		/// Every push is non-negative here and does not need testing: the origin is inside
		/// the tool by the time this runs (<see cref="ToolContainsOrigin"/>), so
		/// <c>h_B(-n) >= -n . 0 = 0</c> for any direction.
		/// </para>
		/// </remarks>
		/// <param name="solid">The convex solid.</param>
		/// <param name="tool">The structuring element.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null. One unit per face.</param>
		/// <param name="normals">The outward unit normals, one per non-degenerate face.</param>
		/// <param name="offsets">The matching pushed-in plane offsets.</param>
		/// <param name="cancelled">True when the pass stopped on the cancel flag.</param>
		/// <returns>True when the planes are usable.</returns>
		private static bool TrySupportPlanes(
			ManifoldImpl solid,
			ManifoldImpl tool,
			CancelToken? token,
			ProgressReporter? progress,
			out List<Vec3> normals,
			out List<double> offsets,
			out bool cancelled)
		{
			int numTri = solid.NumTri();
			normals = new List<Vec3>(numTri);
			offsets = new List<double>(numTri);
			cancelled = false;

			List<Vec3> toolVerts = tool.VertPos;

			for (int tri = 0; tri < numTri; tri++)
			{
				if ((tri & CancelPollMask) == 0 && Cancel.IsCancelled(token))
				{
					cancelled = true;
					return false;
				}

				progress?.Advance(1);

				Vec3 normal = solid.FaceNormal[tri];
				double length = Length(normal);
				if (!(length > 0.0))
				{
					// A degenerate face carries a zero normal and bounds nothing. Dropping
					// it is safe on a convex solid, where every plane that matters is also
					// carried by a face with area.
					continue;
				}

				// Exact when the normal is already unit, which SetNormalsAndCoplanar makes
				// it: dividing by 1.0 does not move a bit, so a box's axis normals stay
				// exactly axis-aligned and its corners come out exactly on their planes.
				normal = normal / length;

				double planeOffset = Dot(normal, solid.VertPos[solid.Halfedge[tri * 3].StartVert]);

				double support = double.NegativeInfinity;
				for (int i = 0; i < toolVerts.Count; i++)
				{
					double reach = -Dot(normal, toolVerts[i]);
					if (reach > support)
					{
						support = reach;
					}
				}

				normals.Add(normal);
				offsets.Add(planeOffset - support);
			}

			// A bounded solid needs four planes. Fewer means the faces were degenerate
			// enough that this has stopped describing the input.
			return normals.Count >= 4;
		}

		/// <summary>
		/// Finds a point strictly inside the eroded body and each plane's slack there.
		/// </summary>
		/// <remarks>
		/// The candidate is the solid's vertex centroid, which is strictly inside the solid
		/// for any convex polytope and, for every shape that reaches this in practice — box,
		/// prism, cylinder, sphere, regular polyhedron — is the incentre or close enough to
		/// it. It is NOT guaranteed inside the eroded body: erode a skew tetrahedron to
		/// within a hair of nothing and what is left surrounds the incentre, not the
		/// centroid. That case declines, which costs a slow erosion of a solid that was
		/// about to vanish anyway. Solving the Chebyshev-centre LP would close it, and is
		/// deliberately not built until something needs it.
		/// </remarks>
		/// <param name="solid">The convex solid.</param>
		/// <param name="normals">The face normals.</param>
		/// <param name="offsets">The pushed-in plane offsets.</param>
		/// <param name="tolerance">The absolute slack the point must clear.</param>
		/// <param name="interior">The interior point.</param>
		/// <param name="slacks">Each plane's slack at that point.</param>
		/// <returns>True when the point is strictly inside every pushed-in plane.</returns>
		private static bool TryInteriorPoint(
			ManifoldImpl solid,
			List<Vec3> normals,
			List<double> offsets,
			double tolerance,
			out Vec3 interior,
			out List<double> slacks)
		{
			List<Vec3> verts = solid.VertPos;
			Vec3 sum = new Vec3(0.0, 0.0, 0.0);
			for (int i = 0; i < verts.Count; i++)
			{
				sum = sum + verts[i];
			}

			interior = sum / verts.Count;
			slacks = new List<double>(normals.Count);

			for (int i = 0; i < normals.Count; i++)
			{
				double slack = offsets[i] - Dot(normals[i], interior);
				if (!(slack > tolerance))
				{
					slacks.Clear();
					return false;
				}

				slacks.Add(slack);
			}

			return true;
		}

		/// <summary>
		/// The vertices of the halfspace intersection, via the polar dual.
		/// </summary>
		/// <remarks>
		/// The dual hull's job is only to say WHICH three planes meet at each vertex; the
		/// vertex itself is solved from the original normals and offsets, so the answer does
		/// not inherit the dual's <c>1/c</c> rounding. Vertices are looked up back to their
		/// planes by the exact bits of the dual point, which works because
		/// <see cref="QuickHullFunctions.ConvexHull"/> selects and compacts its input points
		/// rather than moving them, and because a dual point determines its halfspace
		/// uniquely (n = q/|q|, d = 1/|q|), so two planes sharing one are one constraint.
		/// The dictionary is probe-only, per CLAUDE.md's rule for every map in this port.
		/// </remarks>
		/// <param name="normals">The face normals.</param>
		/// <param name="offsets">The pushed-in plane offsets.</param>
		/// <param name="slacks">Each plane's slack at <paramref name="interior"/>.</param>
		/// <param name="interior">A point strictly inside the eroded body.</param>
		/// <param name="tolerance">The absolute slop a produced vertex is verified within.</param>
		/// <param name="vertices">The eroded body's vertices.</param>
		/// <returns>True when every vertex was solved and verified.</returns>
		private static bool TryVertices(
			List<Vec3> normals,
			List<double> offsets,
			List<double> slacks,
			Vec3 interior,
			double tolerance,
			out List<Vec3> vertices)
		{
			vertices = new List<Vec3>();

			List<Vec3> dualPoints = new List<Vec3>(normals.Count);
			Dictionary<Vec3, int> planeOfDualPoint = new Dictionary<Vec3, int>(normals.Count);
			for (int i = 0; i < normals.Count; i++)
			{
				Vec3 dual = normals[i] / slacks[i];
				dualPoints.Add(dual);
				planeOfDualPoint[dual] = i;
			}

			ManifoldImpl dualHull = QuickHullFunctions.ConvexHull(dualPoints);
			if (dualHull.IsEmpty())
			{
				return false;
			}

			int numTri = dualHull.NumTri();
			for (int tri = 0; tri < numTri; tri++)
			{
				if (!planeOfDualPoint.TryGetValue(dualHull.VertPos[dualHull.Halfedge[tri * 3].StartVert], out int a)
					|| !planeOfDualPoint.TryGetValue(dualHull.VertPos[dualHull.Halfedge[(tri * 3) + 1].StartVert], out int b)
					|| !planeOfDualPoint.TryGetValue(dualHull.VertPos[dualHull.Halfedge[(tri * 3) + 2].StartVert], out int c))
				{
					// A hull vertex that is not one of the points handed in. Nothing in
					// QuickHull is supposed to produce one, and if something ever does the
					// triple cannot be named, so the whole answer is handed back.
					return false;
				}

				if (!TrySolvePlaneTriple(normals[a], normals[b], normals[c], offsets[a], offsets[b], offsets[c], out Vec3 vertex))
				{
					return false;
				}

				vertices.Add(vertex);
			}

			// The backstop. The dual argument is only as good as its interior point and its
			// conditioning, and both are checked above by their own rules; this checks the
			// answer itself, which is the thing the caller is about to be handed.
			for (int v = 0; v < vertices.Count; v++)
			{
				for (int i = 0; i < normals.Count; i++)
				{
					if (Dot(normals[i], vertices[v]) > offsets[i] + tolerance)
					{
						return false;
					}
				}
			}

			return vertices.Count > 0;
		}

		/// <summary>
		/// The point where three planes meet, by Cramer's rule.
		/// </summary>
		/// <param name="na">The first plane's unit normal.</param>
		/// <param name="nb">The second plane's unit normal.</param>
		/// <param name="nc">The third plane's unit normal.</param>
		/// <param name="da">The first plane's offset.</param>
		/// <param name="db">The second plane's offset.</param>
		/// <param name="dc">The third plane's offset.</param>
		/// <param name="point">The meeting point.</param>
		/// <returns>
		/// False when the three normals are close enough to coplanar that the point is
		/// noise. The normals are unit, so the determinant is the triple product and its
		/// magnitude is a pure angle measure — no scale to normalize against, which is why
		/// the threshold can be a bare constant here and is relative everywhere else.
		/// </returns>
		private static bool TrySolvePlaneTriple(
			Vec3 na,
			Vec3 nb,
			Vec3 nc,
			double da,
			double db,
			double dc,
			out Vec3 point)
		{
			point = new Vec3(0.0, 0.0, 0.0);

			Vec3 bCrossC = Cross(nb, nc);
			double det = Dot(na, bCrossC);
			if (Math.Abs(det) < 1e-9)
			{
				return false;
			}

			Vec3 cCrossA = Cross(nc, na);
			Vec3 aCrossB = Cross(na, nb);
			point = ((bCrossC * da) + (cCrossA * db) + (aCrossB * dc)) / det;
			return true;
		}
	}
}
