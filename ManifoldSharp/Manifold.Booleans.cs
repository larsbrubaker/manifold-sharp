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

// Manifold.Booleans.cs — the boolean half of manifold.rs, split out for the
// 800-line file cap. Same class, same order as the Rust; see Manifold.cs for the
// file map and the value-semantics note.
//
// Every entry point here folds into `BooleanWithEngineRuleAndProgress`, which is
// one call to `Boolean3Functions.BooleanDispatchFull`. The overload ladder is
// transcribed rather than collapsed into optional parameters: the Rust names are
// the API other code (and agg-sharp's Phase 12 swap) is written against, and an
// optional-argument version would silently change which overload a caller binds
// to when a new parameter is inserted.
//
// `BatchBoolean` is a pairwise LEFT FOLD, not CsgTree.BatchBoolean's
// most-verts-first heap. That is what manifold.rs does, and the fold order
// reaches the output (boolean results depend on operand order), so it is the
// specification here even though the smarter batcher exists next door.
//
// ── C# operators ─────────────────────────────────────────────────────────────
// The Rust spells out twelve impls — Add/Sub/BitXor over owned and borrowed
// operands, plus the three *Assign forms. C# has one operator per symbol and
// derives `+=` from `+`, so three operators cover all twelve, and they are the
// same three semantics: `+` union, `-` difference, `^` intersection, matching
// C++ `operator+`, `operator-`, `operator^`.

using System.Runtime.InteropServices;

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <content>
	/// The boolean operations, composition, property assignment and the minimum-gap
	/// query.
	/// </content>
	public sealed partial class Manifold
	{
		/// <summary>Union — C++ <c>operator+</c>.</summary>
		/// <param name="a">The left operand.</param>
		/// <param name="b">The right operand.</param>
		/// <returns>The union.</returns>
		public static Manifold operator +(Manifold a, Manifold b)
		{
			ArgumentNullException.ThrowIfNull(a);
			return a.Union(b);
		}

		/// <summary>Difference — C++ <c>operator-</c>.</summary>
		/// <param name="a">The left operand.</param>
		/// <param name="b">The right operand.</param>
		/// <returns>The difference.</returns>
		public static Manifold operator -(Manifold a, Manifold b)
		{
			ArgumentNullException.ThrowIfNull(a);
			return a.Difference(b);
		}

		/// <summary>Intersection — C++ <c>operator^</c>.</summary>
		/// <param name="a">The left operand.</param>
		/// <param name="b">The right operand.</param>
		/// <returns>The intersection.</returns>
		public static Manifold operator ^(Manifold a, Manifold b)
		{
			ArgumentNullException.ThrowIfNull(a);
			return a.Intersection(b);
		}

		/// <summary>Apply batch boolean operations on a list of manifolds.</summary>
		/// <param name="manifolds">The operands, folded left to right.</param>
		/// <param name="op">The operation to apply.</param>
		/// <returns>The result, or the empty manifold for an empty list.</returns>
		public static Manifold BatchBoolean(IReadOnlyList<Manifold> manifolds, OpType op)
		{
			ArgumentNullException.ThrowIfNull(manifolds);

			if (manifolds.Count == 0)
			{
				return Empty();
			}

			Manifold result = manifolds[0].Clone();
			for (int i = 1; i < manifolds.Count; i++)
			{
				result = result.Boolean(manifolds[i], op);
			}

			return result;
		}

		/// <summary>
		/// <see cref="BatchBoolean"/> with an explicit engine choice (pairwise left fold,
		/// like <see cref="BatchBoolean"/>).
		/// </summary>
		/// <param name="manifolds">The operands, folded left to right.</param>
		/// <param name="op">The operation to apply.</param>
		/// <param name="engine">The engine to route to.</param>
		/// <returns>The result, or the empty manifold for an empty list.</returns>
		public static Manifold BatchBooleanWithEngine(
			IReadOnlyList<Manifold> manifolds,
			OpType op,
			BooleanEngine engine)
		{
			ArgumentNullException.ThrowIfNull(manifolds);

			if (manifolds.Count == 0)
			{
				return Empty();
			}

			Manifold result = manifolds[0].Clone();
			for (int i = 1; i < manifolds.Count; i++)
			{
				result = result.BooleanWithEngine(manifolds[i], op, engine);
			}

			return result;
		}

		/// <summary>Concatenate several manifolds into one without any boolean.</summary>
		/// <param name="parts">The meshes to concatenate.</param>
		/// <returns>The composed manifold.</returns>
		public static Manifold Compose(IReadOnlyList<Manifold> parts)
		{
			ArgumentNullException.ThrowIfNull(parts);

			List<ManifoldImpl> impls = new List<ManifoldImpl>(parts.Count);
			foreach (Manifold m in parts)
			{
				impls.Add(m.imp.Clone());
			}

			return FromImpl(Boolean3Functions.ComposeMeshes(impls));
		}

		/// <summary>The boolean operation, on the process-default engine.</summary>
		/// <param name="other">The right operand.</param>
		/// <param name="op">The operation to apply.</param>
		/// <returns>The result.</returns>
		public Manifold Boolean(Manifold other, OpType op)
		{
			return this.BooleanWithEngine(other, op, BooleanConfig.DefaultEngine());
		}

		/// <summary>
		/// <see cref="Boolean"/> with an explicit engine choice, overriding the
		/// process-global default set via <see cref="BooleanConfig.SetDefaultEngine"/>.
		/// </summary>
		/// <param name="other">The right operand.</param>
		/// <param name="op">The operation to apply.</param>
		/// <param name="engine">The engine to route to.</param>
		/// <returns>The result.</returns>
		public Manifold BooleanWithEngine(Manifold other, OpType op, BooleanEngine engine)
		{
			ArgumentNullException.ThrowIfNull(other);
			return FromImpl(
				Boolean3Functions.BooleanDispatch(this.imp, other.imp, op, engine, null));
		}

		/// <summary><see cref="BooleanWithEngine"/> with cooperative cancellation.</summary>
		/// <param name="other">The right operand.</param>
		/// <param name="op">The operation to apply.</param>
		/// <param name="engine">The engine to route to.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The result.</returns>
		public Manifold BooleanWithEngineAndToken(
			Manifold other,
			OpType op,
			BooleanEngine engine,
			CancelToken? token)
		{
			ArgumentNullException.ThrowIfNull(other);
			return FromImpl(
				Boolean3Functions.BooleanDispatch(this.imp, other.imp, op, engine, token));
		}

		/// <summary>
		/// <see cref="BooleanWithEngineAndToken"/> that also reports coarse pipeline
		/// progress.
		/// </summary>
		/// <remarks>
		/// Cancellation and progress travel together because callers that want one almost
		/// always want the other (a UI showing a progress bar next to a cancel button);
		/// pass null for either independently. A null reporter is byte-for-byte the
		/// un-instrumented path — see <see cref="ProgressReporter"/> for the phases
		/// reported and the throttling contract.
		/// </remarks>
		/// <param name="other">The right operand.</param>
		/// <param name="op">The operation to apply.</param>
		/// <param name="engine">The engine to route to.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>The result.</returns>
		public Manifold BooleanWithEngineAndProgress(
			Manifold other,
			OpType op,
			BooleanEngine engine,
			CancelToken? token,
			ProgressReporter? progress)
		{
			ArgumentNullException.ThrowIfNull(other);
			return FromImpl(Boolean3Functions.BooleanDispatchWithProgress(
				this.imp,
				other.imp,
				op,
				engine,
				token,
				progress));
		}

		/// <summary>
		/// <see cref="BooleanWithEngine"/> with an explicit winding rule.
		/// </summary>
		/// <remarks>
		/// <see cref="WindingRule.Nonzero"/> treats inside-out geometry as solid
		/// (<c>w != 0</c> rather than <c>w &gt;= 1</c>), which keeps the inverted regions
		/// of inconsistently wound scans instead of dropping them. The rule is a
		/// robust-engine semantic: the exact engine ignores it, and <c>Auto</c> routes to
		/// the robust engine whenever the rule is <c>Nonzero</c> (see
		/// <see cref="Boolean3Functions.BooleanDispatchFull"/>).
		/// </remarks>
		/// <param name="other">The right operand.</param>
		/// <param name="op">The operation to apply.</param>
		/// <param name="engine">The engine to route to.</param>
		/// <param name="rule">Which winding numbers count as solid material.</param>
		/// <returns>The result.</returns>
		public Manifold BooleanWithEngineAndRule(
			Manifold other,
			OpType op,
			BooleanEngine engine,
			WindingRule rule)
		{
			return this.BooleanWithEngineRuleAndProgress(other, op, engine, rule, null, null);
		}

		/// <summary>
		/// The full per-call boolean path: engine, winding rule, cancellation, and
		/// progress. Every other boolean entry point on <see cref="Manifold"/> is this one
		/// with defaults filled in.
		/// </summary>
		/// <param name="other">The right operand.</param>
		/// <param name="op">The operation to apply.</param>
		/// <param name="engine">The engine to route to.</param>
		/// <param name="rule">Which winding numbers count as solid material.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <param name="progress">The progress reporter, or null.</param>
		/// <returns>The result.</returns>
		public Manifold BooleanWithEngineRuleAndProgress(
			Manifold other,
			OpType op,
			BooleanEngine engine,
			WindingRule rule,
			CancelToken? token,
			ProgressReporter? progress)
		{
			ArgumentNullException.ThrowIfNull(other);
			return FromImpl(Boolean3Functions.BooleanDispatchFull(
				this.imp,
				other.imp,
				op,
				engine,
				rule,
				token,
				progress));
		}

		/// <summary>Union, on an explicit engine.</summary>
		/// <param name="other">The right operand.</param>
		/// <param name="engine">The engine to route to.</param>
		/// <returns>The union.</returns>
		public Manifold UnionWithEngine(Manifold other, BooleanEngine engine)
		{
			return this.BooleanWithEngine(other, OpType.Add, engine);
		}

		/// <summary>Difference, on an explicit engine.</summary>
		/// <param name="other">The right operand.</param>
		/// <param name="engine">The engine to route to.</param>
		/// <returns>The difference.</returns>
		public Manifold DifferenceWithEngine(Manifold other, BooleanEngine engine)
		{
			return this.BooleanWithEngine(other, OpType.Subtract, engine);
		}

		/// <summary>Intersection, on an explicit engine.</summary>
		/// <param name="other">The right operand.</param>
		/// <param name="engine">The engine to route to.</param>
		/// <returns>The intersection.</returns>
		public Manifold IntersectionWithEngine(Manifold other, BooleanEngine engine)
		{
			return this.BooleanWithEngine(other, OpType.Intersect, engine);
		}

		/// <summary>
		/// <see cref="Boolean"/> with cooperative cancellation.
		/// </summary>
		/// <remarks>
		/// Pass null for the uncancellable behaviour of <see cref="Boolean"/> — that path
		/// is unchanged and touches no atomics. With a token, a cancel requested from any
		/// thread (before or during the call) makes this return an empty manifold whose
		/// <see cref="Status"/> is <see cref="Error.Cancelled"/>, mirroring the C++
		/// <c>ExecutionContext</c> contract.
		/// </remarks>
		/// <param name="other">The right operand.</param>
		/// <param name="op">The operation to apply.</param>
		/// <param name="token">The cancellation token, or null.</param>
		/// <returns>The result.</returns>
		public Manifold BooleanWithToken(Manifold other, OpType op, CancelToken? token)
		{
			ArgumentNullException.ThrowIfNull(other);
			return FromImpl(
				Boolean3Functions.BooleanWithToken(this.imp, other.imp, op, token));
		}

		/// <summary>Union.</summary>
		/// <param name="other">The right operand.</param>
		/// <returns>The union.</returns>
		public Manifold Union(Manifold other)
		{
			return this.Boolean(other, OpType.Add);
		}

		/// <summary>Difference.</summary>
		/// <param name="other">The right operand.</param>
		/// <returns>The difference.</returns>
		public Manifold Difference(Manifold other)
		{
			return this.Boolean(other, OpType.Subtract);
		}

		/// <summary>Intersection.</summary>
		/// <param name="other">The right operand.</param>
		/// <returns>The intersection.</returns>
		public Manifold Intersection(Manifold other)
		{
			return this.Boolean(other, OpType.Intersect);
		}

		/// <summary>Compute per-vertex Gaussian and mean curvature into property slots.</summary>
		/// <param name="gaussianIdx">The property slot for Gaussian curvature, or -1.</param>
		/// <param name="meanIdx">The property slot for mean curvature, or -1.</param>
		/// <returns>The manifold with curvature properties.</returns>
		public Manifold CalculateCurvature(int gaussianIdx, int meanIdx)
		{
			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			if (this.IsEmpty())
			{
				return this.Clone();
			}

			ManifoldImpl outR = this.imp.Clone();
			outR.CalculateCurvature(gaussianIdx, meanIdx);
			return FromImpl(outR);
		}

		/// <summary>
		/// Set per-vertex properties via a callback function.
		/// </summary>
		/// <remarks>
		/// <paramref name="numProp"/> is the new number of properties per vertex (&gt;= 0).
		/// The callback receives (new_prop_slice, position, old_prop_slice) for each vertex
		/// and writes into the new slice, whose length is <paramref name="numProp"/>.
		/// </remarks>
		/// <param name="numProp">The new extra-property count per vertex.</param>
		/// <param name="propFunc">The per-vertex writer.</param>
		/// <returns>The manifold with the new properties.</returns>
		public Manifold SetProperties(int numProp, SetPropertiesFunc propFunc)
		{
			ArgumentNullException.ThrowIfNull(propFunc);

			Manifold? e = this.RequirePaired();
			if (e != null)
			{
				return e;
			}

			if (this.IsEmpty())
			{
				return this.Clone();
			}

			ManifoldImpl outR = this.imp.Clone();
			int oldNumProp = outR.NumProp;
			List<double> oldProperties = new List<double>(outR.Properties);

			if (numProp == 0)
			{
				outR.Properties.Clear();
			}
			else
			{
				int numPropVert = outR.NumPropVert();
				outR.Properties.Clear();
				outR.Properties.Capacity = numProp * numPropVert;
				for (int i = 0; i < numProp * numPropVert; i++)
				{
					outR.Properties.Add(0.0);
				}

				int numTri = outR.NumTri();
				for (int tri = 0; tri < numTri; tri++)
				{
					for (int i = 0; i < 3; i++)
					{
						Halfedge edge = outR.Halfedge[(3 * tri) + i];
						int vert = edge.StartVert;
						int propVert = edge.PropVert;
						Vec3 pos = outR.VertPos[vert];

						// The Rust's guard is `prop_vert * old_num_prop < old_properties.len()`
						// on usize operands, so a negative prop_vert would already have been
						// rejected as a huge unsigned value; the C# int comparison has to say so.
						ReadOnlySpan<double> oldSlice =
							oldNumProp > 0
								&& propVert >= 0
								&& propVert * oldNumProp < oldProperties.Count
								? CollectionsMarshal.AsSpan(oldProperties)
									.Slice(oldNumProp * propVert, oldNumProp)
								: ReadOnlySpan<double>.Empty;

						propFunc(
							CollectionsMarshal.AsSpan(outR.Properties).Slice(numProp * propVert, numProp),
							pos,
							oldSlice);
					}
				}
			}

			outR.NumProp = numProp;
			return FromImpl(outR);
		}

		/// <summary>
		/// Compute the minimum gap between two manifolds within
		/// <paramref name="searchLength"/>. Returns <paramref name="searchLength"/> if no
		/// closer points are found within that range.
		/// </summary>
		/// <param name="other">The other manifold.</param>
		/// <param name="searchLength">The largest gap worth reporting.</param>
		/// <returns>The gap.</returns>
		public double MinGap(Manifold other, double searchLength)
		{
			ArgumentNullException.ThrowIfNull(other);
			return this.imp.MinGap(other.imp, searchLength);
		}
	}
}
