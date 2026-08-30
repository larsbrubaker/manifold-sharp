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

// Quality.cs — the process-global settings of types.rs, split out of Types.cs
// for the 800-line cap: circle quantization (Quality), the boolean-engine
// selection enums (BooleanEngine, WindingRule) and the default-engine cell
// (BooleanConfig). The module header for the whole types.rs port lives in
// Types.cs.
//
// Both statics are process-global by design — they are the C++ `Quality` static
// class and `manifoldParams()`-style globals, not per-instance settings — so
// both are shared mutable state and both are guarded. Quality's three fields
// are read together by GetCircularSegments, so they sit behind one lock (the
// Rust's `Mutex<QualityState>`) rather than three independent atomics: a torn
// read of angle-from-one-setting and length-from-another would quantize a
// circle nobody asked for. BooleanConfig is a single cell and stays a single
// atomic, matching the Rust's `AtomicU8`.

namespace ManifoldSharp
{
	/// <summary>
	/// Which 3D boolean implementation to run.
	/// </summary>
	/// <remarks>
	/// Selection is input-based only: <see cref="Auto"/> never catches panics from the
	/// exact engine — a panic there is a bug to report, not a dispatch signal.
	/// </remarks>
	public enum BooleanEngine
	{
		/// <summary>
		/// The ported exact pipeline (default). Byte-identical results to the C++
		/// reference; requires strictly manifold operands.
		/// </summary>
		Exact = 0,

		/// <summary>
		/// The robust engine (<c>src/robust</c>, Barki et al. 2015): exact rational
		/// arithmetic, accepts closed orientable triangle soup (non-manifold,
		/// disconnected, voids). Slower; triangulation may differ from
		/// <see cref="Exact"/>.
		/// </summary>
		Robust = 1,

		/// <summary>
		/// <see cref="Exact"/> unless the input needs more, then <see cref="Robust"/>.
		/// </summary>
		/// <remarks>
		/// <para>
		/// The full rule, as one short-circuiting disjunction in
		/// <c>Boolean3Functions.BooleanDispatchFull</c> — <see cref="Robust"/> when
		/// <em>any</em> of these holds, <see cref="Exact"/> otherwise:
		/// </para>
		/// <list type="number">
		/// <item><description>the winding rule is <see cref="WindingRule.Nonzero"/>, which
		/// only the robust engine can honour;</description></item>
		/// <item><description>either operand carries non-manifold soup geometry (imported
		/// via <c>Manifold.FromMeshGLRobust</c>);</description></item>
		/// <item><description>either operand fails an exact self-intersection scan
		/// (<c>Robust.Soup.HasSelfIntersections</c>, cached per impl).</description></item>
		/// </list>
		/// <para>
		/// The order matters and is the Rust's: the two scans are the expensive terms and
		/// run only when the cheap ones have all failed, and operand B's scan only when
		/// operand A's came back false. Reordering them would change how often each operand
		/// pays for — and caches — its scan.
		/// </para>
		/// </remarks>
		Auto = 2,
	}

	/// <summary>
	/// Which winding numbers count as solid material.
	/// </summary>
	/// <remarks>
	/// The robust engine labels every cell of the arrangement with a winding number per
	/// operand; this rule turns that integer into "inside" or "outside". It is a
	/// <em>robust-engine</em> semantic only — the exact engine has no winding labels to
	/// reinterpret and ignores the rule entirely.
	/// </remarks>
	public enum WindingRule
	{
		/// <summary>
		/// <c>w &gt;= 1</c> (default). Inside-out geometry (negative winding) is not
		/// material, so an inverted region of a self-intersecting scan is discarded — the
		/// mathematically standard interpretation of orientation.
		/// </summary>
		Positive = 0,

		/// <summary>
		/// <c>w != 0</c>. Inside-out geometry is kept as solid, matching the intent of
		/// scans and CAD exports whose shells are wound inconsistently. Chosen per call
		/// for models where dropping the inverted chunk is not what the user wants.
		/// </summary>
		Nonzero = 1,
	}

	/// <summary>
	/// Process-global default engine, in the style of <see cref="Quality"/>: the plain
	/// boolean entry points (<c>Manifold.Boolean</c>, CSG-tree evaluation, Minkowski) read
	/// this; the <c>WithEngine</c> variants override it per call.
	/// </summary>
	public static class BooleanConfig
	{
		// The Rust holds this in an AtomicU8 read and written with Ordering::Relaxed: the
		// cell orders nothing else, and readers only use it to pick a code path. C# has no
		// relaxed atomic; `volatile` is the weakest ordering the CLR memory model exposes
		// and is *stronger* than Relaxed, which is sound in one direction only — meeting
		// more of the memory model than the Rust asked for cannot change which values are
		// observed. See the same note on CancelToken's flag.
		private static volatile int booleanEngineDefault;

		/// <summary>
		/// Sets the engine the plain boolean entry points use, for the whole process.
		/// </summary>
		/// <remarks>
		/// All three values are live: Phase 10 landed the robust engine, so
		/// <see cref="BooleanEngine.Robust"/> and <see cref="BooleanEngine.Auto"/> route
		/// through <c>Robust.RobustFunctions.BooleanWithRule</c> and failures are a status on
		/// the result, as CLAUDE.md requires. (This remark used to carry a
		/// PHASE 12 CARRY-FORWARD warning that the two non-Exact values threw; that is gone
		/// with the throws.)
		/// </remarks>
		/// <param name="engine">The engine to make default.</param>
		public static void SetDefaultEngine(BooleanEngine engine)
		{
			int v;
			switch (engine)
			{
				case BooleanEngine.Exact:
					v = 0;
					break;
				case BooleanEngine.Robust:
					v = 1;
					break;
				case BooleanEngine.Auto:
					v = 2;
					break;
				default:
					throw new ArgumentOutOfRangeException(nameof(engine), $"Unknown BooleanEngine value: {(int)engine}");
			}

			booleanEngineDefault = v;
		}

		/// <summary>
		/// The engine the plain boolean entry points currently use.
		/// </summary>
		/// <returns>The process-global default engine.</returns>
		public static BooleanEngine DefaultEngine()
		{
			switch (booleanEngineDefault)
			{
				case 1:
					return BooleanEngine.Robust;
				case 2:
					return BooleanEngine.Auto;
				default:
					return BooleanEngine.Exact;
			}
		}

		/// <summary>Restores the shipped default (<see cref="BooleanEngine.Exact"/>).</summary>
		public static void ResetToDefaults()
		{
			SetDefaultEngine(BooleanEngine.Exact);
		}
	}

	/// <summary>
	/// Process-global circle quantization: how many segments a circle of a given radius
	/// is built from. The C++ <c>Quality</c> static class.
	/// </summary>
	public static class Quality
	{
		// The Rust guards these three with one Mutex<QualityState> behind a OnceLock; the
		// C# equivalent is one lock object and three fields initialized inline, since a
		// static field initializer already runs exactly once and thread-safely.
		private static readonly object SyncRoot = new object();

		private static double minCircularAngle = Types.DefaultAngle;

		private static double minCircularEdgeLength = Types.DefaultLength;

		private static int circularSegments = Types.DefaultSegments;

		/// <summary>
		/// Sets the minimum angle, in degrees, between consecutive segments of a circle.
		/// </summary>
		/// <param name="angle">The minimum angle in degrees.</param>
		public static void SetMinCircularAngle(double angle)
		{
			lock (SyncRoot)
			{
				minCircularAngle = angle;
			}
		}

		/// <summary>Sets the minimum length of a circle segment.</summary>
		/// <param name="length">The minimum edge length.</param>
		public static void SetMinCircularEdgeLength(double length)
		{
			lock (SyncRoot)
			{
				minCircularEdgeLength = length;
			}
		}

		/// <summary>
		/// Overrides the segment count for every circle, ignoring angle and length. Zero
		/// (the default) turns the override off.
		/// </summary>
		/// <param name="n">The segment count, or 0 for automatic.</param>
		public static void SetCircularSegments(int n)
		{
			lock (SyncRoot)
			{
				circularSegments = n;
			}
		}

		/// <summary>
		/// The number of segments a circle of this radius is drawn with.
		/// </summary>
		/// <param name="radius">The circle radius.</param>
		/// <returns>The segment count, always a multiple of 4 and at least 4.</returns>
		public static int GetCircularSegments(double radius)
		{
			double angle;
			double edgeLength;
			int segments;
			lock (SyncRoot)
			{
				// All three under one lock: GetCircularSegments is a single logical read of
				// the settings, and mixing an old angle with a new length would quantize a
				// circle that was never configured.
				angle = minCircularAngle;
				edgeLength = minCircularEdgeLength;
				segments = circularSegments;
			}

			if (segments > 0)
			{
				return segments;
			}

			// Match C++ exactly: int truncation (not ceil), fmin (not fmax), round down to multiple of 4
			//
			// The two casts truncate toward zero in both languages and both saturate on
			// overflow — Rust's `as i32` by definition, C#'s because .NET 9 made every
			// floating-point to integer conversion saturating and platform-independent
			// (dotnet/core/compatibility/jit/9.0/fp-to-integer). On this net10.0 target a
			// bare cast *is* Rust `as`, so no saturating helper is needed here: a zero
			// angle or edge length — either makes the quotient infinite, and both are
			// reachable through the setters above — degrades identically in both ports.
			int nSegA = (int)(360.0 / angle);
			int nSegL = (int)(2.0 * Math.Abs(radius) * Types.KPi / edgeLength);
			int nSeg = Math.Min(nSegA, nSegL) + 3;
			nSeg -= nSeg % 4;
			return Math.Max(nSeg, 4);
		}

		/// <summary>Restores the shipped defaults for all three settings.</summary>
		public static void ResetToDefaults()
		{
			lock (SyncRoot)
			{
				minCircularAngle = Types.DefaultAngle;
				minCircularEdgeLength = Types.DefaultLength;
				circularSegments = Types.DefaultSegments;
			}
		}
	}
}
