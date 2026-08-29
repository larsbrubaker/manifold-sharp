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

// Phase 2: Core Types — ported from include/manifold/common.h,
// include/manifold/polygon.h, include/manifold/manifold.h (MeshGLP/Error),
// src/shared.h (Halfedge, TriRef, Barycentric, TmpEdge)
//
// The bounding-volume types (Box, Rect) live in Bounds.cs and the MeshGLP
// interchange mesh in the Phase 3 mesh-core files; the Rust re-exports both
// through types.rs so that module is the single public home for all core
// types. C# needs no re-export: everything here shares the flat `ManifoldSharp`
// namespace with them (only Linalg keeps a sub-namespace), so `crate::types::X`
// is simply `X`.
//
// ── File split ───────────────────────────────────────────────────────────────
// types.rs is one 607-line file whose C# expansion does not fit the 800-line
// cap, so it lands as four files, all in this namespace:
//   Types.cs             constants, scalar utilities, the polygon types, OpType,
//                        Error, ExecutionParams, Smoothness, RayHit
//   Types.TrigDegrees.cs Sind/Cosd — the exact-at-multiples-of-90 trig
//   Types.Shared.cs      the src/shared.h structs (Halfedge, Barycentric,
//                        TriRef, TmpEdge), the src/impl.h relation tables
//                        (Relation, MeshRelationD) and the halfedge index
//                        helpers
//   Quality.cs           the process-global Quality and BooleanConfig state,
//                        plus BooleanEngine and WindingRule
// The three files that carry free functions continue the same
// `public static partial class Types`, which is where every free item of the
// Rust module lands (Rust has module-level functions; C# does not).
//
// ── Polygon aliases ──────────────────────────────────────────────────────────
// The Rust type aliases are transcribed as C# global using aliases below, so
// downstream ports keep reading like the Rust (`Polygons polys`). They are
// aliases, not types: outside this assembly `Polygons` is exactly
// `List<List<Vec2>>`, which is also what the public surface exposes.
//
// `List<T>` and not `T[]`: the arena rule (arrays + int indices) applies where
// the Rust Vec is fixed after build, and these are not — polygon.rs,
// constructors.rs and cross_section.rs all start from `Vec::new()` and push
// contours and vertices as they go, so a fixed-size array would force every one
// of those loops to be restructured. The Vec here is a builder, so it stays a
// List.

global using Polygons = System.Collections.Generic.List<System.Collections.Generic.List<ManifoldSharp.Linalg.Vec2>>;
global using PolygonsIdx = System.Collections.Generic.List<System.Collections.Generic.List<ManifoldSharp.PolyVert>>;
global using SimplePolygon = System.Collections.Generic.List<ManifoldSharp.Linalg.Vec2>;
global using SimplePolygonIdx = System.Collections.Generic.List<ManifoldSharp.PolyVert>;

using System.Runtime.CompilerServices;

using ManifoldSharp.Linalg;

namespace ManifoldSharp
{
	/// <summary>
	/// The constants and free functions of <c>types.rs</c> — Rust module-level items
	/// with no C# equivalent, gathered onto one static class named for the module.
	/// </summary>
	public static partial class Types
	{
		// ─── Constants ───────────────────────────────────────────────────────────

		/// <summary>Rust <c>K_PI</c> — pi. Bit-identical to <c>std::f64::consts::PI</c>.</summary>
		public const double KPi = Math.PI;

		/// <summary>Rust <c>K_TWO_PI</c> — tau (2*pi). Bit-identical to <c>std::f64::consts::TAU</c>.</summary>
		public const double KTwoPi = Math.Tau;

		/// <summary>
		/// Rust <c>K_HALF_PI</c> — pi/2. Halving is exact, so this is bit-identical to
		/// <c>std::f64::consts::FRAC_PI_2</c>.
		/// </summary>
		public const double KHalfPi = Math.PI / 2.0;

		/// <summary>Precision used for epsilon calculations relative to bounding-box scale.</summary>
		public const double KPrecision = 1e-12;

		/// <summary>Rust <c>DEFAULT_SEGMENTS</c> — the circular-segment override, off by default.</summary>
		public const int DefaultSegments = 0;

		/// <summary>Rust <c>DEFAULT_ANGLE</c> — the default minimum circular angle, in degrees.</summary>
		public const double DefaultAngle = 10.0;

		/// <summary>Rust <c>DEFAULT_LENGTH</c> — the default minimum circular edge length.</summary>
		public const double DefaultLength = 1.0;

		// ─── Scalar utilities ────────────────────────────────────────────────────

		/// <summary>Degrees to radians.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Radians(double a)
		{
			return a * KPi / 180.0;
		}

		/// <summary>Radians to degrees.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Degrees(double a)
		{
			return a * 180.0 / KPi;
		}

		/// <summary>
		/// Smooth Hermite interpolation between 0 and 1 when edge0 &lt; x &lt; edge1.
		/// </summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public static double Smoothstep(double edge0, double edge1, double a)
		{
			double x = (a - edge0) / (edge1 - edge0);

			// Rust's f64::clamp, transcribed: a pair of comparisons that leaves NaN as
			// NaN. LinalgFunctions.Clamp is max-then-min (the linalg clamp_s), which
			// turns a NaN into lo, so it is *not* a substitute here.
			x = x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
			return x * x * (3.0 - 2.0 * x);
		}
	}

	/// <summary>
	/// Polygon vertex with index — the element type of <c>SimplePolygonIdx</c>.
	/// </summary>
	/// <remarks>
	/// <c>operator==</c> is the Rust's derived <c>PartialEq</c> (IEEE on the position);
	/// <see cref="Equals(PolyVert)"/> and <see cref="GetHashCode"/> are bit-based, per
	/// the equality rule the Linalg folder header states.
	/// </remarks>
	public struct PolyVert : IEquatable<PolyVert>
	{
		/// <summary>The vertex position.</summary>
		public Vec2 Pos;

		/// <summary>The index this vertex carries through triangulation.</summary>
		public int Idx;

		/// <summary>Creates an indexed polygon vertex.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public PolyVert(Vec2 pos, int idx)
		{
			this.Pos = pos;
			this.Idx = idx;
		}

		/// <summary>IEEE equality, the Rust's derived <c>PartialEq</c>.</summary>
		public static bool operator ==(PolyVert a, PolyVert b)
		{
			return a.Pos == b.Pos && a.Idx == b.Idx;
		}

		/// <summary>IEEE inequality, the Rust's derived <c>PartialEq</c>.</summary>
		public static bool operator !=(PolyVert a, PolyVert b)
		{
			return !(a == b);
		}

		/// <summary>Bit-exact equality; the partner of <see cref="GetHashCode"/>.</summary>
		public bool Equals(PolyVert other)
		{
			return this.Pos.Equals(other.Pos) && this.Idx == other.Idx;
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is PolyVert other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.Pos, this.Idx);
		}
	}

	/// <summary>
	/// Which boolean operation a CSG node performs.
	/// </summary>
	public enum OpType
	{
		/// <summary>Union.</summary>
		Add = 0,

		/// <summary>Difference.</summary>
		Subtract = 1,

		/// <summary>Intersection.</summary>
		Intersect = 2,
	}

	/// <summary>
	/// The status a Manifold carries instead of throwing — see PORTING_PLAN.md: errors
	/// are a status enum on the result, never an exception.
	/// </summary>
	/// <remarks>
	/// <b>The declaration order is the FFI surface.</b> The values 0-15 are spelled out
	/// because the FFI maps these variants to status codes positionally, exactly as C++
	/// <c>Manifold::Error</c> does; new variants only ever go on the end.
	/// </remarks>
	public enum Error
	{
		/// <summary>The operation succeeded.</summary>
		NoError = 0,

		/// <summary>A vertex position was infinite or NaN.</summary>
		NonFiniteVertex = 1,

		/// <summary>The input is not a manifold (2-manifold, closed, orientable) mesh.</summary>
		NotManifold = 2,

		/// <summary>A triangle referenced a vertex index outside the vertex list.</summary>
		VertexOutOfBounds = 3,

		/// <summary>The property buffer length is not a multiple of the property count.</summary>
		PropertiesWrongLength = 4,

		/// <summary>Fewer than three properties, so the first three cannot be positions.</summary>
		MissingPositionProperties = 5,

		/// <summary>mergeFromVert and mergeToVert have different lengths.</summary>
		MergeVectorsDifferentLengths = 6,

		/// <summary>A merge index referenced a vertex outside the vertex list.</summary>
		MergeIndexOutOfBounds = 7,

		/// <summary>runTransform is not 12 doubles per run.</summary>
		TransformWrongLength = 8,

		/// <summary>runIndex does not have one more entry than the number of runs.</summary>
		RunIndexWrongLength = 9,

		/// <summary>faceID does not have one entry per triangle.</summary>
		FaceIdWrongLength = 10,

		/// <summary>The requested construction is not valid (bad parameters).</summary>
		InvalidConstruction = 11,

		/// <summary>The result would exceed the index range the kernel can address.</summary>
		ResultTooLarge = 12,

		/// <summary>The supplied halfedge tangents are not usable.</summary>
		InvalidTangents = 13,

		/// <summary>
		/// The operation was interrupted through a <see cref="CancelToken"/> and returned
		/// an empty result.
		/// </summary>
		/// <remarks>
		/// Appended last, matching C++ <c>Manifold::Error::Cancelled</c>
		/// (cpp-reference/manifold/include/manifold/manifold.h:139). The order of every
		/// preceding variant is load-bearing: the FFI maps them to status codes 0-13
		/// positionally, so new variants only ever go on the end.
		/// </remarks>
		Cancelled = 14,

		/// <summary>
		/// The mesh is not geometrically closed and orientable, so even the robust
		/// (non-manifold) boolean engine cannot interpret it as a solid. Produced only by
		/// the <c>FromMeshGLRobust</c> import path; the strict import keeps reporting
		/// <see cref="NotManifold"/>. FFI status code 15.
		/// </summary>
		NotClosed = 15,
	}

	/// <summary>
	/// The Rust <c>Error::to_str</c> / <c>Display</c>, which C# cannot hang off the enum
	/// itself.
	/// </summary>
	/// <remarks>
	/// <see cref="object.ToString"/> on the enum yields the C# member name
	/// (<c>"NoError"</c>); <see cref="ToStr"/> yields the human string the Rust prints
	/// (<c>"No Error"</c>), and is the one that must stay stable.
	/// </remarks>
	public static class ErrorExtensions
	{
		/// <summary>The human-readable text for a status, matching Rust <c>to_str</c>.</summary>
		/// <param name="error">The status to describe.</param>
		/// <returns>The message text.</returns>
		public static string ToStr(this Error error)
		{
			switch (error)
			{
				case Error.NoError:
					return "No Error";
				case Error.NonFiniteVertex:
					return "Non-Finite Vertex";
				case Error.NotManifold:
					return "Not Manifold";
				case Error.VertexOutOfBounds:
					return "Vertex Out of Bounds";
				case Error.PropertiesWrongLength:
					return "Properties Wrong Length";
				case Error.MissingPositionProperties:
					return "Missing Position Properties";
				case Error.MergeVectorsDifferentLengths:
					return "Merge Vectors Different Lengths";
				case Error.MergeIndexOutOfBounds:
					return "Merge Index Out of Bounds";
				case Error.TransformWrongLength:
					return "Transform Wrong Length";
				case Error.RunIndexWrongLength:
					return "Run Index Wrong Length";
				case Error.FaceIdWrongLength:
					return "Face ID Wrong Length";
				case Error.InvalidConstruction:
					return "Invalid Construction";
				case Error.ResultTooLarge:
					return "Result Too Large";
				case Error.InvalidTangents:
					return "Invalid Tangents";
				case Error.Cancelled:
					return "Cancelled";
				case Error.NotClosed:
					return "Not Closed";
				default:
					// The Rust match is exhaustive over the enum and has no fallback; C#
					// enums are open (any int can be cast in), so the unreachable arm
					// names the offending value rather than returning a plausible string.
					throw new ArgumentOutOfRangeException(nameof(error), $"Unknown Error value: {(int)error}");
			}
		}
	}

	/// <summary>
	/// The C++ <c>manifoldParams()</c> debug switches, ported for parity.
	/// </summary>
	/// <remarks>
	/// Nothing in the Rust port reads these yet — the checks they gate are C++ debug
	/// instrumentation — but the type is part of types.rs and carries over so that the
	/// port's public surface matches. <c>new ExecutionParams()</c> reproduces the Rust
	/// <c>Default</c>; <c>default(ExecutionParams)</c> does not, since two of the
	/// defaults are true.
	/// </remarks>
	public struct ExecutionParams
	{
		/// <summary>Run the (expensive) intermediate mesh checks between stages.</summary>
		public bool IntermediateChecks;

		/// <summary>Run the self-intersection checks.</summary>
		public bool SelfIntersectionChecks;

		/// <summary>Process overlapping (coplanar-degenerate) triangles rather than dropping them.</summary>
		public bool ProcessOverlaps;

		/// <summary>Suppress error printing.</summary>
		public bool SuppressErrors;

		/// <summary>Remove degenerate triangles from the result.</summary>
		public bool CleanupTriangles;

		/// <summary>Verbosity level; 0 is silent.</summary>
		public int Verbose;

		/// <summary>Creates the parameter set with the Rust <c>Default</c> values.</summary>
		public ExecutionParams()
		{
			this.IntermediateChecks = false;
			this.SelfIntersectionChecks = false;
			this.ProcessOverlaps = true;
			this.SuppressErrors = false;
			this.CleanupTriangles = true;
			this.Verbose = 0;
		}
	}

	/// <summary>
	/// A per-halfedge smoothness weight, the input to edge sharpening.
	/// </summary>
	public struct Smoothness : IEquatable<Smoothness>
	{
		/// <summary>
		/// The halfedge index = 3 * tri + i.
		/// </summary>
		/// <remarks>
		/// Rust and C++ type this <c>usize</c>/<c>size_t</c>; here it is <c>int</c>,
		/// because every halfedge index in the port is an <c>int</c> (see
		/// <see cref="Halfedge.PairedHalfedge"/>) and the arena rule in PORTING_PLAN.md
		/// keeps indices <c>int</c>. The values stored are always <c>3 * tri + i</c> for a
		/// triangle index that already had to fit an <c>int</c>.
		/// </remarks>
		public int Halfedge;

		/// <summary>
		/// 0 = sharp, 1 = smooth. The Rust field is <c>smoothness</c>; C# forbids a member
		/// named for its enclosing type, so this one carries the <c>Value</c> suffix.
		/// </summary>
		public double SmoothnessValue;

		/// <summary>Creates a smoothness weight for a halfedge.</summary>
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Smoothness(int halfedge, double smoothness)
		{
			this.Halfedge = halfedge;
			this.SmoothnessValue = smoothness;
		}

		/// <summary>IEEE equality, the Rust's derived <c>PartialEq</c>.</summary>
		public static bool operator ==(Smoothness a, Smoothness b)
		{
			return a.Halfedge == b.Halfedge && a.SmoothnessValue == b.SmoothnessValue;
		}

		/// <summary>IEEE inequality, the Rust's derived <c>PartialEq</c>.</summary>
		public static bool operator !=(Smoothness a, Smoothness b)
		{
			return !(a == b);
		}

		/// <summary>Bit-exact equality; the partner of <see cref="GetHashCode"/>.</summary>
		public bool Equals(Smoothness other)
		{
			return this.Halfedge == other.Halfedge
				&& BitConverter.DoubleToUInt64Bits(this.SmoothnessValue)
					== BitConverter.DoubleToUInt64Bits(other.SmoothnessValue);
		}

		/// <inheritdoc/>
		public override bool Equals(object? obj)
		{
			return obj is Smoothness other && this.Equals(other);
		}

		/// <inheritdoc/>
		public override int GetHashCode()
		{
			return HashCode.Combine(this.Halfedge, BitConverter.DoubleToUInt64Bits(this.SmoothnessValue));
		}
	}

	/// <summary>
	/// Result of a RayCast query: a single triangle-ray intersection.
	/// </summary>
	/// <remarks>
	/// The Rust derives <c>Default</c>, which zeroes every field; so does
	/// <c>default(RayHit)</c> here, so the two agree.
	/// </remarks>
	public struct RayHit
	{
		/// <summary>Triangle index that was hit.</summary>
		public ulong FaceId;

		/// <summary>
		/// Parametric distance along the ray segment in [0, 1]. 0 = origin, 1 = endpoint.
		/// </summary>
		public double Distance;

		/// <summary>3D position of the hit point.</summary>
		public Vec3 Position;

		/// <summary>Geometric face normal at the hit.</summary>
		public Vec3 Normal;
	}
}
