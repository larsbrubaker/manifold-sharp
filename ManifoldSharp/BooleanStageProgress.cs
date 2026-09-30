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

// BooleanStageProgress.cs — NOT A PORT. The exact boolean's optional stage sink: an
// `Action<double>?` that Boolean3Functions.BooleanWithToken, Boolean3.NewWithToken and
// BooleanResultAssemble.BooleanResultWithToken take in a C#-only overload, and invoke
// with the cumulative marks below at the cancel gates that close each heavy stage
// (CancelToken.cs lists those gates). The Rust has no such sink; it is part of
// divergence ledger entry 6, whose union tree is its one caller.
//
// A side channel only: the sink is invoked between stages, reads nothing and is handed
// nothing but a constant, so every computed value is the one the overloads without it
// produce. A null sink is the pre-existing code. A throwing sink unwinds out of the
// boolean, as a throwing ProgressReporter callback does.
//
// The marks are one boolean's own completed fraction, from stage times measured in
// Debug on two warm unions: a dilated drilled part with a rotated copy of itself (1200
// triangles each) and two overlapping 160-segment spheres (12800 each). The two
// edge-face intersection passes took 32-54% of the time, SimplifyTopology 22-33%,
// triangulation 7-16%, the closing sort 2-21%, winding and edge assembly the rest.
// They are estimates of where the time goes, not a contract; a mark only has to be
// monotone and at most 1. The last one is reported after SimplifyTopology, leaving the
// closing sort unreported, so a caller never hears 1.0 before the boolean returns.

namespace ManifoldSharp
{
	/// <summary>
	/// The cumulative fractions the exact boolean's stage sink hears, in order.
	/// </summary>
	public static class BooleanStageProgress
	{
		/// <summary>After <c>Intersect12</c> P→Q.</summary>
		public const double AfterIntersectPQ = 0.22;

		/// <summary>After <c>Intersect12</c> Q→P.</summary>
		public const double AfterIntersectQP = 0.44;

		/// <summary>After <c>Winding03</c> on P.</summary>
		public const double AfterWindingP = 0.46;

		/// <summary>After <c>Winding03</c> on Q.</summary>
		public const double AfterWindingQ = 0.48;

		/// <summary>After the edge assembly (<c>AppendWholeEdges</c>).</summary>
		public const double AfterAssembly = 0.53;

		/// <summary>After <c>Face2Tri</c> and <c>ReorderHalfedges</c>.</summary>
		public const double AfterTriangulation = 0.63;

		/// <summary>After <c>CreateProperties</c> and <c>UpdateReference</c>.</summary>
		public const double AfterReference = 0.64;

		/// <summary>After <c>SimplifyTopology</c>; only the closing sort remains.</summary>
		public const double AfterSimplify = 0.90;
	}
}
