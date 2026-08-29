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

// The polygon type aliases, again, for the test assembly.
//
// `global using` aliases are per-*compilation*, not per-assembly-reference: the
// four declared in ManifoldSharp/Types.cs make `PolygonsIdx` a name inside
// ManifoldSharp only, and referencing that assembly does not import them. The
// Rust tests read `PolygonsIdx` / `SimplePolygonIdx` because in Rust those are
// crate-level `type` items visible to the in-crate test modules, so the aliases
// are repeated here to keep the ported tests reading like their source. They
// resolve to the same List types, so an alias and its expansion are the same
// type at the boundary.
//
// This file is the whole test assembly's declaration — a test file that uses
// `PolygonsIdx` without an obvious `using` is finding it here.

global using Polygons = System.Collections.Generic.List<System.Collections.Generic.List<ManifoldSharp.Linalg.Vec2>>;
global using PolygonsIdx = System.Collections.Generic.List<System.Collections.Generic.List<ManifoldSharp.PolyVert>>;
global using SimplePolygon = System.Collections.Generic.List<ManifoldSharp.Linalg.Vec2>;
global using SimplePolygonIdx = System.Collections.Generic.List<ManifoldSharp.PolyVert>;
