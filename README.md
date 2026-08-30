# manifold-sharp

**3D mesh booleans in pure C# — exact on clean geometry, robust on real-world geometry.**

Pure managed port of the [Manifold](https://github.com/elalish/manifold) 3D geometry
kernel, ported from [manifold-rust](https://github.com/larsbrubaker/manifold-rust)
(itself an exact-match Rust port of Manifold C++ v3.5.x). No native library to build,
load, or link — it runs anywhere .NET runs.

## What this is

Two things in one assembly, both inherited from the Rust port:

1. **The ported Manifold kernel** — union / intersection / difference on triangle
   meshes, plus constructors, cross-sections, convex hull, Minkowski, SDF meshing and
   smooth subdivision. The port targets *exact numerical match* with manifold-rust, and
   therefore with the C++ reference: same algorithms, same floating-point values, same
   triangle topology, same counts.

2. **The "robust" boolean engine** the C++ library does not have. It accepts the meshes
   real pipelines actually contain — triangle soup, scans, downloaded models: non-manifold
   connectivity, self-intersections, doubled sheets, disconnected shells, internal voids,
   inside-out bodies. It computes on exact rational arithmetic with a mesh-arrangement
   formulation, so its answers are decided by exact predicates rather than by tolerances.

**The porting source is the Rust, not the C++.** The Rust tree already carries the
documented divergences, the determinism fixes, and file headers written to be ported
from; this port follows it line for line and keeps those headers.

**Relationship to `ManifoldRust`.** The `ManifoldRust` NuGet package is a P/Invoke
binding over the manifold-rust cdylib. It is not this library's ancestor — it is its
**oracle**. The same booleans run through both and are compared bit-for-bit in CI, which
is the reason the binding is a dependency of this repository at all. Nothing here calls
into it at runtime.

## Why pure managed

- **No native artifact anywhere.** No P/Invoke, no `NativeFileReference`, no `.a` on
  mono-wasm's emcc link line, no per-RID native payload to ship or to forget to ship.
  `browser-wasm` is just another target: a project targeting `net10.0-browser` /
  `browser-wasm` that references this library compiles and links with no extra step.
- **One dependency**, [Clipper2](https://www.nuget.org/packages/Clipper2) (pure managed
  too), confined to `CrossSection.Clipper.cs` exactly as the Rust confines
  `clipper2-rust` to `cross_section.rs`. Everything else is BCL-only, and the assembly is
  built `IsAotCompatible`.
- **Debugger-visible internals.** A geometry bug in a consuming application is a
  breakpoint away, not an FFI boundary away. That is not a small thing when the failure
  mode is one wrong float twelve stages into a boolean.

## Features

- **Booleans**: union, intersection, difference, with `+` `-` `^` operators; n-ary
  `BatchBoolean`; full CSG-tree evaluation (`CsgNode` / `CsgOp` / `CsgLeaf`) with
  bounding-box pruning and batching
- **Two engines**, selected per call (`UnionWithEngine`, `DifferenceWithEngine`,
  `IntersectionWithEngine`, `BooleanWithEngine`…) or process-wide
  (`BooleanConfig.SetDefaultEngine`) — see [The two engines](#the-two-engines)
- **Constructors**: `Tetrahedron`, `Cube`, `Cylinder` / `CylinderCentered`, `Sphere`
- **Modeling**: `Extrude` (with twist and top scaling), `Revolve`, `Hull` /
  `ConvexHull` / `HullManifolds`, `MinkowskiSum` / `MinkowskiDifference`, `LevelSet`
  (SDF meshing, with optional iso-level and crossing-refinement tolerance)
- **Smoothing and subdivision**: `Smooth` (from sharpened edges), `SmoothOut`,
  `SmoothByNormals`, `CalculateNormals`, `Refine`, `RefineToLength`, `RefineToTolerance`
- **2D cross-sections**: a full `CrossSection` type over Clipper2 — `Square`, `Circle`,
  booleans and `BatchBoolean`, `Offset` / `OffsetWithParams`, `Simplify`, `MinkowskiSum`,
  hull, `Decompose`, `Compose`, `Warp`, transforms — plus `Slice` and `Project` from a
  `Manifold`
- **Mesh repair**: `RepairOrientation()` for inside-out bodies, `RebuildSolid(rule)` to
  re-derive a solid from winding numbers when rewinding is not enough,
  `HasSelfIntersections()` as an exact self-scan
- **Partitioning and measurement**: `Split`, `SplitByPlane`, `TrimByPlane`, `Decompose`,
  `Compose`, `MinGap`, `Volume`, `SurfaceArea`, `Genus`, `BoundingBox`,
  `NumDegenerateTris`, `MatchesTriNormals`
- **Ray casting**: `RayCast(origin, endpoint)` returns every triangle the segment crosses,
  nearest first, each with its parametric position along the segment, the hit point, the
  geometric face normal and the triangle index
- **Per-vertex properties**: arbitrary property slots carried through booleans, written by
  `SetProperties`, plus `CalculateCurvature` (Gaussian and mean into property slots)
- **Transforms**: `Translate`, `Rotate`, `Scale`, `Mirror`, `Transform`, `Warp` and
  `WarpBatch`
- **I/O**: `MeshGL` (f32) and `MeshGL64` (f64) in and out, a soup-tolerant import
  (`FromMeshGLRobust` / `FromMeshGL64Robust`), and an OBJ text round-trip
  (`WriteObj` / `ReadObj`)
- **Cooperative cancellation** (`CancelToken`, the `…WithToken` overloads) and coarse
  **progress reporting** (`ProgressReporter`) through the boolean pipeline
- **Optional parallelism** — a runtime switch, not a build flag; results stay
  **bit-identical** to the sequential path

## Getting started

There is no NuGet package yet — the library is consumed as a submodule with a project
reference, which is how agg-sharp's `PolygonMesh` consumes it:

```xml
<ProjectReference Include="path/to/manifold-sharp/ManifoldSharp/ManifoldSharp.csproj" />
```

```csharp
using ManifoldSharp;
using ManifoldSharp.Linalg;
```

`ManifoldSharp` holds the geometry types; `ManifoldSharp.Linalg` holds `Vec2`, `Vec3`,
`Mat3x4` and friends. Types the Rust spells as aliases (`Polygons`, `SimplePolygon`) are
plain BCL generics on the public surface: `List<List<Vec2>>` and `List<Vec2>`.

## Usage

### A boolean

```csharp
Manifold cube = Manifold.Cube(new Vec3(1.0, 1.0, 1.0), center: true);
Manifold sphere = Manifold.Sphere(0.6, 32);

Manifold result = cube - sphere;           // or: cube.Difference(sphere)

// Errors are a status on the result, never an exception: a failed operation
// comes back as empty geometry carrying the reason.
if (result.Status() != Error.NoError)
{
    throw new InvalidOperationException(result.Status().ToStr());
}

Console.WriteLine($"volume = {result.Volume()}");
Console.WriteLine($"area   = {result.SurfaceArea()}");
Console.WriteLine($"tris   = {result.NumTri()}");

MeshGL mesh = result.GetMeshGL(-1);        // VertProperties / TriVerts, GPU-ready
```

### A CSG tree

```csharp
Manifold box = Manifold.Cube(new Vec3(2.0, 2.0, 2.0), center: true);
Manifold ball = Manifold.Sphere(1.35, 64);
Manifold barZ = Manifold.Cylinder(4.0, 0.4, 0.4, 64).Translate(new Vec3(0.0, 0.0, -2.0));
Manifold barX = barZ.Rotate(0.0, 90.0, 0.0);

// One evaluation for the whole tree: intermediate results are never handed
// back as Manifolds, n-ary children are batched, and bounding boxes prune.
CsgNode tree = new CsgOp(
    OpType.Subtract,
    new CsgNode[]
    {
        new CsgOp(
            OpType.Intersect,
            new CsgNode[] { new CsgLeaf(box.AsImpl()), new CsgLeaf(ball.AsImpl()) }),
        new CsgOp(
            OpType.Add,
            new CsgNode[] { new CsgLeaf(barZ.AsImpl()), new CsgLeaf(barX.AsImpl()) }),
    });

Manifold drilled = Manifold.FromImpl(tree.Evaluate());

// A flat n-ary operation has a shorthand that builds the same tree for you:
Manifold bars = Manifold.BatchBoolean(new[] { barZ, barX }, OpType.Add);
```

### An SDF level set

```csharp
// DeterministicMath, not System.Math: the port's own transcendentals are
// transcribed from musl and give the same bits on every platform. Your own
// callback is your own determinism budget — System.Math.Cos is platform libm.
static double Gyroid(Vec3 p)
{
    return (DeterministicMath.Cos(p.X) * DeterministicMath.Sin(p.Y))
        + (DeterministicMath.Cos(p.Y) * DeterministicMath.Sin(p.Z))
        + (DeterministicMath.Cos(p.Z) * DeterministicMath.Sin(p.X));
}

Box bounds = new Box(Vec3.Splat(-Math.PI), Vec3.Splat(Math.PI));
Manifold gyroid = Manifold.LevelSet(Gyroid, bounds, 0.3);
```

The SDF callback must be **pure and thread-safe** — with parallelism on it is called from
several threads at once. See the caveat under [Parallelism](#parallelism) for what happens
if it throws.

### A `MeshGL` round trip

```csharp
// Import: positions interleaved in VertProperties, NumProp floats per vertex
// (3 = position only), triangles as flat vertex indices.
MeshGL raw = new MeshGL { NumProp = 3 };
raw.VertProperties.AddRange(new float[]
{
    0f, 0f, 0f,   1f, 0f, 0f,   1f, 1f, 0f,   0f, 1f, 0f,
    0f, 0f, 1f,   1f, 0f, 1f,   1f, 1f, 1f,   0f, 1f, 1f,
});
raw.TriVerts.AddRange(new uint[]
{
    0, 3, 2,  0, 2, 1,  4, 5, 6,  4, 6, 7,
    0, 1, 5,  0, 5, 4,  1, 2, 6,  1, 6, 5,
    2, 3, 7,  2, 7, 6,  3, 0, 4,  3, 4, 7,
});

Manifold imported = Manifold.FromMeshGL(raw);
if (imported.Status() != Error.NoError)
{
    throw new InvalidOperationException(imported.Status().ToStr());
}

// Export, work on the buffers, and come back in. GetMeshGL(-1) asks for no
// normal slot; GetMeshGL64/FromMeshGL64 are the f64 pair, which round-trips
// the kernel's own precision with nothing narrowed.
MeshGL exported = imported.GetMeshGL(-1);
Manifold reimported = Manifold.FromMeshGL(exported);

// Text, for a fixture or a bug report — 19 digits, faces sorted, tolerance
// and epsilon carried in comments so ReadObj restores them.
string obj = imported.WriteObj();
Manifold fromText = Manifold.ReadObj(obj);
```

### Messy input

```csharp
// Imports geometry the strict pipeline would reject as NotManifold, as long
// as it is closed and orientable.
Manifold scan = Manifold.FromMeshGLRobust(soup);

// Per call...
Manifold cut = scan.DifferenceWithEngine(cutter, BooleanEngine.Auto);

// ...or once, process-wide.
BooleanConfig.SetDefaultEngine(BooleanEngine.Auto);
Manifold cut2 = scan.Difference(cutter);

// Repair, when the input is wound inside out or overlaps itself:
bool messy = scan.HasSelfIntersections();
Manifold rewound = scan.RepairOrientation();
Manifold resolved = rewound.RebuildSolid(WindingRule.Nonzero);
```

Geometry that is not even closed imports as empty with `Error.NotClosed`. The default
engine remains `Exact`, so existing code is unchanged.

### Ray casting

```csharp
foreach (RayHit hit in solid.RayCast(new Vec3(0.0, 0.0, -4.0), new Vec3(0.0, 0.0, 4.0)))
{
    // hit.Distance is the parametric position along the segment in [0, 1] —
    // 0 is the origin, 1 is the endpoint — not a length.
    Console.WriteLine(
        $"t={hit.Distance} at ({hit.Position.X}, {hit.Position.Y}, {hit.Position.Z}) tri={hit.FaceId}");
}
```

## The two engines

| | `BooleanEngine.Exact` | `BooleanEngine.Robust` |
|---|---|---|
| what it is | the ported C++ pipeline (floats plus symbolic perturbation) | an exact-rational mesh arrangement |
| input it needs | strictly manifold operands | closed, orientable triangle soup |
| output | bit-identical to manifold-rust, and so to the C++ reference | correct on input the exact engine cannot express |
| cost | fast | slower; triangulation may differ from `Exact` |

`BooleanEngine.Exact` is the default, so `default(BooleanEngine)` — every zero-initialized
options struct — is the fast path, and reaching anything else takes asking for it.

`BooleanEngine.Auto` decides per call, on the input only: it uses `Robust` when the
winding rule is `Nonzero`, when either operand is soup-backed, or when either operand
fails a **cached exact self-intersection scan**; otherwise `Exact`. It never catches a
fault from the exact engine and reads it as a dispatch signal — a fault there is a bug to
report.

`WindingRule` is a robust-engine semantic only. `Positive` (`w >= 1`, the default)
discards inside-out regions, the mathematically standard reading of orientation;
`Nonzero` (`w != 0`) keeps them, which is what scans and CAD exports with inconsistently
wound shells usually mean.

The robust engine's corpus validation (a sweep over ~10k Thingi10K meshes, with a
Monte-Carlo referee arbitrating volume disagreements) belongs to the Rust project and is
reported there; what this port claims is that its robust engine reproduces the Rust's
answers bit for bit.

## Parallelism

Rust gates its parallel loops behind the compile-time `parallel` Cargo feature. A C# class
library ships one assembly to every consumer, so the analogue here is a runtime switch:

```csharp
// Set once, at startup, before any geometry runs.
ManifoldParallel.Enabled = true;
```

It is **off by default** and is seeded at process start from the `MANIFOLD_PARALLEL`
environment variable (`1` or `true`), which is how the whole test suite is run with the
parallel loops live. Eleven sites participate — the six manifold-rust blesses by name
(`intersect12`, `winding03`, `face2tri`, the SDF voxel fill, the Minkowski per-face hulls,
`calculate_vert_normals`) plus the robust engine's five per-triangle maps. Each writes
`result[i]` for its own `i` into a pre-allocated array and reads nothing another index
writes, which is why the switch cannot change an answer.

**One caveat, at one site.** Exceptions from a parallel body propagate unwrapped, so a
caller catches what the sequential loop would have thrown — *unless several workers fault
at once*, which has no sequential counterpart at all. That case keeps the
`AggregateException` the TPL raises. The only map that runs caller-supplied code is the
SDF voxel fill, so `Manifold.LevelSet` and its overloads are the only public entry points
where a throwing callback can surface as an `AggregateException`, and only with
`ManifoldParallel.Enabled` set.

## Determinism guarantees

1. **Identical results to manifold-rust** on identical inputs: identical floating-point
   values, triangle counts, vertex positions, topology. "Close enough" is a bug, not a
   tolerance.
2. **Bit-identical across platforms.** Every transcendental the kernel depends on goes
   through `DeterministicMath` — Sin, Cos, Tan, Asin, Acos, Atan, Atan2 transcribed from
   FreeBSD msun via musl — never `System.Math`, whose results are the platform's libm.
   `Types.Sind` / `Types.Cosd` add the degree-argument reduction that is exact at
   multiples of 90, because those feed circle vertex positions and a 1-ULP difference
   there moves real geometry.
3. **Bit-identical sequential versus parallel.** Stricter than upstream C++
   `MANIFOLD_PAR`, which permits nondeterministic vertex ordering in some phases. The
   suite is run both ways, and a failure that appears only under `MANIFOLD_PARALLEL=1` is
   treated as a real determinism bug.
4. **One documented exception**, at the Minkowski call site: parallel per-face hulls
   consume mesh-**ID values** from the process-global counter in worker order rather than
   face order. Those handles are opaque and already move between two sequential runs;
   geometry, topology, counts and every relationship are bit-identical either way, which
   is measured rather than asserted.

Deliberate divergences from the Rust are a ledger, not a footnote —
[`docs/RUST_DIVERGENCES.md`](docs/RUST_DIVERGENCES.md) holds all five, and each is a case
where the Rust behaviour is unreproducible in a managed runtime, unspecified in Rust
itself, or a provable defect — never an accuracy change:

1. [`Vec2`'s hash is the plain field-order bit hash](docs/RUST_DIVERGENCES.md#1-vec2s-hash-is-the-plain-field-order-bit-hash-2026-08-29)
   — the Rust impl mixes in a freshly seeded `RandomState` and so is not a function of its
   input; the value is unreachable from any output.
2. [Signed-zero ties in `MinF64` / `MaxF64` are pinned to .NET semantics](docs/RUST_DIVERGENCES.md#2-signed-zero-ties-in-minf64--maxf64-are-pinned-to-net-semantics-2026-08-29)
   — Rust documents the `+0.0` / `-0.0` tie as returning either input
   non-deterministically, so there is no "the Rust result" to match.
3. [`Slice` seeds its polygon loops from the smallest remaining triangle](docs/RUST_DIVERGENCES.md#3-slice-seeds-its-polygon-loops-from-the-smallest-remaining-triangle-2026-08-29)
   — the Rust seeds from a `HashSet` iterator, which returns a different member on each
   run of the same binary. Every contour is bit-for-bit a Rust contour; only the order and
   the starting rotation are pinned.
4. [The centered cylinder rebuilds its collider](docs/RUST_DIVERGENCES.md#4-the-centered-cylinder-rebuilds-its-collider-2026-08-30)
   — the Rust centers by editing vertex positions in place and refreshing only the
   bounding box, leaving the cached face BVH describing the pre-shift mesh, so every
   boolean against a centered cylinder or cone failed. The C++ this both ports does not
   have the defect. No geometry bit moves; only the stale cache is rebuilt.
5. [`SubdivideImpl` runs the whole of `refine`'s finishing tail](docs/RUST_DIVERGENCES.md#5-subdivideimpl-runs-the-whole-of-refines-finishing-tail-2026-08-30)
   — the Rust runs two of that tail's six steps, leaving a stale collider and a `VertNormal`
   shorter than `VertPos` behind the same defect as entry 4. The only entry that is not
   bit-free: the missing `SortGeometry` means the repair reorders the output.

## Performance

Measured against the manifold-rust release build (`lto = "fat"`, `codegen-units = 1`) on
the same machine, same session, same operations. Ratios are C# ÷ Rust; greater than 1
means C# is slower. Full methodology, the memory table, the per-stage breakdown and the
honest qualifications are in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

| workload | seq ratio | par ratio |
|---|---|---|
| Generic_Twin_7081 union | 1.31 | 1.31 |
| stretchy bracelet, build + MinGap | 1.63 | 1.63 |
| Menger sponge depth 4 + convex hull | 1.65 | **0.98** |
| large scene, 7 999-sphere union | 1.70 | 1.13 |
| sphere difference, 2 097 152 tri | 1.86 | 1.84 |
| SDF blobs, edge length 0.05 | 2.04 | 2.03 |
| peak memory (max RSS, 2M boolean) | 1.12 | 1.12 |

Read the top of the table first: the composed, real-world workloads — many booleans over
irregular meshes, which is what an application actually asks a kernel for — land at
1.3–1.7× of a fat-LTO Rust binary, and the Menger sponge is *faster* in parallel than the
Rust is (0.98×), because Rust's `parallel` feature makes that CSG-tree workload slower and
this port's does not. The single enormous boolean reads 1.86× / 1.84×, which is roughly
what a JIT costs against `codegen-units = 1` with no bounds checks to elide. The SDF case
(2.04×) is a tight scalar callback over 8 million voxels and is the clearest measurement
of raw per-call codegen in the set. Peak resident memory is within **12 %**.

One stage is outright faster here than in the Rust: `intersect12` in parallel, 0.012 s
against 0.024 s, because this port moved that stage's output from a per-edge collection to
a per-chunk one while the Rust still carries the per-edge `Vec`-of-`Vec`s its own plan
flags. That is not a claim about languages — allocation was what limited the scaling on
both sides.

Every row above produced **identical triangle counts on both sides**. That is what makes
the comparison meaningful: the two implementations are doing the same work, not similar
work.

The drivers are checked in, so the way to disagree with a number is to re-run it:

```bash
dotnet run -c Release --project ManifoldSharp.Benchmarks -- <driver> [size] --reps 3
MANIFOLD_PARALLEL=1 dotnet run -c Release --project ManifoldSharp.Benchmarks -- <driver> [size] --reps 3
```

Drivers: `perf`, `large-scene`, `mem`, `menger`, `bracelet`, `twins`, `sdf-blobs`,
`robust`, or `all`. The Rust side's mirror crate is checked in as
[`ManifoldSharp.Benchmarks/rust-mirror/main.rs.txt`](ManifoldSharp.Benchmarks/rust-mirror/main.rs.txt),
with the whole recipe in its header.

## Verification

Three independent nets, none of which can be satisfied by the port agreeing with itself.

**1. A test-for-test port of the Rust suite.** Every Rust test module is ported 1:1 by
name and expected value, including the same **9 `#[ignore]`d tests**, carried over with
their reasons. The suite additionally carries clearly-labeled adaptation tests for C#-only
machinery (bit-pattern regressions, file-size compliance, the parallelism sites); those
are counted separately and never stand in for a ported test. Current: **805 tests, 796
passing, 9 skipped**. The Rust helpers that parse meshes out of the pinned C++ test source
at test time are replaced by transcribed, checked-in fixtures, so this repository has no
cpp-reference dependency.

**2. The oracle lane.** `ManifoldSharp.OracleTests` runs the same booleans through this
port and through the native manifold-rust cdylib (the `ManifoldRust` P/Invoke binding) and
compares the exported meshes **row for row, with zero slack** — no canonicalization, no
sorting, no epsilon: triangle triples in corner order, vertex positions bit-for-bit in
index order. 34 rows, covering both engines and both winding rules, plus
`RepairOrientation` and `RebuildSolid`. The published package carries a linux-x64 native,
so this lane runs in CI on every push rather than being a local-only luxury.

**3. Differential harnesses, per module.** The porting loop for every module was: port it,
then dump phase-boundary intermediates from both implementations and diff them
byte-for-byte until they agree. Roughly **3.4M compared state lines** across all harnesses
through the robust engine, with zero diffs. When a ported test disagrees, the Rust's expected
value is the specification and the C# output is the bug — tests are never weakened to
pass.

On top of those: the whole suite is run a second time with `MANIFOLD_PARALLEL=1`, in Debug
and Release, which forces every ported expected value to survive the parallel loops. CI
(`.github/workflows/ci.yml`) builds the solution in Release and runs both the main suite
and the oracle lane on every push and pull request.

## Building and testing

```bash
dotnet build ManifoldSharp.sln
dotnet test --project ManifoldSharp.Tests/ManifoldSharp.Tests.csproj
dotnet test --project ManifoldSharp.OracleTests/ManifoldSharp.OracleTests.csproj
```

Run these **from the repository root**: `global.json` is what opts `dotnet test` into
Microsoft.Testing.Platform, which TUnit requires. The `--project` flag is load-bearing —
a positional path that fails to resolve silently falls back to the solution in the current
directory and can report "Zero tests ran" (exit 5).

The determinism run, which must be green in Debug and in Release:

```bash
MANIFOLD_PARALLEL=1 dotnet test --project ManifoldSharp.Tests/ManifoldSharp.Tests.csproj -c Release
```

## Status and known limits

**The port is complete** — every module, both engines, the parallel feature, and the
integration. agg-sharp's `PolygonMesh` boolean kernel is this library (the `ManifoldRust`
P/Invoke binding it replaced is retired to the oracle role), and MatterCAD runs its full
suite on it. What is deliberately left undone is one line each in
[`docs/FOLLOW_UPS.md`](docs/FOLLOW_UPS.md).

Honestly:

- **No NuGet package yet**, and no published API-stability promise with it.
- **Parallelism is off by default** and is a host decision, not an automatic one.
- **It is a managed port, and it costs.** 1.3–2.0× the sequential Rust on the benchmark
  set, with the outliers named in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) rather than
  averaged away.
- **The robust engine requires input to be closed**; anything else imports empty with
  `Error.NotClosed`. Large, heavily self-intersecting meshes can be slow in it — a limit
  inherited from the Rust, whose corpus sweep documents it.
- **Errors are a status; faults are not.** A rejected input comes back as empty geometry
  carrying an `Error` value, but an invariant the exact engine cannot maintain — a
  non-manifold intermediate, say — throws, mirroring the Rust's panic at the same place.
  If you feed the exact engine geometry you have not validated, guard it, or hand the
  decision to `BooleanEngine.Auto`.

## License and credits

[Apache-2.0](LICENSE), matching the original Manifold library.

The upstream chain: **Manifold C++** (Emmett Lalish) → **manifold-rust** (Lars Brubaker's
exact-match Rust port, plus the robust engine) → **manifold-sharp** (this pure C# port).

- **[Emmett Lalish](https://github.com/elalish)** — author of the original
  [Manifold](https://github.com/elalish/manifold) C++ library, which both ports follow.
- **[Lars Brubaker](https://github.com/larsbrubaker)** — port author.
- **[MatterHackers](https://www.matterhackers.com)** — sponsor.
</content>
