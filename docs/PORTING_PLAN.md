# manifold-rust → C# Porting Plan

This is the roadmap for porting [manifold-rust](https://github.com/larsbrubaker/manifold-rust)
— itself an exact-match Rust port of [Manifold](https://github.com/elalish/manifold) C++
v3.5.x — to pure managed C#. It tracks what is left to do; use `git log` for history.

**Why a pure port when a binding exists:** `manifold-rust/dotnet/ManifoldRust` is P/Invoke
over a Rust cdylib. It works, but it drags a native artifact through every platform MatterCAD
targets — including a static `.a` on mono-wasm's emcc link line for the browser. A fully
managed port runs anywhere .NET runs with zero native plumbing, and its debugger-visible
internals make MatterCAD-side geometry bugs tractable. The binding does not die: it becomes
the **oracle** this port is verified against (see Verification).

**Reference:** the Rust tree at `~/Development/rust-apps/manifold-rust` (crate v0.14.0),
which matches C++ Manifold v3.5.0 semantics (the Rust parent pins `541c33bd`; its
cpp-reference working tree sits at v3.5.2 / `11235e6b`, delta audited as nothing-to-port
— transcribed fixtures in this repo cite the v3.5.2 commit they were read from). The Rust code, not the C++, is the porting source — it already contains
the documented divergences, the determinism fixes, and file headers written to be ported
from.

**Status:** Phases 0–10 complete (2026-08-30, 785 tests / 9 skips + 32 oracle-lane
tests): everything through the public façade, smoothing/subdivision/SDF, CrossSection,
the exact-arithmetic tier and the whole robust engine — every step differentially
verified bit-exact against the compiled Rust (~3.4M compared state lines across all
harnesses, zero diffs). The 9 skips are the Rust's 9 `#[ignore]`s, same reasons. The
oracle lane compares booleans row-for-row against the native library with zero slack,
on both engines and both winding rules. Remaining: Phases 11, 12 below.

---

## The exactness bar (non-negotiable, inherited from the Rust port)

1. **Identical results on identical inputs** versus manifold-rust: identical
   floating-point values, triangle counts, vertex positions, topology.
2. **Deliberate divergence only for provable defects/improvements or genuinely
   unspecified Rust behavior**, documented in `docs/RUST_DIVERGENCES.md` — kept short,
   and never for convenience. No entry may change a specified numerical result.
3. **No stubs.** No `NotImplementedException`, no placeholders. Dependencies get
   implemented first, in dependency order.
4. **Tests ported with (or before) each module.** Every Rust test module is ported
   1:1 by test name and expected value (the current Rust tree holds 722 `#[test]`s,
   9 `#[ignore]`d with reasons — all nine carried over verbatim). The C# suite
   additionally carries its own clearly-labeled adaptation tests for C#-only logic
   (bit-pattern regressions, compliance gates, invented-here machinery); those are
   counted separately and never stand in for a ported test.
5. **Never weaken a test to make it pass.** Every failure is a real bug, resolved by
   instrumentation and root-cause analysis (dump phase-boundary intermediates from both
   implementations, diff byte-for-byte).
6. **Sequential and parallel builds must be bit-identical** (stricter than upstream C++).
   Parallelism arrives last, and only at determinism-preserving sites: the six the Rust
   port blesses by name (`intersect12`, `winding03`, `face2tri`, SDF voxel fill,
   minkowski hulls, `calculate_vert_normals`) plus the robust engine's five per-triangle
   maps, which reach the same helper through `Progress.MaybeParMapCtProgress` exactly as
   they do in the Rust — eleven in all, and the Rust `parallel` feature's own scope.

## Project shape

- Repo: `larsbrubaker/manifold-sharp`, Apache-2.0, submodule of agg-sharp at
  `Submodules/manifold-sharp`.
- `ManifoldSharp/ManifoldSharp.csproj` — net10.0 class library, no dependencies except
  `Clipper2` (official C# Clipper2Lib, same upstream author/numerics as the
  `clipper2-rust` the Rust port uses).
- `ManifoldSharp.Tests/` — TUnit, mirroring the Rust `#[cfg(test)]` modules one file per
  Rust test module. The 20 Thingi10K STL fixtures (7.2 MB) are test-project content,
  never packaged.
- `ManifoldSharp.OracleTests/` — lane comparing this port bit-for-bit against the
  native library through the existing `ManifoldRust` binding, consumed as the published
  NuGet package (which ships natives for win-x64/linux-x64/osx-arm64/osx-x64), so it
  runs in CI too. `MANIFOLD_RS_NATIVE` can still point it at an unpublished cargo build.
- File conventions from the Rust port carry over: 800-line file cap (documented
  exceptions only), every file opens with a header stating purpose and relations —
  **port the Rust headers verbatim**, they are the best porting artifact in the repo.
- Errors are a status enum on the result (`ManifoldImpl.Status`, empty result on
  failure), not exceptions — `error_propagation` tests define the semantics. The Rust
  `Error` enum's declaration order is FFI-meaningful; keep it.

## Dependency replacements

| Rust crate | Confined to | C# replacement |
|---|---|---|
| `dashu-int`/`dashu-ratio` | `src/robust/exact/backend.rs` only | `System.Numerics.BigInteger` + a hand-written canonical `BigRational` (auto-reduced, sign on numerator). The 7-item "backend-coupled hot spots" checklist at the top of `backend.rs` is the acceptance spec. `rat_to_f64` (correctly-rounded rational→double) is hand-ported, never delegated. |
| `clipper2-rust` | `src/cross_section.rs` only | `Clipper2` NuGet — same API names (`Union`, `InflatePaths`, `Area`, `PathsD`…). Lowest-risk dependency in the port. |
| `rustc-hash` | 7 robust files, all probe-only maps | Plain `Dictionary`/`HashSet` (safe *because* every site is documented probe-only; keep those comments). `hash_rational`'s limb-level hash becomes an `IEqualityComparer<BigRational>`. |
| `rayon` (optional) | `src/par.rs` only | Done: `Parallel.For` writing into pre-allocated arrays — index-ordered, bit-identical to sequential, at all **eleven** sites the Rust feature covers (the six blessed by name, plus the robust engine's five per-triangle maps, which reach `Par` through `Progress.MaybeParMapCtProgress`). Rust's compile-time `parallel` feature becomes the runtime switch `ManifoldParallel.Enabled` (default off, seeded from `MANIFOLD_PARALLEL`), since one C# assembly ships to every consumer. |
| `num-traits` | re-exported from `backend.rs` | Nothing; concrete `BigInteger` methods cover it. |

## Phases

Line counts are production Rust to port; tests are the Rust tests that must pass to close
the phase. Order follows the module dependency graph; each phase compiles and is fully
tested before the next begins. Phases marked ∥ are independent islands a second worker
can take in parallel.

**Phase 11 — Parallel & performance.** The parallel feature is done: `Parallel.For`
behind `ManifoldParallel.Enabled` at all eleven sites the Rust feature covers — the six
blessed by name and the robust engine's five per-triangle maps, which share the helper
via `Progress.MaybeParMapCtProgress` — with `ParallelismTests.cs` asserting bit-identity
sequential-vs-parallel at each (one robust-pipeline test covers all five robust maps at
once, since a robust boolean drives every one of them). Run the whole suite against
it with `MANIFOLD_PARALLEL=1` — that forced-on run is the strongest determinism net the
port has, and it must stay green in Debug and Release. Remaining: port
`perf_test`/`large_scene_test`/`mem_profile` drivers; benchmark against the Rust
release build. C# perf levers replacing Rust's fat-LTO: `AggressiveInlining` on linalg
operators, pre-sized arrays, struct discriminated unions on hot enums (`TriLoc`),
tiered-PGO. Set expectations honestly: parity with C++/Rust is the goal, but
`BigInteger` on the `intpred` tier will profile differently than dashu's inline words —
measure before optimizing. **Deliverables (user-requested):** (a) a measured
performance comparison against the Rust release build using the ported drivers, on the
same benchmarks the Rust README reports (sphere/menger booleans, bracelet, twins,
sdf_blobs, large-scene, peak memory), published as a table; (b) the README brought to
parity with manifold-rust's — features, usage examples, API tour, engine explanation
(exact vs robust), determinism guarantees, the wasm story (pure managed — the port's
whole point), the performance table from (a), and the verification story (test-for-test
suite, oracle lane, differential harnesses).

**Phase 12 — Integration.** agg-sharp `PolygonMesh` switches from the `ManifoldRust`
NuGet package (0.5.0) to a project reference on `Submodules/manifold-sharp`; the wasm
head drops the `NativeFileReference`/`wasm-opt` special-casing; MatterCAD main takes the agg-sharp bump and runs its full suite (the port
lives on main everywhere; the old manifold-sharp branches are retired).

## Verification strategy (three independent nets)

1. **Test-for-test port** of the Rust suite — every Rust test module 1:1 by name and
   expected value, same 9 ignores (plus the port's separately-counted adaptation tests). The `manifold_tests/mod.rs` helpers that parse meshes out of the
   pinned C++ test source at test time are replaced by transcribing those meshes into
   checked-in test data (no cpp-reference dependency here).
2. **Oracle lane**: the same operations through the `ManifoldRust` P/Invoke binding,
   MeshGL outputs compared bit-for-bit. Local/env-gated (needs a cargo build).
3. **Trace diffing** for debugging: `MANIFOLD_TIMING`-style phase instrumentation on
   both sides; when a test diverges, dump intermediates at phase boundaries and diff.

## C# translation rules (each one has burned someone before)

- **Stable sorts:** `Array.Sort`/`List<T>.Sort` are unstable introsort. Every Rust
  `sort_by_key` site (stable) becomes LINQ `OrderBy` (documented stable) or an explicit
  index tie-break comparator. Only the 16 audited `sort_unstable` sites may use `Sort`.
- **Ordered assertions:** a ported `assert_eq!` on a Vec must pass
  `CollectionOrdering.Matching` — TUnit's `IsEquivalentTo` default is order-insensitive
  and silently turns a sequence comparison into a set comparison.
- **No FMA, ever:** the Rust code has zero `mul_add`; RyuJIT doesn't auto-contract
  today, but keep it a tested invariant, and never introduce `Math.FusedMultiplyAdd`
  or `Vector<T>` into math paths.
- `f64::EPSILON` → the literal `2.220446049250313E-16`. C# `double.Epsilon` is the
  smallest subnormal and is **wrong**.
- `f64::NAN` → a positive-quiet-NaN constant built from bits `0x7ff8000000000000`.
  C# `double.NaN` carries the sign bit set (`0xfff8...`) and diverges from Rust in any
  bit-based hash/weld/compare. (Found by differential fuzzing in the math.rs port.)
- **Float→integer casts are settled, no audit needed:** Rust `as i32`/`as i64` saturate,
  and since .NET 9 so does C# — every float→integer conversion clamps over-range to the
  target's Max, under-range to its Min, and maps NaN to zero, deterministically on every
  platform (`dotnet/core/compatibility/jit/9.0/fp-to-integer`). On this net10.0 target a
  bare `(int)`/`(long)` cast *is* Rust `as`, so no clamping helper is required
  (`DeterministicMath.SaturatingToInt32` is kept as executable documentation and as the
  guard if this is ever retargeted below net9.0). The audit obligation is only for
  **integer→integer** narrowing: C#'s default `unchecked` wraps, which matches Rust `as`
  and Rust *release* arithmetic but not the overflow panic a Rust debug build would have
  raised, so a site relying on wrapping should say so in a comment.
- Float hashing/welding via `BitConverter.DoubleToInt64Bits`, with the `-0.0 → 0.0`
  normalization from `robust/soup.rs` reproduced byte-for-byte.
- `lerp` is `a*(1-t) + b*t`, never `a + (b-a)*t`.
- Enums-with-data: `CsgNode` → abstract base + sealed subclasses; `TriLoc` and other
  hot-loop enums → struct with `Kind` tag + payload fields (no per-call allocation).
- Arena style stays: arrays + `int` indices, no object graphs where Rust used indices.
- `recursive_edge_swap` takes its `tag` as `ref int`, keeps the explicit `int` stack
  with the `edge < 0` guard.
- Statics: mesh-ID counter via `Interlocked.Increment`; quality/engine defaults and the
  partition cache behind locks; the `SelfIntersectCache` clone-invalidation trap carries
  over to any copy constructor.

## Working agreement

Implementation is delegated stepwise (one module or coherent module group per step),
each step closing with its ported tests green. The Rust file being ported is read in
full first, including its header and its tests. When a Rust comment documents an
invariant, the C# port keeps the comment.
