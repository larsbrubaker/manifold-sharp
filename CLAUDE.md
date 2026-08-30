# manifold-sharp — agent guidelines

Pure C# port of [manifold-rust](https://github.com/larsbrubaker/manifold-rust) (itself an
exact-match port of Manifold C++ v3.5.x). **The port is complete** — agg-sharp's
`PolygonMesh` and MatterCAD run on it. What follows is therefore a *maintenance* contract,
not a plan: the whole value of this library is that its output is indistinguishable from the
Rust's, and any change — a bug fix, an optimization, a new feature — inherits the bar below.

## The maintenance contract (non-negotiable)

1. **Identical results on identical inputs** versus manifold-rust: identical floating-point
   values, triangle counts, vertex positions, topology. "Close enough" is a bug. This binds
   new work, not just ported code: a change must leave every existing bit where it was, and
   must be argued that way rather than assumed.
2. **Deliberate divergence only for provable defects/improvements or genuinely unspecified
   Rust behavior**, and only as a numbered entry in `docs/RUST_DIVERGENCES.md` — the single
   sanctioned record, kept short, never for convenience. No entry may change a specified
   numerical result. An undocumented divergence is a bug even when it looks like an
   improvement.
3. **No stubs.** No `NotImplementedException`, no placeholders.
4. **The tests are the Rust's, 1:1** by name and expected value, including the same 9
   `#[ignore]`s with their reasons. C#-only machinery (bit-pattern regressions, compliance
   gates, the parallelism sites) gets clearly-labeled adaptation tests, counted separately,
   which never stand in for a ported test.
5. **Never weaken a test to make it pass.** Every failure is a real bug, resolved by
   instrumentation and root-cause analysis. For a ported test the Rust's expected value is
   the specification and the C# output is the bug.
6. **Sequential and parallel builds must be bit-identical** — stricter than upstream C++,
   which permits nondeterministic vertex ordering in some phases. Parallelism lives at
   exactly eleven determinism-preserving sites: the six manifold-rust blesses by name
   (`intersect12`, `winding03`, `face2tri`, SDF voxel fill, Minkowski hulls,
   `calculate_vert_normals`) plus the robust engine's five per-triangle maps, which reach the
   same helper through `Progress.MaybeParMapCtProgress`. That set is the Rust `parallel`
   feature's own scope; widening it needs the same proof each existing site carries — every
   worker writes `result[i]` for its own `i` and reads nothing another index writes.

## Reference and oracle

- **Reference:** the Rust tree at `~/Development/rust-apps/manifold-rust` (crate v0.14.0),
  which matches C++ Manifold v3.5.0 semantics — the Rust parent pins `541c33bd`; its
  cpp-reference working tree sits at v3.5.2 / `11235e6b`, delta audited as nothing-to-port,
  and transcribed fixtures in this repo cite the v3.5.2 commit they were read from. The
  Rust, never the C++, is the authority: it already carries the documented divergences, the
  determinism fixes, and the file headers this port inherited. **Two paths need the Rust at
  `fa18cc5` or later to agree bit-for-bit**: `cylinder`'s `center` branch and
  `subdivide_impl`, whose stale-cache defects this port repaired first and upstream fixed in
  that commit. Both the published crate 0.14.0 and the NuGet 0.5.0 natives predate it and
  still carry the old behaviour there, so a differential harness built against an older
  checkout will disagree with this port on those two functions and be right to — check the
  Rust's commit before chasing it. No oracle row exercises either function on the native
  side, so the lane is unaffected.
- **Oracle:** `manifold-rust/dotnet/ManifoldRust`, a P/Invoke binding over the Rust cdylib,
  consumed as the published NuGet package (natives for win-x64/linux-x64/osx-arm64/osx-x64,
  so the lane runs in CI). It is not this library's ancestor and nothing here calls into it
  at runtime — it exists so the two implementations can be compared. `MANIFOLD_RS_NATIVE`
  points the lane at an unpublished cargo build instead.

## Verification nets (three; none can be satisfied by the port agreeing with itself)

1. **Test-for-test suite** — `ManifoldSharp.Tests`, one file per Rust test module. The Rust
   helpers that parse meshes out of the pinned C++ test source at test time are replaced by
   transcribed, checked-in fixtures, so this repo has no cpp-reference dependency.
2. **Oracle lane** — `ManifoldSharp.OracleTests` runs the same operations through this port
   and through the native library and compares exported meshes **row for row with zero
   slack**: no canonicalization, no sorting, no epsilon, on both engines and both winding
   rules.
3. **Differential harnesses** — the discipline that built the port, and the one a nontrivial
   change still owes. Instrument both implementations, dump intermediates at phase
   boundaries, diff byte-for-byte until they agree. The suite alone cannot prove
   bit-exactness of an interior stage it only observes through a final mesh, so a change to
   one needs its own scratchpad harness against the compiled Rust.

## C# translation rules (each one has burned someone before)

- **Stable sorts:** `Array.Sort`/`List<T>.Sort` are unstable introsort. Every Rust
  `sort_by_key` site (stable) is LINQ `OrderBy` (documented stable) or an explicit index
  tie-break comparator. Only the 16 audited `sort_unstable` sites use `Sort`.
- **Ordered assertions:** a ported `assert_eq!` on a Vec must pass
  `CollectionOrdering.Matching` — TUnit's `IsEquivalentTo` default is order-insensitive and
  silently turns a sequence comparison into a set comparison.
- **No FMA, ever:** the Rust code has zero `mul_add`; RyuJIT doesn't auto-contract today, but
  it is a tested invariant — never introduce `Math.FusedMultiplyAdd` or `Vector<T>` into math
  paths.
- `f64::EPSILON` is the literal `2.220446049250313E-16`. C# `double.Epsilon` is the smallest
  subnormal and is **wrong**.
- `f64::NAN` is a positive-quiet-NaN constant built from bits `0x7ff8000000000000`. C#
  `double.NaN` carries the sign bit set (`0xfff8...`) and diverges from Rust in any bit-based
  hash/weld/compare. (Found by differential fuzzing in the math.rs port.)
- **Float→integer casts are settled, no audit needed:** Rust `as i32`/`as i64` saturate, and
  since .NET 9 so does C# — over-range clamps to the target's Max, under-range to its Min,
  NaN maps to zero, deterministically on every platform
  (`dotnet/core/compatibility/jit/9.0/fp-to-integer`). On this net10.0 target a bare
  `(int)`/`(long)` cast *is* Rust `as` (`DeterministicMath.SaturatingToInt32` is kept as
  executable documentation and as the guard if this is ever retargeted below net9.0). The
  audit obligation is only for **integer→integer** narrowing: C#'s default `unchecked` wraps,
  which matches Rust `as` and Rust *release* arithmetic but not the overflow panic a Rust
  debug build would have raised, so a site relying on wrapping says so in a comment.
- Float hashing/welding goes through `BitConverter.DoubleToInt64Bits`, with the
  `-0.0 → 0.0` normalization from `robust/soup.rs` reproduced byte-for-byte.
- `lerp` is `a*(1-t) + b*t`, never `a + (b-a)*t`.
- Enums-with-data: `CsgNode` is an abstract base plus sealed subclasses; `TriLoc` and other
  hot-loop enums are a struct with a `Kind` tag plus payload fields (no per-call allocation).
- Arena style stays: arrays plus `int` indices, no object graphs where Rust used indices.
- `recursive_edge_swap` takes its `tag` as `ref int` and keeps the explicit `int` stack with
  the `edge < 0` guard.
- Statics: the mesh-ID counter is `Interlocked.Increment`; quality/engine defaults and the
  partition cache sit behind locks; the `SelfIntersectCache` clone-invalidation trap applies
  to any copy constructor.

## Dependency replacements (confined on purpose — keep them confined)

| Rust crate | Confined to | C# replacement |
|---|---|---|
| `dashu-int`/`dashu-ratio` | `Robust/Exact/` backend only | `System.Numerics.BigInteger` plus a hand-written canonical `BigRational` (auto-reduced, sign on numerator). The 7-item "backend-coupled hot spots" checklist at the top of the Rust `backend.rs` is the acceptance spec. `rat_to_f64` (correctly-rounded rational→double) is hand-ported, never delegated. |
| `clipper2-rust` | `CrossSection.Clipper.cs` only | `Clipper2` NuGet — the official C# Clipper2Lib, same upstream author and numerics as the Rust's. |
| `rustc-hash` | 7 robust files, all probe-only maps | Plain `Dictionary`/`HashSet` — sound *because* every site is documented probe-only (never iterated), so the hasher cannot affect determinism. Keep those comments; a new map that is iterated does not get this exemption. `hash_rational`'s limb-level hash is an `IEqualityComparer<BigRational>`. |
| `rayon` (optional) | `Par.cs` only | `Parallel.For` writing into pre-allocated arrays, index-ordered and bit-identical to sequential, at the eleven sites above. Rust's compile-time `parallel` feature becomes the runtime switch `ManifoldParallel.Enabled` (default off, seeded from `MANIFOLD_PARALLEL`), since one C# assembly ships to every consumer. |
| `num-traits` | re-exported from the exact backend | Nothing; concrete `BigInteger` methods cover it. |

## Conventions

- **Preserve the ported headers and invariant comments.** Every file opens with a header
  stating purpose and relations, ported from the Rust's, and roughly twenty of them document
  why the "obvious" implementation is wrong. Changing code under one without updating it is
  how that knowledge is lost.
- **Naming:** a Rust module's free functions live in a static class named for the module
  (`types.rs` → `Types`), taking a `Functions` suffix only when the bare name would collide
  with a primary type or a namespace (`LinalgFunctions`, `SvdFunctions`). Type-name spellings
  that aid three-way diffing against the C++ and the Rust may keep their source casing
  instead of C# casing (`SVDSet`).
- **Errors are a status enum on the result** (`ManifoldImpl.Status`, empty result on
  failure), not exceptions — `ManifoldErrorPropagationTests` is the specification. The Rust
  `Error` enum's declaration order is FFI-meaningful; keep it.

## File-size rule

An **800-line file cap**, documented exceptions only, enforced by `FileComplianceTests` with
an explicit exemption list. Current exemptions: `QuickHull.Algo.cs` (867 now, ceiling 880 —
inherits the Rust quickhull_algo exemption; a further split would cut between
`SetupInitialTetrahedron`'s degenerate branches and the loop that assumes them away). The
other Rust exemptions (linalg, edge_op) did not carry over — those C# files split instead.
Adding an exemption requires the same documented justification the Rust files carry.
Reducing line count by deleting comments or blank lines is not compliance.

## Build and test

Build: `dotnet build ManifoldSharp.sln` · Test: `dotnet test --project
ManifoldSharp.Tests/ManifoldSharp.Tests.csproj` (TUnit; run from repo root where
`global.json` opts into Microsoft.Testing.Platform). The `--project` flag is load-bearing: a
positional path silently falls back to the cwd solution and can report "Zero tests ran"
(exit 5).

Prefixing any test command with `MANIFOLD_PARALLEL=1` seeds `ManifoldParallel.Enabled` on
for that process, which is how the suite is run with the parallel loops live — the port's
strongest determinism net, since every ported expected value then has to survive them.
**Both configurations must be green, in Debug and in Release**, and a failure that only
appears under `MANIFOLD_PARALLEL=1` is a real determinism bug: fix the code, never the test.

## Orchestration pattern

The main session (the supervisor) acts as planner and orchestrator only — it does not write
or edit code directly. All implementation is delegated to the `implementer` subagent
(`.claude/agents/implementer.md`), one scoped step at a time. All post-change review is
delegated to the `reviewer` subagent. Test failures go to the `fix-test-failures` agent,
which treats every failure as a real bug. The main session handles planning, architecture
decisions, and synthesizing results.

The loop that held for every phase of the port still applies to changes: the implementer
works with a scratchpad differential harness against the compiled manifold-rust → the
reviewer independently audits text-level fidelity against the Rust source (the net a harness
cannot provide) → a fix round → the supervisor commits selectively and watches CI.

## Open items

`docs/FOLLOW_UPS.md` carries what is deliberately left undone, one line each with a pointer.
