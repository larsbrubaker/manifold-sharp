# manifold-sharp — agent guidelines

Pure C# port of manifold-rust (which is itself an exact-match port of Manifold C++
v3.5.x). **Read `PORTING_PLAN.md` before doing anything** — it carries the phase order,
the dependency-replacement table, and the C# translation rules. The porting source is
the Rust tree at `~/Development/rust-apps/manifold-rust`, not the C++.

Rules that override instinct:

- **Exact numerical match with manifold-rust.** Identical floats, counts, positions,
  topology. "Close enough" is a bug.
- **No stubs, no `NotImplementedException`.** Port dependencies first, in the phase
  order from the plan.
- **Port tests with the code**, same expected values as the Rust suite. Never weaken a
  test to make it pass; instrument both implementations and diff intermediates instead.
- **Port each Rust file's header comment and invariant comments verbatim** (adjusted for
  C# names). They document why the "obvious" implementation is wrong in ~20 places.
- 800-line file cap, documented exceptions only (linalg, edge_op, quickhull_algo
  inherit their exemptions from the Rust port).
- **Naming:** a Rust module's free functions land in a static class named for the module
  (`types.rs` → `Types`), taking a `Functions` suffix only when that bare name would
  collide with a primary type or a namespace (`LinalgFunctions`, `SvdFunctions`).
  Type-name spellings that aid three-way diffing against the C++ and the Rust may keep
  their source casing instead of C# casing (`SVDSet`).
- Errors are a status enum on the result, not exceptions.
- Stable-sort discipline, no FMA, `2.220446049250313E-16` not `double.Epsilon` — the
  full list is in PORTING_PLAN.md "C# translation rules".

Build: `dotnet build ManifoldSharp.sln` · Test: `dotnet test --project
ManifoldSharp.Tests/ManifoldSharp.Tests.csproj` (TUnit; run from repo root where
`global.json` opts into Microsoft.Testing.Platform). The `--project` flag is
load-bearing: a positional path silently falls back to the cwd solution and can
report "Zero tests ran" (exit 5).
