# manifold-sharp — agent guidelines

Pure C# port of manifold-rust (which is itself an exact-match port of Manifold C++
v3.5.x). **Read `docs/PORTING_PLAN.md` before doing anything** — it carries the phase order,
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
- **Ported `assert_eq!` on a Vec must pass `CollectionOrdering.Matching`:** TUnit's
  `IsEquivalentTo` defaults to `CollectionOrdering.Any`, which is order-insensitive and
  silently turns a sequence comparison into a set comparison.
- Errors are a status enum on the result, not exceptions.
- Stable-sort discipline, no FMA, `2.220446049250313E-16` not `double.Epsilon` — the
  full list is in docs/PORTING_PLAN.md "C# translation rules".

Build: `dotnet build ManifoldSharp.sln` · Test: `dotnet test --project
ManifoldSharp.Tests/ManifoldSharp.Tests.csproj` (TUnit; run from repo root where
`global.json` opts into Microsoft.Testing.Platform). The `--project` flag is
load-bearing: a positional path silently falls back to the cwd solution and can
report "Zero tests ran" (exit 5).

Parallelism is off by default and is the C# stand-in for Rust's `parallel` Cargo
feature (`ManifoldParallel.Enabled`, see Par.cs). Prefixing any test command with
`MANIFOLD_PARALLEL=1` seeds the switch on for that process, which is how the suite is
run with the parallel loops live — the port's strongest determinism net, since every
ported expected value then has to survive them. **Both configurations must be green,
in Debug and in Release**, and a failure that only appears under `MANIFOLD_PARALLEL=1`
is a real determinism bug: fix the code, never the test.

## Orchestration pattern

The main session (the supervisor) acts as planner and orchestrator only — it does not
write or edit code directly. All implementation is delegated to the `implementer`
subagent (`.claude/agents/implementer.md`), one scoped step at a time. All post-change
review is delegated to the `reviewer` subagent. Test failures go to the
`fix-test-failures` agent, which treats every failure as a real bug — for a ported test
that means the Rust's expected value is the specification, never the C# output. The
main session handles planning, architecture decisions, and synthesizing results.

The porting loop that has held since Phase 0: the implementer ports a module with its
own scratchpad differential harness against the compiled manifold-rust → the reviewer
independently audits text-level fidelity against the Rust source (the net a harness
cannot provide) → a fix round → the supervisor commits selectively and watches CI.

## File-size rule

The 800-line file cap is enforced by `FileComplianceTests` in the test suite, with an
explicit exemption list. Current exemptions: `QuickHull.Algo.cs` (867 now, ceiling 880
— inherits the
Rust quickhull_algo exemption; a further split would cut between
SetupInitialTetrahedron's degenerate branches and the loop that assumes them away).
The other Rust exemptions (linalg, edge_op) did not carry over — those C# files split
instead. Adding an exemption requires the same documented justification the Rust files
carry. Reducing line count by deleting comments or blank lines is not compliance.
