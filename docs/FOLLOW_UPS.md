# Follow-ups

Open items left behind by the completed port. One line each, with the pointer that lets a
future session pick it up. Delete an entry when it is done — see `docs/CLAUDE.md`.

## Consumers (agg-sharp / MatterCAD)

- **Browser CSG performance is unmeasured.** The kernel is managed everywhere now, but the
  browser leg has never been timed; the shape to expect is that mesh *import* dominates the
  boolean itself, and the AOT/relink story (`wasm-tools`, trimming) is untested against it.
  Pointer: agg-sharp commit `e2229217`, `PolygonMesh/Csg/ManifoldKernel.cs` static ctor
  (parallelism is off in the browser — the one agg platform with no worker threads).
- **The `Rust*` / `ManifoldRust*` identifier names in agg-sharp are now misleading.** They
  were left alone deliberately, to keep the swap commit reviewable line-for-line against the
  FFI source; renaming is a separate mechanical pass. `ManifoldKernel.cs`: `RustManifold`,
  `RustCsgNode`, `RustCsgOp`, `RustCsgLeaf`, `RustMeshGL64`, `RustStatus`, `RustOpType`,
  `RustPhases`, `RustParallel`, `RustProgressReporter`, `RustCancelToken`,
  `RustBooleanConfig`, `RustBooleanEngine`, `RustWindingRule`; also
  `BooleanProcessing.DoArrayViaManifoldRust`, the file `PolygonMesh/Csg/MeshRepairRust.cs`,
  and `Tests/Agg.Tests/Agg.PolygonMesh/ManifoldRustBackendTests.cs`.
- **`ManifoldKernel`'s `BatchBoolean` is a CSG-tree composite living in a consumer.** It
  spells out leaves-over-clones → one `CsgOp` → `EvaluateWithToken` because
  `Manifold.BatchBoolean` is the pairwise-fold shorthand, not the tree the P/Invoke binding
  exposed under that name. Consider promoting the composite to a `ManifoldSharp` API so the
  consumer stops carrying kernel logic. Pointer: `PolygonMesh/Csg/ManifoldKernel.cs:875-933`.

## Upstream (manifold-rust)

- **Ledger entry 4** — `cylinder`'s `center` branch should center by transform as the C++
  does, instead of editing `vert_pos` in place and leaving a stale face BVH. Fix upstream and
  retire the entry on the re-sync. Pointer: `docs/RUST_DIVERGENCES.md#4`.
- **Ledger entry 5** — `subdivide_impl` should run the whole of `refine`'s finishing tail,
  not two of its six steps. Fix upstream and retire the entry on the re-sync. Pointer:
  `docs/RUST_DIVERGENCES.md#5`.

## Performance

- **The next levers are named but untouched**: the `UnionWithEngine` triangulation outlier,
  `Triangulation` generally, `Kernel12`'s three allocations per call, `Collider.TraverseBvh`'s
  unzeroed stackalloc, and `Collider`'s `Action<int,int>` recorder. Pointer:
  `docs/BENCHMARKS.md`, "Next levers", which is also the baseline any of them is measured
  against.

## Coverage

- **Two branches are differentially unproven** — no harness input has reached them, so they
  are ported-and-reviewed but not diffed against the Rust at run time: `WindingOffSurface`'s
  graze retry (`ManifoldSharp/Robust/RayShoot.cs:355`, the `if (grazed)` leg that advances to
  the next candidate direction) and `RetriangulateCorridor`'s constrained-pinch re-mark
  (`ManifoldSharp/Robust/Cdt.Constraints.cs:394`, the `interiorCon` loop). Both are documented
  at the code; an input that reaches either is worth keeping as a fixture.
</content>
