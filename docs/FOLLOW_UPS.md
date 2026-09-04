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

- **An n-ary CSG-tree union can produce a self-intersecting result where the pairwise fold of
  the same operands does not.** Found from MatterCAD's bevel: 27 operands (17 swept run
  cutters plus 10 corner-patch shells), every one of them individually `Clean` and manifold by
  `HasSelfIntersections`, unioned through `CsgOp` / `EvaluateWithToken` - the tree path
  `ManifoldKernel.BatchBoolean` builds - come back **`SelfIntersecting`, manifold=True, 2824
  verts**. Folding the identical operands pairwise gives **`Clean`, 2941 verts**; unioning them
  as two groups (the 17 runs, then the 10 patches, then those two) gives **`Clean`, 2823
  verts** - within one vertex of the tree result, so the two agree on the shape and disagree
  only on validity. Narrowed: one patch alone with the 17 runs is enough to reproduce it
  (`SelfIntersecting`), the other nine are not. What the batch carries is many small coplanar
  contacts between operands, but coplanar overlap is not sufficient on its own - two cutters
  sharing 4 mm^2 of wall, and a patch sharing 124 mm^2 with a run, both union without
  complaint - so it is the count or some particular pair. Consumer-side workaround in place:
  `BevelFeatureMeshBuilder.UnionForOperandAsync` tries the tree first and falls back to two
  groups and then to a pairwise fold, so the fast path is unchanged. Reproducer:
  MatterCAD `BevelMinkowskiOracleTests.AnLShapedPrismAgreesOnItsConvexEdgesAndBothLeaveTheConcaveOneSharp`
  at radius 1 with every edge selected, with the bevel drive-through corner enabled; the
  coplanar contacts are listable with MatterCAD's `BevelCoplanarProbe`.

- **`refine`'s tail order must be harmonized in both repos at once.** manifold-rust's
  `manifold_smooth.rs` tail — which this port transcribes as `FinishRefine` in
  `ManifoldSharp/Manifold.Smooth.cs` — runs `calculate_bbox` + `set_epsilon`
  unconditionally and computes normals *after* `sort_geometry`, where C++
  `Impl::Refine` (`smoothing.cpp:1120-1128`) sorts last and never calls `SetEpsilon`. On
  the tangent-free leg (the one `SubdivideImpl` uses) the two orders are bit-free
  equivalents; the `hadTangents` leg is the open question, because running
  `SetNormalsAndCoplanar` after the sort rather than before it can group faces into
  different coplanar IDs, and nothing asserts those groupings today. The oracle lane
  compares with no slack, so changing the order on one side alone breaks the other.
  Pointer: manifold-rust `PORTING_PLAN.md`, "Needs investigation".

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
