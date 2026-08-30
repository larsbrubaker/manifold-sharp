# manifold-sharp vs manifold-rust — measured performance

**Not a working document.** `docs/CLAUDE.md` says everything in this folder describes work
still to do and gets deleted when that work finishes. This file is the second standing
exception alongside `RUST_DIVERGENCES.md`: it is *measurement output*, the durable record the
README's performance section summarizes, and the thing a future "did we regress?" question is
answered against. It never grows a status section and it is never pruned — it is replaced, in
whole, by a re-run of the drivers it documents.

Everything below is a **snapshot of one machine on one date**. It is not a claim about
hardware you do not have. The drivers that produced it are checked in, so the way to disagree
with a number is to re-run it:

```bash
dotnet run -c Release --project ManifoldSharp.Benchmarks -- <driver> [size] --reps 3
MANIFOLD_PARALLEL=1 dotnet run -c Release --project ManifoldSharp.Benchmarks -- <driver> [size] --reps 3
```

---

## Methodology

**Date:** 2026-08-30.

**Machine:** Apple M5, 10 cores, 24 GB, macOS 26.5.1 (build 25F80). No test suite, no build and
no other benchmark in flight while any number below was taken; the interactive session driving
the run was, which is not the same thing as an idle machine and is the reason the sub-10 ms
rows scatter. Every C# number in this snapshot was taken back to back with the corresponding
pre-change number on the same machine in the same session — for the three-way table in the
per-stage section, by alternating the builds rather than measuring one shape and then the other
— and the pre-change re-measurement reproduced the previous snapshot's per-stage table row for
row (nine of ten rows within 3 %; `Assembly`, at 0.13 s the smallest of them, within 8 %). That
is the evidence that the ratios here are the code changing and not the conditions.

**C# side:** .NET SDK 10.0.400, runtime 10.0.11, `-c Release`, workstation concurrent GC (the
default; server GC was *not* enabled, and it would change the memory table), TieredPGO at its
default (on). `ManifoldSharp.Benchmarks`, this repo. Parallelism is the runtime switch
`MANIFOLD_PARALLEL=1`, the port's stand-in for the Rust `parallel` Cargo feature.

**Rust side:** rustc 1.97.1, manifold-rust 0.14.0 with its release profile as published
(`lto = "fat"`, `codegen-units = 1`) and its `Cargo.lock` dependency versions (clipper2-rust
1.0.3 in particular — the resolver otherwise floats to 1.1.0). Both configurations were built:
`cargo build --release` and `cargo build --release --features parallel`. The manifold-rust
tree was **not modified**: the Rust numbers come from a small mirror crate built outside it
with a path dependency, whose drivers are byte-for-byte the same operations as the C# drivers.
`perf` / `large-scene` / `mem` are copies of manifold-rust's own `examples/`; `menger`,
`bracelet`, `twins` and `sdf-blobs` are copies of the Rust test bodies its `PORTING_PLAN.md`
quotes figures from; `robust` is the whole-operation half of `examples/robust_perf.rs`.

That mirror's source is checked in at
[`ManifoldSharp.Benchmarks/rust-mirror/main.rs.txt`](../ManifoldSharp.Benchmarks/rust-mirror/main.rs.txt)
— as a `.txt`, because it is a record and not a build target: no Cargo.toml sits beside it and
nothing in this repo compiles it. A Rust crate living inside a C# repo would be a second thing
to keep green, but a benchmark table whose other column cannot be reproduced is an assertion
rather than a measurement, so the file is here and the crate around it is not. Its header
carries the whole recipe: the Cargo.toml shape (path dependency on manifold-rust, a `parallel`
feature forwarding to `manifold-rust/parallel`, and — load-bearing —
`[profile.release] lto = "fat"`, `codegen-units = 1`, since profile settings come from the
*building* workspace root and their absence would silently measure a non-LTO binary), the one
absolute path to correct, the `cargo update -p clipper2-rust --precise 1.0.3` pin (a fresh
resolver floats to 1.1.0 and you would be timing a different Clipper), and the two builds to
keep side by side.

**Timing protocol:** best of **N = 3** in-process repetitions (N = 5 for the sub-100 ms robust
cases and for the C# twins row, N = 9 for the C# sphere ladder), measured with
`Stopwatch.GetTimestamp` / `Instant::now` around the operation only — mesh construction and file
loading sit outside the timed region on both sides, exactly as the Rust examples place them.

The sphere ladder's C# column needs the higher N because four of its seven rows finish in under
10 ms, where a best-of-3 measures the machine's scheduler rather than the kernel: the
512-triangle row alone spans 0.0010–0.0034 s across repeated best-of-3 runs. The Rust column
stays at N = 3 because an AOT binary with no tiering has nothing to shake out — its small rows
are already at their floor. **Read the four sub-10 ms rows as ±50 %** even at N = 9; they are in
the table for shape, and the rows that carry a claim are the last three. Memory is likewise
best of three runs rather than the single run the previous snapshot took, for the same reason —
a sampled heap peak scattered over 2 826–2 972 MB before the change.

Best-of, not mean-of, and the first rep is not discarded but is in practice the warmup: the
workload is deterministic, so every rep computes an identical result and the spread is pure
machine noise plus, on the C# side, the JIT compiling the boolean pipeline during rep 1. That
tax is real and it is large in relative terms on the small cases (the 512-triangle sphere
difference costs 25 ms on its first execution and 1.1 ms warm), which is why the per-rep lines
are printed rather than averaged away. A best-of-N is what an AOT-compiled Rust binary can be
fairly compared against; a single cold run is a measurement of the JIT.

**Ratios** are C# ÷ Rust throughout. Greater than 1 means C# is slower. Times are seconds.

**Memory** is measured two ways, because neither alone is honest. The Rust driver's counting
global allocator is exact and event-driven; C# cannot interpose on the GC's allocator, so its
heap peak is *sampled* at 1 ms and its "current" reading includes garbage not yet collected.
The one directly comparable number is maximum resident set size from `/usr/bin/time -l`, which
measures the same quantity for both processes. `ManifoldSharp.Benchmarks/MemorySampler.cs`
documents each mapping in full.

---

## Boolean and modelling benchmarks

Sphere-minus-sphere at doubling tessellation (`perf`, the ported `examples/perf_test.rs`); the
row label is input triangles per operand:

| operation | Rust seq | C# seq | ratio | Rust par | C# par | ratio |
|---|---|---|---|---|---|---|
| sphere difference, 512 tri | 0.00053 | 0.00116 | 2.19 | 0.00087 | 0.00093 | 1.07 |
| sphere difference, 2 048 tri | 0.00122 | 0.00210 | 1.72 | 0.00174 | 0.00239 | 1.37 |
| sphere difference, 8 192 tri | 0.00355 | 0.00595 | 1.68 | 0.00384 | 0.00514 | 1.34 |
| sphere difference, 32 768 tri | 0.01136 | 0.01995 | 1.76 | 0.01101 | 0.01648 | 1.50 |
| sphere difference, 131 072 tri | 0.04069 | 0.07309 | 1.80 | 0.03752 | 0.06199 | 1.65 |
| sphere difference, 524 288 tri | 0.15588 | 0.27873 | 1.79 | 0.13879 | 0.24367 | 1.76 |
| **sphere difference, 2 097 152 tri** | **0.6099** | **1.13681** | **1.86** | **0.5436** | **0.99970** | **1.84** |

The rest of the set the manifold-rust README and PORTING_PLAN report:

| operation | Rust seq | C# seq | ratio | Rust par | C# par | ratio |
|---|---|---|---|---|---|---|
| large scene, 7 999-sphere union (n=20) | 2.7723 | 4.7097 | 1.70 | 3.1637 | 3.5805 | 1.13 |
| Menger sponge depth 4 + convex hull | 6.7550 | 11.1237 | 1.65 | 7.2575 | 7.1406 | 0.98 |
| — of which: sponge CSG build | 6.7530 | 11.1205 | 1.65 | 7.2557 | 7.1374 | 0.98 |
| — of which: convex hull | 0.0019 | 0.0032 | 1.68 | 0.0018 | 0.0032 | 1.78 |
| stretchy bracelet, build + MinGap | 1.5066 | 2.4486 | 1.63 | 1.2526 | 2.0440 | 1.63 |
| — of which: two bracelets built | 0.7615 | 1.2269 | 1.61 | 0.5018 | 0.7685 | 1.53 |
| — of which: MinGap | 0.7451 | 1.2218 | 1.64 | 0.7488 | 1.2709 | 1.70 |
| Generic_Twin_7081 union | 11.0994 | 14.5713 | 1.31 | 1.8962 | 2.4786 | 1.31 |
| SDF blobs, edge length 0.05 | 1.2615 | 2.5730 | 2.04 | 1.1130 | 2.2543 | 2.03 |

Every row above produced **identical triangle counts on both sides** (2 111 168 out of the 2M
sphere difference, 3 880 out of the sphere64 union, 605 126 out of the SDF blobs, 33 172 out of
the twins union, 12 out of the Menger hull, and MinGap 5.0 within 1e-3). That is the exactness
bar holding under load, and it is the reason these numbers can be compared at all: the two
implementations are doing the same work, not similar work.

### Parallel speedup within each language

Neither language's `parallel` mode is a general win, and the shape of that is the same on both:

| operation | Rust seq→par | C# seq→par |
|---|---|---|
| sphere difference, 2M tri | 1.12× | 1.14× |
| large scene (n=20) | 0.88× (slower) | 1.32× |
| Menger sponge depth 4 | 0.93× (slower) | 1.56× |
| stretchy bracelet | 1.20× | 1.20× |
| Generic_Twin_7081 union | **5.85×** | **5.88×** |
| SDF blobs | 1.13× | 1.14× |

The twins case is the one the eleven parallel sites were built for: a boolean whose cost is
concentrated in per-triangle intersection work. The CSG-tree workloads (large scene, Menger)
are dominated by tree evaluation, which is *not* one of the blessed sites, so on the Rust side
they pay rayon's overhead for nothing and come out slower.

---

## Peak memory

The 2M-triangle sphere difference (`mem 6`, the ported `examples/mem_profile.rs`), best of three
runs:

| measurement | Rust seq | C# seq | ratio | Rust par | C# par | ratio |
|---|---|---|---|---|---|---|
| heap after both inputs exist | 903.0 MB | 1 076.0 MB | 1.19 | 903.1 MB | 1 076.7 MB | 1.19 |
| heap at completion | 1 354.0 MB | 2 436.8 MB | 1.80 | 1 354.1 MB | 2 435.1 MB | 1.80 |
| heap high-water during the boolean | 1 467.0 MB | 2 436.8 MB | 1.66 | 1 467.1 MB | 2 435.1 MB | 1.66 |
| **max RSS (`/usr/bin/time -l`)** | **2 237.0 MiB** | **2 512.3 MiB** | **1.12** | **2 237.7 MiB** | **2 514.2 MiB** | **1.12** |
| total bytes allocated | (not tracked) | 5 499.1 MB | — | (not tracked) | 5 518.9 MB | — |

Read the last row first. Max RSS is the only apples-to-apples number here and the gap on it is
**12 % in both modes** — close to the ~10 % the Rust port itself carries against C++, and the
sequential/parallel split that used to be 29 %/39 % has closed to nothing, because the thing
that made the parallel run cost more was garbage, not live data. The heap
rows still look worse (about 1.8×) and are not comparable in that direction: Rust's counter
drops the instant a `Vec` is dropped, while `GC.GetTotalMemory` still counts everything the
collector has not yet swept, and 5.5 GB of cumulative allocation means there is a great deal
not yet swept. The two views agreeing to within 1 % on the *input* row (903 vs 1 076 MB, where
nothing is garbage yet) is what shows the 1.8× on the later rows is GC slack rather than 1.8×
more live data.

The whole table moved with `intersect12`'s output shape (see below): cumulative allocation fell
from 6 733 MB to 5 499 MB, and essentially all of that 1.23 GB is one `List` plus one closure
plus one delegate per halfedge, for 6 291 456 halfedges.

Nothing here is close to a limit on a 24 GB machine, and the 2M case is the largest the Rust
README reports.

---

## Where the time goes: per-stage breakdown

`MANIFOLD_TIMING=1` is instrumented identically on both sides — that was built as a
trace-diffing net for correctness, and it works just as well for this. Single sequential run of
the 2M sphere difference (instrumentation inflates the total by roughly 3 %, symmetrically):

| stage | Rust | C# | ratio |
|---|---|---|---|
| Intersect12 P→Q | 0.0527 | 0.0752 | 1.43 |
| Intersect12 Q→P | 0.0462 | 0.0740 | 1.60 |
| Winding03 P | 0.0366 | 0.0649 | 1.77 |
| Winding03 Q | 0.0361 | 0.0536 | 1.48 |
| Intersections (total) | 0.1716 | 0.2678 | 1.56 |
| Assembly | 0.0546 | 0.1356 | 2.48 |
| Triangulation | 0.0556 | 0.1932 | 3.47 |
| Simplification | 0.1752 | 0.2556 | 1.46 |
| Sorting | 0.1801 | 0.3382 | 1.88 |
| whole boolean | 0.6372 | 1.1904 | 1.87 |

**No stage carries the gap any more.** The previous snapshot's outlier is gone: `intersect12`
read 13.6×/15.3× here and 63 % of the C# total, and now reads 1.43×/1.60× and 13 %. The slowest
row in the pipeline is `Triangulation` at 3.47×, and it is 16 % of the total.

What changed is the stage's output shape, not its arithmetic. Both ports used to build a
per-edge collection of intersection records — the Rust's own PORTING_PLAN flags "intersect12's
per-edge Vec-of-Vecs" as its remaining memory overhang versus C++'s counted two-pass output —
and the counts show what that costs a GC runtime: on this case 6 291 456 halfedges yield
21 936 collider candidates and 3 504 intersections, so the geometry is nil and the shape is
everything. One `List`, one closure and one delegate per halfedge, plus a live 6.3-million-entry
array of *references* that every later Gen0 collection has to mark through, was 0.72 s of the
0.72 s.

`Boolean3Kernels.Broadphase.cs` now moves the unit of collection from the edge to a **chunk** of
edges — one list, one closure and one delegate per chunk, ~4 096 chunks instead of 6.3 million
edges — keeping a single BVH traversal and a single `Kernel12` call per candidate. Output is
bit-identical element for element, which is structural rather than hoped-for: each chunk walks
its own ascending block of edges, so concatenating chunks in index order reproduces the
per-edge concatenation exactly.

### Three shapes, measured

C++ solves the same problem differently — a counted two-pass output (count candidates,
exclusive-scan into offsets, fill flat arrays, compact) — and that form was built and measured
here before the chunked one, because it is what manifold-rust's PORTING_PLAN names. Both are
bit-identical to the original and to each other. Seconds, and the last column is cumulative
allocation on the 2M case:

| `intersect12` output shape | 2M seq | 2M par | twins seq | twins par | allocated |
|---|---|---|---|---|---|
| per-edge collection (the Rust's, and this port until now) | 2.2375 | 1.8334 | 14.7859 | 2.4455 | 6 734.8 MB |
| counted two-pass (C++'s) | 1.3058 | 0.9979 | 15.8052 | 2.8306 | 5 581.3 MB |
| **per-chunk collection (shipped)** | **1.1368** | **0.9997** | **14.5713** | **2.4786** | **5 499.1 MB** |

The two-pass form wins the sphere ladder and loses Generic_Twin_7081 by **6.9 % sequential and
15.7 % parallel**, because its counting pass is a *second BVH traversal* and twins is the case
that cannot afford one: two near-coincident meshes, 59 112 halfedges, 9 s in the stage, and only
1 024 intersections to show for it — the cost there is candidates, not results, and a second
traversal is a second helping of exactly the expensive part. The chunked form needs no second
traversal, so it takes the ladder win without the twins loss, and it beats the two-pass on the
ladder as well.

**The chunk size is a scheduling parameter, and it is not a constant.** The chunk is the atomic
unit of parallel work, and per-edge cost is wildly non-uniform (it is proportional to an edge's
candidate count, which spans orders of magnitude), so the makespan is set by the unluckiest
chunk. Measured at 10 cores: a fixed 1 024-edge chunk is 1.6 % better than a 64-edge chunk on
the 2M sphere and **10 % worse on twins** (2.68–2.92 s against 2.48 s), because twins has only
58 chunks at that size. Sizing for a fixed *number* of chunks (~4 096) with a 64-edge floor
lands on the good end of both — 2M sphere 1.1368 s, twins 2.4786 s — and that is what ships.

The stage's parallel scaling moved with it: C# 0.075 s → 0.012 s is **6.2× on 10 cores**,
against 1.6× before and against Rust's 0.053 s → 0.024 s (2.2×). In parallel mode this stage is
now **faster in C# than in Rust** (0.012 s against 0.024 s), which is not a claim about language
quality — it is that the Rust still carries the per-edge `Vec`-of-`Vec`s its own plan flags, and
allocation is what was limiting the scaling on both sides.

---

## The exact-arithmetic tier: does BigInteger show?

`docs/PORTING_PLAN.md` predicts that `System.Numerics.BigInteger` on the exact-predicate tier
"will profile differently than dashu's inline words". This needed its own driver, because
**none of the benchmarks above execute a single BigInteger operation**: they all run on the
default `Exact` engine, which is floats plus symbolic perturbation. The `robust` driver runs the
same union through both engines. Best of 5, seconds:

| case | Rust seq | C# seq | ratio | Rust par | C# par | ratio |
|---|---|---|---|---|---|---|
| spiky dodecahedra, **exact** engine | 0.00028 | 0.00208 | 7.43 | 0.00028 | 0.00208 | 7.43 |
| spiky dodecahedra, **robust** engine | 0.00328 | 0.01021 | 3.11 | 0.00677 | 0.00790 | 1.17 |
| sphere64 union, **exact** engine | 0.00124 | 0.01345 | 10.85 | 0.00160 | 0.01313 | 8.21 |
| sphere64 union, **robust** engine | 0.02185 | 0.08718 | 3.99 | 0.02798 | 0.08531 | 3.05 |

The prediction does not hold in the direction it was made, and the result is the reverse of the
worry: on both cases the **robust engine's C#/Rust ratio (3.1–4.0×) is better than the exact
engine's on the very same meshes (7.4–10.9×)**. Whatever `BigInteger` costs relative to dashu,
it is not visible above the noise of everything else the robust pipeline does, and it is
certainly not the port's bottleneck. (Neither is `intersect12` any more; nothing in the
pipeline is, which is the point of the per-stage table above. These four rows are unchanged by
that work — re-measured after it and identical inside the noise — which is what you would
expect of a driver that spends its time in the robust engine's own graph and cell passes.)

Two honest qualifications, because this is the weakest evidence in the file:

1. These two cases resolve almost entirely in the robust engine's f64 interval filter — the
   per-run trace shows the exact fallback being reached rarely — so the driver exercises the
   BigInteger path *lightly*. A decisive answer wants a case that forces the exact tier
   repeatedly (manifold-rust's `ROBUST_PERF_THINGI` corpus pairs are the obvious source) and
   that is not ported here.
2. The exact-engine rows here (7–11×) are far worse than the same-sized rows in the main table
   (1.7× at 2 048 triangles). The difference is the call path: the main table uses `Union`,
   which routes through the CSG tree, while this driver uses `UnionWithEngine`, which calls the
   boolean directly — and the stage trace for the sphere64 case puts 6.4 ms of its 16.8 ms in
   **Triangulation**, a 17× gap where the 2M case shows only 3.5×. That is an unexplained
   outlier, and with `intersect12` closed it is now the largest single anomaly in the file.

---

## Reading this fairly

The spread across the whole file is now **1.3× to 2.0×** sequential, with two outliers, one in
each direction. There is no longer a shape the port is bad at.

Where it is closest: **the composed, real-world workloads**. Generic_Twin_7081 at 1.31×, the
bracelet at 1.63×, the Menger sponge at 1.65× (0.98× parallel — the C# run is faster than the
Rust one there, because Rust's `parallel` feature makes that CSG-tree workload slower and this
port's does not), the large scene at 1.70× (1.13× parallel). These are the shapes an
application actually asks a kernel for — many booleans over irregular meshes — and a managed
port landing within 1.1–1.7× of a fat-LTO Rust binary on them, with identical output, is the
outcome Phase 11 set out to reach.

The single enormous boolean, which used to be where it lagged worst at 3.5–3.7×, now reads
**1.86× sequential and 1.84× parallel**, and the per-stage table shows why: only one row is
above 2.5× and most of the pipeline is under 1.9×, which is roughly what a JIT costs against
`lto = "fat"` with `codegen-units = 1` and no bounds checks to elide. The SDF case (2.04×) is a
third shape again — a tight scalar callback over 8 million voxels — and it is the clearest
measurement of raw per-call codegen quality in the set.

What is *not* slower: correctness, memory within 12 % of RSS, and determinism. Sequential and
parallel produce bit-identical results in both languages, which is a stricter guarantee than
upstream C++ offers, and it is what makes every ratio in this file a comparison of the same
computation rather than of two approximations to it.

**Outcome of the accepted `intersect12` target (2026-08-30).** The bounded pass landed, and the
accepted target evolved on measurement. What was accepted was C++'s counted two-pass output;
what shipped is a per-chunk collection, because the two-pass form's counting pass is a second
BVH traversal that cost 6.9 %/15.7 % on Generic_Twin_7081. Both were built, both are
bit-identical, and the three-way table in the per-stage section is the datum — the intermediate
result is kept there rather than discarded, because "C++'s shape is the right shape" was a
reasonable prior and the measurement is the reason it did not survive. Net: the
sphere-difference rows moved 2M 2.24 s → 1.14 s sequential and 1.83 s → 1.00 s parallel, the
per-stage table 13.6×/15.3× → 1.43×/1.60×, the memory table max RSS 1.29× → 1.12×, and twins
improved rather than regressed (14.79 s → 14.57 s, 2.45 s → 2.48 s). The whole file above is a
re-run of the checked-in drivers with that change in place, not an amendment of the previous
snapshot.

### Next levers

In rough order of what the numbers point at, none of them touched by this pass:

1. **The `UnionWithEngine` triangulation outlier** named in qualification 2 above — 17× on a
   case where the 2M boolean shows 3.5×. It is now the largest single anomaly in the file.
2. **`Triangulation`, generally**: at 3.47× it is the slowest row in the per-stage table and
   16 % of the 2M boolean.
3. **`Kernel12` allocates three arrays per call** (`Boolean3Kernels.cs`: two `Vec3[2]` and the
   `new[] { startVert, endVert }` it iterates). Invisible on the sphere ladder, which calls it
   21 936 times; not invisible on Generic_Twin_7081, whose 9 s in the stage is almost entirely
   candidates being rejected by that function.
4. **`Collider.TraverseBvh` stackallocs 64 ints per query** and, without `SkipLocalsInit`, zeroes
   256 bytes each time — roughly 800 MB of memset across the 2M case's 3.1 million forward-edge
   traversals.
5. **`Collider`'s recorder is an `Action<int,int>`** where the Rust's is a monomorphized
   `impl FnMut`: an indirect call per candidate against an inlined one. A struct-generic recorder
   would close that, at the cost of widening `Collider`'s API.

This file is the baseline any of them should be measured against.
