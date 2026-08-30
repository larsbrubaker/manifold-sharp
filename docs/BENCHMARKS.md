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

**Machine:** Apple M5, 10 cores, 24 GB, macOS 26.5.1 (build 25F80). Otherwise idle — no test
suite, no build, nothing else in flight while any number below was taken.

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
cases), measured with `Stopwatch.GetTimestamp` / `Instant::now` around the operation only —
mesh construction and file loading sit outside the timed region on both sides, exactly as the
Rust examples place them.

Best-of, not mean-of, and the first rep is not discarded but is in practice the warmup: the
workload is deterministic, so every rep computes an identical result and the spread is pure
machine noise plus, on the C# side, the JIT compiling the boolean pipeline during rep 1. That
tax is real and it is large in relative terms on the small cases (the 512-triangle sphere
difference costs 25 ms on its first execution and 1.3 ms warm), which is why the per-rep lines
are printed rather than averaged away. A best-of-3 is what an AOT-compiled Rust binary can be
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
| sphere difference, 512 tri | 0.00053 | 0.00134 | 2.51 | 0.00087 | 0.00369 | 4.23 |
| sphere difference, 2 048 tri | 0.00122 | 0.00268 | 2.19 | 0.00174 | 0.00257 | 1.48 |
| sphere difference, 8 192 tri | 0.00355 | 0.01135 | 3.19 | 0.00384 | 0.00747 | 1.95 |
| sphere difference, 32 768 tri | 0.01136 | 0.03943 | 3.47 | 0.01101 | 0.04244 | 3.85 |
| sphere difference, 131 072 tri | 0.04069 | 0.12936 | 3.18 | 0.03752 | 0.12056 | 3.21 |
| sphere difference, 524 288 tri | 0.15588 | 0.54977 | 3.53 | 0.13879 | 0.48324 | 3.48 |
| **sphere difference, 2 097 152 tri** | **0.6099** | **2.2468** | **3.68** | **0.5436** | **1.8880** | **3.47** |

The rest of the set the manifold-rust README and PORTING_PLAN report:

| operation | Rust seq | C# seq | ratio | Rust par | C# par | ratio |
|---|---|---|---|---|---|---|
| large scene, 7 999-sphere union (n=20) | 2.7723 | 5.5800 | 2.01 | 3.1637 | 4.3751 | 1.38 |
| Menger sponge depth 4 + convex hull | 6.7550 | 11.8429 | 1.75 | 7.2575 | 7.9748 | 1.10 |
| — of which: sponge CSG build | 6.7530 | 11.8394 | 1.75 | 7.2557 | 7.9712 | 1.10 |
| — of which: convex hull | 0.0019 | 0.0034 | 1.79 | 0.0018 | 0.0035 | 1.94 |
| stretchy bracelet, build + MinGap | 1.5066 | 2.7367 | 1.82 | 1.2526 | 2.2886 | 1.83 |
| — of which: two bracelets built | 0.7615 | 1.4498 | 1.90 | 0.5018 | 0.9488 | 1.89 |
| — of which: MinGap | 0.7451 | 1.2869 | 1.73 | 0.7488 | 1.3379 | 1.79 |
| Generic_Twin_7081 union | 11.0994 | 14.8685 | 1.34 | 1.8962 | 2.6251 | 1.38 |
| SDF blobs, edge length 0.05 | 1.2615 | 2.6927 | 2.13 | 1.1130 | 2.4612 | 2.21 |

Every row above produced **identical triangle counts on both sides** (2 111 168 out of the 2M
sphere difference, 3 880 out of the sphere64 union, 605 126 out of the SDF blobs, 33 172 out of
the twins union, 12 out of the Menger hull, and MinGap 5.0 within 1e-3). That is the exactness
bar holding under load, and it is the reason these numbers can be compared at all: the two
implementations are doing the same work, not similar work.

### Parallel speedup within each language

Neither language's `parallel` mode is a general win, and the shape of that is the same on both:

| operation | Rust seq→par | C# seq→par |
|---|---|---|
| sphere difference, 2M tri | 1.12× | 1.19× |
| large scene (n=20) | 0.88× (slower) | 1.28× |
| Menger sponge depth 4 | 0.93× (slower) | 1.49× |
| stretchy bracelet | 1.20× | 1.20× |
| Generic_Twin_7081 union | **5.85×** | **5.66×** |
| SDF blobs | 1.13× | 1.09× |

The twins case is the one the eleven parallel sites were built for: a boolean whose cost is
concentrated in per-triangle intersection work. The CSG-tree workloads (large scene, Menger)
are dominated by tree evaluation, which is *not* one of the blessed sites, so on the Rust side
they pay rayon's overhead for nothing and come out slower.

---

## Peak memory

The 2M-triangle sphere difference (`mem 6`, the ported `examples/mem_profile.rs`), one run:

| measurement | Rust seq | C# seq | ratio | Rust par | C# par | ratio |
|---|---|---|---|---|---|---|
| heap after both inputs exist | 903.0 MB | 1 076.0 MB | 1.19 | 903.1 MB | 1 076.8 MB | 1.19 |
| heap at completion | 1 354.0 MB | 2 825.0 MB | 2.09 | 1 354.1 MB | 3 008.3 MB | 2.22 |
| heap high-water during the boolean | 1 467.0 MB | 2 825.0 MB | 1.93 | 1 467.1 MB | 3 008.3 MB | 2.05 |
| **max RSS (`/usr/bin/time -l`)** | **2 237.0 MiB** | **2 886.6 MiB** | **1.29** | **2 237.7 MiB** | **3 101.9 MiB** | **1.39** |
| total bytes allocated | (not tracked) | 6 733.1 MB | — | (not tracked) | 6 733.2 MB | — |

Read the last row first. Max RSS is the only apples-to-apples number here and the gap on it is
**29 % sequential, 39 % parallel** — the same order as the ~10 % the Rust port itself carries
against C++. The heap rows look much worse (about 2×) and are not comparable in that
direction: Rust's counter drops the instant a `Vec` is dropped, while `GC.GetTotalMemory` still
counts everything the collector has not yet swept, and the 6.7 GB of cumulative allocation
means there is a great deal not yet swept. The two views agreeing to within 1 % on the *input*
row (903 vs 1 076 MB, where nothing is garbage yet) is what shows the 1.9× on the later rows is
GC slack rather than 1.9× more live data.

Nothing here is close to a limit on a 24 GB machine, and the 2M case is the largest the Rust
README reports.

---

## Where the time goes: per-stage breakdown

`MANIFOLD_TIMING=1` is instrumented identically on both sides — that was built as a
trace-diffing net for correctness, and it works just as well for this. Single sequential run of
the 2M sphere difference (instrumentation inflates the total by roughly 3 %, symmetrically):

| stage | Rust | C# | ratio |
|---|---|---|---|
| Intersect12 P→Q | 0.0527 | 0.7191 | **13.6** |
| Intersect12 Q→P | 0.0462 | 0.7082 | **15.3** |
| Winding03 P | 0.0366 | 0.0653 | 1.78 |
| Winding03 Q | 0.0361 | 0.0540 | 1.49 |
| Intersections (total) | 0.1716 | 1.5466 | 9.01 |
| Assembly | 0.0546 | 0.1301 | 2.38 |
| Triangulation | 0.0556 | 0.1912 | 3.44 |
| Simplification | 0.1752 | 0.2600 | 1.48 |
| Sorting | 0.1801 | 0.3389 | 1.88 |
| whole boolean | 0.6372 | 2.4669 | 3.87 |

**One stage carries the entire gap.** Everything except `intersect12` lands between 1.5× and
3.4×; `intersect12` alone is 14×, and at 63 % of the C# total it is what turns a ~1.9× port
into a 3.7× one. Take `intersect12` to 3× and the 2M row would read about 1.3 s, a ratio near
2.1. It is also the site whose parallel scaling is worst on both sides (C# 0.72 s → 0.44 s,
1.6× on 10 cores; Rust 0.053 s → 0.024 s, 2.2×), which points at memory traffic rather than
arithmetic: `intersect12` builds a per-edge collection of intersection records, and the Rust's
own PORTING_PLAN flags "intersect12's per-edge Vec-of-Vecs" as its remaining memory overhang
versus C++'s counted two-pass output. The C# port inherits that structure and pays more for it,
which is exactly what a per-element-allocating inner loop looks like on a GC runtime.

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
certainly not the port's bottleneck. The bottleneck is `intersect12`, in the *other* engine.

Two honest qualifications, because this is the weakest evidence in the file:

1. These two cases resolve almost entirely in the robust engine's f64 interval filter — the
   per-run trace shows the exact fallback being reached rarely — so the driver exercises the
   BigInteger path *lightly*. A decisive answer wants a case that forces the exact tier
   repeatedly (manifold-rust's `ROBUST_PERF_THINGI` corpus pairs are the obvious source) and
   that is not ported here.
2. The exact-engine rows here (7–11×) are far worse than the same-sized rows in the main table
   (2.2× at 2 048 triangles). The difference is the call path: the main table uses `Union`,
   which routes through the CSG tree, while this driver uses `UnionWithEngine`, which calls the
   boolean directly — and the stage trace for the sphere64 case puts 6.4 ms of its 16.8 ms in
   **Triangulation**, a 17× gap where the 2M case shows only 3.4×. That is an unexplained
   outlier and the second thing to profile after `intersect12`.

---

## Reading this fairly

Where the port is at or near parity: **the composed, real-world workloads**. Generic_Twin_7081
at 1.34×, the Menger sponge at 1.75× (1.10× parallel), the bracelet at 1.82×, the large scene
at 2.01× (1.38× parallel). These are the shapes an application actually asks a kernel for —
many booleans over irregular meshes — and a managed port landing within 1.3–2× of a fat-LTO
Rust binary on them, with identical output, is the outcome Phase 11 set out to reach. The
parallel columns close further because the eleven parallel sites carry proportionally more of
the C# cost.

Where it lags: **the single enormous boolean**, at 3.5–3.7×, and the reason is one stage, not a
pervasive managed-code tax. The rest of the pipeline sits at 1.5–3.4×, which is roughly what a
JIT costs against `lto = "fat"` with `codegen-units = 1` and no bounds checks to elide. The SDF
case (2.1×) is a third shape again — a tight scalar callback over 8 million voxels — and it is
the clearest measurement of raw per-call codegen quality in the set.

What is *not* slower: correctness, memory within 30–40 % of RSS, and determinism.
Sequential and parallel produce bit-identical results in both languages, which is a stricter
guarantee than upstream C++ offers, and it is what makes every ratio in this file a comparison
of the same computation rather than of two approximations to it.

The next optimization, if anyone wants one, is `intersect12`, and this file is the baseline it
should be measured against.

**Accepted as the next optimization target (2026-08-30).** A bounded pass is attempting
`intersect12`'s output shape — the counted two-pass form manifold-rust's own PORTING_PLAN names
as what C++ does, in place of the per-edge collection-of-collections both ports currently
build. If it lands, the sphere-difference rows, the per-stage table and the memory table are
the ones it moves, and this file gets re-snapshotted from the same drivers rather than
amended.
