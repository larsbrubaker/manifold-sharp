# Deliberate divergences from manifold-rust

Per `CLAUDE.md`, producing identical floating-point values to
`manifold-rust` is the default and "close enough" is a bug; the entries below
are the deliberate exceptions. Each one is a case where the Rust behaviour is
either unreproducible in a managed runtime, unspecified in Rust itself, or a
provable defect, with the evidence that justified the choice made instead.
Trace-diff debugging against the Rust must expect these.

The plan predicted this file would stay empty. The first two entries arrived
with `linalg.rs` in Phase 1, and neither is an accuracy change: one replaces a
Rust hash that is not reproducible even across two runs of the same Rust binary,
and the other pins a tie that Rust explicitly leaves unspecified. The third pins
an iteration order the Rust randomizes per process. Those three are the
contract's "genuinely unspecified Rust behavior" clause — in each case there is
no single Rust result to match. The fourth is of a different kind: an *appended*
progress phase for a pipeline the Rust does not instrument at all. None of the
four changes a specified numerical value.

## 1. `Vec2`'s hash is the plain field-order bit hash (2026-08-29)

**What differs:** `Vec2.GetHashCode` hashes the bit patterns of `X` then `Y` in
field order, the same shape as `Vec3` and `Vec4`. The Rust `impl Hash for Vec2`
does something else: it hashes `x` into the caller's hasher, hashes `y` into a
private `DefaultHasher`, then writes
`RandomState::new().build_hasher().finish() ^ (h2.finish() << 1)` — mixing in a
freshly seeded `RandomState` whose seed is drawn per call.

**Where:** `ManifoldSharp/Linalg/Vec2.cs`, `GetHashCode`. The Rust is
`src/linalg.rs` lines 2194-2206, whose comment describes the intent as
"XOR with shift — mirrors C++ std::hash specialization". The neighbouring
`Vec3` and `Vec4` impls, commented "Simpler hash that is consistent: just hash
all fields sequentially", are ported verbatim; only `Vec2` diverges.

**Why:** the Rust impl cannot be ported, because it is not a function of its
input. `RandomState::new()` seeds from thread-local random state, so the same
`Vec2` hashes differently on successive calls within one process, and the `x`
contribution written into `state` is then followed by an unrelated random word.
Reproducing that would import nondeterminism into a port whose entire premise is
bit-identical output. Nothing is lost by declining: the Rust type derives
`PartialEq` but never implements `Eq`, so this `Hash` impl cannot key a
`HashMap` or `HashSet` in Rust at all, and no call site uses it. On the C# side
`GetHashCode` must agree with `Equals`, which is bit-based, and the field-order
bit hash is the impl that does.

**Evidence:** the hash value is unreachable from any output. Per
`CLAUDE.md`'s dependency table every map in the port is documented
probe-only (`rustc-hash` → plain `Dictionary`/`HashSet`), so no iteration order
derived from a hash reaches a mesh. Equality, which *is* observable, stays
exactly bit-based — see `LinalgTests`.

## 2. Signed-zero ties in `MinF64` / `MaxF64` are pinned to .NET semantics (2026-08-29)

**What differs:** when the two arguments compare equal but differ in sign of
zero, `LinalgFunctions.MinF64` returns `-0.0` and `MaxF64` returns `+0.0`,
regardless of argument order. Rust's `f64::min` / `f64::max` do not specify a
result in that case, and what they actually produce depends on the target.

**Where:** `ManifoldSharp/Linalg/LinalgFunctions.cs`, `MinF64` / `MaxF64`, which
every Rust `.min()` / `.max()` in the linalg port routes through — so this
reaches `Min`, `Max`, `Clamp`, `MinElem`, `MaxElem`, `RotationQuatMat` and
`QSlerp`.

**Why:** Rust documents `f64::min` as following IEEE-754 `minNum` except for
signaling NaNs, and states outright that "if the inputs compare equal (such as
for the case of +0.0 and -0.0), either input may be returned
non-deterministically". That makes the tie a property of the backend, not of the
source: arm64's `FMIN`/`FMAX` are sign-of-zero aware, while x86's
`minsd`/`maxsd` resolve ties positionally and ignore the sign of zero, so the
answer there also depends on which operand LLVM put in which register. There is
no single "the Rust result" to match. This port therefore pins the tie to
.NET's `Math.Min`/`Math.Max`, whose signed-zero behaviour *is* specified and is
sign-of-zero aware in both operand orders — which is the arm64 Rust build, the
one the oracle lane compares against.

The NaN behaviour is not part of this divergence and is ported exactly:
`Math.Min`/`Math.Max` propagate NaN whereas Rust returns the non-NaN operand, so
`MinF64`/`MaxF64` screen both arguments for NaN before delegating. That screen
is the reason the wrappers exist at all.

**Evidence:** `LinalgTests.MinMaxF64SignedZeroTiesAreSignAware` pins the tie and
`LinalgTests.MinMaxF64NaNLoses` pins the NaN rule. The first implementation of
these helpers used `a < b ? a : b`, which returned `+0.0` for `MinF64(-0.0, 0.0)`
and `-0.0` for `MaxF64(0.0, -0.0)` — measured against Rust on arm64 during
review and corrected. Should the port ever need to match an x86 Rust build
bit-for-bit at a tie, this is the single place to change.

## 3. `Slice` seeds its polygon loops from the smallest remaining triangle (2026-08-29)

**What differs:** `ManifoldImpl.Slice` holds the plane-straddling triangles in a
`SortedSet<int>` and seeds each output loop from `tris.Min` — the smallest
triangle index still in the set. The Rust holds them in a
`std::collections::HashSet<usize>` and seeds from `tris.iter().next()`, an
arbitrary member. Two consequences, and they are the whole of the divergence:

- the returned contours come out ordered by the smallest triangle index each
  one contains, ascending;
- each contour's point list starts at the crossing contributed by that seed
  triangle, so the contour is a *pinned rotation* of the Rust's cycle.

On a manifold mesh everything else is the Rust: the set of straddling triangles,
which triangles fall in which loop, the walk, the contour count, and every
coordinate bit. Off that path there is one more difference, and it is a
stop-vs-stop one: where the Rust widens an unpaired `-1` to a huge `usize` and
panics on the next halfedge access, this port throws
`InvalidOperationException` at the same step. C# integer division makes the
literal transcription unsafe rather than merely different — `-1 / 3 == 0` would
silently restart the walk at triangle 0 and can spin forever. `Manifold.Slice`
screens `IsSoup` before calling, so only a direct `ManifoldImpl.Slice` on a soup
impl reaches either behaviour. `Project` is untouched — `assemble_halfedges`
seeds from a `BTreeMap`, so it is already ordered — and is ported literally.

**Where:** `ManifoldSharp/FaceOp.Slice.cs`, `ManifoldImpl.Slice`, whose file
header carries the same rule at the code. Reached by `Manifold.Slice` in
`Manifold.Regions.cs`. The Rust is `src/face_op.rs` lines 594-674.

**Why:** the Rust's polygon order is not a function of its input. `HashSet`'s
default `RandomState` seeds from thread-local random state at construction, so
`iter().next()` returns a different member on each *run of the same binary* —
the same non-reproducibility that entry 1 declines to port, arriving here
through iteration order instead of a hash value. There is therefore no "the
Rust order" to match, which is exactly the exactness bar's "genuinely
unspecified Rust behavior" clause; a port that reproduced the shape of the Rust
(any `HashSet`-like probe order) would inherit output that changes run to run,
against the port's whole premise. `SortedSet` is the smallest change that makes
the choice a function of the input, and `Min` is the seed rule because it needs
no extra state — the set is already sorted.

Nothing specified is changed by the pin. `CrossSection` — the only consumer, via
`Manifold.Slice` — reports area, bounds and Clipper results, none of which
depend on contour order or on where a closed contour starts.

**Evidence:** a differential harness (scratchpad, `slice_` prefix) dumped
`ManifoldImpl::slice` / `::project` output for 12 meshes — cube, tetrahedron,
two spheres, a cone, a tilted cube, two- and four-body unions, a hollow cube, a
two-hole slab, a sphere difference — over 73 slice heights plus 12 projections:
85 cases, 121 contours, 1,910 points, dumped as raw f64 bit patterns.

- Five runs of the *Rust* differ from each other on 58-65 of the 73 slice cases
  and agree on all 85 after canonicalization (smallest cyclic rotation per
  contour, contours then sorted). For `two_cubes` at z=0.5 the five runs
  produced five different first vertices and both contour orders.
- The C# side is byte-identical across five runs.
- C# versus each of the five Rust runs: 0 contour-count mismatches, 0 canonical
  content mismatches — every C# contour is bit-for-bit a Rust contour — and the
  raw mismatches are confined to `SLICE`. All 12 `PROJECT` cases match the Rust
  *raw*, order and rotation included, in every run.

The ported tests `ManifoldBasicTests.CppManifoldSlice` /
`CppManifoldSliceEmptyObject` / `CppManifoldProject` keep the Rust's exact
`assert_eq!` expected values, because area does not observe the pin.

## 4. The progress module gains a tenth `Phase` and a `CompletePhase` emit (2026-08-30)

**What differs:** two additions to `progress.rs`'s port, both C#-only, both made
for the same caller.

1. `Phase` declares `Minkowski = 9`, appended after the Rust's nine, and
   `Phases.All` therefore has ten members instead of nine — so
   `Phases.FromId(9)` answers `Minkowski` where the Rust's `Phase::from_id(9)`
   answers `None`. The Rust's own nine ids and names are untouched, value for
   value and string for string.
2. `ProgressReporter.CompletePhase()` (plus the null-aware
   `Progress.CompletePhase`) emits the current phase at exactly 1.0,
   unconditionally. The Rust has no such method.

**Where:** `ManifoldSharp/Progress.cs` (the enum, `Phases.All`, `Phases.Name`,
`ProgressReporter.CompletePhase`) and `ManifoldSharp/Minkowski.cs`, the only
thing that reports the phase or calls the emit. The Rust is `src/progress.rs`'s
`#[repr(u32)] enum Phase`, whose ids are the FFI surface of
`manifold_rs_progress_phase_name`, and its `ProgressReporter`.

**Why the second one:** it repairs a defect the Rust shares. `Advance` emits only
when `done` crosses a step boundary, and `step` is `total / 100`, so every unit
after the last boundary is swallowed: at `total = 514` the final report is
510/514, and only when `total <= 100` — where `step` is 1 — does a finished phase
happen to land on 1.0. That is invisible in the boolean pipeline because the
consumer closes the window itself (agg-sharp's `BooleanProgressAdapter`
publishes a step boundary per pairwise operation), and it is not invisible for a
single-phase operation like Minkowski, where the bar simply stops short of full.
Reporting is write-only from the kernel's point of view, so an extra emit cannot
reach a computed value; the risk it does carry — a straggler `Advance` reporting
a *lower* fraction after the 1.0 — is closed by parking the throttle inside
`CompletePhase`. It is deliberately not called on a cancelled or failed path,
where a full bar would be a false claim.

**Why the first one:** the Rust threads neither a progress sink nor a cancel token into
`minkowski`, and `cancel.rs`'s own deviation list says so — "C++ also threads ctx
into the non-Boolean entry points (`FromMeshGL`, `Smooth`, `LevelSet`, `Hull`,
`Minkowski`, `Refine`). Here only the boolean / CSG pipeline is cancellable at
all". A Minkowski erosion is the slowest operation this library performs (one
convex hull and one boolean per triangle of the eroded solid — minutes on a few
thousand triangles), and agg-sharp's morphological entry points had to be
synchronous purely because the kernel offered nothing to report or interrupt
with. Adding progress there needs a phase to report *as*: `ProgressReporter`
takes a `Phase`, and reusing one of the boolean's — "assemble", say — would name
the wrong pipeline in a user-facing string and break the monotonic-phase
contract `ProgressTests` asserts, since the batch reductions inside Minkowski run
whole booleans of their own.

Appending is what the enum's own doc comment already prescribes for growth
("new phases are appended rather than inserted"), and it is the minimal shape:
every Rust id keeps its Rust meaning, so a diff of reported ids on the nine
shared phases is unchanged, and only a consumer that enumerates `Phase::ALL` or
probes id 9 can tell the difference. No FFI consumes this enum from C# — the
Rust's `#[repr(u32)]` matters to `manifold-rs-ffi`, which this port does not
build.

**Evidence:** the whole progress module is write-only — the reporter cannot feed
a value back into the kernel, and `Minkowski.Compute` with a null reporter runs
the code that ran before either addition existed
(`Progress.MaybeParMapCtProgress` with a null reporter *is* `Par.MaybeParMapCt`,
and with a null token *is* `Par.MaybeParMap`). `MinkowskiTests`'
`DefaultParametersAreBitIdenticalToAnInstrumentedRun` pins that: the same fixture
run with defaults and with a live token plus a reporter produces bit-identical
vertex positions and halfedges.
`ALargeRunEndsAtExactlyOneInBothParallelModes` is the regression test for the
swallowed tail — it uses a total past 100, where the throttle's step exceeds 1,
and asserts the last fraction is 1.0 sequentially and in parallel.
`ProgressTests.PhaseIdsRoundTrip` reads `Phases.All` dynamically and so covers
the tenth member without being edited.
