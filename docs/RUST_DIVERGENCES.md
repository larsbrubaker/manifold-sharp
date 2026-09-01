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
progress phase for a pipeline the Rust does not instrument at all, plus a closing
emit that repairs a reporting defect the Rust shares. The fifth is of that same
additive kind and goes one step further — a whole algorithm the Rust does not
have, reachable only by name. None of the five changes a specified numerical
value, and none of them moves a bit produced by a ported function.

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
`ProgressReporter.CompletePhase`). `ManifoldSharp/Minkowski.cs` is the only thing
that *reports the appended phase*; the *emit* has six more callers, added
2026-08-30 — the robust engine's determinate phases, which had the same swallowed
tail. `IntersectionGraphBuild.cs` closes narrow phase, self intersections,
candidate points, registries and arrangements, and `Cells.cs` closes cells, each
at its success point only. The three indeterminate phases (`winding`, `assemble`
in `RobustFunctions.cs`, `exact boolean` in `Boolean3.Functions.cs`) deliberately
do not call it: with no total there is no bar to leave short, and `CompletePhase`
on one would only repeat the null fraction `BeginPhase` already emitted. The Rust
is `src/progress.rs`'s `#[repr(u32)] enum Phase`, whose ids are the FFI surface of
`manifold_rs_progress_phase_name`, and its `ProgressReporter`.

**Why the second one:** it repairs a defect the Rust shares. `Advance` emits only
when `done` crosses a step boundary, and `step` is `total / 100`, so every unit
after the last boundary is swallowed: at `total = 514` the final report is
510/514, and only when `total <= 100` — where `step` is 1 — does a finished phase
happen to land on 1.0. It bites every determinate phase, the boolean pipeline's
six included: on the union of two 96-segment spheres the narrow phase's last
report was 4600/4608, self intersections' and arrangements' 9200/9216, candidate
points' 312/314 and registries' 624/628, with cells (a variable-size advance per
edge group) stopping 0.000255 short. What made it look like a Minkowski-only
defect is that the boolean's *consumer* hides it — agg-sharp's
`BooleanProgressAdapter` keeps a high-water mark and closes each pairwise
operation's window itself, so a phase stopping at 0.9983 of its own bar is
invisible there, while a single-phase operation like Minkowski simply stops short
of full. Hiding is not fixing, so `CompletePhase` closes all seven.
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
`ProgressTests.EveryDeterminateRobustPhaseEndsAtExactlyOneInBothParallelModes` is
its counterpart for the six boolean phases, on a fixture sized so that every one
of them is both past 100 units and off a step boundary — which it also asserts
(each phase's second-to-last report is below 1.0), so a fixture that drifted into
landing on a boundary would fail rather than quietly stop proving anything.
Dropping any one of the six `CompletePhase` calls fails it, naming that phase.
`ProgressTests.PhaseIdsRoundTrip` reads `Phases.All` dynamically and so covers
the tenth member without being edited.

## 5. A closed-form convex erosion the Rust has no counterpart for (2026-09-01)

**What differs:** this port adds a second erosion algorithm.
`ManifoldSharp/ConvexErosion.cs` computes the Minkowski difference of a *convex*
solid in closed form — as the intersection of the solid's face halfspaces, each
plane pushed inward by the tool's support in that direction — where
`minkowski.rs` has exactly one erosion, the per-triangle sweep, and applies it to
convex and non-convex solids alike. `Manifold.TryConvexErosion` is the only way
in; `ManifoldSharp/Manifold.Shapes.cs` names it right below
`MinkowskiDifference`.

**What does not differ, and this is the point:** `Minkowski.Compute`,
`Minkowski.Sum`, `Minkowski.Difference`, `Manifold.MinkowskiSum` and
`Manifold.MinkowskiDifference` are untouched. Nothing is rerouted. Every ported
entry point still runs the ported algorithm and still produces the Rust's bits,
including for a convex solid, where the closed form would have been available and
is deliberately not taken. So this entry adds a capability rather than changing
an answer — the same shape as entry 4's appended phase, one level up.

**Where:** `ManifoldSharp/ConvexErosion.cs` (new file, not a port, its header
carries the derivation) and `Manifold.TryConvexErosion` in
`ManifoldSharp/Manifold.Shapes.cs`. The Rust is `src/minkowski.rs`, whose three
branches are transcribed in `ManifoldSharp/Minkowski.cs`; none of them is this.

**Why:** erosion is the only operation in this library with no cheap branch. The
Rust's cost model is explicit that convex ⊕ convex is one hull and milliseconds,
but *every* erosion — convex operands included — is one convex hull and one
boolean per triangle of the eroded solid. Measured on this port, in Release, on
an M-series mac: a 12-triangle box eroded by a 72-triangle ball is 15-20 ms, a
436-triangle cylinder is 184 ms, a 288-triangle sphere is 400 ms and a
4024-triangle sphere is 2.7 s. The closed form answers the same four in 0.01,
0.07, 1.0 and 29 ms — 100x to 2500x, and the gap widens with triangle count
because the sweep is superlinear in it and the closed form is two convex hulls
whatever the count.

That matters to one caller in particular. agg-sharp's `RoundAllEdgesObject3D` is
a morphological *opening* — erode by r, dilate by r — which is the uniform
rolling-ball fillet by construction. The dilation was always milliseconds and the
erosion was always the whole bill; on the box shape class the opening drops from
~16 ms to ~0.12 ms, about 130x, and on a part with a few thousand triangles it is
the difference between a progress bar and a cancel button and neither being
needed.

The mathematics is exact rather than approximate, which is why this is worth
having as more than a heuristic. A convex A *is* `{ x : n_i·x <= d_i }` over its
faces; erosion distributes over an intersection of halfspaces; and eroding one
halfspace by B slides its plane by B's support. So
`A ⊖ B = { x : n_i·x <= d_i - h_B(-n_i) }` with no tessellation of the offset
surface and no boolean anywhere. The sign is `h_B(-n_i)` because `Minkowski.cs`'s
erosion is `A \ (∂A ⊕ B)` — it sweeps B, not -B — so what it computes is
`{ x : x - B ⊆ A }`, and the closed form is written to answer the same question.
For the centred ball every real caller erodes with, `B = -B` and the distinction
is invisible; it is matched anyway so an asymmetric tool cannot silently answer
something else.

Two implementation choices are worth recording. The halfspace intersection is
built through the **polar dual** — with a point p strictly inside the result, a
facet of `hull{ n_i / c_i }` names the three planes meeting at a vertex — because
the obvious alternative, clipping plane by plane, is one boolean per face and
costs more than the sweep it replaces for anything past a prism. And the dual is
used *only* to enumerate those triples: each vertex is then solved in the primal
from the original `n` and `d`, so a box's corners land exactly on their planes
instead of round-tripping through `1/c`. That is what makes the 20-cube eroded by
a unit ball come out as volume 5832.0 exactly.

**Why it is a `Try` and not a reroute:** the closed form has a real applicability
gate, and a gate is a thing a caller should see rather than a thing hidden inside
an answer. It declines — returns false, and the caller runs the sweep — on: a
non-convex solid (the premise), a non-convex tool (not needed by the geometry,
needed by the promise: the sweep swaps its operands in that case and computes
something else), a tool that does not contain the origin, a vertex centroid that
is not strictly inside the eroded body, a degenerate plane triple, and any output
vertex that fails verification against every constraint it was built from.
Declining is always safe, because the sweep is the specification.

The tool-misses-the-origin decline is the interesting one and is a defect
*found* by this work rather than introduced by it. `A \ (∂A ⊕ B)` drops a point
of A exactly when its swept copy `x - B` *meets* the boundary. With the origin in
B that is the erosion, because `x` is itself in `x - B`: a swept copy that leaves
A has to cross out of it. With the origin outside B the swept copy can sail clear
over the boundary and land wholly outside, and the sweep keeps that point too — so
its answer stops being an erosion of anything. A 20-cube swept by a unit ball
centred at (2,0,0) comes back — in Rust and in this port alike — as a 32-triangle
solid of volume 5908 whose bounding box is still the whole cube. Being right and
being different is still a behaviour change, so the fast path stands down there
and the divergence is written here instead of shipped.

**Origin containment is the gate, and nothing weaker is.** The first version of
this tested the *pushes* instead — decline if `h_B(-n_i) < 0` for one of A's face
normals — on the reasoning that a negative push is a tool sticking out past that
face. That is a necessary condition and not a sufficient one, because it only ever
samples the directions A happens to have faces in. Review found the case that
walks through it: the same unit ball centred at (0.8,0.8,0), whose centre is 1.13
from the origin and which therefore reaches past none of a cube's six face planes.
It passed, the closed form answered the true erosion 5832, the sweep answered
5832.547, and the two disagreed in silence — the one outcome this whole design is
built to make impossible. The gate now tests the actual condition, origin ∈ B,
against the *tool's* own face planes: exact, since the tool is already known
convex, so a convex tool contains the origin precisely when every outward face
plane has a non-negative offset. It also subsumes the push test, since `0 ∈ B`
gives `h_B(-n) ≥ 0` in every direction, so that check is gone rather than kept
alongside.

**Where it is not exact.** The arithmetic is exact and beats the sweep at it — the
20-cube's 5832.0 has no rounding in it at all — and agreement holds at 1e-15
relative up to about a thousand faces. On a denser solid it does not, and the
cause is the dual hull rather than the arithmetic: QuickHull discards points
within its relative epsilon of an existing facet, and on a finely tessellated
solid many dual points sit that close, so a few halfspaces are dropped as if
redundant when they are very slightly not. Measured: a 2048-triangle sphere eroded
by a unit ball gives 4016 triangles against the sweep's 4024 and a relative volume
difference of 4.9e-7; a 1152-triangle sphere already differs in triangle count
(2544 against 2550) while the volumes still agree to 4e-15, so the triangulation
parts company first and the volume follows. Both are far inside the error the
tessellated ball itself introduces, so the fast path is still the right answer for
a fillet — but it is not the sweep's answer, and a caller that needs a dense
convex erosion right to the last bit wants the sweep.

**Evidence:** `ManifoldSharp.Tests/ConvexErosionTests.cs`, 22 cases, against two
oracles. The independent one enumerates every triple of offset face planes,
keeps the candidates where the tool actually fits — the *definition* of erosion,
not the plane-offset shortcut the production path is built on — and hulls the
survivors; it shares no code with the dual-hull construction it checks and is
cubic in the face count, so it could only ever be a test. Cube, tetrahedron and
icosahedron agree with it on volume and surface area to a relative 1e-6. The
second oracle is `Minkowski.Difference` itself, on the erosion and again on the
full opening (`erode` then `dilate`), which is the only thing Round All Edges
actually shows a user.

`TheGeneralDifferenceStillRunsTheSweep` is the regression test for the "nothing
is rerouted" claim above, and it is pinned on a shape where the two paths are
observably different meshes of the same solid — a 40x20x10 box, 36 triangles from
the sweep and 12 from the closed form, equal volumes — so a future edit that
quietly routed `MinkowskiDifference` through the fast path fails rather than
passing because the answers happen to match.
`AToolThatMissesTheOriginIsDeclined` runs both tool positions from the gate
paragraph above, the (0.8,0.8,0) regression included, and asserts the sweep's own
answer at each is unchanged. Neutering the gate fails both cases.

`AnAsymmetricToolErodesTheWayTheSweepDoes` is what makes the `h_B(-n)` sign a
measured claim rather than a written one. Every other fixture here uses a centred
ball, where `B = -B` and the reflection is invisible — flipping the sign failed
*nothing*, so the file header, the brute-force oracle's comment and this entry
were all asserting something no test could see. The tool reaches 1.0 in +x against
0.3 in -x and 0.8 in +z against 0.4 in -z, and the eroded 40-cube's bounding box
is asserted bit-exactly against the sweep's: X [-19, 19.7], Z [-19.2, 19.6]. With
the sign flipped those mirror and the test fails; it was run flipped to confirm
that, and it is the only test in the file that catches it.

`ADenseSolidAgreesWithTheSweepOnlyToATolerance` pins the inexactness paragraph on
the 2048-triangle sphere, at 5e-6 with an order of magnitude of headroom over the
measured 4.9e-7, and also asserts the difference is *above* 1e-9 — so if the dual
hull ever stopped dropping those points, the test fails and says that the header
and this entry now describe a problem that no longer exists.

`ProgressIsReportedOnSuccessAndNotOnADeclineBeforeAnyWork` pins the reporting
contract: the appended `Phase.Minkowski` of entry 4, one unit per face of the
solid for the support pass plus one for everything after it, a bar that only rises
and ends at exactly 1.0, and silence on the decline that actually happens (a
non-convex solid, rejected before the phase opens) so the sweep's own `BeginPhase`
is the first thing a caller sees. The rare numeric declines *after* the phase has
opened leave it short on purpose — that is the price of having a seam mid-run at
all, and `ACancelRaisedDuringTheSupportPassIsObservedThere` is what it buys: a
token tripped from the second progress report is observed inside the support pass,
within 30 of the ~103 reports a completed pass emits.
`APreCancelledTokenAnswersCancelledRatherThanDeclining` pins the other end, that a
cancelled token comes back as *applied* — false would send a cancelled caller off
to run the minutes-long path it was cancelled out of — and the closing
`Cancel.IsCancelled` after the final hull is the twin of the one
`Minkowski.Compute` makes after its last `BatchBoolean`, so `CancelToken.cs`'s
invariant ("a cancelled token can never produce a `NoError` result") holds on this
path too.
