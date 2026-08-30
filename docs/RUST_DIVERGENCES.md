# Deliberate divergences from manifold-rust

Per `docs/PORTING_PLAN.md`, producing identical floating-point values to
`manifold-rust` is the default and "close enough" is a bug; the entries below
are the deliberate exceptions. Each one is a case where the Rust behaviour is
either unreproducible in a managed runtime, unspecified in Rust itself, or a
provable defect, with the evidence that justified the choice made instead.
Trace-diff debugging against the Rust must expect these.

The plan predicted this file would stay empty. The first two entries arrived
with `linalg.rs` in Phase 1, and neither is an accuracy change: one replaces a
Rust hash that is not reproducible even across two runs of the same Rust binary,
and the other pins a tie that Rust explicitly leaves unspecified. The third pins
an iteration order the Rust randomizes per process. The fourth is the first of a
different kind — a Rust defect against the C++ it ports, reproduced in the Rust
and repaired here.

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
`docs/PORTING_PLAN.md`'s dependency table every map in the port is documented
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

## 4. The centered cylinder rebuilds its collider (2026-08-30)

**What differs:** `Constructors.Cylinder`'s `center` branch ends with
`m.SortGeometry()`. The Rust ends at `calculate_bbox()`. Nothing about the
geometry changes — this is the first entry that repairs a *defect* rather than
pinning an unspecified behaviour, and the repair is provably bit-free.

**Where:** `ManifoldSharp/Constructors.cs`, `Cylinder`, the `if (center)` block,
whose comment carries the same rule at the code. The Rust is
`src/constructors.rs` lines 398-403. Reached by `Manifold.Cylinder`,
`Manifold.CylinderCentered`, and — through the recursive
`Cylinder(height, radiusHigh, 0.0, n, true)` its cone branch starts from — every
cone as well.

**Why:** the Rust's `cylinder` centers by editing `vert_pos` in place and then
refreshing only the bounding box. `extrude`, which produced that mesh, finished
with `sort_geometry`, and `sort_geometry` is where the face BVH is built
(`src/sort.rs:306`, `mesh.collider = Collider::new(face_box, face_morton)`). So
once the vertices move, the impl carries a collider describing the *pre-shift*
positions, a half-height out in Z. Every boolean against a centered cylinder then
queried that collider, missed every intersection against both cap fans, and
tripped the odd-crossing assert in `pair_up`.

**The C++ does not have this defect.** `cpp-reference/manifold` at v3.5.2
(`src/constructors.cpp:155-157`) centers with a transform, not an in-place edit:

```cpp
Manifold cylinder = Manifold::Extrude({circle}, height, 0, 0.0, vec2(scale));
if (center)
  cylinder = cylinder.Translate(vec3(0.0, 0.0, -height / 2.0)).AsOriginal();
return cylinder;
```

`Impl::Transform` maintains the collider (it maps the existing boxes for an
axis-aligned transform and refits otherwise — the C# port of that is
`ManifoldImpl.Shapes.cs:280`). manifold-rust replaced the `Translate` call with
the in-place loop and dropped the maintenance with it. **manifold-rust should be
fixed to match the C++, and this entry retired on the re-sync.** Its own cone
branch still goes through `transform`, which is why the two halves of one
function disagree.

`SortGeometry` is the repair rather than `Impl::Transform` because it is the
smaller claim: it recomputes the derived caches from the positions that are
already there, where re-centering through a transform would also recompute the
positions and land `+0.0` where the in-place subtraction leaves `-0.0`.
`SetEpsilon` is deliberately *not* re-run — epsilon must stay the value the
un-centered mesh carried, which is what `Impl::Transform` propagates (it scales
by the spectral norm, `1.0` for a translation) and therefore what the C++ path
produces too.

**Evidence:**

1. *The Rust reproduces it.* A scratchpad crate with a path dependency on
   manifold-rust 0.14.0, calling
   `Manifold::cube(Vec3::new(2,2,2), true).difference(&Manifold::cylinder_centered(4.0, 0.4, 0.4, 64, true))`:

   ```
   bore(center=true): tris=252 vol=2.007391033949401 status=NoError
   === cube - cylinder_centered(center=true) ===
   thread 'main' panicked at src/boolean_result.rs:317:5:
   Non-manifold edge! Not an even number of points. Got 1 points: starts=0, ends=1
   === cube - cone(radius_low=0, center=true) ===
   thread 'main' panicked at src/boolean_result.rs:317:5:
   Non-manifold edge! Not an even number of points. Got 1 points: starts=1, ends=0
   === cube - cylinder(uncentered).translate(-2) ===
   OK  tris=272 status=NoError vol=6.996304483025299
   ```

   The port's pre-repair failure was the same message with the same
   `starts`/`ends` counts on both cases, from the ported assert in
   `BooleanResult.PairUp`. The third line is the control: the same geometry
   reached through a transform has always worked, on both sides.

2. *The repair moves no bits.* Morton codes are computed relative to the bounding
   box, and a pure translation moves the box with the points, so re-sorting
   finds the order the mesh already has. A dump of raw f64 bit patterns for nine
   constructors — centered cylinder, centered cone, centered frustum, the
   8-segment case the ported `ConstructorsTests.CylinderCentered` uses, their
   un-centered counterparts, cube, sphere and revolve, 6 951 lines of positions,
   indices and run boundaries plus epsilon and tolerance — is byte-identical
   before and after.

3. *The repaired output matches the native library.* The port's boolean on a
   directly-constructed centered cylinder, against manifold-rust's own cdylib
   through the `ManifoldRust` binding fed the identical f64 arrays: 272
   triangles both sides, and all 2 448 f64 coordinates agree bit for bit
   (triangles resolved to position triples and canonicalised, because the oracle
   re-imports the cutter as a single run and the two exporters therefore emit the
   same triangles in a different run order). The centered frustum likewise: 112
   triangles, 1 008 coordinates, zero mismatches. The centered *cone* agrees on
   every position bit and on volume and surface area bit for bit, but
   triangulates one coplanar cap region into 25 different triangles — provenance,
   not arithmetic: hand the port the same round-tripped cone the oracle gets and
   the mismatch goes to zero.

4. *The root cause is pinned directly.* `ConstructorsTests` (C#-only region)
   queries every face's true box against the cached collider and requires each
   leaf to find itself. Before the repair, 124 of the centered cylinder's 252
   faces could not — exactly the two 62-triangle cap fans, which are flat and so
   cannot overlap a box a half-height away, while the side walls still could.

**Not fixed here, same defect class:** `Subdivision.SubdivideImpl` (Rust
`subdivide_impl`, `src/subdivision.rs:558`) also mutates geometry with only
`calculate_bbox` + `set_epsilon`, and a boolean against its output throws
`ArgumentOutOfRangeException` — the collider still has the pre-subdivision leaf
count, so its leaf indices address triangles that no longer exist. It is left
alone deliberately: unlike the translation above, subdivision appends vertices,
so re-sorting genuinely reorders them (47 of 78 position slots and 132 of 144
indices move at one level), and `SortGeometry` alone does not even repair it —
the boolean still throws. It needs its own investigation and its own decision
about diverging in observable output. Nothing in either port calls it outside
tests.
