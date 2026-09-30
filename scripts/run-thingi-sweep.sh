#!/usr/bin/env bash
# Thingi10K sweep of the fast convex Dilate/Erode and the boolean (opt-in, not part of CI).
#
# Usage:
#   scripts/run-thingi-sweep.sh --root <Thingi10K meshes dir> [options]
#
#   --root DIR        the folder holding Thingi10K-meshes-{1,2,3}/meshes/<id>.stl.zip
#   --out FILE        CSV to write (default thingi-sweep.csv). Resumable: ids already in
#                     the file are skipped, so rerunning the same command continues a run.
#   --max-faces N     skip meshes with more STL faces than this (default 20000)
#   --timeout-s S     per-mesh budget in seconds (default 120); a mesh still running
#                     30 s past it is killed and recorded as "hung"
#   --jobs N          meshes in flight at once (default 1), each in its own process
#   --ids A,B,...     only these ids
#
# Each mesh is imported through ManifoldSharp.Tests' StlFixtures.ImportStlLikeDemo and
# classified: not_closed, not_manifold, self_intersecting, misoriented (a shell wound for
# the wrong side, e.g. a reversed outer shell or a cavity wound as material; skipped) or solid. Solids get
# five checks with a sphere tool of radius 0.02 x bbox diagonal (12 segments):
#   dilate        TryDilateByConvex: valid, re-imports, volume >= solid, genus sane
#   dilate_sweep  fast vs Minkowski sum: volume within 1e-6 relative, same genus
#   erode         TryErodeByConvex: valid, re-imports, volume <= solid
#   erode_sweep   fast vs Minkowski difference: volume within 1e-6 relative, same genus
#   incl_excl     A and A shifted by 0.3 x its size: vol A + vol B = vol(A u B) + vol(A n B)
# Verdicts: ok, FAIL, timeout, declined (the fast path passed, e.g. a convex part), n/a.
# A summary by class and verdict prints at the end.
#
# Full corpus (hours; run on a quiet machine):
#   scripts/run-thingi-sweep.sh --root ~/Development/rust-apps/Thingi10K/meshes --jobs 4
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
dotnet build "$repo/ManifoldSharp.ThingiSweep/ManifoldSharp.ThingiSweep.csproj" -c Release -v quiet -nologo
exec dotnet "$repo/ManifoldSharp.ThingiSweep/bin/Release/net10.0/ManifoldSharp.ThingiSweep.dll" "$@"
