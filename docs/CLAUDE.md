# docs/ — Working Documents Only

Every document in this folder is a **working document**: it describes work still **to do**, not
work that was done. History lives in git, not here.

Rules:

- A doc exists only while the work it plans is incomplete. **When the work completes, delete the
  doc** in the same change that finishes it. `PORTING_PLAN.md` follows this rule: phases already
  shipped live in its one-line Status, not as narrative; when the port is complete, the plan goes
  with it.
- Prune as you go: remove completed steps, stale findings, and superseded decisions instead of
  appending status updates. A reader should never have to skip "done" sections to find the open
  work.
- Do not add post-mortems, changelogs, or "how we got here" narrative. If a decision or caveat
  must outlive the doc, record it as a comment at the code it constrains (or in the root
  CLAUDE.md), then delete it from here.
- When checking whether a doc is still needed, verify against the code — a doc's own status line
  can be stale.

**Two permanent exceptions:**

- `RUST_DIVERGENCES.md` is a ledger, not a working document. The exactness bar in
  `PORTING_PLAN.md` mandates it as the single sanctioned record of deliberate behavioral
  divergence from manifold-rust, and it outlives the plan. It never gets pruned — entries are
  removed only if the divergence itself is removed from the code.
- `BENCHMARKS.md` is measurement output, not a plan. It is the durable record the README's
  performance section summarizes and the baseline a future regression is argued against, and
  it names the machine and date it snapshots. It is never appended to with status updates:
  it is replaced wholesale by re-running the checked-in `ManifoldSharp.Benchmarks` drivers.
