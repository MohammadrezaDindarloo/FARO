# Eq. 15 (KSO) — exact paper statement, and what our implementation must change

Transcribed from the paper (Section II-E). This is authoritative; where the code
disagrees, the code is wrong.

```
min_{q_0:K}   sum_{s=0}^{K} (q_s - q_nom)^T W (q_s - q_nom)

s.t.   q_0 = q_init,      q_K in Q_goal,

       for all s in {1, ..., K}:
           contact(q_s, c_{s-1} U c_s) <= 0      (7a), (7b)
           collision(q_s)              <= 0      (9)
           limits(q_s)                 <= 0      (13a)

       for all s in {0, ..., K-1}:
           for all a in I satisfying (16a) or (16b):
               contact(q_s, q_{s+1}, c_s) = 0    (8)
```

with `c_K := c_{K-1}`, "so that the terminal configuration can be handled using the
same indexing as the intermediate configurations".

Eq. 16, verbatim:

```
    b_s = b_{s+1} != empty                (16a)   persistent contact
    b_s != empty  AND  b_{s+1} = empty    (16b)   contact release
```

## Status (2026-09-16): implemented, on acados SQP

Every line of Eq. 15 above is built by `faro/kso/problem.py` and solved by
`faro/kso/acados_solver.py`, with the solver §II-G names. **An earlier version of
this section said acados had been tried and removed in favour of Ipopt. That is no
longer true.** acados came back once the GJK witnesses became parameters and the
Hessian was switched to EXACT + CONVEXIFY. The Ipopt KSO path is deleted. See
`kso_acados_design.md` for how the encoding problems below were resolved.

What remains open is listed in `paper_alignment.md` §4: how `q_init` is chosen, the
gap in the build-cache key, and code generation for every sequence.

## What an earlier implementation got wrong — ALL FIXED, kept as history

| # | Paper | What `faro/kso/problem.py` USED to do (all five now match the paper) | Severity |
|---|---|---|---|
| 1 | **K+1 configurations** `q_0..q_K` for K modes `c_0..c_{K-1}`, with `c_K := c_{K-1}` | one configuration per mode (K) | structural |
| 2 | **`contact(q_s, c_{s-1} U c_s)`** — the EDGE, the union of the previous and current mode | `contact(q_s, c_s)` — the current mode only | **the big one** |
| 3 | **`q_0 = q_init`** — the initial configuration is pinned, as an equality | nothing pinned; `anchor_initial` is optional and pins only OBJECT poses | structural |
| 4 | **`q_K in Q_goal`** — a terminal goal set | absent entirely | missing feature |
| 5 | contact / collision / limits for `s in {1..K}` — **not at s = 0** | applied at every step including 0 | minor, but changes row counts |

### Why #2 matters more than the rest

`c_{s-1} U c_s` is exactly the EDGE of Section II-D — the same union the edge filter E
tests. So the paper's KSO already embeds the edge check at every knot: each
configuration must satisfy the contacts of the mode it is leaving AND the mode it is
entering, simultaneously.

That resolves the disagreement `scripts/04_kso_demo.py` currently prints on the
`regrasp` demo, where our KSO says feasible and the edge filter says infeasible. Under
the paper's formulation the KSO would say **infeasible** too, because `q_1` would have
to put the left hand flat on `box_left` and `box_front` at once. Our version is a
strictly weaker filter than the paper's.

It also means a KSO pass implies all its edges pass, which changes how Alg. 1 should
order the filters.

## Resolved ambiguities

* **#27 (K vs K+1) — RESOLVED.** K+1 configurations, K modes, `c_K := c_{K-1}`.
* **#29 (initial state) — RESOLVED as a formulation question.** `q_0 = q_init` is in the paper. It is
  not optional, and it pins the whole configuration, not just object poses. **Still
  open as a data question:** where `q_init` comes from (see below).
* **#28 (the cost) — CONFIRMED.** `sum_s (q_s - q_nom)^T W (q_s - q_nom)`, exactly the
  generalization of Eq. 14 we defaulted to. The `smoothness` term is ours and is not
  in the paper; it should stay at 0.0 to be faithful.

## New open questions

* **`Q_goal`** — PARTLY ADDRESSED: `goal=` takes a partial mode and adds its contact rows at knot K. The paper gives it a name and no definition here. For box-placement it
  is presumably "box bottom in contact with the tabletop", i.e. a set expressed through
  contact rather than a fixed configuration. Needs to be a config-level object.
* **`q_init`** — the full scene state, including the robot. Our
  `Scene.nominal_configuration()` is a nominal POSTURE, not a task initial state; those
  are the same thing only by coincidence today.
