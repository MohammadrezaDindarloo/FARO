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

## Status: implemented exactly, on Ipopt

Every line of Eq. 15 above is built by `faro/kso/problem.py::build_problem` and
verified by `scripts/04_kso_demo.py`. The ONE deviation from the paper is the solver:
Section II-G specifies acados SQP for the KSO and we use Ipopt.

acados was built, integrated as a multi-phase OCP (one phase per knot) and removed
again. It is recorded here because the reasons are worth keeping:

* expressing Eq. 15 in acados needs a control `u` with `x_{s+1} = x_s + u_s`, because
  an acados path constraint sees only `(x_k, u_k)` and Eq. 8 couples `q_s` with
  `q_{s+1}`. Eq. 15 has no controls, so `u` is an artifact of the encoding.
* Gauss-Newton then has no curvature in the `u` directions, and the only fixes are to
  put a cost on `u` -- which Eq. 15 does not have, and which changes the problem -- or
  to regularize, which is the same thing by another name.
* with `SQP_WITH_FEASIBLE_QP` it iterates rather than failing at the first QP, but
  diverges from the cold start: 1237 iterations, base at z = 5.8 m, box 2.8 m away.
  Ipopt solves the identical problem in 15 s.

The rule applied was the user's: the equations are the paper's and only the solver may
differ. Adding a cost term to make acados converge would have inverted that.

## What an earlier implementation got wrong

| # | Paper | What `faro/kso/problem.py` currently does | Severity |
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
* **#29 (initial state) — RESOLVED.** `q_0 = q_init` is in the paper. Not optional,
  and it pins the whole configuration, not just object poses.
* **#28 (the cost) — CONFIRMED.** `sum_s (q_s - q_nom)^T W (q_s - q_nom)`, exactly the
  generalization of Eq. 14 we defaulted to. The `smoothness` term is ours and is not
  in the paper; it should stay at 0.0 to be faithful.

## New open questions

* **`Q_goal`** — the paper gives it a name and no definition here. For box-placement it
  is presumably "box bottom in contact with the tabletop", i.e. a set expressed through
  contact rather than a fixed configuration. Needs to be a config-level object.
* **`q_init`** — the full scene state, including the robot. Our
  `Scene.nominal_configuration()` is a nominal POSTURE, not a task initial state; those
  are the same thing only by coincidence today.
