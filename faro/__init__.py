"""FARO -- Feasibility-Aware Robot motion Optimization.

A from-scratch reimplementation of the FARO hierarchy of feasibility checks over
discrete contact-mode sequences (arXiv 2607.18362). The authors did not release
the method code, so everything here is rebuilt from the paper.

Subpackage map (which paper section each one implements):

    core/         shared vocabulary: ContactMode, ContactPatch, Interface, plans
    robots/       URDF loading + Pinocchio wrappers (numeric and symbolic)
    scene/        scene definition: objects, allowed contact interfaces
    constraints/  shared CasADi constraint blocks, Section II-C / Eqs. 7-13
    solvers/      backend-agnostic NLP interface; Ipopt
    mode_edge/    single-mode and edge feasibility IK-NLP, Eq. 14
    kso/          kinematic sequence optimization, Eq. 15
    to/           full dynamic trajectory optimization, Eq. 17
    search/       feasibility-guided UCT tree search, Algorithm 1
    llm/          LLM contact-plan sampling
    viz/          Meshcat + MuJoCo visualization helpers
    utils/        config loading, paths, logging

Build order is bottom-up; see README.md for milestone status.
"""

__version__ = "0.0.1"

# BEFORE ANY HEAVY IMPORT. Eq. 14's answer is a solver status, so a threaded BLAS's
# non-deterministic reductions can flip a mode's verdict between identical calls --
# measured, and it did. `OMP_NUM_THREADS` is only read when BLAS loads, so this has
# to happen here and nowhere later. See faro/utils/determinism.py.
from faro.utils.determinism import ensure_single_threaded_blas as _pin_blas  # noqa: E402

_pin_blas()
