"""FARO -- Feasibility-Aware Robot motion Optimization.

A from-scratch reimplementation of the FARO hierarchy of feasibility checks over
discrete contact-mode sequences (arXiv 2607.18362). The authors did not release
the method code, so everything here is rebuilt from the paper.

Subpackage map (which paper section each one implements):

    core/         shared vocabulary: ContactMode, ContactPatch, Interface, plans
    robots/       URDF loading + Pinocchio wrappers (numeric and symbolic)
    scene/        scene definition: objects, allowed contact interfaces
    constraints/  shared CasADi constraint blocks, Section II-C / Eqs. 7-13
    solvers/      backend-agnostic NLP interface; Ipopt now, acados later
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
