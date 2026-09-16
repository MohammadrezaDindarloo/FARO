/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_multiphase_ocp_H_
#define ACADOS_SOLVER_multiphase_ocp_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"


#ifdef __cplusplus
extern "C" {
#endif














// ** capsule for solver data **
typedef struct multiphase_ocp_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;


    /* external functions phase 0 */
    // dynamics

    external_function_external_param_casadi *discr_dyn_phi_fun_0;
    external_function_external_param_casadi *discr_dyn_phi_fun_jac_ut_xt_0;

    // constraints


    // cost

    /* external functions phase 1 */
    // dynamics

    external_function_external_param_casadi *discr_dyn_phi_fun_1;
    external_function_external_param_casadi *discr_dyn_phi_fun_jac_ut_xt_1;

    // constraints
    external_function_external_param_casadi *nl_constr_h_fun_jac_1;
    external_function_external_param_casadi *nl_constr_h_fun_1;




    // cost







    external_function_external_param_casadi nl_constr_h_0_fun_jac;
    external_function_external_param_casadi nl_constr_h_0_fun;







    external_function_external_param_casadi nl_constr_h_e_fun_jac;
    external_function_external_param_casadi nl_constr_h_e_fun;



} multiphase_ocp_solver_capsule;

ACADOS_SYMBOL_EXPORT multiphase_ocp_solver_capsule * multiphase_ocp_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int multiphase_ocp_acados_free_capsule(multiphase_ocp_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int multiphase_ocp_acados_create(multiphase_ocp_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int multiphase_ocp_acados_reset(multiphase_ocp_solver_capsule* capsule, int reset_qp_solver_mem, int reset_numerical_values, int reset_solver_options, int reset_x_to_x0_bar);
ACADOS_SYMBOL_EXPORT int multiphase_ocp_acados_create_with_discretization(multiphase_ocp_solver_capsule* capsule, int N, double* new_time_steps);


ACADOS_SYMBOL_EXPORT int multiphase_ocp_acados_update_params(multiphase_ocp_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int multiphase_ocp_acados_update_params_sparse(multiphase_ocp_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);
ACADOS_SYMBOL_EXPORT int multiphase_ocp_acados_set_p_global_and_precompute_dependencies(multiphase_ocp_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT int multiphase_ocp_acados_solve(multiphase_ocp_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int multiphase_ocp_acados_setup_qp_matrices_and_factorize(multiphase_ocp_solver_capsule* capsule);


ACADOS_SYMBOL_EXPORT int multiphase_ocp_acados_free(multiphase_ocp_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void multiphase_ocp_acados_print_stats(multiphase_ocp_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int multiphase_ocp_acados_custom_update(multiphase_ocp_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *multiphase_ocp_acados_get_nlp_in(multiphase_ocp_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *multiphase_ocp_acados_get_nlp_out(multiphase_ocp_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *multiphase_ocp_acados_get_sens_out(multiphase_ocp_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *multiphase_ocp_acados_get_nlp_solver(multiphase_ocp_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *multiphase_ocp_acados_get_nlp_config(multiphase_ocp_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *multiphase_ocp_acados_get_nlp_opts(multiphase_ocp_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *multiphase_ocp_acados_get_nlp_dims(multiphase_ocp_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *multiphase_ocp_acados_get_nlp_plan(multiphase_ocp_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_multiphase_ocp_H_





