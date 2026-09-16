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













// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h> // memcpy

// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific
#include "kso_p0_4ea03e70c6_model/kso_p0_4ea03e70c6_model.h"
#include "kso_p0_4ea03e70c6_constraints/kso_p0_4ea03e70c6_constraints.h"
#include "kso_p1_4ea03e70c6_model/kso_p1_4ea03e70c6_model.h"
#include "kso_p1_4ea03e70c6_constraints/kso_p1_4ea03e70c6_constraints.h"





#include "acados_solver_multiphase_ocp.h"


#define MULTIPHASE_OCP_N      2
#define NP_0     6201
#define NP_1     6201









multiphase_ocp_solver_capsule * multiphase_ocp_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(multiphase_ocp_solver_capsule));
    multiphase_ocp_solver_capsule *capsule = (multiphase_ocp_solver_capsule *) capsule_mem;

    return capsule;
}


int multiphase_ocp_acados_free_capsule(multiphase_ocp_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int multiphase_ocp_acados_create(multiphase_ocp_solver_capsule* capsule)
{
    int N_shooting_intervals = MULTIPHASE_OCP_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return multiphase_ocp_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


/**
 * Internal function for multiphase_ocp_acados_create: step 1
 */
void multiphase_ocp_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->relaxed_ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;

    nlp_solver_plan->regularization = NO_REGULARIZE;
    nlp_solver_plan->globalization = MERIT_BACKTRACKING;

    nlp_solver_plan->nlp_cost[0] = LINEAR_LS;
    nlp_solver_plan->nlp_constraints[0] = BGH;
    for (int i = 1; i < 1; i++)
    {
        nlp_solver_plan->nlp_cost[i] = LINEAR_LS;
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    for (int i = 0; i < 1; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = DISCRETE_MODEL;
        // discrete dynamics does not need sim solver option, this field is ignored
        nlp_solver_plan->sim_solver_plan[i].sim_solver = INVALID_SIM_SOLVER;
    }
    for (int i = 1; i < 2; i++)
    {
        nlp_solver_plan->nlp_cost[i] = LINEAR_LS;
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    for (int i = 1; i < 2; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = DISCRETE_MODEL;
        // discrete dynamics does not need sim solver option, this field is ignored
        nlp_solver_plan->sim_solver_plan[i].sim_solver = INVALID_SIM_SOLVER;
    }

    nlp_solver_plan->nlp_cost[N] = LINEAR_LS;
    nlp_solver_plan->nlp_constraints[N] = BGH;
}



/**
 * Internal function for multiphase_ocp_acados_create: step 2
 */
ocp_nlp_dims* multiphase_ocp_acados_create_setup_dimensions(multiphase_ocp_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    int i;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np    = intNp1mem + (N+1)*17;
    for (i = 0; i < 1; i++)
    {
        // common
        nx[i] = 43;
        nu[i] = 43;
        nz[i] = 0;
        ns[i] = 0;
        np[i] = 6201;
        // cost
        ny[i] = 86;
        // constraints
        nbu[i] = 0;
        nbx[i] = 0;
        ng[i] = 0;
        nh[i] = 0;
        nphi[i] = 0;
        nr[i] = 0;
        // slacks
        nsbu[i] = 0;
        nsbx[i] = 0;
        nsg[i] = 0;
        nsh[i] = 0;
        nsphi[i] = 0;
        nbxe[i] = 0;
    }
    for (i = 1; i < 2; i++)
    {
        // common
        nx[i] = 43;
        nu[i] = 43;
        nz[i] = 0;
        ns[i] = 0;
        np[i] = 6201;
        // cost
        ny[i] = 86;
        // constraints
        nbu[i] = 0;
        nbx[i] = 0;
        ng[i] = 0;
        nh[i] = 869;
        nphi[i] = 0;
        nr[i] = 0;
        // slacks
        nsbu[i] = 0;
        nsbx[i] = 0;
        nsg[i] = 0;
        nsh[i] = 0;
        nsphi[i] = 0;
        nbxe[i] = 0;
    }

    /* initial node*/
    i = 0;
    // common
    nx[i] = 43;
    nu[i] = 43;
    nz[i] = 0;
    ns[i] = 0;
    np[i] = 6201;
    // cost
    ny[i] = 86;
    // constraints
    nbu[i] = 0;
    nbx[i] = 43;
    nbxe[i] = 43;
    ng[i] = 0;
    nh[i] = 12;
    nphi[i] = 0;
    nr[i] = 0;
    // slacks
    nsbu[i] = 0;
    nsbx[i] = 0;
    nsg[i] = 0;
    nsh[i] = 0;
    nsphi[i] = 0;

    /* terminal node */
    // common
    i = N;
    nx[i] = 43;
    nu[i] = 0;
    nz[i] = 0;
    ns[i] = 0;
    np[i] = 6201;
    // cost
    ny[i] = 43;
    // constraints
    nbu[i] = 0;
    nbx[i] = 0;
    ng[i] = 0;
    nh[i] = 849;
    nphi[i] = 0;
    nr[i] = 0;
    // slacks
    nsbu[i] = 0;
    nsbx[i] = 0;
    nsg[i] = 0;
    nsh[i] = 0;
    nsphi[i] = 0;
    nbxe[i] = 0;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 0);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 0);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }


    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);


    for (int i = 1; i < 1; i++)
    {
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    for (int i = 1; i < 2; i++)
    {
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }



    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);

    free(intNp1mem);

    return nlp_dims;
}



/**
 * Internal function for multiphase_ocp_acados_create: step 3
 */
void multiphase_ocp_acados_create_setup_functions(multiphase_ocp_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);



    ext_fun_opts.external_workspace = true;


    MAP_CASADI_FNC(nl_constr_h_0_fun_jac, kso_p0_4ea03e70c6_constr_h_0_fun_jac_uxt_zt);
    MAP_CASADI_FNC(nl_constr_h_0_fun, kso_p0_4ea03e70c6_constr_h_0_fun);



/////////////// PATH
    int n_path, n_cost_path;
    n_path = 1;
    n_cost_path = 0;


    // discrete dynamics
    capsule->discr_dyn_phi_fun_0 = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*n_path);
    for (int i = 0; i < n_path; i++)
    {
        MAP_CASADI_FNC(discr_dyn_phi_fun_0[i], kso_p0_4ea03e70c6_dyn_disc_phi_fun);
    }

    capsule->discr_dyn_phi_fun_jac_ut_xt_0 = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*n_path);
    for (int i = 0; i < n_path; i++)
    {
        MAP_CASADI_FNC(discr_dyn_phi_fun_jac_ut_xt_0[i], kso_p0_4ea03e70c6_dyn_disc_phi_fun_jac);
    }
    n_path = 1;
    n_cost_path = 1;
    capsule->nl_constr_h_fun_jac_1 = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*n_cost_path);
    for (int i = 0; i < n_cost_path; i++) {
        MAP_CASADI_FNC(nl_constr_h_fun_jac_1[i], kso_p1_4ea03e70c6_constr_h_fun_jac_uxt_zt);
    }
    capsule->nl_constr_h_fun_1 = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*n_cost_path);
    for (int i = 0; i < n_cost_path; i++) {
        MAP_CASADI_FNC(nl_constr_h_fun_1[i], kso_p1_4ea03e70c6_constr_h_fun);
    }
    



    // discrete dynamics
    capsule->discr_dyn_phi_fun_1 = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*n_path);
    for (int i = 0; i < n_path; i++)
    {
        MAP_CASADI_FNC(discr_dyn_phi_fun_1[i], kso_p1_4ea03e70c6_dyn_disc_phi_fun);
    }

    capsule->discr_dyn_phi_fun_jac_ut_xt_1 = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*n_path);
    for (int i = 0; i < n_path; i++)
    {
        MAP_CASADI_FNC(discr_dyn_phi_fun_jac_ut_xt_1[i], kso_p1_4ea03e70c6_dyn_disc_phi_fun_jac);
    }




    MAP_CASADI_FNC(nl_constr_h_e_fun_jac, kso_p1_4ea03e70c6_constr_h_e_fun_jac_uxt_zt);
    MAP_CASADI_FNC(nl_constr_h_e_fun, kso_p1_4ea03e70c6_constr_h_e_fun);
    
    

#undef MAP_CASADI_FNC
}



/**
 * Internal function for multiphase_ocp_acados_create: step 4
 */
void multiphase_ocp_acados_create_set_default_parameters(multiphase_ocp_solver_capsule* capsule) {

    // initialize parameters to nominal value
    double* p = calloc(6201, sizeof(double));

    

    

    for (int i = 0; i < 1; i++) {
        multiphase_ocp_acados_update_params(capsule, i, p, NP_0);
    }

    

    

    for (int i = 1; i < 2; i++) {
        multiphase_ocp_acados_update_params(capsule, i, p, NP_1);
    }
    free(p);
}




/**
 * Internal function for multiphase_ocp_acados_create: step 5
 */
void multiphase_ocp_acados_create_setup_nlp_in_numerical_values(multiphase_ocp_solver_capsule* capsule, int N)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;


    // set up time_steps


    double time_step = 1;
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
    }
    // set cost scaling
    double* cost_scaling = malloc((N+1)*sizeof(double));
    cost_scaling[0] = 1;
    cost_scaling[1] = 1;
    cost_scaling[2] = 1;
    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
    }
    free(cost_scaling);

    /* INITIAL NODE */
    double* yref_0 = calloc(86, sizeof(double));
    // change only the non-zero elements:
    yref_0[2] = 0.7779237030901003;
    yref_0[6] = 1;
    yref_0[7] = -0.2;
    yref_0[10] = 0.42;
    yref_0[11] = -0.22;
    yref_0[13] = -0.2;
    yref_0[16] = 0.42;
    yref_0[17] = -0.22;
    yref_0[22] = 0.25;
    yref_0[23] = 0.2;
    yref_0[25] = 0.9;
    yref_0[29] = 0.25;
    yref_0[30] = -0.2;
    yref_0[32] = 0.9;
    yref_0[36] = 0.45;
    yref_0[38] = 0.15;
    yref_0[42] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);

    double* W_0 = calloc(86*86, sizeof(double));
    // change only the non-zero elements:
    W_0[0+(86) * 0] = 1;
    W_0[1+(86) * 1] = 1;
    W_0[2+(86) * 2] = 1;
    W_0[3+(86) * 3] = 10;
    W_0[4+(86) * 4] = 10;
    W_0[5+(86) * 5] = 10;
    W_0[6+(86) * 6] = 10;
    W_0[7+(86) * 7] = 1;
    W_0[8+(86) * 8] = 1;
    W_0[9+(86) * 9] = 1;
    W_0[10+(86) * 10] = 1;
    W_0[11+(86) * 11] = 1;
    W_0[12+(86) * 12] = 1;
    W_0[13+(86) * 13] = 1;
    W_0[14+(86) * 14] = 1;
    W_0[15+(86) * 15] = 1;
    W_0[16+(86) * 16] = 1;
    W_0[17+(86) * 17] = 1;
    W_0[18+(86) * 18] = 1;
    W_0[19+(86) * 19] = 1;
    W_0[20+(86) * 20] = 1;
    W_0[21+(86) * 21] = 1;
    W_0[22+(86) * 22] = 1;
    W_0[23+(86) * 23] = 1;
    W_0[24+(86) * 24] = 1;
    W_0[25+(86) * 25] = 1;
    W_0[26+(86) * 26] = 1;
    W_0[27+(86) * 27] = 1;
    W_0[28+(86) * 28] = 1;
    W_0[29+(86) * 29] = 1;
    W_0[30+(86) * 30] = 1;
    W_0[31+(86) * 31] = 1;
    W_0[32+(86) * 32] = 1;
    W_0[33+(86) * 33] = 1;
    W_0[34+(86) * 34] = 1;
    W_0[35+(86) * 35] = 1;
    W_0[36+(86) * 36] = 1;
    W_0[37+(86) * 37] = 1;
    W_0[38+(86) * 38] = 1;
    W_0[39+(86) * 39] = 1;
    W_0[40+(86) * 40] = 1;
    W_0[41+(86) * 41] = 1;
    W_0[42+(86) * 42] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);
    double* Vx_0 = calloc(86*43, sizeof(double));
    // change only the non-zero elements:
    Vx_0[0+(86) * 0] = 1;
    Vx_0[1+(86) * 1] = 1;
    Vx_0[2+(86) * 2] = 1;
    Vx_0[3+(86) * 3] = 1;
    Vx_0[4+(86) * 4] = 1;
    Vx_0[5+(86) * 5] = 1;
    Vx_0[6+(86) * 6] = 1;
    Vx_0[7+(86) * 7] = 1;
    Vx_0[8+(86) * 8] = 1;
    Vx_0[9+(86) * 9] = 1;
    Vx_0[10+(86) * 10] = 1;
    Vx_0[11+(86) * 11] = 1;
    Vx_0[12+(86) * 12] = 1;
    Vx_0[13+(86) * 13] = 1;
    Vx_0[14+(86) * 14] = 1;
    Vx_0[15+(86) * 15] = 1;
    Vx_0[16+(86) * 16] = 1;
    Vx_0[17+(86) * 17] = 1;
    Vx_0[18+(86) * 18] = 1;
    Vx_0[19+(86) * 19] = 1;
    Vx_0[20+(86) * 20] = 1;
    Vx_0[21+(86) * 21] = 1;
    Vx_0[22+(86) * 22] = 1;
    Vx_0[23+(86) * 23] = 1;
    Vx_0[24+(86) * 24] = 1;
    Vx_0[25+(86) * 25] = 1;
    Vx_0[26+(86) * 26] = 1;
    Vx_0[27+(86) * 27] = 1;
    Vx_0[28+(86) * 28] = 1;
    Vx_0[29+(86) * 29] = 1;
    Vx_0[30+(86) * 30] = 1;
    Vx_0[31+(86) * 31] = 1;
    Vx_0[32+(86) * 32] = 1;
    Vx_0[33+(86) * 33] = 1;
    Vx_0[34+(86) * 34] = 1;
    Vx_0[35+(86) * 35] = 1;
    Vx_0[36+(86) * 36] = 1;
    Vx_0[37+(86) * 37] = 1;
    Vx_0[38+(86) * 38] = 1;
    Vx_0[39+(86) * 39] = 1;
    Vx_0[40+(86) * 40] = 1;
    Vx_0[41+(86) * 41] = 1;
    Vx_0[42+(86) * 42] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vx", Vx_0);
    free(Vx_0);
    double* Vu_0 = calloc(86*43, sizeof(double));
    // change only the non-zero elements:
    Vu_0[43+(86) * 0] = 1;
    Vu_0[44+(86) * 1] = 1;
    Vu_0[45+(86) * 2] = 1;
    Vu_0[46+(86) * 3] = 1;
    Vu_0[47+(86) * 4] = 1;
    Vu_0[48+(86) * 5] = 1;
    Vu_0[49+(86) * 6] = 1;
    Vu_0[50+(86) * 7] = 1;
    Vu_0[51+(86) * 8] = 1;
    Vu_0[52+(86) * 9] = 1;
    Vu_0[53+(86) * 10] = 1;
    Vu_0[54+(86) * 11] = 1;
    Vu_0[55+(86) * 12] = 1;
    Vu_0[56+(86) * 13] = 1;
    Vu_0[57+(86) * 14] = 1;
    Vu_0[58+(86) * 15] = 1;
    Vu_0[59+(86) * 16] = 1;
    Vu_0[60+(86) * 17] = 1;
    Vu_0[61+(86) * 18] = 1;
    Vu_0[62+(86) * 19] = 1;
    Vu_0[63+(86) * 20] = 1;
    Vu_0[64+(86) * 21] = 1;
    Vu_0[65+(86) * 22] = 1;
    Vu_0[66+(86) * 23] = 1;
    Vu_0[67+(86) * 24] = 1;
    Vu_0[68+(86) * 25] = 1;
    Vu_0[69+(86) * 26] = 1;
    Vu_0[70+(86) * 27] = 1;
    Vu_0[71+(86) * 28] = 1;
    Vu_0[72+(86) * 29] = 1;
    Vu_0[73+(86) * 30] = 1;
    Vu_0[74+(86) * 31] = 1;
    Vu_0[75+(86) * 32] = 1;
    Vu_0[76+(86) * 33] = 1;
    Vu_0[77+(86) * 34] = 1;
    Vu_0[78+(86) * 35] = 1;
    Vu_0[79+(86) * 36] = 1;
    Vu_0[80+(86) * 37] = 1;
    Vu_0[81+(86) * 38] = 1;
    Vu_0[82+(86) * 39] = 1;
    Vu_0[83+(86) * 40] = 1;
    Vu_0[84+(86) * 41] = 1;
    Vu_0[85+(86) * 42] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vu", Vu_0);
    free(Vu_0);



    // constraints at initial node
    // x0
    int nbx_0 = 43;
    int* idxbx0 = malloc(nbx_0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;
    idxbx0[6] = 6;
    idxbx0[7] = 7;
    idxbx0[8] = 8;
    idxbx0[9] = 9;
    idxbx0[10] = 10;
    idxbx0[11] = 11;
    idxbx0[12] = 12;
    idxbx0[13] = 13;
    idxbx0[14] = 14;
    idxbx0[15] = 15;
    idxbx0[16] = 16;
    idxbx0[17] = 17;
    idxbx0[18] = 18;
    idxbx0[19] = 19;
    idxbx0[20] = 20;
    idxbx0[21] = 21;
    idxbx0[22] = 22;
    idxbx0[23] = 23;
    idxbx0[24] = 24;
    idxbx0[25] = 25;
    idxbx0[26] = 26;
    idxbx0[27] = 27;
    idxbx0[28] = 28;
    idxbx0[29] = 29;
    idxbx0[30] = 30;
    idxbx0[31] = 31;
    idxbx0[32] = 32;
    idxbx0[33] = 33;
    idxbx0[34] = 34;
    idxbx0[35] = 35;
    idxbx0[36] = 36;
    idxbx0[37] = 37;
    idxbx0[38] = 38;
    idxbx0[39] = 39;
    idxbx0[40] = 40;
    idxbx0[41] = 41;
    idxbx0[42] = 42;

    double* lubx0 = calloc(2*nbx_0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + nbx_0;
    // change only the non-zero elements:
    lbx0[0] = 0.09317839969727612;
    ubx0[0] = 0.09317839969727612;
    lbx0[1] = 0.13045619700661976;
    ubx0[1] = 0.13045619700661976;
    lbx0[2] = 0.4514808774673858;
    ubx0[2] = 0.4514808774673858;
    lbx0[3] = 0.07370095777944052;
    ubx0[3] = 0.07370095777944052;
    lbx0[4] = -0.3423530429054911;
    ubx0[4] = -0.3423530429054911;
    lbx0[5] = 0.023451630060614684;
    ubx0[5] = 0.023451630060614684;
    lbx0[6] = 0.936382714514931;
    ubx0[6] = 0.936382714514931;
    lbx0[7] = -0.6233811583995398;
    ubx0[7] = -0.6233811583995398;
    lbx0[8] = 0.04320584036993826;
    ubx0[8] = 0.04320584036993826;
    lbx0[9] = 0.00512586167006415;
    ubx0[9] = 0.00512586167006415;
    lbx0[10] = 0.8120430340777451;
    ubx0[10] = 0.8120430340777451;
    lbx0[11] = 0.5235994610924907;
    ubx0[11] = 0.5235994610924907;
    lbx0[12] = -0.1198743825910852;
    ubx0[12] = -0.1198743825910852;
    lbx0[13] = -0.6546972765756901;
    ubx0[13] = -0.6546972765756901;
    lbx0[14] = 0.2927752375691052;
    ubx0[14] = 0.2927752375691052;
    lbx0[15] = -0.1348090578958186;
    ubx0[15] = -0.1348090578958186;
    lbx0[16] = 0.874966313023328;
    ubx0[16] = 0.874966313023328;
    lbx0[17] = 0.5236000056945823;
    ubx0[17] = 0.5236000056945823;
    lbx0[18] = -0.26175983101238426;
    ubx0[18] = -0.26175983101238426;
    lbx0[19] = 0.0071623851845860325;
    ubx0[19] = 0.0071623851845860325;
    lbx0[20] = -0.1325147066143501;
    ubx0[20] = -0.1325147066143501;
    lbx0[21] = -0.3276110679257776;
    ubx0[21] = -0.3276110679257776;
    lbx0[22] = 0.9627558816703656;
    ubx0[22] = 0.9627558816703656;
    lbx0[23] = 0.3175265387725001;
    ubx0[23] = 0.3175265387725001;
    lbx0[24] = -0.019495194568615357;
    ubx0[24] = -0.019495194568615357;
    lbx0[25] = 1.12929731054022;
    ubx0[25] = 1.12929731054022;
    lbx0[26] = -0.30529467129872;
    ubx0[26] = -0.30529467129872;
    lbx0[27] = 0.16308386484701326;
    ubx0[27] = 0.16308386484701326;
    lbx0[28] = -1.6144295664682768;
    ubx0[28] = -1.6144295664682768;
    lbx0[29] = 0.6459958835714067;
    ubx0[29] = 0.6459958835714067;
    lbx0[30] = -0.33601217123306154;
    ubx0[30] = -0.33601217123306154;
    lbx0[31] = -1.1382458326715306;
    ubx0[31] = -1.1382458326715306;
    lbx0[32] = 1.928217408542408;
    ubx0[32] = 1.928217408542408;
    lbx0[33] = 1.357881807378167;
    ubx0[33] = 1.357881807378167;
    lbx0[34] = -0.08977701145062987;
    ubx0[34] = -0.08977701145062987;
    lbx0[35] = -1.6144295675703706;
    ubx0[35] = -1.6144295675703706;
    lbx0[36] = -0.03541367173626337;
    ubx0[36] = -0.03541367173626337;
    lbx0[37] = 0.1577303922893885;
    ubx0[37] = 0.1577303922893885;
    lbx0[38] = 0.15;
    ubx0[38] = 0.15;
    lbx0[39] = 0.0000000000000000000000000000000000000000000000000000000000000000009156426765968716;
    ubx0[39] = 0.0000000000000000000000000000000000000000000000000000000000000000009156426765968716;
    lbx0[40] = -0.0000000000000000000000000000000000000000000000000000000000000000004391943496223828;
    ubx0[40] = -0.0000000000000000000000000000000000000000000000000000000000000000004391943496223828;
    lbx0[41] = 0.24073518763855625;
    ubx0[41] = 0.24073518763855625;
    lbx0[42] = 0.970590835253482;
    ubx0[42] = 0.970590835253482;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);
    // idxbxe_0
    int* idxbxe_0 = malloc(43 * sizeof(int));
    idxbxe_0[0] = 0;
    idxbxe_0[1] = 1;
    idxbxe_0[2] = 2;
    idxbxe_0[3] = 3;
    idxbxe_0[4] = 4;
    idxbxe_0[5] = 5;
    idxbxe_0[6] = 6;
    idxbxe_0[7] = 7;
    idxbxe_0[8] = 8;
    idxbxe_0[9] = 9;
    idxbxe_0[10] = 10;
    idxbxe_0[11] = 11;
    idxbxe_0[12] = 12;
    idxbxe_0[13] = 13;
    idxbxe_0[14] = 14;
    idxbxe_0[15] = 15;
    idxbxe_0[16] = 16;
    idxbxe_0[17] = 17;
    idxbxe_0[18] = 18;
    idxbxe_0[19] = 19;
    idxbxe_0[20] = 20;
    idxbxe_0[21] = 21;
    idxbxe_0[22] = 22;
    idxbxe_0[23] = 23;
    idxbxe_0[24] = 24;
    idxbxe_0[25] = 25;
    idxbxe_0[26] = 26;
    idxbxe_0[27] = 27;
    idxbxe_0[28] = 28;
    idxbxe_0[29] = 29;
    idxbxe_0[30] = 30;
    idxbxe_0[31] = 31;
    idxbxe_0[32] = 32;
    idxbxe_0[33] = 33;
    idxbxe_0[34] = 34;
    idxbxe_0[35] = 35;
    idxbxe_0[36] = 36;
    idxbxe_0[37] = 37;
    idxbxe_0[38] = 38;
    idxbxe_0[39] = 39;
    idxbxe_0[40] = 40;
    idxbxe_0[41] = 41;
    idxbxe_0[42] = 42;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);



    // set up nonlinear constraints for first stage
    double* luh_0 = calloc(2*12, sizeof(double));
    double* lh_0 = luh_0;
    double* uh_0 = luh_0 + 12;
    lh_0[9] = -1000000000;
    lh_0[10] = -1000000000;
    lh_0[11] = -1000000000;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lh", lh_0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "uh", uh_0);
    free(luh_0);








    /* Path related delarations */
    int i_fun;

    // cost
    double* yref;
    double* Vx;
    double* Vu;
    double* Vz;
    double* W;

    // bounds on u
    int* idxbu;
    double* lubu;
    double* lbu;
    double* ubu;

    // bounds on x
    double* lubx;
    double* lbx;
    double* ubx;
    int* idxbx;

    // general linear constraints
    double* D;
    double* C;
    double* lug;
    double* lg;
    double* ug;

    // nonlinear constraints
    double* luh;
    double* lh;
    double* uh;
    double* luphi;
    double* lphi;
    double* uphi;

    // general slack related
    double* zlumem;
    double* Zl;
    double* Zu;
    double* zl;
    double* zu;

    // specific slack types
    int* idxsbx;
    double* lusbx;
    double* lsbx;
    double* usbx;

    int* idxsbu;
    double* lusbu;
    double* lsbu;
    double* usbu;

    int* idxsg;
    double* lusg;
    double* lsg;
    double* usg;

    int* idxsh;
    double* lush;
    double* lsh;
    double* ush;

    int* idxsphi;
    double* lusphi;
    double* lsphi;
    double* usphi;

    // slacks idxs_rev
    int* idxs_rev;
    double* lus;
    double* ls;
    double* us;

    /*********************
     *  Phase 0
     * *******************/

    /**** Cost phase 0 ****/
    yref = calloc(86, sizeof(double));
    // change only the non-zero elements:
    yref[2] = 0.7779237030901003;
    yref[6] = 1;
    yref[7] = -0.2;
    yref[10] = 0.42;
    yref[11] = -0.22;
    yref[13] = -0.2;
    yref[16] = 0.42;
    yref[17] = -0.22;
    yref[22] = 0.25;
    yref[23] = 0.2;
    yref[25] = 0.9;
    yref[29] = 0.25;
    yref[30] = -0.2;
    yref[32] = 0.9;
    yref[36] = 0.45;
    yref[38] = 0.15;
    yref[42] = 1;

    for (int i = 1; i < 1; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(yref);
    W = calloc(86*86, sizeof(double));
    // change only the non-zero elements:
    W[0+(86) * 0] = 1;
    W[1+(86) * 1] = 1;
    W[2+(86) * 2] = 1;
    W[3+(86) * 3] = 10;
    W[4+(86) * 4] = 10;
    W[5+(86) * 5] = 10;
    W[6+(86) * 6] = 10;
    W[7+(86) * 7] = 1;
    W[8+(86) * 8] = 1;
    W[9+(86) * 9] = 1;
    W[10+(86) * 10] = 1;
    W[11+(86) * 11] = 1;
    W[12+(86) * 12] = 1;
    W[13+(86) * 13] = 1;
    W[14+(86) * 14] = 1;
    W[15+(86) * 15] = 1;
    W[16+(86) * 16] = 1;
    W[17+(86) * 17] = 1;
    W[18+(86) * 18] = 1;
    W[19+(86) * 19] = 1;
    W[20+(86) * 20] = 1;
    W[21+(86) * 21] = 1;
    W[22+(86) * 22] = 1;
    W[23+(86) * 23] = 1;
    W[24+(86) * 24] = 1;
    W[25+(86) * 25] = 1;
    W[26+(86) * 26] = 1;
    W[27+(86) * 27] = 1;
    W[28+(86) * 28] = 1;
    W[29+(86) * 29] = 1;
    W[30+(86) * 30] = 1;
    W[31+(86) * 31] = 1;
    W[32+(86) * 32] = 1;
    W[33+(86) * 33] = 1;
    W[34+(86) * 34] = 1;
    W[35+(86) * 35] = 1;
    W[36+(86) * 36] = 1;
    W[37+(86) * 37] = 1;
    W[38+(86) * 38] = 1;
    W[39+(86) * 39] = 1;
    W[40+(86) * 40] = 1;
    W[41+(86) * 41] = 1;
    W[42+(86) * 42] = 1;

    for (int i = 1; i < 1; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
    }
    free(W);
    Vx = calloc(86*43, sizeof(double));
    // change only the non-zero elements:
    Vx[0+(86) * 0] = 1;
    Vx[1+(86) * 1] = 1;
    Vx[2+(86) * 2] = 1;
    Vx[3+(86) * 3] = 1;
    Vx[4+(86) * 4] = 1;
    Vx[5+(86) * 5] = 1;
    Vx[6+(86) * 6] = 1;
    Vx[7+(86) * 7] = 1;
    Vx[8+(86) * 8] = 1;
    Vx[9+(86) * 9] = 1;
    Vx[10+(86) * 10] = 1;
    Vx[11+(86) * 11] = 1;
    Vx[12+(86) * 12] = 1;
    Vx[13+(86) * 13] = 1;
    Vx[14+(86) * 14] = 1;
    Vx[15+(86) * 15] = 1;
    Vx[16+(86) * 16] = 1;
    Vx[17+(86) * 17] = 1;
    Vx[18+(86) * 18] = 1;
    Vx[19+(86) * 19] = 1;
    Vx[20+(86) * 20] = 1;
    Vx[21+(86) * 21] = 1;
    Vx[22+(86) * 22] = 1;
    Vx[23+(86) * 23] = 1;
    Vx[24+(86) * 24] = 1;
    Vx[25+(86) * 25] = 1;
    Vx[26+(86) * 26] = 1;
    Vx[27+(86) * 27] = 1;
    Vx[28+(86) * 28] = 1;
    Vx[29+(86) * 29] = 1;
    Vx[30+(86) * 30] = 1;
    Vx[31+(86) * 31] = 1;
    Vx[32+(86) * 32] = 1;
    Vx[33+(86) * 33] = 1;
    Vx[34+(86) * 34] = 1;
    Vx[35+(86) * 35] = 1;
    Vx[36+(86) * 36] = 1;
    Vx[37+(86) * 37] = 1;
    Vx[38+(86) * 38] = 1;
    Vx[39+(86) * 39] = 1;
    Vx[40+(86) * 40] = 1;
    Vx[41+(86) * 41] = 1;
    Vx[42+(86) * 42] = 1;
    for (int i = 1; i < 1; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vx", Vx);
    }
    free(Vx);

    
    Vu = calloc(86*43, sizeof(double));
    // change only the non-zero elements:
    Vu[43+(86) * 0] = 1;
    Vu[44+(86) * 1] = 1;
    Vu[45+(86) * 2] = 1;
    Vu[46+(86) * 3] = 1;
    Vu[47+(86) * 4] = 1;
    Vu[48+(86) * 5] = 1;
    Vu[49+(86) * 6] = 1;
    Vu[50+(86) * 7] = 1;
    Vu[51+(86) * 8] = 1;
    Vu[52+(86) * 9] = 1;
    Vu[53+(86) * 10] = 1;
    Vu[54+(86) * 11] = 1;
    Vu[55+(86) * 12] = 1;
    Vu[56+(86) * 13] = 1;
    Vu[57+(86) * 14] = 1;
    Vu[58+(86) * 15] = 1;
    Vu[59+(86) * 16] = 1;
    Vu[60+(86) * 17] = 1;
    Vu[61+(86) * 18] = 1;
    Vu[62+(86) * 19] = 1;
    Vu[63+(86) * 20] = 1;
    Vu[64+(86) * 21] = 1;
    Vu[65+(86) * 22] = 1;
    Vu[66+(86) * 23] = 1;
    Vu[67+(86) * 24] = 1;
    Vu[68+(86) * 25] = 1;
    Vu[69+(86) * 26] = 1;
    Vu[70+(86) * 27] = 1;
    Vu[71+(86) * 28] = 1;
    Vu[72+(86) * 29] = 1;
    Vu[73+(86) * 30] = 1;
    Vu[74+(86) * 31] = 1;
    Vu[75+(86) * 32] = 1;
    Vu[76+(86) * 33] = 1;
    Vu[77+(86) * 34] = 1;
    Vu[78+(86) * 35] = 1;
    Vu[79+(86) * 36] = 1;
    Vu[80+(86) * 37] = 1;
    Vu[81+(86) * 38] = 1;
    Vu[82+(86) * 39] = 1;
    Vu[83+(86) * 40] = 1;
    Vu[84+(86) * 41] = 1;
    Vu[85+(86) * 42] = 1;

    for (int i = 1; i < 1; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vu", Vu);
    }
    free(Vu);


    /**** Constraints phase 0 ****/

    /* constraints that are the same for initial and intermediate */






    /* Path constraints */














    /*********************
     *  Phase 1
     * *******************/

    /**** Cost phase 1 ****/
    yref = calloc(86, sizeof(double));
    // change only the non-zero elements:
    yref[2] = 0.7779237030901003;
    yref[6] = 1;
    yref[7] = -0.2;
    yref[10] = 0.42;
    yref[11] = -0.22;
    yref[13] = -0.2;
    yref[16] = 0.42;
    yref[17] = -0.22;
    yref[22] = 0.25;
    yref[23] = 0.2;
    yref[25] = 0.9;
    yref[29] = 0.25;
    yref[30] = -0.2;
    yref[32] = 0.9;
    yref[36] = 0.45;
    yref[38] = 0.15;
    yref[42] = 1;

    for (int i = 1; i < 2; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(yref);
    W = calloc(86*86, sizeof(double));
    // change only the non-zero elements:
    W[0+(86) * 0] = 1;
    W[1+(86) * 1] = 1;
    W[2+(86) * 2] = 1;
    W[3+(86) * 3] = 10;
    W[4+(86) * 4] = 10;
    W[5+(86) * 5] = 10;
    W[6+(86) * 6] = 10;
    W[7+(86) * 7] = 1;
    W[8+(86) * 8] = 1;
    W[9+(86) * 9] = 1;
    W[10+(86) * 10] = 1;
    W[11+(86) * 11] = 1;
    W[12+(86) * 12] = 1;
    W[13+(86) * 13] = 1;
    W[14+(86) * 14] = 1;
    W[15+(86) * 15] = 1;
    W[16+(86) * 16] = 1;
    W[17+(86) * 17] = 1;
    W[18+(86) * 18] = 1;
    W[19+(86) * 19] = 1;
    W[20+(86) * 20] = 1;
    W[21+(86) * 21] = 1;
    W[22+(86) * 22] = 1;
    W[23+(86) * 23] = 1;
    W[24+(86) * 24] = 1;
    W[25+(86) * 25] = 1;
    W[26+(86) * 26] = 1;
    W[27+(86) * 27] = 1;
    W[28+(86) * 28] = 1;
    W[29+(86) * 29] = 1;
    W[30+(86) * 30] = 1;
    W[31+(86) * 31] = 1;
    W[32+(86) * 32] = 1;
    W[33+(86) * 33] = 1;
    W[34+(86) * 34] = 1;
    W[35+(86) * 35] = 1;
    W[36+(86) * 36] = 1;
    W[37+(86) * 37] = 1;
    W[38+(86) * 38] = 1;
    W[39+(86) * 39] = 1;
    W[40+(86) * 40] = 1;
    W[41+(86) * 41] = 1;
    W[42+(86) * 42] = 1;

    for (int i = 1; i < 2; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
    }
    free(W);
    Vx = calloc(86*43, sizeof(double));
    // change only the non-zero elements:
    Vx[0+(86) * 0] = 1;
    Vx[1+(86) * 1] = 1;
    Vx[2+(86) * 2] = 1;
    Vx[3+(86) * 3] = 1;
    Vx[4+(86) * 4] = 1;
    Vx[5+(86) * 5] = 1;
    Vx[6+(86) * 6] = 1;
    Vx[7+(86) * 7] = 1;
    Vx[8+(86) * 8] = 1;
    Vx[9+(86) * 9] = 1;
    Vx[10+(86) * 10] = 1;
    Vx[11+(86) * 11] = 1;
    Vx[12+(86) * 12] = 1;
    Vx[13+(86) * 13] = 1;
    Vx[14+(86) * 14] = 1;
    Vx[15+(86) * 15] = 1;
    Vx[16+(86) * 16] = 1;
    Vx[17+(86) * 17] = 1;
    Vx[18+(86) * 18] = 1;
    Vx[19+(86) * 19] = 1;
    Vx[20+(86) * 20] = 1;
    Vx[21+(86) * 21] = 1;
    Vx[22+(86) * 22] = 1;
    Vx[23+(86) * 23] = 1;
    Vx[24+(86) * 24] = 1;
    Vx[25+(86) * 25] = 1;
    Vx[26+(86) * 26] = 1;
    Vx[27+(86) * 27] = 1;
    Vx[28+(86) * 28] = 1;
    Vx[29+(86) * 29] = 1;
    Vx[30+(86) * 30] = 1;
    Vx[31+(86) * 31] = 1;
    Vx[32+(86) * 32] = 1;
    Vx[33+(86) * 33] = 1;
    Vx[34+(86) * 34] = 1;
    Vx[35+(86) * 35] = 1;
    Vx[36+(86) * 36] = 1;
    Vx[37+(86) * 37] = 1;
    Vx[38+(86) * 38] = 1;
    Vx[39+(86) * 39] = 1;
    Vx[40+(86) * 40] = 1;
    Vx[41+(86) * 41] = 1;
    Vx[42+(86) * 42] = 1;
    for (int i = 1; i < 2; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vx", Vx);
    }
    free(Vx);

    
    Vu = calloc(86*43, sizeof(double));
    // change only the non-zero elements:
    Vu[43+(86) * 0] = 1;
    Vu[44+(86) * 1] = 1;
    Vu[45+(86) * 2] = 1;
    Vu[46+(86) * 3] = 1;
    Vu[47+(86) * 4] = 1;
    Vu[48+(86) * 5] = 1;
    Vu[49+(86) * 6] = 1;
    Vu[50+(86) * 7] = 1;
    Vu[51+(86) * 8] = 1;
    Vu[52+(86) * 9] = 1;
    Vu[53+(86) * 10] = 1;
    Vu[54+(86) * 11] = 1;
    Vu[55+(86) * 12] = 1;
    Vu[56+(86) * 13] = 1;
    Vu[57+(86) * 14] = 1;
    Vu[58+(86) * 15] = 1;
    Vu[59+(86) * 16] = 1;
    Vu[60+(86) * 17] = 1;
    Vu[61+(86) * 18] = 1;
    Vu[62+(86) * 19] = 1;
    Vu[63+(86) * 20] = 1;
    Vu[64+(86) * 21] = 1;
    Vu[65+(86) * 22] = 1;
    Vu[66+(86) * 23] = 1;
    Vu[67+(86) * 24] = 1;
    Vu[68+(86) * 25] = 1;
    Vu[69+(86) * 26] = 1;
    Vu[70+(86) * 27] = 1;
    Vu[71+(86) * 28] = 1;
    Vu[72+(86) * 29] = 1;
    Vu[73+(86) * 30] = 1;
    Vu[74+(86) * 31] = 1;
    Vu[75+(86) * 32] = 1;
    Vu[76+(86) * 33] = 1;
    Vu[77+(86) * 34] = 1;
    Vu[78+(86) * 35] = 1;
    Vu[79+(86) * 36] = 1;
    Vu[80+(86) * 37] = 1;
    Vu[81+(86) * 38] = 1;
    Vu[82+(86) * 39] = 1;
    Vu[83+(86) * 40] = 1;
    Vu[84+(86) * 41] = 1;
    Vu[85+(86) * 42] = 1;

    for (int i = 1; i < 2; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vu", Vu);
    }
    free(Vu);


    /**** Constraints phase 1 ****/

    /* constraints that are the same for initial and intermediate */






    /* Path constraints */



    // set up nonlinear constraints for stage 1 to N-1
    luh = calloc(2*869, sizeof(double));
    lh = luh;
    uh = luh + 869;
    lh[32] = -1000000000;
    lh[33] = -1000000000;
    lh[34] = -1000000000;
    lh[35] = -1000000000;
    lh[36] = -1000000000;
    lh[37] = -1000000000;
    lh[38] = -1000000000;
    lh[39] = -1000000000;
    lh[40] = -1000000000;
    lh[41] = -1000000000;
    lh[42] = -1000000000;
    lh[43] = -1000000000;
    lh[44] = -1000000000;
    lh[45] = -1000000000;
    lh[46] = -1000000000;
    lh[47] = -1000000000;
    lh[48] = -1000000000;
    lh[49] = -1000000000;
    lh[50] = -1000000000;
    lh[51] = -1000000000;
    lh[52] = -1000000000;
    lh[53] = -1000000000;
    lh[54] = -1000000000;
    lh[55] = -1000000000;
    lh[56] = -1000000000;
    lh[57] = -1000000000;
    lh[58] = -1000000000;
    lh[59] = -1000000000;
    lh[60] = -1000000000;
    lh[61] = -1000000000;
    lh[62] = -1000000000;
    lh[63] = -1000000000;
    lh[64] = -1000000000;
    lh[65] = -1000000000;
    lh[66] = -1000000000;
    lh[67] = -1000000000;
    lh[68] = -1000000000;
    lh[69] = -1000000000;
    lh[70] = -1000000000;
    lh[71] = -1000000000;
    lh[72] = -1000000000;
    lh[73] = -1000000000;
    lh[74] = -1000000000;
    lh[75] = -1000000000;
    lh[76] = -1000000000;
    lh[77] = -1000000000;
    lh[78] = -1000000000;
    lh[79] = -1000000000;
    lh[80] = -1000000000;
    lh[81] = -1000000000;
    lh[82] = -1000000000;
    lh[83] = -1000000000;
    lh[84] = -1000000000;
    lh[85] = -1000000000;
    lh[86] = -1000000000;
    lh[87] = -1000000000;
    lh[88] = -1000000000;
    lh[89] = -1000000000;
    lh[90] = -1000000000;
    lh[91] = -1000000000;
    lh[92] = -1000000000;
    lh[93] = -1000000000;
    lh[94] = -1000000000;
    lh[95] = -1000000000;
    lh[96] = -1000000000;
    lh[97] = -1000000000;
    lh[98] = -1000000000;
    lh[99] = -1000000000;
    lh[100] = -1000000000;
    lh[101] = -1000000000;
    lh[102] = -1000000000;
    lh[103] = -1000000000;
    lh[104] = -1000000000;
    lh[105] = -1000000000;
    lh[106] = -1000000000;
    lh[107] = -1000000000;
    lh[108] = -1000000000;
    lh[109] = -1000000000;
    lh[110] = -1000000000;
    lh[111] = -1000000000;
    lh[112] = -1000000000;
    lh[113] = -1000000000;
    lh[114] = -1000000000;
    lh[115] = -1000000000;
    lh[116] = -1000000000;
    lh[117] = -1000000000;
    lh[118] = -1000000000;
    lh[119] = -1000000000;
    lh[120] = -1000000000;
    lh[121] = -1000000000;
    lh[122] = -1000000000;
    lh[123] = -1000000000;
    lh[124] = -1000000000;
    lh[125] = -1000000000;
    lh[126] = -1000000000;
    lh[127] = -1000000000;
    lh[128] = -1000000000;
    lh[129] = -1000000000;
    lh[130] = -1000000000;
    lh[131] = -1000000000;
    lh[132] = -1000000000;
    lh[133] = -1000000000;
    lh[134] = -1000000000;
    lh[135] = -1000000000;
    lh[136] = -1000000000;
    lh[137] = -1000000000;
    lh[138] = -1000000000;
    lh[139] = -1000000000;
    lh[140] = -1000000000;
    lh[141] = -1000000000;
    lh[142] = -1000000000;
    lh[143] = -1000000000;
    lh[144] = -1000000000;
    lh[145] = -1000000000;
    lh[146] = -1000000000;
    lh[147] = -1000000000;
    lh[148] = -1000000000;
    lh[149] = -1000000000;
    lh[150] = -1000000000;
    lh[151] = -1000000000;
    lh[152] = -1000000000;
    lh[153] = -1000000000;
    lh[154] = -1000000000;
    lh[155] = -1000000000;
    lh[156] = -1000000000;
    lh[157] = -1000000000;
    lh[158] = -1000000000;
    lh[159] = -1000000000;
    lh[160] = -1000000000;
    lh[161] = -1000000000;
    lh[162] = -1000000000;
    lh[163] = -1000000000;
    lh[164] = -1000000000;
    lh[165] = -1000000000;
    lh[166] = -1000000000;
    lh[167] = -1000000000;
    lh[168] = -1000000000;
    lh[169] = -1000000000;
    lh[170] = -1000000000;
    lh[171] = -1000000000;
    lh[172] = -1000000000;
    lh[173] = -1000000000;
    lh[174] = -1000000000;
    lh[175] = -1000000000;
    lh[176] = -1000000000;
    lh[177] = -1000000000;
    lh[178] = -1000000000;
    lh[179] = -1000000000;
    lh[180] = -1000000000;
    lh[181] = -1000000000;
    lh[182] = -1000000000;
    lh[183] = -1000000000;
    lh[184] = -1000000000;
    lh[185] = -1000000000;
    lh[186] = -1000000000;
    lh[187] = -1000000000;
    lh[188] = -1000000000;
    lh[189] = -1000000000;
    lh[190] = -1000000000;
    lh[191] = -1000000000;
    lh[192] = -1000000000;
    lh[193] = -1000000000;
    lh[194] = -1000000000;
    lh[195] = -1000000000;
    lh[196] = -1000000000;
    lh[197] = -1000000000;
    lh[198] = -1000000000;
    lh[199] = -1000000000;
    lh[200] = -1000000000;
    lh[201] = -1000000000;
    lh[202] = -1000000000;
    lh[203] = -1000000000;
    lh[204] = -1000000000;
    lh[205] = -1000000000;
    lh[206] = -1000000000;
    lh[207] = -1000000000;
    lh[208] = -1000000000;
    lh[209] = -1000000000;
    lh[210] = -1000000000;
    lh[211] = -1000000000;
    lh[212] = -1000000000;
    lh[213] = -1000000000;
    lh[214] = -1000000000;
    lh[215] = -1000000000;
    lh[216] = -1000000000;
    lh[217] = -1000000000;
    lh[218] = -1000000000;
    lh[219] = -1000000000;
    lh[220] = -1000000000;
    lh[221] = -1000000000;
    lh[222] = -1000000000;
    lh[223] = -1000000000;
    lh[224] = -1000000000;
    lh[225] = -1000000000;
    lh[226] = -1000000000;
    lh[227] = -1000000000;
    lh[228] = -1000000000;
    lh[229] = -1000000000;
    lh[230] = -1000000000;
    lh[231] = -1000000000;
    lh[232] = -1000000000;
    lh[233] = -1000000000;
    lh[234] = -1000000000;
    lh[235] = -1000000000;
    lh[236] = -1000000000;
    lh[237] = -1000000000;
    lh[238] = -1000000000;
    lh[239] = -1000000000;
    lh[240] = -1000000000;
    lh[241] = -1000000000;
    lh[242] = -1000000000;
    lh[243] = -1000000000;
    lh[244] = -1000000000;
    lh[245] = -1000000000;
    lh[246] = -1000000000;
    lh[247] = -1000000000;
    lh[248] = -1000000000;
    lh[249] = -1000000000;
    lh[250] = -1000000000;
    lh[251] = -1000000000;
    lh[252] = -1000000000;
    lh[253] = -1000000000;
    lh[254] = -1000000000;
    lh[255] = -1000000000;
    lh[256] = -1000000000;
    lh[257] = -1000000000;
    lh[258] = -1000000000;
    lh[259] = -1000000000;
    lh[260] = -1000000000;
    lh[261] = -1000000000;
    lh[262] = -1000000000;
    lh[263] = -1000000000;
    lh[264] = -1000000000;
    lh[265] = -1000000000;
    lh[266] = -1000000000;
    lh[267] = -1000000000;
    lh[268] = -1000000000;
    lh[269] = -1000000000;
    lh[270] = -1000000000;
    lh[271] = -1000000000;
    lh[272] = -1000000000;
    lh[273] = -1000000000;
    lh[274] = -1000000000;
    lh[275] = -1000000000;
    lh[276] = -1000000000;
    lh[277] = -1000000000;
    lh[278] = -1000000000;
    lh[279] = -1000000000;
    lh[280] = -1000000000;
    lh[281] = -1000000000;
    lh[282] = -1000000000;
    lh[283] = -1000000000;
    lh[284] = -1000000000;
    lh[285] = -1000000000;
    lh[286] = -1000000000;
    lh[287] = -1000000000;
    lh[288] = -1000000000;
    lh[289] = -1000000000;
    lh[290] = -1000000000;
    lh[291] = -1000000000;
    lh[292] = -1000000000;
    lh[293] = -1000000000;
    lh[294] = -1000000000;
    lh[295] = -1000000000;
    lh[296] = -1000000000;
    lh[297] = -1000000000;
    lh[298] = -1000000000;
    lh[299] = -1000000000;
    lh[300] = -1000000000;
    lh[301] = -1000000000;
    lh[302] = -1000000000;
    lh[303] = -1000000000;
    lh[304] = -1000000000;
    lh[305] = -1000000000;
    lh[306] = -1000000000;
    lh[307] = -1000000000;
    lh[308] = -1000000000;
    lh[309] = -1000000000;
    lh[310] = -1000000000;
    lh[311] = -1000000000;
    lh[312] = -1000000000;
    lh[313] = -1000000000;
    lh[314] = -1000000000;
    lh[315] = -1000000000;
    lh[316] = -1000000000;
    lh[317] = -1000000000;
    lh[318] = -1000000000;
    lh[319] = -1000000000;
    lh[320] = -1000000000;
    lh[321] = -1000000000;
    lh[322] = -1000000000;
    lh[323] = -1000000000;
    lh[324] = -1000000000;
    lh[325] = -1000000000;
    lh[326] = -1000000000;
    lh[327] = -1000000000;
    lh[328] = -1000000000;
    lh[329] = -1000000000;
    lh[330] = -1000000000;
    lh[331] = -1000000000;
    lh[332] = -1000000000;
    lh[333] = -1000000000;
    lh[334] = -1000000000;
    lh[335] = -1000000000;
    lh[336] = -1000000000;
    lh[337] = -1000000000;
    lh[338] = -1000000000;
    lh[339] = -1000000000;
    lh[340] = -1000000000;
    lh[341] = -1000000000;
    lh[342] = -1000000000;
    lh[343] = -1000000000;
    lh[344] = -1000000000;
    lh[345] = -1000000000;
    lh[346] = -1000000000;
    lh[347] = -1000000000;
    lh[348] = -1000000000;
    lh[349] = -1000000000;
    lh[350] = -1000000000;
    lh[351] = -1000000000;
    lh[352] = -1000000000;
    lh[353] = -1000000000;
    lh[354] = -1000000000;
    lh[355] = -1000000000;
    lh[356] = -1000000000;
    lh[357] = -1000000000;
    lh[358] = -1000000000;
    lh[359] = -1000000000;
    lh[360] = -1000000000;
    lh[361] = -1000000000;
    lh[362] = -1000000000;
    lh[363] = -1000000000;
    lh[364] = -1000000000;
    lh[365] = -1000000000;
    lh[366] = -1000000000;
    lh[367] = -1000000000;
    lh[368] = -1000000000;
    lh[369] = -1000000000;
    lh[370] = -1000000000;
    lh[371] = -1000000000;
    lh[372] = -1000000000;
    lh[373] = -1000000000;
    lh[374] = -1000000000;
    lh[375] = -1000000000;
    lh[376] = -1000000000;
    lh[377] = -1000000000;
    lh[378] = -1000000000;
    lh[379] = -1000000000;
    lh[380] = -1000000000;
    lh[381] = -1000000000;
    lh[382] = -1000000000;
    lh[383] = -1000000000;
    lh[384] = -1000000000;
    lh[385] = -1000000000;
    lh[386] = -1000000000;
    lh[387] = -1000000000;
    lh[388] = -1000000000;
    lh[389] = -1000000000;
    lh[390] = -1000000000;
    lh[391] = -1000000000;
    lh[392] = -1000000000;
    lh[393] = -1000000000;
    lh[394] = -1000000000;
    lh[395] = -1000000000;
    lh[396] = -1000000000;
    lh[397] = -1000000000;
    lh[398] = -1000000000;
    lh[399] = -1000000000;
    lh[400] = -1000000000;
    lh[401] = -1000000000;
    lh[402] = -1000000000;
    lh[403] = -1000000000;
    lh[404] = -1000000000;
    lh[405] = -1000000000;
    lh[406] = -1000000000;
    lh[407] = -1000000000;
    lh[408] = -1000000000;
    lh[409] = -1000000000;
    lh[410] = -1000000000;
    lh[411] = -1000000000;
    lh[412] = -1000000000;
    lh[413] = -1000000000;
    lh[414] = -1000000000;
    lh[415] = -1000000000;
    lh[416] = -1000000000;
    lh[417] = -1000000000;
    lh[418] = -1000000000;
    lh[419] = -1000000000;
    lh[420] = -1000000000;
    lh[421] = -1000000000;
    lh[422] = -1000000000;
    lh[423] = -1000000000;
    lh[424] = -1000000000;
    lh[425] = -1000000000;
    lh[426] = -1000000000;
    lh[427] = -1000000000;
    lh[428] = -1000000000;
    lh[429] = -1000000000;
    lh[430] = -1000000000;
    lh[431] = -1000000000;
    lh[432] = -1000000000;
    lh[433] = -1000000000;
    lh[434] = -1000000000;
    lh[435] = -1000000000;
    lh[436] = -1000000000;
    lh[437] = -1000000000;
    lh[438] = -1000000000;
    lh[439] = -1000000000;
    lh[440] = -1000000000;
    lh[441] = -1000000000;
    lh[442] = -1000000000;
    lh[443] = -1000000000;
    lh[444] = -1000000000;
    lh[445] = -1000000000;
    lh[446] = -1000000000;
    lh[447] = -1000000000;
    lh[448] = -1000000000;
    lh[449] = -1000000000;
    lh[450] = -1000000000;
    lh[451] = -1000000000;
    lh[452] = -1000000000;
    lh[453] = -1000000000;
    lh[454] = -1000000000;
    lh[455] = -1000000000;
    lh[456] = -1000000000;
    lh[457] = -1000000000;
    lh[458] = -1000000000;
    lh[459] = -1000000000;
    lh[460] = -1000000000;
    lh[461] = -1000000000;
    lh[462] = -1000000000;
    lh[463] = -1000000000;
    lh[464] = -1000000000;
    lh[465] = -1000000000;
    lh[466] = -1000000000;
    lh[467] = -1000000000;
    lh[468] = -1000000000;
    lh[469] = -1000000000;
    lh[470] = -1000000000;
    lh[471] = -1000000000;
    lh[472] = -1000000000;
    lh[473] = -1000000000;
    lh[474] = -1000000000;
    lh[475] = -1000000000;
    lh[476] = -1000000000;
    lh[477] = -1000000000;
    lh[478] = -1000000000;
    lh[479] = -1000000000;
    lh[480] = -1000000000;
    lh[481] = -1000000000;
    lh[482] = -1000000000;
    lh[483] = -1000000000;
    lh[484] = -1000000000;
    lh[485] = -1000000000;
    lh[486] = -1000000000;
    lh[487] = -1000000000;
    lh[488] = -1000000000;
    lh[489] = -1000000000;
    lh[490] = -1000000000;
    lh[491] = -1000000000;
    lh[492] = -1000000000;
    lh[493] = -1000000000;
    lh[494] = -1000000000;
    lh[495] = -1000000000;
    lh[496] = -1000000000;
    lh[497] = -1000000000;
    lh[498] = -1000000000;
    lh[499] = -1000000000;
    lh[500] = -1000000000;
    lh[501] = -1000000000;
    lh[502] = -1000000000;
    lh[503] = -1000000000;
    lh[504] = -1000000000;
    lh[505] = -1000000000;
    lh[506] = -1000000000;
    lh[507] = -1000000000;
    lh[508] = -1000000000;
    lh[509] = -1000000000;
    lh[510] = -1000000000;
    lh[511] = -1000000000;
    lh[512] = -1000000000;
    lh[513] = -1000000000;
    lh[514] = -1000000000;
    lh[515] = -1000000000;
    lh[516] = -1000000000;
    lh[517] = -1000000000;
    lh[518] = -1000000000;
    lh[519] = -1000000000;
    lh[520] = -1000000000;
    lh[521] = -1000000000;
    lh[522] = -1000000000;
    lh[523] = -1000000000;
    lh[524] = -1000000000;
    lh[525] = -1000000000;
    lh[526] = -1000000000;
    lh[527] = -1000000000;
    lh[528] = -1000000000;
    lh[529] = -1000000000;
    lh[530] = -1000000000;
    lh[531] = -1000000000;
    lh[532] = -1000000000;
    lh[533] = -1000000000;
    lh[534] = -1000000000;
    lh[535] = -1000000000;
    lh[536] = -1000000000;
    lh[537] = -1000000000;
    lh[538] = -1000000000;
    lh[539] = -1000000000;
    lh[540] = -1000000000;
    lh[541] = -1000000000;
    lh[542] = -1000000000;
    lh[543] = -1000000000;
    lh[544] = -1000000000;
    lh[545] = -1000000000;
    lh[546] = -1000000000;
    lh[547] = -1000000000;
    lh[548] = -1000000000;
    lh[549] = -1000000000;
    lh[550] = -1000000000;
    lh[551] = -1000000000;
    lh[552] = -1000000000;
    lh[553] = -1000000000;
    lh[554] = -1000000000;
    lh[555] = -1000000000;
    lh[556] = -1000000000;
    lh[557] = -1000000000;
    lh[558] = -1000000000;
    lh[559] = -1000000000;
    lh[560] = -1000000000;
    lh[561] = -1000000000;
    lh[562] = -1000000000;
    lh[563] = -1000000000;
    lh[564] = -1000000000;
    lh[565] = -1000000000;
    lh[566] = -1000000000;
    lh[567] = -1000000000;
    lh[568] = -1000000000;
    lh[569] = -1000000000;
    lh[570] = -1000000000;
    lh[571] = -1000000000;
    lh[572] = -1000000000;
    lh[573] = -1000000000;
    lh[574] = -1000000000;
    lh[575] = -1000000000;
    lh[576] = -1000000000;
    lh[577] = -1000000000;
    lh[578] = -1000000000;
    lh[579] = -1000000000;
    lh[580] = -1000000000;
    lh[581] = -1000000000;
    lh[582] = -1000000000;
    lh[583] = -1000000000;
    lh[584] = -1000000000;
    lh[585] = -1000000000;
    lh[586] = -1000000000;
    lh[587] = -1000000000;
    lh[588] = -1000000000;
    lh[589] = -1000000000;
    lh[590] = -1000000000;
    lh[591] = -1000000000;
    lh[592] = -1000000000;
    lh[593] = -1000000000;
    lh[594] = -1000000000;
    lh[595] = -1000000000;
    lh[596] = -1000000000;
    lh[597] = -1000000000;
    lh[598] = -1000000000;
    lh[599] = -1000000000;
    lh[600] = -1000000000;
    lh[601] = -1000000000;
    lh[602] = -1000000000;
    lh[603] = -1000000000;
    lh[604] = -1000000000;
    lh[605] = -1000000000;
    lh[606] = -1000000000;
    lh[607] = -1000000000;
    lh[608] = -1000000000;
    lh[609] = -1000000000;
    lh[610] = -1000000000;
    lh[611] = -1000000000;
    lh[612] = -1000000000;
    lh[613] = -1000000000;
    lh[614] = -1000000000;
    lh[615] = -1000000000;
    lh[616] = -1000000000;
    lh[617] = -1000000000;
    lh[618] = -1000000000;
    lh[619] = -1000000000;
    lh[620] = -1000000000;
    lh[621] = -1000000000;
    lh[622] = -1000000000;
    lh[623] = -1000000000;
    lh[624] = -1000000000;
    lh[625] = -1000000000;
    lh[626] = -1000000000;
    lh[627] = -1000000000;
    lh[628] = -1000000000;
    lh[629] = -1000000000;
    lh[630] = -1000000000;
    lh[631] = -1000000000;
    lh[632] = -1000000000;
    lh[633] = -1000000000;
    lh[634] = -1000000000;
    lh[635] = -1000000000;
    lh[636] = -1000000000;
    lh[637] = -1000000000;
    lh[638] = -1000000000;
    lh[639] = -1000000000;
    lh[640] = -1000000000;
    lh[641] = -1000000000;
    lh[642] = -1000000000;
    lh[643] = -1000000000;
    lh[644] = -1000000000;
    lh[645] = -1000000000;
    lh[646] = -1000000000;
    lh[647] = -1000000000;
    lh[648] = -1000000000;
    lh[649] = -1000000000;
    lh[650] = -1000000000;
    lh[651] = -1000000000;
    lh[652] = -1000000000;
    lh[653] = -1000000000;
    lh[654] = -1000000000;
    lh[655] = -1000000000;
    lh[656] = -1000000000;
    lh[657] = -1000000000;
    lh[658] = -1000000000;
    lh[659] = -1000000000;
    lh[660] = -1000000000;
    lh[661] = -1000000000;
    lh[662] = -1000000000;
    lh[663] = -1000000000;
    lh[664] = -1000000000;
    lh[665] = -1000000000;
    lh[666] = -1000000000;
    lh[667] = -1000000000;
    lh[668] = -1000000000;
    lh[669] = -1000000000;
    lh[670] = -1000000000;
    lh[671] = -1000000000;
    lh[672] = -1000000000;
    lh[673] = -1000000000;
    lh[674] = -1000000000;
    lh[675] = -1000000000;
    lh[676] = -1000000000;
    lh[677] = -1000000000;
    lh[678] = -1000000000;
    lh[679] = -1000000000;
    lh[680] = -1000000000;
    lh[681] = -1000000000;
    lh[682] = -1000000000;
    lh[683] = -1000000000;
    lh[684] = -1000000000;
    lh[685] = -1000000000;
    lh[686] = -1000000000;
    lh[687] = -1000000000;
    lh[688] = -1000000000;
    lh[689] = -1000000000;
    lh[690] = -1000000000;
    lh[691] = -1000000000;
    lh[692] = -1000000000;
    lh[693] = -1000000000;
    lh[694] = -1000000000;
    lh[695] = -1000000000;
    lh[696] = -1000000000;
    lh[697] = -1000000000;
    lh[698] = -1000000000;
    lh[699] = -1000000000;
    lh[700] = -1000000000;
    lh[701] = -1000000000;
    lh[702] = -1000000000;
    lh[703] = -1000000000;
    lh[704] = -1000000000;
    lh[705] = -1000000000;
    lh[706] = -1000000000;
    lh[707] = -1000000000;
    lh[708] = -1000000000;
    lh[709] = -1000000000;
    lh[710] = -1000000000;
    lh[711] = -1000000000;
    lh[712] = -1000000000;
    lh[713] = -1000000000;
    lh[714] = -1000000000;
    lh[715] = -1000000000;
    lh[716] = -1000000000;
    lh[717] = -1000000000;
    lh[718] = -1000000000;
    lh[719] = -1000000000;
    lh[720] = -1000000000;
    lh[721] = -1000000000;
    lh[722] = -1000000000;
    lh[723] = -1000000000;
    lh[724] = -1000000000;
    lh[725] = -1000000000;
    lh[726] = -1000000000;
    lh[727] = -1000000000;
    lh[728] = -1000000000;
    lh[729] = -1000000000;
    lh[730] = -1000000000;
    lh[731] = -1000000000;
    lh[732] = -1000000000;
    lh[733] = -1000000000;
    lh[734] = -1000000000;
    lh[735] = -1000000000;
    lh[736] = -1000000000;
    lh[737] = -1000000000;
    lh[738] = -1000000000;
    lh[739] = -1000000000;
    lh[740] = -1000000000;
    lh[741] = -1000000000;
    lh[742] = -1000000000;
    lh[743] = -1000000000;
    lh[744] = -1000000000;
    lh[745] = -1000000000;
    lh[746] = -1000000000;
    lh[747] = -1000000000;
    lh[748] = -1000000000;
    lh[749] = -1000000000;
    lh[750] = -1000000000;
    lh[751] = -1000000000;
    lh[752] = -1000000000;
    lh[753] = -1000000000;
    lh[754] = -1000000000;
    lh[755] = -1000000000;
    lh[756] = -1000000000;
    lh[757] = -1000000000;
    lh[758] = -1000000000;
    lh[759] = -1000000000;
    lh[760] = -1000000000;
    lh[761] = -1000000000;
    lh[762] = -1000000000;
    lh[763] = -1000000000;
    lh[764] = -1000000000;
    lh[765] = -1000000000;
    lh[766] = -1000000000;
    lh[767] = -1000000000;
    lh[768] = -1000000000;
    lh[769] = -1000000000;
    lh[770] = -1000000000;
    lh[771] = -1000000000;
    lh[772] = -1000000000;
    lh[773] = -1000000000;
    lh[774] = -1000000000;
    lh[775] = -1000000000;
    lh[776] = -1000000000;
    lh[777] = -1000000000;
    lh[778] = -1000000000;
    lh[779] = -1000000000;
    lh[780] = -1000000000;
    lh[781] = -1000000000;
    lh[782] = -1000000000;
    lh[783] = -1000000000;
    lh[784] = -1000000000;
    lh[785] = -1000000000;
    lh[786] = -1000000000;
    lh[787] = -1000000000;
    lh[788] = -1000000000;
    lh[789] = -1000000000;
    lh[790] = -1000000000;
    lh[791] = -1000000000;
    lh[792] = -1000000000;
    lh[793] = -1000000000;
    lh[794] = -1000000000;
    lh[795] = -1000000000;
    lh[796] = -1000000000;
    lh[797] = -1000000000;
    lh[798] = -1000000000;
    lh[799] = -1000000000;
    lh[800] = -1000000000;
    lh[801] = -1000000000;
    lh[802] = -1000000000;
    lh[803] = -1000000000;
    lh[804] = -1000000000;
    lh[805] = -1000000000;
    lh[806] = -1000000000;
    lh[807] = -1000000000;
    lh[808] = -1000000000;
    lh[809] = -1000000000;
    lh[810] = -1000000000;
    lh[811] = -1000000000;
    lh[812] = -1000000000;
    lh[813] = -1000000000;
    lh[814] = -1000000000;
    lh[815] = -1000000000;
    lh[816] = -1000000000;
    lh[817] = -1000000000;
    lh[818] = -1000000000;
    lh[819] = -1000000000;
    lh[820] = -1000000000;
    lh[821] = -1000000000;
    lh[822] = -1000000000;
    lh[823] = -1000000000;
    lh[824] = -1000000000;
    lh[825] = -1000000000;
    lh[826] = -1000000000;
    lh[827] = -1000000000;
    lh[828] = -1000000000;
    lh[829] = -1000000000;
    lh[830] = -1000000000;
    lh[831] = -1000000000;
    lh[832] = -1000000000;
    lh[833] = -1000000000;
    lh[834] = -1000000000;
    lh[835] = -1000000000;
    lh[836] = -1000000000;
    lh[837] = -1000000000;
    lh[838] = -1000000000;
    lh[839] = -1000000000;
    lh[840] = -1000000000;
    lh[841] = -1000000000;
    lh[842] = -1000000000;
    lh[843] = -1000000000;
    lh[844] = -1000000000;
    lh[845] = -1000000000;
    lh[846] = -1000000000;
    lh[847] = -1000000000;
    lh[848] = -1000000000;
    lh[849] = -1000000000;
    lh[850] = -1000000000;
    lh[851] = -1000000000;
    lh[852] = -1000000000;
    lh[853] = -1000000000;
    lh[854] = -1000000000;
    lh[855] = -1000000000;
    lh[856] = -1000000000;
    lh[857] = -1000000000;
    lh[858] = -1000000000;
    lh[859] = -1000000000;
    lh[860] = -1000000000;
    lh[861] = -1000000000;
    lh[862] = -1000000000;
    lh[863] = -1000000000;
    lh[864] = -1000000000;
    lh[865] = -1000000000;
    lh[866] = -1000000000;
    lh[867] = -1000000000;
    lh[868] = -1000000000;

    for (int i = 1; i < 2; i++)
    {
        i_fun = i - 1;
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "uh", uh);
    }
    free(luh);













    // TERMINAL node
    double* yref_e = calloc(43, sizeof(double));
    // change only the non-zero elements:
    yref_e[2] = 0.7779237030901003;
    yref_e[6] = 1;
    yref_e[7] = -0.2;
    yref_e[10] = 0.42;
    yref_e[11] = -0.22;
    yref_e[13] = -0.2;
    yref_e[16] = 0.42;
    yref_e[17] = -0.22;
    yref_e[22] = 0.25;
    yref_e[23] = 0.2;
    yref_e[25] = 0.9;
    yref_e[29] = 0.25;
    yref_e[30] = -0.2;
    yref_e[32] = 0.9;
    yref_e[36] = 0.45;
    yref_e[38] = 0.15;
    yref_e[42] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);
    free(yref_e);

    double* W_e = calloc(43*43, sizeof(double));
    // change only the non-zero elements:
    W_e[0+(43) * 0] = 1;
    W_e[1+(43) * 1] = 1;
    W_e[2+(43) * 2] = 1;
    W_e[3+(43) * 3] = 10;
    W_e[4+(43) * 4] = 10;
    W_e[5+(43) * 5] = 10;
    W_e[6+(43) * 6] = 10;
    W_e[7+(43) * 7] = 1;
    W_e[8+(43) * 8] = 1;
    W_e[9+(43) * 9] = 1;
    W_e[10+(43) * 10] = 1;
    W_e[11+(43) * 11] = 1;
    W_e[12+(43) * 12] = 1;
    W_e[13+(43) * 13] = 1;
    W_e[14+(43) * 14] = 1;
    W_e[15+(43) * 15] = 1;
    W_e[16+(43) * 16] = 1;
    W_e[17+(43) * 17] = 1;
    W_e[18+(43) * 18] = 1;
    W_e[19+(43) * 19] = 1;
    W_e[20+(43) * 20] = 1;
    W_e[21+(43) * 21] = 1;
    W_e[22+(43) * 22] = 1;
    W_e[23+(43) * 23] = 1;
    W_e[24+(43) * 24] = 1;
    W_e[25+(43) * 25] = 1;
    W_e[26+(43) * 26] = 1;
    W_e[27+(43) * 27] = 1;
    W_e[28+(43) * 28] = 1;
    W_e[29+(43) * 29] = 1;
    W_e[30+(43) * 30] = 1;
    W_e[31+(43) * 31] = 1;
    W_e[32+(43) * 32] = 1;
    W_e[33+(43) * 33] = 1;
    W_e[34+(43) * 34] = 1;
    W_e[35+(43) * 35] = 1;
    W_e[36+(43) * 36] = 1;
    W_e[37+(43) * 37] = 1;
    W_e[38+(43) * 38] = 1;
    W_e[39+(43) * 39] = 1;
    W_e[40+(43) * 40] = 1;
    W_e[41+(43) * 41] = 1;
    W_e[42+(43) * 42] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    free(W_e);
    double* Vx_e = calloc(43*43, sizeof(double));
    // change only the non-zero elements:
    Vx_e[0+(43) * 0] = 1;
    Vx_e[1+(43) * 1] = 1;
    Vx_e[2+(43) * 2] = 1;
    Vx_e[3+(43) * 3] = 1;
    Vx_e[4+(43) * 4] = 1;
    Vx_e[5+(43) * 5] = 1;
    Vx_e[6+(43) * 6] = 1;
    Vx_e[7+(43) * 7] = 1;
    Vx_e[8+(43) * 8] = 1;
    Vx_e[9+(43) * 9] = 1;
    Vx_e[10+(43) * 10] = 1;
    Vx_e[11+(43) * 11] = 1;
    Vx_e[12+(43) * 12] = 1;
    Vx_e[13+(43) * 13] = 1;
    Vx_e[14+(43) * 14] = 1;
    Vx_e[15+(43) * 15] = 1;
    Vx_e[16+(43) * 16] = 1;
    Vx_e[17+(43) * 17] = 1;
    Vx_e[18+(43) * 18] = 1;
    Vx_e[19+(43) * 19] = 1;
    Vx_e[20+(43) * 20] = 1;
    Vx_e[21+(43) * 21] = 1;
    Vx_e[22+(43) * 22] = 1;
    Vx_e[23+(43) * 23] = 1;
    Vx_e[24+(43) * 24] = 1;
    Vx_e[25+(43) * 25] = 1;
    Vx_e[26+(43) * 26] = 1;
    Vx_e[27+(43) * 27] = 1;
    Vx_e[28+(43) * 28] = 1;
    Vx_e[29+(43) * 29] = 1;
    Vx_e[30+(43) * 30] = 1;
    Vx_e[31+(43) * 31] = 1;
    Vx_e[32+(43) * 32] = 1;
    Vx_e[33+(43) * 33] = 1;
    Vx_e[34+(43) * 34] = 1;
    Vx_e[35+(43) * 35] = 1;
    Vx_e[36+(43) * 36] = 1;
    Vx_e[37+(43) * 37] = 1;
    Vx_e[38+(43) * 38] = 1;
    Vx_e[39+(43) * 39] = 1;
    Vx_e[40+(43) * 40] = 1;
    Vx_e[41+(43) * 41] = 1;
    Vx_e[42+(43) * 42] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Vx", Vx_e);
    free(Vx_e);




    /* terminal constraints */






    // set up nonlinear constraints for last stage
    double* luh_e = calloc(2*849, sizeof(double));
    double* lh_e = luh_e;
    double* uh_e = luh_e + 849;
    lh_e[17] = -1000000000;
    lh_e[18] = -1000000000;
    lh_e[19] = -1000000000;
    lh_e[20] = -1000000000;
    lh_e[21] = -1000000000;
    lh_e[22] = -1000000000;
    lh_e[23] = -1000000000;
    lh_e[24] = -1000000000;
    lh_e[25] = -1000000000;
    lh_e[26] = -1000000000;
    lh_e[27] = -1000000000;
    lh_e[28] = -1000000000;
    lh_e[29] = -1000000000;
    lh_e[30] = -1000000000;
    lh_e[31] = -1000000000;
    lh_e[32] = -1000000000;
    lh_e[33] = -1000000000;
    lh_e[34] = -1000000000;
    lh_e[35] = -1000000000;
    lh_e[36] = -1000000000;
    lh_e[37] = -1000000000;
    lh_e[38] = -1000000000;
    lh_e[39] = -1000000000;
    lh_e[40] = -1000000000;
    lh_e[41] = -1000000000;
    lh_e[42] = -1000000000;
    lh_e[43] = -1000000000;
    lh_e[44] = -1000000000;
    lh_e[45] = -1000000000;
    lh_e[46] = -1000000000;
    lh_e[47] = -1000000000;
    lh_e[48] = -1000000000;
    lh_e[49] = -1000000000;
    lh_e[50] = -1000000000;
    lh_e[51] = -1000000000;
    lh_e[52] = -1000000000;
    lh_e[53] = -1000000000;
    lh_e[54] = -1000000000;
    lh_e[55] = -1000000000;
    lh_e[56] = -1000000000;
    lh_e[57] = -1000000000;
    lh_e[58] = -1000000000;
    lh_e[59] = -1000000000;
    lh_e[60] = -1000000000;
    lh_e[61] = -1000000000;
    lh_e[62] = -1000000000;
    lh_e[63] = -1000000000;
    lh_e[64] = -1000000000;
    lh_e[65] = -1000000000;
    lh_e[66] = -1000000000;
    lh_e[67] = -1000000000;
    lh_e[68] = -1000000000;
    lh_e[69] = -1000000000;
    lh_e[70] = -1000000000;
    lh_e[71] = -1000000000;
    lh_e[72] = -1000000000;
    lh_e[73] = -1000000000;
    lh_e[74] = -1000000000;
    lh_e[75] = -1000000000;
    lh_e[76] = -1000000000;
    lh_e[77] = -1000000000;
    lh_e[78] = -1000000000;
    lh_e[79] = -1000000000;
    lh_e[80] = -1000000000;
    lh_e[81] = -1000000000;
    lh_e[82] = -1000000000;
    lh_e[83] = -1000000000;
    lh_e[84] = -1000000000;
    lh_e[85] = -1000000000;
    lh_e[86] = -1000000000;
    lh_e[87] = -1000000000;
    lh_e[88] = -1000000000;
    lh_e[89] = -1000000000;
    lh_e[90] = -1000000000;
    lh_e[91] = -1000000000;
    lh_e[92] = -1000000000;
    lh_e[93] = -1000000000;
    lh_e[94] = -1000000000;
    lh_e[95] = -1000000000;
    lh_e[96] = -1000000000;
    lh_e[97] = -1000000000;
    lh_e[98] = -1000000000;
    lh_e[99] = -1000000000;
    lh_e[100] = -1000000000;
    lh_e[101] = -1000000000;
    lh_e[102] = -1000000000;
    lh_e[103] = -1000000000;
    lh_e[104] = -1000000000;
    lh_e[105] = -1000000000;
    lh_e[106] = -1000000000;
    lh_e[107] = -1000000000;
    lh_e[108] = -1000000000;
    lh_e[109] = -1000000000;
    lh_e[110] = -1000000000;
    lh_e[111] = -1000000000;
    lh_e[112] = -1000000000;
    lh_e[113] = -1000000000;
    lh_e[114] = -1000000000;
    lh_e[115] = -1000000000;
    lh_e[116] = -1000000000;
    lh_e[117] = -1000000000;
    lh_e[118] = -1000000000;
    lh_e[119] = -1000000000;
    lh_e[120] = -1000000000;
    lh_e[121] = -1000000000;
    lh_e[122] = -1000000000;
    lh_e[123] = -1000000000;
    lh_e[124] = -1000000000;
    lh_e[125] = -1000000000;
    lh_e[126] = -1000000000;
    lh_e[127] = -1000000000;
    lh_e[128] = -1000000000;
    lh_e[129] = -1000000000;
    lh_e[130] = -1000000000;
    lh_e[131] = -1000000000;
    lh_e[132] = -1000000000;
    lh_e[133] = -1000000000;
    lh_e[134] = -1000000000;
    lh_e[135] = -1000000000;
    lh_e[136] = -1000000000;
    lh_e[137] = -1000000000;
    lh_e[138] = -1000000000;
    lh_e[139] = -1000000000;
    lh_e[140] = -1000000000;
    lh_e[141] = -1000000000;
    lh_e[142] = -1000000000;
    lh_e[143] = -1000000000;
    lh_e[144] = -1000000000;
    lh_e[145] = -1000000000;
    lh_e[146] = -1000000000;
    lh_e[147] = -1000000000;
    lh_e[148] = -1000000000;
    lh_e[149] = -1000000000;
    lh_e[150] = -1000000000;
    lh_e[151] = -1000000000;
    lh_e[152] = -1000000000;
    lh_e[153] = -1000000000;
    lh_e[154] = -1000000000;
    lh_e[155] = -1000000000;
    lh_e[156] = -1000000000;
    lh_e[157] = -1000000000;
    lh_e[158] = -1000000000;
    lh_e[159] = -1000000000;
    lh_e[160] = -1000000000;
    lh_e[161] = -1000000000;
    lh_e[162] = -1000000000;
    lh_e[163] = -1000000000;
    lh_e[164] = -1000000000;
    lh_e[165] = -1000000000;
    lh_e[166] = -1000000000;
    lh_e[167] = -1000000000;
    lh_e[168] = -1000000000;
    lh_e[169] = -1000000000;
    lh_e[170] = -1000000000;
    lh_e[171] = -1000000000;
    lh_e[172] = -1000000000;
    lh_e[173] = -1000000000;
    lh_e[174] = -1000000000;
    lh_e[175] = -1000000000;
    lh_e[176] = -1000000000;
    lh_e[177] = -1000000000;
    lh_e[178] = -1000000000;
    lh_e[179] = -1000000000;
    lh_e[180] = -1000000000;
    lh_e[181] = -1000000000;
    lh_e[182] = -1000000000;
    lh_e[183] = -1000000000;
    lh_e[184] = -1000000000;
    lh_e[185] = -1000000000;
    lh_e[186] = -1000000000;
    lh_e[187] = -1000000000;
    lh_e[188] = -1000000000;
    lh_e[189] = -1000000000;
    lh_e[190] = -1000000000;
    lh_e[191] = -1000000000;
    lh_e[192] = -1000000000;
    lh_e[193] = -1000000000;
    lh_e[194] = -1000000000;
    lh_e[195] = -1000000000;
    lh_e[196] = -1000000000;
    lh_e[197] = -1000000000;
    lh_e[198] = -1000000000;
    lh_e[199] = -1000000000;
    lh_e[200] = -1000000000;
    lh_e[201] = -1000000000;
    lh_e[202] = -1000000000;
    lh_e[203] = -1000000000;
    lh_e[204] = -1000000000;
    lh_e[205] = -1000000000;
    lh_e[206] = -1000000000;
    lh_e[207] = -1000000000;
    lh_e[208] = -1000000000;
    lh_e[209] = -1000000000;
    lh_e[210] = -1000000000;
    lh_e[211] = -1000000000;
    lh_e[212] = -1000000000;
    lh_e[213] = -1000000000;
    lh_e[214] = -1000000000;
    lh_e[215] = -1000000000;
    lh_e[216] = -1000000000;
    lh_e[217] = -1000000000;
    lh_e[218] = -1000000000;
    lh_e[219] = -1000000000;
    lh_e[220] = -1000000000;
    lh_e[221] = -1000000000;
    lh_e[222] = -1000000000;
    lh_e[223] = -1000000000;
    lh_e[224] = -1000000000;
    lh_e[225] = -1000000000;
    lh_e[226] = -1000000000;
    lh_e[227] = -1000000000;
    lh_e[228] = -1000000000;
    lh_e[229] = -1000000000;
    lh_e[230] = -1000000000;
    lh_e[231] = -1000000000;
    lh_e[232] = -1000000000;
    lh_e[233] = -1000000000;
    lh_e[234] = -1000000000;
    lh_e[235] = -1000000000;
    lh_e[236] = -1000000000;
    lh_e[237] = -1000000000;
    lh_e[238] = -1000000000;
    lh_e[239] = -1000000000;
    lh_e[240] = -1000000000;
    lh_e[241] = -1000000000;
    lh_e[242] = -1000000000;
    lh_e[243] = -1000000000;
    lh_e[244] = -1000000000;
    lh_e[245] = -1000000000;
    lh_e[246] = -1000000000;
    lh_e[247] = -1000000000;
    lh_e[248] = -1000000000;
    lh_e[249] = -1000000000;
    lh_e[250] = -1000000000;
    lh_e[251] = -1000000000;
    lh_e[252] = -1000000000;
    lh_e[253] = -1000000000;
    lh_e[254] = -1000000000;
    lh_e[255] = -1000000000;
    lh_e[256] = -1000000000;
    lh_e[257] = -1000000000;
    lh_e[258] = -1000000000;
    lh_e[259] = -1000000000;
    lh_e[260] = -1000000000;
    lh_e[261] = -1000000000;
    lh_e[262] = -1000000000;
    lh_e[263] = -1000000000;
    lh_e[264] = -1000000000;
    lh_e[265] = -1000000000;
    lh_e[266] = -1000000000;
    lh_e[267] = -1000000000;
    lh_e[268] = -1000000000;
    lh_e[269] = -1000000000;
    lh_e[270] = -1000000000;
    lh_e[271] = -1000000000;
    lh_e[272] = -1000000000;
    lh_e[273] = -1000000000;
    lh_e[274] = -1000000000;
    lh_e[275] = -1000000000;
    lh_e[276] = -1000000000;
    lh_e[277] = -1000000000;
    lh_e[278] = -1000000000;
    lh_e[279] = -1000000000;
    lh_e[280] = -1000000000;
    lh_e[281] = -1000000000;
    lh_e[282] = -1000000000;
    lh_e[283] = -1000000000;
    lh_e[284] = -1000000000;
    lh_e[285] = -1000000000;
    lh_e[286] = -1000000000;
    lh_e[287] = -1000000000;
    lh_e[288] = -1000000000;
    lh_e[289] = -1000000000;
    lh_e[290] = -1000000000;
    lh_e[291] = -1000000000;
    lh_e[292] = -1000000000;
    lh_e[293] = -1000000000;
    lh_e[294] = -1000000000;
    lh_e[295] = -1000000000;
    lh_e[296] = -1000000000;
    lh_e[297] = -1000000000;
    lh_e[298] = -1000000000;
    lh_e[299] = -1000000000;
    lh_e[300] = -1000000000;
    lh_e[301] = -1000000000;
    lh_e[302] = -1000000000;
    lh_e[303] = -1000000000;
    lh_e[304] = -1000000000;
    lh_e[305] = -1000000000;
    lh_e[306] = -1000000000;
    lh_e[307] = -1000000000;
    lh_e[308] = -1000000000;
    lh_e[309] = -1000000000;
    lh_e[310] = -1000000000;
    lh_e[311] = -1000000000;
    lh_e[312] = -1000000000;
    lh_e[313] = -1000000000;
    lh_e[314] = -1000000000;
    lh_e[315] = -1000000000;
    lh_e[316] = -1000000000;
    lh_e[317] = -1000000000;
    lh_e[318] = -1000000000;
    lh_e[319] = -1000000000;
    lh_e[320] = -1000000000;
    lh_e[321] = -1000000000;
    lh_e[322] = -1000000000;
    lh_e[323] = -1000000000;
    lh_e[324] = -1000000000;
    lh_e[325] = -1000000000;
    lh_e[326] = -1000000000;
    lh_e[327] = -1000000000;
    lh_e[328] = -1000000000;
    lh_e[329] = -1000000000;
    lh_e[330] = -1000000000;
    lh_e[331] = -1000000000;
    lh_e[332] = -1000000000;
    lh_e[333] = -1000000000;
    lh_e[334] = -1000000000;
    lh_e[335] = -1000000000;
    lh_e[336] = -1000000000;
    lh_e[337] = -1000000000;
    lh_e[338] = -1000000000;
    lh_e[339] = -1000000000;
    lh_e[340] = -1000000000;
    lh_e[341] = -1000000000;
    lh_e[342] = -1000000000;
    lh_e[343] = -1000000000;
    lh_e[344] = -1000000000;
    lh_e[345] = -1000000000;
    lh_e[346] = -1000000000;
    lh_e[347] = -1000000000;
    lh_e[348] = -1000000000;
    lh_e[349] = -1000000000;
    lh_e[350] = -1000000000;
    lh_e[351] = -1000000000;
    lh_e[352] = -1000000000;
    lh_e[353] = -1000000000;
    lh_e[354] = -1000000000;
    lh_e[355] = -1000000000;
    lh_e[356] = -1000000000;
    lh_e[357] = -1000000000;
    lh_e[358] = -1000000000;
    lh_e[359] = -1000000000;
    lh_e[360] = -1000000000;
    lh_e[361] = -1000000000;
    lh_e[362] = -1000000000;
    lh_e[363] = -1000000000;
    lh_e[364] = -1000000000;
    lh_e[365] = -1000000000;
    lh_e[366] = -1000000000;
    lh_e[367] = -1000000000;
    lh_e[368] = -1000000000;
    lh_e[369] = -1000000000;
    lh_e[370] = -1000000000;
    lh_e[371] = -1000000000;
    lh_e[372] = -1000000000;
    lh_e[373] = -1000000000;
    lh_e[374] = -1000000000;
    lh_e[375] = -1000000000;
    lh_e[376] = -1000000000;
    lh_e[377] = -1000000000;
    lh_e[378] = -1000000000;
    lh_e[379] = -1000000000;
    lh_e[380] = -1000000000;
    lh_e[381] = -1000000000;
    lh_e[382] = -1000000000;
    lh_e[383] = -1000000000;
    lh_e[384] = -1000000000;
    lh_e[385] = -1000000000;
    lh_e[386] = -1000000000;
    lh_e[387] = -1000000000;
    lh_e[388] = -1000000000;
    lh_e[389] = -1000000000;
    lh_e[390] = -1000000000;
    lh_e[391] = -1000000000;
    lh_e[392] = -1000000000;
    lh_e[393] = -1000000000;
    lh_e[394] = -1000000000;
    lh_e[395] = -1000000000;
    lh_e[396] = -1000000000;
    lh_e[397] = -1000000000;
    lh_e[398] = -1000000000;
    lh_e[399] = -1000000000;
    lh_e[400] = -1000000000;
    lh_e[401] = -1000000000;
    lh_e[402] = -1000000000;
    lh_e[403] = -1000000000;
    lh_e[404] = -1000000000;
    lh_e[405] = -1000000000;
    lh_e[406] = -1000000000;
    lh_e[407] = -1000000000;
    lh_e[408] = -1000000000;
    lh_e[409] = -1000000000;
    lh_e[410] = -1000000000;
    lh_e[411] = -1000000000;
    lh_e[412] = -1000000000;
    lh_e[413] = -1000000000;
    lh_e[414] = -1000000000;
    lh_e[415] = -1000000000;
    lh_e[416] = -1000000000;
    lh_e[417] = -1000000000;
    lh_e[418] = -1000000000;
    lh_e[419] = -1000000000;
    lh_e[420] = -1000000000;
    lh_e[421] = -1000000000;
    lh_e[422] = -1000000000;
    lh_e[423] = -1000000000;
    lh_e[424] = -1000000000;
    lh_e[425] = -1000000000;
    lh_e[426] = -1000000000;
    lh_e[427] = -1000000000;
    lh_e[428] = -1000000000;
    lh_e[429] = -1000000000;
    lh_e[430] = -1000000000;
    lh_e[431] = -1000000000;
    lh_e[432] = -1000000000;
    lh_e[433] = -1000000000;
    lh_e[434] = -1000000000;
    lh_e[435] = -1000000000;
    lh_e[436] = -1000000000;
    lh_e[437] = -1000000000;
    lh_e[438] = -1000000000;
    lh_e[439] = -1000000000;
    lh_e[440] = -1000000000;
    lh_e[441] = -1000000000;
    lh_e[442] = -1000000000;
    lh_e[443] = -1000000000;
    lh_e[444] = -1000000000;
    lh_e[445] = -1000000000;
    lh_e[446] = -1000000000;
    lh_e[447] = -1000000000;
    lh_e[448] = -1000000000;
    lh_e[449] = -1000000000;
    lh_e[450] = -1000000000;
    lh_e[451] = -1000000000;
    lh_e[452] = -1000000000;
    lh_e[453] = -1000000000;
    lh_e[454] = -1000000000;
    lh_e[455] = -1000000000;
    lh_e[456] = -1000000000;
    lh_e[457] = -1000000000;
    lh_e[458] = -1000000000;
    lh_e[459] = -1000000000;
    lh_e[460] = -1000000000;
    lh_e[461] = -1000000000;
    lh_e[462] = -1000000000;
    lh_e[463] = -1000000000;
    lh_e[464] = -1000000000;
    lh_e[465] = -1000000000;
    lh_e[466] = -1000000000;
    lh_e[467] = -1000000000;
    lh_e[468] = -1000000000;
    lh_e[469] = -1000000000;
    lh_e[470] = -1000000000;
    lh_e[471] = -1000000000;
    lh_e[472] = -1000000000;
    lh_e[473] = -1000000000;
    lh_e[474] = -1000000000;
    lh_e[475] = -1000000000;
    lh_e[476] = -1000000000;
    lh_e[477] = -1000000000;
    lh_e[478] = -1000000000;
    lh_e[479] = -1000000000;
    lh_e[480] = -1000000000;
    lh_e[481] = -1000000000;
    lh_e[482] = -1000000000;
    lh_e[483] = -1000000000;
    lh_e[484] = -1000000000;
    lh_e[485] = -1000000000;
    lh_e[486] = -1000000000;
    lh_e[487] = -1000000000;
    lh_e[488] = -1000000000;
    lh_e[489] = -1000000000;
    lh_e[490] = -1000000000;
    lh_e[491] = -1000000000;
    lh_e[492] = -1000000000;
    lh_e[493] = -1000000000;
    lh_e[494] = -1000000000;
    lh_e[495] = -1000000000;
    lh_e[496] = -1000000000;
    lh_e[497] = -1000000000;
    lh_e[498] = -1000000000;
    lh_e[499] = -1000000000;
    lh_e[500] = -1000000000;
    lh_e[501] = -1000000000;
    lh_e[502] = -1000000000;
    lh_e[503] = -1000000000;
    lh_e[504] = -1000000000;
    lh_e[505] = -1000000000;
    lh_e[506] = -1000000000;
    lh_e[507] = -1000000000;
    lh_e[508] = -1000000000;
    lh_e[509] = -1000000000;
    lh_e[510] = -1000000000;
    lh_e[511] = -1000000000;
    lh_e[512] = -1000000000;
    lh_e[513] = -1000000000;
    lh_e[514] = -1000000000;
    lh_e[515] = -1000000000;
    lh_e[516] = -1000000000;
    lh_e[517] = -1000000000;
    lh_e[518] = -1000000000;
    lh_e[519] = -1000000000;
    lh_e[520] = -1000000000;
    lh_e[521] = -1000000000;
    lh_e[522] = -1000000000;
    lh_e[523] = -1000000000;
    lh_e[524] = -1000000000;
    lh_e[525] = -1000000000;
    lh_e[526] = -1000000000;
    lh_e[527] = -1000000000;
    lh_e[528] = -1000000000;
    lh_e[529] = -1000000000;
    lh_e[530] = -1000000000;
    lh_e[531] = -1000000000;
    lh_e[532] = -1000000000;
    lh_e[533] = -1000000000;
    lh_e[534] = -1000000000;
    lh_e[535] = -1000000000;
    lh_e[536] = -1000000000;
    lh_e[537] = -1000000000;
    lh_e[538] = -1000000000;
    lh_e[539] = -1000000000;
    lh_e[540] = -1000000000;
    lh_e[541] = -1000000000;
    lh_e[542] = -1000000000;
    lh_e[543] = -1000000000;
    lh_e[544] = -1000000000;
    lh_e[545] = -1000000000;
    lh_e[546] = -1000000000;
    lh_e[547] = -1000000000;
    lh_e[548] = -1000000000;
    lh_e[549] = -1000000000;
    lh_e[550] = -1000000000;
    lh_e[551] = -1000000000;
    lh_e[552] = -1000000000;
    lh_e[553] = -1000000000;
    lh_e[554] = -1000000000;
    lh_e[555] = -1000000000;
    lh_e[556] = -1000000000;
    lh_e[557] = -1000000000;
    lh_e[558] = -1000000000;
    lh_e[559] = -1000000000;
    lh_e[560] = -1000000000;
    lh_e[561] = -1000000000;
    lh_e[562] = -1000000000;
    lh_e[563] = -1000000000;
    lh_e[564] = -1000000000;
    lh_e[565] = -1000000000;
    lh_e[566] = -1000000000;
    lh_e[567] = -1000000000;
    lh_e[568] = -1000000000;
    lh_e[569] = -1000000000;
    lh_e[570] = -1000000000;
    lh_e[571] = -1000000000;
    lh_e[572] = -1000000000;
    lh_e[573] = -1000000000;
    lh_e[574] = -1000000000;
    lh_e[575] = -1000000000;
    lh_e[576] = -1000000000;
    lh_e[577] = -1000000000;
    lh_e[578] = -1000000000;
    lh_e[579] = -1000000000;
    lh_e[580] = -1000000000;
    lh_e[581] = -1000000000;
    lh_e[582] = -1000000000;
    lh_e[583] = -1000000000;
    lh_e[584] = -1000000000;
    lh_e[585] = -1000000000;
    lh_e[586] = -1000000000;
    lh_e[587] = -1000000000;
    lh_e[588] = -1000000000;
    lh_e[589] = -1000000000;
    lh_e[590] = -1000000000;
    lh_e[591] = -1000000000;
    lh_e[592] = -1000000000;
    lh_e[593] = -1000000000;
    lh_e[594] = -1000000000;
    lh_e[595] = -1000000000;
    lh_e[596] = -1000000000;
    lh_e[597] = -1000000000;
    lh_e[598] = -1000000000;
    lh_e[599] = -1000000000;
    lh_e[600] = -1000000000;
    lh_e[601] = -1000000000;
    lh_e[602] = -1000000000;
    lh_e[603] = -1000000000;
    lh_e[604] = -1000000000;
    lh_e[605] = -1000000000;
    lh_e[606] = -1000000000;
    lh_e[607] = -1000000000;
    lh_e[608] = -1000000000;
    lh_e[609] = -1000000000;
    lh_e[610] = -1000000000;
    lh_e[611] = -1000000000;
    lh_e[612] = -1000000000;
    lh_e[613] = -1000000000;
    lh_e[614] = -1000000000;
    lh_e[615] = -1000000000;
    lh_e[616] = -1000000000;
    lh_e[617] = -1000000000;
    lh_e[618] = -1000000000;
    lh_e[619] = -1000000000;
    lh_e[620] = -1000000000;
    lh_e[621] = -1000000000;
    lh_e[622] = -1000000000;
    lh_e[623] = -1000000000;
    lh_e[624] = -1000000000;
    lh_e[625] = -1000000000;
    lh_e[626] = -1000000000;
    lh_e[627] = -1000000000;
    lh_e[628] = -1000000000;
    lh_e[629] = -1000000000;
    lh_e[630] = -1000000000;
    lh_e[631] = -1000000000;
    lh_e[632] = -1000000000;
    lh_e[633] = -1000000000;
    lh_e[634] = -1000000000;
    lh_e[635] = -1000000000;
    lh_e[636] = -1000000000;
    lh_e[637] = -1000000000;
    lh_e[638] = -1000000000;
    lh_e[639] = -1000000000;
    lh_e[640] = -1000000000;
    lh_e[641] = -1000000000;
    lh_e[642] = -1000000000;
    lh_e[643] = -1000000000;
    lh_e[644] = -1000000000;
    lh_e[645] = -1000000000;
    lh_e[646] = -1000000000;
    lh_e[647] = -1000000000;
    lh_e[648] = -1000000000;
    lh_e[649] = -1000000000;
    lh_e[650] = -1000000000;
    lh_e[651] = -1000000000;
    lh_e[652] = -1000000000;
    lh_e[653] = -1000000000;
    lh_e[654] = -1000000000;
    lh_e[655] = -1000000000;
    lh_e[656] = -1000000000;
    lh_e[657] = -1000000000;
    lh_e[658] = -1000000000;
    lh_e[659] = -1000000000;
    lh_e[660] = -1000000000;
    lh_e[661] = -1000000000;
    lh_e[662] = -1000000000;
    lh_e[663] = -1000000000;
    lh_e[664] = -1000000000;
    lh_e[665] = -1000000000;
    lh_e[666] = -1000000000;
    lh_e[667] = -1000000000;
    lh_e[668] = -1000000000;
    lh_e[669] = -1000000000;
    lh_e[670] = -1000000000;
    lh_e[671] = -1000000000;
    lh_e[672] = -1000000000;
    lh_e[673] = -1000000000;
    lh_e[674] = -1000000000;
    lh_e[675] = -1000000000;
    lh_e[676] = -1000000000;
    lh_e[677] = -1000000000;
    lh_e[678] = -1000000000;
    lh_e[679] = -1000000000;
    lh_e[680] = -1000000000;
    lh_e[681] = -1000000000;
    lh_e[682] = -1000000000;
    lh_e[683] = -1000000000;
    lh_e[684] = -1000000000;
    lh_e[685] = -1000000000;
    lh_e[686] = -1000000000;
    lh_e[687] = -1000000000;
    lh_e[688] = -1000000000;
    lh_e[689] = -1000000000;
    lh_e[690] = -1000000000;
    lh_e[691] = -1000000000;
    lh_e[692] = -1000000000;
    lh_e[693] = -1000000000;
    lh_e[694] = -1000000000;
    lh_e[695] = -1000000000;
    lh_e[696] = -1000000000;
    lh_e[697] = -1000000000;
    lh_e[698] = -1000000000;
    lh_e[699] = -1000000000;
    lh_e[700] = -1000000000;
    lh_e[701] = -1000000000;
    lh_e[702] = -1000000000;
    lh_e[703] = -1000000000;
    lh_e[704] = -1000000000;
    lh_e[705] = -1000000000;
    lh_e[706] = -1000000000;
    lh_e[707] = -1000000000;
    lh_e[708] = -1000000000;
    lh_e[709] = -1000000000;
    lh_e[710] = -1000000000;
    lh_e[711] = -1000000000;
    lh_e[712] = -1000000000;
    lh_e[713] = -1000000000;
    lh_e[714] = -1000000000;
    lh_e[715] = -1000000000;
    lh_e[716] = -1000000000;
    lh_e[717] = -1000000000;
    lh_e[718] = -1000000000;
    lh_e[719] = -1000000000;
    lh_e[720] = -1000000000;
    lh_e[721] = -1000000000;
    lh_e[722] = -1000000000;
    lh_e[723] = -1000000000;
    lh_e[724] = -1000000000;
    lh_e[725] = -1000000000;
    lh_e[726] = -1000000000;
    lh_e[727] = -1000000000;
    lh_e[728] = -1000000000;
    lh_e[729] = -1000000000;
    lh_e[730] = -1000000000;
    lh_e[731] = -1000000000;
    lh_e[732] = -1000000000;
    lh_e[733] = -1000000000;
    lh_e[734] = -1000000000;
    lh_e[735] = -1000000000;
    lh_e[736] = -1000000000;
    lh_e[737] = -1000000000;
    lh_e[738] = -1000000000;
    lh_e[739] = -1000000000;
    lh_e[740] = -1000000000;
    lh_e[741] = -1000000000;
    lh_e[742] = -1000000000;
    lh_e[743] = -1000000000;
    lh_e[744] = -1000000000;
    lh_e[745] = -1000000000;
    lh_e[746] = -1000000000;
    lh_e[747] = -1000000000;
    lh_e[748] = -1000000000;
    lh_e[749] = -1000000000;
    lh_e[750] = -1000000000;
    lh_e[751] = -1000000000;
    lh_e[752] = -1000000000;
    lh_e[753] = -1000000000;
    lh_e[754] = -1000000000;
    lh_e[755] = -1000000000;
    lh_e[756] = -1000000000;
    lh_e[757] = -1000000000;
    lh_e[758] = -1000000000;
    lh_e[759] = -1000000000;
    lh_e[760] = -1000000000;
    lh_e[761] = -1000000000;
    lh_e[762] = -1000000000;
    lh_e[763] = -1000000000;
    lh_e[764] = -1000000000;
    lh_e[765] = -1000000000;
    lh_e[766] = -1000000000;
    lh_e[767] = -1000000000;
    lh_e[768] = -1000000000;
    lh_e[769] = -1000000000;
    lh_e[770] = -1000000000;
    lh_e[771] = -1000000000;
    lh_e[772] = -1000000000;
    lh_e[773] = -1000000000;
    lh_e[774] = -1000000000;
    lh_e[775] = -1000000000;
    lh_e[776] = -1000000000;
    lh_e[777] = -1000000000;
    lh_e[778] = -1000000000;
    lh_e[779] = -1000000000;
    lh_e[780] = -1000000000;
    lh_e[781] = -1000000000;
    lh_e[782] = -1000000000;
    lh_e[783] = -1000000000;
    lh_e[784] = -1000000000;
    lh_e[785] = -1000000000;
    lh_e[786] = -1000000000;
    lh_e[787] = -1000000000;
    lh_e[788] = -1000000000;
    lh_e[789] = -1000000000;
    lh_e[790] = -1000000000;
    lh_e[791] = -1000000000;
    lh_e[792] = -1000000000;
    lh_e[793] = -1000000000;
    lh_e[794] = -1000000000;
    lh_e[795] = -1000000000;
    lh_e[796] = -1000000000;
    lh_e[797] = -1000000000;
    lh_e[798] = -1000000000;
    lh_e[799] = -1000000000;
    lh_e[800] = -1000000000;
    lh_e[801] = -1000000000;
    lh_e[802] = -1000000000;
    lh_e[803] = -1000000000;
    lh_e[804] = -1000000000;
    lh_e[805] = -1000000000;
    lh_e[806] = -1000000000;
    lh_e[807] = -1000000000;
    lh_e[808] = -1000000000;
    lh_e[809] = -1000000000;
    lh_e[810] = -1000000000;
    lh_e[811] = -1000000000;
    lh_e[812] = -1000000000;
    lh_e[813] = -1000000000;
    lh_e[814] = -1000000000;
    lh_e[815] = -1000000000;
    lh_e[816] = -1000000000;
    lh_e[817] = -1000000000;
    lh_e[818] = -1000000000;
    lh_e[819] = -1000000000;
    lh_e[820] = -1000000000;
    lh_e[821] = -1000000000;
    lh_e[822] = -1000000000;
    lh_e[823] = -1000000000;
    lh_e[824] = -1000000000;
    lh_e[825] = -1000000000;
    lh_e[826] = -1000000000;
    lh_e[827] = -1000000000;
    lh_e[828] = -1000000000;
    lh_e[829] = -1000000000;
    lh_e[830] = -1000000000;
    lh_e[831] = -1000000000;
    lh_e[832] = -1000000000;
    lh_e[833] = -1000000000;
    lh_e[834] = -1000000000;
    lh_e[835] = -1000000000;
    lh_e[836] = -1000000000;
    lh_e[837] = -1000000000;
    lh_e[838] = -1000000000;
    lh_e[839] = -1000000000;
    lh_e[840] = -1000000000;
    lh_e[841] = -1000000000;
    lh_e[842] = -1000000000;
    lh_e[843] = -1000000000;
    lh_e[844] = -1000000000;
    lh_e[845] = -1000000000;
    lh_e[846] = -1000000000;
    lh_e[847] = -1000000000;
    lh_e[848] = -1000000000;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "lh", lh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "uh", uh_e);
    free(luh_e);

    /* terminal soft constraints */













}




void multiphase_ocp_acados_create_setup_nlp_in(multiphase_ocp_solver_capsule* capsule, int N)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;

    /* INITIAL NODE */


    // set up nonlinear constraints for first stage
    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nl_constr_h_fun_jac", &capsule->nl_constr_h_0_fun_jac);
    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nl_constr_h_fun", &capsule->nl_constr_h_0_fun);
    
    
    

    /* Path related delarations */
    int i_fun;

    /*********************
     *  Phase 0
     * *******************/
    /**** Dynamics ****/
    for (int i = 0; i < 1; i++)
    {
        i_fun = i - 0;
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun", &capsule->discr_dyn_phi_fun_0[i_fun]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun_jac",
                                   &capsule->discr_dyn_phi_fun_jac_ut_xt_0[i_fun]);
    }

    /**** Constraints phase 0 ****/





    /*********************
     *  Phase 1
     * *******************/
    /**** Dynamics ****/
    for (int i = 1; i < 2; i++)
    {
        i_fun = i - 1;
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun", &capsule->discr_dyn_phi_fun_1[i_fun]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun_jac",
                                   &capsule->discr_dyn_phi_fun_jac_ut_xt_1[i_fun]);
    }

    /**** Constraints phase 1 ****/


    // set up nonlinear constraints for stage 1 to N-1

    for (int i = 1; i < 2; i++)
    {
        i_fun = i - 1;
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun_jac",
                                      &capsule->nl_constr_h_fun_jac_1[i_fun]);
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun",
                                      &capsule->nl_constr_h_fun_1[i_fun]);
        
        
        
    }



    // TERMINAL node

    /* terminal constraints */

    // set up nonlinear constraints for last stage
    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun_jac", &capsule->nl_constr_h_e_fun_jac);
    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun", &capsule->nl_constr_h_e_fun);
    
    
    
}


/**
 * Internal function for multiphase_ocp_acados_create: step 6
 */
void multiphase_ocp_acados_create_set_opts(multiphase_ocp_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    // declare
    bool tmp_bool;
    int newton_iter_val;
    double newton_tol_val;
    /************************************************
    *  opts
    ************************************************/



    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);
    double globalization_alpha_min = 0.05;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_alpha_min", &globalization_alpha_min);

    double globalization_alpha_reduction = 0.7;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_alpha_reduction", &globalization_alpha_reduction);



    // ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization", "merit_backtracking");

    int globalization_line_search_use_sufficient_descent = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_line_search_use_sufficient_descent", &globalization_line_search_use_sufficient_descent);

    int globalization_use_SOC = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_use_SOC", &globalization_use_SOC);

    double globalization_eps_sufficient_descent = 0.0001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_eps_sufficient_descent", &globalization_eps_sufficient_descent);

    int with_solution_sens_wrt_params_forw = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params_forw", &with_solution_sens_wrt_params_forw);

    int with_solution_sens_wrt_params_adj = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params_adj", &with_solution_sens_wrt_params_adj);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    double solution_sens_qp_t_lam_min = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "solution_sens_qp_t_lam_min", &solution_sens_qp_t_lam_min);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    double levenberg_marquardt = 0.01;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 2;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    int log_primal_step_norm = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "log_primal_step_norm", &log_primal_step_norm);

    int log_dual_step_norm = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "log_dual_step_norm", &log_dual_step_norm);

    double nlp_solver_tol_min_step_norm = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_min_step_norm", &nlp_solver_tol_min_step_norm);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");



    int qp_solver_t0_init = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_t0_init", &qp_solver_t0_init);




    // set SQP specific options
    double nlp_solver_tol_stat = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_stat", &nlp_solver_tol_stat);

    double nlp_solver_tol_eq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_eq", &nlp_solver_tol_eq);

    double nlp_solver_tol_ineq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_ineq", &nlp_solver_tol_ineq);

    double nlp_solver_tol_comp = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_comp", &nlp_solver_tol_comp);

    int nlp_solver_max_iter = 200;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "max_iter", &nlp_solver_max_iter);

    // set options for adaptive Levenberg-Marquardt Update
    bool with_adaptive_levenberg_marquardt = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_adaptive_levenberg_marquardt", &with_adaptive_levenberg_marquardt);

    double adaptive_levenberg_marquardt_lam = 5;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_lam", &adaptive_levenberg_marquardt_lam);

    double adaptive_levenberg_marquardt_mu_min = 0.0000000000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_mu_min", &adaptive_levenberg_marquardt_mu_min);

    double adaptive_levenberg_marquardt_mu0 = 0.001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_mu0", &adaptive_levenberg_marquardt_mu0);

    double adaptive_levenberg_marquardt_obj_scalar = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_obj_scalar", &adaptive_levenberg_marquardt_obj_scalar);

    bool eval_residual_at_max_iter = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "eval_residual_at_max_iter", &eval_residual_at_max_iter);

    // QP scaling
    double qpscaling_ub_max_abs_eig = 100000;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qpscaling_ub_max_abs_eig", &qpscaling_ub_max_abs_eig);

    double qpscaling_lb_norm_inf_grad_obj = 0.0001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qpscaling_lb_norm_inf_grad_obj", &qpscaling_lb_norm_inf_grad_obj);

    qpscaling_scale_objective_type qpscaling_scale_objective = NO_OBJECTIVE_SCALING;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qpscaling_scale_objective", &qpscaling_scale_objective);

    ocp_nlp_qpscaling_constraint_type qpscaling_scale_constraints = NO_CONSTRAINT_SCALING;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qpscaling_scale_constraints", &qpscaling_scale_constraints);

    // NLP QP tol strategy
    ocp_nlp_qp_tol_strategy_t nlp_qp_tol_strategy = FIXED_QP_TOL;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_strategy", &nlp_qp_tol_strategy);

    double nlp_qp_tol_reduction_factor = 0.1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_reduction_factor", &nlp_qp_tol_reduction_factor);

    double nlp_qp_tol_safety_factor = 0.1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_safety_factor", &nlp_qp_tol_safety_factor);

    double nlp_qp_tol_min_stat = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_min_stat", &nlp_qp_tol_min_stat);

    double nlp_qp_tol_min_eq = 0.0000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_min_eq", &nlp_qp_tol_min_eq);

    double nlp_qp_tol_min_ineq = 0.0000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_min_ineq", &nlp_qp_tol_min_ineq);

    double nlp_qp_tol_min_comp = 0.00000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "nlp_qp_tol_min_comp", &nlp_qp_tol_min_comp);

    bool with_anderson_acceleration = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_anderson_acceleration", &with_anderson_acceleration);

    double anderson_activation_threshold = 10;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "anderson_activation_threshold", &anderson_activation_threshold);

    int qp_solver_iter_max = 500;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);



    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);



    /* Stage varying options */
    int ext_cost_num_hess;
    bool output_z_val = true;
    bool sens_algebraic_val = true;
    sim_collocation_type collocation_type;

    // set up sim_method_num_stages
    int* sim_method_num_stages = malloc(N*sizeof(int));
    sim_method_num_stages[0] = 4;
    sim_method_num_stages[1] = 4;

    // set up sim_method_num_steps
    int* sim_method_num_steps = malloc(N*sizeof(int));
    sim_method_num_steps[0] = 1;
    sim_method_num_steps[1] = 1;

    // set up sim_method_jac_reuse
    bool* sim_method_jac_reuse = malloc(N*sizeof(bool));
    sim_method_jac_reuse[0] = (bool)0;
    sim_method_jac_reuse[1] = (bool)0;

    // free arrays
    free(sim_method_num_steps);
    free(sim_method_num_stages);
    free(sim_method_jac_reuse);
}

/**
 * Internal function for multiphase_ocp_acados_create: step 7
 */
void multiphase_ocp_acados_create_set_nlp_out(multiphase_ocp_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;


    int nx_max = 43;
    int nu_max = 43;

    // initialize primal solution
    double* xu0 = calloc(nx_max+nu_max, sizeof(double));
    double* x0 = xu0;

    // initialize with x0
    x0[0] = 0.09317839969727612;
    x0[1] = 0.13045619700661976;
    x0[2] = 0.4514808774673858;
    x0[3] = 0.07370095777944052;
    x0[4] = -0.3423530429054911;
    x0[5] = 0.023451630060614684;
    x0[6] = 0.936382714514931;
    x0[7] = -0.6233811583995398;
    x0[8] = 0.04320584036993826;
    x0[9] = 0.00512586167006415;
    x0[10] = 0.8120430340777451;
    x0[11] = 0.5235994610924907;
    x0[12] = -0.1198743825910852;
    x0[13] = -0.6546972765756901;
    x0[14] = 0.2927752375691052;
    x0[15] = -0.1348090578958186;
    x0[16] = 0.874966313023328;
    x0[17] = 0.5236000056945823;
    x0[18] = -0.26175983101238426;
    x0[19] = 0.0071623851845860325;
    x0[20] = -0.1325147066143501;
    x0[21] = -0.3276110679257776;
    x0[22] = 0.9627558816703656;
    x0[23] = 0.3175265387725001;
    x0[24] = -0.019495194568615357;
    x0[25] = 1.12929731054022;
    x0[26] = -0.30529467129872;
    x0[27] = 0.16308386484701326;
    x0[28] = -1.6144295664682768;
    x0[29] = 0.6459958835714067;
    x0[30] = -0.33601217123306154;
    x0[31] = -1.1382458326715306;
    x0[32] = 1.928217408542408;
    x0[33] = 1.357881807378167;
    x0[34] = -0.08977701145062987;
    x0[35] = -1.6144295675703706;
    x0[36] = -0.03541367173626337;
    x0[37] = 0.1577303922893885;
    x0[38] = 0.15;
    x0[39] = 0.0000000000000000000000000000000000000000000000000000000000000000009156426765968716;
    x0[40] = -0.0000000000000000000000000000000000000000000000000000000000000000004391943496223828;
    x0[41] = 0.24073518763855625;
    x0[42] = 0.970590835253482;


    double* u0 = xu0 + nx_max;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x0);
    free(xu0);
}



/**
 * Internal function for multiphase_ocp_acados_create: step 9
 */
int multiphase_ocp_acados_create_precompute(multiphase_ocp_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}



int multiphase_ocp_acados_create_with_discretization(multiphase_ocp_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (new_time_steps) {
        fprintf(stderr, "multiphase_ocp_acados_create_with_discretization: new_time_steps should be NULL " \
            "for multi-phase solver!\n");
        return 1;
    }

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    multiphase_ocp_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = multiphase_ocp_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    multiphase_ocp_acados_create_set_opts(capsule);

    // 4) create and set nlp_out
    // 4.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 4.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    multiphase_ocp_acados_create_set_nlp_out(capsule);

    // 5) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 6) set default parameters in functions
    multiphase_ocp_acados_create_setup_functions(capsule);
    multiphase_ocp_acados_create_setup_nlp_in(capsule, N);
    multiphase_ocp_acados_create_setup_nlp_in_numerical_values(capsule, N);
    multiphase_ocp_acados_create_set_default_parameters(capsule);

    // 7) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);


    // 8) do precomputations
    int status = multiphase_ocp_acados_create_precompute(capsule);

    return status;
}




int multiphase_ocp_acados_reset(multiphase_ocp_solver_capsule* capsule, int reset_qp_solver_mem, int reset_numerical_values, int reset_solver_options, int reset_x_to_x0_bar)
{
    // set initialization to all zeros
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    // sets primal and dual iterates to zero
    ocp_nlp_out_set_values_to_zero(nlp_config, nlp_dims, nlp_out);

    // reset integrator memory
    ocp_nlp_solver_reset_integrator_memory(nlp_solver, nlp_in, nlp_out);
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    if (reset_numerical_values)
    {
        // reset parameters to initial values
        multiphase_ocp_acados_create_set_default_parameters(capsule);

        // reset numerical values in nlp_in
        multiphase_ocp_acados_create_setup_nlp_in_numerical_values(capsule, N);
    }

    if (reset_solver_options)
    {
        // reset solver options to initial values
        multiphase_ocp_acados_create_set_opts(capsule);
    }

    if (reset_x_to_x0_bar)
    {double* buffer = calloc(43, sizeof(double));
        ocp_nlp_constraints_model_get(nlp_config, nlp_dims, nlp_in, 0, "lbx", buffer);
        for (int i=0; i<N+1; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", buffer);
        }
        free(buffer);
    }
    return 0;
}



int multiphase_ocp_acados_update_params(multiphase_ocp_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;
    if (stage >= 0 && stage < 1)
    {
        if (NP_0 != np)
        {
            printf("acados_update_params: trying to set %i parameters at stage %i."
                " Parameters should be of length %i. Exiting.\n", np, stage, NP_0);
            exit(1);
        }
    }
    
    if (stage >= 1 && stage < 2)
    {
        if (NP_1 != np)
        {
            printf("acados_update_params: trying to set %i parameters at stage %i."
                " Parameters should be of length %i. Exiting.\n", np, stage, NP_1);
            exit(1);
        }
    }
    
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int multiphase_ocp_acados_update_params_sparse(multiphase_ocp_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int multiphase_ocp_acados_set_p_global_and_precompute_dependencies(multiphase_ocp_solver_capsule* capsule, double* data, int data_len)
{

    // printf("No global_data, multiphase_ocp_acados_set_p_global_and_precompute_dependencies does nothing.\n");
    return 0;
}



int multiphase_ocp_acados_solve(multiphase_ocp_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}


int multiphase_ocp_acados_setup_qp_matrices_and_factorize(multiphase_ocp_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_setup_qp_matrices_and_factorize(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



void multiphase_ocp_acados_print_stats(multiphase_ocp_solver_capsule* capsule)
{
    int sqp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "sqp_iter", &sqp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    double stat[2400];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = sqp_iter+1 < stat_m ? sqp_iter+1 : stat_m;

    printf("iter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tqp_stat\tqp_iter\talpha");
    if (stat_n > 8)
        printf("\t\tqp_res_stat\tqp_res_eq\tqp_res_ineq\tqp_res_comp");
    printf("\n");

    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            if (j == 0 || j == 5 || j == 6)
            {
                tmp_int = (int) stat[i + j * nrow];
                printf("%d\t", tmp_int);
            }
            else
            {
                printf("%e\t", stat[i + j * nrow]);
            }
        }
        printf("\n");
    }

}




int multiphase_ocp_acados_free(multiphase_ocp_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // initial node
    external_function_external_param_casadi_free(&capsule->nl_constr_h_0_fun_jac);
    external_function_external_param_casadi_free(&capsule->nl_constr_h_0_fun);
    /* Path phase {jj} */
    // dynamics
    for (int i_fun = 0; i_fun < 1; i_fun++)
    {
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun_0[i_fun]);
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun_jac_ut_xt_0[i_fun]);
    }
    free(capsule->discr_dyn_phi_fun_0);
    free(capsule->discr_dyn_phi_fun_jac_ut_xt_0);

    // constraints
    /* Path phase {jj} */
    // dynamics
    for (int i_fun = 0; i_fun < 1; i_fun++)
    {
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun_1[i_fun]);
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun_jac_ut_xt_1[i_fun]);
    }
    free(capsule->discr_dyn_phi_fun_1);
    free(capsule->discr_dyn_phi_fun_jac_ut_xt_1);

    // constraints
    for (int i_fun = 0; i_fun < 1; i_fun++)
    {
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun_jac_1[i_fun]);
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun_1[i_fun]);
    }
    free(capsule->nl_constr_h_fun_jac_1);
    free(capsule->nl_constr_h_fun_1);


    /* Terminal node */
    external_function_external_param_casadi_free(&capsule->nl_constr_h_e_fun_jac);
    external_function_external_param_casadi_free(&capsule->nl_constr_h_e_fun);



    return 0;
}




int multiphase_ocp_acados_custom_update(multiphase_ocp_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}


ocp_nlp_in *multiphase_ocp_acados_get_nlp_in(multiphase_ocp_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *multiphase_ocp_acados_get_nlp_out(multiphase_ocp_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *multiphase_ocp_acados_get_sens_out(multiphase_ocp_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *multiphase_ocp_acados_get_nlp_solver(multiphase_ocp_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *multiphase_ocp_acados_get_nlp_config(multiphase_ocp_solver_capsule* capsule) { return capsule->nlp_config; }
void *multiphase_ocp_acados_get_nlp_opts(multiphase_ocp_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *multiphase_ocp_acados_get_nlp_dims(multiphase_ocp_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *multiphase_ocp_acados_get_nlp_plan(multiphase_ocp_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
