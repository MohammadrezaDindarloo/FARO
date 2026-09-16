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
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_multiphase_ocp.h"

// blasfeo
#include "blasfeo_d_aux_ext_dep.h"

#define MULTIPHASE_OCP_N      2


int main()
{

    int nx_max = 43;
    int nu_max = 43;

    multiphase_ocp_solver_capsule *acados_ocp_capsule = multiphase_ocp_acados_create_capsule();
    // there is an opportunity to change the number of shooting intervals in C without new code generation
    int N = MULTIPHASE_OCP_N;
    // allocate the array and fill it accordingly
    double* new_time_steps = NULL;
    int status = multiphase_ocp_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);

    if (status)
    {
        printf("multiphase_ocp_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    ocp_nlp_config *nlp_config = multiphase_ocp_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = multiphase_ocp_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = multiphase_ocp_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = multiphase_ocp_acados_get_nlp_out(acados_ocp_capsule);
    ocp_nlp_solver *nlp_solver = multiphase_ocp_acados_get_nlp_solver(acados_ocp_capsule);
    void *nlp_opts = multiphase_ocp_acados_get_nlp_opts(acados_ocp_capsule);

    // initialization for state values
    double* x_init = calloc(nx_max, sizeof(double));
    double* u_init = calloc(nu_max, sizeof(double));
    double* x_val = calloc(nx_max, sizeof(double));
    double* u_val = calloc(nu_max, sizeof(double));

    // prepare evaluation
    int NTIMINGS = 1;
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    // solve ocp in loop
    for (int ii = 0; ii < NTIMINGS; ii++)
    {
        // initialize solution
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x_init);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u_init);
        }
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x_init);
        status = multiphase_ocp_acados_solve(acados_ocp_capsule);
        ocp_nlp_get(nlp_solver, "time_tot", &elapsed_time);

        min_time = MIN(elapsed_time, min_time);
    }

    int nx;
    int nu;
    /* print solution and statistics */
    printf("\n--- x trajectory ---\n");
    for (int ii = 0; ii <= nlp_dims->N; ii++)
    {
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", x_val);
        nx = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nlp_out, ii, "x");
        d_print_exp_tran_mat(nx, 1, x_val, 1);
    }
    printf("\n--- u trajectory ---\n");
    for (int ii = 0; ii < nlp_dims->N; ii++)
    {
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", u_val);
        nu = ocp_nlp_dims_get_from_attr(nlp_config, nlp_dims, nlp_out, ii, "u");
        d_print_exp_tran_mat(nu, 1, u_val, 1);    }
    printf("\nsolved ocp %d times, solution printed above\n\n", NTIMINGS);

    if (status == ACADOS_SUCCESS)
    {
        printf("multiphase_ocp_acados_solve(): SUCCESS!\n");
    }
    else
    {
        printf("multiphase_ocp_acados_solve() failed with status %d.\n", status);
    }

    // get solution
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_solver, "sqp_iter", &sqp_iter);

    multiphase_ocp_acados_print_stats(acados_ocp_capsule);

    printf("\nSolver info:\n");
    printf("SQP iterations %2d\n minimum time for %d solve %f [ms]\n KKT %e\n",
           sqp_iter, NTIMINGS, min_time*1000, kkt_norm_inf);

    // free solver
    status = multiphase_ocp_acados_free(acados_ocp_capsule);
    if (status) {
        printf("multiphase_ocp_acados_free() returned status %d. \n", status);
    }
    // free solver capsule
    status = multiphase_ocp_acados_free_capsule(acados_ocp_capsule);
    if (status) {
        printf("multiphase_ocp_acados_free_capsule() returned status %d. \n", status);
    }

    return status;
}
