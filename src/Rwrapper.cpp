#include "multinomial_probit.h"
#include <cstring>

extern "C" {

char errmsg[512];

SEXP R_get_num_mnp_opt_vars(SEXP k, SEXP n)
{
    return Rf_ScalarInteger(get_num_mnp_opt_vars(Rf_asInteger(k), Rf_asInteger(n)));
}

SEXP R_get_mnp_starting_point(SEXP k_, SEXP n_)
{
    int k = Rf_asInteger(k_);
    int n = Rf_asInteger(n_);
    int nvars = get_num_mnp_opt_vars(k, n);
    SEXP optvars = PROTECT(Rf_allocVector(REALSXP, nvars));
    bool success = true;
    try {
        get_mnp_starting_point(REAL(optvars), k, n);
    }
    catch (std::exception &e) {
        success = false;
        std::strncpy(errmsg, e.what(), 512);
    }
    if (!success) {
        Rf_error(errmsg);
    }
    UNPROTECT(1);
    return optvars;
}

SEXP R_get_mnp_prediction_matrices(SEXP k_, SEXP Lflat)
{
    const int k = Rf_asInteger(k_);
    const char* names_out[] = {"L", "class_Mats", "class_Rhos", "class_vars", "class_check_Rho", ""};
    SEXP out =  PROTECT(Rf_mkNamed(VECSXP, names_out));
    SEXP L = PROTECT(Rf_allocMatrix(REALSXP, k, k));
    SEXP class_Mats = PROTECT(Rf_alloc3DArray(REALSXP, k,k-1,k));
    SEXP class_Rhos = PROTECT(Rf_alloc3DArray(REALSXP, k,k-1,k-1));
    SEXP class_vars = PROTECT(Rf_allocMatrix(REALSXP, k,k-1));
    SEXP class_check_Rho = PROTECT(Rf_allocVector(INTSXP, k));
    bool success = true;
    try {
        get_mnp_prediction_matrices(
            k,
            REAL(Lflat),
            REAL(L),
            REAL(class_Mats),
            REAL(class_Rhos),
            REAL(class_vars),
            INTEGER(class_check_Rho)
        );
    }
    catch (std::exception &e) {
        success = false;
        std::strncpy(errmsg, e.what(), 512);
    }
    if (!success) {
        Rf_error(errmsg);
    }
    SET_VECTOR_ELT(out, 0, L);
    SET_VECTOR_ELT(out, 1, class_Mats);
    SET_VECTOR_ELT(out, 2, class_Rhos);
    SET_VECTOR_ELT(out, 3, class_vars);
    SET_VECTOR_ELT(out, 4, class_check_Rho);
    UNPROTECT(6);
    return out;
}

SEXP R_wrapped_mnp_fun_grad
(
    SEXP y,
    SEXP pred,
    SEXP Lflat,
    SEXP weights,
    SEXP nthreads,
    SEXP only_x,
    SEXP fdiff
)
{
    const char* names_out[] = {"ll", "gradX", "gradL", ""};
    SEXP out =  PROTECT(Rf_mkNamed(VECSXP, names_out));
    int m = Rf_xlength(y);
    int k = Rf_nrows(pred) + 1;
    SEXP gradX = PROTECT(Rf_allocMatrix(REALSXP, k-1, m));
    SEXP gradL = PROTECT(Rf_allocVector(REALSXP, k + ((k*(k-1))>>1) - 1));
    double *ptr_weights = NULL;
    if (Rf_xlength(weights)) {
        ptr_weights = REAL(weights);
    }
    SEXP ll = PROTECT(Rf_allocVector(REALSXP, 1));
    bool success = true;
    try {
        if (Rf_asLogical(fdiff)) {
            REAL(ll)[0] = mnp_fun_grad(
                m, k,
                Rf_asInteger(nthreads),
                (bool) Rf_asLogical(only_x),
                REAL(gradX),
                REAL(gradL),
                INTEGER(y),
                REAL(pred),
                REAL(Lflat),
                ptr_weights
            );
        }
        else {
            REAL(ll)[0] = mnp_fun_grad_fdiff(
                m, k,
                Rf_asInteger(nthreads),
                (bool) Rf_asLogical(only_x),
                REAL(gradX),
                REAL(gradL),
                INTEGER(y),
                REAL(pred),
                REAL(Lflat),
                ptr_weights
            );
        }
    }
    catch (std::exception &e) {
        success = false;
        std::strncpy(errmsg, e.what(), 512);
    }
    if (!success) {
        Rf_error(errmsg);
    }
    SET_VECTOR_ELT(out, 0, ll);
    SET_VECTOR_ELT(out, 1, gradX);
    SET_VECTOR_ELT(out, 2, gradL);
    UNPROTECT(4);
    return out;
}

SEXP R_wrapped_mnp_fun
(
    SEXP y,
    SEXP pred,
    SEXP Lflat,
    SEXP weights,
    SEXP nthreads
)
{
    int m = Rf_xlength(y);
    int k = Rf_nrows(pred) + 1;
    double *ptr_weights = NULL;
    if (Rf_xlength(weights)) {
        ptr_weights = REAL(weights);
    }
    double out;
    bool success = true;
    try {
        out = mnp_likelihood(
            m, k,
            Rf_asInteger(nthreads),
            INTEGER(y),
            REAL(pred),
            REAL(Lflat),
            ptr_weights
        );
    }
    catch (std::exception &e) {
        success = false;
        std::strncpy(errmsg, e.what(), 512);
    }
    if (!success) {
        Rf_error(errmsg);
    }
    return Rf_ScalarReal(-out);
}

SEXP R_wrapped_mnp_classpred
(
    SEXP m_, SEXP k_, SEXP nthreads,
    SEXP pred,
    SEXP class_Mats,
    SEXP class_Rhos,
    SEXP class_vars,
    SEXP class_check_Rho,
    SEXP logp
)
{
    int m = Rf_asInteger(m_);
    int k = Rf_asInteger(k_);
    SEXP out = PROTECT(Rf_allocMatrix(REALSXP, k, m));
    bool success = true;
    try {
        mnp_classpred(
            m, k,
            Rf_asInteger(nthreads),
            REAL(out),
            REAL(pred),
            REAL(class_Mats),
            REAL(class_Rhos),
            REAL(class_vars),
            INTEGER(class_check_Rho),
            (bool) Rf_asLogical(logp)
        );
    }
    catch (std::exception &e) {
        success = false;
        std::strncpy(errmsg, e.what(), 512);
    }
    if (!success) {
        Rf_error(errmsg);
    }
    UNPROTECT(1);
    return out;
}

static const R_CallMethodDef callMethods [] = {
    {"R_get_num_mnp_opt_vars", (DL_FUNC) &R_get_num_mnp_opt_vars, 2},
    {"R_get_mnp_starting_point", (DL_FUNC) &R_get_mnp_starting_point, 2},
    {"R_get_mnp_prediction_matrices", (DL_FUNC) &R_get_mnp_prediction_matrices, 2},
    {"R_wrapped_mnp_fun_grad", (DL_FUNC) &R_wrapped_mnp_fun_grad, 7},
    {"R_wrapped_mnp_fun", (DL_FUNC) &R_wrapped_mnp_fun, 5},
    {"R_wrapped_mnp_classpred", (DL_FUNC) &R_wrapped_mnp_classpred, 9},
    {NULL, NULL, 0}
}; 

void attribute_visible R_init_multinomial_probit(DllInfo *info)
{
    R_registerRoutines(info, NULL, callMethods, NULL, NULL);
    R_useDynamicSymbols(info, TRUE);
}

}
