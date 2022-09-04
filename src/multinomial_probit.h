#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <iterator>
#include <cassert>
#include <vector>
#ifdef _OPENMP
#   include <omp.h>
#endif

/* 'restrict' qualifier from C, if supported */
#if defined(__GNUG__) || defined(__GNUC__) || defined(_MSC_VER) || defined(__clang__) || \
    defined(__INTEL_COMPILER) || defined(__IBMCPP__) || defined(__ibmxl__) || defined(SUPPORTS_RESTRICT)
#   define restrict __restrict
#else
#   define restrict 
#endif

/* matrix_helpers.cpp */
void swap_entries_sq_matrix(double *restrict v, double *restrict X, const int ld_X,
                            const int n, const int pos1, const int pos2);
void fill_lower_triangle(double *restrict A, int n);
void matrix_inverse(double *restrict X, double *restrict Xinv, const int n);
void L_square_from_flat(const double *restrict Lflat, double *restrict Lsq, const int n);
void schur_complement01(const double *restrict X, const int n, double *restrict out, double *restrict buffer);

/* norm_grad.cpp */
double loggrad_x0
(
    const int n,
    const double *restrict x,
    const double *restrict R,
    double *restrict newX, /* (n-1)*/
    double *restrict newR, /* (n-1)^2 */
    double *restrict newV, /* (n-1) */
    double *restrict buffer1,
    int *restrict buffer2
);
double loggrad_R01
(
    const int n,
    const double *restrict x,
    const double *restrict R,
    const double *restrict iC22, /* inv(C[2:n,2:n]) */
    double *restrict newX, /* (n-2)*/
    double *restrict newR, /* (n-2)^2 */
    double *restrict newV, /* (n-2) */
    double *restrict buffer1,
    int *restrict buffer2
);

/* univariate.cpp */
double norm_logpdf_1d(double x);
double norm_cdf_1d(double a);
double norm_logcdf_1d(double a);

/* bhat.cpp */
double norm_logcdf_2d(double x1, double x2, double rho);
double norm_logcdf_3d(double x1, double x2, double x3, double rho12, double rho13, double rho23);
double norm_logcdf_4d_internal(const double x[4], const double rho[6]);
double norm_logcdf_4d(const double x[4], const double rho[6]);

/* ldl.cpp */
void factorize_ldl_2by2blocks(const double *restrict X, const int n,
                              double *restrict diag, double *restrict L,
                              double *restrict temp, double *restrict temp2);
void update_ldl_rank2(double *restrict L, const int ld_L,
                      double *restrict D, const int ld_D,
                      double *restrict O, const int ld_O, /* O is 2x2 */
                      const int n,
                      double *restrict newL, /* buffer dim (n-2)^2 */
                      double *restrict newD, /* buffer dim (n-2)^2 */
                      double *restrict b, /* buffer dim n-2 */
                      double *restrict Z /* buffer dim n-2 */
                      );

/* tvbs.cpp */
void truncate_bvn_2by2block(const double mu1, const double mu2,
                            const double v1, const double v2, const double cv,
                            const double t1, const double t2,
                            double &restrict mu1_out, double &restrict mu2_out,
                            double &restrict v1_out, double &restrict v2_out, double &restrict cv_out);
double norm_logcdf
(
    double *restrict x_reordered,
    double *restrict rho_reordered,
    const int n,
    const bool check_rho,
    double *restrict buffer, /* dim: 4*n^2 + 3*n - 8 */
    int *restrict buffer2
);

/* multinomial_probit.cpp */
int get_num_mnp_opt_vars(const int k, const int n);
void get_mnp_starting_point(double optvars[], const int k, const int n);
void get_mnp_prediction_matrices
(
    const int k,
    const double *restrict Lflat,
    double *restrict L, /* k*k */
    double *restrict class_Mats, /* k*(k-1)*k */
    double *restrict class_Rhos, /* k*(k-1)*(k-1) */
    double *restrict class_vars, /* k*(k-1) */
    int *restrict class_check_Rho /* k */
);
double mnp_likelihood
(
    const int m, const int k,
    int nthreads,
    const int *restrict y,
    const double *restrict pred,
    const double *restrict Lflat,
    const double *restrict weights /* optional row weights */
);
void mnp_classpred
(
    const int m, const int k,
    int nthreads,
    double *restrict out, /* m*k */
    const double *restrict pred, /* m*(k-1) */
    const double *restrict class_Mats,
    const double *restrict class_Rhos,
    const double *restrict class_vars,
    const int *restrict class_check_Rho,
    bool logp
);
double mnp_fun_grad
(
    const int m, const int k,
    int nthreads,
    const bool only_x,
    double *restrict gradX,
    double *restrict gradL,
    const int *restrict y,
    const double *restrict pred,
    const double *restrict Lflat,
    const double *restrict weights /* optional row weights */
);
/* Same thing as above, but obtained through finite differencing */
double mnp_fun_grad_fdiff
(
    const int m, const int k,
    int nthreads,
    const bool only_x,
    double *restrict gradX,
    double *restrict gradL,
    const int *restrict y,
    const double *restrict pred,
    const double *restrict Lflat,
    const double *restrict weights /* optional row weights */
);


#define EPS_BLOCK 1e-20
#define LOW_RHO 1e-9
constexpr const static double HIGH_RHO = 1. - 1e-3;

#define ncomb2(n) (    ( (n)*((n)-1) ) >> 1     )

#ifdef FOR_R
    #include <R.h>
    #include <Rinternals.h>
    #include <R_ext/Visibility.h>
    #include <R_ext/BLAS.h>
    #include <R_ext/Lapack.h>
#endif

#ifdef __GNUC__
#   define likely(x) __builtin_expect((bool)(x), true)
#   define unlikely(x) __builtin_expect((bool)(x), false)
#else
#   define likely(x) (x)
#   define unlikely(x) (x)
#endif

#ifndef FCONE
    #define FCONE 
#endif

#ifndef F77_CALL
    #define F77_CALL(fn) fn ## _
#endif

#ifndef M_PI
#   define M_PI 3.14159265358979323846
#endif

#ifndef _OPENMP
#   define omp_get_thread_num() 0
#endif

#ifndef CBLAS_H
#   include "cblas.h"
#endif
