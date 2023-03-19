#include "multinomial_probit.h"

void standardize_norm_prob
(
    double *restrict x,
    double *restrict R,
    double *restrict V,
    const int n
)
{
    for (int ix = 0; ix < n; ix++) {
        V[ix] = std::sqrt(R[ix*(n+1)]);
    }
    for (int row = 0; row < n-1; row++) {
        #ifndef _MSC_VER
        #pragma omp simd
        #endif
        for (int col = row+1; col < n; col++) {
            R[col + row*n] /= (V[row] * V[col]);
        }
    }
    for (int ix = 0; ix < n; ix++) {
        R[ix*(n+1)] = 1.;
    }
    fill_lower_triangle(R, n);
    #ifndef _MSC_VER
    #pragma omp simd
    #endif
    for (int ix = 0; ix < n; ix++) {
        x[ix] /= V[ix];
    }
}

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
)
{
    if (n == 1) {
        return norm_logpdf_1d(x[0]);
    }
    else if (n == 2) {
        return norm_logpdf_1d(x[0]) + norm_logcdf_1d((x[1] - R[1]*x[0]) / std::sqrt(std::fma(-R[1], R[1], 1.)));
    }

    int n1 = n - 1;
    std::copy(x + 1, x + n, newX);
    cblas_daxpy(n1, -x[0], R + 1, 1, newX, 1);

    F77_CALL(dlacpy)(
        "?", &n1, &n1,
        R + n + 1, &n,
        newR, &n1 FCONE
    );
    cblas_dsyr(
        CblasRowMajor, CblasUpper, n1,
        -1., R + 1, 1,
        newR, n1
    );

    standardize_norm_prob(
        newX,
        newR,
        newV,
        n1
    );

    double mult1 = norm_logpdf_1d(x[0]);
    double mult2 = norm_logcdf(
        newX,
        newR,
        n1,
        true,
        buffer1,
        buffer2
    );

    return mult1 + mult2;
}

const double log_inv_twoPI = std::log(1. / (2. * M_PI));
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
)
{
    if (n <= 1) return 0.;

    const double rtilde = std::fma(-R[1], R[1], 1.);
    const double mult2 = (
        log_inv_twoPI
        - std::log(std::sqrt(rtilde))
        - (x[0]*x[0] + x[1]*x[1] - 2.*R[1]*x[0]*x[1]) / (2. * rtilde)
    );
    if (n == 2) {
        return mult2;
    }

    const int n2 = n - 2;
    std::copy(iC22, iC22 + n2*n2, newR);


    for (int ix = 0; ix < n2; ix++) {
        newX[ix] = - (
            x[0] * (R[ix+2] - R[ix+2+n]*R[1]) +
            x[1] * (R[ix+2+n] - R[ix+2]*R[1])
        ) / rtilde;
    }
    cblas_daxpy(n2, 1., x + 2, 1, newX, 1);

    standardize_norm_prob(
        newX,
        newR,
        newV,
        n2
    );

    double cdf = norm_logcdf(
        newX,
        newR,
        n2,
        true,
        buffer1,
        buffer2
    );

    return cdf + mult2;
}
