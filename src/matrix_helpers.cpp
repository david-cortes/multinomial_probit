#include "multinomial_probit.h"

/* X[i,:] = X[j,:]; X[:,i] = X[:,j] */
void swap_entries_sq_matrix(double *restrict v, double *restrict X, const int ld_X,
                            const int n, const int pos1, const int pos2)
{
    if (pos1 == pos2) {
        return;
    }

    int row_st1 = pos1 * n;
    int row_st2 = pos2 * n;
    for (int ix = 0; ix < n; ix++) {
        std::swap(X[ix + row_st1], X[ix + row_st2]);
    }
    for (int ix = 0; ix < n; ix++) {
        std::swap(X[pos1 + ix*n], X[pos2 + ix*n]);
    }
    if (v) {
        std::swap(v[pos1], v[pos2]);
    }
}

void fill_lower_triangle(double *restrict A, int n)
{
    for (int row = 1; row < n; row++)
        for (int col = 0; col < row; col++)
            A[col + row*n] = A[row + col*n];
}

/* TODO: maybe this one would be more robust by using eigendecomposition */
void matrix_inverse(double *restrict X, double *restrict Xinv, const int n)
{
    if (n <= 0) return;
    else if (n == 1) {
        Xinv[0] = 1. / std::fmax(X[0], std::numeric_limits<double>::min());
        return;
    }
    else if (n == 2) {
        double det = X[0]*X[3] - X[1]*X[2];
        double reg = 0.;
        if (det <= 0.) {
            reg = std::sqrt(std::fabs(det - std::numeric_limits<double>::min()));
            reg = std::fmax(reg, std::sqrt(std::numeric_limits<double>::min()));
        }
        if (det >= 0) {
            Xinv[0] = X[3] + reg; Xinv[1] = -X[1];
            Xinv[2] = -X[2]; Xinv[3] = X[0] + reg;
        }
        #pragma GCC unroll 4
        for (int ix = 0; ix < 4; ix++) {
            Xinv[ix] /= det;
        }
    }
    int status;
    double reg = 1e-8;
    std::copy(X, X + n*n, Xinv);
    while (true) {
        F77_CALL(dpotrf)("L", &n, Xinv, &n, &status);
        if (status <= 0) break;
        std::copy(X, X + n*n, Xinv);
        for (int ix = 0; ix < n; ix++) {
            Xinv[ix*(n+1)] += reg;
        }
        reg *= 1.5;
    }
    assert(status == 0);
    F77_CALL(dpotri)("L", &n, Xinv, &n, &status);
    assert(status == 0);
    fill_lower_triangle(Xinv, n);
}

void L_square_from_flat(const double *restrict Lflat, double *restrict Lsq, const int n)
{
    std::fill(Lsq, Lsq + n*n, 0.);
    Lsq[0] = 1.;
    int counter = 0;
    for (int row = 1; row < n; row++) {
        for (int col = 0; col <= row; col++) {
            Lsq[col + row*n] = Lflat[counter++];
        }
    }
}
