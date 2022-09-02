#include "multinomial_probit.h"

void truncate_bvn_2by2block(const double mu1, const double mu2,
                            const double v1, const double v2, const double cv,
                            const double t1, const double t2,
                            double &restrict mu1_out, double &restrict mu2_out,
                            double &restrict v1_out, double &restrict v2_out, double &restrict cv_out)
{
    double s1 = std::sqrt(v1);
    double s2 = std::sqrt(v2);
    s1 = std::fmax(s1, 1e-8);
    s2 = std::fmax(s2, 1e-8);
    double ntp1 = (t1 - mu1) / s1;
    double ntp2 = (t2 - mu2) / s2;
    double rho = cv / (s1 * s2);

    double logp = norm_logcdf_2d(ntp1, ntp2, rho);
    double rhotilde = std::sqrt(std::fma(-rho, rho, 1.));
    rhotilde = std::fmax(rhotilde, 1e-16);
    double tr1 = std::fma(-rho, ntp2, ntp1) / rhotilde;
    double tr2 = std::fma(-rho, ntp1, ntp2) / rhotilde;
    double logpd1 = norm_logpdf_1d(ntp1);
    double logpd2 = norm_logpdf_1d(ntp2);
    double logcd1 = norm_logcdf_1d(tr1);
    double logcd2 = norm_logcdf_1d(tr2);

    double log_pd1_cd2 = logpd1 + logcd2;
    double log_pd2_cd1 = logpd2 + logcd1;
    double log_pdf_tr1 = norm_logpdf_1d(tr1);
    double log_pdf_tr2 = norm_logpdf_1d(tr2);

    double log_rho = std::log(std::fabs(rho));
    double sign_rho = (rho >= 0.)? 1. : -1.;
    double log_rho_pd2_cd1 = log_rho + log_pd2_cd1;
    double log_rho_pd1_cd2 = log_rho + log_pd1_cd2;

    double temp1 = sign_rho * std::exp(log_rho_pd2_cd1 - log_pd1_cd2);
    double temp2 = sign_rho * std::exp(log_rho_pd1_cd2 - log_pd2_cd1);

    double sign_m1 = -1.;
    double sign_m2 = -1.;
    double log_m1, log_m2;
    if (temp1 > -1.) {
        log_m1 = log_pd1_cd2 + std::log1p(temp1) - logp;
    }
    else {
        if (sign_rho > 0.) {
            goto no_log_m1;
        }

        if (log_rho_pd2_cd1 > log_pd1_cd2) {
            sign_m1 = 1.;
            log_m1 = log_rho_pd2_cd1 + std::log1p(-std::exp(log_pd1_cd2 - log_rho_pd2_cd1)) - logp;
        }
        else
        {
            no_log_m1:
            log_m1 = -std::fma(rho, std::exp(log_pd2_cd1), std::exp(log_pd1_cd2)) / std::exp(logp);
            sign_m1 = (log_m1 >= 0.)? 1. : -1.;
            log_m1 = std::log(log_m1);
        }
    }
    if (temp2 > -1.) {
        log_m2 = log_pd2_cd1 + std::log1p(temp2) - logp;
    }
    else {
        if (sign_rho > 0.) {
            goto no_log_m2;
        }

        if (log_rho_pd1_cd2 > log_pd2_cd1) {
            sign_m2 = 1.;
            log_m2 = log_rho_pd1_cd2 + std::log1p(-std::exp(log_pd2_cd1 - log_rho_pd1_cd2)) - logp;
        }
        else
        {
            no_log_m2:
            log_m2 = -std::fma(rho, std::exp(log_pd1_cd2), std::exp(log_pd2_cd1)) / std::exp(logp);
            sign_m2 = (log_m2 >= 0.)? 1. : -1.;
            log_m2 = std::log(log_m2);
        }
    }

    double sign_ntp1 = (ntp1 >= 0.)? 1. : -1.;
    double sign_ntp2 = (ntp2 >= 0.)? 1. : -1.;
    double log_ntp1 = std::log(std::fabs(ntp1));
    double log_ntp2 = std::log(std::fabs(ntp2));
    double log_rhotilde = std::log(rhotilde);

    double os1 = 1. - (
        sign_ntp1 * std::exp(log_ntp1 + log_pd1_cd2 - logp)
        + sign_ntp2 * std::exp(log_ntp2 + 2. * log_rho + log_pd2_cd1 - logp)
        - sign_rho * std::exp(log_rhotilde + log_rho + logpd2 + log_pdf_tr1 - logp)
    ) - std::exp(2. * log_m1);
    double os2 = 1. - (
        sign_ntp2 * std::exp(log_ntp2 + log_pd2_cd1 - logp)
        + sign_ntp1 * std::exp(log_ntp1 + 2. * log_rho + log_pd1_cd2 - logp)
        - sign_rho * std::exp(log_rhotilde + log_rho + logpd1 + log_pdf_tr2 - logp)
    ) - std::exp(2. * log_m2);
    double orho = rho * (
        1.
        - sign_ntp1 * std::exp(log_ntp1 + log_pd1_cd2 - logp)
        - sign_ntp2 * std::exp(log_ntp2 + log_pd2_cd1 - logp)
    ) + std::exp(log_rhotilde + logpd1 + log_pdf_tr2 - logp)
     - std::exp(log_m1 + log_m2);

    mu1_out = std::fma(sign_m1 * std::exp(log_m1), s1, mu1);
    mu2_out = std::fma(sign_m2 * std::exp(log_m2), s2, mu2);
    v1_out = v1 * os1;
    v2_out = v2 * os2;
    cv_out = s1 * s2 * orho;

    v1_out = std::fmax(v1_out, std::numeric_limits<double>::min());
    v2_out = std::fmax(v2_out, std::numeric_limits<double>::min());
}

void copy_and_standardize
(
    const int m, const int n,
    const double *restrict X,
    const double *restrict Sigma,
    double *restrict X_out,
    double *restrict Rho_out,
    double *restrict buffer_sdtdev /* dimension is 'n' */
)
{
    std::copy(Sigma, Sigma + n*n, Rho_out);

    for (int ix = 0; ix < n; ix++) {
        buffer_sdtdev[ix] = std::sqrt(Rho_out[ix*(n+1)]);
    }

    for (int row = 0; row < m; row++) {
        double *restrict Xrow_out = X_out + row * m;
        const double *restrict Xrow = X + row * m;
        #ifndef _MSC_VER
        #pragma omp simd
        #endif
        for (int col = 0; col < n; col++) {
            Xrow_out[col] = Xrow[col] / buffer_sdtdev[col];
        }
    }

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            Rho_out[col + row*n] /= buffer_sdtdev[row] * buffer_sdtdev[col];
        }
    }
}

void sort_ascending
(
    double *restrict x,
    double *restrict R,
    const int n,
    double *restrict buffer_x,
    double *restrict buffer_R,
    int *restrict buffer_idx
)
{
    std::iota(buffer_idx, buffer_idx + n, 0);
    std::sort(buffer_idx, buffer_idx + n, [&x](const int a, const int b){return x[a] < x[b];});

    std::copy(x, x + n, buffer_x);
    for (int ix = 0; ix < n; ix++)
        x[ix] = buffer_x[buffer_idx[ix]];

    const int n2 = n * n;
    std::copy(R, R + n2, buffer_R);

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            R[col + row*n] = buffer_R[buffer_idx[col] + buffer_idx[row]*n];
        }
    }
}

static inline
double nonstd_logcdf_2d(double x1, double x2, double m1, double m2, double v1, double v2, double cv)
{
    double s1 = std::sqrt(v1);
    double s2 = std::sqrt(v2);
    return norm_logcdf_2d((x1 - m1) / s1, (x2 - m2) / s2, cv / (s1 * s2));
}

static inline
double nonstd_logcdf_3d(const double x[3], const double mu[3], const double cov[], const int ld_cov)
{
    double stdevs[] = {
        std::sqrt(cov[0]),
        std::sqrt(cov[1 + ld_cov]),
        std::sqrt(cov[2 + 2*ld_cov])
    };
    return norm_logcdf_3d((x[0] - mu[0]) / stdevs[0],
                          (x[1] - mu[1]) / stdevs[1],
                          (x[2] - mu[2]) / stdevs[2],
                          cov[1] / (stdevs[0] * stdevs[1]),
                          cov[2] / (stdevs[0] * stdevs[2]),
                          cov[2 + ld_cov] / (stdevs[1] * stdevs[2]));
}

static inline
double nonstd_logcdf_4d(const double x[4], const double mu[4], const double cov[], const int ld_cov)
{
    double stdevs[] = {
        std::sqrt(cov[0]),
        std::sqrt(cov[1 + ld_cov]),
        std::sqrt(cov[2 + 2*ld_cov]),
        std::sqrt(cov[3 + 3*ld_cov])
    };
    double stdx[] = {
        (x[0] - mu[0]) / stdevs[0],
        (x[1] - mu[1]) / stdevs[1],
        (x[2] - mu[2]) / stdevs[2],
        (x[3] - mu[3]) / stdevs[3]
    };
    double rho[] = {
        cov[1] / (stdevs[0] * stdevs[1]),
        cov[2] / (stdevs[0] * stdevs[2]),
        cov[3] / (stdevs[0] * stdevs[3]),
        cov[2 + ld_cov] / (stdevs[1] * stdevs[2]),
        cov[3 + ld_cov] / (stdevs[1] * stdevs[3]),
        cov[3 + 2*ld_cov] / (stdevs[2] * stdevs[3])
    };
    return norm_logcdf_4d(stdx, rho);
}

[[gnu::always_inline]]
static inline
bool rho_is_zero(double x)
{
    return std::fabs(x) <= LOW_RHO;
}

[[gnu::always_inline]]
static inline
bool rho_is_one(double x)
{
    return std::fabs(x) >= HIGH_RHO;
}

/* This function pre-processes a correlation matrix by:
   - Eliminating perfectly correlated variables (only the one implying
     the lowest upper bound is required).
   - Eliminating independent variables (the final probability is a simple
     product of their probabilities and the probability of the remaining
     variables with have correlations among themselves).
   - Trying to split up variables into two independent blocks if possible
     (in which case the final probability is the product of the block
     probabilities).
  
  The output will be a potentially reduced R matrix, with rows/columns
  swapped accordingly as necessary, and the entries in 'x' also swapped
  in order to match the new correlation matrix.

  The output matrix will have the eliminated entries put as the first rows. */
void preprocess_rho(double *restrict R, const int ld_R, const int n, double *restrict x,
                    int &restrict pos_st, double &restrict p_independent,
                    int &restrict size_block1, int &restrict size_block2,
                    const int min_n_check_to_check)
{
    p_independent = 0.;
    size_block1 = n;
    size_block2 = 0;
    pos_st = 0;

    /* Step 1: look for perfect correlations */
    for (int row = pos_st; row < n-1; row++) {
        for (int col = row+1; col < n; col++) {
            if (rho_is_one(R[col + row*ld_R])) {
                if (x[row] < R[col + row*ld_R] * x[col]) {
                    swap_entries_sq_matrix(x, R, ld_R, n, col, row);
                }
                swap_entries_sq_matrix(x, R, ld_R, n, pos_st, row);
                pos_st++;
            }
        }
    }
    size_block1 = n - pos_st;
    if (size_block1 <= min_n_check_to_check) {
        return;
    }

    /* Step 2: look for independent variables. */
    int row_zeros;
    for (int row = pos_st; row < n; row++) {
        
        row_zeros = 0;
        for (int col = pos_st; col < n; col++) {
            row_zeros += rho_is_zero(R[col + row*ld_R]);
        }
        if (row_zeros >= n-pos_st-1) {
            p_independent += norm_logcdf_1d(x[row]);
            swap_entries_sq_matrix(x, R, ld_R, n, pos_st, row);
            pos_st++;
        }
    }
    size_block1 = n - pos_st;
    if (size_block1 <= min_n_check_to_check) {
        return;
    }

    /* Step 3: look for independent blocks.
    This one is more tricky to do. In broad terms, imagine that we want to put
    a block of zeros at the top-left corner that would imply independence, with
    the rows that have the largest numbers of zeros coming first.
    
    Examples:
    [1, r, 0, 0, 0]    <- Left is what we seek.            [1, r, r, 0, 0]
    [r, 1, 0, 0, 0]       Right one is an equivalent  ->   [r, 1, r, 0, 0]
    [0, 0, 1, r, r]       reordering of the matrix.        [r, r, 1, 0, 0]
    [0, 0, r, 1, r]                                        [0, 0, 0, 1, r]
    [0, 0, r, r, 1]                                        [0, 0, 0, r, 1]

    Note that diagonals cannot be zeros, so something like this would
    not be a valid correlation matrix (it's not possible by design),
    and thus such pattern we need not bother with:
    [r, r, 0, 0, 0, 0]
    [r, r, 0, 0, 0, 0]
    [r, r, 0, 0, 0, 0]   <- This is not a correlation matrix
    [0, 0, r, r, r, r]      (must always have ones at diagonals,
    [0, 0, r, r, r, r]       and must alawys be symmetric)
    [0, 0, r, r, r, r]
    
    The best possible scenario is something that splits the matrix into two
    equally-sized blocks - e.g.

    [1, r, r, 0, 0, 0]                      [1, r, 0, 0, 0, 0]
    [r, 1, r, 0, 0, 0]                      [r, 1, 0, 0, 0, 0]
    [r, r, 1, 0, 0, 0]   <- Left one is ->  [0, 0, 1, r, r, r]
    [0, 0, 0, 1, r, r]      preferrable     [0, 0, r, 1, r, r]
    [0, 0, 0, r, 1, r]                      [0, 0, r, r, 1, r]
    [0, 0, 0, r, r, 1]                      [0, 0, r, r, r, 1]

    Note in the left pattern that there must be at least n/2 rows with
    at least n/2 zeros each, while in the right pattern there must be at
    least (n/2 - 1) rows with at least (n/2 + 1) zeros each, and in each
    case, the columns with zeros must be the exact same ones among the
    rows that belong to the group.

    In the first group, there's a maximum of C(n, n/2) possible such
    blocks, while in the second, there's a maximum of C(n, n/2 - 1)
    possible such blocks, where C(n,k) = n!/((n-k)!*k!).

    Thus, finding such a block directly is hard (although it could also
    be done with graph-based methods, but that's a more complicated
    approach).

    Instead, we can approach it the other way around: first find two
    rows that are dependent (perhaps starting with the ones that have
    the largest amounts of non-zeros):
    [1, r]
    [r, 1]

    Then try to expand this set by adding another row that would be
    dependent on at least one of those two:
    [1, r, r]          [1, r, 0]          [1, r, r]
    [r, 1, r]   (or)   [r, 1, r]   (or)   [r, 1, 0]
    [r, r, 1]          [0, r, 1]          [r, 0, 1]

    And so on until being unable to add any further rows. This way, it's
    O(n^3) to check the full matrix for independent blocks (up to O(n^4)
    if we take into account the row swapping along the way).

    In any event, an independent block would imply a minimum number of
    zeros in the triangular part of the matrix of at least 2*(n-2) (this
    is the case in which two rows are independent of the rest, and any case
    of a single row being independent would have already been handled in the
    earlier conditions), so if there aren't enough zeros left, can avoid
    bothering to check at all.

    As a greedy trick to speed it up, depending on which exit is more likely
    to happen, could either:
      (a) Put the row with the highest number of non-zeros first and the
          row with the lowest number of non-zeros last, or perhaps do a
          full sort by descending number of non-zeros. This way, it will
          speed up the search for non-zero rows and reach the end of the
          full loop faster.
      (b) Sort the rows in ascending order of zeros. This way, the more
          sparse rows will be on top and it is more likely to encounter
          the exit condition early on.
    Since we expect that most cases will not have any independent blocks,
    strategy (a) is likely to result in a faster procedure.
    */
    int n_zeros = 0;
    for (int row = pos_st; row < n-1; row++) {
        for (int col = row+1; col < n; col++) {
            n_zeros += rho_is_zero(R[col + row*ld_R]);
        }
    }
    if (n_zeros < 2*(n-pos_st-2)) {
        return;
    }

    int max_zeros = 0;
    int min_zeros = n;
    int row_max_zeros = pos_st;
    int row_min_zeros = pos_st;
    int n_zeros_row;
    for (int row = pos_st; row < n; row++) {
        n_zeros_row = 0;
        for (int col = pos_st; col < n; col++) {
            n_zeros += rho_is_zero(R[col + row*ld_R]);
        }
        if (n_zeros_row > max_zeros) {
            max_zeros = n_zeros_row;
            row_max_zeros = row;
        }
        if (n_zeros_row < min_zeros) {
            min_zeros = n_zeros_row;
            row_min_zeros = row;
        }
    }
    if (max_zeros < 2) {
        return;
    }
    if (max_zeros - min_zeros >= 2) {
        swap_entries_sq_matrix(x, R, ld_R, n, row_max_zeros, pos_st);
        swap_entries_sq_matrix(x, R, ld_R, n, row_min_zeros, n-1);
    }

    int pos;
    for (pos = pos_st; pos < n-1; pos++) {
        
        for (int row = pos+1; row < n; row++) {
            for (int col = pos_st; col <= pos; col++) {
                if (!rho_is_zero(R[col + row*ld_R])) {
                    swap_entries_sq_matrix(x, R, ld_R, n, pos+1, row);
                    goto next_pos;
                }
            }
        }
        break;
        next_pos:
        {}
    }

    if (pos < n-2) {
        size_block1 = pos - pos_st + 1;
        size_block2 = (n - pos_st) - size_block1;
    }
}

double norm_logcdf
(
    double *restrict x_reordered,
    double *restrict rho_reordered,
    const int n,
    const bool check_rho,
    double *restrict buffer, /* dim: 5*n^2 + 3*n - 8 */
    int *restrict buffer2
)
{
    if (n <= 4) {
        switch (n) {
            case 1: {
                return norm_logcdf_1d(x_reordered[0]);
            }
            case 2: {
                return norm_logcdf_2d(x_reordered[0], x_reordered[1], rho_reordered[1]);
            }
            case 3: {
                return norm_logcdf_3d(x_reordered[0], x_reordered[1], x_reordered[2],
                                      rho_reordered[1], rho_reordered[2], rho_reordered[5]);
            }
            case 4: {
                const double rho_flat[] = {
                    rho_reordered[1], rho_reordered[2], rho_reordered[3],
                    rho_reordered[6], rho_reordered[7], rho_reordered[11]
                };
                return norm_logcdf_4d(x_reordered, rho_flat);
            }
            default: {
                assert(0);
                return NAN;
            }
        }
    }

    int size_block1, size_block2, pos_st;
    double p_independent;
    if (unlikely(check_rho)) {
        preprocess_rho(rho_reordered, n, n, x_reordered,
                       pos_st, p_independent,
                       size_block1, size_block2,
                       4);

        if (pos_st != 0 || size_block1 != n) {
            double p1 = 0.;
            double p2 = 0.;

            /* TODO: maybe should move these into separate buffers so that it can reuse
               'rho_reordered' and avoid doing this allocation */
            std::vector<double> rho_copy(n*n);
            if (size_block1) {
                F77_CALL(dlacpy)(
                    "A", &size_block1, &size_block1,
                    rho_reordered + pos_st * (n + 1), &n,
                    rho_copy.data(), &size_block1 FCONE
                );
                p1 = norm_logcdf(
                    x_reordered + pos_st,
                    rho_copy.data(),
                    size_block1,
                    false,
                    buffer,
                    buffer2
                );
            }

            if (size_block2) {
                if (size_block2) {
                    F77_CALL(dlacpy)(
                        "A", &size_block2, &size_block2,
                        rho_reordered + (pos_st + size_block1) * (n + 1), &n,
                        rho_copy.data(), &size_block2 FCONE
                    );
                }
                p2 = norm_logcdf(
                    x_reordered + pos_st + size_block1,
                    rho_copy.data(),
                    size_block2,
                    false,
                    buffer,
                    buffer2
                );
            }

            return p1 + p2 + p_independent;
        }
    }

    double *restrict const mu_trunc = buffer; buffer += n;
    double *restrict const L = buffer; buffer += n*n;
    double *restrict const D = buffer; buffer += n*n;
    double *restrict const temp1 = buffer; buffer += n*n;
    double *restrict const temp2 = buffer; buffer += n*n;
    double *restrict const temp3 = buffer; buffer += 2*(n-2);
    double *restrict const temp4 = buffer; buffer += 2*(n-2);

    sort_ascending(
        x_reordered,
        rho_reordered,
        n,
        temp1,
        temp2,
        buffer2
    );

    factorize_ldl_2by2blocks(rho_reordered, n,
                             D, L,
                             temp1, temp2);

    
    temp1[0] = rho_reordered[1]; temp1[1] = rho_reordered[2]; temp1[2] = rho_reordered[3];
    temp1[3] = rho_reordered[2 + n]; temp1[4] = rho_reordered[3 + n]; temp1[5] = rho_reordered[3 + 2*n];
    double cumP = norm_logcdf_4d(x_reordered, temp1);
    int n_steps = (n / 2) - !(n & 1) - 1;

    double bvn_trunc_mu[2];
    double bvn_trunc_cv[3];
    double bvn_trunc_cv_square[4];
    std::fill(mu_trunc, mu_trunc + n, 0.);
    double qvn_cv[16];
    double p2, p3, p4;

    for (int step = 0; step < n_steps; step++) {

        truncate_bvn_2by2block(mu_trunc[2*step], mu_trunc[2*step + 1],
                               D[2*step*(n+1)], D[(2*step+1)*(n+1)], D[2*step*(n+1) + 1],
                               x_reordered[2*step], x_reordered[2*step + 1],
                               bvn_trunc_mu[0], bvn_trunc_mu[1],
                               bvn_trunc_cv[0], bvn_trunc_cv[1], bvn_trunc_cv[2]);

        bvn_trunc_mu[0] -= mu_trunc[2*step];
        bvn_trunc_mu[1] -= mu_trunc[2*step + 1];

        cblas_dgemv(
            CblasRowMajor, CblasNoTrans,
            n - 2*(step+1), 2,
            1., L + 2*step*(n+1) + 2*n, n,
            bvn_trunc_mu, 1,
            1., mu_trunc + 2*(step+1), 1
        );


        bvn_trunc_cv_square[0] = bvn_trunc_cv[0];  bvn_trunc_cv_square[1] = bvn_trunc_cv[2];
        bvn_trunc_cv_square[2] = bvn_trunc_cv[2];  bvn_trunc_cv_square[3] = bvn_trunc_cv[1];
        update_ldl_rank2(L + (n+1)*2*(step), n,
                         D + (n+1)*2*(step), n,
                         bvn_trunc_cv_square, 2,
                         n - 2*(step),
                         temp1,
                         temp2,
                         temp3,
                         temp4);


        if (n >= 2*(step+1)+4) {
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                4, 4, 4,
                1., L + (n+1)*2*(step+1), n,
                D + (n+1)*2*(step+1), n,
                0., qvn_cv, 4
            );
            cblas_dtrmm(
                CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                4, 4,
                1., L + (n+1)*2*(step+1), n,
                qvn_cv, 4
            );

            p4 = nonstd_logcdf_4d(x_reordered + 2*(step+1), mu_trunc + 2*(step+1), qvn_cv, 4);
            p2 = nonstd_logcdf_2d(x_reordered[2*(step+1)], x_reordered[2*(step+1)+1],
                                  mu_trunc[2*(step+1)], mu_trunc[2*(step+1)+1],
                                  qvn_cv[0], qvn_cv[5], qvn_cv[1]);

            cumP += p4 - p2;
        }
        else {
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                3, 3, 3,
                1., L + (n+1)*2*(step+1), n,
                D + (n+1)*2*(step+1), n,
                0., qvn_cv, 3
            );
            cblas_dtrmm(
                CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                3, 3,
                1., L + (n+1)*2*(step+1), n,
                qvn_cv, 3
            );

            p3 = nonstd_logcdf_3d(x_reordered + 2*(step+1), mu_trunc + 2*(step+1), qvn_cv, 3);
            p2 = nonstd_logcdf_2d(x_reordered[2*(step+1)], x_reordered[2*(step+1)+1],
                                  mu_trunc[2*(step+1)], mu_trunc[2*(step+1)+1],
                                  qvn_cv[0], qvn_cv[4], qvn_cv[1]);

            cumP += p3 - p2;
        }

    }

    cumP = std::fmin(cumP, 0.);
    return cumP;
}
