#include "multinomial_probit.h"

/* Likelihood and gradients for Multinomial Probit Regression (MNP)

Under this model, the classes follow a latent process as follows:
    pred[nrows, nclasses] = X[nrows,ncols] * t(coefs[nclasses, ncols])
    scores[nrows, nclasses] = pred + eps[nrows, nclasses]
    eps(row) ~ MVN(0, Sigma[nclasses, nclasses])

(where coefs and Sigma are model parameters)

For each row, the probability that it will belong to each class corresponds
to the probability that each class score will be the highest.
In order for a given score to be the highest, it means that the difference
between that score and every other score (for every other class) has to be
greater than zero.

Thus, we can imagine a combination matrix for each class having ones at
the column of the class and each row having a minus one at a single column
for another classes. Say, we have k=4 and we want to get the differences for
the first class:
    M1 = [1, -1,  0,  0]
         [1,  0, -1,  0]
         [1,  0,  0, -1]
    diff1 = M1 * scores

(Note that M1 has dimensions [k-1, k])

Then, the probability that each of them is greater than zero can be calculated
as the probability that each of these random draws will be larger than zero:
    MVN( M1*pred(row) ,  M1*Sigma*t(M1) )

Which in turn corresponds to the CDF of a standardized MVN distribution:
    V = diag(1 / sqrt(diag(Sigma)))
    Rho = V*M1*Sigma*t(M1)*t(V)
    MVN( 0 , Rho )

And prob:
    CDF( V . (M1*pred(row)), Rho  )
    (where "." denotes elementwise multiplication)

In this case, we have the constraint that Sigma (the covariance matrix)
needs to be positive semi-definite - to ensure that, we can try to optimize
instead its Cholesky:
    Sigma = L * t(L)

If we take a given combination matrix, say M1, then we can calculate the covariance
of the translated class as follows:
    A1 = M1 * L
    SigmaHat = A1 * t(A1)
    VHat = diag(1 / sqrt(diag(SigmaHat)))
    RhoHat = VHat * SigmaHat * t(VHat)
    XHat = Vhat . (M1 * pred)

(Note that each of these class correlation matrices will have dimension k-1, not k)

Having these probabilities, one can then take the gradients with respect to each
element of the standardized normal distribution and apply the chain rule to
propagate it to the original model parameters (coefs, L).

For the bounds of the CDF, the gradient is given as follows:
    grad(x1) = pdf(x1) * CDF(x2..n | x1=x1 )
For the gradient of correlations of the CDF, one can consult the following reference:
    Plackett, Robin L.
    "A reduction formula for normal multivariate integrals."
    Biometrika 41.3/4 (1954): 351-360.

The gradients of each successive step from the correlation coefficient back to the
Cholesky have a rather complicated matrix form, and are easier to deal with in
loop form. There's no exact reference for them, but one can consult the following
reference for an intuition:
    Bolduc, Denis.
    "A practical technique to estimate multinomial probit models in transportation."
    Transportation Research Part B: Methodological 33.1 (1999): 63-79.

One might realize that the optimization problem as-is is ill-defined, since basically
it's a relative comparison and different translations and rescalings will produce the
same result.

As such, one can leave a base class with fixed values as the reference class to which
the others are compared (e.g. assuming all coefficients for this class are zeros and do
not get optimized). Here it is assumed that the first class is the reference class,
and thus the functions take no predictions for them (assume their prediction is always
zero).

Likewise, one of the variances in Sigma needs to be fixed - this is easily achieved
by the setting the element of L at the first row and first column equal to 1, and
hence the functions here take the Cholesky with one fewer element (they assume the
ommited entry is equal to 1).
 */

int get_num_mnp_opt_vars(const int k, const int n)
{
    int ncoef = n * (k-1);
    int numL = k + ncomb2(k);
    return ncoef + numL - 1;
}

void get_mnp_starting_point(double optvars[], const int k, const int n)
{
    const int nvars = get_num_mnp_opt_vars(k, n);
    std::fill(optvars, optvars + nvars, 0.);
    int ix_prev = 0;
    int ix_curr;
    for (int idx = 2; idx < k+1; idx++) {
        ix_curr = idx + ix_prev;
        optvars[ix_curr - 1] = 1.;
        ix_prev = ix_curr;
    }
}

void get_mnp_prediction_matrices
(
    const int k,
    const double *restrict Lflat,
    double *restrict L, /* k*k */
    double *restrict class_Mats, /* k*(k-1)*k */
    double *restrict class_Rhos, /* k*(k-1)*(k-1) */
    double *restrict class_vars, /* k*(k-1) */
    int *restrict class_check_Rho /* k */
)
{
    assert(k >= 3);
    int k1 = k - 1;

    L_square_from_flat(Lflat, L, k);

    std::vector<double> class_As(k*k1*k);
    std::vector<double> class_Sigmas(k*k1*k1);

    for (int cl = 0; cl < k; cl++) {

        double *classMat = class_Mats + cl * k1*k;
        std::fill(classMat, classMat + k1*k, 0.);
        for (int ix = 0; ix < k1; ix++) {
            classMat[ix * (k+1) + (ix>=cl)] = -1.;
        }
        for (int row = 0; row < k1; row++) {
            classMat[cl + row*k] = 1.;
        }

        double *classA = class_As.data() + cl * k1*k;
        std::copy(classMat, classMat + k1*k, classA);
        cblas_dtrmm(
            CblasRowMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
            k1, k,
            1., L, k,
            classA, k
        );

        double *classSigma = class_Sigmas.data() + cl * k1*k1;
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            k1, k1, k,
            1., classA, k,
            classA, k,
            0., classSigma, k1
        );

        double *classVar = class_vars + cl*k1;
        for (int ix = 0; ix < k1; ix++) {
            classVar[ix] = std::sqrt(classSigma[ix * k]);
        }

        double *classRho = class_Rhos + cl*k1*k1;
        for (int ix = 0; ix < k1; ix++) {
            classRho[ix * k] = 1.;
        }
        for (int row = 0; row < k1-1; row++) {
            for (int col = row + 1; col < k1; col++) {
                classRho[col + row*k1] = classSigma[col + row*k1] / (classVar[row] * classVar[col]);
            }
        }
        fill_lower_triangle(classRho, k1);

        bool check_rho = false;
        for (int row = 0; row < k1-1; row++) {
            for (int col = row + 1; col < k1; col++) {
                double abs_rho = std::fabs(classRho[col + row*k1]);
                check_rho |= abs_rho >= HIGH_RHO;
                check_rho |= abs_rho <= LOW_RHO;
            }
        }
        class_check_Rho[cl] = check_rho;
    }
}

double mnp_likelihood
(
    const int m, const int k,
    int nthreads,
    const int *restrict y,
    const double *restrict pred,
    const double *restrict Lflat,
    const double *restrict weights /* optional row weights */
)
{
    assert(k >= 3);
    std::vector<int> argsorted_y(m);
    std::iota(argsorted_y.begin(), argsorted_y.end(), 0);
    std::sort(argsorted_y.begin(), argsorted_y.end(), [&y](const int a, const int b){return y[a] < y[b];});

    std::vector<int> y_reordered(m);
    for (int ix = 0; ix < m; ix++) {
        y_reordered[ix] = y[argsorted_y[ix]];
    }

    int k1 = k - 1;
    std::vector<double> pred_reordered(m * k1);
    for (int row = 0; row < m; row++) {
        std::copy(pred + argsorted_y[row]*k1, pred + (argsorted_y[row]+1)*k1, pred_reordered.begin() + row*k1);
    }

    std::vector<int> class_indptr(k+1);
    class_indptr[0] = 0;
    for (int cl = 1; cl < k;) {
        int offset = std::distance(
            y_reordered.data(),
            std::lower_bound(
                y_reordered.data() + class_indptr[cl-1],
                y_reordered.data() + m,
                cl
            )
        );
        if (offset == m) {
            for (; cl < k; cl++) {
                class_indptr[cl] = m;
            }
            break;
        }
        int n_skip = y_reordered[offset] - cl;
        for (int ix = 0; ix < n_skip + 1; ix++) {
            class_indptr[cl] = offset;
            cl++;
        }
    }
    class_indptr[k] = m;

    std::vector<double> L(k*k);
    L_square_from_flat(Lflat, L.data(), k);

    std::vector<double> class_Mats(k*k1*k);
    std::vector<double> class_As(k*k1*k);
    std::vector<double> class_vars(k*k1);
    std::vector<double> class_Rhos(k*k1*k1);
    std::vector<bool> class_checkRho(k);
    for (int cl = 0; cl < k; cl++) {
        if (class_indptr[cl] == class_indptr[cl+1]) continue;

        double *classMat = class_Mats.data() + cl * k1*k;
        std::fill(classMat, classMat + k1*k, 0.);
        for (int ix = 0; ix < k1; ix++) {
            classMat[ix * (k+1) + (ix>=cl)] = -1.;
        }
        for (int row = 0; row < k1; row++) {
            classMat[cl + row*k] = 1.;
        }

        double *classA = class_As.data() + cl * k1*k;
        std::copy(classMat, classMat + k1*k, classA);
        cblas_dtrmm(
            CblasRowMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
            k1, k,
            1., L.data(), k,
            classA, k
        );

        double *classSigma = class_Rhos.data() + cl * k1*k1;
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            k1, k1, k,
            1., classA, k,
            classA, k,
            0., classSigma, k1
        );

        double *classVar = class_vars.data() + cl*k1;
        for (int ix = 0; ix < k1; ix++) {
            classVar[ix] = std::sqrt(classSigma[ix * k]);
        }

        double *classRho = classSigma;
        for (int ix = 0; ix < k1; ix++) {
            classRho[ix * k] = 1.;
        }
        for (int row = 0; row < k1-1; row++) {
            for (int col = row + 1; col < k1; col++) {
                classRho[col + row*k1] /= (classVar[row] * classVar[col]);
            }
        }
        fill_lower_triangle(classRho, k1);

        bool check_rho = false;
        for (int row = 0; row < k1-1; row++) {
            for (int col = row + 1; col < k1; col++) {
                double abs_rho = std::fabs(classRho[col + row*k1]);
                check_rho |= abs_rho >= HIGH_RHO;
                check_rho |= abs_rho <= LOW_RHO;
            }
        }
        class_checkRho[cl] = check_rho;
    }

    std::vector<double> shifted_pred(m * k1);
    for (int cl = 0; cl < k; cl++) {
        int n_this = class_indptr[cl+1] - class_indptr[cl];
        if (!n_this) continue;
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            n_this, k1, k1,
            1., pred_reordered.data() + class_indptr[cl]*k1, k1,
            class_Mats.data() + cl * k1*k + 1, k,
            0., shifted_pred.data() + class_indptr[cl]*k1, k1
        );
    }

    /* standardize them */
    for (int row = 0; row < m; row++) {
        double *classVars = class_vars.data() + y_reordered[row]*k1;
        double *predRow = shifted_pred.data() + row*k1;

        #ifndef _WIN32
        #pragma omp simd
        #endif
        for (int col = 0; col < k1; col++) {
            predRow[col] /= classVars[col];
        }
    }

    
    std::vector<std::vector<double>> thread_buffer1(nthreads);
    std::vector<std::vector<int>> thread_buffer2(nthreads);
    int size_thread_dwork = 5*k*k + 3*k - 8  +  k1+k1*k1;
    /* shorthand: 6*k^2 + 4*k */
    for (int tid = 0; tid < nthreads; tid++) {
        thread_buffer1[tid].resize(size_thread_dwork);
        thread_buffer2[tid].resize(k);
    }

    std::vector<double> logp(m);
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(m, shifted_pred, k, k1, class_Rhos, y_reordered, class_checkRho, \
                   thread_buffer1, thread_buffer2, logp)
    for (int row = 0; row < m; row++) {
        int tid = omp_get_thread_num();
        double *buffer1 = thread_buffer1[tid].data();
        double *x_local = buffer1; buffer1 += k1;
        double *R_local = buffer1; buffer1 += k1*k1;
        std::copy(shifted_pred.data() + row*k1, shifted_pred.data() + (row+1)*k1, x_local);
        std::copy(class_Rhos.data() + y_reordered[row]*k1*k1, class_Rhos.data() + y_reordered[row]*k1*k1 + k1*k1, R_local);

        logp[row] = norm_logcdf(
            x_local,
            R_local,
            k1,
            class_checkRho[y_reordered[row]],
            thread_buffer1[tid].data(),
            thread_buffer2[tid].data()
        );
    }

    /* Note: beware bad behavior when using multiple threads */
    if (!weights) {
        return std::accumulate(logp.begin(), logp.end(), 0.);
    }
    else {
        return cblas_ddot(m, logp.data(), 1, weights, 1);
    }
}

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
)
{
    assert(k >= 3);
    int k1 = k - 1;

    std::vector<std::vector<double>> thread_buffer1(nthreads);
    std::vector<std::vector<int>> thread_buffer2(nthreads);
    int size_thread_dwork = 5*k*k + 3*k - 8 + k1*k1;
    for (int tid = 0; tid < nthreads; tid++) {
        thread_buffer1[tid].resize(size_thread_dwork);
        thread_buffer2[tid].resize(k);
    }

    /* Note: one could think of making the last class' probability as 1 - sum(other_p),
       but since these numbers are very inexact, it will oftentimes not end up summing
       to 1, hence this calculated log-probabilities also for the last class. */
    std::vector<double> classBounds(m*k1);
    for (int cl = 0; cl < k; cl++) {
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            m, k1, k1,
            1., pred, k1,
            class_Mats + cl * k1*k + 1, k,
            0., classBounds.data(), k1
        );

        const double *classVar = class_vars + cl*k1;
        const double *classRho = class_Rhos + cl*k1*k1;

        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                shared(m, cl, k, k1, classRho, classVar, class_check_Rho, \
                       thread_buffer1, thread_buffer2)
        for (int row = 0; row < m; row++) {
            int tid = omp_get_thread_num();
            double *thisRow = classBounds.data() + row*k1;
            double *buffer1 = thread_buffer1[tid].data();
            double *R_local = buffer1; buffer1 += k1*k1;
            std::copy(classRho, classRho + k1*k1, R_local);
            #ifndef _WIN32
            #pragma omp simd
            #endif
            for (int col = 0; col < k1; col++) {
                thisRow[col] /= classVar[col];
            }

            out[cl + row*k] = norm_logcdf(
                thisRow,
                R_local,
                k1,
                (bool) class_check_Rho[cl],
                buffer1,
                thread_buffer2[tid].data()
            );
        }
    }

    if (!logp) {
        for (int row = 0; row < m; row++) {
            double *outRow = out + row*k;
            #ifndef _WIN32
            #pragma omp simd
            #endif
            for (int col = 0; col < k; col++) {
                outRow[col] = std::exp(outRow[col]);
            }

            double cumP = 0.;
            #ifndef _WIN32
            #pragma omp simd
            #endif
            for (int col = 0; col < k; col++) {
                cumP += outRow[col];
            }

            #ifndef _WIN32
            #pragma omp simd
            #endif
            for (int col = 0; col < k; col++) {
                outRow[col] /= cumP;
            }
        }
    }
}

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
)
{
    /* dwork = (5/4)*k^4 + (7/2)*k^3 + (1/4)*k^2 + 5*k*m - 13*k - 4*m + 18
       shorthand = 2*k^4 + 3*k^3 + k^2 + 5*k*m */
    /* iwork = 3*m + 2*k + nthreads + 1 */
    assert(k >= 3);
    std::vector<int> argsorted_y(m);
    std::iota(argsorted_y.begin(), argsorted_y.end(), 0);
    std::sort(argsorted_y.begin(), argsorted_y.end(), [&y](const int a, const int b){return y[a] < y[b];});

    std::vector<int> resort_out(m);
    std::iota(resort_out.begin(), resort_out.end(), 0);
    std::sort(resort_out.begin(), resort_out.end(), [&argsorted_y](const int a, const int b){return argsorted_y[a] < argsorted_y[b];});

    std::vector<int> y_reordered(m);
    for (int ix = 0; ix < m; ix++) {
        y_reordered[ix] = y[argsorted_y[ix]];
    }

    int k1 = k - 1;
    int ktri = ncomb2(k1);
    int numL = k + ncomb2(k);
    std::vector<double> pred_reordered(m * k1);
    for (int row = 0; row < m; row++) {
        std::copy(pred + argsorted_y[row]*k1, pred + (argsorted_y[row]+1)*k1, pred_reordered.begin() + row*k1);
    }

    std::vector<double> weights_reordered;
    if (weights) {
        weights_reordered.resize(m);
        for (int row = 0; row < m; row++) {
            weights_reordered[row] = weights[argsorted_y[row]];
        }
    }

    std::vector<int> class_indptr(k+1);
    class_indptr[0] = 0;
    for (int cl = 1; cl < k;) {
        int offset = std::distance(
            y_reordered.data(),
            std::lower_bound(
                y_reordered.data() + class_indptr[cl-1],
                y_reordered.data() + m,
                cl
            )
        );
        if (offset == m) {
            for (; cl < k; cl++) {
                class_indptr[cl] = m;
            }
            break;
        }
        int n_skip = y_reordered[offset] - cl;
        for (int ix = 0; ix < n_skip + 1; ix++) {
            class_indptr[cl] = offset;
            cl++;
        }
    }
    class_indptr[k] = m;

    std::vector<double> L(k*k);
    L_square_from_flat(Lflat, L.data(), k);

    std::vector<double> class_Mats(k*k1*k);
    std::vector<double> class_As(k*k1*k);
    std::vector<double> class_Sigmas(k*k1*k1);
    std::vector<double> class_vars(k*k1);
    std::vector<double> class_crossv(k*k1);
    std::vector<double> class_Rhos(k*k1*k1);
    std::vector<double> class_Cs(k*k1*k1);
    std::vector<bool> class_checkRho(k);
    for (int cl = 0; cl < k; cl++) {
        if (class_indptr[cl] == class_indptr[cl+1]) continue;

        double *classMat = class_Mats.data() + cl * k1*k;
        std::fill(classMat, classMat + k1*k, 0.);
        for (int ix = 0; ix < k1; ix++) {
            classMat[ix * (k+1) + (ix>=cl)] = -1.;
        }
        for (int row = 0; row < k1; row++) {
            classMat[cl + row*k] = 1.;
        }

        double *classA = class_As.data() + cl * k1*k;
        std::copy(classMat, classMat + k1*k, classA);
        cblas_dtrmm(
            CblasRowMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
            k1, k,
            1., L.data(), k,
            classA, k
        );

        double *classSigma = class_Sigmas.data() + cl * k1*k1;
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            k1, k1, k,
            1., classA, k,
            classA, k,
            0., classSigma, k1
        );

        double *classCrossv = class_crossv.data() + cl*k1;
        for (int ix = 0; ix < k1; ix++) {
            classCrossv[ix] = -1. / (2. * std::pow(classSigma[ix * k], 1.5));
        }

        double *classVar = class_vars.data() + cl*k1;
        for (int ix = 0; ix < k1; ix++) {
            classVar[ix] = std::sqrt(classSigma[ix * k]);
        }

        double *classRho = class_Rhos.data() + cl*k1*k1;
        for (int ix = 0; ix < k1; ix++) {
            classRho[ix * k] = 1.;
        }
        for (int row = 0; row < k1-1; row++) {
            for (int col = row + 1; col < k1; col++) {
                classRho[col + row*k1] = classSigma[col + row*k1] / (classVar[row] * classVar[col]);
            }
        }
        fill_lower_triangle(classRho, k1);

        bool check_rho = false;
        for (int row = 0; row < k1-1; row++) {
            for (int col = row + 1; col < k1; col++) {
                double abs_rho = std::fabs(classRho[col + row*k1]);
                check_rho |= abs_rho >= HIGH_RHO;
                check_rho |= abs_rho <= LOW_RHO;
            }
        }
        class_checkRho[cl] = check_rho;

        double *classC = class_Cs.data() + cl*k1*k1;
        if (!only_x) {
            matrix_inverse(classRho, classC, k1);
        }
    }

    std::vector<double> shifted_pred(m * k1);
    for (int cl = 0; cl < k; cl++) {
        int n_this = class_indptr[cl+1] - class_indptr[cl];
        if (!n_this) continue;
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            n_this, k1, k1,
            1., pred_reordered.data() + class_indptr[cl]*k1, k1,
            class_Mats.data() + cl * k1*k + 1, k,
            0., shifted_pred.data() + class_indptr[cl]*k1, k1
        );
    }

    /* standardize them */
    std::vector<double> shifted_pred_std(m * k1);
    for (int row = 0; row < m; row++) {
        double *classVars = class_vars.data() + y_reordered[row]*k1;
        double *predRow = shifted_pred.data() + row*k1;
        double *predRowStd = shifted_pred_std.data() + row*k1;

        #ifndef _WIN32
        #pragma omp simd
        #endif
        for (int col = 0; col < k1; col++) {
            predRowStd[col] = predRow[col] / classVars[col];
        }
    }

    
    std::vector<std::vector<double>> thread_buffer1(nthreads);
    std::vector<std::vector<int>> thread_buffer2(nthreads);
    int size_dwork_cdf = 5*k1*k1 + 3*k1 - 8;
    int size_dwork_X = k1 + 2*(k1-1) + k1*k1 + (k1-1)*(k1-1);
    int size_dwork_R = (k1-2)*(k1-2) + (k1-1)*(k1-1) + 2*(k1-1) + k1;
    int size_thread_dwork = size_dwork_cdf + std::max(size_dwork_X, size_dwork_R);
    /* shorthand 8*k^2 + 6*k */

    for (int tid = 0; tid < nthreads; tid++) {
        thread_buffer1[tid].resize(size_thread_dwork);
        thread_buffer2[tid].resize(k);
    }

    std::vector<double> log_probs(m);
    #pragma omp parallel for schedule(static) num_threads(nthreads) \
            shared(m, shifted_pred_std, k, k1, class_Rhos, y_reordered, class_checkRho, \
                   thread_buffer1, thread_buffer2, log_probs)
    for (int row = 0; row < m; row++) {
        int tid = omp_get_thread_num();
        double *buffer1 = thread_buffer1[tid].data();
        double *x_local = buffer1; buffer1 += k1;
        double *R_local = buffer1; buffer1 += k1*k1;

        std::copy(shifted_pred_std.data() + row*k1, shifted_pred_std.data() + (row+1)*k1, x_local);
        std::copy(class_Rhos.data() + y_reordered[row]*k1*k1, class_Rhos.data() + y_reordered[row]*k1*k1 + k1*k1, R_local);

        log_probs[row] = norm_logcdf(
            x_local,
            R_local,
            k1,
            class_checkRho[y_reordered[row]],
            buffer1,
            thread_buffer2[tid].data()
        );
    }
    double logp;
    if (!weights) {
        logp = std::accumulate(log_probs.begin(), log_probs.end(), 0.);
    }
    else {
        logp = cblas_ddot(m, log_probs.data(), 1, weights_reordered.data(), 1);
    }

    
    std::vector<double> grad_X(k * k1);
    std::vector<double> gradX_reordered(m * k1);
    std::vector<double> gradX_mult(m * k1);
    std::vector<double> nonsummed_gradX(m * k1);
    for (int cl = 0; cl < k; cl++) {
        
        if (class_indptr[cl] == class_indptr[cl+1]) continue;
        double *classRho = class_Rhos.data() + cl*k1*k1;
        
        for (int cc = 0; cc < k1; cc++) {
            swap_entries_sq_matrix(
                (double*)nullptr, classRho, k1,
                k1, 0, cc
            );

            if (cc != 0) {
                for (int row = class_indptr[cl]; row < class_indptr[cl+1]; row++) {
                    std::swap(shifted_pred_std[row*k1], shifted_pred_std[cc + row*k1]);    
                }
            }

            #pragma omp parallel for schedule(static) num_threads(nthreads) \
                    shared(m, k1, thread_buffer1, thread_buffer2, shifted_pred_std, gradX_reordered, \
                           classRho, shifted_pred, class_indptr, nonsummed_gradX)
            for (int row = class_indptr[cl]; row < class_indptr[cl+1]; row++) {
                int tid = omp_get_thread_num();
                double *buffer1 = thread_buffer1[tid].data();
                int *buffer2 = thread_buffer2[tid].data();

                double *x_local = buffer1; buffer1 += k1;
                double *R_local = buffer1; buffer1 += k1*k1;
                double *newX = buffer1; buffer1 += k1-1;
                double *newR = buffer1; buffer1 += (k1-1)*(k1-1);
                double *newV = buffer1; buffer1 += k1-1;

                std::copy(shifted_pred_std.data() + row*k1, shifted_pred_std.data() + (row+1)*k1, x_local);
                std::copy(classRho, classRho + k1*k1, R_local);

                double logp_this = loggrad_x0(
                    k1,
                    x_local,
                    R_local,
                    newX,
                    newR,
                    newV,
                    buffer1,
                    buffer2
                );

                double grad_this = std::exp(logp_this - log_probs[row]);
                nonsummed_gradX[cc + row*k1] = grad_this  * shifted_pred[cc + row*k1];
                gradX_reordered[cc + row*k1] = grad_this;
            }


            /* now restore like they were */
            swap_entries_sq_matrix(
                (double*)nullptr, classRho, k1,
                k1, cc, 0
            );
            if (cc != 0) {
                for (int row = 0; row < m; row++) {
                    std::swap(shifted_pred_std[cc + row*k1], shifted_pred_std[row*k1]);    
                }
            }
        }

        double *grad_xvals = grad_X.data() + cl*k1;
        std::fill(grad_xvals, grad_xvals + k1, 0.);
        for (int row = 0; row < m; row++) {
            cblas_daxpy(k1, 1., nonsummed_gradX.data() + row*k1, 1, grad_xvals, 1);
        }
        double *classCrossv = class_crossv.data() + cl*k1;
        #ifndef _WIN32
        #pragma omp simd
        #endif
        for (int col = 0; col < k1; col++) {
            grad_xvals[col] *= classCrossv[col];
        }

        double *classV = class_vars.data() + cl*k1;
        for (int row = class_indptr[cl]; row < class_indptr[cl+1]; row++) {
            double *row_gradX = gradX_reordered.data() + row*k1;
            #ifndef _WIN32
            #pragma omp simd
            #endif
            for (int col = 0; col < k1; col++) {
                row_gradX[col] /= classV[col];
            }
        }

        double *classMat = class_Mats.data() + cl * k1*k;
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            class_indptr[cl+1] - class_indptr[cl], k1, k1,
            -1., gradX_reordered.data() + class_indptr[cl]*k1, k1,
            classMat + 1, k,
            0., gradX_mult.data() + class_indptr[cl]*k1, k1
        );
    }

    for (int row = 0; row < m; row++) {
        std::copy(
            gradX_mult.data() + resort_out[row]*k1,
            gradX_mult.data() + (resort_out[row] + 1)*k1,
            gradX + row*k1
        );
    }

    logp = -logp;
    if (only_x) return logp;

    std::vector<double> grad_Rho(k * ktri);
    std::vector<double> iC22(  (k1-2)*(k1-2)  );
    std::vector<double> iC22_temp(  (k1-2)*(k1-2)  );
    std::vector<double> nonsummed_gradRho(m * ktri);
    for (int cl = 0; cl < k; cl++) {

        if (class_indptr[cl] == class_indptr[cl+1]) continue;
        
        double *classRho = class_Rhos.data() + cl*k1*k1;
        double *classC = class_Cs.data() + cl*k1*k1;
        
        int combcounter = 0;
        for (int rrow = 0; rrow < k1-1; rrow++) {
            for (int rcol = rrow + 1; rcol < k1; rcol++) {
                
                if (rrow != 0) {
                    for (int row = class_indptr[cl]; row < class_indptr[cl+1]; row++) {
                        std::swap(shifted_pred_std[row*k1], shifted_pred_std[row*k1 + rrow]);
                    }
                }
                if (rcol != 1) {
                    for (int row = class_indptr[cl]; row < class_indptr[cl+1]; row++) {
                        std::swap(shifted_pred_std[row*k1 + 1], shifted_pred_std[row*k1 + rcol]);
                    }
                }
                swap_entries_sq_matrix(
                    (double*)nullptr, classRho, k1,
                    k1, 0, rrow
                );
                swap_entries_sq_matrix(
                    (double*)nullptr, classRho, k1,
                    k1, 1, rcol
                );

                swap_entries_sq_matrix(
                    (double*)nullptr, classC, k1,
                    k1, 0, rrow
                );
                swap_entries_sq_matrix(
                    (double*)nullptr, classC, k1,
                    k1, 1, rcol
                );

                int k3 = k1 - 2;
                if (k3) {
                    F77_CALL(dlacpy)(
                        "?", &k3, &k3,
                        classC + 2*(k1 + 1), &k1,
                        iC22_temp.data(), &k3
                    );
                    matrix_inverse(iC22_temp.data(), iC22.data(), k3);
                }

                #pragma omp parallel for schedule(static) num_threads(nthreads) \
                        shared(class_indptr, m, k1, ktri, thread_buffer1, thread_buffer2, \
                               shifted_pred_std, classRho, log_probs, nonsummed_gradRho)
                for (int row = class_indptr[cl]; row < class_indptr[cl+1]; row++) {
                    int tid = omp_get_thread_num();
                    double *buffer1 = thread_buffer1[tid].data();
                    int *buffer2 = thread_buffer2[tid].data();

                    double *x_local = buffer1; buffer1 += k1;
                    double *iC22_local = buffer1; buffer1 += (k1-2)*(k1-2);
                    double *newX = buffer1; buffer1 += k1-1;
                    double *newR = buffer1; buffer1 += (k1-1)*(k1-1);
                    double *newV = buffer1; buffer1 += k1-1;

                    std::copy(shifted_pred_std.data() + row*k1, shifted_pred_std.data() + (row+1)*k1, x_local);
                    if (k3) std::copy(iC22.data(), iC22.data() + k3*k3, iC22_local);

                    double logp_this = loggrad_R01(
                        k1,
                        x_local,
                        classRho,   
                        iC22_local,
                        newX,
                        newR,
                        newV,
                        buffer1,
                        buffer2
                    );

                    nonsummed_gradRho[combcounter + row*ktri] = std::exp(logp_this - log_probs[row]);
                }


                /* now restore them */
                if (rrow != 0) {
                    for (int row = class_indptr[cl]; row < class_indptr[cl+1]; row++) {
                        std::swap(shifted_pred_std[row*k1 + rrow], shifted_pred_std[row*k1]);
                    }
                }
                if (rcol != 1) {
                    for (int row = class_indptr[cl]; row < class_indptr[cl+1]; row++) {
                        std::swap(shifted_pred_std[row*k1 + rcol], shifted_pred_std[row*k1 + 1]);
                    }
                }
                swap_entries_sq_matrix(
                    (double*)nullptr, classRho, k1,
                    k1, rcol, 1
                );
                swap_entries_sq_matrix(
                    (double*)nullptr, classRho, k1,
                    k1, rrow, 0
                );
                swap_entries_sq_matrix(
                    (double*)nullptr, classC, k1,
                    k1, rcol, 1
                );
                swap_entries_sq_matrix(
                    (double*)nullptr, classC, k1,
                    k1, rrow, 0
                );
                combcounter++;
            }
        }

        if (!weights) {
            for (int row = 0; row < m; row++) {
                cblas_daxpy(ktri, 1., nonsummed_gradRho.data() + row*ktri, 1, grad_Rho.data() + cl*ktri, 1);
            }
        }
        else {
            for (int row = 0; row < m; row++) {
                cblas_daxpy(ktri, weights_reordered[row], nonsummed_gradRho.data() + row*ktri, 1, grad_Rho.data() + cl*ktri, 1);
            }
        }
    }

    std::vector<double> grad_R_S( ktri * (k1+ktri) );
    std::vector<double> grad_fx_S(k1+ktri);
    std::vector<double> grad_S_A( (k1+ktri) * (k1*k) );
    std::vector<double> grad_fx_A(k1*k);
    std::vector<double> grad_A_L( (k1*k) * numL );
    /* TODO: get rid of this, should assing indices directly */
    std::vector<double> grad_unshaped(k*k);

    std::fill(gradL, gradL + numL - 1, 0.);

    for (int cl = 0; cl < k; cl++) {
        if (class_indptr[cl] == class_indptr[cl+1]) continue;

        double *classSigma = class_Sigmas.data() + cl * k1*k1;
        double *classVar = class_vars.data() + cl*k1;
        double *classCrossv = class_crossv.data() + cl*k1;

        int rcounter = 0;
        std::fill(grad_R_S.begin(), grad_R_S.end(), 0.);
        for (int rrow = 0; rrow < k1-1; rrow++) {
            for (int rcol = rrow+1; rcol < k1; rcol++) {
                std::fill(grad_unshaped.data(), grad_unshaped.data() + k1*k1, 0.);
                grad_unshaped[rrow + rrow*k1] = classSigma[rcol + rrow*k1] * classCrossv[rrow] / classVar[rcol];
                grad_unshaped[rcol + rrow*k1] = 1. / (classVar[rrow] * classVar[rcol]);
                grad_unshaped[rcol + rcol*k1] = classSigma[rcol + rrow*k1] * classCrossv[rcol] / classVar[rrow];

                int ccounter = 0;
                double *gradRS = grad_R_S.data() + rcounter*(k1+ktri);
                for (int row = 0; row < k1; row++) {
                    for (int col = row; col < k1; col++) {
                        gradRS[ccounter++] = grad_unshaped[col + row*k1];
                    }
                }
                rcounter++;
            }
        }

        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            1, k1+ktri, ktri,
            1., grad_Rho.data() + cl*ktri, ktri,
            grad_R_S.data(), k1+ktri,
            0., grad_fx_S.data(), k1+ktri
        );

        double *gradX = grad_X.data() + cl*k1;
        grad_fx_S[0] += gradX[0];
        int ccounter = 0;
        for (int ix = 0; ix < k1-1;ix++) {
            ccounter += k1 - ix;
            grad_fx_S[ccounter] += gradX[ix+1];
        }
        
        double *classA = class_As.data() + cl * k1*k;
        rcounter = 0;
        std::fill(grad_S_A.begin(), grad_S_A.end(), 0.);
        for (int rrow = 0; rrow < k1; rrow++) {
            for (int rcol = rrow; rcol < k1; rcol++) {
                double *grad_this = grad_S_A.data() + rcounter*(k1*k);
                cblas_daxpy(k, 1., classA + rcol*k, 1, grad_this + rrow*k, 1);
                cblas_daxpy(k, 1., classA + rrow*k, 1, grad_this + rcol*k, 1);
                rcounter++;
            }
        }

        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            1, k1*k, k1+ktri,
            1., grad_fx_S.data(), k1+ktri,
            grad_S_A.data(), k1*k,
            0., grad_fx_A.data(), k1*k
        );

        double *classMat = class_Mats.data() + cl * k1*k;
        rcounter = 0;
        for (int rrow = 0; rrow < k1; rrow++) {
            for (int rcol = 0; rcol < k; rcol++) {
                std::fill(grad_unshaped.data(), grad_unshaped.data() + k*k, 0.);
                cblas_dcopy(k-rcol, classMat + rcol + rrow*k, 1, grad_unshaped.data() + rcol*(k+1), k);
                double *grad_this = grad_A_L.data() + rcounter*numL;
                int ccounter = 0;
                for (int row = 0; row < k; row++) {
                    for (int col = 0; col <= row; col++) {
                        grad_this[ccounter++] = grad_unshaped[col + row*k];
                    }
                }
                rcounter++;
            }
        }

        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            1, numL - 1, k1*k,
            -1., grad_fx_A.data(), k1*k,
            grad_A_L.data() + 1, numL,
            1., gradL, numL - 1
        );
    }

    return logp;
}
