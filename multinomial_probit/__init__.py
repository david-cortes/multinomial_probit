import numpy as np
from . import _cpp_wrapper
import ctypes
from scipy.optimize import minimize, BFGS, approx_fprime
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import multiprocessing
from scipy.sparse import issparse, isspmatrix_csr, hstack, coo_matrix

__all__ = ["MultinomialProbitRegression"]

class MultinomialProbitRegression(BaseEstimator):
    """
    Multinomial Probit Regression

    Fits a multinomial probit regression (for multi-class classification with 3 or more classes), with
    likelihoods and gradients computed through a fast approximation to the CDF of the MVN distribution
    based on the TVBS method (see references for details) and optimized through the BFGS method from SciPy.
    
    In general, the approximation for the likelihood and gradients is not good enough for the resulting
    model to be very usable - typically, results are not competitive against simpler multinomial logistic
    regression (which assumes a unit covariance matrix), and as such, is not recommended for serious usage.

    Note
    ----
    If passing ``n_jobs>1``, it is recommended to use a BLAS / LAPACK implementation that would be
    openmp-aware, such as "openblas-openmp" (typically not the default as e.g. numpy from pip bundles
    "openblas-pthreads") or MKL.

    Note
    ----
    One is likely to observe better results from the R interface of this package, which uses a more
    precise implementation of the BFGS solver and more precise row summing algorithm.

    Parameters
    ----------
    lambda_ : float
        Amount of L2 regularization. Note that the regularization is added following GLMNET's formula
        (more regularization means more shrinkage, and it is scaled by the number of rows), which is
        different from scikit-learn's formula (higher values mean less shrinkage and it is not scaled
        by the number of rows).
    fit_intercept : bool
        Whether to add intercepts to the coefficients for each class.
    grad : str, one of "analytical" or finite_diff"
        How to calculate gradients of the MVN CDF parameters.

        If passing "analytical", will use the theoretically correct formula for the gradients given by
        the MVN PDF plus conditioned CDF (using Plackett's identity for the correlation coefficient
        gradients), with the same CDF approximation as used for calculating the likelihood.

        If passing "finite_diff", will approximate the gradients through finite differencing.

        In theory, analytical gradients should be more accurate and faster, but in practive, since
        the CDF is calculated by an imprecise approximation, the analytical gradients of the theoretical
        CDF might not be too accurate as gradients of this approximation. Both approaches involve the
        same number of CDF calculations, but for analytical gradients, the problems are reduced by
        one / two dimensions, which makes them slightly faster.

        Note that this refers to gradients w.r.t the parameters of MVN distributions, not w.r.t to the
        actual model parameters, which are still obtained by applying the chain rule.

        On smaller datasets in particular, finite differencing can result in more accurate gradients,
        but on larger datasets, as errors average out, analytical gradients tend to be more accurate.
    warm_start : bool
        In successive calls to ``fit``, whether to reuse previous solutions as starting point
        for the optimization procedure.
    presolve_logitstic : bool
        Whether to pre-solve for the coefficients by fitting a multinomial *logistic* regression
        first and use that solution as starting point for optimization (after a smaller second
        presolve pass solving only for the Cholesky of the covariance matrix).
    n_jobs : int
        Number of parallel threads to use. Negative values are interpreted according to joblib's
        formula: n_cpus - n_jobs + 1

    Attributes
    ----------
    k : int
        Number of classes to which the model was fitted
    coef_ : array (n_classes - 1, n_features)
        The fitted model coefficients. Note that there are no coefficients
        for the first class - since the model is about relative differences,
        the first class is set as the base to which the others are compared.
    intercept_ : array(n_classes - 1)
        Intercepts for each class. If passing ``fit_intercept=False``, will be
        filled with zeros.
    Sigma_ : array(n_classes, n_classes)
        The fitted covariance matrix for the classes. Note that the value on the
        first row and first column (the variance of the first class) is fixed to 1,
        as the model is about relative differences for each class and the first class
        is taken as a base.
    ll_ : float
        The negative of the log-likelihood for the fitted model (divided by the number of rows).
        If using regularization, will be summed to this number.

    References
    ----------
    .. [1] Bhat, Chandra R.
           "New matrix-based methods for the analytic evaluation of the multivariate cumulative normal distribution function."
           Transportation Research Part B: Methodological 109 (2018): 238-256.
    .. [2] Plackett, Robin L. "A reduction formula for normal multivariate integrals." Biometrika 41.3/4 (1954): 351-360.
    """
    def __init__(self, lambda_=0., fit_intercept=True, grad="analytical", warm_start=False, presolve_logistic=True, n_jobs=-1):
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
        self.grad = grad
        self.warm_start = warm_start
        self.presolve_logistic = presolve_logistic
        self.n_jobs = n_jobs
    
    def fit(self, X, y, sample_weights=None):
        """
        Fit the model to data

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            The data / covariates.
        y : array (n_samples,)
            The class / label for each row. Must be an integer vector with
            numeration starting at zero, and must contain a minimum of 3 classes.
        sample_weights : array (n_samples,)
            Optional row weights.

        Returns
        -------
        self : MultinomialProbitRegression
            The fitted model (note that it is also modified in-place)
        """
        assert self.lambda_ >= 0
        assert self.grad in ["finite_diff", "analytical"]

        y, sample_weights = self._check_y_and_weights(y, sample_weights)
        self.k_ = y.max() + 1
        assert self.k_ >= 3

        m = X.shape[0]
        if self.fit_intercept:
            if not issparse(X):
                X = np.c_[np.ones((m,1)), X]
            else:
                X = hstack([coo_matrix(np.ones((m,1))), X], format="csr", dtype=np.float64)
        if not issparse(X) and not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        if issparse(X):
            if not isspmatrix_csr(X):
                X = X.tocsr()
        n = X.shape[1]

        assert y.shape[0] == X.shape[0]

        if sample_weights is not None:
            if not isinstance(sample_weights, np.ndarray):
                sample_weights = np.array(sample_weights)
            sample_weights = sample_weights.reshape(-1)
            if sample_weights.dtype != np.float64:
                sample_weights = sample_weights.astype(np.float64)

        n_jobs = self._get_njobs()
        lam = float(self.lambda_) * m

        numL = self.k_ + int(self.k_*(self.k_-1)/2) - 1
        optvars = _cpp_wrapper.wrapped_mnp_starting_point(self.k_, n)
        if self.warm_start and self.__is_fitted__():
            old_opt = self._Lflat
            if optvars.shape[0] == self._Lflat.shape[0]:
                optvars = self._Lflat
        elif self.presolve_logistic:
            assert np.all(np.unique(y) == np.arange(self.k_))
            model_logistic = LogisticRegression(
                multi_class="multinomial",
                fit_intercept=False,
                C=0.5/max(1.,lam)
            ).fit(X,y)
            optvars[numL:] = (model_logistic.coef_[1:,:] - model_logistic.coef_[0,:]).reshape(-1)
            # now update chol(Sigma) indepdendently before proceeding
            coefs = optvars[numL:].reshape((self.k_-1,n))
            pred = X @ coefs.T
            args_onlyrho = (pred, y, sample_weights, self.k_, lam, self.fit_intercept, n_jobs)
            optvars_onlyrho = optvars[:numL]

            # Note: finite differencing here is roughly equally as slow as gradient calculations.
            # What's more, since the CDFs calculated here are not exact, the gradients for Rho
            # are usually more precise when obtained through finite differencing
            res = minimize(_mnp_fun_onlyrho, optvars_onlyrho, args_onlyrho, method="BFGS")
            optvars[:numL] = res["x"]

        finite_diff = self.grad == "finite_diff"
        args = (X, y, sample_weights, self.k_, lam, self.fit_intercept, n_jobs, finite_diff)
        res = minimize(_mnp_fun_grad, optvars, args, jac=True, method="BFGS")
        optvars = res["x"]

        # Parameters for the second-pass optimization procedure
        # (basically continuing the BFGS updated, but with a backtracking line search
        #  and with dampening for bad correction pairs)
        alpha = 0.001
        beta = 0.5
        maxiter = 10000
        maxls = 50

        bfgs = BFGS(exception_strategy="damp_update", min_curvature=1e-8)
        bfgs.initialize(n=optvars.shape[0], approx_type="inv_hess")
        bfgs.H = res["hess_inv"]
    
        for opt_iter in range(maxiter):
            f, g = _mnp_fun_grad(optvars, X, y, sample_weights, self.k_, lam, self.fit_intercept, n_jobs, finite_diff)

            if opt_iter > 0:
                bfgs.update(f - prev_f, g - prev_g)
            search_dir_ = -bfgs.dot(g)

            step = 1.0
            took_step = False

            for search_dir in [search_dir_, -g]:
                for ls in range(maxls):
                    newvars = optvars + step*search_dir
                    newf = _mnp_fun(newvars, X, y, sample_weights, self.k_, lam, self.fit_intercept, n_jobs, finite_diff)
                    if (ls == 0):
                        newf0 = newf
                    predf = step * np.dot(g, search_dir)


                    if predf < 0:
                        if (newf <= f + alpha*predf):
                            optvars = newvars
                            took_step = True
                            break
                    else:
                        if newf < f:
                            optvars = newvars
                            took_step = True
                            break
                    step *= beta

                if not took_step:
                    if newf0 < f:
                        optvars += search_dir
                        newf = newf0
                        took_step = True
                    elif newf < f:
                        optvars = newvars
                        took_step = True

                if took_step:
                    break

            if (not took_step) or np.abs(f - newf) <= 1e-8:
                break

            prev_g = g
            prev_f = newf

        L, class_Mats, class_Rhos, class_Rhos, class_vars, class_check_Rho \
            = _cpp_wrapper.wrapped_get_mnp_prediction_matrices(optvars, self.k_)
        self._class_Mats = class_Mats
        self._class_Rhos = class_Rhos
        self._class_vars = class_vars
        self._class_check_Rho = class_check_Rho
        self._Lflat = optvars

        self.ll_ = f / (2 * m)
        self._opt_iter = opt_iter
        self.Sigma_ = L @ L.T
        coefs = optvars[numL:].reshape((self.k_-1,n))
        if self.fit_intercept:
            self.intercept_ = coefs[:,0].reshape(-1)
            self.coef_ = coefs[:,1:]
        else:
            self.intercept_ = np.zeros(self.k_-1)
            self.coef_ = coefs
        return self

    def _get_class_scores(self, X):
        assert self.__is_fitted__()
        return (X @ self.coef_.T) + self.intercept_.reshape((1,-1))
    
    def _predict(self, X, logp):
        pred = self._get_class_scores(X)
        m = X.shape[0]
        return _cpp_wrapper.wrapped_mnp_classpred(
            m, self.k_, self._get_njobs(),
            pred,
            self._class_Mats,
            self._class_Rhos,
            self._class_vars,
            self._class_check_Rho,
            logp
        )

    def predict(self, X):
        """
        Predict the most likely class for each row

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            The data / covariates.

        Returns
        -------
        y_pred : array(n_samples,)
            The predicted class for each row.
        """
        classpreds = self._predict(X, False)
        return classpreds.argmax(axis=1)
    
    def predict_proba(self, X):
        """"
        Calculate the probability of each row being of each class
        
        Note
        ----
        The calculation of log-probabilities by the model uses a not-so-accurate
        approximation, thus the outputs for each row are rescaled to sum up
        exactly to one.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            The data / covariates.

        Returns
        -------
        class_probs : array(n_samples, n_classes)
            The predicted probabilities for each row and class
        """
        return self._predict(X, False)
    
    def decision_funtion(self, X):
        """
        Calculate log-probabilities for each row and class

        Note
        ----
        The calculation of log-probabilities by the model uses a not-so-accurate
        approximation, thus if these outputs are exponentiated, the values for
        each row might not sum up to exactly one.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            The data / covariates.

        Returns
        -------
        class_log_probs : array(n_samples, n_classes)
            The predicted log-probabilities for each row and class
        """
        return self._predict(X, True)
    
    def score(self, X, y, sample_weights=None):
        """
        Calculates the average log-likelihood

        Parameters
        ----------
        X : array-like (n_samples, n_features)
            The data / covariates.
        y : array (n_samples, )
            The labels for X (see the documentation for 'fit' for more details)
        sample_weights : None or array(n_samples,)
            Optional row weights.

        Returns
        -------
        score : float
            Obtained log-likelihood, divided by the number of rows
        """
        pred = self._get_class_scores(X)
        y, sample_weights = self._check_y_and_weights(y, sample_weights)
        return -_cpp_wrapper.wrapped_mnp_fun(
            y, pred, self._Lflat, sample_weights, self._get_njobs(),
        ) / X.shape[0]

    def __is_fitted__(self):
        return hasattr(self, "coef_")

    def _get_njobs(self):
        n_jobs = multiprocessing.cpu_count() + 1 + self.n_jobs
        if n_jobs < 1:
            raise ValueError("Passed invalid 'n_jobs'.")
        return n_jobs

    def _check_y_and_weights(self, y, sample_weights):
        y = np.require(y, dtype=ctypes.c_int, requirements=["C_CONTIGUOUS"])

        if sample_weights is not None:
            sample_weights = sample_weights.reshape(-1)
            assert sample_weights.shape[0] == y.shape[0]
            if not isinstance(sample_weights, np.ndarray):
                sample_weights = np.array(sample_weights)
            sample_weights = np.require(sample_weights, dtype=np.float64, requirements=["C_CONTIGUOUS"])
        else:
            sample_weights = np.ones(y.shape[0])
        return y, sample_weights

def _mnp_fun_grad(optvars, X, y, w, k, lam, fit_intercept, nthreads, finite_diff):
    m = X.shape[0]
    n = X.shape[1]
    numL = k + int(k*(k-1)/2) - 1
    Lflat = optvars[:numL]
    coefs = optvars[numL:].reshape((k-1,n))
    pred = X @ coefs.T

    fun, gradX, gradL = _cpp_wrapper.wrapped_mnp_fun_grad(
        y, pred, Lflat, w, nthreads, False, finite_diff
    )

    grad_out = np.empty(optvars.shape[0])
    grad_out[:numL] = gradL
    gradX = gradX.reshape((pred.shape[0], pred.shape[1]))

    slice_begin = numL
    slice_chunk = n
    for cl in range(k-1):
        if not w.shape[0]:
            grad_out[slice_begin:slice_begin+slice_chunk] = np.einsum("i,ij->j", gradX[:, cl], X)
        else:
            grad_out[slice_begin:slice_begin+slice_chunk] = np.einsum("i,ij->j", gradX[:, cl]*w, X)
        slice_begin += slice_chunk

    if lam:
        if fit_intercept:
            fun += lam*(np.dot(optvars[numL:], optvars[numL:]) - np.dot(coefs[:,0], coefs[:,0]))
        else:
            fun += lam*np.dot(optvars[numL:], optvars[numL:])
        grad_out[numL:] += 2*lam*optvars[numL:]
        ## don't regularize the intercepts
        if fit_intercept:
            slice_begin = numL
            for cl in range(k-1):
                grad_out[slice_begin] -= 2*lam*coefs[cl,0]
                slice_begin += slice_chunk

    return fun, grad_out

def _mnp_fun(optvars, X, y, w, k, lam, fit_intercept, nthreads, finite_diff):
    m = X.shape[0]
    n = X.shape[1]
    numL = k + int(k*(k-1)/2) - 1
    Lflat = optvars[:numL]
    coefs = optvars[numL:].reshape((k-1,n))
    pred = X @ coefs.T

    fun = _cpp_wrapper.wrapped_mnp_fun(
        y, pred, Lflat, w, nthreads
    )
    if lam:
        if fit_intercept:
            fun += lam*(np.dot(optvars[numL:], optvars[numL:]) - np.dot(coefs[:,0], coefs[:,0]))
        else:
            fun += lam*np.dot(optvars[numL:], optvars[numL:])
    return fun

def _mnp_fun_onlyrho(optvars, pred, y, w, k, lam, fit_intercept, nthreads):
    numL = k + int(k*(k-1)/2) - 1
    Lflat = optvars[:numL]

    return _cpp_wrapper.wrapped_mnp_fun(
        y, pred, Lflat, w, nthreads
    )
