#' @importFrom parallel detectCores
#' @importFrom utils head
#' @importFrom stats coef predict
#' @useDynLib multinomial.probit, .registration=TRUE
NULL

#' @export
#' @title Multinomial Probit Regression
#' @description Fits a multinomial probit regression (for multi-class classification with 3 or more classes), with
#' likelihoods and gradients computed through a fast approximation to the CDF of the MVN distribution
#' based on the TVBS method (see references for details) and optimized through the BFGS method.
#'
#' In general, the approximation for the likelihood and gradients is not good enough for the resulting
#' model to be very usable - typically, results are not competitive against simpler multinomial logistic
#' regression (which assumes a unit covariance matrix), and as such, is not recommended for serious usage.
#' @details If the data is a sparse matrix, it is highly recommended to load library
#' "MatrixExtra" before running this function, and to pass the "x" data as a CSR matrix of
#' class "dgRMatrix".
#'
#' Note that this library uses OpenMP for parallelization. As such, it is recommended to use
#' BLAS / LAPACK libraries that are OpenMP aware, such as `openblas-openmp` (which is usually
#' not the default openblas variant) or MKL (you might also want to set environment variable
#' `MKL_THREADING_LAYER=GNU`).
#' @references \enumerate{
#' \item Bhat, Chandra R.
#' "New matrix-based methods for the analytic evaluation of the multivariate cumulative normal distribution function."
#' Transportation Research Part B: Methodological 109 (2018): 238-256.
#' \item Plackett, Robin L. "A reduction formula for normal multivariate integrals." Biometrika 41.3/4 (1954): 351-360.
#' }
#' @examples
#' library(multinomial.probit)
#' data(iris)
#' x <- iris[, 1:4]
#' y <- iris$Species
#' m <- multinomial.probit(x, y)
#' predict(m, x, type="prob")
#' predict(m, x, type="class")
#' @param x The covariates. If passing a `data.frame`, will be encoded through `model.matrix`, and the user is responsible
#' for handling the levels of factor variables.
#' @param y The labels. Should be passed as a `factor` type.
#' @param weights Optional row weights.
#' @param lambda Amount of L2 regularization. Note that the regularization is added following GLMNET's formula.
#' @param intercept Whether to add intercepts to the coefficients for each class.
#' @param grad How to calculate gradients of the MVN CDF parameters.
#'
#' If passing "analytical", will use the theoretically correct formula for the gradients given by
#' the MVN PDF plus conditioned CDF (using Plackett's identity for the correlation coefficient
#' gradients), with the same CDF approximation as used for calculating the likelihood.
#'
#' If passing "finite_diff", will approximate the gradients through finite differencing.
#'
#' In theory, analytical gradients should be more accurate and faster, but in practive, since
#' the CDF is calculated by an imprecise approximation, the analytical gradients of the theoretical
#' CDF might not be too accurate as gradients of this approximation. Both approaches involve the
#' same number of CDF calculations, but for analytical gradients, the problems are reduced by
#' one / two dimensions, which makes them slightly faster.
#'
#' Note that this refers to gradients w.r.t the parameters of MVN distributions, not w.r.t to the
#' actual model parameters, which are still obtained by applying the chain rule.
#'
#' On smaller datasets in particular, finite differencing can result in more accurate gradients,
#' but on larger datasets, as errors average out, analytical gradients tend to be more accurate.
#' @param presolve_logistic Whether to pre-solve for the coefficients by fitting a multinomial \bold{logistic} regression
#' first and use that solution as starting point for optimization (after a smaller second
#' presolve pass solving only for the Cholesky of the covariance matrix).
#' @param nthreads Number of parallel threads to use.
#' @return The fitted model object (class "multinomial_probit"), on which methods such as
#' \link{predict.multinomial_probit} or \link{coef.multinomial_probit} can be called.
multinomial.probit <- function(
    x, y, weights=NULL,
    lambda = 0.,
    intercept=TRUE,
    grad=c("analytical", "finite_diff"),
    presolve_logistic=TRUE,
    nthreads=parallel::detectCores()
) {
    stopifnot(lambda >= 0)
    grad <- head(grad, 1)
    stopifnot(grad %in% c("analytical", "finite_diff"))
    stopifnot(nrow(x) == length(y))
    if (is.data.frame(x)) {
        if (intercept)
            x <- model.matrix(~ ., x)
        else
            x <- model.matrix(~ . - 1, x)
    } else if (intercept) {
        oldnames <- colnames(x)
        x <- cbind(matrix(rep(1, nrow(x)), nrow=nrow(x)), x)
        if (!is.null(oldnames)) {
            newnames <- c("(Intercept)", oldnames)
            colnames(x) <- newnames
        }
    }
    temp <- check.y.and.weights(y, weights)
    stopifnot(length(levels(y)) >= 3)
    ylevs <- levels(y)
    y <- temp[[1]]
    weights <- temp[[2]]
    k <- max(y) + 1L
    m <- nrow(x)
    n <- ncol(x)
    lam <- as.numeric(lambda) * m
    numL <- k + (k*(k-1)/2) - 1
    optvars <- .Call(R_get_mnp_starting_point, k, n)
    if (presolve_logistic) {
        if (requireNamespace("glmnet")) {
            if (intercept) {
                model <- glmnet::glmnet(
                    x[, 2:ncol(x)], y,
                    weights = weights,
                    family = "multinomial",
                    lambda = lambda, alpha = 0,
                    intercept = TRUE,
                    standardize = FALSE
                )
                coef <- sapply(coef(model), function(x) x[,ncol(x),drop=TRUE])
            } else {
                model <- glmnet::glmnet(
                    x, y,
                    weights = weights,
                    family = "multinomial",
                    lambda = lambda, alpha = 0,
                    intercept = FALSE,
                    standardize = FALSE
                )
                coef <- sapply(coef(model), function(x) x[,ncol(x),drop=TRUE])
                coef <- coef[2:nrow(coef),]
            }
            coef <- coef[,2:ncol(coef)] - coef[,1]
            optvars[(numL+1):length(optvars)] <- as.numeric(coef)

            newL <- optvars[1:numL]
            pred <- t(as.matrix(x %*% coef))
            res <- optim(
                newL, mnp.fun.onlyrho, NULL, pred, y, weights, nthreads,
                method = "BFGS",
                control = list(maxit = 10000)
            )
            optvars[1:numL] <- res$par
            rm(newL)
            rm(model)
            rm(coef)
        } else {
            stop("Package 'glmnet' is required for 'presolve_logistic=TRUE'.")
        }
    }

    finite_diff <- grad == "finite_diff"
    res <- optim(
        optvars, mnp.fun, mnp.grad,
        x, y, weights, k, lam, intercept, nthreads, finite_diff,
        method = "BFGS",
        control = list(maxit = 10000)
    )

    coefs <- matrix(res$par[(numL+1):length(optvars)], nrow=n)
    if (!is.null(colnames(x)))
        row.names(coefs) <- colnames(x)
    colnames(coefs) <- ylevs[2:k]
    temp <- .Call(R_get_mnp_prediction_matrices, k, res$par)
    this <- list(
        coefs = coefs,
        levs = ylevs,
        lambda = lambda,
        Chol = temp$L,
        Lflat = res$par[1:numL],
        Sigma = crossprod(temp$L),
        class_Mats = temp$class_Mats,
        class_Rhos = temp$class_Rhos,
        class_vars = temp$class_vars,
        class_check_Rho = temp$class_check_Rho,
        intercept = intercept,
        nthreads = nthreads,
        info = within(res, rm(par))
    )
    class(this) <- "multinomial_probit"
    return(this)
}

#' @export
#' @title Print method for multinomial probit models
#' @description Displays summary information from a multinomial probit
#' regression model.
#' @param x A multinomial probit regression model as output by function \link{multinomial.probit}.
#' @param ... Not used.
#' @return The same `x` argument, as invisible.
print.multinomial_probit <- function(x, ...) {
    cat("Multinomial Probit Model\n\n")
    cat(sprintf("Number of classes: %d\n", length(x$levs)))
    cat(sprintf("Number of features: %d\n", nrow(x$coefs) - as.numeric(as.logical(x$intercept))))
    cat(sprintf("Regularization: %f\n\n", x$lambda))
    cat("Coefficients:\n")
    print(head(x$coefs))
    return(invisible(x))
}

#' @export
#' @title Summary method for multinomial probit models
#' @description Displays summary information from a multinomial probit
#' regression model (same as method `print`).
#' @param object A multinomial probit regression model as output by function \link{multinomial.probit}.
#' @param ... Not used.
#' @return The same `object` argument, as invisible.
summary.multinomial_probit <- function(object, ...) {
    return(print.multinomial_probit(object))
}

#' @export
#' @title Extract multinomial probit model parameters
#' @description Extract the estimated coefficients and covariance from a
#' multinomial probit regression model.
#' @param object A multinomial probit regression model as output by function \link{multinomial.probit}.
#' @param ... Not used.
#' @return A list with two entries:\itemize{
#' \item `coef` Containing the fitted coefficients, as a matrix of dimensions [n_features + intercep, n_classes - 1].
#' The first level of `y` is used as the base class and thus has no coefficients (could be thought of as being all
#' zeros).
#' \item `Sigma` Containing the fitted class covariance matrix, as a square matrix of dimension `n_classes`.
#' The entry in the first row and first column is always fixed to 1.
#' }
coef.multinomial_probit <- function(object, ...) {
    return(list(coef = object$coefs, Sigma = object$Sigma))
}

#' @export
#' @title Prediction function for multinomial probit model
#' @description Calculate predictions from a multinomial probit
#' model on new data.
#' @details The calculation of log-probabilities by the model uses a not-so-accurate
#' approximation, thus the outputs for each row are rescaled to sum up
#' exactly to one.
#' @param object A multinomial probit regression model as output by function \link{multinomial.probit}.
#' @param newdata The new covariates for which to make predictions.
#' @param type The type of prediction to output. Options are:\itemize{
#' \item `"prob"` Will return the probabilities of each row belonging to each class (returned as a matrix).
#' \item `"class"` Will return the class with the highest predicted probability (returned as a factor vector).
#' \item `"logprob"` Will return log-probabilities for each row and class (returned as a matrix).
#' }
#' @param nthreads Number of parallel threads to use.
#' @return Either a matrix or a vector depending on argument `type`.
predict.multinomial_probit <- function(object, newdata, type = c("prob", "class", "logprob"), nthreads = parallel::detectCores()) {
    type <- head(type, 1)
    stopifnot(type %in% c("prob", "class", "logprob"))
    if (is.data.frame(newdata)) {
        newdata <- model.matrix(~ . - 1, newdata)
    }
    if (object$intercept) {
        pred <- newdata %*% object$coefs[2:nrow(object$coefs), , drop=FALSE]
        pred <- sweep(pred, 2, object$coefs[1,,drop=TRUE], `+`)
        pred <- t(pred)
    } else {
        pred <- t(newdata %*% object$coefs)
    }
    logp <- type == "logprob"
    m <- ncol(pred)
    res <- .Call(
        R_wrapped_mnp_classpred,
        m, nrow(object$Sigma), nthreads,
        pred,
        object$class_Mats,
        object$class_Rhos,
        object$class_vars,
        object$class_check_Rho,
        logp
    )
    res <- t(res)
    row.names(res) <- row.names(newdata)
    if (type != "class") {
        colnames(res) <- object$levs
    } else {
        res <- max.col(res)
        attributes(res)$levels <- object$levs
        attributes(res)$class <- "factor"
    }
    return(res)
}

#' @export
#' @title Multinomial Probit Log-Likelihood
#' @description Calculates the log-likelihood on a given set of covariates and labels
#' from a fitted multinomial probit model.
#' @param model A multinomial probit regression model as output by function \link{multinomial.probit}.
#' @param x The covariates. If it is passed as a `data.frame`, factor columns should have
#' the same levels as the `x` data to which the model was fitted. Note that this package
#' does not perform column matching - the number of columns must match exactly with the
#' `x` data to which the model was fitted.
#' @param y The labels. Should be passed as a factor variable, with the same levels as the
#' `y` to which the model was fitted.
#' @param weights Optional row weights.
#' @param nthreads Number of parallel threads to use.
#' @return The obtained log-likelihood (a negative number, with higher values meaning a better
#' math between predictions and labels).
mnp.likelihood <- function(model, x, y, weights = NULL, nthreads = parallel::detectCores()) {
    if (is.data.frame(x)) {
        x <- model.matrix(~ . - 1, x)
    }
    if (model$intercept) {
        pred <- x %*% model$coefs[2:nrow(model$coefs), , drop=FALSE]
        pred <- sweep(pred, 2, model$coefs[1,,drop=TRUE], `+`)
        pred <- t(pred)
    } else {
        pred <- t(as.matrix(x %*% model$coefs))
    }
    temp <- check.y.and.weights(y, weights)
    y <- temp[[1]]
    weights <- temp[[2]]
    res <- .Call(R_wrapped_mnp_fun, y, pred, model$Lflat, weights, nthreads)
    return(-res)
}

check.y.and.weights <- function(y, weights) {
    stopifnot(is.factor(y))
    y <- as.integer(y) - 1L
    if (!is.null(weights)) {
        weights <- as.numeric(weights)
        stopifnot(length(weights) == length(y))
    } else {
        weights <- rep(1., length(y))
    }
    return(list(y, weights))
}

mnp.grad <- function(optvars, X, y, w, k, lam, intercept, nthreads, finite_diff) {
    m <- nrow(X)
    n <- ncol(X)
    numL <- k + (k*(k-1)/2) - 1
    coefs <- matrix(optvars[(numL+1L):length(optvars)], nrow=n)
    pred <- t(as.matrix(X %*% coefs))
    temp <- .Call(
        R_wrapped_mnp_fun_grad,
        y, pred, optvars, w, nthreads, FALSE, finite_diff
    )
    gradX <- matrix(temp$gradX, nrow=nrow(pred))
    gradL <- temp$gradL

    grad.out <- numeric(length(optvars))
    grad.out[1L:numL] <- gradL

    slice_begin <- numL
    slice_chunk <- n
    for (cl in 1:(k-1)) {
        if (!NROW(w)) {
            grad.out[(slice_begin+1L):(slice_begin+slice_chunk)] <- as.numeric(gradX[cl, , drop=TRUE] %*% X)
        } else {
            grad.out[(slice_begin+1L):(slice_begin+slice_chunk)] <- as.numeric((w * gradX[cl, , drop=TRUE]) %*% X)
        }
        slice_begin <- slice_begin + slice_chunk
    }

    if (lam) {
        grad.out[(numL+1L):length(grad.out)] <- grad.out[(numL+1L):length(grad.out)] + 2*lam*optvars[(numL+1L):length(grad.out)]
        if (intercept) {
            slice_begin <- numL
            for (cl in 1:(k-1)) {
                grad.out[slice_begin+1L] <- grad.out[slice_begin+1L] - 2*lam*coefs[1,cl]
                slice_begin <- slice_begin + slice_chunk
            }
        }
    }
    return(grad.out)
}

mnp.fun <- function(optvars, X, y, w, k, lam, intercept, nthreads, finite_diff) {
    m <- nrow(X)
    n <- ncol(X)
    numL <- k + (k*(k-1)/2) - 1
    coefs <- matrix(optvars[(numL+1L):length(optvars)], nrow=n)
    pred <- t(as.matrix(X %*% coefs))

    fun <- .Call(
        R_wrapped_mnp_fun,
        y, pred, optvars, w, nthreads
    )
    if (lam) {
        if (intercept) {
            fun <- fun + lam*(as.numeric(coefs) %*% as.numeric(coefs) - coefs[1,,drop=TRUE] %*% coefs[1,,drop=TRUE])
        } else {
            fun <- fun + lam*as.numeric(coefs) %*% as.numeric(coefs)
        }
    }
    return(fun)
}

mnp.fun.onlyrho <- function(optvars, pred, y, w, nthreads) {
    return(.Call(R_wrapped_mnp_fun, y, pred, optvars, w, nthreads))
}
