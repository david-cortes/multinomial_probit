# Multinomial Probit Regression

This is an R / Python package for fitting multinomial probit (MNP) models, a form of generalized linear regression model for multi-class classification.

It uses a fast approximation for the CDF of multivariate normal distributions alongside with analytical gradients, which is not accurate enough to make the models competitive against logistic ones in practice (see below for more details), and as such, is not recommended for serious usage but is nevertheless made available for research purposes.

Note: one is more likely to see better results in the R interface compared to the Python one, due to a more careful implementation of the BFGS optimization procedure in R as comapred to SciPy.

# Description

When dealing with multi-class classification, one typically uses models such as multinomial logistic regression or separate one-vs-rest binary logistic regressions. Under these models, predictions for each class are implicitly assumed as being independent - that is, increasing the prediction for one class will not affect the predictions for other classes.

Multinomial probit regression, on the other hand, explicitly models the correlations among the predictions of each class, and models the variance of each class' predictions - thus, if e.g. the prediction / score for one class increases, then the probabilities for every other class will shift accordingly, given the correlation matrix, which is a property that is not shared by e.g. one-vs-rest logistic regressions.

In more detail, if we take a given row and fitted model parameters, a multinomial probit model with $k$ classes would make predictions through a latent process as follows:

$$
\text{pred} = \mathbf{\Beta^{k, n}} \mathbf{x^n}
$$

$$
\text{score} = \text{pred} + \mathbf{\epsilon^k}
$$

$$
\mathbf{\epsilon} \sim \text{MVN}(\mathbf{0}, \mathbf{\Sigma})
$$

(where $\mathbf{\Beta^{k, n}}$ and $\mathbf{\Sigma}$ are model parameters, and MVN denotes the multivariate normal distribution)

Then, from $\text{score}^k$, the probability that this row will belong to each class is given by the probability that each corresponding entry in $\text{score}^k$ is higher than the other entries (considering that it contains random multivariate-distributed noise).

The calculation of these probabilities involves calculating the CDF (cumulative distribution function) of multivariate normal distributions, whose calculation at high accuracy is not computationally tractable - as such, this type of model is typically fitted through maximum simulated likelihood methods with limited random samples, but this package instead uses much faster and stable approximations to the MVN CDF given by the TVBS method (see references for more details) and follows a typicall optimization procedure (BFGS) with analytical gradients (no randomization is involved).

In theory, this model is more expressive than others and should be able to capture more complex relationships in the data and provide better coefficients and estimates.

In practice, the approximation is not good enough for a model fitted by maximizing this approximated likelihood to be competitive against multinomial logistic - for example, it does not manage to achieve 100% accuracy on the iris dataset - and as such, is not recommended for serious usage.

Note that this package only provides functionality for fitting the model parameters and producing predictions (i.e. no calculation of standard errors, p-values, or variance-covariance of predictors).

# Installation

* R:

```r
remotes::install_github("david-cortes/multinomial_probit")
```

* Python:

```shell
pip install git+https://github.com/david-cortes/multinomial_probit.git
```

** *
**IMPORTANT:** the setup script will try to add compilation flag `-march=native`. This instructs the compiler to tune the package for the CPU in which it is being installed (by e.g. using AVX instructions if available), but the result might not be usable in other computers. If building a binary wheel of this package or putting it into a docker image which will be used in different machines, this can be overriden either by (a) defining an environment variable `DONT_SET_MARCH=1`, or by (b) manually supplying compilation `CFLAGS` as an environment variable with something related to architecture. For maximum compatibility (but slowest speed), it's possible to do something like this:

```
export DONT_SET_MARCH=1
pip install git+https://github.com/david-cortes/multinomial_probit.git
```
** *

# Sample usage

* R:

```r
library(multinomial.probit)
data(iris)
x <- iris[, 1:4]
y <- iris$Species
m <- multinomial.probit(x, y)
predict(m, x, type="prob")
predict(m, x, type="class")
mnp.likelihood(m, x, y)
```

* Python:

(_Recommended to use the R interface as the BFGS optimizer is better there_)

(The package is scikit-learn-compatible and resembles their `LogisticRegression` class)

```python
from multinomial_probit import MultinomialProbitRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = MultinomialProbitRegression()
model.fit(X, y)
model.predict(X)
model.predict_proba(X)
model.score(X, y)
```

# References

* Bhat, Chandra R. "New matrix-based methods for the analytic evaluation of the multivariate cumulative normal distribution function." Transportation Research Part B: Methodological 109 (2018): 238-256.
* Plackett, Robin L. "A reduction formula for normal multivariate integrals." Biometrika 41.3/4 (1954): 351-360.
* Bolduc, Denis. "A practical technique to estimate multinomial probit models in transportation." Transportation Research Part B: Methodological 33.1 (1999): 63-79.
