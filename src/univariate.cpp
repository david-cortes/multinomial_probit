#include "multinomial_probit.h"

/* Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

/* Adapted from SciPy:
   https://github.com/scipy/scipy/blob/main/scipy/stats/_continuous_distns.py */
static const double log_sqrt_twoPI = std::log(std::sqrt(2. * M_PI));
double norm_logpdf_1d(double x)
{
    return -0.5 * (x*x) - log_sqrt_twoPI;
}

/* Adapted from cephes */
const static double inv_sqrt2 = 1. / std::sqrt(2.);
double norm_cdf_1d(double a)
{
    if (std::isinf(a)) return 0.;
    double x, y, z;
    x = a * inv_sqrt2;
    z = std::fabs(x);

    if (z < inv_sqrt2) {
        y = .5 + .5*std::erf(x);
    }
    else {
        y = .5*std::erfc(z);
        if(x > 0.) {
            y = 1. - y;
        }
    }
    return y;
}

/* Adapted from SciPy:
   https://github.com/scipy/scipy/blob/8a64c938ddf1ae4c02a08d2c5e38daeb8d061d38/scipy/special/cephes/ndtr.c */
static const double half_log_twoPI = 0.5 * std::log(2. * M_PI);;
double norm_logcdf_1d(double a)
{
    if (std::isinf(a)) {
        return (a >= 0.)? 0. : -std::numeric_limits<double>::infinity();
    }
    const double a_sq = a * a;
    double log_LHS;              /* we compute the left hand side of the approx (LHS) in one shot */
    double last_total = 0;       /* variable used to check for convergence */
    double right_hand_side = 1;  /* includes first term from the RHS summation */
    double numerator = 1;        /* numerator for RHS summand */
    double denom_factor = 1;     /* use reciprocal for denominator to avoid division */
    double denom_cons = 1./a_sq; /* the precomputed division we use to adjust the denominator */
    long sign = 1;
    long i = 0;

    if (a > 6.) {
        return -norm_cdf_1d(-a);        /* log(1+x) \approx x */
    }
    if (a > -20.) {
        return std::log(norm_cdf_1d(a));
    }
    log_LHS = -0.5*a_sq - std::log(-a) - half_log_twoPI;

    while (std::fabs(last_total - right_hand_side) > std::numeric_limits<double>::epsilon()) {
        i++;
        last_total = right_hand_side;
        sign = -sign;
        denom_factor *= denom_cons;
        numerator *= 2 * i - 1;
        right_hand_side = std::fma(sign*numerator, denom_factor, right_hand_side);
    }
    
    return log_LHS + std::log(right_hand_side);
}
