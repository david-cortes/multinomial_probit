#include "multinomial_probit.h"

double norm_logcdf_2d(double x1, double x2, double rho)
{
    double abs_rho = std::fabs(rho);
    if (unlikely(abs_rho <= LOW_RHO)) {
        return norm_logcdf_1d(x1) + norm_logcdf_1d(x2);
    }
    else if (unlikely(abs_rho >= HIGH_RHO)) {
        if (rho >= 0.) {
            return norm_logcdf_1d(std::fmin(x1, x2));
        }
        else {
            return norm_logcdf_1d(std::fmin(x1, -x2));
        }
    }
    if (x2 < x1) {
        std::swap(x1, x2);
    }
    double log_d1 = norm_logpdf_1d(x1);
    double log_p1 = norm_logcdf_1d(x1);
    double log_l1 = log_d1 - log_p1;
    double sign_l1 = -1.;
    double log_rho = std::log(std::fabs(rho));
    double sign_rho = (rho >= 0.)? 1. : -1.;
    double log_rl1 = log_rho + log_l1;
    double sign_rl1 = sign_rho * sign_l1;
    double log_x1 = std::log(std::fabs(x1));
    double sign_x1 = (x1 >= 0.)? 1. : -1.;

    double v2 = sign_rho * sign_x1 * sign_rl1 * std::exp(log_rho + log_x1 + log_rl1);
    double rl1 = sign_rl1 * std::exp(log_rl1);
    v2 += std::fma(-rl1, rl1, 1.);
    v2 = std::fmax(v2, std::numeric_limits<double>::min());

    return norm_logcdf_1d(x1) + norm_logcdf_1d((x2 - rl1) / std::sqrt(v2));
}

double norm_logcdf_3d(double x1, double x2, double x3, double rho12, double rho13, double rho23)
{
    if (unlikely(rho12*rho12 + rho13*rho13 <= EPS_BLOCK)) {
        return norm_logcdf_1d(x1) + norm_logcdf_2d(x2, x3, rho23);
    }
    else if (unlikely(rho12*rho12 + rho23*rho23 <= EPS_BLOCK)) {
        return norm_logcdf_1d(x2) + norm_logcdf_2d(x1, x3, rho13);
    }
    else if (unlikely(rho13*rho13 + rho23*rho23 <= EPS_BLOCK)) {
        return norm_logcdf_1d(x3) + norm_logcdf_2d(x1, x2, rho12);
    }

    if (x3 < x2) {
        std::swap(x2, x3);
        std::swap(rho12, rho13);
    }
    if (x2 < x1) {
        std::swap(x1, x2);
        std::swap(rho13, rho23);
    }

    double temp = norm_logpdf_1d(x1) - norm_logcdf_1d(x1);
    double mutilde = -std::exp(temp);
    double omega = 1. + (mutilde * (x1 - mutilde));

    double rho12_sq = rho12 * rho12;
    double rho13_sq = rho13 * rho13;
    double omega_m1 = omega - 1.;

    double t1 = std::fma(rho12_sq, omega_m1, 1.);
    t1 = std::fmax(t1, std::numeric_limits<double>::min());
    double t2 = std::fma(rho13_sq, omega_m1, 1.);
    t2 = std::fmax(t2, std::numeric_limits<double>::min());

    double s11 = std::sqrt(t1);
    double s22 = std::sqrt(t2);
    double v12 = rho23 + rho12 * rho13 * omega_m1;

    double p1 = norm_logcdf_2d(x1, x2, rho12);
    double p2 = norm_logcdf_2d(
        std::fma(-rho12, mutilde, x2) / s11,
        std::fma(-rho13, mutilde, x3) / s22,
        v12 / (s11 * s22)
    );
    double p3 = norm_logcdf_1d(std::fma(-rho12, mutilde, x2) / s11);
    return p1 + p2 - p3;
}

void bv_trunc_std4d_loweronly(const double rho[6], const double tp[2],
                              double *restrict mu_out, double *restrict Omega_out)
{
    double detV11 = std::fma(-rho[0], rho[0], 1.);
    double invV11v = 1. / detV11;
    double invV11d = -rho[0] / detV11;
    double reg = 1e-8;
    double d = 1.;
    while (detV11 <= 0.) {
        d += reg;
        detV11 = std::fma(-rho[0], rho[0], d*d);
        invV11v = d / detV11;
        invV11d = -rho[0] / detV11;
        reg *= 1.5;
    }

    double Omega11[3];
    double mu_half[2];
    truncate_bvn_2by2block(0., 0., 1., 1., rho[0], tp[0], tp[1],
                           mu_half[0], mu_half[1],
                           Omega11[0], Omega11[1], Omega11[2]);

    mu_out[0] = (invV11v*rho[1] + invV11d*rho[3]) * (mu_half[0]) +
                (invV11d*rho[1] + invV11v*rho[3]) * (mu_half[1]);
    mu_out[1] = (invV11v*rho[2] + invV11d*rho[4]) * (mu_half[0]) +
                (invV11d*rho[2] + invV11v*rho[4]) * (mu_half[1]);

    double Omega11_invV11[] = {
        Omega11[0]*invV11v + Omega11[2]*invV11d, Omega11[0]*invV11d + Omega11[2]*invV11v,
        Omega11[2]*invV11v + Omega11[1]*invV11d, Omega11[2]*invV11d + Omega11[1]*invV11v
    };
    /* O12 */
    double O12[] = {
        Omega11_invV11[0]*rho[1] + Omega11_invV11[1]*rho[3], Omega11_invV11[0]*rho[2] + Omega11_invV11[1]*rho[4],
        Omega11_invV11[2]*rho[1] + Omega11_invV11[3]*rho[3], Omega11_invV11[2]*rho[2] + Omega11_invV11[3]*rho[4]
    };
    /* V12 - O12 */
    double temp1[] = {
        rho[1] - O12[0], rho[2] - O12[1],
        rho[3] - O12[2], rho[4] - O12[3]
    };
    /* iV11 * (V12 - O12) */
    double temp2[] = {
        invV11v*temp1[0] + invV11d*temp1[2], invV11v*temp1[1] + invV11d*temp1[3],
        invV11d*temp1[0] + invV11v*temp1[2], invV11d*temp1[1] + invV11v*temp1[3]
    };
    /* V22 - V21 * (iV11 * (V12 - O12)) */
    Omega_out[0] = 1. - rho[1]*temp2[0] - rho[3]*temp2[2];
    Omega_out[1] = 1. - rho[2]*temp2[1] - rho[4]*temp2[3];
    Omega_out[2] = rho[5] - rho[1]*temp2[1] - rho[3]*temp2[3];

    Omega_out[0] = std::fmax(Omega_out[0], 0.0000005);
    Omega_out[1] = std::fmax(Omega_out[1], 0.0000005);
}

/* A[ind,ind], when 'A' is represented by its upper triangle only. */
static inline
void rearrange_tri(double x[6], int ordering[4])
{
    double Xfull[] = {
        1.,    x[0],   x[1],   x[2],
        x[0],     1.,  x[3],   x[4],
        x[1],  x[3],     1.,   x[5],
        x[2],  x[4],   x[5],     1.
    };

    double Xnew[16];
    const int n = 4;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            Xnew[col + row*n] = Xfull[ordering[col] + ordering[row]*n];
        }
    }

    x[0] = Xnew[1];
    x[1] = Xnew[2];
    x[2] = Xnew[3];
    x[3] = Xnew[6];
    x[4] = Xnew[7];
    x[5] = Xnew[11];
}

double norm_logcdf_4d_internal(const double x[4], const double rho[6])
{
    double mu[2];
    double Sigma[3];
    bv_trunc_std4d_loweronly(rho, x, mu, Sigma);

    double p1 = norm_logcdf_1d((x[2] - mu[0]) / std::sqrt(Sigma[0]));
    double p2 = norm_logcdf_2d(
        (x[2] - mu[0]) / std::sqrt(Sigma[0]),
        (x[3] - mu[1]) / std::sqrt(Sigma[1]),
        Sigma[2] / (std::sqrt(Sigma[0]) * std::sqrt(Sigma[1]))
    );
    double p3 = norm_logcdf_3d(x[0], x[1], x[2], rho[0], rho[1], rho[3]);
    return p2 - p1 + p3;
}

double norm_logcdf_4d(const double x[4], const double rho[6])
{
    const double rho_sq[] = {
        rho[0]*rho[0], rho[1]*rho[1], rho[2]*rho[2], rho[3]*rho[3], rho[4]*rho[4], rho[5]*rho[5]
    };
    if (unlikely(rho_sq[1] + rho_sq[2] + rho_sq[3] + rho_sq[4] < EPS_BLOCK)) {
        return norm_logcdf_2d(x[2], x[3], rho[5]) + norm_logcdf_2d(x[0], x[1], rho[0]);
    }
    else if (unlikely(rho_sq[0] + rho_sq[1] + rho_sq[4] + rho_sq[5] < EPS_BLOCK)) {
        return norm_logcdf_2d(x[0], x[3], rho[2]) + norm_logcdf_2d(x[2], x[1], rho[3]);
    }
    else if (unlikely(rho_sq[0] + rho_sq[2] + rho_sq[3] + rho_sq[5] < EPS_BLOCK)) {
        return norm_logcdf_2d(x[2], x[0], rho[1]) + norm_logcdf_2d(x[3], x[1], rho[4]);
    }
    else if (unlikely(rho_sq[0] + rho_sq[1] + rho_sq[2] < EPS_BLOCK)) {
        return norm_logcdf_1d(x[0]) + norm_logcdf_3d(x[1], x[2], x[3], rho[3], rho[4], rho[5]);
    }
    else if (unlikely(rho_sq[0] + rho_sq[3] + rho_sq[4] < EPS_BLOCK)) {
        return norm_logcdf_1d(x[1]) + norm_logcdf_3d(x[0], x[2], x[3], rho[1], rho[2], rho[5]);
    }
    else if (unlikely(rho_sq[1] + rho_sq[3] + rho_sq[5] < EPS_BLOCK)) {
        return norm_logcdf_1d(x[2]) + norm_logcdf_3d(x[0], x[1], x[3], rho[0], rho[2], rho[4]);
    }
    else if (unlikely(rho_sq[2] + rho_sq[4] + rho_sq[5] < EPS_BLOCK)) {
        return norm_logcdf_1d(x[3]) + norm_logcdf_3d(x[0], x[1], x[2], rho[0], rho[1], rho[3]);
    }
    /* If rho(1,2):+1 -> x1 ==  x2
       If rho(1,2):-1 -> x1 == -x2 */
    else if (unlikely(std::fabs(rho[0]) >= HIGH_RHO)) {
        if (x[0] <= rho[0] * x[1]) {
            return norm_logcdf_3d(x[0], x[2], x[3], rho[1], rho[2], rho[5]);
        }
        else {
            return norm_logcdf_3d(rho[0]*x[1], x[2], x[3], rho[3], rho[4], rho[5]);
        }
    }
    else if (unlikely(std::fabs(rho[1]) >= HIGH_RHO)) {
        if (x[0] <= rho[1] * x[2]) {
            return norm_logcdf_3d(x[0], x[1], x[3], rho[0], rho[2], rho[4]);
        }
        else {
            return norm_logcdf_3d(rho[1]*x[2], x[1], x[3], rho[3], rho[5], rho[4]);
        }
    }
    else if (unlikely(std::fabs(rho[2]) >= HIGH_RHO)) {
        if (x[0] <= rho[2] * x[3]) {
            return norm_logcdf_3d(x[0], x[1], x[2], rho[0], rho[1], rho[3]);
        }
        else {
            return norm_logcdf_3d(rho[2]*x[3], x[1], x[2], rho[4], rho[5], rho[3]);
        }
    }
    else if (unlikely(std::fabs(rho[3]) >= HIGH_RHO)) {
        if (x[1] <= rho[3] * x[2]) {
            return norm_logcdf_3d(x[1], x[0], x[3], rho[0], rho[4], rho[2]);
        }
        else {
            return norm_logcdf_3d(rho[3]*x[2], x[0], x[3], rho[1], rho[5], rho[2]);
        }
    }
    else if (unlikely(std::fabs(rho[4]) >= HIGH_RHO)) {
        if (x[1] <= rho[2] * x[3]) {
            return norm_logcdf_3d(x[1], x[0], x[2], rho[0], rho[3], rho[1]);
        }
        else {
            return norm_logcdf_3d(rho[2]*x[3], x[0], x[2], rho[2], rho[5], rho[1]);
        }
    }
    else if (unlikely(std::fabs(rho[5]) >= HIGH_RHO)) {
        if (x[2] <= rho[5] * x[3]) {
            return norm_logcdf_3d(x[2], x[0], x[1], rho[1], rho[3], rho[0]);
        }
        else {
            return norm_logcdf_3d(rho[5]*x[3], x[0], x[1], rho[2], rho[4], rho[0]);
        }
    }

    int argsorted[] = {0, 1, 2, 3};
    std::sort(argsorted, argsorted + 4, [&x](const int a, const int b){return x[a] < x[b];});
    const double xpass[] = {x[argsorted[0]], x[argsorted[1]], x[argsorted[2]], x[argsorted[3]]};
    double rhopass[6];
    std::copy(rho, rho + 6, rhopass);
    rearrange_tri(rhopass, argsorted);
    return norm_logcdf_4d_internal(xpass, rhopass);
}
