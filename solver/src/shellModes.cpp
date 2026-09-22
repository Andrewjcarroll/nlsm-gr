#include "shellModes.h"

#include <gsl/gsl_sf_legendre.h>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dsolve {
ShellModeTransform::ShellModeTransform(unsigned int nt, unsigned int np,
                                     unsigned int lmax)
    : nt_(nt), np_(np), lmax_(lmax) {
    // Integrate products through degree 2*lmax and avoid azimuthal aliasing
    // within the retained band. Unresolved higher input modes can still alias.
    if (nt < 3 || np < 3 || lmax > (nt - 1) / 2 || lmax > (np - 1) / 2 ||
        lmax > static_cast<unsigned int>(std::numeric_limits<int>::max() / 2))
        throw std::invalid_argument(
            "Shell modes require N_THETA >= 2*LMAX+1 and N_PHI >= 2*LMAX+1 "
            "(both angular counts >= 3)");
    const unsigned int n = nt - 1;
    const double pi = std::acos(-1.0), nn = double(n) * n;
    weights_.resize(nt);
    weights_.front() = weights_.back() = 1.0 / (nn - (n % 2 == 0 ? 1.0 : 0.0));
    for (unsigned int i = 1; i < n; ++i) {
        const double theta = pi * i / n;
        double v = 1.0;
        for (unsigned int k = 1; k <= (n - 1) / 2; ++k)
            v -= 2 * std::cos(2 * k * theta) / (4.0 * k * k - 1);
        if (n % 2 == 0) v -= std::cos(n * theta) / (nn - 1);
        weights_[i] = 2 * v / n;
    }
}

std::vector<std::complex<double>> ShellModeTransform::project(
    const std::vector<std::complex<double>>& samples) const {
    const std::size_t count = 2 + std::size_t(nt_ - 2) * np_;
    if (samples.size() != count)
        throw std::invalid_argument("Shell mode sample count does not match grid");
    for (const auto& f : samples)
        if (!std::isfinite(f.real()) || !std::isfinite(f.imag()))
            throw std::invalid_argument("Non-finite shell mode sample");
    const double pi = std::acos(-1.0);
    const int limit = static_cast<int>(lmax_);
    std::vector<std::complex<double>> result(std::size_t(lmax_ + 1) * (lmax_ + 1));
    std::vector<double> legendre(gsl_sf_legendre_array_n(lmax_));
    for (unsigned int i = 0; i < nt_; ++i) {
        const bool pole = i == 0 || i + 1 == nt_;
        const double x = i == 0 ? 1.0 : (i + 1 == nt_ ? -1.0 :
                         std::cos(pi * i / (nt_ - 1)));
        // Explicit -1 selects the Condon–Shortley phase in GSL's array API.
        if (gsl_sf_legendre_array_e(GSL_SF_LEGENDRE_SPHARM, lmax_, x, -1.0,
                                    legendre.data()) != 0)
            throw std::runtime_error("Shell harmonic evaluation failed");
        for (int m = -limit; m <= limit; ++m) {
            if (pole && m != 0) continue;  // Y_lm vanishes at poles for m != 0.
            std::complex<double> azimuth = 0;
            if (pole) {
                azimuth = 2 * pi * (i == 0 ? samples.front() : samples.back());
            } else {
                const std::size_t offset = 1 + std::size_t(i - 1) * np_;
                for (unsigned int j = 0; j < np_; ++j)
                    azimuth += samples[offset + j] *
                               std::polar(1.0, -2 * pi * m * j / np_);
                azimuth *= 2 * pi / np_;
            }
            const int am = std::abs(m);
            const double sign = m < 0 && am % 2 ? -1.0 : 1.0;
            for (int l = am; l <= limit; ++l)
                result[index(l, m)] += weights_[i] * sign * azimuth *
                    legendre[gsl_sf_legendre_array_index(l, am)];
        }
    }
    return result;
}
}  // namespace dsolve
