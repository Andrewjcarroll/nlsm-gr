#ifndef NLSM_SHELL_MODES_H
#define NLSM_SHELL_MODES_H

#include <complex>
#include <cstddef>
#include <vector>

namespace dsolve {
// Scalar orthonormal complex Y_lm, including the Condon–Shortley phase:
// Y_lm = sqrt((2l+1)/(4pi) (l-m)!/(l+m)!) P_l^m(cos(theta)) exp(i m phi).
// P includes (-1)^m; Y_l,-m = (-1)^m conj(Y_lm). Projection uses conj(Y).
// Samples: north pole, interior theta rings (phi varying fastest), south pole.
// theta_i=pi*i/(nt-1), phi_j=2*pi*j/np; poles occur only once.
class ShellModeTransform {
 public:
    ShellModeTransform(unsigned int nt, unsigned int np, unsigned int lmax);
    std::vector<std::complex<double>> project(
        const std::vector<std::complex<double>>& samples) const;
    static std::size_t index(int l, int m) {
        return std::size_t(l) * l + std::size_t(l + m);
    }

 private:
    unsigned int nt_, np_, lmax_;
    std::vector<double> weights_;  // Clenshaw–Curtis weights for d(cos(theta)).
};
}  // namespace dsolve
#endif
