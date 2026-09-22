// Standalone: c++ -std=c++17 -O2 -Isolver/include tests/shell_modes.cpp
// solver/src/shellModes.cpp -lgsl -lgslcblas -o /tmp/shellModesTest
#include "shellModes.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

using Complex = std::complex<double>;
using dsolve::ShellModeTransform;

template <class F>
std::vector<Complex> sample(unsigned int nt, unsigned int np, F f) {
    const double pi = std::acos(-1.0);
    std::vector<Complex> values{f(0, 0)};
    for (unsigned int i = 1; i + 1 < nt; ++i)
        for (unsigned int j = 0; j < np; ++j)
            values.push_back(f(pi * i / (nt - 1), 2 * pi * j / np));
    values.push_back(f(pi, 0));
    return values;
}

void check(unsigned int nt, unsigned int np, int limit) {
    const double pi = std::acos(-1.0);
    const ShellModeTransform transform(nt, np, limit);
    // Independent closed-form Y_21 tests phase, conjugation and normalization.
    auto y21 = [pi](double t, double p) {
        return -std::sqrt(15 / (8 * pi)) * std::sin(t) * std::cos(t) *
               std::polar(1.0, p);
    };
    double worst = 0;
    for (unsigned int test = 0; test < 4; ++test) {
        const auto a = transform.project(sample(nt, np, [&](double t, double p) {
            if (test == 0) return Complex(1, 0);
            if (test == 1) return y21(t, p);
            if (test == 2) return -std::conj(y21(t, p)); // Y_2,-1
            return Complex(y21(t, p).real(), 0);
        }));
        double targetError = 0, leakage = 0;
        for (int l = 0; l <= limit; ++l)
            for (int m = -l; m <= l; ++m) {
                double expected = 0;
                if (test == 0 && l == 0 && m == 0) expected = std::sqrt(4 * pi);
                if (test == 1 && l == 2 && m == 1) expected = 1;
                if (test == 2 && l == 2 && m == -1) expected = 1;
                if (test == 3 && l == 2 && m == 1) expected = 0.5;
                if (test == 3 && l == 2 && m == -1) expected = -0.5;
                const double err = std::abs(a[ShellModeTransform::index(l, m)] - expected);
                if (expected != 0) targetError = std::max(targetError, err);
                else leakage = std::max(leakage, err);
            }
        std::cout << "grid=" << nt << 'x' << np << " lmax=" << limit
                  << " field=" << (test == 0 ? "constant" : test == 1 ? "Y21" :
                                    test == 2 ? "Y2,-1" : "Re(Y21)")
                  << " target_error=" << targetError << " leakage=" << leakage << '\n';
        worst = std::max({worst, targetError, leakage});
    }
    if (worst > 2e-12) throw std::runtime_error("Harmonic recovery failed");
}

int main() {
    try {
        check(65, 128, 8);
        check(18, 25, 8); // odd Clenshaw–Curtis interval count
        check(5, 5, 2);   // minimum resolution for lmax=2
        const auto monopole = ShellModeTransform(3, 3, 0).project(
            std::vector<Complex>(5, 1.0));
        if (std::abs(monopole[0] - std::sqrt(4 * std::acos(-1.0))) > 2e-12)
            throw std::runtime_error("LMAX=0 failed");
        for (const auto& grid : {std::pair<unsigned int, unsigned int>{16, 25}, {17, 16}}) {
            bool rejected = false;
            try { ShellModeTransform invalid(grid.first, grid.second, 8); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Undersampled grid accepted");
        }
        std::cout << "PASS\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
