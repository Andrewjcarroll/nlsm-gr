#include "shellOutput.h"
#include "shellModes.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "grDef.h"
#include "parameters.h"
#include "shell_compat/daUtils.h"

namespace dsolve {
namespace {
// The solver allocates legacy extra slots; only these named fields are evolved.
constexpr unsigned int shellVariables =
    sizeof(SOLVER_VAR_NAMES) / sizeof(SOLVER_VAR_NAMES[0]);
struct Shell {
    std::vector<double> xyz, theta, phi;
    std::vector<unsigned int> connectivity, offsets, types;
};

Shell makeShell() {
    const auto nt = SOLVER_SHELL_N_THETA, np = SOLVER_SHELL_N_PHI;
    const double r = SOLVER_SHELL_RADIUS, pi = std::acos(-1.0);
    const uint64_t count = 2 + uint64_t(nt - 2) * np;
    if (nt < 3 || np < 3 || count > INT_MAX / 4 || !std::isfinite(r) ||
        r <= 0)
        throw std::runtime_error(
            "Invalid shell radius, resolution or frequency");
    // Dendro's SearchKey represents half-open octants. Keep the whole surface
    // strictly inside the domain, including its extrema between sample points.
    for (unsigned int d = 0; d < 3; ++d)
        if (!std::isfinite(SOLVER_SHELL_CENTER[d]) ||
            !(SOLVER_SHELL_CENTER[d] - r > SOLVER_COMPD_MIN[d]) ||
            !(SOLVER_SHELL_CENTER[d] + r < SOLVER_COMPD_MAX[d]))
            throw std::runtime_error(
                "Shell must lie strictly inside the physical domain");
    Shell s;
    auto point = [&](double theta, double phi, bool pole) {
        const double xy = pole ? 0.0 : r * std::sin(theta);
        const double p[3] = {SOLVER_SHELL_CENTER[0] + xy * std::cos(phi),
                             SOLVER_SHELL_CENTER[1] + xy * std::sin(phi),
                             SOLVER_SHELL_CENTER[2] + r * std::cos(theta)};
        for (unsigned int d = 0; d < 3; ++d) {
            if (!std::isfinite(p[d]) || !(p[d] > SOLVER_COMPD_MIN[d]) ||
                !(p[d] < SOLVER_COMPD_MAX[d]))
                throw std::runtime_error("Shell point outside physical domain");
            s.xyz.push_back(p[d]);
        }
        s.theta.push_back(theta);
        s.phi.push_back(phi);
    };
    point(0, 0, true);
    for (unsigned int i = 1; i + 1 < nt; ++i)
        for (unsigned int j = 0; j < np; ++j)
            point(pi * i / (nt - 1), 2 * pi * j / np, false);
    point(pi, 0, true);
    auto ring = [np](unsigned int i, unsigned int j) {
        return 1 + (i - 1) * np + j % np;
    };
    auto cell = [&](std::initializer_list<unsigned int> ids) {
        s.connectivity.insert(s.connectivity.end(), ids);
        s.offsets.push_back(s.connectivity.size());
        s.types.push_back(ids.size() == 3 ? 5 : 9);  // VTK_TRIANGLE / VTK_QUAD
    };
    for (unsigned int j = 0; j < np; ++j) {
        cell({0, ring(1, j), ring(1, j + 1)});
        for (unsigned int i = 1; i + 2 < nt; ++i)
            cell({ring(i, j), ring(i + 1, j), ring(i + 1, j + 1),
                  ring(i, j + 1)});
        cell({ring(nt - 2, j), static_cast<unsigned int>(count - 1),
              ring(nt - 2, j + 1)});
    }
    return s;
}

// Return samples in angular-grid order on active rank zero. Dendro replicates
// search coordinates, but evaluates the basis only on the owning rank.
std::vector<double> sample(ot::Mesh* mesh, double* field, const Shell& s) {
    const unsigned int n = s.theta.size();
    const Point grid[2] = {
        Point(SOLVER_OCTREE_MIN[0], SOLVER_OCTREE_MIN[1], SOLVER_OCTREE_MIN[2]),
        Point(SOLVER_OCTREE_MAX[0], SOLVER_OCTREE_MAX[1],
              SOLVER_OCTREE_MAX[2])};
    const Point domain[2] = {
        Point(SOLVER_COMPD_MIN[0], SOLVER_COMPD_MIN[1], SOLVER_COMPD_MIN[2]),
        Point(SOLVER_COMPD_MAX[0], SOLVER_COMPD_MAX[1], SOLVER_COMPD_MAX[2])};
    mesh->readFromGhostBegin(field, 1);
    mesh->readFromGhostEnd(field, 1);
    std::vector<double> local(n, std::numeric_limits<double>::quiet_NaN());
    std::vector<unsigned int> ids;
    ot::shell_da::interpolateToCoords(mesh, field, s.xyz.data(), s.xyz.size(),
                                      grid, domain, local.data(), ids);
    std::vector<int> owned(n, 0), total(n, 0);
    std::vector<double> values(n, 0), result(n, 0);
    int bad = 0, anyBad = 0;
    for (auto id : ids) {
        if (id >= n) {
            bad = 1;
            continue;
        }
        ++owned[id];
        if (!std::isfinite(local[id])) bad = 1;
        values[id] = local[id];
    }
    MPI_Comm comm = mesh->getMPICommunicator();
    MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, comm);
    MPI_Reduce(owned.data(), total.data(), n, MPI_INT, MPI_SUM, 0, comm);
    MPI_Reduce(values.data(), result.data(), n, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (mesh->getMPIRank() == 0)
        for (unsigned int i = 0; i < n; ++i)
            if (total[i] != 1 || !std::isfinite(result[i])) {
                std::cerr << "[shell] invalid ownership/value at point " << i
                          << ": owners=" << total[i] << '\n';
                anyBad = 1;
                break;
            }
    MPI_Bcast(&anyBad, 1, MPI_INT, 0, comm);
    if (anyBad)
        throw std::runtime_error(
            "Shell interpolation failed coverage/finite-value checks");
    return result;
}

template <typename T>
void array(std::ostream& out, const char* type, const char* name,
           const std::vector<T>& data, unsigned int components = 1) {
    out << "<DataArray type=\"" << type << "\" Name=\"" << name
        << "\" NumberOfComponents=\"" << components << "\" format=\"ascii\">\n";
    for (auto value : data) out << value << ' ';
    out << "\n</DataArray>\n";
}

void write(const Shell& s, const std::vector<std::vector<double>>& fields,
           unsigned int step, double time) {
    std::ostringstream name;
    name << SOLVER_SHELL_FILE_PREFIX << '_' << std::setfill('0') << std::setw(6)
         << step << ".vtu";
    const std::filesystem::path path(name.str());
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(17)
        << "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" "
           "version=\"0.1\" "
           "byte_order=\"LittleEndian\">\n<UnstructuredGrid>\n<FieldData>\n";
    out << "<DataArray type=\"Float64\" Name=\"TimeValue\" "
           "NumberOfTuples=\"1\" format=\"ascii\">"
        << time << "</DataArray>\n"
        << "<DataArray type=\"UInt32\" Name=\"step\" NumberOfTuples=\"1\" "
           "format=\"ascii\">"
        << step << "</DataArray>\n</FieldData>\n"
        << "<Piece NumberOfPoints=\"" << s.theta.size() << "\" NumberOfCells=\""
        << s.types.size() << "\">\n<Points>\n";
    array(out, "Float64", "Points", s.xyz, 3);
    out << "</Points>\n<Cells>\n";
    array(out, "UInt32", "connectivity", s.connectivity);
    array(out, "UInt32", "offsets", s.offsets);
    array(out, "UInt8", "types", s.types);
    out << "</Cells>\n<PointData>\n";
    for (unsigned int v = 0; v < shellVariables; ++v)
        array(out, "Float64", SOLVER_VAR_NAMES[v], fields[v]);
    array(out, "Float64", "radius",
          std::vector<double>(s.theta.size(), SOLVER_SHELL_RADIUS));
    array(out, "Float64", "theta", s.theta);
    array(out, "Float64", "phi", s.phi);
    out << "</PointData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
    out.close();
    std::cout << "[shell] wrote " << path << " (" << s.theta.size()
              << " points)\n";
}
// One file per extraction step avoids ambiguous append/restart histories.
void writeModes(const std::vector<std::vector<double>>& fields,
                unsigned int step, double time) {
    ShellModeTransform transform(SOLVER_SHELL_N_THETA, SOLVER_SHELL_N_PHI,
                                 SOLVER_SHELL_LMAX);
    const auto& chi = fields[VAR::U_CHI];
    const auto& phi = fields[VAR::U_PHI];
    const auto chiModes = transform.project(
        std::vector<std::complex<double>>(chi.begin(), chi.end()));
    const auto phiModes = transform.project(
        std::vector<std::complex<double>>(phi.begin(), phi.end()));
    std::ostringstream name;
    name << SOLVER_SHELL_FILE_PREFIX << "_modes_" << std::setfill('0')
         << std::setw(6) << step << ".tsv";
    const std::filesystem::path path(name.str());
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(17)
        << "# orthonormal complex Y_lm; Condon-Shortley phase; integral f conj(Y) dOmega\n"
        << "# Clenshaw-Curtis in cos(theta), periodic trapezoidal in phi\n"
        << "# radius=" << SOLVER_SHELL_RADIUS << " center="
        << SOLVER_SHELL_CENTER[0] << ',' << SOLVER_SHELL_CENTER[1] << ','
        << SOLVER_SHELL_CENTER[2] << " n_theta=" << SOLVER_SHELL_N_THETA
        << " n_phi=" << SOLVER_SHELL_N_PHI << " lmax=" << SOLVER_SHELL_LMAX
        << "\n# time\tstep\tl\tm\tchi_real\tchi_imag\tphi_real\tphi_imag\n";
    const int limit = static_cast<int>(SOLVER_SHELL_LMAX);
    for (int l = 0; l <= limit; ++l)
        for (int m = -l; m <= l; ++m) {
            const auto k = ShellModeTransform::index(l, m);
            out << time << '\t' << step << '\t' << l << '\t' << m << '\t'
                << chiModes[k].real() << '\t' << chiModes[k].imag() << '\t'
                << phiModes[k].real() << '\t' << phiModes[k].imag() << '\n';
        }
    out.close();
    std::cout << "[shell] wrote " << path << " (" << chiModes.size()
              << " modes per field)\n";
}
}  // namespace

bool shellOutputDue(unsigned int step) {
    return (SOLVER_SHELL_OUTPUT_ENABLED && SOLVER_SHELL_OUTPUT_FREQ &&
            step % SOLVER_SHELL_OUTPUT_FREQ == 0) ||
           (SOLVER_SHELL_MODES_ENABLE && SOLVER_SHELL_MODE_FREQ &&
            step % SOLVER_SHELL_MODE_FREQ == 0);
}

void writeShell(ot::Mesh* mesh, double* const* fields, unsigned int step,
                double time) {
    if (!shellOutputDue(step) || !mesh->isActive()) return;
    try {
        const Shell s = makeShell();
        if (SOLVER_SHELL_VALIDATE) {
            // Independent scratch vectors: never overwrite evolved variables.
            double maxError = 0;
            for (unsigned int test = 0; test < 2; ++test) {
                std::vector<double> analytic;
                auto f = [test](double x, double y, double z) {
                    return test == 0 ? x + 2 * y - 0.5 * z
                                     : x * x + y * y + z * z;
                };
                mesh->createVector<double>(
                    analytic, std::function<double(double, double, double)>(
                                  [&](double x, double y, double z) {
                                      return f(GRIDX_TO_X(x), GRIDY_TO_Y(y),
                                               GRIDZ_TO_Z(z));
                                  }));
                const auto values = sample(mesh, analytic.data(), s);
                if (mesh->getMPIRank() == 0) {
                    for (unsigned int i = 0; i < values.size(); ++i) {
                        const double expected =
                            f(s.xyz[3 * i], s.xyz[3 * i + 1], s.xyz[3 * i + 2]);
                        maxError = std::max(
                            maxError, std::abs(values[i] - expected) /
                                          std::max(1.0, std::abs(expected)));
                    }
                }
            }
            MPI_Bcast(&maxError, 1, MPI_DOUBLE, 0, mesh->getMPICommunicator());
            if (mesh->getMPIRank() == 0)
                std::cout << "[shell] analytic interpolation max scaled error="
                          << maxError << '\n';
            if (maxError > 1e-8)
                throw std::runtime_error(
                    "Shell analytic interpolation validation failed");
        }
        std::vector<std::vector<double>> samples;
        for (unsigned int v = 0; v < shellVariables; ++v)
            samples.push_back(sample(mesh, fields[v], s));
        if (mesh->getMPIRank() == 0) {
            if (SOLVER_SHELL_OUTPUT_ENABLED &&
                step % SOLVER_SHELL_OUTPUT_FREQ == 0)
                write(s, samples, step, time);
            if (SOLVER_SHELL_MODES_ENABLE &&
                step % SOLVER_SHELL_MODE_FREQ == 0)
                writeModes(samples, step, time);
        }
        // Complete collective output before ranks resume other
        // output/evolution.
        MPI_Barrier(mesh->getMPICommunicator());
    } catch (const std::exception& e) {
        std::cerr << "[shell] rank " << mesh->getMPIRankGlobal() << ": "
                  << e.what() << std::endl;
        MPI_Abort(mesh->getMPIGlobalCommunicator(), 1);
    }
}
}  // namespace dsolve
