# NLSM with CFDs

Optional spherical-shell output samples the evolved `U_CHI` and `U_PHI`
fields on a physical spherical surface and writes `nlsm_shell_<step>.vtu`.
It is disabled by default and independent of volume/slice output and its
compile-time switch. Both the context time stepper (argument `1`, default)
and the legacy RK path (argument `0`) call the same extraction routine.

Add these **top-level quoted keys** to an existing valid TOML configuration:

```toml
"dsolve::SOLVER_SHELL_OUTPUT_ENABLED" = true
"dsolve::SOLVER_SHELL_RADIUS" = 10.0
"dsolve::SOLVER_SHELL_CENTER" = [0.0, 0.0, 0.0]
"dsolve::SOLVER_SHELL_N_THETA" = 65
"dsolve::SOLVER_SHELL_N_PHI" = 128
"dsolve::SOLVER_SHELL_OUTPUT_FREQ" = 10
"dsolve::SOLVER_SHELL_FILE_PREFIX" = "vtu/nlsm_shell"
"dsolve::SOLVER_SHELL_VALIDATE" = false
```

These are the defaults except `OUTPUT_ENABLED`, whose default is `false`.
Radius is in physical coordinates; the default center is the domain origin,
independent of the two configurable initial-data centers. Frequency is a
positive number of steps; output includes step zero and occurs before evolution
at each matching step. Each angular count must be at least 3. The entire
sphere must be **strictly inside** the physical domain: touching the outer
boundary is rejected because Dendro's point search uses half-open octants.
The writer creates parent directories for its own prefix.

`SOLVER_SHELL_VALIDATE = true` additionally samples independent scratch fields
`x + 2*y - 0.5*z` and `x*x + y*y + z*z` on the current mesh at every shell output.
It aborts if maximum scaled error exceeds `1e-8`; use element order at least 2.
At the origin the second field should equal the extraction radius squared.
This optional check does not replace or modify the evolved solution. Coverage,
unique ownership, domain containment, and finite-value checks are always on.
Failures, including file-writing errors, report a shell error and abort MPI.

The implementation reuses Dendro's `interpolateToCoords`: physical coordinates
are mapped into octree coordinates, SFC partition splitters select owned points,
and the owning rank locates each containing element and evaluates its
order-p tensor-product Lagrange basis. `getElementNodalValues` supplies nodal
values, including hanging-node interpolation across AMR levels. Ghost values
are refreshed before sampling. All active ranks provide the coordinate list,
as required by that API, but only the owning rank interpolates each point.
Per-point values and ownership counts are reduced to active communicator rank
zero; inactive ranks skip extraction. No complete AMR state is gathered or
copied. Only the optional analytic check allocates a scratch mesh field.

The dependency's upper-z-face interpolation branch incorrectly snaps to the
lower face. `cmake/ShellInterpolation.cmake` generates a corrected copy of the
existing interpolation headers in the build tree, in a private namespace.
This preserves the dependency checkout and avoids conflicting template
instantiations with its original headers. It also preserves the upstream
`1e-6` physical-coordinate face-snapping tolerance; users working on extremely
small physical domains should account for that limitation. No interpolation
algorithm or VTK dependency is added.

The shell uses `theta_i = i*pi/(N_theta-1)` and `phi_j = 2*pi*j/N_phi`.
Point order is north pole, interior latitude rings in increasing theta/phi,
then south pole. There are `2 + (N_theta-2)*N_phi` points and
`(N_theta-1)*N_phi` cells. Each pole is a single vertex (`phi=0`); pole caps
are triangles, interior bands are quadrilaterals, and longitude wraps
periodically. Faces point outward. ASCII VTK UnstructuredGrid output contains
`U_CHI`, `U_PHI`, `radius`, `theta`, `phi`, plus time and step field data.
The solver's legacy eight-slot allocation is not interpreted as eight physical
fields: the NLSM equation enums and field-name table define only two.
No NLSM energy-density diagnostic is implemented in this checkout, so none is
invented here.

Scalar spherical-harmonic extraction uses these same samples; see below.
Pole samples must not be counted as independent pole vertices when reshaping
to a rectangular angular array.
The repository's BSSN sibling has GW harmonic extraction but is not linked by
this NLSM executable. Existing Dendro VTU helpers write octree volume/slice
meshes, so the small surface writer handles only the new connectivity.

## NLSM spherical-harmonic coefficients

Enable modes independently of shell VTU output with these top-level keys
(edit existing keys instead of repeating them):

```toml
"dsolve::SOLVER_SHELL_MODES_ENABLE" = true  # default false
"dsolve::SOLVER_SHELL_LMAX" = 8             # default 8; inclusive, starting at l=0
"dsolve::SOLVER_SHELL_MODE_FREQ" = 10       # default 10; includes step zero
"dsolve::SOLVER_SHELL_RADIUS" = 10.0
"dsolve::SOLVER_SHELL_CENTER" = [0.0, 0.0, 0.0]
"dsolve::SOLVER_SHELL_N_THETA" = 65
"dsolve::SOLVER_SHELL_N_PHI" = 128
"dsolve::SOLVER_SHELL_FILE_PREFIX" = "vtu/nlsm_shell"
"dsolve::SOLVER_SHELL_OUTPUT_ENABLED" = false # modes do not require VTU
```

Run `mpirun -np 2 build/solver/nlsmSolver your_config.toml 1` from a run
directory with the checkpoint/profile directories required by your configuration.
Both time steppers support extraction. When VTU and modes are due together,
the fields are sampled only once. AMR and Sobolev derivative calculations are
unaffected.

For both `U_CHI` and `U_PHI`, the output is
`a_lm = integral f(theta,phi) conj(Y_lm(theta,phi)) dOmega`.
For nonnegative m we use
`Y_lm = sqrt((2l+1)/(4*pi) * (l-m)!/(l+m)!) P_l^m(cos(theta)) exp(i*m*phi)`;
`P_l^m` includes the Condon–Shortley `(-1)^m` phase, and
`Y_l,-m = (-1)^m conj(Y_lm)`. These are orthonormal complex scalar harmonics.
The implementation uses the already-linked GSL normalized Legendre array
routine with the phase explicitly enabled. It adds no library dependency.
See the [GSL harmonic normalization documentation](https://www.gnu.org/software/gsl/doc/html/specfunc.html#associated-legendre-polynomials-and-spherical-harmonics).
The integral uses solid angle, without an `R^2` surface-area factor. Coefficients
are field amplitudes; their squared magnitudes are angular modal power, not
a newly defined NLSM physical-energy diagnostic.

The existing equally spaced theta grid becomes Chebyshev–Lobatto nodes under
`x=cos(theta)`. We integrate in x with Clenshaw–Curtis weights and in phi with
periodic trapezoidal weights `2*pi/N_PHI`. Each unique pole receives the full
longitude-integrated weight `2*pi*w_endpoint`. No duplicated seam or pole
samples and no uniform solid-angle sum are introduced.
We require `N_THETA >= 2*LMAX+1` and `N_PHI >= 2*LMAX+1` (both at least 3):
this integrates products within the retained harmonic band to roundoff.
Higher angular content in the actual solution can still alias; increase both
angular counts to check convergence independently of spatial interpolation.

Active mesh rank zero writes `<prefix>_modes_<step:06>.tsv`, one file per
extraction step, with `(LMAX+1)^2` rows sorted by l, then m from -l to l.
Comment headers record convention, quadrature, radius, center, angular counts,
and lmax. Whitespace-separated numeric columns are:

```text
time  step  l  m  chi_real  chi_imag  phi_real  phi_imag
```

Each file is self-contained. Re-extracting the same step overwrites its file;
use separate prefixes/directories for different runs or restarted branches.
For example, `numpy.loadtxt(path)` reads the numeric data; complex chi
coefficients are `data[:,4] + 1j*data[:,5]`. Compare runs at matching physical
times, radii, centers, and normalization.

Build and run the independent transform test with:

```bash
cmake --build build --target shellModesTest
build/solver/shellModesTest
```

It tests a constant (only `a_00=sqrt(4*pi)`), analytic complex `Y_21`
(only `a_21=1`), `Y_2,-1`, and `Re(Y_21)` on default, odd-interval, and
minimum-resolution grids. It prints target errors and maximum leakage and
rejects errors above `2e-12`. It also checks lmax zero and undersampling
rejection. These tests isolate angular projection from mesh interpolation.

For an end-to-end output check, enable both modes and shell VTU and run:

```bash
python3 tests/check_shell_modes.py vtu/nlsm_shell_modes_000000.tsv \
  --shell vtu/nlsm_shell_000000.vtu
```

The checker verifies row ordering, finite values, real-field conjugate symmetry,
and independently projects the VTU samples onto analytic `Y_00` and `Y_21`.
Use `--compare other_run/nlsm_shell_modes_000000.tsv` to check MPI agreement
for otherwise identical runs.

To build and run the reproducible refined-mesh smoke test from the repository
root (MPI, GSL, Eigen, BLAS/LAPACK and existing project dependencies required):

```bash
cmake -S . -B build
cmake --build build -j 4
repo="$PWD"
mkdir -p /tmp/nlsm-shell-test/serial/{cp,prof,vtu}
mkdir -p /tmp/nlsm-shell-test/mpi/{cp,prof,vtu}
cd /tmp/nlsm-shell-test/serial
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpirun -np 1 \
  "$repo/build/solver/nlsmSolver" "$repo/tests/shell.param.toml" 1
cd /tmp/nlsm-shell-test/mpi
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpirun -np 2 \
  "$repo/build/solver/nlsmSolver" "$repo/tests/shell.param.toml" 1
python3 "$repo/tests/check_shell.py" --compare \
  /tmp/nlsm-shell-test/serial/vtu/nlsm_shell_000000.vtu \
  /tmp/nlsm-shell-test/mpi/vtu/nlsm_shell_000000.vtu
```

The test configuration uses a level-2/3 mesh, radius 13, and a 17-by-24
angular grid. It is a self-contained valid configuration.
`nlsm_simplified.param.toml` also includes all shell parameters, disabled by default.
The Python checker needs only the standard library and checks finite fields,
spherical geometry, unique poles, cell indices and orientation, two cells per
edge, Euler characteristic 2, and optional serial/MPI field agreement.
After dependencies are downloaded, configure with
`-DFETCHCONTENT_UPDATES_DISCONNECTED=ON` to avoid automatic dependency updates.

Additional checks: set shell frequency to 2 and end time to `0.65` to get
steps 0 and 2 despite volume frequency 100; set enabled to false to suppress
shell files; set radius to `25.0` to check domain rejection. Test the minimal
surface with `N_THETA=3`, `N_PHI=3`. A high `SOLVER_DENDRO_GRAIN_SZ` exercises
inactive MPI ranks on this small mesh when run with `mpirun -np 4`
(Dendro keeps at least two ranks active).

Validation performed: full project build; serial and two-rank extraction on
levels 2/3; linear/quadratic errors below `8e-15`; serial/MPI agreement;
closed outward connectivity; independent output cadence; disabled output;
domain rejection; a minimal 5-point/6-triangle surface; and four MPI ranks
with two inactive ranks. The legacy RK path also wrote a valid matching shell,
but subsequently segfaulted in normal step handling. The same legacy failure
was reproduced with shell output disabled; use the default context stepper
for the successful end-to-end smoke test.

In ParaView, open `nlsm_shell_000000.vtu` (or select the numbered series),
click **Apply**, choose **Surface With Edges**, and color by `U_CHI` or `U_PHI`.
The Information panel should show 362 points and 384 cells for the test grid.
Inspect the poles and longitude seam for a closed connected surface. The
Spreadsheet View exposes physical coordinates, angular coordinates, and fields.
