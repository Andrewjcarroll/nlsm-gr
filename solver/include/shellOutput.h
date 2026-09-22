#ifndef NLSM_SHELL_OUTPUT_H
#define NLSM_SHELL_OUTPUT_H
#include "mesh.h"
namespace dsolve {
// True when either surface VTU or harmonic output is due at this step.
bool shellOutputDue(unsigned int step);
// Collective on active mesh ranks. Refreshes ghosts but never alters owned
// state.
void writeShell(ot::Mesh* mesh, double* const* fields, unsigned int step,
                double time);
}  // namespace dsolve
#endif
