/// ====================================================
// NLSM RHS assembly (chi, phi)
//   chi_rhs = phi
//   phi_rhs = Laplacian(chi) - sin(2*chi)/r^2
// ====================================================

#if defined(USE_FIRST_DERIV_TWICE)
#  pragma message("Compiled WITH USE_FIRST_DERIV_TWICE (NLSM)")
#else
#  pragma message("Compiled WITHOUT USE_FIRST_DERIV_TWICE (NLSM)")
#endif

// --- scratch arrays you must provide/allocate (same size as chi/phi) ---
// double* grad_0_chi; double* grad_1_chi; double* grad_2_chi;
// double* grad2_0_0_chi; double* grad2_1_1_chi; double* grad2_2_2_chi;

// --- inputs ---
// chi[pp], phi[pp]
// hx, hy, hz : grid spacings
// sz         : local array size struct used by dendro (e.g. {nx,ny,nz})
// bflag      : boundary flags used by dendro stencils
// x0,y0,z0   : coordinates of gridpoint pp OR arrays X[pp],Y[pp],Z[pp] (see below)

// ====================================================
// Derivatives for chi (needed for Laplacian)
// ====================================================

// Always compute 1st derivs for chi (needed if USE_FIRST_DERIV_TWICE)
dendro_derivs::deriv_x(grad_0_chi, chi, hx, sz, bflag);
dendro_derivs::deriv_y(grad_1_chi, chi, hy, sz, bflag);
dendro_derivs::deriv_z(grad_2_chi, chi, hz, sz, bflag);

#ifdef USE_FIRST_DERIV_TWICE
  // Compute "2nd derivs" using first derivs
  dendro_derivs::deriv_x(grad2_0_0_chi, grad_0_chi, hx, sz, bflag);
  dendro_derivs::deriv_y(grad2_1_1_chi, grad_1_chi, hy, sz, bflag);
  dendro_derivs::deriv_z(grad2_2_2_chi, grad_2_chi, hz, sz, bflag);
#else
  // Normal 2nd derivs
  dendro_derivs::deriv_xx(grad2_0_0_chi, chi, hx, sz, bflag);
  dendro_derivs::deriv_yy(grad2_1_1_chi, chi, hy, sz, bflag);
  dendro_derivs::deriv_zz(grad2_2_2_chi, chi, hz, sz, bflag);
#endif

SOLVER_DERIVS->grad_x(grad_0_phi, phi_cpy, hx, sz, bflag);
SOLVER_DERIVS->grad_y(grad_1_phi, phi_cpy, hy, sz, bflag);
SOLVER_DERIVS->grad_z(grad_2_phi, phi_cpy, hz, sz, bflag);
