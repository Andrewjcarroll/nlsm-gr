
// ====================================================
// NLSM derivative scratch layout
// ====================================================
//
// Fields:
//   chi : sigma-model scalar
//   phi : conjugate momentum (chi_t)
//
// Only chi needs 2nd derivatives
//

double *grad_0_chi      = deriv_base +  0 * BLK_SZ;
double *grad_1_chi      = deriv_base +  1 * BLK_SZ;
double *grad_2_chi      = deriv_base +  2 * BLK_SZ;

double *grad2_0_0_chi   = deriv_base +  3 * BLK_SZ;
double *grad2_1_1_chi   = deriv_base +  4 * BLK_SZ;
double *grad2_2_2_chi   = deriv_base +  5 * BLK_SZ;

double *grad_0_phi   = deriv_base +  6 * BLK_SZ;
double *grad_1_phi   = deriv_base +  7 * BLK_SZ;
double *grad_2_phi   = deriv_base +  8 * BLK_SZ;
