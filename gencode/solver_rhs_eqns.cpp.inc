// Dendro: {{{ 
// Dendro: original ops:  15
// Dendro: printing temp variables

// Dendro: printing variables
//--
chi_rhs[pp] = phi[pp];
//--
phi_rhs[pp] = grad2_0_0_chi[pp] + grad2_1_1_chi[pp] + grad2_2_2_chi[pp] - sin(2*chi[pp])/(pow(x, 2) + pow(y, 2) + pow(z, 2) + 9.9999999999999995e-3);
// Dendro: reduced ops:  15
// Dendro: }}} 
