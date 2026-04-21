// Dendro: {{{ 
// Dendro: original ops:  15
// Dendro: printing temp variables

// Dendro: printing variables
//--
double r =  sqrt((pow(x, 2) + pow(y, 2) + pow(z, 2)));
phi_rhs[pp] = grad2_0_0_chi[pp] + grad2_1_1_chi[pp] + grad2_2_2_chi[pp] - sin(2*chi[pp])/pow(r, 2);
//--
chi_rhs[pp] = phi[pp];
// Dendro: reduced ops:  15
// Dendro: }}} 
