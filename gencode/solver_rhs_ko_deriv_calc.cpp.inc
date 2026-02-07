
/**
 * NLSM KO derivative include (analog of EM2 ko_derivs include)
 *
 * Evolved vars:
 *   chi, phi
 */

/*[[[cog
import cog
import sys
import importlib.util
import dendrosym

spec = importlib.util.spec_from_file_location("dendroconf", CONFIG_FILE_PATH)
dendroconf = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dendroconf
spec.loader.exec_module(dendroconf)

cog.outl('//// dendro_derivs::ko GRADIENT CALCULATION (NLSM)')
# If you have a config-based generator for NLSM evolution vars,
# you could call it here similarly, but I’m writing the explicit include
# to match your current EM2 style.
]]]*/
//// dendro_derivs::ko GRADIENT CALCULATION (NLSM)

// ----------------------------------------------------
// KO "derivatives" for chi
// ----------------------------------------------------
dendro_derivs::ko_deriv_x(grad_0_chi, chi, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_chi, chi, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_chi, chi, hz, sz, bflag);

// ----------------------------------------------------
// KO "derivatives" for phi
// ----------------------------------------------------
dendro_derivs::ko_deriv_x(grad_0_phi, phi, hx, sz, bflag);
dendro_derivs::ko_deriv_y(grad_1_phi, phi, hy, sz, bflag);
dendro_derivs::ko_deriv_z(grad_2_phi, phi, hz, sz, bflag);

//[[[end]]]
