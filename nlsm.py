import dendro
from sympy import *

###############################################################
# initialize
###############################################################

chi = dendro.scalar("chi", "[pp]")
phi = dendro.scalar("phi", "[pp]")

# IMPORTANT: coordinates are pointwise scalars in the C++ kernel,
# so they must NOT be declared as dendro.scalar(...,"[pp]")
x, y, z = symbols("x y z", real=True)

d  = dendro.set_first_derivative("grad")
d2 = dendro.set_second_derivative("grad2")

###############################################################
# helpers
###############################################################

r2 = x*x + y*y + z*z
r2_safe = r2 + 1.0e-6

lap_chi = sum(d2(i, i, chi) for i in dendro.e_i)

###############################################################
# evolution equations
###############################################################

chi_rhs = phi
phi_rhs = lap_chi - sin(2*chi) / r2_safe

outs   = [chi_rhs, phi_rhs]
vnames = ["chi_rhs", "phi_rhs"]

dendro.generate(outs, vnames, "[pp]")
