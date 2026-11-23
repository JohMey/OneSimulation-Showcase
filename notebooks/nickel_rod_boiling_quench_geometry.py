"""Geometry of the OneSimulation example "Cooling of a nickel rod by nucleate boiling".

Builds the nickel rod, 10 mm in diameter and 50 mm long, with the CAD objects of OneSimulation and
writes nickel_rod_boiling_quench.step. The cylindrical face is named mantle, the two end faces form
the group ends.
"""

from onesim.cad.d3 import Cylinder

rod = Cylinder(radius=0.005, length=0.050)
f = rod.boundaries
rod = rod.grouped(mantle=f.outer_wall, ends=f.top | f.bottom)
rod.save("nickel_rod_boiling_quench.step")
