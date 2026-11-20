"""Geometry of the OneSimulation example "AC resistance and sheath losses of a three-core power cable".

Builds with the CAD objects of OneSimulation, as thin plates, the cross-section of a three-core
220 kV cable with 500 mm2 copper conductors: three conductors of radius 12.6 mm whose centres form an
equilateral triangle of side 101 mm (cores touching), each inside a lead sheath from 45 to 48 mm
radius, and the surrounding region out to a radius of 400 mm.
Files: three_core_cable_ac_losses.step (the surrounding disc), three_core_cable_ac_losses_conductor_<k>.step
and three_core_cable_ac_losses_sheath_<k>.step for the phases k = a, b, c.
"""

import numpy as np
from onesim.cad.d3 import AnnularCylinder, Cylinder

spacing, conductor, sheath_in, sheath_out, outer, depth = (
    0.101,
    0.0126,
    0.045,
    0.048,
    0.4,
    0.002,
)
Cylinder(radius=outer, length=depth).save("three_core_cable_ac_losses.step")
for k, angle in zip("abc", (90.0, 210.0, 330.0)):
    centre = (
        spacing / np.sqrt(3) * np.cos(np.radians(angle)),
        spacing / np.sqrt(3) * np.sin(np.radians(angle)),
        0.0,
    )
    Cylinder(radius=conductor, length=depth, origin=centre).save(
        f"three_core_cable_ac_losses_conductor_{k}.step"
    )
    AnnularCylinder(
        inner_radius=sheath_in, outer_radius=sheath_out, length=depth, origin=centre
    ).save(f"three_core_cable_ac_losses_sheath_{k}.step")
