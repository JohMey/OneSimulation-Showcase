"""Geometry of the OneSimulation example "Magnetic field of a three-phase power line over three spans".

Builds two towers of a 400 kV line, 300 m apart, and the three phase conductors over the middle
span and the two neighbouring spans with the CAD objects of OneSimulation. Each tower is a mast
2 m square and 32 m high with a crossarm; the phases hang 9 m apart from the crossarm at 30 m and
sag in parabolic arcs to 20 m at mid-span. Each phase bundle is swept as a tube of its equivalent
radius 0.2 m along 60 straight pieces per span.

Files: power_line_magnetic_field_3d.step holds the two towers; power_line_magnetic_field_3d_phase_a.step,
_phase_b.step and _phase_c.step hold the three conductors.
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Sweep

span, attach, lowest, spacing, bundle = 300.0, 30.0, 20.0, 9.0, 0.2
towers = None
for x in (-span / 2, span / 2):
    tower = Box(
        length=2.0, width=2.0, height=attach + 2.0, origin=(x - 1.0, -1.0, 0.0)
    ) | Box(
        length=1.5,
        width=2 * spacing + 4.0,
        height=1.5,
        origin=(x - 0.75, -spacing - 2.0, attach),
    )
    towers = tower if towers is None else towers | tower
towers.save("power_line_magnetic_field_3d.step")

x = np.linspace(-1.5 * span, 1.5 * span, 181)
height = (
    lowest
    + 4 * (attach - lowest) * ((np.mod(x + span / 2, span) - span / 2) / span) ** 2
)
for name, y in (("phase_a", -spacing), ("phase_b", 0.0), ("phase_c", spacing)):
    path = tuple((float(a), y, float(b)) for a, b in zip(x, height))
    Sweep(profile=Profile.circle(bundle), path=path).save(
        f"power_line_magnetic_field_3d_{name}.step"
    )
