"""Geometry of the OneSimulation example "Viscoelastic arterial wall under biaxial stretch".

Builds a square sample of the adventitia of an artery, 10 x 10 mm and 0.43 mm thick (the wall
thickness of the human iliac artery), with the CAD objects of OneSimulation and writes
arterial_wall_viscoelastic_relaxation.step. x is the circumferential, y the axial and z the radial
direction of the vessel; the faces keep the names x_min ... z_max.
"""

from onesim.cad.d3 import Box

Box(length=10.0e-3, width=10.0e-3, height=0.43e-3).save(
    "arterial_wall_viscoelastic_relaxation.step"
)
