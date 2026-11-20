"""Geometry of the OneSimulation example "Q factors and resonant frequencies of cavity resonators".

Builds the air inside three closed metal cavities with the CAD objects of OneSimulation: a
rectangular box of 30 x 15 x 40 mm (cavity_resonator_q_factors_rectangular.step), a cylinder of
radius 25 mm and length 40 mm (cavity_resonator_q_factors_cylindrical.step) and a sphere of radius
25 mm (cavity_resonator_q_factors_spherical.step). Every file names its whole surface `walls`.
"""

from onesim.cad.d3 import Box, Cylinder, Sphere

cavities = {
    "rectangular": Box(length=0.030, width=0.015, height=0.040),
    "cylindrical": Cylinder(radius=0.025, length=0.040),
    "spherical": Sphere(radius=0.025),
}
for name, body in cavities.items():
    body.grouped(walls=body.boundaries.all).save(
        f"cavity_resonator_q_factors_{name}.step"
    )
