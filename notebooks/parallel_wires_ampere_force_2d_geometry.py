"""Geometry of the OneSimulation example "Force between two parallel current-carrying wires".

Builds a 1 m long piece of two parallel straight wires (radius 0.1 m, axes 1 m apart) and the air
around them, a cylinder of 10 m radius about the midpoint between the wires, with the CAD objects
of OneSimulation. The air cylinder (the modelled domain, wires included) is written to
parallel_wires_ampere_force_2d.step; the wires go to parallel_wires_ampere_force_2d_left.step and
parallel_wires_ampere_force_2d_right.step. The example computes on the cross-section z = 0.5 m.
"""

from onesim.cad.d3 import Cylinder

spacing, radius, outer, length = 1.0, 0.1, 10.0, 1.0
Cylinder(radius=outer, length=length).save("parallel_wires_ampere_force_2d.step")
Cylinder(radius=radius, length=length, origin=(-spacing / 2, 0.0, 0.0)).save(
    "parallel_wires_ampere_force_2d_left.step"
)
Cylinder(radius=radius, length=length, origin=(spacing / 2, 0.0, 0.0)).save(
    "parallel_wires_ampere_force_2d_right.step"
)
