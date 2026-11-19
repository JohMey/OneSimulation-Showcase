"""Geometry of the OneSimulation example "Stresses in a rotating pulley".

Builds the spoked pulley (outer radius 125 mm, bore radius 25 mm, 10 mm thick, eight rounded
trapezoidal cut-outs) with the CAD objects of OneSimulation and writes rotating_pulley_stresses.step
with the face groups bore, rim and cutouts. The pulley blank without cut-outs, with the groups bore
and rim, goes to rotating_pulley_stresses_blank.step. The example computes on the mid-plane sections.
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Cylinder, Prism

mm = 1e-3
outer_radius, bore_radius, thickness = 125 * mm, 25 * mm, 10 * mm
inner_edge, outer_edge, inner_fillet, outer_fillet = 48 * mm, 100 * mm, 11 * mm, 16 * mm
inner_half_width, outer_half_width = 17.18 * mm, 30.96 * mm

corners = [
    (inner_half_width, inner_edge),
    (outer_half_width, outer_edge),
    (-outer_half_width, outer_edge),
    (-inner_half_width, inner_edge),
]
cutout = Prism(
    profile=Profile.rounded_polygon(
        corners, (inner_fillet, outer_fillet, outer_fillet, inner_fillet)
    ),
    height=3 * thickness,
    origin=(0, 0, -thickness),
)

shaft = Cylinder(
    radius=bore_radius, length=3 * thickness, origin=(0, 0, -thickness)
).named("shaft")
blank = Cylinder(radius=outer_radius, length=thickness) - shaft
pulley = blank
for k in range(8):
    pulley = pulley - cutout.rotated((0, 0, 1), k * np.pi / 4).named("window")

faces = blank.boundaries
blank = blank.grouped(bore=faces["shaft.outer_wall"], rim=faces.outer_wall)
faces = pulley.boundaries
pulley = pulley.grouped(
    bore=faces["shaft.outer_wall"], rim=faces.outer_wall, cutouts=faces["window.side"]
)
pulley.save("rotating_pulley_stresses.step")
blank.save("rotating_pulley_stresses_blank.step")
