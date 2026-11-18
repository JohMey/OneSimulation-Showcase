"""Geometry of the OneSimulation example "Stresses in a rotating pulley".

Builds the spoked pulley (outer radius 125 mm, bore radius 25 mm, eight rounded trapezoidal
cut-outs) with the CAD objects of OneSimulation and writes rotating_pulley_stresses.step. The file
holds the half of the 10 mm plate above its mid-plane and names the faces bore, rim, cutouts and
midplane. The pulley blank without cut-outs goes to rotating_pulley_stresses_blank.step.
"""

import numpy as np

from onesim.cad.d2 import Profile
from onesim.cad.d3 import Cylinder, Prism

mm = 1e-3
outer_radius, bore_radius, half_thickness = 125 * mm, 25 * mm, 5 * mm
inner_edge, outer_edge, inner_fillet, outer_fillet = 48 * mm, 100 * mm, 11 * mm, 16 * mm
inner_half_width, outer_half_width = 17.18 * mm, 30.96 * mm

corners = [(inner_half_width, inner_edge), (outer_half_width, outer_edge), (-outer_half_width, outer_edge), (-inner_half_width, inner_edge)]
cutout = Prism(profile=Profile.rounded_polygon(corners, (inner_fillet, outer_fillet, outer_fillet, inner_fillet)),
               height=3 * half_thickness, origin=(0, 0, -half_thickness))

shaft = Cylinder(radius=bore_radius, length=3 * half_thickness, origin=(0, 0, -half_thickness)).named("shaft")
blank = Cylinder(radius=outer_radius, length=half_thickness) - shaft
pulley = blank
for k in range(8):
    pulley = pulley - cutout.rotated((0, 0, 1), k * np.pi / 4).named("window")

faces = blank.boundaries
blank = blank.grouped(bore=faces["shaft.outer_wall"], rim=faces.outer_wall, midplane=faces.bottom)
faces = pulley.boundaries
pulley = pulley.grouped(bore=faces["shaft.outer_wall"], rim=faces.outer_wall, midplane=faces.bottom, cutouts=faces["window.side"])
pulley.save("rotating_pulley_stresses.step")
blank.save("rotating_pulley_stresses_blank.step")
