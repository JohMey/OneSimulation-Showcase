"""Geometry of the OneSimulation example "Substrate integrated waveguide".

Builds with the CAD objects of OneSimulation a substrate integrated waveguide (SIW) on a 0.787 mm
substrate of relative permittivity 2.2: two rows of 12 plated vias (diameter 1.0 mm, pitch 1.8 mm,
rows 11.81 mm apart) under a top plate 23.4 mm long and 13.81 mm wide, fed at both ends by 50-ohm
microstrip lines
(2.42 mm wide, 5 mm long) through linear tapers 8 mm long that widen to 4.7 mm. The top plate
reaches 0.5 mm beyond the outer edges of the vias and covers them. The box of air above the board is
5 mm high. Four STEP files: substrate_integrated_waveguide.step (the air with the vias removed; faces
port_1 and port_2 at the ends of the feed lines, ground below, vias, open for the lid and the long
sides), substrate_integrated_waveguide_substrate.step (the substrate with the via holes),
substrate_integrated_waveguide_copper.step (a thin layer of air in the shape of the top copper, whose
face on the substrate is the copper) and substrate_integrated_waveguide_vias.step (the via barrels,
for the figures).
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Cylinder, Prism

height, permittivity, lid, layer = 0.787e-3, 2.2, 5e-3, 0.4e-3
diameter, pitch, vias, rows = 1.0e-3, 1.8e-3, 12, 11.81e-3
line, feed, taper, mouth, margin = 2.42e-3, 5e-3, 8e-3, 4.7e-3, 0.5e-3
plate = (vias - 1) * pitch + 2 * pitch
x0 = -(feed + taper + plate / 2)
side = rows / 2 + 6e-3
box = Box(length=-2 * x0, width=2 * side, height=lid, origin=(x0, -side, 0.0))
barrels = [Cylinder(radius=diameter / 2, length=height, origin=(-(vias - 1) * pitch / 2 + k * pitch, sign * rows / 2, 0.0))
           for k in range(vias) for sign in (1, -1)]
holes = barrels[0]
for barrel in barrels[1:]:
    holes = holes | barrel
air = box - holes
near = lambda value, target: abs(value - target) < 1e-9
faces = air.boundaries
air = air.grouped(
    port_1=faces.x_min, port_2=faces.x_max, ground=faces.z_min,
    vias=air.select(faces=lambda face: (face.kind == "cylinder" and abs(face.radius - diameter / 2) < 1e-6)
                    or (face.kind == "plane" and near(face.center[2], height))),
    open=faces.z_max | faces.y_min | faces.y_max,
)
air.save("substrate_integrated_waveguide.step")
(Box(length=-2 * x0, width=2 * side, height=height, origin=(x0, -side, 0.0)) - holes).save("substrate_integrated_waveguide_substrate.step")
strip = lambda x, y, dx, dy: Box(length=dx, width=dy, height=layer, origin=(x, y, height))
cone = lambda start, sign: Prism(profile=Profile.polygon([(start, -line / 2), (start + sign * taper, -mouth / 2), (start + sign * taper, mouth / 2), (start, line / 2)]),
                                 height=layer, origin=(0.0, 0.0, height))
span = rows + diameter + 2 * margin
copper = (strip(x0, -line / 2, feed + 1e-4, line) | cone(x0 + feed, 1) | strip(-plate / 2, -span / 2, plate, span)
          | cone(-x0 - feed, -1) | strip(-x0 - feed - 1e-4, -line / 2, feed + 1e-4, line))
copper.save("substrate_integrated_waveguide_copper.step")
holes.save("substrate_integrated_waveguide_vias.step")
effective_width = rows - diameter**2 / (0.95 * pitch)
print(f"effective width {effective_width * 1e3:.3f} mm, TE10 cutoff {299792458.0 / (2 * effective_width * np.sqrt(permittivity)) / 1e9:.3f} GHz")
