"""Geometry of the OneSimulation example "Probe-fed microstrip patch antenna".

Builds with the CAD objects of OneSimulation a rectangular patch antenna for 5.8 GHz on a 0.787 mm
substrate of relative permittivity 2.2 (patch 20.43 mm wide and 17.02 mm long from the transmission
line model with the Hammerstad fringing length), fed from below by an air-filled 50-ohm coaxial line
(inner radius 0.635 mm) whose inner conductor meets the patch 3.0 mm from its centre along its
length. Three STEP files: microstrip_patch_antenna.step, the air of a box 56 mm x 56 mm x 20 mm
above the ground plane with the coaxial line below it (faces radiation on the top and the sides,
ground, coax_port at the end of the line 5 mm below the ground, coax_wall, pin); the substrate
microstrip_patch_antenna_substrate.step; and microstrip_patch_antenna_patch.step, a thin layer of air
on the substrate in the shape of the patch, open over the end of the inner conductor, whose face on the
substrate is the patch.
"""

import numpy as np
from onesim.cad.d3 import Box, Cylinder

frequency, permittivity, height, light = 5.8e9, 2.2, 0.787e-3, 299792458.0
width = light / (2 * frequency) * np.sqrt(2 / (permittivity + 1))
effective = (permittivity + 1) / 2 + (permittivity - 1) / 2 / np.sqrt(
    1 + 12 * height / width
)
fringe = (
    0.412
    * height
    * (effective + 0.3)
    * (width / height + 0.264)
    / ((effective - 0.258) * (width / height + 0.8))
)
length = light / (2 * frequency * np.sqrt(effective)) - 2 * fringe
size, top, coax, feed, layer = 0.056, 0.02, 0.005, 0.003, 0.4e-3
inner, outer = 0.635e-3, 0.635e-3 * np.exp(50 / 60)
box = Box(length=size, width=size, height=top, origin=(-size / 2, -size / 2, 0.0))
line = Cylinder(radius=outer, length=coax, origin=(feed, 0.0, -coax))
pin = Cylinder(radius=inner, length=coax + height, origin=(feed, 0.0, -coax))
air = (box | line) - pin
near = lambda value, target: abs(value - target) < 1e-9
air = air.grouped(
    coax_port=air.select(
        faces=lambda face: face.kind == "plane" and near(face.center[2], -coax)
    ),
    ground=air.select(
        faces=lambda face: face.kind == "plane" and near(face.center[2], 0.0)
    ),
    coax_wall=air.select(
        faces=lambda face: face.kind == "cylinder" and abs(face.radius - outer) < 1e-6
    ),
    pin=air.select(
        faces=lambda face: (face.kind == "cylinder" and abs(face.radius - inner) < 1e-6)
        or (face.kind == "plane" and near(face.center[2], height))
    ),
)
faces = air.boundaries
taken = faces.coax_port | faces.ground | faces.coax_wall | faces.pin
air.grouped(radiation=air.select(faces=lambda face: face.id not in taken.members)).save(
    "microstrip_patch_antenna.step"
)
(
    Box(length=size, width=size, height=height, origin=(-size / 2, -size / 2, 0.0))
    - pin
).save("microstrip_patch_antenna_substrate.step")
(
    Box(
        length=length,
        width=width,
        height=layer,
        origin=(-length / 2, -width / 2, height),
    )
    - Cylinder(radius=inner, length=2 * layer, origin=(feed, 0.0, height - layer / 2))
).save("microstrip_patch_antenna_patch.step")
print(
    f"patch {width * 1e3:.2f} mm wide, {length * 1e3:.2f} mm long; effective permittivity {effective:.3f}, fringe {fringe * 1e3:.3f} mm"
)
