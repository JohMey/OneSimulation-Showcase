"""Geometry of the OneSimulation example "Trimmer capacitor with a rotating electrode".

Builds the air around an air-gap trimmer capacitor with the CAD objects of OneSimulation: two
half-disc electrodes of 5 mm radius and 0.1 mm thickness, 0.5 mm apart, in a cylinder of air. The
lower electrode (stator) is fixed; the upper one (rotor) is turned about the common axis by 0 to 180
degrees. The electrodes are cut out of the air, their surfaces are the faces stator and rotor, the
cylinder of air carries the face group outer.

One file per rotor angle: trimmer_capacitor_rotation.step (0 degrees, full overlap) and
trimmer_capacitor_rotation_030.step ... trimmer_capacitor_rotation_180.step.
"""

import numpy as np
from onesim.cad.d3 import Box, Cylinder

radius, thickness, gap, air_radius, air_height = 0.005, 0.0001, 0.0005, 0.012, 0.010
half = lambda bottom: Cylinder(
    radius=radius, length=thickness, origin=(0.0, 0.0, bottom)
) & Box(
    length=radius, width=2 * radius, height=thickness, origin=(0.0, -radius, bottom)
)
stator = half(-gap / 2 - thickness)
for angle in range(0, 181, 30):
    rotor = half(gap / 2).rotated((0.0, 0.0, 1.0), np.radians(angle))
    air = (
        Cylinder(
            radius=air_radius, length=air_height, origin=(0.0, 0.0, -air_height / 2)
        )
        - stator.named("stator")
        - rotor.named("rotor")
    )
    faces = air.boundaries
    air = air.grouped(
        stator=air.select(faces=lambda face: face.id.startswith("stator.")),
        rotor=air.select(faces=lambda face: face.id.startswith("rotor.")),
        outer=faces.bottom | faces.top | faces.outer_wall,
    )
    air.save(
        "trimmer_capacitor_rotation.step"
        if angle == 0
        else f"trimmer_capacitor_rotation_{angle:03d}.step"
    )
