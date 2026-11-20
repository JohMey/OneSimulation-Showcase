"""Geometry of the OneSimulation example "Coaxial probe to waveguide transition".

Builds with the CAD objects of OneSimulation one half of a coaxial-to-waveguide transition, cut along
its plane of symmetry y = 0: a WR-90 rectangular waveguide (22.86 mm x 10.16 mm, half width
11.43 mm) along x, shorted behind the probe, and an air-filled 50-ohm coaxial line (inner radius
0.65 mm, outer radius 1.4956 mm) entering from below through the lower broad wall, whose inner
conductor reaches 6.1 mm up into the guide as a probe. The matched transition, with the short
5.5 mm behind the probe, is coaxial_probe_waveguide_transition.step; the same transition with the
short 4.5, 6.5, 8.0 and 9.9 mm (a quarter guide wavelength at 10 GHz) behind the probe is
coaxial_probe_waveguide_transition_short_<mm>.step (4p5, 6p5, 8p0, 9p9). Faces: coax_port (the
coaxial end, z = -8 mm), guide_port (the waveguide end, x = 25 mm), probe (the inner conductor),
symmetry (the cut plane y = 0) and metal (all other walls, with the short).
"""

import numpy as np
from onesim.cad.d3 import Box, Cylinder

a, b, run = 0.02286, 0.01016, 0.025
inner, outer, coax, probe = 0.00065, 0.00065 * np.exp(50 / 60), 0.008, 0.0061
near = lambda value, target: abs(value - target) < 1e-9


def transition(backshort):
    half = Box(
        length=backshort + run + 0.01,
        width=a,
        height=b + coax + probe,
        origin=(-backshort - 0.005, 0.0, -coax - 0.001),
    )
    guide = Box(
        length=backshort + run, width=a / 2, height=b, origin=(-backshort, 0.0, 0.0)
    )
    line = Cylinder(radius=outer, length=coax + 0.001, origin=(0.0, 0.0, -coax))
    pin = Cylinder(radius=inner, length=coax + probe, origin=(0.0, 0.0, -coax))
    air = (guide | (line & half)) - pin
    air = air.grouped(
        coax_port=air.select(
            faces=lambda face: face.kind == "plane" and near(face.center[2], -coax)
        ),
        guide_port=air.select(
            faces=lambda face: face.kind == "plane" and near(face.center[0], run)
        ),
        symmetry=air.select(
            faces=lambda face: face.kind == "plane" and near(face.center[1], 0.0)
        ),
        probe=air.select(
            faces=lambda face: np.hypot(face.center[0], face.center[1]) < inner * 1.01
            and face.center[2] > -coax + 1e-9
        ),
    )
    faces = air.boundaries
    named = faces.coax_port | faces.guide_port | faces.symmetry | faces.probe
    return air.grouped(
        metal=air.select(faces=lambda face: face.id not in named.members)
    )


transition(0.0055).save("coaxial_probe_waveguide_transition.step")
for millimetres in (4.5, 6.5, 8.0, 9.9):
    transition(millimetres * 1e-3).save(
        f"coaxial_probe_waveguide_transition_short_{millimetres:.1f}".replace(".", "p")
        + ".step"
    )
