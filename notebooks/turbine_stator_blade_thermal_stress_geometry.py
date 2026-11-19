"""Geometry of the OneSimulation example "Thermal stress in a turbine stator blade".

Builds a cooled stator vane (a cambered airfoil of 60 mm chord and 50 mm span between an inner and
an outer platform, with three radial cooling channels) with the CAD objects of OneSimulation,
names the faces the simulation refers to and writes turbine_stator_blade_thermal_stress.step.
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Cylinder, Prism, UnionFromObjects

chord, span, camber, thickness, platform = 0.060, 0.050, 0.10, 0.22, 0.005
channels = ((0.012, 0.0045), (0.027, 0.0040), (0.042, 0.0030))

s = 0.95 * 0.5 * (1 - np.cos(np.linspace(0.0, np.pi, 24)))
half = (
    5
    * thickness
    * (0.2969 * np.sqrt(s) - 0.1260 * s - 0.3516 * s**2 + 0.2843 * s**3 - 0.1036 * s**4)
)
mean = camber * 4 * s * (1 - s)
slope = np.arctan(camber * 4 * (1 - 2 * s))
upper = np.column_stack((s - half * np.sin(slope), mean + half * np.cos(slope)))
lower = np.column_stack((s + half * np.sin(slope), mean - half * np.cos(slope)))
outline = chord * np.vstack((upper[::-1][:-1], lower))
camber_line = lambda x: chord * camber * 4 * (x / chord) * (1 - x / chord)

airfoil = Prism(
    profile=Profile.spline([tuple(point) for point in outline]),
    height=span + 2 * platform,
    origin=(0.0, 0.0, -platform),
)
inner = Box(
    length=0.080, width=0.050, height=platform, origin=(-0.010, -0.020, -platform)
)
outer = Box(length=0.080, width=0.050, height=platform, origin=(-0.010, -0.020, span))
vane = UnionFromObjects(objects=[airfoil, inner, outer])
for x, radius in channels:
    vane = vane - Cylinder(
        radius=radius,
        length=span + 4 * platform,
        origin=(x, camber_line(x), -2 * platform),
    )

vane = vane.grouped(
    airfoil=vane.select(
        faces=lambda face: face.kind not in ("plane", "cylinder")
        and 1e-6 < face.center[2] < span - 1e-6
    ),
    channels=vane.select(faces=lambda face: face.kind == "cylinder"),
    gas_platforms=vane.select(
        faces=lambda face: face.kind == "plane"
        and (abs(face.center[2]) < 1e-9 or abs(face.center[2] - span) < 1e-9)
    ),
    inner_back=vane.select(
        faces=lambda face: face.kind == "plane"
        and abs(face.center[2] + platform) < 1e-9
    ),
    outer_back=vane.select(
        faces=lambda face: face.kind == "plane"
        and abs(face.center[2] - span - platform) < 1e-9
    ),
)
vane.save("turbine_stator_blade_thermal_stress.step")
