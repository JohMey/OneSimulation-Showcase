"""Geometry of the OneSimulation example "Conical dielectric probe on skin".

Builds, as half solids of revolution about the z axis with the CAD objects of OneSimulation and in
metres, the air around a PTFE-filled circular waveguide of 2.5 mm radius that ends in a PTFE cone
touching the skin, with the skin and fat below (conical_dielectric_probe.step; faces port (the
waveguide cross-section), metal (the guide wall), outer (the faces behind the absorbing layers) and
axis), the PTFE (conical_dielectric_probe_ptfe.step), the skin layer 1.5 mm thick
(conical_dielectric_probe_skin.step), the fat below it (conical_dielectric_probe_fat.step) and a
lesion 5 mm across and 1 mm deep in the skin under the tip (conical_dielectric_probe_lesion.step).
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

mm = 1e-3
radius, wall, tip, cone, guide, width, depth, top, skin, lesion, deep = (
    2.5 * mm,
    0.3 * mm,
    1.0 * mm,
    6 * mm,
    20 * mm,
    8 * mm,
    6 * mm,
    10 * mm,
    1.5 * mm,
    2.5 * mm,
    1.0 * mm,
)
half = lambda path: Revolution(profile=Profile(path), angle=np.pi).rotated(
    (0, 0, 1), -np.pi / 2
)
half(
    Path(
        (0.0, -depth),
        (
            Segment((width, -depth), name="outer"),
            Segment((width, top), name="outer"),
            Segment((radius + wall, top), name="outer"),
            Segment((radius + wall, cone), name="metal"),
            Segment((radius, cone), name="metal"),
            Segment((radius, guide), name="metal"),
            Segment((0.0, guide), name="port"),
            Segment((0.0, -depth), name="axis"),
        ),
    )
).save("conical_dielectric_probe.step")
half(
    Path(
        (0.0, 0.0),
        (
            Segment((tip, 0.0)),
            Segment((radius, cone)),
            Segment((radius, guide)),
            Segment((0.0, guide)),
            Segment((0.0, 0.0)),
        ),
    )
).save("conical_dielectric_probe_ptfe.step")
half(
    Path(
        (0.0, -skin),
        (
            Segment((width, -skin)),
            Segment((width, 0.0)),
            Segment((lesion, 0.0)),
            Segment((lesion, -deep)),
            Segment((0.0, -deep)),
            Segment((0.0, -skin)),
        ),
    )
).save("conical_dielectric_probe_skin.step")
half(
    Path(
        (0.0, -depth),
        (
            Segment((width, -depth)),
            Segment((width, -skin)),
            Segment((0.0, -skin)),
            Segment((0.0, -depth)),
        ),
    )
).save("conical_dielectric_probe_fat.step")
half(
    Path(
        (0.0, -deep),
        (
            Segment((lesion, -deep)),
            Segment((lesion, 0.0)),
            Segment((0.0, 0.0)),
            Segment((0.0, -deep)),
        ),
    )
).save("conical_dielectric_probe_lesion.step")
