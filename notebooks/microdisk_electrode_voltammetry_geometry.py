"""Geometry of the OneSimulation example "Voltammetry at a microdisk electrode".

Builds the electrolyte above a disk electrode of radius 10 µm set flush in an insulating plane, a
hemisphere of radius 2 mm, as half a solid of revolution about the z axis with the CAD objects of
OneSimulation and writes microdisk_electrode_voltammetry.step with the faces disk (the bottom
plane), insulator and outer. A small cylinder of the disk radius above the electrode is written to
microdisk_electrode_voltammetry_core.step; the mesh embeds it, so that a mesh vertex sits on the rim
of the disk.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Revolution

disk, outer = 10e-6, 2e-3
outline = Path(
    (0.0, 0.0),
    (
        Segment((disk, 0.0), name="disk"),
        Segment((outer, 0.0), name="insulator"),
        Segment((0.0, outer), center=(0.0, 0.0), name="outer"),
        Segment((0.0, 0.0), name="axis"),
    ),
)
Revolution(profile=Profile(outline), angle=np.pi).rotated((0, 0, 1), -np.pi / 2).save(
    "microdisk_electrode_voltammetry.step"
)
core = Path(
    (0.0, 0.0),
    (
        Segment((disk, 0.0)),
        Segment((disk, disk)),
        Segment((0.0, disk)),
        Segment((0.0, 0.0)),
    ),
)
Revolution(profile=Profile(core), angle=np.pi).rotated((0, 0, 1), -np.pi / 2).save(
    "microdisk_electrode_voltammetry_core.step"
)
