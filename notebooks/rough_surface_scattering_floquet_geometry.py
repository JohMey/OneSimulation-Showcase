"""Geometry of the OneSimulation example "Light scattered by a rough metal coating".

Builds with the CAD objects of OneSimulation, drawn in micrometres as thin plates, one period
(4 um wide along x) of a glass substrate with a gold coating, 1 um of glass below the coating and
1.5 um of air above it: the whole cell (rough_surface_scattering_floquet.step; y_min in the glass
and y_max in the air are the ports, x_min and x_max the periodic sides), the glass
(rough_surface_scattering_floquet_glass.step), a flat coating 50 nm thick
(rough_surface_scattering_floquet_flat.step) and a coating whose top surface is rough
(rough_surface_scattering_floquet_rough.step): a periodic sum of eight cosines with random phases,
15 nm rms, sampled at 400 points.
"""

import numpy as np
from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Box, Prism

width, glass, air, film, depth = 4.0, 1.0, 1.5, 0.05, 0.05
Box(
    length=width, width=glass + film + air, height=depth, origin=(0.0, -glass, 0.0)
).save("rough_surface_scattering_floquet.step")
Box(length=width, width=glass, height=depth, origin=(0.0, -glass, 0.0)).save(
    "rough_surface_scattering_floquet_glass.step"
)
Box(length=width, width=film, height=depth).save(
    "rough_surface_scattering_floquet_flat.step"
)
rng = np.random.default_rng(7)
x = np.linspace(0.0, width, 401)
height = sum(
    np.cos(2 * np.pi * j * x / width + rng.uniform(0, 2 * np.pi)) / j
    for j in range(1, 9)
)
height = film + 0.015 * height / np.sqrt(np.mean(height**2))
outline = Path(
    (0.0, 0.0),
    (
        Segment((width, 0.0)),
        *(Segment((float(a), float(h))) for a, h in zip(x[::-1], height[::-1])),
        Segment((0.0, 0.0)),
    ),
)
Prism(profile=Profile(outline), height=depth).save(
    "rough_surface_scattering_floquet_rough.step"
)
