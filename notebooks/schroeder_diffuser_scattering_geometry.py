"""Geometry of the OneSimulation example "Schroeder diffuser".

Builds, as thin plates with the CAD objects of OneSimulation, the air in front of a rigid wall with
a quadratic-residue diffuser set into it, out to a half circle of radius 3 m. The diffuser has
N = 7 wells per period, 45 mm wide with 5 mm fins, and depths (n^2 mod 7) x 49 mm for a design
frequency of 500 Hz. schroeder_diffuser_scattering.step holds four periods,
schroeder_diffuser_scattering_single.step one; both name the faces outer, wall and wells.
"""

from onesim.cad.d2 import Path, Profile, Segment
from onesim.cad.d3 import Prism

cell, fin, unit, outer = 0.05, 0.005, 343.0 / 500.0 / 14, 3.0
depths = [(n * n % 7) * unit for n in range(7)]
for name, periods in (
    ("schroeder_diffuser_scattering", 4),
    ("schroeder_diffuser_scattering_single", 1),
):
    left = -periods * 7 * cell / 2
    segments = [Segment((left, 0.0), name="wall")]
    for index in range(7 * periods):
        depth, start = depths[index % 7], left + index * cell
        if depth > 0:
            segments += [
                Segment((start + fin / 2, 0.0), name="wells"),
                Segment((start + fin / 2, -depth), name="wells"),
                Segment((start + cell - fin / 2, -depth), name="wells"),
                Segment((start + cell - fin / 2, 0.0), name="wells"),
            ]
    segments += [
        Segment((-left, 0.0), name="wells"),
        Segment((outer, 0.0), name="wall"),
        Segment((-outer, 0.0), center=(0.0, 0.0), name="outer"),
    ]
    Prism(profile=Profile(Path((-outer, 0.0), tuple(segments))), height=0.01).save(
        f"{name}.step"
    )
