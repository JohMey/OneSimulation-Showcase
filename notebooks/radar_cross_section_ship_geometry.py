"""Geometry of the OneSimulation example "Radar cross section of a ship".

Builds, as thin plates with the CAD objects of OneSimulation, the air around the side silhouette of
a 12 m patrol boat (hull, deck house and mast; radar_cross_section_ship.step, the silhouette cut out
as a hole) in a 22 m x 20 m box whose outer 1.5 m hold the absorbing layer, and the same box around
a conducting cylinder of 1 m radius for the validation (radar_cross_section_ship_cylinder.step).
"""

from onesim.cad.d2 import Profile
from onesim.cad.d3 import Box, Cylinder, Prism

width, height, thickness = 22.0, 20.0, 0.2
air = Box(
    length=width, width=height, height=thickness, origin=(-width / 2, -height / 2, 0.0)
)
silhouette = [
    (-6.0, -1.0),
    (4.5, -1.0),
    (6.2, 0.6),
    (6.0, 0.9),
    (2.0, 0.9),
    (2.0, 2.1),
    (0.6, 2.1),
    (0.6, 3.6),
    (0.35, 3.6),
    (0.35, 2.1),
    (-1.8, 2.1),
    (-2.2, 0.9),
    (-6.0, 0.9),
]
ship = Prism(profile=Profile.polygon(silhouette), height=2 * thickness).moved(
    (0.0, 0.0, -thickness / 2)
)
(air - ship).save("radar_cross_section_ship.step")
(
    air - Cylinder(radius=1.0, length=2 * thickness, origin=(0.0, 0.0, -thickness / 2))
).save("radar_cross_section_ship_cylinder.step")
