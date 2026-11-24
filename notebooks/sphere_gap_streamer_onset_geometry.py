"""Geometry of the OneSimulation example "Streamer onset in a sphere gap".

Builds the air around two equal conducting spheres of 62.5 mm radius (diameter 125 mm) whose surfaces are 60 mm apart
with the CAD objects of OneSimulation: a quarter of a sphere of 2 m radius, cut by the two symmetry planes through the
common axis, with the two conductors removed. The part is written to sphere_gap_streamer_onset.step with the face groups
sphere_1, sphere_2, outer, section and symmetry.
"""

from onesim.cad.d3 import Box, Sphere

radius, gap, outer_radius = 0.0625, 0.06, 2.0
centre = radius + gap / 2
quarter = Box(
    length=2 * outer_radius,
    width=outer_radius,
    height=outer_radius,
    origin=(-outer_radius, 0.0, 0.0),
)
air = Sphere(radius=outer_radius) & quarter.named("plane")
air = air - Sphere(radius=radius, origin=(-centre, 0.0, 0.0)).named("sphere_1")
air = air - Sphere(radius=radius, origin=(centre, 0.0, 0.0)).named("sphere_2")
faces = air.boundaries
air = air.grouped(
    sphere_1=faces.sphere_1,
    sphere_2=faces.sphere_2,
    outer=faces.surface,
    section=faces["plane.z_min"],
    symmetry=faces["plane.y_min"] | faces["plane.z_min"],
)
air.save("sphere_gap_streamer_onset.step")
