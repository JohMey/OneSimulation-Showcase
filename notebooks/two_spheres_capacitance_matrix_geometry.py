"""Geometry of the OneSimulation example "Capacitance matrix of two spheres".

Builds the air around two conducting spheres (radii 0.1 m and 0.3 m, centres 0.5 m apart) with the
CAD objects of OneSimulation: a quarter of a sphere of 4 m radius, cut by the two symmetry planes
through the common axis, with the two conductors removed. The part is written to
two_spheres_capacitance_matrix.step with the face groups sphere_1, sphere_2, outer, section and
symmetry.
"""

from onesim.cad.d3 import Box, Sphere

a, b, c, outer_radius = 0.1, 0.3, 0.5, 4.0
quarter = Box(
    length=2 * outer_radius,
    width=outer_radius,
    height=outer_radius,
    origin=(-outer_radius, 0.0, 0.0),
)
air = Sphere(radius=outer_radius) & quarter.named("plane")
air = air - Sphere(radius=a, origin=(-c / 2, 0.0, 0.0)).named("sphere_1")
air = air - Sphere(radius=b, origin=(c / 2, 0.0, 0.0)).named("sphere_2")
faces = air.boundaries
air = air.grouped(
    sphere_1=faces.sphere_1,
    sphere_2=faces.sphere_2,
    outer=faces.surface,
    section=faces["plane.z_min"],
    symmetry=faces["plane.y_min"] | faces["plane.z_min"],
)
air.save("two_spheres_capacitance_matrix.step")
