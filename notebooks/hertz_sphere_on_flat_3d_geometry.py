"""Geometry of the OneSimulation example "Hertz contact of a sphere on a block".

Builds one quarter of the two bodies with the CAD objects of OneSimulation: the block, 5 mm wide
and 5 mm deep with its upper face at z = 0, goes to hertz_sphere_on_flat_3d.step; the quarter of
the lower half of the sphere (radius 10 mm, touching the block at the origin) goes to
hertz_sphere_on_flat_3d_sphere.step with the named faces plane_x, plane_y, equator and surface.
"""

from onesim.cad.d3 import Box, Sphere

radius, block_size = 10e-3, 5e-3
block = Box(
    length=block_size,
    width=block_size,
    height=block_size,
    origin=(0.0, 0.0, -block_size),
)
block.save("hertz_sphere_on_flat_3d.step")

quarter = Box(length=radius, width=radius, height=radius).named("plane")
sphere = Sphere(radius=radius, origin=(0.0, 0.0, radius)) & quarter
faces = sphere.boundaries
sphere = sphere.grouped(
    plane_x=faces["plane.x_min"],
    plane_y=faces["plane.y_min"],
    equator=faces["plane.z_max"],
    surface=faces.surface,
)
sphere.save("hertz_sphere_on_flat_3d_sphere.step")
