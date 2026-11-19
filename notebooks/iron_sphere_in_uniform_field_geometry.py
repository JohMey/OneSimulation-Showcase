"""Geometry of the OneSimulation example "Iron sphere in a uniform magnetic field".

Builds one octant of the air region, a sphere of radius 0.5 m around the iron sphere, with the CAD
objects of OneSimulation and writes iron_sphere_in_uniform_field.step. The file names the faces
plane_x, plane_y, plane_z and outer. The octant of the iron sphere (radius 0.125 m) goes to
iron_sphere_in_uniform_field_iron.step; the notebook embeds it in the mesh of the air region.
"""

from onesim.cad.d3 import Box, Sphere

iron_radius, air_radius = 0.125, 0.5
octant = Box(length=air_radius, width=air_radius, height=air_radius).named("plane")
air = Sphere(radius=air_radius) & octant
faces = air.boundaries
air = air.grouped(
    plane_x=faces["plane.x_min"],
    plane_y=faces["plane.y_min"],
    plane_z=faces["plane.z_min"],
    outer=faces.surface,
)
air.save("iron_sphere_in_uniform_field.step")

iron = Sphere(radius=iron_radius) & octant
iron.save("iron_sphere_in_uniform_field_iron.step")
