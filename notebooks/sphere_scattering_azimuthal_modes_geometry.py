"""Geometry of the OneSimulation example "Plane-wave scattering by a body of revolution on its meridian".

Builds with the CAD objects of OneSimulation the meridian half plane of the computational domain as a
thin plate, radius 1.2 m from the axis and 2.4 m along it (sphere_scattering_azimuthal_modes.step;
faces x_min on the axis, x_max, z_min and z_max the outer walls), and the half of a sphere of radius
0.3 m about the origin on that side (sphere_scattering_azimuthal_modes_sphere.step).
"""

from onesim.cad.d3 import Box, Sphere

outer, radius, thickness = 1.2, 0.3, 0.01
Box(
    length=outer,
    width=thickness,
    height=2 * outer,
    origin=(0.0, -thickness / 2, -outer),
).save("sphere_scattering_azimuthal_modes.step")
(
    Sphere(radius=radius)
    & Box(
        length=radius,
        width=2 * radius,
        height=2 * radius,
        origin=(0.0, -radius, -radius),
    )
).save("sphere_scattering_azimuthal_modes_sphere.step")
