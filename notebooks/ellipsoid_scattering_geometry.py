"""Geometry of the OneSimulation example "Sound scattered by a rigid ellipsoid".

Builds, with the CAD objects of OneSimulation, the air around a rigid triaxial ellipsoid with
semi-axes 100 mm, 60 mm and 40 mm (a sphere scaled along the axes) out to an outer sphere. The outer
radius is eight wavenumbers at each of three frequencies (ka = 0.25, 0.5 and 1): 3.2 m, 1.6 m and
0.8 m, written to ellipsoid_scattering_ka025.step, ellipsoid_scattering_ka050.step and
ellipsoid_scattering.step. The ellipsoid is the part body, the outer sphere the part air.
"""

from onesim.cad.d3 import Sphere

body = Sphere(radius=0.1).scaled((1.0, 0.6, 0.4)).named("body")
for name, outer in (
    ("ellipsoid_scattering_ka025", 3.2),
    ("ellipsoid_scattering_ka050", 1.6),
    ("ellipsoid_scattering", 0.8),
):
    air = Sphere(radius=outer).named("air") - body
    air.save(f"{name}.step")
