"""Geometry of the OneSimulation example "Horseshoe permanent magnet".

Builds one quarter of a horseshoe magnet and its surroundings with the CAD objects of OneSimulation.
The horseshoe has a square cross-section of 20 mm, a half-ring of 20 mm inner radius on top and two
legs 60 mm long below it; the lowest 20 mm of each leg are the magnetized pole pieces, the rest is
soft iron. The quarter on the side x > 0, z > 0 is cut by the two symmetry planes.

Files: horseshoe_permanent_magnet.step holds the air box (the modelled region) with the face groups
symmetry_x, symmetry_z and outer; horseshoe_permanent_magnet_yoke.step holds the iron of the
quarter (half-ring and upper leg), horseshoe_permanent_magnet_pole.step the magnetized pole piece.
"""

from onesim.cad.d3 import AnnularCylinder, Box

gap, width, leg, pole, air = 0.040, 0.020, 0.060, 0.020, 0.250
half_thickness = width / 2

region = Box(length=air, width=2 * air, height=air, origin=(0.0, -air, 0.0))
faces = region.boundaries
region = region.grouped(
    symmetry_x=faces.x_min,
    symmetry_z=faces.z_min,
    outer=faces.x_max | faces.y_min | faces.y_max | faces.z_max,
)
region.save("horseshoe_permanent_magnet.step")

ring = AnnularCylinder(
    inner_radius=gap / 2, outer_radius=gap / 2 + width, length=half_thickness
) & Box(length=gap / 2 + width, width=gap / 2 + width, height=half_thickness)
yoke = ring | Box(
    length=width,
    width=leg - pole,
    height=half_thickness,
    origin=(gap / 2, pole - leg, 0.0),
)
yoke.save("horseshoe_permanent_magnet_yoke.step")

Box(length=width, width=pole, height=half_thickness, origin=(gap / 2, -leg, 0.0)).save(
    "horseshoe_permanent_magnet_pole.step"
)
