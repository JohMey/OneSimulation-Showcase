"""Geometry of the OneSimulation example "Magnetic prospecting of an iron ore body".

Builds the half of the ground and air around a buried magnetite body that lies on one side of the
magnetic meridian plane, with the CAD objects of OneSimulation. The ore body is a sphere of 50 m
radius whose centre lies 100 m below the surface; the modelled region is a half ball of 800 m
radius about that centre. The region is written to magnetic_prospecting_ore_body.step with the face
groups meridian (the plane y = 0) and outer; the half of the ore body goes to
magnetic_prospecting_ore_body_ore.step.
"""

from onesim.cad.d3 import Box, Sphere

ore_radius, region_radius = 50.0, 800.0
half = Box(
    length=2 * region_radius,
    width=region_radius,
    height=2 * region_radius,
    origin=(-region_radius, 0.0, -region_radius),
).named("cut")
region = Sphere(radius=region_radius) & half
faces = region.boundaries
region = region.grouped(meridian=faces["cut.y_min"], outer=faces.surface)
region.save("magnetic_prospecting_ore_body.step")
(Sphere(radius=ore_radius) & half).save("magnetic_prospecting_ore_body_ore.step")
