"""Geometry of the OneSimulation example "Saturation of a magnetic ring core on a straight conductor".

Builds one eighth of the region around a finite ring core (inner radius 5 mm, outer radius 10 mm,
10 mm high) that sits on a straight conductor of 3 mm radius, with the CAD objects of OneSimulation:
the space between the conductor surface and a cylinder of 30 mm radius, from the mid-plane of the
core (z = 0) up to z = 30 mm. Its faces form the groups conductor, planes (the two cut planes
through the axis), midplane, top and outer; it is written to nonlinear_bh_ferrite_ring.step. The
eighth of the core (z from 0 to 5 mm) is written to nonlinear_bh_ferrite_ring_core.step.
"""

from onesim.cad.d3 import Box, Cylinder

mm = 1e-3
height, conductor, region, inner, outer, core = (
    30 * mm,
    3 * mm,
    30 * mm,
    5 * mm,
    10 * mm,
    5 * mm,
)
eighth = Box(length=region, width=region, height=height).named("cut")
space = (
    Cylinder(radius=region, length=height).named("air")
    - Cylinder(radius=conductor, length=height).named("wire")
) & eighth
faces = space.boundaries
space = space.grouped(
    conductor=space.select(faces=lambda face: face.id.startswith("wire")),
    outer=space.select(faces=lambda face: face.id == "air.outer_wall"),
    planes=faces["cut.x_min"] | faces["cut.y_min"],
    midplane=space.select(faces=lambda face: face.id in ("air.bottom", "cut.z_min")),
    top=space.select(faces=lambda face: face.id in ("air.top", "cut.z_max")),
)
space.save("nonlinear_bh_ferrite_ring.step")
(
    (Cylinder(radius=outer, length=core) - Cylinder(radius=inner, length=core)) & eighth
).save("nonlinear_bh_ferrite_ring_core.step")
