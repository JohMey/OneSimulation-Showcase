"""Geometry of the OneSimulation example "Swirl flow around a rotating disk".

Builds the water above the rotating disk with the CAD objects of OneSimulation: a cylinder of
10 mm radius and 9 mm height, and a second one of twice the size (20 mm and 18 mm) for the study of
the domain. The model is axisymmetric, so each file holds the half x >= 0 of its cylinder; the cut
with the plane y = 0 is the meridian section. Writes rotating_disk_swirl_flow.step and
rotating_disk_swirl_flow_large.step with the face groups axis (the cut through the axis), disk (the
bottom, which rotates), top (the open upper boundary) and rim (the open outer boundary).
"""

from onesim.cad.d3 import Box, Cylinder


def water(radius, height):
    body = Cylinder(radius=radius, length=height).named("water") & Box(
        length=radius, width=2 * radius, height=height, origin=(0.0, -radius, 0.0)
    ).named("half")
    named = lambda condition: body.select(faces=condition)
    return body.grouped(
        axis=named(lambda face: face.id.startswith("half")),
        disk=named(
            lambda face: not face.id.startswith("half") and abs(face.center[2]) < 1e-9
        ),
        top=named(
            lambda face: not face.id.startswith("half")
            and abs(face.center[2] - height) < 1e-9
        ),
        rim=named(lambda face: face.kind != "plane"),
    )


water(0.010, 0.009).save("rotating_disk_swirl_flow.step")
water(0.020, 0.018).save("rotating_disk_swirl_flow_large.step")
