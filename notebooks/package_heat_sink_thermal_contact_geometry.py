"""Geometry of the OneSimulation example "Thermal contact between an electronic package and a finned heat sink".

Builds one sixteenth of the device with the CAD objects of OneSimulation: a sector of 22.5 degrees,
5 cm long, cut from a cylindrical silicon package of 1 cm radius and from the aluminium heat sink
around it, a hub of 3 mm wall with eight radial fins of 2 mm thickness that reach 2 cm from the axis
(the sector holds half of one fin). Both parts go to package_heat_sink_thermal_contact.step as an
assembly of package and sink. Groups: symmetry (the two cut planes), ends (z = 0 and z = 5 cm),
contact (the package surface and the bore of the sink) and cooled (the air-side faces of the sink).
"""

import math

from onesim.cad.d2 import Profile
from onesim.cad.d3 import AnnularCylinder, Assembly, Box, Cylinder, Prism

radius, hub, tip, fin, length, angle = (
    0.01,
    0.013,
    0.02,
    0.002,
    0.05,
    math.radians(22.5),
)
wedge = Prism(
    profile=Profile.polygon([(0.0, 0.0), (0.05, 0.0), (0.05, 0.05 * math.tan(angle))]),
    height=length,
)
package = Cylinder(radius=radius, length=length) & wedge
sink = (
    (
        AnnularCylinder(inner_radius=radius, outer_radius=hub, length=length)
        | (
            Box(length=tip, width=fin / 2, height=length)
            & Cylinder(radius=tip, length=length)
        )
    )
    - Cylinder(radius=radius, length=length)
) & wedge


def groups(part):
    ends = part.select(
        faces=lambda face: face.kind == "plane"
        and (abs(face.center[2]) < 1e-6 or abs(face.center[2] - length) < 1e-6)
    )
    symmetry = part.select(
        faces=lambda face: face.kind == "plane"
        and abs(face.center[2] - length / 2) < 1e-6
        and (
            abs(face.center[1]) < 1e-6
            or abs(math.atan2(face.center[1], face.center[0]) - angle) < 1e-4
        )
    )
    contact = part.select(
        faces=lambda face: face.kind == "cylinder"
        and math.hypot(face.center[0], face.center[1]) < 1.02 * radius
    )
    return dict(ends=ends, symmetry=symmetry, contact=contact)


named = groups(package)
package = package.grouped(
    ends=named["ends"],
    contact=named["contact"],
    symmetry=package.boundaries.all - named["ends"] - named["contact"],
)
named = groups(sink)
sink = sink.grouped(
    **named,
    cooled=sink.boundaries.all - named["ends"] - named["symmetry"] - named["contact"]
)
Assembly(parts=(package.named("package"), sink.named("sink"))).save(
    "package_heat_sink_thermal_contact.step"
)
