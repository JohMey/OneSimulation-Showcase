"""Geometry of the OneSimulation example "Stresses in a gantry crane with internal hinges".

Builds a solid model of the whole gantry crane with the CAD objects of OneSimulation and writes
portal_crane_frame_stress.step: two box-section main girders (12 m span, centre lines 8 m above the
rails and 3 m apart), two box-section fixed legs on one rail, two tubular pendulum legs on the other
rail and an end girder across each end. The rails run along y; the span is along x. The example
computes the crane as a frame along the centre lines of these members and uses the solid for the
picture.
"""

from onesim.cad.d3 import AnnularCylinder, Box, UnionFromObjects

span, height, half_gauge = 12.0, 8.0, 1.5
parts = []
for y in (-half_gauge, half_gauge):
    parts.append(
        Box(length=0.4, width=0.6, height=height - 0.4, origin=(-0.2, y - 0.3, 0.0))
    )
    parts.append(
        Box(
            length=span + 0.4,
            width=0.4,
            height=0.8,
            origin=(-0.2, y - 0.2, height - 0.4),
        )
    )
    parts.append(
        AnnularCylinder(
            inner_radius=0.14,
            outer_radius=0.15,
            length=height - 0.4,
            origin=(span, y, 0.0),
        )
    )
for x in (0.0, span):
    parts.append(
        Box(
            length=0.4,
            width=2 * half_gauge + 0.4,
            height=0.6,
            origin=(x - 0.2, -half_gauge - 0.2, height - 0.3),
        )
    )
crane = UnionFromObjects(objects=tuple(parts))
crane.save("portal_crane_frame_stress.step")
