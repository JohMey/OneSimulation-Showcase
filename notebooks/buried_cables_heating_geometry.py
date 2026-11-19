"""Geometry of the OneSimulation example "Heating of buried power cables".

Builds a 0.5 m long piece of the ground and of the three cables with the CAD objects of
OneSimulation. The soil block, 80 m wide and 40 m deep with the ground surface at y = 0, is written
to buried_cables_heating.step with the faces surface (y = 0), bottom (y = -40 m) and sides. The
layers of the three cables, laid 1 m deep and 0.3 m apart, go to one file each:
buried_cables_heating_conductor.step, ..._insulation.step, ..._sheath.step and ..._jacket.step.
The example computes on the cross-section z = 0.25 m.
"""

from onesim.cad.d3 import AnnularCylinder, Box, Cylinder

half_width, piece = 40.0, 0.5
radii = (0.019, 0.035, 0.0375, 0.0415)
centres = ((-0.3, -1.0), (0.0, -1.0), (0.3, -1.0))
soil = Box(
    length=2 * half_width,
    width=half_width,
    height=piece,
    origin=(-half_width, -half_width, 0.0),
)
faces = soil.boundaries
soil.grouped(
    surface=faces.y_max, bottom=faces.y_min, sides=faces.x_min | faces.x_max
).save("buried_cables_heating.step")
for name, inner, outer in zip(
    ("conductor", "insulation", "sheath", "jacket"), (0.0,) + radii[:-1], radii
):
    layer = [
        (
            Cylinder(radius=outer, length=piece, origin=(x, y, 0.0))
            if inner == 0.0
            else AnnularCylinder(
                inner_radius=inner, outer_radius=outer, length=piece, origin=(x, y, 0.0)
            )
        )
        for x, y in centres
    ]
    (layer[0] | layer[1] | layer[2]).save(f"buried_cables_heating_{name}.step")
