"""Geometry of the OneSimulation example "Static stress in a bracket".

Builds one half of the sheet-metal bracket with the CAD objects of OneSimulation, names the faces
the simulation refers to and writes the part to bracket_static_stress.step.
"""

from onesim.cad.d3 import Box, UnionFromObjects

sheet, height, half_gap = 0.005, 0.040, 0.020

arm = Box(length=0.130, width=sheet, height=height, origin=(0.0, half_gap, 0.0))
lug = Box(length=0.030, width=sheet, height=0.045, origin=(0.100, half_gap, -0.045))
tab = Box(length=0.040, width=0.045, height=sheet, origin=(0.0, 0.0, height - sheet))

bracket = UnionFromObjects(objects=[arm, lug, tab])
bracket = bracket.hole(
    0.012, at=(0.115, half_gap + sheet, -0.015), direction=(0.0, -1.0, 0.0)
)
bracket = bracket.hole(0.009, at=(0.010, 0.035, height)).hole(
    0.009, at=(0.030, 0.035, height)
)

faces = bracket.boundaries
bracket = bracket.grouped(
    pin=faces["hole.outer_wall"],
    bolts=faces["hole_2.outer_wall"] | faces["hole_3.outer_wall"],
    symmetry=bracket.select(faces=lambda face: abs(face.center[1]) < 1e-9),
)
bracket.save("bracket_static_stress.step")
