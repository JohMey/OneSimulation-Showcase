"""Geometry of the OneSimulation example "Stiffness of a mast diagonal with welded end plates".

Builds the diagonal of a lattice mast with the CAD objects of OneSimulation: a steel tube 60 mm in
outer diameter with a 3 mm wall and 1.2 m long along x, slotted at both ends to take a welded
gusset plate 8 mm thick and 60 mm high that reaches 50 mm into the tube and 100 mm beyond it. Each
plate carries two bolt holes of 18 mm diameter. Writes mast_diagonal_mounting_stiffness.step with
the face groups fixed_bolts (holes of the plate at x < 0), loaded_bolts (holes of the plate at
x > 1.2 m) and tube (outer tube surface).
"""

from onesim.cad.d3 import AnnularCylinder, Box

length, outer, wall, plate, height = 1.2, 0.030, 0.003, 0.008, 0.060
tube = AnnularCylinder(
    inner_radius=outer - wall, outer_radius=outer, length=length, axis=(1.0, 0.0, 0.0)
)
gussets = [
    Box(length=0.150, width=plate, height=height, origin=(x0, -plate / 2, -height / 2))
    for x0 in (-0.100, length - 0.050)
]
diagonal = tube | gussets[0] | gussets[1]
for x in (-0.080, -0.040, length + 0.040, length + 0.080):
    diagonal = diagonal.hole(0.018, at=(x, plate / 2, 0.0), direction=(0.0, -1.0, 0.0))
bolt = lambda face: face.kind == "cylinder" and abs(face.radius - 0.009) < 1e-9
diagonal = diagonal.grouped(
    fixed_bolts=diagonal.select(faces=lambda face: bolt(face) and face.center[0] < 0),
    loaded_bolts=diagonal.select(
        faces=lambda face: bolt(face) and face.center[0] > length
    ),
    tube=diagonal.select(
        faces=lambda face: face.kind == "cylinder" and abs(face.radius - outer) < 1e-9
    ),
)
diagonal.save("mast_diagonal_mounting_stiffness.step")
