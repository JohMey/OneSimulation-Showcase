"""Geometry of the OneSimulation example "Porous structure on the microscopic scale: a sintered grain packing".

Builds a quarter of the periodic unit cell of a body-centred cubic packing of spherical grains with
the CAD objects of OneSimulation. The cell has the side 1 mm; grains sit on its corners and in its
centre. The quarter holds 1 mm x 0.5 mm x 0.5 mm (x along the flow, y and z between mirror planes)
with the grains at (0, 0, 0), (1, 0, 0) and (0.5, 0.5, 0.5) mm cut out. Above a grain radius of
0.433 mm the centre grain overlaps the corner grains, as in a sintered material, and the pore space
is no longer a sum of simple shapes. One file per grain radius in per cent of the cell side:
porous_microstructure_hexagonal_array_<percent>.step for 38, 40, 45 and 47 per cent and
porous_microstructure_hexagonal_array.step for 45 per cent. The cell faces keep the names x_min,
x_max, y_min, y_max, z_min and z_max; the grain surfaces form the group grains.
"""

from onesim.cad.d3 import Box, Sphere

side = 1e-3
for percent in (38, 40, 45, 47):
    radius = percent / 100 * side
    grains = (Sphere(radius=radius) | Sphere(radius=radius, origin=(side, 0.0, 0.0))
              | Sphere(radius=radius, origin=(side / 2, side / 2, side / 2)))
    quarter = Box(length=side, width=side / 2, height=side / 2) - grains
    quarter = quarter.grouped(grains=quarter.select(faces=lambda face: face.kind == "sphere"))
    quarter.save(f"porous_microstructure_hexagonal_array_{percent}.step")
    if percent == 45:
        quarter.save("porous_microstructure_hexagonal_array.step")
