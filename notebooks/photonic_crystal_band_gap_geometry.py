"""Geometry of the OneSimulation example "Photonic crystal of dielectric rods".

Builds with the CAD objects of OneSimulation, as plates 1 mm thick, a square lattice (a = 10 mm) of
alumina rods of radius 2 mm:

* photonic_crystal_band_gap_cell.step and _cell_rod.step: the unit cell and its rod,
* photonic_crystal_band_gap_column.step and _column_rods.step: one period of a slab seven rows
  thick with air above and below (faces x_min, x_max periodic, y_min, y_max ports),
* photonic_crystal_band_gap.step and _rods.step: an 11 x 11 crystal with a waveguide bent by 90
  degrees (a row and a column of missing rods) in air with room for absorbing layers.
"""

from onesim.cad.d3 import Box, Cylinder, TablePattern

a, radius, thickness = 0.01, 0.002, 0.001
rod = lambda x, y: Cylinder(origin=(x, y, 0.0), radius=radius, length=thickness)

Box(origin=(-a / 2, -a / 2, 0.0), length=a, width=a, height=thickness).save(
    "photonic_crystal_band_gap_cell.step"
)
rod(0.0, 0.0).save("photonic_crystal_band_gap_cell_rod.step")

Box(origin=(-a / 2, -5.5 * a, 0.0), length=a, width=11 * a, height=thickness).save(
    "photonic_crystal_band_gap_column.step"
)
TablePattern(rod(0.0, -3 * a), [(0.0, j * a, 0.0) for j in range(1, 7)]).save(
    "photonic_crystal_band_gap_column_rods.step"
)

Box(origin=(-8 * a, -8 * a, 0.0), length=16 * a, width=16 * a, height=thickness).save(
    "photonic_crystal_band_gap.step"
)
channel = {(i, 0) for i in range(-5, 1)} | {(0, j) for j in range(0, 6)}
TablePattern(
    rod(-5 * a, -5 * a),
    [
        ((i + 5) * a, (j + 5) * a, 0.0)
        for i in range(-5, 6)
        for j in range(-5, 6)
        if (i, j) not in channel | {(-5, -5)}
    ],
).save("photonic_crystal_band_gap_rods.step")
