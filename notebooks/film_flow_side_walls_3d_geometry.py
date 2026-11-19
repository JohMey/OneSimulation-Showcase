"""Geometry of the OneSimulation example "Film flow down an inclined channel with side walls".

Builds a short slab of the liquid film, 2 mm deep and 0.125 mm long, for five channel widths with
the CAD objects of OneSimulation. The channel of 8 mm width goes to film_flow_side_walls_3d.step,
the others to film_flow_side_walls_3d_1mm.step, ..._2mm.step, ..._4mm.step and ..._16mm.step. The
bottom is the face z_min, the free surface z_max and the side walls y_min and y_max; the faces
keep their names in the files.
"""

from onesim.cad.d3 import Box

depth = 0.002
for width_mm in (1, 2, 4, 8, 16):
    film = Box(length=depth / 16, width=width_mm / 1000, height=depth)
    film.save(
        "film_flow_side_walls_3d.step"
        if width_mm == 8
        else f"film_flow_side_walls_3d_{width_mm}mm.step"
    )
