"""Geometry of the OneSimulation example "Curvature of a prestressed micromirror".

Builds one quarter of a square micromirror, 1 mm x 1 mm, made of a 10 µm polysilicon plate
plated with a 0.5 µm gold film on top, with the CAD objects of OneSimulation. The whole stack is
written to prestressed_micromirror_curvature.step (faces x_min, x_max, y_min, y_max, z_min,
z_max; x_min and y_min are the symmetry planes through the centre of the mirror); the film, which
marks the region of the second material, is written to prestressed_micromirror_curvature_film.step.
"""

from onesim.cad.d3 import Box

half_side, plate, film = 500e-6, 10e-6, 0.5e-6
mirror = Box(length=half_side, width=half_side, height=plate + film)
coating = Box(length=half_side, width=half_side, height=film, origin=(0.0, 0.0, plate))
mirror.save("prestressed_micromirror_curvature.step")
coating.save("prestressed_micromirror_curvature_film.step")
