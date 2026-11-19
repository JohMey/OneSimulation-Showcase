"""Geometry of the OneSimulation example "Phase separation of a quenched binary mixture".

Builds a square film, 0.5 mm wide and 20 µm thick, with the CAD objects of OneSimulation and writes
binary_alloy_phase_separation.step. The faces keep their names x_min, x_max, y_min, y_max, z_min
and z_max in the file; the example computes on the mid-plane of the film.
"""

from onesim.cad.d3 import Box

film = Box(length=0.5e-3, width=0.5e-3, height=0.02e-3)
film.save("binary_alloy_phase_separation.step")
