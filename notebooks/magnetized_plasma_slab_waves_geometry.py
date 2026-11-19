"""Geometry of the OneSimulation example "Circularly polarized waves in a lossy magnetized plasma slab".

Builds a 50 mm thick plasma layer as a 10 mm x 10 mm column of the unbounded slab with the CAD
objects of OneSimulation and writes magnetized_plasma_slab_waves.step. The faces keep their names
x_min, x_max, y_min, y_max (periodic sides) and z_min, z_max (the two surfaces of the layer).
"""

from onesim.cad.d3 import Box

layer = Box(length=0.010, width=0.010, height=0.050)
layer.save("magnetized_plasma_slab_waves.step")
