"""Geometry of the OneSimulation example "Free convection in a porous cavity".

Builds the square porous cavity (side 1, nondimensional) as a thin block with the CAD objects of
OneSimulation and writes porous_cavity_free_convection.step; the model lives on its section. The
faces keep the names x_min (hot wall), x_max (cold wall), y_min and y_max (insulated).
"""

from onesim.cad.d3 import Box

Box(length=1.0, width=1.0, height=0.1).save("porous_cavity_free_convection.step")
