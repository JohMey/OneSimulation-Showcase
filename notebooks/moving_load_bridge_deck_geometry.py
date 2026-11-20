"""Geometry of the OneSimulation example "Moving load on a simply supported deck".

Builds a solid concrete deck slab, 10 m long, 1 m wide and 0.5 m deep, with the CAD objects of
OneSimulation and writes moving_load_bridge_deck.step. The faces keep their names x_min, x_max,
y_min, y_max, z_min and z_max in the file.
"""

from onesim.cad.d3 import Box

deck = Box(length=10.0, width=1.0, height=0.5, origin=(0.0, -0.5, -0.25))
deck.save("moving_load_bridge_deck.step")
