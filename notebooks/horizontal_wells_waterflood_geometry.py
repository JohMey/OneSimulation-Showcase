"""Geometry of the OneSimulation example "Waterflood of an oil reservoir with two horizontal wells".

Builds the reservoir block, 600 m long, 300 m wide and 30 m thick, with the CAD objects of
OneSimulation and writes horizontal_wells_waterflood.step. The faces keep the names x_min, x_max,
y_min, y_max, z_min (base) and z_max (cap rock); all of them are sealed in the simulation.
"""

from onesim.cad.d3 import Box

Box(length=600.0, width=300.0, height=30.0).save("horizontal_wells_waterflood.step")
