"""Geometry of the OneSimulation example "Gaussian pulse and perfectly matched layers".

Builds the quarter of the air volume, a plate of 0.36 m x 0.36 m and 10 mm thickness, with the CAD
objects of OneSimulation and writes gaussian_pulse_pml.step. The faces keep their names x_min,
x_max, y_min, y_max, z_min and z_max in the file; the example computes on the cross-section
z = 5 mm.
"""

from onesim.cad.d3 import Box

air = Box(length=0.36, width=0.36, height=0.010)
air.save("gaussian_pulse_pml.step")
