"""Geometry of the OneSimulation example "Reflection of a circularly polarized wave at glass".

Builds a column of air, 10 mm x 10 mm in cross-section and 200 mm long, with the CAD objects of
OneSimulation and writes circular_wave_glass_reflection.step. The faces keep their names x_min,
x_max, y_min, y_max (periodic sides), z_min (the plane through which the wave enters) and z_max
(the air-glass interface).
"""

from onesim.cad.d3 import Box

Box(length=0.010, width=0.010, height=0.200).save("circular_wave_glass_reflection.step")
