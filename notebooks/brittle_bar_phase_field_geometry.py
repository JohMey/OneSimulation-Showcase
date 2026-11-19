"""Geometry of the OneSimulation example "Brittle damage of a bar in uniaxial tension".

Builds the upper half of the bar with the CAD objects of OneSimulation: 100 mm long, 1 mm high
(half of the 2 mm bar height) and 1 mm thick. The part is written to brittle_bar_phase_field.step;
the faces keep the names x_min (held end), x_max (pulled end), y_min (symmetry plane) and y_max
(free surface).
"""

from onesim.cad.d3 import Box

Box(length=0.1, width=0.001, height=0.001).save("brittle_bar_phase_field.step")
