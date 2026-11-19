"""Geometry of the OneSimulation example "Wicking in a paper strip".

Builds a strip of filter paper, 20 mm wide, 40 mm high and 0.2 mm thick, with the CAD objects of
OneSimulation and writes paper_strip_wicking.step. The faces keep their names x_min, x_max, y_min
(the edge dipped into water), y_max, z_min and z_max in the file; the example computes on the
mid-plane of the sheet.
"""

from onesim.cad.d3 import Box

strip = Box(length=0.020, width=0.040, height=0.0002)
strip.save("paper_strip_wicking.step")
