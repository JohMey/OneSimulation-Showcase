"""Geometry of the OneSimulation example "Thermal warpage of a printed circuit board".

Builds one quarter of a 40 x 40 mm circuit board with the CAD objects of OneSimulation: the FR-4
laminate (1.0 mm) with a full copper layer (0.1 mm) on top, written to pcb_thermal_warpage.step;
its faces keep the names x_min, x_max, y_min, y_max, z_min and z_max (x_min and y_min are the
symmetry planes through the centre of the board). The copper layer, which marks the region of the
second material, is written to pcb_thermal_warpage_copper.step.
"""

from onesim.cad.d3 import Box

mm = 1e-3
half, laminate, copper = 20 * mm, 1.0 * mm, 0.1 * mm
Box(length=half, width=half, height=laminate + copper).save("pcb_thermal_warpage.step")
Box(length=half, width=half, height=copper, origin=(0.0, 0.0, laminate)).save(
    "pcb_thermal_warpage_copper.step"
)
