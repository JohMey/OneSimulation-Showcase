"""Geometry of the OneSimulation example "Hartmann layers in a liquid-metal duct".

Builds a 50 mm long piece of an insulating rectangular duct with a 140 mm x 20 mm flow cross-section
with the CAD objects of OneSimulation and writes hartmann_layer_duct_section.step. The faces keep their
names: z_min and z_max are the wide walls normal to the magnetic field (Hartmann walls), y_min and y_max
the narrow side walls, x_min and x_max inlet and outlet.
"""

from onesim.cad.d3 import Box

duct = Box(length=0.050, width=0.140, height=0.020)
duct.save("hartmann_layer_duct_section.step")
