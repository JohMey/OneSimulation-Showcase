"""Geometry of the OneSimulation example "Plasmonic wire grating".

Builds, as thin plates with the CAD objects of OneSimulation and in micrometres, one period (400 nm)
of a grating of silver wires of 40 nm radius held 20 nm above a glass substrate: the cell, 0.6 um of
air over 0.6 um of glass (plasmonic_wire_grating.step; x_min and x_max periodic, y_max the port in
air, y_min the port in the glass), the glass (plasmonic_wire_grating_glass.step) and the wire
(plasmonic_wire_grating_wire.step).
"""

from onesim.cad.d3 import Box, Cylinder

period, layer, radius, gap, depth = 0.4, 0.6, 0.04, 0.02, 0.1
Box(length=period, width=2 * layer, height=depth, origin=(0, -layer, 0)).save(
    "plasmonic_wire_grating.step"
)
Box(length=period, width=layer, height=depth, origin=(0, -layer, 0)).save(
    "plasmonic_wire_grating_glass.step"
)
Cylinder(radius=radius, length=depth, origin=(period / 2, gap + radius, 0)).save(
    "plasmonic_wire_grating_wire.step"
)
