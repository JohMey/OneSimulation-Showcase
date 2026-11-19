"""Geometry of the OneSimulation example "Fabry-Perot etalon".

Builds, as thin plates with the CAD objects of OneSimulation, one period of a silicon etalon in air:
the cell, 0.3 mm wide and 6 mm high (fabry_perot_etalon_transmission.step; faces x_min and x_max
periodic, y_max and y_min the ports), and the 2 mm thick silicon plate in its middle
(fabry_perot_etalon_transmission_silicon.step).
"""

from onesim.cad.d3 import Box

period, height, plate, thickness = 0.3e-3, 6e-3, 2e-3, 0.1e-3
Box(length=period, width=height, height=thickness).save(
    "fabry_perot_etalon_transmission.step"
)
Box(
    origin=(0.0, (height - plate) / 2, 0.0),
    length=period,
    width=plate,
    height=thickness,
).save("fabry_perot_etalon_transmission_silicon.step")
