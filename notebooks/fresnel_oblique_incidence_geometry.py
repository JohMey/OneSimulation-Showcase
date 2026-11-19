"""Geometry of the OneSimulation example "Fresnel equations at oblique incidence".

Builds, as thin plates with the CAD objects of OneSimulation, one period of a plane interface between
air and glass at 1 GHz: the whole cell, 90 mm wide (0.3 wavelengths) and 600 mm high
(fresnel_oblique_incidence.step, faces x_min and x_max periodic, y_max the port in air, y_min the
port in glass), and its lower half, the glass (fresnel_oblique_incidence_glass.step).
"""

from onesim.cad.d3 import Box

period, height, thickness = 0.09, 0.6, 0.01
Box(length=period, width=height, height=thickness).save(
    "fresnel_oblique_incidence.step"
)
Box(length=period, width=height / 2, height=thickness).save(
    "fresnel_oblique_incidence_glass.step"
)
