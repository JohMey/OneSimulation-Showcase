"""Geometry of the OneSimulation example "Waveguide with several propagating modes".

Builds with the CAD objects of OneSimulation, as thin plates, the H-plane section of a metal
waveguide 50 mm wide and 200 mm long (multimode_waveguide_scattering.step; faces x_min and x_max
are the ports, y_min and y_max the metal walls) and two dielectric inserts 40 mm long in its middle:
one filling the whole width (multimode_waveguide_scattering_full.step) and one filling the lower
half only (multimode_waveguide_scattering_half.step).
"""

from onesim.cad.d3 import Box

length, width, insert, thickness = 0.2, 0.05, 0.04, 0.005
start = (length - insert) / 2
Box(length=length, width=width, height=thickness).save(
    "multimode_waveguide_scattering.step"
)
Box(origin=(start, 0.0, 0.0), length=insert, width=width, height=thickness).save(
    "multimode_waveguide_scattering_full.step"
)
Box(origin=(start, 0.0, 0.0), length=insert, width=width / 2, height=thickness).save(
    "multimode_waveguide_scattering_half.step"
)
