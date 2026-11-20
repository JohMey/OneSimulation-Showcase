"""Geometry of the OneSimulation example "Band gaps of a one-dimensional photonic crystal".

Builds with the CAD objects of OneSimulation one period of a multilayer stack as a thin plate:
the unit cell, a = 1 mm along x (the stacking direction) and a / 4 along y, in
one_dimensional_photonic_crystal_bands.step (faces x_min, x_max, y_min, y_max), and the first
layer, half a period thick, in one_dimensional_photonic_crystal_bands_layer.step.
"""

from onesim.cad.d3 import Box

a, thickness = 1e-3, 1e-4
Box(length=a, width=a / 4, height=thickness).save(
    "one_dimensional_photonic_crystal_bands.step"
)
Box(length=a / 2, width=a / 4, height=thickness).save(
    "one_dimensional_photonic_crystal_bands_layer.step"
)
