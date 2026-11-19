"""Geometry of the OneSimulation example "Metal scatterer near an optical waveguide".

Builds, as thin plates with the CAD objects of OneSimulation and in micrometres, a section 6 um long
and 5 um high of silica cladding (waveguide_metal_scatterer.step; x_min and x_max are the ports,
y_min and y_max the outer faces behind the absorbing layers), the 0.4 um silicon nitride core along
its middle (waveguide_metal_scatterer_core.step) and a gold rod of 0.1 um radius 0.15 um above the
core (waveguide_metal_scatterer_rod.step).
"""

from onesim.cad.d3 import Box, Cylinder

length, height, core, radius, gap, depth = 6.0, 5.0, 0.4, 0.1, 0.15, 0.1
Box(
    length=length, width=height, height=depth, origin=(-length / 2, -height / 2, 0)
).save("waveguide_metal_scatterer.step")
Box(length=length, width=core, height=depth, origin=(-length / 2, -core / 2, 0)).save(
    "waveguide_metal_scatterer_core.step"
)
Cylinder(radius=radius, length=depth, origin=(0.0, core / 2 + gap + radius, 0)).save(
    "waveguide_metal_scatterer_rod.step"
)
