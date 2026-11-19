"""Geometry of the OneSimulation example "Surface plasmon polariton on a silver-air interface".

Builds, as thin plates with the CAD objects of OneSimulation and in micrometres, a section 2 um long
of 0.4 um silver under 3.5 um of air (metal_air_surface_plasmon_propagation.step; x_min and x_max are
the waveguide ports, y_min and y_max the far walls), the deep silver
(metal_air_surface_plasmon_propagation_silver.step), its top 80 nm (_skin.step) and 80 nm of air
above the interface (_band.step), the last two meshed finely.
"""

from onesim.cad.d3 import Box

length, air, metal, band, depth = 2.0, 3.5, 0.4, 0.16, 0.1
Box(length=length, width=air + metal, height=depth, origin=(0, -metal, 0)).save(
    "metal_air_surface_plasmon_propagation.step"
)
Box(length=length, width=metal - band / 2, height=depth, origin=(0, -metal, 0)).save(
    "metal_air_surface_plasmon_propagation_silver.step"
)
Box(length=length, width=band / 2, height=depth, origin=(0, -band / 2, 0)).save(
    "metal_air_surface_plasmon_propagation_skin.step"
)
Box(length=length, width=band / 2, height=depth, origin=(0, 0, 0)).save(
    "metal_air_surface_plasmon_propagation_band.step"
)
