"""Geometry of the OneSimulation example "Optical ring resonator notch filter".

Builds, as thin plates with the CAD objects of OneSimulation and in micrometres, a section 7 um wide
and 6.9 um high of silica cladding (ring_resonator_notch_filter.step; x_min and x_max are the ports,
y_min and y_max the outer faces behind the absorbing layers), a straight silicon waveguide 0.24 um
wide along it (ring_resonator_notch_filter_bus.step) and a silicon ring of 2 um centre radius and
the same width 0.08 um above it (ring_resonator_notch_filter_ring.step).
"""

from onesim.cad.d3 import AnnularCylinder, Box

width, height, core, radius, gap, depth, bottom = 7.0, 6.9, 0.24, 2.0, 0.08, 0.1, -3.9
centre = -(radius + core + gap)
Box(length=width, width=height, height=depth, origin=(-width / 2, bottom, 0)).save(
    "ring_resonator_notch_filter.step"
)
Box(
    length=width, width=core, height=depth, origin=(-width / 2, centre - core / 2, 0)
).save("ring_resonator_notch_filter_bus.step")
AnnularCylinder(
    inner_radius=radius - core / 2, outer_radius=radius + core / 2, length=depth
).save("ring_resonator_notch_filter_ring.step")
