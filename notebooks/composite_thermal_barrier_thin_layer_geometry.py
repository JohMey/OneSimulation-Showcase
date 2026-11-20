"""Geometry of the OneSimulation example "Composite thermal barrier: resolved layers and a thin-layer condition".

Builds a steel cylinder of 2 cm radius and 4 cm height with the CAD objects of OneSimulation. A barrier
of two ceramic layers, 50 µm and 75 µm, sits at mid-height (z = 20 mm to 20.125 mm).
composite_thermal_barrier_thin_layer.step holds the whole cylinder, in which the layers are resolved by
the mesh; composite_thermal_barrier_thin_layer_parts.step holds the lower and the upper steel part,
which meet in the middle plane of the barrier. Groups: bottom (z = 0), top (z = 4 cm), mantle.
"""

from onesim.cad.d3 import Assembly, Cylinder

radius, height, barrier = 0.02, 0.04, 0.02 + 62.5e-6
whole = Cylinder(radius=radius, length=height)
faces = whole.boundaries
whole.grouped(bottom=faces.bottom, top=faces.top, mantle=faces.side).save(
    "composite_thermal_barrier_thin_layer.step"
)
lower = Cylinder(radius=radius, length=barrier)
upper = Cylinder(radius=radius, length=height - barrier, origin=(0.0, 0.0, barrier))
lower = lower.grouped(bottom=lower.boundaries.bottom, mantle=lower.boundaries.side)
upper = upper.grouped(top=upper.boundaries.top, mantle=upper.boundaries.side)
Assembly(parts=(lower.named("lower"), upper.named("upper"))).save(
    "composite_thermal_barrier_thin_layer_parts.step"
)
