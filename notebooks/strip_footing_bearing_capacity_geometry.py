"""Geometry of the OneSimulation example "Bearing capacity of a strip footing on clay".

Builds one half of the clay stratum below a strip footing of width B = 1 m: a block 5 m wide and
3 m deep, 1 m thick, with the symmetry plane at x = 0 and the ground surface at y = 0. The block
is made of two parts, the soil under the footing (x < 0.5 m) named footing and the rest, so the
ground surface is split at the edge of the footing. The part is written to
strip_footing_bearing_capacity.step; its faces are footing.x_min (symmetry plane under the
footing), footing.y_max (footing), footing.y_min, soil.x_max (far side), soil.y_max (free ground
surface) and soil.y_min (base).
"""

from onesim.cad.d3 import Box

width, depth, half_footing = 5.0, 3.0, 0.5
stratum = Box(
    length=half_footing, width=depth, height=1.0, origin=(0.0, -depth, 0.0)
).named("footing") | Box(
    length=width - half_footing,
    width=depth,
    height=1.0,
    origin=(half_footing, -depth, 0.0),
).named(
    "soil"
)
stratum.save("strip_footing_bearing_capacity.step")
