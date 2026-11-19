"""Geometry of the OneSimulation example "Laser heating of an absorbing slab (Beer-Lambert law)".

Builds the glass slab, 60 mm x 60 mm and 1 mm thick, with the CAD objects of OneSimulation and writes
beer_lambert_slab_heating.step. The irradiated face is named front, the face on the heat sink back and
the four edges form the group edges.
"""

from onesim.cad.d3 import Box

slab = Box(length=0.06, width=0.06, height=0.001, origin=(-0.03, -0.03, 0.0))
f = slab.boundaries
slab = slab.grouped(
    front=f.z_min, back=f.z_max, edges=f.x_min | f.x_max | f.y_min | f.y_max
)
slab.save("beer_lambert_slab_heating.step")
