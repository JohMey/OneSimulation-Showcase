"""Geometry of the OneSimulation example "Gold nanoparticle on a glass substrate".

Builds, with the CAD objects of OneSimulation and in micrometres, a cube of 0.6 um edge centred on
the origin (gold_nanoparticle_on_substrate.step; all faces lie behind the absorbing layers), the
glass half below z = 0 (gold_nanoparticle_on_substrate_glass.step) and a gold sphere of 50 nm radius
5 nm above the glass (gold_nanoparticle_on_substrate_gold.step).
"""

from onesim.cad.d3 import Box, Sphere

half, radius, gap = 0.3, 0.05, 0.005
Box(
    length=2 * half, width=2 * half, height=2 * half, origin=(-half, -half, -half)
).save("gold_nanoparticle_on_substrate.step")
Box(length=2 * half, width=2 * half, height=half, origin=(-half, -half, -half)).save(
    "gold_nanoparticle_on_substrate_glass.step"
)
Sphere(radius=radius, origin=(0.0, 0.0, radius + gap)).save(
    "gold_nanoparticle_on_substrate_gold.step"
)
