"""Geometry of the OneSimulation example "Split-and-recombine micromixer at five flow rates".

Builds a micromixer of square channels 1 mm wide with two split-and-recombine elements with the CAD
objects of OneSimulation, as a straight square channel from which the dividing walls are cut out.
In each element the stream is cut into an upper and a lower half; the upper half is squeezed into
the left half of the channel and the lower half into the right half, both are stretched back to the
full height, and the two halves are joined side by side again, so that the number of layers
doubles. Walls 0.2 mm thick separate the branches. The part is written to
split_recombine_micromixer_flow_rates.step with the face groups inlet (x = 0) and outlet (x = 13 mm).
"""

from onesim.cad.d3 import Box

mm = 1e-3
side, wall, elements = 1.0 * mm, 0.2 * mm, 2
length = (5 + 4 * elements) * mm
low, high = (side - wall) / 2, (side + wall) / 2
box = lambda x0, x1, y0, y1, z0, z1: Box(
    length=x1 - x0, width=y1 - y0, height=z1 - z0, origin=(x0, y0, z0)
)
mixer = box(0, length, 0, side, 0, side)
for start in [(1 + 4 * k) * mm for k in range(elements)]:
    mixer = (
        mixer
        - box(start, start + mm, -side, 2 * side, low, high)
        - box(start + mm, start + 2 * mm, low, 2 * side, low, 2 * side)
        - box(start + mm, start + 2 * mm, -side, high, -side, high)
        - box(start + 2 * mm, start + 3 * mm, low, high, -side, 2 * side)
    )
mixer = mixer.grouped(
    inlet=mixer.select(faces=lambda face: abs(face.center[0]) < 1e-9),
    outlet=mixer.select(faces=lambda face: abs(face.center[0] - length) < 1e-9),
)
mixer.save("split_recombine_micromixer_flow_rates.step")
