"""Geometry of the OneSimulation example "Waveguide iris bandpass filter".

Builds with the CAD objects of OneSimulation the air inside a WR-90 rectangular waveguide
(22.86 mm x 10.16 mm, axis along x, broad side along y, narrow side along z) with four symmetric
inductive irises: metal plates 1 mm thick across the full height that leave a centred aperture.
The dimensions realize a three-pole Chebyshev bandpass filter (0.1 dB ripple, 9.7 to 10.3 GHz):
the coupled-resonator design with impedance inverters and the thin-iris reactance (Matthaei, Young
and Jones; Marcuvitz) gives apertures of 10.76 and 6.99 mm and iris spacings of 16.50 and 17.95 mm;
refined on the H-plane section for the 1 mm iris thickness they become the values below. 25 mm of
guide lead to the first and from the last iris.

File waveguide_iris_bandpass_filter.step holds the air with the faces port_in (x = 0), port_out
(the far end) and the group metal (walls and irises).
"""

import numpy as np
from onesim.cad.d3 import Box

a, b, thickness, lead = 0.02286, 0.01016, 0.001, 0.025
apertures = (0.01160, 0.00818, 0.00818, 0.01160)
spacings = (0.01691, 0.01841, 0.01691)
planes = lead + np.concatenate(([0.0], np.cumsum(spacings)))
air = Box(length=planes[-1] + lead, width=a, height=b).named("guide")
for x, d in zip(planes, apertures):
    for y in (-0.001, (a + d) / 2):
        air = air - Box(
            length=thickness,
            width=(a - d) / 2 + 0.001,
            height=b + 0.002,
            origin=(x - thickness / 2, y, -0.001),
        ).named("iris")
faces = air.boundaries
air.grouped(
    port_in=faces["guide.x_min"],
    port_out=faces["guide.x_max"],
    metal=air.select(faces=lambda face: face.id.startswith("iris."))
    | faces["guide.y_min"]
    | faces["guide.y_max"]
    | faces["guide.z_min"]
    | faces["guide.z_max"],
).save("waveguide_iris_bandpass_filter.step")
