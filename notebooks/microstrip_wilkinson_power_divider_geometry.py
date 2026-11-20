"""Geometry of the OneSimulation example "Microstrip Wilkinson power divider".

Builds with the CAD objects of OneSimulation an equal-split Wilkinson divider for 3 GHz on a 0.813 mm
substrate of relative permittivity 3.55, in a box of air 5 mm high: a 50-ohm input line meets a ring
of two 70.71-ohm arms, each a quarter guide wavelength long on its centre line (widths and lengths
from the Hammerstad-Jensen formulas); the arms end 1 mm apart, where the isolation resistor bridges
them, and continue as 50-ohm output lines that turn up and down to the sides of the box. Four STEP
files: microstrip_wilkinson_power_divider.step (faces port_1 at the input, port_2 and port_3 at the
outputs, wall, ground below, lid above), microstrip_wilkinson_power_divider_substrate.step,
microstrip_wilkinson_power_divider_strips.step (a thin layer of air on the substrate in the shape of
the copper: its face on the substrate is the conductor pattern) and
microstrip_wilkinson_power_divider_resistor.step (a thin piece of the substrate under the gap: its top face
is the resistor between the arms).
"""

import numpy as np
from onesim.cad.d3 import AnnularCylinder, Box
from scipy.optimize import brentq

height, permittivity, frequency, lid, feed, layer, gap, stub = (
    0.813e-3,
    3.55,
    3e9,
    5e-3,
    7e-3,
    0.4e-3,
    1e-3,
    4.5e-3,
)
eta0, light = 376.730313668, 299792458.0


def microstrip(width):
    u = width / height
    a = (
        1
        + np.log((u**4 + (u / 52) ** 2) / (u**4 + 0.432)) / 49
        + np.log(1 + (u / 18.1) ** 3) / 18.7
    )
    b = 0.564 * ((permittivity - 0.9) / (permittivity + 3)) ** 0.053
    effective = (permittivity + 1) / 2 + (permittivity - 1) / 2 * (1 + 10 / u) ** (
        -a * b
    )
    f = 6 + (2 * np.pi - 6) * np.exp(-((30.666 / u) ** 0.7528))
    return (
        eta0 / (2 * np.pi) * np.log(f / u + np.sqrt(1 + 4 / u**2)) / np.sqrt(effective),
        effective,
    )


arm_width = brentq(
    lambda w: microstrip(w)[0] - 50 * np.sqrt(2), 0.05 * height, 20 * height
)
line_width = brentq(lambda w: microstrip(w)[0] - 50.0, 0.05 * height, 20 * height)
quarter = light / frequency / np.sqrt(microstrip(arm_width)[1]) / 4
radius = brentq(
    lambda r: r * (np.pi - np.arcsin(gap / (2 * r))) - quarter, gap, 10 * quarter
)
x0, x1, y1 = -radius - feed, radius + stub + 3e-3, radius + feed
box = Box(length=x1 - x0, width=2 * y1, height=lid, origin=(x0, -y1, 0.0))
resistor = Box(
    length=arm_width,
    width=gap,
    height=0.2e-3,
    origin=(radius - arm_width / 2, -gap / 2, height - 0.2e-3),
)
substrate = (
    Box(length=x1 - x0, width=2 * y1, height=height, origin=(x0, -y1, 0.0)) - resistor
)
strip = lambda x, y, dx, dy: Box(
    length=dx, width=dy, height=layer, origin=(x, y, height)
)
inner, outer = radius - arm_width / 2, radius + arm_width / 2
ring = AnnularCylinder(
    inner_radius=inner, outer_radius=outer, length=layer, origin=(0.0, 0.0, height)
) - strip(0.0, -gap / 2, 2 * outer, gap)
pattern = ring | strip(
    x0, -line_width / 2, x0 * -1 - radius + 0.2 * arm_width, line_width
)
for side in (1, -1):
    low = gap / 2 if side > 0 else -gap / 2 - line_width
    pattern = pattern | strip(inner, low, radius + stub - inner, line_width)
    pattern = pattern | strip(
        radius + stub - line_width,
        gap / 2 if side > 0 else -y1,
        line_width,
        y1 - gap / 2,
    )
faces = box.boundaries
box.grouped(
    port_1=faces.x_min,
    port_2=faces.y_max,
    port_3=faces.y_min,
    wall=faces.x_max,
    ground=faces.z_min,
    lid=faces.z_max,
).save("microstrip_wilkinson_power_divider.step")
substrate.save("microstrip_wilkinson_power_divider_substrate.step")
pattern.save("microstrip_wilkinson_power_divider_strips.step")
resistor.save("microstrip_wilkinson_power_divider_resistor.step")
print(
    f"arms: width {arm_width * 1e3:.3f} mm, quarter wave {quarter * 1e3:.3f} mm, ring radius {radius * 1e3:.3f} mm; "
    f"lines: width {line_width * 1e3:.3f} mm"
)
