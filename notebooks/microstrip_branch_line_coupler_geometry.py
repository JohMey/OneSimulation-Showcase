"""Geometry of the OneSimulation example "Microstrip branch-line coupler".

Builds with the CAD objects of OneSimulation a 3 dB branch-line coupler for 3 GHz on a 0.813 mm
substrate of relative permittivity 3.55, in a box of air 5 mm high. The strip widths and the
quarter-wave lengths follow from the Hammerstad-Jensen formulas: 35.36-ohm series arms (horizontal)
and 50-ohm shunt arms (vertical) between the centre lines of a rectangle, and 50-ohm feed lines that
leave the four corners towards the four sides of the box. Three STEP files:
microstrip_branch_line_coupler.step (the box: faces port_1 ... port_4 on the sides the feeds reach,
ground below, lid above), microstrip_branch_line_coupler_substrate.step (the substrate) and
microstrip_branch_line_coupler_strips.step (a thin layer of air on the substrate in the shape of the
copper: its face on the substrate is the conductor pattern).
"""

import numpy as np
from onesim.cad.d3 import Box
from scipy.optimize import brentq

height, permittivity, frequency, lid, feed, layer = (
    0.813e-3,
    3.55,
    3e9,
    5e-3,
    6e-3,
    0.4e-3,
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


width = {
    z: brentq(lambda w: microstrip(w)[0] - z, 0.05 * height, 20 * height)
    for z in (50.0, 50 / np.sqrt(2))
}
quarter = {
    z: light / frequency / np.sqrt(microstrip(w)[1]) / 4 for z, w in width.items()
}
series, shunt = quarter[50 / np.sqrt(2)], quarter[50.0]
ws, wp = width[50 / np.sqrt(2)], width[50.0]
x0, y0 = -feed, -feed
box = Box(
    length=series + 2 * feed, width=shunt + 2 * feed, height=lid, origin=(x0, y0, 0.0)
)
substrate = Box(
    length=series + 2 * feed,
    width=shunt + 2 * feed,
    height=height,
    origin=(x0, y0, 0.0),
)
strip = lambda x, y, dx, dy: Box(
    length=dx, width=dy, height=layer, origin=(x, y, height)
)
pattern = (
    strip(-wp / 2, -ws / 2, series + wp, ws)
    | strip(-wp / 2, shunt - ws / 2, series + wp, ws)
    | strip(-wp / 2, -ws / 2, wp, shunt + ws)
    | strip(series - wp / 2, -ws / 2, wp, shunt + ws)
    | strip(x0, -wp / 2, feed, wp)
    | strip(series - wp / 2, y0, wp, feed)
    | strip(series, shunt - wp / 2, feed, wp)
    | strip(-wp / 2, shunt, wp, feed)
)
faces = box.boundaries
box.grouped(
    port_1=faces.x_min,
    port_2=faces.y_min,
    port_3=faces.x_max,
    port_4=faces.y_max,
    ground=faces.z_min,
    lid=faces.z_max,
).save("microstrip_branch_line_coupler.step")
substrate.save("microstrip_branch_line_coupler_substrate.step")
pattern.save("microstrip_branch_line_coupler_strips.step")
print(
    f"widths: 50 ohm {wp * 1e3:.3f} mm, 35.36 ohm {ws * 1e3:.3f} mm; quarter waves {series * 1e3:.3f} mm (series), "
    f"{shunt * 1e3:.3f} mm (shunt)"
)
