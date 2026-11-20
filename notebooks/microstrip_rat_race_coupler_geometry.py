"""Geometry of the OneSimulation example "Microstrip rat-race coupler".

Builds with the CAD objects of OneSimulation a rat-race (hybrid ring) coupler for 3 GHz on a 0.813 mm
substrate of relative permittivity 3.55: a 70.71-ohm ring one and a half guide wavelengths around
(widths and lengths from the Hammerstad-Jensen formulas, mean radius from the circumference) and four
50-ohm feed lines leaving it radially at 0, 60, 120 and 180 degrees. The board is a regular hexagon
whose sides face the feed directions, in a box of air 5 mm high. Three STEP files:
microstrip_rat_race_coupler.step (faces port_1 ... port_4 at 0, 60, 120 and 180 degrees, wall for
the two other sides, ground below, lid above), microstrip_rat_race_coupler_substrate.step and
microstrip_rat_race_coupler_strips.step (a thin layer of air on the substrate in the shape of the
copper: its face on the substrate is the conductor pattern).
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import AnnularCylinder, Box, Prism
from scipy.optimize import brentq

height, permittivity, frequency, lid, feed, layer = (
    0.813e-3,
    3.55,
    3e9,
    5e-3,
    5e-3,
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


ring_width = brentq(
    lambda w: microstrip(w)[0] - 50 * np.sqrt(2), 0.05 * height, 20 * height
)
feed_width = brentq(lambda w: microstrip(w)[0] - 50.0, 0.05 * height, 20 * height)
radius = 1.5 * light / frequency / np.sqrt(microstrip(ring_width)[1]) / (2 * np.pi)
apothem = radius + ring_width / 2 + feed
corner = apothem / np.cos(np.pi / 6)
outline = Profile.polygon(
    [
        (
            corner * np.cos(np.radians(30 + 60 * k)),
            corner * np.sin(np.radians(30 + 60 * k)),
        )
        for k in range(6)
    ],
    names=["port_2", "port_3", "port_4", "wall", "wall", "port_1"],
)
box = Prism(profile=outline, height=lid)
substrate = Prism(profile=outline, height=height)
ring = AnnularCylinder(
    inner_radius=radius - ring_width / 2,
    outer_radius=radius + ring_width / 2,
    length=layer,
    origin=(0.0, 0.0, height),
)
arm = Box(
    length=apothem - radius + 0.2 * ring_width,
    width=feed_width,
    height=layer,
    origin=(radius - 0.2 * ring_width, -feed_width / 2, height),
)
pattern = ring
for angle in (0, 60, 120, 180):
    pattern = pattern | arm.rotated((0.0, 0.0, 1.0), np.radians(angle))
faces = box.boundaries
box.grouped(ground=faces.bottom, lid=faces.top).save("microstrip_rat_race_coupler.step")
substrate.save("microstrip_rat_race_coupler_substrate.step")
pattern.save("microstrip_rat_race_coupler_strips.step")
print(
    f"ring: width {ring_width * 1e3:.3f} mm, mean radius {radius * 1e3:.3f} mm; feeds: width {feed_width * 1e3:.3f} mm"
)
