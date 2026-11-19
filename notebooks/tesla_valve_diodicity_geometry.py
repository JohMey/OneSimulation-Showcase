"""Geometry of the OneSimulation example "Tesla valve: diodicity and loop length".

Builds a four-stage Tesla-type valve with the CAD objects of OneSimulation, channels 1 mm wide: the
main channel meanders in right angles (rises of 4 mm, runs of 4 mm), and at each corner where it
turns into a run a loop leaves straight on, against the run, turns through a half circle of 1 mm
centreline radius and rejoins the rising or falling part of the main channel 2 mm before the
corner. The part is a slab 0.5 mm thick; the straight ends are the faces inlet_bar.x_min and
outlet_bar.x_max. One file per length of the straight part of the loop: tesla_valve_diodicity_<tenths
of a mm>.step for 1.0, 1.5 and 2.0 mm, and tesla_valve_diodicity.step for 1.5 mm. The example
computes on the cross-section z = 0.25 mm.
"""

from onesim.cad.d3 import AnnularCylinder, Box

mm = 1e-3
width, thickness, radius, run, rise = 1 * mm, 0.5 * mm, 1 * mm, 4 * mm, 4 * mm
bar = lambda x0, y0, x1, y1: Box(
    length=abs(x1 - x0) + width,
    width=abs(y1 - y0) + width,
    height=thickness,
    origin=(min(x0, x1) - width / 2, min(y0, y1) - width / 2, 0.0),
)
centreline = [
    (-3 * mm, 0),
    (0, 0),
    (0, rise),
    (run, rise),
    (run, 0),
    (2 * run, 0),
    (2 * run, rise),
    (3 * run, rise),
    (3 * run, 0),
]
for tenths in (10, 15, 20):
    loop = tenths / 10 * mm
    valve = Box(
        length=2 * mm, width=width, height=thickness, origin=(-5 * mm, -width / 2, 0.0)
    ).named("inlet_bar")
    for (x0, y0), (x1, y1) in zip(centreline[:-1], centreline[1:]):
        valve = valve | bar(x0, y0, x1, y1)
    for x, y, side in (
        (0, rise, -1),
        (run, 0, 1),
        (2 * run, rise, -1),
        (3 * run, 0, 1),
    ):
        ring = AnnularCylinder(
            inner_radius=radius - width / 2,
            outer_radius=radius + width / 2,
            length=thickness,
            origin=(x - loop, y + side * radius, 0.0),
        )
        valve = (
            valve
            | bar(x - loop, y, x, y)
            | bar(x - loop, y + 2 * side * radius, x, y + 2 * side * radius)
            | (
                ring
                & Box(
                    length=2 * radius,
                    width=4 * radius,
                    height=thickness,
                    origin=(x - loop - 2 * radius, y + side * radius - 2 * radius, 0.0),
                )
            )
        )
    valve = valve | Box(
        length=4 * mm, width=width, height=thickness, origin=(3 * run, -width / 2, 0.0)
    ).named("outlet_bar")
    valve.save(f"tesla_valve_diodicity_{tenths}.step")
    if tenths == 15:
        valve.save("tesla_valve_diodicity.step")
