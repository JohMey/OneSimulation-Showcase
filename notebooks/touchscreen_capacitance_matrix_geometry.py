"""Geometry of the OneSimulation example "Capacitance matrix of a touchscreen with and without a finger".

Builds one half of a small projected-capacitive touchscreen with the CAD objects of OneSimulation.
Three transmit rows (along x, at y = -5, 0 and 5 mm) lie in a polymer substrate (0 to 1.2 mm above
the grounded display shield), three receive columns (along y, at x = -5, 0 and 5 mm) on top of it,
under a 0.5 mm cover glass. The electrodes are 1.5 mm wide and 0.2 mm thick, rows and columns 0.2 mm
apart, and run across the whole modelled area. A finger (a grounded cylinder of 4 mm radius) can rest on the glass above the middle
crossing. The touchscreen is mirror-symmetric about the plane x = 0; the half x > 0 is modelled.

Files: touchscreen_capacitance_matrix.step (without finger) and touchscreen_capacitance_matrix_finger.step
hold the air and dielectrics with the electrodes (and the finger) cut out, with the face groups
tx_low, tx_middle, tx_high (rows at y = -5, 0, 5 mm), rx_middle, rx_side (columns at x = 0 and
5 mm), finger, shield, symmetry and outer; touchscreen_capacitance_matrix_substrate.step and
touchscreen_capacitance_matrix_glass.step hold the two dielectric layers.
"""

from onesim.cad.d3 import Box, Cylinder

mm = 1e-3
half_x, half_y, top, substrate, glass = 10 * mm, 10 * mm, 10 * mm, 1.2 * mm, 1.7 * mm
width, thickness, strip = 1.5 * mm, 0.2 * mm, 10 * mm
rows = {
    name: Box(
        length=strip,
        width=width,
        height=thickness,
        origin=(0.0, y - width / 2, 0.6 * mm),
    )
    for name, y in (("tx_low", -5 * mm), ("tx_middle", 0.0), ("tx_high", 5 * mm))
}
columns = {
    "rx_middle": Box(
        length=width / 2,
        width=2 * strip,
        height=thickness,
        origin=(0.0, -strip, substrate - thickness),
    ),
    "rx_side": Box(
        length=width,
        width=2 * strip,
        height=thickness,
        origin=(5 * mm - width / 2, -strip, substrate - thickness),
    ),
}
electrodes = {**rows, **columns}
finger = Cylinder(radius=4 * mm, length=top - glass, origin=(0.0, 0.0, glass))

for name, cut in (
    ("touchscreen_capacitance_matrix.step", None),
    ("touchscreen_capacitance_matrix_finger.step", finger),
):
    region = Box(
        length=half_x, width=2 * half_y, height=top, origin=(0.0, -half_y, 0.0)
    ).named("box")
    for electrode, body in electrodes.items():
        region = region - body.named(electrode)
    if cut is not None:
        region = region - cut.named("finger")
    faces = region.boundaries
    groups = {
        electrode: region.select(
            faces=lambda face, electrode=electrode: face.id.startswith(electrode + ".")
        )
        for electrode in electrodes
    }
    if cut is not None:
        groups["finger"] = region.select(
            faces=lambda face: face.id.startswith("finger.")
        )
    region.grouped(
        **groups,
        shield=faces["box.z_min"],
        symmetry=faces["box.x_min"],
        outer=faces["box.x_max"]
        | faces["box.y_min"]
        | faces["box.y_max"]
        | faces["box.z_max"],
    ).save(name)

layers = {
    "substrate": Box(
        length=half_x, width=2 * half_y, height=substrate, origin=(0.0, -half_y, 0.0)
    ),
    "glass": Box(
        length=half_x,
        width=2 * half_y,
        height=glass - substrate,
        origin=(0.0, -half_y, substrate),
    ),
}
for name, layer in layers.items():
    for body in electrodes.values():
        layer = layer - body
    layer.save(f"touchscreen_capacitance_matrix_{name}.step")
