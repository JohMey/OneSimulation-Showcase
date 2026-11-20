"""Geometry of the OneSimulation example "Capacitive position sensor".

Builds the air inside a grounded shielding box (40 x 40 x 20 mm) above a printed circuit board with
the CAD objects of OneSimulation. Four square copper pads (8 x 8 x 0.5 mm, 2 mm apart) on the board
and a metal cube (10 mm) hovering 2 mm above them are cut out of the air; their faces form the
groups pad_1 ... pad_4 and target, the walls of the box the group ground. One file is written for
each lateral position of the cube: capacitive_position_sensor.step (centred) and
capacitive_position_sensor_x2.step, _x4.step, _x2y2.step, _x4y3.step (offsets in mm). The board
(FR-4, 1.6 mm) is written to capacitive_position_sensor_board.step.
"""

from onesim.cad.d3 import Box

mm = 1e-3
box, height, board, pad, copper, gap, cube, lift = (
    40 * mm,
    20 * mm,
    1.6 * mm,
    8 * mm,
    0.5 * mm,
    2 * mm,
    10 * mm,
    2 * mm,
)
centres = {"pad_1": (1, 1), "pad_2": (-1, 1), "pad_3": (-1, -1), "pad_4": (1, -1)}
for name, (dx, dy) in {
    "": (0, 0),
    "_x2": (2, 0),
    "_x4": (4, 0),
    "_x2y2": (2, 2),
    "_x4y3": (4, 3),
}.items():
    air = Box(
        length=box, width=box, height=height, origin=(-box / 2, -box / 2, 0.0)
    ).named("wall")
    for label, (sx, sy) in centres.items():
        air = air - Box(
            length=pad,
            width=pad,
            height=copper,
            origin=(
                sx * (gap + pad) / 2 - pad / 2,
                sy * (gap + pad) / 2 - pad / 2,
                board,
            ),
        ).named(label)
    air = air - Box(
        length=cube,
        width=cube,
        height=cube,
        origin=(dx * mm - cube / 2, dy * mm - cube / 2, board + copper + lift),
    ).named("target")
    groups = {
        label: air.select(
            faces=lambda face, label=label: face.id.startswith(label + ".")
        )
        for label in (*centres, "target", "wall")
    }
    air = air.grouped(ground=groups.pop("wall"), **groups)
    air.save(f"capacitive_position_sensor{name}.step")
Box(length=box, width=box, height=board, origin=(-box / 2, -box / 2, 0.0)).save(
    "capacitive_position_sensor_board.step"
)
