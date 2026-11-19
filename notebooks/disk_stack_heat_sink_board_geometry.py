"""Geometry of the OneSimulation example "Circuit board with a disk-stack heat sink".

Builds, with the CAD objects of OneSimulation, a circuit board (100 mm x 80 mm x 1.6 mm) carrying three
chip packages and a disk-stack heat sink (a post with six disks) on the largest chip. Two assemblies are
written: the board with its chips alone (disk_stack_heat_sink_board_bare.step) and with the heat sink
(disk_stack_heat_sink_board.step). Every chip and the heat sink also get a region file
disk_stack_heat_sink_board_<name>.step.
"""

from onesim.cad.d3 import Box, Cylinder, UnionFromObjects

board_t, chip_t = 0.0016, 0.002
board = Box(length=0.100, width=0.080, height=board_t)
chips = {
    "chip_main": Box(
        length=0.020, width=0.020, height=chip_t, origin=(0.040, 0.030, board_t)
    ),
    "chip_a": Box(
        length=0.012, width=0.012, height=chip_t, origin=(0.012, 0.012, board_t)
    ),
    "chip_b": Box(
        length=0.012, width=0.012, height=chip_t, origin=(0.074, 0.054, board_t)
    ),
}
top = board_t + chip_t
sink = UnionFromObjects(
    objects=[
        Cylinder(radius=0.005, length=0.020, origin=(0.050, 0.040, top)),
        *[
            Cylinder(
                radius=0.014,
                length=0.001,
                origin=(0.050, 0.040, top + 0.004 + 0.003 * i),
            )
            for i in range(6)
        ],
    ]
)
UnionFromObjects(objects=[board, *chips.values()]).save(
    "disk_stack_heat_sink_board_bare.step"
)
UnionFromObjects(objects=[board, *chips.values(), sink]).save(
    "disk_stack_heat_sink_board.step"
)
for name, part in {**chips, "sink": sink}.items():
    part.save(f"disk_stack_heat_sink_board_{name}.step")
