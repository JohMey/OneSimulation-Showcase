"""Geometry of the OneSimulation example "Free-convection cooling of vertical circuit boards".

Builds the cross-section region of one channel of a stack of vertical circuit boards with the CAD
objects of OneSimulation, as slabs 5 mm deep: two boards (1.6 mm thick, 100 mm high) 10 mm apart,
the left one carrying three chips (2 mm thick, 12 mm high) on the side facing the channel, and the
air in the channel and 20 mm below and 30 mm above the boards. Writes

- vertical_circuit_boards_free_convection_air.step: the air, the domain of the model, with the
  groups inlet (bottom), outlet (top), sides and solids (every face touching a board or a chip),
- vertical_circuit_boards_free_convection_board_left.step, ..._board_right.step and
  ..._chip_1.step, ..._chip_2.step, ..._chip_3.step (from the bottom up): the boards and chips,
- vertical_circuit_boards_free_convection.step: the whole channel (air, boards and chips), the
  domain of the heat transport, with the groups inlet, outlet and sides.
"""

from onesim.cad.d3 import Box

thickness, gap, board, below, above, depth = 1.6e-3, 10e-3, 0.1, 0.02, 0.03, 5e-3
chip_thickness, chip_height, chip_levels = 2e-3, 12e-3, (0.016, 0.046, 0.076)
slug = "vertical_circuit_boards_free_convection"

whole = Box(length=2 * thickness + gap, width=below + board + above, height=depth)
solids = {
    "board_left": Box(
        length=thickness, width=board, height=depth, origin=(0.0, below, 0.0)
    ),
    "board_right": Box(
        length=thickness,
        width=board,
        height=depth,
        origin=(thickness + gap, below, 0.0),
    ),
    **{
        f"chip_{index}": Box(
            length=chip_thickness,
            width=chip_height,
            height=depth,
            origin=(thickness, below + level, 0.0),
        )
        for index, level in enumerate(chip_levels, start=1)
    },
}
air = whole.named("box")
for name, solid in solids.items():
    air = air - solid.named(name)
named = lambda *prefixes: air.select(faces=lambda face: face.id.startswith(prefixes))
air = air.grouped(
    inlet=named("box.y_min"),
    outlet=named("box.y_max"),
    sides=named("box.x_min", "box.x_max"),
    solids=named(*solids),
)
air.save(f"{slug}_air.step")
whole.grouped(
    inlet=whole.boundaries.y_min,
    outlet=whole.boundaries.y_max,
    sides=whole.boundaries.x_min | whole.boundaries.x_max,
).save(f"{slug}.step")
for name, solid in solids.items():
    solid.save(f"{slug}_{name}.step")
