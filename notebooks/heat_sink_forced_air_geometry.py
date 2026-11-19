"""Geometry of the OneSimulation example "Heat sink in a forced air flow".

Builds one half (y > 0) of a rectangular air duct, 80 mm long (x), 30 mm wide and 20 mm high, with
an aluminium heat sink on its floor, with the CAD objects of OneSimulation. The heat sink has a
20 x 18 x 3 mm base and three fins (2 mm thick, 10 mm high, 6 mm apart; the middle fin is cut in
half by the symmetry plane). The duct with the heat sink is written to heat_sink_forced_air.step;
its faces are named after the two bodies: air.x_min (inlet), air.x_max (outlet), air.y_min and
sink.y_min (symmetry plane), sink.z_min (heated bottom of the base), and the duct walls air.y_max,
air.z_min, air.z_max. The air alone, which carries the flow mesh, is written to
heat_sink_forced_air_air.step, the heat sink to heat_sink_forced_air_sink.step.
"""

from onesim.cad.d3 import Box

mm = 1e-3
duct = Box(length=80 * mm, width=15 * mm, height=20 * mm)
sink = (
    Box(length=20 * mm, width=9 * mm, height=3 * mm, origin=(20 * mm, 0.0, -3 * mm))
    | Box(length=20 * mm, width=1 * mm, height=13 * mm, origin=(20 * mm, 0.0, -3 * mm))
    | Box(
        length=20 * mm, width=2 * mm, height=13 * mm, origin=(20 * mm, 7 * mm, -3 * mm)
    )
)
air = duct - sink
(air.named("air") | sink.named("sink")).save("heat_sink_forced_air.step")
air.save("heat_sink_forced_air_air.step")
sink.save("heat_sink_forced_air_sink.step")
