"""Geometry of the OneSimulation example "Aquifer-system compaction above a bedrock step".

Builds a 100 m thick slice of the basin with the CAD objects of OneSimulation: the deep part of
the lower aquifer left of the bedrock step, the lower aquifer above the step, the confining unit
and the upper aquifer. Writes terzaghi_aquifer_compaction.step with the whole slice and
terzaghi_aquifer_compaction_confining.step with the confining unit alone. The faces keep the name
of their layer, for example deep.x_min or upper.y_max; the example computes on the cross-section
z = 50 m.
"""

from onesim.cad.d3 import Box

width, step, thickness = 5200.0, 4000.0, 100.0
deep = Box(length=step, width=320.0, height=thickness, origin=(0.0, -440.0, 0.0)).named(
    "deep"
)
lower = Box(
    length=width, width=80.0, height=thickness, origin=(0.0, -120.0, 0.0)
).named("lower")
confining = Box(
    length=width, width=20.0, height=thickness, origin=(0.0, -40.0, 0.0)
).named("confining")
upper = Box(length=width, width=20.0, height=thickness, origin=(0.0, -20.0, 0.0)).named(
    "upper"
)

(deep | lower | confining | upper).save("terzaghi_aquifer_compaction.step")
confining.save("terzaghi_aquifer_compaction_confining.step")
