"""Geometry of the OneSimulation example "Four-point bending of a layered power module".

Builds one quarter of the layered strip (30 mm half length, 5 mm half width, 4.13 mm thick) with
the CAD objects of OneSimulation and writes four_point_bending_layered_module.step; its faces keep
the names x_min, x_max, y_min, y_max, z_min and z_max. The solder, ceramic and top copper layers,
which mark the regions of the other materials, are written to
four_point_bending_layered_module_solder.step, ..._ceramic.step and ..._copper.step.
"""

from onesim.cad.d3 import Box

half_length, half_width = 0.030, 0.005
layers = {"baseplate": 3.0e-3, "solder": 0.2e-3, "ceramic": 0.63e-3, "copper": 0.3e-3}

strip = Box(length=half_length, width=half_width, height=sum(layers.values()))
strip.save("four_point_bending_layered_module.step")
bottom = layers["baseplate"]
for name in ("solder", "ceramic", "copper"):
    Box(
        length=half_length,
        width=half_width,
        height=layers[name],
        origin=(0.0, 0.0, bottom),
    ).save(f"four_point_bending_layered_module_{name}.step")
    bottom += layers[name]
