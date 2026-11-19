"""Geometry of the OneSimulation example "Plasmonic waveguide filter".

Builds, as thin plates with the CAD objects of OneSimulation and in micrometres, a metal-insulator-
metal waveguide: a 50 nm air channel 1.2 um long between two silver blocks 0.35 um thick
(plasmonic_waveguide_stub_filter.step; x_min and x_max are the ports, y_min and y_max the outer
faces of the silver), the channel (plasmonic_waveguide_stub_filter_channel.step) and a side stub of
the same width, 0.25 um deep into the upper silver (plasmonic_waveguide_stub_filter_stub.step).
"""

from onesim.cad.d3 import Box

length, gap, metal, stub, depth = 1.2, 0.05, 0.35, 0.25, 0.1
Box(
    length=length, width=gap + 2 * metal, height=depth, origin=(0, -gap / 2 - metal, 0)
).save("plasmonic_waveguide_stub_filter.step")
Box(length=length, width=gap, height=depth, origin=(0, -gap / 2, 0)).save(
    "plasmonic_waveguide_stub_filter_channel.step"
)
Box(
    length=gap, width=stub, height=depth, origin=(length / 2 - gap / 2, gap / 2, 0)
).save("plasmonic_waveguide_stub_filter_stub.step")
