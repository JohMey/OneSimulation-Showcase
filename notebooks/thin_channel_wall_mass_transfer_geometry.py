"""Geometry of the OneSimulation example "Mass transfer out of a thin flow channel".

Builds the thin channel, 20 mm long, 10 mm wide and 0.5 mm thick, with the CAD objects of
OneSimulation and writes thin_channel_wall_mass_transfer.step. The faces keep their names: x_min
(inlet), x_max (outlet), y_min and y_max (side walls), z_min and z_max (the two permeable walls).
The example computes on the mid-plane z = 0.25 mm and uses the full channel as its 3D check.
"""

from onesim.cad.d3 import Box

Box(length=0.020, width=0.010, height=0.0005).save("thin_channel_wall_mass_transfer.step")
