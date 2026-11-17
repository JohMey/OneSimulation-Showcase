"""Geometry of the OneSimulation example "Lid-driven cubic cavity in three dimensions".

Builds one half of the cube of side 1 m, cut at its mid-plane z = 0.5 m, with the CAD objects of
OneSimulation and writes lid_driven_cube_3d.step. The lid is the face y_max, the mid-plane z_max;
the faces keep their names x_min, x_max, y_min, y_max, z_min and z_max in the file.
"""

from onesim.cad.d3 import Box

half_cube = Box(length=1.0, width=1.0, height=0.5)
half_cube.save("lid_driven_cube_3d.step")
