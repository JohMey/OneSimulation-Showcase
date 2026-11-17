"""Geometry of the OneSimulation example "Magnetic field of a Helmholtz coil".

Builds the air region around the coil pair, a sphere of 2 m radius about the centre of the pair,
with the CAD objects of OneSimulation and writes helmholtz_coil_field.step. Its one face keeps the
name surface in the file. The two windings are no part of the file: they are current sources of
the model and are placed inside the air by the mesher.
"""

from onesim.cad.d3 import Sphere

air = Sphere(radius=2.0)
air.save("helmholtz_coil_field.step")
