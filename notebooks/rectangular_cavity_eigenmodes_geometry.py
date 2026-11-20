"""Geometry of the OneSimulation example "Eigenmodes of a rectangular metallic cavity".

Builds with the CAD objects of OneSimulation a thin plate whose cross-section is the cavity,
200 mm x 100 mm, and writes rectangular_cavity_eigenmodes.step; all its side faces are metal walls.
"""

from onesim.cad.d3 import Box

Box(length=0.2, width=0.1, height=0.005).save("rectangular_cavity_eigenmodes.step")
