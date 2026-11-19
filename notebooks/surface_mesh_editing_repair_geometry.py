"""Input file of the OneSimulation example "Editing and repairing surface meshes".

Writes surface_mesh_editing_repair.stl: the dense triangle skin of a ball of 30 mm radius, built
with the CAD objects of OneSimulation, with the two defects scans typically have: the cap above
z = 25 mm is missing (the part the scanner could not see) and a small patch on the side is torn out.
"""

import numpy as np
from onesim.cad.d3 import FacetBody, Sphere

ball = Sphere(radius=0.03).faceted()
middles = ball.points[ball.triangles].mean(axis=1)
torn = np.linalg.norm(middles - np.array([0.0, -0.03, 0.0]), axis=1) < 0.004
FacetBody(
    points=ball.points, triangles=ball.triangles[(middles[:, 2] < 0.025) & ~torn]
).save("surface_mesh_editing_repair.stl")
