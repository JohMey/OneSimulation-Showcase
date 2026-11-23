"""Geometry of the OneSimulation example "Creep relaxation of thermal stresses in a turbine stator vane".

Builds the airfoil of a stator vane (NACA 4420 section, 40 mm chord, leading edge at x = 0, with a
semicircular trailing edge at 90 % chord) extruded
over a span of 80 mm along z with the CAD objects of OneSimulation and writes
turbine_stator_blade_creep.step. The faces are named bottom (root, z = 0), top (tip, z = 80 mm)
and side (the airfoil surface).
"""

import numpy as np
from onesim.cad.d2 import Profile
from onesim.cad.d3 import Prism

chord, span, camber, position, thickness = 0.040, 0.080, 0.04, 0.4, 0.20
beta = np.linspace(0.0, np.pi, 13)
x = 0.9 * 0.5 * (1 - np.cos(beta))
half = 5 * thickness * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
mean = np.where(x < position, camber / position**2 * (2 * position * x - x**2), camber / (1 - position) ** 2 * (1 - 2 * position + 2 * position * x - x**2))
slope = np.arctan(np.where(x < position, 2 * camber / position**2 * (position - x), 2 * camber / (1 - position) ** 2 * (position - x)))
upper = np.column_stack((x - half * np.sin(slope), mean + half * np.cos(slope)))
lower = np.column_stack((x + half * np.sin(slope), mean - half * np.cos(slope)))
arc = slope[-1] + np.linspace(-0.5 * np.pi, 0.5 * np.pi, 5)[1:-1]
tail = np.column_stack((x[-1] + half[-1] * np.cos(arc), mean[-1] + half[-1] * np.sin(arc)))
outline = chord * np.vstack((upper[::-1][:-1], lower[1:], tail))
vane = Prism(profile=Profile.spline(outline), height=span)
vane.save("turbine_stator_blade_creep.step")
