"""Geometry of the OneSimulation example "Small-signal inductance of an inductor with a nonlinear core".

Builds with the CAD objects of OneSimulation the meridian half plane (x = r >= 0, z) of a pot-core
inductor as thin plates: the air region, 80 mm from the axis and 160 mm along it
(inductor_small_signal_axisymmetric.step; x_min is the axis); the steel pot core, 30 mm in radius and
40 mm high, with a winding window from r = 10 to 26 mm, |z| <= 14 mm, and an air gap of 1 mm in its
centre post (inductor_small_signal_axisymmetric_core.step); and the winding in the window
(inductor_small_signal_axisymmetric_winding.step).
"""

from onesim.cad.d3 import Box

t = 0.002
plate = lambda r0, r1, z0, z1: Box(
    length=r1 - r0, width=t, height=z1 - z0, origin=(r0, -t / 2, z0)
)
plate(0.0, 0.08, -0.08, 0.08).save("inductor_small_signal_axisymmetric.step")
(
    plate(0.0, 0.03, -0.02, 0.02)
    - plate(0.010, 0.026, -0.014, 0.014)
    - plate(-0.001, 0.0105, -0.0005, 0.0005)
).save("inductor_small_signal_axisymmetric_core.step")
plate(0.0115, 0.0245, -0.013, 0.013).save(
    "inductor_small_signal_axisymmetric_winding.step"
)
