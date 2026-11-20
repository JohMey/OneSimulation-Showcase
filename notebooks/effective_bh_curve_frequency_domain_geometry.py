"""Geometry of the OneSimulation example "Effective B-H curve: harmonic against transient saturation".

Builds with the CAD objects of OneSimulation a square steel core frame of 100 mm x 100 mm with a
60 mm x 60 mm window (legs of 20 mm), a winding of two coil sides around its left leg (inside the
window and outside the core, 8 mm x 40 mm each) and the air around them, a square of 300 mm; all are
10 mm deep. Files: effective_bh_curve_frequency_domain.step (air), effective_bh_curve_frequency_domain_core.step,
effective_bh_curve_frequency_domain_go.step and effective_bh_curve_frequency_domain_return.step.
The example computes on the cross-section z = 5 mm.
"""

from onesim.cad.d3 import Box

depth = 0.01
Box(length=0.3, width=0.3, height=depth, origin=(-0.15, -0.15, 0.0)).save(
    "effective_bh_curve_frequency_domain.step"
)
(
    Box(length=0.1, width=0.1, height=depth, origin=(-0.05, -0.05, 0.0))
    - Box(length=0.06, width=0.06, height=depth, origin=(-0.03, -0.03, 0.0))
).save("effective_bh_curve_frequency_domain_core.step")
Box(length=0.008, width=0.04, height=depth, origin=(-0.028, -0.02, 0.0)).save(
    "effective_bh_curve_frequency_domain_go.step"
)
Box(length=0.008, width=0.04, height=depth, origin=(-0.06, -0.02, 0.0)).save(
    "effective_bh_curve_frequency_domain_return.step"
)
