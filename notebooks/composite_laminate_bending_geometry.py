"""Geometry of the OneSimulation example "Bending of a simply supported cross-ply laminate".

Builds one quarter of a square laminated plate of side a = 100 mm with the CAD objects of
OneSimulation: the box 0 <= x, y <= a/2, 0 <= z <= h. The plate with h = a/4 is written to
composite_laminate_bending.step, the thinner plates with h = a/10, a/20 and a/50 to
composite_laminate_bending_s10.step, _s20.step and _s50.step. The faces keep their names x_min and
y_min (simply supported edges), x_max and y_max (symmetry planes), z_min and z_max (lower and
loaded upper face).
"""

from onesim.cad.d3 import Box

side = 0.100
for ratio, suffix in ((4, ""), (10, "_s10"), (20, "_s20"), (50, "_s50")):
    Box(length=side / 2, width=side / 2, height=side / ratio).save(
        f"composite_laminate_bending{suffix}.step"
    )
