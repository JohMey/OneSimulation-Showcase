"""Geometry of the OneSimulation example "Fabry-Perot resonator with Bragg mirrors".

Builds with the CAD objects of OneSimulation, drawn in micrometres as thin plates, a strip 0.1 um wide
(y) along the optical axis x: air from x = 0 to 0.5, a Bragg mirror of five pairs of quarter-wave
layers for 1 um (titanium dioxide, n = 2.4, 104.2 nm; silica, n = 1.45, 172.4 nm), an air gap of 5 um,
the mirror in reverse order and 0.5 um of air. Files: fabry_perot_bragg_mirror_resonator.step (the strip,
faces x_min and x_max are the ports, y_min and y_max the periodic sides),
fabry_perot_bragg_mirror_resonator_high.step and ..._low.step (the layers of each material).
"""

from onesim.cad.d3 import Box

width, depth, design, pairs, gap, lead = 0.1, 0.05, 1.0, 5, 5.0, 0.5
high, low = design / (4 * 2.4), design / (4 * 1.45)
mirror = [high, low] * pairs
layers = mirror + [gap] + mirror[::-1]
x, slabs = lead, {"high": None, "low": None}
for index, thickness in enumerate(layers):
    kind = "high" if thickness == high else ("low" if thickness == low else None)
    if kind:
        slab = Box(length=thickness, width=width, height=depth, origin=(x, 0.0, 0.0))
        slabs[kind] = slab if slabs[kind] is None else slabs[kind] | slab
    x += thickness
Box(length=x + lead, width=width, height=depth).save("fabry_perot_bragg_mirror_resonator.step")
for kind, solid in slabs.items():
    solid.save(f"fabry_perot_bragg_mirror_resonator_{kind}.step")
