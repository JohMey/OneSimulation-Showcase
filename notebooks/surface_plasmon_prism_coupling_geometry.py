"""Geometry of the OneSimulation example "Surface plasmon excitation in the Otto and Kretschmann configurations".

Builds, as thin plates with the CAD objects of OneSimulation and in micrometres, one period
(0.15 um wide) of the two layered prism couplers at 633 nm:

* Kretschmann: surface_plasmon_prism_coupling_kretschmann.step (0.6 um of glass, a 50 nm gold film,
  0.6 um of air; x_min and x_max periodic, y_max the port in the glass, y_min the port in the air)
  with its parts _kretschmann_glass.step and _kretschmann_gold.step;
* Otto: surface_plasmon_prism_coupling_otto.step (0.6 um of glass, a 450 nm air gap, 0.3 um of gold;
  y_max the port in the glass, y_min the bottom of the gold) with _otto_glass.step and _otto_gold.step.
"""

from onesim.cad.d3 import Box

period, glass, depth = 0.15, 0.6, 0.1
layers = {
    "kretschmann": (("gold", 0.05), ("air", 0.6)),
    "otto": (("air", 0.45), ("gold", 0.3)),
}
for name, ((first, a), (second, b)) in layers.items():
    total = glass + a + b
    Box(length=period, width=total, height=depth, origin=(0, -(a + b), 0)).save(
        f"surface_plasmon_prism_coupling_{name}.step"
    )
    Box(length=period, width=glass, height=depth, origin=(0, 0, 0)).save(
        f"surface_plasmon_prism_coupling_{name}_glass.step"
    )
    bottom = -a if first == "gold" else -(a + b)
    Box(
        length=period,
        width=a if first == "gold" else b,
        height=depth,
        origin=(0, bottom, 0),
    ).save(f"surface_plasmon_prism_coupling_{name}_gold.step")
