"""Geometry of the OneSimulation example "Hyperbolic waves in a metal-dielectric layered metamaterial".

Builds, as thin plates with the CAD objects of OneSimulation and in micrometres, (1) one period of
width 0.1 um of an infinite stack of eight bilayers (20 nm dielectric on 20 nm silver) between air
(hyperbolic_metamaterial_layers_cell.step, with _cell_metal and _cell_dielectric; x_min and x_max
are the periodic sides, y_max and y_min the ports), and (2) a section 3.2 um wide and 1.8 um high
through the same stack with a line source 30 nm above it (hyperbolic_metamaterial_layers.step, with
_metal, _dielectric and _source; all outer faces lie behind the absorbing layers).
"""

from onesim.cad.d3 import Box, Cylinder

layer, pairs, depth = 0.02, 8, 0.1
stack = lambda width, kind: [
    Box(
        length=width,
        width=layer,
        height=depth,
        origin=(-width / 2, -(2 * k + kind + 1) * layer, 0),
    )
    for k in range(pairs)
]
union = lambda boxes: boxes[0] if len(boxes) == 1 else boxes[0] | union(boxes[1:])
Box(length=0.1, width=1.12, height=depth, origin=(-0.05, -0.72, 0)).save(
    "hyperbolic_metamaterial_layers_cell.step"
)
union(stack(0.1, 1)).save("hyperbolic_metamaterial_layers_cell_metal.step")
union(stack(0.1, 0)).save("hyperbolic_metamaterial_layers_cell_dielectric.step")
Box(length=3.2, width=1.8, height=depth, origin=(-1.6, -1.1, 0)).save(
    "hyperbolic_metamaterial_layers.step"
)
union(stack(3.2, 1)).save("hyperbolic_metamaterial_layers_metal.step")
union(stack(3.2, 0)).save("hyperbolic_metamaterial_layers_dielectric.step")
Cylinder(radius=0.01, length=depth, origin=(0.0, 0.03, 0)).save(
    "hyperbolic_metamaterial_layers_source.step"
)
