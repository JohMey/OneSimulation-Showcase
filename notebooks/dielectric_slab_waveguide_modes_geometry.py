"""Geometry of the OneSimulation example "Dielectric slab waveguide".

Builds with the CAD objects of OneSimulation, as thin plates, a polyethylene slab 4 mm thick in air:
the air around a 100 mm long piece of it, 24 mm high, including the absorbing layers
(dielectric_slab_waveguide_modes.step), the slab in it (dielectric_slab_waveguide_modes_core.step),
and a 0.5 mm long periodic segment of the guide between metal walls 24 mm apart for the mode analysis
(dielectric_slab_waveguide_modes_cell.step with its slab dielectric_slab_waveguide_modes_cell_core.step).
The faces keep the names x_min, x_max, y_min, y_max.
"""

from onesim.cad.d3 import Box

core, thickness = 4e-3, 1e-3
for name, length, height in (("", 0.1, 0.024), ("_cell", 0.5e-3, 0.024)):
    Box(
        origin=(0.0, -height / 2, 0.0), length=length, width=height, height=thickness
    ).save(f"dielectric_slab_waveguide_modes{name}.step")
    Box(origin=(0.0, -core / 2, 0.0), length=length, width=core, height=thickness).save(
        f"dielectric_slab_waveguide_modes{name}_core.step"
    )
