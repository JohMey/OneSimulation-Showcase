"""Geometry of the OneSimulation example "Thermal bridge in 3D" (EN ISO 10211 case 4).

Builds one quarter of the insulation layer (0.5 m x 0.5 m, 0.2 m thick) with the quarter of the
iron bar (0.05 m x 0.025 m, 0.6 m long) with the CAD objects of OneSimulation. The wall is written
to thermal_bridge_3d_iso10211.step with the face groups external, internal and adiabatic; the bar,
which marks the region of the second material, is written to thermal_bridge_3d_iso10211_bar.step.
"""

from onesim.cad.d3 import Box

x_bar, z_bar, t_insulation, y_tip = 0.55, 0.525, 0.2, 0.6
layer = Box(length=0.5, width=t_insulation, height=0.5, origin=(0.5, 0.0, 0.5))
bar = Box(length=x_bar - 0.5, width=y_tip, height=z_bar - 0.5, origin=(0.5, 0.0, 0.5))
wall = layer | bar.named("bar")
faces = wall.boundaries
wall = wall.grouped(
    external=faces.y_min,
    internal=faces.y_max | faces["bar.x_max"] | faces["bar.z_max"] | faces["bar.y_max"],
    adiabatic=faces.x_min
    | faces.z_min
    | faces["bar.x_min"]
    | faces["bar.z_min"]
    | faces.x_max
    | faces.z_max,
)
wall.save("thermal_bridge_3d_iso10211.step")
bar.save("thermal_bridge_3d_iso10211_bar.step")
