"""OneSimulation example: the geometry of an incandescent light bulb.
Requires the installed, activated OneSimulation runtime.

Glass envelope, filling gas, metal base and coiled filament of a light bulb are built with the CAD
objects of OneSimulation: revolved profiles of lines and arcs for glass and gas, a cylinder for
the base and a wire swept along a helix for the filament. The four parts form an assembly; the glass, the
space inside it and the base are meshed as one with a volume for every part.

The volumes of the revolved parts are compared with the theorem of Pappus and Guldin applied to
their profiles, the filament with the length of its helix.
"""

# # Light bulb geometry
#
# Lamps are a classic case for heat transfer and radiation models, and a good exercise in building
# a model geometry from its natural pieces:
#
# | part | built from | material |
# |---|---|---|
# | `glass` | revolved outline of the envelope less revolved outline of its inside | glass, 0.8 mm wall |
# | `gas` | revolved inside outline less the filament | argon |
# | `base` | cylinder below the neck | brass |
# | `filament` | circle of 0.3 mm radius swept along a helix of six turns | tungsten |
#
# The outline of the envelope is a closed path in the $(r,z)$ half-plane: along the axis, out to
# the neck of radius 13 mm, up the neck, out along a straight shoulder and round a circular arc of
# 30 mm radius to the top. The inner outline is the same path 0.8 mm smaller.
#
# **Reference.** A plane region of area $A$ whose centroid lies at the distance $\bar r$ from the
# axis sweeps the volume $V = 2\pi\bar r A$ when it turns about the axis (Pappus and Guldin);
# area and centroid of each outline follow exactly from its lines and arcs. The filament is a tube
# of radius $a$ around a helix of $N$ turns, radius $R$ and pitch $p$:
# $V = \pi a^2 N\sqrt{(2\pi R)^2 + p^2}$.

import numpy as np
from onesim.cad.d2 import Path, PlanarRegion, Profile, Segment
from onesim.cad.d3 import Assembly, Cylinder, Helix, Revolution, Sweep
from onesim.meshing import TetrahedralMesher
from onesim.meshing.quality import quality
from onesim.visualization import plots, theme

theme.set()

# ## Outlines and parts
#
# The inner outline is the outer one made smaller by the wall thickness: the arc keeps its centre,
# the neck moves in, the top moves down and the bottom moves up, so the glass is closed at the base.

neck, neck_height, sphere, centre, wall = 0.013, 0.025, 0.030, 0.065, 0.0008
outer, inner = [
    Profile(
        Path(
            (0.0, t),
            (
                Segment((neck - t, t)),
                Segment((neck - t, neck_height)),
                Segment(
                    (
                        (sphere - t) * np.cos(np.pi / 3),
                        centre - (sphere - t) * np.sin(np.pi / 3),
                    )
                ),
                Segment((0.0, centre + sphere - t), center=(0.0, centre)),
                Segment((0.0, t)),
            ),
        )
    )
    for t in (0.0, wall)
]
helix = Helix(
    radius=0.0015, pitch=0.001, turns=6, origin=(-0.003, 0.0, 0.06), axis=(1, 0, 0)
)
filament = Sweep(profile=Profile.circle(0.0003), path=helix).named("filament")
glass = (Revolution(profile=outer) - Revolution(profile=inner).named("inside")).named(
    "glass"
)
gas = (Revolution(profile=inner) - filament).named("gas")
base = Cylinder(radius=neck, length=0.02, origin=(0.0, 0.0, -0.02)).named("base")
bulb = Assembly(parts=[glass, gas, base, filament])
print(
    f"parts: {', '.join(part.id for part in bulb.volumes)}; contacts: "
    f"{', '.join(' / '.join(contact.parts) for contact in bulb.interfaces)}"
)

lamp = Assembly(parts=[glass, Revolution(profile=inner).named("inside"), base])
mesh = TetrahedralMesher(target_size=0.003, gap_cells=1).generate(lamp)
view = plots.Camera(elevation=15, azimuth=-60)
fig = plots.figure(1, 2, size=(15, 8))
plots.geometry(
    bulb,
    selection={part.id: bulb.volume_of(part.id) for part in bulb.volumes},
    labels=False,
    camera=view,
    title="Light bulb: glass, gas, base and filament",
    ax=fig[0],
)
plots.mesh(
    mesh,
    cell_values=quality(mesh)["shape"],
    limits=(0, 1),
    plane=((0.0, 0.0, 0.04), (0.0, 1.0, 0.0)),
    label="shape measure",
    camera=plots.Camera(elevation=5, azimuth=90),
    title=f"Glass, inside and base meshed as one, cut: {mesh.cell_count} cells",
    ax=fig[1],
)
plots.show()

# ## Results
#
# Pappus and Guldin from the section values of the two outlines and the helix length of the
# filament against the volume of every part of the CAD model, then the three meshed parts.

pappus = lambda profile: 2 * np.pi * profile.section_values().centroid[0] * profile.area
coil = np.pi * 0.0003**2 * 6 * np.hypot(2 * np.pi * 0.0015, 0.001)
closed = {
    "glass": pappus(outer) - pappus(inner),
    "gas": pappus(inner) - coil,
    "base": np.pi * neck**2 * 0.02,
    "filament": coil,
}
print("part        CAD [cm3]   closed form [cm3]   deviation")
for part in bulb.volumes:
    print(
        f"{part.id:9s} {part.measure * 1e6:12.5f} {closed[part.id] * 1e6:16.5f} {part.measure / closed[part.id] - 1:+12.1e}"
    )
cells = quality(mesh)["size"]
reference = {"glass": closed["glass"], "inside": pappus(inner), "base": closed["base"]}
for part in lamp.volumes:
    meshed = cells[list(mesh.region_cells(lamp.volume_of(part.id)))].sum()
    print(
        f"mesh of {part.id:7s}: {meshed * 1e6:10.5f} cm3 against {reference[part.id] * 1e6:10.5f} cm3 ({meshed / reference[part.id] - 1:+.2%})"
    )
print(
    f"filament wire length {helix.length * 1e3:.3f} mm, closed form {6 * np.hypot(2 * np.pi * 0.0015, 0.001) * 1e3:.3f} mm"
)

# ## Figure
#
# The two outlines in the half-plane and the filament with its helix, rendered.

fig = plots.figure(1, 3, size=(20, 7))
plots.geometry(
    PlanarRegion(profile=outer), title="Outline of the envelope (r, z)", ax=fig[0]
)
plots.geometry(
    PlanarRegion(profile=inner),
    title="Outline of the inside, 0.8 mm smaller",
    ax=fig[1],
)
plots.render(
    filament,
    camera=plots.Camera(elevation=20, azimuth=-60),
    material="steel",
    title="Filament: six turns of 0.6 mm wire",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Give the parts their materials with `regionwise`, heat the filament and let the gas carry the
# heat to the glass (`onesim.physics.heat_transfer`), or add the two support wires as swept paths
# from the base to the coil.
