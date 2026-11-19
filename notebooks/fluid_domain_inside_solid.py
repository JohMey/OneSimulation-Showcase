"""OneSimulation example: creating the fluid domain inside a solid structure.
Requires the installed, activated OneSimulation runtime.

A manifold block with a through bore and a branch bore is built with the CAD objects of
OneSimulation. The space inside the bores becomes a solid of its own, the fluid domain, with the
bore walls as walls and new faces closing the three openings; block and fluid form an assembly
that shares the wall.

Volume, wall areas and opening areas of the fluid domain are compared with closed forms of the
intersecting cylinders, and both domains are meshed, the fluid with boundary layers on its wall.
"""

# # Creating a fluid domain inside a solid structure
#
# CAD models of valves, manifolds, pumps and heat exchangers describe the metal. A flow or
# conjugate heat transfer model needs the opposite: the space the fluid fills, closed at the inlets
# and outlets. `part.interior(walls)` builds that space from the inner faces of the part. The walls
# keep the identifiers they have on the part, and every opening is closed by a new flat face, so
# inlet and outlet conditions have a face to refer to.
#
# | face group of the fluid | meaning |
# |---|---|
# | `wall` (`main.outer_wall`, `branch.outer_wall`) | the bore walls of the block: no-slip, heat exchange with the metal |
# | `opening` (`opening_1` ... `opening_3`) | the three ends of the bores: inlet and outlets |
#
# The block is $L\times W\times H$ = 120 x 60 x 60 mm; a bore of radius $r$ = 12 mm runs through
# its length at mid-height and a branch bore of the same radius is drilled from the top down to its
# axis.
#
# **Reference.** Two cylinders of equal radius that cross at right angles share the Steinmetz solid
# of volume $16r^3/3$; the branch, which ends at the axis of the main bore, shares half of it.
# Each cylinder wall loses the area $4r^2$ to the other. Hence
#
# $$V_\text{fluid} = \pi r^2 L + \pi r^2 \frac{H}{2} - \frac{8r^3}{3},\qquad
#   A_\text{main} = 2\pi r L - 4r^2,\qquad A_\text{branch} = \pi r H - 4r^2,\qquad A_\text{opening} = \pi r^2 .$$

import numpy as np
from onesim.cad.d3 import Assembly, Box, Cylinder
from onesim.meshing import BoundaryLayers, TetrahedralMesher
from onesim.meshing.quality import quality
from onesim.visualization import plots, theme

theme.set()

# ## The solid structure
#
# The bores are named cylinders subtracted from the block, so their walls are the faces
# `main.outer_wall` and `branch.outer_wall`.

length, width, height, radius = 0.12, 0.06, 0.06, 0.012
block = (
    Box(length=length, width=width, height=height)
    - Cylinder(
        radius=radius,
        length=length,
        origin=(0.0, width / 2, height / 2),
        axis=(1, 0, 0),
    ).named("main")
    - Cylinder(
        radius=radius, length=height / 2, origin=(length / 2, width / 2, height / 2)
    ).named("branch")
)
walls = block.boundaries["main.outer_wall"] | block.boundaries["branch.outer_wall"]
print(f"block faces: {', '.join(face.id for face in block.faces)}")

# ## The fluid domain
#
# `interior` takes the faces that enclose the fluid and returns a solid. Housing and fluid together
# are an `Assembly`: one mesh with a volume for each part that meet node by node at the wall.

fluid = block.interior(walls)
manifold = Assembly(parts=[block.named("housing"), fluid.named("water")])
groups = fluid.boundary_groups()
print(f"fluid faces: {', '.join(face.id for face in fluid.faces)}")
print(f"walls: {', '.join(groups['wall'])}; openings: {', '.join(groups['opening'])}")

mesh = TetrahedralMesher(target_size=0.006, curvature_angle=0.3).generate(manifold)
view = plots.Camera(elevation=30, azimuth=-55)
fig = plots.figure(1, 2, size=(16, 6))
plots.geometry(
    block,
    selection={"bore walls": walls},
    labels=False,
    camera=view,
    title="Manifold block with its bores",
    ax=fig[0],
)
plots.mesh(
    mesh,
    cell_values=quality(mesh)["shape"],
    limits=(0, 1),
    plane=((length / 2, width / 2, height / 2), (0.0, 1.0, 0.0)),
    label="shape measure",
    camera=plots.Camera(elevation=15, azimuth=90),
    title=f"Assembly mesh of housing and water cut open: {mesh.cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Results
#
# Volume and face areas of the fluid domain against the closed forms; the volume of the housing is
# the block less the fluid, and the contact face between the two parts is the whole wall. The
# linear tetrahedra cut the round walls as chords, so the meshed fluid is about
# $1-\sin\alpha/\alpha \approx 1.5\,\%$ smaller for a turn of $\alpha$ = 0.3 rad per cell, and the
# housing larger by the same volume.

exact = {
    "fluid volume [cm3]": (
        fluid.measure,
        np.pi * radius**2 * length + np.pi * radius**2 * height / 2 - 8 * radius**3 / 3,
        1e6,
    ),
    "main bore wall [cm2]": (
        fluid.topology.face("main.outer_wall").area,
        2 * np.pi * radius * length - 4 * radius**2,
        1e4,
    ),
    "branch bore wall [cm2]": (
        fluid.topology.face("branch.outer_wall").area,
        np.pi * radius * height - 4 * radius**2,
        1e4,
    ),
    **{
        f"{name} [cm2]": (fluid.topology.face(name).area, np.pi * radius**2, 1e4)
        for name in groups["opening"]
    },
    "housing volume [cm3]": (
        block.measure,
        length * width * height - fluid.measure,
        1e6,
    ),
}
print("quantity                    CAD           closed form    deviation")
for name, (value, reference, scale) in exact.items():
    print(
        f"{name:24s} {value * scale:12.5f} {reference * scale:14.5f} {value / reference - 1:+12.1e}"
    )
(contact,) = manifold.interfaces
print(
    f"contact between {' and '.join(contact.parts)}: {contact.area * 1e4:.5f} cm2, wall area "
    f"{(exact['main bore wall [cm2]'][1] + exact['branch bore wall [cm2]'][1]) * 1e4:.5f} cm2"
)
volumes = quality(mesh)["size"]
for part in manifold.volumes:
    print(
        f"mesh of {part.id:8s}: {len(mesh.region_cells(manifold.volume_of(part.id))):6d} cells, volume {volumes[list(mesh.region_cells(manifold.volume_of(part.id)))].sum() * 1e6:.4f} cm3 "
        f"(CAD {part.measure * 1e6:.4f} cm3)"
    )

# ## Fluid mesh with boundary layers
#
# For the flow alone the fluid gets its own mesh with five prism-like layers of tetrahedra on the
# wall, 0.2 mm high at the wall and growing by 1.3, where the velocity rises from zero.

layers = BoundaryLayers(
    walls=fluid.boundaries.wall, count=5, first_height=2e-4, growth=1.3
)
flow_mesh = TetrahedralMesher(
    target_size=0.004, curvature_angle=0.3, boundary_layers=(layers,)
).generate(fluid)
report = flow_mesh.layer_report
print(
    f"fluid mesh: {flow_mesh.cell_count} cells, {report['layer_count']} layers, heights "
    f"{', '.join(f'{1e3 * item['mean']:.3f}' for item in report['layer_heights'])} mm, stated "
    f"{', '.join(f'{1e3 * h:.3f}' for h in layers.layer_heights())} mm"
)
print(
    f"fluid mesh volume {quality(flow_mesh)['size'].sum() * 1e6:.4f} cm3, closed form {exact['fluid volume [cm3]'][1] * 1e6:.4f} cm3"
)

# ## Figure
#
# The fluid domain with its wall and its three openings, and its layered mesh cut through the axis
# of the main bore.

fig = plots.figure(1, 2, size=(16, 6))
plots.geometry(
    fluid,
    selection={"wall": fluid.boundaries.wall, "openings": fluid.boundaries.opening},
    labels=False,
    camera=view,
    title="Fluid domain: wall and openings",
    ax=fig[0],
)
plots.mesh(
    flow_mesh,
    cell_values=flow_mesh.cell_layers,
    limits=(0, 5),
    plane=((length / 2, width / 2, height / 2), (0.0, 1.0, 0.0)),
    label="layer (0 = core)",
    camera=plots.Camera(elevation=15, azimuth=90),
    title=f"Fluid mesh with 5 wall layers: {flow_mesh.cell_count} cells",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Solve laminar flow from `opening_1` to the two other openings on `flow_mesh`
# (`onesim.physics.fluid_flow`), or heat the housing and let the assembly mesh carry the heat across
# the shared wall into the water.
