"""OneSimulation example: bearing capacity of a smooth strip footing on undrained clay.
Requires the installed, activated OneSimulation runtime.

A strip footing is pressed into a stratum of clay. The half section is read from a STEP file and
cut to its plane, and the elastic-perfectly plastic soil is computed in plane strain with
quadratic triangles. The footing pressure approaches a collapse value, which is compared with
Prandtl's bearing capacity (2 + π) s_u on two meshes.
"""

# # Bearing capacity of a strip footing on clay
#
# How much load can a shallow foundation carry before the soil under it fails? For a long
# (strip) footing on saturated clay loaded quickly, the clay behaves undrained: it deforms at
# constant volume and fails when the shear stress reaches the undrained shear strength $s_u$.
# Prandtl's solution of 1920 for a rigid-plastic half-space gives the collapse pressure of a
# smooth footing
#
# $$q_f=(2+\pi)\,s_u\approx5.14\,s_u ,$$
#
# independent of the footing width and of the stiffness of the soil (L. Prandtl, Z. Angew. Math.
# Mech. 1, 1921, 15-20; e.g. D.M. Potts and L. Zdravković, *Finite Element Analysis in
# Geotechnical Engineering: Application*, Telford 2001, where this case is a standard benchmark).
#
# **Model.** The clay is elastic-perfectly plastic with the Drucker-Prager criterion at zero
# friction angle, which is the von Mises criterion; in plane strain its cohesion
# $c=\tfrac{\sqrt3}{2}s_u$ gives the shear strength $s_u$.
#
# **Setup.** A footing of width $B=1$ m on a clay stratum 3 m deep; one half is modelled, 5 m
# wide. $E=10$ MPa, $\nu=0.3$, $s_u=50$ kPa. The footing is smooth and is pushed down by 0.15 m.
# The stratum comes from the CAD file `strip_footing_bearing_capacity.step`, written by
# `strip_footing_bearing_capacity_geometry.py` with the CAD objects of OneSimulation; the soil under
# the footing is its own block, so the ground surface is split at the edge of the footing.
#
# | side | condition |
# |---|---|
# | `footing.y_max` (under the footing) | $u_y=-\delta$, smooth |
# | `soil.y_max` (ground surface) | free |
# | `footing.x_min` (symmetry), `soil.x_max` | $u_x=0$ |
# | `footing.y_min`, `soil.y_min` (rigid base) | $u_y=0$, smooth |
#
# **Reference.** Prandtl's collapse pressure $(2+\pi)\,s_u=257.1$ kPa.

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import TriangularMesher
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import DampedNewton
from onesim.physics.geotechnics import DruckerPragerSoil, FrictionalSolid
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `section` cuts the stratum at mid-thickness; the cut faces keep their names. The triangles are
# graded towards the edge of the footing, where the stresses change fastest: three meshes with
# 6, 3 and 1.5 cm at the edge.

stratum = load("strip_footing_bearing_capacity.step")
plane = stratum.section((0, 0, 1), origin=(0, 0, stratum.size[2] / 2))
faces = plane.boundaries
edge_sizes = [0.06, 0.03, 0.015]
meshes = [
    TriangularMesher(
        target_size=0.3,
        size_field=lambda p, h=h: np.minimum(
            h + 0.15 * np.hypot(p[:, 0] - 0.5, p[:, 1]), 0.3
        ),
    ).generate(plane)
    for h in edge_sizes
]

fig = plots.figure(1, 2, size=(15, 5))
plots.geometry(
    plane,
    selection={
        "footing": faces["footing.y_max"],
        "ground surface": faces["soil.y_max"],
        "rollers": faces["footing.x_min"]
        | faces["soil.x_max"]
        | faces["footing.y_min"]
        | faces["soil.y_min"],
    },
    labels=False,
    title="Half section of the clay stratum",
    ax=fig[0],
)
plots.mesh(
    meshes[-1],
    groups=("all",),
    legend=False,
    title=f"Finer mesh: {meshes[-1].cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Plane strain with quadratic triangles; the footing goes down in twelve steps to 0.15 m.

undrained_strength = 50e3
clay = FrictionalSolid(
    material=DruckerPragerSoil(
        youngs_modulus=10e6,
        poisson_ratio=0.3,
        cohesion=np.sqrt(3) / 2 * undrained_strength,
        friction_angle=0.0,
    ),
    kinematics="plane_strain",
)
settlement = np.r_[np.arange(0.0, 0.06, 0.01), np.arange(0.06, 0.151, 0.015)]
results = [
    clay.solve(
        mesh=mesh,
        times=settlement,
        spatial=FiniteElement(element=Lagrange(2)),
        nonlinear=DampedNewton(
            relative_tolerance=1e-8,
            absolute_tolerance=1e-6,
            maximum_iterations=60,
            linear_solver=SparseDirect(),
        ),
        displacements=(
            (faces["footing.x_min"], "x", 0.0),
            (faces["soil.x_max"], "x", 0.0),
            (faces["footing.y_min"] | faces["soil.y_min"], "y", 0.0),
            (faces["footing.y_max"], "y", lambda t: -t),
        ),
    )
    for mesh in meshes
]
pressure = [
    -result.reactions[
        :,
        (result.coordinates[:, 0] <= 0.5 + 1e-9)
        & (np.abs(result.coordinates[:, 1]) < 1e-9),
        1,
    ].sum(axis=1)
    / 0.5
    for result in results
]

# ## Comparison with the reference
#
# The mean footing pressure at the largest settlement against Prandtl's value, and its last
# increase, which shows how close the soil is to collapse.

prandtl = (2 + np.pi) * undrained_strength
print(f"Prandtl: q_f = (2 + π) s_u = {prandtl / 1e3:.1f} kPa")
print(
    f"{'size at the edge':>16s} {'triangles':>9s} {'q at δ = 0.15 m [kPa]':>22s} {'q / s_u':>8s} {'deviation':>10s} {'increase over the last step':>28s}"
)
for size, mesh, values in zip(edge_sizes, meshes, pressure):
    print(
        f"{100 * size:13.1f} cm {mesh.cell_count:9d} {values[-1] / 1e3:22.1f} {values[-1] / undrained_strength:8.3f}"
        f" {100 * (values[-1] / prandtl - 1):+9.1f} % {100 * (values[-1] / values[-2] - 1):27.2f} %"
    )

# ## Figure
#
# Load-settlement curves of the three meshes with Prandtl's value, and how far the soil moves in the
# last step per unit footing movement: near collapse this outlines the failure mechanism, the
# rigid wedge under the footing, the fan and the passive wedge beside it.

result, mesh = results[-1], meshes[-1]
nodes = len(mesh.coordinates)
increment = np.hypot(
    *(result.displacement[-1, :nodes] - result.displacement[-2, :nodes]).T
)
fig = plots.figure(1, 2, size=(16, 5.5))
plots.line(
    y={
        f"{mesh.cell_count} triangles": (100 * settlement, values / undrained_strength)
        for mesh, values in zip(meshes, pressure)
    },
    hlines={"Prandtl, 2 + π": 2 + np.pi},
    markers=True,
    ylimits=(0, 6),
    xlabel="settlement δ [cm]",
    ylabel="mean footing pressure q / s_u",
    title="Load-settlement curve",
    legend="lower right",
    ax=fig[0],
)
plots.heatmap(
    result.coordinates[:nodes, 0],
    result.coordinates[:nodes, 1],
    100 * increment / (settlement[-1] - settlement[-2]),
    triangles=np.asarray(mesh.cells),
    limits=(0, 100),
    contours=10,
    label="displacement increment / footing increment [%]",
    xlabel="x [m]",
    ylabel="y [m]",
    xlimits=(0, 2.5),
    ylimits=(-1.5, 0),
    title="Collapse mechanism in the last step",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Give the clay a friction angle and compare with the bearing capacity factor
# $N_c=(N_q-1)\cot\varphi$; make the footing rough by fixing $u_x$ under it; or move the base of
# the stratum up in `strip_footing_bearing_capacity_geometry.py` until the shallow layer raises
# the collapse load.
