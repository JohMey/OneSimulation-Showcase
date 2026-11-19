"""OneSimulation example: horseshoe permanent magnet (3D magnetostatics with remanent magnetization).
Requires the installed, activated OneSimulation runtime.

A horseshoe of soft iron whose two pole pieces are magnetized in opposite directions is read from
STEP files and solved for its 3D field with edge elements. Without the iron the field is checked
against the closed-form field of uniformly magnetized blocks; with the iron a mesh study follows.
"""

# # Horseshoe permanent magnet
#
# A horseshoe magnet of square cross-section (20 mm) consists of a half-ring and two straight legs.
# The lowest 20 mm of each leg are pole pieces of a rare-earth magnet (remanence $B_r=1.2$ T),
# magnetized along the leg and in opposite directions: one pole piece points down, the other up.
# The rest of the horseshoe is soft iron ($\mu_r=1000$) that carries the flux from one pole piece to
# the other, so that the field concentrates in the gap between the poles.
#
# There are no free currents. The magnet enters through its magnetization $\mathbf M$ in
# $\mathbf B=\mu(\mathbf H+\mathbf M)$, with $\mathbf M=B_r/\mu_0$ along the leg in the pole pieces
# and zero elsewhere, and OneSimulation solves
#
# $$\nabla\times\big(\mu^{-1}\nabla\times\mathbf A\big)=\nabla\times\mathbf M,\qquad \mathbf B=\nabla\times\mathbf A$$
#
# with second-order Nédélec edge elements and a Coulomb gauge. The recoil permeability of the magnet
# is taken as 1.
#
# One quarter is modelled. The air box comes from `horseshoe_permanent_magnet.step` with the face
# groups `symmetry_x`, `symmetry_z` and `outer`; the iron of the quarter and its pole piece come
# from `horseshoe_permanent_magnet_yoke.step` and `horseshoe_permanent_magnet_pole.step`. The script
# `horseshoe_permanent_magnet_geometry.py` builds all three with the CAD objects of OneSimulation.
#
# | faces | condition |
# |---|---|
# | `symmetry_z`, the mid-plane of the horseshoe | field tangential, $\mathbf n\times\mathbf A=0$ |
# | `symmetry_x`, the plane between the legs | field normal, $\mathbf n\times\mathbf H=0$ |
# | `outer`, 25 cm from the magnet | $\mathbf n\times\mathbf A=0$ |
#
# **Reference.** A uniformly magnetized block in air acts like two sheets of magnetic charge
# $\pm\mathbf M\cdot\mathbf n$ on its end faces, whose field has a closed form (E. P. Furlani,
# *Permanent Magnet and Electromechanical Devices*, Academic Press 2001, Sec. 4.3; Akoun and Yonnet,
# *IEEE Trans. Magn.* 20, 1984). With the iron replaced by air the model must reproduce the field of
# the two pole blocks exactly. With the iron there is no closed form; a mesh study checks the result.

import numpy as np
from onesim.cad.d3 import Box
from onesim.cad.exchange import load
from onesim.materials import VacuumPermeability
from onesim.meshing import LocalRefinement, TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import SparseDirect
from onesim.physics.magnetostatics import (
    CoulombGauge,
    MagneticFieldBoundary,
    MagneticInsulation,
    MagneticVectorPotential,
    MagnetizationContribution,
    Magnetostatics,
    SampledMagnetizationLoads,
    curl_magnetization_load,
    curl_space_magnetization_patches,
)
from onesim.simulation import Problem, Simulation, regionwise
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the air box with its named faces, the iron and the pole piece. The mesher embeds
# both bodies in the mesh of the air, refines them and the air around the pole pieces, and grows
# the cells towards the outer faces. Three meshes differ in the cell size in and around the magnet.

air, yoke, pole = (
    load(f"horseshoe_permanent_magnet{name}.step") for name in ("", "_yoke", "_pole")
)
faces = air.boundaries
low, high = (np.array(corner) for corner in pole.bounding_box)
sizes = (0.0060, 0.0045, 0.0034)
meshes = [
    TetrahedralMesher(
        target_size=0.05,
        embedded_regions=(yoke, pole),
        refinements=(
            LocalRefinement(region=yoke, target_size=size),
            LocalRefinement(region=pole, target_size=size),
            LocalRefinement(
                region=Box(
                    length=0.06, width=0.06, height=0.03, origin=(0.0, -0.09, 0.0)
                ),
                target_size=1.5 * size,
            ),
        ),
    ).generate(air)
    for size in sizes
]

view = plots.Camera(elevation=25, azimuth=-60)
fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    [yoke.named("soft iron"), pole.named("pole piece")],
    camera=view,
    title="Quarter of the horseshoe",
    ax=fig[0],
)
plots.mesh(
    meshes[0],
    groups=("symmetry_z",),
    legend=False,
    camera=plots.Camera(elevation=90, azimuth=-90),
    title=f"Coarsest mesh: {meshes[0].cell_count} tetrahedra (mid-plane)",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The magnetization is $B_r/\mu_0$ along $-y$ in the pole piece of the quarter (its mirror image
# points along $+y$). The permeability is $\mu_0\mu_r$ in the iron, which `regionwise` assigns from a
# test of the half-ring and the upper leg. The magnetization enters the weak form as
# $\int\mathbf M\cdot\nabla\times\mathbf v\,\mathrm dV$ on the curl space of the edge elements.

remanence, mu_iron = 1.2, 1000.0
mu0 = float(VacuumPermeability().evaluate())
in_pole = lambda points: np.all((points >= low) & (points <= high), axis=1)
radius = lambda points: np.hypot(points[:, 0], points[:, 1])
in_iron = lambda points: (points[:, 2] <= high[2]) & np.where(
    points[:, 1] > 0,
    (radius(points) >= low[0]) & (radius(points) <= high[0]),
    (points[:, 1] >= high[1]) & (points[:, 0] >= low[0]) & (points[:, 0] <= high[0]),
)
magnetization = MagnetizationContribution(
    lambda points, time: np.where(
        in_pole(points)[:, None], (0.0, -remanence / mu0, 0.0), 0.0
    ),
    SampledMagnetizationLoads(
        curl_space_magnetization_patches,
        lambda points, cell, offset, context: offset(points, context.time),
        curl_magnetization_load,
    ),
)
numerics = Numerics(
    spatial=FiniteElement(element=Nedelec(order=2)), linear_solver=SparseDirect()
)

models = {
    mu_r: Magnetostatics(
        domain=air.volume,
        formulation=MagneticVectorPotential(gauge=CoulombGauge()),
        magnetization_provider=magnetization,
        permeability=regionwise(((in_iron, mu0 * mu_r),), default=mu0, unit="H/m"),
    )
    for mu_r in (1.0, mu_iron)
}
problems = {
    mu_r: Problem(
        name="Horseshoe magnet",
        geometry=air,
        physics=(model,),
        boundary_conditions=(
            MagneticInsulation(boundary=faces.symmetry_z | faces.outer, physics=model),
            MagneticFieldBoundary(
                boundary=faces.symmetry_x, physics=model, value=(0.0, 0.0, 0.0)
            ),
        ),
    )
    for mu_r, model in models.items()
}
without_iron = (
    Simulation(problem=problems[1.0], mesh=meshes[-1], numerics=numerics)
    .solve_stationary()
    .field(models[1.0].magnetic_flux_density)
)
with_iron = [
    Simulation(problem=problems[mu_iron], mesh=mesh, numerics=numerics)
    .solve_stationary()
    .field(models[mu_iron].magnetic_flux_density)
    for mesh in meshes
]

# ## Validation without the iron
#
# The closed-form field of the two pole blocks of the full horseshoe, along the line across the
# gap at the height of the pole pieces and along the axis of symmetry below the magnet. Points
# closer than 1.5 mm to a pole face, where the field jumps, are left out of the deviation. `block` is
# the charge-sheet formula of Furlani (2001, Sec. 4.3) for a block magnetized along $y$, summed over
# its two end faces and four corners.

block = (
    lambda p, low, high, br: br
    / (4 * np.pi)
    * np.einsum(
        "fab,pkfab->pk",
        np.multiply.outer((1, -1), ((1, -1), (-1, 1))),
        np.stack(
            (
                np.log(
                    (
                        du := np.reshape((low[2], high[2]), (2, 1))
                        - p[:, None, None, None, 2]
                    )
                    + (
                        r := np.sqrt(
                            du**2
                            + (dv := (low[0], high[0]) - p[:, None, None, None, 0]) ** 2
                            + (
                                dw := p[:, None, None, None, 1]
                                - np.reshape((high[1], low[1]), (2, 1, 1))
                            )
                            ** 2
                        )
                    )
                ),
                np.arctan2(du * dv, dw * r),
                np.log(dv + r),
            ),
            axis=1,
        ),
    )
)
right = (np.array((low[0], low[1], -high[2])), high)
left = (np.array((-high[0], low[1], -high[2])), np.array((-low[0], high[1], high[2])))
blocks = lambda points: block(points, *right, -remanence) + block(
    points, *left, remanence
)
middle = (low[1] + high[1]) / 2
lines = {
    "across the gap": np.column_stack(
        (np.linspace(0.001, 0.07, 70), np.full(70, middle), np.full(70, 0.001))
    ),
    "below the magnet": np.column_stack(
        (np.full(70, 0.001), np.linspace(-0.065, -0.2, 70), np.full(70, 0.001))
    ),
}
print(
    "line               largest deviation from the closed form   field at the start of the line"
)
for label, points in lines.items():
    exact, computed = blocks(points), without_iron.evaluate(points)
    away = np.all(np.abs(points[:, :1] - np.array((low[0], high[0]))) > 0.0015, axis=1)
    print(
        f"{label:18s} {100 * np.max(np.linalg.norm(computed - exact, axis=1)[away]) / np.max(np.linalg.norm(exact, axis=1)):18.2f} % of the largest field"
        f"   {1e3 * np.linalg.norm(computed[0]):8.2f} mT (closed form {1e3 * np.linalg.norm(exact[0]):8.2f} mT)"
    )

# ## Results with the soft iron
#
# The field strength 5 mm below the opening between the pole ends on the three meshes, and the
# gain from the iron.

centre = np.array([[0.001, low[1] - 0.005, 0.001]])
gap_field = [np.linalg.norm(field.evaluate(centre)) for field in with_iron]
print("cell size [mm]   tetrahedra   |B| 5 mm below the pole ends [mT]")
for size, mesh, value in zip(sizes, meshes, gap_field):
    print(f"{1e3 * size:14.1f} {mesh.cell_count:12d} {1e3 * value:22.2f}")
print(
    f"spread over the three meshes {100 * (max(gap_field) - min(gap_field)) / gap_field[-1]:.2f} % of the finest value"
)
print(
    f"without the iron: {1e3 * np.linalg.norm(without_iron.evaluate(centre)):.2f} mT (closed form {1e3 * np.linalg.norm(blocks(centre)):.2f} mT); "
    f"the soft iron raises the field there by a factor {gap_field[-1] / np.linalg.norm(blocks(centre)):.2f}"
)

# ## Figure
#
# The field strength in the mid-plane of the horseshoe on a logarithmic scale (the quarter
# mirrored), and the field across the gap with and without the iron.

strength = lambda B, position: 1e3 * np.linalg.norm(B, axis=1)
gap_line = lines["across the gap"]
fig = plots.figure(1, 2, size=(15, 6.5))
plots.surface(
    with_iron[-1],
    boundary=faces.symmetry_z,
    component=lambda B, position: np.log10(np.maximum(strength(B, position), 1e-3)),
    symmetry={"planes": [((1, 0, 0), (0, 0, 0))]},
    limits=(0, 3),
    contours=13,
    subdivisions=2,
    label="log10(|B| / mT)",
    title="Field strength in the mid-plane",
    xlimits=(-0.1, 0.1),
    ylimits=(-0.13, 0.07),
    ax=fig[0],
)
plots.line(
    points={
        "with soft iron": (
            1e3 * gap_line[::3, 0],
            strength(with_iron[-1].evaluate(gap_line[::3]), None),
        ),
        "without iron": (
            1e3 * gap_line[::3, 0],
            strength(without_iron.evaluate(gap_line[::3]), None),
        ),
    },
    reference={
        "closed form without iron": (
            1e3 * gap_line[:, 0],
            strength(blocks(gap_line), None),
        )
    },
    vlines={"pole face": 1e3 * low[0]},
    xlabel="distance from the middle of the gap x [mm]",
    ylabel="|B| [mT]",
    title="Field across the pole gap",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Lengthen the pole pieces or narrow the gap in `horseshoe_permanent_magnet_geometry.py` and write
# the STEP files again; give the magnet its recoil permeability of 1.05; or replace the linear iron
# by a saturating magnetization curve to see where the half-ring saturates.
