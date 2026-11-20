"""OneSimulation example: stresses and shear-failure risk around a multilateral well junction.
Requires the installed, activated OneSimulation runtime.

A vertical well with an inclined lateral leg is drilled into rock under anisotropic in-situ stresses.
Half of the rock block is read from a STEP file and meshed with curved quadratic tetrahedra refined
at the bore walls; the far-field stresses act on the outer faces and the mud pressure on the walls.
The hoop stress around the main bore away from the junction is compared with the Kirsch solution,
and a Mohr-Coulomb failure index shows where the junction is weakest.
"""

# # Stresses around a multilateral well junction
#
# A multilateral well taps several reservoir zones from one main bore through branching legs. Where
# a leg leaves the main bore, the rock between the two openings carries the stress that the removed
# rock carried before, from two directions at once. This junction is the weak spot of the well: if
# the stress concentration there exceeds the shear strength of the rock, the wall breaks out.
#
# **Setup.** A rock block of 2 m × 2 m × 2 m; by the symmetry of the well about the vertical plane
# $y=0$ only the half $y\ge0$ is modelled. A vertical main bore of 100 mm diameter runs through the
# block; at mid height a lateral leg of 80 mm diameter branches off downwards at 45° in the plane
# $y=0$ and ends 0.8 m from the axis. Rock with $E=20$ GPa and $\nu=0.25$; in-situ stresses
# (compression) $\sigma_H=30$ MPa along $x$, $\sigma_h=20$ MPa along $y$, $\sigma_v=40$ MPa vertical;
# mud pressure $p_w=12$ MPa in both bores. The block comes from the CAD file
# `multilateral_well_stability.step`, written by `multilateral_well_stability_geometry.py` with the
# CAD objects of OneSimulation.
#
# | face | condition |
# |---|---|
# | `x_high`, `y_high`, `z_high` | traction $-\sigma_H$, $-\sigma_h$, $-\sigma_v$ normal to the face |
# | `x_low`, `z_low` | normal displacement 0 |
# | `symmetry` | $u_y=0$ |
# | `main_bore`, `lateral_bore` | mud pressure $p_w$ |
#
# **Reference.** Far from the junction and from the block faces the main bore is a hole in a plate
# under plane strain. With $\theta$ measured from the $x$ axis, the wall stresses (compression
# positive) are (Kirsch 1898; Fjaer et al., *Petroleum Related Rock Mechanics*, 2nd ed., Ch. 4)
#
# $$\sigma_\theta=\sigma_H+\sigma_h-2(\sigma_H-\sigma_h)\cos2\theta-p_w,\qquad \sigma_r=p_w,\qquad
#   \sigma_z=\sigma_v-2\nu(\sigma_H-\sigma_h)\cos2\theta .$$
#
# The Mohr-Coulomb failure index $F=\tfrac12(\sigma_1-\sigma_3)/\big(c\cos\varphi+\tfrac12(\sigma_1+\sigma_3)\sin\varphi\big)$,
# with cohesion $c=10$ MPa and friction angle $\varphi=30°$, reaches 1 where the rock fails in shear.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import LocalRefinement, TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import (
    FixedDisplacementComponents,
    LinearElasticity,
    Pressure,
    Traction,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the half block with its named faces; quadratic tetrahedra of 0.15 m, refined to
# 15 mm on both bore walls and bound to their curved surfaces.

rock = load("multilateral_well_stability.step")
faces = rock.boundaries
radius = 0.05
mesh = TetrahedralMesher(
    target_size=0.15,
    geometry_order=2,
    refinements=(
        LocalRefinement(region=faces.main_bore | faces.lateral_bore, target_size=0.015),
    ),
).generate(rock)

view = plots.Camera(elevation=20, azimuth=-110)
fig = plots.figure(1, 2, size=(15, 6.5))
plots.geometry(
    rock,
    selection={
        "main bore": faces.main_bore,
        "lateral leg": faces.lateral_bore,
        "symmetry plane": faces.symmetry,
    },
    labels=False,
    camera=view,
    title="Half rock block around the well",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{mesh.cell_count} curved tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# `LinearElasticity` with the in-situ stresses as tractions on the far faces and the mud pressure as
# `Pressure` on the bore walls, solved with a sparse direct solver.

stone = IsotropicSolid(
    density=2500.0,
    specific_heat_capacity=900.0,
    thermal_conductivity=2.5,
    youngs_modulus=20e9,
    poisson_ratio=0.25,
    thermal_expansion_coefficient=8e-6,
)
model = LinearElasticity(domain=rock.volume, material=stone)
major, minor, vertical, mud, poisson = 30e6, 20e6, 40e6, 12e6, 0.25
problem = Problem(
    name="Multilateral junction",
    geometry=rock,
    physics=(model,),
    boundary_conditions=(
        FixedDisplacementComponents(
            boundary=faces.x_low, physics=model, components=("x",)
        ),
        FixedDisplacementComponents(
            boundary=faces.symmetry, physics=model, components=("y",)
        ),
        FixedDisplacementComponents(
            boundary=faces.z_low, physics=model, components=("z",)
        ),
        Traction(boundary=faces.x_high, physics=model, value=(-major, 0.0, 0.0)),
        Traction(boundary=faces.y_high, physics=model, value=(0.0, -minor, 0.0)),
        Traction(boundary=faces.z_high, physics=model, value=(0.0, 0.0, -vertical)),
        Pressure(
            boundary=faces.main_bore | faces.lateral_bore, physics=model, value=mud
        ),
    ),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)
result = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary()
stress = result.field(model.stress)

# ## Comparison with the Kirsch solution
#
# The wall stresses of the main bore half-way between the junction and the top face, 0.5 m above
# the junction, for $0\le\theta\le180°$, against the Kirsch solution; then the failure index there and
# at the junction.

theta = np.linspace(0.0, np.pi, 37)
wall = np.column_stack(
    (
        radius * (1 + 2e-4) * np.cos(theta),
        radius * (1 + 2e-4) * np.sin(theta),
        np.full_like(theta, 0.5),
    )
)
tensors = lambda s: np.stack(
    [
        [s[:, 0], s[:, 3], s[:, 5]],
        [s[:, 3], s[:, 1], s[:, 4]],
        [s[:, 5], s[:, 4], s[:, 2]],
    ]
).transpose(2, 0, 1)
local = tensors(np.asarray(stress.evaluate(wall)))
hoop = -np.einsum(
    "ni,nij,nj->n",
    np.column_stack((-np.sin(theta), np.cos(theta), 0 * theta)),
    local,
    np.column_stack((-np.sin(theta), np.cos(theta), 0 * theta)),
)
radial = -np.einsum(
    "ni,nij,nj->n",
    np.column_stack((np.cos(theta), np.sin(theta), 0 * theta)),
    local,
    np.column_stack((np.cos(theta), np.sin(theta), 0 * theta)),
)
axial = -local[:, 2, 2]
kirsch = {
    "hoop": major + minor - 2 * (major - minor) * np.cos(2 * theta) - mud,
    "radial": mud + 0 * theta,
    "axial": vertical - 2 * poisson * (major - minor) * np.cos(2 * theta),
}
cohesion, friction = 10e6, np.radians(30.0)
failure = lambda s, position: (
    lambda e: 0.5
    * (e[:, 2] - e[:, 0])
    / (cohesion * np.cos(friction) + 0.5 * (e[:, 2] + e[:, 0]) * np.sin(friction))
)(np.linalg.eigvalsh(-tensors(np.asarray(s))))
for name, values in (("hoop", hoop), ("radial", radial), ("axial", axial)):
    print(
        f"{name:6s} stress: largest deviation from Kirsch {np.abs(values - kirsch[name]).max() / 1e6:.2f} MPa "
        f"({100 * np.abs(values - kirsch[name]).max() / kirsch['hoop'].max():.2f} % of the peak hoop stress)"
    )
print(
    f"hoop stress at θ = 0: {hoop[0] / 1e6:.2f} MPa (Kirsch {kirsch['hoop'][0] / 1e6:.2f}); at θ = 90°: {hoop[18] / 1e6:.2f} MPa (Kirsch {kirsch['hoop'][18] / 1e6:.2f})"
)
along_main = failure(np.asarray(stress.evaluate(wall)), None)
far_field = failure(
    np.stack(
        [
            -kirsch["radial"],
            -kirsch["hoop"],
            -kirsch["axial"],
            0 * theta,
            0 * theta,
            0 * theta,
        ],
        axis=1,
    ),
    None,
)
walls = stress.sample(along=faces.main_bore | faces.lateral_bore, subdivisions=2)
near = np.linalg.norm(walls.positions, axis=1) < 5 * radius
index = failure(walls.values[near], walls.positions[near])
worst = walls.positions[near][int(np.argmax(index))]
print(
    f"failure index on the main bore 0.5 m above the junction: {along_main.max():.3f} at θ = {np.degrees(theta[np.argmax(along_main)]):.0f}° "
    f"(Kirsch: {far_field.max():.3f})"
)
print(
    f"within 5 bore radii of the junction centre: largest failure index {index.max():.3f} at ({worst[0]:.3f}, {worst[1]:.3f}, {worst[2]:.3f}) m; "
    f"{100 * np.mean(index > 1):.1f} % of the wall samples there exceed 1"
)

# ## Figure
#
# Left the wall stresses of the main bore against the Kirsch solution; right the failure index on the
# walls of both bores. Away from the junction the main bore stays at the plane-strain value of about
# 0.88; the highest values, above 1, sit on the flank of the lateral leg just below the junction,
# where the leg cuts into the most loaded side of the main bore. The flat closed end of the lateral
# leg, an idealisation with a sharp edge, is left out of the junction statistics.

fig = plots.figure(1, 2, size=(17, 6.5))
plots.line(
    points={
        name: (np.degrees(theta), values / 1e6)
        for name, values in (("hoop", hoop), ("radial", radial), ("axial", axial))
    },
    reference={
        f"{name}, Kirsch": (np.degrees(theta), values / 1e6)
        for name, values in kirsch.items()
    },
    xlabel="θ from the x axis [°]",
    ylabel="stress, compression positive [MPa]",
    title="Main bore wall 0.5 m above the junction",
    legend="center right",
    ax=fig[0],
)
plots.surface(
    stress,
    boundary=faces.main_bore | faces.lateral_bore,
    component=failure,
    subdivisions=3,
    limits=(0.5, 1.25),
    camera=plots.Camera(elevation=15, azimuth=55),
    label="Mohr-Coulomb failure index",
    title="Failure index on the bore walls",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Raise the mud weight until the failure index stays below 1 everywhere, turn the lateral towards
# the direction of the smaller horizontal stress, or replace the elastic rock by a Mohr-Coulomb
# plastic material to follow the breakout itself.
