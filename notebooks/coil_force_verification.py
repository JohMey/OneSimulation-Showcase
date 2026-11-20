"""OneSimulation example: force between two coaxial coils - a verification series (3D magnetostatics).
Requires the installed, activated OneSimulation runtime.

The force between two current-carrying coils is computed from the 3D field on second-order edge
elements, as the Lorentz force on the windings, for three coil distances and two meshes. Every
force is compared with the exact force from Maxwell's mutual inductance of coaxial loops,
integrated over both winding cross-sections, and the forces on the two coils are checked against
Newton's third law.
"""

# # Force between two coaxial coils: a verification series
#
# Two coaxial coils of mean radius $R=50$ mm with a square winding cross-section of 10 x 10 mm and
# 100 turns each carry 1 A in the same direction and attract each other. The force on a winding is
# the Lorentz force
#
# $$\mathbf F=\int_V\mathbf J\times\mathbf B\,dV,\qquad F_z=-J\int_V B_r\,dV$$
#
# for the uniform azimuthal current density $J=NI/A$. The field solves
# $\nabla\times(\mu_0^{-1}\nabla\times\mathbf A)=\mathbf J$ with second-order Nédélec elements and a
# Coulomb gauge; one quarter is computed, and the windings are cut by its two symmetry planes. The
# coil distance is varied from $R/2$ to $2R$, and the middle distance is also solved on a finer mesh.
#
# The air, a quarter of a ball of 0.4 m radius, comes from the CAD file
# `coil_force_verification.step` (face groups `planes` and `outer`), the quarter of the lower winding
# from `coil_force_verification_lower.step` and the upper winding for the three distances from
# `..._upper_25mm.step`, `..._upper_50mm.step` and `..._upper_100mm.step`. The script
# `coil_force_verification_geometry.py` builds them with the CAD objects of OneSimulation.
#
# | face | condition |
# |---|---|
# | `planes` (through the axis), `outer` | $\mathbf n\times\mathbf A=0$ |
#
# **Reference.** Two coaxial loops of radii $a$, $b$ at the axial distance $z$ attract with
# $F_z=I_1I_2\,\partial M/\partial z$, with Maxwell's mutual inductance
# $M=\mu_0\sqrt{ab}\,[(2/k-k)K(k)-(2/k)E(k)]$, $k^2=4ab/((a+b)^2+z^2)$ (Smythe, *Static and Dynamic
# Electricity*, 3rd ed., Sec. 7.13). Integrating it over both rectangular cross-sections with
# Gauss-Legendre points gives the exact force between the windings.

import numpy as np
from onesim.cad.d2 import Plane
from onesim.cad.d3 import AnnularCylinder
from onesim.cad.exchange import load
from onesim.materials import VacuumPermeability
from onesim.meshing import LocalRefinement, TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import GMRES, AuxiliarySpaceMaxwell
from onesim.physics.magnetostatics import (
    CoulombGauge,
    MagneticInsulation,
    MagneticVectorPotential,
    Magnetostatics,
    SolenoidCurrent,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme
from scipy.special import ellipe, ellipk

theme.set()

# ## Geometry and mesh
#
# Each winding is declared whole, as `SolenoidCurrent` with its own annular support; with
# `clip_regions=True` the mesher keeps the quarter of it that lies inside the air region as an
# embedded region. The cells are 4 mm (2.8 mm on the finer mesh) in and around the windings and grow
# towards the outer boundary.

air = load("coil_force_verification.step")
faces = air.boundaries
radius, width, turns, current = 0.05, 0.01, 100, 1.0
cases = [(0.025, 0.004), (0.05, 0.004), (0.05, 0.0028), (0.10, 0.004)]
lower = load("coil_force_verification_lower.step")
windings = [
    (lower, load(f"coil_force_verification_upper_{round(1e3 * d)}mm.step"))
    for d, _ in cases
]
coils = [
    tuple(
        SolenoidCurrent(
            inner_radius=radius - width / 2,
            outer_radius=radius + width / 2,
            length=width,
            turns=turns,
            current=current,
            center=(0.0, 0.0, z),
            support_factory=AnnularCylinder,
        )
        for z in (0.0, d)
    )
    for d, _ in cases
]
meshes = [
    TetrahedralMesher(
        target_size=0.06,
        embedded_regions=tuple(coil.support for coil in pair),
        clip_regions=True,
        refinements=tuple(
            LocalRefinement(region=coil.support, target_size=h) for coil in pair
        ),
        size_field=lambda points, d=d, h=h: np.clip(
            h
            + 0.6
            * np.maximum(
                np.hypot(
                    np.hypot(points[:, 0], points[:, 1]) - radius, points[:, 2] - d / 2
                )
                - d / 2
                - width,
                0,
            ),
            h,
            0.06,
        ),
    ).generate(air)
    for (d, h), pair in zip(cases, coils)
]

view = plots.Camera(elevation=20, azimuth=-60)
fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    [coil.named(name) for coil, name in zip(windings[1], ("lower coil", "upper coil"))],
    camera=view,
    title="The two windings, 50 mm apart",
    ax=fig[0],
)
plots.mesh(
    meshes[1],
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{meshes[1].cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Each winding carries $NI=100$ ampere-turns spread uniformly over its cross-section
# (`SolenoidCurrent`, with the same dimensions as the STEP windings). The linear systems are solved
# with GMRES and the auxiliary-space preconditioner for edge elements (`AuxiliarySpaceMaxwell`: a
# discrete gradient and a nodal interpolation with algebraic multigrid on the nodal spaces).

numerics = Numerics(
    spatial=FiniteElement(element=Nedelec(order=2)),
    linear_solver=GMRES(preconditioner=AuxiliarySpaceMaxwell()),
)
models = [
    Magnetostatics(
        domain=air.volume,
        permeability=VacuumPermeability(),
        formulation=MagneticVectorPotential(gauge=CoulombGauge()),
        current_sources=pair,
    )
    for pair in coils
]
fields = [
    Simulation(
        problem=Problem(
            name="Two coils",
            geometry=air,
            physics=(model,),
            boundary_conditions=(
                MagneticInsulation(boundary=faces.planes | faces.outer, physics=model),
            ),
        ),
        mesh=mesh,
        numerics=numerics,
    )
    .solve_stationary()
    .field(model.magnetic_flux_density)
    for model, mesh in zip(models, meshes)
]

# ## Results
#
# The force on each winding, $F_z=-J\int B_r\,dV$, integrated over the cells of the winding (its
# embedded region, `region=coil.support`), four times the quarter, against the exact force.

mu0 = float(VacuumPermeability().evaluate())
density = turns * current / width**2
radial = lambda B, p: (B[:, 0] * p[:, 0] + B[:, 1] * p[:, 1]) / np.hypot(
    p[:, 0], p[:, 1]
)
forces = [
    [
        -4 * density * field.integrate(region=coil.support, component=radial).value
        for coil in pair
    ]
    for field, pair in zip(fields, coils)
]
nodes, weights = np.polynomial.legendre.leggauss(8)
r, z = np.meshgrid(radius + width / 2 * nodes, width / 2 * nodes, indexing="ij")
w = (np.outer(weights, weights) * (width / 2) ** 2 * density).ravel()
mutual = lambda a, b, s: (
    lambda k: mu0
    * np.sqrt(a * b)
    * ((2 / k - k) * ellipk(k * k) - 2 / k * ellipe(k * k))
)(np.sqrt(4 * a * b / ((a + b) ** 2 + s**2)))
exact = lambda d: np.sum(
    w[:, None]
    * w[None, :]
    * (
        mutual(
            r.ravel()[:, None],
            r.ravel()[None, :],
            d + z.ravel()[None, :] - z.ravel()[:, None] + 1e-9,
        )
        - mutual(
            r.ravel()[:, None],
            r.ravel()[None, :],
            d + z.ravel()[None, :] - z.ravel()[:, None] - 1e-9,
        )
    )
    / 2e-9
)
print(
    "distance   cell size   tetrahedra   force on the upper coil [mN]   exact [mN]   deviation   (F_lower + F_upper) / F_upper"
)
for (d, h), mesh, (f_lower, f_upper) in zip(cases, meshes, forces):
    print(
        f"{1e3 * d:6.0f} mm {1e3 * h:8.1f} mm {mesh.cell_count:12d} {1e3 * f_upper:22.4f} {1e3 * exact(d):17.4f} {100 * (f_upper / exact(d) - 1):+10.3f} %"
        f" {(f_lower + f_upper) / f_upper:+16.1e}"
    )
thin = (
    lambda d: current**2
    * turns**2
    * (mutual(radius, radius, d + 1e-9) - mutual(radius, radius, d - 1e-9))
    / 2e-9
)
print(
    "thin-loop formula for comparison: "
    + ", ".join(
        f"{1e3 * thin(d):.4f} mN at {1e3 * d:.0f} mm" for d in (0.025, 0.05, 0.10)
    )
)

# ## Figure
#
# The field strength through the axis for the middle distance and the force against the distance.

wide = np.linspace(0.012, 0.15, 120)
fig = plots.figure(1, 2, size=(16, 6))
plots.section(
    fields[1],
    plane=Plane(origin=(0, 1e-6, 0), normal=(0, 1, 0)),
    component=lambda B, p: 1e3 * np.linalg.norm(B, axis=1),
    limits=(0, 3),
    contours=13,
    subdivisions=2,
    symmetry={"planes": [((1, 0, 0), (0, 0, 0))]},
    label="|B| [mT]",
    xlimits=(-0.12, 0.12),
    ylimits=(-0.07, 0.12),
    xlabel="x [m]",
    ylabel="z [m]",
    title="Field of the two coils, 50 mm apart",
    ax=fig[0],
)
plots.line(
    1e3 * wide,
    {
        "exact, windings": [-1e3 * exact(d) for d in wide],
        "thin loops": [-1e3 * thin(d) for d in wide],
    },
    points={
        "simulation, upper coil": (
            [1e3 * d for d, _ in cases],
            [-1e3 * f[1] for f in forces],
        ),
        "simulation, lower coil (sign reversed)": (
            [1e3 * d for d, _ in cases],
            [1e3 * f[0] for f in forces],
        ),
    },
    scale="logy",
    xlabel="coil distance [mm]",
    ylabel="attractive force [mN]",
    title="Force between the coils",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Reverse one current to make the force repulsive, tilt the upper winding in
# `coil_force_verification_geometry.py` (half model) and compute the torque, or put an iron plate
# between the coils and watch the force change.
