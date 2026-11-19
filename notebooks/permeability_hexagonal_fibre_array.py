"""OneSimulation example: permeability of a hexagonal array of fibres from a 3D periodic unit cell.
Requires the installed, activated OneSimulation runtime.

The periodic unit cell of a hexagonal fibre array is read from STEP files for seven solid fractions
and meshed with tetrahedra whose opposite faces match. The 3D creeping flow across the fibres is
solved with Taylor-Hood elements and periodic conditions in all three directions. Porosity,
tortuosity and the Darcy permeability follow; the permeability is compared with Drummond and Tahir
(dilute) and Gebart (dense), its in-plane isotropy is checked, with a mesh study.
"""

# # Permeability of a hexagonal array of fibres
#
# Unidirectional composite preforms, hollow-fibre modules and fibrous filters are often close to a
# hexagonal packing of parallel fibres. For the flow of resin or fluid across the fibres a model of
# the part needs the porosity $\varepsilon$ and the permeability $K$ of Darcy's law. Both follow
# from one periodic cell of the microstructure (representative volume element): a uniform pressure
# gradient $G$, applied as a body force, drives the creeping flow, and $K=\mu\langle u\rangle/G$ with
# the superficial velocity $\langle u\rangle$. The tortuosity $\tau=\langle|\mathbf u|\rangle/\langle u\rangle$
# measures how far the paths wind around the fibres.
#
# The cell holds a fibre on its axis and a quarter fibre on each long edge, pitch $L=1$ mm, solid
# fraction $\phi=1-\varepsilon=2\pi a^2/(\sqrt3L^2)$. It comes from the CAD files
# `permeability_hexagonal_fibre_array.step` ($\phi=0.4$) and `..._10.step` to `..._70.step`, written
# by `permeability_hexagonal_fibre_array_geometry.py`, as prisms 0.1 mm long along the fibres.
#
# | face | condition |
# |---|---|
# | `x_min` / `x_max` | periodic pair |
# | `y_min` / `y_max` | periodic pair |
# | `z_min` / `z_max` (along the fibres) | periodic pair |
# | `fibre.outer_wall` | no slip |
#
# **Reference.** A hexagonal array is isotropic across the fibres: the permeabilities along $x$ and
# $y$ agree. J. E. Drummond and M. I. Tahir, "Laminar viscous flow through regular arrays of
# parallel solid cylinders", *Int. J. Multiphase Flow* 10, 515-540 (1984), for dilute arrays:
#
# $$\frac{K}{a^2}=\frac{1}{8\phi}\left(-\ln\phi-1.498+2\phi-\frac{\phi^2}{2}-0.739\,\phi^4
#   +\frac{2.534\,\phi^5}{1+1.2758\,\phi}\right);$$
#
# B. R. Gebart, *J. Compos. Mater.* 26, 1100-1133 (1992), for dense arrays:
# $K/a^2=\frac{16}{9\pi\sqrt6}\left(\sqrt{\phi_\max/\phi}-1\right)^{5/2}$ with $\phi_\max=\pi/(2\sqrt3)$.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import NewtonianFluid
from onesim.meshing import LocalRefinement, PeriodicPair, TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, TaylorHood
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.hydrodynamics import IncompressibleNavierStokes, NoSlipWall
from onesim.results.derived import derived_field
from onesim.simulation import PeriodicBoundary, Problem, Simulation
from onesim.simulation.field_constraints import MeanValue
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# Tetrahedra of $L/20$, on the fibres at most $L/30$ and a fifth of the narrowest gap; the three
# pairs of opposite faces carry the same triangles. The mesh study at $\phi=0.4$ scales all sizes by
# 1.5.

fractions = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
cells = {
    fraction: load(f"permeability_hexagonal_fibre_array_{round(100 * fraction)}.step")
    for fraction in fractions
}
pitch, height, length = cells[0.40].size
radius = {
    fraction: pitch * np.sqrt(fraction * np.sqrt(3) / (2 * np.pi))
    for fraction in fractions
}
pairs = (("x_min", "x_max"), ("y_min", "y_max"), ("z_min", "z_max"))
meshes = {
    (fraction, scale): TetrahedralMesher(
        target_size=scale * pitch / 20,
        refinements=(
            LocalRefinement(
                region=cells[fraction].boundaries["fibre.outer_wall"],
                target_size=scale * min(pitch / 30, (pitch - 2 * radius[fraction]) / 5),
            ),
        ),
        periodic_pairs=tuple(
            PeriodicPair(
                first=cells[fraction].boundaries[low],
                second=cells[fraction].boundaries[high],
            )
            for low, high in pairs
        ),
    ).generate(cells[fraction])
    for fraction, scale in [(fraction, 1.0) for fraction in fractions] + [(0.40, 1.5)]
}

cell = load("permeability_hexagonal_fibre_array.step")
view = plots.Camera(elevation=45, azimuth=-60)
fig = plots.figure(1, 2, size=(13, 6))
plots.geometry(
    cell,
    selection={"fibres": cell.boundaries["fibre.outer_wall"]},
    labels=False,
    camera=view,
    title="Periodic cell, solid fraction 0.4",
    ax=fig[0],
)
plots.mesh(
    meshes[0.40, 1.0],
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{meshes[0.40, 1.0].cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Water and a pressure gradient of 1 Pa/m across the fibres, along $x$ for all solid fractions, and
# along $y$ and on the coarser mesh for $\phi=0.4$.

water, gradient = NewtonianFluid(density=1000.0, dynamic_viscosity=1.0e-3), 1.0
cases = [(fraction, "x", 1.0) for fraction in fractions] + [
    (0.40, "y", 1.0),
    (0.40, "x", 1.5),
]
flows = {}
for fraction, axis, scale in cases:
    faces = cells[fraction].boundaries
    model = IncompressibleNavierStokes(
        domain=cells[fraction].volume,
        material=water,
        body_force=(gradient, 0.0, 0.0) if axis == "x" else (0.0, gradient, 0.0),
    )
    problem = Problem(
        name="Hexagonal fibre array cell",
        geometry=cells[fraction],
        physics=(model,),
        boundary_conditions=(
            NoSlipWall(boundary=faces["fibre.outer_wall"], physics=model),
            *[
                PeriodicBoundary(
                    pair=PeriodicPair(first=faces[low], second=faces[high]),
                    physics=model,
                )
                for low, high in pairs
            ],
        ),
        constraints=(MeanValue(model.pressure, 0.0),),
    )
    numerics = Numerics(
        spatial=FiniteElement(element=TaylorHood()),
        initial_guess={model.velocity: (0.0, 0.0, 0.0)},
        nonlinear_solver=Newton(
            linear_solver=SparseDirect(),
            absolute_tolerance=1e-20,
            relative_tolerance=1e-10,
        ),
    )
    flows[fraction, axis, scale] = (
        Simulation(problem=problem, mesh=meshes[fraction, scale], numerics=numerics)
        .solve_stationary()
        .field(model.velocity)
    )

# ## Results
#
# Porosity and tortuosity are integrals over the fluid; the superficial velocity is the volume
# integral of the velocity over the cell volume.

volume = pitch * height * length
permeability = {
    key: water.dynamic_viscosity
    * velocity.integrate().value["xy".index(key[1])]
    / volume
    / gradient
    for key, velocity in flows.items()
}
porosity = {
    fraction: derived_field(
        flows[fraction, "x", 1.0],
        lambda u, points: np.ones(len(points)),
        name="one",
        unit="1",
    )
    .integrate()
    .value
    / volume
    for fraction in fractions
}
tortuosity = {
    fraction: derived_field(
        flows[fraction, "x", 1.0],
        lambda u, points: np.linalg.norm(u, axis=1),
        name="speed",
        unit="m/s",
    )
    .integrate()
    .value
    / flows[fraction, "x", 1.0].integrate().value[0]
    for fraction in fractions
}
drummond_tahir = lambda phi: (
    -np.log(phi)
    - 1.498
    + 2 * phi
    - phi**2 / 2
    - 0.739 * phi**4
    + 2.534 * phi**5 / (1 + 1.2758 * phi)
) / (8 * phi)
gebart = (
    lambda phi: 16
    / (9 * np.pi * np.sqrt(6))
    * (np.sqrt(np.pi / (2 * np.sqrt(3) * phi)) - 1) ** 2.5
)

print(
    " φ     porosity  exact    K/a²     Drummond-Tahir  deviation   Gebart   deviation   tortuosity"
)
for fraction in fractions:
    scaled = permeability[fraction, "x", 1.0] / radius[fraction] ** 2
    print(
        f"{fraction:4.2f} {porosity[fraction]:9.4f} {1 - fraction:7.4f} {scaled:9.4f} {drummond_tahir(fraction):12.4f} {100 * (scaled / drummond_tahir(fraction) - 1):+9.1f} %"
        f" {gebart(fraction):8.4f} {100 * (scaled / gebart(fraction) - 1):+8.1f} % {tortuosity[fraction]:10.3f}"
    )
print(
    f"\nisotropy at φ = 0.4: K_y / K_x - 1 = {permeability[0.40, 'y', 1.0] / permeability[0.40, 'x', 1.0] - 1:+.1e}"
)
print(
    f"mesh study at φ = 0.4: K/a² = {permeability[0.40, 'x', 1.5] / radius[0.40]**2:.5f} ({meshes[0.40, 1.5].cell_count} tetrahedra), "
    f"{permeability[0.40, 'x', 1.0] / radius[0.40]**2:.5f} ({meshes[0.40, 1.0].cell_count} tetrahedra)"
)

# ## Figure
#
# The flow speed on the cell faces at $\phi=0.4$ for the flow along $x$ and along $y$, and the
# permeability over the solid fraction.

phi = np.linspace(0.05, 0.88, 200)
fig = plots.figure(layout=[["x", "y", "permeability"]], size=(18, 6))
for axis in ("x", "y"):
    plots.surface(
        flows[0.40, axis, 1.0],
        component=lambda u, position: 1e6 * np.linalg.norm(u, axis=1),
        limits="robust",
        contours=6,
        subdivisions=2,
        camera=view,
        label="speed [µm/s]",
        title=f"Flow along {axis}, φ = 0.4",
        ax=fig[axis],
    )
plots.line(
    reference={
        "Drummond & Tahir (1984), dilute": (
            phi[phi < 0.45],
            drummond_tahir(phi[phi < 0.45]),
        ),
        "Gebart (1992), dense": (phi[phi > 0.25], gebart(phi[phi > 0.25])),
    },
    points={
        "simulation": (
            list(fractions),
            [permeability[f, "x", 1.0] / radius[f] ** 2 for f in fractions],
        )
    },
    scale="logy",
    ylimits=(1e-4, 10),
    xlabel="solid fraction φ",
    ylabel="permeability K / a²",
    title="Permeability across the fibres",
    legend="upper right",
    ax=fig["permeability"],
)
plots.show()

# ## Going further
#
# Replace the regular array by a random arrangement of fibres in a larger cell in
# `permeability_hexagonal_fibre_array_geometry.py`: the permeability then scatters around the value
# of the regular array, and averaging over several cells shows how large a representative volume
# must be. A force along $z$ gives the permeability along the fibres.
