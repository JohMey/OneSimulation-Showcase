"""OneSimulation example: homogenized thermal properties of a unidirectional fibre composite.
Requires the installed, activated OneSimulation runtime.

The unit cell of a square array of glass fibres in epoxy is read from a STEP file. Two steady heat
conduction problems on the cell give the effective conductivity across and along the fibres; the
effective density and heat capacity follow from the volume and mass fractions. The transverse
conductivity is compared with the series solution of Rayleigh for a square array and with the
Voigt-Reuss and Hashin-Shtrikman bounds, on three meshes.
"""

# # Homogenized thermal properties of a fibre composite
#
# A unidirectional composite looks homogeneous at the scale of a part, but conducts heat
# differently along and across its fibres. Its effective properties follow from one unit cell of
# the fibre arrangement. Glass fibres ($k_f=1.05$ W/(m K), $\rho_f=2550$ kg/m³,
# $c_f=840$ J/(kg K)) fill $\varphi=50$ % of an epoxy matrix ($k_m=0.19$ W/(m K), $\rho_m=1200$ kg/m³,
# $c_m=1100$ J/(kg K)) in a square array with a spacing of 10 µm.
#
# The effective conductivity $k_\text{eff}$ relates the mean heat flux to the mean temperature
# gradient. A cell with mirror symmetry about its mid-planes has the same solution for a temperature
# jump across it with insulated sides as for the periodic cell problem, so
#
# $$k_{xx}=\frac{Q_x\,L_x}{A_x\,\Delta T},\qquad k_{zz}=\frac{Q_z\,L_z}{A_z\,\Delta T},$$
#
# with $Q$ the heat flow through the cell for a temperature difference $\Delta T$ between opposite
# faces. Density and heat capacity per volume are exact volume averages:
# $\rho_\text{eff}=\varphi\rho_f+(1-\varphi)\rho_m$ and
# $c_\text{eff}=(\varphi\rho_f c_f+(1-\varphi)\rho_m c_m)/\rho_\text{eff}$.
#
# | cell problem | faces with $T=1$ K and $T=0$ | insulated |
# |---|---|---|
# | across the fibres | `left`, `right` (x) | `sides` (y), `bottom`, `top` (z) |
# | along the fibres | `bottom`, `top` (z) | `left`, `right`, `sides` |
#
# The cell and its fibre come from the CAD files `fiber_composite_thermal_homogenization.step` and
# `fiber_composite_thermal_homogenization_fibre.step`; the script `fiber_composite_thermal_homogenization_geometry.py` builds
# it with the CAD objects of OneSimulation.
#
# **Reference.** Along the fibres the parallel rule $k_{zz}=\varphi k_f+(1-\varphi)k_m$ is exact.
# Across them, Rayleigh's multipole series for a square array of cylinders, in the form of Perrins,
# McKenzie and McPhedran (*Proc. R. Soc. Lond. A* 369, 1979, 207–225), with
# $\beta=(k_f-k_m)/(k_f+k_m)$:
#
# $$\frac{k_{xx}}{k_m}=1+\frac{2\beta\varphi}{1-\beta\varphi-0.305827\,\beta^2\varphi^4\big/\big(1-1.402958\,\beta^2\varphi^8\big)-0.013362\,\beta^2\varphi^8}.$$
#
# It must lie between the Reuss (series) and Voigt (parallel) values and between the two-dimensional
# Hashin-Shtrikman bounds.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid, regionwise
from onesim.meshing import TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the cell with its named faces and the fibre, which the mesher embeds as a material
# region. Three meshes of quadratic tetrahedra, 1.4, 1 and 0.8 µm, check the convergence.

cell, fibre_body = load("fiber_composite_thermal_homogenization.step"), load(
    "fiber_composite_thermal_homogenization_fibre.step"
).named("fibre")
faces = cell.boundaries
side, depth = cell.size[0], cell.size[2]
meshes = [
    TetrahedralMesher(target_size=size, embedded_regions=(fibre_body,)).generate(cell)
    for size in (1.4e-6, 1.0e-6, 0.8e-6)
]

fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    [cell.named("matrix"), fibre_body],
    labels=False,
    camera=plots.Camera(elevation=35, azimuth=-55),
    length_unit="um",
    title="Unit cell: fibre in matrix",
    ax=fig[0],
)
plots.mesh(
    meshes[1],
    groups=("all",),
    legend=False,
    length_unit="um",
    title=f"Middle mesh: {meshes[1].cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Cell problems
#
# `regionwise` gives each part its conductivity. Each cell problem holds two opposite faces at 1 K
# and 0 K and insulates the others; the heat flow through the cold face, `reaction`, gives the
# effective conductivity.

fibre, matrix, fraction = (
    {"k": 1.05, "rho": 2550.0, "c": 840.0},
    {"k": 0.19, "rho": 1200.0, "c": 1100.0},
    0.5,
)
material = IsotropicSolid(
    density=1.0,
    specific_heat_capacity=1.0,
    youngs_modulus=1.0,
    poisson_ratio=0.3,
    thermal_expansion_coefficient=0.0,
    thermal_conductivity=regionwise(
        {"fibre": fibre["k"]}, default=matrix["k"], unit="W/m/K"
    ),
)
directions = {
    "across": (faces.left, faces.right, side),
    "along": (faces.bottom, faces.top, depth),
}
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)
conductivity, meshed = {}, []
for direction, (hot, cold, length) in directions.items():
    model = HeatConduction(domain=cell.domain(), material=material, heating=0.0)
    problem = Problem(
        name=f"Cell problem {direction} the fibres",
        geometry=cell,
        physics=(model,),
        boundary_conditions=(
            FixedTemperature(boundary=hot, physics=model, value=1.0),
            FixedTemperature(boundary=cold, physics=model, value=0.0),
            HeatFlux(boundary=faces.all - hot - cold, physics=model, value=0.0),
        ),
    )
    area = side * side * depth / length
    temperatures = [
        Simulation(problem=problem, mesh=mesh, numerics=numerics)
        .solve_stationary()
        .field(model.temperature)
        for mesh in meshes
    ]
    conductivity[direction] = [
        temperature.reaction(hot) * length / area for temperature in temperatures
    ]
    meshed = [
        temperature.integrate(
            region=fibre_body, component=lambda T, position: np.ones_like(T)
        ).value
        / (side * side * depth)
        for temperature in temperatures
    ]

# ## Results
#
# The effective conductivities on the three meshes against the references, then density and heat
# capacity. The faceted fibre of a mesh holds slightly less than half of the cell; the references
# are evaluated at the fibre fraction each mesh holds (the volume of its fibre cells).

kf, km = fibre["k"], matrix["k"]
beta = (kf - km) / (kf + km)
rayleigh = km * (
    1
    + 2
    * beta
    * fraction
    / (
        1
        - beta * fraction
        - 0.305827 * beta**2 * fraction**4 / (1 - 1.402958 * beta**2 * fraction**8)
        - 0.013362 * beta**2 * fraction**8
    )
)
voigt, reuss = fraction * kf + (1 - fraction) * km, 1 / (
    fraction / kf + (1 - fraction) / km
)
lower = km + fraction / (1 / (kf - km) + (1 - fraction) / (2 * km))
upper = kf + (1 - fraction) / (1 / (km - kf) + fraction / (2 * kf))
print(
    f"bounds across the fibres: Reuss {reuss:.4f} < Hashin-Shtrikman {lower:.4f} ... {upper:.4f} < Voigt {voigt:.4f} W/(m K)"
)
series = lambda f: km * (
    1
    + 2
    * beta
    * f
    / (
        1
        - beta * f
        - 0.305827 * beta**2 * f**4 / (1 - 1.402958 * beta**2 * f**8)
        - 0.013362 * beta**2 * f**8
    )
)
print(
    "tetrahedra   fibre fraction of the mesh   k_xx [W/(m K)]   Rayleigh at that fraction   deviation   k_zz [W/(m K)]   parallel rule   deviation"
)
for mesh, f, across, along in zip(
    meshes, meshed, conductivity["across"], conductivity["along"]
):
    parallel = f * kf + (1 - f) * km
    print(
        f"{mesh.cell_count:10d} {100 * f:27.3f} % {across:15.5f} {series(f):26.5f} {100 * (across / series(f) - 1):+10.3f} % "
        f"{along:15.5f} {parallel:15.5f} {100 * (along / parallel - 1):+10.4f} %"
    )
density = fraction * fibre["rho"] + (1 - fraction) * matrix["rho"]
capacity = (
    fraction * fibre["rho"] * fibre["c"] + (1 - fraction) * matrix["rho"] * matrix["c"]
) / density
print(
    f"effective density {density:.1f} kg/m³, effective specific heat capacity {capacity:.1f} J/(kg K)"
)
print(
    f"anisotropy k_zz / k_xx = {conductivity['along'][-1] / conductivity['across'][-1]:.3f}"
)

# ## Figure
#
# The transverse conductivity of the three meshes at the fibre fraction each holds, on the curve of
# the Rayleigh series.

fractions = np.linspace(0.488, 0.5, 61)
plots.line(
    x=100 * fractions,
    y={"Rayleigh series": series(fractions)},
    points={
        "simulation on three meshes": (
            [100 * f for f in meshed],
            conductivity["across"],
        )
    },
    xlabel="fibre volume fraction [%]",
    ylabel="k_xx [W/(m K)]",
    title="Effective conductivity across the fibres",
    legend="lower right",
)
plots.show()

# ## Going further
#
# Change the fibre fraction in `fiber_composite_thermal_homogenization_geometry.py` (up to the
# touching limit of 78.5 %) and follow the Rayleigh curve, or replace the fibre by an anisotropic
# carbon fibre with a conductivity tensor.
