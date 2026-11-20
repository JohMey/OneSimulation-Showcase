"""OneSimulation example: effective thermal conductivity of a porous solid and its averaging models.
Requires the installed, activated OneSimulation runtime.

A porous medium in local thermal equilibrium conducts heat with one effective conductivity that
combines solid and pore fluid. The classical averaging models — volume average, reciprocal average,
power law, Maxwell-Eucken — give it from the porosity alone and differ widely. Here the unit cells of
a solid with water-filled spherical pores, read from STEP files for four porosities, are solved in 3D
and placed among the models and the Hashin-Shtrikman bounds.
"""

# # Effective thermal conductivity of a porous solid
#
# A porous medium of a solid with conductivity $k_s$ whose pores of volume fraction $\varepsilon$
# (the porosity) are filled with a fluid of conductivity $k_f$ is described, under local thermal
# equilibrium, by one effective conductivity $k_\text{eff}$. Common averaging models are
#
# | model | $k_\text{eff}$ |
# |---|---|
# | volume average (parallel layers) | $\varepsilon k_f+(1-\varepsilon)k_s$ |
# | reciprocal average (series layers) | $\big(\varepsilon/k_f+(1-\varepsilon)/k_s\big)^{-1}$ |
# | power law (geometric mean) | $k_f^{\,\varepsilon}\,k_s^{\,1-\varepsilon}$ |
# | Maxwell-Eucken, pores dispersed in the solid | $k_s\,\dfrac{2k_s+k_f-2\varepsilon(k_s-k_f)}{2k_s+k_f+\varepsilon(k_s-k_f)}$ |
#
# Volume and reciprocal average are the absolute bounds; Maxwell-Eucken equals the upper
# Hashin-Shtrikman bound of an isotropic two-phase material, with the solid as the connected phase.
#
# Which model applies depends on the microstructure. For separate pores in a connected solid it is
# Maxwell-Eucken. To check this, unit cells of a simple cubic array of spherical pores are solved: a
# ceramic of $k_s=10$ W/(m K) with water in the pores, $k_f=0.6$ W/(m K), at porosities of 10, 20, 30
# and 45 %. A cell with mirror symmetry about its mid-planes gives its effective conductivity from a
# temperature difference between two opposite faces with the other faces insulated,
# $k_\text{eff}=Q\,L/(A\,\Delta T)$, the same as the periodic cell problem.
#
# | face | condition |
# |---|---|
# | `hot`, $x=0$ | $T=1$ K |
# | `cold`, $x=1$ mm | $T=0$ |
# | all other faces | insulated |
#
# The cell comes from the CAD file `porous_effective_conductivity_models.step`, the pores from
# `porous_effective_conductivity_models_pore_<percent>.step`; the script `porous_effective_conductivity_models_geometry.py`
# builds them with the CAD objects of OneSimulation.
#
# **Reference.** The averaging models above (Nield and Bejan, *Convection in Porous Media*, 5th ed.,
# 2017, section 2.2.1) and the Hashin-Shtrikman bounds (Hashin and Shtrikman, *J. Appl. Phys.* 33,
# 1962, 3125). For dilute pores the cell must approach Maxwell-Eucken; at high porosity the cubic
# array lies slightly below it, as the pores start to interact.

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
# `load` returns the cell with its named faces and each pore, which the mesher embeds as a material
# region. Quadratic tetrahedra of 0.1 mm resolve the spherical pores; the faceted pores hold a
# little less volume than the spheres, so each cell is compared at the porosity its mesh holds.

porosities = (10, 20, 30, 45)
cell = load("porous_effective_conductivity_models.step")
faces = cell.boundaries
pores = {
    percent: load(f"porous_effective_conductivity_models_pore_{percent}.step").named(
        "pore"
    )
    for percent in porosities
}
meshes = {
    percent: TetrahedralMesher(target_size=1e-4, embedded_regions=(pore,)).generate(
        cell
    )
    for percent, pore in pores.items()
}

fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    [cell.named("solid"), pores[30]],
    labels=False,
    camera=plots.Camera(elevation=30, azimuth=-55),
    length_unit="mm",
    title="Cell with a 30 % pore",
    ax=fig[0],
)
plots.mesh(
    meshes[30],
    groups=("all",),
    legend=False,
    length_unit="mm",
    title=f"Mesh: {meshes[30].cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Cell problems
#
# `regionwise` gives solid and pore their conductivities; the heat flow out of the cold face,
# `reaction`, gives the effective conductivity.

solid, fluid = 10.0, 0.6
material = IsotropicSolid(
    density=1.0,
    specific_heat_capacity=1.0,
    youngs_modulus=1.0,
    poisson_ratio=0.3,
    thermal_expansion_coefficient=0.0,
    thermal_conductivity=regionwise({"pore": fluid}, default=solid, unit="W/m/K"),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)
effective, meshed = {}, {}
for percent, mesh in meshes.items():
    model = HeatConduction(domain=cell.domain(), material=material, heating=0.0)
    problem = Problem(
        name=f"Porous cell, {percent} %",
        geometry=cell,
        physics=(model,),
        boundary_conditions=(
            FixedTemperature(boundary=faces.hot, physics=model, value=1.0),
            FixedTemperature(boundary=faces.cold, physics=model, value=0.0),
            HeatFlux(
                boundary=faces.all - faces.hot - faces.cold, physics=model, value=0.0
            ),
        ),
    )
    temperature = (
        Simulation(problem=problem, mesh=mesh, numerics=numerics)
        .solve_stationary()
        .field(model.temperature)
    )
    effective[percent] = (
        temperature.reaction(faces.hot) * cell.size[0] / (cell.size[1] * cell.size[2])
    )
    meshed[percent] = temperature.integrate(
        region=pores[percent], component=lambda T, position: np.ones_like(T)
    ).value / np.prod(cell.size)

# ## Results
#
# The cell conductivities against the four averaging models, each evaluated at the porosity the
# mesh of the cell holds (the volume of its pore cells).

models = {
    "volume average": lambda e: e * fluid + (1 - e) * solid,
    "reciprocal average": lambda e: 1 / (e / fluid + (1 - e) / solid),
    "power law": lambda e: fluid**e * solid ** (1 - e),
    "Maxwell-Eucken": lambda e: solid
    * (2 * solid + fluid - 2 * e * (solid - fluid))
    / (2 * solid + fluid + e * (solid - fluid)),
}
print(
    "porosity   meshed   cell [W/(m K)]   volume avg.   reciprocal avg.   power law   Maxwell-Eucken   cell vs Maxwell-Eucken"
)
for percent in porosities:
    e = meshed[percent]
    print(
        f"{percent:7d} % {100 * e:7.2f} % {effective[percent]:14.4f} {models['volume average'](e):13.4f} {models['reciprocal average'](e):17.4f} "
        f"{models['power law'](e):11.4f} {models['Maxwell-Eucken'](e):16.4f} {100 * (effective[percent] / models['Maxwell-Eucken'](e) - 1):+18.2f} %"
    )

# The cells follow Maxwell-Eucken within a fraction of a percent at low porosity and stay just
# below it at 45 %, while the power law underestimates and the volume average overestimates the
# conductivity by tens of percent: for a connected solid with closed pores, Maxwell-Eucken is the
# model to choose.
#
# ## Figure
#
# All averaging models over the porosity with the four cell results.

e = np.linspace(0, 0.6, 121)
plots.line(
    x=e * 100,
    y={name: law(e) for name, law in models.items()},
    points={
        "3D unit cells, spherical pores": (
            [100 * meshed[p] for p in porosities],
            [effective[p] for p in porosities],
        )
    },
    xlabel="porosity [%]",
    ylabel="k_eff [W/(m K)]",
    title="Effective conductivity: ceramic with water-filled pores",
    legend="upper right",
)
plots.show()

# ## Going further
#
# Swap solid and fluid roles (solid particles in a fluid) and compare with the other
# Maxwell-Eucken form, or replace the sphere by a cylinder in
# `porous_effective_conductivity_models_geometry.py` to model aligned channels.
