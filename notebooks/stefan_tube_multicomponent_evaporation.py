"""OneSimulation example: the Stefan tube, evaporation of a liquid mixture through stagnant air.
Requires the installed, activated OneSimulation runtime.

A liquid mixture of acetone and methanol evaporates at the bottom of a vertical tube; the vapours
diffuse up through air and are swept away at the open top. Air does not dissolve in the liquid, so
it is at rest, and the vapours ride on the mean motion they create themselves, the Stefan flow.
In a ternary gas the species interact: the Maxwell-Stefan equations couple every flux to every
mole-fraction gradient. OneSimulation computes the steady state on the tube axis and compares the
evaporation fluxes and the mole-fraction profiles with the exact solution and with the
measurements of Carty and Schrodt.
"""

# # The Stefan tube: multicomponent evaporation through stagnant air
#
# The tube of length $L=0.238$ m holds a liquid acetone (1) and methanol (2) mixture at the bottom
# ($z=0$); air (3) flows over the open top ($z=L$). At 328.5 K and 99.4 kPa the total molar
# concentration is $c=36.375$ mol/m³. The molar fluxes $N_i$ obey the steady balances and the
# Maxwell-Stefan relations
#
# $$\frac{dN_i}{dz}=0,\qquad
#   -c\,\frac{dx_i}{dz}=\sum_{j\ne i}\frac{x_j N_i-x_i N_j}{D_{ij}},\qquad N_3=0 .$$
#
# The last condition is the stagnant air: it fixes the molar-average velocity
# $u=(N_1+N_2)/c$, so no velocity has to be prescribed. OneSimulation's `SpeciesTransport` with
# the binary diffusivities $D_{ij}$ solves the coupled Maxwell-Stefan fluxes on finite volumes;
# `stagnant_species="air"` makes the model determine the Stefan flow itself and close the total
# concentration with air.
#
# | Quantity | Value |
# |---|---|
# | $D_{12}$ acetone-methanol | $8.48\cdot10^{-6}$ m²/s |
# | $D_{13}$ acetone-air | $13.72\cdot10^{-6}$ m²/s |
# | $D_{23}$ methanol-air | $19.91\cdot10^{-6}$ m²/s |
# | liquid surface, $z=0$ | $x_1=0.319$, $x_2=0.528$, $x_3=0.153$ |
# | open top, $z=L$ | pure air |
#
# **Reference.** Ternary evaporation experiment of R. Carty and T. Schrodt,
# *Ind. Eng. Chem. Fundam.* 14 (1975) 276 (run T-5): measured fluxes
# $N_1=1.779\cdot10^{-3}$ and $N_2=3.121\cdot10^{-3}$ mol/(m² s) and seven measured composition
# points. The exact solution of the one-dimensional problem, computed independently by integrating
# the Maxwell-Stefan relations and shooting the two fluxes onto the top composition (the method of
# R. Taylor and R. Krishna, *Multicomponent Mass Transfer*, Wiley 1993, §2.2), gives
# $N_1=1.78213\cdot10^{-3}$ and $N_2=3.12645\cdot10^{-3}$ mol/(m² s). For given fluxes the composition along the tube is linear in
# $x_1, x_2$ with constant coefficients,
#
# $$\frac{d}{dz}\begin{pmatrix}x_1\\x_2\\1\end{pmatrix}=\frac{1}{c}
#   \begin{pmatrix}\frac{N_2}{D_{12}}+\frac{N_1}{D_{13}} & \frac{N_1}{D_{13}}-\frac{N_1}{D_{12}} & -\frac{N_1}{D_{13}}\\
#   \frac{N_2}{D_{23}}-\frac{N_2}{D_{12}} & \frac{N_1}{D_{12}}+\frac{N_2}{D_{23}} & -\frac{N_2}{D_{23}}\\ 0&0&0\end{pmatrix}
#   \begin{pmatrix}x_1\\x_2\\1\end{pmatrix},$$
#
# so the exact profile is a matrix exponential, and it must reach pure air at the top.

import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fvm import CentralDifference, FiniteVolume, LeastSquaresGradient
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.physics.chemistry import (
    FixedConcentration,
    SpeciesTransport,
    TransportSpecies,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme
from scipy.linalg import expm

theme.set()


# ## Model
#
# The tube axis is a line; the bottom carries the composition in equilibrium with the liquid, the
# top pure air. The lateral wall needs no condition in one dimension.

length, total = 0.238, 36.375
names = ("acetone", "methanol", "air")
d12, d13, d23 = 8.48e-6, 13.72e-6, 19.91e-6
liquid_surface = np.array([0.319, 0.528, 0.153])
open_top = np.array([0.0, 0.0, 1.0])
tube = Line(start=(0.0, 0.0, 0.0), end=(length, 0.0, 0.0))
mixture = SpeciesTransport(
    domain=tube.line,
    species=tuple(TransportSpecies(name) for name in names),
    binary_diffusivities=((0.0, d12, d13), (d12, 0.0, d23), (d13, d23, 0.0)),
    stagnant_species="air",
)
problem = Problem(
    name="Stefan tube",
    geometry=tube,
    physics=(mixture,),
    boundary_conditions=(
        FixedConcentration(
            physics=mixture,
            boundary=tube.boundaries.start,
            value=dict(zip(names, total * liquid_surface)),
        ),
        FixedConcentration(
            physics=mixture,
            boundary=tube.boundaries.end,
            value=dict(zip(names, total * open_top)),
        ),
    ),
)
numerics = Numerics(
    spatial=FiniteVolume(
        gradient=LeastSquaresGradient(), diffusion=CentralDifference()
    ),
    nonlinear_solver=Newton(linear_solver=SparseDirect(), absolute_tolerance=1e-12),
)

# ## Steady state on five meshes
#
# The steady problem is solved directly with Newton's method on 10 to 160 cells. The fluxes are
# the outward boundary fluxes of each species at the open top (per unit cross section).

meshes = (10, 20, 40, 80, 160)
results = {
    cells: Simulation(
        problem=problem,
        mesh=StructuredLineMesher(cells=cells).generate(tube),
        numerics=numerics,
    ).solve_stationary()
    for cells in meshes
}
fluxes = {
    cells: np.array(
        [
            result.field(mixture.field(name)).boundary_flux(tube.boundaries.end)
            for name in names
        ]
    )
    for cells, result in results.items()
}

# ## Comparison of the fluxes
#
# The evaporation fluxes against the exact solution and the measurement; the air flux and the
# balance between bottom and top show the stagnant gas and the conservation of the scheme.

exact = np.array([1.78213e-3, 3.12645e-3])
measured = np.array([1.779e-3, 3.121e-3])
print(
    " cells   N acetone [mol/m²s]   N methanol [mol/m²s]   deviation from exact   N air [mol/m²s]"
)
for cells, flux in fluxes.items():
    print(
        f"{cells:6d} {flux[0]:19.6e} {flux[1]:21.6e}   {flux[0] / exact[0] - 1:+9.1e} {flux[1] / exact[1] - 1:+9.1e} {flux[2]:15.1e}"
    )
finest = results[meshes[-1]]
inflow = np.array(
    [
        finest.field(mixture.field(name)).boundary_flux(tube.boundaries.start)
        for name in names
    ]
)
print(
    f"bottom and top fluxes agree to {np.max(np.abs(inflow + fluxes[meshes[-1]])):.1e} mol/m²s"
)
print(
    f"deviation from the measured fluxes: acetone {fluxes[meshes[-1]][0] / measured[0] - 1:+.2%}, "
    f"methanol {fluxes[meshes[-1]][1] / measured[1] - 1:+.2%}"
)
print(
    f"Stefan flow: molar-average velocity (N1 + N2)/c = {fluxes[meshes[-1]][:2].sum() / total * 1e3:.4f} mm/s"
)

# ## Comparison of the profiles
#
# With the computed fluxes the matrix exponential gives the exact composition along the tube; it
# must arrive at pure air at the top. The measured compositions of Carty and Schrodt are shown as
# well.

n1, n2 = fluxes[meshes[-1]][:2]
generator = (
    np.array(
        [
            [n2 / d12 + n1 / d13, n1 / d13 - n1 / d12, -n1 / d13],
            [n2 / d23 - n2 / d12, n1 / d12 + n2 / d23, -n2 / d23],
            [0.0, 0.0, 0.0],
        ]
    )
    / total
)
heights = np.linspace(0.0, length, 121)
profile = np.array(
    [expm(generator * z) @ np.r_[liquid_surface[:2], 1.0] for z in heights]
)[:, :2]
print(
    f"exact profile with the computed fluxes reaches x1 = {profile[-1, 0]:.1e}, x2 = {profile[-1, 1]:.1e} at the top"
)
measured_height = 0.01 * np.array([4.45, 7.0, 9.45, 11.9, 14.4, 17.0, 19.4])
measured_acetone = np.array([0.263, 0.251, 0.232, 0.217, 0.185, 0.153, 0.113])
measured_methanol = np.array([0.457, 0.460, 0.421, 0.379, 0.307, 0.243, 0.168])
centres = {
    cells: np.asarray(result.field(mixture.field("air")).space.dof_coordinates)[:, 0]
    for cells, result in results.items()
}
fractions = {
    cells: np.stack(
        [np.asarray(result.field(mixture.field(name)).values) for name in names]
    )
    / total
    for cells, result in results.items()
}
print(
    " cells   largest profile deviation acetone   methanol   air      sum of mole fractions - 1"
)
for cells in meshes:
    reference = np.array(
        [np.interp(centres[cells], heights, profile[:, k]) for k in range(2)]
    )
    reference = np.vstack((reference, 1 - reference.sum(axis=0)))
    print(
        f"{cells:6d} {np.max(np.abs(fractions[cells][0] - reference[0])):27.1e} {np.max(np.abs(fractions[cells][1] - reference[1])):10.1e} "
        f"{np.max(np.abs(fractions[cells][2] - reference[2])):8.1e} {np.max(np.abs(fractions[cells].sum(axis=0) - 1)):18.1e}"
    )
rms = [
    np.sqrt(
        np.mean(
            (
                np.interp(
                    measured_height, centres[meshes[-1]], fractions[meshes[-1]][k]
                )
                - data
            )
            ** 2
        )
    )
    for k, data in enumerate((measured_acetone, measured_methanol))
]
print(
    f"root-mean-square deviation from the measured compositions: acetone {rms[0]:.4f}, methanol {rms[1]:.4f}"
)

# ## Figure
#
# Left: the three mole fractions along the tube on 20 cells (solid), the exact profile (dashed) and
# the measured points. Near the liquid the methanol profile is almost flat although methanol carries the larger
# flux: the Stefan flow and the friction with the acetone flux do part of its transport, a coupling
# that a Fick law with one diffusivity per species cannot show. The measured acetone fractions lie
# about 0.03 below both the computed and the exact profile, so this offset stems from the experiment
# or the property data, not from the discretisation. Right: the flux deviation from the exact
# solution over the cell size, with second-order convergence.

fig = plots.figure(1, 2, size=(15, 5.5))
plots.line(
    y={
        "acetone": (100 * centres[20], fractions[20][0]),
        "methanol": (100 * centres[20], fractions[20][1]),
        "air": (100 * centres[20], fractions[20][2]),
    },
    points={
        "measured acetone": (100 * measured_height, measured_acetone),
        "measured methanol": (100 * measured_height, measured_methanol),
    },
    reference={
        "acetone": (100 * heights, profile[:, 0]),
        "methanol": (100 * heights, profile[:, 1]),
        "air": (100 * heights, 1 - profile.sum(axis=1)),
    },
    colors={"measured acetone": 0, "measured methanol": 1},
    xlabel="height above the liquid z [cm]",
    ylabel="mole fraction",
    title="Composition in the tube (dashed: exact, circles: measured)",
    legend="upper left",
    ax=fig[0],
)
plots.line(
    points={
        "acetone": (
            np.array([length / cells for cells in meshes]),
            np.array([abs(fluxes[cells][0] / exact[0] - 1) for cells in meshes]),
        ),
        "methanol": (
            np.array([length / cells for cells in meshes]),
            np.array([abs(fluxes[cells][1] / exact[1] - 1) for cells in meshes]),
        ),
    },
    markers=True,
    scale="loglog",
    slope=2,
    xlabel="cell size [m]",
    ylabel="relative flux deviation",
    title="Mesh convergence of the evaporation fluxes",
    legend="lower right",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# The same declaration runs on the revolved r-z section of the tube (`RevolvedCellMeshView` in
# `FiniteVolume`) and on a three-dimensional tube; the fluxes then come out per tube rather than per
# unit area. With two species it reduces to the binary Stefan problem,
# $N_A=\frac{cD}{L}\ln\frac{1-x_{A,L}}{1-x_{A,0}}$. With initial values and a time integrator
# the same model computes the start-up of the evaporation from a tube initially filled with air.
