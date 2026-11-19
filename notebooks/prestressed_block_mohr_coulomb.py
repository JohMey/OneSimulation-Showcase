"""OneSimulation example: uniaxial compression of a prestressed rock block with Mohr-Coulomb plasticity.
Requires the installed, activated OneSimulation runtime.

A rock block is read from a STEP file, meshed with tetrahedra, brought to an isotropic initial
stress and then compressed vertically while the lateral stress is held. The vertical yield
stress, the hardening slope and the lateral expansion of the 3D elastoplastic solution are
compared with the closed-form Mohr-Coulomb solution for two prestress levels, and the
Drucker-Prager cone fitted to the compression meridian is run alongside.
"""

# # Uniaxial compression of a prestressed rock block
#
# A rock sample in the ground is never stress free. In this verification a cubic block first
# carries an isotropic stress $p_0$; then its top is pushed down while the stress on its sides
# stays at $p_0$. Because the stress state remains uniform, the vertical stress at which the rock
# yields can be written down: a classical check of a 3D plasticity implementation.
#
# The rock follows the **Mohr-Coulomb** criterion with linear isotropic hardening of the
# cohesion and a non-associated plastic potential (dilatancy angle $\psi$):
#
# $$f=\sigma_1-\sigma_3+(\sigma_1+\sigma_3)\sin\varphi-2\,c(\kappa)\cos\varphi\le0,\qquad
#   c(\kappa)=c_0+\frac{H\kappa}{2\cos\varphi}$$
#
# (tension positive, $\sigma_1\ge\sigma_2\ge\sigma_3$, $\kappa$ the accumulated plastic
# multiplier).
#
# **Setup.** A cube of side 1 m, $E=10$ GPa, $\nu=0.25$, $c_0=5$ MPa, $\varphi=35°$, $\psi=10°$,
# hardening modulus $H=1$ GPa; prestress $p_0=10$ and 20 MPa; vertical compression to
# 1.2 % strain. The block comes from the CAD file `prestressed_block_mohr_coulomb.step`, written
# by `prestressed_block_mohr_coulomb_geometry.py` with the CAD objects of OneSimulation.
#
# | face | condition |
# |---|---|
# | `x_min`, `y_min`, `z_min` | roller (symmetry) |
# | `x_max`, `y_max` | normal stress $-p_0$ |
# | `z_max` | vertical displacement, first the elastic isotropic compression, then pushed down |
#
# **Reference.** With the vertical strain $\varepsilon$ counted from the prestressed state, the
# vertical stress rises as $p_0+E\varepsilon$ up to the yield stress
#
# $$\sigma_y=p_0\,\frac{1+\sin\varphi}{1-\sin\varphi}+\frac{2c_0\cos\varphi}{1-\sin\varphi},$$
#
# and afterwards with the slope $E\,H_\varphi/(H_\varphi+E(1-\sin\psi))$, $H_\varphi=H/(1-\sin\varphi)$,
# from the consistency condition (e.g. M.A. Crisfield, *Non-linear Finite Element Analysis of
# Solids and Structures*, Vol. 2, Wiley 1997; D.M. Potts and L. Zdravković, *Finite Element
# Analysis in Geotechnical Engineering: Theory*, Telford 1999).

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import TetrahedralMesher
from onesim.physics.geotechnics import (
    DruckerPragerSoil,
    FrictionalSolid,
    MohrCoulombRock,
)
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the block with its named faces; an unstructured tetrahedral mesh fills it.

block = load("prestressed_block_mohr_coulomb.step")
faces = block.boundaries
side = block.size[2]
mesh = TetrahedralMesher(target_size=0.35).generate(block)

view = plots.Camera(elevation=25, azimuth=-55)
fig = plots.figure(1, 2, size=(13, 6))
plots.geometry(
    block,
    selection={
        "rollers": faces.x_min | faces.y_min | faces.z_min,
        "prestress": faces.x_max | faces.y_max,
        "pushed down": faces.z_max,
    },
    labels=False,
    camera=view,
    title="Rock block with its named faces",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{mesh.cell_count} tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The same parameters for both criteria; the Drucker-Prager cone coincides with Mohr-Coulomb on
# the compression meridian. 25 load steps.

parameters = dict(
    youngs_modulus=10e9,
    poisson_ratio=0.25,
    cohesion=5e6,
    friction_angle=np.radians(35),
    dilation_angle=np.radians(10),
    hardening_modulus=1e9,
)
materials = {
    "Mohr-Coulomb": MohrCoulombRock(**parameters),
    "Drucker-Prager": DruckerPragerSoil(**parameters),
}
prestresses = (10e6, 20e6)
strain = np.linspace(0.0, 0.012, 25)
young, poisson = parameters["youngs_modulus"], parameters["poisson_ratio"]
results = {
    (name, p0): FrictionalSolid(material=material, kinematics="volume").solve(
        mesh=mesh,
        times=strain * side,
        normal_tractions=((faces.x_max, -p0), (faces.y_max, -p0)),
        displacements=(
            (faces.x_min, "x", 0.0),
            (faces.y_min, "y", 0.0),
            (faces.z_min, "z", 0.0),
            (
                faces.z_max,
                "z",
                (lambda start: lambda t: start - t)(
                    -p0 * (1 - 2 * poisson) / young * side
                ),
            ),
        ),
    )
    for name, material in materials.items()
    for p0 in prestresses
}
top = np.isclose(results["Mohr-Coulomb", 10e6].coordinates[:, 2], side)
corner = np.argmax(results["Mohr-Coulomb", 10e6].coordinates.sum(axis=1))
vertical = {
    key: -result.reactions[:, top, 2].sum(axis=1) / side**2
    for key, result in results.items()
}
lateral = {
    key: result.displacement[:, corner, 0] - result.displacement[0, corner, 0]
    for key, result in results.items()
}

# ## Comparison with the reference
#
# The vertical yield stress from the first plastic step, the slope after yield, and the uniformity
# of the 3D stress field.

sine, cosine, dilation = (
    np.sin(parameters["friction_angle"]),
    np.cos(parameters["friction_angle"]),
    np.sin(parameters["dilation_angle"]),
)
yield_stress = lambda p0: p0 * (1 + sine) / (1 - sine) + 2 * parameters[
    "cohesion"
] * cosine / (1 - sine)
hardening = {
    "Mohr-Coulomb": parameters["hardening_modulus"] / (1 - sine),
    "Drucker-Prager": parameters["hardening_modulus"] * (3 - sine) / (3 * (1 - sine)),
}
flow = {
    "Mohr-Coulomb": 1 - dilation,
    "Drucker-Prager": 1 - 2 * dilation / (3 - dilation),
}
slope = {
    name: young * hardening[name] / (hardening[name] + young * flow[name])
    for name in materials
}
print(
    f"{'criterion':15s} {'p0 [MPa]':>8s} {'yield stress [MPa]':>19s} {'closed form':>12s} {'deviation':>10s}"
    f" {'slope after yield [GPa]':>24s} {'closed form':>12s} {'stress spread [Pa]':>19s}"
)
for (name, p0), sigma in vertical.items():
    measured_slope = (sigma[-1] - sigma[-2]) / (strain[-1] - strain[-2])
    measured = p0 + young * (sigma[-1] - measured_slope * strain[-1] - p0) / (
        young - measured_slope
    )
    print(
        f"{name:15s} {p0 / 1e6:8.0f} {measured / 1e6:19.4f} {yield_stress(p0) / 1e6:12.4f} {100 * (measured / yield_stress(p0) - 1):+9.1e} %"
        f" {measured_slope / 1e9:24.4f} {slope[name] / 1e9:12.4f} {np.ptp(results[name, p0].stress[-1, :, 2, 2]):19.1e}"
    )

# ## Figure
#
# Vertical stress and lateral expansion against vertical strain for both criteria and both
# prestress levels.

line = np.linspace(0.0, 0.012, 601)
fig = plots.figure(1, 2, size=(15, 6))
plots.line(
    points={
        f"{name}, p0 = {p0 / 1e6:.0f} MPa": (100 * strain, sigma / 1e6)
        for (name, p0), sigma in vertical.items()
    },
    reference={
        f"Mohr-Coulomb closed form, p0 = {p0 / 1e6:.0f} MPa": (
            100 * line,
            np.where(
                p0 + young * line < yield_stress(p0),
                p0 + young * line,
                yield_stress(p0)
                + slope["Mohr-Coulomb"] * (line - (yield_stress(p0) - p0) / young),
            )
            / 1e6,
        )
        for p0 in prestresses
    },
    xlabel="vertical strain from the prestressed state [%]",
    ylabel="vertical stress [MPa] (compression positive)",
    title="Vertical stress",
    legend="lower right",
    ax=fig[0],
)
plots.line(
    y={
        f"{name}, p0 = {p0 / 1e6:.0f} MPa": (100 * strain, 1e3 * values)
        for (name, p0), values in lateral.items()
    },
    reference={"elastic, ν ε": (100 * line, 1e3 * poisson * line * side)},
    xlabel="vertical strain from the prestressed state [%]",
    ylabel="lateral expansion of the block [mm]",
    title="Lateral expansion: dilatancy after yield",
    legend="upper left",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Set the hardening modulus to zero for perfect plasticity, or the dilatancy angle equal to the
# friction angle for associated flow. Pull the block instead of pushing it: in triaxial extension
# the Drucker-Prager cone fitted to compression overestimates the strength of the rock, while the
# Mohr-Coulomb hexagon keeps its corners.
