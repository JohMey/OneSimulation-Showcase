"""OneSimulation example: plasticity and creep in a heated, pressurised thick-walled cylinder.
Requires the installed, activated OneSimulation runtime.

A thick-walled steel cylinder is first heated from the inside, which yields its bore by thermal
stress, then pressurised, and held for 2000 hours while it creeps. A sector of the cylinder is read
from a STEP file and computed in 3D with one material that combines thermal expansion, von Mises
plasticity with hardening and Norton creep. The thermoelastic start and the stationary creep stresses
are compared with closed-form solutions.
"""

# # Plasticity and creep in a heated thick-walled cylinder
#
# Components of power plants are loaded by temperature differences and by pressure, and at their
# working temperature they creep. Such a load history is *non-proportional*: the thermal stresses come
# first, the pressure later, and the stresses redistribute by plastic flow and by creep in between.
# One material law here carries all three effects at every quadrature point,
#
# $$\boldsymbol\varepsilon=\boldsymbol\varepsilon_e+\boldsymbol\varepsilon_p+\boldsymbol\varepsilon_c+\alpha(T-T_0)\mathbf I,\qquad
#   q\le\sigma_y+H\bar\varepsilon_p,\qquad \dot{\bar\varepsilon}_c=A\,q^n ,$$
#
# with von Mises plasticity and Norton creep acting in the direction of the stress deviator; both
# are integrated in one implicit radial return per step.
#
# **Setup.** Inner radius $a=100$ mm, outer radius $b=200$ mm, a long cylinder (plane strain).
# $E=180$ GPa, $\nu=0.3$, $\alpha=14\cdot10^{-6}$ 1/K, $\sigma_y=120$ MPa, $H=5$ GPa, Norton creep
# $A=3\cdot10^{-40}$ Pa$^{-4}$s$^{-1}$, $n=4$. History: within 6 minutes the bore heats by 250 K and the outside
# by 100 K above the reference temperature (steady logarithmic profile); then an internal pressure of
# 40 MPa is applied within the next 6 minutes and held to 2000 h. A 90° sector, 20 mm long, comes from the CAD
# file `thick_cylinder_plasticity_creep.step`, written by `thick_cylinder_plasticity_creep_geometry.py`
# with the CAD objects of OneSimulation.
#
# | face | condition |
# |---|---|
# | `inner` | pressure $p(t)$ |
# | `outer` | free |
# | `symmetry_x`, `symmetry_y` | normal displacement 0 |
# | `ends` | axial displacement 0 (long cylinder) |
#
# **Reference.** (1) While elastic, the thermal stresses of the logarithmic profile with the
# temperature difference $\Delta T$ across the wall are (Timoshenko and Goodier, *Theory of
# Elasticity*, §150), with $k=\alpha E\Delta T/\big(2(1-\nu)\ln(b/a)\big)$,
#
# $$\sigma_r=k\Big[-\ln\frac br-\frac{a^2}{b^2-a^2}\Big(1-\frac{b^2}{r^2}\Big)\ln\frac ba\Big],\qquad
#   \sigma_\theta=k\Big[1-\ln\frac br-\frac{a^2}{b^2-a^2}\Big(1+\frac{b^2}{r^2}\Big)\ln\frac ba\Big].$$
#
# (2) In the stationary creep state only strain *rates* matter: constant thermal and plastic strains
# drop out, and the stresses tend to the Norton distribution of the pressurised tube
# (Kraus, *Creep Analysis*, 1980), $\sigma_\theta=p\,\big(1+(\beta-1)(b/r)^\beta\big)/\big((b/a)^\beta-1\big)$
# with $\beta=2/n$, whatever the thermal and plastic history.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials.solids import CreepMaterialPoint, PowerLawCreep
from onesim.meshing import TetrahedralMesher
from onesim.physics.solid_mechanics import SmallStrainSolid
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the sector with its named faces; quadratic tetrahedra of 15 mm.

sector = load("thick_cylinder_plasticity_creep.step")
faces = sector.boundaries
mesh = TetrahedralMesher(target_size=0.015).generate(sector)

view = plots.Camera(elevation=35, azimuth=-60)
fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    sector,
    selection={
        "inner (pressure)": faces.inner,
        "outer": faces.outer,
        "symmetry": faces.symmetry_x | faces.symmetry_y,
    },
    labels=False,
    camera=view,
    title="Sector of the cylinder",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{mesh.cell_count} quadratic tetrahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The temperature at a point follows the logarithmic profile, scaled up during the first 6 minutes; the
# pressure acts radially on the bore. `CreepMaterialPoint` with a yield stress and a thermal expansion
# coefficient carries all strains; the steps follow the history. Plastic flow and creep keep the volume
# constant, and quadratic tetrahedra lock under such flow (the stresses then scatter wildly from point
# to point) unless the volumetric strain is relaxed: `volumetric_strain="element_mean"` replaces it
# by its mean over each element (B-bar method), and the pressure reported is that element mean.

inner, outer, hour = 0.100, 0.200, 3600.0
heating = lambda t: min(t / (0.1 * hour), 1.0)
temperature = lambda point, time: 293.15 + heating(time) * (
    100.0 + 150.0 * np.log(outer / np.hypot(point[0], point[1])) / np.log(outer / inner)
)
pressure = lambda t: 40e6 * np.clip((t - 0.1 * hour) / (0.1 * hour), 0.0, 1.0)
material = CreepMaterialPoint(
    laws=(PowerLawCreep(coefficient=3e-40, stress_exponent=4.0),),
    temperature=temperature,
    yield_stress=120e6,
    hardening_modulus=5e9,
    thermal_expansion=14e-6,
    reference_temperature=293.15,
)
times = (
    np.r_[
        0.0,
        0.002,
        np.linspace(0.01, 0.1, 10),
        np.linspace(0.12, 0.2, 5),
        0.2 + np.geomspace(0.01, 1999.8, 25),
    ]
    * hour
)
result = SmallStrainSolid(
    youngs_modulus=180e9,
    poisson_ratio=0.3,
    kinematics="volume",
    material_point=material,
).solve(
    mesh=mesh,
    times=times,
    order=2,
    displacements=(
        (faces.symmetry_x, "x", 0.0),
        (faces.symmetry_y, "y", 0.0),
        (faces.ends, "z", 0.0),
    ),
    tractions=(
        (
            faces.inner,
            lambda x, y, z, t: np.column_stack(
                (
                    pressure(t) * x / np.hypot(x, y),
                    pressure(t) * y / np.hypot(x, y),
                    0 * x,
                )
            ),
        ),
    ),
    volumetric_strain="element_mean",
)
points = result.quadrature_coordinates
radius = np.hypot(points[:, 0], points[:, 1])
radial = np.column_stack((points[:, 0] / radius, points[:, 1] / radius, 0 * radius))
hoop_direction = np.column_stack(
    (-points[:, 1] / radius, points[:, 0] / radius, 0 * radius)
)
component = lambda k, direction: np.einsum(
    "ni,nij,nj->n", direction, result.stress[k], direction
)
plastic = lambda k: np.array(
    [p.equivalent_plastic_strain for cell in result.material_history[k] for p in cell]
)

# ## Comparison with the closed forms
#
# The thermoelastic stresses at 2 % of the heating, the plastic zone at the end of the heating, and
# the stresses at 2000 h against the stationary creep solution.

k_factor = (
    lambda fraction: 14e-6
    * 180e9
    * fraction
    * 150.0
    / (2 * 0.7 * np.log(outer / inner))
)
ratio = inner**2 / (outer**2 - inner**2)
thermal = {
    "σ_r": lambda r, f: k_factor(f)
    * (-np.log(outer / r) - ratio * (1 - outer**2 / r**2) * np.log(outer / inner)),
    "σ_θ": lambda r, f: k_factor(f)
    * (1 - np.log(outer / r) - ratio * (1 + outer**2 / r**2) * np.log(outer / inner)),
}
beta = 2 / 4.0
creep = {
    "σ_r": lambda r: -40e6 * ((outer / r) ** beta - 1) / ((outer / inner) ** beta - 1),
    "σ_θ": lambda r: 40e6
    * (1 + (beta - 1) * (outer / r) ** beta)
    / ((outer / inner) ** beta - 1),
}
directions = {"σ_r": radial, "σ_θ": hoop_direction}
end_heating = int(np.argmin(np.abs(times - 0.1 * hour)))
rms = (
    lambda values, reference, scale: np.sqrt(np.mean((values - reference) ** 2)) / scale
)
for name, direction in directions.items():
    print(
        f"{name}: at 2 % heating rms deviation from the thermoelastic solution {100 * rms(component(1, direction), thermal[name](radius, 0.02), np.abs(thermal['σ_θ'](radius, 0.02)).max()):.2f} % "
        f"of its peak hoop stress; after 2000 h rms deviation from the stationary creep solution {100 * rms(component(-1, direction), creep[name](radius), 40e6):.2f} % of p"
    )
bore, mantle = int(np.argmin(radius)), int(np.argmax(radius))
print(
    f"after 2000 h: hoop stress at the bore {component(-1, hoop_direction)[bore] / 1e6:.2f} MPa (stationary {creep['σ_θ'](radius[bore]) / 1e6:.2f}), "
    f"at the outer surface {component(-1, hoop_direction)[mantle] / 1e6:.2f} MPa (stationary {creep['σ_θ'](radius[mantle]) / 1e6:.2f})"
)
print(
    f"end of heating: elastic hoop stress at the bore would be {thermal['σ_θ'](inner, 1.0) / 1e6:.1f} MPa; yielded zone up to r = "
    f"{1e3 * radius[plastic(end_heating) > 0].max():.1f} mm, largest plastic strain {100 * plastic(end_heating).max():.3f} %"
)
print(
    f"after 2000 h: largest creep strain {100 * max(p.equivalent_creep_strain for cell in result.material_history[-1] for p in cell):.3f} %, "
    f"plastic strain grew to {100 * plastic(-1).max():.3f} %"
)

# ## Figure
#
# Left the hoop stress across the wall at the end of the heating, after the pressure is applied and
# after 2000 h, with the thermoelastic and the stationary creep solutions; right the hoop stress at the
# bore and at the outer surface over the history.

line = np.linspace(inner, outer, 200)
steps = {
    "end of heating": end_heating,
    "pressure applied": int(np.argmin(np.abs(times - 0.2 * hour))),
    "after 2000 h": len(times) - 1,
}
fig = plots.figure(1, 2, size=(16, 6))
plots.line(
    points={
        name: (1e3 * radius, component(k, hoop_direction) / 1e6)
        for name, k in steps.items()
    },
    reference={
        "thermoelastic, end of heating": (1e3 * line, thermal["σ_θ"](line, 1.0) / 1e6),
        "stationary creep": (1e3 * line, creep["σ_θ"](line) / 1e6),
    },
    xlabel="radius r [mm]",
    ylabel="hoop stress σ_θ [MPa]",
    title="Hoop stress across the wall",
    legend="upper right",
    ax=fig[0],
)
plots.line(
    y={
        where: (
            times[1:] / hour,
            np.array(
                [component(k, hoop_direction)[index] for k in range(1, len(times))]
            )
            / 1e6,
        )
        for where, index in (("bore", bore), ("outer surface", mantle))
    },
    hlines={
        "stationary, bore": creep["σ_θ"](radius[bore]) / 1e6,
        "stationary, outer surface": creep["σ_θ"](radius[mantle]) / 1e6,
    },
    scale="logx",
    xlabel="time [h]",
    ylabel="hoop stress σ_θ [MPa]",
    title="History of the hoop stress",
    legend="lower left",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Cool the cylinder down after the hold and follow the reversed plastic flow, make the creep
# coefficient temperature dependent through an activation energy, or cycle the pressure.
