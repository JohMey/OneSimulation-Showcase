"""OneSimulation example: primary and secondary creep combined in a hot thick-walled tube.
Requires the installed, activated OneSimulation runtime.

The hot pressurised pipe of the companion creep example is given a primary creep stage on top of its
secondary creep: a Norton-Bailey law with strain hardening and a Norton law act together at every
quadrature point of the 2D-axisymmetric meridian section. The combined material relaxes faster at
first and reaches the same stationary creep stresses, which are compared with their closed form.
"""

# # Primary and secondary creep combined in a hot thick-walled tube
#
# Real creep curves start with a primary stage, in which the creep rate falls, before the steady
# secondary stage. A common model adds the two contributions to the equivalent creep strain rate,
#
# $$\dot{\bar\varepsilon}_c=\underbrace{m\,A_p^{1/m}\,q^{n_p/m}\,\bar\varepsilon_p^{(m-1)/m}}_{\text{Norton-Bailey, strain hardening}}
#   +\underbrace{A_s\,q^{n_s}}_{\text{Norton}},\qquad
#   \dot{\boldsymbol\varepsilon}_c=\tfrac32\,\dot{\bar\varepsilon}_c\,\frac{\mathbf s}{q},$$
#
# where each law keeps its own accumulated strain. Under constant stress the primary part dies out
# and the secondary law alone remains.
#
# **Setup.** The tube of the companion example: inner radius $a=50$ mm, outer radius $b=100$ mm,
# internal pressure $p=40$ MPa, long pipe (plane strain), $E=180$ GPa, $\nu=0.3$, temperature 850 K.
# Secondary creep: $A_s=2\cdot10^{-48}$ Pa$^{-5}$s$^{-1}$, $n_s=5$. Primary creep:
# $A_p=10^{-44}$ Pa$^{-5}$s$^{-0.3}$, $n_p=5$, $m=0.3$. Two runs over 1000 h: secondary
# creep alone and both laws together. The ring comes from the CAD file
# `combined_primary_secondary_creep.step`, written by `combined_primary_secondary_creep_geometry.py`
# with the CAD objects of OneSimulation; the computation uses its meridian section.
#
# | edge of the section | condition |
# |---|---|
# | `inner` | pressure $p$ |
# | `outer` | free |
# | `bottom`, `top` | axial displacement 0 (long pipe) |
#
# **Reference.** With the same stress exponent $n=5$ in both laws, the creep strain rate is
# proportional to $q^5$ at every point at any time, so the stresses tend to the stationary Norton
# distribution of the thick tube (Kraus, *Creep Analysis*, Wiley 1980, Sec. 4.2), with $\beta=2/n$,
#
# $$\sigma_\theta=p\,\frac{1+(\beta-1)(b/r)^\beta}{(b/a)^\beta-1},\qquad
#   \sigma_r=-p\,\frac{(b/r)^\beta-1}{(b/a)^\beta-1},\qquad \sigma_z=\tfrac12(\sigma_r+\sigma_\theta),$$
#
# starting from the elastic Lamé stresses. The primary stage only changes how fast it gets there.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials.solids import CreepMaterialPoint, PowerLawCreep
from onesim.meshing import TriangularMesher
from onesim.physics.solid_mechanics import SmallStrainSolid
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the sector of the ring with its named faces; `section` cuts the meridian plane
# $y=0$, whose $x$ coordinate is the radius. Quadratic triangles of 3 mm.

ring = load("combined_primary_secondary_creep.step")
faces = ring.boundaries
meridian = ring.section((0, 1, 0), origin=(0, 0, 0))
edges = meridian.boundaries
mesh = TriangularMesher(target_size=0.003).generate(meridian)

fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    ring,
    selection={
        "inner (pressure)": faces.inner,
        "outer": faces.outer,
        "ends": faces.bottom | faces.top,
    },
    labels=False,
    camera=plots.Camera(elevation=30, azimuth=-60),
    title="Sector of the pipe ring",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"Meridian section: {mesh.cell_count} quadratic triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Two `CreepMaterialPoint`s for `SmallStrainSolid` with axisymmetric kinematics: one with the Norton
# law alone, one with the Norton-Bailey law added (strain hardening). The load steps grow
# logarithmically from 36 s to 1000 h.

inner, outer, pressure, exponent = 0.050, 0.100, 40e6, 5.0
secondary = PowerLawCreep(coefficient=2e-48, stress_exponent=exponent)
primary = PowerLawCreep(coefficient=1e-44, stress_exponent=exponent, time_exponent=0.3)
hour = 3600.0
times = np.r_[0.0, np.geomspace(0.01, 1000.0, 45)] * hour
materials = {
    "secondary only": (secondary,),
    "primary + secondary": (primary, secondary),
}
results = {
    name: SmallStrainSolid(
        youngs_modulus=180e9,
        poisson_ratio=0.3,
        kinematics="axisymmetric",
        material_point=CreepMaterialPoint(laws=laws, hardening="strain"),
    ).solve(
        mesh=mesh,
        times=times,
        order=2,
        displacements=((edges.bottom, "z", 0.0), (edges.top, "z", 0.0)),
        tractions=((edges.inner, (pressure, 0.0)),),
    )
    for name, laws in materials.items()
}
radius = results["secondary only"].quadrature_coordinates[:, 0]
bore, mantle = np.argmin(radius), np.argmax(radius)

# ## Comparison with the closed form
#
# The stresses at the quadrature points after 1000 h against the stationary creep solution, the
# time at which the bore hoop stress has relaxed halfway, and the creep strain carried by each law.

beta, ratio = 2 / exponent, outer / inner
lame_hoop = lambda r: pressure / (ratio**2 - 1) * (1 + (outer / r) ** 2)
creep = {
    "σ_r": lambda r: -pressure * ((outer / r) ** beta - 1) / (ratio**beta - 1),
    "σ_z": lambda r: pressure
    * (1 + (beta / 2 - 1) * (outer / r) ** beta)
    / (ratio**beta - 1),
    "σ_θ": lambda r: pressure
    * (1 + (beta - 1) * (outer / r) ** beta)
    / (ratio**beta - 1),
}
deviation = lambda values, reference: np.abs(values - reference(radius)) / pressure
for name, result in results.items():
    print(
        f"{name}: after 1000 h "
        + ", ".join(
            f"{component} within {100 * deviation(result.stress[-1, :, k, k], creep[component]).max():.2f} % of p "
            f"(rms {100 * np.sqrt(np.mean(deviation(result.stress[-1, :, k, k], creep[component]) ** 2)):.2f} %)"
            for k, component in enumerate(creep)
        )
    )
for name, result in results.items():
    halfway = (
        times[
            np.argmax(
                result.stress[:, bore, 2, 2]
                <= 0.5 * (lame_hoop(radius[bore]) + creep["σ_θ"](radius[bore]))
            )
        ]
        / hour
    )
    print(f"{name}: bore hoop stress halfway relaxed after {halfway:.2f} h")
state = results["primary + secondary"].material_history[-1][0][0]
print(
    f"creep strain at the bore after 1000 h: primary {100 * state.accumulated[0]:.3f} %, secondary {100 * state.accumulated[1]:.3f} % "
    f"(secondary-only run: {100 * results['secondary only'].material_history[-1][0][0].equivalent_creep_strain:.3f} %)"
)

# ## Figure
#
# Left the hoop stress at the bore and at the outer surface over time for both materials, with the
# stationary values; right the hoop stress across the wall after 1 h and after 1000 h. The primary
# stage carries most of the early redistribution, the secondary law the final state.

early = int(np.argmin(np.abs(times - hour)))
line = np.linspace(inner, outer, 200)
mid = np.abs(results["secondary only"].quadrature_coordinates[:, 1] - 0.010) < 0.004
fig = plots.figure(1, 2, size=(16, 6))
plots.line(
    y={
        f"{where}, {name}": (times[1:] / hour, result.stress[1:, index, 2, 2] / 1e6)
        for name, result in results.items()
        for where, index in (("bore", bore), ("outer surface", mantle))
    },
    hlines={
        "stationary, bore": creep["σ_θ"](radius[bore]) / 1e6,
        "stationary, outer surface": creep["σ_θ"](radius[mantle]) / 1e6,
    },
    scale="logx",
    xlabel="time [h]",
    ylabel="hoop stress σ_θ [MPa]",
    title="Relaxation of the hoop stress",
    legend="center left",
    ax=fig[0],
)
plots.line(
    points={
        f"{name}, {times[k] / hour:.3g} h": (
            1e3 * radius[mid],
            result.stress[k, mid, 2, 2] / 1e6,
        )
        for name, result in results.items()
        for k in (early, len(times) - 1)
    },
    reference={
        "Lamé": (1e3 * line, lame_hoop(line) / 1e6),
        "stationary creep": (1e3 * line, creep["σ_θ"](line) / 1e6),
    },
    xlabel="radius r [mm]",
    ylabel="hoop stress σ_θ [MPa]",
    title="Hoop stress across the wall",
    legend="upper right",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Give the two laws different stress exponents: the stationary distribution then follows the
# secondary exponent, while the transient shows a mixture of both. Switch the primary law to time
# hardening to compare the two hardening rules in a non-uniform stress field.
