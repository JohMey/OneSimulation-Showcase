"""OneSimulation example: primary creep under a changing load, time hardening against strain hardening.
Requires the installed, activated OneSimulation runtime.

A block of a high-temperature alloy creeps under uniaxial tension for 50 hours; then a transverse
compression is added. Primary creep follows a Norton-Bailey law, integrated at every quadrature point
of a 3D solid with either the time-hardening or the strain-hardening rule. Both are compared with
their closed-form solutions for piecewise constant stress.
"""

# # Primary creep under a changing load
#
# At high temperature metals creep: they keep straining under constant stress. In the primary stage
# the creep rate falls with time, which the Norton-Bailey law describes for constant stress as
#
# $$\bar\varepsilon_c=A\,q^n\,t^m ,\qquad m<1 ,$$
#
# with the von Mises stress $q$ and the equivalent creep strain $\bar\varepsilon_c$. The creep strain
# rate is deviatoric and follows the stress deviator $\mathbf s$,
# $\dot{\boldsymbol\varepsilon}_c=\tfrac32\,\dot{\bar\varepsilon}_c\,\mathbf s/q$.
#
# When the stress changes, the law has to say what the material "remembers". Two classical rules
# differentiate the law differently:
#
# | rule | creep rate | memory |
# |---|---|---|
# | time hardening | $\dot{\bar\varepsilon}_c=m\,A\,q^n\,t^{m-1}$ | time since the start of creep |
# | strain hardening | $\dot{\bar\varepsilon}_c=m\,A^{1/m}q^{n/m}\,\bar\varepsilon_c^{(m-1)/m}$ | creep strain accumulated so far |
#
# Under constant stress both give the same curve; after a load change they differ.
#
# **Setup.** A block of 10 mm × 10 mm × 2 mm with $E=200$ GPa, $\nu=0.3$ and the creep constants
# $A=8\cdot10^{-30}$ Pa$^{-3}$s$^{-1/2}$, $n=3$, $m=0.5$. For the first 50 h the block carries a
# tensile stress $\sigma_x=100$ MPa; from 50 h to 100 h a compressive stress $\sigma_y=-100$ MPa is
# added, which raises the von Mises stress to $100\sqrt3$ MPa and turns the creep direction. The block
# comes from the CAD file `primary_creep_nonconstant_load.step`, written by
# `primary_creep_nonconstant_load_geometry.py` with the CAD objects of OneSimulation.
#
# | face | condition |
# |---|---|
# | `x_min`, `y_min`, `z_min` | symmetry (normal displacement 0) |
# | `x_max` | traction $\sigma_x=100$ MPa |
# | `y_max` | traction $\sigma_y=0$, from 50 h on $-100$ MPa |
# | `z_max` | free |
#
# **Reference.** The stress is uniform and constant within each stage, so the equivalent creep strain
# has closed forms (e.g. Kraus, *Creep Analysis*, Wiley 1980, Ch. 2): with $t_1=50$ h,
# $\bar\varepsilon_1=A q_1^n t_1^m$ and $t>t_1$,
#
# $$\text{time hardening: }\bar\varepsilon_c=\bar\varepsilon_1+A q_2^n\,(t^m-t_1^m),\qquad
#   \text{strain hardening: }\bar\varepsilon_c=\big(\bar\varepsilon_1^{1/m}+A^{1/m}q_2^{n/m}(t-t_1)\big)^m .$$
#
# The tensor components follow from the flow direction of each stage: $(1,-\tfrac12,-\tfrac12)$ in
# uniaxial tension, $(\tfrac{\sqrt3}2,-\tfrac{\sqrt3}2,0)$ under the combined load.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials.solids import CreepMaterialPoint, PowerLawCreep
from onesim.meshing import CadHexahedralMesher
from onesim.physics.solid_mechanics import SmallStrainSolid
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the block with its named faces; it is meshed with quadratic hexahedra.

block = load("primary_creep_nonconstant_load.step")
faces = block.boundaries
mesh = CadHexahedralMesher(cells=(4, 4, 1)).generate(block)

view = plots.Camera(elevation=35, azimuth=40)
fig = plots.figure(1, 2, size=(13, 6))
plots.geometry(
    block,
    selection={
        "σ_x": faces.x_max,
        "σ_y": faces.y_max,
        "symmetry": faces.x_min | faces.y_min | faces.z_min,
    },
    labels=False,
    camera=view,
    title="Creep block with its named faces",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    camera=view,
    title=f"{mesh.cell_count} quadratic hexahedra",
    ax=fig[1],
)
plots.show()

# ## Model
#
# `SmallStrainSolid` in 3D carries a `CreepMaterialPoint` with the Norton-Bailey law; the hardening
# rule is the only difference between the two runs. Each load step is integrated implicitly at
# every quadrature point. The steps are spaced logarithmically from the start of creep and again from
# the load change, where the creep rate is highest.

hour, switch = 3600.0, 50 * 3600.0
coefficient, exponent, time_exponent = 8e-30, 3.0, 0.5
law = PowerLawCreep(
    coefficient=coefficient, stress_exponent=exponent, time_exponent=time_exponent
)
times = (
    np.r_[0.0, np.geomspace(0.01, 50.0, 30), 50.0 + np.geomspace(0.01, 50.0, 30)] * hour
)
transverse = lambda x, y, z, t: (0.0, -100e6 if t > switch + 1.0 else 0.0, 0.0)
results = {
    rule: SmallStrainSolid(
        youngs_modulus=200e9,
        poisson_ratio=0.3,
        kinematics="volume",
        material_point=CreepMaterialPoint(laws=(law,), hardening=rule),
    ).solve(
        mesh=mesh,
        times=times,
        order=2,
        displacements=(
            (faces.x_min, "x", 0.0),
            (faces.y_min, "y", 0.0),
            (faces.z_min, "z", 0.0),
        ),
        tractions=((faces.x_max, (100e6, 0.0, 0.0)), (faces.y_max, transverse)),
    )
    for rule in ("time", "strain")
}
states = {
    rule: [step[0][0] for step in result.material_history]
    for rule, result in results.items()
}
creep = {
    rule: np.array([state.creep_strain[:3] for state in history])
    for rule, history in states.items()
}
equivalent = {
    rule: np.array([state.equivalent_creep_strain for state in history])
    for rule, history in states.items()
}
spread = {
    rule: np.ptp(
        [
            point.equivalent_creep_strain
            for cell in result.material_history[-1]
            for point in cell
        ]
    )
    for rule, result in results.items()
}

# ## Comparison with the closed-form solution
#
# The equivalent creep strain and the three normal creep strains against the closed forms, at the
# load change and at the end.

first, second = 100e6, 100e6 * np.sqrt(3)
before = (
    lambda t: coefficient * first**exponent * np.minimum(t, switch) ** time_exponent
)
reference = {
    "time": lambda t: np.where(
        t <= switch,
        before(t),
        before(switch)
        + coefficient
        * second**exponent
        * (np.maximum(t, switch) ** time_exponent - switch**time_exponent),
    ),
    "strain": lambda t: np.where(
        t <= switch,
        before(t),
        (
            before(switch) ** (1 / time_exponent)
            + coefficient ** (1 / time_exponent)
            * second ** (exponent / time_exponent)
            * np.maximum(t - switch, 0.0)
        )
        ** time_exponent,
    ),
}
components = lambda total: (
    np.outer(np.minimum(total, before(switch)), [1.0, -0.5, -0.5])
    + np.outer(
        total - np.minimum(total, before(switch)),
        [np.sqrt(3) / 2, -np.sqrt(3) / 2, 0.0],
    )
)
print(
    f"stress at the end: σ_x = {results['strain'].stress[-1, :, 0, 0].mean() / 1e6:.3f} MPa, σ_y = {results['strain'].stress[-1, :, 1, 1].mean() / 1e6:.3f} MPa"
)
for rule in results:
    exact = reference[rule](times)
    print(
        f"{rule} hardening: equivalent creep strain {100 * equivalent[rule][times == switch][0]:.4f} % at 50 h and {100 * equivalent[rule][-1]:.4f} % at 100 h "
        f"(closed form {100 * exact[times == switch][0]:.4f} % and {100 * exact[-1]:.4f} %), "
        f"largest deviation {np.abs(equivalent[rule] - exact).max() / exact[-1]:.1e} of the final value, "
        f"spread over all quadrature points {spread[rule]:.1e}"
    )
    print(
        f"   creep strains at 100 h: ε_xx {100 * creep[rule][-1, 0]:.4f} %, ε_yy {100 * creep[rule][-1, 1]:.4f} %, ε_zz {100 * creep[rule][-1, 2]:.4f} %"
        f"  (closed form {', '.join(f'{100 * value:.4f} %' for value in components(exact)[-1])})"
    )
print(
    f"after the load change strain hardening creeps faster: at 100 h it is {100 * (equivalent['strain'][-1] / equivalent['time'][-1] - 1):.1f} % ahead of time hardening"
)

# ## Figure
#
# Left the equivalent creep strain of both rules with the closed forms; right the normal creep
# strains of both rules, with the closed form of the strain-hardening rule as lines. In the first stage the curves coincide. After the load change, time hardening continues
# on the flat part of the creep curve, whereas strain hardening restarts on a steeper part, because
# the higher stress has less accumulated creep strain behind it.

line = np.linspace(0.0, times[-1], 600)
fig = plots.figure(1, 2, size=(16, 6))
plots.line(
    points={
        f"{rule} hardening": (times / hour, 100 * equivalent[rule]) for rule in results
    },
    reference={
        f"{rule} hardening, closed form": (line / hour, 100 * reference[rule](line))
        for rule in results
    },
    vlines={"load change": 50.0},
    xlabel="time [h]",
    ylabel="equivalent creep strain [%]",
    title="Equivalent creep strain",
    legend="upper left",
    ax=fig[0],
)
plots.line(
    points={
        f"ε_{axis}, {rule}": (times / hour, 100 * creep[rule][:, k])
        for rule in results
        for k, axis in enumerate(("xx", "yy", "zz"))
    },
    reference={
        f"ε_{axis}, closed form": (
            line / hour,
            100 * components(reference["strain"](line))[:, k],
        )
        for k, axis in enumerate(("xx", "yy", "zz"))
    },
    vlines={"load change": 50.0},
    xlabel="time [h]",
    ylabel="creep strain [%]",
    title="Normal creep strains",
    legend="lower left",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Add a secondary Norton term as a second law in `laws=(...)`, make the creep constant depend on
# temperature through an activation energy, or remove the load after 50 h and follow the elastic
# recovery; under a reversed load strain hardening and time hardening part even further.
