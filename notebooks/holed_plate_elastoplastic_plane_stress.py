"""OneSimulation example: elastoplastic perforated plate in plane stress, loading and unloading.
Requires the installed, activated OneSimulation runtime.

An aluminium strip with a large central hole is pulled into the plastic range and released again.
The quarter section is cut from a STEP file and computed in plane stress with von Mises plasticity
and linear hardening on quadratic triangles. The elastic stress concentration, the onset of yield,
the net-section limit and the residual stress after unloading are compared with handbook values and
elastic superposition.
"""

# # Elastoplastic perforated plate in plane stress
#
# A hole concentrates the stress in a loaded plate. As the load grows, the material at the edge of
# the hole yields first; a plastic zone spreads across the ligament between hole and edge, and the
# plate stiffens again only through hardening once the whole net section is plastic. On unloading
# the elastic spring-back of the surrounding plate puts the plastic zone into compression: residual
# stresses remain at the hole edge.
#
# The case follows the classical perforated strip of Zienkiewicz, Valliappan and King (Int. J.
# Numer. Meth. Eng. 1, 1969, 75-100), also in O.C. Zienkiewicz and R.L. Taylor, *The Finite Element
# Method*, Vol. 2. In plane stress the out-of-plane stress vanishes; the von Mises yield condition
# with linear isotropic hardening is
#
# $$\sqrt{\tfrac32\,\mathbf s:\mathbf s}=\sigma_y+H\,\bar\varepsilon_p ,$$
#
# and the out-of-plane strain is found at every quadrature point so that $\sigma_{zz}=0$.
#
# **Setup.** A strip 200 mm wide, 360 mm long and 1 mm thick with a central hole of $d=100$ mm;
# $E=68.7$ GPa, $\nu=0.2$, $\sigma_y=238$ MPa, $H=2.21$ GPa (the values of the original study,
# 7000, 24.3 and 225 kgf/mm²). The ends carry a uniform tension $\sigma$ that rises to
# $0.55\,\sigma_y$ and returns to zero. A quarter of the plate comes from the CAD file
# `holed_plate_elastoplastic_plane_stress.step`, written by
# `holed_plate_elastoplastic_plane_stress_geometry.py` with the CAD objects of OneSimulation; the
# computation uses its mid-plane section.
#
# | edge of the section | condition |
# |---|---|
# | `symmetry_x` | $u_x=0$ |
# | `symmetry_y` | $u_y=0$ |
# | `loaded` | traction $\sigma$ in $y$ |
# | `hole`, `side` | free |
#
# **Reference.** (1) For an elastic strip with $d/W=0.5$ the stress at the hole edge is
# $K_{t,n}\,\sigma_\text{net}$ with $\sigma_\text{net}=\sigma\,W/(W-d)$ and $K_{t,n}=2.156$ (Pilkey and
# Pilkey, *Peterson's Stress Concentration Factors*, 3rd ed., Chart 4.1). (2) Yield therefore starts
# at $\sigma=\sigma_y(1-d/W)/K_{t,n}$. (3) Without hardening the plate collapses when the whole net
# section yields, at $\sigma=\sigma_y(1-d/W)=0.5\,\sigma_y$. (4) Wherever the material yields, the
# von Mises stress equals the current yield stress $\sigma_y+H\bar\varepsilon_p$. (5) If the plate
# unloads elastically, the residual stress is the peak stress minus the elastic solution for the
# peak load (Melan's superposition).

import numpy as np
from onesim.cad.exchange import load
from onesim.materials.solids import IsotropicHardening, J2Plasticity
from onesim.meshing import TriangularMesher
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import DampedNewton
from onesim.physics.solid_mechanics import SmallStrainSolid
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the quarter plate with its named faces; `section` cuts its mid-plane. The
# quadratic triangles are 3 mm at the hole and grow to 15 mm.

quarter = load("holed_plate_elastoplastic_plane_stress.step")
faces = quarter.boundaries
plane = quarter.section((0, 0, 1), origin=(0, 0, quarter.size[2] / 2))
edges = plane.boundaries
mesh = TriangularMesher(
    target_size=0.015,
    size_field=lambda p: np.minimum(
        0.003 + 0.08 * np.abs(np.hypot(p[:, 0], p[:, 1]) - 0.05), 0.015
    ),
).generate(plane)

fig = plots.figure(1, 2, size=(14, 7))
plots.geometry(
    quarter,
    selection={
        "hole": faces.hole,
        "loaded end": faces.loaded,
        "symmetry": faces.symmetry_x | faces.symmetry_y,
    },
    labels=False,
    camera=plots.Camera(elevation=60, azimuth=-70),
    title="Quarter of the perforated strip",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"Mid-plane section: {mesh.cell_count} quadratic triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# `SmallStrainSolid` in plane stress with a `J2Plasticity` material; the out-of-plane condensation
# runs at every quadrature point. The end traction follows the load history step by step.

youngs_modulus, poisson_ratio, yield_stress, hardening = 68.7e9, 0.2, 238e6, 2.21e9
width, diameter = 0.200, 0.100
history = np.r_[0.0, np.arange(0.2, 0.551, 0.025), 0.40, 0.25, 0.10, 0.0] * yield_stress
result = SmallStrainSolid(
    youngs_modulus=youngs_modulus,
    poisson_ratio=poisson_ratio,
    kinematics="plane_stress",
    thickness=quarter.size[2],
    material=J2Plasticity(
        youngs_modulus / (2 * (1 + poisson_ratio)),
        IsotropicHardening(yield_stress, hardening),
    ),
).solve(
    mesh=mesh,
    times=np.arange(len(history), dtype=float),
    order=2,
    displacements=((edges.symmetry_x, "x", 0.0), (edges.symmetry_y, "y", 0.0)),
    tractions=((edges.loaded, lambda x, y, t: (0.0, history[int(round(t))])),),
    nonlinear=DampedNewton(
        linear_solver=SparseDirect(),
        relative_tolerance=1e-12,
        absolute_tolerance=1e-6,
        maximum_iterations=40,
    ),
)
points = result.quadrature_coordinates
edge = int(np.argmin(np.hypot(points[:, 0] - diameter / 2, points[:, 1])))
plastic = np.array(
    [
        [point.material.equivalent_plastic_strain for cell in step for point in cell]
        for step in result.material_history
    ]
)
deviator = (
    result.stress
    - np.trace(result.stress, axis1=2, axis2=3)[..., None, None] * np.eye(3) / 3
)
mises = np.sqrt(1.5 * np.sum(deviator**2, axis=(2, 3)))
top = np.isclose(result.space.dof_coordinates[:, 1], 0.180)
stretch = result.displacement[:, top, 1].mean(axis=1)

# ## Comparison with the handbook values
#
# The stress at the hole edge in the first, still elastic step; the load at which the edge yields;
# the net-section limit; and the von Mises stress at the hole edge against the hardened yield stress
# at peak load and after unloading.

peterson, ratio = 2.156, 1 - diameter / width
onset = yield_stress * ratio / peterson
peak, final = int(np.argmax(history)), len(history) - 1
print(
    f"hole edge at σ = {history[1] / 1e6:.1f} MPa: σ_yy = {result.stress[1, edge, 1, 1] / 1e6:.1f} MPa, "
    f"K_t,n = {result.stress[1, edge, 1, 1] * ratio / history[1]:.3f} (Peterson {peterson}, point {1e3 * (np.hypot(*points[edge]) - diameter / 2):.2f} mm inside the edge)"
)
first = int(np.argmax(plastic.max(axis=1) > 0))
print(
    f"first yield between σ = {history[first - 1] / yield_stress:.3f} σ_y and {history[first] / yield_stress:.3f} σ_y (reference {onset / yield_stress:.3f} σ_y)"
)
ligament = np.abs(points[:, 1]) < 0.004
print(
    f"net section without hardening collapses at σ = {ratio:.2f} σ_y; at peak load {history[peak] / yield_stress:.3f} σ_y "
    f"{100 * np.mean(plastic[peak][ligament] > 0):.0f} % of the ligament points have yielded"
)
for k, label in ((peak, "peak load"), (final, "after unloading")):
    print(
        f"{label}: hole edge σ_yy = {result.stress[k, edge, 1, 1] / 1e6:7.1f} MPa, von Mises {mises[k, edge] / 1e6:6.1f} MPa, "
        f"current yield stress σ_y + H ε_p = {(yield_stress + hardening * plastic[k, edge]) / 1e6:6.1f} MPa, ε_p = {100 * plastic[k, edge]:.3f} %"
    )
print(
    f"residual stress at the hole edge by elastic superposition: {(result.stress[peak, edge, 1, 1] - result.stress[1, edge, 1, 1] / history[1] * history[peak]) / 1e6:.1f} MPa "
    f"(simulation {result.stress[final, edge, 1, 1] / 1e6:.1f} MPa): the unloading stayed elastic"
)
print(
    f"permanent elongation of the quarter after unloading: {1e6 * stretch[-1]:.1f} µm; largest out-of-plane stress {np.abs(result.stress[..., 2, 2]).max() / 1e3:.2f} kPa"
)

# ## Figure
#
# Left the end tension against the stress at the hole edge, with the elastic concentration, the
# onset of yield and the net-section limit; in the middle against the elongation of the quarter;
# right the yielded quadrature points at three loads. The plastic zone starts at the hole edge and
# crosses the ligament around the net-section limit, where the plate softens to the hardening slope.
# On unloading the hole edge springs back elastically into compression.

fig = plots.figure(1, 3, size=(19, 6))
plots.line(
    y={
        "simulation, loading and unloading": (
            result.stress[:, edge, 1, 1] / 1e6,
            history / 1e6,
        )
    },
    markers=True,
    reference={
        "elastic concentration": (
            np.array([0.0, 1.1 * yield_stress]) / 1e6,
            np.array([0.0, 1.1 * yield_stress]) * ratio / peterson / 1e6,
        )
    },
    hlines={
        "onset of yield": onset / 1e6,
        "net-section limit": ratio * yield_stress / 1e6,
    },
    xlabel="σ_yy at the hole edge [MPa]",
    ylabel="end tension σ [MPa]",
    title="Stress at the hole edge",
    legend="upper left",
    ax=fig[0],
)
plots.line(
    y={"simulation, loading and unloading": (1e6 * stretch, history / 1e6)},
    markers=True,
    hlines={
        "onset of yield": onset / 1e6,
        "net-section limit": ratio * yield_stress / 1e6,
    },
    xlabel="elongation of the quarter [µm]",
    ylabel="end tension σ [MPa]",
    title="Load-elongation curve",
    legend="lower right",
    ax=fig[1],
)
plots.line(
    points={
        f"yielded at σ = {history[k] / yield_stress:.3f} σ_y": (
            1e3 * points[plastic[k] > 0, 0],
            1e3 * points[plastic[k] > 0, 1],
        )
        for k in (peak, 9, first + 1)
    },
    xlimits=(0, 100),
    ylimits=(0, 100),
    xlabel="x [mm]",
    ylabel="y [mm]",
    title="Spread of the plastic zone",
    legend="upper right",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Set the hardening modulus to zero and drive the end by displacement to see the load level off at
# the net-section limit, add kinematic hardening to model the Bauschinger effect on unloading, or
# repeat the cycle and watch whether the hole edge shakes down or keeps yielding.
