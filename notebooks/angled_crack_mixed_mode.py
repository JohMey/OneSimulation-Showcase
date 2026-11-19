"""OneSimulation example: mixed-mode stress intensity factors of an angled crack in a plate.
Requires the installed, activated OneSimulation runtime.

A steel plate with an inclined central crack is pulled at its ends. The plate is read from a
STEP file for four crack angles and cut to its cross-section; the opening and sliding stress
intensity factors K_I and K_II follow from the interaction integral with the Williams crack-tip
fields, on two rings around the tip, and are compared with the handbook solution.
"""

# # Mixed-mode stress intensity of an angled crack
#
# A crack that is not perpendicular to the load is opened and sheared at the same time. Near its
# tip the stresses are the sum of the two singular fields of Williams, weighted by the stress
# intensity factors $K_I$ (opening) and $K_{II}$ (in-plane sliding); both decide whether and in
# which direction the crack grows.
#
# The **interaction integral** separates them. It superposes the computed field with an
# auxiliary Williams field of a single mode and evaluates the J-integral of the sum as a domain
# integral over a ring around the tip; the cross term is proportional to the stress intensity
# factor of that mode:
#
# $$I^{(m)}=\int_A\Big(\sigma_{ij}\frac{\partial u^{(m)}_i}{\partial x_1}+\sigma^{(m)}_{ij}\frac{\partial u_i}{\partial x_1}
#   -\sigma_{kl}\varepsilon^{(m)}_{kl}\,\delta_{1j}\Big)\frac{\partial q}{\partial x_j}\,\mathrm dA,\qquad
#   K_m=\tfrac12E'\,I^{(m)},$$
#
# with $x_1$ along the crack, the virtual advance $q$ and $E'=E/(1-\nu^2)$ in plane strain.
#
# **Setup.** A steel plate ($E=200$ GPa, $\nu=0.3$) 200 mm × 200 mm with a central crack of length
# $2a=20$ mm at the angle $\beta=0°$, 30°, 45° and 60° to the plate edges, pulled by
# $\sigma=100$ MPa; plane strain. The plates come from the CAD files
# `angled_crack_mixed_mode_<β>.step`, written by `angled_crack_mixed_mode_geometry.py` with the CAD
# objects of OneSimulation; the crack is a slender cut with sharp tips.
#
# | side | condition |
# |---|---|
# | `y_min` | roller, $u_y=0$ |
# | `x_min` | roller, $u_x=0$ (the plate contracts freely) |
# | `y_max` | tension $\sigma$ |
# | `x_max`, `crack.side` | traction free |
#
# **Reference.** For a crack in a large plate, Y. Murakami (ed.), *Stress Intensity Factors
# Handbook*, Pergamon 1987, Case 1.1 (after Erdogan and Sih):
# $K_I=\sigma\sqrt{\pi a}\cos^2\beta$, $K_{II}=\sigma\sqrt{\pi a}\sin\beta\cos\beta$. With
# $a$ one tenth of the half width, the finite plate raises both by about 0.6 %.

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import TriangularMesher
from onesim.physics.solid_mechanics import (
    SmallStrainSolid,
    crack_tip_advance,
    small_strain_interaction_integrals,
    williams_auxiliary_fields,
)
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns each plate, `section` its cross-section with the names of the cut faces. The
# triangles are 0.25 mm at both crack tips and grow to 15 mm away from them.

angles = (0, 30, 45, 60)
plates = {beta: load(f"angled_crack_mixed_mode_{beta}.step") for beta in angles}
side, _, thickness = plates[0].size
sections = {
    beta: plate.section((0, 0, 1), origin=(0, 0, thickness / 2))
    for beta, plate in plates.items()
}
half_length = 0.010
direction = {
    beta: np.array([np.cos(np.radians(beta)), np.sin(np.radians(beta))])
    for beta in angles
}
tips = {beta: side / 2 + half_length * direction[beta] for beta in angles}
meshes = {
    beta: TriangularMesher(
        target_size=0.015,
        size_field=lambda p, beta=beta: np.minimum(
            0.00025
            + 0.25
            * np.minimum(
                np.hypot(*(p[:, :2] - tips[beta]).T),
                np.hypot(*(p[:, :2] - (side - tips[beta])).T),
            ),
            0.015,
        ),
    ).generate(section)
    for beta, section in sections.items()
}

fig = plots.figure(1, 2, size=(13, 6.5))
plots.geometry(
    plates[30],
    selection={
        "rollers": plates[30].boundaries.y_min | plates[30].boundaries.x_min,
        "pulled": plates[30].boundaries.y_max,
        "crack": plates[30].boundaries["crack.side"],
    },
    labels=False,
    camera=plots.Camera(elevation=55, azimuth=-60),
    title="Plate with the crack at 30°",
    ax=fig[0],
)
plots.mesh(
    meshes[30],
    groups=("all",),
    legend=False,
    title=f"β = 30°: {meshes[30].cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# Plane strain with quadratic triangles. For each angle the Williams fields are set up in the
# frame of the right-hand tip, and the interaction integral is evaluated on two rings,
# 1.5-4 mm and 2.5-6 mm around it.

stress, youngs_modulus, poisson = 100e6, 200e9, 0.3
steel = SmallStrainSolid(
    youngs_modulus=youngs_modulus,
    poisson_ratio=poisson,
    kinematics="plane_strain",
    thickness=thickness,
)
results = {
    beta: steel.solve(
        mesh=meshes[beta],
        times=(0.0,),
        order=2,
        tractions=((section.boundaries.y_max, (0.0, stress)),),
        displacements=(
            (section.boundaries.y_min, "y", 0.0),
            (section.boundaries.x_min, "x", 0.0),
        ),
    )
    for beta, section in sections.items()
}
rings = ((0.0015, 0.004), (0.0025, 0.006))
intensity = {}
for beta, result in results.items():
    fields = williams_auxiliary_fields(
        tips[beta], np.radians(beta), youngs_modulus, poisson, "plane_strain"
    )
    intensity[beta] = [
        0.5
        * fields.effective_modulus
        * small_strain_interaction_integrals(
            result, crack_tip_advance(tips[beta], direction[beta], *ring), fields
        )
        for ring in rings
    ]

# ## Comparison with the reference
#
# $K_I$ and $K_{II}$ on both rings for every angle against the handbook values.

scale = stress * np.sqrt(np.pi * half_length)
murakami = lambda beta: scale * np.array(
    [np.cos(np.radians(beta)) ** 2, np.sin(np.radians(beta)) * np.cos(np.radians(beta))]
)
print(
    f"{'β':>4s} {'triangles':>9s} {'K_I ring 1':>11s} {'ring 2':>8s} {'handbook':>9s} {'dev.':>7s}   {'K_II ring 1':>11s} {'ring 2':>8s} {'handbook':>9s} {'dev.':>7s}  [MPa m^0.5]"
)
for beta in angles:
    ring1, ring2 = (np.asarray(values) / 1e6 for values in intensity[beta])
    exact = murakami(beta) / 1e6
    deviation = [
        (
            f"{100 * ((ring1[k] + ring2[k]) / 2 / exact[k] - 1):+6.2f}%"
            if exact[k]
            else f"{'—':>7s}"
        )
        for k in range(2)
    ]
    print(
        f"{beta:3d}° {meshes[beta].cell_count:9d} {ring1[0]:11.3f} {ring2[0]:8.3f} {exact[0]:9.3f} {deviation[0]}"
        f"   {ring1[1]:11.3f} {ring2[1]:8.3f} {exact[1]:9.3f} {deviation[1]}"
    )

# ## Figure
#
# The vertical displacement around the crack at 45°, and both stress intensity factors over the
# crack angle with the handbook curves.

result, nodes = results[45], len(meshes[45].coordinates)
beta_line = np.linspace(0, 90, 181)
fig = plots.figure(1, 2, size=(15, 6))
plots.heatmap(
    1e3 * result.coordinates[:nodes, 0],
    1e3 * result.coordinates[:nodes, 1],
    1e6 * result.displacement[0, :nodes, 1],
    triangles=np.asarray(meshes[45].cells),
    limits=(25, 60),
    contours=12,
    label="u_y [µm]",
    xlabel="x [mm]",
    ylabel="y [mm]",
    xlimits=(80, 120),
    ylimits=(80, 120),
    title="Vertical displacement around the crack at 45°",
    ax=fig[0],
)
plots.line(
    points={
        f"K_{mode}, ring {k + 1}": (
            np.array(angles),
            [intensity[beta][k][m] / scale for beta in angles],
        )
        for m, mode in enumerate(("I", "II"))
        for k in range(2)
    },
    reference={
        "K_I, handbook": (beta_line, np.cos(np.radians(beta_line)) ** 2),
        "K_II, handbook": (
            beta_line,
            np.sin(np.radians(beta_line)) * np.cos(np.radians(beta_line)),
        ),
    },
    xlabel="crack angle β [°]",
    ylabel="K / (σ √(πa))",
    title="Stress intensity factors over the crack angle",
    legend="upper right",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# The two factors give the direction of crack growth, for example by the maximum tangential
# stress criterion (`maximum_tangential_stress_angle`). Lengthen the crack in
# `angled_crack_mixed_mode_geometry.py` to see the finite width raise both factors.
