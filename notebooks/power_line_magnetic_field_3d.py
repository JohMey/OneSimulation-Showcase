"""OneSimulation example: magnetic field of a three-phase 400 kV power line over three spans (3D).
Requires the installed, activated OneSimulation runtime.

Towers and sagging phase conductors are read from STEP files; the 3D magnetic field of the 50 Hz
currents is integrated with the Biot-Savart law along the conductor lines. The rms flux density
1 m above ground is mapped over the span and compared with the two-dimensional field of straight
lines at mid-span and at the tower; a quadrature study checks the integration.
"""

# # Magnetic field of a three-phase power line over three spans
#
# A 400 kV line carries its three phase bundles in one horizontal row, 9 m apart. Between two towers
# 300 m apart the conductors sag from the crossarm at 30 m to 20 m above ground at mid-span, so the
# field on the ground changes along the span: it is strongest under the lowest point. Each phase
# carries $I=1000$ A (rms) of a symmetric three-phase system,
# $i_k(t)=\sqrt2I\cos(\omega t-2\pi k/3)$.
#
# Air and ground are not magnetic, and at 50 Hz the currents induced in the ground spread over a
# depth of the order of a kilometre (Carson), so the field near the line is the free-space field of
# the conductor currents. OneSimulation integrates it in 3D with the Biot-Savart law along the
# centre lines of the conductors,
#
# $$\mathbf B(\mathbf r)=\frac{\mu_0}{4\pi}\sum_k i_k\int\frac{\mathrm d\boldsymbol\ell_k\times(\mathbf r-\mathbf r')}{|\mathbf r-\mathbf r'|^3},$$
#
# over the middle span and the two neighbouring ones. The steel of the towers is left out, and so
# are the shield wires, which carry no current in this model. The rms flux density follows from
# the two quadrature parts, $B_\mathrm{rms}^2=\tfrac12\big(|\mathbf B(0)|^2+|\mathbf B(90°)|^2\big)$.
#
# The towers come from `power_line_magnetic_field_3d.step`, the conductors from
# `power_line_magnetic_field_3d_phase_a.step`, `_phase_b.step` and `_phase_c.step`. The script
# `power_line_magnetic_field_3d_geometry.py` builds them with the CAD objects of OneSimulation.
#
# **Reference.** Far from the towers the conductors are nearly straight and horizontal; the field of
# long straight currents at the local conductor height (Ampère's law, superposed; CIGRE Technical
# Brochure 373, 2009) is then a close estimate:
#
# $$\mathbf B=\frac{\mu_0}{2\pi}\sum_k i_k\,\frac{\hat{\mathbf x}\times(\mathbf r-\mathbf r_k)}{|\mathbf r-\mathbf r_k|^2}.$$
#
# At mid-span it uses the lowest height (20 m), at the tower the attachment height (30 m); the sag
# makes the 3D field differ from both. A study over the number of line pieces checks the
# integration.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VacuumPermeability
from onesim.physics.magnetostatics import magnetic_induction, polygon_line_quadrature
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and line discretization
#
# `load` returns the towers and the conductors. Span, attachment height and lowest point are read
# from the parts; the conductor centre lines are parabolic arcs through them. Each span is divided
# into straight pieces carrying a two-point Gauss rule.

towers = load("power_line_magnetic_field_3d.step")
phases = [
    load(f"power_line_magnetic_field_3d_{name}.step")
    for name in ("phase_a", "phase_b", "phase_c")
]
bundle = 0.2
span = towers.size[0] - 2.0
lowest = phases[0].bounding_box[0][2] + bundle
attach = phases[0].bounding_box[1][2] - bundle
lateral = [0.5 * (part.bounding_box[0][1] + part.bounding_box[1][1]) for part in phases]
sag_height = (
    lambda x: lowest
    + 4 * (attach - lowest) * ((np.mod(x + span / 2, span) - span / 2) / span) ** 2
)
gauss, gauss_weights = (
    np.array((0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3))),
    np.array((0.5, 0.5)),
)
pieces = (8, 32, 128)
lines = {
    n: [
        polygon_line_quadrature(
            np.column_stack((x, np.full_like(x, y), sag_height(x))),
            gauss,
            gauss_weights,
            closed=False,
        )
        for y in lateral
        for x in (np.linspace(-1.5 * span, 1.5 * span, 3 * n + 1),)
    ]
    for n in pieces
}

fig = plots.figure(1, 2, size=(15, 5.5))
plots.geometry(
    [
        towers.named("towers"),
        *(
            part.named(name)
            for part, name in zip(phases, ("phase a", "phase b", "phase c"))
        ),
    ],
    camera=plots.Camera(elevation=20, azimuth=-60),
    title="Towers and conductors over three spans",
    ax=fig[0],
)
plots.line(
    points={
        f"{n} pieces per span": (lines[n][0].points[:, 0], lines[n][0].points[:, 2])
        for n in pieces[:2]
    },
    vlines=[-span / 2, span / 2],
    xlabel="x along the line [m]",
    ylabel="height [m]",
    title="Quadrature points on phase a",
    ax=fig[1],
)
plots.show()

# ## Field of the three phases
#
# The field is linear in the currents: the field of each phase for 1 A is integrated once, and the
# instantaneous fields at $\omega t=0$ and $90°$ are their weighted sums.

current = 1000.0
mu0 = float(VacuumPermeability().evaluate())
weights = {
    phase: np.sqrt(2) * current * np.cos(phase - 2 * np.pi * np.arange(3) / 3)
    for phase in (0.0, np.pi / 2)
}
x_grid, y_grid = np.linspace(-150.0, 150.0, 31), np.linspace(-60.0, 60.0, 81)
ground = np.stack(np.meshgrid(x_grid, y_grid, indexing="ij"), axis=-1).reshape(-1, 2)
ground = np.column_stack((ground, np.ones(len(ground))))
unit_fields = {
    n: [
        magnetic_induction(ground, line.points, line.line_elements, permeability=mu0)
        for line in lines[n]
    ]
    for n in pieces
}
rms = {
    n: np.sqrt(
        0.5
        * sum(
            np.sum(
                sum(w * B for w, B in zip(weights[phase], unit_fields[n])) ** 2, axis=1
            )
            for phase in weights
        )
    )
    for n in pieces
}

# ## Results
#
# The quadrature study at the strongest point, the field under the lowest point and at the tower
# against the straight-line estimate, and the change along the span.

line_field = (
    lambda h, i: mu0
    / (2 * np.pi)
    * np.sum(
        i * 1j / np.conj(y_grid[:, None] + 1j - (np.array(lateral) + 1j * h)), axis=1
    )
)
straight = lambda h: np.sqrt(
    0.5 * sum(np.abs(line_field(h, i)) ** 2 for i in weights.values())
)
finest = rms[pieces[-1]].reshape(len(x_grid), len(y_grid))
peak = np.argmax(rms[pieces[-1]])
print("pieces per span   largest rms flux density 1 m above ground [µT]")
for n in pieces:
    print(
        f"{n:15d} {1e6 * rms[n][peak]:20.4f}   ({100 * (rms[n][peak] / rms[pieces[-1]][peak] - 1):+.4f} % from the finest)"
    )
mid, tower = np.argmin(np.abs(x_grid)), np.argmin(np.abs(x_grid - span / 2))
for label, row, z in (("mid-span", mid, lowest), ("at the tower", tower, attach)):
    estimate = straight(z)
    print(
        f"{label:13s}: largest {1e6 * finest[row].max():.3f} µT, straight lines at {z:.0f} m {1e6 * estimate.max():.3f} µT "
        f"({100 * (finest[row].max() / estimate.max() - 1):+.1f} %)"
    )
print(
    f"the field under the lowest point is {finest[mid].max() / finest[tower].max():.2f} times the field at the tower; "
    "public reference level (ICNIRP 2010, 50 Hz): 200 µT"
)

# ## Figure
#
# The rms flux density 1 m above ground over the middle span, and the profiles across the line at
# mid-span and at the tower against the straight-line estimates.

fig = plots.figure(1, 2, size=(16, 6))
plots.heatmap(
    x_grid,
    y_grid,
    1e6 * finest,
    limits=(0, 1e6 * finest.max()),
    contours=10,
    label="rms |B| 1 m above ground [µT]",
    xlabel="x along the line [m]",
    ylabel="y across the line [m]",
    title="Flux density over the span",
    aspect="auto",
    ax=fig[0],
)
plots.line(
    points={
        "3D, mid-span": (y_grid[::4], 1e6 * finest[mid, ::4]),
        "3D, at the tower": (y_grid[::4], 1e6 * finest[tower, ::4]),
    },
    reference={
        f"straight lines at {lowest:.0f} m": (y_grid, 1e6 * straight(lowest)),
        f"straight lines at {attach:.0f} m": (y_grid, 1e6 * straight(attach)),
    },
    xlabel="y across the line [m]",
    ylabel="rms flux density 1 m above ground [µT]",
    title="Profiles across the line",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Add a second circuit with reversed phase order in `power_line_magnetic_field_3d_geometry.py` to
# see the field drop away from the line, give the shield wires their induced currents, or raise
# the lowest point to find the height at which the field under the line stays below a design value.
