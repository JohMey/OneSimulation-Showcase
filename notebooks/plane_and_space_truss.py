"""OneSimulation example: member forces and deflection of a plane truss and of its space variant.
Requires the installed, activated OneSimulation runtime.

A pin-jointed steel truss carries a load at mid-span, first as a plane truss and then as a space
truss with a triangular cross-section. Both are assembled from axial bar elements; the member
forces are checked against the method of joints and the mid-span deflection against the unit-load
method.
"""

# # Plane and space truss
#
# A truss is a framework of straight bars joined by pins. Its members carry only axial forces, no
# bending and no torsion, so each bar is one element with two nodes and three displacements per
# joint. This notebook first solves a **plane truss**: a 6 m span with two top joints, simply
# supported and loaded by $F=100$ kN at the middle of the bottom chord. The same truss is then
# built as a **space truss**: two inclined side trusses share the top chord, three ties and two
# braces join the two bottom chords, and the load is split between the two loaded bottom joints.
#
# | quantity | value |
# |---|---|
# | panel length $a$, truss height $h$ | 3 m, 2 m |
# | half width of the space truss $b$ | 1 m |
# | steel bars | $E=200$ GPa, $A=10$ cm² |
# | load | $F=100$ kN at mid-span |
#
# | joint (plane truss) | x [m] | z [m] | support |
# |---|---|---|---|
# | 0 | 0 | 0 | pin: $u_x=u_z=0$ |
# | 1 | 3 | 0 | load $F$ |
# | 2 | 6 | 0 | roller: $u_z=0$ |
# | 3 | 1.5 | 2 | |
# | 4 | 4.5 | 2 | |
#
# In the space truss joints 0, 1, 2 lie at $y=-b$, joints 5, 6, 7 at $y=+b$, and the top joints 3
# and 4 at $y=0$; joints 1 and 6 carry $F/2$ each.
#
# **Reference.** The truss is statically determinate, so the **method of joints** gives every
# member force from equilibrium alone (Hibbeler, *Structural Analysis*, ch. 3). With the diagonal
# angle $\tan\theta=2h/a$:
#
# $$N_\text{diagonal}=\pm\frac{F}{2\sin\theta},\qquad N_\text{bottom}=\frac{F}{2\tan\theta},\qquad
#   N_\text{top}=-\frac{F}{\tan\theta}.$$
#
# Each side of the space truss is the same plane truss, inclined by $\tan\beta=b/h$ and of height
# $h'=\sqrt{h^2+b^2}$; it carries the in-plane load $F'=F/(2\cos\beta)$, the shared top chord
# collects both sides, and the ties take the horizontal components $\mp\tfrac F2\tan\beta$ at the
# loaded joints and $\tfrac F4\tan\beta$ at the supports; the braces of the bottom plane stay free of
# force under this symmetric load. The **unit-load method** (virtual work)
# gives the deflection under the load from the member forces,
# $\delta=\sum_i N_i^2L_i/(EAF)$; in the space truss it is the mean of the two loaded joints.

import numpy as np
from onesim.physics.solid_mechanics import FrameMember, SpatialFrame
from onesim.visualization import plots, theme

theme.set()

# ## Truss layouts
#
# The joints are given as a table of coordinates; every member names its two end joints. Each bar
# is a `FrameMember` with only $E$ and $A$, which makes it an axial truss element; `SpatialFrame`
# joins the bars at shared joints.

a, h, b, force, youngs_modulus, area = 3.0, 2.0, 1.0, 100e3, 200e9, 10e-4
plane_joints = np.array(
    [[0, 0, 0], [a, 0, 0], [2 * a, 0, 0], [a / 2, 0, h], [1.5 * a, 0, h]]
)
plane_bars = {
    "bottom 0-1": (0, 1),
    "bottom 1-2": (1, 2),
    "top 3-4": (3, 4),
    "diagonal 0-3": (0, 3),
    "diagonal 3-1": (3, 1),
    "diagonal 1-4": (1, 4),
    "diagonal 4-2": (4, 2),
}
space_joints = np.array(
    [
        [0, -b, 0],
        [a, -b, 0],
        [2 * a, -b, 0],
        [a / 2, 0, h],
        [1.5 * a, 0, h],
        [0, b, 0],
        [a, b, 0],
        [2 * a, b, 0],
    ]
)
space_bars = {
    "bottom 0-1": (0, 1),
    "bottom 1-2": (1, 2),
    "bottom 5-6": (5, 6),
    "bottom 6-7": (6, 7),
    "top 3-4": (3, 4),
    "diagonal 0-3": (0, 3),
    "diagonal 3-1": (3, 1),
    "diagonal 1-4": (1, 4),
    "diagonal 4-2": (4, 2),
    "diagonal 5-3": (5, 3),
    "diagonal 3-6": (3, 6),
    "diagonal 6-4": (6, 4),
    "diagonal 4-7": (4, 7),
    "tie 0-5": (0, 5),
    "tie 1-6": (1, 6),
    "tie 2-7": (2, 7),
    "brace 0-6": (0, 6),
    "brace 6-2": (6, 2),
}
plane = SpatialFrame(
    joints=plane_joints,
    members=tuple(
        FrameMember(name=name, start=s, end=e, youngs_modulus=youngs_modulus, area=area)
        for name, (s, e) in plane_bars.items()
    ),
)
space = SpatialFrame(
    joints=space_joints,
    members=tuple(
        FrameMember(name=name, start=s, end=e, youngs_modulus=youngs_modulus, area=area)
        for name, (s, e) in space_bars.items()
    ),
)

drawing = {
    name: np.concatenate(
        (joints[np.array(list(bars.values()))], np.full((len(bars), 1, 3), np.nan)),
        axis=1,
    ).reshape(-1, 3)
    for name, joints, bars in (
        ("plane", plane_joints, plane_bars),
        ("space", space_joints, space_bars),
    )
}
fig = plots.figure(1, 3, size=(17, 4.5))
plots.line(
    None,
    {"bars": (drawing["plane"][:, 0], drawing["plane"][:, 2])},
    points={"joints": (plane_joints[:, 0], plane_joints[:, 2])},
    legend=False,
    xlabel="x [m]",
    ylabel="z [m]",
    ylimits=(-0.5, 2.5),
    title="Plane truss, 7 bars",
    ax=fig[0],
)
plots.line(
    None,
    {"bars": (drawing["space"][:, 0], drawing["space"][:, 1])},
    points={"joints": (space_joints[:, 0], space_joints[:, 1])},
    legend=False,
    xlabel="x [m]",
    ylabel="y [m]",
    ylimits=(-1.5, 1.5),
    title="Space truss, plan view, 18 bars",
    ax=fig[1],
)
plots.line(
    None,
    {"bars": (drawing["space"][:, 1], drawing["space"][:, 2])},
    points={"joints": (space_joints[:, 1], space_joints[:, 2])},
    legend=False,
    xlabel="y [m]",
    ylabel="z [m]",
    xlimits=(-2, 2),
    ylimits=(-0.5, 2.5),
    title="Space truss, end view",
    ax=fig[2],
)
plots.show()

# ## Model
#
# The plane truss is held at $u_y=0$ in every joint, which keeps it in its plane. The space truss
# rests on its four corner joints: joint 0 is fixed, joint 2 slides in $x$, and joints 5 and 7
# slide in both horizontal directions.

plane_result = plane.solve(
    supports=[(0, "x", 0), (0, "z", 0), (2, "z", 0)]
    + [(joint, "y", 0) for joint in range(5)],
    forces={1: (0, 0, -force)},
)
space_result = space.solve(
    supports=[
        (0, "x", 0),
        (0, "y", 0),
        (0, "z", 0),
        (2, "y", 0),
        (2, "z", 0),
        (5, "z", 0),
        (7, "z", 0),
    ],
    forces={1: (0, 0, -force / 2), 6: (0, 0, -force / 2)},
)
axial = {
    name: {bar: result.members[bar].local_end_actions[1] for bar in bars}
    for name, result, bars in (
        ("plane", plane_result, plane_bars),
        ("space", space_result, space_bars),
    )
}

# ## Results
#
# Member forces of both trusses against the method of joints (tension positive) and the
# deflection under the load against the unit-load method.

theta, tilt = np.arctan(2 * h / a), np.arctan(b / h)
side_load, side_theta = force / (2 * np.cos(tilt)), np.arctan(2 * np.hypot(h, b) / a)
joints_method = lambda load, angle: {
    "bottom": load / (2 * np.tan(angle)),
    "top": -load / np.tan(angle),
    "diagonal": load / (2 * np.sin(angle)),
}
plane_reference = joints_method(force, theta)
side_reference = joints_method(side_load, side_theta)
reference = {
    "plane": {
        bar: plane_reference[bar.split()[0]]
        * (-1 if bar in ("diagonal 0-3", "diagonal 4-2") else 1)
        for bar in plane_bars
    },
    "space": {
        **{
            bar: side_reference[bar.split()[0]]
            * (-1 if bar[-3:] in ("0-3", "4-2", "5-3", "4-7") else 1)
            for bar in space_bars
            if bar.split()[0] in side_reference
        },
        "top 3-4": 2 * side_reference["top"],
        "tie 0-5": force / 4 * np.tan(tilt),
        "tie 1-6": -force / 2 * np.tan(tilt),
        "tie 2-7": force / 4 * np.tan(tilt),
        "brace 0-6": 0.0,
        "brace 6-2": 0.0,
    },
}
lengths = {
    name: {bar: np.linalg.norm(joints[e] - joints[s]) for bar, (s, e) in bars.items()}
    for name, joints, bars in (
        ("plane", plane_joints, plane_bars),
        ("space", space_joints, space_bars),
    )
}
unit_load = {
    name: sum(reference[name][bar] ** 2 * lengths[name][bar] for bar in reference[name])
    / (youngs_modulus * area * force)
    for name in reference
}

for name in ("plane", "space"):
    print(f"{name} truss          simulation [kN]   method of joints [kN]")
    for bar in axial[name]:
        print(
            f"  {bar:16s} {axial[name][bar] / 1e3:14.3f} {reference[name][bar] / 1e3:20.3f}"
        )
deflections = {
    "plane": -plane_result.displacement[1, 2],
    "space": -space_result.displacement[[1, 6], 2].mean(),
}
for name, value in deflections.items():
    print(
        f"{name} truss deflection under the load: simulation {1e3 * value:.4f} mm, unit-load method "
        f"{1e3 * unit_load[name]:.4f} mm, deviation {100 * (value / unit_load[name] - 1):+.1e} %"
    )
print(
    f"vertical reactions of the space truss: {space_result.reaction_force[[0, 2, 5, 7], 2] / 1e3} kN"
)

# ## Figure
#
# The deformed trusses (displacements magnified 100 times) and the member forces against the
# method of joints. The diagonals running down towards the load are in tension, the outer
# diagonals and the top chord in compression.

scale = 100
deformed = {
    name: np.concatenate(
        (
            (joints + scale * result.displacement)[np.array(list(bars.values()))],
            np.full((len(bars), 1, 3), np.nan),
        ),
        axis=1,
    ).reshape(-1, 3)
    for name, joints, bars, result in (
        ("plane", plane_joints, plane_bars, plane_result),
        ("space", space_joints, space_bars, space_result),
    )
}
fig = plots.figure(2, 2, size=(16, 10))
plots.line(
    None,
    {"deformed (×100)": (deformed["plane"][:, 0], deformed["plane"][:, 2])},
    reference={"undeformed": (drawing["plane"][:, 0], drawing["plane"][:, 2])},
    ylimits=(-1.2, 2.6),
    legend="lower center",
    xlabel="x [m]",
    ylabel="z [m]",
    title="Plane truss",
    ax=fig[0],
)
plots.line(
    None,
    {"deformed (×100)": (deformed["space"][:, 0], deformed["space"][:, 2])},
    reference={"undeformed": (drawing["space"][:, 0], drawing["space"][:, 2])},
    ylimits=(-1.2, 2.6),
    legend="lower center",
    xlabel="x [m]",
    ylabel="z [m]",
    title="Space truss, side view",
    ax=fig[1],
)
plots.bars(
    list(plane_bars),
    {"simulation": [axial["plane"][bar] / 1e3 for bar in plane_bars]},
    reference={
        "method of joints": [reference["plane"][bar] / 1e3 for bar in plane_bars]
    },
    ylabel="axial force [kN]",
    title="Plane truss: member forces",
    ax=fig[2],
)
ranked = sorted(space_bars, key=axial["space"].get)
plots.bars(
    ranked,
    {"simulation": [axial["space"][bar] / 1e3 for bar in ranked]},
    reference={"method of joints": [reference["space"][bar] / 1e3 for bar in ranked]},
    horizontal=True,
    ylabel="axial force [kN]",
    title="Space truss: member forces",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# Change the joint table, for example a higher truss or more panels, and the solver follows; a
# statically indeterminate truss (an extra diagonal) needs no change at all, while the method of
# joints no longer suffices. Give the members their bending stiffness (`bending_inertia_y`, ...)
# and the same frame becomes a rigid-jointed frame with secondary moments.
