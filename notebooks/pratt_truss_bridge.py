"""OneSimulation example: member forces and deflection of a Pratt truss bridge.
Requires the installed, activated OneSimulation runtime.

A 24 m steel through-truss bridge with two Pratt trusses, floor beams and lateral bracing carries
its deck load at the panel points. One truss is solved as a pin-jointed plane truss and checked
against the method of sections and the unit-load method; the whole bridge is then solved as a
rigid-jointed space frame of tubular beams to show the secondary bending and the share of the
bracing.
"""

# # Pratt truss bridge
#
# A Pratt truss is recognised by its diagonals: apart from the inclined end posts they all slant
# down towards the middle of the span. Under gravity loads this puts the long diagonals in
# **tension** and the short verticals in **compression**, which suits steel, whose slender members
# buckle in compression long before they yield in tension.
#
# The bridge has two Pratt trusses 5 m apart, six panels of 4 m and a height of 4 m. The deck
# rests on floor beams at the bottom panel points; top struts, top and bottom lateral braces and
# two portal braces in the plane of the end posts make the structure stiff across the span. In
# the usual bridge notation the bottom panel points are $L_0\dots L_6$ and the top panel points
# $U_1\dots U_5$:
#
# | joint | x [m] | z [m] | |
# |---|---|---|---|
# | $L_k$, $k=0\dots6$ | $4k$ | 0 | $L_0$, $L_6$ on bearings |
# | $U_k$, $k=1\dots5$ | $4k$ | 4 | |
#
# each at $y=+2.5$ m (north truss) and $y=-2.5$ m (south truss).
#
# | members | steel section |
# |---|---|
# | chords, end posts | tube 323.9 × 12.5 mm |
# | diagonals, verticals | tube 219.1 × 10 mm |
# | top struts, lateral and portal braces | tube 114.3 × 6 mm |
# | floor beams | I-section 600 × 300 mm, web 12 mm, flanges 20 mm |
#
# Deck and traffic load $P=150$ kN on each interior bottom panel point of each truss; steel with
# $E=210$ GPa and $G=81$ GPa.
#
# **Reference.** For a pin-jointed plane Pratt truss under panel loads the **method of sections**
# (Hibbeler, *Structural Analysis*, ch. 3) gives every member force from the bending moment
# $M(x)$ and the shear force $V$ of the equivalent simply supported beam: the chords carry
# $\pm M/h$ about the opposite panel point, a diagonal carries $V/\sin\theta$, a vertical the shear
# of the neighbouring panel. The **unit-load method** gives the mid-span deflection
# $\delta=\sum_i N_i\,n_i\,L_i/(E A_i)$ with the member forces $n_i$ of a unit load at $L_3$.

import numpy as np
from onesim.meshing import BeamSection
from onesim.physics.solid_mechanics import FrameMember, SpatialFrame
from onesim.visualization import plots, theme

theme.set()

# ## Joints and members
#
# The joints are a table of coordinates; each member names its two end joints and its kind. The
# Pratt pattern is written once for a truss and repeated for both sides. `BeamSection` supplies
# area and second moments of the sections; the torsion constant of a thin tube is twice its
# second moment.

panel, height, width, load, youngs_modulus, shear_modulus = (
    4.0,
    4.0,
    5.0,
    150e3,
    210e9,
    81e9,
)
sides = {"N": width / 2, "S": -width / 2}
joints = {
    **{f"{s}L{k}": (panel * k, y, 0.0) for s, y in sides.items() for k in range(7)},
    **{
        f"{s}U{k}": (panel * k, y, height)
        for s, y in sides.items()
        for k in range(1, 6)
    },
}
pratt = {
    "chord": [(f"L{k}", f"L{k + 1}") for k in range(6)]
    + [(f"U{k}", f"U{k + 1}") for k in range(1, 5)]
    + [("L0", "U1"), ("L6", "U5")],
    "web": [("U1", "L2"), ("U2", "L3"), ("U5", "L4"), ("U4", "L3")]
    + [(f"L{k}", f"U{k}") for k in range(1, 6)],
}
members = {
    f"{s}{p}{q}": (kind, s + p, s + q)
    for s in sides
    for kind, pairs in pratt.items()
    for p, q in pairs
}
members |= {f"floor beam {k}": ("floor beam", f"NL{k}", f"SL{k}") for k in range(7)}
members |= {f"strut {k}": ("brace", f"NU{k}", f"SU{k}") for k in range(1, 6)}
members |= {f"top brace {k}": ("brace", f"NU{k}", f"SU{k + 1}") for k in range(1, 5)}
members |= {f"bottom brace {k}": ("brace", f"NL{k}", f"SL{k + 1}") for k in range(6)}
members |= {
    "portal west": ("brace", "NL0", "SU1"),
    "portal east": ("brace", "NL6", "SU5"),
}
sections = {
    "chord": BeamSection("pipe", (0.16195, 0.0125)),
    "web": BeamSection("pipe", (0.10955, 0.010)),
    "brace": BeamSection("pipe", (0.05715, 0.006)),
    "floor beam": BeamSection("I", (0.600, 0.300, 0.012, 0.020)),
}
index = {name: number for number, name in enumerate(joints)}
coordinates = np.array(list(joints.values()))

drawing = np.concatenate(
    (
        coordinates[[[index[p], index[q]] for _, p, q in members.values()]],
        np.full((len(members), 1, 3), np.nan),
    ),
    axis=1,
).reshape(-1, 3)
fig = plots.figure(1, 3, size=(17, 4.2))
plots.line(
    None,
    {"members": (drawing[:, 0], drawing[:, 2])},
    points={"joints": (coordinates[:, 0], coordinates[:, 2])},
    legend=False,
    xlabel="x [m]",
    ylabel="z [m]",
    ylimits=(-1, 5),
    title="Elevation: Pratt truss",
    ax=fig[0],
)
plots.line(
    None,
    {"members": (drawing[:, 0], drawing[:, 1])},
    points={"joints": (coordinates[:, 0], coordinates[:, 1])},
    legend=False,
    xlabel="x [m]",
    ylabel="y [m]",
    ylimits=(-4, 4),
    title="Plan: floor beams and lateral braces",
    ax=fig[1],
)
plots.line(
    None,
    {"members": (drawing[:, 1], drawing[:, 2])},
    points={"joints": (coordinates[:, 1], coordinates[:, 2])},
    legend=False,
    xlabel="y [m]",
    ylabel="z [m]",
    xlimits=(-4, 4),
    ylimits=(-1, 5),
    title="Cross-section",
    ax=fig[2],
)
plots.show()

# ## Models
#
# 1. **One truss, pin-jointed**: the members of the north truss as axial bars, held in their plane.
# 2. **The whole bridge, rigid-jointed**: all 66 members as Euler-Bernoulli beams joined rigidly,
#    the floor beams upright, the tubes with their full bending and torsion stiffness.
#
# Both rest on bearings at $L_0$ and $L_6$; the panel loads act at $L_1\dots L_5$.

north = [name for name in joints if name.startswith("N")]
truss = SpatialFrame(
    joints=coordinates[[index[name] for name in north]],
    members=tuple(
        FrameMember(
            name=name,
            start=north.index(p),
            end=north.index(q),
            youngs_modulus=youngs_modulus,
            area=sections[kind].area,
        )
        for name, (kind, p, q) in members.items()
        if name[0] == "N"
    ),
)
bridge = SpatialFrame(
    joints=coordinates,
    members=tuple(
        FrameMember(
            name=name,
            start=index[p],
            end=index[q],
            youngs_modulus=youngs_modulus,
            area=sections[kind].area,
            bending_inertia_z=sections[kind].inertia[0],
            bending_inertia_y=sections[kind].inertia[1],
            torsion_constant=(
                2 * sections[kind].inertia[0]
                if kind != "floor beam"
                else (2 * 0.3 * 0.02**3 + 0.56 * 0.012**3) / 3
            ),
            shear_modulus=shear_modulus,
            local_y=(
                (0, 0, 1)
                if joints[p][0] != joints[q][0] or kind == "floor beam"
                else (1, 0, 0)
            ),
        )
        for name, (kind, p, q) in members.items()
    ),
)
truss_result = truss.solve(
    supports=[
        (north.index("NL0"), "x", 0),
        (north.index("NL0"), "z", 0),
        (north.index("NL6"), "z", 0),
    ]
    + [(number, "y", 0) for number in range(len(north))],
    forces={north.index(f"NL{k}"): (0, 0, -load) for k in range(1, 6)},
)
bridge_result = bridge.solve(
    supports=[(index["NL0"], c, 0) for c in "xyz"]
    + [
        (index["SL0"], "x", 0),
        (index["SL0"], "z", 0),
        (index["NL6"], "z", 0),
        (index["SL6"], "z", 0),
    ],
    forces={index[f"{s}L{k}"]: (0, 0, -load) for s in sides for k in range(1, 6)},
)

# ## Results
#
# The method of sections for the left half of the truss (the right half mirrors it), with the
# bending moment and the shear of the equivalent beam under the given bottom-panel loads.


def method_of_sections(panel_loads):
    """Member forces of the left half of a symmetric Pratt truss under bottom-panel loads (tension positive)."""
    x, reaction = panel * np.arange(7), panel_loads.sum() / 2
    moment = [reaction * xk - panel_loads[:k] @ (xk - x[:k]) for k, xk in enumerate(x)]
    shear, sine = reaction - np.cumsum(panel_loads)[:3], height / np.hypot(
        panel, height
    )
    return {
        "L0L1": moment[1] / height,
        "L1L2": moment[1] / height,
        "L2L3": moment[2] / height,
        "U1U2": -moment[2] / height,
        "U2U3": -moment[3] / height,
        "L0U1": -shear[0] / sine,
        "U1L2": shear[1] / sine,
        "U2L3": shear[2] / sine,
        "L1U1": panel_loads[1],
        "L2U2": -shear[2],
        "L3U3": 0.0,
    }


reference = method_of_sections(np.array([0, 1, 1, 1, 1, 1, 0]) * load)
unit = method_of_sections(np.array([0, 0, 0, 1, 0, 0, 0.0]))
area = {m: sections[members["N" + m][0]].area for m in reference}
length = {
    m: np.linalg.norm(np.subtract(joints["N" + m[2:]], joints["N" + m[:2]]))
    for m in reference
}
unit_load = sum(
    (1 if m == "L3U3" else 2)
    * reference[m]
    * unit[m]
    * length[m]
    / (youngs_modulus * area[m])
    for m in reference
)
pinned = {m: truss_result.members["N" + m].local_end_actions[1] for m in reference}
rigid = {m: bridge_result.members["N" + m].local_end_actions[6] for m in reference}

print(
    "member   method of sections [kN]   pin-jointed truss [kN]   rigid-jointed bridge [kN]   difference"
)
for m in reference:
    print(
        f"{m:8s} {reference[m] / 1e3:22.1f} {pinned[m] / 1e3:24.1f} {rigid[m] / 1e3:26.1f} {(rigid[m] - reference[m]) / load:+10.3f} P"
    )
print(
    f"mid-span deflection: unit-load method {1e3 * unit_load:.3f} mm, pin-jointed truss "
    f"{-1e3 * truss_result.displacement[north.index('NL3'), 2]:.3f} mm, rigid-jointed bridge "
    f"{-1e3 * bridge_result.displacement[index['NL3'], 2]:.3f} mm"
)
print(
    f"bearing reactions of the bridge: {bridge_result.reaction_force[[index[j] for j in ('NL0', 'SL0', 'NL6', 'SL6')], 2] / 1e3} kN, "
    f"total load {10 * load / 1e3:.0f} kN"
)

# ## Secondary bending
#
# The rigid joints make the members bend a little as the truss deflects. For a tube the largest
# bending stress follows from the resultant end moment, $\sigma_b=M\,r/I$.

axial_stress = {m: rigid[m] / area[m] for m in reference}
bending_stress = {
    m: max(
        np.hypot(*bridge_result.members["N" + m].local_end_actions[[4, 5]]),
        np.hypot(*bridge_result.members["N" + m].local_end_actions[[10, 11]]),
    )
    * sections[members["N" + m][0]].dimensions[0]
    / sections[members["N" + m][0]].inertia[0]
    for m in reference
}
for m in reference:
    print(
        f"{m:8s} axial stress {axial_stress[m] / 1e6:7.1f} MPa   largest secondary bending stress {bending_stress[m] / 1e6:6.1f} MPa"
    )

# ## Figure
#
# The deflected bridge (displacements magnified 100 times), the member forces of the half truss
# from the three approaches, and the axial against the secondary bending stress in the
# rigid-jointed bridge.

deformed = np.concatenate(
    (
        (coordinates + 100 * bridge_result.displacement)[
            [[index[p], index[q]] for _, p, q in members.values()]
        ],
        np.full((len(members), 1, 3), np.nan),
    ),
    axis=1,
).reshape(-1, 3)
fig = plots.figure(1, 3, size=(18, 5.2))
plots.line(
    None,
    {"deflected (×100)": (deformed[:, 0], deformed[:, 2])},
    reference={"unloaded": (drawing[:, 0], drawing[:, 2])},
    ylimits=(-2.6, 5),
    legend="lower center",
    xlabel="x [m]",
    ylabel="z [m]",
    title="Rigid-jointed bridge, elevation",
    ax=fig[0],
)
plots.bars(
    list(reference),
    {
        "pin-jointed truss": [pinned[m] / 1e3 for m in reference],
        "rigid-jointed bridge": [rigid[m] / 1e3 for m in reference],
    },
    reference={"method of sections": [reference[m] / 1e3 for m in reference]},
    ylabel="axial force [kN]",
    title="Member forces of the left half truss",
    ax=fig[1],
)
plots.bars(
    list(reference),
    {
        "axial": [axial_stress[m] / 1e6 for m in reference],
        "secondary bending": [bending_stress[m] / 1e6 for m in reference],
    },
    ylabel="stress [MPa]",
    title="Rigid-jointed bridge: member stresses",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Change the joint table, for example eight panels or a curved top chord (a Parker truss), and the
# solver follows. Flip the diagonals to obtain a Howe truss and watch tension and compression
# swap between diagonals and verticals.
