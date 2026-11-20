"""OneSimulation example: natural frequencies of an in-plane frame carrying a mass with rotary inertia.
Requires the installed, activated OneSimulation runtime.

An L-shaped steel frame, clamped at its base, carries a compact machine at the end of its arm,
modelled as a point mass with a mass moment of inertia. The frame is built from beam elements that
move in their plane; the two lowest natural frequencies are compared with the closed-form solution
of the massless frame, and the effect of the frame's own mass is shown on refined meshes.
"""

# # Natural frequencies of an in-plane frame with a mass and its rotary inertia
#
# A heavy component on a light supporting structure, such as a motor on a bracket or a camera on
# a mast, dominates the dynamics: the structure acts as a spring, the component as a mass. A compact
# component does not only translate but also rotates, so besides its mass $m$ its mass moment of
# inertia $J$ enters the natural frequencies.
#
# **Setup.** A frame in the $xz$ plane: a column of height $H=2$ m, clamped at the base, and a
# horizontal arm of length $L=1.5$ m, rigidly connected at the corner. Both are steel sections with
# $E=210$ GPa, $A=5\cdot10^{-3}$ m² and $I=8\cdot10^{-6}$ m⁴ for in-plane bending. At the end of
# the arm sits a machine of $m=200$ kg and $J=60$ kg m². Only in-plane motion is allowed: at every
# joint the out-of-plane translation and the two out-of-plane rotations are held.
#
# | joint | condition |
# |---|---|
# | base | clamped |
# | corner | rigid connection of column and arm |
# | arm end | point mass $m$ and rotary inertia $J$ |
# | all joints | in-plane motion only ($u_y=0$, $\theta_x=\theta_z=0$) |
#
# **Reference.** If the frame's own mass is neglected, the arm end has three degrees of freedom,
# $(u_x, u_z, \theta_y)$. The unit-load method gives the flexibility matrix of the clamped L-frame at
# that point (e.g. Hibbeler, *Structural Analysis*, Ch. 9), from bending and the axial strain of arm
# and column,
#
# $$\mathbf F=\frac1{EI}\begin{pmatrix} H^3/3 & LH^2/2 & H^2/2\\ LH^2/2 & L^3/3+L^2H & L^2/2+LH\\
#   H^2/2 & L^2/2+LH & L+H\end{pmatrix}+\frac1{EA}\begin{pmatrix}L&0&0\\0&H&0\\0&0&0\end{pmatrix},$$
#
# and the natural frequencies follow from $\det\!\big(\mathbf F^{-1}-\omega^2\,\mathrm{diag}(m,m,J)\big)=0$.

import numpy as np
from onesim.numerics.linear import SparseSymmetricEigenSolver
from onesim.physics.solid_mechanics import FrameMember, SpatialFrame, frame_modes
from onesim.visualization import plots, theme

theme.set()

# ## Frame model
#
# The column and the arm are divided into $n$ beam elements each; the joints lie in the $xz$
# plane. Four meshes, $n=1$ to $8$.

height, arm, youngs_modulus, area, inertia = 2.0, 1.5, 210e9, 5e-3, 8e-6
mass, rotary = 200.0, 60.0
joints = lambda n: np.vstack(
    (
        np.column_stack(
            (np.zeros(n + 1), np.zeros(n + 1), np.linspace(0, height, n + 1))
        ),
        np.column_stack(
            (np.linspace(0, arm, n + 1)[1:], np.zeros(n), np.full(n, height))
        ),
    )
)
frame = lambda n: SpatialFrame(
    joints=joints(n),
    members=tuple(
        FrameMember(
            name=f"member {i}",
            start=i,
            end=i + 1,
            youngs_modulus=youngs_modulus,
            area=area,
            bending_inertia_y=inertia,
            bending_inertia_z=inertia,
            torsion_constant=2 * inertia,
            shear_modulus=81e9,
        )
        for i in range(2 * n)
    ),
)
supports = lambda n: (
    *((0, dof, 0.0) for dof in ("x", "y", "z", "rx", "ry", "rz")),
    *((joint, dof, 0.0) for joint in range(1, 2 * n + 1) for dof in ("y", "rx", "rz")),
)
divisions = [1, 2, 4, 8]
solve = lambda n, density: frame_modes(
    frame(n),
    supports=supports(n),
    density=density,
    mode_count=3,
    target_frequency=0.0,
    eigensolver=SparseSymmetricEigenSolver(tolerance=1e-12),
    joint_masses={2 * n: (mass, (0.0, rotary, 0.0))},
)
light = {n: solve(n, 1e-6) for n in divisions}
steel = {n: solve(n, 7850.0) for n in divisions}

# ## Comparison with the massless frame
#
# With a negligible member density the frame is exactly the massless spring of the reference, for
# any number of elements. With the steel density the column adds 79 kg and the arm 59 kg of distributed mass and
# lower the frequencies; the refined meshes converge to the frame with its own mass.

flexibility = np.array(
    [
        [height**3 / 3, arm * height**2 / 2, height**2 / 2],
        [arm * height**2 / 2, arm**3 / 3 + arm**2 * height, arm**2 / 2 + arm * height],
        [height**2 / 2, arm**2 / 2 + arm * height, arm + height],
    ]
) / (youngs_modulus * inertia) + np.diag([arm, height, 0.0]) / (youngs_modulus * area)
reference = np.sort(
    np.sqrt(
        np.linalg.eigvals(
            np.linalg.inv(flexibility) @ np.diag([1 / mass, 1 / mass, 1 / rotary])
        ).real
    )
) / (2 * np.pi)
print(f"{'':28s} {'f1 [Hz]':>10s} {'f2 [Hz]':>10s} {'f3 [Hz]':>10s}")
print(
    f"{'closed form, massless frame':28s} " + " ".join(f"{f:10.3f}" for f in reference)
)
for n, modes in light.items():
    print(
        f"{f'massless members, n = {n}':28s} "
        + " ".join(f"{f:10.3f}" for f in np.sort(modes.frequencies))
        + "   deviation "
        + ", ".join(
            f"{100 * (f / r - 1):+.1e} %"
            for f, r in zip(np.sort(modes.frequencies), reference)
        )
    )
for n, modes in steel.items():
    print(
        f"{f'steel members, n = {n}':28s} "
        + " ".join(f"{f:10.3f}" for f in np.sort(modes.frequencies))
    )
point_only = frame_modes(
    frame(8),
    supports=supports(8),
    density=1e-6,
    mode_count=3,
    target_frequency=0.0,
    eigensolver=SparseSymmetricEigenSolver(tolerance=1e-12),
    joint_masses={16: mass},
)
print(
    "the rotary inertia matters: without J the massless frame would have "
    + ", ".join(f"{f:.3f} Hz" for f in np.sort(point_only.frequencies)[:2])
    + " instead of "
    + ", ".join(f"{f:.3f} Hz" for f in reference[:2])
)

# ## Figure
#
# The two lowest mode shapes of the steel frame ($n=8$) drawn over the undeformed frame, and the
# frequencies of both models against the number of elements.

shape = lambda k: steel[8].mode_displacements(int(np.argsort(steel[8].frequencies)[k]))
scale = lambda k: 0.4 / np.abs(shape(k)[:, [0, 2]]).max()
fig = plots.figure(1, 3, size=(18, 6))
for k in range(2):
    plots.line(
        y={
            "undeformed": (joints(8)[:, 0], joints(8)[:, 2]),
            f"mode {k + 1}, {np.sort(steel[8].frequencies)[k]:.2f} Hz": (
                joints(8)[:, 0] + scale(k) * shape(k)[:, 0],
                joints(8)[:, 2] + scale(k) * shape(k)[:, 2],
            ),
        },
        styles={"undeformed": "dashed"},
        xlimits=(-0.8, 2.3),
        ylimits=(-0.2, 2.8),
        xlabel="x [m]",
        ylabel="z [m]",
        title=f"Mode {k + 1} of the steel frame",
        legend="lower right",
        ax=fig[k],
    )
plots.line(
    points={
        f"mode {k + 1}, massless members": (
            divisions,
            [np.sort(light[n].frequencies)[k] for n in divisions],
        )
        for k in range(2)
    }
    | {
        f"mode {k + 1}, steel members": (
            divisions,
            [np.sort(steel[n].frequencies)[k] for n in divisions],
        )
        for k in range(2)
    },
    hlines={f"closed form, mode {k + 1}": reference[k] for k in range(2)},
    markers=True,
    scale="logx",
    xlabel="elements per member",
    ylabel="natural frequency [Hz]",
    title="Frequencies and mesh",
    legend="center right",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Move the machine to the corner and watch the rotary inertia lose its influence on the first mode,
# add a second mass on the column, or replace the clamped base by a rotational spring.
