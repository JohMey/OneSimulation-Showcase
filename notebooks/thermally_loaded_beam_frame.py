"""OneSimulation example: a 3D beam bent by temperature differences across its cross-section.
Requires the installed, activated OneSimulation runtime.

A simply supported steel beam, modelled with 3D beam elements, is warmer on one side and on its
underside than on the opposite faces. The prescribed temperature field bends it about both axes
without any heat-conduction solve; deflections and, for a beam clamped at both ends, the support
reactions are checked against beam theory.
"""

# # Thermally loaded beam with 3D beam elements
#
# A straight steel beam of length $L=3$ m with a square $40\times40$ mm cross-section is heated
# unevenly. Across its height the bottom face is $\Delta T_z=50$ K warmer than the top face, across its
# width the face at $+y$ is $\Delta T_y=50$ K warmer than the face at $-y$, and the mean temperature lies
# 50 K above the stress-free reference. The temperature therefore varies linearly over the section,
#
# $$T(y,z)-T_\text{ref}=\Delta T_m+g_y\,y+g_z\,z,\qquad g_y=\frac{\Delta T_y}{b}=1250\ \text{K/m},\quad
#   g_z=-\frac{\Delta T_z}{h}=-1250\ \text{K/m}.$$
#
# A linear temperature over the section is exactly what a beam element can carry: the mean value
# stretches the axis, the two gradients impose the free thermal curvatures $\kappa_y=\alpha g_y$ and
# $\kappa_z=\alpha g_z$. OneSimulation models the beam as a `SpatialFrame` of 3D Euler-Bernoulli beam
# elements (six degrees of freedom per joint) and applies the temperature field with
# `ThermalFrameMemberLoads.linear_temperature`; no temperature is computed, it is prescribed.
#
# | case | supports | beam theory |
# |---|---|---|
# | simply supported | $x=0$: displacements and twist held; $x=L$: transverse displacements held | statically determinate: free bending, $v(x)=\kappa_y\,x(L-x)/2$, $w(x)=\kappa_z\,x(L-x)/2$, $u(L)=\alpha\Delta T_mL$, no reactions |
# | clamped | both ends fully held | no displacement; axial force $N=-EA\,\alpha\Delta T_m$, end moments $EI\,\alpha g$ about both axes |
#
# Steel: $E=210$ GPa, $G=E/2.6$, $\alpha=11\times10^{-6}$ 1/K.
#
# **Reference** (B. A. Boley and J. H. Weiner, *Theory of Thermal Stresses*, Wiley 1960, §9.2;
# W. C. Young and R. G. Budynas, *Roark's Formulas for Stress and Strain*, 7th ed., McGraw-Hill 2002,
# §8.1, thermal loading of beams). The midspan deflection of the simply supported beam is
#
# $$|v(L/2)|=|w(L/2)|=\frac{\alpha\,g\,L^2}{8}=15.47\ \text{mm},\qquad
#   \sqrt{v^2+w^2}=21.88\ \text{mm}.$$

import numpy as np
from onesim.meshing import BeamSection
from onesim.physics.solid_mechanics import (
    FrameMember,
    SpatialFrame,
    ThermalFrameMemberLoads,
)
from onesim.visualization import plots, theme

theme.set()

# ## Beam model
#
# The beam is divided into $n$ equal elements; three subdivisions are compared. The section supplies
# area and second moments.

length, youngs_modulus, alpha = 3.0, 210e9, 11e-6
section = BeamSection("rectangle", (0.04, 0.04))
inertia_y, inertia_z = section.inertia
beam = lambda n: SpatialFrame(
    joints=np.column_stack(
        (np.linspace(0, length, n + 1), np.zeros(n + 1), np.zeros(n + 1))
    ),
    members=tuple(
        FrameMember(
            name=f"element {i}",
            start=i,
            end=i + 1,
            youngs_modulus=youngs_modulus,
            area=section.area,
            bending_inertia_y=inertia_y,
            bending_inertia_z=inertia_z,
            shear_modulus=youngs_modulus / 2.6,
            torsion_constant=0.1406 * 0.04**4,
        )
        for i in range(n)
    ),
)
temperature = ThermalFrameMemberLoads.linear_temperature(
    temperature=343.15,
    reference_temperature=293.15,
    expansion_coefficient=alpha,
    gradient=(1250.0, -1250.0),
)
simply_supported = lambda n: (
    *((0, dof, 0.0) for dof in ("x", "y", "z", "rx")),
    (n, "y", 0.0),
    (n, "z", 0.0),
)
clamped = lambda n: tuple(
    (joint, dof, 0.0) for joint in (0, n) for dof in ("x", "y", "z", "rx", "ry", "rz")
)
runs = {
    n: beam(n).solve(supports=simply_supported(n), member_loads=temperature)
    for n in (2, 4, 12)
}
held = beam(12).solve(supports=clamped(12), member_loads=temperature)

# ## Comparison with beam theory

camber = lambda x, gradient: alpha * gradient * x * (length - x) / 2
print(
    "elements   v(L/2) [mm]   w(L/2) [mm]   total [mm]   u(L) [mm]   largest deviation from beam theory at the joints"
)
for n, result in runs.items():
    x = np.linspace(0, length, n + 1)
    exact = np.column_stack((alpha * 50.0 * x, camber(x, 1250.0), camber(x, -1250.0)))
    mid = result.displacement[n // 2]
    print(
        f"{n:8d} {1e3 * mid[1]:13.4f} {1e3 * mid[2]:13.4f} {1e3 * np.hypot(mid[1], mid[2]):12.4f} {1e3 * result.displacement[-1, 0]:11.4f}"
        f" {np.abs(result.displacement - exact).max():14.1e} m"
    )
print(
    f"beam theory {1e3 * camber(length / 2, 1250.0):11.4f} {1e3 * camber(length / 2, -1250.0):13.4f} "
    f"{1e3 * np.sqrt(2) * camber(length / 2, 1250.0):12.4f} {1e3 * alpha * 50.0 * length:11.4f}"
)
print(
    f"simply supported: largest reaction force {np.abs(runs[12].reaction_force).max():.1e} N, "
    f"largest reaction moment {np.abs(runs[12].reaction_moment).max():.1e} N m (statically determinate, free bending)"
)

axial = youngs_modulus * section.area * alpha * 50.0
moment = youngs_modulus * inertia_y * alpha * 1250.0
reactions = {
    "axial force [kN]": (abs(held.reaction_force[0, 0]) / 1e3, axial / 1e3),
    "moment about y [N m]": (abs(held.reaction_moment[0, 1]), moment),
    "moment about z [N m]": (abs(held.reaction_moment[0, 2]), moment),
}
print("clamped beam, reactions at x = 0 (magnitudes)   simulation    beam theory")
for name, (value, exact) in reactions.items():
    print(f"{name:44s} {value:12.3f} {exact:14.3f}")
print(f"clamped beam: largest displacement {np.abs(held.displacement).max():.1e} m")

# ## Figure
#
# Left: the deflection of the simply supported beam towards the warm side ($+y$) and towards the warm
# underside ($-z$), at the joints of the 12-element model, against the beam-theory parabola. Right:
# the reactions of the clamped beam that hold back the thermal elongation and curvatures.

fine = np.linspace(0, length, 200)
joints = np.linspace(0, length, 13)
fig = plots.figure(1, 2, size=(14, 4.8))
plots.line(
    points={
        "v (towards +y)": (joints, 1e3 * runs[12].displacement[:, 1]),
        "w (along z)": (joints, 1e3 * runs[12].displacement[:, 2]),
    },
    reference={
        "v (towards +y)": (fine, 1e3 * camber(fine, 1250.0)),
        "w (along z)": (fine, 1e3 * camber(fine, -1250.0)),
    },
    xlabel="position along the beam x [m]",
    ylabel="deflection [mm]",
    title="Simply supported beam (markers) and beam theory (dashed)",
    legend="center",
    ax=fig[0],
)
plots.bars(
    [
        "axial force / 10 [kN]",
        "moment about y [N m] / 100",
        "moment about z [N m] / 100",
    ],
    {
        "simulation": [
            abs(held.reaction_force[0, 0]) / 1e4,
            abs(held.reaction_moment[0, 1]) / 100,
            abs(held.reaction_moment[0, 2]) / 100,
        ]
    },
    reference={"beam theory": [axial / 1e4, moment / 100, moment / 100]},
    ylabel="magnitude (scaled)",
    title="Clamped beam: reactions at x = 0",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Give the gradients different values, or let the temperature vary in time with the `time` argument of
# `linear_temperature`: the beam follows quasi-statically. Holding only one end against rotation turns
# the free curvature into a cantilever deflection $\kappa L^2/2$ at the tip.
