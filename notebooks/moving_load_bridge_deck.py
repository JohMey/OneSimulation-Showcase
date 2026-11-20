"""OneSimulation example: a wheel load moving across a simply supported deck.
Requires the installed, activated OneSimulation runtime.

A concentrated wheel load crosses a solid concrete deck slab that rests on two end supports. The slab
is read from a STEP file and meshed with curved quadratic tetrahedra; the load position is a function
of time and every position is a static 3D solve. The midspan deflection traced over the crossing
(the influence line) is compared with Timoshenko beam theory and with the deflection line of the
midspan load case through Maxwell-Betti reciprocity.
"""

# # A wheel load moving across a simply supported deck
#
# Bridges, crane runways and rails carry loads that move. As long as the load moves slowly compared
# with the natural vibrations of the structure, each position is a static load case and the response
# follows the position of the load. Plotting a response quantity against the load position gives its
# *influence line*, the basic design tool for moving loads.
#
# **Setup.** A deck slab of length $L=10$ m, width $b=1$ m and depth $h=0.5$ m, concrete with
# $E=30$ GPa and $\nu=0.2$. Both end faces rest on supports that hold them vertically and leave them
# free to rotate; one end is also held horizontally at mid-depth. A wheel load $P=100$ kN, spread
# smoothly over 0.4 m of the top face, crosses the deck at $v=20$ m/s, so its position is
# $a(t)=0.5\text{ m}+v\,t$. The slab comes from the CAD file `moving_load_bridge_deck.step`,
# written by `moving_load_bridge_deck_geometry.py` with the CAD objects of OneSimulation.
#
# | face | condition |
# |---|---|
# | `x_min` | $u_z=0$; $u_x=0$ on the mid-depth line, $u_y=0$ on the centre line |
# | `x_max` | $u_z=0$ |
# | `z_max` | moving traction $-q(x-a(t))$ |
# | other faces | free |
#
# **Reference.** For a simply supported Timoshenko beam with a load $P$ at $x=a\le L/2$ the midspan
# deflection is (Timoshenko and Gere, *Mechanics of Materials*)
#
# $$w\!\left(\tfrac L2\right)=\frac{P\,a\,(3L^2-4a^2)}{48\,EI}+\frac{P\,a}{2\,\kappa GA},\qquad \kappa=\tfrac56 ,$$
#
# and symmetric for $a>L/2$. Independently of any beam theory, Maxwell-Betti reciprocity says that the
# midspan deflection under a load at $a$ equals the deflection at $a$ under the same load at midspan,
# so the influence line must coincide with the deflection line of the midspan load case.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import TetrahedralMesher, with_order
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import (
    FixedDisplacementComponents,
    LinearElasticity,
    Traction,
    TractionFree,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the slab with its named faces; quadratic tetrahedra of 0.2 m.

deck = load("moving_load_bridge_deck.step")
faces = deck.boundaries
length, width, depth = deck.size
mesh = with_order(TetrahedralMesher(target_size=0.2).generate(deck), 2)

view = plots.Camera(elevation=25, azimuth=-70)
fig = plots.figure(1, 2, size=(15, 5.5))
plots.geometry(
    deck,
    selection={
        "supported ends": faces.x_min | faces.x_max,
        "deck surface": faces.z_max,
    },
    labels=False,
    camera=view,
    title="Deck slab with its named faces",
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
# The wheel pressure is a smooth bump of width $w=0.4$ m, $q=\frac{2P}{bw}\cos^2\!\big(\pi(x-a)/w\big)$
# for $|x-a|<w/2$, whose resultant is exactly $P$. The crossing is sampled every 25 ms, i.e. every
# 0.5 m of travel; each sample is one static solve on the same mesh.

concrete = IsotropicSolid(
    density=2400.0,
    specific_heat_capacity=880.0,
    thermal_conductivity=1.8,
    youngs_modulus=30e9,
    poisson_ratio=0.2,
    thermal_expansion_coefficient=10e-6,
)
model = LinearElasticity(domain=deck.volume, material=concrete)
force, patch, speed = 100e3, 0.4, 20.0
times = np.arange(0.0, 0.4501, 0.025)
position = lambda t: 0.5 + speed * t
wheel = lambda a: (
    lambda x, y, z: np.column_stack(
        (
            0 * x,
            0 * x,
            -2
            * force
            / (width * patch)
            * np.cos(np.pi * (x - a) / patch) ** 2
            * (np.abs(x - a) < patch / 2),
        )
    )
)
supports = (
    FixedDisplacementComponents(
        boundary=faces.x_min,
        physics=model,
        components=("x", "y", "z"),
        where={
            "x": lambda x, y, z: np.abs(z) < 1e-6,
            "y": lambda x, y, z: np.abs(y) < 1e-6,
        },
    ),
    FixedDisplacementComponents(boundary=faces.x_max, physics=model, components=("z",)),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)
results = [
    Simulation(
        problem=Problem(
            name=f"Wheel at t = {t:.3f} s",
            geometry=deck,
            physics=(model,),
            boundary_conditions=(
                *supports,
                Traction(boundary=faces.z_max, physics=model, value=wheel(position(t))),
                TractionFree(
                    boundary=faces.all - faces.x_min - faces.x_max - faces.z_max,
                    physics=model,
                ),
            ),
        ),
        mesh=mesh,
        numerics=numerics,
    ).solve_stationary()
    for t in times
]
displacements = [result.field(model.displacement) for result in results]
centre_line = lambda x: np.column_stack((x, 0 * x, 0 * x))
midspan = np.array(
    [
        field.evaluate(centre_line(np.array([length / 2])))[0, 2]
        for field in displacements
    ]
)
middle = int(np.argmin(np.abs(position(times) - length / 2)))
line = displacements[middle].evaluate(centre_line(position(times)))[:, 2]

# ## Comparison with beam theory and reciprocity
#
# The midspan deflection on the axis for every load position against the Timoshenko influence line,
# and against the deflection line of the midspan load case at the same positions.

inertia, area, shear = width * depth**3 / 12, width * depth, 30e9 / (2 * 1.2)
near = lambda a: np.minimum(a, length - a)
timoshenko = lambda a: -(
    force * near(a) * (3 * length**2 - 4 * near(a) ** 2) / (48 * 30e9 * inertia)
    + force * near(a) / (2 * 5 / 6 * shear * area)
)
print(
    f"{'t [s]':>6s} {'a [m]':>6s} {'w(L/2) [mm]':>12s} {'Timoshenko [mm]':>16s} {'deviation':>10s} {'w(a), load at L/2 [mm]':>23s} {'Betti gap':>10s}"
)
for t, a, w, w_line in zip(times, position(times), midspan, line):
    print(
        f"{t:6.3f} {a:6.2f} {1e3 * w:12.4f} {1e3 * timoshenko(a):16.4f} {100 * (w / timoshenko(a) - 1):+9.2f} % {1e3 * w_line:23.4f} {100 * (w - w_line) / abs(midspan).max():+9.3f} %"
    )
print(
    f"largest midspan deflection {1e3 * abs(midspan).max():.3f} mm with the load at a = {position(times[np.argmax(abs(midspan))]):.2f} m; "
    f"of it {100 * force * length / 2 / (2 * 5 / 6 * shear * area) / abs(timoshenko(length / 2)):.1f} % from shear"
)
print(
    f"reaction at the far support with the load at midspan: {abs(displacements[middle].reaction(faces.x_max)[2]) / 1e3:.2f} kN (statics: {force / 2e3:.2f} kN)"
)

# ## Figure
#
# Left the influence line of the midspan deflection with the Timoshenko curve and the deflection line
# of the midspan load case; right the deflected deck with the wheel at midspan, 100 times enlarged.

axis = np.linspace(0.0, length, 201)
fig = plots.figure(1, 2, size=(16, 6))
plots.line(
    points={
        "midspan deflection under the moving load": (position(times), 1e3 * midspan),
        "deflection line, load at midspan (Betti)": (position(times), 1e3 * line),
    },
    reference={"Timoshenko beam": (axis, 1e3 * timoshenko(axis))},
    xlabel="load position a [m]",
    ylabel="deflection [mm]",
    title="Influence line of the midspan deflection",
    legend="upper center",
    ax=fig[0],
)
plots.surface(
    displacements[middle],
    component=lambda u, position: 1e3 * u[:, 2],
    deformation=displacements[middle],
    deformation_scale=100.0,
    camera=view,
    label="u_z [mm]",
    title="Deck with the wheel at midspan (×100)",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Add a second axle at a fixed distance behind the first and superpose, trace the bending stress at
# the bottom fibre instead of the deflection, or replace the slab by a box girder read from its own
# STEP file. At speeds that approach the natural frequencies of the deck, inertia adds a dynamic
# amplification that this quasi-static crossing leaves out.
