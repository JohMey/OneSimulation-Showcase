"""OneSimulation example: position optimisation of grading rings on a high-voltage insulator (2D axisymmetric).
Requires the installed, activated OneSimulation runtime.

Grading rings at both ends of a rod insulator take the field off the critical junction of metal,
insulator and air. Too close to the ends they do little, too far along the rod they press the
field onto the insulator themselves. The axisymmetric electrostatic field is solved for seven ring
positions read from STEP files; the optimum follows from a fit through the sweep and is checked
by a mesh study and the charge balance.
"""

# # Position optimisation of grading rings on an insulator
#
# A rod insulator of length $L=1$ m and radius 4 cm (silicone rubber, $\varepsilon_r=4.3$) carries
# a metal end fitting at each end: the upper one at the peak phase voltage $V_0=100$ kV, the lower
# one grounded. Without any field control, the field on the insulator surface peaks next to the
# fittings. Grading rings (torus, ring radius 15 cm, tube radius 2 cm) at the potential of the
# neighbouring fitting take over part of the charge and lower that peak. Their distance $s$ from the
# end of the rod is the design parameter: the surface field is lowest for one position, and with it
# the risk of flashover.
#
# The insulator and the rings are rotationally symmetric, so the potential satisfies
#
# $$\frac1r\frac{\partial}{\partial r}\Bigl(r\,\varepsilon\frac{\partial V}{\partial r}\Bigr)
#   +\frac{\partial}{\partial z}\Bigl(\varepsilon\frac{\partial V}{\partial z}\Bigr)=0$$
#
# in the meridian plane $(r,z)$, solved with quadratic elements and the axisymmetric measure
# $2\pi r\,dr\,dz$.
#
# Half of the air around the insulator, with the fittings and rings cut out, comes from the CAD file
# `grading_ring_position_optimisation.step` ($s=10$ cm) and from one file for each other ring
# position, `..._00cm.step` to `..._30cm.step`; the rod is `grading_ring_position_optimisation_rod.step`.
# The script `grading_ring_position_optimisation_geometry.py` builds them with the CAD objects of
# OneSimulation.
#
# | boundary | condition |
# |---|---|
# | upper fitting and ring (`hv.*`) | $V=V_0$ |
# | lower fitting and ring (`ground.*`) | $V=0$ |
# | outer faces of the air (`air.*`) | $D_n=0$ |
# | axis $r=0$ (`part.x_min`) | symmetry, $D_n=0$ |
#
# **Reference.** There is no closed form for the optimum; it is the minimum of the computed surface
# field over the ring distance (Kuffel, Zaengl and Kuffel, *High Voltage Engineering:
# Fundamentals*, 2nd ed., Ch. 4 on field control; IEC/TS 60815). The computation is checked by a
# mesh study of the surface field at the optimum and by the charge balance: without any charge
# leaving through the outer faces, the charge on the high-voltage electrodes must equal minus the
# charge on the grounded ones.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VacuumPermittivity
from onesim.meshing import TriangularMesher, joined
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange, RevolvedH1SpaceFactory
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import (
    FixedPotential,
    PoissonEquation,
    ZeroNormalElectricDisplacement,
)
from onesim.simulation import Problem, Simulation, regionwise
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `section` with the plane $y=0$ returns the meridian half-plane of each solid. The air with the rod
# cut out and the rod itself are meshed with triangles that match on the rod surface and joined into
# one mesh, which carries the edge names of both: the ends of the rod (`top`, `bottom`) touch the
# fittings. The cells are 5 mm on the rod surface and on the conductors and grow with the distance.

offsets = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
solids = {
    s: load(
        "grading_ring_position_optimisation.step"
        if s == 0.10
        else f"grading_ring_position_optimisation_{round(100 * s):02d}cm.step"
    )
    for s in offsets
}
rod = load("grading_ring_position_optimisation_rod.step").section((0, 1, 0))
planes = {s: solid.section((0, 1, 0)) for s, solid in solids.items()}
rod_radius, length = rod.size[:2]
selections = {
    s: {edge.id: plane.boundaries[edge.id] for edge in plane.edges}
    for s, plane in planes.items()
}
edges = lambda s, prefix: [name for name in selections[s] if name.startswith(prefix)]
mesh_for = lambda s, h: joined(
    [
        TriangularMesher(
            target_size=0.3,
            grading=0.2,
            hard_sizes={"cut.outer_wall": h},
            local_sizes={edge: h for edge in edges(s, "hv.") + edges(s, "ground.")},
        ).generate(planes[s] - rod),
        TriangularMesher(target_size=2 * h, hard_sizes={"outer_wall": h}).generate(rod),
    ],
    geometry=planes[s],
)
meshes = {s: mesh_for(s, 0.005) for s in offsets}

fig = plots.figure(1, 2, size=(14, 6.5))
plots.geometry(
    [
        solids[0.10].named("air"),
        load("grading_ring_position_optimisation_rod.step").named("insulator"),
    ],
    camera=plots.Camera(elevation=20, azimuth=-60),
    title="Half of the air around the insulator",
    ax=fig[0],
)
plots.mesh(
    meshes[0.10],
    groups=("all",),
    legend=False,
    title=f"Meridian plane, {meshes[0.10].cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# `RevolvedH1SpaceFactory` makes the quadratic Lagrange space of the meridian plane axisymmetric.
# The permittivity is $4.3\,\varepsilon_0$ in the rod and $\varepsilon_0$ in the air. Each ring
# position is one simulation; the position of 10 cm is also solved on two coarser meshes.

voltage, eps_r = 100e3, 4.3
eps0 = float(VacuumPermittivity().evaluate())
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2), provider=RevolvedH1SpaceFactory()),
    linear_solver=SparseDirect(),
)
runs = [(s, meshes[s]) for s in offsets] + [
    (0.10, mesh_for(0.10, h)) for h in (0.020, 0.010)
]
models = [
    PoissonEquation(
        domain=planes[s].domain(),
        permittivity=regionwise(((rod, eps_r * eps0),), default=eps0, unit="F/m"),
        charge_density=0.0,
    )
    for s, mesh in runs
]
results = [
    Simulation(
        problem=Problem(
            name="Grading rings",
            geometry=planes[s],
            physics=(model,),
            boundary_conditions=(
                *(
                    FixedPotential(
                        boundary=selections[s][edge], physics=model, value=voltage
                    )
                    for edge in edges(s, "hv.")
                ),
                *(
                    FixedPotential(
                        boundary=selections[s][edge], physics=model, value=0.0
                    )
                    for edge in edges(s, "ground.")
                ),
                *(
                    ZeroNormalElectricDisplacement(
                        boundary=selections[s][edge], physics=model
                    )
                    for edge in edges(s, "air.") + edges(s, "part.")
                ),
            ),
        ),
        mesh=mesh,
        numerics=numerics,
    ).solve_stationary()
    for model, (s, mesh) in zip(models, runs)
]

# ## Results
#
# The design value is the largest field on the insulator surface, read just outside the rod and
# leaving out 1 cm next to each fitting, where the junction of metal, insulator and air makes the
# field singular. A parabola through the three best positions gives the optimum.

z = np.linspace(0.01, length - 0.01, 393)
surface = np.column_stack((np.full_like(z, rod_radius + 1e-6), z))
profiles = [
    np.linalg.norm(result.field(model.electric_field).evaluate(surface), axis=1)
    for model, result in zip(models, results)
]
peaks = np.array([profile.max() for profile in profiles[: len(offsets)]])
best = np.clip(np.argmin(peaks), 1, len(offsets) - 2)
curve = np.polyfit(offsets[best - 1 : best + 2], peaks[best - 1 : best + 2], 2)
optimum = -curve[1] / (2 * curve[0])
print("ring distance s   largest surface field [kV/cm]")
for s, peak in zip(offsets, peaks):
    print(f"{100 * s:13.0f} cm {peak / 1e5:20.3f}")
print(
    f"optimum ring distance {100 * optimum:.1f} cm with {np.polyval(curve, optimum) / 1e5:.3f} kV/cm; "
    f"rings at the ends (s = 0): {peaks[0] / 1e5:.3f} kV/cm, uniform field V0/L = {voltage / length / 1e5:.3f} kV/cm"
)

study = [profiles[-2].max(), profiles[-1].max(), profiles[2].max()]
print(
    f"mesh study at s = 10 cm (20, 10, 5 mm on the rod): {', '.join(f'{v / 1e5:.4f}' for v in study)} kV/cm; "
    f"change between the two finest meshes {100 * (study[2] / study[1] - 1):+.3f} %"
)
for (s, mesh), model, result in list(zip(runs, models, results))[: len(offsets) : 3]:
    potential = result.field(model.potential)
    high = sum(potential.reaction(selections[s][edge]) for edge in edges(s, "hv."))
    low = sum(potential.reaction(selections[s][edge]) for edge in edges(s, "ground."))
    print(
        f"s = {100 * s:4.0f} cm: charge on the high-voltage side {1e9 * high:.4f} nC, on the grounded side {1e9 * low:.4f} nC, "
        f"balance {(high + low) / high:+.1e}"
    )

# ## Figure
#
# The field strength around the upper end at the optimum, the surface field along the rod for
# three ring positions, and the largest surface field against the ring distance.

fine = np.linspace(offsets[best - 1], offsets[best + 1], 100)
index = int(np.argmin(np.abs(np.array(offsets) - optimum)))
fig = plots.figure(layout=[["map", "profile"], ["map", "sweep"]], size=(15, 10))
plots.surface(
    results[index].field(models[index].electric_field),
    component=lambda E, position: np.linalg.norm(E[:, :2], axis=1) / 1e5,
    limits=(0, 4),
    contours=17,
    subdivisions=2,
    label="|E| [kV/cm]",
    xlimits=(0, 0.35),
    ylimits=(0.55, 1.2),
    xlabel="r [m]",
    ylabel="z [m]",
    title=f"Field near the high-voltage end, s = {100 * offsets[index]:.0f} cm",
    ax=fig["map"],
)
plots.line(
    z,
    {
        f"s = {100 * offsets[i]:.0f} cm": profiles[i] / 1e5
        for i in (0, index, len(offsets) - 1)
    },
    xlabel="height on the insulator z [m]",
    ylabel="surface field [kV/cm]",
    title="Field along the insulator surface",
    ax=fig["profile"],
)
plots.line(
    fine * 100,
    {"parabola through the three best positions": np.polyval(curve, fine) / 1e5},
    points={"simulation": (100 * np.array(offsets), peaks / 1e5)},
    vlines={"optimum": 100 * optimum},
    ylimits=(0.95 * peaks.min() / 1e5, 1.05 * peaks.max() / 1e5),
    xlabel="ring distance from the rod end s [cm]",
    ylabel="largest surface field [kV/cm]",
    title="Optimisation of the ring position",
    ax=fig["sweep"],
)
plots.show()

# ## Going further
#
# Change the ring radius or the tube radius in `grading_ring_position_optimisation_geometry.py` and
# optimise both together; give the rod a weather shed profile; or put the ring only at the
# high-voltage end, where most of the stress is.
