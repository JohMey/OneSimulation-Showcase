"""OneSimulation example: optimizing the currents of a coil system for a uniform field (2D-axisymmetric).
Requires the installed, activated OneSimulation runtime.

Six coaxial coils, placed symmetrically about a midplane, should produce a uniform axial field of
1 mT over 40 cm of their axis. The meridian section is cut from STEP parts and solved with the
axisymmetric magnetostatic model; the simulation study service fits the three coil currents by
Levenberg-Marquardt with real solves. The optimum is checked against a least-squares fit built from
the closed-form axial field of thick coils.
"""

# # Optimizing the currents of a coil system for a uniform field
#
# Magnetic resonance, beam lines and magnetometer calibration need a magnetic field that is uniform
# over a long stretch of an axis. Here six identical coils (inner radius 0.10 m, outer radius
# 0.12 m, 20 mm high) sit symmetrically about the plane $z=0$, at $z=\pm0.05$, $\pm0.15$ and
# $\pm0.25$ m. Each symmetric pair carries the same ampere-turns $NI_k$; the three values are the
# design variables. The goal is $B_z=B_0=1$ mT on the axis for $|z|\le0.2$ m, measured by the mean
# squared relative deviation at 21 points,
#
# $$J(NI_1,NI_2,NI_3)=\frac1{21}\sum_j\Big(\frac{B_z(0,z_j)}{B_0}-1\Big)^2 .$$
#
# The coils are rotationally symmetric, so OneSimulation solves the axisymmetric magnetostatic
# equation for the azimuthal vector potential $A_\varphi(r,z)$ with quadratic elements on the
# meridian half-plane above the midplane. `SimulationStudyService.fit` minimizes $J$ with real
# solves by the Levenberg-Marquardt method: the 21 relative deviations are the residuals, their
# Jacobian over the three currents comes from difference quotients of solved simulations.
#
# The domain is the meridian section of `coil_current_optimization_axisymmetric.step`, a half
# cylinder of 1 m radius and height; the windings are the sections of
# `coil_current_optimization_axisymmetric_coil_1.step` to `_coil_3.step`. The script
# `coil_current_optimization_axisymmetric_geometry.py` builds them with the CAD objects of
# OneSimulation.
#
# | edge | condition |
# |---|---|
# | midplane $z=0$ | field normal, $\mathbf n\times\mathbf H=0$ (`MagneticSymmetry`) |
# | axis $r=0$ | axis of the revolved space |
# | $r=1$ m and $z=1$ m | $A_\varphi=0$ (`MagneticInsulation`) |
#
# **Reference.** On its axis a coil of rectangular winding section (radii $R_1$, $R_2$, height $h$,
# centre $z_k$) with uniform current density has the closed-form field (e.g. Montgomery, *Solenoid
# Magnet Design*, 1969)
#
# $$B_z(z)=\frac{\mu_0NI}{2h(R_2-R_1)}\Big[u\ln\frac{R_2+\sqrt{R_2^2+u^2}}{R_1+\sqrt{R_1^2+u^2}}\Big]_{u=z-z_k-h/2}^{u=z-z_k+h/2}.$$
#
# The field is linear in the three ampere-turns, so the best currents in free space follow from a
# linear least-squares fit; the currents fitted with the simulation must come out close to it.
# The insulated boundary 1 m away changes the axial field by about 0.1 %.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import VacuumPermeability
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange, RevolvedH1SpaceFactory
from onesim.numerics.linear import SparseDirect
from onesim.physics.magnetostatics import (
    MagneticInsulation,
    MagneticSymmetry,
    Magnetostatics,
    RegionCurrent,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.studies import SimulationStudyService, StudyParameter
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `section` cuts the parts in the plane $y=0$; the section coordinates are $(r, z)$. The current
# density is assigned at the quadrature points inside the winding sections, so the triangles are
# small at the windings and grow towards the far boundary.

meridian = lambda name: load(
    f"coil_current_optimization_axisymmetric{name}.step"
).section((0, 1, 0))
domain = meridian("")
windings = [meridian(f"_coil_{k}") for k in (1, 2, 3)]
edges = domain.boundaries
midplane = domain.select(
    edges=lambda edge: all(abs(point[1]) < 1e-12 for point in edge.points)
)
axis = domain.select(
    edges=lambda edge: all(abs(point[0]) < 1e-12 for point in edge.points)
)
inner, outer = windings[0].bounding_box[0][0], windings[0].bounding_box[1][0]
height = windings[0].size[1]
centres = np.array(
    [0.5 * (w.bounding_box[0][1] + w.bounding_box[1][1]) for w in windings]
)
near = lambda points: np.min(
    [
        np.hypot(
            np.maximum(
                np.abs(points[:, 0] - 0.5 * (inner + outer)) - 0.5 * (outer - inner), 0
            ),
            np.maximum(np.abs(points[:, 1] - c) - 0.5 * height, 0),
        )
        for c in centres
    ],
    axis=0,
)
mesh = TriangularMesher(
    target_size=0.12,
    refinements=1,
    size_field=lambda points: np.clip(0.005 + near(points) / 2, 0.005, 0.12),
).generate(domain)

fig = plots.figure(1, 2, size=(14, 6))
plots.geometry(
    [
        load("coil_current_optimization_axisymmetric.step").named(
            "space around the coils"
        ),
        *(
            load(f"coil_current_optimization_axisymmetric_coil_{k}.step").named(
                f"coil {k}"
            )
            for k in (1, 2, 3)
        ),
    ],
    camera=plots.Camera(elevation=20, azimuth=-60),
    title="Upper half of the coil system (x > 0)",
    ax=fig[0],
)
plots.mesh(
    mesh,
    groups=("all",),
    legend=False,
    title=f"Meridian section: {mesh.cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model and optimisation
#
# Each winding has 100 turns; the parameters are the coil currents (A). The study service builds
# and solves one simulation per point and reads the axial field at the 21 points.

turns, target = 100, 1e-3
mu0 = float(VacuumPermeability().evaluate())
z_target = np.linspace(0.0, 0.2, 21)
on_axis = np.column_stack((np.zeros_like(z_target), z_target))
numerics = Numerics(
    spatial=FiniteElement(
        element=Lagrange(order=2), quadrature_order=4, provider=RevolvedH1SpaceFactory()
    ),
    linear_solver=SparseDirect(),
)
model_for = lambda values: Magnetostatics(
    domain=domain.domain(),
    permeability=VacuumPermeability(),
    current_sources=tuple(
        RegionCurrent.coil(
            winding,
            turns=turns,
            current=values[f"I{k}"],
            cross_section=(outer - inner) * height,
            within=domain.domain(),
        )
        for k, winding in zip((1, 2, 3), windings)
    ),
)
models = []
simulation_for = lambda model: Simulation(
    problem=Problem(
        name="Coil system",
        geometry=domain,
        physics=(model,),
        boundary_conditions=(
            MagneticSymmetry(boundary=midplane, physics=model),
            MagneticInsulation(boundary=edges.all - midplane - axis, physics=model),
        ),
    ),
    mesh=mesh,
    numerics=numerics,
)
service = SimulationStudyService(
    tuple(StudyParameter(f"I{k}", 0.0, 10.0) for k in (1, 2, 3)),
    lambda values: simulation_for(models.append(model_for(values)) or models[-1]),
    lambda result: {
        f"b{j}": value
        for j, value in enumerate(
            result.field(models[-1].magnetic_flux_density).evaluate(on_axis)[:, 2]
            / target
        )
    },
)
best = service.fit(
    residuals=lambda m: np.array([m[f"b{j}"] - 1 for j in range(len(z_target))]),
    start={"I1": 2.0, "I2": 2.0, "I3": 2.0},
    method="levenberg_marquardt",
    maximum_runs=100,
)
deviation = lambda run: float(
    np.mean([(run.value(f"b{j}") - 1) ** 2 for j in range(len(z_target))])
)

# ## Reference: least squares with the closed-form coil field

coil_field = (
    lambda z, zc: mu0
    * turns
    / (2 * height * (outer - inner))
    * sum(
        sign * u * np.log((outer + np.hypot(outer, u)) / (inner + np.hypot(inner, u)))
        for sign, u in ((1, z - zc + height / 2), (-1, z - zc - height / 2))
    )
)
matrix = np.column_stack(
    [coil_field(z_target, c) + coil_field(z_target, -c) for c in centres]
)
reference, *_ = np.linalg.lstsq(matrix, np.full_like(z_target, target), rcond=None)

# ## Results
#
# The two optima belong to two models: the fit minimizes the deviation of the simulated field (with
# the insulated boundary 1 m away and the windings on the mesh), the closed-form least squares that
# of the free-space formula. Each set of currents is therefore best in its own model; both rms values
# are shown in both models.

optimum = np.array([best.parameters[f"I{k}"] for k in (1, 2, 3)])
print(
    f"{best.run_count} solved simulations, converged: {best.converged} ({best.message})"
)
print(
    "coil   centre [m]   fitted current [A]   least squares, closed form [A]   difference"
)
for k, (c, value, ref) in enumerate(zip(centres, optimum, reference), start=1):
    print(
        f"{k:4d} {c:12.2f} {value:23.4f} {ref:32.4f} {100 * (value / ref - 1):+11.2f} %"
    )
start_run = service.executed_runs[0]
closed_rms = lambda currents: 100 * np.sqrt(
    np.mean((matrix @ currents / target - 1) ** 2)
)
print(
    f"rms deviation from 1 mT on |z| <= 0.2 m at the start: {100 * np.sqrt(deviation(start_run)):.3f} % (simulation)"
)
print(
    f"fitted currents:       simulation {100 * np.sqrt(np.mean(best.residuals ** 2)):.3f} %, closed form {closed_rms(optimum):.3f} %"
)
print(f"least-squares currents: closed form {closed_rms(reference):.3f} %")

# ## Figure
#
# The flux density in the meridian half-plane above the midplane with the fitted currents,
# the axial field before and after the optimisation, and the course of the objective over the runs.

final = service.build(dict(best.parameters)).solve_stationary()
field = final.field(models[-1].magnetic_flux_density)
z = np.linspace(0.0, 0.4, 161)
fig = plots.figure(layout=[["map", "axis"], ["map", "runs"]], size=(15, 10))
plots.surface(
    field,
    component=lambda B, position: 1e3 * np.linalg.norm(B, axis=1),
    limits=(0, 2),
    contours=21,
    subdivisions=2,
    label="|B| [mT]",
    title="Flux density, fitted currents",
    xlimits=(0, 0.3),
    ylimits=(0, 0.4),
    xlabel="r [m]",
    ylabel="z [m]",
    ax=fig["map"],
)
plots.line(
    points={
        "simulation, fitted currents": (
            z[::4],
            1e3
            * field.evaluate(np.column_stack((np.zeros_like(z[::4]), z[::4])))[:, 2],
        )
    },
    reference={
        "closed form, fitted currents": (
            z,
            1e3
            * np.column_stack([coil_field(z, c) + coil_field(z, -c) for c in centres])
            @ optimum,
        ),
        "closed form, start currents": (
            z,
            1e3
            * np.column_stack([coil_field(z, c) + coil_field(z, -c) for c in centres])
            @ np.full(3, 2.0),
        ),
    },
    vlines=[0.2],
    xlabel="z on the axis [m]",
    ylabel="B_z [mT]",
    title="Axial field",
    ax=fig["axis"],
)
plots.line(
    points={
        "objective J": (
            np.arange(len(service.executed_runs)),
            [deviation(run) for run in service.executed_runs],
        )
    },
    scale="logy",
    xlabel="solved simulation",
    ylabel="mean squared relative deviation",
    title="Optimisation history",
    ax=fig["runs"],
)
plots.show()

# ## Going further
#
# Add the coil positions as further `StudyParameter`s (one geometry file per position or a
# `move_faces` on the winding parts), weight the off-axis field to widen the uniform region, or
# add a second objective such as the total ampere-turns to trade uniformity against power.
