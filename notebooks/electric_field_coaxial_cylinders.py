"""OneSimulation example: electric field between coaxial cylinders (1D radial electrostatics).
Requires the installed, activated OneSimulation runtime.

The dielectric between the conductors of a coaxial capacitor is a radial line from the inner to the
outer radius; OneSimulation solves the axisymmetric Poisson equation on it with cylindrical volumes
and surfaces. Potential, field and capacitance per length are checked against the closed-form
coaxial solution on linear and quadratic elements.
"""

# # Electric field between coaxial cylinders
#
# A coaxial arrangement: an inner conductor (radius $a=10$ mm) at the potential $V_0=100$ V and a
# grounded outer conductor (radius $b=30$ mm), with vacuum in between. The arrangement is long and
# rotationally symmetric, so the potential depends on the radius only, and Laplace's equation in
# cylindrical coordinates becomes one-dimensional:
#
# $$\frac{1}{r}\frac{\mathrm d}{\mathrm dr}\Big(\varepsilon_0\,r\,\frac{\mathrm dV}{\mathrm dr}\Big)=0,
#   \qquad V(a)=V_0,\quad V(b)=0 .$$
#
# OneSimulation solves the Poisson equation of `onesim.physics.electrostatics` on a line from $a$ to
# $b$; the radial line space weights every integral with the cylinder area $2\pi r\,\ell$, so the
# reaction of the potential on the inner end is the physical charge of a conductor piece of length
# $\ell=20$ mm. The model is one-dimensional and needs no CAD file.
#
# | end | condition |
# |---|---|
# | $r=a$, inner conductor | $V=V_0$ |
# | $r=b$, outer conductor | $V=0$ |
#
# **Reference.** The coaxial capacitor has the closed-form solution (e.g. Jackson, *Classical
# Electrodynamics*, 3rd ed., Sec. 1.13)
#
# $$V(r)=V_0\,\frac{\ln(b/r)}{\ln(b/a)},\qquad E_r(r)=\frac{V_0}{r\,\ln(b/a)},\qquad
#   \frac{C}{\ell}=\frac{2\pi\varepsilon_0}{\ln(b/a)} .$$

import numpy as np
from onesim.cad.d1 import Line
from onesim.materials import VacuumPermittivity
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import (
    EmbeddedLineH1SpaceFactory,
    FiniteElement,
    Lagrange,
    RevolvedLineH1SpaceFactory,
)
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import FixedPotential, PoissonEquation
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Model
#
# The radial line runs along $x$ from $a$ to $b$; the axis of revolution is the $z$ axis.
# Linear and quadratic elements on 8, 16, 32 and 64 cells form the mesh study.

a, b, length, voltage = 0.010, 0.030, 0.020, 100.0
radial_line = Line(start=(a, 0.0, 0.0), end=(b, 0.0, 0.0))
ends = radial_line.boundaries
model = PoissonEquation(
    domain=radial_line.line, permittivity=VacuumPermittivity(), charge_density=0.0
)
problem = Problem(
    name="Coaxial capacitor",
    geometry=radial_line,
    physics=(model,),
    boundary_conditions=(
        FixedPotential(boundary=ends.start, physics=model, value=voltage),
        FixedPotential(boundary=ends.end, physics=model, value=0.0),
    ),
)
radial_space = EmbeddedLineH1SpaceFactory(
    line_factory=RevolvedLineH1SpaceFactory(axis=(0.0, 0.0, 1.0), axial_length=length)
)
cells = (8, 16, 32, 64)
runs = {
    (order, n): Simulation(
        problem=problem,
        mesh=StructuredLineMesher(cells=n).generate(radial_line),
        numerics=Numerics(
            spatial=FiniteElement(
                element=Lagrange(order=order), quadrature_order=5, provider=radial_space
            ),
            linear_solver=SparseDirect(),
        ),
    ).solve_stationary()
    for order in (1, 2)
    for n in cells
}

# ## Comparison with the closed form
#
# Potential and field on a fine set of radii, and the capacitance per length from the charge on the
# inner conductor.

eps0 = float(VacuumPermittivity().evaluate())
r = np.linspace(a + 1e-5, b - 1e-5, 401)
points = np.column_stack((r, np.zeros_like(r), np.zeros_like(r)))
exact_potential = voltage * np.log(b / r) / np.log(b / a)
exact_field = voltage / (r * np.log(b / a))
exact_capacitance = 2 * np.pi * eps0 / np.log(b / a)
errors = {}
print(
    "elements    cells   potential error [% of V0]   field error [%]   C' [pF/m]   C' error"
)
for (order, n), result in runs.items():
    potential = result.field(model.potential).evaluate(points).reshape(-1)
    field = result.field(model.electric_field).evaluate(points)[:, 0]
    capacitance = result.field(model.potential).reaction(ends.start) / (
        voltage * length
    )
    errors[order, n] = (
        np.max(np.abs(potential - exact_potential)) / voltage,
        np.max(np.abs(field / exact_field - 1)),
        abs(capacitance / exact_capacitance - 1),
    )
    print(
        f"P{order} {n:12d} {100 * errors[order, n][0]:24.5f} {100 * errors[order, n][1]:17.4f} "
        f"{1e12 * capacitance:11.4f} {errors[order, n][2]:10.1e}"
    )
print(f"closed form: C' = 2 pi eps0 / ln(b/a) = {1e12 * exact_capacitance:.4f} pF/m")

# ## Figures
#
# Potential and field on the coarsest quadratic mesh against the closed form, and the mesh study.

coarse = runs[2, 8]
markers = np.linspace(a, b, 25)
marks = np.column_stack((markers, np.zeros_like(markers), np.zeros_like(markers)))
fig = plots.figure(1, 2, size=(12, 4.5))
plots.line(
    points={
        "P2, 8 cells": (
            1e3 * markers,
            coarse.field(model.potential).evaluate(marks).reshape(-1),
        )
    },
    reference={"closed form": (1e3 * r, exact_potential)},
    xlabel="radius r [mm]",
    ylabel="potential [V]",
    title="Potential V(r)",
    ax=fig[0],
)
plots.line(
    points={
        "P2, 8 cells": (
            1e3 * markers,
            coarse.field(model.electric_field).evaluate(marks)[:, 0],
        )
    },
    reference={"closed form": (1e3 * r, exact_field)},
    xlabel="radius r [mm]",
    ylabel="radial field [V/m]",
    title="Radial field E_r(r)",
    ax=fig[1],
)
plots.show()

fig = plots.figure(1, 2, size=(12, 4.5))
plots.convergence(
    [(b - a) / n for n in cells],
    {f"P{order}": [errors[order, n][1] for n in cells] for order in (1, 2)},
    xlabel="cell size h [m]",
    ylabel="largest relative field error",
    title="Field",
    ax=fig[0],
)
plots.convergence(
    [(b - a) / n for n in cells],
    {f"P{order}": [errors[order, n][2] for n in cells] for order in (1, 2)},
    xlabel="cell size h [m]",
    ylabel="relative error of C'",
    title="Capacitance per length",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Change the radii and see how the field at the inner conductor grows as $b/a$ shrinks; for a given
# outer radius it is smallest at $b/a=e$, the classic design rule for high-voltage cables. A
# `charge_density` turns the model into a space-charge problem, and a second dielectric layer shows
# the field jump at the interface.
