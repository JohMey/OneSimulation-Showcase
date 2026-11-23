"""OneSimulation example: effective diffusivity of a porous medium from its resolved pore geometry.
Requires the installed, activated OneSimulation runtime.

Diffusion through the pore space between impermeable grains is solved on the resolved
two-dimensional microstructure, a square array of cylindrical grains read from STEP files for four
solid fractions. The flux through the strip gives the effective diffusivity of the homogenised
medium, which is checked against the Rayleigh-Perrins formula for a square array of cylinders.
"""

# # Effective diffusivity of a porous medium
#
# Transport through porous materials is usually described by a homogeneous model with an
# effective diffusivity $D_\mathrm{eff}$, because the pores are far smaller than the device. The
# effective value follows from the microstructure: a representative piece of it is resolved, a
# concentration difference is applied across it, and the steady flux is measured,
#
# $$\nabla\cdot(D\,\nabla c)=0\ \text{in the pores},\qquad
#   D_\mathrm{eff}=\frac{J\,L}{H\,(c_\mathrm{in}-c_\mathrm{out})},$$
#
# with the molar flux $J$ per unit depth through a strip of length $L$ and height $H$, so that
# $D_\mathrm{eff}$ refers to the total cross-section. Here the grains are impermeable cylinders on
# a square lattice with the solid fraction $f$; OneSimulation solves the pore diffusion with the
# `SpeciesTransport` model and quadratic triangles on the cross-section of each STEP file.
#
# | Boundary | Condition |
# |---|---|
# | $x=0$ (`x_min`) | $c=c_\mathrm{in}=1$ mol/m³ |
# | $x=L$ (`x_max`) | $c=c_\mathrm{out}=0$ |
# | top, bottom, grain surfaces | no flux |
#
# By symmetry of the square array the planes between grain rows and columns are planes of
# symmetry, so a strip of 6 x 2 cells gives the value of the infinite array.
#
# **Reference.** Lord Rayleigh, *On the influence of obstacles arranged in rectangular order upon
# the properties of a medium*, Phil. Mag. 34, 481-502 (1892); W. T. Perrins, D. R. McKenzie and
# R. C. McPhedran, Proc. R. Soc. Lond. A 369, 207-225 (1979). For impermeable cylinders on a square
# lattice
#
# $$\frac{D_\mathrm{eff}}{D}=1-\frac{2f}{1+f-\dfrac{0.305827\,f^4}{1-1.402958\,f^8}-0.013362\,f^8},$$
#
# which reduces to Maxwell's dilute limit $(1-f)/(1+f)$ for small $f$. The tortuosity factor
# $\tau=\varepsilon D/D_\mathrm{eff}$ with the porosity $\varepsilon=1-f$ follows directly.

import numpy as np
from onesim.cad.d1 import Line
from onesim.cad.exchange import load
from onesim.meshing import StructuredLineMesher, TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import Newton
from onesim.numerics.time import BDF, BackwardEuler
from onesim.physics.chemistry import (
    FixedConcentration,
    SpeciesTransport,
    TransportSpecies,
)
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()


# ## Geometry and mesh
#
# Four strips with solid fractions of 10 % to 70 %, cut at mid-thickness; quadratic triangles
# with 0.04 mm edges, refined to 0.015 mm on the grain surfaces.

fractions = (0.1, 0.3, 0.5, 0.7)
files = {f: f"porous_effective_diffusivity_{round(100 * f)}.step" for f in fractions}
files[0.5] = "porous_effective_diffusivity.step"
parts = {f: load(name) for f, name in files.items()}
planes = {f: part.section((0, 0, 1), origin=(0, 0, 5e-5)) for f, part in parts.items()}
meshes = {
    f: TriangularMesher(target_size=4e-5, local_sizes={"grains": 1.5e-5}).generate(plane)
    for f, plane in planes.items()
}
length, height = planes[0.5].size[:2]
fig = plots.figure(1, 2, size=(15, 5))
plots.geometry(
    parts[0.5],
    selection={
        "grain surfaces": parts[0.5].boundaries.grains,
        "inlet face": parts[0.5].boundaries.x_min,
    },
    labels=False,
    camera=plots.Camera(elevation=35, azimuth=-60),
    title="Porous strip, solid fraction 50 % (STEP)",
    ax=fig[0],
)
plots.mesh(
    meshes[0.5],
    groups=("all",),
    legend=False,
    title=f"Pore space: {meshes[0.5].cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model and solution
#
# One stationary diffusion problem per solid fraction, with the molecular diffusivity of a
# small solute in water.

diffusivity = 1e-9
spatial = FiniteElement(element=Lagrange(order=2))
fields = {}
for fraction, plane in planes.items():
    model = SpeciesTransport(
        domain=plane.domain(), species=(TransportSpecies("A", diffusivity=diffusivity),)
    )
    ends = (
        FixedConcentration(physics=model, boundary=plane.boundaries.x_min, value=1.0),
        FixedConcentration(physics=model, boundary=plane.boundaries.x_max, value=0.0),
    )
    problem = Problem(name="Pore strip", geometry=plane, physics=(model,), boundary_conditions=ends)
    simulation = Simulation(
        problem=problem,
        mesh=meshes[fraction],
        numerics=Numerics(spatial=spatial, linear_solver=SparseDirect()),
    )
    fields[fraction] = simulation.solve_stationary().field(model.field("A"))

# ## Comparison with the Rayleigh-Perrins formula
#
# The outflow and the inflow must agree (conservation); their mean gives $D_\mathrm{eff}$.

rayleigh = lambda f: 1 - 2 * f / (1 + f - 0.305827 * f**4 / (1 - 1.402958 * f**8) - 0.013362 * f**8)
print(
    "  f    porosity   D_eff/D computed   Rayleigh-Perrins   deviation   Maxwell   tortuosity   in/out flux balance"
)
ratios = {}
for f, field in fields.items():
    inflow, outflow = field.boundary_flux(planes[f].boundaries.x_min), field.boundary_flux(
        planes[f].boundaries.x_max
    )
    ratios[f] = 0.5 * (outflow - inflow) * length / (height * diffusivity)
    print(
        f"{f:4.1f} {1 - f:9.2f} {ratios[f]:16.5f} {rayleigh(f):17.5f} {100 * (ratios[f] / rayleigh(f) - 1):+10.3f} % "
        f"{(1 - f) / (1 + f):9.4f} {(1 - f) / ratios[f]:11.3f} {abs(inflow + outflow) / outflow:16.1e}"
    )

# ## Does a homogeneous strip fill at the same rate?
#
# Initially the pores contain no solute. Fix the left and right end concentrations
# at one and zero. Compare the resolved 50 % strip with a homogeneous line.
# Its storage is the pore fraction $\varepsilon$, so the diffusion coefficient
# in its concentration equation is $D_\mathrm{eff}/\varepsilon$.
# Normalize both inventories by the available pore volume.

porosity = 0.5
line = Line(start=(0, 0, 0), end=(length, 0, 0))
plane = planes[0.5]
domains = [
    (
        "2D pores",
        plane,
        plane.domain(),
        plane.boundaries.x_min,
        plane.boundaries.x_max,
        TriangularMesher(target_size=9e-5).generate(plane),
        diffusivity,
        length * height * porosity,
    ),
    (
        "1D homogeneous",
        line,
        line.line,
        line.boundaries.start,
        line.boundaries.end,
        StructuredLineMesher(cells=60).generate(line),
        diffusivity * ratios[0.5] / porosity,
        length,
    ),
]
inventory = {}
for name, geometry, domain, left, right, mesh, coefficient, measure in domains:
    model = SpeciesTransport(
        domain=domain, species=(TransportSpecies("A", diffusivity=coefficient),)
    )
    ends = (
        FixedConcentration(physics=model, boundary=left, value=1.0),
        FixedConcentration(physics=model, boundary=right, value=0.0),
    )
    problem = Problem(
        name=name,
        geometry=geometry,
        physics=(model,),
        boundary_conditions=ends,
        initial_conditions=(InitialValue(model.field("A"), 0.0),),
    )
    numerics = Numerics(
        spatial=spatial,
        nonlinear_solver=Newton(linear_solver=SparseDirect(), absolute_tolerance=1e-15),
        time_integrator=BDF(order=2, startup=BackwardEuler()),
    )
    history = (
        Simulation(problem=problem, mesh=mesh, numerics=numerics)
        .run(start_time=0.0, end_time=14400.0, time_step=600.0)
        .field(model.field("A"))
    )
    times = history.times[1:]
    inventory[name] = np.array([history.at(t).integrate().value / measure for t in times])
print(
    f"After 4 h: pore-average concentration {inventory['2D pores'][-1]:.5f} mol/m³; "
    f"homogeneous model {inventory['1D homogeneous'][-1]:.5f} mol/m³."
)
print(
    f"Largest inventory difference across the recorded steps: "
    f"{max(abs(inventory['2D pores'] - inventory['1D homogeneous'])):.5f} mol/m³."
)

# ## Figure
#
# The concentration in the pores of the 50 % strip, and the effective diffusivity over the solid
# fraction against the Rayleigh-Perrins formula and Maxwell's dilute limit. The third panel compares
# the resolved and homogenized inventories over four hours.

fine = np.linspace(0.0, 0.75, 151)
fig = plots.figure(1, 3, size=(19, 5))
plots.surface(
    fields[0.5],
    limits=(0.0, 1.0),
    contours=11,
    subdivisions=2,
    xlabel="x [m]",
    ylabel="y [m]",
    label="c [mol/m³]",
    title="Concentration in the pores, f = 50 %",
    ax=fig[0],
)
plots.line(
    points={"OneSimulation": (np.array(fractions), np.array([ratios[f] for f in fractions]))},
    reference={
        "Rayleigh-Perrins": (fine, rayleigh(fine)),
        "Maxwell (1 - f) / (1 + f)": (fine, (1 - fine) / (1 + fine)),
    },
    xlabel="solid fraction f",
    ylabel="D_eff / D",
    title="Effective diffusivity of the square array",
    legend="upper right",
    ax=fig[1],
)
plots.line(
    times / 3600,
    {"1D homogeneous": inventory["1D homogeneous"]},
    points={"2D pores": (times / 3600, inventory["2D pores"])},
    xlabel="Time [h]",
    ylabel="Pore-average concentration [mol/m³]",
    title="Filling the 50 % strip",
    legend="lower right",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Replace the square array by a random packing of grains of different sizes, or let the grains
# overlap into a connected skeleton: the formula no longer applies, but the same strip gives
# $D_\mathrm{eff}$. Shift the array by half a cell between rows (a staggered array) and compare
# with the hexagonal-array result of Perrins et al.; at high solid fractions the two lattices
# differ strongly because the narrowest pore throat differs.
