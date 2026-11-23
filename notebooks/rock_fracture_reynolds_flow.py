"""Conductance of a water-filled rock fracture under a 20 mm head difference."""

# # Water flowing through a rough fracture
#
# A change in aperture has a large effect on flow. For a gap $b$, the discharge per unit width is $-b^3\nabla p/(12\mu)$. A 20 mm water-head difference drives flow from top to bottom of an 80 × 50 mm fracture. Three synthetic aperture maps separate the effects of wide channels, narrow bottlenecks, and a gap that varies in both directions.
#
# **Reference.** R. W. Zimmerman and G. S. Bodvarsson, *Hydraulic Conductivity of Rock Fractures*, Transport in Porous Media 23 (1996), 1–30. When the aperture varies across the flow, the channels conduct in parallel. When it varies downstream, their resistances add in series.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import NewtonianFluid
from onesim.meshing import TriangularMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import DiscontinuousLagrange, FiniteElement, MixedDiffusionElement, RaviartThomas
from onesim.numerics.linear import SparseDirect
from onesim.physics.hydrodynamics import DarcyFlow, DarcyFluxBoundary, DarcyPressureBoundary
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and aperture
#
# The STEP carrier supplies the planform and its named edges. The simulated aperture is a spatial coefficient in the plane. For the two striped maps it varies sinusoidally from 0.15 to 0.45 mm. The mesh view enlarges a 13 × 9 mm patch. Reducing the target edge length from 1.20 to 0.55 mm gives 30,508 triangles instead of 6,458.

part = load("rock_fracture_reynolds_flow.step")
plane = part.section((0, 0, 1), origin=(0, 0, part.size[2] / 2))
sides = plane.boundaries
mesh = TriangularMesher(target_size=0.00055).generate(plane)
print(f"Mesh: {mesh.cell_count:,} triangles, {len(mesh.coordinates):,} vertices")
fig = plots.figure(1, 2, size=(11, 4.2))
plots.geometry(part, selection={"Inlet": part.boundaries.y_max, "Outlet": part.boundaries.y_min},
               labels=False, length_unit="mm", title="Fracture plane", ax=fig[0])
plots.mesh(mesh, groups=("all",), legend=False, length_unit="mm", ax=fig[1],
           title=f"Mesh detail · {mesh.cell_count:,} triangles in total").limits((0.030, 0.043), (0.040, 0.049))
plots.show()

width, length = plane.size[:2]
gap, contrast = 3e-4, 0.5
kx, ky = 2 * np.pi / width, 2 * np.pi / length
maps = {"Parallel": lambda x, y: gap * (1 + contrast * np.cos(2 * kx * (x - 0.010))),
        "Series": lambda x, y: gap * (1 + contrast * np.cos(2 * ky * (y - 0.020))),
        "Crossed": lambda x, y: gap * (1 + 0.32 * np.cos(2 * kx * x + ky * y) + 0.18 * np.sin(kx * x - 2 * ky * y))}

# ## Solve for pressure and discharge
#
# A mixed Darcy element returns both the pressure and a conservative flux. Multiplying its normal flux integral by the nominal gap converts the two-dimensional result to m³/s. The side edges carry zero normal flux.

water = NewtonianFluid(density=1000.0, dynamic_viscosity=1e-3)
drop = 1000 * 9.81 * 0.020
element = MixedDiffusionElement(flux=RaviartThomas(0), scalar=DiscontinuousLagrange(0))
numerics = Numerics(spatial=FiniteElement(element=element), linear_solver=SparseDirect())
rates, fields = {}, {}
for name, aperture in maps.items():
    model = DarcyFlow(domain=plane.domain(), material=water, permeability=lambda x, y: aperture(x, y)**3 / (12 * gap))
    boundaries = (DarcyPressureBoundary(physics=model, boundary=sides.y_max, value=drop),
                  DarcyPressureBoundary(physics=model, boundary=sides.y_min, value=0),
                  DarcyFluxBoundary(physics=model, boundary=sides.x_min | sides.x_max, flux=0))
    problem = Problem(name=name, geometry=plane, physics=(model,), boundary_conditions=boundaries)
    result = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary()
    fields[name] = result.field(model.darcy_velocity)
    rates[name] = gap * fields[name].integrate(boundary=sides.y_min, normal_component=True).value

# ## Compare the channel directions
#
# Refinement reduces the error for the parallel channels from 0.182 % to 0.038 %. Normalize the discharge by that of a flat slot of aperture $b_0$. For sinusoidal stripes, the arithmetic mean of $(b/b_0)^3$ is $1+3m^2/2$. Its harmonic mean is $2(1-m^2)^{5/2}/(2+m^2)$, where $m$ is the fractional aperture variation.

flat = gap**3 * width * drop / (12e-3 * length)
exact = {"Parallel": 1 + 1.5 * contrast**2, "Series": 2 * (1 - contrast**2)**2.5 / (2 + contrast**2)}
for name, rate in rates.items():
    print(f"{name:9s}   {rate * 1e9:.3f} µL/s   Q/Q_flat = {rate / flat:.5f}")
for name, value in exact.items():
    print(f"{name:9s}   deviation from the cubic-law reference: {100 * (rates[name] / (flat * value) - 1):+.3f} %")

x, y = mesh.coordinates[:, :2].T
fig = plots.figure(1, 2, size=(12, 4.6))
plots.heatmap(x * 1e3, y * 1e3, maps["Crossed"](x, y) * 1e3, triangles=mesh.cells, contours=8,
              xlabel="x [mm]", ylabel="y [mm]", label="Aperture [mm]", title="Crossed aperture bands", ax=fig[0])
plots.surface(fields["Crossed"], component=lambda v, p: 1e6 * gap * np.linalg.norm(v, axis=1),
              label="Discharge per width [mm²/s]", length_unit="mm", title="Computed flow channels", ax=fig[1])
plots.show()
