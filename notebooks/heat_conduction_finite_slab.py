# # Cooling a finite slab from both faces
#
# A hot slab touches two cold surfaces at once. How long does the core stay warm? The coordinate $x$ runs from −1 to 1 in units of the half-thickness $b$. Temperature $\\theta=(T-T_s)/(T_i-T_s)$ starts at one. Time is measured by the Fourier number $\\tau=\\alpha t/b^2$, giving $\\partial_\\tau\\theta=\\partial_x^2\\theta$.

import numpy as np
from onesim.cad.d1 import Line
from onesim.materials import IsotropicSolid
from onesim.meshing import NonuniformLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.numerics.time import AdaptiveTimeStep, BackwardEuler, BDF
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction
from onesim.simulation import Problem, Simulation, smooth_step
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme

theme.set()

# ## Cooling through the faces
#
# A smooth step lowers both boundary temperatures from one to zero over the first $2\\times10^{-6}$ units of time. The line elements are shorter near the surfaces, where the early temperature gradients are steepest. Adaptive BDF2 steps follow the cooling transient.

slab = Line(start=(-1, 0, 0), end=(1, 0, 0))
solid = IsotropicSolid(density=1, specific_heat_capacity=1, thermal_conductivity=1,
                       youngs_modulus=1, poisson_ratio=0.3, thermal_expansion_coefficient=0)
heat = HeatConduction(domain=slab.line, material=solid)
wall = smooth_step(0, 2e-6, low=1, high=0)
problem = Problem(name="Slab cooling", geometry=slab, physics=(heat,),
    initial_conditions=(InitialValue(heat.temperature, 1),),
    boundary_conditions=tuple(FixedTemperature(physics=heat, boundary=face, value=wall)
                              for face in (slab.boundaries.start, slab.boundaries.end)))
method = Numerics(spatial=FiniteElement(element=Lagrange(order=1)),
                  time_integrator=BDF(order=2, startup=BackwardEuler()), linear_solver=SparseDirect())

# ## Check the full temperature profile
#
# For an instantaneous change at both faces, Bird, Stewart and Lightfoot give the exact cosine series in *Transport Phenomena*, 2nd ed., Eq. 12.1-31. Its modes have wave numbers $q_n=(n+\\tfrac12)\\pi$ and decay at rates $q_n^2$. Compare the profiles at six times on two meshes, using the relative $L^2$ error over the whole slab.

times = np.array([0.01, 0.04, 0.1, 0.2, 0.4, 0.6])
x = np.linspace(-1, 1, 201)
probes = np.column_stack((x, np.zeros((len(x), 2))))
n = np.arange(50)
q = (n + 0.5) * np.pi
modes = (2 * (-1.)**n / q)[:, None] * np.cos(q[:, None] * x)
reference = np.exp(-times[:, None] * q**2) @ modes
errors = {}
for cells in (64, 128):
    faces = 1 - np.cos(np.linspace(0, np.pi, cells + 1))
    mesh = NonuniformLineMesher(arclength_faces=tuple(faces)).generate(slab)
    steps = AdaptiveTimeStep(initial=2e-7, minimum=1e-10, maximum=0.005,
                             relative_tolerance=0, absolute_tolerance=0.00002 * 64 / cells)
    result = Simulation(problem=problem, mesh=mesh, numerics=method).run(start_time=0, end_time=0.6, time_step=steps)
    history = result.field(heat.temperature).time_interpolant()
    profile = np.array([history(probes, t) for t in times])
    errors[cells] = np.linalg.norm(profile-reference, axis=1) / np.linalg.norm(reference, axis=1)
    print(f"{cells} cells: maximum relative profile error {errors[cells].max():.3%}")
print(f"Core at τ = 0.6: computed {profile[-1, len(x)//2]:.5f}, series {reference[-1, len(x)//2]:.5f}")

fig = plots.figure(1, 2, size=(11.5, 4.8))
plots.line(x, {f"τ = {t:g}": row for t, row in zip(times, profile)},
           title="Cooling moves inward", xlabel="Position / half-thickness", ylabel="Reduced temperature", legend="lower center", ax=fig[0])
plots.line(times, {f"{cells} cells": 100*error for cells, error in errors.items()}, markers=True,
           title="Agreement with the cosine series", xlabel="Fourier number", ylabel="Relative profile error [%]",
           scale="loglog", ax=fig[1])
plots.show()
