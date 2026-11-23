"""Generated from the OneSimulation example notebook.
Needs the installed, activated OneSimulation runtime. See
https://johmey.github.io/OneSimulation-Showcase/examples/notebooks.html
"""


# # Sod shock tube with WENO5 and RK4 in Python
#
# This notebook sets up the one-dimensional compressible Euler equations with `Problem` and `Simulation`. It builds the initial state directly and computes the subsequent flow with the solver. You will need an activated OneSimulation Runtime Alpha **0.2.1a4**, NumPy and Matplotlib.
#
# The initial left and right states have densities of 1 and 0.125, pressures of 1 and 0.1, and zero velocity. The ratio of specific heats is γ = 1.4. The domain is x ∈ [0, 1], both ends are transmissive, and the final time is t = 0.2. The Jiang–Shu WENO5 scheme reconstructs the split fluxes component by component. It uses high-order reconstruction in smooth regions and nonlinear weights at discontinuities. Positivity is checked for this run. The example has no experimental comparison.

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.numerics import Numerics
from onesim.numerics.fdm import (
    FiniteDifference, GlobalLaxFriedrichsSplit, WENO5Flux, WENOFluxDifference,
)
from onesim.numerics.time import ClassicalRungeKutta4, TimeStepLimits
from onesim.physics.hydrodynamics import (
    CompressibleEuler, IdealGasEquationOfState, TransmissiveBoundary,
)
from onesim.results.storage import MemoryOutputSink, Output, ResultRetentionPolicy
from onesim.simulation import ExecutionOptions, Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue

POINTS = 1600
DT = 0.00005
OUTPUT = Path('sod_results')


# ## Mesh and time integration
#
# The mesh has 1,600 equally spaced points. Classical RK4 advances the solution in time. The runtime may reduce the requested maximum step `DT` to satisfy the CFL limit. It retains the final fields and saves 21 states for the positivity checks.

# Initial states: left (rho,p,u)=(1,1,0), right (0.125,0.1,0).
gamma, end_time = 1.4, 0.2
geometry = Line(start=(0.0, 0.0, 0.0), end=(1.0, 0.0, 0.0))
gas = CompressibleEuler(
    domain=geometry.line, equation_of_state=IdealGasEquationOfState(gamma=gamma),
)
problem = Problem(
    name='Sod shock tube with the Euler API', geometry=geometry, physics=(gas,),
    initial_conditions=(
        InitialValue(gas.density, lambda x: np.where(x < 0.5, 1.0, 0.125)),
        InitialValue(gas.momentum, 0.0),
        InitialValue(gas.total_energy, lambda x: np.where(x < 0.5, 1.0, 0.1) / (gamma - 1)),
    ),
    boundary_conditions=(TransmissiveBoundary(physics=gas, boundary=geometry.boundaries.all),),
)
mesh = StructuredLineMesher(cells=POINTS).generate(geometry)
numerics = Numerics(
    spatial=FiniteDifference(gradient=WENOFluxDifference(
        flux_splitting=GlobalLaxFriedrichsSplit(), reconstruction=WENO5Flux(),
    )),
    time_integrator=ClassicalRungeKutta4(),
    step_limits=TimeStepLimits(cfl_limit=0.4, stability_safety_factor=0.4),
)
simulation = Simulation(problem=problem, mesh=mesh, numerics=numerics)
simulation.validate().require_valid()
snapshots = MemoryOutputSink()
started = time.perf_counter()
result = simulation.run(
    start_time=0.0, end_time=end_time, time_step=DT,
    execution_options=ExecutionOptions(result_retention=ResultRetentionPolicy.final_only()),
    output=Output(fields=(gas.density, gas.pressure), interval=0.01, sink=snapshots),
)
elapsed = time.perf_counter() - started


# ## Positivity and conservation
#
# Check that density and pressure remain positive in every saved state. At t = 0.2 the waves have yet to reach the open boundaries, so the total mass and energy should still match their initial values to within numerical precision. The pressure difference at the ends increases the total momentum. The momentum check includes that boundary contribution. The CSV contains the fields computed by the solver.

# Check the computed fields.
x = np.asarray(mesh.cell_centers[:, 0])
rho, velocity, pressure, energy, momentum = [
    np.asarray(result.field(field).values)
    for field in (gas.density, gas.velocity, gas.pressure, gas.total_energy, gas.momentum)
]
assert np.isfinite([rho, velocity, pressure, energy, momentum]).all()
assert rho.min() > 0 and pressure.min() > 0
assert all(np.min(field.values) > 0 for snap in snapshots.snapshots for field in snap.fields.values())
mass_error = abs(float(rho @ mesh.cell_volumes) - 0.5625)
energy_error = abs(float(energy @ mesh.cell_volumes) - 1.375)
# The pressure difference across the open ends produces a momentum gain of (1-0.1)*t.
momentum_balance_error = abs(float(momentum @ mesh.cell_volumes) - (1.0 - 0.1) * end_time)
assert max(mass_error, energy_error, momentum_balance_error) < 1e-9
verification = dict(
    passed=True, points=POINTS, requested_dt=DT, final_time=end_time,
    solve_seconds=elapsed, minimum_density=float(rho.min()), minimum_pressure=float(pressure.min()),
    mass_absolute_error=mass_error, energy_absolute_error=energy_error,
    momentum_boundary_balance_absolute_error=momentum_balance_error,
    positive_saved_snapshots=len(snapshots.snapshots),
    final_accepted_step=snapshots.snapshots[-1].accepted_step,
    spatial='WENO5-JS finite differences, componentwise global LF flux splitting',
    time_integrator='classical RK4', analytic_solution_used=False,
)
OUTPUT.mkdir(parents=True, exist_ok=True)
np.savetxt(OUTPUT / 'sod_fields.csv', np.column_stack((x, rho, velocity, pressure, energy, momentum)),
           delimiter=',', header='x,density,velocity,pressure,total_energy,momentum', comments='')
(OUTPUT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')


# ## Plot the fields
#
# The solid curves show the computed solution. The dashed curves show the prescribed initial state. To study convergence, rerun the model with different values of `POINTS` and `DT` and compare the numerical solutions. Measuring the exact error would require an independent reference solution.

# Plot the computed fields and show the initial state with a dashed line.
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor='#f6f1e7')
for ax, values, initial, title in zip(axes.flat, (rho, velocity, pressure, energy),
        (np.where(x < 0.5, 1.0, 0.125), np.zeros_like(x), np.where(x < 0.5, 1.0, 0.1),
         np.where(x < 0.5, 2.5, 0.25)), ('Density ρ', 'Velocity u', 'Pressure p', 'Total energy density E')):
    ax.set_facecolor('#fffdf7')
    ax.plot(x, initial, '--', color='#b4aa98', lw=1.1, label='Initial state at t = 0')
    ax.plot(x, values, color='#39715e', lw=2, label=f'Numerical solution at t = {end_time:g}')
    ax.set(xlabel='Position x', ylabel=title, xlim=(0, 1), title=title)
    ax.grid(alpha=0.16)
axes[0, 0].legend(frameon=False, fontsize=10)
fig.suptitle('Shock formation from a pressure discontinuity', fontsize=26, color='#203f36', x=.07, ha='left')
point_label = f'{POINTS:,}'
fig.text(.07, .915, f'Sod shock tube · {point_label} grid points · WENO5 + RK4 · t = 0.2', color='#39715e', fontsize=14)
fig.text(.07, .025, 'Direct Python API · dimensionless units · numerical solution and initial conditions · no experimental comparison', color='#6d746c')
fig.subplots_adjust(top=.84, bottom=.10, left=.08, right=.96, hspace=.34, wspace=.24)
fig.savefig(OUTPUT / 'sod_weno5.png', dpi=300)
fig.savefig(OUTPUT / 'sod_weno5.svg')
plt.close(fig)
print(json.dumps(verification, ensure_ascii=False))
