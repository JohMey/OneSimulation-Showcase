"""Generated from the OneSimulation example notebook.
Needs the installed, activated OneSimulation runtime. See
https://johmey.github.io/OneSimulation-Showcase/examples/notebooks.html
"""


# # A damped oscillator with OneSimulation
#
# Use an activated installation of Runtime Alpha 0.2.1a4. The cells run in sequence and use only data defined here. Keep your licence key outside the notebook.
#
# $$m\,\ddot u+c\,\dot u+k\,u=0.$$
#
# A mass $m$ moves on a linear spring with stiffness $k$ and a viscous damper with coefficient $c$. There is no external force. The model represents a single degree of freedom with damping proportional to velocity.

from pathlib import Path
from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.results.storage import MemoryOutputSink, Output
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue

runtime_version = version("onesimulation")
print("OneSimulation:", runtime_version)
assert runtime_version == "0.2.1a4", "This example requires Runtime Alpha 0.2.1a4."

from onesim.physics.solid_mechanics import StructuralOscillator


# ## Parameters and model
#
# The reference solution assumes underdamping, $0<c<2\sqrt{mk}$. Supply the initial displacement and velocity in SI units.

mass = 1.0                    # kg
stiffness = 4.0               # N/m
damping = 0.2                 # N s/m
initial_displacement = 0.01   # m
initial_velocity = 0.0        # m/s
end_time = 10.0               # s
time_step = 0.02              # s

assert mass > 0 and stiffness > 0
assert 0 < damping < 2 * np.sqrt(mass * stiffness)

domain = LumpedDomain()
model = StructuralOscillator(
    domain=domain.system,
    mass=mass,
    stiffness=stiffness,
    damping=damping,
)
simulation = Simulation(
    problem=Problem(
        name="Damped oscillator reference case",
        geometry=domain,
        physics=(model,),
        initial_conditions=(
            InitialValue(model.displacement, initial_displacement),
            InitialValue(model.velocity, initial_velocity),
        ),
    ),
    mesh=LumpedMesh(domain),
    numerics=Numerics(
        spatial=LumpedParameters(),
        time_integrator=ClassicalRungeKutta4(),
    ),
)
simulation.validate().require_valid()

sink = MemoryOutputSink()
result = simulation.run(
    start_time=0.0,
    end_time=end_time,
    time_step=time_step,
    output=Output(
        fields=(model.displacement, model.velocity, model.mechanical_energy),
        interval=0.1,
        sink=sink,
    ),
)
times = np.asarray([float(s.time) for s in sink.snapshots])
displacements = np.asarray([
    float(np.asarray(s.fields[model.displacement.field_id].values).reshape(-1)[0])
    for s in sink.snapshots
])
velocities = np.asarray([
    float(np.asarray(s.fields[model.velocity.field_id].values).reshape(-1)[0])
    for s in sink.snapshots
])
energies = np.asarray([
    float(np.asarray(s.fields[model.mechanical_energy.field_id].values).reshape(-1)[0])
    for s in sink.snapshots
])
assert len(times) > 1 and np.isclose(times[-1], end_time)


# ## Comparison with the closed-form solution
#
# With $\alpha=c/(2m)$, $\omega=\sqrt{k/m-\alpha^2}$, $A=u_0$ and $B=(v_0+\alpha u_0)/\omega$, the displacement is
#
# $$u(t)=e^{-\alpha t}\left(A\cos(\omega t)+B\sin(\omega t)\right).$$
#
# Damping reduces mechanical energy $E=\tfrac12m v^2+\tfrac12k u^2$. The checks compare displacement and velocity with the solution and test this energy balance for the supplied parameters. The reference is mathematical and uses no measured data.

alpha = damping / (2 * mass)
omega = np.sqrt(stiffness / mass - alpha**2)
a = initial_displacement
b = (initial_velocity + alpha * a) / omega
reference_displacement = np.exp(-alpha * times) * (
    a * np.cos(omega * times) + b * np.sin(omega * times)
)
reference_velocity = np.exp(-alpha * times) * (
    (-alpha * a + omega * b) * np.cos(omega * times)
    + (-alpha * b - omega * a) * np.sin(omega * times)
)
max_displacement_error = float(np.max(np.abs(displacements - reference_displacement)))
max_velocity_error = float(np.max(np.abs(velocities - reference_velocity)))
assert max_displacement_error < 2e-8
assert max_velocity_error < 5e-8
assert np.allclose(energies, 0.5 * mass * velocities**2 + 0.5 * stiffness * displacements**2)
assert np.all(np.diff(energies) <= 1e-12), "Mechanical energy should not increase in this example."
assert energies[-1] < energies[0]
print(f"Largest displacement error: {max_displacement_error:.3e} m")
print(f"Largest velocity error: {max_velocity_error:.3e} m/s")
print(f"Initial / final energy: {energies[0]:.8f} / {energies[-1]:.8f} J")

output_dir = Path("onesimulation_results")
output_dir.mkdir(exist_ok=True)
np.savetxt(
    output_dir / "damped_oscillator.csv",
    np.column_stack((times, displacements, reference_displacement, velocities, reference_velocity, energies)),
    delimiter=",",
    header="time_s,displacement_m,reference_displacement_m,velocity_m_per_s,reference_velocity_m_per_s,mechanical_energy_J",
    comments="",
)
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
fig.patch.set_facecolor("#fbf5eb")
for ax in axes:
    ax.set_facecolor("#fffaf2")
    ax.grid(alpha=0.25)
axes[0].plot(times, reference_displacement * 1000, color="#6b442f", label="Analytical reference")
axes[0].plot(times, displacements * 1000, ".", color="#c56838", label="OneSimulation RK4")
axes[0].set(ylabel="Displacement / mm", title="Linear damped oscillator")
axes[0].legend()
axes[1].plot(times, energies, color="#c56838")
axes[1].set(xlabel="Time / s", ylabel="Mechanical energy / J")
fig.savefig(output_dir / "damped_oscillator.svg")
plt.show()
print("CSV and SVG saved in", output_dir)


# ## Explore further
#
# Halve the time step and compare the errors. Vary damping within the underdamped range and examine the envelope and energy loss. Critical and overdamped motion require different reference formulas. New parameter values may need shorter time steps and revised tolerances. Rerunning the notebook overwrites both output files.
