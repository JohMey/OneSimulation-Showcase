"""Generated from the OneSimulation example notebook.
Needs the installed, activated OneSimulation runtime. See
https://johmey.github.io/OneSimulation-Showcase/examples/notebooks.html
"""


# # Radiative cooling with OneSimulation
#
# A body at a uniform temperature cools by radiation. Use an activated installation of Runtime Alpha 0.2.1a4 and the Python environment described in the release instructions. Run the cells in order. Keep your licence key outside the notebook.
#
# $$C\,\frac{dT}{dt}=-\varepsilon\,\sigma\,A\,(T^4-T_\mathrm{U}^4).$$
#
# Here $C$ is the total heat capacity in J/K, $A$ is the radiating area in m², $\varepsilon$ is emissivity and $\sigma$ is the Stefan–Boltzmann constant. Setting the surroundings to $T_\mathrm{U}=0$ K gives an idealized mathematical reference. The model neglects internal temperature gradients, convection and other heat sources.

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

from onesim.physics.heat_transfer import RadiationCooling, ThermalGreyDiffuseRadiationBoundary


# ## Parameters and model
#
# You can vary the parameters. The reference formula assumes constant heat capacity, area and emissivity, with $T_\mathrm{U}=0$ K. The error tolerances below apply to the supplied values.

heat_capacity = 100.0          # J/K; total heat capacity
area = 0.1                     # m²
emissivity = 0.8               # dimensionless, from 0 to 1
initial_temperature = 600.0    # K
ambient_temperature = 0.0      # K; idealized analytical case
end_time = 20.0                # s
time_step = 0.5                # s

assert heat_capacity > 0 and area > 0 and initial_temperature > 0
assert 0 < emissivity <= 1 and ambient_temperature == 0

domain = LumpedDomain()
radiation = ThermalGreyDiffuseRadiationBoundary(
    condition_id="notebook-radiation",
    emissivity=emissivity,
    ambient_temperature=ambient_temperature,
)
model = RadiationCooling(
    domain=domain.system,
    heat_capacity=heat_capacity,
    radiating_area=area,
    radiation=radiation,
)
simulation = Simulation(
    problem=Problem(
        name="Radiative cooling reference case",
        geometry=domain,
        physics=(model,),
        initial_conditions=(InitialValue(model.temperature, initial_temperature),),
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
        fields=(model.temperature, model.radiated_power),
        interval=1.0,
        sink=sink,
    ),
)
times = np.asarray([float(s.time) for s in sink.snapshots])
temperatures = np.asarray([
    float(np.asarray(s.fields[model.temperature.field_id].values).reshape(-1)[0])
    for s in sink.snapshots
])
powers = np.asarray([
    float(np.asarray(s.fields[model.radiated_power.field_id].values).reshape(-1)[0])
    for s in sink.snapshots
])
assert len(times) > 1 and np.isclose(times[-1], end_time)


# ## Comparison with the closed-form solution
#
# For $T_\mathrm{U}=0$ K, separation of variables gives
#
# $$T(t)=\left(T_0^{-3}+\frac{3\varepsilon\sigma A}{C}\,t\right)^{-1/3}.$$
#
# Compare the numerical temperature at every saved time with the formula. This checks the numerical solution against the equation. Experimental validation would require measurements.

sigma = 5.670374419e-8  # W/(m² K⁴)
reference_temperature = (
    initial_temperature**(-3) + 3 * emissivity * sigma * area * times / heat_capacity
) ** (-1 / 3)
max_temperature_error = float(np.max(np.abs(temperatures - reference_temperature)))
assert max_temperature_error < 1e-6, "The reference error exceeds the example tolerance."
assert np.all(np.diff(temperatures) < 0), "The body should cool in this example."
assert np.all(powers >= 0)
print(f"Final temperature: {temperatures[-1]:.8f} K")
print(f"Largest absolute reference error: {max_temperature_error:.3e} K")

output_dir = Path("onesimulation_results")
output_dir.mkdir(exist_ok=True)
np.savetxt(
    output_dir / "radiative_cooling.csv",
    np.column_stack((times, temperatures, reference_temperature, powers)),
    delimiter=",",
    header="time_s,temperature_K,reference_temperature_K,radiated_power_W",
    comments="",
)
fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
fig.patch.set_facecolor("#fbf5eb")
ax.set_facecolor("#fffaf2")
ax.plot(times, reference_temperature, color="#6b442f", label="Analytical reference")
ax.plot(times, temperatures, "o", color="#c56838", markersize=4, label="OneSimulation RK4")
ax.set(xlabel="Time / s", ylabel="Temperature / K", title="Idealized radiative cooling")
ax.grid(alpha=0.25)
ax.legend()
fig.savefig(output_dir / "radiative_cooling.svg")
plt.show()
print("CSV and SVG saved in", output_dir)


# ## Explore further
#
# Halve the time step and compare the error. Change area, heat capacity and emissivity one at a time, and use the equation to explain the effects. For other environmental conditions, select a reference solution with those same conditions. Rerunning the notebook overwrites its two output files.
