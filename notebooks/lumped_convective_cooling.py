"""OneSimulation example: a lumped thermal mass cooling by convection.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against Newton's law of cooling.
"""


# # Convective cooling of a lumped thermal mass
#
# A compact body with total heat capacity $C$ sits in a still environment at
# temperature $T_\infty$ and loses heat through its surface by convection. When the
# body is small enough that its interior stays nearly uniform (the lumped-capacitance
# limit), a single temperature $T(t)$ describes it and Newton's law of cooling holds:
#
# $$C\,\frac{dT}{dt}=-h\,A\,(T-T_\infty).$$
#
# Here $h$ is the convective heat-transfer coefficient and $A$ the exposed area; the
# product $hA$ is the thermal conductance to the surroundings. OneSimulation treats
# this as a zero-dimensional (lumped) model and integrates it in time with the
# classical fourth-order Runge-Kutta method.
#
# **Reference.** Separating variables gives the exponential decay towards the
# ambient temperature,
#
# $$T(t)=T_\infty+(T_0-T_\infty)\,e^{-hA\,t/C},$$
#
# with time constant $\tau=C/(hA)$. We compare every stored time against this
# expression.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.physics.systems import ThermalCapacitance
from onesim.results.storage import MemoryOutputSink, Output
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model
#
# The parameters can be changed. The reference formula below stays valid as long as
# the heat capacity, area, heat-transfer coefficient and ambient temperature are
# constant. The values describe a small metal part cooling in quiet air.

heat_capacity = 900.0           # J/K, total heat capacity C
heat_transfer_coefficient = 15.0  # W/(m²·K), convective coefficient h
area = 0.06                     # m², exposed surface area A
ambient_temperature = 300.0     # K, surrounding temperature
initial_temperature = 350.0     # K, body temperature at t = 0

conductance = heat_transfer_coefficient * area  # W/K, equals h*A
time_constant = heat_capacity / conductance
end_time = 3.0 * time_constant
time_step = 0.05 * time_constant
print(f"conductance hA = {conductance:.3f} W/K,  time constant C/(hA) = {time_constant:.1f} s")

domain = LumpedDomain()
model = ThermalCapacitance(
    domain=domain.system,
    heat_capacity=heat_capacity,
    conductance=conductance,
    ambient_temperature=ambient_temperature,
    heating=0.0,
)
simulation = Simulation(
    problem=Problem(
        name="Convective cooling of a lumped thermal mass",
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
    output=Output(fields=(model.temperature,), interval=time_step, sink=sink),
)
times = np.asarray([float(s.time) for s in sink.snapshots])
temperatures = np.asarray([
    float(np.asarray(s.fields[model.temperature.field_id].values).reshape(-1)[0])
    for s in sink.snapshots
])


# ## Comparison with the closed-form solution
#
# We evaluate the exponential reference at every stored time. A passing check shows
# that this numerical example agrees with Newton's law of cooling; it is not an
# experimental validation of a real object.

reference = ambient_temperature + (initial_temperature - ambient_temperature) * np.exp(
    -conductance * times / heat_capacity
)
max_abs_error = float(np.max(np.abs(temperatures - reference)))
print("Error per stored time (K):")
for t, err in zip(times, np.abs(temperatures - reference)):
    print(f"  t={t:7.1f} s  |dT|={err:.3e} K")
print(f"Maximum absolute error against the exponential: {max_abs_error:.3e} K")


# ## Figure

fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
fig.patch.set_facecolor("#fbf5eb")
ax.set_facecolor("#fffaf2")
dense_t = np.linspace(0.0, end_time, 400)
dense_ref = ambient_temperature + (initial_temperature - ambient_temperature) * np.exp(
    -conductance * dense_t / heat_capacity
)
ax.plot(dense_t, dense_ref, color="#6b442f", lw=1.8, label="Newton's law of cooling")
ax.plot(times, temperatures, "o", color="#c56838", ms=4, label="OneSimulation RK4")
ax.axhline(ambient_temperature, color="#9a8f7d", ls="--", lw=1.0, label="Ambient")
ax.set(xlabel="Time / s", ylabel="Temperature / K",
       title="Convective cooling of a lumped thermal mass")
ax.grid(alpha=0.25)
ax.legend()
plt.show()


# ## Going further
#
# Halve the time step and confirm that the error against the exponential shrinks
# as expected for a fourth-order method. Then vary the conductance $hA$ or the heat
# capacity $C$ and relate the change in the time constant $\tau=C/(hA)$ to the
# equation. Adding a constant internal power through the `heating` input turns the
# body towards a shifted steady state $T_\infty+P/(hA)$ instead of the ambient
# temperature; replace the reference accordingly before studying that case.
