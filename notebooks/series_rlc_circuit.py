"""OneSimulation example: free discharge of a series RLC circuit.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the underdamped closed-form solution.
"""


# # Capacitor discharge through a series RLC circuit
#
# A capacitor $C$, initially charged to the voltage $U_0$, is connected at $t=0$ to
# a resistor $R$ and an inductor $L$ in series. There is no source in the loop.
# Kirchhoff's voltage law, with the capacitor charge $q$ and the loop current
# $i=\dot q$, gives the familiar second-order system
#
# $$\dot q = i,\qquad L\,\frac{di}{dt}+R\,i+\frac{q}{C}=0.$$
#
# OneSimulation treats the circuit as a zero-dimensional (lumped) model with two
# differential states, the charge and the current, and integrates them with the
# classical fourth-order Runge-Kutta method.
#
# **Reference.** With $\omega_0=1/\sqrt{LC}$, $\alpha=R/(2L)$ and, for the
# underdamped case $\alpha<\omega_0$, $\omega_d=\sqrt{\omega_0^2-\alpha^2}$, the
# charge and the current are
#
# $$q(t)=q_0\,e^{-\alpha t}\left(\cos\omega_d t+\frac{\alpha}{\omega_d}\sin\omega_d t\right),
# \qquad i(t)=-q_0\,\frac{\omega_0^2}{\omega_d}\,e^{-\alpha t}\sin\omega_d t,$$
#
# with $q_0=C\,U_0$. The current vanishes exactly at $t_n=n\pi/\omega_d$, where the
# charge takes the values $q_0 e^{-\alpha t_n}(-1)^n$. We use these facts to check
# not only the time series but also the damped angular frequency and the decay rate.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.physics.electromagnetics import SeriesRLC
from onesim.results.storage import MemoryOutputSink, Output
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model
#
# The parameters can be changed. The reference formula below stays valid as long
# as the circuit remains underdamped, $R<2\sqrt{L/C}$. The values describe a
# small laboratory circuit: a 100 µF capacitor charged to 10 V discharging
# through 2 Ω and 10 mH.

resistance = 2.0          # Ohm, R
inductance = 10.0e-3      # H, L
capacitance = 100.0e-6    # F, C
initial_voltage = 10.0    # V, capacitor voltage at t = 0
initial_current = 0.0     # A, loop current at t = 0

omega_0 = 1.0 / np.sqrt(inductance * capacitance)
alpha = resistance / (2 * inductance)
omega_d = np.sqrt(omega_0**2 - alpha**2)
initial_charge = capacitance * initial_voltage
damped_period = 2 * np.pi / omega_d
end_time = 5.0 / alpha                 # five decay times, about eight oscillations
time_step = damped_period / 200.0
print(f"omega_0 = {omega_0:.1f} rad/s,  alpha = {alpha:.1f} 1/s,  omega_d = {omega_d:.1f} rad/s")
print(f"damped period = {damped_period * 1e3:.3f} ms,  decay time 1/alpha = {1e3 / alpha:.1f} ms")

domain = LumpedDomain()
model = SeriesRLC(
    domain=domain.system,
    resistance=resistance,
    inductance=inductance,
    capacitance=capacitance,
    source_voltage=0.0,
)
simulation = Simulation(
    problem=Problem(
        name="Series RLC capacitor discharge",
        geometry=domain,
        physics=(model,),
        initial_conditions=(
            InitialValue(model.charge, initial_charge),
            InitialValue(model.current, initial_current),
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
        fields=(model.charge, model.current, model.capacitor_voltage),
        interval=time_step,
        sink=sink,
    ),
)
times = np.asarray([float(s.time) for s in sink.snapshots])


def series(field):
    return np.asarray([
        float(np.asarray(s.fields[field.field_id].values).reshape(-1)[0])
        for s in sink.snapshots
    ])


charges = series(model.charge)
currents = series(model.current)
capacitor_voltages = series(model.capacitor_voltage)


# ## Comparison with the closed-form solution
#
# We evaluate the reference charge and current at every stored time. In addition
# we read the damped frequency off the zero crossings of the current and the decay
# rate off the charge values at those crossings, exactly as one would from an
# oscilloscope trace. A passing check shows that this numerical example agrees
# with the linear circuit equations; it is not a measurement of a real circuit.

envelope = initial_charge * np.exp(-alpha * times)
reference_charge = envelope * (np.cos(omega_d * times) + (alpha / omega_d) * np.sin(omega_d * times))
reference_current = -initial_charge * (omega_0**2 / omega_d) * np.exp(-alpha * times) * np.sin(omega_d * times)

max_charge_error = float(np.max(np.abs(charges - reference_charge)))
max_current_error = float(np.max(np.abs(currents - reference_current)))
print(f"Maximum charge error:  {max_charge_error:.3e} C  (q0 = {initial_charge:.1e} C)")
print(f"Maximum current error: {max_current_error:.3e} A  (peak current ~ {initial_charge * omega_0:.2f} A)")

# Zero crossings of the current (linear interpolation between stored samples).
sign_change = np.where(np.sign(currents[:-1]) * np.sign(currents[1:]) < 0)[0]
crossing_times = times[sign_change] - currents[sign_change] * (
    (times[sign_change + 1] - times[sign_change]) / (currents[sign_change + 1] - currents[sign_change])
)
measured_omega_d = np.pi / np.mean(np.diff(crossing_times))
# Charge at the crossings: |q(t_n)| = q0 exp(-alpha t_n), so the log-slope is -alpha.
charge_at_crossings = np.interp(crossing_times, times, charges)
measured_alpha = -np.polyfit(crossing_times, np.log(np.abs(charge_at_crossings)), 1)[0]
print(f"Damped angular frequency: measured {measured_omega_d:.2f} rad/s, exact {omega_d:.2f} rad/s")
print(f"Decay rate:               measured {measured_alpha:.3f} 1/s,  exact {alpha:.3f} 1/s")
# Physical sanity: the stored energy can only be dissipated in the resistor.


# ## Figure

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
fig.patch.set_facecolor("#fbf5eb")
for ax in axes:
    ax.set_facecolor("#fffaf2")
    ax.grid(alpha=0.25)
dense_t = np.linspace(0.0, end_time, 2000)
dense_env = initial_voltage * np.exp(-alpha * dense_t)
dense_voltage = dense_env * (np.cos(omega_d * dense_t) + (alpha / omega_d) * np.sin(omega_d * dense_t))
axes[0].plot(dense_t * 1e3, dense_voltage, color="#6b442f", lw=1.6, label="Closed-form solution")
axes[0].plot(times[::10] * 1e3, capacitor_voltages[::10], "o", color="#c56838", ms=3.5, label="OneSimulation RK4")
axes[0].plot(dense_t * 1e3, dense_env, "--", color="#9a8f7d", lw=1.0, label=r"Envelope $U_0 e^{-\alpha t}$")
axes[0].plot(dense_t * 1e3, -dense_env, "--", color="#9a8f7d", lw=1.0)
axes[0].set(ylabel="Capacitor voltage / V", title="Series RLC capacitor discharge")
axes[0].legend(loc="upper right")
axes[1].plot(times * 1e3, currents, color="#c56838", lw=1.4, label="Loop current")
axes[1].plot(crossing_times * 1e3, np.zeros_like(crossing_times), "k|", ms=10, label="Zero crossings")
axes[1].set(xlabel="Time / ms", ylabel="Current / A")
axes[1].legend(loc="upper right")
plt.show()


# ## Going further
#
# Raise the resistance towards the critical value $2\sqrt{L/C}$ and watch the
# oscillation disappear; beyond it the circuit is overdamped and the reference
# formula must be replaced by the sum of two real exponentials. A constant
# `source_voltage` turns the discharge into a step response that settles at the
# charge $C\,U_\mathrm{s}$ with the same $\alpha$ and $\omega_d$. Halving the time step
# should shrink the time-series errors by about a factor of sixteen.
