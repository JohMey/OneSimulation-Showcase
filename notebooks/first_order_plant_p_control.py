"""OneSimulation example: proportional control of a first-order plant.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form closed-loop response.
"""


# # Proportional control of a first-order plant
#
# A first-order plant with the time constant $\tau$ and the static gain $K$,
#
# $$\tau\,\frac{dy}{dt}=-y+K\,u,$$
#
# is driven by a proportional controller $u=K_p\,(r-y)$ that compares the output
# $y$ with a constant setpoint $r$. OneSimulation builds the loop from signal
# blocks (setpoint, summing junction, controller, plant) and integrates the single
# continuous state with the classical fourth-order Runge-Kutta method.
#
# **Reference.** Substituting the control law gives another first-order system,
# so the closed-loop step response from rest is
#
# $$y(t)=y_\infty\left(1-e^{-t/\tau_\mathrm{cl}}\right),\qquad
#   y_\infty=\frac{K K_p}{1+K K_p}\,r,\qquad
#   \tau_\mathrm{cl}=\frac{\tau}{1+K K_p}.$$
#
# A proportional controller speeds the plant up by the factor $1+KK_p$ but leaves
# the steady-state offset $r-y_\infty=r/(1+KK_p)$. Both facts are checked below.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.physics.systems import ControlSignalFlowProvider, SignalBlock

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model
#
# The parameters can be changed; the reference stays valid for any positive gains
# and time constant. The values describe a sluggish process (20 s time constant,
# gain 2) under a moderate proportional gain, so the offset remains clearly visible.

time_constant = 20.0        # s, plant time constant tau
plant_gain = 2.0            # static plant gain K
proportional_gain = 4.0     # controller gain K_p
setpoint = 1.0              # r, unit step

loop_gain = plant_gain * proportional_gain
steady_output = loop_gain / (1.0 + loop_gain) * setpoint
closed_loop_time_constant = time_constant / (1.0 + loop_gain)
end_time = 6.0 * closed_loop_time_constant
time_step = 0.01            # s
print(f"loop gain K*Kp = {loop_gain:.1f},  closed-loop time constant = {closed_loop_time_constant:.3f} s "
      f"(open loop {time_constant:.1f} s),  steady output = {steady_output:.4f}")

loop = ControlSignalFlowProvider(
    blocks=(
        SignalBlock("setpoint", "constant", parameters={"value": setpoint}),
        SignalBlock("error", "sum", inputs=("setpoint", "plant"), parameters={"weights": (1.0, -1.0)}),
        SignalBlock("controller", "pid", inputs=("error",), parameters={"proportional_gain": proportional_gain}),
        SignalBlock("actuator", "gain", inputs=("controller",), parameters={"gain": plant_gain}),
        SignalBlock("plant", "first_order", inputs=("actuator",),
                    parameters={"time_constant": time_constant, "initial_value": 0.0}),
    ),
    final_time=end_time,
    time_step=time_step,
    time_integration="classical_rk4",
)
run = loop.execute()
times = np.asarray(run.times)
outputs = np.asarray(run.signals["plant"])
commands = np.asarray(run.signals["controller"])
errors = np.asarray(run.signals["error"])
assert len(times) > 1 and np.isclose(times[-1], end_time)
print("evaluation order:", " -> ".join(run.evaluation_order))


# ## Comparison with the closed-form solution
#
# The closed-loop exponential is evaluated at every stored time, the closed-loop
# time constant is fitted to the simulated approach, and the control law is
# verified sample by sample. A passing check shows that this numerical example
# agrees with the linear loop equations; it is not a measurement of a real process.

reference = steady_output * (1.0 - np.exp(-times / closed_loop_time_constant))
max_output_error = float(np.max(np.abs(outputs - reference)))
print(f"Maximum output error against the closed-loop exponential: {max_output_error:.3e}")
assert max_output_error < 1e-8 * setpoint, "Output deviates from the closed-loop solution."

# The controller applies exactly u = Kp (r - y) at every sample.
assert np.allclose(errors, setpoint - outputs, rtol=0.0, atol=1e-13)
assert np.allclose(commands, proportional_gain * (setpoint - outputs), rtol=0.0, atol=1e-12)

# Recover the closed-loop time constant from the log-slope of the remaining approach.
approach = steady_output - outputs
fitted_rate = -np.polyfit(times, np.log(approach), 1)[0]
fitted_time_constant = 1.0 / fitted_rate
print(f"Closed-loop time constant: fitted {fitted_time_constant:.5f} s, exact {closed_loop_time_constant:.5f} s")
assert abs(fitted_time_constant / closed_loop_time_constant - 1) < 1e-4
# The controller speeds the plant up by (1 + K Kp) but leaves the proportional offset.
assert fitted_time_constant < time_constant / loop_gain
final_offset = setpoint - outputs[-1]
expected_offset = setpoint / (1.0 + loop_gain)
print(f"Steady-state offset: simulated {final_offset:.5f}, exact r/(1+K Kp) = {expected_offset:.5f}")
assert abs(final_offset - expected_offset) < 1.1 * steady_output * np.exp(-end_time / closed_loop_time_constant)
assert final_offset > 0.0
# Physical sanity: a first-order loop rises monotonically and never overshoots.
assert np.all(np.diff(outputs) > 0.0)
assert np.max(outputs) < steady_output
assert abs(outputs[0]) < 1e-15


# ## Figure

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
fig.patch.set_facecolor("#fbf5eb")
for ax in axes:
    ax.set_facecolor("#fffaf2")
    ax.grid(alpha=0.25)
dense_t = np.linspace(0.0, end_time, 400)
dense_y = steady_output * (1.0 - np.exp(-dense_t / closed_loop_time_constant))
stride = max(1, len(times) // 40)
axes[0].plot(dense_t, dense_y, color="#6b442f", lw=1.6, label=r"$y_\infty\,(1-e^{-t/\tau_{cl}})$")
axes[0].plot(times[::stride], outputs[::stride], "o", color="#c56838", ms=4, label="OneSimulation RK4")
axes[0].axhline(setpoint, color="#3f6b8a", ls="--", lw=1.0, label="Setpoint $r$")
axes[0].axhline(steady_output, color="#9a8f7d", ls="--", lw=1.0, label=r"$y_\infty=KK_p r/(1+KK_p)$")
axes[0].axvline(closed_loop_time_constant, color="#9a8f7d", ls=":", lw=1.0, label=r"$t=\tau_{cl}$")
axes[0].set(ylabel="Plant output $y$", title="First-order plant under proportional control")
axes[0].legend(loc="lower right", fontsize=9)
axes[1].plot(times, commands, color="#c56838", lw=1.4, label=r"Controller output $u=K_p(r-y)$")
axes[1].axhline(proportional_gain * expected_offset, color="#9a8f7d", ls="--", lw=1.0,
                label=r"$u_\infty=K_p r/(1+KK_p)$")
axes[1].set(xlabel="Time / s", ylabel="Control signal $u$")
axes[1].legend(loc="upper right")
plt.show()


# ## Going further
#
# Raise `proportional_gain` and the offset shrinks as $1/(1+KK_p)$ while the loop
# becomes faster; it can never vanish with proportional action alone. Add an
# `integral_gain` to the controller block and the offset disappears, but the loop
# becomes second order and may overshoot; derive the new characteristic equation
# $\tau s^2+(1+KK_p)s+KK_i=0$ before adapting the checks. A `saturation` block
# between controller and actuator or a `transport_delay` in the feedback path
# turns the example into the standard anti-windup and dead-time studies.
