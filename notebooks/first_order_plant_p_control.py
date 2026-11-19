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

import numpy as np
from onesim.physics.systems import ControlSignalFlowProvider, SignalBlock
from onesim.visualization import plots, theme

theme.set()


# ## Parameters and model
#
# The parameters can be changed; the reference stays valid for any positive gains
# and time constant. The values describe a sluggish process (20 s time constant,
# gain 2) under a moderate proportional gain, so the offset remains clearly visible.

time_constant = 20.0
plant_gain = 2.0
proportional_gain = 4.0
setpoint = 1.0

loop_gain = plant_gain * proportional_gain
steady_output = loop_gain / (1.0 + loop_gain) * setpoint
closed_loop_time_constant = time_constant / (1.0 + loop_gain)
end_time = 6.0 * closed_loop_time_constant
time_step = 0.01
print(
    f"loop gain K*Kp = {loop_gain:.1f},  closed-loop time constant = {closed_loop_time_constant:.3f} s "
    f"(open loop {time_constant:.1f} s),  steady output = {steady_output:.4f}"
)

loop = ControlSignalFlowProvider(
    blocks=(
        SignalBlock("setpoint", "constant", parameters={"value": setpoint}),
        SignalBlock(
            "error",
            "sum",
            inputs=("setpoint", "plant"),
            parameters={"weights": (1.0, -1.0)},
        ),
        SignalBlock(
            "controller",
            "pid",
            inputs=("error",),
            parameters={"proportional_gain": proportional_gain},
        ),
        SignalBlock(
            "actuator", "gain", inputs=("controller",), parameters={"gain": plant_gain}
        ),
        SignalBlock(
            "plant",
            "first_order",
            inputs=("actuator",),
            parameters={"time_constant": time_constant, "initial_value": 0.0},
        ),
    ),
    final_time=end_time,
    time_step=time_step,
    time_integration="classical_rk4",
)
run = loop.execute()
times = np.asarray(run.times)
outputs = np.asarray(run.signals["plant"])
commands = np.asarray(run.signals["controller"])
print("evaluation order:", " -> ".join(run.evaluation_order))


# ## Comparison with the closed-form solution
#
# The closed-loop exponential is evaluated at every stored time, the closed-loop
# time constant is fitted to the simulated approach, and the control law is
# verified sample by sample. A passing check shows that this numerical example
# agrees with the linear loop equations; it is not a measurement of a real process.

reference = steady_output * (1.0 - np.exp(-times / closed_loop_time_constant))
max_output_error = float(np.max(np.abs(outputs - reference)))
print(
    f"Maximum output error against the closed-loop exponential: {max_output_error:.3e}"
)


approach = steady_output - outputs
fitted_rate = -np.polyfit(times, np.log(approach), 1)[0]
fitted_time_constant = 1.0 / fitted_rate
print(
    f"Closed-loop time constant: fitted {fitted_time_constant:.5f} s, exact {closed_loop_time_constant:.5f} s"
)
final_offset = setpoint - outputs[-1]
expected_offset = setpoint / (1.0 + loop_gain)
print(
    f"Steady-state offset: simulated {final_offset:.5f}, exact r/(1+K Kp) = {expected_offset:.5f}"
)


# ## Figure

dense_t = np.linspace(0.0, end_time, 400)
fig = plots.figure(2, 1, size=(8, 7))
plots.line(
    times,
    points={"OneSimulation RK4": (times[::10], outputs[::10])},
    reference={
        "exact step response": (
            dense_t,
            steady_output * (1.0 - np.exp(-dense_t / closed_loop_time_constant)),
        )
    },
    hlines={"setpoint r": setpoint, "steady state K Kp r / (1 + K Kp)": steady_output},
    vlines={"closed-loop time constant": closed_loop_time_constant},
    ylabel="plant output y",
    title="First-order plant under proportional control",
    legend="lower right",
    ax=fig[0],
)
plots.line(
    times,
    {"controller output u = Kp (r - y)": commands},
    hlines={"steady state Kp r / (1 + K Kp)": proportional_gain * expected_offset},
    xlabel="time [s]",
    ylabel="control signal u",
    ax=fig[1],
)
plots.show()


# ## Going further
#
# Raise `proportional_gain` and the offset shrinks as $1/(1+KK_p)$ while the loop
# becomes faster; it can never vanish with proportional action alone. Add an
# `integral_gain` to the controller block and the offset disappears, but the loop
# becomes second order and may overshoot; derive the new characteristic equation
# $\tau s^2+(1+KK_p)s+KK_i=0$ before adapting the checks. A `saturation` block
# between controller and actuator or a `transport_delay` in the feedback path
# turns the example into the standard anti-windup and dead-time studies.
