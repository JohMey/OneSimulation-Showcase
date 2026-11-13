"""OneSimulation example: a liquid tank draining through an orifice (Torricelli's law).
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form square-root law.
"""


# # Tank draining through an orifice
#
# An open tank of cross-section $A$ is filled to the level $h_0$ and drains through
# a small orifice of area $a$ in its bottom. With the discharge coefficient $C_d$
# the outflow follows Torricelli's law, and the volume balance of the tank reads
#
# $$A\,\frac{dh}{dt}=-C_d\,a\,\sqrt{2\,g\,h}.$$
#
# OneSimulation models the tank as a compliant pressure node of a lumped hydraulic
# network: the gauge pressure at the bottom is $p=\rho g h$, so the node compliance
# is $C=A/(\rho g)$ and the orifice is a branch to the atmosphere with the
# nonlinear flow law $q=C_d\,a\sqrt{2\,(p-p_\mathrm{atm})/\rho}$. The network is
# integrated with the classical fourth-order Runge-Kutta method.
#
# **Reference.** The square root of the level falls linearly in time,
#
# $$\sqrt{h(t)}=\sqrt{h_0}-\frac{C_d\,a}{2A}\sqrt{2g}\;t,\qquad
#   t_e=\frac{A}{C_d\,a}\sqrt{\frac{2h_0}{g}},$$
#
# where $t_e$ is the time at which the tank is empty. We compare $h(t)$ at every
# stored time, recover $t_e$ from the simulated curve, and check that the drained
# volume equals the integrated orifice flow.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.physics.systems import HydraulicBranch, HydraulicNode, HydraulicSystemsProvider

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model
#
# The parameters can be changed; the reference stays valid for a prismatic tank
# (constant $A$), a constant discharge coefficient and a liquid that is much
# deeper than the orifice. The values describe a 0.5 m² water tank, 1.2 m deep,
# emptying through a sharp-edged 20 mm hole.

gravity = 9.81                       # m/s^2
density = 1000.0                     # kg/m^3, water
tank_area = 0.5                      # m^2, A
orifice_diameter = 0.02              # m
orifice_area = np.pi * (orifice_diameter / 2.0) ** 2   # m^2, a
discharge_coefficient = 0.62         # C_d, sharp-edged orifice
initial_level = 1.2                  # m, h0

drain_rate = discharge_coefficient * orifice_area * np.sqrt(2.0 * gravity) / (2.0 * tank_area)  # m^0.5/s
empty_time = np.sqrt(initial_level) / drain_rate
end_time = 0.9 * empty_time          # stop shortly before the orifice runs dry (h = 1 % of h0)
time_step = 1.0                      # s
print(f"orifice area = {orifice_area * 1e4:.3f} cm^2,  initial outflow = "
      f"{discharge_coefficient * orifice_area * np.sqrt(2 * gravity * initial_level) * 1e3:.3f} L/s")
print(f"emptying time t_e = {empty_time:.1f} s ({empty_time / 60:.1f} min)")

compliance = tank_area / (density * gravity)   # m^3/Pa, C = A/(rho g)


def orifice_flow(upstream_pressure, downstream_pressure, time, parameters):
    """Torricelli outflow q = C_d a sqrt(2 (p_up - p_down) / rho); no reverse flow."""
    head = max(upstream_pressure - downstream_pressure, 0.0)
    return parameters["discharge_coefficient"] * parameters["area"] * np.sqrt(2.0 * head / parameters["density"])


network = HydraulicSystemsProvider(
    nodes=(
        HydraulicNode("tank", compliance=compliance, initial_pressure=density * gravity * initial_level),
        HydraulicNode("atmosphere", imposed_pressure=0.0),   # gauge pressure reference
    ),
    branches=(
        HydraulicBranch(
            "orifice",
            from_node="tank",
            to_node="atmosphere",
            parameters={
                "discharge_coefficient": discharge_coefficient,
                "area": orifice_area,
                "density": density,
            },
            flow_law=orifice_flow,
        ),
    ),
    final_time=end_time,
    time_step=time_step,
    time_integration="classical_rk4",
)
run = network.execute()
times = np.asarray(run.times)
levels = np.asarray(run.pressures["tank"]) / (density * gravity)
outflows = np.asarray(run.flows["orifice"])


# ## Comparison with the closed-form solution
#
# We evaluate the square-root law at every stored time, extrapolate the emptying
# time from the simulated $\sqrt{h}$ trace, and check the volume balance. A passing
# check shows that this numerical example agrees with Torricelli's law; it is not
# a measurement of a real tank.

reference_levels = (np.sqrt(initial_level) - drain_rate * times) ** 2
max_level_error = float(np.max(np.abs(levels - reference_levels)))
print(f"Maximum level error: {max_level_error:.3e} m  (h0 = {initial_level} m, final level {levels[-1] * 1e2:.2f} cm)")

# sqrt(h) is a straight line; its intercept with zero is the emptying time.
slope, intercept = np.polyfit(times, np.sqrt(levels), 1)
fitted_empty_time = -intercept / slope
print(f"Emptying time: extrapolated {fitted_empty_time:.2f} s, exact {empty_time:.2f} s")
# ...and the integrated outflow equals the volume that left the tank.
drained_volume = tank_area * (initial_level - levels[-1])
integrated_outflow = float(np.trapezoid(outflows, times))
print(f"Drained volume: {drained_volume * 1e3:.2f} L, integrated orifice flow: {integrated_outflow * 1e3:.2f} L")


# ## Figure

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
fig.patch.set_facecolor("#fbf5eb")
for ax in axes:
    ax.set_facecolor("#fffaf2")
    ax.grid(alpha=0.25)
dense_t = np.linspace(0.0, empty_time, 600)
dense_h = (np.sqrt(initial_level) - drain_rate * dense_t) ** 2
stride = max(1, len(times) // 40)
axes[0].plot(dense_t / 60, dense_h, color="#6b442f", lw=1.6,
             label=r"$\left(\sqrt{h_0}-\frac{C_d a}{2A}\sqrt{2g}\,t\right)^2$")
axes[0].plot(times[::stride] / 60, levels[::stride], "o", color="#c56838", ms=4, label="OneSimulation RK4")
axes[0].axvline(empty_time / 60, color="#9a8f7d", ls=":", lw=1.0, label=r"Emptying time $t_e$")
axes[0].set(ylabel="Liquid level / m", title="Tank draining through an orifice")
axes[0].legend(loc="upper right")
axes[1].plot(dense_t / 60, np.sqrt(dense_h), color="#6b442f", lw=1.6, label="Closed-form (straight line)")
axes[1].plot(times[::stride] / 60, np.sqrt(levels[::stride]), "o", color="#c56838", ms=4, label="OneSimulation RK4")
axes[1].set(xlabel="Time / min", ylabel=r"$\sqrt{h}$ / m$^{1/2}$")
axes[1].legend(loc="upper right")
plt.show()


# ## Going further
#
# Double the orifice diameter and the tank empties four times faster; double the
# tank area and it takes twice as long. Add a constant `external_flow` to the tank
# node and the level settles where inflow and Torricelli outflow balance,
# $h_\infty=\left(Q_\mathrm{in}/(C_d a)\right)^2/(2g)$. A second tank node fed by
# the orifice branch turns the example into the classical two-tank system, whose
# levels no longer have an elementary closed form but can be checked against a
# fine-step reference integration.
