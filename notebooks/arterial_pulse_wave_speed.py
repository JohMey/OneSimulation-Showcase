"""OneSimulation example: the speed of a pressure pulse in an elastic artery.
Requires the installed, activated OneSimulation runtime.

The measured pulse speed is checked against the Moens-Korteweg wave speed derived
from the model's own elastic wall law, and the pulse's pressure-to-flow ratio
against the characteristic impedance rho*c0/A0.
"""

# # Arterial pulse wave: how fast a pressure pulse travels along an elastic vessel
#
# Blood vessels are compliant tubes: a local rise in pressure widens the vessel,
# and the interplay between the inertia of the blood and the elasticity of the
# wall lets a pressure pulse run along the artery much faster than the blood
# itself moves. For a thin-walled elastic tube of resting radius $r_0$, wall
# thickness $h$, Young's modulus $E$ and Poisson ratio $\nu$ the small-amplitude
# pulse-wave speed is the Moens-Korteweg speed
#
# $$c_0=\sqrt{\frac{E\,h}{2\,\rho\,(1-\nu^2)\,r_0}} .$$
#
# We launch a small, smooth area pulse in a straight vessel, let it split into a
# forward- and a backward-running pulse, and measure how fast they travel.
#
# OneSimulation integrates the one-dimensional cross-section-averaged blood-flow
# equations for the lumen area $A(x,t)$ and the volume flow $Q(x,t)$,
#
# $$\partial_t A + \partial_x Q = 0,\qquad
#   \partial_t Q + \partial_x\!\Big(\frac{Q^2}{A}\Big) + \frac{A}{\rho}\,\partial_x p = 0,$$
#
# closed by the elastic tube law of `ArterialSegment`,
#
# $$p(A) = p_\mathrm{ext} + \beta\Big(\sqrt{A/A_0}-1\Big),$$
#
# with a second-order finite-volume scheme (minmod reconstruction, Rusanov flux)
# and characteristic-based junction and terminal conditions.
#
# **Reference.** Linearising the system about the resting area $A_0$ gives the
# wave speed $c^2 = (A/\rho)\,\mathrm{d}p/\mathrm{d}A$. For the tube law above
# $\mathrm{d}p/\mathrm{d}A = \beta/(2\sqrt{A A_0})$, so at $A=A_0$
#
# $$c_0=\sqrt{\frac{\beta}{2\rho}} .$$
#
# Matching the tube law to a thin elastic wall gives
# $\beta=\sqrt{\pi}\,E h/\big((1-\nu^2)\sqrt{A_0}\big)$, and with
# $A_0=\pi r_0^2$ this $c_0$ is exactly the Moens-Korteweg speed. A linear
# forward-running pulse also fixes the ratio of its pressure and flow
# perturbations, the characteristic impedance $Z_0=\rho c_0/A_0$; we check both.

import numpy as np
from onesim.cad.d1 import Line
from onesim.meshing import StructuredLineMesher
from onesim.physics.biophysics import (
    ArterialNetwork,
    ArterialPressure,
    ArterialSegment,
    WindkesselTerminal,
)
from onesim.visualization import plots, theme

theme.set()


# ## Parameters
#
# The vessel is of aortic size with a wall of moderate stiffness, filled with
# blood. The initial disturbance is a narrow Gaussian area bump of one part in a
# thousand of the resting area, so the wave stays linear. The run ends before the
# two pulses reach the ends of the vessel, so the terminals do not influence the
# speed measurement.

rho = 1060.0
r0 = 0.010
h = 1.5e-3
E = 4.0e5
nu = 0.5
L = 1.0
x0 = 0.5 * L
sigma = 0.03
amplitude = 1.0e-3
cells = 400
final_time = 0.05

A0 = np.pi * r0**2
beta = np.sqrt(np.pi) * E * h / ((1.0 - nu**2) * np.sqrt(A0))
c_formula = np.sqrt(E * h / (2.0 * rho * (1.0 - nu**2) * r0))
c_tube_law = np.sqrt(beta / (2.0 * rho))
Z0 = rho * c_formula / A0
print(f"A0 = {A0*1e4:.3f} cm^2,  beta = {beta:.0f} Pa")
print(
    f"Moens-Korteweg speed c0 = {c_formula:.4f} m/s,  tube-law speed sqrt(beta/2rho) = {c_tube_law:.4f} m/s,"
    f"  c0*t = {c_formula*final_time:.3f} m"
)


def initial_area(x):
    return A0 * (1.0 + amplitude * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2)))


vessel = ArterialSegment(
    name="aorta",
    start="inlet",
    end="outlet",
    mesh=StructuredLineMesher(cells=cells).generate(
        Line(start=(0.0, 0.0, 0.0), end=(L, 0.0, 0.0))
    ),
    reference_area=A0,
    wall_stiffness=beta,
)
network = ArterialNetwork(
    segments=(vessel,),
    density=rho,
    terminals=(
        ArterialPressure(node="inlet", pressure=0.0),
        WindkesselTerminal(
            node="outlet",
            proximal_resistance=Z0,
            distal_resistance=5.0 * Z0,
            compliance=1.0e-9,
            reference_pressure=0.0,
            initial_pressure=0.0,
        ),
    ),
)
result = network.solve(
    times=np.linspace(0.0, final_time, 6),
    max_step=1.0e-3,
    initial_area={"aorta": initial_area},
    initial_flow={"aorta": lambda x: np.zeros_like(x)},
)
print(
    f"Integrated to t = {result.times[-1]:.4f} s with {len(result.stage_times)} explicit stages"
)


# ## Measure the pulse speed and the characteristic impedance
#
# The relative area perturbation $\delta A/A_0$ at the final time consists of
# two pulses that keep their shape. The centroid of $|\delta A|$ on each half of
# the vessel marks the pulse position; the travelled distance divided by the
# elapsed time is the measured speed. On the forward pulse the flow and pressure
# perturbations are tied by $\delta p = Z_0\,Q$ (and $\delta p=-Z_0 Q$ on the
# backward pulse), which gives an independent check of the wall law.

x = result.coordinates["aorta"]
area_final = result.area["aorta"][-1]
flow_final = result.flow["aorta"][-1]
pressure_final = result.pressure["aorta"][-1]
t_end = float(result.times[-1])

delta_a = area_final / A0 - 1.0
weight = np.abs(delta_a)
right = x > x0
left = x < x0
centroid_right = np.sum(x[right] * weight[right]) / np.sum(weight[right])
centroid_left = np.sum(x[left] * weight[left]) / np.sum(weight[left])
speed_right = (centroid_right - x0) / t_end
speed_left = (x0 - centroid_left) / t_end
speed_measured = 0.5 * (speed_right + speed_left)
rel_error = abs(speed_measured - c_formula) / c_formula

print(f"forward  pulse speed = {speed_right:.4f} m/s")
print(f"backward pulse speed = {speed_left:.4f} m/s")
print(f"measured pulse speed = {speed_measured:.4f} m/s")
print(f"relative error vs Moens-Korteweg = {100*rel_error:.2f} %")

peak_right = np.argmax(np.where(right, delta_a, -np.inf))
peak_left = np.argmax(np.where(left, delta_a, -np.inf))
Z_right = pressure_final[peak_right] / flow_final[peak_right]
Z_left = -pressure_final[peak_left] / flow_final[peak_left]
Z_measured = 0.5 * (Z_right + Z_left)
Z_error = abs(Z_measured - Z0) / Z0
print(
    f"peak pressure of the forward pulse = {pressure_final[peak_right]:.2f} Pa,"
    f"  peak flow = {flow_final[peak_right]*1e6:.3f} ml/s"
)
print(
    f"characteristic impedance: measured {Z_measured:.4e}, rho*c0/A0 = {Z0:.4e} Pa·s/m^3"
    f"  (relative error {100*Z_error:.2f} %)"
)


# ## Figure

predicted = [x0 - c_formula * t_end, x0 + c_formula * t_end]
fig = plots.figure(1, 2, size=(12, 4.8))
plots.line(
    x,
    {
        f"computed, t = {t_end * 1e3:.0f} ms": delta_a * 1e3,
        "initial area bump": (result.area["aorta"][0] / A0 - 1.0) * 1e3,
    },
    points={"predicted x0 ± c0 t": (predicted, np.interp(predicted, x, delta_a * 1e3))},
    vlines={"measured pulse centroids": centroid_left, "": centroid_right},
    xlabel="position along the vessel x [m]",
    ylabel="area perturbation dA/A0 [‰]",
    title=f"Pulse speed: measured {speed_measured:.3f} m/s, Moens-Korteweg {c_formula:.3f} m/s",
    ax=fig[0],
)
plots.line(
    x,
    {"pressure perturbation": pressure_final},
    reference={"Z0 Q with Z0 = rho c0 / A0": Z0 * flow_final},
    xlabel="position along the vessel x [m]",
    ylabel="pressure [Pa]",
    title="Forward pulse: p = +Z0 Q, backward pulse: p = -Z0 Q",
    ax=fig[1],
)
plots.show()


# ## Going further
#
# Stiffen the wall (`E`) or thin the lumen (`r0`) and watch the pulse run faster
# as $\sqrt{E h/r_0}$: arterial stiffening with age shows up directly as a
# higher pulse-wave velocity. Lengthen `final_time` so the forward pulse reaches
# the outlet: with the proximal resistance matched to $Z_0$ it is absorbed almost
# without reflection, while a sealed outlet (`ArterialInflow(flow=0)`) sends it
# back doubled in pressure. Raise `amplitude` towards unity and the crest, which
# travels faster than the trough, steepens into a shock-like front.
