"""OneSimulation example: water hammer after a fast valve closure.
Requires the installed, activated OneSimulation runtime.

A reservoir feeds a long pipe that ends in a valve. The valve closes in a fraction
of the pipe period, and the OneSimulation pressure-wave solution is checked against
the Joukowsky pressure rise, the wave period 4L/a and the exact characteristic
solution of the frictionless pipe.
"""

# # Water hammer after a fast valve closure
#
# A constant-head reservoir feeds a horizontal pipe of length $L$ and diameter $D$
# that ends in a valve discharging to the atmosphere. Water flows steadily at the
# mean velocity $v_0$. At $t=0$ the valve starts to close and is shut after the
# closure time $t_c$. The decelerated water column is slightly compressible and the
# pipe wall slightly elastic, so the stopped flow sends a pressure wave up the pipe
# at the speed $a$; it is reflected at the reservoir and returns to the valve.
#
# OneSimulation solves the one-dimensional water-hammer equations for the pressure
# $p(x,t)$ and the volume flow $Q(x,t)$,
#
# $$\frac{A}{\rho a^2}\,\frac{\partial p}{\partial t}+\frac{\partial Q}{\partial x}=0,\qquad
#   \frac{\rho}{A}\,\frac{\partial Q}{\partial t}+\frac{\partial p}{\partial x}
#   +\frac{f\rho}{2DA^2}\,Q\,|Q|=0,$$
#
# on a staggered pipe grid with an energy-conserving implicit-midpoint time step.
#
# | End | Condition |
# |---|---|
# | $x=0$, reservoir | $p=p_r$ (constant head) |
# | $x=L$, valve | $p=R\,Q\,|Q|/\tau(t)^2$, opening $\tau$ from 1 to 0 in $t_c$ |
#
# **Reference.** For a closure faster than the return time $2L/a$ the pressure at the
# valve rises by the Joukowsky value (N. Joukowsky, 1898; E. B. Wylie and V. L.
# Streeter, *Fluid Transients in Systems*, Prentice Hall 1993, ch. 1-3)
#
# $$\Delta p_J=\rho\,a\,v_0,$$
#
# stays there for $2L/a$, then drops to $p_r-\Delta p_J$ for another $2L/a$: without
# friction the valve pressure is a square wave of period $T=4L/a$. The exact solution
# of the frictionless pipe follows from the characteristics: with the valve wave
# $F(t)=Z\,(Q_0-Q_v(t))$, $Z=\rho a/A$, and the reflection $-1$ at the reservoir,
#
# $$p_v(t)-p_r=F(t)+2\sum_{k\ge1}(-1)^k\,F\!\left(t-\frac{2kL}{a}\right),$$
#
# where $Q_v$ is the valve flow during the closure, from the valve law and the
# characteristic $p_v=p_r+Z\,(Q_0-Q_v)$.

import numpy as np
from onesim.physics.systems import (
    HydraulicPressurePort,
    HydraulicValvePort,
    WaterHammerNetwork,
    WaterHammerPipe,
)
from onesim.results.sampling import crossing_times
from onesim.visualization import plots, theme

theme.set()


# ## Pipe and valve
#
# A steel water main of 600 m and 0.5 m diameter with a wave speed of 1200 m/s, fed by
# a reservoir 150 m above the valve. The valve loss coefficient is chosen so that the
# initial flow is 1 m/s; the valve closes with a smooth cosine schedule in 0.2 s, a
# fifth of the return time $2L/a=1$ s.

density, length, diameter, wave_speed, velocity = 1000.0, 600.0, 0.5, 1200.0, 1.0
reservoir_pressure, closure_time = density * 9.81 * 150.0, 0.2
area = np.pi * diameter**2 / 4
flow = velocity * area
resistance = reservoir_pressure / flow**2
impedance = density * wave_speed / area
closing = lambda t: 0.5 + 0.5 * np.cos(np.pi * np.clip(t / closure_time, 0.0, 1.0))
joukowsky, period = density * wave_speed * velocity, 4.0 * length / wave_speed
print(
    f"initial flow {flow * 1e3:.1f} L/s, reservoir pressure {reservoir_pressure / 1e5:.2f} bar"
)
print(
    f"Joukowsky rise ρ a v0 = {joukowsky / 1e5:.2f} bar, period 4L/a = {period:.2f} s, closure in {closure_time / (2 * length / wave_speed):.0%} of 2L/a"
)

# ## Simulation on three grids
#
# Eight seconds (four periods) on 100, 200 and 400 cells with a time step of half the
# cell transit time. The pipe starts in the steady state: uniform reservoir pressure
# and the initial flow everywhere.

times = np.linspace(0.0, 8.0, 1601)
runs = {
    cells: WaterHammerNetwork(
        pipes=(
            WaterHammerPipe(
                name="main",
                start="reservoir",
                end="valve",
                length=length,
                diameter=diameter,
                wave_speed=wave_speed,
                cells=cells,
            ),
        ),
        ports=(
            HydraulicPressurePort(node="reservoir", pressure=reservoir_pressure),
            HydraulicValvePort(
                node="valve",
                downstream_pressure=0.0,
                resistance=resistance,
                opening=closing,
            ),
        ),
        density=density,
    ).solve(
        times=times,
        max_step=0.5 * length / wave_speed / cells,
        initial_pressure=reservoir_pressure,
        initial_flow=flow,
    )
    for cells in (100, 200, 400)
}
result = runs[400]
surge = {
    cells: run.pressure["main"][:, -1] - reservoir_pressure
    for cells, run in runs.items()
}
distance = {
    cells: length - run.pressure_coordinates["main"][-1] for cells, run in runs.items()
}

# ## Characteristic solution
#
# The pressure is compared in the cell next to the valve, half a cell upstream of it.
# At the distance $d$ from the valve the outgoing wave arrives $d/a$ later and the wave
# reflected at the reservoir $d/a$ earlier than at the valve itself. The valve wave
# $F(t)$ solves the quadratic valve law along the characteristic; before $t=0$ it is
# zero by construction.


def characteristic_surge(t, d):
    total = reservoir_pressure + impedance * flow
    wave = lambda s: impedance * (
        flow
        - 2
        * total
        * closing(s)
        / (
            impedance * closing(s)
            + np.sqrt((impedance * closing(s)) ** 2 + 4 * total * resistance)
        )
    )
    return sum(
        (-1) ** k
        * (
            wave(t - (2 * k * length + d) / wave_speed)
            - wave(t - (2 * (k + 1) * length - d) / wave_speed)
        )
        for k in range(9)
    )


exact = {cells: characteristic_surge(times, d) for cells, d in distance.items()}

# ## Comparison
#
# The plateau pressure is the median of the surge in the first half period after the
# closure; the period comes from the interpolated downward zero crossings. The
# overshoot at the steep wave fronts and the rms deviation from the exact solution
# shrink as the grid is refined.

plateau = (times > 2 * closure_time) & (times < 2 * length / wave_speed)
crossings = {
    cells: crossing_times(times, p, 0.0, direction="falling")
    for cells, p in surge.items()
}
print(
    " cells   plateau rise [bar]   Joukowsky [bar]   peak [bar]   period [s]   rms deviation from exact"
)
for cells, p in surge.items():
    print(
        f"{cells:6d} {np.median(p[plateau]) / 1e5:17.4f} {joukowsky / 1e5:17.4f} {p.max() / 1e5:12.3f} "
        f"{np.mean(np.diff(crossings[cells])):12.5f} {np.sqrt(np.mean((p - exact[cells]) ** 2)) / joukowsky:15.2%}"
    )
print(
    f"period 4L/a = {period:.5f} s;  valve flow after closure: {np.abs(result.node_inflow['valve'][times > closure_time]).max():.1e} m³/s"
)
energy = result.stored_energy + result.dissipated_energy - result.boundary_work
print(
    f"energy balance of the pipe (400 cells): drift {np.ptp(energy) / energy[0]:.1e} of the initial energy"
)

# ## Figure
#
# The pressure next to the valve on the three grids against the characteristic solution, and the
# pressure in the whole pipe over the first two periods: the fronts travel at $a$ and
# reverse sign at the reservoir. Dotted lines: the Joukowsky values $\pm\rho a v_0$.

window = times <= 2.0 * period
fig = plots.figure(1, 2, size=(15, 5))
plots.line(
    times,
    {f"{cells} cells": p / 1e5 for cells, p in surge.items()},
    reference={"characteristic solution": exact[400] / 1e5},
    hlines=[joukowsky / 1e5, -joukowsky / 1e5],
    xlimits=(0.0, 3.5),
    xlabel="time t [s]",
    ylabel="pressure surge p - p_r [bar]",
    title="Pressure next to the valve",
    legend="lower left",
    ax=fig[0],
)
plots.heatmap(
    result.pressure_coordinates["main"],
    times[window],
    (result.pressure["main"][window] - reservoir_pressure).T / 1e5,
    colormap="diverging",
    centre=0.0,
    limits=(-1.1 * joukowsky / 1e5, 1.1 * joukowsky / 1e5),
    label="p - p_r [bar]",
    xlabel="position along the pipe x [m]",
    ylabel="time t [s]",
    aspect="auto",
    title="Pressure waves in the pipe",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Close the valve more slowly than $2L/a$ and the first relief wave from the reservoir
# arrives before the closure is complete: the peak falls below $\rho a v_0$ (Michaud's
# estimate $\Delta p\approx 2\rho L v_0/t_c$). Give the pipe a `darcy_friction` and
# the square wave decays, with the line-packing rise above the Joukowsky value. A
# `HydraulicPumpPort` instead of the reservoir, or a branch to a second valve, turns
# the case into the classical pump-trip and network studies.
