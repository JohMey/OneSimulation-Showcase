"""OneSimulation example: overdischarge protection of a two-cell battery string with shunt resistors.
Requires the installed, activated OneSimulation runtime.

Two lumped lithium-ion cells in series, one with an internal short, are discharged by a current
source; event-switched shunts bypass a cell once its state of charge falls to 20 %. The circuit is
integrated with adaptive BDF2 and Dormand-Prince RK45, both with event location, and checked
against charge conservation, the closed-form switching time and bypass state, the exact series
solution of the particle diffusion and the published protection behaviour.
"""

# # Battery overdischarge protection using shunt resistances
#
# A string of two identical 12 Ah lithium-ion cells is discharged by a current source
# $I_\mathrm{app}=C_\mathrm{rate}\times 12$ A. From $t=50$ s cell 1 develops a small internal
# short (conductance $G_\mathrm{short}$), so it drains faster than cell 2. Without protection the
# string ends with cell 1 overdischarged and cell 2 under-used. With protection, a shunt resistor
# parallel to each cell is switched from $10^8\,\Omega$ to about $2\cdot10^{-8}\,\Omega$ the moment
# that cell's state of charge reaches 0.2: the cell is bypassed and empties into its shunt. This is
# the cell balancing idea of G. L. Plett's battery-management lectures.
#
# **Cell.** The state of charge follows coulomb counting, $Q\,d\mathrm{SOC}/dt=-I_\mathrm{int}$,
# with the internal current $I_\mathrm{int}$ (discharge positive). The terminal voltage is
#
# $$V = E_\mathrm{OCV}(\mathrm{SOC}_\mathrm{surf})
#      - \eta_{IR,1C}\,\frac{I_\mathrm{int}}{I_{1C}}
#      - \frac{2R_gT}{F}\,\operatorname{asinh}\frac{I_\mathrm{int}}{2J_0I_{1C}} ,$$
#
# where the surface state of charge belongs to a spherical diffusion particle with time constant
# $\tau$: $\partial s/\partial t=\tau^{-1}x^{-2}\partial_x(x^2\partial_x s)$ on $0\le x\le1$ with
# $\tau^{-1}\partial_x s=-I_\mathrm{int}/(3Q)$ at the surface. The particle is divided into equal
# spherical shells (finite volumes).
#
# **Circuit.** The source forces $I_\mathrm{app}$ through each cell-shunt pair, so
# $I_\mathrm{int}=I_\mathrm{app}+V/R_\mathrm{shunt}+G_\mathrm{short}(t)\,V$. Together with the
# voltage equation this is an algebraic equation for $I_\mathrm{int}$, solved by Newton's method.
#
# **Time integration.** OneSimulation's variable-step BDF2 and, for comparison, the adaptive
# Dormand-Prince Runge-Kutta 5(4) method. The shunt switching is an event on
# $\mathrm{SOC}-0.2$; the integrator locates it, and the run continues from there with the shunt
# switched and a small first step.
#
# **Reference.** (i) Charge conservation: the particle average must equal
# $1-\int I_\mathrm{int}dt/Q$. (ii) Cell 2 carries the constant source current until it switches,
# so $t_2=0.8\,Q/I_\mathrm{app}$ (up to the leakage through $10^8\,\Omega$), and its surface state
# of charge has the exact series solution of a sphere with constant flux (Crank, *The Mathematics
# of Diffusion*). (iii) A bypassed cell relaxes to $V=-I_\mathrm{app}R_\mathrm{on}$. (iv) The
# published behaviour: the shunts fire near 4400 s and 5800 s at 0.5C and near 1350 s and 1450 s
# at 2C; without shunts cell 1 ends at a state of charge of about $-0.1$ and cell 2 at about 0.07.

from time import perf_counter

import numpy as np
from onesim.numerics.time import AdaptiveBDF, AdaptiveRungeKutta45, EventDefinition
from onesim.simulation import smooth_step
from onesim.visualization import plots, theme
from scipy.optimize import brentq

theme.set()

# ## Cell and circuit
#
# Capacity, short and shunts follow the published model; the open-circuit curve and the loss
# parameters are representative of a lithium-ion cell, with a voltage that collapses to 0 V at an
# empty cell and continues linearly below.

capacity, one_c, ohmic_1c, exchange_ratio, diffusion_time = (
    12.0 * 3600.0,
    12.0,
    0.02,
    0.85,
    1000.0,
)
shunt_off, shunt_on, threshold = 1.0e8, 1.0e8 * np.finfo(float).eps, 0.2
plateau, slope, knee, tafel = 3.55, 0.6, 0.025, 2 * 8.314462618 * 298.15 / 96485.33212
short = smooth_step(49.5, 50.5, low=0.0, high=0.5)
open_circuit = lambda s: np.where(
    s >= 0,
    plateau + slope * s - plateau * np.exp(-np.maximum(s, 0) / knee),
    (slope + plateau / knee) * s,
)
open_circuit_slope = lambda s: slope + plateau / knee * np.exp(-np.maximum(s, 0) / knee)
losses = lambda current: ohmic_1c * current / one_c + tafel * np.arcsinh(
    current / (2 * exchange_ratio * one_c)
)
loss_slope = lambda current: ohmic_1c / one_c + tafel / np.hypot(
    1.0, current / (2 * exchange_ratio * one_c)
) / (2 * exchange_ratio * one_c)


def simulate(rate, shunts, end_time, solver, shells=20):
    """Discharge the string at the C-rate `rate` with or without shunts; history of both cells."""
    source, faces = rate * one_c, np.linspace(0.0, 1.0, shells + 1)
    volume, link = np.diff(faces**3) / 3, faces[1:-1] ** 2 * shells / diffusion_time
    diffusion = (
        np.diag(link, 1)
        + np.diag(link, -1)
        - np.diag(np.r_[link, 0.0] + np.r_[0.0, link])
    ) / volume[:, None]
    lever, sink, bypassed = (
        diffusion_time / (6 * capacity * shells),
        np.eye(shells)[-1] / (3 * capacity * volume[-1]),
        np.zeros(2, dtype=bool),
    )
    conductance = lambda time: shunts / np.where(
        bypassed, shunt_on, shunt_off
    ) + np.array([float(short(time)), 0.0])
    surface = lambda state, current: state[:, -1] - lever * current

    def current(time, state):
        """Internal currents of both cells: Newton solution of the circuit equation."""
        g, i = conductance(time), np.full(2, source)
        for _ in range(100):
            update = (
                i - source - g * (open_circuit(surface(state, i)) - losses(i))
            ) / (
                1 + g * (open_circuit_slope(surface(state, i)) * lever + loss_slope(i))
            )
            i = i - update
            if np.all(np.abs(update) <= 1e-13 * (1 + np.abs(i))):
                break
        return i

    feedback = lambda time, state, i: (
        conductance(time)
        * open_circuit_slope(surface(state, i))
        / (
            1
            + conductance(time)
            * (open_circuit_slope(surface(state, i)) * lever + loss_slope(i))
        )
    )
    rate_of_change = lambda time, y: (
        y.reshape(2, shells) @ diffusion.T
        - np.outer(current(time, y.reshape(2, shells)), sink)
    ).ravel()
    jacobian = lambda time, y: np.kron(np.eye(2), diffusion) - np.diag(
        np.kron(
            feedback(time, y.reshape(2, shells), current(time, y.reshape(2, shells))),
            sink,
        )
    )
    events = (
        [
            EventDefinition(
                f"shunt {cell + 1} on",
                lambda time, y, cell=cell: (
                    -1.0
                    if bypassed[cell]
                    else 3 * y.reshape(2, shells)[cell] @ volume - threshold
                ),
                direction=-1,
                terminal=True,
            )
            for cell in range(2)
        ]
        if shunts
        else ()
    )
    options = {"jacobian": jacobian} if isinstance(solver, AdaptiveBDF) else {}
    time, state, first_step, parts, fired, steps, start = (
        0.0,
        np.ones(2 * shells),
        0.1,
        [],
        {},
        0,
        perf_counter(),
    )
    while time < end_time:
        segment = solver.integrate(
            rate_of_change,
            state,
            time,
            end_time,
            initial_step=first_step,
            events=events,
            **options,
        )
        states = np.array(segment.states).reshape(-1, 2, shells)
        parts.append(
            (
                np.array(segment.times),
                states,
                np.array([current(t, s) for t, s in zip(segment.times, states)]),
            )
        )
        for event in segment.events:
            fired[event.event_id] = event.time
            bypassed[int(event.event_id.split()[1]) - 1] = True
        time, state, first_step, steps = (
            segment.times[-1],
            segment.states[-1],
            1e-3,
            steps + segment.accepted_steps,
        )
    times, states, currents = (np.concatenate(part) for part in zip(*parts))
    voltage = open_circuit(states[:, :, -1] - lever * currents) - losses(currents)
    return dict(
        times=times,
        soc=3 * states @ volume,
        current=currents,
        surface=states[:, :, -1] - lever * currents,
        voltage=voltage,
        events=fired,
        steps=steps,
        seconds=perf_counter() - start,
    )


# ## Simulation
#
# Both C-rates without and with shunts (BDF2, 20 shells), a Dormand-Prince reference at a
# tolerance of $10^{-10}$, and at 0.5C with shunts both integrators on 10, 20 and 40 shells and
# for tolerances from $10^{-4}$ to $10^{-7}$.

methods = {
    "BDF2": lambda tolerance: AdaptiveBDF(
        order=2,
        relative_tolerance=tolerance,
        absolute_tolerance=tolerance,
        maximum_step=25.0,
    ),
    "RK45": lambda tolerance: AdaptiveRungeKutta45(
        relative_tolerance=tolerance, absolute_tolerance=tolerance, maximum_step=25.0
    ),
}
cases = {
    (0.5, False): 6700.0,
    (0.5, True): 10000.0,
    (2.0, False): 1700.0,
    (2.0, True): 10000.0,
}
runs = {
    case: simulate(*case, end, methods["BDF2"](1e-6)) for case, end in cases.items()
}
reference = simulate(0.5, True, 10000.0, methods["RK45"](1e-10))
study = {
    (shells, name, tolerance): simulate(0.5, True, 10000.0, method(tolerance), shells)
    for name, method in methods.items()
    for shells, tolerance in (
        (10, 1e-6),
        (20, 1e-6),
        (40, 1e-6),
        (20, 1e-4),
        (20, 1e-5),
        (20, 1e-7),
    )
}

# ## Comparison
#
# Before its shunt fires, cell 2 carries the constant source current. For a sphere with a constant
# surface flux the surface value lies below the average by
# $(I\tau/3Q)\,[\,1/5-2\sum_n e^{-\lambda_n^2t/\tau}/\lambda_n^2\,]$ with $\tan\lambda_n=\lambda_n$.

roots = np.array(
    [
        brentq(lambda x: np.sin(x) - x * np.cos(x), n * np.pi, (n + 0.5) * np.pi)
        for n in range(1, 400)
    ]
)
exact_lag = lambda time: -(6.0 * diffusion_time / (3 * capacity)) * (
    0.2
    - 2 * np.sum(np.exp(-np.outer(time, roots**2) / diffusion_time) / roots**2, axis=1)
)
probe = np.linspace(20.0, 5700.0, 300)
lag = {
    shells: np.interp(
        probe,
        study[shells, "BDF2", 1e-6]["times"],
        (study[shells, "BDF2", 1e-6]["surface"] - study[shells, "BDF2", 1e-6]["soc"])[
            :, 1
        ],
    )
    for shells in (10, 20, 40)
}
mesh_error = {
    shells: np.abs(values - exact_lag(probe)).max() for shells, values in lag.items()
}

print(
    "C-rate  shunts   shunt 1 fires   shunt 2 fires   final state of charge (cell 1, cell 2)"
)
for (rate, shunts), run in runs.items():
    fired = "".join(
        f"{run['events'].get(name, np.nan):13.2f} s "
        for name in ("shunt 1 on", "shunt 2 on")
    )
    print(
        f"{rate:5.1f}   {'on ' if shunts else 'off'}   {fired}  {run['soc'][-1, 0]:+.4f}, {run['soc'][-1, 1]:+.4f}"
    )
print(
    "published: about 4400 s and 5800 s at 0.5C, 1350 s and 1450 s at 2C; without shunts at 0.5C about -0.1 and 0.07"
)
for rate in (0.5, 2.0):
    run, source = runs[rate, True], rate * one_c
    before = run["times"] <= run["events"]["shunt 2 on"]
    switching = (
        0.8 * capacity
        - np.trapezoid(run["voltage"][before, 1] / shunt_off, run["times"][before])
    ) / source
    print(
        f"{rate}C: cell 2 switches at {run['events']['shunt 2 on']:.6f} s, closed form {switching:.6f} s; bypassed cells end at "
        f"{run['voltage'][-1, 0]:.3e} V and {run['voltage'][-1, 1]:.3e} V, closed form {-source * shunt_on:.3e} V"
    )
balance = max(
    np.abs(
        run["soc"][-1]
        - 1
        + np.trapezoid(run["current"], run["times"], axis=0) / capacity
    ).max()
    for run in (*runs.values(), reference)
)
print(f"charge balance, largest |SOC - (1 - ∫I dt / Q)| over all runs: {balance:.1e}")
print(
    "particle against the exact series, largest error of surface minus average: "
    + ", ".join(
        f"{error:.2e} ({shells} shells)" for shells, error in mesh_error.items()
    )
    + f"; observed order {np.log2(mesh_error[10] / mesh_error[20]):.2f} and {np.log2(mesh_error[20] / mesh_error[40]):.2f}"
)
print(
    "shells  tolerance  method   accepted steps   run time [s]   |t1 - reference| [s]"
)
for (shells, name, tolerance), run in sorted(study.items()):
    print(
        f"{shells:6d}  {tolerance:9.0e}  {name:6s} {run['steps']:16d} {run['seconds']:14.1f} "
        f"{abs(run['events']['shunt 1 on'] - reference['events']['shunt 1 on']):22.1e}"
    )

# ## Figure
#
# The 0.5C case: terminal voltages and states of charge without and with shunts, and the bypass
# transient of cell 1. Below: the particle of cell 2 against the exact series, and the number of
# steps both integrators need when the particle mesh or the tolerance changes.

curves = lambda key: {
    f"cell {cell + 1}"
    + (", shunts" if shunts else ""): (
        runs[0.5, shunts]["times"],
        runs[0.5, shunts][key][:, cell],
    )
    for shunts in (False, True)
    for cell in (0, 1)
}
dashed = {"cell 1, shunts": "dashed", "cell 2, shunts": "dashed"}
colours = {"cell 1": 0, "cell 2": 1, "cell 1, shunts": 0, "cell 2, shunts": 1}
fired = reference["events"]["shunt 1 on"]
near = lambda run: (run["times"] > fired - 20) & (run["times"] < fired + 300)
shells_axis, tolerances = [10, 20, 40], [1e-4, 1e-5, 1e-6, 1e-7]
fig = plots.figure(2, 3, size=(17, 10))
plots.line(
    y=curves("voltage"),
    colors=colours,
    styles=dashed,
    xlimits=(0, 7000),
    ylimits=(-11, 5),
    xlabel="time [s]",
    ylabel="terminal voltage [V]",
    title="Cell voltages at 0.5C",
    legend="lower left",
    ax=fig[0],
)
plots.line(
    y=curves("soc"),
    colors=colours,
    styles=dashed,
    hlines={"shunt threshold": threshold},
    xlimits=(0, 7000),
    xlabel="time [s]",
    ylabel="state of charge",
    title="State of charge at 0.5C",
    legend="upper right",
    ax=fig[1],
)
plots.line(
    y={
        "current, RK45 reference": (
            reference["times"][near(reference)] - fired,
            np.maximum(reference["current"][near(reference), 0], 1e-2),
        )
    },
    points={
        "current, BDF2 steps": (
            (runs[0.5, True]["times"] - fired)[near(runs[0.5, True])][::12],
            np.maximum(runs[0.5, True]["current"][near(runs[0.5, True]), 0], 1e-2)[
                ::12
            ],
        )
    },
    right={
        "surface state of charge": (
            reference["times"][near(reference)] - fired,
            reference["surface"][near(reference), 0],
        ),
        "average state of charge": (
            reference["times"][near(reference)] - fired,
            reference["soc"][near(reference), 0],
        ),
    },
    scale="logy",
    xlabel="time after shunt 1 fires [s]",
    ylabel="current of cell 1 [A]",
    right_label="state of charge",
    title="Bypass transient of cell 1",
    legend="upper right",
    ax=fig[2],
)
plots.line(
    y={f"{shells} shells": (probe, 1e3 * values) for shells, values in lag.items()},
    reference={"exact series": (probe, 1e3 * exact_lag(probe))},
    scale="logx",
    xlabel="time [s]",
    ylabel="surface - average state of charge [10⁻³]",
    title="Particle of cell 2 and the exact series",
    legend="upper right",
    ax=fig[3],
)
plots.line(
    shells_axis,
    {
        name: [study[shells, name, 1e-6]["steps"] for shells in shells_axis]
        for name in methods
    },
    markers=True,
    scale="loglog",
    xlabel="particle shells",
    ylabel="accepted steps over 10000 s",
    title="Explicit steps grow with the particle mesh",
    legend="upper left",
    ax=fig[4],
)
plots.line(
    tolerances,
    {
        name: [study[20, name, tolerance]["steps"] for tolerance in tolerances]
        for name in methods
    },
    markers=True,
    scale="loglog",
    xlabel="relative tolerance, 20 shells",
    ylabel="accepted steps over 10000 s",
    title="... and do not follow the tolerance",
    legend="upper right",
    ax=fig[5],
)
plots.show()

# ## Going further
#
# Replace the representative open-circuit curve by measured data and the near-short by a balancing
# resistor of a few ohms: the bypassed cell then drains slowly and a second event on rising state
# of charge can switch the shunt off again. Several cells in series only add blocks to the state
# vector. The step counts show the classic stiffness signature: the explicit method's step is set
# by the fastest diffusion mode, which grows with the square of the shell count, while BDF2 follows
# the accuracy of the slow dynamics.
