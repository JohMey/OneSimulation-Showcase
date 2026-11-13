"""OneSimulation example: overdischarge protection of a two-cell battery string with shunt resistors.
Requires the installed, activated OneSimulation runtime.

Two lumped lithium-ion cells in series, one with an internal short, are
discharged by a current source; event-switched shunts bypass a cell once its
state of charge falls to 20 %. The circuit is integrated with adaptive BDF2
(analytic Jacobian) and Dormand-Prince RK45, both with event location, and graded
against charge conservation, the closed-form switching time and bypass state, the
exact series solution of the particle diffusion and the published protection
behaviour.
"""


# # Battery overdischarge protection using shunt resistances
#
# A 2s string of two identical 12 Ah lithium-ion cells feeds a 10 m$\Omega$ load
# from a current source $I_\mathrm{app}=C_\mathrm{rate}\times 12$ A. From $t=50$ s
# cell 1 develops a small internal short (conductance $G_\mathrm{short}$), so it
# drains faster than cell 2. Without protection the string ends with cell 1
# overdischarged and cell 2 under-used. With protection, a shunt resistor parallel
# to each cell is switched from $10^8\,\Omega$ to $\varepsilon\cdot10^8\,\Omega
# \approx 2.2\cdot10^{-8}\,\Omega$ (machine epsilon $\varepsilon$) the moment that
# cell's state of charge reaches 0.2: the cell is bypassed and empties into its
# shunt, while the load current continues through the shunt. This is the cell
# balancing idea of G. L. Plett's battery-management lectures (*Battery Pack
# Balancing and Power Estimation*, Coursera).
#
# **Cell model (lumped battery).** Each cell is zero-dimensional. Its state of
# charge obeys coulomb counting, $Q\,d\mathrm{SOC}/dt=-I_\mathrm{int}$, where the
# internal current $I_\mathrm{int}$ (discharge positive) includes the internal
# short. The terminal voltage is
#
# $$V = E_\mathrm{OCV}(\mathrm{SOC}_\mathrm{surf})
#      - \eta_{IR,1C}\,\frac{I_\mathrm{int}}{I_{1C}}
#      - \frac{2R_gT}{F}\,\operatorname{asinh}\frac{I_\mathrm{int}}{2J_0I_{1C}} ,$$
#
# with ohmic and activation losses, and the concentration overpotential enters
# through the surface state of charge of a dimensionless spherical diffusion
# particle with time constant $\tau$: $\partial s/\partial t
# = \tau^{-1}x^{-2}\partial_x(x^2\partial_x s)$ on $0\le x\le1$, symmetry at $x=0$
# and $\tau^{-1}\partial_x s = -I_\mathrm{int}/(3Q)$ at $x=1$, so that the volume
# average of $s$ is exactly the coulomb-counted SOC.
#
# **Circuit.** The current source forces $I_\mathrm{app}$ through each cell-shunt
# pair, so Kirchhoff's current law at a pair and the internal short give
# $I_\mathrm{int}=I_\mathrm{app}+V/R_\mathrm{shunt}+G_\mathrm{short}(t)\,V$. Together
# with the voltage equation this is an algebraic (index-one) constraint for
# $I_\mathrm{int}$, solved by Newton's method inside every right-hand-side
# evaluation; the remaining unknowns are the particle states.
#
# **Numerics.** The particle is discretised with a cell-centred finite-volume
# method on spherical shells (method of lines, second order). Time integration
# uses OneSimulation's variable-step BDF2 with Newton iterations on the exact
# (analytic) Jacobian, and, for comparison, the adaptive Dormand-Prince
# Runge-Kutta 5(4) method. In both, the shunt switching is an event on
# $\mathrm{SOC}-0.2$ with a reset of the shunt state, exactly the discrete-state
# logic of the published model; the event is located on the step, the shunt
# state is reset, and the integration restarts from the event with a small
# step, so that the 2 s bypass spike that follows is resolved.
#
# **Reference.** (i) Charge conservation: the volume-averaged particle SOC must
# equal $1-\int I_\mathrm{int}dt/Q$; (ii) cell 2 carries the constant source
# current until it switches, so its switching time is
# $t_2=0.8\,Q/I_\mathrm{app}$ (up to the $10^8\,\Omega$ leakage) and its surface SOC
# has the exact series solution of diffusion in a sphere with constant flux;
# (iii) a bypassed cell relaxes to zero current, so its terminal voltage becomes
# $-I_\mathrm{app}R_\mathrm{on}$ and its SOC $-I_\mathrm{app}R_\mathrm{on}/E'_\mathrm{OCV}(0)$;
# (iv) the published result: with shunts both cells end at SOC $\approx 0$;
# without them cell 1 is overdischarged (SOC about $-0.1$) and cell 2 stops at
# about 0.07 (0.5C); the shunts fire near 4400 s and 5800 s at 0.5C and near
# 1350 s and 1450 s at 2C.

from importlib.metadata import version
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# OneSimulation figure style: cream paper, dark-green ink, green-sand-copper field colours
CREAM, INK, GREEN, COPPER = "#F7F2E7", "#223D35", "#336B57", "#B66E43"
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation", ["#214D43", "#5F9175", "#A8BC91", "#E4D6A0", "#CD9561", "#A85539"])
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation_diverging", ["#104F54", "#5E9996", "#F7F2E7", "#D2996E", "#A4472D"])
for _cmap in (FIELD_CMAP, DIVERGING_CMAP):
    if _cmap.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cmap)
plt.rcParams.update({
    "figure.facecolor": CREAM, "savefig.facecolor": CREAM, "axes.facecolor": "#FFFAF2",
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": "#9FA593",
    "xtick.color": INK, "ytick.color": INK, "grid.color": "#D5D7C8", "image.cmap": "onesimulation",
    "axes.prop_cycle": cycler(color=[GREEN, COPPER, "#5F9175", "#A85539", "#214D43", "#CD9561", "#8B9A8D"]),
})

from onesim.numerics.discretization.execution.backends.dynamic_analysis import (
    AdaptiveBDF,
    AdaptiveRungeKutta45,
    EventDefinition,
)

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# Documented in the published model: two identical cells in series, initial SOC
# 1, current source $C_\mathrm{rate}\times I_{1C}$ with C-rates 0.5 and 2, load
# 10 m$\Omega$, shunts $10^8\,\Omega$ switched to $\varepsilon\cdot10^8\,\Omega$ at
# SOC $\le 0.2$, internal short of cell 1 from 50 s (smoothed unit step of 1 s
# width). Read off the published current curves: the 12 Ah capacity (0.5C draws
# 6 A) and the short conductance of 0.5 S (2 A extra current at about 4 V, and
# $-4$ A at the $-8$ V of the overdischarged cell). The open-circuit curve and the
# loss parameters are not published; the values below are representative of a
# lithium-ion cell, with an open-circuit voltage that collapses to 0 V at SOC 0
# and continues linearly below it.

FARADAY, R_GAS, temperature = 96485.33212, 8.314462618, 298.15
capacity_ah = 12.0
capacity = capacity_ah * 3600.0           # C, Q
one_c_current = capacity_ah               # A, I_1C
initial_soc = 1.0
eta_ir_1c = 0.02                          # V, ohmic overpotential at 1C
exchange_ratio = 0.85                     # J_0, exchange current in units of I_1C
diffusion_time = 1000.0                   # s, tau
short_conductance = 0.5                   # S, G_short (cell 1 only)
short_start = 50.0                        # s
shunt_off = 1.0e8                         # Ohm, shunt with SOC > 0.2
machine_epsilon = np.finfo(float).eps
shunt_on = shunt_off * machine_epsilon    # Ohm, about 2.2e-8
load_resistance = 10e-3                   # Ohm
threshold = 0.2
ocv_plateau, ocv_slope, ocv_knee = 3.55, 0.6, 0.025


def open_circuit_voltage(s):
    s = np.asarray(s, dtype=float)
    upper = ocv_plateau + ocv_slope * s - ocv_plateau * np.exp(-np.maximum(s, 0.0) / ocv_knee)
    return np.where(s >= 0.0, upper, (ocv_slope + ocv_plateau / ocv_knee) * s)


def open_circuit_slope(s):
    s = np.asarray(s, dtype=float)
    return np.where(s >= 0.0, ocv_slope + ocv_plateau / ocv_knee * np.exp(-np.maximum(s, 0.0) / ocv_knee),
                    ocv_slope + ocv_plateau / ocv_knee)


tafel = 2.0 * R_GAS * temperature / FARADAY


def losses(current):
    return eta_ir_1c * current / one_c_current + tafel * np.arcsinh(current / (2 * exchange_ratio * one_c_current))


def loss_slope(current):
    return eta_ir_1c / one_c_current + tafel / np.sqrt(
        1.0 + (current / (2 * exchange_ratio * one_c_current)) ** 2) / (2 * exchange_ratio * one_c_current)


def short_ramp(time):
    z = np.clip(time - short_start + 0.5, 0.0, 1.0)
    return z**3 * (10.0 - 15.0 * z + 6.0 * z * z)


print(f"E_OCV(1) = {open_circuit_voltage(1.0):.3f} V, E_OCV(0.2) = {open_circuit_voltage(0.2):.3f} V, "
      f"E_OCV'(0) = {open_circuit_slope(0.0):.1f} V; shunt on = {shunt_on:.3e} Ohm")


# ## Geometry and mesh
#
# Both cells and the circuit are zero-dimensional. The only mesh is the
# pseudo-dimensional radius $x\in[0,1]$ of each cell's diffusion particle, divided
# into $N$ equal spherical shells (cell-centred finite volumes). Its two boundaries
# carry explicit conditions: zero flux (symmetry) at the centre $x=0$ and the
# current-driven flux $-I_\mathrm{int}/(3Q)$ at the surface $x=1$, where the surface
# value is reconstructed from the outermost cell and that flux. The production
# runs use $N=20$; $N=10$ and $40$ form the mesh study.


class StringModel:
    """Two-cell string: particle finite volumes, nested Kirchhoff solve, events."""

    def __init__(self, shells, source_current, shunts):
        self.n, self.source, self.shunts = shells, source_current, shunts
        faces = np.linspace(0.0, 1.0, shells + 1)
        self.dx = 1.0 / shells
        self.volume = (faces[1:] ** 3 - faces[:-1] ** 3) / 3.0
        self.area = faces**2
        self.surface_lever = self.dx * diffusion_time / (6.0 * capacity)   # s_surf = s_N - lever * I
        # Tridiagonal diffusion operator of one particle (zero flux at x = 0).
        laplace = np.zeros((shells, shells))
        for i in range(shells - 1):
            g = self.area[i + 1] / (self.dx * diffusion_time)
            laplace[i, i] -= g / self.volume[i]
            laplace[i, i + 1] += g / self.volume[i]
            laplace[i + 1, i + 1] -= g / self.volume[i + 1]
            laplace[i + 1, i] += g / self.volume[i + 1]
        self.laplace = laplace
        self.calls = 0
        self.jacobians = 0

    def initial_state(self):
        return np.concatenate([np.full(2 * self.n, initial_soc), [1.0, 1.0]])

    def conductance(self, time, state):
        """Parallel conductance seen by each cell: shunt (if present) plus internal short."""
        shunt = 1.0 / (shunt_off * state[2 * self.n:]) if self.shunts else np.zeros(2)
        return shunt + np.array([short_conductance * short_ramp(time), 0.0])

    def circuit(self, time, state):
        """Newton solve of I_int - I_app - G (E_OCV(s_surf) - losses) = 0 for both cells."""
        outer = state[[self.n - 1, 2 * self.n - 1]]
        total = self.conductance(time, state)
        current = np.full(2, self.source, dtype=float)
        for _ in range(100):
            surface = outer - self.surface_lever * current
            residual = current - self.source - total * (open_circuit_voltage(surface) - losses(current))
            slope = 1.0 + total * (open_circuit_slope(surface) * self.surface_lever + loss_slope(current))
            update = residual / slope
            current -= update
            if np.all(np.abs(update) <= 1e-13 * (1.0 + np.abs(current))):
                break
        surface = outer - self.surface_lever * current
        voltage = open_circuit_voltage(surface) - losses(current)
        return current, voltage, surface, residual

    def rhs(self, time, state):
        self.calls += 1
        current = self.circuit(time, state)[0]
        rates = np.zeros_like(state)
        for k in range(2):
            block = slice(k * self.n, (k + 1) * self.n)
            rates[block] = self.laplace @ state[block]
            rates[(k + 1) * self.n - 1] -= current[k] / (3.0 * capacity * self.volume[-1])
        return rates

    def jacobian(self, time, state):
        """Exact Jacobian: diffusion operator plus the implicit-function derivative dI/ds_N."""
        self.jacobians += 1
        current, _, surface, _ = self.circuit(time, state)
        total = self.conductance(time, state)
        slope_e = open_circuit_slope(surface)
        d_current = total * slope_e / (1.0 + total * (slope_e * self.surface_lever + loss_slope(current)))
        matrix = np.zeros((state.size, state.size))
        for k in range(2):
            block = slice(k * self.n, (k + 1) * self.n)
            matrix[block, block] = self.laplace
            last = (k + 1) * self.n - 1
            matrix[last, last] -= d_current[k] / (3.0 * capacity * self.volume[-1])
        return matrix

    def soc(self, state):
        return 3.0 * np.array([state[:self.n] @ self.volume, state[self.n:2 * self.n] @ self.volume])

    def events(self):
        if not self.shunts:
            return ()

        def switch(k):
            def indicator(time, state):
                return self.soc(state)[k] - threshold if state[2 * self.n + k] > 0.5 else -1.0

            def reset(time, state):
                state = state.copy()
                state[2 * self.n + k] = machine_epsilon
                return state

            return EventDefinition(f"shunt {k + 1} on", indicator, direction=-1, terminal=True, reset=reset)

        return (switch(0), switch(1))


def simulate(c_rate, shunts, end_time, method="bdf", shells=20, tolerance=1e-6):
    """Integrate segment by segment: every shunt event is terminal, then the run restarts with a small step.

    The restart matters for BDF2, which otherwise re-enters with its last (large) step and an unchecked
    implicit-Euler start step right across the 2 s bypass spike.
    """
    model = StringModel(shells, c_rate * one_c_current, shunts)
    if method == "bdf":
        solver = AdaptiveBDF(order=2, relative_tolerance=tolerance, absolute_tolerance=tolerance,
                             maximum_step=25.0)
        options = {"jacobian": model.jacobian}
    else:
        solver = AdaptiveRungeKutta45(relative_tolerance=tolerance, absolute_tolerance=tolerance,
                                      maximum_step=25.0)
        options = {}
    start = perf_counter()
    time, state, first_step = 0.0, model.initial_state(), 0.1
    times, states, events, accepted, rejected = [time], [state], {}, 0, 0
    while time < end_time:
        segment = solver.integrate(model.rhs, state, time, end_time, initial_step=first_step,
                                   events=model.events(), **options)
        times.extend(segment.times[1:])
        states.extend(segment.states[1:])
        events.update({event.event_id: event.time for event in segment.events})
        accepted, rejected = accepted + segment.accepted_steps, rejected + segment.rejected_steps
        time, state, first_step = segment.times[-1], segment.states[-1], 1e-3
    wall = perf_counter() - start
    times, states = np.asarray(times), np.asarray(states)
    circuit = [model.circuit(t, y) for t, y in zip(times, states)]
    # Left limits at the stored points: the shunt state of the previous point (it only changes at events).
    left_limit = [circuit[0][0]] + [
        model.circuit(t, np.concatenate([y[:2 * shells], previous[2 * shells:]]))[0]
        for t, y, previous in zip(times[1:], states[1:], states[:-1])]
    return {
        "model": model, "wall": wall, "times": times, "accepted": accepted, "rejected": rejected,
        "soc": np.array([model.soc(y) for y in states]),
        "current": np.array([c[0] for c in circuit]), "voltage": np.array([c[1] for c in circuit]),
        "surface": np.array([c[2] for c in circuit]),
        "kcl": np.array([np.abs(c[3]) / (np.abs(c[0]) + abs(model.source)) for c in circuit]),
        "left_current": np.array(left_limit),
        "events": events,
    }


# ## Solve
#
# **Production runs**: both C-rates, without and with shunts, BDF2 on $N=20$
# shells. Without shunts the runs end where the published curves end (6700 s at
# 0.5C, 1700 s at 2C); with shunts they run to 10000 s.

cases = {(0.5, False): 6700.0, (0.5, True): 10000.0, (2.0, False): 1700.0, (2.0, True): 10000.0}
runs = {key: simulate(key[0], key[1], end) for key, end in cases.items()}
for (c_rate, shunts), result in runs.items():
    print(f"C-rate {c_rate}, shunts {'on ' if shunts else 'off'}: {result['accepted']} BDF2 steps, "
          f"{result['model'].calls} rhs and {result['model'].jacobians} Jacobian evaluations, {result['wall']:.1f} s")

# **Reference run** for the method comparison: Dormand-Prince at $10^{-10}$ on
# $N=20$, 0.5C with shunts.
reference = simulate(0.5, True, 10000.0, method="rk45", tolerance=1e-10)

# **Method and mesh comparison** at 0.5C with shunts: BDF2 (exact Jacobian)
# against Dormand-Prince RK45 on 10, 20 and 40 shells at a relative tolerance of
# $10^{-6}$, and on 20 shells for tolerances $10^{-4}\ldots10^{-7}$.
comparison = {}
for shells in (10, 20, 40):
    for method in ("bdf", "rk45"):
        comparison[(shells, method, 1e-6)] = simulate(0.5, True, 10000.0, method=method, shells=shells)
for tolerance in (1e-4, 1e-5, 1e-7):
    for method in ("bdf", "rk45"):
        comparison[(20, method, tolerance)] = simulate(0.5, True, 10000.0, method=method, tolerance=tolerance)


# **Particle mesh study against the exact series.** Before its shunt fires, cell 2
# carries the constant current $I_\mathrm{app}$ (plus a $4\cdot10^{-8}$ A leakage).
# For a sphere with a constant surface flux the surface-minus-average deviation is
# $-(I\tau/3Q)\,[\,1/5-2\sum_n e^{-\lambda_n^2t/\tau}/\lambda_n^2\,]$ with
# $\tan\lambda_n=\lambda_n$ (Crank, *The Mathematics of Diffusion*, 1975).

roots = []
for n in range(1, 400):
    low, high = n * np.pi, n * np.pi + 0.5 * np.pi      # f = sin - x cos changes sign in between
    for _ in range(80):
        middle = 0.5 * (low + high)
        same_side = np.sign(np.sin(middle) - middle * np.cos(middle)) == np.sign(np.sin(low) - low * np.cos(low))
        low, high = (middle, high) if same_side else (low, middle)
    roots.append(0.5 * (low + high))
roots = np.array(roots)


def exact_surface_deviation(time, current):
    series = np.sum(np.exp(-np.outer(time, roots**2) / diffusion_time) / roots**2, axis=1)
    return -(current * diffusion_time / (3.0 * capacity)) * (0.2 - 2.0 * series)


source_half_c = 0.5 * one_c_current
probe = np.linspace(20.0, 5700.0, 300)
mesh_errors = {}
for shells in (10, 20, 40):
    result = comparison[(shells, "bdf", 1e-6)]
    window = result["times"] < 5750.0
    deviation = result["surface"][window, 1] - result["soc"][window, 1]
    mesh_errors[shells] = np.max(np.abs(np.interp(probe, result["times"][window], deviation)
                                        - exact_surface_deviation(probe, source_half_c)))


# ## Comparison with the reference

print("\nPublished protection behaviour vs OneSimulation (BDF2, N = 20):")
published = {(0.5, True): (4400, 5800), (2.0, True): (1350, 1450)}
for key, (t1_pub, t2_pub) in published.items():
    result = runs[key]
    t1, t2 = result["events"]["shunt 1 on"], result["events"]["shunt 2 on"]
    print(f"  C-rate {key[0]}: shunt 1 fires at {t1:8.2f} s (published about {t1_pub} s), "
          f"shunt 2 at {t2:8.2f} s (published about {t2_pub} s)")
    print(f"             final SOC cell 1 {result['soc'][-1, 0]: .3e}, cell 2 {result['soc'][-1, 1]: .3e} "
          f"(published: both about 0)")
for key in ((0.5, False), (2.0, False)):
    result = runs[key]
    print(f"  C-rate {key[0]} without shunts, t = {result['times'][-1]:.0f} s: SOC cell 1 {result['soc'][-1, 0]: .4f} "
          f"(overdischarged), cell 2 {result['soc'][-1, 1]:.4f} (under-used); terminal voltages "
          f"{result['voltage'][-1, 0]:.2f} V and {result['voltage'][-1, 1]:.2f} V")
print("  published without shunts at 0.5C, 6700 s: cell 1 about -0.1, cell 2 about 0.07, cell 1 at about -8 V")

print("\nClosed-form checks:")
for key in ((0.5, True), (2.0, True)):
    result = runs[key]
    source = key[0] * one_c_current
    times = result["times"]
    before = times <= result["events"]["shunt 2 on"]
    leakage = np.trapezoid(result["voltage"][before, 1] / shunt_off, times[before])
    exact_t2 = (0.8 * capacity - leakage) / source
    print(f"  C-rate {key[0]}: cell 2 switching time {result['events']['shunt 2 on']:.6f} s vs "
          f"(0.8 Q - leakage)/I_app = {exact_t2:.6f} s, rel. dev. "
          f"{abs(result['events']['shunt 2 on'] - exact_t2) / exact_t2:.1e}")
    bypass_voltage = -source * shunt_on
    bypass_soc = bypass_voltage / open_circuit_slope(0.0)
    print(f"             bypassed cells at the end: V = {result['voltage'][-1, 0]:.4e} / {result['voltage'][-1, 1]:.4e} V "
          f"vs -I_app R_on = {bypass_voltage:.4e} V")
    print(f"             final SOC {result['soc'][-1, 0]:.4e} / {result['soc'][-1, 1]:.4e} vs "
          f"-I_app R_on / E'_OCV(0) = {bypass_soc:.4e}")
    print(f"             shunt current of cell 1 at the end: {result['voltage'][-1, 0] / shunt_on:.6f} A "
          f"(carries the load current, sign: from - to + terminal) vs I_app = {source:.1f} A")

print("\nCharge conservation, particle average vs 1 - (trapezoidal integral of the solved current)/Q:")
# The residual is the error of the trapezoidal rule over the solver's own steps (the particle conserves charge
# exactly); with shunts it is dominated by the 2 s bypass spike of up to 900 A.
balance_runs = {f"BDF2 tol 1e-6, {key[0]}C, shunts {'on ' if key[1] else 'off'}": result for key, result in runs.items()}
balance_runs["RK45 tol 1e-10, 0.5C, shunts on "] = reference
for label, result in balance_runs.items():
    steps = np.diff(result["times"])[:, None]
    charge = np.sum(0.5 * steps * (result["current"][:-1] + result["left_current"][1:]), axis=0)
    counted = initial_soc - charge / capacity
    print(f"  {label:32s}: |SOC - (1 - int I dt / Q)| = {abs(result['soc'][-1, 0] - counted[0]):.1e} (cell 1), "
          f"{abs(result['soc'][-1, 1] - counted[1]):.1e} (cell 2)")
print("  max relative Kirchhoff residual of the nested current solve over all production runs: "
      f"{max(result['kcl'].max() for result in runs.values()):.1e}")

print("\nParticle finite volumes vs the exact sphere series (cell 2, 0.5C, 20 s to 5700 s):")
for shells, error in mesh_errors.items():
    print(f"  N = {shells:2d}: max |surface deviation error| = {error:.3e}  (steady deviation "
          f"{source_half_c * diffusion_time / (15 * capacity):.3e})")
print(f"  observed order: {np.log2(mesh_errors[10] / mesh_errors[20]):.2f} (10->20), "
      f"{np.log2(mesh_errors[20] / mesh_errors[40]):.2f} (20->40)")

t1_ref, t2_ref = reference["events"]["shunt 1 on"], reference["events"]["shunt 2 on"]
print(f"\nTime integrators, 0.5C with shunts (reference RK45 at 1e-10, N = 20: t1 = {t1_ref:.6f} s, "
      f"t2 = {t2_ref:.6f} s):")
for (shells, method, tolerance), result in sorted(comparison.items(), key=lambda item: (item[0][2] != 1e-6, item[0])):
    name = "BDF2 + exact Jacobian" if method == "bdf" else "Dormand-Prince RK45  "
    event_text = (f", |t1 - t1_ref| {abs(result['events']['shunt 1 on'] - t1_ref):.1e} s, "
                  f"|t2 - t2_ref| {abs(result['events']['shunt 2 on'] - t2_ref):.1e} s") if shells == 20 else ""
    print(f"  N = {shells:2d}, tol {tolerance:.0e}, {name}: steps {result['accepted']:6d} (+{result['rejected']:3d}), "
          f"rhs {result['model'].calls:7d}, Jacobians {result['model'].jacobians:5d}, wall {result['wall']:5.1f} s"
          f"{event_text}")


# ## Figure

fig, axes = plt.subplots(3, 2, figsize=(12, 13.5), constrained_layout=True)
styles = {False: "-", True: "--"}
colours = {0.5: (GREEN, COPPER), 2.0: ("#8B9A8D", "#CD9561")}
ax = axes[0, 0]
for (c_rate, shunts), result in runs.items():
    cell1, cell2 = colours[c_rate]
    ax.plot(result["times"], result["voltage"][:, 0], styles[shunts], color=cell1, lw=1.5,
            label=f"{c_rate}C cell 1" if not shunts else None)
    ax.plot(result["times"], result["voltage"][:, 1], styles[shunts], color=cell2, lw=1.5,
            label=f"{c_rate}C cell 2" if not shunts else None)
ax.set(xlabel="Time / s", ylabel="Terminal voltage / V", title="Cell voltages (solid: no shunts, dashed: shunts)",
       ylim=(-11, 4.6))
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[0, 1]
for (c_rate, shunts), result in runs.items():
    cell1, cell2 = colours[c_rate]
    ax.plot(result["times"], result["current"][:, 0], styles[shunts], color=cell1, lw=1.5)
    ax.plot(result["times"], result["current"][:, 1], styles[shunts], color=cell2, lw=1.5)
peak = reference["current"].max()
ax.set(xlabel="Time / s", ylabel="Internal cell current (discharge > 0) / A", ylim=(-3, 32),
       title=f"Cell currents (bypass discharge peaks at {peak:.0f} A, clipped)")
ax.grid(alpha=0.3)

ax = axes[1, 0]
for (c_rate, shunts), result in runs.items():
    cell1, cell2 = colours[c_rate]
    ax.plot(result["times"], result["soc"][:, 0], styles[shunts], color=cell1, lw=1.5,
            label=f"{c_rate}C cell 1" if not shunts else None)
    ax.plot(result["times"], result["soc"][:, 1], styles[shunts], color=cell2, lw=1.5,
            label=f"{c_rate}C cell 2" if not shunts else None)
    for event_id, event_time in result["events"].items():
        ax.plot(event_time, threshold, "o", color=INK, ms=4)
ax.axhline(threshold, color=INK, ls=":", lw=1.0)
ax.axhline(0.0, color="#9FA593", lw=0.8)
ax.set(xlabel="Time / s", ylabel="State of charge", title="State of charge (dots: shunt events at SOC = 0.2)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1, 1]
zoom, coarse = reference, runs[(0.5, True)]
t1 = zoom["events"]["shunt 1 on"]
window = (zoom["times"] > t1 - 20) & (zoom["times"] < t1 + 300)
marks = np.flatnonzero((coarse["times"] > t1 - 20) & (coarse["times"] < t1 + 300))
marks = marks[np.unique(np.linspace(0, marks.size - 1, 30).astype(int))]
ax.semilogy(zoom["times"][window] - t1, np.maximum(zoom["current"][window, 0], 1e-2), color=GREEN, lw=1.6,
            label="Cell 1 current, RK45 reference")
ax.semilogy(coarse["times"][marks] - coarse["events"]["shunt 1 on"], np.maximum(coarse["current"][marks, 0], 1e-2),
            "s", color=GREEN, ms=4, mfc="none", label="Cell 1 current, BDF2 steps")
ax2 = ax.twinx()
ax2.plot(zoom["times"][window] - t1, zoom["surface"][window, 0], color=COPPER, lw=1.6, label="Surface SOC (reference)")
ax2.plot(zoom["times"][window] - t1, zoom["soc"][window, 0], "--", color=COPPER, lw=1.2, label="Average SOC (reference)")
ax2.plot(coarse["times"][marks] - coarse["events"]["shunt 1 on"], coarse["soc"][marks, 0], "s", color=COPPER, ms=4,
         mfc="none", label="Average SOC, BDF2 steps")
ax2.set_ylabel("State of charge", color=COPPER)
ax.set(xlabel="Time after shunt 1 fires / s", ylabel="Current / A", title="Bypass transient of cell 1 at 0.5C")
lines, lines2 = ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()
ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper right", fontsize=8.5)
ax.grid(alpha=0.3, which="both")

ax = axes[2, 0]
fine = np.geomspace(1.0, 5700.0, 600)
ax.plot(fine, exact_surface_deviation(fine, source_half_c) * 1e3, color=INK, lw=2.2, label="Exact sphere series")
for shells, colour in zip((10, 20, 40), (COPPER, GREEN, "#8B9A8D")):
    result = comparison[(shells, "bdf", 1e-6)]
    window = result["times"] < 5750.0
    ax.plot(result["times"][window], (result["surface"][window, 1] - result["soc"][window, 1]) * 1e3, "--",
            color=colour, lw=1.4, label=f"Finite volumes, N = {shells}")
ax.set(xlabel="Time / s", ylabel="(surface - average) SOC  x 1000", xscale="log", xlim=(1, 6000),
       title="Cell 2 particle: finite volumes vs exact series")
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3, which="both")

ax = axes[2, 1]
shell_counts = np.array([10, 20, 40])
tolerances = np.array([1e-4, 1e-5, 1e-6, 1e-7])
for method, marker, colour, name in (("bdf", "s", COPPER, "BDF2 + exact Jacobian"),
                                     ("rk45", "o", GREEN, "Dormand-Prince RK45")):
    steps = [comparison[(n, method, 1e-6)]["accepted"] for n in shell_counts]
    ax.loglog(shell_counts, steps, marker + "-", color=colour, lw=1.8, ms=7, label=f"{name}, tol 1e-6")
ax.set(xlabel="Particle shells N (solid)", ylabel="Accepted steps over 10000 s")
ax.set_title("Stiffness: explicit steps grow with the mesh and ignore the tolerance", pad=28)
ax.set_xticks(shell_counts, [str(n) for n in shell_counts])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.grid(alpha=0.3, which="both")
top = ax.twiny()
for method, marker, colour, name in (("bdf", "s", COPPER, "BDF2 + exact Jacobian"),
                                     ("rk45", "o", GREEN, "Dormand-Prince RK45")):
    steps = [comparison[(20, method, tol)]["accepted"] for tol in tolerances]
    top.loglog(tolerances, steps, marker + ":", color=colour, lw=1.3, ms=5, label=f"{name}, N = 20")
top.set_xlabel("Relative tolerance at N = 20 (dotted)")
top.invert_xaxis()
lines, lines2 = ax.get_legend_handles_labels(), top.get_legend_handles_labels()
ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper left", fontsize=8.5)
plt.show()


# ## Going further
#
# Replace the representative open-circuit curve by measured data and the
# switching logic by a balancing resistor of a few ohms instead of a near-short:
# the bypassed cell then drains slowly and the event logic can switch the shunt
# off again (a second event on rising SOC). Several cells in series only add
# blocks to the state vector and rows to the nested Kirchhoff solve. The step
# counts show the classic stiffness signature: the explicit method's step is set
# by the fastest diffusion mode, which grows with the square of the shell count,
# while BDF2 with an exact Jacobian follows the accuracy of the slow dynamics.
