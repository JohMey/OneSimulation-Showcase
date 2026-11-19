"""OneSimulation example: electrode balancing of a lithium-ion cell from its open-circuit voltage.
Requires the installed, activated OneSimulation runtime.

The open-circuit voltage of a cell is the difference of the two electrode potentials, each taken at
the lithiation the electrode has at the given state of charge. The four lithiation limits that fix
this balance are fitted to an open-circuit voltage curve with the parameter-study optimizer of
OneSimulation and compared with the values that generated the data.
"""

# # Electrode balancing of a lithium-ion cell
#
# Ageing changes a lithium-ion cell in two ways that the terminal voltage alone cannot tell apart:
# lithium is lost to side reactions, and active material of one electrode stops taking part. Both
# shift the **balance** of the electrodes, the stoichiometry (degree of lithiation) $x$ of the
# negative and $y$ of the positive electrode at 0 % and at 100 % state of charge. Between the
# limits both change linearly with the state of charge $s$,
#
# $$x(s)=x_0+s\,(x_{100}-x_0),\qquad y(s)=y_0+s\,(y_{100}-y_0),$$
#
# and the open-circuit voltage of the cell is the difference of the two half-cell potentials,
#
# $$U_\mathrm{oc}(s)=U_+\big(y(s)\big)-U_-\big(x(s)\big).$$
#
# With the half-cell curves $U_\pm$ known from coin-cell measurements, the four limits follow from
# a least-squares fit of $U_\mathrm{oc}(s)$ to the measured cell curve. There is no transport, no
# current and no geometry: the model is zero-dimensional.
#
# **Reference.** C.-H. Chen, F. Brosa Planella, K. O'Regan, D. Gastol, W. D. Widanage and
# E. Kendrick, *J. Electrochem. Soc.* 167 (2020) 080534: half-cell potentials of the graphite-SiOx
# negative and the NMC811 positive electrode of the LG M50 cell, and its balance $x_0=0.0279$,
# $x_{100}=0.9014$, $y_0=0.9077$, $y_{100}=0.2661$. The open-circuit voltage curve used here is
# **synthetic**: it is generated from these four values at 101 states of charge, with normally
# distributed noise of 2 mV as a measurement would have. A correct fit recovers the four values
# within the uncertainty that the noise leaves.

import numpy as np
from onesim.simulation.studies import SimulationStudyService, StudyParameter
from onesim.visualization import plots, theme

theme.set()

# ## Half-cell potentials and the measured curve
#
# The two half-cell potentials of Chen et al. as functions of the stoichiometry, the open-circuit
# voltage of a balance as the difference of the two along the state of charge, and the synthetic
# measurement. The charge windows of the electrodes follow from their loading (active volume
# times maximum lithium concentration times Faraday's constant): 5.83 Ah for the negative and
# 8.73 Ah for the positive electrode of the 0.1027 m² electrode pair.

negative_ocp = lambda x: (
    1.9793 * np.exp(-39.3631 * x)
    + 0.2482
    - 0.0909 * np.tanh(29.8538 * (x - 0.1234))
    - 0.04478 * np.tanh(14.9159 * (x - 0.2769))
    - 0.0205 * np.tanh(30.4444 * (x - 0.6103))
)
positive_ocp = lambda y: (
    4.4875
    - 0.8090 * y
    - 0.0428 * np.tanh(18.5138 * (y - 0.5542))
    - 17.7326 * np.tanh(15.7890 * (y - 0.3117))
    + 17.5842 * np.tanh(15.9308 * (y - 0.3120))
)
soc = np.linspace(0.0, 1.0, 101)
open_circuit = lambda b: (
    positive_ocp(b["y0"] + soc * (b["y100"] - b["y0"]))
    - negative_ocp(b["x0"] + soc * (b["x100"] - b["x0"]))
)
names = ("x0", "x100", "y0", "y100")
true = dict(zip(names, (0.0279, 0.9014, 0.9077, 0.2661)))
noise = 2e-3
measured = open_circuit(true) + np.random.default_rng(2026).normal(0.0, noise, soc.size)
capacity_negative, capacity_positive = (
    96485.33212 * fraction * thickness * concentration * 0.1027 / 3600
    for fraction, thickness, concentration in (
        (0.75, 85.2e-6, 33133),
        (0.665, 75.6e-6, 63104),
    )
)
print(
    f"measured open-circuit voltage from {measured[0]:.3f} V at 0 % to {measured[-1]:.3f} V at 100 % state of charge"
)
print(
    f"electrode charge windows: negative {capacity_negative:.3f} Ah, positive {capacity_positive:.3f} Ah"
)

# ## Fit of the four limits
#
# `SimulationStudyService` evaluates the open-circuit voltage for trial balances and measures the
# mean squared deviation from the data in mV²; its projected-gradient optimizer (forward
# differences of $10^{-6}$ of each interval, backtracking line search) minimises it within the
# bounds $0\le x_0\le0.2$, $0.7\le x_{100}\le1$, $0.75\le y_0\le1$, $0.1\le y_{100}\le0.45$. It
# starts from a guess several per cent off: $x_0=0.06$, $x_{100}=0.85$, $y_0=0.85$, $y_{100}=0.32$.

bounds = {"x0": (0.0, 0.2), "x100": (0.7, 1.0), "y0": (0.75, 1.0), "y100": (0.1, 0.45)}
start = {"x0": 0.06, "x100": 0.85, "y0": 0.85, "y100": 0.32}
study = SimulationStudyService(
    tuple(StudyParameter(name, *bounds[name]) for name in names),
    lambda balance: balance,
    lambda curve: {"msd": float(np.mean((1e3 * (curve - measured)) ** 2))},
    solve=open_circuit,
)
best = study.optimize(
    ("msd",),
    start=start,
    step=1e-6,
    tolerance=1e-5,
    maximum_iterations=3000,
    maximum_runs=400000,
)
fitted = open_circuit(best.parameters)

# ## Comparison with the generating values
#
# The uncertainty of each limit follows from the Gauss-Newton covariance $(J^TJ)^{-1}$ of the
# residuals divided by the noise, with the sensitivities $J$ from one more evaluation per limit.
# From the fitted windows follow the charge the cell can deliver, $C_-\,(x_{100}-x_0)$, and the
# excess of the positive electrode.

sensitivity = np.column_stack(
    [
        (
            open_circuit(dict(best.parameters) | {name: best.parameters[name] + 1e-6})
            - fitted
        )
        / (1e-6 * noise)
        for name in names
    ]
)
sigma = dict(zip(names, np.sqrt(np.diag(np.linalg.inv(sensitivity.T @ sensitivity)))))
print(
    f"optimizer: {best.iterations} iterations, {best.run_count} evaluations of the open-circuit voltage, "
    f"χ² = N·MSD/s² = {soc.size * best.objectives['msd'] / (1e3 * noise) ** 2:.1f} for {soc.size - 4} degrees of freedom"
)
print("limit      start     fitted     ± 1 σ       true   deviation")
for name in names:
    print(
        f"{name:5s} {start[name]:10.4f} {best.parameters[name]:10.5f} {sigma[name]:9.5f} {true[name]:10.4f}"
        f"   {(best.parameters[name] - true[name]) / sigma[name]:+5.2f} σ"
    )
cell = {
    label: (
        capacity_negative * (b["x100"] - b["x0"]),
        capacity_positive * (b["y0"] - b["y100"]),
    )
    for label, b in (("fitted", best.parameters), ("true", true))
}
for label, (negative, positive) in cell.items():
    print(
        f"{label:6s} cell capacity {negative:.4f} Ah (negative window), positive window {positive:.4f} Ah, "
        f"excess of the positive {100 * (positive / negative - 1):.1f} %"
    )
print(
    f"rms voltage residual {1e3 * np.sqrt(np.mean((fitted - measured) ** 2)):.3f} mV (noise {1e3 * noise:.1f} mV), "
    f"largest deviation of the fitted from the true curve {1e3 * np.max(np.abs(fitted - open_circuit(true))):.2f} mV"
)

# ## Figure
#
# Left: the measured curve with the start guess and the fit. Middle: the residuals against the
# noise level. Right: the balancing diagram, both half-cell potentials over the state of charge
# of the cell with the fitted limits; the cell voltage is their difference.

fig = plots.figure(1, 3, size=(17, 5))
plots.line(
    soc,
    {"fitted balance": fitted},
    points={"measured (synthetic)": (soc[::2], measured[::2])},
    reference={"start guess": open_circuit(start)},
    xlabel="state of charge",
    ylabel="open-circuit voltage [V]",
    title="Cell open-circuit voltage",
    legend="lower right",
    ax=fig[0],
)
plots.line(
    soc,
    {"measured - fitted": 1e3 * (measured - fitted)},
    hlines=[1e3 * noise, -1e3 * noise],
    ylimits=(-8, 8),
    xlabel="state of charge",
    ylabel="voltage residual [mV]",
    title="Residuals (dotted: ± noise)",
    legend="upper right",
    ax=fig[1],
)
plots.line(
    soc,
    {
        "positive electrode U₊(y)": positive_ocp(
            best.parameters["y0"]
            + soc * (best.parameters["y100"] - best.parameters["y0"])
        )
    },
    right={
        "negative electrode U₋(x)": negative_ocp(
            best.parameters["x0"]
            + soc * (best.parameters["x100"] - best.parameters["x0"])
        )
    },
    xlabel="state of charge of the cell",
    ylabel="positive potential vs Li [V]",
    right_label="negative potential vs Li [V]",
    title="Balancing diagram with the fitted limits",
    legend="center right",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Shift both $y$ limits by the same amount to mimic loss of lithium inventory, or shrink one window
# to mimic loss of active material, generate the curve again and fit it: the fit tells the two
# ageing modes apart. Fitting the derivative $\mathrm dU_\mathrm{oc}/\mathrm ds$ as well sharpens the
# graphite limits, whose plateaus leave the voltage itself nearly flat.
