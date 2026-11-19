"""OneSimulation example: single-particle model of a lithium-ion cell against the full porous-electrode model.
Requires the installed, activated OneSimulation runtime.

The single-particle model (SPM) of a graphite-SiOx | NMC811 cell is checked against the exact series
solution of spherical diffusion and compared with the single-particle model with electrolyte (SPMe)
and the Doyle-Fuller-Newman model (DFN) from C/10 to 1C. The SPM error grows linearly with the rate;
the SPMe removes most of it.
"""

# # Single-particle model against the full porous-electrode model
#
# When the electrolyte conducts well and the current is moderate, every particle in an electrode
# sees almost the same reaction rate. The single-particle model then replaces each electrode by one
# representative sphere that carries the mean reaction current $j_\pm=\pm I/(a_\pm L_\pm)$:
#
# $$\frac{\partial c_s}{\partial t}=\frac{D_s}{r^2}\frac{\partial}{\partial r}\Big(r^2\frac{\partial c_s}{\partial r}\Big),\qquad
#   -D_s\frac{\partial c_s}{\partial r}\Big|_{R}=\frac{j}{F},\qquad
#   V=U_+(\theta_+^\mathrm{surf})-U_-(\theta_-^\mathrm{surf})-\frac{2RT}{F}\Big(\operatorname{asinh}\frac{|j_+|}{2j_{0,+}}+\operatorname{asinh}\frac{j_-}{2j_{0,-}}\Big) .$$
#
# The SPMe adds the salt diffusion and the ohmic and concentration losses of the electrolyte, still
# with a uniform reaction; the DFN resolves the reaction distribution through both electrodes.
#
# **Reference.** For a constant current the surface concentration of a sphere follows the exact
# series (Crank, *The Mathematics of Diffusion*, 1975, eq. 6.62)
#
# $$c_s(R,t)=c_0-\frac{jR}{FD_s}\Big[3\tau+\frac15-2\sum_{n}\frac{e^{-\lambda_n^2\tau}}{\lambda_n^2}\Big],\qquad \tau=\frac{D_st}{R^2},\quad\tan\lambda_n=\lambda_n .$$
#
# Marquis, Sulzer, Timms, Please and Chapman, *J. Electrochem. Soc.* 166 (2019) A3693, derive the SPM
# and the SPMe as asymptotic limits of the DFN for well-conducting electrodes: the voltage error of the
# SPM grows linearly with the C-rate; the SPMe adds the electrolyte correction and leaves a much
# smaller error. In this cell the positive electrode conducts poorly (0.18 S/m), so the reaction
# spreads unevenly through it and the SPMe error grows more slowly than the quadratic law of the
# ideal case, but stays several times below the SPM error. Cell parameters of Chen et al., *J. Electrochem. Soc.* 167
# (2020) 080534 (LG M50, 5 Ah).

import numpy as np
from onesim.physics.chemistry import (
    BatteryElectrodeLayer,
    BatteryElectrolyteLayer,
    PorousElectrodeBattery,
)
from onesim.visualization import plots, theme
from scipy.optimize import brentq

theme.set()

# ## Cell

faraday, area = 96485.33212, 0.1027
negative_ocp = lambda x, T: (
    1.9793 * np.exp(-39.3631 * x)
    + 0.2482
    - 0.0909 * np.tanh(29.8538 * (x - 0.1234))
    - 0.04478 * np.tanh(14.9159 * (x - 0.2769))
    - 0.0205 * np.tanh(30.4444 * (x - 0.6103))
)
positive_ocp = lambda y, T: (
    4.4875
    - 0.8090 * y
    - 0.0428 * np.tanh(18.5138 * (y - 0.5542))
    - 17.7326 * np.tanh(15.7890 * (y - 0.3117))
    + 17.5842 * np.tanh(15.9308 * (y - 0.3120))
)
negative = BatteryElectrodeLayer(
    thickness=85.2e-6,
    porosity=0.25,
    electrolyte_diffusivity=1.7694e-10 * 0.25**1.5,
    electrolyte_conductivity=0.9487 * 0.25**1.5,
    particle_radius=5.86e-6,
    particle_diffusivity=3.3e-14,
    active_fraction=0.75,
    maximum_concentration=33133,
    initial_stoichiometry=0.9014,
    reference_exchange_current=6.48e-7 * np.sqrt(1000) * 33133 / 2,
    electronic_conductivity=215,
    open_circuit_potential=negative_ocp,
)
positive = BatteryElectrodeLayer(
    thickness=75.6e-6,
    porosity=0.335,
    electrolyte_diffusivity=1.7694e-10 * 0.335**1.5,
    electrolyte_conductivity=0.9487 * 0.335**1.5,
    particle_radius=5.22e-6,
    particle_diffusivity=4e-15,
    active_fraction=0.665,
    maximum_concentration=63104,
    initial_stoichiometry=0.2661,
    reference_exchange_current=3.42e-6 * np.sqrt(1000) * 63104 / 2,
    electronic_conductivity=0.18,
    open_circuit_potential=positive_ocp,
)
separator = BatteryElectrolyteLayer(
    thickness=12e-6,
    porosity=0.47,
    electrolyte_diffusivity=1.7694e-10 * 0.47**1.5,
    electrolyte_conductivity=0.9487 * 0.47**1.5,
)
cell = PorousElectrodeBattery(
    negative=negative,
    separator=separator,
    positive=positive,
    electrolyte_concentration=1000,
    transference_number=0.2594,
)

# ## Particle against the exact series
#
# 50 minutes at 1C, the SPM on particle meshes of 8 to 64 shells; the exact series with 200 roots. The
# surface value is extrapolated from the outer shell, so the error falls at first order.

one_c = 5.0 / area
probe = np.linspace(0, 3000, 61)
spm_meshes = {
    n: cell.solve(
        mode="SPM",
        current_density=one_c,
        final_time=3000,
        radial_cells=n,
        output_times=probe,
    )
    for n in (8, 16, 32, 64)
}
roots = np.array(
    [
        brentq(lambda z: np.sin(z) - z * np.cos(z), n * np.pi, (n + 0.5) * np.pi - 1e-9)
        for n in range(1, 201)
    ]
)
series = (
    lambda layer, j, t: layer.initial_stoichiometry * layer.maximum_concentration
    - j
    * layer.particle_radius
    / (faraday * layer.particle_diffusivity)
    * (
        3 * layer.particle_diffusivity * t / layer.particle_radius**2
        + 0.2
        - 2
        * np.sum(
            np.exp(
                -np.outer(
                    layer.particle_diffusivity * t / layer.particle_radius**2, roots**2
                )
            )
            / roots**2,
            axis=1,
        )
    )
)
exact = [
    series(
        layer,
        sign
        * one_c
        / (3 * layer.active_fraction / layer.particle_radius * layer.thickness),
        probe,
    )
    for layer, sign in ((negative, 1), (positive, -1))
]
errors = {
    n: [
        np.abs(run.particle_surface_concentration[k][:, 0] - exact[k]).max()
        / layer.maximum_concentration
        for k, layer in enumerate((negative, positive))
    ]
    for n, run in spm_meshes.items()
}
print(
    "particle shells   largest error of the surface stoichiometry (negative, positive)"
)
for n, (en, ep) in errors.items():
    print(f"{n:15d}   {en:.3e}   {ep:.3e}")
print(
    f"observed orders 32 -> 64 shells: {np.log2(errors[32][0] / errors[64][0]):.2f} (negative), {np.log2(errors[32][1] / errors[64][1]):.2f} (positive)"
)

# ## Three models over the rate
#
# Each discharge covers 90 % of the nominal capacity, so all three models stay away from the steep end
# of the voltage curve.

rates = (0.1, 0.2, 0.5, 1.0)
runs = {
    (mode, rate): cell.solve(
        mode=mode,
        current_density=rate * one_c,
        final_time=0.9 * 3600 / rate,
        radial_cells=16,
        through_cells=(10, 5, 10),
        output_times=np.linspace(0, 0.9 * 3600 / rate, 300),
    )
    for mode in ("SPM", "SPMe", "DFN")
    for rate in rates
}
deviation = {
    (mode, rate): runs[mode, rate].terminal_voltage - runs["DFN", rate].terminal_voltage
    for mode in ("SPM", "SPMe")
    for rate in rates
}
print("C-rate   largest |V_SPM - V_DFN|   largest |V_SPMe - V_DFN|   [mV]")
for rate in rates:
    print(
        f"{rate:6.1f}   {1e3 * np.abs(deviation['SPM', rate]).max():20.2f}   {1e3 * np.abs(deviation['SPMe', rate]).max():23.2f}"
    )
slopes = {
    mode: np.polyfit(
        np.log(rates),
        np.log([np.abs(deviation[mode, rate]).max() for rate in rates]),
        1,
    )[0]
    for mode in ("SPM", "SPMe")
}
print(
    f"growth of the error with the C-rate: SPM ~ C^{slopes['SPM']:.2f}, SPMe ~ C^{slopes['SPMe']:.2f} (ideal asymptotic limits: 1 and 2)"
)

# ## Figure

capacity = lambda rate: 5.0 * rate * runs["DFN", rate].times / 3600
fig = plots.figure(2, 2, size=(14, 10))
plots.line(
    y={
        f"DFN {rate:g}C": (capacity(rate), runs["DFN", rate].terminal_voltage)
        for rate in (0.1, 0.5, 1.0)
    },
    reference={
        f"DFN {rate:g}C": (capacity(rate), runs["SPM", rate].terminal_voltage)
        for rate in (0.1, 0.5, 1.0)
    },
    xlabel="discharged capacity [Ah]",
    ylabel="terminal voltage [V]",
    title="DFN (solid) and SPM (dashed)",
    legend="lower left",
    ax=fig[0],
)
plots.line(
    y={
        f"{mode} {rate:g}C": (capacity(rate), 1e3 * deviation[mode, rate])
        for mode in ("SPM", "SPMe")
        for rate in (0.5, 1.0)
    },
    styles={"SPMe 0.5C": "dashed", "SPMe 1C": "dashed"},
    colors={"SPM 0.5C": 0, "SPM 1C": 1, "SPMe 0.5C": 0, "SPMe 1C": 1},
    xlabel="discharged capacity [Ah]",
    ylabel="V - V_DFN [mV]",
    title="Voltage error of the reduced models",
    legend="lower left",
    ax=fig[1],
)
plots.line(
    rates,
    {
        mode: [1e3 * np.abs(deviation[mode, rate]).max() for rate in rates]
        for mode in ("SPM", "SPMe")
    },
    markers=True,
    slope=1,
    scale="loglog",
    xlabel="C-rate",
    ylabel="largest |V - V_DFN| [mV]",
    title="Error over the C-rate",
    legend="upper left",
    ax=fig[2],
)
plots.convergence(
    [1 / n for n in errors],
    {
        "negative particle": [e[0] for e in errors.values()],
        "positive particle": [e[1] for e in errors.values()],
    },
    order=1,
    xlabel="shell width / particle radius",
    ylabel="largest error of the surface stoichiometry",
    title="SPM particle against the exact series",
    ax=fig[3],
)
plots.show()

# ## Going further
#
# The SPM runs in a fraction of the DFN time: use it for long cycling studies or state estimation,
# and switch to the SPMe or the DFN where the electrolyte limits the cell, as in the rate-capability
# example.
