"""OneSimulation example: capacity and power fade of a lumped lithium-ion battery from a parasitic film reaction.
Requires the installed, activated OneSimulation runtime.

The solid-electrolyte interphase on the negative electrode grows by solvent diffusion through the
film; the reaction consumes cyclable lithium and adds film resistance. Together with an empirical
cycle fade, the lumped cell is aged over ten years at three temperatures. Film thickness, end of
life and the discharge energy of aged cells are compared with closed-form solutions.
"""

# # Capacity and power fade of a lumped battery
#
# Most of the calendar ageing of a lithium-ion cell comes from a parasitic reaction at the negative
# electrode: solvent molecules diffuse through the solid-electrolyte interphase (SEI) and react at
# the graphite surface. The film thickens, every mole of film product consumes two moles of
# cyclable lithium, and the film resistance grows:
#
# $$\frac{dL}{dt}=\bar V\,\frac{c_\mathrm{sol}D(T)}{L},\qquad
#   Q_\mathrm{loss}=\frac{2F\,A_s}{\bar V}\,(L-L_0),\qquad
#   R=R_0+\frac{L}{\kappa_\mathrm{SEI}A_s},\qquad D(T)=D_\mathrm{ref}\,e^{-\frac{E_a}{R_g}\left(\frac1T-\frac1{T_\mathrm{ref}}\right)},$$
#
# with the partial molar volume $\bar V$ of the film, the solvent concentration $c_\mathrm{sol}$ and the
# active surface $A_s$ of the negative electrode. Cycling adds an empirical fade of the capacity,
# $Q_\mathrm{cyc}(n)=Q_0\,e^{-k\,\max(n-n_0,0)}$ after $n$ equivalent full cycles; the cell does one
# cycle per day, and $Q=Q_\mathrm{cyc}(n)-Q_\mathrm{loss}$. The aged cell is a lumped electrothermal
# battery (the model of the constant-current example) with capacity $Q$ and resistance $R$.
#
# **Reference.** The film law has the closed form $L(t)=\sqrt{L_0^2+2\bar Vc_\mathrm{sol}D\,t}$, so the
# calendar loss grows with $\sqrt t$ (M. B. Pinson and M. Z. Bazant, *J. Electrochem. Soc.* 160 (2013)
# A243; H. J. Ploehn, P. Ramadass and R. E. White, *J. Electrochem. Soc.* 151 (2004) A456). The end of
# life, 80 % of the initial capacity, follows from the closed forms by root finding. For a constant
# current $I$ the lumped cell delivers $E=Q\int_{s_1}^{s_0}U_\mathrm{oc}\,ds-I^2R\,t$ between the states
# of charge $s_0$ and $s_1$. The open-circuit curve and the fresh cell (5 Ah, 30 mΩ) follow the LG M50
# cell of Chen et al., *J. Electrochem. Soc.* 167 (2020) 080534.

import numpy as np
from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.time import AdaptiveBDF, ClassicalRungeKutta4, EventDefinition
from onesim.physics.chemistry import (
    BatteryChemistry,
    BatteryCycleCapacityFade,
    LumpedBattery,
    sei_area_resistance,
    sei_thickness_rate,
    solvent_limited_sei_flux,
)
from onesim.results.storage import MemoryOutputSink, Output
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue
from onesim.visualization import plots, theme
from scipy.optimize import brentq

theme.set()

# ## Cell and ageing parameters

faraday, gas, year = 96485.33212, 8.314462618, 365.25 * 86400
q0, r0, surface = 5.0 * 3600, 0.030, 3.36
solvent, molar_volume, film0, kappa_film = 4541.0, 9.585e-5, 5e-9, 5e-6
d_ref, activation = 5e-22, 40e3
cycling = BatteryCycleCapacityFade(
    initial_capacity=q0, fade_rate=4e-5, induction_cycles=200
)
diffusivity = lambda T: d_ref * np.exp(-activation / gas * (1 / T - 1 / 298.15))
open_circuit = lambda s: (
    4.4875
    - 0.8090 * (y := 0.2661 + (1 - s) * 0.5829)
    - 0.0428 * np.tanh(18.5138 * (y - 0.5542))
    - 17.7326 * np.tanh(15.7890 * (y - 0.3117))
    + 17.5842 * np.tanh(15.9308 * (y - 0.3120))
    - 1.9793 * np.exp(-39.3631 * (x := 0.0279 + s * 0.8735))
    - 0.2482
    + 0.0909 * np.tanh(29.8538 * (x - 0.1234))
    + 0.04478 * np.tanh(14.9159 * (x - 0.2769))
    + 0.0205 * np.tanh(30.4444 * (x - 0.6103))
)

# ## Ten years at 25, 35 and 45 °C
#
# The film thickness (in nm, over time in years) is integrated with OneSimulation's adaptive BDF2
# method, using the library's film laws; an event marks the end of life at 80 % of the initial
# capacity.

capacity = lambda years, film: cycling.remaining_capacity(
    365.25 * years
) - 2 * faraday * surface / molar_volume * (1e-9 * film - film0)
ageing = {}
for celsius in (25, 35, 45):
    D = diffusivity(273.15 + celsius)
    growth = (
        lambda t, y: year
        * 1e9
        * sei_thickness_rate(
            solvent_limited_sei_flux(solvent, D, 1e-9 * y), molar_volume
        )
    )
    end_of_life = EventDefinition(
        "end of life", lambda t, y: capacity(t, y[0]) - 0.8 * q0, direction=-1
    )
    run = AdaptiveBDF(
        order=2, relative_tolerance=1e-6, absolute_tolerance=1e-6, maximum_step=0.05
    ).integrate(
        growth,
        np.array([1e9 * film0]),
        0.0,
        10.0,
        initial_step=1e-6,
        events=(end_of_life,),
    )
    times, film = np.array(run.times), np.array(run.states)[:, 0]
    exact = lambda t, D=D: 1e9 * np.sqrt(
        film0**2 + 2 * molar_volume * solvent * D * t * year
    )
    ageing[celsius] = dict(
        times=times,
        film=film,
        exact=exact(times),
        capacity=capacity(times, film),
        life=[event.time for event in run.events],
        exact_life=brentq(lambda t: capacity(t, exact(t)) - 0.8 * q0, 1e-6, 50.0),
    )

# ## Discharge of aged cells
#
# The cell aged at 25 °C, taken after 0, 3, 6 and 9 years, discharged at 5 A from full to 2 % state of
# charge (the fresh cell's 1C).

cell = LumpedDomain()
numerics = Numerics(spatial=LumpedParameters(), time_integrator=ClassicalRungeKutta4())
discharges = {}
for age in (0, 3, 6, 9):
    film = np.interp(age, ageing[25]["times"], ageing[25]["film"])
    q, resistance = (
        capacity(age, film),
        r0 + float(sei_area_resistance(1e-9 * film, kappa_film)) / surface,
    )
    model = LumpedBattery(
        domain=cell.system,
        chemistry=BatteryChemistry(
            "aged",
            open_circuit_voltage=lambda s, T: open_circuit(s),
            internal_resistance=lambda s, T: resistance,
        ),
        charge_capacity=q,
        current=5.0,
        thermal_capacity=70.0,
        heat_transfer=0.3,
        ambient_temperature=298.15,
    )
    record = MemoryOutputSink()
    end = 0.98 * q / 5.0
    Simulation(
        problem=Problem(
            name="aged discharge",
            geometry=cell,
            physics=(model,),
            initial_conditions=(
                InitialValue(model.state_of_charge, 1.0),
                InitialValue(model.temperature, 298.15),
            ),
        ),
        mesh=LumpedMesh(cell),
        numerics=numerics,
    ).run(
        start_time=0.0,
        end_time=end,
        time_step=end / 1000,
        output=Output(
            fields=(model.terminal_voltage, model.temperature),
            interval=end / 1000,
            sink=record,
        ),
    )
    s = np.linspace(0.02, 1.0, 4001)
    discharges[age] = dict(
        times=record.times,
        voltage=record.time_series(model.terminal_voltage)[1],
        q=q,
        resistance=resistance,
        energy=5.0
        * np.trapezoid(record.time_series(model.terminal_voltage)[1], record.times),
        exact=q * np.trapezoid(open_circuit(s), s) - 25.0 * resistance * end,
        heating=record.time_series(model.temperature)[1].max() - 298.15,
    )

# ## Results

for celsius, a in ageing.items():
    print(
        f"{celsius} °C: film after 10 years {a['film'][-1]:.2f} nm (closed form {a['exact'][-1]:.2f} nm, largest deviation "
        f"{np.abs(a['film'] - a['exact']).max():.1e} nm); capacity {a['capacity'][-1] / q0:.2%}; end of life after "
        + (
            f"{a['life'][0]:.4f} years (closed form {a['exact_life']:.4f})"
            if a["life"]
            else f"more than 10 years (closed form {a['exact_life']:.2f})"
        )
    )
print(
    "age [y]   capacity [Ah]   resistance [mΩ]   energy [Wh]   closed form [Wh]   deviation   heating [K]"
)
for age, d in discharges.items():
    print(
        f"{age:7d}   {d['q'] / 3600:13.3f}   {1e3 * d['resistance']:15.2f}   {d['energy'] / 3600:11.4f}   {d['exact'] / 3600:16.4f}"
        f"   {100 * (d['energy'] / d['exact'] - 1):+8.3f} %   {d['heating']:11.2f}"
    )

# ## Figure

fig = plots.figure(1, 3, size=(19, 5.4))
plots.line(
    y={f"{c} °C": (a["times"], a["film"]) for c, a in ageing.items()},
    reference={f"{c} °C": (a["times"], a["exact"]) for c, a in ageing.items()},
    xlabel="time [years]",
    ylabel="SEI thickness [nm]",
    title="Film growth (dashed: closed form)",
    legend="upper left",
    ax=fig[0],
)
plots.line(
    y={f"{c} °C": (a["times"], 100 * a["capacity"] / q0) for c, a in ageing.items()},
    reference={
        "cycling only": (
            ageing[25]["times"],
            100 * cycling.remaining_capacity(365.25 * ageing[25]["times"]) / q0,
        )
    },
    hlines={"end of life": 80.0},
    xlabel="time [years], one cycle per day",
    ylabel="capacity [% of initial]",
    title="Capacity fade",
    legend="lower left",
    ax=fig[1],
)
plots.line(
    y={
        f"after {age} years": (5.0 * d["times"] / 3600, d["voltage"])
        for age, d in discharges.items()
    },
    xlabel="discharged capacity [Ah]",
    ylabel="terminal voltage [V]",
    title="5 A discharge of the cell aged at 25 °C",
    legend="lower left",
    ax=fig[2],
)
plots.show()

# ## Going further
#
# Make the film growth depend on the state of charge through the negative-electrode potential, add
# lithium plating at low temperature, or fit the cycle fade to measured capacity checks.
