"""OneSimulation example: Butler-Volmer kinetics and the Tafel line of a single electrode.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form Butler-Volmer
current-overpotential relation, its Tafel slope and its charge-transfer resistance.
"""


# # Butler-Volmer kinetics and the Tafel line of a single electrode
#
# The rate of a charge-transfer reaction at an electrode depends exponentially on the
# overpotential $\eta=\phi-E_\mathrm{eq}$, the distance of the electrode potential from
# its equilibrium value. The Butler-Volmer law combines the anodic and the cathodic
# branch,
#
# $$i_\mathrm{F}(\eta)=i_0\left[\exp\!\left(\frac{\alpha_a F\eta}{RT}\right)
# -\exp\!\left(-\frac{\alpha_c F\eta}{RT}\right)\right],$$
#
# with the exchange current density $i_0$ and the transfer coefficients
# $\alpha_a+\alpha_c=1$ for a one-electron step. The electrode surface also stores
# charge in its double layer (capacitance $C_\mathrm{dl}$ per unit area), so when a
# current density $i_\mathrm{app}$ is forced through the interface the potential obeys
#
# $$C_\mathrm{dl}\,\frac{d\phi}{dt}=i_\mathrm{app}-i_\mathrm{F}(\phi-E_\mathrm{eq}).$$
#
# OneSimulation treats the electrode as a zero-dimensional (lumped) model with the
# potential and the species concentrations as states and integrates it with the
# classical fourth-order Runge-Kutta method. We sweep the applied current density
# over three decades, let the potential settle at each level (a galvanostatic
# staircase) and so trace the current-overpotential curve point by point.
#
# **Reference.** At steady state $i_\mathrm{F}=i_\mathrm{app}$. For the symmetric case
# $\alpha_a=\alpha_c=1/2$ the law reads $i_\mathrm{F}=2i_0\sinh(F\eta/2RT)$ and inverts
# in closed form,
#
# $$\eta_\infty=\frac{2RT}{F}\,\operatorname{arsinh}\frac{i_\mathrm{app}}{2i_0}.$$
#
# Two limits are classical: for $|\eta|\ll RT/F$ the interface is a linear resistor
# with the charge-transfer resistance $R_\mathrm{ct}=RT/(i_0F)$, and for
# $\eta\gg RT/F$ the cathodic branch is negligible and $\eta$ grows linearly with
# $\log_{10}i_\mathrm{F}$ at the Tafel slope $b=2.303\,RT/(\alpha_a F)$, about
# 118 mV per decade at 25 °C for $\alpha_a=1/2$. In the linear regime the potential
# settles exponentially with the time constant $\tau=R_\mathrm{ct}C_\mathrm{dl}$.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.geometry import LumpedDomain
from onesim.meshing import LumpedMesh
from onesim.numerics import Numerics
from onesim.numerics.discretization import LumpedParameters
from onesim.numerics.time import ClassicalRungeKutta4
from onesim.physics.chemistry import ElectrochemicalReaction, LumpedElectrode
from onesim.results.storage import MemoryOutputSink, Output
from onesim.simulation import Problem, Simulation
from onesim.simulation.initial_conditions import InitialValue

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model
#
# The parameters can be changed. The closed-form inversion below assumes the
# symmetric transfer coefficient $\alpha=1/2$ and a one-electron reaction; the
# Tafel-slope and charge-transfer-resistance checks hold for any $\alpha$. The
# values describe a moderately fast redox couple at room temperature. The
# equilibrium potential is held fixed and the electrolyte volume is large, so the
# concentration of the dissolved species barely changes during the short settling
# transients and $\eta$ equals the electrode potential $\phi$.

FARADAY = 96485.33212           # C/mol
GAS_CONSTANT = 8.314462618      # J/(mol K)

temperature = 298.15            # K
exchange_current_density = 1.0  # A/m², i_0
transfer_coefficient = 0.5      # alpha (alpha_a = alpha, alpha_c = 1 - alpha)
double_layer_capacitance = 0.2  # F/m², C_dl
electrode_area = 1.0e-4         # m²
electrolyte_volume = 1.0e-3     # m³
initial_concentration = 1000.0  # mol/m³, dissolved species
applied_current_densities = exchange_current_density * np.logspace(-1.5, 2.0, 15)  # A/m²

thermal_voltage = GAS_CONSTANT * temperature / FARADAY            # RT/F
charge_transfer_resistance = thermal_voltage / exchange_current_density  # Ohm m²
tafel_slope = np.log(10.0) * thermal_voltage / transfer_coefficient      # V per decade
settling_time = charge_transfer_resistance * double_layer_capacitance    # R_ct C_dl
print(f"RT/F = {thermal_voltage * 1e3:.3f} mV,  R_ct = {charge_transfer_resistance * 1e3:.2f} mOhm m²")
print(f"Tafel slope = {tafel_slope * 1e3:.1f} mV/decade,  R_ct C_dl = {settling_time * 1e3:.3f} ms")


def butler_volmer(eta):
    return exchange_current_density * (
        np.exp(transfer_coefficient * eta / thermal_voltage)
        - np.exp(-(1.0 - transfer_coefficient) * eta / thermal_voltage)
    )


reaction = ElectrochemicalReaction(
    "redox",
    {"Ox": -1.0, "Red": 1.0},
    exchange_current_density,
    lambda concentrations, temperature: 0.0,   # fixed equilibrium potential: eta = phi
    electron_number=1.0,
    transfer_coefficient=transfer_coefficient,
)


def galvanostatic_step(applied_current_density, end_time, time_step):
    """Hold a constant current density and return the potential and faradaic current traces."""
    domain = LumpedDomain()
    electrode = LumpedElectrode(
        domain=domain.system,
        reactions=(reaction,),
        area=electrode_area,
        volume=electrolyte_volume,
        double_layer_capacitance=double_layer_capacitance,
        applied_current_density=applied_current_density,
        temperature=temperature,
    )
    simulation = Simulation(
        problem=Problem(
            name="Galvanostatic step on a Butler-Volmer electrode",
            geometry=domain,
            physics=(electrode,),
            initial_conditions=(
                InitialValue(electrode.potential, 0.0),
                InitialValue(electrode.concentration("Ox"), initial_concentration),
                InitialValue(electrode.concentration("Red"), initial_concentration),
            ),
        ),
        mesh=LumpedMesh(domain),
        numerics=Numerics(
            spatial=LumpedParameters(),
            time_integrator=ClassicalRungeKutta4(),
        ),
    )
    simulation.validate().require_valid()
    sink = MemoryOutputSink()
    simulation.run(
        start_time=0.0,
        end_time=end_time,
        time_step=time_step,
        output=Output(
            fields=(electrode.potential, electrode.faradaic_current_density),
            interval=time_step,
            sink=sink,
        ),
    )

    def series(field):
        return np.asarray([
            float(np.asarray(s.fields[field.field_id].values).reshape(-1)[0])
            for s in sink.snapshots
        ])

    times = np.asarray([float(s.time) for s in sink.snapshots])
    return times, series(electrode.potential), series(electrode.faradaic_current_density)


# ## Galvanostatic staircase
#
# At small currents the potential settles with the linear time constant
# $\tau=R_\mathrm{ct}C_\mathrm{dl}$ and each level is held for $40\,\tau$. At large
# anodic overpotentials the interface stiffens: its differential conductance grows to
# $\alpha_a F i/(RT)$, so the local time constant shrinks to about $2\tau\,i_0/i$,
# while the initial charging ramp of the double layer at the rate $i_\mathrm{app}/C_\mathrm{dl}$
# takes at most about $10\,\tau\,i_0/i$. Hold time and explicit time step are therefore
# both divided by the ratio $i_\mathrm{app}/i_0$ (floored at one), which keeps 200 steps
# per level and the explicit scheme well inside its stability limit.

steady_potentials = []
steady_currents = []
full_curve_error = 0.0
for i_app in applied_current_densities:
    stiffness = max(1.0, i_app / exchange_current_density)   # differential conductance ratio
    times, potentials, faradaic = galvanostatic_step(
        i_app, end_time=40.0 * settling_time / stiffness, time_step=0.2 * settling_time / stiffness
    )
    # The reported faradaic current density is the Butler-Volmer law evaluated at the
    # instantaneous potential, during the transient as well as at steady state.
    full_curve_error = max(full_curve_error, float(np.max(np.abs(faradaic - butler_volmer(potentials)))))
    steady_potentials.append(potentials[-1])
    steady_currents.append(faradaic[-1])
    # Under anodic load the potential rises and then rests at the settled value.
    assert np.all(np.diff(potentials) >= -1e-12), "The potential must not fall under anodic load."
    assert potentials[-1] > potentials[0]
steady_potentials = np.asarray(steady_potentials)
steady_currents = np.asarray(steady_currents)
print(f"Maximum deviation of the reported current from i_F(phi): {full_curve_error:.3e} A/m²")
assert full_curve_error < 1e-9 * np.max(applied_current_densities)


# ## Comparison with the closed-form solution
#
# At every level the settled potential is compared with the arsinh inversion and the
# settled faradaic current with the applied current. The Tafel slope is read off the
# three highest levels and the charge-transfer resistance off the three lowest. A
# passing check shows that this numerical example agrees with the Butler-Volmer
# law; it is not a measurement of a real electrode.

reference_potentials = 2.0 * thermal_voltage * np.arcsinh(applied_current_densities / (2.0 * exchange_current_density))
potential_error = np.abs(steady_potentials - reference_potentials)
current_error = np.abs(steady_currents / applied_current_densities - 1.0)
print("Applied current, settled overpotential, closed form (mV):")
for i_app, eta, ref in zip(applied_current_densities, steady_potentials, reference_potentials):
    print(f"  i = {i_app:8.3f} A/m²  eta = {eta * 1e3:8.3f} mV  reference = {ref * 1e3:8.3f} mV")
print(f"Maximum overpotential error: {np.max(potential_error) * 1e6:.3f} µV")
print(f"Maximum relative current error at steady state: {np.max(current_error):.2e}")
assert np.max(potential_error) < 1e-6, "Settled overpotential deviates from the arsinh inversion."
assert np.max(current_error) < 1e-6, "Settled faradaic current must equal the applied current."

# Tafel slope from the highest three levels, where i_app >= 30 i_0 and the cathodic branch is negligible.
tafel = applied_current_densities >= 30.0 * exchange_current_density
measured_tafel_slope = np.polyfit(np.log10(steady_currents[tafel]), steady_potentials[tafel], 1)[0]
# Charge-transfer resistance from the lowest three levels, where eta < 0.1 RT/F.
linear = applied_current_densities <= 0.1 * exchange_current_density
measured_resistance = np.polyfit(steady_currents[linear], steady_potentials[linear], 1)[0]
print(f"Tafel slope:                measured {measured_tafel_slope * 1e3:.2f} mV/decade, exact {tafel_slope * 1e3:.2f} mV/decade")
print(f"Charge-transfer resistance: measured {measured_resistance * 1e3:.3f} mOhm m², exact {charge_transfer_resistance * 1e3:.3f} mOhm m²")
assert abs(measured_tafel_slope / tafel_slope - 1) < 1e-2
assert abs(measured_resistance / charge_transfer_resistance - 1) < 1e-3

# Small-signal step response: the potential settles exponentially with tau = R_ct C_dl.
# At i_app = 0.01 i_0 the overpotential is 0.01 RT/F and the cubic Butler-Volmer term
# shifts the settled value by only (eta/2V_T)^2/6, about 4e-6 relative.
small_current = 0.01 * exchange_current_density
step_times, step_potentials, _ = galvanostatic_step(
    small_current, end_time=5.0 * settling_time, time_step=0.02 * settling_time
)
step_reference = charge_transfer_resistance * small_current * (1.0 - np.exp(-step_times / settling_time))
step_error = float(np.max(np.abs(step_potentials - step_reference)))
print(f"Small-signal step: maximum deviation from the R_ct C_dl exponential {step_error * 1e9:.3f} nV "
      f"(final overpotential {step_potentials[-1] * 1e6:.3f} µV)")
assert step_error < 1e-4 * charge_transfer_resistance * small_current
measured_step_tau = -1.0 / np.polyfit(
    step_times[1:-1], np.log(1.0 - step_potentials[1:-1] / (charge_transfer_resistance * small_current)), 1
)[0]
print(f"Small-signal time constant: measured {measured_step_tau * 1e3:.4f} ms, exact {settling_time * 1e3:.4f} ms")
assert abs(measured_step_tau / settling_time - 1) < 1e-3


# ## Figure

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
fig.patch.set_facecolor("#fbf5eb")
for ax in axes:
    ax.set_facecolor("#fffaf2")
    ax.grid(alpha=0.25)
dense_eta = np.linspace(-0.3, 0.3, 600)
axes[0].plot(dense_eta * 1e3, butler_volmer(dense_eta), color="#6b442f", lw=1.8, label="Butler-Volmer law")
axes[0].plot(steady_potentials * 1e3, steady_currents, "o", color="#c56838", ms=5, label="OneSimulation steady states")
axes[0].plot(dense_eta * 1e3, dense_eta / charge_transfer_resistance, "--", color="#9a8f7d", lw=1.0,
             label=r"Linear regime $\eta/R_\mathrm{ct}$")
axes[0].set(xlabel="Overpotential / mV", ylabel="Current density / A m$^{-2}$",
            title="Current-overpotential curve", ylim=(-30, 110))
axes[0].legend(loc="upper left")
positive = dense_eta > 0.0
axes[1].semilogx(butler_volmer(dense_eta[positive]), dense_eta[positive] * 1e3, color="#6b442f", lw=1.8,
                 label="Butler-Volmer law")
axes[1].semilogx(steady_currents, steady_potentials * 1e3, "o", color="#c56838", ms=5,
                 label="OneSimulation steady states")
tafel_line = tafel_slope * np.log10(steady_currents / exchange_current_density)
axes[1].semilogx(steady_currents, tafel_line * 1e3, "--", color="#9a8f7d", lw=1.0,
                 label=rf"Tafel line, {tafel_slope * 1e3:.0f} mV/decade")
axes[1].set(xlabel="Current density / A m$^{-2}$", ylabel="Overpotential / mV", title="Tafel plot")
axes[1].legend(loc="upper left")
plt.show()


# ## Going further
#
# Change the transfer coefficient to $\alpha=0.3$; the arsinh inversion no longer
# applies, but the Tafel slope becomes $2.303\,RT/(0.3F)$ and the charge-transfer
# resistance is unchanged. Apply negative current densities to trace the cathodic
# branch, whose Tafel slope is set by $\alpha_c=1-\alpha$. Making the equilibrium
# potential depend on the concentrations through the Nernst equation and shrinking
# the electrolyte volume turns the staircase into a coupled kinetics-and-depletion
# problem in which the settled potential drifts with time.
