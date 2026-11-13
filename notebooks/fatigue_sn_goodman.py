"""OneSimulation example: stress-life fatigue with mean-stress corrections.
Requires the installed, activated OneSimulation runtime.

The fatigue lives and Miner damage sums returned by the OneSimulation fatigue
model are checked against the Basquin relation and the Goodman, Gerber and
Soderberg mean-stress corrections evaluated by hand.
"""


# # Stress-life fatigue with Goodman, Gerber and Soderberg corrections
#
# A component is loaded by a constant-amplitude stress cycle with amplitude
# $\sigma_a$ about a mean stress $\sigma_m$. Fully reversed stress-life (S-N) data
# describe the material: for the high-cycle regime the Basquin relation
#
# $$\sigma_a=\sigma_f'\,N^{b}\qquad\Longleftrightarrow\qquad
#   N=\left(\frac{\sigma_a}{\sigma_f'}\right)^{1/b}$$
#
# links the amplitude to the number of cycles to failure $N$. A tensile mean stress
# shortens the life. The classical corrections replace $\sigma_a$ by an equivalent
# fully reversed amplitude $\sigma_{a,\mathrm{eq}}$ that is then looked up on the
# fully reversed curve:
#
# $$\text{Goodman: }\ \frac{\sigma_a}{1-\sigma_m/\sigma_u},\qquad
#   \text{Gerber: }\ \frac{\sigma_a}{1-(\sigma_m/\sigma_u)^2},\qquad
#   \text{Soderberg: }\ \frac{\sigma_a}{1-\sigma_m/\sigma_y},$$
#
# with the ultimate tensile strength $\sigma_u$ and the yield strength $\sigma_y$.
# Damage from several load blocks accumulates linearly by Miner's rule,
# $D=\sum_i n_i/N_i$, and failure is predicted at $D=1$.
#
# OneSimulation stores the S-N curve as paired (cycles, amplitude) data, interpolates
# it linearly in log-log coordinates, applies the selected mean-stress correction on
# the tensile side (a compressive mean stress receives no life credit) and
# accumulates Miner damage cycle block by cycle block.
#
# **Reference.** The S-N table below is sampled exactly from a Basquin curve, so the
# log-log interpolation must reproduce the closed form $N=(\sigma_{a,\mathrm{eq}}/\sigma_f')^{1/b}$
# for every correction, the corrected amplitudes must match the three formulas, and
# the two-block Miner sum must equal $n_1/N_1+n_2/N_2$.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.physics.solid_mechanics.fatigue.models.fatigue_data import (
    FatigueCycle,
    FatigueState,
    SNDataset,
)
from onesim.physics.solid_mechanics.inelasticity.models.damage_laws import DamageLaw, DamageLawKind
from onesim.physics.solid_mechanics.inelasticity.models.damage_provider import (
    StructuralDamageFractureFatigueProvider,
)

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# A medium-strength steel: fatigue strength coefficient $\sigma_f'=900$ MPa, Basquin
# exponent $b=-0.10$, ultimate tensile strength 600 MPa, yield strength 450 MPa.
# The load cycle has an amplitude of 200 MPa about a tensile mean of 150 MPa.

MPa = 1.0e6
fatigue_strength_coefficient = 900.0 * MPa   # sigma_f'
basquin_exponent = -0.10                     # b
ultimate_strength = 600.0 * MPa              # sigma_u
yield_strength = 450.0 * MPa                 # sigma_y

stress_amplitude = 200.0 * MPa               # sigma_a
mean_stress = 150.0 * MPa                    # sigma_m (tensile)


def basquin_life(amplitude):
    return (amplitude / fatigue_strength_coefficient) ** (1.0 / basquin_exponent)


# The S-N table: five points taken exactly from the Basquin curve between 10^3 and 10^7 cycles.
table_cycles = tuple(10.0**k for k in range(3, 8))
table_amplitudes = tuple(fatigue_strength_coefficient * N**basquin_exponent for N in table_cycles)
for N, amplitude in zip(table_cycles, table_amplitudes):
    print(f"S-N point: N = {N:8.0f} cycles,  sigma_a = {amplitude/MPa:6.1f} MPa")


# ## Model
#
# One `SNDataset` per mean-stress convention shares the same fully reversed curve.
# The fatigue provider accumulates Miner damage from `FatigueCycle` blocks; a
# `FatigueState` carries the damage sum and the accumulated cycle count.

corrections = {
    "none": None,
    "Goodman": "goodman",
    "Gerber": "gerber",
    "Soderberg": "soderberg",
}
datasets = {
    label: SNDataset(
        dataset_id=f"steel-sn-{label.lower()}",
        cycles_to_failure=table_cycles,
        stress_amplitudes=table_amplitudes,
        mean_stress_correction=method,
        unit="Pa",
        ultimate_tensile_strength=ultimate_strength,
        yield_strength=yield_strength,
    )
    for label, method in corrections.items()
}
provider = StructuralDamageFractureFatigueProvider(
    DamageLaw("miner", DamageLawKind.SMEARED_CRACKING, critical_history=1.0)
)

applied_cycles = 1000.0
cycle = FatigueCycle("service", stress_amplitude, 0.0, mean_stress, count=applied_cycles)


# ## Comparison with the closed-form corrections
#
# For each convention the equivalent amplitude returned by the dataset is compared
# with the formula, and the life implied by the Miner damage of one block,
# $N=n/D$, is compared with the Basquin life of the corrected amplitude.

expected_amplitude = {
    "none": stress_amplitude,
    "Goodman": stress_amplitude / (1.0 - mean_stress / ultimate_strength),
    "Gerber": stress_amplitude / (1.0 - (mean_stress / ultimate_strength) ** 2),
    "Soderberg": stress_amplitude / (1.0 - mean_stress / yield_strength),
}

lives = {}
for label, data in datasets.items():
    corrected = data.corrected_amplitude(cycle)
    state = provider.update_fatigue(FatigueState(), cycle, sn_data=data)
    life_numeric = applied_cycles / state.damage
    life_reference = basquin_life(expected_amplitude[label])
    lives[label] = (corrected, life_numeric, life_reference)
    amplitude_error = abs(corrected - expected_amplitude[label]) / expected_amplitude[label]
    life_error = abs(life_numeric - life_reference) / life_reference
    print(f"{label:9s}: sigma_a,eq = {corrected/MPa:6.1f} MPa (formula {expected_amplitude[label]/MPa:6.1f} MPa, "
          f"error {amplitude_error:.1e}),  N = {life_numeric:.4e} (Basquin {life_reference:.4e}, error {life_error:.1e})")
    assert amplitude_error < 1e-12, f"{label} corrected amplitude deviates from the closed form."
    assert life_error < 1e-9, f"{label} life deviates from the Basquin value of the corrected amplitude."
    assert state.accumulated_cycles == applied_cycles

life_none, life_goodman, life_gerber, life_soderberg = (lives[k][1] for k in ("none", "Goodman", "Gerber", "Soderberg"))
assert life_soderberg < life_goodman < life_gerber < life_none, \
    "Soderberg must be the most conservative, Gerber the least, and all below the uncorrected life."
print(f"Life reduction by the tensile mean: Goodman {life_none/life_goodman:.1f}x, "
      f"Gerber {life_none/life_gerber:.1f}x, Soderberg {life_none/life_soderberg:.1f}x")


# ### Compressive mean stress
#
# A compressive mean of the same magnitude receives no life credit: all three
# corrections return the uncorrected amplitude and the fully reversed life.

compressive_cycle = FatigueCycle("compressive", stress_amplitude, 0.0, -mean_stress, count=applied_cycles)
for label, data in datasets.items():
    state = provider.update_fatigue(FatigueState(), compressive_cycle, sn_data=data)
    assert abs(data.corrected_amplitude(compressive_cycle) - stress_amplitude) < 1e-9 * MPa
    assert abs(applied_cycles / state.damage - life_none) / life_none < 1e-9
print(f"Compressive mean of {-mean_stress/MPa:.0f} MPa: all conventions give N = {life_none:.4e} (no credit)")


# ### Two-block loading and Miner's rule
#
# A second block with a larger amplitude follows the first. The damage after both
# blocks must equal $n_1/N_1+n_2/N_2$ with the Goodman-corrected lives, and the
# remaining life is read off from the damage still available.

second_amplitude = 260.0 * MPa
second_cycles = 500.0
goodman = datasets["Goodman"]
block_1 = FatigueCycle("block-1", stress_amplitude, 0.0, mean_stress, count=applied_cycles)
block_2 = FatigueCycle("block-2", second_amplitude, 0.0, mean_stress, count=second_cycles)
state = provider.update_fatigue(FatigueState(), block_1, sn_data=goodman)
state = provider.update_fatigue(state, block_2, sn_data=goodman)

life_1 = basquin_life(stress_amplitude / (1.0 - mean_stress / ultimate_strength))
life_2 = basquin_life(second_amplitude / (1.0 - mean_stress / ultimate_strength))
miner_reference = applied_cycles / life_1 + second_cycles / life_2
miner_error = abs(state.damage - miner_reference) / miner_reference
print(f"Two-block Miner damage: model {state.damage:.6e},  reference {miner_reference:.6e} (error {miner_error:.1e})")
print(f"Accumulated cycles: {state.accumulated_cycles:.0f};  repeats of the block pair to failure: {1.0/state.damage:.1f}")
assert miner_error < 1e-9, "Miner damage of two blocks deviates from n1/N1 + n2/N2."
assert state.accumulated_cycles == applied_cycles + second_cycles

# A mean stress at or above the ultimate strength is static failure, not fatigue: the
# model rejects it instead of extrapolating the correction.
static_cycle = FatigueCycle("static", stress_amplitude, 0.0, ultimate_strength)
try:
    goodman.corrected_amplitude(static_cycle)
except ValueError as error:
    print(f"Mean stress at sigma_u rejected: {error}")
else:
    raise AssertionError("A mean stress at the ultimate strength must be rejected.")


# ## Figure
#
# Left: the S-N curve with the equivalent amplitudes and lives of the four
# conventions. Right: the Haigh diagram with the Goodman line, Gerber parabola and
# Soderberg line through the fully reversed amplitude that gives the uncorrected life,
# and the applied cycle.

cream = "#fbf5eb"
colors = {"none": "#214D43", "Goodman": "#A85539", "Gerber": "#1f5c73", "Soderberg": "#8a6d1f"}
fig, (ax_sn, ax_haigh) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
for ax in (ax_sn, ax_haigh):
    ax.set_facecolor("#fffaf2")
    ax.grid(alpha=0.25, which="both")

N_fine = np.logspace(3, 7, 200)
ax_sn.plot(N_fine, fatigue_strength_coefficient * N_fine**basquin_exponent / MPa, color="#333", lw=1.6,
           label="Basquin curve (fully reversed)")
ax_sn.plot(table_cycles, np.asarray(table_amplitudes) / MPa, "s", color="#333", ms=5, mfc="none",
           label="S-N table points")
for label, (corrected, life_numeric, _) in lives.items():
    ax_sn.plot(life_numeric, corrected / MPa, "o", color=colors[label], ms=7,
               label=f"{label}: N = {life_numeric:.2e}")
ax_sn.set(xscale="log", xlabel="Cycles to failure N", ylabel="Equivalent stress amplitude / MPa",
          title="S-N curve and equivalent fully reversed amplitudes")
ax_sn.legend(fontsize=8)

mean_fine = np.linspace(0.0, ultimate_strength, 200)
fully_reversed = stress_amplitude  # the amplitude whose fully reversed life is the uncorrected one
ax_haigh.plot(mean_fine / MPa, fully_reversed * (1.0 - mean_fine / ultimate_strength) / MPa,
              color=colors["Goodman"], lw=1.8, label="Goodman line")
ax_haigh.plot(mean_fine / MPa, fully_reversed * (1.0 - (mean_fine / ultimate_strength) ** 2) / MPa,
              color=colors["Gerber"], lw=1.8, label="Gerber parabola")
mean_sod = np.linspace(0.0, yield_strength, 200)
ax_haigh.plot(mean_sod / MPa, fully_reversed * (1.0 - mean_sod / yield_strength) / MPa,
              color=colors["Soderberg"], lw=1.8, label="Soderberg line")
ax_haigh.plot(mean_stress / MPa, stress_amplitude / MPa, "o", color="#A85539", ms=8,
              label="Applied cycle (σ_m, σ_a)")
ax_haigh.axvline(yield_strength / MPa, color="#999", lw=1, ls=":")
ax_haigh.axvline(ultimate_strength / MPa, color="#999", lw=1, ls="--")
ax_haigh.set(xlabel="Mean stress σ_m / MPa", ylabel="Stress amplitude σ_a / MPa",
             title=f"Haigh diagram for the N = {life_none:.1e} cycle line", ylim=(0, 1.3 * stress_amplitude / MPa))
ax_haigh.legend(fontsize=8)
plt.show()


# ## Going further
#
# Replace the Basquin table by measured S-N data with an endurance limit and watch
# amplitudes below the last point receive the run-out life. Use an `ENDataset` for a
# strain-life (Coffin-Manson) analysis in the low-cycle regime, where the life is
# counted in reversals. Feed the cycle blocks from a rainflow count of a measured
# load history, or couple the fatigue provider to a stress analysis so that the
# amplitude and mean at the critical location come from the finite element solution.
