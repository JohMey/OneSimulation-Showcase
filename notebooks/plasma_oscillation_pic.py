"""OneSimulation example: Langmuir (plasma) oscillation in an electrostatic particle-in-cell run.
Requires the installed, activated OneSimulation runtime.

Displaced electrons oscillate in the electric field they create. The frequency measured in a
particle-in-cell simulation is compared with the plasma frequency and its Bohm-Gross thermal
correction.
"""

# # Plasma oscillation in a particle-in-cell simulation
#
# A uniform electron plasma of density $n_0$ sits in a neutralising background of immobile ions.
# If the electrons are displaced by a small sinusoidal amount, the space charge pulls them back
# and the plasma oscillates. The simulation follows the electrons as macro-particles on a periodic
# grid, so the oscillation comes from the self-consistent electric field.
#
# **Reference.** A cold plasma oscillates at the plasma frequency; a warm one with the thermal
# speed $v_{th}=\sqrt{k_BT_e/m_e}$ follows the Bohm-Gross relation:
#
# $$\omega_p=\sqrt{\frac{n_0 e^2}{\varepsilon_0 m_e}},\qquad
#   \omega^2=\omega_p^2+3k^2v_{th}^2=\omega_p^2\big(1+3\,k^2\lambda_D^2\big) .$$

import numpy as np
from onesim.physics.plasma import PeriodicParticleInCell, PICSpecies
from onesim.results import oscillation_fit
from onesim.visualization import plots, theme
from scipy.constants import e, epsilon_0, m_e
from scipy.special import erfinv

theme.set()

density, length, cells, particles = 1.0e16, 5.0e-3, 64, 64 * 400
plasma_frequency = np.sqrt(density * e**2 / (epsilon_0 * m_e))
k = 2 * np.pi / length
thermal_speed = lambda temperature: np.sqrt(e * temperature / m_e)
bohm_gross = lambda temperature: np.sqrt(
    plasma_frequency**2 + 3 * k**2 * thermal_speed(temperature) ** 2
)


# ## Simulation
#
# The electrons start as a travelling wave of displacement $0.01\,L\,\sin kx$, with a Maxwellian
# velocity spread for the warm cases. Every five time steps the Fourier amplitude of the electric
# field at the wavenumber $k$ is recorded; it rotates at the oscillation frequency.

plasma = PeriodicParticleInCell(
    lengths=(length,), cells=(cells,), background_charge_density=e * density
)
grid = np.arange(cells) * length / cells
start = (np.arange(particles) + 0.5) * length / particles
time_step = 1 / (50 * plasma_frequency)


def simulate(temperature):
    """Field amplitude and total energy over six oscillation periods for an electron temperature in eV."""
    velocities = np.zeros((particles, 3))
    velocities[:, 0] = -0.01 * length * bohm_gross(temperature) * np.cos(k * start)
    maxwellian = (
        np.sqrt(2)
        * thermal_speed(temperature)
        * erfinv(2 * (np.arange(particles) + 0.5) / particles - 1)
    )
    velocities[:, 0] += np.random.default_rng(7).permutation(maxwellian)
    electrons = PICSpecies(
        positions=(start + 0.01 * length * np.sin(k * start))[:, None],
        velocities=velocities,
        mass=m_e,
        charge=-e,
        weights=density * length / particles,
    )
    times, amplitude, energy = [], [], []
    for sample in range(int(6 * 2 * np.pi / bohm_gross(temperature) / (5 * time_step))):
        result = plasma.solve(species=(electrons,), time_step=time_step, steps=5)
        electrons = result.species[0]
        times.append(5 * time_step * (sample + 1))
        amplitude.append(np.sum(result.electric_field[:, 0] * np.exp(-1j * k * grid)))
        energy.append(result.kinetic_energy[-1] + result.field_energy[-1])
    return np.array(times), np.array(amplitude), np.array(energy)


runs = {temperature: simulate(temperature) for temperature in (0.0, 1.0, 3.0)}


# ## Results
#
# `oscillation_fit` reads the frequency off the rotating field amplitude.

measured = {
    temperature: abs(oscillation_fit(times, amplitude).frequency)
    for temperature, (times, amplitude, _) in runs.items()
}
print(
    "T_e [eV]   measured ω/ω_p   Bohm-Gross   deviation   largest change of the total energy"
)
for temperature, (times, amplitude, energy) in runs.items():
    print(
        f"{temperature:8.0f} {measured[temperature] / plasma_frequency:16.4f} {bohm_gross(temperature) / plasma_frequency:12.4f}"
        f" {100 * (measured[temperature] / bohm_gross(temperature) - 1):+10.2f} % {np.abs(energy / energy[0] - 1).max():14.1e}"
    )


# ## Figure
#
# The field amplitude over the first periods against a cosine at the Bohm-Gross frequency, and
# the measured frequencies on the dispersion relation.

debye = lambda temperature: np.sqrt(epsilon_0 * e * temperature / (density * e**2))
labels = {temperature: f"T_e = {temperature:.0f} eV" for temperature in runs}
fine = np.linspace(0, 2.5 * 2 * np.pi / plasma_frequency, 600)
kd = np.linspace(0, 0.2, 100)

fig = plots.figure(1, 2, size=(13, 4.8))
plots.line(
    points={
        labels[T]: (1e9 * times, (amplitude / amplitude[0]).real)
        for T, (times, amplitude, _) in runs.items()
    },
    reference={
        labels[T]: (1e9 * fine, np.cos(bohm_gross(T) * (fine - times[0])))
        for T, (times, _, _) in runs.items()
    },
    xlimits=(0, 1e9 * fine[-1]),
    xlabel="time [ns]",
    ylabel="field amplitude / initial amplitude",
    title="Simulation (markers) and Bohm-Gross frequency (dashed)",
    legend="upper right",
    ax=fig[0],
)
plots.line(
    kd,
    {"Bohm-Gross": np.sqrt(1 + 3 * kd**2)},
    points={
        "simulation": (
            [k * debye(T) for T in runs],
            [measured[T] / plasma_frequency for T in runs],
        )
    },
    hlines={"cold plasma": 1.0},
    xlabel="k λ_D",
    ylabel="ω / ω_p",
    title="Dispersion relation",
    legend="upper left",
    ax=fig[1],
)
plots.show()


# ## Going further
#
# Raise the temperature until $k\lambda_D$ exceeds 0.3 and Landau damping makes the oscillation
# decay; `oscillation_fit` returns the damping as `growth_rate`. Two counter-streaming electron
# beams turn the oscillation into the two-stream instability.
