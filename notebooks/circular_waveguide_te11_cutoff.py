"""OneSimulation example: TE11 cutoff and single-mode band of a circular waveguide.
Requires the installed, activated OneSimulation runtime.

The numerically computed TE11 and TM01 port eigenmodes and the driven 3D field
are checked against the Bessel-function cutoff frequencies, the TE11 mode
profile and the standing-wave pattern of a shorted guide.
"""


# # TE11 cutoff of a circular waveguide
#
# An air-filled circular waveguide of radius $a$ is fed through a **waveguide
# port** on its bottom disk ($z=0$) and terminated by a perfectly conducting
# short at $z=\ell$; the cylindrical wall is a perfect electric conductor. The
# port mode is not prescribed analytically: OneSimulation solves the transverse
# eigenvalue problem on the meshed port cross-section and uses its first TE
# eigenmode (the fundamental TE11 mode) as the excitation.
#
# The frequency-domain Maxwell equations are solved with Nédélec (H(curl))
# elements of order 2 on a tetrahedral mesh:
#
# $$\nabla\times\big(\mu_r^{-1}\nabla\times\mathbf{E}\big)-k_0^2\,\varepsilon_r\,\mathbf{E}=0,
#   \qquad k_0=\omega/c.$$
#
# **Reference.** The cutoff wavenumbers of a circular guide are zeros of Bessel
# functions, $k_c=p'_{nm}/a$ for TE$_{nm}$ (zeros of $J_n'$) and $k_c=p_{nm}/a$
# for TM$_{nm}$ (zeros of $J_n$):
#
# $$f_c=\frac{c\,k_c}{2\pi},\qquad p'_{11}=1.8412,\quad p_{01}=2.4048,\quad p'_{21}=3.0542 .$$
#
# TE11 is the fundamental mode, TM01 the first higher mode, and the guide is
# single-moded between the two cutoffs. The TE11 electric field through the
# centre of the guide follows $2J_1(k_c r)/(k_c r)$ along the polarisation
# direction and $2J_1'(k_c r)$ perpendicular to it. With
# $\beta=\sqrt{k_0^2-k_c^2}$ the shorted guide carries the standing wave
# $|\mathbf{E}_t(z)|\propto|\sin\beta(\ell-z)|$ on its axis.

from importlib.metadata import version
from math import pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, jvp

from onesim.cad.d3 import Cylinder
from onesim.materials.electromagnetic import IsotropicElectromagneticMaterial
from onesim.materials.properties.units.constants import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from onesim.meshing import TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import ComplexSparseDirect
from onesim.physics.electromagnetics import (
    FrequencyDomainMaxwell, NumericalWaveguideMode, PerfectElectricConductor, WaveguidePort,
)
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# With a radius of 20 mm the TE11 mode cuts off at about 4.39 GHz, TM01 at
# 5.74 GHz and TE21 at 7.29 GHz. The guide is driven at 5.4 GHz, inside the
# single-mode band, and a second solve at 6.5 GHz (between the TM01 and TE21
# cutoffs) binds the TM01 port mode to verify the mode ordering numerically.

a = 0.020               # m, guide radius
length = 0.060          # m, guide length to the short (z direction)
frequency = 5.4e9       # Hz, drive frequency in the single-mode band
frequency_tm = 6.5e9    # Hz, drive frequency for the TM01 check
p11_te, p01_tm, p21_te = 1.841183781, 2.404825558, 3.054236928
c0 = 1.0 / sqrt(VACUUM_PERMEABILITY * VACUUM_PERMITTIVITY)
kc_te11 = p11_te / a
f_c_te11 = c0 * kc_te11 / (2 * pi)
f_c_tm01 = c0 * p01_tm / (2 * pi * a)
f_c_te21 = c0 * p21_te / (2 * pi * a)
k0 = 2 * pi * frequency / c0
beta = sqrt(k0**2 - kc_te11**2)
print(f"f_c(TE11) = {f_c_te11 / 1e9:.4f} GHz,  f_c(TM01) = {f_c_tm01 / 1e9:.4f} GHz,  f_c(TE21) = {f_c_te21 / 1e9:.4f} GHz")
print(f"guide wavelength of TE11 at {frequency / 1e9:.1f} GHz: {2 * pi / beta * 1e3:.1f} mm")

guide = Cylinder(radius=a, length=length)
mesh = TetrahedralMesher(target_size=0.004).generate(guide)
print("Mesh:", mesh.cell_count, "tetrahedra")


def shorted_guide(drive_frequency, mode):
    model = FrequencyDomainMaxwell(
        domain=guide.volume, frequency=drive_frequency,
        material=IsotropicElectromagneticMaterial(),      # air: eps_r = mu_r = 1, lossless
        phasor_convention="exp(-i*omega*t)", amplitude_convention="peak",
    )
    problem = Problem(
        name="circular waveguide with a short",
        geometry=guide,
        physics=(model,),
        boundary_conditions=(
            # Port on the bottom disk (z = 0), excited with a numerically computed eigenmode.
            WaveguidePort(boundary=guide.boundaries.bottom, physics=model, mode=mode, incident_power=1.0),
            # The cylindrical wall and the short at z = length are perfect electric conductors.
            PerfectElectricConductor(boundary=guide.boundaries.all - guide.boundaries.bottom, physics=model),
        ),
    )
    simulation = Simulation(
        problem=problem,
        mesh=mesh,
        numerics=Numerics(spatial=FiniteElement(element=Nedelec(order=2)), linear_solver=ComplexSparseDirect()),
    )
    simulation.validate().require_valid()
    return model, simulation.solve_stationary()


model, result = shorted_guide(frequency, NumericalWaveguideMode("TE"))
model_tm, result_tm = shorted_guide(frequency_tm, NumericalWaveguideMode("TM"))


# ## Comparison with the Bessel-function solution
#
# Each bound port mode exposes its cutoff frequency and its transverse electric
# field; the driven TE11 solution is sampled along the guide axis.

_, te11 = result.operators[model.model_id].ports[0]
_, tm01 = result_tm.operators[model_tm.model_id].ports[0]
f_te11_num = float(te11.cutoff_frequency)
f_tm01_num = float(tm01.cutoff_frequency)
te11_error = abs(f_te11_num / f_c_te11 - 1.0)
tm01_error = abs(f_tm01_num / f_c_tm01 - 1.0)
print(f"TE11 cutoff: computed {f_te11_num / 1e9:.4f} GHz, reference {f_c_te11 / 1e9:.4f} GHz, deviation {100 * te11_error:.2f} %")
print(f"TM01 cutoff: computed {f_tm01_num / 1e9:.4f} GHz, reference {f_c_tm01 / 1e9:.4f} GHz, deviation {100 * tm01_error:.2f} %")

# TE11 is linearly polarised at the centre; the mesh picks the polarisation
# direction of the degenerate pair, so it is read off the computed mode.
centre_field = np.asarray(te11.electric_field(np.zeros((1, 3))))[0]
polarisation = centre_field / np.linalg.norm(centre_field)
perpendicular = np.array([-polarisation[1], polarisation[0], 0.0])
radii = np.linspace(-0.9 * a, 0.9 * a, 25)
along = np.asarray(te11.electric_field(np.outer(radii, polarisation))) @ polarisation
across = np.asarray(te11.electric_field(np.outer(radii, perpendicular))) @ polarisation
along, across = along / along[len(radii) // 2], across / across[len(radii) // 2]
x = kc_te11 * radii
shape_along_ref = np.where(np.abs(x) > 1e-12, 2 * jv(1, x) / np.where(np.abs(x) > 1e-12, x, 1.0), 1.0)
shape_across_ref = 2 * jvp(1, x)
shape_error = float(max(np.max(np.abs(along - shape_along_ref)), np.max(np.abs(across - shape_across_ref))))
print(f"TE11 mode profile vs 2 J1(x)/x and 2 J1'(x): max deviation {shape_error:.3f} (of centre value 1)")

# Standing wave along the axis of the shorted guide.
zs = np.linspace(0.02 * length, 0.98 * length, 25)
axis_points = np.column_stack((np.zeros_like(zs), np.zeros_like(zs), zs))
E_axis = np.asarray(result.field(model.electric_field).evaluate(axis_points))
transverse = np.sqrt(np.abs(E_axis[:, 0]) ** 2 + np.abs(E_axis[:, 1]) ** 2)
longitudinal = float(np.max(np.abs(E_axis[:, 2])) / transverse.max())
standing_ref = np.abs(np.sin(beta * (length - zs)))
standing_error = float(np.max(np.abs(transverse / transverse.max() - standing_ref / standing_ref.max())))
print(f"Standing wave |E_t(z)| vs |sin(beta (l - z))|: max deviation {standing_error:.3f} (of peak 1), |E_z|/|E_t| = {longitudinal:.3f}")

absorbed = result.field(model.absorbed_power_density).integrate()
print(f"Absorbed power in the lossless guide: {absorbed.value:.3e} {absorbed.unit} (incident 1 W)")

assert te11_error < 0.015, "TE11 cutoff deviates more than 1.5 % from c p'_11 / (2 pi a)."
assert tm01_error < 0.015, "TM01 cutoff deviates more than 1.5 % from c p_01 / (2 pi a)."
assert f_te11_num < frequency < f_tm01_num, "The TE11 drive frequency must lie in the computed single-mode band."
assert f_c_tm01 < frequency_tm < f_c_te21, "The TM01 check frequency must lie between the TM01 and TE21 cutoffs."
assert shape_error < 0.10, "TE11 mode profile deviates more than 10 % of its centre value from the Bessel forms."
assert standing_error < 0.04, "Standing-wave pattern deviates more than 4 % of its peak from the closed form (the discrete cutoff shifts beta by about 1 %)."
assert longitudinal < 0.05, "The TE11 field on the axis must be transverse."
assert abs(absorbed.value) < 1e-9, "A lossless shorted guide absorbs no power."


# ## Figure

cream = "#fbf5eb"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)

ax = axes[0]
ax.set_facecolor("#fffaf2")
fine = np.linspace(-a, a, 300)
xf = kc_te11 * fine
ax.plot(fine * 1e3, np.where(np.abs(xf) > 1e-12, 2 * jv(1, xf) / np.where(np.abs(xf) > 1e-12, xf, 1.0), 1.0),
        color="#39715e", lw=1.8, label="2 J1(k_c r) / (k_c r)  (along E)")
ax.plot(fine * 1e3, 2 * jvp(1, xf), color="#7a9a3a", lw=1.8, ls="--", label="2 J1'(k_c r)  (across E)")
ax.plot(radii * 1e3, along, "o", color="#b4572b", ms=4, label="OneSimulation port mode, along E")
ax.plot(radii * 1e3, across, "s", color="#b4572b", ms=4, mfc="none", label="OneSimulation port mode, across E")
ax.set(xlabel="Position through the centre / mm", ylabel="E / E(centre)",
       title=f"TE11 profile — cutoff {f_te11_num / 1e9:.3f} GHz (reference {f_c_te11 / 1e9:.3f} GHz)")
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.25)

ax = axes[1]
ax.set_facecolor("#fffaf2")
fine_z = np.linspace(0, length, 300)
ax.plot(fine_z * 1e3, np.abs(np.sin(beta * (length - fine_z))), color="#39715e", lw=1.8, label="|sin(beta (l - z))|")
ax.plot(zs * 1e3, transverse / transverse.max(), "o", color="#b4572b", ms=4, label="OneSimulation |E_t| on the axis")
ax.set(xlabel="Position z along the guide / mm", ylabel="|E_t| / max",
       title=f"Standing wave in the shorted guide at {frequency / 1e9:.1f} GHz")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Ask for `NumericalWaveguideMode("TE", order=3)` above 7.3 GHz to bind the TE21
# mode and compare it with $p'_{21}$, give the material a finite conductivity to
# switch the absorbed power on, or refine `target_size` to tighten the cutoff
# comparisons, which converge quadratically with the port mesh size.
