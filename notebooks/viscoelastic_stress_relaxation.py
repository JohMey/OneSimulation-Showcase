"""OneSimulation example: stress relaxation of a viscoelastic bar (standard linear solid).
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form relaxation
function of the standard linear solid in uniaxial tension.
"""


# # Stress relaxation of a viscoelastic bar
#
# A polymer bar (length $L$ along $x$, square cross section) is stretched by a
# step strain $\varepsilon_0$ at $t=0$ and then held at constant length. The
# material is a **standard linear solid** (one Prony branch): the shear modulus
# relaxes from its glassy value $G_0=G_\infty+G_1$ to the rubbery value
# $G_\infty$ with the relaxation time $\tau$, while the bulk modulus $K$ stays
# elastic, as is typical for polymers,
#
# $$G(t)=G_\infty+G_1\,e^{-t/\tau},\qquad K=\text{const}.$$
#
# OneSimulation solves the 3D small-strain equilibrium on a hexahedral mesh
# (Lagrange order 2) and advances the viscous strain of the Prony branch at
# every quadrature point with an implicit (backward Euler) update.
#
# **Reference.** The stress state in the bar is homogeneous and uniaxial. By
# the elastic-viscoelastic correspondence principle the uniaxial relaxation
# modulus of a material with elastic bulk and standard-linear-solid shear
# response is again a standard linear solid,
#
# $$\sigma(t)=\varepsilon_0\left[E_\infty+(E_0-E_\infty)\,e^{-t/\tau_E}\right],\qquad
#   E_0=\frac{9KG_0}{3K+G_0},\quad E_\infty=\frac{9KG_\infty}{3K+G_\infty},\quad
#   \tau_E=\tau\,\frac{3K+G_0}{3K+G_\infty}.$$
#
# We verify the instantaneous modulus $E_0$, the long-time modulus $E_\infty$,
# the uniaxial relaxation time $\tau_E$ and the whole relaxation curve.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Box
from onesim.meshing import StructuredHexahedralMesher
from onesim.physics.solid_mechanics import SmallStrainSolid
from onesim.physics.solid_mechanics.inelasticity.models.constitutive_updates import StructuralAdvancedMaterialProvider

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model

length, width, height = 0.100, 0.010, 0.010     # m
K = 3.0e9                                       # Pa, elastic bulk modulus
G_inf, G_1, tau = 0.30e9, 0.70e9, 10.0          # Pa, Pa, s  rubbery shear modulus, relaxing branch, relaxation time
strain_0 = 5e-3                                 # -, step strain held constant

G_0 = G_inf + G_1
E_0 = 9 * K * G_0 / (3 * K + G_0)               # instantaneous (glassy) Young's modulus
nu_0 = (3 * K - 2 * G_0) / (2 * (3 * K + G_0))  # instantaneous Poisson ratio
E_inf = 9 * K * G_inf / (3 * K + G_inf)         # long-time (rubbery) Young's modulus
tau_E = tau * (3 * K + G_0) / (3 * K + G_inf)   # uniaxial relaxation time
print(f"E_0 = {E_0/1e9:.3f} GPa,  E_inf = {E_inf/1e9:.3f} GPa,  nu_0 = {nu_0:.3f},  tau_E = {tau_E:.3f} s")

# The Prony law acts on the strain deviator, s = 2 G e, so the moduli of the
# parallel spring and of the Maxwell branch are entered as 2 G.
material = StructuralAdvancedMaterialProvider(
    viscoelastic_kind="prony",
    equilibrium_modulus=2 * G_inf,
    prony_terms=((2 * G_1, tau),),
)
model = SmallStrainSolid(youngs_modulus=E_0, poisson_ratio=nu_0, kinematics="volume", material=material)

geometry = Box(length=length, width=width, height=height)
mesh = StructuredHexahedralMesher(cells=(8, 2, 2)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")

# Time grid: t = 0 is the instant after the step; then 5 relaxation times.
end_time = 5.0 * tau_E
times = np.concatenate(([0.0], np.linspace(0.0, end_time, 121)[1:]))

# Boundary conditions: axial clamp and step displacement, two symmetry-type
# supports against rigid-body motion (lateral contraction stays free), and
# traction-free remaining faces.
result = model.solve(
    mesh=mesh,
    times=times,
    order=2,
    displacements=(
        (geometry.boundaries.x_min, "x", 0.0),
        (geometry.boundaries.x_max, "x", strain_0 * length),
        (geometry.boundaries.y_min, "y", 0.0),
        (geometry.boundaries.z_min, "z", 0.0),
    ),
    tractions=(
        (geometry.boundaries.y_max, (0.0, 0.0, 0.0)),
        (geometry.boundaries.z_max, (0.0, 0.0, 0.0)),
    ),
)


# ## Comparison with the standard-linear-solid relaxation

stress_reference = strain_0 * (E_inf + (E_0 - E_inf) * np.exp(-times / tau_E))

sigma_xx = result.stress[:, :, 0, 0]
stress_fem = sigma_xx.mean(axis=1)
stress_spread = np.ptp(sigma_xx, axis=1)
relaxation_modulus = stress_fem / strain_0

# Instantaneous and long-time moduli (the latter from the last stored time,
# where the exponential term has decayed to e^-5).
E_0_fem = relaxation_modulus[0]
E_inf_fem = (stress_fem[-1] - strain_0 * (E_0 - E_inf) * np.exp(-end_time / tau_E)) / strain_0

# Relaxation time from the log-linear decay of the relaxing part of the stress.
window = (times > 0.0) & (times < 3.0 * tau_E)
decay = (stress_fem[window] - strain_0 * E_inf) / (strain_0 * (E_0 - E_inf))
tau_E_fem = -1.0 / np.polyfit(times[window], np.log(decay), 1)[0]

# Lateral strain: the Poisson ratio rises from nu_0 towards the rubbery value.
face_y = np.isclose(result.coordinates[:, 1], width)
poisson_fem = -result.displacement[:, face_y, 1].mean(axis=1) / width / strain_0
nu_inf = (3 * K - 2 * G_inf) / (2 * (3 * K + G_inf))

rel_curve = np.max(np.abs(stress_fem - stress_reference)) / (strain_0 * E_0)
rel_E0 = abs(E_0_fem - E_0) / E_0
rel_Einf = abs(E_inf_fem - E_inf) / E_inf
rel_tau = abs(tau_E_fem - tau_E) / tau_E

print(f"Relaxation curve:     max deviation {100*rel_curve:.3f} % of the instantaneous stress")
print(f"Homogeneity:          max stress spread over the bar {stress_spread.max():.2e} Pa")
print(f"Instantaneous modulus: FEM {E_0_fem/1e9:.4f} GPa,  theory {E_0/1e9:.4f} GPa  ({100*rel_E0:.4f} %)")
print(f"Long-time modulus:     FEM {E_inf_fem/1e9:.4f} GPa,  theory {E_inf/1e9:.4f} GPa  ({100*rel_Einf:.3f} %)")
print(f"Relaxation time:       FEM {tau_E_fem:.3f} s,  theory {tau_E:.3f} s  ({100*rel_tau:.3f} %)")
print(f"Poisson ratio:         t=0 {poisson_fem[0]:.4f} (theory {nu_0:.4f}),  t=5 tau_E {poisson_fem[-1]:.4f} (rubbery limit {nu_inf:.4f})")

assert np.all(np.diff(stress_fem) < 0.0), "The stress must relax monotonically at constant strain."
assert rel_E0 < 1e-6, "Instantaneous modulus deviates from 9 K G_0 / (3 K + G_0)."
assert rel_Einf < 5e-3, "Long-time modulus deviates more than 0.5 % from 9 K G_inf / (3 K + G_inf)."
assert rel_curve < 1e-2, "Relaxation curve deviates more than 1 % of the instantaneous stress."
assert rel_tau < 3e-2, "Fitted uniaxial relaxation time deviates more than 3 % from tau (3K+G_0)/(3K+G_inf)."
assert stress_spread.max() < 1e-6 * strain_0 * E_0, "The stress state in the bar is not homogeneous."
assert abs(poisson_fem[0] - nu_0) < 1e-4 and poisson_fem[-1] > poisson_fem[0], \
    "The Poisson ratio must start at nu_0 and rise during relaxation."


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
fine = np.linspace(0.0, end_time, 400)
ax.plot(fine, (E_inf + (E_0 - E_inf) * np.exp(-fine / tau_E)) / 1e9, color="#214D43", lw=1.8,
        label="theory  E(t) = E_inf + (E_0 - E_inf) exp(-t / tau_E)")
ax.plot(times[::3], relaxation_modulus[::3] / 1e9, "o", color="#A85539", ms=4, label="OneSimulation 3D FEM, Prony branch")
ax.axhline(E_0 / 1e9, color="#999", lw=0.8, ls=":")
ax.axhline(E_inf / 1e9, color="#999", lw=0.8, ls=":")
ax.text(end_time, E_0 / 1e9, " E_0", va="center", color="#666", fontsize=8)
ax.text(end_time, E_inf / 1e9, " E_inf", va="center", color="#666", fontsize=8)
ax.set(xlabel="Time t / s", ylabel="Relaxation modulus sigma(t) / eps_0  /  GPa",
       title="Stress relaxation of a standard linear solid under a step strain")
ax.legend()
plt.show()


# ## Going further
#
# Add a second Prony branch, e.g. `prony_terms=((2*G_1, tau), (2*G_2, 20*tau))`,
# to obtain a broader relaxation spectrum; the reference then becomes a sum of
# two exponentials with their own uniaxial time constants. Halve the time step
# and watch the backward-Euler deviation from the exact exponential shrink.
