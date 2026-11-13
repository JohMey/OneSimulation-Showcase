"""OneSimulation example: elastoplastic bar in tension with J2 plasticity (linear isotropic hardening).
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form bilinear
stress-strain law of a uniaxially loaded elastoplastic bar.
"""


# # Elastoplastic bar: yielding under a displacement ramp
#
# A prismatic steel bar (length $L$ along $x$, square cross section $A=w\,h$) is
# clamped axially on its left face and stretched by a prescribed displacement on
# its right face. The displacement is ramped well beyond the elastic limit and
# then partly released again. The material is elastic-plastic: isotropic Hooke
# elasticity, a von Mises (J2) yield surface and linear isotropic hardening with
# modulus $H$.
#
# OneSimulation solves the 3D small-strain equilibrium $-\nabla\!\cdot\boldsymbol\sigma=0$
# on a hexahedral mesh (Lagrange order 2). At every quadrature point the stress
# follows from the implicit radial-return mapping, and the plastic history
# (plastic strain tensor, equivalent plastic strain) is carried from load step
# to load step.
#
# **Reference.** The stress state in the bar is homogeneous and uniaxial, so
# the classical one-dimensional elastoplastic law applies. With
# $\varepsilon_y=\sigma_y/E$ and the tangent modulus $E_T = E H/(E+H)$,
#
# $$\sigma(\varepsilon)=\begin{cases}
#   E\,\varepsilon, & \varepsilon\le\varepsilon_y,\\[2pt]
#   \sigma_y + E_T\,(\varepsilon-\varepsilon_y), & \varepsilon>\varepsilon_y,
# \end{cases}\qquad
# \varepsilon^p=\frac{E}{E+H}\,(\varepsilon-\varepsilon_y).$$
#
# During unloading the response is elastic again with slope $E$. Plastic flow
# is volume preserving, so the lateral strain is
# $\varepsilon_{yy}=-\nu\,\varepsilon^e-\tfrac12\,\varepsilon^p$.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Box
from onesim.meshing import StructuredHexahedralMesher
from onesim.physics.solid_mechanics import SmallStrainSolid
from onesim.physics.solid_mechanics.inelasticity.models.plasticity import IsotropicHardening, J2Plasticity

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model

length, width, height = 0.200, 0.020, 0.020     # m
E, nu = 200e9, 0.30                             # Pa, -   structural steel
sigma_y, H = 250e6, 2.0e9                       # Pa, Pa  initial yield stress, isotropic hardening modulus
area = width * height
shear_modulus = E / (2 * (1 + nu))
strain_yield = sigma_y / E
tangent_modulus = E * H / (E + H)

# Loading programme: ramp the end displacement to 4 eps_y, then release half a
# yield strain again (elastic unloading, no reverse yielding).
strain_max = 4.0 * strain_yield
strain_release = 0.5 * strain_yield
times = np.concatenate((np.linspace(0.0, 1.0, 25), np.linspace(1.0, 1.25, 7)[1:]))
strain_programme = np.where(times <= 1.0, strain_max * times,
                            strain_max - strain_release * (times - 1.0) / 0.25)


def end_displacement(t):
    """Prescribed axial displacement of the right face as a function of time."""
    return float(np.interp(t, times, strain_programme)) * length


geometry = Box(length=length, width=width, height=height)
material = J2Plasticity(shear_modulus=shear_modulus,
                        hardening=IsotropicHardening(sigma_y, H))
model = SmallStrainSolid(youngs_modulus=E, poisson_ratio=nu, kinematics="volume", material=material)
mesh = StructuredHexahedralMesher(cells=(8, 2, 2)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")

# Boundary conditions: axial clamp and ramp, two symmetry-type supports that
# suppress rigid-body motion but leave the lateral contraction free, and
# traction-free remaining faces.
result = model.solve(
    mesh=mesh,
    times=times,
    order=2,
    displacements=(
        (geometry.boundaries.x_min, "x", 0.0),
        (geometry.boundaries.x_max, "x", end_displacement),
        (geometry.boundaries.y_min, "y", 0.0),
        (geometry.boundaries.z_min, "z", 0.0),
    ),
    tractions=(
        (geometry.boundaries.y_max, (0.0, 0.0, 0.0)),
        (geometry.boundaries.z_max, (0.0, 0.0, 0.0)),
    ),
)


# ## Comparison with the bilinear law

def reference_stress(strain_history):
    """Closed-form uniaxial elastoplastic response to a monotone ramp followed by elastic release."""
    stress = np.where(strain_history <= strain_yield, E * strain_history,
                      sigma_y + tangent_modulus * (strain_history - strain_yield))
    peak = int(np.argmax(strain_history))
    stress[peak:] = stress[peak] - E * (strain_history[peak] - strain_history[peak:])
    return stress


strain = strain_programme
stress_reference = reference_stress(strain)

# Axial stress: mean over all quadrature points, and spread as a homogeneity check.
sigma_xx = result.stress[:, :, 0, 0]
stress_fem = sigma_xx.mean(axis=1)
stress_spread = np.ptp(sigma_xx, axis=1)

# Axial force from the reactions on the pulled face.
pulled = np.isclose(result.coordinates[:, 0], length)
force = result.reactions[:, pulled, 0].sum(axis=1)
stress_from_force = force / area

# Yield point: intersection of the fitted elastic and hardening branches.
loading = times <= 1.0
elastic_part = loading & (strain < 0.9 * strain_yield)
plastic_part = loading & (strain > 1.5 * strain_yield)
slope_elastic, offset_elastic = np.polyfit(strain[elastic_part], stress_fem[elastic_part], 1)
slope_plastic, offset_plastic = np.polyfit(strain[plastic_part], stress_fem[plastic_part], 1)
strain_yield_fem = (offset_plastic - offset_elastic) / (slope_elastic - slope_plastic)
stress_yield_fem = slope_elastic * strain_yield_fem + offset_elastic

# Unloading slope (elastic again).
unloading = times > 1.0
slope_unloading = np.polyfit(strain[unloading], stress_fem[unloading], 1)[0]

# Equivalent plastic strain at the end of the ramp, read from the material history.
peak = int(np.argmax(strain))
plastic_fem = np.mean([state.equivalent_plastic_strain
                       for cell in result.material_history[peak] for state in cell])
plastic_reference = E / (E + H) * (strain_max - strain_yield)

# Lateral strain at the peak from the displacement of the free y face.
face_y = np.isclose(result.coordinates[:, 1], width)
lateral_fem = result.displacement[peak, face_y, 1].mean() / width
lateral_reference = -nu * (strain_max - plastic_reference) - 0.5 * plastic_reference

rel_curve = np.max(np.abs(stress_fem - stress_reference)) / sigma_y
rel_force = np.max(np.abs(stress_from_force - stress_reference)) / sigma_y
rel_yield_stress = abs(stress_yield_fem - sigma_y) / sigma_y
rel_yield_strain = abs(strain_yield_fem - strain_yield) / strain_yield
rel_E = abs(slope_elastic - E) / E
rel_ET = abs(slope_plastic - tangent_modulus) / tangent_modulus
rel_unload = abs(slope_unloading - E) / E
rel_plastic = abs(plastic_fem - plastic_reference) / plastic_reference
rel_lateral = abs(lateral_fem - lateral_reference) / abs(lateral_reference)

print(f"Stress-strain curve:   max deviation {100*rel_curve:.4f} % of sigma_y (quadrature stress)")
print(f"                       max deviation {100*rel_force:.4f} % of sigma_y (reaction force / A)")
print(f"Homogeneity:           max stress spread over the bar {stress_spread.max()/1e6:.2e} MPa")
print(f"Yield stress:   FEM {stress_yield_fem/1e6:.3f} MPa,  input {sigma_y/1e6:.3f} MPa  ({100*rel_yield_stress:.4f} %)")
print(f"Yield strain:   FEM {strain_yield_fem:.6e},  theory {strain_yield:.6e}  ({100*rel_yield_strain:.4f} %)")
print(f"Elastic slope:  FEM {slope_elastic/1e9:.3f} GPa,  E = {E/1e9:.3f} GPa  ({100*rel_E:.4f} %)")
print(f"Hardening slope: FEM {slope_plastic/1e9:.4f} GPa,  E_T = {tangent_modulus/1e9:.4f} GPa  ({100*rel_ET:.4f} %)")
print(f"Unloading slope: FEM {slope_unloading/1e9:.3f} GPa,  E = {E/1e9:.3f} GPa  ({100*rel_unload:.4f} %)")
print(f"Plastic strain at peak: FEM {plastic_fem:.6e},  theory {plastic_reference:.6e}  ({100*rel_plastic:.4f} %)")
print(f"Lateral strain at peak: FEM {lateral_fem:.6e},  theory {lateral_reference:.6e}  ({100*rel_lateral:.4f} %)")

assert stress_fem[peak] > sigma_y, "The bar must be loaded beyond the yield stress."
assert rel_curve < 1e-3, "Stress-strain curve deviates more than 0.1 % of sigma_y from the bilinear law."
assert rel_force < 1e-3, "Reaction force deviates more than 0.1 % of sigma_y A from the bilinear law."
assert stress_spread.max() < 1e-4 * sigma_y, "The stress state in the bar is not homogeneous."
assert rel_yield_stress < 5e-3, "Yield stress deviates more than 0.5 % from the input."
assert rel_yield_strain < 5e-3, "Yield strain deviates more than 0.5 % from sigma_y / E."
assert rel_E < 1e-3 and rel_unload < 1e-3, "Elastic loading or unloading slope deviates from E."
assert rel_ET < 1e-2, "Hardening slope deviates more than 1 % from E H / (E + H)."
assert rel_plastic < 1e-3, "Equivalent plastic strain deviates more than 0.1 % from the reference."
assert rel_lateral < 1e-2, "Lateral strain does not reflect plastic incompressibility."


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
fine = np.linspace(0.0, strain_max, 200)
ax.plot(fine * 1e3, np.where(fine <= strain_yield, E * fine, sigma_y + tangent_modulus * (fine - strain_yield)) / 1e6,
        color="#214D43", lw=1.8, label="theory  (bilinear, loading)")
ax.plot(strain[unloading] * 1e3, stress_reference[unloading] / 1e6, "--", color="#214D43", lw=1.4,
        label="theory  (elastic unloading)")
ax.plot(strain * 1e3, stress_fem / 1e6, "o", color="#A85539", ms=4, label="OneSimulation 3D FEM, J2 plasticity")
ax.axhline(sigma_y / 1e6, color="#999", lw=0.8, ls=":")
ax.set(xlabel="Axial strain / 1e-3", ylabel="Axial stress / MPa",
       title="Elastoplastic bar: stress-strain curve with yield, hardening and unloading")
ax.legend(loc="lower right")
plt.show()


# ## Going further
#
# Set the hardening modulus $H$ to zero for an elastic-perfectly-plastic
# plateau at $\sigma_y$, or increase the released strain beyond
# $2\,\varepsilon_y$ to trigger reverse yielding in compression. Replace the
# isotropic law by `CombinedHardening` with a `KinematicHardening` branch to
# see the Bauschinger effect shift the reverse yield point.
