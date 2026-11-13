"""OneSimulation example: capacitance of a parallel-plate capacitor from the field energy.
Requires the installed, activated OneSimulation runtime.

The capacitance extracted from the stored electrostatic energy is checked against the
closed-form parallel-plate value and against the charge-flux route.
"""


# # Capacitance from the electrostatic field energy
#
# A circuit designer rarely wants the field itself; what matters is the lumped
# parameter it implies. For a capacitor that parameter follows from the energy stored
# in the field: with one plate at $V_0$ and the other grounded,
#
# $$W=\frac12\int_\Omega \varepsilon\,|\mathbf E|^2\,\mathrm dV,\qquad
#   C=\frac{2W}{V_0^2}.$$
#
# The same capacitance also follows from the charge on the grounded plate,
# $Q=\varepsilon\oint \mathbf E\cdot\mathbf n\,\mathrm dA$ and $C=Q/V_0$. OneSimulation
# solves Laplace's equation $\nabla\cdot(\varepsilon\nabla V)=0$ on a real 3D hexahedral
# mesh (Lagrange order 2), and this notebook extracts $C$ by both routes.
#
# **Reference.** For the ideal parallel-plate capacitor (plate area $A$, gap $d$, no
# fringing because the side walls carry a zero normal displacement),
#
# $$C=\frac{\varepsilon_0\,\varepsilon_r\,A}{d},\qquad
#   W=\frac12\,C\,V_0^2 .$$
#
# The field is exactly uniform, so both numerical routes must reproduce this value to
# solver precision, and filling the gap with a dielectric must scale $C$ linearly in
# $\varepsilon_r$.

from importlib.metadata import version
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Box
from onesim.materials import VacuumPermittivity
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import (
    FixedPotential, PoissonEquation, ZeroNormalElectricDisplacement,
)
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The gap runs along x, so the two plates are the x_min face (at $V_0$) and the x_max
# face (grounded). The four remaining faces carry a zero normal electric displacement.

d = 0.002             # m, plate separation (gap along x)
W = 0.02              # m, plate width (y)
H = 0.02              # m, plate height (z)
V0 = 50.0             # V, potential on the x_min plate
eps0 = float(VacuumPermittivity().evaluate())
A = W * H             # m^2, plate area

geometry = Box(length=d, width=W, height=H)
mesh = StructuredHexahedralMesher(cells=(12, 6, 6)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")


@dataclass(frozen=True)
class Dielectric:
    """Linear isotropic dielectric: a constant permittivity eps0 * eps_r in F/m."""
    relative_permittivity: float
    unit = "F/m"

    def evaluate(self, **state):
        return eps0 * self.relative_permittivity


# ## Solver and energy integration
#
# The energy density $\tfrac12\varepsilon|\mathbf E|^2$ is integrated over the volume
# with a 2x2x2 Gauss rule on every hexahedral cell of the mesh (exact for the
# quadratic energy density of a cell-wise linear field). The charge-flux route uses the
# built-in surface integral of the normal field component over the grounded plate.

gauss = np.array([-1.0, 1.0]) / np.sqrt(3.0)
corners = np.asarray(mesh.coordinates)[np.asarray(mesh.cells)]        # (cells, 8, 3)
lo, hi = corners.min(axis=1), corners.max(axis=1)                    # axis-aligned cells
centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
offsets = np.array([[gx, gy, gz] for gx in gauss for gy in gauss for gz in gauss])
quad_points = (centre[:, None, :] + half[:, None, :] * offsets[None, :, :]).reshape(-1, 3)
quad_weights = np.repeat(np.prod(2.0 * half, axis=1) / 8.0, len(offsets))
print(f"Quadrature: {len(quad_points)} points, total volume {quad_weights.sum()*1e9:.3f} mm^3 "
      f"(exact {d*W*H*1e9:.3f} mm^3)")


def solve_capacitor(permittivity, eps_value):
    """Solve the electrostatic problem and return (C_energy, C_flux, W_energy)."""
    model = PoissonEquation(domain=geometry.volume, permittivity=permittivity, charge_density=0.0)
    problem = Problem(
        name="Parallel-plate capacitor energy",
        geometry=geometry,
        physics=(model,),
        boundary_conditions=(
            FixedPotential(boundary=geometry.boundaries.x_min, physics=model, value=V0),
            FixedPotential(boundary=geometry.boundaries.x_max, physics=model, value=0.0),
            ZeroNormalElectricDisplacement(
                boundary=geometry.boundaries.y_min | geometry.boundaries.y_max
                | geometry.boundaries.z_min | geometry.boundaries.z_max,
                physics=model),
        ),
    )
    result = Simulation(
        problem=problem, mesh=mesh,
        numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()),
    ).solve_stationary()
    E_field = result.field(model.electric_field)

    E_quad = np.asarray(E_field.evaluate(quad_points))
    energy = 0.5 * eps_value * float(np.sum(quad_weights * np.sum(E_quad**2, axis=1)))
    C_energy = 2.0 * energy / V0**2

    flux = E_field.integrate(boundary=geometry.boundaries.x_max, normal_component=True)
    C_flux = eps_value * abs(float(flux.value)) / V0
    return C_energy, C_flux, energy


# ## Vacuum gap: energy route versus charge route versus closed form

C_energy, C_flux, W_energy = solve_capacitor(VacuumPermittivity(), eps0)
C_ana = eps0 * A / d
W_ana = 0.5 * C_ana * V0**2

err_energy = abs(C_energy - C_ana) / C_ana
err_flux = abs(C_flux - C_ana) / C_ana
err_routes = abs(C_energy - C_flux) / C_flux

print(f"stored energy:         {W_energy*1e9:.6f} nJ   (closed form {W_ana*1e9:.6f} nJ)")
print(f"C from field energy:   {C_energy*1e12:.6f} pF  ({100*err_energy:.2e} % off eps0*A/d)")
print(f"C from charge flux:    {C_flux*1e12:.6f} pF  ({100*err_flux:.2e} % off eps0*A/d)")
print(f"C closed form eps0A/d: {C_ana*1e12:.6f} pF")
print(f"energy route vs flux route: {100*err_routes:.2e} %")

assert np.isfinite(W_energy) and W_energy > 0, "The stored energy must be finite and positive."
assert err_energy < 1e-3, "Energy-route capacitance deviates from eps0*A/d by more than 0.1 %."
assert err_flux < 1e-3, "Flux-route capacitance deviates from eps0*A/d by more than 0.1 %."
assert err_routes < 1e-6, "The energy and charge-flux routes must agree to solver precision."


# ## Dielectric filling: C scales linearly with eps_r

eps_r_values = (1.0, 2.0, 4.0, 8.0)
C_energy_eps, C_flux_eps = [], []
for eps_r in eps_r_values:
    material = VacuumPermittivity() if eps_r == 1.0 else Dielectric(relative_permittivity=eps_r)
    C_e, C_f, _ = solve_capacitor(material, eps0 * eps_r)
    C_energy_eps.append(C_e); C_flux_eps.append(C_f)
    print(f"eps_r = {eps_r:>3.0f}:  C_energy {C_e*1e12:8.4f} pF,  C_flux {C_f*1e12:8.4f} pF,  "
          f"eps0*eps_r*A/d {eps_r*C_ana*1e12:8.4f} pF")

C_energy_eps = np.array(C_energy_eps); C_flux_eps = np.array(C_flux_eps)
C_ana_eps = np.array(eps_r_values) * C_ana
scaling_error = float(np.max(np.abs(C_energy_eps - C_ana_eps) / C_ana_eps))
print(f"max deviation of the dielectric sweep from linear scaling: {100*scaling_error:.2e} %")
assert scaling_error < 1e-3, "Capacitance does not scale linearly with the relative permittivity."


# ## Figure

cream = "#fbf5eb"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
fig.patch.set_facecolor(cream)
for ax in (ax1, ax2):
    ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
routes = ("field energy\n$2W/V_0^2$", "charge flux\n$Q/V_0$", "closed form\n$\\varepsilon_0 A/d$")
values = np.array([C_energy, C_flux, C_ana]) * 1e12
ax1.bar(routes, values, color=("#A85539", "#3887c5", "#214D43"), width=0.55)
ax1.set_ylim(0, 1.15 * values.max())
for i, v in enumerate(values):
    ax1.text(i, v * 1.02, f"{v:.4f} pF", ha="center", va="bottom", fontsize=9)
ax1.set(ylabel="Capacitance / pF", title="Vacuum gap: three routes to C")
ax2.plot(eps_r_values, C_ana_eps * 1e12, color="#214D43", lw=1.8, ls="--", label="$\\varepsilon_0\\varepsilon_r A/d$")
ax2.plot(eps_r_values, C_energy_eps * 1e12, "o", color="#A85539", ms=7, label="from field energy")
ax2.plot(eps_r_values, C_flux_eps * 1e12, "x", color="#3887c5", ms=8, mew=2, label="from charge flux")
ax2.set(xlabel="Relative permittivity $\\varepsilon_r$", ylabel="Capacitance / pF",
        title="Dielectric filling: linear scaling"); ax2.legend()
plt.show()


# ## Going further
#
# Add fringing by making the plates smaller than the box and keeping the outer walls
# charge-free: the two routes still agree with each other, but both now exceed
# $\varepsilon_0 A/d$, and the energy route becomes the convenient way to read off the
# fringing correction. The same recipe, with $\tfrac12\int\mathbf B\cdot\mathbf H\,\mathrm dV$,
# yields inductances from magnetostatic solutions.
