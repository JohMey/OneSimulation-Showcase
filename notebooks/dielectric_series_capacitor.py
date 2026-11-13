"""OneSimulation example: a two-layer dielectric capacitor.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form series-dielectric result.
"""


# # Series dielectric capacitor
#
# A parallel-plate gap is filled with two stacked dielectric layers. Layer 1
# (thickness $d_1$, relative permittivity $\varepsilon_{r1}$) sits against the live
# plate at $V_0$; layer 2 (thickness $d_2$, relative permittivity
# $\varepsilon_{r2}$) sits against the grounded plate. The stack behaves as two
# capacitors in series.
#
# With no free charge in either layer the potential satisfies
#
# $$\nabla\cdot(\varepsilon(x)\,\nabla V)=0,\qquad V(0)=V_0,\quad V(d)=0,$$
#
# where the permittivity $\varepsilon(x)$ jumps at the interface. OneSimulation
# solves this on a 3D hexahedral mesh with finite elements (Lagrange order 2); the
# interface is placed on element boundaries so the piecewise-linear potential is
# represented exactly.
#
# **Reference.** The normal electric displacement $D$ is continuous across the
# interface, so with $a_i = d_i/(\varepsilon_0\varepsilon_{ri})$
#
# $$D=\frac{V_0}{a_1+a_2},\qquad
#   E_i=\frac{D}{\varepsilon_0\varepsilon_{ri}},\qquad
#   C=\frac{A}{a_1+a_2}
#    =\left(\frac{1}{C_1}+\frac{1}{C_2}\right)^{-1}.$$
#
# The field is piecewise uniform, larger in the lower-permittivity layer. We check
# the potential, the two field levels, and the series capacitance.

from importlib.metadata import version
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Box
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.electrostatics import (
    FixedPotential, PoissonEquation, ZeroNormalElectricDisplacement,
)
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model
#
# The gap runs along x. Layer 1 occupies 0 < x < d1, layer 2 occupies d1 < x < d.
# A spatially varying permittivity returns the local value at each quadrature point.

eps0 = 8.8541878128e-12
d1 = 0.001            # m, thickness of layer 1 (against the live plate)
d2 = 0.001            # m, thickness of layer 2 (against ground)
d = d1 + d2           # m, total gap
W = 0.02              # m, plate width (y)
H = 0.02              # m, plate height (z)
V0 = 50.0             # V, potential on the x_min plate
eps_r1 = 2.0          # relative permittivity of layer 1
eps_r2 = 5.0          # relative permittivity of layer 2
A = W * H             # m^2, plate area


@dataclass(frozen=True)
class LayeredPermittivity:
    """Piecewise-constant permittivity: eps_r1 for x < d1, eps_r2 beyond."""

    unit = "F/m"

    def evaluate(self, **state):
        x = state["x"]
        eps_r = np.where(np.asarray(x) < d1, eps_r1, eps_r2)
        return eps0 * (eps_r if eps_r.ndim else float(eps_r))


geometry = Box(length=d, width=W, height=H)
# Even cell count along x puts the interface at d1 on an element boundary.
mesh = StructuredHexahedralMesher(cells=(12, 6, 6)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")

model = PoissonEquation(domain=geometry.volume, permittivity=LayeredPermittivity(), charge_density=0.0)
problem = Problem(
    name="Series dielectric capacitor",
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
    problem=problem,
    mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()),
).solve_stationary()


# ## Comparison with the analytic solution
#
# Closed-form series solution.

a1 = d1 / (eps0 * eps_r1)
a2 = d2 / (eps0 * eps_r2)
D = V0 / (a1 + a2)
E1 = D / (eps0 * eps_r1)
E2 = D / (eps0 * eps_r2)
V_interface = V0 - E1 * d1


def V_analytic(xv):
    return np.where(xv <= d1, V0 - E1 * xv, V_interface - E2 * (xv - d1))


# Axial cut along x, kept strictly inside each layer to avoid the jump at x = d1.
x1 = np.linspace(1e-5, d1 - 1e-5, 20)
x2 = np.linspace(d1 + 1e-5, d - 1e-5, 20)
x = np.concatenate((x1, x2))
pts = np.column_stack((x, np.full_like(x, W / 2), np.full_like(x, H / 2)))
V_num = np.asarray(result.field(model.potential).evaluate(pts)).reshape(-1)
Ex_num = np.asarray(result.field(model.electric_field).evaluate(pts))[:, 0]

V_ana = V_analytic(x)
Ex_ana = np.where(x <= d1, E1, E2)

V_rel = float(np.max(np.abs(V_num - V_ana)) / V0)
Ex_rel = float(np.max(np.abs(Ex_num - Ex_ana) / Ex_ana))

# Capacitance from the charge on the grounded plate: Q = eps0*eps_r2 * flux there.
flux = result.field(model.electric_field).integrate(
    boundary=geometry.boundaries.x_max, normal_component=True)
C_num = eps0 * eps_r2 * abs(float(flux.value)) / V0
C_ana = A / (a1 + a2)
C_rel = abs(C_num - C_ana) / C_ana

print(f"max relative potential error: {100*V_rel:.3f} %")
print(f"max relative field error:     {100*Ex_rel:.3f} %")
print(f"field in layer 1 / layer 2:   {E1:.1f} / {E2:.1f} V/m (analytic)")
print(f"analytic capacitance: {C_ana*1e12:.4f} pF")
print(f"numeric  capacitance: {C_num*1e12:.4f} pF  ({100*C_rel:.3f} % off)")


# ## Figure

cream = "#fbf5eb"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
fig.patch.set_facecolor(cream)
xs = np.linspace(0, d, 200)
for ax in (ax1, ax2):
    ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25); ax.set_xlabel("Position x / mm")
    ax.axvline(d1 * 1000, color="#999", ls="--", lw=1, label="interface")
ax1.plot(xs * 1000, V_analytic(xs), color="#6b442f", label="analytic")
ax1.plot(x * 1000, V_num, "o", color="#c56838", ms=4, label="OneSimulation FEM")
ax1.set(ylabel="Potential / V", title="Potential V(x)"); ax1.legend()
ax2.plot(xs * 1000, np.where(xs <= d1, E1, E2), color="#2f5b6b", label="analytic")
ax2.plot(x * 1000, Ex_num, "o", color="#3887c5", ms=4, label="OneSimulation FEM")
ax2.set(ylabel="Axial field E_x / (V/m)", title="Axial field E_x(x)"); ax2.legend()
plt.show()


# ## Going further
#
# Swap the layer order or the permittivity ratio: the series capacitance is
# unchanged, but the field redistributes so that the thinner or lower-permittivity
# layer always carries the larger field, which is where dielectric breakdown starts.
