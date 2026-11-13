"""OneSimulation example: bending of a clamped beam (linear elasticity).
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against Euler-Bernoulli beam theory.
"""


# # Bending of a clamped beam
#
# A beam (length $L$, width $b$, height $h$) fully clamped on the left carries a
# uniformly distributed transverse load with total force $F$ pointing down on the right.
#
# OneSimulation solves the stationary 3D equilibrium
# $-\nabla\!\cdot\boldsymbol\sigma=0$ with an isotropic Hooke law (Q2 hexahedra) and
# computes displacement and stress.
#
# **Reference.** Euler-Bernoulli beam theory gives the tip deflection
#
# $$u_z(L)=-\frac{F L^3}{3EI},\qquad I=\frac{b h^3}{12}.$$
#
# It neglects shear and 3D clamping effects and is therefore an approximate target value.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# OneSimulation figure style: cream paper, dark-green ink, green-sand-copper field colours
CREAM, INK, GREEN, COPPER = "#F7F2E7", "#223D35", "#336B57", "#B66E43"
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation", ["#214D43", "#5F9175", "#A8BC91", "#E4D6A0", "#CD9561", "#A85539"])
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "onesimulation_diverging", ["#104F54", "#5E9996", "#F7F2E7", "#D2996E", "#A4472D"])
for _cmap in (FIELD_CMAP, DIVERGING_CMAP):
    if _cmap.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cmap)
plt.rcParams.update({
    "figure.facecolor": CREAM, "savefig.facecolor": CREAM, "axes.facecolor": "#FFFAF2",
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": "#9FA593",
    "xtick.color": INK, "ytick.color": INK, "grid.color": "#D5D7C8", "image.cmap": "onesimulation",
    "axes.prop_cycle": cycler(color=[GREEN, COPPER, "#5F9175", "#A85539", "#214D43", "#CD9561", "#8B9A8D"]),
})

from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import FixedDisplacement, LinearElasticity, Traction, TractionFree
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model

length, width, height = 0.120, 0.018, 0.012   # m
E, nu = 70e9, 0.30                             # Pa
force = 30.0                                   # N, total transverse load pointing down
inertia = width * height**3 / 12.0
tip_reference = -force * length**3 / (3.0 * E * inertia)

geometry = Box(length=length, width=width, height=height)
material = IsotropicSolid(
    density=2700.0, specific_heat_capacity=900.0, thermal_conductivity=200.0,
    youngs_modulus=E, poisson_ratio=nu, thermal_expansion_coefficient=23e-6,
)
model = LinearElasticity(domain=geometry.volume, material=material)
problem = Problem(
    name="Clamped beam under end-face load",
    geometry=geometry, physics=(model,),
    boundary_conditions=(
        FixedDisplacement(boundary=geometry.boundaries.x_min, physics=model, value=(0.0, 0.0, 0.0)),
        Traction(boundary=geometry.boundaries.x_max, physics=model,
                 value=(0.0, 0.0, -force / (width * height))),
        TractionFree(boundary=geometry.boundaries.all - geometry.boundaries.x_min - geometry.boundaries.x_max,
                     physics=model),
    ),
)
mesh = StructuredHexahedralMesher(cells=(24, 4, 4)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")
result = Simulation(
    problem=problem, mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()),
).solve_stationary()


# ## Comparison with beam theory

displacement = result.field(model.displacement)
tip = float(displacement.evaluate([[length, width / 2, height / 2]])[0, 2])
rel_error = abs(tip - tip_reference) / abs(tip_reference)
print(f"Tip deflection: FEM {tip*1e3:.4f} mm,  Euler-Bernoulli {tip_reference*1e3:.4f} mm")
print(f"relative deviation: {100*rel_error:.2f} %  (3D FEM is slightly lower due to shear/clamping)")

# Deflection line along the central axis
xs = np.linspace(0.0, length, 25)
line = np.column_stack((xs, np.full_like(xs, width / 2), np.full_like(xs, height / 2)))
uz = np.asarray(displacement.evaluate(line))[:, 2]
uz_ref = -force * (3 * length * xs**2 - xs**3) / (6 * E * inertia)


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
ax.plot(xs * 1000, uz_ref * 1e3, color="#214D43", lw=1.8, label="Euler-Bernoulli")
ax.plot(xs * 1000, uz * 1e3, "o", color="#A85539", ms=4, label="OneSimulation 3D FEM")
ax.set(xlabel="Position along the beam x / mm", ylabel="Deflection u_z / mm",
       title="Clamped beam: deflection line")
ax.legend()
plt.show()


# ## Going further
#
# Refine the mesh (more cells through the height) and watch the FEM deflection
# approach the Euler-Bernoulli value, or slightly exceed it because of shear.
# Change the L/h ratio; for stubby beams the shear contribution grows and beam
# theory becomes less accurate.
