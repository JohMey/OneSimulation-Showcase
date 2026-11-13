"""OneSimulation example: uniaxial tension of a prismatic bar (linear elasticity).
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form homogeneous
uniaxial stress state.
"""


# # Uniaxial tension of a bar
#
# A prismatic bar (length $L$ along $x$, square cross section $A=w\,h$) is clamped
# on its left face and pulled on its right face by a uniform axial traction
# $\sigma = F/A$. The four long side faces carry no load.
#
# OneSimulation solves the stationary 3D equilibrium
# $-\nabla\!\cdot\boldsymbol\sigma=0$ with an isotropic Hooke law (Q2 hexahedra).
#
# **Reference.** Away from the clamped end the stress state is homogeneous and
# uniaxial, so classical elasticity gives
#
# $$\sigma_{xx}=\frac{F}{A},\qquad \varepsilon_{xx}=\frac{\sigma_{xx}}{E},\qquad
#   \varepsilon_{yy}=\varepsilon_{zz}=-\nu\,\varepsilon_{xx}.$$
#
# We read the axial strain from the slope of $u_x(x)$ in the central region (the
# clamp introduces a thin Saint-Venant boundary layer) and the lateral
# contraction from $u_y$ across the mid-span cross section.

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

length, width, height = 0.400, 0.050, 0.050     # m
E, nu = 200e9, 0.30                             # Pa, -
force = 1.25e5                                  # N, total axial pull
area = width * height
sigma_reference = force / area                  # Pa, homogeneous axial stress
strain_reference = sigma_reference / E          # -, axial strain
elongation_reference = strain_reference * length

geometry = Box(length=length, width=width, height=height)
material = IsotropicSolid(
    density=7850.0, specific_heat_capacity=460.0, thermal_conductivity=50.0,
    youngs_modulus=E, poisson_ratio=nu, thermal_expansion_coefficient=12e-6,
)
model = LinearElasticity(domain=geometry.volume, material=material)
problem = Problem(
    name="Bar under uniaxial tension",
    geometry=geometry, physics=(model,),
    boundary_conditions=(
        FixedDisplacement(boundary=geometry.boundaries.x_min, physics=model, value=(0.0, 0.0, 0.0)),
        Traction(boundary=geometry.boundaries.x_max, physics=model, value=(sigma_reference, 0.0, 0.0)),
        TractionFree(boundary=geometry.boundaries.all - geometry.boundaries.x_min - geometry.boundaries.x_max,
                     physics=model),
    ),
)
mesh = StructuredHexahedralMesher(cells=(32, 4, 4)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")
result = Simulation(
    problem=problem, mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()),
).solve_stationary()


# ## Comparison with the homogeneous uniaxial state

displacement = result.field(model.displacement)

# Axial strain from the slope of u_x(x) in the central region (0.2 L .. 0.8 L).
xs = np.linspace(0.2 * length, 0.8 * length, 13)
axis_points = np.column_stack((xs, np.full_like(xs, width / 2), np.full_like(xs, height / 2)))
ux = np.asarray(displacement.evaluate(axis_points))[:, 0]
strain_axial = np.polyfit(xs, ux, 1)[0]

# Total elongation measured end to end along the central axis.
ends = np.array([[0.0, width / 2, height / 2], [length, width / 2, height / 2]])
elongation = float(np.diff(np.asarray(displacement.evaluate(ends))[:, 0])[0])

# Lateral contraction across the mid-span cross section (y direction).
mid = length / 2
lateral = np.array([[mid, 0.0, height / 2], [mid, width, height / 2]])
uy = np.asarray(displacement.evaluate(lateral))[:, 1]
strain_lateral = float(np.diff(uy)[0]) / width
poisson_measured = -strain_lateral / strain_axial

rel_strain = abs(strain_axial - strain_reference) / strain_reference
rel_elong = abs(elongation - elongation_reference) / elongation_reference
rel_poisson = abs(poisson_measured - nu) / nu

print(f"Axial stress (applied): {sigma_reference/1e6:.3f} MPa")
print(f"Axial strain: FEM {strain_axial:.6e},  theory {strain_reference:.6e}  ({100*rel_strain:.3f} %)")
print(f"Elongation:   FEM {elongation*1e6:.3f} um,  theory {elongation_reference*1e6:.3f} um  ({100*rel_elong:.3f} %)")
print(f"Poisson ratio: FEM {poisson_measured:.4f},  input {nu:.4f}  ({100*rel_poisson:.3f} %)")


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
ax.plot(xs * 1000, strain_reference * xs * 1e6, color="#214D43", lw=1.8, label="theory  eps*x")
ax.plot(xs * 1000, ux * 1e6, "o", color="#A85539", ms=4, label="OneSimulation 3D FEM")
ax.set(xlabel="Position along the bar x / mm", ylabel="Axial displacement u_x / um",
       title="Uniaxial tension: axial displacement in the central region")
ax.legend()
plt.show()


# ## Going further
#
# Increase the length-to-width ratio and watch the clamp boundary layer shrink,
# so the end-to-end elongation approaches F L / (A E) ever more closely. Change
# the Poisson ratio and confirm the measured lateral contraction follows it.
