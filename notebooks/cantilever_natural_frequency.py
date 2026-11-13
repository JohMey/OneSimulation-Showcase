"""OneSimulation example: fundamental bending frequency of a clamped beam (modal analysis).
Requires the installed, activated OneSimulation runtime.

The OneSimulation modal solution is checked against Euler-Bernoulli beam theory.
"""


# # Natural frequency of a clamped beam
#
# A prismatic beam (length $L$, square cross-section $b\times h$) is fully clamped on
# the left and free on every other face. OneSimulation assembles the 3D linear-elastic
# stiffness and mass operators (Q2 hexahedra) and solves the generalised eigenproblem
# $K\boldsymbol\phi=\omega^2 M\boldsymbol\phi$ on the free degrees of freedom to obtain
# the natural modes and frequencies.
#
# **Reference.** Euler-Bernoulli beam theory gives the bending natural frequencies
#
# $$f_n=\frac{\beta_n^2}{2\pi}\sqrt{\frac{E I}{\rho A L^4}},\qquad
#   I=\frac{b h^3}{12},\quad A=b h,$$
#
# with $\beta_1=1.875104$ for the first clamped-free mode. Beam theory neglects shear
# deformation and rotary inertia and is therefore an approximate target value for a
# three-dimensional solid of finite slenderness.

from importlib.metadata import version
from math import pi, sqrt
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
from onesim.physics.solid_mechanics import FixedDisplacement, LinearElasticity, TractionFree, structural_modes
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model

length, side = 0.120, 0.012                     # m, square cross-section
E, nu, density = 210e9, 0.30, 7850.0            # Pa, -, kg/m^3 (structural steel)
inertia = side * side**3 / 12.0                 # I = b h^3 / 12, square section
area = side * side                              # A = b h
beta_1 = 1.875104                               # first clamped-free eigenvalue beta_1 * L
frequency_reference = beta_1**2 / (2.0 * pi) * sqrt(E * inertia / (density * area * length**4))

geometry = Box(length=length, width=side, height=side)
material = IsotropicSolid(
    density=density, specific_heat_capacity=460.0, thermal_conductivity=50.0,
    youngs_modulus=E, poisson_ratio=nu, thermal_expansion_coefficient=12e-6,
)
model = LinearElasticity(domain=geometry.volume, material=material)
problem = Problem(
    name="Clamped beam natural modes",
    geometry=geometry, physics=(model,),
    boundary_conditions=(
        FixedDisplacement(boundary=geometry.boundaries.x_min, physics=model, value=(0.0, 0.0, 0.0)),
        TractionFree(boundary=geometry.boundaries.all - geometry.boundaries.x_min, physics=model),
    ),
)
mesh = StructuredHexahedralMesher(cells=(24, 3, 3)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")
result = Simulation(
    problem=problem, mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()),
).solve_stationary()


# ## Modal analysis and comparison with beam theory
#
# The square cross-section makes bending about the two transverse axes degenerate, so
# the first two computed frequencies form one bending pair. The fundamental is the
# lowest of them.

modes = structural_modes(result, model, mode_count=4)
fundamental = modes.frequencies[0]
rel_error = abs(fundamental - frequency_reference) / frequency_reference
print(f"Fundamental frequency: modal {fundamental:.2f} Hz,  Euler-Bernoulli {frequency_reference:.2f} Hz")
print(f"relative deviation: {100*rel_error:.2f} %  (3D solid is slightly stiffer than beam theory)")
print("First four frequencies / Hz:", [round(f, 2) for f in modes.frequencies])


# ## Figure

cream = "#fbf5eb"
labels = ["Mode 1\n(bending)", "Mode 2\n(bending)", "Mode 3", "Mode 4"]
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25, axis="y")
ax.bar(range(4), modes.frequencies, color="#A85539", width=0.6, label="OneSimulation modal")
ax.axhline(frequency_reference, color="#214D43", lw=1.8, ls="--", label="Euler-Bernoulli $f_1$")
ax.set(xticks=range(4), xticklabels=labels, ylabel="Natural frequency / Hz",
       title="Clamped beam: natural frequencies")
ax.legend()
plt.show()


# ## Going further
#
# Refine the mesh and watch the computed fundamental frequency descend towards the
# Euler-Bernoulli value from above, as the discrete model becomes less stiff. Change
# the slenderness L/h; for stubby beams shear and rotary inertia grow and beam theory
# increasingly overestimates the higher bending frequencies.
