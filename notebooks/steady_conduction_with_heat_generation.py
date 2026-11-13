"""OneSimulation example: steady conduction in a wall with uniform heat generation.
Requires the installed, activated OneSimulation runtime.

The OneSimulation solution is checked against the closed-form parabolic profile.
"""


# # Steady conduction with internal heat generation
#
# A plane wall of thickness $L$ generates heat uniformly at a volumetric rate
# $q$ (W/m^3). Both faces are held at the same temperature $T_s$ and the lateral
# faces are insulated, so the temperature varies only along $x$.
#
# OneSimulation solves the stationary heat equation
#
# $$0 = k\,\frac{d^2 T}{dx^2} + q,\qquad T(0)=T(L)=T_s,$$
#
# on a real 3D mesh with finite elements (Lagrange order 2); the lateral faces
# carry a zero-flux condition so the behaviour is one-dimensional.
#
# **Reference.** Integrating twice with the symmetric boundary conditions gives
# the parabola
#
# $$T(x) = T_s + \frac{q}{2k}\,x\,(L-x),$$
#
# with the peak temperature $T_s + qL^2/(8k)$ at the centre.

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
from onesim.physics.heat_transfer import FixedTemperature, HeatConduction, HeatFlux
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The slab lies along $x$; its lateral size is small and only serves to form a
# slender 3D mesh. The source $q$ and conductivity $k$ set the peak temperature
# rise $qL^2/(8k)$ above the surface value.

L = 0.10              # m, wall thickness (x direction)
cross = 0.02          # m, lateral size (insulated)
k = 20.0              # W/(m·K), thermal conductivity
q = 5.0e5             # W/m^3, uniform volumetric heat generation
T_s = 350.0           # K, temperature of both faces
peak_rise = q * L * L / (8.0 * k)
print(f"Analytic peak temperature rise above the surface: {peak_rise:.3f} K")

geometry = Box(length=L, width=cross, height=cross)
# Structured hexahedral mesh: many cells along x (the 1D direction), few across.
mesh = StructuredHexahedralMesher(cells=(40, 2, 2)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")

material = IsotropicSolid(
    density=1000.0, specific_heat_capacity=500.0, thermal_conductivity=k,
    youngs_modulus=100e9, poisson_ratio=0.3, thermal_expansion_coefficient=1e-5,
)
model = HeatConduction(domain=geometry.volume, material=material, heating=q)
problem = Problem(
    name="Steady conduction with internal heat generation",
    geometry=geometry,
    physics=(model,),
    boundary_conditions=(
        FixedTemperature(boundary=geometry.boundaries.x_min, physics=model, value=T_s),
        FixedTemperature(boundary=geometry.boundaries.x_max, physics=model, value=T_s),
        # laterally insulated (zero flux) -> 1D behaviour
        HeatFlux(
            boundary=(geometry.boundaries.y_min | geometry.boundaries.y_max
                      | geometry.boundaries.z_min | geometry.boundaries.z_max),
            physics=model, value=0.0,
        ),
    ),
)
result = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(element=Lagrange(order=2)),
        linear_solver=SparseDirect(),
    ),
).solve_stationary()


# ## Comparison with the closed-form parabola
#
# We sample the computed temperature along the centre line and compare it with
# $T(x) = T_s + q/(2k)\,x(L-x)$.

def analytic(x):
    return T_s + q / (2.0 * k) * x * (L - x)

sample_x = np.linspace(0.0, L, 41)
line = np.column_stack((sample_x, np.full_like(sample_x, cross / 2), np.full_like(sample_x, cross / 2)))
num = np.asarray(result.field(model.temperature).evaluate(line)).reshape(-1)
ana = analytic(sample_x)
max_abs_error = float(np.max(np.abs(num - ana)))
rel_error = max_abs_error / peak_rise
print(f"Maximum absolute error against the parabola: {max_abs_error:.4f} K  ({100*rel_error:.3f} % of the peak rise)")

# Physical sanity: the peak sits at the centre and both faces stay at T_s.
center_T = float(np.asarray(result.field(model.temperature).evaluate([[L / 2, cross / 2, cross / 2]])).reshape(-1)[0])
print(f"Centre temperature: FEM {center_T:.3f} K,  analytic {T_s + peak_rise:.3f} K")


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
ax.plot(sample_x * 1000, ana, color="#214D43", lw=1.8, label="Closed-form parabola")
ax.plot(sample_x * 1000, num, "o", color="#A85539", ms=4, label="OneSimulation 3D FEM")
ax.set(xlabel="Position x / mm", ylabel="Temperature / K",
       title="Wall with internal heat generation: temperature profile")
ax.legend()
plt.show()


# ## Going further
#
# Raise the volumetric source $q$ or lower the conductivity $k$ and watch the
# parabola grow; the peak rise scales as $qL^2/(8k)$. Replace one fixed-temperature
# face with an insulated face to model one-sided cooling; the maximum then moves to
# the insulated face and the reference becomes $T_s + q/(2k)(L^2 - x^2)$.
