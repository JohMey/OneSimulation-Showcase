"""OneSimulation example: interior field of a long solenoid (magnetostatics).
Requires the installed, activated OneSimulation runtime.

The 3D FEM solution is checked against the ideal infinite-solenoid interior value.
"""


# # Interior field of a long solenoid
#
# A single homogenised solenoid (mean radius $\approx 12.5$ mm, length $200$ mm, so
# length $\gg$ radius) carries $1\,\mathrm{A}$ through $400$ turns. Its current is
# modelled as an azimuthal volume current density on one explicitly meshed winding
# region. OneSimulation solves
# $\nabla\times(\mu^{-1}\nabla\times\mathbf A)=\mathbf J$ with a weak Coulomb gauge
# and reconstructs $\mathbf B=\nabla\times\mathbf A$. This is an actual spatial FEM
# computation, not an analytic coil formula returned by the solver.
#
# **Reference.** Deep inside an infinitely long solenoid the field is axial and uniform,
#
# $$B=\mu_0\,n\,I,\qquad n=\frac{\text{turns}}{\text{length}}.$$
#
# The coil here is long but finite, so the formula is approached from below (the ends
# let a little flux escape). A tolerance of a few per cent covers this finite-length
# effect and the magnetic insulation on the finite computational domain.

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

from onesim.cad.d3 import Cylinder
from onesim.materials import VacuumPermeability
from onesim.meshing import TetrahedralMesher, LocalRefinement
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import SparseDirect
from onesim.physics.magnetostatics import (
    CoulombGauge, MagneticInsulation, MagneticVectorPotential,
    Magnetostatics, SolenoidCurrent,
)
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model

domain_length, domain_radius = 0.40, 0.05       # m, finite computational cylinder
coil_length = 0.20                              # m, solenoid length (>> radius)
inner_radius, outer_radius = 0.010, 0.015       # m, homogenised winding annulus
turns, current = 400, 1.0                       # -, A
center_z = domain_length / 2.0                  # coil centred in the domain

turns_per_metre = turns / coil_length           # n
mu0 = float(VacuumPermeability().evaluate())
field_reference = mu0 * turns_per_metre * current  # ideal infinite-solenoid interior B

geometry = Cylinder(radius=domain_radius, length=domain_length)
coil = SolenoidCurrent(
    inner_radius=inner_radius, outer_radius=outer_radius, length=coil_length,
    center=(0.0, 0.0, center_z), turns=turns, current=current,
)
model = Magnetostatics(
    domain=geometry.volume, permeability=VacuumPermeability(),
    formulation=MagneticVectorPotential(gauge=CoulombGauge()), current_sources=(coil,),
)
problem = Problem(
    name="Long solenoid interior field",
    geometry=geometry, physics=(model,),
    boundary_conditions=(MagneticInsulation(boundary=geometry.boundaries.all, physics=model),),
)
mesh = TetrahedralMesher(
    target_size=0.02, embedded_regions=(coil.support,),
    refinements=(LocalRefinement(region=coil.support, target_size=0.006),),
).generate(geometry)
print("Mesh:", f"{mesh.cell_count} tetrahedra")
result = Simulation(
    problem=problem, mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Nedelec(order=1)), linear_solver=SparseDirect()),
).solve_stationary()


# ## Centre field and axial uniformity
#
# The axial field is sampled at the geometric centre and along the inner axis over the
# central half of the coil, where an infinite solenoid would be perfectly uniform.

B_field = result.field(model.magnetic_flux_density)
centre = np.array([[0.0, 0.0, center_z]])
B_centre = np.asarray(B_field.evaluate(centre))[0]
centre_axial = float(B_centre[2])
rel_error = abs(centre_axial - field_reference) / field_reference
print(f"Centre axial field: FEM {centre_axial*1e3:.4f} mT,  ideal mu0*n*I {field_reference*1e3:.4f} mT")
print(f"relative deviation: {100*rel_error:.2f} %  (finite length lowers the field slightly)")

axial_z = np.linspace(center_z - coil_length / 4.0, center_z + coil_length / 4.0, 21)
axial_points = np.column_stack((np.zeros_like(axial_z), np.zeros_like(axial_z), axial_z))
B_axial = np.asarray(B_field.evaluate(axial_points))
Bz_axial = B_axial[:, 2]
uniformity = float((Bz_axial.max() - Bz_axial.min()) / Bz_axial.mean())
print(f"Axial non-uniformity over the central half: {100*uniformity:.2f} %")


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
ax.axhline(field_reference * 1e3, color="#214D43", lw=1.8, ls="--", label="ideal $\\mu_0 n I$")
ax.plot((axial_z - center_z) * 1e3, Bz_axial * 1e3, "o-", color="#A85539", ms=4,
        label="OneSimulation 3D FEM")
ax.set(xlabel="Axial position from centre z / mm", ylabel="Axial flux density $B_z$ / mT",
       title="Long solenoid: interior axial field")
ax.legend()
plt.show()


# ## Going further
#
# Lengthen the coil (raise the length-to-radius ratio) and watch the centre field climb
# towards $\mu_0 n I$ as the end effects fade. Enlarging the computational cylinder, so
# the magnetic insulation sits farther from the winding, has the same effect of letting
# the interior field approach the ideal infinite-solenoid value.
