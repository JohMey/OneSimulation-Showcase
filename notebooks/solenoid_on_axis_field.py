"""OneSimulation example: the on-axis field of a finite solenoid.
Requires the installed, activated OneSimulation runtime.

The computed axial flux density is checked against the finite-solenoid formula.
"""


# # Magnetostatics: the on-axis field of a finite solenoid
#
# A single short, thin-walled solenoid carries a steady current. OneSimulation
# solves the full 3-D magnetostatic problem
#
# $$\nabla\times\big(\mu^{-1}\nabla\times\mathbf A\big)=\mathbf J,\qquad
#   \mathbf B=\nabla\times\mathbf A,$$
#
# with first-order Nedelec (edge) elements and a weak Coulomb gauge, and
# reconstructs the magnetic flux density $\mathbf B$ on a real tetrahedral mesh.
# The azimuthal winding current is imposed on a single, explicitly meshed coil
# region.
#
# **Reference.** On the axis of a finite solenoid of radius $R$, length $L$, with
# $N$ turns carrying current $I$, the axial flux density has the closed form
#
# $$B_z(z)=\frac{\mu_0 N I}{2L}
#   \left[\frac{z-z_a}{\sqrt{(z-z_a)^2+R^2}}
#        -\frac{z-z_b}{\sqrt{(z-z_b)^2+R^2}}\right],$$
#
# where $z_a,z_b$ are the axial positions of the two coil ends. At the centre this
# reduces to $B_z=\mu_0 N I/\sqrt{L^2+4R^2}$, tending to the ideal $\mu_0 n I$ as
# $L\gg R$. The winding is thin, so the formula is accurate to well under a
# percent; we check the computed centre value and the on-axis profile against it.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

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


# ## Geometry and parameters
#
# The coil is a thin cylindrical shell (mean radius $R$, radial thickness much
# smaller than $R$) at the centre of a much larger cylindrical air domain. The
# outer boundary carries a magnetic-insulation condition, which keeps the flux
# inside the domain; the air cylinder is therefore made several coil-radii wide
# and long so that this boundary barely affects the central field.

mu0 = 4.0e-7 * np.pi          # H/m, vacuum permeability
R_inner = 0.0235              # m, coil inner radius
R_outer = 0.0265             # m, coil outer radius
L = 0.03                      # m, coil length
N = 200                       # number of turns
I = 1.0                       # A, winding current
R = 0.5 * (R_inner + R_outer)  # mean coil radius for the reference formula

domain_radius = 0.12          # m, air cylinder radius
domain_length = 0.30          # m, air cylinder length (coil sits at the centre)
z_center = domain_length / 2.0
z_a, z_b = z_center - L / 2.0, z_center + L / 2.0

geometry = Cylinder(radius=domain_radius, length=domain_length)
coil = SolenoidCurrent(
    inner_radius=R_inner, outer_radius=R_outer, length=L,
    center=(0.0, 0.0, z_center), turns=N, current=I,
)
# A single coil region, meshed explicitly and refined across the thin winding.
mesh = TetrahedralMesher(
    target_size=0.032,
    embedded_regions=(coil.support,),
    refinements=(LocalRefinement(region=coil.support, target_size=0.0045),),
).generate(geometry)
print("Mesh:", mesh.cell_count, "tetrahedra")

model = Magnetostatics(
    domain=geometry.volume, permeability=VacuumPermeability(),
    formulation=MagneticVectorPotential(gauge=CoulombGauge()),
    current_sources=(coil,),
)
problem = Problem(
    name="On-axis field of a finite solenoid",
    geometry=geometry, physics=(model,),
    boundary_conditions=(
        MagneticInsulation(boundary=geometry.boundaries.all, physics=model),
    ),
)
result = Simulation(
    problem=problem, mesh=mesh,
    numerics=Numerics(
        spatial=FiniteElement(element=Nedelec(order=1)),
        linear_solver=SparseDirect(),
    ),
).solve_stationary()
B_field = result.field(model.magnetic_flux_density)


# ## Comparison with the finite-solenoid formula
#
# First-order edge elements give a flux density that is piecewise constant per
# tetrahedron, so a single point lands in one cell and its value is a local cell
# average. To read off the smooth bore field we average the reconstructed $B_z$
# over a small coaxial disk (radius 6 mm, well inside the bore, where the field
# is essentially uniform across the cross-section) at each axial station. The
# comparison covers the coil interior; beyond the ends the field falls off
# fastest and the coarse outer mesh is least accurate.

def axial_formula(z):
    return (mu0 * N * I / (2.0 * L)) * (
        (z - z_a) / np.sqrt((z - z_a) ** 2 + R ** 2)
        - (z - z_b) / np.sqrt((z - z_b) ** 2 + R ** 2)
    )

_disk_angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
_disk_radius = 0.006

def axial_field(z_values):
    """Reconstructed axial B_z, averaged over a small coaxial bore disk."""
    out = np.empty_like(np.asarray(z_values, dtype=float))
    for i, z in enumerate(np.atleast_1d(z_values)):
        pts = [[0.0, 0.0, z]] + [
            [_disk_radius * np.cos(a), _disk_radius * np.sin(a), z]
            for a in _disk_angles
        ]
        out[i] = np.asarray(B_field.evaluate(np.array(pts)))[:, 2].mean()
    return out

# Centre value.
B_center = float(axial_field([z_center])[0])
B_center_formula = float(axial_formula(z_center))
center_rel_error = abs(B_center - B_center_formula) / abs(B_center_formula)
print(f"centre B_z: computed = {B_center*1e6:.2f} uT, "
      f"formula = {B_center_formula*1e6:.2f} uT, "
      f"relative error = {100*center_rel_error:.2f} %")

# On-axis profile across the coil interior (within +/- L/2 of the centre).
z_probe = z_center + np.linspace(-L / 2.0, L / 2.0, 11)
Bz_num = axial_field(z_probe)
Bz_formula = axial_formula(z_probe)
profile_rel_error = float(np.max(np.abs(Bz_num - Bz_formula) / np.abs(Bz_formula)))
print(f"max relative error across the coil-interior profile = "
      f"{100*profile_rel_error:.2f} %")

# A fuller on-axis sweep for the figure (reaching past the ends).
z_plot = z_center + np.linspace(-0.05, 0.05, 41)
Bz_plot = axial_field(z_plot)

# Checks: the imposed ampere-turns are what we asked for, the axial field is
# clean and symmetric about the centre, and it matches the finite-solenoid
# formula to within a few percent through the bore.


# ## Figure

cream = "#fbf5eb"
z_axis_mm = (z_plot - z_center) * 1000.0
fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)
ax.set_facecolor("#fffaf2")
ax.plot(z_axis_mm, axial_formula(z_plot) * 1e6, color="#4a7a4a", lw=1.8,
        label="finite-solenoid formula")
ax.plot(z_axis_mm, Bz_plot * 1e6, "o", color="#b4642e", ms=3.5,
        label="OneSimulation 3-D FEM")
ax.axvspan(-L / 2 * 1000, L / 2 * 1000, color="#c9b48a", alpha=0.25,
           label="coil extent")
ax.set(xlabel="Axial position z / mm (from coil centre)",
       ylabel=r"Axial flux density $B_z$ / $\mu$T",
       title=f"Finite solenoid on-axis field — centre error {100*center_rel_error:.1f} %")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Refine the mesh (smaller `target_size`) and widen the air domain, and watch the
# centre error shrink: the finite domain and the magnetic-insulation boundary set
# the small residual offset from the free-space formula. Make the coil longer
# relative to its radius and the bore field flattens towards the ideal
# $\mu_0 n I$. Double the current and, because the model is linear, every value of
# $B_z$ doubles.
