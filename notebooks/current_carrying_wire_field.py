"""OneSimulation example: magnetic field of a long current-carrying wire (magnetostatics).
Requires the installed, activated OneSimulation runtime.

The 3D FEM solution is checked against Ampere's law for an infinitely long conductor.
"""


# # Magnetic field of a current-carrying wire
#
# A straight conductor of radius $R_w$ carries a total current $I$ uniformly over its
# cross section. It runs along the axis of a much wider cylinder of air and spans the
# whole length of the computational domain, so the current density
# $\mathbf J = J\,\hat{\mathbf z}$ is divergence-free and the problem does not depend
# on $z$. OneSimulation solves
# $\nabla\times(\mu^{-1}\nabla\times\mathbf A)=\mathbf J$ with a weak Coulomb gauge on a
# structured hexahedral mesh (Nedelec elements of order 2) and reconstructs
# $\mathbf B=\nabla\times\mathbf A$. The conductor is supplied as a small user-defined
# current source: the model accepts any object with a meshed `support` region and a
# `current_density(points)` method, and the mesher resolves that support exactly.
#
# The conductor has a hairline axial bore of radius $r_0=R_w/10$. It lets the
# structured mesh stay clear of the axis; since no current flows inside the bore,
# the tangential magnetic field on its wall is exactly zero, and the bore enters the
# reference only through the enclosed current.
#
# **Reference.** Ampere's law for an infinitely long conductor gives a purely
# azimuthal field,
#
# $$B_\varphi(r)=\frac{\mu_0 I}{2\pi r}\,\frac{r^2-r_0^2}{R_w^2-r_0^2}\ (r_0\le r\le R_w),
#   \qquad B_\varphi(r)=\frac{\mu_0 I}{2\pi r}\ (r\ge R_w),$$
#
# which for $r_0\to 0$ is the familiar linear rise $\mu_0 I r/(2\pi R_w^2)$ inside
# and the $1/r$ decay outside. Magnetic insulation ($\mathbf n\times\mathbf A=0$) on the
# outer wall and the end faces is compatible with this field: $\mathbf B$ is tangential
# to the outer wall and $\mathbf A=A_z(r)\,\hat{\mathbf z}$ has no tangential component
# on the end faces. The comparison runs along a radial line at mid-length from inside
# the conductor out into the air.

from dataclasses import dataclass, field
from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import AnnularCylinder
from onesim.materials import VacuumPermeability
from onesim.meshing import StructuredAnnularHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import SparseDirect
from onesim.physics.magnetostatics import (
    CoulombGauge, MagneticFieldBoundary, MagneticInsulation, MagneticVectorPotential,
    Magnetostatics,
)
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters and model

wire_radius = 0.005                             # m, conductor radius R_w
bore_radius = wire_radius / 10.0                # m, hairline axial bore r_0
domain_radius, domain_length = 0.030, 0.010     # m, air cylinder (insulated far wall), axial slice
current = 100.0                                 # A, total current along +z

mu0 = float(VacuumPermeability().evaluate())
cross_section = np.pi * (wire_radius**2 - bore_radius**2)
print(f"J = {current/cross_section/1e6:.3f} A/mm^2,  "
      f"B at the conductor surface = {mu0*current/(2*np.pi*wire_radius)*1e3:.3f} mT")


@dataclass(frozen=True)
class AxialWireCurrent:
    """Uniform axial current density on a resolved (bored) cylindrical conductor."""

    inner_radius: float
    outer_radius: float
    length: float
    current: float
    support: AnnularCylinder = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "support", AnnularCylinder(
            inner_radius=self.inner_radius, outer_radius=self.outer_radius, length=self.length))

    def current_density(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        inside = (radius >= self.inner_radius) & (radius <= self.outer_radius)
        values = np.zeros_like(pts)
        values[inside, 2] = self.current / (np.pi * (self.outer_radius**2 - self.inner_radius**2))
        return values.reshape(np.shape(points))


geometry = AnnularCylinder(inner_radius=bore_radius, outer_radius=domain_radius, length=domain_length)
wire = AxialWireCurrent(inner_radius=bore_radius, outer_radius=wire_radius,
                        length=domain_length, current=current)
model = Magnetostatics(
    domain=geometry.volume, permeability=VacuumPermeability(),
    formulation=MagneticVectorPotential(gauge=CoulombGauge()), current_sources=(wire,),
)
problem = Problem(
    name="Current-carrying wire field",
    geometry=geometry, physics=(model,),
    boundary_conditions=(
        MagneticInsulation(
            boundary=geometry.boundaries.outer_wall | geometry.boundaries.top | geometry.boundaries.bottom,
            physics=model),
        # no current inside the bore -> tangential H vanishes on its wall (Ampere's law)
        MagneticFieldBoundary(boundary=geometry.boundaries.inner_wall, physics=model, value=(0.0, 0.0, 0.0)),
    ),
)
mesh = StructuredAnnularHexahedralMesher(
    radial_cells=24, azimuthal_cells=24, axial_cells=2, embedded_regions=(wire.support,),
).generate(geometry)
print("Mesh:", f"{mesh.cell_count} hexahedra, {len(mesh.region_cells(wire.support))} of them in the conductor")
result = Simulation(
    problem=problem, mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Nedelec(order=2), quadrature_order=5),
                      linear_solver=SparseDirect()),
).solve_stationary()


# ## Azimuthal field along a radial line
#
# The flux density is sampled on a radial line at mid-length and projected onto the
# azimuthal direction. Inside the conductor the field rises almost linearly, outside it
# decays as $1/r$.

B_field = result.field(model.magnetic_flux_density)
r = np.linspace(0.3 * wire_radius, 0.9 * domain_radius, 40)
points = np.column_stack((r, np.zeros_like(r), np.full_like(r, domain_length / 2.0)))
B = np.asarray(B_field.evaluate(points))
B_phi = B[:, 1]                                  # on the +x axis the azimuthal direction is +y
B_ref = mu0 * current / (2 * np.pi * r) * np.clip(
    (r**2 - bore_radius**2) / (wire_radius**2 - bore_radius**2), None, 1.0)

rel_error = np.abs(B_phi - B_ref) / B_ref
inside, outside = r <= wire_radius, r > wire_radius
for name, mask in (("inside the conductor", inside), ("outside (air)", outside)):
    print(f"max relative deviation {name}: {100*rel_error[mask].max():.2f} %")
stray = np.abs(B[:, [0, 2]]).max() / B_ref.max()
print(f"off-azimuthal components |B_r|, |B_z| at most {100*stray:.2f} % of the peak field")

assert np.isfinite(B).all(), "The flux density must be finite."
assert np.all(B_phi > 0), "The field must circulate counter-clockwise around a +z current."
assert rel_error.max() < 0.05, "Deviation larger than the example tolerance (5 % against Ampere's law)."
assert stray < 0.03, "The field should be purely azimuthal."
assert B_phi[inside][-1] > B_phi[inside][0], "The field must grow with r inside the conductor."
assert B_phi[outside][-1] < B_phi[outside][0], "The field must decay with r outside the conductor."


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
ax.plot(r * 1e3, B_ref * 1e3, color="#214D43", lw=1.8, label="Ampere's law (infinite conductor)")
ax.plot(r * 1e3, B_phi * 1e3, "o", color="#A85539", ms=4, label="OneSimulation 3D FEM")
ax.axvline(wire_radius * 1e3, color="#6b442f", lw=1, ls=":", label="conductor surface $R_w$")
ax.set(xlabel="Radius r / mm", ylabel="Azimuthal flux density $B_\\varphi$ / mT",
       title="Current-carrying wire: azimuthal field at mid-length")
ax.legend()
plt.show()


# ## Going further
#
# Give the conductor a permeability other than $\mu_0$ (an iron wire): the field inside
# scales with $\mu_r$ while the exterior $1/r$ law is unchanged, since Ampere's law only
# sees the enclosed current. Doubling the radial and azimuthal cell counts shows the
# second-order convergence of the Nedelec discretisation towards the closed form.
