"""OneSimulation example: bending mode shapes of a clamped-free beam (modal analysis).
Requires the installed, activated OneSimulation runtime.

The beam is read from a STEP file and meshed with hexahedra; the mode shapes of the modal analysis
are checked against the closed-form Euler-Bernoulli eigenfunctions.
"""

# # Mode shapes of a clamped-free beam
#
# A prismatic steel beam (length $L$, rectangular cross-section $b\times h$ with
# $b>h$) is fully clamped at $x=0$ and free on every other face. OneSimulation
# assembles the 3D linear-elastic stiffness and mass operators (Q2 hexahedra) and
# solves $K\boldsymbol\phi=\omega^2 M\boldsymbol\phi$ on the free degrees of freedom.
# Here we look at the **shapes** of the first two bending modes about the weak axis
# (deflection in $z$, the thin direction), not only at their frequencies.
#
# The beam comes from the CAD file `beam_mode_shapes.step`; its faces are named `x_min`, `x_max`,
# `y_min`, `y_max`, `z_min` and `z_max`. The script `beam_mode_shapes_geometry.py` builds it with
# the CAD objects of OneSimulation.
#
# **Reference.** Euler-Bernoulli beam theory gives the clamped-free eigenfunctions
#
# $$\phi_n(x)=\cosh\beta_n x-\cos\beta_n x-\sigma_n\left(\sinh\beta_n x-\sin\beta_n x\right),\qquad
#   \sigma_n=\frac{\sinh\beta_n L-\sin\beta_n L}{\cosh\beta_n L+\cos\beta_n L},$$
#
# with $\beta_1 L=1.8751$ and $\beta_2 L=4.6941$ the roots of $\cos\beta L\cosh\beta L=-1$, and the
# frequencies $f_n=\frac{\beta_n^2}{2\pi}\sqrt{EI/(\rho A L^4)}$, so that
# $f_2/f_1=(4.6941/1.8751)^2=6.267$ independent of the material and the section.
# Beam theory neglects shear deformation and rotary inertia, which lower the second
# frequency of a solid beam of finite slenderness by a few percent.

import numpy as np
from onesim.cad.exchange import load
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import (
    FixedDisplacement,
    LinearElasticity,
    TractionFree,
    structural_modes,
)
from onesim.simulation import Problem, Simulation
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# The section is wider ($b$, along $y$) than thick ($h$, along $z$), so bending about the two axes
# is not degenerate.

beam = load("beam_mode_shapes.step")
faces = beam.boundaries
length, width, height = beam.size
mesh = StructuredHexahedralMesher(cells=(24, 2, 2)).generate(beam)

fig = plots.figure(1, 2, size=(14, 4.5))
plots.geometry(
    beam,
    selection={"clamped": faces.x_min},
    labels=False,
    title="Beam with its clamped face",
    ax=fig[0],
)
plots.mesh(
    mesh, groups=("all",), legend=False, title=f"{mesh.cell_count} hexahedra", ax=fig[1]
)
plots.show()

# ## Model and modal analysis
#
# `structural_modes` returns the natural frequencies and, through `mode_displacements(i)`, the
# displacement of each mode at the nodes of the finite-element space. The two weak-axis bending
# modes are the modes whose centre-line motion is dominated by $u_z$.

youngs_modulus, density = 210e9, 7850.0
steel = IsotropicSolid(
    density=density,
    specific_heat_capacity=460.0,
    thermal_conductivity=50.0,
    youngs_modulus=youngs_modulus,
    poisson_ratio=0.30,
    thermal_expansion_coefficient=12e-6,
)
model = LinearElasticity(domain=beam.volume, material=steel)
problem = Problem(
    name="Clamped-free beam mode shapes",
    geometry=beam,
    physics=(model,),
    boundary_conditions=(
        FixedDisplacement(boundary=faces.x_min, physics=model, value=(0.0, 0.0, 0.0)),
        TractionFree(boundary=faces.all - faces.x_min, physics=model),
    ),
)
numerics = Numerics(
    spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()
)
result = Simulation(problem=problem, mesh=mesh, numerics=numerics).solve_stationary()
modes = structural_modes(result, model, mode_count=4)

nodes = np.asarray(result.operators[model.model_id].space.dof_coordinates)
centre_line = np.flatnonzero(
    np.isclose(nodes[:, 1], width / 2) & np.isclose(nodes[:, 2], height / 2)
)
centre_line = centre_line[np.argsort(nodes[centre_line, 0])]
x = nodes[centre_line, 0]
motion = [modes.mode_displacements(i)[centre_line] for i in range(modes.mode_count)]
first, second = [
    i
    for i, u in enumerate(motion)
    if np.abs(u[:, 2]).max() > 10 * np.abs(u[:, :2]).max()
][:2]

# ## Comparison with the Euler-Bernoulli eigenfunctions
#
# Computed and closed-form shapes are both scaled to unit tip deflection.


def eigenfunction(x, root):
    """Clamped-free Euler-Bernoulli mode shape for the root beta L, scaled to unit tip deflection."""
    sigma = (np.sinh(root) - np.sin(root)) / (np.cosh(root) + np.cos(root))
    shape = (
        lambda s: np.cosh(root * s)
        - np.cos(root * s)
        - sigma * (np.sinh(root * s) - np.sin(root * s))
    )
    return shape(x / length) / shape(1.0)


roots = (1.8751, 4.6941)
computed = [motion[i][:, 2] / motion[i][-1, 2] for i in (first, second)]
beam_theory = [
    root**2
    / (2 * np.pi)
    * np.sqrt(
        youngs_modulus * width * height**3 / 12 / (density * width * height * length**4)
    )
    for root in roots
]
frequencies = [modes.frequencies[first], modes.frequencies[second]]

print("first four frequencies [Hz]:", [round(f, 1) for f in modes.frequencies])
print(f"weak-axis bending modes: number {first + 1} and number {second + 1}")
for number, (shape, root) in enumerate(zip(computed, roots), start=1):
    print(
        f"mode {number}: largest deviation of the shape from beam theory {np.abs(shape - eigenfunction(x, root)).max():.4f} (unit tip deflection)"
    )
print(
    f"f1: simulation {frequencies[0]:.1f} Hz, Euler-Bernoulli {beam_theory[0]:.1f} Hz, "
    f"deviation {100 * (frequencies[0] / beam_theory[0] - 1):+.2f} %"
)
print(
    f"f2/f1: simulation {frequencies[1] / frequencies[0]:.3f}, Euler-Bernoulli {beam_theory[1] / beam_theory[0]:.3f}, "
    f"deviation {100 * (frequencies[1] / frequencies[0] * beam_theory[0] / beam_theory[1] - 1):+.2f} % (shear and rotary inertia)"
)

# ## Figure

x_fine = np.linspace(0.0, length, 300)
labels = [
    f"mode {number} ({frequency:.0f} Hz)"
    for number, frequency in enumerate(frequencies, start=1)
]
plots.line(
    points={label: (1e3 * x, shape) for label, shape in zip(labels, computed)},
    reference={
        label: (1e3 * x_fine, eigenfunction(x_fine, root))
        for label, root in zip(labels, roots)
    },
    hlines=0.0,
    xlabel="position along the beam x [mm]",
    ylabel="normalised deflection w / w_tip",
    title="Clamped-free beam: simulation (markers) and Euler-Bernoulli (dashed)",
    legend="lower left",
)
plots.show()

# ## Going further
#
# Raise `mode_count` and look at the strong-axis pair, whose shapes follow the same $\phi_n$ but
# whose frequencies scale with $b/h$. Shorten the beam in `beam_mode_shapes_geometry.py` and write
# the STEP file again: the frequency ratio falls further below 6.267 as shear deformation and
# rotary inertia grow, while the mode shapes remain close to the beam-theory ones.
