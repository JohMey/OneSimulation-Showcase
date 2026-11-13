"""OneSimulation example: bending mode shapes of a clamped-free beam (modal analysis).
Requires the installed, activated OneSimulation runtime.

The OneSimulation mode shapes are checked against the closed-form Euler-Bernoulli eigenfunctions.
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
# **Reference.** Euler-Bernoulli beam theory gives the clamped-free eigenfunctions
#
# $$\phi_n(x)=\cosh\beta_n x-\cos\beta_n x-\sigma_n\left(\sinh\beta_n x-\sin\beta_n x\right),\qquad
#   \sigma_n=\frac{\sinh\beta_n L-\sin\beta_n L}{\cosh\beta_n L+\cos\beta_n L},$$
#
# where $\sigma_n$ follows from the free-end conditions $\phi_n''(L)=\phi_n'''(L)=0$
# ($\sigma_1=0.7341$, $\sigma_2=1.0185$), with $\beta_1 L=1.8751$ and
# $\beta_2 L=4.6941$ the roots of $\cos\beta L\cosh\beta L=-1$, and the frequencies
# $f_n=\frac{\beta_n^2}{2\pi}\sqrt{EI/(\rho A L^4)}$, so that
# $f_2/f_1=(4.6941/1.8751)^2=6.267$ independent of the material and the section.
# Beam theory neglects shear deformation and rotary inertia, which lower the second
# frequency of a solid beam of finite slenderness by a few percent.

from importlib.metadata import version
from math import pi, sqrt
import numpy as np
import matplotlib.pyplot as plt

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
#
# The section is wider ($b$, along $y$) than thick ($h$, along $z$) so that bending
# about the two axes is not degenerate: the lowest modes are then, in order, the
# first weak-axis bending mode, the first strong-axis bending mode and the second
# weak-axis bending mode.

length, width, height = 0.120, 0.012, 0.008          # m  (x, y, z)
E, nu, density = 210e9, 0.30, 7850.0                 # Pa, -, kg/m^3 (structural steel)
inertia = width * height**3 / 12.0                   # weak-axis second moment of area
area = width * height
beta_L = (1.8751, 4.6941)                            # clamped-free eigenvalues beta_n L
frequency_reference = tuple(bl**2 / (2.0 * pi) * sqrt(E * inertia / (density * area * length**4)) for bl in beta_L)
ratio_reference = (beta_L[1] / beta_L[0]) ** 2
print(f"Euler-Bernoulli: f1 = {frequency_reference[0]:.1f} Hz, f2 = {frequency_reference[1]:.1f} Hz, f2/f1 = {ratio_reference:.3f}")

geometry = Box(length=length, width=width, height=height)
material = IsotropicSolid(
    density=density, specific_heat_capacity=460.0, thermal_conductivity=50.0,
    youngs_modulus=E, poisson_ratio=nu, thermal_expansion_coefficient=12e-6,
)
model = LinearElasticity(domain=geometry.volume, material=material)
problem = Problem(
    name="Clamped-free beam mode shapes",
    geometry=geometry, physics=(model,),
    boundary_conditions=(
        FixedDisplacement(boundary=geometry.boundaries.x_min, physics=model, value=(0.0, 0.0, 0.0)),
        TractionFree(boundary=geometry.boundaries.all - geometry.boundaries.x_min, physics=model),
    ),
)
mesh = StructuredHexahedralMesher(cells=(24, 2, 2)).generate(geometry)
print("Mesh:", mesh.cell_count, "hexahedra")
result = Simulation(
    problem=problem, mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)), linear_solver=SparseDirect()),
).solve_stationary()


# ## Modal analysis
#
# `structural_modes` returns the natural frequencies and, through
# `mode_displacements(i)`, the nodal displacement of each mode on the Q2 node set.
# The node coordinates of that set are exposed by the finite-element space, so the
# centre line $y=b/2,\ z=h/2$ can be sampled directly. The two weak-axis bending
# modes are identified as the modes whose centre-line motion is dominated by $u_z$.

modes = structural_modes(result, model, mode_count=4)
print("First four frequencies / Hz:", [round(f, 1) for f in modes.frequencies])

nodes = np.asarray(result.operators[model.model_id].space.dof_coordinates)
on_axis = np.isclose(nodes[:, 1], width / 2) & np.isclose(nodes[:, 2], height / 2)
order = np.argsort(nodes[on_axis, 0])
x_axis = nodes[on_axis, 0][order]

bending_z = []
for i in range(modes.mode_count):
    u = modes.mode_displacements(i)[on_axis][order]
    peak = np.abs(u).max(axis=0)
    if peak[2] > 10.0 * max(peak[0], peak[1]):
        bending_z.append((i, u[:, 2]))
(index_1, w_1), (index_2, w_2) = bending_z[:2]
f_1, f_2 = modes.frequencies[index_1], modes.frequencies[index_2]
print(f"Weak-axis bending modes: #{index_1 + 1} at {f_1:.1f} Hz and #{index_2 + 1} at {f_2:.1f} Hz")


# ## Comparison with the Euler-Bernoulli eigenfunctions
#
# Both the computed and the analytical shapes are normalised to unit tip deflection.
# The deviation is reported as the maximum over the centre line.

def phi(x, bl):
    beta = bl / length
    sigma = (np.sinh(bl) - np.sin(bl)) / (np.cosh(bl) + np.cos(bl))
    return np.cosh(beta * x) - np.cos(beta * x) - sigma * (np.sinh(beta * x) - np.sin(beta * x))

shape_errors = []
shapes = []
for n, (w, bl) in enumerate(zip((w_1, w_2), beta_L), start=1):
    computed = w / w[-1]
    reference = phi(x_axis, bl) / phi(length, bl)
    err = float(np.max(np.abs(computed - reference)))
    shape_errors.append(err)
    shapes.append((computed, reference))
    print(f"mode {n}: max |phi_FEM - phi_EB| = {err:.4f}  (unit tip deflection)")

ratio = f_2 / f_1
ratio_error = abs(ratio - ratio_reference) / ratio_reference
f1_error = abs(f_1 - frequency_reference[0]) / frequency_reference[0]
print(f"f1: modal {f_1:.1f} Hz vs Euler-Bernoulli {frequency_reference[0]:.1f} Hz  ({100*f1_error:.2f} %)")
print(f"f2/f1: modal {ratio:.3f} vs Euler-Bernoulli {ratio_reference:.3f}  ({100*ratio_error:.2f} %, "
      "shear and rotary inertia lower the second mode)")
# The second mode must have exactly one interior node (sign change) along the beam.


# ## Figure

cream = "#fbf5eb"
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream); ax.set_facecolor("#fffaf2"); ax.grid(alpha=0.25)
colors = ("#A85539", "#214D43")
x_fine = np.linspace(0.0, length, 300)
for n, ((computed, _), bl, c, f) in enumerate(zip(shapes, beta_L, colors, (f_1, f_2)), start=1):
    ax.plot(x_fine * 1000, phi(x_fine, bl) / phi(length, bl), "-", color=c, lw=1.8,
            label=f"Euler-Bernoulli $\\phi_{n}$")
    ax.plot(x_axis * 1000, computed, "o", color=c, ms=3.5, label=f"OneSimulation mode {n} ({f:.0f} Hz)")
ax.axhline(0.0, color="#7a7a7a", lw=0.8)
ax.set(xlabel="Position along the beam x / mm", ylabel="Normalised deflection $w / w_{tip}$",
       title="Clamped-free beam: first two weak-axis bending mode shapes")
ax.legend(loc="lower left")
plt.show()


# ## Going further
#
# Add `mode_count` and look at the strong-axis pair, whose shapes follow the same
# $\phi_n$ but whose frequencies scale with $b/h$. Shorten the beam to see the
# second-mode frequency ratio fall further below 6.267 as shear deformation and
# rotary inertia grow, while the mode shapes remain close to the beam-theory ones.
