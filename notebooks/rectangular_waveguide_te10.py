"""OneSimulation example: TE10 mode of a rectangular waveguide.
Requires the installed, activated OneSimulation runtime.

The computed port eigenmode and the driven 3D field are checked against the
closed-form TE10 cutoff frequency, mode shape and standing-wave pattern.
"""


# # TE10 mode of a rectangular waveguide
#
# An air-filled rectangular waveguide of cross-section $a\times b$ is fed
# through a **waveguide port** at $z=0$ and terminated by a perfectly conducting
# short at $z=\ell$. All other walls are perfect electric conductors. The port
# mode is not prescribed analytically: OneSimulation solves the transverse
# eigenvalue problem on the port cross-section numerically and uses the first
# TE eigenmode as the excitation.
#
# The frequency-domain Maxwell equations are solved with Nédélec (H(curl))
# elements of order 2 on a tetrahedral mesh:
#
# $$\nabla\times\big(\mu_r^{-1}\nabla\times\mathbf{E}\big)-k_0^2\,\varepsilon_r\,\mathbf{E}=0,
#   \qquad k_0=\omega/c.$$
#
# **Reference.** For the TE$_{mn}$ modes of a rectangular guide
#
# $$f_{c,mn}=\frac{c}{2}\sqrt{\Big(\frac{m}{a}\Big)^2+\Big(\frac{n}{b}\Big)^2},\qquad
#   f_{c,10}=\frac{c}{2a},\qquad
#   E_y\propto\sin\!\Big(\frac{\pi x}{a}\Big),$$
#
# and, with $\beta=\sqrt{k_0^2-(\pi/a)^2}$, the shorted guide carries the
# standing wave $|E_y(z)|\propto|\sin\beta(\ell-z)|$ without absorbing power.
# We check the cutoff frequency, the port mode shape, the standing-wave pattern
# and the absorbed power against these closed forms.

from importlib.metadata import version
from math import pi, sqrt
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad.d3 import Box
from onesim.materials.electromagnetic import IsotropicElectromagneticMaterial
from onesim.materials.properties.units.constants import VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY
from onesim.meshing import TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Nedelec
from onesim.numerics.linear import ComplexSparseDirect
from onesim.physics.electromagnetics import (
    FrequencyDomainMaxwell, NumericalWaveguideMode, PerfectElectricConductor, WaveguidePort,
)
from onesim.simulation import Problem, Simulation

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The guide is 100 mm wide and 50 mm high, so TE10 cuts off at about 1.5 GHz
# and the next modes (TE20, TE01) at 3.0 GHz. At 2.5 GHz only TE10 propagates.

a = 0.100             # m, broad wall (x direction)
b = 0.050             # m, narrow wall (y direction)
length = 0.080        # m, guide length to the short (z direction)
frequency = 2.5e9     # Hz, drive frequency
c0 = 1.0 / sqrt(VACUUM_PERMEABILITY * VACUUM_PERMITTIVITY)
f_c10 = c0 / (2 * a)
f_c20 = c0 / a
f_c01 = c0 / (2 * b)
k0 = 2 * pi * frequency / c0
beta = sqrt(k0**2 - (pi / a) ** 2)
print(f"f_c(TE10) = {f_c10 / 1e9:.4f} GHz,  f_c(TE20) = {f_c20 / 1e9:.3f} GHz,  f_c(TE01) = {f_c01 / 1e9:.3f} GHz")
print(f"guide wavelength at {frequency / 1e9:.1f} GHz: {2 * pi / beta * 1e3:.1f} mm")

guide = Box(length=a, width=b, height=length)
mesh = TetrahedralMesher(target_size=0.010).generate(guide)
print("Mesh:", mesh.cell_count, "tetrahedra")

model = FrequencyDomainMaxwell(
    domain=guide.volume, frequency=frequency,
    material=IsotropicElectromagneticMaterial(),      # air: eps_r = mu_r = 1, lossless
    phasor_convention="exp(-i*omega*t)", amplitude_convention="peak",
)
problem = Problem(
    name="TE10 mode of a rectangular waveguide",
    geometry=guide,
    physics=(model,),
    boundary_conditions=(
        # Port at z = 0 excited with the first numerically computed TE eigenmode.
        WaveguidePort(boundary=guide.boundaries.z_min, physics=model,
                      mode=NumericalWaveguideMode("TE"), incident_power=1.0),
        # Every other face (four side walls and the short at z = length) is a PEC.
        PerfectElectricConductor(boundary=guide.boundaries.all - guide.boundaries.z_min, physics=model),
    ),
)
simulation = Simulation(
    problem=problem,
    mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Nedelec(order=2)), linear_solver=ComplexSparseDirect()),
)
simulation.validate().require_valid()
result = simulation.solve_stationary()


# ## Comparison with the closed-form TE10 solution
#
# The bound port mode exposes its cutoff frequency and its transverse electric
# field; the driven 3D solution is sampled along the guide axis.

_, port_mode = result.operators[model.model_id].ports[0]
f_c_num = float(port_mode.cutoff_frequency)
cutoff_error = abs(f_c_num / f_c10 - 1.0)
print(f"TE10 cutoff: computed {f_c_num / 1e9:.4f} GHz, reference {f_c10 / 1e9:.4f} GHz, deviation {100 * cutoff_error:.2f} %")

# Mode shape across the broad wall at mid-height of the port plane.
xs = np.linspace(0.05 * a, 0.95 * a, 19)
port_points = np.column_stack((xs, np.full_like(xs, b / 2), np.zeros_like(xs)))
E_port = np.asarray(port_mode.electric_field(port_points))
Ey_port = E_port[:, 1]
Ey_port = Ey_port / Ey_port[np.argmax(np.abs(Ey_port))]      # normalised, positive peak
shape_ref = np.sin(pi * xs / a)
shape_error = float(np.max(np.abs(Ey_port - shape_ref)))
cross_component = float(np.max(np.abs(E_port[:, 0])) / np.max(np.abs(E_port[:, 1])))
print(f"Port mode shape vs sin(pi x / a): max deviation {shape_error:.3f} (of peak 1), |E_x|/|E_y| = {cross_component:.3f}")

# Standing wave along the axis of the shorted guide.
zs = np.linspace(0.02 * length, 0.98 * length, 25)
axis_points = np.column_stack((np.full_like(zs, a / 2), np.full_like(zs, b / 2), zs))
E_axis = np.asarray(result.field(model.electric_field).evaluate(axis_points))
amplitude = np.abs(E_axis[:, 1])
standing_ref = np.abs(np.sin(beta * (length - zs)))
standing_error = float(np.max(np.abs(amplitude / amplitude.max() - standing_ref / standing_ref.max())))
print(f"Standing wave |E_y(z)| vs |sin(beta (l - z))|: max deviation {standing_error:.3f} (of peak 1)")

absorbed = result.field(model.absorbed_power_density).integrate()
print(f"Absorbed power in the lossless guide: {absorbed.value:.3e} {absorbed.unit} (incident 1 W)")


# ## Figure

cream = "#fbf5eb"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
fig.patch.set_facecolor(cream)

ax = axes[0]
ax.set_facecolor("#fffaf2")
fine = np.linspace(0, a, 200)
ax.plot(fine * 1e3, np.sin(pi * fine / a), color="#39715e", lw=1.8, label="sin(pi x / a)")
ax.plot(xs * 1e3, Ey_port, "o", color="#b4572b", ms=4, label="OneSimulation port eigenmode")
ax.set(xlabel="Position x across the broad wall / mm", ylabel="E_y / E_y,max",
       title=f"TE10 mode shape — cutoff {f_c_num / 1e9:.3f} GHz (reference {f_c10 / 1e9:.3f} GHz)")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25)

ax = axes[1]
ax.set_facecolor("#fffaf2")
fine_z = np.linspace(0, length, 300)
ax.plot(fine_z * 1e3, np.abs(np.sin(beta * (length - fine_z))), color="#39715e", lw=1.8, label="|sin(beta (l - z))|")
ax.plot(zs * 1e3, amplitude / amplitude.max(), "o", color="#b4572b", ms=4, label="OneSimulation |E_y| on the axis")
ax.set(xlabel="Position z along the guide / mm", ylabel="|E_y| / max",
       title=f"Standing wave in the shorted guide at {frequency / 1e9:.1f} GHz")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25)
plt.show()


# ## Going further
#
# Ask for `NumericalWaveguideMode("TE", order=2)` to excite the next mode and
# compare its cutoff with $f_{c,20}$ or $f_{c,01}$ (raise the drive frequency
# above that cutoff first). Giving the material a finite conductivity turns the
# absorbed power on, and refining `target_size` tightens all four comparisons.
