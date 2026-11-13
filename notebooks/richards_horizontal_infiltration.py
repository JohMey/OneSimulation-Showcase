"""OneSimulation example: horizontal infiltration into an unsaturated soil column.
Requires the installed, activated OneSimulation runtime.

The OneSimulation Richards solution is checked against the Boltzmann similarity
solution of horizontal absorption: the self-similar water-content profile and the
square-root-of-time law of the cumulative infiltration.
"""


# # Horizontal infiltration into an unsaturated soil column
#
# A long horizontal column of soil is initially at a uniform (fairly dry) water
# content $\theta_i$. From $t=0$ on, the inlet face $x=0$ is held close to
# saturation, so water is drawn into the column by capillarity alone; gravity
# plays no role in a horizontal column. The far end stays dry during the whole
# experiment.
#
# OneSimulation solves the mixed form of the Richards equation with a
# conservative finite-volume scheme,
#
# $$\frac{\partial\theta(h)}{\partial t}
#   =\frac{\partial}{\partial x}\Big(K(h)\,\frac{\partial h}{\partial x}\Big),\qquad
#   h(x,0)=h_i,\quad h(0,t)=h_0,\quad q(L,t)=0,$$
#
# with the van Genuchten-Mualem closure for the water content $\theta(h)$ and
# the hydraulic conductivity $K(h)$. Written in terms of $\theta$ this is a
# nonlinear diffusion equation with the soil-water diffusivity
# $D(\theta)=K/C$, $C=d\theta/dh$.
#
# **Reference.** Because there is no gravity term and no intrinsic length scale,
# the semi-infinite absorption problem admits Boltzmann's similarity variable
# $\varphi=x/\sqrt{t}$: the profile is $\theta(x,t)=\Theta(\varphi)$ at all
# times, where $\Theta$ solves the ordinary differential equation
#
# $$\frac{d}{d\varphi}\Big(D(\Theta)\frac{d\Theta}{d\varphi}\Big)
#   +\frac{\varphi}{2}\frac{d\Theta}{d\varphi}=0,\qquad
#   \Theta(0)=\theta_0,\quad\Theta(\infty)=\theta_i,$$
#
# and the cumulative infiltration grows exactly as the square root of time,
#
# $$I(t)=S\sqrt{t},\qquad
#   S=\int_{\theta_i}^{\theta_0}\varphi\,d\theta=-2\,D(\theta_0)\,\Theta'(0),$$
#
# with $S$ the sorptivity of the soil. For a constant diffusivity the profile
# is the closed form $\Theta=\theta_i+(\theta_0-\theta_i)\,
# \mathrm{erfc}\big(\varphi/2\sqrt{D}\big)$ with $S=2(\theta_0-\theta_i)\sqrt{D/\pi}$;
# for the real soil $D$ varies by a factor of forty across the wetting front,
# so $\Theta(\varphi)$ is obtained by integrating the ordinary differential
# equation with a shooting method, completely independently of the
# finite-volume solver. We check the profiles at several times, the collapse of
# all profiles onto one curve in $\varphi$, the sorptivity and the
# $\sqrt{t}$ law.

from importlib.metadata import version
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from onesim.physics.hydrodynamics.porous.numerics.equations.multiphysics.richards import Richards1DProvider

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# The values are freely adjustable and describe a loam-type soil. The inlet is
# held at a small suction of 0.1 m (95 % of saturation): the van Genuchten-Mualem
# diffusivity diverges at exact saturation, which neither the similarity
# solution nor an explicit scheme can represent. The column is long enough that
# the wetting front never reaches the far end within the simulated hour.

theta_r = 0.08          # residual water content
theta_s = 0.43          # saturated water content
alpha = 3.6             # 1/m, van Genuchten alpha
n_vg = 1.56             # van Genuchten n
K_s = 1.0e-5            # m/s, saturated hydraulic conductivity
h_0 = -0.10             # m, pressure head held at the inlet face x = 0
h_i = -1.00             # m, initial pressure head of the column
L = 0.5                 # m, column length (far end stays dry)
cells = 400             # finite-volume cells
time_step = 0.02        # s, explicit time step
segment = 60.0          # s, length of one chained run segment (= sampling interval)
end_time = 3600.0       # s, simulated duration
probe_times = (600.0, 1200.0, 2400.0, 3600.0)

m_vg = 1.0 - 1.0 / n_vg
d_theta = theta_s - theta_r


def water_content(h):
    return theta_r + d_theta * (1.0 + np.abs(alpha * np.minimum(h, 0.0)) ** n_vg) ** (-m_vg)


def diffusivity(theta):
    """Soil-water diffusivity D = K / C of the van Genuchten-Mualem soil."""
    S_e = (theta - theta_r) / d_theta
    K_rel = np.sqrt(S_e) * (1.0 - (1.0 - S_e ** (1.0 / m_vg)) ** m_vg) ** 2
    scaled_head = (S_e ** (-1.0 / m_vg) - 1.0) ** (1.0 / n_vg)
    capacity = d_theta * m_vg * n_vg * alpha * scaled_head ** (n_vg - 1.0) * (1.0 + scaled_head**n_vg) ** (-m_vg - 1.0)
    return K_s * K_rel / capacity


theta_0 = float(water_content(h_0))
theta_i = float(water_content(h_i))
print(f"theta_0 = {theta_0:.4f} (inlet),  theta_i = {theta_i:.4f} (initial),  "
      f"D(theta_0) = {diffusivity(theta_0):.2e} m^2/s,  D(theta_i) = {diffusivity(theta_i):.2e} m^2/s")


# ## Similarity solution (reference)
#
# With the flux-like variable $q=D(\Theta)\,\Theta'$ the similarity equation is
# the first-order system $\Theta'=q/D$, $q'=-\tfrac{\varphi}{2}\,q/D$. Its initial
# value $q(0)=-S/2$ is found by shooting: too steep a start reaches $\theta_i$
# with a finite slope, too shallow a start levels off above $\theta_i$.


def shoot(q_0, dense=False):
    def rhs(phi, y):
        theta, q = y
        D = diffusivity(theta)
        return [q / D, -0.5 * phi * q / D]

    def reached_initial_content(phi, y):
        return y[0] - theta_i

    def slope_vanished(phi, y):
        return y[1] + 1e-9 * abs(q_0)

    reached_initial_content.terminal = slope_vanished.terminal = True
    reached_initial_content.direction, slope_vanished.direction = -1, 1
    sol = solve_ivp(rhs, (0.0, 0.05), [theta_0, q_0], events=(reached_initial_content, slope_vanished),
                    rtol=1e-10, atol=1e-14, method="LSODA", dense_output=dense)
    if sol.t_events[0].size:
        return sol.y[1, -1] / abs(q_0), sol            # too steep: negative
    return (sol.y[0, -1] - theta_i) / (theta_0 - theta_i), sol   # too shallow: positive


q_0 = brentq(lambda q: shoot(q)[0], -0.5e-2, -0.5e-5, xtol=1e-16, rtol=1e-12)
mismatch, similarity = shoot(q_0, dense=True)
sorptivity = -2.0 * q_0
phi_front = float(similarity.t[-1])
sorptivity_integral = float(np.trapezoid(similarity.y[0] - theta_i, similarity.t))


def theta_similarity(phi):
    phi = np.asarray(phi, dtype=float)
    inside = np.minimum(phi, phi_front)
    return np.where(phi < phi_front, similarity.sol(inside)[0], theta_i)


print(f"Sorptivity S = {sorptivity:.4e} m/sqrt(s)  (check: integral of phi dtheta = {sorptivity_integral:.4e}),  "
      f"wetting front at phi = {phi_front:.4f} m/sqrt(s)")
# The same constitutive law must be used on both sides: compare D with the solver's own closure.
law = Richards1DProvider(cell_count=cells, length=L, residual_water_content=theta_r, saturated_water_content=theta_s,
                         van_genuchten_alpha=alpha, van_genuchten_n=n_vg, saturated_hydraulic_conductivity=K_s)
for theta in np.linspace(theta_i, theta_0, 7):
    h = law.pressure_head(theta)


# ## Richards run
#
# The inlet face carries the prescribed pressure head $h_0$; the far end is a
# no-flow face. Gravity is switched off for the horizontal column. The solver
# advances the conserved water storage with an explicit step and the harmonic
# mean of the cell conductivities; the time series is obtained by chaining
# one-minute segments, each restarted from the previous pressure-head field,
# which reproduces one long run step for step. The reported boundary exchange
# of every segment is accumulated into the cumulative infiltration.

soil = Richards1DProvider(
    cell_count=cells, length=L,
    residual_water_content=theta_r, saturated_water_content=theta_s,
    van_genuchten_alpha=alpha, van_genuchten_n=n_vg, saturated_hydraulic_conductivity=K_s,
    gravity=0.0,                      # horizontal column
    initial_pressure_head=h_i,
    bottom_pressure_head=h_0,         # inlet face x = 0: prescribed pressure head
    top_pressure_head=None,           # far end x = L: no-flow face
    final_time=segment, time_step=time_step,
)
x = (np.arange(cells) + 0.5) * L / cells
print(f"Grid: {cells} cells of {1e3 * L / cells:.2f} mm, dt = {time_step} s, {int(round(end_time / time_step))} explicit steps")

head = None
t = 0.0
infiltration = 0.0
times, cumulative, profiles, mass_residual = [0.0], [0.0], {}, 0.0
while t < end_time - 1e-9:
    res = soil.run(head)
    head = res.pressure_head
    t += res.final_time
    infiltration += res.boundary_exchange
    mass_residual = max(mass_residual, res.mass_balance_residual)
    times.append(t)
    cumulative.append(infiltration)
    if any(abs(t - probe) < 1e-6 for probe in probe_times):
        profiles[float(round(t))] = np.asarray(res.water_content)
times, cumulative = np.asarray(times), np.asarray(cumulative)
print(f"Run finished at t = {t:.0f} s;  cumulative infiltration I = {infiltration * 1e3:.2f} mm;  "
      f"max mass-balance residual {mass_residual:.1e} m")


# ## Comparison with the similarity solution
#
# The water-content profiles at the probe times are compared with
# $\Theta(x/\sqrt{t})$, the cumulative infiltration with $S\sqrt{t}$, and the
# growth exponent of $I(t)$ is fitted. The far end must still be at $\theta_i$
# for the semi-infinite reference to apply.

print("Water-content profile error per time (relative to theta_0 - theta_i):")
max_profile_error = rms_profile_error = 0.0
for t_probe, theta_num in profiles.items():
    err = np.abs(theta_num - theta_similarity(x / np.sqrt(t_probe))) / (theta_0 - theta_i)
    print(f"  t = {t_probe:6.0f} s   max = {100 * err.max():5.2f} %   rms = {100 * np.sqrt(np.mean(err**2)):5.2f} %   "
          f"front at x = {phi_front * np.sqrt(t_probe) * 1e2:.1f} cm")
    max_profile_error = max(max_profile_error, float(err.max()))
    rms_profile_error = max(rms_profile_error, float(np.sqrt(np.mean(err**2))))

# Self-similarity: the profiles at the probe times must coincide in phi = x / sqrt(t).
phi_common = np.linspace(0.0, 0.9 * phi_front, 200)
collapsed = np.array([np.interp(phi_common, x / np.sqrt(t_probe), theta_num) for t_probe, theta_num in profiles.items()])
collapse_error = float(np.max(np.ptp(collapsed, axis=0)) / (theta_0 - theta_i))

late = times >= probe_times[0]
infiltration_error = np.max(np.abs(cumulative[late] / (sorptivity * np.sqrt(times[late])) - 1.0))
exponent = np.polyfit(np.log(times[late]), np.log(cumulative[late]), 1)[0]
S_measured = float(np.mean(cumulative[late] / np.sqrt(times[late])))
print(f"Collapse of the profiles in phi: max spread {100 * collapse_error:.2f} % of (theta_0 - theta_i)")
print(f"Cumulative infiltration for t >= {probe_times[0]:.0f} s: max |I / (S sqrt t) - 1| = {100 * infiltration_error:.2f} %,  "
      f"fitted exponent {exponent:.4f} (reference 0.5),  I / sqrt(t) = {S_measured:.4e} vs S = {sorptivity:.4e} m/sqrt(s)")

# Physical sanity: the water content stays between the initial and inlet values and decreases along x.


# ## Figure
#
# Left: water-content profiles (dots: OneSimulation, lines: $\Theta(x/\sqrt{t})$).
# Centre: the same profiles against the similarity variable $\varphi$. Right:
# cumulative infiltration against $\sqrt{t}$, which the similarity solution
# predicts to be a straight line of slope $S$.

cream = "#fbf5eb"
fig, (ax_x, ax_phi, ax_i) = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
fig.patch.set_facecolor(cream)
colors = plt.cm.copper(np.linspace(0.2, 0.85, len(profiles)))
x_fine = np.linspace(0.0, L, 1500)
for (t_probe, theta_num), c in zip(profiles.items(), colors):
    ax_x.plot(x_fine * 1e2, theta_similarity(x_fine / np.sqrt(t_probe)), "-", color=c, lw=1.5)
    ax_x.plot(x[::6] * 1e2, theta_num[::6], "o", color=c, ms=3, label=f"OneSimulation, t = {t_probe / 60:.0f} min")
    ax_phi.plot(x / np.sqrt(t_probe) * 1e3, theta_num, "o", color=c, ms=2.5)
ax_x.set_facecolor("#fffaf2"); ax_x.grid(alpha=0.25)
ax_x.set(xlabel="position x / cm", ylabel="water content θ", title="Wetting profiles (lines: similarity solution)")
ax_x.set_xlim(0.0, 1.15 * phi_front * np.sqrt(end_time) * 1e2)
ax_x.legend(frameon=False, fontsize=8)

phi_fine = np.linspace(0.0, 1.2 * phi_front, 600)
ax_phi.plot(phi_fine * 1e3, theta_similarity(phi_fine), "-", color="#39715e", lw=1.8, label="Θ(φ) from the Boltzmann equation")
ax_phi.set_facecolor("#fffaf2"); ax_phi.grid(alpha=0.25)
ax_phi.set(xlabel="φ = x / √t  / (mm/√s)", ylabel="water content θ", title="Collapse onto the similarity variable")
ax_phi.legend(frameon=False, fontsize=8)

ax_i.plot(np.sqrt(times), sorptivity * np.sqrt(times) * 1e3, "-", color="#39715e", lw=1.8, label="S √t (similarity solution)")
ax_i.plot(np.sqrt(times[::2]), cumulative[::2] * 1e3, "o", color="#b4572b", ms=3, label="OneSimulation")
ax_i.set_facecolor("#fffaf2"); ax_i.grid(alpha=0.25)
ax_i.set(xlabel="√t / √s", ylabel="cumulative infiltration I / mm", title="Sorptivity law I = S √t")
ax_i.legend(frameon=False, fontsize=8)
plt.show()


# ## Going further
#
# Lower `h_i` to start from a drier soil: the diffusivity contrast across the
# front grows and the harmonic-mean conductivity needs finer cells to keep the
# early infiltration right. Set `gravity=1.0` to turn the column vertical and
# watch the late-time infiltration rate approach the saturated conductivity
# instead of decaying as $1/\sqrt{t}$ (Philip's two-term law $I=S\sqrt{t}+At$).
