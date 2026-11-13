"""OneSimulation example: oblique whistler and kinetic-Alfven-type waves in three-dimensional Hall MHD.
Requires the installed, activated OneSimulation runtime.

Fourteen small-amplitude Hall-MHD eigenmodes with wave vectors at angles between 0 and 90 degrees
to the background magnetic field are launched together in a triply periodic box, so the computed
field varies in all three directions. Their frequencies are measured from the time series of each
launched Fourier mode and compared with the exact linear Hall-MHD dispersion relation for oblique
propagation, together with the polarisation, the energy budget, a resolution study and a time-step
study. The Fourier pseudo-spectral discretisation is contrasted with second-order central
differences on the same grids.
"""


# # Oblique whistler waves in Hall MHD
#
# A uniform plasma (density $\rho_0$, pressure $p_0$, adiabatic exponent $\gamma$) is threaded by
# the uniform field $\mathbf B_0=B_0\hat{\mathbf z}$. In Hall magnetohydrodynamics the electric
# field carries, besides the ideal part $-\mathbf v\times\mathbf B$, the Hall term
#
# $$\mathbf E_{\rm Hall}=\eta_H\,\frac{\mathbf J\times\mathbf B}{\rho},\qquad
#   \mathbf J=\nabla\times\mathbf B\qquad(\mu_0=1),$$
#
# which decouples the ions from the field on scales below the ion inertial length
# $d_i=\eta_H/\sqrt{\rho_0}$. The shear-Alfvén wave then splits into a right-hand **whistler**
# branch whose frequency grows like $k^2$, and a left-hand **ion-cyclotron / kinetic-Alfvén-type**
# branch. Both depend on the angle $\theta$ between the wave vector $\mathbf k$ and $\mathbf B_0$.
#
# **Exact reference.** Linearising the compressible, adiabatic Hall-MHD equations about the uniform
# state gives the dispersion relation for oblique propagation (E. Hameiri, A. Ishizawa and
# A. Ishida, *Waves in the Hall-magnetohydrodynamics model*, Phys. Plasmas 12, 072109, 2005)
#
# $$\left(\omega^2-k_\parallel^2v_A^2\right)\left(\omega^4-k^2(v_A^2+c_s^2)\,\omega^2+k^2k_\parallel^2v_A^2c_s^2\right)
#   =\omega^2\,k_\parallel^2k^2\,v_A^2\,d_i^2\left(\omega^2-k^2c_s^2\right),$$
#
# with $k_\parallel=k\cos\theta$, the Alfvén speed $v_A=B_0/\sqrt{\rho_0}$ and the sound speed
# $c_s=\sqrt{\gamma p_0/\rho_0}$. Its three positive roots are the slow, the ion-cyclotron /
# kinetic-Alfvén-type and the fast / whistler branch. For $d_i=0$ it factorises into the ideal
# Alfvén and magnetosonic waves; for $\theta=0$ it reduces to the parallel whistler
# $\omega=kv_A[\sqrt{1+(kd_i/2)^2}+kd_i/2]$ and the ion-cyclotron wave with the minus sign; for
# large $k$ the whistler approaches $\omega\simeq k\,k_\parallel d_i v_A$. The eigenvectors of the
# same linear system give the polarisation of each branch.
#
# **Numerical method.** OneSimulation's three-dimensional resistive/Hall-MHD solver integrates the
# eight conservative fields (density, three momenta, total energy, three magnetic components) in
# conservative form. Two spatial discretisations of the same equations are compared on a periodic
# collocation grid:
#
# * Fourier pseudo-spectral derivatives (exact for every resolved Fourier mode),
# * second-order central finite differences, whose derivative of $e^{ikx}$ is $i\tilde k\,e^{ikx}$
#   with the modified wavenumber $\tilde k=\sin(kh)/h$.
#
# Time integration is the explicit classical fourth-order Runge-Kutta method (RK4).
#
# **What is checked.** Frequencies of all launched modes against the exact dispersion relation;
# polarisation; total-energy conservation and amplitude preservation of every mode; a resolution
# study for both spatial schemes (with the modified-wavenumber prediction for the difference
# scheme); and a time-step study with the observed order of RK4.

from importlib.metadata import version
import dataclasses
import time
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
from matplotlib.colors import TwoSlopeNorm

from onesim.physics.magnetohydrodynamics.resistive.numerics.backends.equations.resistive_mhd_3d import (
    ResistiveMHD3DSolver,
)

print("OneSimulation:", version("onesimulation"))


# ## Parameters
#
# Dimensionless units with $\mu_0=1$: $\rho_0=1$, $B_0=1$ (so $v_A=1$), $\eta_H=0.5$ (so
# $d_i=0.5$), $\gamma=5/3$ and $c_s=0.5$ (plasma beta $2p_0/B_0^2=0.3$). The periodic box has the
# side $2\pi$, so the admissible wave vectors are integer triples $(k_x,k_y,k_z)$.
#
# Eight modes are launched on the whistler branch and six on the ion-cyclotron / kinetic-Alfvén
# branch, each with a magnetic amplitude of $10^{-7}B_0$ (small enough that the quadratic coupling
# between the harmonics $\mathbf k$, $2\mathbf k$, $3\mathbf k$ stays below the time-integration error): a family with $|\mathbf k|=3$ at
# $\theta=0^\circ$, $48.2^\circ$, $70.5^\circ$ and $90^\circ$, and families along $\theta=0^\circ$
# and $\theta=45^\circ$ with $|\mathbf k|$ between 1 and 4.24 ($kd_i$ up to 2.1).

rho_0, B_0, eta_H, gamma, c_s = 1.0, 1.0, 0.5, 5.0 / 3.0, 0.5
p_0 = rho_0 * c_s**2 / gamma
B0_vec = np.array([0.0, 0.0, B_0])
v_A, d_i = B_0 / np.sqrt(rho_0), eta_H / np.sqrt(rho_0)
amplitude = 1.0e-7                   # magnetic amplitude of every launched mode, in units of B_0
final_time = 4.0 * np.pi             # simulated time
sample_interval = 0.08               # time between two stored samples

WHISTLER, ION_CYCLOTRON = -1, -2     # index of the branch in the ascending list of positive roots
modes = [((0, 0, 1), WHISTLER), ((0, 0, 3), WHISTLER), ((1, 0, 1), WHISTLER), ((2, 0, 2), WHISTLER),
         ((3, 0, 3), WHISTLER), ((1, 2, 2), WHISTLER), ((2, 2, 1), WHISTLER), ((0, 3, 0), WHISTLER),
         ((0, 0, 2), ION_CYCLOTRON), ((0, 1, 1), ION_CYCLOTRON), ((0, 2, 2), ION_CYCLOTRON),
         ((0, 3, 3), ION_CYCLOTRON), ((2, 1, 2), ION_CYCLOTRON), ((2, -2, 1), ION_CYCLOTRON)]
branch_name = {WHISTLER: "whistler", ION_CYCLOTRON: "ion-cyclotron/KAW"}
print(f"v_A = {v_A:.3f}, d_i = {d_i:.3f}, c_s = {c_s:.3f}, beta = {2*p_0/B_0**2:.3f}")


# ## Geometry and mesh
#
# The domain is the cube $[0,2\pi)^3$, periodic in $x$, $y$ and $z$ on all six faces (the solver's
# only boundary condition). The grids have $N^3$ points with $N=8$, 16 and 32 for the spectral
# scheme and $N=16$, 32 and 64 for the difference scheme; the main run uses the spectral scheme
# on $32^3$ points. All fourteen modes are superposed, so the field is a genuinely
# three-dimensional interference pattern.

# ## Model
#
# **Linear theory.** For a plane wave $\propto e^{i\mathbf k\cdot\mathbf x}$ the linearised equations for
# $X=(\rho',\mathbf v,p',\mathbf b)$ read $\dot X=L(\mathbf k)X$ with
#
# $$\dot\rho'=-\rho_0\,i\mathbf k\cdot\mathbf v,\quad
#   \rho_0\dot{\mathbf v}=-i\mathbf k\,p'-i\mathbf k(\mathbf B_0\cdot\mathbf b)+i(\mathbf k\cdot\mathbf B_0)\mathbf b,\quad
#   \dot p'=-\gamma p_0\,i\mathbf k\cdot\mathbf v,$$
# $$\dot{\mathbf b}=i\mathbf k\times(\mathbf v\times\mathbf B_0)-\frac{\eta_H}{\rho_0}\,i\mathbf k\times\big[(i\mathbf k\times\mathbf b)\times\mathbf B_0\big].$$
#
# The eigenvalues of $L$ are $-i\omega$; the positive frequencies agree with the three roots of the
# closed-form relation above (both are printed below), and the eigenvectors supply the initial
# state of each launched mode. For the difference scheme every derivative of a Fourier mode is
# multiplied by $i\tilde{\mathbf k}$ instead of $i\mathbf k$, so its exact discrete frequency is
# $\omega(\tilde{\mathbf k})$ with $\tilde k_j=\sin(k_jh)/h$; the difference runs are launched with
# the eigenvectors of $L(\tilde{\mathbf k})$ so that each excites a single discrete branch.

def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=complex)


def linear_operator(k):
    ik = 1j * np.asarray(k, float)
    K, C = skew(ik), skew(B0_vec)
    L = np.zeros((8, 8), complex)
    L[0, 1:4] = -rho_0 * ik
    L[1:4, 4] = -ik / rho_0
    L[1:4, 5:8] = (-np.outer(ik, B0_vec) + (ik @ B0_vec) * np.eye(3)) / rho_0
    L[4, 1:4] = -gamma * p_0 * ik
    L[5:8, 1:4] = -K @ C                                   # curl(v x B0)
    L[5:8, 5:8] = (eta_H / rho_0) * K @ C @ K              # -(eta_H/rho0) curl((curl b) x B0)
    return L


def eigenmodes(k):
    """Positive frequencies (ascending) and eigenvectors of the linearised Hall-MHD system."""
    lam, vectors = np.linalg.eig(linear_operator(k))
    omega = (1j * lam).real
    keep = [i for i in np.argsort(omega) if omega[i] > 1e-9]
    return omega[keep], vectors[:, keep]


def dispersion_roots(k_abs, theta):
    """Positive roots of the closed-form oblique Hall-MHD dispersion relation (Hameiri et al. 2005)."""
    k2, kp2 = k_abs**2, (k_abs * np.cos(theta)) ** 2
    lhs = np.polymul([1.0, -kp2 * v_A**2], [1.0, -k2 * (v_A**2 + c_s**2), k2 * kp2 * v_A**2 * c_s**2])
    rhs = kp2 * k2 * v_A**2 * d_i**2 * np.array([1.0, -k2 * c_s**2, 0.0])
    w2 = np.roots(np.polysub(lhs, rhs))
    w2 = w2.real[(np.abs(w2.imag) < 1e-9 * np.abs(w2).max()) & (w2.real > 1e-14)]
    return np.sort(np.sqrt(w2))


def frame(k):
    """Unit vectors e1 (perpendicular to k and B0) and e2 (perpendicular to k, in the k-B0 plane)."""
    khat = np.asarray(k, float) / np.linalg.norm(k)
    e1 = np.cross(khat, B0_vec)
    e1 = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(e1) < 1e-12 else e1 / np.linalg.norm(e1)
    return e1, np.cross(khat, e1)


print(" mode (kx,ky,kz)   |k|    theta    branch               omega (eigenvalue)   omega (closed form)")
for k, branch in modes:
    kv = np.array(k, float)
    k_abs, theta = np.linalg.norm(kv), np.arccos(abs(kv[2]) / np.linalg.norm(kv))
    w_eig = eigenmodes(kv)[0][branch]
    w_closed = dispersion_roots(k_abs, theta)[branch]
    print(f" {str(k):15s} {k_abs:5.3f}  {np.degrees(theta):5.1f} deg  {branch_name[branch]:18s}"
          f"  {w_eig:12.7f}        {w_closed:12.7f}")


# **Runs.** Each run starts from the uniform state plus the real parts of the fourteen eigenvectors
# (normalised to $|\hat{\mathbf b}|=10^{-7}$). The conservative state is advanced in short
# segments of length `sample_interval`; after each segment the complex amplitude of every launched
# mode is extracted by projecting $\mathbf B-\mathbf B_0$ onto $e^{-i\mathbf k\cdot\mathbf x}$ and
# onto the launched polarisation vector. The frequency is the slope of the unwrapped phase of that
# amplitude over the whole run (least squares).

def run(N, scheme, time_step):
    solver = ResistiveMHD3DSolver(mesh_size=N, gamma=gamma, resistivity=0.0, hall_coefficient=eta_H,
                                  final_time=sample_interval, time_step=time_step, time_method="classical_rk4")
    solver.operators = dataclasses.replace(solver.operators, derivative_scheme=scheme)
    h = 2 * np.pi / N
    X = np.meshgrid(*solver.axes, indexing="ij")
    perturbation = np.zeros((8,) + X[0].shape)
    launched = []
    for k, branch in modes:
        kv = np.array(k, float)
        k_scheme = np.sin(kv * h) / h if scheme == "centered_fdm" else kv
        omega, vectors = eigenmodes(k_scheme)
        vector = vectors[:, branch] / np.linalg.norm(vectors[5:8, branch])
        wave = np.exp(1j * (kv[0] * X[0] + kv[1] * X[1] + kv[2] * X[2]))
        perturbation += amplitude * (vector[:, None, None, None] * wave).real
        launched.append(dict(k=kv, branch=branch, polarisation=vector[5:8], omega_discrete=omega[branch],
                             conj_wave=np.conj(wave)))
    density = rho_0 + perturbation[0]
    velocity = perturbation[1:4]
    magnetic = tuple(B0_vec[i] + perturbation[5 + i] for i in range(3))
    momentum = tuple(density * velocity[i] for i in range(3))
    energy = (p_0 + perturbation[4]) / (gamma - 1) + 0.5 * density * np.sum(velocity**2, axis=0) \
        + 0.5 * sum(b * b for b in magnetic)
    state = (density, momentum, energy, magnetic)

    def amplitudes(magnetic):
        b = [magnetic[i] - B0_vec[i] for i in range(3)]
        out = []
        for mode in launched:
            coefficient = np.array([np.mean(b[i] * mode["conj_wave"]) for i in range(3)])
            out.append(coefficient)
        return np.array(out)

    times, series = [0.0], [amplitudes(magnetic)]
    energy_total = [energy.sum()]
    div_b = 0.0
    start = time.perf_counter()
    t = 0.0
    for _ in range(int(round(final_time / sample_interval))):
        result = solver.run(state)
        state = (result.density, result.momentum, result.total_energy, result.magnetic_field)
        t += result.time
        times.append(t)
        series.append(amplitudes(result.magnetic_field))
        energy_total.append(result.total_energy.sum())
        div_b = max(div_b, float(np.max(np.abs(result.divergence))))
    elapsed = time.perf_counter() - start
    times, series = np.array(times), np.array(series)
    for m, mode in enumerate(launched):
        pol = mode["polarisation"]
        a = series[:, m, :] @ np.conj(pol) / (pol @ np.conj(pol))
        mode["omega"] = -np.polyfit(times, np.unwrap(np.angle(a)), 1)[0]
        mode["amplitude_ratio"] = abs(a[-1]) / abs(a[0])
        mode["signal"] = a
        e1, e2 = frame(mode["k"])
        ratio = (series[:, m, :] @ e1) / (series[:, m, :] @ e2)
        mode["polarisation_measured"] = ratio
        del mode["conj_wave"]
    steps = int(round(final_time / time_step))
    print(f"{'spectral' if scheme == 'spectral' else 'central differences':19s} N = {N:2d}, dt = {time_step:.3f}:"
          f" {steps:5d} RK4 steps in {elapsed:6.1f} s")
    final = dict(magnetic=[state[3][i] - B0_vec[i] for i in range(3)], axes=solver.axes)
    return dict(N=N, scheme=scheme, dt=time_step, h=h, times=times, modes=launched,
                energy_drift=np.max(np.abs(np.array(energy_total) / energy_total[0] - 1)), div_b=div_b,
                final=final)


# ## Solve
#
# Main run: spectral, $32^3$, $\Delta t=0.008$. Resolution study: spectral on $8^3$ and $16^3$ with
# the same step, central differences on $16^3$, $32^3$, $64^3$ with $\Delta t=0.02$. Time-step
# study: spectral $16^3$ with $\Delta t=0.04$, 0.02, 0.01.

main = run(32, "spectral", 0.008)
spectral_study = [run(8, "spectral", 0.008), run(16, "spectral", 0.008), main]
time_study = [run(16, "spectral", dt) for dt in (0.04, 0.02, 0.01)]
fd_study = [run(N, "centered_fdm", 0.02) for N in (16, 32, 64)]


# ## Comparison with the reference
#
# **Frequencies** of the main run (spectral, $32^3$, $\Delta t=0.008$) against the exact
# dispersion relation, with the polarisation $b_1/b_2$ in the frame $e_1\perp(\mathbf k,\mathbf B_0)$,
# $e_2\perp\mathbf k$ in the $(\mathbf k,\mathbf B_0)$ plane: a purely imaginary ratio
# $\pm i\,|b_1/b_2|$ means elliptical polarisation, its sign gives the sense of rotation, and the
# value is printed as measured (time average) and from the eigenvector.

def exact_omega(mode):
    return eigenmodes(mode["k"])[0][mode["branch"]]


print(" mode         theta   branch               omega measured   omega exact     rel. dev.  "
      " b1/b2 measured        b1/b2 theory       |a(T)/a(0)|-1")
for mode in main["modes"]:
    k = mode["k"]
    theta = np.degrees(np.arccos(abs(k[2]) / np.linalg.norm(k)))
    w_ref = exact_omega(mode)
    e1, e2 = frame(k)
    pol_theory = (mode["polarisation"] @ e1) / (mode["polarisation"] @ e2)
    pol = mode["polarisation_measured"].mean()
    print(f" {str(tuple(int(v) for v in k)):11s} {theta:5.1f}  {branch_name[mode['branch']]:18s}"
          f"  {mode['omega']:13.8f}  {w_ref:13.8f}   {mode['omega']/w_ref-1:+.2e}"
          f"   {pol.real:+.4f}{pol.imag:+.4f}i   {pol_theory.real:+.4f}{pol_theory.imag:+.4f}i"
          f"   {mode['amplitude_ratio']-1:+.1e}")
spread = max(np.max(np.abs(m["polarisation_measured"] - m["polarisation_measured"].mean())) /
             (1 + abs(m["polarisation_measured"].mean())) for m in main["modes"])
print(f"\nlargest change of b1/b2 during the run, relative to 1 + |b1/b2|: {spread:.1e}")
print(f"total energy: largest relative change {main['energy_drift']:.1e};  max |div B| = {main['div_b']:.1e}")

# **Resolution study.** The spectral derivative is exact for every resolved Fourier mode, so the
# spectral frequencies should not change with $N$ (all three spectral runs use $\Delta t=0.008$;
# what remains is the RK4 time error). The central-difference frequencies converge to
# the exact value with second order and must equal the modified-wavenumber prediction
# $\omega(\tilde{\mathbf k})$ on each grid. From the three grids $N=16,32,64$ the observed order is
# $p=\log_2[(\omega_{16}-\omega_{32})/(\omega_{32}-\omega_{64})]$ and the Richardson estimate
# $\omega_{64}+(\omega_{64}-\omega_{32})/(2^p-1)$, which uses no knowledge of the exact value.
# The difference runs use $\Delta t=0.02$, so their agreement with $\omega(\tilde{\mathbf k})$ is
# limited by the RK4 time error of a few $10^{-6}$ (see the time-step study).

print("\nlargest relative deviation from the exact frequency over all 14 modes")
for run_ in spectral_study:
    dev = max(abs(m["omega"] / exact_omega(m) - 1) for m in run_["modes"])
    print(f"  spectral            N = {run_['N']:2d}: {dev:.2e}")
for run_ in fd_study:
    dev = max(abs(m["omega"] / exact_omega(m) - 1) for m in run_["modes"])
    pred = max(abs(m["omega"] / m["omega_discrete"] - 1) for m in run_["modes"])
    print(f"  central differences N = {run_['N']:2d}: {dev:.2e}   (vs modified-wavenumber prediction: {pred:.1e})")

print("\ncentral differences, per mode: omega on N = 16, 32, 64, observed order, Richardson estimate vs exact")
for m in range(len(modes)):
    w = np.array([r["modes"][m]["omega"] for r in fd_study])
    w_ref = exact_omega(fd_study[0]["modes"][m])
    p = np.log2((w[0] - w[1]) / (w[1] - w[2]))
    rich = w[2] + (w[2] - w[1]) / (2**p - 1)
    print(f"  {str(modes[m][0]):11s} {branch_name[modes[m][1]]:18s} " + "  ".join(f"{v:.6f}" for v in w)
          + f"   p = {p:.2f}   Richardson {rich:.6f} ({rich/w_ref-1:+.1e})   exact {w_ref:.6f}")

# **Time-step study** (spectral, $16^3$): the error of RK4 must fall with the fourth power of
# $\Delta t$ until it reaches the level set by the small nonlinearity of the waves.

print("\nspectral 16^3: largest relative frequency deviation over all modes")
errors = []
for run_ in time_study:
    errors.append(max(abs(m["omega"] / exact_omega(m) - 1) for m in run_["modes"]))
for i, run_ in enumerate(time_study):
    order = f"   observed order {np.log2(errors[i-1]/errors[i]):.2f}" if i else ""
    print(f"  dt = {run_['dt']:.3f}: {errors[i]:.2e}{order}")
fastest = max(range(len(modes)), key=lambda m: exact_omega(main["modes"][m]))
print(f"  (dominated by the fastest mode {modes[fastest][0]}, omega dt = "
      f"{exact_omega(main['modes'][fastest])*time_study[0]['dt']:.2f} at the largest step)")


# ## Figure
#
# Top: two slices of the computed magnetic perturbation at $t=4\pi$ (spectral, $32^3$, shown
# with spectral interpolation): $b_x$ in the plane $z=0$ perpendicular to $\mathbf B_0$ and $b_y$
# in the plane $y=0$ containing $\mathbf B_0$, and the time series of two launched modes against
# $e^{-i\omega t}$ from the dispersion relation. Bottom: the dispersion relation, frequency
# against the angle $\theta$ for $|\mathbf k|=3$ and against $|\mathbf k|$ for $\theta=0^\circ$
# and $45^\circ$ (lines: closed form, symbols: measured), and the resolution and time-step
# studies.

def upsample(field2d, n):
    """Trigonometric interpolation of a periodic 2D slice onto n x n points."""
    N = field2d.shape[0]
    spectrum = np.fft.fftshift(np.fft.fft2(field2d))
    padded = np.zeros((n, n), complex)
    lo = n // 2 - N // 2
    padded[lo:lo + N, lo:lo + N] = spectrum
    return np.fft.ifft2(np.fft.ifftshift(padded)).real * (n / N) ** 2


fig = plt.figure(figsize=(16, 9.2), constrained_layout=True)
grid = fig.add_gridspec(2, 4)
n_fine = 160
s = np.linspace(0, 2 * np.pi, n_fine, endpoint=False)
b_final = main["final"]["magnetic"]
for col, (data, labels, title) in enumerate((
        (b_final[0][:, :, 0], ("x", "y"), "b_x on the plane z = 0, normal to B_0"),
        (b_final[1][:, 0, :], ("x", "z"), "b_y on the plane y = 0, containing B_0"))):
    ax = fig.add_subplot(grid[0, col])
    fine = upsample(data, n_fine) / amplitude
    limit = np.max(np.abs(fine))
    image = ax.pcolormesh(s, s, fine.T, cmap=DIVERGING_CMAP, norm=TwoSlopeNorm(0.0, -limit, limit), shading="auto")
    ax.set_aspect("equal")
    ax.set(xlabel=labels[0], ylabel=labels[1], title=title)
    if col == 1:
        ax.annotate("", xy=(0.92, 0.9), xytext=(0.92, 0.65), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
        ax.text(0.86, 0.77, "B_0", transform=ax.transAxes, color=INK, ha="right")
    fig.colorbar(image, ax=ax, shrink=0.85, label="b / (10^-7 B_0)")

ax = fig.add_subplot(grid[0, 2:4])
for m, colour, every in ((6, COPPER, 1), (13, GREEN, 2)):
    mode = main["modes"][m]
    w_ref = exact_omega(mode)
    a = mode["signal"] / mode["signal"][0]
    t_fine = np.linspace(0, final_time, 1200)
    ax.plot(t_fine, np.cos(w_ref * t_fine), "-", color=colour, lw=1.2)
    ax.plot(main["times"][::every], a.real[::every], "o", ms=3.0, color=colour,
            label=f"k = {tuple(int(v) for v in mode['k'])}, {branch_name[mode['branch']]}, omega = {mode['omega']:.4f}")
ax.set(xlabel="time t", ylabel="Re a(t) / a(0)", title="Launched-mode amplitudes (symbols) and cos(omega t) from the dispersion relation")
ax.legend(loc="lower left", fontsize=8.5)
ax.grid(alpha=0.3)

ax = fig.add_subplot(grid[1, 0])
thetas = np.linspace(0.0, np.pi / 2, 181)
roots = [dispersion_roots(3.0, th) for th in thetas]
for b, (colour, label) in enumerate(((COPPER, "fast / whistler"), (GREEN, "ion-cyclotron / KAW"), ("#8B9A8D", "slow"))):
    ax.plot(np.degrees(thetas), [r[-1 - b] if len(r) > b else np.nan for r in roots], color=colour, lw=1.6, label=label)
for mode in main["modes"]:
    if abs(np.linalg.norm(mode["k"]) - 3.0) < 1e-9:
        theta = np.degrees(np.arccos(abs(mode["k"][2]) / 3.0))
        ax.plot(theta, mode["omega"], "o" if mode["branch"] == WHISTLER else "s", color=INK, ms=7, mfc="none")
ax.set(xlabel="angle theta between k and B_0 [deg]", ylabel="omega", title="|k| = 3: frequency vs angle")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = fig.add_subplot(grid[1, 1])
k_line = np.linspace(0.05, 4.6, 200)
for theta_deg, style in ((0.0, "-"), (45.0, "--")):
    th = np.radians(theta_deg)
    r = np.array([dispersion_roots(k, th) for k in k_line])
    ax.plot(k_line, r[:, -1], style, color=COPPER, lw=1.5, label=f"whistler, theta = {theta_deg:.0f} deg")
    if theta_deg == 0.0:
        ion = k_line * v_A * (np.sqrt(1 + (k_line * d_i / 2) ** 2) - k_line * d_i / 2)
        ax.plot(k_line, ion, style, color=GREEN, lw=1.5, label="ion-cyclotron, theta = 0 deg")
    else:
        ax.plot(k_line, r[:, -2], style, color=GREEN, lw=1.5, label=f"ion-cyclotron/KAW, theta = {theta_deg:.0f} deg")
ax.plot(k_line, k_line * v_A, ":", color=INK, lw=1, label="Alfven omega = k v_A")
for mode in main["modes"]:
    k_abs = np.linalg.norm(mode["k"])
    theta = np.degrees(np.arccos(abs(mode["k"][2]) / k_abs))
    if min(abs(theta), abs(theta - 45.0)) < 1e-6:
        ax.plot(k_abs, mode["omega"], "o" if mode["branch"] == WHISTLER else "s", color=INK, ms=7, mfc="none")
ax.set(xlabel="|k|", ylabel="omega", title="Frequency vs |k| (symbols: measured)")
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(alpha=0.3)

ax = fig.add_subplot(grid[1, 2])
hs = np.array([r["h"] for r in fd_study])
for m, marker in ((4, "o-"), (6, "s-"), (11, "^-")):
    ax.loglog(hs, [abs(r["modes"][m]["omega"] / exact_omega(r["modes"][m]) - 1) for r in fd_study], marker,
              color=COPPER, mfc="none", label=f"central diff., k = {modes[m][0]}")
spec_h = np.array([r["h"] for r in spectral_study])
ax.loglog(spec_h, [max(abs(m["omega"] / exact_omega(m) - 1) for m in r["modes"]) for r in spectral_study], "D-",
          color=GREEN, label="spectral, all modes (max), dt = 0.008")
first = abs(fd_study[0]["modes"][4]["omega"] / exact_omega(fd_study[0]["modes"][4]) - 1)
ax.loglog(hs, first * (hs / hs[0]) ** 2, ":", color=INK, lw=1, label="slope 2")
ax.set_xticks([2 * np.pi / n for n in (64, 32, 16, 8)], ["2pi/64", "2pi/32", "2pi/16", "2pi/8"])
ax.minorticks_off()
ax.set(xlabel="grid spacing h", ylabel="|omega / omega_exact - 1|", title="Resolution study")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.3, which="both")

ax = fig.add_subplot(grid[1, 3])
dts = np.array([r["dt"] for r in time_study])
ax.loglog(dts, errors, "o-", color=COPPER, label="spectral 16^3, max over modes")
ax.loglog(dts, errors[0] * (dts / dts[0]) ** 4, ":", color=INK, lw=1, label="slope 4")
ax.set(xlabel="time step dt", ylabel="|omega / omega_exact - 1|", title="Time-step study (RK4)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
plt.show()


# ## Going further
#
# Raise the amplitude to $10^{-2}B_0$ to see the nonlinear frequency shift and the transfer of
# energy between modes that share wave-vector triads, or set `eta_H = 0` to recover the ideal
# Alfvén and magnetosonic waves (the whistler and ion-cyclotron branches then merge into the
# shear-Alfvén wave $\omega=k_\parallel v_A$). A finite `resistivity` adds the damping rate
# $\eta k^2/2$ to every mode, which the same phase-and-amplitude fit measures directly.
