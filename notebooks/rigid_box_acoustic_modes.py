"""OneSimulation example: acoustic eigenmodes of a closed rigid rectangular box.
Requires the installed, activated OneSimulation runtime.

The first twenty standing pressure waves of an air-filled 1.0 x 0.7 x 0.5 m box with
sound-hard walls - axial, tangential and oblique modes, three of them degenerate pairs -
are computed as a generalized eigenvalue problem with high-order hexahedral finite
elements (Q1 to Q4) and with unstructured tetrahedra, and graded against the exact
frequencies and mode shapes.
"""


# # Acoustic modes of a rigid rectangular box
#
# A closed room or enclosure filled with air only "rings" at discrete frequencies: its
# acoustic eigenmodes. For small-amplitude sound the pressure $p(\mathbf x)\,e^{i\omega t}$
# obeys the Helmholtz equation
#
# $$\nabla\cdot\Big(\frac{1}{\rho_0}\nabla p\Big)+\frac{\omega^2}{\rho_0 c^2}\,p=0
#   \quad\text{in the box},\qquad \frac{\partial p}{\partial n}=0\quad\text{on all six walls},$$
#
# where the wall condition says that a rigid (sound-hard) wall does not move: the normal
# particle velocity $v_n=-\partial_n p/(i\omega\rho_0)$ vanishes. The box measures
# $L_x\times L_y\times L_z=1.0\times0.7\times0.5$ m and holds air at 20 °C
# ($c=343$ m/s, $\rho_0=1.204$ kg/m³).
#
# | boundary | condition |
# |---|---|
# | walls $x=0$ and $x=L_x$ | rigid, $\partial p/\partial n=0$ |
# | walls $y=0$ and $y=L_y$ | rigid, $\partial p/\partial n=0$ |
# | walls $z=0$ and $z=L_z$ | rigid, $\partial p/\partial n=0$ |
#
# **Reference (exact solution of the 3D problem).** The eigenmodes separate into
# products of cosines (Lord Rayleigh, *The Theory of Sound*, Vol. II, §267, 1896;
# P. M. Morse and K. U. Ingard, *Theoretical Acoustics*, McGraw-Hill 1968, §9.4),
#
# $$p_{lmn}=\cos\frac{l\pi x}{L_x}\,\cos\frac{m\pi y}{L_y}\,\cos\frac{n\pi z}{L_z},\qquad
#   f_{lmn}=\frac{c}{2}\sqrt{\Big(\frac{l}{L_x}\Big)^2+\Big(\frac{m}{L_y}\Big)^2+\Big(\frac{n}{L_z}\Big)^2}.$$
#
# Modes with one non-zero index are *axial*, with two *tangential* and with three
# *oblique*. Because $L_x=2L_z$, the modes $(2,0,0)$ and $(0,0,1)$ have exactly the same
# frequency (343 Hz), and so do $(2,1,0)$/$(0,1,1)$ and $(2,2,0)$/$(0,2,1)$: there the
# eigenvalue is double and every combination of the two shapes is an eigenmode too. The
# first twenty non-zero modes reach up to 665 Hz; the twenty-first lies at 686 Hz.
#
# **Numerical method.**
# - Spatial: continuous Galerkin finite elements. Primary: tensor-product Lagrange
#   elements $Q_p$ of degree $p=1,2,3,4$ on hexahedra; cross-check: Lagrange $P_p$
#   ($p=1,2,3$) on an unstructured tetrahedral mesh from OneSimulation's CAD mesher.
#   The element stiffness $\int\rho_0^{-1}\nabla\phi_i\cdot\nabla\phi_j$ and the
#   compressibility mass $\int(\rho_0c^2)^{-1}\phi_i\phi_j$ come from OneSimulation's
#   acoustic Helmholtz element kernel on the quadrature of each finite-element space.
# - The rigid walls are the natural boundary condition of this weak form: no wall
#   term enters the matrices, and the computed modes are checked for
#   $\partial p/\partial n\approx 0$ afterwards.
# - Eigenproblem: the generalized symmetric eigenvalue problem $K\mathbf p=\omega^2M\mathbf p$
#   is solved with shift-and-invert Lanczos iterations (OneSimulation's sparse symmetric
#   eigensolver, sparse LU of $K-\sigma M$ with a small negative shift $\sigma$). The
#   constant pressure mode at $\omega=0$ is the expected null mode of a closed box.
# - Time integration: none (eigenvalue problem).
#
# **Discretisation error.** For a conforming Galerkin method the computed eigenvalues
# lie above the exact ones and their error falls like $h^{2p}$ (I. Babuška and
# J. Osborn, "Eigenvalue problems", *Handbook of Numerical Analysis* II, 1991). In wave
# terms this is the *dispersion error*: on a given mesh the error grows quickly with
# frequency, i.e. with fewer elements per wavelength, and high-order elements push it
# down by orders of magnitude at the same number of unknowns.

from importlib.metadata import version
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
from matplotlib.colors import Normalize
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, splu

from onesim.cad.d3 import Box
from onesim.meshing import StructuredHexahedralMesher, TetrahedralMesher
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear.kernels.algorithms.sparse_symmetric_eigen import SparseSymmetricEigenSolver
from onesim.physics.acoustics import pressure_eigenmodes, pressure_helmholtz_element

print("OneSimulation:", version("onesimulation"))
start_time = time.perf_counter()


# ## Parameters

Lx, Ly, Lz = 1.0, 0.7, 0.5          # m, box dimensions
c0 = 343.0                          # m/s, speed of sound in air at 20 C
rho0 = 1.204                        # kg/m^3, air density at 20 C
compressibility = 1.0 / (rho0 * c0**2)   # 1/Pa
mode_count = 20                     # graded non-zero modes

# Mesh study: element size h = 0.1 m / k for refinement level k, per polynomial degree p.
hex_levels = {1: (1, 2, 3, 4), 2: (1, 2, 3), 3: (1, 2), 4: (1,)}
tet_size = 0.1                      # m, target size of the unstructured tetrahedral mesh
tet_degrees = (1, 2, 3)

# Exact modes, sorted by frequency.
exact = sorted(
    (c0 / 2 * np.sqrt((l / Lx) ** 2 + (m / Ly) ** 2 + (n / Lz) ** 2), (l, m, n))
    for l in range(8) for m in range(8) for n in range(8) if l + m + n
)
f_exact = np.array([f for f, _ in exact[:mode_count]])
indices = [lmn for _, lmn in exact[:mode_count]]
kind = {1: "axial", 2: "tangential", 3: "oblique"}
# Degenerate groups: modes with the same exact frequency.
group_of = [int(np.flatnonzero(np.isclose(f_exact, f, rtol=1e-12, atol=0))[0]) for f in f_exact]
print(f"first {mode_count} exact modes: {f_exact[0]:.2f} ... {f_exact[-1]:.2f} Hz; "
      f"next mode {exact[mode_count][0]:.2f} Hz {exact[mode_count][1]}")
print("degenerate pairs:", [(indices[i], indices[j]) for j, i in enumerate(group_of) if i != j])


def exact_shape(lmn, points):
    l, m, n = lmn
    x, y, z = np.asarray(points).T
    return np.cos(l * np.pi * x / Lx) * np.cos(m * np.pi * y / Ly) * np.cos(n * np.pi * z / Lz)


# ## Geometry and mesh
#
# The air volume is a `Box`. The hexahedral meshes are uniform with cubic cells of side
# $h=0.1\,\mathrm m/k$ (10 x 7 x 5 cells at $k=1$, 40 x 28 x 20 at $k=4$); the element
# degree $p$ adds $p-1$ nodes per cell edge. The tetrahedral mesh is generated by
# OneSimulation's CAD mesher with target size 0.1 m. It is genuinely unstructured, so its
# discrete operator is not a tensor product of one-dimensional ones: it tests the 3D
# discretisation without any help from the box's separable shape.

geometry = Box(length=Lx, width=Ly, height=Lz)


def hex_mesh(level):
    return StructuredHexahedralMesher(cells=(10 * level, 7 * level, 5 * level)).generate(geometry)


tet_mesh = TetrahedralMesher(target_size=tet_size).generate(geometry)
print(f"unstructured tetrahedral mesh: {tet_mesh.cell_count} cells")


# ## Model
#
# Global stiffness $K$ and compressibility mass $M$ are assembled cell by cell from the
# Lagrange space bound to the mesh; `pressure_helmholtz_element` evaluates
# $\int\rho_0^{-1}\nabla\phi_i\cdot\nabla\phi_j\,\mathrm dV$ and
# $\int(\rho_0c^2)^{-1}\phi_i\phi_j\,\mathrm dV$ on the space's own quadrature rule. The
# rigid walls need no boundary term (natural condition, see the parameters table above).

def acoustic_matrices(mesh, degree):
    space = FiniteElement(element=Lagrange(order=degree)).bind_space(mesh)
    n, local = space.dof_count, space.cell_dofs.shape[1]
    rows, cols, k_values, m_values = [], [], [], []
    for cell, dofs in enumerate(space.cell_dofs):
        _, weights, (basis, gradients, _) = space.element_data(cell)
        k_cell, m_cell = pressure_helmholtz_element(
            basis, gradients, weights, density=rho0, compressibility=compressibility)
        rows.append(np.repeat(dofs, local))
        cols.append(np.tile(dofs, local))
        k_values.append(np.asarray(k_cell).ravel())
        m_values.append(np.asarray(m_cell).ravel())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    K = sp.csr_matrix((np.concatenate(k_values), (rows, cols)), shape=(n, n))
    M = sp.csr_matrix((np.concatenate(m_values), (rows, cols)), shape=(n, n))
    return space, K, M


def shift_invert(stiffness, mass, shift):
    factor = splu((stiffness - shift * mass).tocsc())
    return LinearOperator(stiffness.shape, matvec=factor.solve)


eigensolver = SparseSymmetricEigenSolver(shift_inverse=shift_invert)


def acoustic_modes(mesh, degree):
    started = time.perf_counter()
    space, K, M = acoustic_matrices(mesh, degree)
    # Shift sigma = -1 rad^2/s^2 sits just below the null mode; the 21 lowest pairs are
    # the constant mode plus the 20 graded modes.
    frequencies, vectors = pressure_eigenmodes(
        K, M, zero_tolerance=1e-3,
        eigensolver=lambda mass, stiffness: eigensolver(mass, stiffness, mode_count=mode_count + 1,
                                                        target=-1.0))
    return dict(space=space, M=M, f=frequencies[1:], null=frequencies[0], modes=vectors[:, 1:],
                dofs=space.dof_count, seconds=time.perf_counter() - started)


# ## Solve

hex_runs = {}
for degree, levels in hex_levels.items():
    for level in levels:
        run = acoustic_modes(hex_mesh(level), degree)
        hex_runs[degree, level] = run
        print(f"Q{degree}, h = {10 / level:5.2f} cm: {run['dofs']:6d} unknowns, null mode {run['null']:.1e} Hz, "
              f"max error {np.max(np.abs(run['f'] / f_exact - 1)):.2e}, {run['seconds']:.1f} s")
tet_runs = {}
for degree in tet_degrees:
    run = acoustic_modes(tet_mesh, degree)
    tet_runs[degree] = run
    print(f"P{degree} tetrahedra: {run['dofs']:6d} unknowns, null mode {run['null']:.1e} Hz, "
          f"max error {np.max(np.abs(run['f'] / f_exact - 1)):.2e}, {run['seconds']:.1f} s")
final = hex_runs[3, 2]


# ## Comparison with the reference
#
# **Mode shapes (MAC).** The modal assurance criterion compares a computed mode
# $\mathbf p_h$ with the exact shape $\mathbf q$ sampled at the nodes, in the mass inner
# product: $\mathrm{MAC}=(\mathbf p_h^\top M\mathbf q)^2/
# \big((\mathbf p_h^\top M\mathbf p_h)(\mathbf q^\top M\mathbf q)\big)$, 1 for identical
# shapes. For a degenerate pair the solver may return any rotation of the two shapes, so
# the criterion measures the fraction of $\mathbf p_h$ that lies in the span of both exact
# shapes (subspace MAC).

def subspace_mac(run):
    space, M, V = run["space"], run["M"], run["modes"]
    exact_nodal = np.column_stack([exact_shape(lmn, space.dof_coordinates) for lmn in indices])
    mac = np.empty(mode_count)
    for j in range(mode_count):
        members = [i for i in range(mode_count) if group_of[i] == group_of[j]]
        Q = exact_nodal[:, members]
        gram = Q.T @ (M @ Q)
        projection = Q.T @ (M @ V[:, j])
        mac[j] = projection @ np.linalg.solve(gram, projection) / (V[:, j] @ (M @ V[:, j]))
    return mac


mac_final, mac_tet = subspace_mac(final), subspace_mac(tet_runs[3])
# Three discretisations with exactly the same number of unknowns (24,969): Q1 with
# h = 2.5 cm, Q2 with h = 5 cm and Q4 with h = 10 cm.
same = {1: hex_runs[1, 4], 2: hex_runs[2, 2], 4: hex_runs[4, 1]}
print(f"\nFinal mesh: Q3 hexahedra, h = 5 cm, {final['dofs']} unknowns; P3 tetrahedra: {tet_runs[3]['dofs']} unknowns;"
      f" equal-cost comparison Q1/Q2/Q4 with {same[1]['dofs']} unknowns each")
print(" #   (l,m,n)   type         exact [Hz]   Q3 [Hz]       rel. dev.  | Q1        Q2        Q4       "
      "| P3-tet    | 1-MAC Q3  1-MAC P3-tet")
for j in range(mode_count):
    lmn = indices[j]
    print(f"{j + 1:2d}   {str(lmn):9s} {kind[np.count_nonzero(lmn)]:11s} {f_exact[j]:10.4f}   {final['f'][j]:11.6f}"
          f"   {final['f'][j] / f_exact[j] - 1:+.2e} | "
          + "  ".join(f"{same[p]['f'][j] / f_exact[j] - 1:+.1e}" for p in (1, 2, 4))
          + f" | {tet_runs[3]['f'][j] / f_exact[j] - 1:+.1e} | {1 - mac_final[j]:.1e}   {1 - mac_tet[j]:.1e}")
print(f"smallest MAC: Q3 hexahedra {mac_final.min():.10f}, P3 tetrahedra {mac_tet.min():.10f}")

# **Splitting of the degenerate pairs.** On the tensor-product hexahedral mesh the two
# members of a pair remain exactly degenerate; on the unstructured tetrahedra the
# discrete symmetry is broken and the pair splits slightly - by much less than the
# discretisation error itself.
for j, i in enumerate(group_of):
    if i != j:
        print(f"pair {indices[i]}/{indices[j]} at {f_exact[j]:.3f} Hz: splitting Q3-hex "
              f"{abs(final['f'][j] - final['f'][i]):.1e} Hz, P1/P2/P3-tet "
              + ", ".join(f"{abs(tet_runs[p]['f'][j] - tet_runs[p]['f'][i]):.1e}" for p in tet_degrees) + " Hz")

# **Mesh convergence and observed order.** Error measure: the largest relative
# frequency error over the twenty modes. With $h$ halved the error should fall by
# $2^{2p}$, i.e. the observed order of the eigenvalue error is $2p$ (frequency error
# $\propto h^{2p}$ as well, since $f\propto\sqrt{\lambda}$).
print("\nConvergence of the largest relative frequency error over the 20 modes")
for degree, levels in hex_levels.items():
    errors = [np.max(np.abs(hex_runs[degree, k]["f"] / f_exact - 1)) for k in levels]
    text = "  ".join(f"h = {10 / k:.2f} cm: {e:.3e}" for k, e in zip(levels, errors))
    print(f"Q{degree}: {text}")
    if len(levels) > 1:
        orders = [np.log(errors[i] / errors[i + 1]) / np.log(levels[i + 1] / levels[i]) for i in range(len(levels) - 1)]
        last, prev = hex_runs[degree, levels[-1]]["f"], hex_runs[degree, levels[-2]]["f"]
        print(f"     observed order(s) {', '.join(f'{o:.2f}' for o in orders)} (theory {2 * degree});"
              f" largest change between the last two meshes {np.max(np.abs(last / prev - 1)):.2e}")
# Richardson estimate for Q3 from the two levels with the theoretical order 6.
f_rich = final["f"] + (final["f"] - hex_runs[3, 1]["f"]) / (2**6 - 1)
print(f"Q3 Richardson estimate (order 6): largest deviation from exact {np.max(np.abs(f_rich / f_exact - 1)):.1e}")

# **Wall condition.** The rigid-wall condition is not imposed but follows from the weak
# form. The normal pressure gradient on the six walls, relative to the RMS pressure
# gradient in the volume, shows how well the computed modes satisfy it.
space = final["space"]
grid = np.linspace(0.02, 0.98, 25)
U, W = np.meshgrid(grid, grid)
u, w = U.ravel(), W.ravel()
eps = 1e-9
walls = {
    "x = 0": (np.column_stack((np.full_like(u, eps), u * Ly, w * Lz)), 0),
    "x = Lx": (np.column_stack((np.full_like(u, Lx - eps), u * Ly, w * Lz)), 0),
    "y = 0": (np.column_stack((u * Lx, np.full_like(u, eps), w * Lz)), 1),
    "y = Ly": (np.column_stack((u * Lx, np.full_like(u, Ly - eps), w * Lz)), 1),
    "z = 0": (np.column_stack((u * Lx, w * Ly, np.full_like(u, eps))), 2),
    "z = Lz": (np.column_stack((u * Lx, w * Ly, np.full_like(u, Lz - eps))), 2),
}
interior = np.column_stack([g.ravel() for g in np.meshgrid(np.linspace(0.05, 0.95, 9) * Lx,
                                                            np.linspace(0.05, 0.95, 9) * Ly,
                                                            np.linspace(0.05, 0.95, 9) * Lz)])
worst = 0.0
for j in range(mode_count):
    values = final["modes"][:, j]
    scale = np.sqrt(np.mean(np.sum(np.asarray(space.gradient(values, interior)) ** 2, axis=-1)))
    for points, axis in walls.values():
        normal = np.asarray(space.gradient(values, points))[:, axis]
        worst = max(worst, np.sqrt(np.mean(normal**2)) / scale)
print(f"\nRMS normal pressure gradient on the walls / RMS gradient in the volume, worst of 20 modes x 6 walls: {worst:.1e}")
print(f"Total run time: {time.perf_counter() - start_time:.0f} s")


# ## Figure
#
# Top and bottom left: computed pressure modes on the final Q3 mesh, shown on the three
# visible walls of the box ($x=L_x$, $y=0$, $z=L_z$); teal and copper are opposite signs. For the degenerate pair at 343 Hz the
# computed two-dimensional eigenspace is rotated onto the exact shapes $(2,0,0)$ and
# $(0,0,1)$. Right: convergence of the largest frequency error with the number of
# unknowns, and the dispersion error over the twenty modes on the coarsest mesh.

def mode_on_faces(run, vector, n=41):
    """Pressure of a nodal vector on the three visible walls, normalised to max |p| = 1."""
    space = run["space"]
    s, t = np.meshgrid(np.linspace(1e-6, 1 - 1e-6, n), np.linspace(1e-6, 1 - 1e-6, n))
    faces = {
        "x": (np.full_like(s, Lx * (1 - 1e-9)), s * Ly, t * Lz),
        "y": (s * Lx, np.full_like(s, 1e-9 * Ly), t * Lz),
        "z": (s * Lx, t * Ly, np.full_like(s, Lz * (1 - 1e-9))),
    }
    values = {key: np.asarray(space.evaluate(vector, np.column_stack([c.ravel() for c in xyz]))).reshape(s.shape)
              for key, xyz in faces.items()}
    peak = max(np.max(np.abs(v)) for v in values.values())
    corner = np.asarray(space.evaluate(vector, np.array([[Lx * (1 - 1e-9), 1e-9, Lz * (1 - 1e-9)]])))[0]
    sign = 1.0 if corner >= 0 else -1.0
    return faces, {k: sign * v / peak for k, v in values.items()}


def rotated_pair(run, members):
    """Rotate the computed degenerate eigenspace onto the exact shapes of the pair."""
    space, M, V = run["space"], run["M"], run["modes"][:, members]
    out = []
    for i in members:
        q = exact_shape(indices[i], space.dof_coordinates)
        out.append(V @ np.linalg.solve(V.T @ (M @ V), V.T @ (M @ q)))
    return out


shown = [indices.index(lmn) for lmn in ((1, 1, 0), (1, 1, 1), (2, 0, 0), (0, 0, 1), (2, 1, 1), (3, 1, 1))]
pair = [indices.index((0, 0, 1)), indices.index((2, 0, 0))]
pair_vectors = dict(zip(pair, rotated_pair(final, pair)))

norm = Normalize(-1.0, 1.0)
fig = plt.figure(figsize=(17, 9.6), constrained_layout=True)
layout = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1, 1.25))
for k, j in enumerate(shown):
    ax = fig.add_subplot(layout[k // 3, k % 3], projection="3d")
    ax.set_facecolor(CREAM)
    vector = pair_vectors.get(j, final["modes"][:, j])
    faces, values = mode_on_faces(final, vector)
    for key in ("x", "y", "z"):
        ax.plot_surface(*faces[key], facecolors=DIVERGING_CMAP(norm(values[key])), rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False)
    for xs, ys, zs in (([0, Lx], [0, 0], [Lz, Lz]), ([Lx, Lx], [0, Ly], [Lz, Lz]), ([Lx, Lx], [0, 0], [0, Lz]),
                       ([0, Lx], [Ly, Ly], [Lz, Lz]), ([0, 0], [0, Ly], [Lz, Lz]), ([0, Lx], [0, 0], [0, 0]),
                       ([Lx, Lx], [0, Ly], [0, 0]), ([0, 0], [0, 0], [0, Lz]), ([Lx, Lx], [Ly, Ly], [0, Lz])):
        ax.plot(xs, ys, zs, color=INK, lw=0.7)
    ax.set_box_aspect((Lx, Ly, Lz))
    ax.view_init(elev=24, azim=-58)
    ax.set_xticks([0, 0.5, 1.0]), ax.set_yticks([0, 0.35, 0.7]), ax.set_zticks([0, 0.25, 0.5])
    ax.tick_params(labelsize=7, pad=0)
    ax.set_xlabel("x / m", fontsize=8, labelpad=-4), ax.set_ylabel("y / m", fontsize=8, labelpad=-4)
    ax.set_zlabel("z / m", fontsize=8, labelpad=-2)
    lmn = indices[j]
    label = kind[np.count_nonzero(lmn)] + (", degenerate pair" if group_of[j] != j or j in group_of[j + 1:] else "")
    ax.set_title(f"mode {lmn}: {final['f'][j]:.3f} Hz (exact {f_exact[j]:.3f})\n{label}", fontsize=10)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING_CMAP), ax=fig.axes[:6], shrink=0.5,
             label="pressure p / max|p|", location="bottom", aspect=40, pad=0.02)

ax = fig.add_subplot(layout[0, 3])
markers = {1: "o", 2: "s", 3: "^", 4: "D"}
for degree, levels in hex_levels.items():
    dofs = np.array([hex_runs[degree, k]["dofs"] for k in levels])
    errors = np.array([np.max(np.abs(hex_runs[degree, k]["f"] / f_exact - 1)) for k in levels])
    line, = ax.loglog(dofs, errors, markers[degree] + "-", label=f"Q{degree} hexahedra")
    if len(levels) > 1:
        guide = errors[0] * (dofs / dofs[0]) ** (-2 * degree / 3)
        ax.loglog(dofs, guide, ":", color=line.get_color(), lw=1)
for degree in tet_degrees:
    run = tet_runs[degree]
    ax.loglog(run["dofs"], np.max(np.abs(run["f"] / f_exact - 1)), markers[degree], mfc="none", ms=9,
              color="#8B9A8D", label=f"P{degree} tetrahedra (unstructured)")
ax.set(xlabel="number of pressure unknowns", ylabel="largest relative frequency error (20 modes)",
       title="Convergence: dotted guides ~ h^(2p)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)

ax = fig.add_subplot(layout[1, 3])
for degree in (1, 2, 3, 4):
    run = hex_runs[degree, 1]
    ax.semilogy(f_exact, np.abs(run["f"] / f_exact - 1), markers[degree] + "-", ms=4,
                label=f"Q{degree}, h = 10 cm ({run['dofs']} unknowns)")
ax.semilogy(f_exact, np.abs(final["f"] / f_exact - 1), "*-", color=INK, ms=6,
            label=f"Q3, h = 5 cm ({final['dofs']} unknowns)")
ax.set(xlabel="exact frequency / Hz", ylabel="relative frequency error",
       title="Dispersion error over the first 20 modes")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
plt.show()


# ## Going further
#
# Make one wall absorbing: a locally reacting impedance $Z$ adds the boundary matrix
# $i\omega\int Z^{-1}\phi_i\phi_j\,\mathrm dS$ and turns the problem into a quadratic
# eigenproblem with complex frequencies whose imaginary parts give the reverberation
# decay. Or replace the box by a room with a non-rectangular floor plan from the CAD
# kernel: the unstructured tetrahedral route above carries over unchanged, while the
# exact cosine modes no longer exist and the mesh study becomes the reference.
