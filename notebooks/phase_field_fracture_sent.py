"""OneSimulation example: phase-field fracture of a single-edge-notched specimen.
Requires the installed, activated OneSimulation runtime.

A notched plate under tension is read from a STEP file and cut to its cross-section. The crack is
computed with the AT2 phase-field model, and the force-displacement curve is compared with the
published result of Miehe et al. (2010); the dissipated energy is compared with Griffith's G_c
times the crack length, for two regularisation lengths and two element sizes.
"""

# # Phase-field fracture: single-edge-notched tension test
#
# A phase-field model replaces a sharp crack by a damage field $d(\mathbf x)\in[0,1]$
# ($d=1$: broken) smeared over a width set by the regularisation length $\ell$. The
# displacement $\mathbf u$ and the damage minimise the total energy (AT2 model)
#
# $$\Pi(\mathbf u,d)=\int_\Omega\Big[g(d)\,\psi^+(\boldsymbol\varepsilon)+\psi^-(\boldsymbol\varepsilon)\Big]\,\mathrm dV
#   +\frac{G_c}{2\ell}\int_\Omega\big(d^2+\ell^2|\nabla d|^2\big)\,\mathrm dV
#   -W_\mathrm{ext},\qquad g(d)=(1-d)^2(1-k)+k ,$$
#
# subject to irreversibility, $\dot d\ge0$. The second integral converges to the Griffith
# surface energy $G_c\,A_\mathrm{crack}$ as $\ell\to0$. The strain energy is split into a
# part $\psi^+$ that drives the crack and a part $\psi^-$ that does not (volumetric energy in
# compression), after Amor, Marigo and Maurini (J. Mech. Phys. Solids 57, 2009).
#
# **Setup** (C. Miehe, M. Hofacker, F. Welschinger, Comput. Methods Appl. Mech. Eng. 199,
# 2010, 2765-2778): a 1 mm x 1 mm plate with a sharp notch from the left side to the centre;
# $E=210$ GPa, $\nu=0.3$, $G_c=2.7$ N/mm, plane strain. The specimen comes from the CAD file
# `phase_field_fracture_sent.step`, written by `phase_field_fracture_sent_geometry.py` with the
# CAD objects of OneSimulation; the model lives on its cross-section.
#
# | side | displacement | damage |
# |---|---|---|
# | `y_min` (bottom) | $u_x=u_y=0$ | zero flux |
# | `y_max` (top) | $u_x=0$, $u_y=u$ | zero flux |
# | `x_min`, `x_max` | traction free | zero flux |
# | `notch.side` (both flanks) | traction free | zero flux |
#
# **Reference.** The published force-displacement curve: peak force about 0.70 kN per mm
# thickness at $u\approx5.5$ µm (points digitised from the figure), with a straight crack across
# the ligament. And the Griffith balance: the energy dissipated in forming the crack equals $G_c$
# times the 0.5 mm ligament times the thickness, up to the overestimate $h/2\ell$ of the
# regularised functional on a mesh of size $h$.

import numpy as np
from onesim.cad.exchange import load
from onesim.meshing import TriangularMesher
from onesim.physics.solid_mechanics import PhaseFieldFracture
from onesim.visualization import plots, theme

theme.set()

# ## Geometry and mesh
#
# `load` returns the solid, `section` its cross-section at half thickness with the names of the
# cut faces on its sides. Each case has its own triangle mesh: the size is $h$ in a band of
# $\pm5\ell$ around the crack plane, from $6\ell$ in front of the notch tip to the right side, and
# grows to 50 µm away from it.

specimen = load("phase_field_fracture_sent.step")
side, _, thickness = specimen.size
plate = specimen.section((0, 0, 1), origin=(0, 0, thickness / 2))
faces = plate.boundaries
tip, mm = side / 2, 1e-3
cases = {
    "l = 0.0075 mm, h = l/2": (0.0075 * mm, 0.00375 * mm),
    "l = 0.015 mm, h = l/2": (0.015 * mm, 0.0075 * mm),
    "l = 0.015 mm, h = l/4": (0.015 * mm, 0.00375 * mm),
}
main = "l = 0.015 mm, h = l/4"
band = lambda ell, h: lambda p: np.minimum(
    h
    + 0.25
    * np.hypot(
        np.maximum(tip - 6 * ell - p[:, 0], 0.0),
        np.maximum(np.abs(p[:, 1] - side / 2) - 5 * ell, 0.0),
    ),
    0.05 * mm,
)
meshes = {
    label: TriangularMesher(target_size=0.05 * mm, size_field=band(ell, h)).generate(
        plate
    )
    for label, (ell, h) in cases.items()
}

fig = plots.figure(1, 2, size=(14, 6.5))
plots.geometry(
    plate, title="Cross-section of the specimen with its named sides", ax=fig[0]
)
plots.mesh(
    meshes[main],
    groups=("all",),
    legend=False,
    title=f"{main}: {meshes[main].cell_count} triangles",
    ax=fig[1],
)
plots.show()

# ## Model
#
# The top side is moved up in 39 steps, fine around the expected peak; the load parameter of the
# solver is the displacement itself. Each step alternates between equilibrium at fixed damage and
# the damage at fixed displacement until the damage changes by less than $10^{-4}$.

youngs_modulus, poisson_ratio, toughness = 210e9, 0.3, 2700.0
lift = (
    np.r_[
        0.0,
        np.arange(0.5e-3, 4.01e-3, 0.5e-3),
        np.arange(4.25e-3, 5.0e-3, 0.25e-3),
        np.arange(5.0e-3, 6.31e-3, 0.05e-3),
    ]
    * mm
)
results = {
    label: PhaseFieldFracture(
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        fracture_energy=toughness,
        length_scale=ell,
        thickness=thickness,
    ).solve(
        mesh=meshes[label],
        times=lift,
        tolerance=1e-4,
        max_iterations=5000,
        displacements=(
            (faces.y_min, "x", 0.0),
            (faces.y_min, "y", 0.0),
            (faces.y_max, "x", 0.0),
            (faces.y_max, "y", lambda t: t),
        ),
        tractions=(
            (faces.x_min, (0.0, 0.0)),
            (faces.x_max, (0.0, 0.0)),
            (faces["notch.side"], (0.0, 0.0)),
        ),
    )
    for label, (ell, h) in cases.items()
}
force = {
    label: result.reactions[:, np.isclose(result.coordinates[:, 1], side), 1].sum(
        axis=1
    )
    for label, result in results.items()
}
peak = {label: int(np.argmax(values)) for label, values in force.items()}

# ## Comparison with the reference
#
# Peak force and the displacement at the peak against the published values, and the energy
# dissipated after the peak against Griffith's $G_c$ times the ligament, without and with the
# mesh overestimate $h/2\ell$.

published = np.array(
    [
        (0.0, 0.0),
        (0.00055, 0.0768),
        (0.00128, 0.1774),
        (0.00197, 0.2716),
        (0.00290, 0.3955),
        (0.00369, 0.4961),
        (0.00421, 0.5619),
        (0.00471, 0.6200),
        (0.00501, 0.6535),
        (0.00524, 0.6780),
        (0.00543, 0.6960),
        (0.00555, 0.7012),
        (0.00557, 0.6584),
        (0.00562, 0.5613),
        (0.00568, 0.4629),
        (0.00575, 0.3192),
        (0.00580, 0.1833),
        (0.00583, 0.0214),
        (0.00583, 0.0007),
    ]
)
griffith = toughness * (side - tip) * thickness
print(
    f"{'case':24s} {'triangles':>9s} {'peak force [kN]':>16s} {'u at peak [µm]':>15s} {'dissipated / G_c a t':>21s} {'with h/2l':>10s}"
)
print(f"{'Miehe et al. (2010)':24s} {'':9s} {0.70:16.3f} {5.5:15.2f}")
for label, (ell, h) in cases.items():
    released = (
        results[label].fracture_energy[-1] - results[label].fracture_energy[peak[label]]
    )
    print(
        f"{label:24s} {meshes[label].cell_count:9d} {force[label][peak[label]] / 1e3:16.3f} {1e6 * lift[peak[label]]:15.2f}"
        f" {released / griffith:21.3f} {released / (griffith * (1 + h / (2 * ell))):10.3f}"
    )
x, y = results[main].coordinates.T
broken = (results[main].damage[-1] > 0.95) & (x > tip)
print(
    f"crack path ({main}): mean height {y[broken].mean() / mm:.4f} mm, between {y[broken].min() / mm:.4f} and {y[broken].max() / mm:.4f} mm"
)
print(
    f"smallest damage change between two steps: {np.diff(results[main].damage, axis=0).min():.1e}"
)

# ## Figure
#
# The damage before the peak, at the peak and after separation; the force-displacement curves
# with the published points; and the energies of the main case.

shown = {
    "before the peak": int(np.argmin(np.abs(lift - 4e-6))),
    "at the peak": peak[main],
    "separated": len(lift) - 1,
}
energy = results[main]
fig = plots.figure(layout=[list(shown), ["force", "force", "energy"]], size=(16, 10.5))
for panel, step in shown.items():
    plots.heatmap(
        x / mm,
        y / mm,
        energy.damage[step],
        triangles=np.asarray(meshes[main].cells),
        limits=(0, 1),
        label="damage d",
        xlabel="x [mm]",
        ylabel="y [mm]",
        title=f"{panel}: u = {1e6 * lift[step]:.2f} µm",
        ax=fig[panel],
    )
plots.line(
    1e6 * lift,
    {label: values / 1e3 for label, values in force.items()},
    points={"Miehe et al. (2010), digitised": (1e3 * published[:, 0], published[:, 1])},
    xlabel="top displacement u [µm]",
    ylabel="force [kN per mm thickness]",
    title="Force-displacement curve",
    legend="upper left",
    ax=fig["force"],
)
plots.line(
    1e6 * lift,
    {
        "external work": 1e3 * energy.external_work,
        "elastic energy": 1e3 * energy.elastic_energy,
        "crack energy": 1e3 * energy.fracture_energy,
    },
    hlines={
        "crack energy at the peak + G_c a t": 1e3
        * (energy.fracture_energy[peak[main]] + griffith)
    },
    xlabel="top displacement u [µm]",
    ylabel="energy [mJ]",
    title=f"Energies, {main}",
    legend="upper left",
    ax=fig["energy"],
)
plots.show()

# ## Going further
#
# Change the notch in `phase_field_fracture_sent_geometry.py` and write the STEP file again, or
# move the top side sideways instead of up (the shear test of Miehe et al.): the crack then
# curves downwards. Try a smaller $\ell$ together with a finer band.
