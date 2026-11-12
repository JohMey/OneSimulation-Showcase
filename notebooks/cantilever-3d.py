"""Generated from the OneSimulation example notebook.
Needs the installed, activated OneSimulation runtime. See
https://johmey.github.io/OneSimulation-Showcase/examples/notebooks.html
"""


# # 3D-Kragträger: Verformung und Spannung
#
# **OneSimulation Runtime Alpha 0.2.1a4 · räumliche lineare Elastizität · Q2-FEM**
#
# Ein 120 × 18 × 12 mm großer, links vollständig eingespannter Quader trägt rechts
# eine gleichmäßig verteilte Flächenlast mit insgesamt 30 N nach unten.
# OneSimulation erzeugt das dreidimensionale Hexaedernetz, assembliert die
# Elastizitätsgleichung und berechnet Verschiebung und Spannung mit einem direkten
# Sparse-Solver. NumPy dient hier nur zur Auswertung und Matplotlib zur Darstellung.
#
# Voraussetzung: das regulär installierte und aktivierte Runtime-Paket, NumPy und
# Matplotlib. Jupyter ist nur eine mögliche Oberfläche. Alle Codezellen in Reihenfolge
# ausführen; es werden keine Daten oder Materialkataloge aus dem Internet geladen.
#
# **Auflösung wählen:** In der ersten Codezelle steht `PROFILE = "SCHNELL"`.
# Dieses Standardprofil löst das bisherige Netz mit 16 × 3 × 3 Q2-Zellen.
# Mit `PROFILE = "HOCHAUFGELOEST"` wird tatsächlich ein neues Netz mit
# 48 × 8 × 8 Q2-Volumenzellen gelöst und mit dem 16 × 3 × 3-Netz verglichen.
# Das große Profil braucht mehrere GiB freien Arbeitsspeicher und deutlich mehr
# Rechenzeit. Die höhere Bildauflösung ersetzt keine Netzverfeinerung.
#
# Die gewählten konstanten Materialparameter sind **illustrative Eingabewerte**, keine
# zertifizierten Werkstoffdaten. Die zusätzlichen thermischen Materialparameter gehen
# in diese rein mechanische Rechnung nicht ein. Die Grafik überhöht Verschiebungen
# **200-fach**; die Farbskala zeigt die tatsächlich berechnete von-Mises-Spannung.

from pathlib import Path
from time import perf_counter
from importlib.metadata import version
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from onesim.cad.d3 import Box
from onesim.materials import IsotropicSolid
from onesim.meshing import StructuredHexahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import (
    FixedDisplacement, LinearElasticity, Traction, TractionFree,
)
from onesim.simulation import Problem, Simulation

assert version("onesimulation") == "0.2.1a4"
PROFILE = "SCHNELL"  # Alternativ bewusst "HOCHAUFGELOEST" wählen.
PROFILES = {
    "SCHNELL": {"coarse": (8, 2, 2), "fine": (16, 3, 3),
                "surface_x": 64, "surface_cross": 12, "dpi": 180},
    "HOCHAUFGELOEST": {"coarse": (16, 3, 3), "fine": (48, 8, 8),
                      "surface_x": 192, "surface_cross": 32, "dpi": 300},
}
config = PROFILES[PROFILE]
output_dir = Path("ergebnisse_onesimulation")
output_dir.mkdir(exist_ok=True)
length, width, height = 0.120, 0.018, 0.012  # m
youngs_modulus, poisson_ratio = 70e9, 0.30  # Pa, dimensionslos
force = 30.0  # N, Betrag der gesamten Last in negativer z-Richtung
deformation_scale = 200.0
inertia = width * height**3 / 12.0
tip_reference = -force * length**3 / (3.0 * youngs_modulus * inertia)


# ## Das physikalische Modell lösen
#
# Gelöst wird die stationäre dreidimensionale Gleichgewichtsbedingung
# \(-\nabla\!\cdot\boldsymbol{\sigma}=0\) mit isotropem Hooke-Gesetz und kleinen Dehnungen.
# An der linken Stirnfläche gilt \(\mathbf u=0\), rechts
# \(\mathbf t=(0,0,-F/(bh))\); die vier Längsflächen sind frei.
#
# Zwei Netze zeigen, wie stark sich die Enddurchbiegung durch Verfeinerung ändert.
# Die Referenz \(u_z(L)=-FL^3/(3EI)\), \(I=bh^3/12\), stammt aus der vereinfachten
# Euler–Bernoulli-Balkentheorie. Sie berücksichtigt weder dreidimensionale
# Einspanneffekte noch Schubverformung und ist deshalb kein exakter 3D-Sollwert.

def solve_cantilever(cells):
    started = perf_counter()
    geometry = Box(length=length, width=width, height=height)
    material = IsotropicSolid(
        density=2700.0, specific_heat_capacity=900.0,
        thermal_conductivity=200.0, youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio, thermal_expansion_coefficient=23e-6,
    )
    model = LinearElasticity(domain=geometry.volume, material=material)
    problem = Problem(
        name="3D-Kragträger unter verteilter Stirnflächenlast",
        geometry=geometry, physics=(model,),
        boundary_conditions=(
            FixedDisplacement(boundary=geometry.boundaries.x_min,
                              physics=model, value=(0.0, 0.0, 0.0)),
            Traction(boundary=geometry.boundaries.x_max, physics=model,
                     value=(0.0, 0.0, -force / (width * height))),
            TractionFree(boundary=geometry.boundaries.all
                         - geometry.boundaries.x_min - geometry.boundaries.x_max,
                         physics=model),
        ),
    )
    mesh = StructuredHexahedralMesher(cells=cells).generate(geometry)
    result = Simulation(
        problem=problem, mesh=mesh,
        numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)),
                          linear_solver=SparseDirect()),
    ).solve_stationary()
    displacement = result.field(model.displacement)
    tip = float(displacement.evaluate([[length, width / 2, height / 2]])[0, 2])
    return dict(geometry=geometry, model=model, mesh=mesh, result=result,
                displacement=displacement, tip=tip,
                nodes=len(displacement.values), cells=list(cells),
                solve_seconds=perf_counter() - started)

coarse = solve_cantilever(config["coarse"])
fine = solve_cantilever(config["fine"])
geometry, model, mesh, result = [fine[k] for k in ("geometry", "model", "mesh", "result")]
displacement = fine["displacement"]
reference_relative_difference = abs((fine["tip"] - tip_reference) / tip_reference)
mesh_relative_change = abs((fine["tip"] - coarse["tip"]) / fine["tip"])
print(f"Q2-Zellen: {mesh.cell_count}; FEM-Knoten: {fine['nodes']}; "
      f"Freiheitsgrade: {3 * fine['nodes']}")
print(f"Durchbiegung: {abs(fine['tip']) * 1000:.6f} mm; "
      f"Balkenreferenz: {abs(tip_reference) * 1000:.6f} mm")
print(f"Abweichung zur Balkenreferenz: {100 * reference_relative_difference:.3f} %; "
      f"Netzänderung: {100 * mesh_relative_change:.3f} %")


# ## Numerische Prüfungen und Ergebnisdaten
#
# Wir prüfen die festgehaltene Einspannfläche, die tatsächlich integrierte Gesamtlast,
# die Symmetrie, endliche Feldwerte, den Netzvergleich und die Übereinstimmung mit der
# Balkennäherung. Die von-Mises-Spannung wird direkt über das Ergebnisfeld von
# OneSimulation ausgewertet. Die CSV enthält die berechnete Mittellinie und den
# analytischen Vergleich; die Grafik wird zusätzlich als PNG und SVG gespeichert.

x = np.linspace(0.0, length, 81)
centerline = np.column_stack((x, np.full_like(x, width / 2), np.full_like(x, height / 2)))
centerline_u = displacement.evaluate(centerline)
beam_reference = -force * x**2 * (3 * length - x) / (6 * youngs_modulus * inertia)
fixed_points = np.array([[0.0, y, z]
                        for y in np.linspace(0, width, 7)
                        for z in np.linspace(0, height, 7)])
fixed_max = float(np.max(np.abs(displacement.evaluate(fixed_points))))
operator = result.operators[model.model_id]
_, _, nodal_loads = operator.conditions(0.0)
integrated_load = np.asarray(nodal_loads).reshape(-1, 3).sum(axis=0)
load_relative_error = float(np.linalg.norm(integrated_load - [0, 0, -force]) / force)
symmetry_max = float(np.max(np.abs(centerline_u[:, 1])))
stress_point = np.array([[length / 2, width / 2, height * 0.9]])
stress_sample = np.asarray(result.field(model.stress).evaluate(stress_point))
beam_stress_sample = force * (length - stress_point[0, 0]) * (stress_point[0, 2] - height / 2) / inertia
stress_relative_difference = float(abs(stress_sample[0, 0] / beam_stress_sample - 1))

assert np.isfinite(centerline_u).all() and np.isfinite(stress_sample).all()
assert fine["tip"] < 0
assert fixed_max < 1e-12
assert load_relative_error < 1e-10
assert symmetry_max < 1e-10
assert reference_relative_difference < 0.02
assert mesh_relative_change < 0.01
assert stress_relative_difference < 0.05

csv_path = output_dir / "kragtraeger_3d.csv"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["x_m", "ux_m", "uy_m", "uz_m", "euler_bernoulli_uz_m"])
    writer.writerows(np.column_stack((x, centerline_u, beam_reference)))

# Die Oberflächen-Farbwerte kommen aus dem berechneten 3D-Spannungsfeld.
# Unterteilung dient nur der Darstellung, nicht einer weiteren FEM-Lösung.
surface_quads, stress_centers = [], []
sx, sc = config["surface_x"], config["surface_cross"]
for fixed_axis, fixed_value, a_axis, b_axis, na, nb in (
    (2, 0.0, 0, 1, sx, sc), (2, height, 0, 1, sx, sc),
    (1, 0.0, 0, 2, sx, sc), (1, width, 0, 2, sx, sc),
    (0, 0.0, 1, 2, sc, sc), (0, length, 1, 2, sc, sc),
):
    dimensions = (length, width, height)
    a = np.linspace(0, dimensions[a_axis], na + 1)
    b = np.linspace(0, dimensions[b_axis], nb + 1)
    for i in range(na):
        for j in range(nb):
            quad = np.zeros((4, 3))
            quad[:, fixed_axis] = fixed_value
            quad[:, a_axis] = (a[i], a[i + 1], a[i + 1], a[i])
            quad[:, b_axis] = (b[j], b[j], b[j + 1], b[j + 1])
            surface_quads.append(quad)
            stress_centers.append(quad.mean(axis=0))
surface_quads = np.asarray(surface_quads)
surface_points = surface_quads.reshape(-1, 3)
surface_u = displacement.evaluate(surface_points).reshape(surface_quads.shape)
stress_centers = np.asarray(stress_centers)
surface_mises = np.asarray(result.field(model.von_mises_stress).evaluate(stress_centers))
assert np.isfinite(surface_u).all() and np.isfinite(surface_mises).all()
assert np.min(surface_mises) >= 0
deformed_quads_mm = (surface_quads + deformation_scale * surface_u) * 1000.0

# Nur Kanten des ursprünglichen FEM-Oberflächennetzes als Linien zeichnen.
edge_ids, visible_edge_ids = set(), set()
for facet in mesh.facets(geometry.boundaries.all):
    coordinates = mesh.coordinates[list(facet)]
    visible = (np.allclose(coordinates[:, 1], 0.0)
               or np.allclose(coordinates[:, 2], height)
               or np.allclose(coordinates[:, 0], length))
    for i in range(len(facet)):
        edge = tuple(sorted((facet[i], facet[(i + 1) % len(facet)])))
        edge_ids.add(edge)
        if visible:
            visible_edge_ids.add(edge)
edge_lines, visible_edge_lines = [], []
for first, second in sorted(edge_ids):
    line = np.linspace(mesh.coordinates[first], mesh.coordinates[second], 5)
    deformed_line = (line + deformation_scale * displacement.evaluate(line)) * 1000.0
    edge_lines.append(deformed_line)
    if (first, second) in visible_edge_ids:
        visible_edge_lines.append(deformed_line)
edge_lines = np.asarray(edge_lines)
visible_edge_lines = np.asarray(visible_edge_lines)

np.savez_compressed(output_dir / "kragtraeger_3d_felddaten.npz",
    mesh_coordinates_m=mesh.coordinates, mesh_cells=mesh.cells,
    surface_quads_m=surface_quads, surface_displacement_m=surface_u,
    stress_centers_m=stress_centers, surface_von_mises_pa=surface_mises,
    deformed_quads_mm=deformed_quads_mm, deformed_mesh_edges_mm=edge_lines,
    centerline_m=centerline, centerline_displacement_m=centerline_u,
    beam_reference_uz_m=beam_reference, deformation_scale=deformation_scale)

cream, ink, green, copper = "#F7F2E7", "#223D35", "#336B57", "#B66E43"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                    "text.color": ink, "axes.labelcolor": ink,
                    "xtick.color": ink, "ytick.color": ink,
                    "svg.fonttype": "none"})
cmap = LinearSegmentedColormap.from_list("onesimulation_stress",
    ["#214D43", "#5F9175", "#A8BC91", "#E4D6A0", "#CD9561", "#A85539"])
norm = Normalize(vmin=0.0, vmax=float(surface_mises.max() / 1e6))

fig = plt.figure(figsize=(16, 9), facecolor=cream)
fig.text(0.045, 0.937, "ONESIMULATION   /   RÄUMLICHE ELASTIZITÄT", fontsize=11,
         color=green, weight="bold")
fig.text(0.045, 0.875, "Biegung wird sichtbar.", fontsize=34, weight="bold")
fig.text(0.045, 0.828, "3D-FEM · quadratische Hexaeder · tatsächlich berechnetes Spannungsfeld",
         fontsize=13)

ax = fig.add_axes([0.015, 0.18, 0.72, 0.65], projection="3d", facecolor=cream)
body = Poly3DCollection(deformed_quads_mm, facecolors=cmap(norm(surface_mises / 1e6)),
                        edgecolors="none", linewidths=0, antialiased=False)
ax.add_collection3d(body)
ax.add_collection3d(Line3DCollection(visible_edge_lines, colors="#244F4580",
                                    linewidths=0.25 if PROFILE == "HOCHAUFGELOEST" else 0.55))

# Unverformte äußere Kanten: geometrische Referenz, keine zweite Lösung.
corners = np.array([(xx, yy, zz) for xx in (0, length) for yy in (0, width) for zz in (0, height)]) * 1000
original_edges = []
for i in range(8):
    for j in range(i + 1, 8):
        if np.count_nonzero(corners[i] != corners[j]) == 1:
            original_edges.append([corners[i], corners[j]])
ax.add_collection3d(Line3DCollection(original_edges, colors="#7C898480",
                                    linewidths=0.75, linestyles="dashed"))
clamp = np.array([[-1.5, -3.0, -3.0], [-1.5, width * 1000 + 3, -3.0],
                  [-1.5, width * 1000 + 3, height * 1000 + 3], [-1.5, -3.0, height * 1000 + 3]])
ax.add_collection3d(Poly3DCollection([clamp], facecolors="#AEB7A6", alpha=0.55,
                                    edgecolors=green, linewidths=1))
for y_value in np.linspace(2, width * 1000 - 2, 4):
    p = np.array([[length, y_value / 1000, height]])
    endpoint = (p + deformation_scale * displacement.evaluate(p))[0] * 1000
    ax.quiver(*[endpoint[0], endpoint[1], endpoint[2] + 12], 0, 0, -8,
              color=copper, linewidth=1.5, arrow_length_ratio=0.25)
ax.text(0, width * 1000 + 7, 15, "fest eingespannt", fontsize=10, color=green)
ax.text(length * 1000 - 4, -5, 8, "30 N", fontsize=12, weight="bold", color=copper)
ax.set_xlim(-4, length * 1000 + 6)
ax.set_ylim(-5, width * 1000 + 10)
ax.set_zlim(-24, 26)
ax.set_box_aspect((130, 33, 50), zoom=1.22)
ax.view_init(elev=23, azim=-62)
ax.set_axis_off()
ax.set_proj_type("ortho")
for artist in ax.collections:
    artist.set_clip_on(False)

cax = fig.add_axes([0.11, 0.235, 0.43, 0.019])
bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
bar.set_label("von-Mises-Spannung [MPa]", fontsize=11, labelpad=8)
bar.outline.set_visible(False)
bar.ax.tick_params(length=0, labelsize=9)
fig.text(0.11, 0.157, "Verformung ×200   ·   gestrichelt: unverformte Geometrie", fontsize=11, color=green)

plot = fig.add_axes([0.755, 0.46, 0.205, 0.24], facecolor=cream)
plot.plot(x * 1000, -centerline_u[:, 2] * 1000, color=green, linewidth=2.5, label="3D-FEM")
plot.plot(x * 1000, -beam_reference * 1000, color=copper, linewidth=1.6,
          linestyle="--", label="Balkennäherung")
plot.set_xlabel("Position x [mm]", fontsize=10)
plot.set_ylabel("Durchbiegung [mm]", fontsize=10)
plot.spines[["top", "right"]].set_visible(False)
plot.spines[["bottom", "left"]].set_color("#B8C2B4")
plot.tick_params(labelsize=9, length=3)
plot.grid(alpha=0.18)
plot.legend(frameon=False, loc="upper left", fontsize=9)
fig.text(0.755, 0.746, "DIE RECHNUNG IM VERGLEICH", fontsize=10, color=green, weight="bold")
fig.text(0.755, 0.367, f"{abs(fine['tip']) * 1000:.4f} mm", fontsize=27, weight="bold")
fig.text(0.755, 0.333, "berechnete Enddurchbiegung", fontsize=10)
fig.text(0.755, 0.274, f"{mesh.cell_count} Q2-Zellen  ·  {3 * fine['nodes']:,} Freiheitsgrade".replace(",", "."), fontsize=11)
fig.text(0.755, 0.242, f"{100 * reference_relative_difference:.2f} % Abstand zur Balkennäherung", fontsize=10)
fig.text(0.755, 0.210, f"{100 * mesh_relative_change:.2f} % Änderung bei Netzverfeinerung", fontsize=10)

fig.add_artist(plt.Line2D([0.045, 0.96], [0.115, 0.115], color="#D5D7C8", linewidth=1))
fig.text(0.045, 0.077, "120 × 18 × 12 mm  ·  E = 70 GPa  ·  ν = 0,30  ·  gleichmäßige Stirnflächenlast", fontsize=10)
fig.text(0.045, 0.045, "Illustratives, linear-elastisches Beispiel. Balkentheorie ist eine Näherung; keine experimentelle Validierung.",
         fontsize=9, color="#596B61")
fig.text(0.96, 0.054, "RUNTIME ALPHA 0.2.1a4", fontsize=9, color=green, ha="right")
png_path = output_dir / "kragtraeger_3d.png"
svg_path = png_path.with_suffix(".svg")
fig.savefig(png_path, dpi=config["dpi"], facecolor=cream)
fig.savefig(svg_path, facecolor=cream)
plt.show()


# ## Ergebnisse weiterverwenden
#
# `onesim.visualization.result` zeichnet ein berechnetes Feld direkt auf seinem
# eigenen Träger. Das ist eine Zeile anstelle der von Hand geschriebenen
# Abtastung und Einfärbung weiter oben; die gestaltete Abbildung bleibt davon
# unberührt, hier steht der kurze Weg daneben.
#
# `onesim.results.storage.write_for_paraview` schreibt ausdrücklich genannte
# Felder auf genau das Netz, auf dem gerechnet wurde, als `.vtk` und `.xdmf`.
# Beide Dateien öffnet ParaView oder VisIt ohne Zwischenschritt.
#
# Grenzen dieses Exports: die Werte werden an den geschriebenen Netzknoten
# ausgewertet. Eine quadratische oder Nédélec-Lösung erscheint dort als lineare
# Ansicht auf denselben Knoten, nicht als die Lösung selbst; bei Bedarf das Netz
# verfeinern. Geschrieben wird nur, was aufgezählt ist — es gibt keine
# stillschweigende Feldauswahl. Es wird nichts neu gerechnet.

import onesim.visualization as visualization
from onesim.results.storage import write_for_paraview

PARAVIEW_FELDER = (model.displacement, model.von_mises_stress, model.stress)
figure = visualization.result(
    result.result_set, model.von_mises_stress.field_id, title="von-Mises-Spannung (Pa)",
)
assert figure.data, "Die Bibliotheksgrafik enthält keine Spur"

paraview_dateien = [
    write_for_paraview(result, mesh, PARAVIEW_FELDER,
                       output_dir / f"kragtraeger_3d.{suffix}")
    for suffix in ("vtk", "xdmf")
]
for pfad in paraview_dateien:
    assert pfad.is_file() and pfad.stat().st_size > 1000, pfad
    print(f"ParaView-Datei: {pfad} ({pfad.stat().st_size / 1024:.1f} KiB)")
figure


# ## Aussagekraft und Grenzen
#
# Die Darstellung zeigt eine stationäre, isotrope, linear-elastische Rechnung mit
# kleinen Dehnungen und idealisierter Einspannung. Die 200-fach gezeichnete
# Verformung ist **keine** geometrisch nichtlineare Rechnung. Die Balkennäherung ist
# eine unabhängige Plausibilitätsreferenz, kein exaktes 3D-Vergleichsproblem.
# Zwei Netze ersetzen keine vollständige Konvergenzstudie. Lokale Spannungsspitzen an
# der scharfkantigen, vollständig eingespannten Fläche sind netzabhängig; aus ihrem
# Maximum wird hier weder ein Sicherheitsfaktor noch eine Werkstofffreigabe abgeleitet.
# Keine plastische Verformung, Kontakt-, Ermüdungs- oder Bruchanalyse wird behauptet.
# Es liegt keine experimentelle oder vergleichende COMSOL-Validierung vor.

verification = {
    "passed": True,
    "profile": PROFILE,
    "runtime_version": version("onesimulation"),
    "example": "3D linear elastic cantilever with Q2 hexahedra",
    "fine_mesh_cells": fine["cells"],
    "fine_mesh_element_count": int(mesh.cell_count),
    "fine_mesh_nodes": int(fine["nodes"]),
    "fine_mesh_displacement_dofs": int(3 * fine["nodes"]),
    "coarse_mesh_cells": coarse["cells"],
    "fine_tip_displacement_m": fine["tip"],
    "coarse_tip_displacement_m": coarse["tip"],
    "euler_bernoulli_tip_m": tip_reference,
    "reference_relative_difference": reference_relative_difference,
    "mesh_relative_change": mesh_relative_change,
    "clamp_max_abs_displacement_m": fixed_max,
    "integrated_force_n": integrated_load.tolist(),
    "load_relative_error": load_relative_error,
    "centerline_symmetry_max_abs_uy_m": symmetry_max,
    "midspan_axial_stress_pa": float(stress_sample[0, 0]),
    "midspan_beam_stress_pa": float(beam_stress_sample),
    "midspan_stress_reference_relative_difference": stress_relative_difference,
    "sampled_surface_peak_von_mises_pa": float(surface_mises.max()),
    "coarse_solve_seconds": coarse["solve_seconds"],
    "fine_solve_seconds": fine["solve_seconds"],
    "deformation_display_factor": deformation_scale,
    "output_files": [str(csv_path), str(png_path), str(svg_path),
                     str(output_dir / "kragtraeger_3d_felddaten.npz")],
    "limitations": [
        "Illustrative material input; no certified material data",
        "Small-strain stationary isotropic linear elasticity",
        "Display deformation magnified 200 times; no large-deformation physics",
        "Euler-Bernoulli is an approximate beam reference, not an exact 3D solution",
        "Two meshes are a refinement check, not a full convergence study",
        "Clamped-edge peak stress depends on mesh; not a strength certification",
        "No experimental or COMSOL validation",
    ],
}
(output_dir / "kragtraeger_3d_pruefung.json").write_text(
    json.dumps(verification, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(verification, indent=2, ensure_ascii=False, allow_nan=False))
