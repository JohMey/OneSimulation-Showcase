"""Generated from the OneSimulation example notebook.
Needs the installed, activated OneSimulation runtime. See
https://johmey.github.io/OneSimulation-Showcase/examples/notebooks.html
"""


# # STEP import: a steel sleeve in axial tension
#
# Create the sleeve or set STEP_FILE to an input STEP file. Runtime 0.2.1a8, CPython 3.13 and the meshing, compute and visualization extras are required.
#
# The benchmark sleeve is 100 mm long, with 12 mm inner and 18 mm outer radius. Its upper face carries 40 MPa axial tension. The lower face is fixed and the cylindrical surfaces are traction-free.

from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from onesim.cad import CadGeometryBuilder
from onesim.cad.d3 import AnnularCylinder, ImportedSolid
from onesim.cad.exchange import survey_step
from onesim.materials.library import standard_material
from onesim.meshing import TetrahedralMesher
from onesim.numerics import Numerics
from onesim.numerics.fem import FiniteElement, Lagrange
from onesim.numerics.linear import SparseDirect
from onesim.physics.solid_mechanics import (
    FixedDisplacement, LinearElasticity, Traction, TractionFree,
)
from onesim.simulation import Problem, Simulation

assert version("onesimulation") == "0.2.1a8"
STEP_FILE = None
PROFILE = "FAST"
PROFILES = {"FAST": {"target_size": 0.006}, "REFINED": {"target_size": 0.003}}
config = PROFILES[PROFILE]
output_dir = Path("onesimulation_results")
output_dir.mkdir(exist_ok=True)

INNER_RADIUS, OUTER_RADIUS, LENGTH = 0.012, 0.018, 0.100
TENSILE_STRESS = 40.0e6
MATERIAL = standard_material("stainless_steel_304")

if STEP_FILE is None:

    design = AnnularCylinder(inner_radius=INNER_RADIUS,
                              outer_radius=OUTER_RADIUS, length=LENGTH)
    built_geometry = CadGeometryBuilder().build(design)
    step_path = output_dir / "sleeve.step"
    built_geometry.kernel.export_step(built_geometry.reference, step_path)
    expected_volume = design.measure
else:
    step_path = Path(STEP_FILE)
    expected_volume = None
assert step_path.is_file() and step_path.stat().st_size > 0, step_path
print(f"STEP file: {step_path} ({step_path.stat().st_size / 1024:.1f} KiB)")


# ## Inspect the imported faces
#
# Read the STEP survey before assigning boundary conditions. STEP exchange does not carry the names used by the simulation model. The four face groups below describe this sleeve and should be checked for every replacement file.

survey = survey_step(step_path)
print(f"Faces: {survey.face_count}; Volume: {survey.volume:.6e} m^3; "
      f"Unit: {survey.unit}")
print(f"Bounding box: "
      + ", ".join(f"{value:+.4f}" for value in survey.bounds))
print()
print(f"{'Index':>5}  {'Name':<10} {'Area / m^2':>14}  Normal")
for face in survey:
    normal = ", ".join(f"{value:+.3f}" for value in face.normal)
    print(f"{face.index:>5}  {face.member:<10} {face.area:>14.6e}  ({normal})")

if expected_volume is not None:

    relative_volume_error = abs(survey.volume / expected_volume - 1.0)
    assert relative_volume_error < 1e-9, relative_volume_error
    print(f"\nRelative volume difference from the design: {relative_volume_error:.3e}")


# ## Bind the faces, mesh the volume and solve
#
# The setup uses stainless steel 304 and small-strain linear elasticity. The fast profile uses a nominal 6 mm mesh size and the refined profile uses 3 mm. The displacement field is computed with quadratic finite elements.

FACE_COUNT = 4
FACE_GROUPS = (
    ("inner_wall", (0,)),
    ("fixed_end", (1,)),
    ("outer_wall", (2,)),
    ("loaded_end", (3,)),
)
solid = ImportedSolid(path=str(step_path), unit="m",
                        face_count=FACE_COUNT, face_groups=FACE_GROUPS)
material = MATERIAL.create()
model = LinearElasticity(domain=solid.volume, material=material)
problem = Problem(
    name="Imported STEP sleeve under uniform tension",
    geometry=solid, physics=(model,),
    boundary_conditions=(
        FixedDisplacement(boundary=solid.boundaries.fixed_end,
                          physics=model, value=(0.0, 0.0, 0.0)),
        Traction(boundary=solid.boundaries.loaded_end, physics=model,
                 value=(0.0, 0.0, TENSILE_STRESS)),
        TractionFree(boundary=solid.boundaries.inner_wall
                     | solid.boundaries.outer_wall, physics=model),
    ),
)

start_time = perf_counter()
mesh = TetrahedralMesher(target_size=config["target_size"]).generate(solid)
meshing_seconds = perf_counter() - start_time
start_time = perf_counter()
result = Simulation(
    problem=problem, mesh=mesh,
    numerics=Numerics(spatial=FiniteElement(element=Lagrange(order=2)),
                      linear_solver=SparseDirect()),
).solve_stationary()
solve_seconds = perf_counter() - start_time
displacement = result.field(model.displacement)
print(f"Tetrahedra: {mesh.cell_count}; Mesh nodes: {len(mesh.coordinates)}; "
      f"FEM nodes: {len(displacement.values)}")
print(f"Meshing: {meshing_seconds:.2f} s; Solving: {solve_seconds:.2f} s")


# ## Verify the loads and displacement
#
# Check that the named faces partition the boundary and integrate the applied traction. The ideal axial force is the load times annulus area. Sample axial displacement along the sleeve and compare it with u(z) = sigma z / E. The fixed end restricts transverse contraction, so the bar solution is an approximate reference.

facets_by_group = {
    name: len(mesh.facets(getattr(solid.boundaries, name)))
    for name, _ in FACE_GROUPS
}
total_boundary_facets = len(mesh.facets(solid.boundaries.all))
assert sum(facets_by_group.values()) == total_boundary_facets
print("Boundary facets per named face:", facets_by_group)

mean_radius = 0.5 * (INNER_RADIUS + OUTER_RADIUS)
angles = np.linspace(0.0, 2.0 * np.pi, 13)[:-1]
loaded_end_points = np.column_stack((
    mean_radius * np.cos(angles), mean_radius * np.sin(angles),
    np.full_like(angles, LENGTH),
))
loaded_end_displacements = np.asarray(displacement.evaluate(loaded_end_points))
fixed_end_points = np.column_stack((
    mean_radius * np.cos(angles), mean_radius * np.sin(angles),
    np.zeros_like(angles),
))
max_fixed_displacement = float(np.max(np.abs(displacement.evaluate(fixed_end_points))))

z = np.linspace(0.0, LENGTH, 61)
axial_points = np.column_stack((np.full_like(z, mean_radius), np.zeros_like(z), z))
axial_displacements = np.asarray(displacement.evaluate(axial_points))
youngs_modulus = float(MATERIAL.properties["youngs_modulus"])
bar_reference = TENSILE_STRESS * z / youngs_modulus
computed_elongation = float(loaded_end_displacements[:, 2].mean())
reference_elongation = TENSILE_STRESS * LENGTH / youngs_modulus
relative_elongation_error = abs(computed_elongation / reference_elongation - 1.0)

operator = result.operators[model.model_id]
_, _, nodal_loads = operator.conditions(0.0)
integrated_load = np.asarray(nodal_loads).reshape(-1, 3).sum(axis=0)
cross_section = np.pi * (OUTER_RADIUS**2 - INNER_RADIUS**2)
expected_force = TENSILE_STRESS * cross_section

relative_transverse_load = float(np.linalg.norm(integrated_load[:2]) / expected_force)
relative_axial_load_error = float(abs(integrated_load[2] / expected_force - 1.0))

mises = np.asarray(result.field(model.von_mises_stress).evaluate(
    np.array([[mean_radius, 0.0, LENGTH / 2]])))
relative_mises_error = float(abs(mises[0] / TENSILE_STRESS - 1.0))

assert np.isfinite(axial_displacements).all() and np.isfinite(mises).all()
assert max_fixed_displacement < 1e-12
assert relative_transverse_load < 1e-12
assert relative_axial_load_error < 0.01
assert float(np.ptp(loaded_end_displacements[:, 2])) / abs(computed_elongation) < 0.01
assert relative_elongation_error < 0.03
assert relative_mises_error < 0.02

print(f"Elongation: {computed_elongation * 1e6:.3f} um; "
      f"Bar formula: {reference_elongation * 1e6:.3f} um "
      f"(difference {100 * relative_elongation_error:.2f} %)")
print(f"Integrated tensile force: {integrated_load[2]:.6f} N; "
      f"exact annulus: {expected_force:.6f} N "
      f"(faceting difference {100 * relative_axial_load_error:.3f} %)")
print(f"Midheight von Mises stress: {mises[0] / 1e6:.3f} MPa "
      f"(difference {100 * relative_mises_error:.2f} %)")

csv_path = output_dir / "step_sleeve.csv"
with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["z_m", "ux_m", "uy_m", "uz_m", "bar_reference_uz_m"])
    writer.writerows(np.column_stack((z, axial_displacements, bar_reference)))


# ## Export the results
#
# Write sampled displacements to CSV, the comparison chart to PNG and SVG, and the volumetric fields to VTK and XDMF for ParaView. Output files are written under onesimulation_results/.

import onesim.visualization as visualization
from onesim.results.storage import write_for_paraview

cream, ink, green, copper = "#F7F2E7", "#223D35", "#336B57", "#B66E43"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "text.color": ink, "axes.labelcolor": ink,
                     "xtick.color": ink, "ytick.color": ink,
                     "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=cream)
ax.set_facecolor(cream)
ax.plot(z * 1000, axial_displacements[:, 2] * 1e6, color=green, linewidth=2.4,
        label="OneSimulation, quadratic FEM on the imported sleeve")
ax.plot(z * 1000, bar_reference * 1e6, color=copper, linewidth=1.6,
        linestyle="--", label=r"Bar formula $u_z=\sigma z/E$")
ax.set_xlabel("Axial position z / mm")
ax.set_ylabel(r"Elongation $u_z$ / µm")
ax.set_title("STEP sleeve under 40 MPa tensile stress", color=ink)
ax.legend(frameon=False)
ax.grid(True, color=ink, alpha=0.12)
fig.tight_layout()
png_path = output_dir / "step_sleeve.png"
svg_path = output_dir / "step_sleeve.svg"
fig.savefig(png_path, dpi=200)
fig.savefig(svg_path)

PARAVIEW_FIELDS = (model.displacement, model.von_mises_stress, model.stress)
paraview_files = [
    write_for_paraview(result, mesh, PARAVIEW_FIELDS,
                       output_dir / f"step_sleeve.{suffix}")
    for suffix in ("vtk", "xdmf")
]
for path in paraview_files:
    assert path.is_file() and path.stat().st_size > 1000, path
    print(f"ParaView file: {path} ({path.stat().st_size / 1024:.1f} KiB)")

figure = visualization.result(result.result_set, model.von_mises_stress.field_id,
                              title="von Mises stress (Pa)")
assert figure.data, "The result plot has no data traces"
figure


# ## Record the check and its limits
#
# The reference record contains the mesh size, integrated load and elongation comparison. This is a small-strain, linearly elastic load case. To adapt it, check the face groups, material, units and reference tolerances for the new geometry.

verification = {
    "passed": True,
    "profile": PROFILE,
    "runtime_version": version("onesimulation"),
    "example": "linear elastic tension on a solid imported from STEP",
    "step_file": str(step_path),
    "step_file_bytes": int(step_path.stat().st_size),
    "imported_face_count": int(survey.face_count),
    "declared_face_count": FACE_COUNT,
    "imported_volume_m3": float(survey.volume),
    "imported_bounds_m": [float(value) for value in survey.bounds],
    "face_group_boundary_facets": facets_by_group,
    "boundary_facets_total": int(total_boundary_facets),
    "mesh_cells": int(mesh.cell_count),
    "mesh_nodes": int(len(mesh.coordinates)),
    "fem_nodes": int(len(displacement.values)),
    "material": MATERIAL.name,
    "material_quality": MATERIAL.quality,
    "youngs_modulus_pa": youngs_modulus,
    "applied_tension_pa": TENSILE_STRESS,
    "cross_section_m2": float(cross_section),
    "computed_elongation_m": computed_elongation,
    "bar_formula_elongation_m": reference_elongation,
    "elongation_relative_difference": relative_elongation_error,
    "integrated_force_n": integrated_load.tolist(),
    "target_force_n": float(expected_force),
    "axial_load_relative_difference": relative_axial_load_error,
    "transverse_load_relative_error": relative_transverse_load,
    "midheight_von_mises_pa": float(mises[0]),
    "midheight_von_mises_relative_difference": relative_mises_error,
    "mesh_seconds": float(meshing_seconds),
    "solve_seconds": float(solve_seconds),
    "output_files": [str(csv_path), str(png_path), str(svg_path),
                     *(str(path) for path in paraview_files)],
    "limitations": [
        "STEP read with the bundled CAD kernel; other exchange formats are not qualified",
        "Boundaries are named by kernel face index; recheck the face table after a CAD change",
        "Nominal handbook material values at 20 degC; no certified material data",
        "Small-strain stationary isotropic linear elasticity; no contact, plasticity or thermal load",
        "Bar formula is an approximate reference; the clamped end suppresses lateral contraction",
        "One mesh is not a convergence study",
        "The loaded end face is triangulated, so the integrated force is slightly below the exact annulus",
        "ParaView export samples the quadratic solution at the written mesh vertices",
        "No experimental or COMSOL validation",
    ],
}
assert all(Path(path).stat().st_size > 0 for path in verification["output_files"])
report_path = output_dir / "step_sleeve_verification.json"
report_path.write_text(json.dumps(verification, indent=2, ensure_ascii=False),
                     encoding="utf-8")
print(json.dumps(verification, indent=2, ensure_ascii=False))
