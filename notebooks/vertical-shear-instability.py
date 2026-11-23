"""Generated from the OneSimulation example notebook.
Needs the installed, activated OneSimulation runtime. See
https://johmey.github.io/OneSimulation-Showcase/examples/notebooks.html
"""


# # Vertical shear instability in a rotating disk
#
# This example follows a small angular momentum perturbation in a locally isothermal disk. **OneSimulation Runtime Alpha 0.2.1a4** solves the axisymmetric compressible flow equations. NumPy prepares the inputs and processes the results. Matplotlib plots the computed fields.
#
# The parameters follow the disk model in [Johannes Meyer’s dissertation (2024)](https://doi.org/10.11588/heidok.00035158). This calculation uses OneSimulation’s own implementation. It does not reproduce or validate the MATRICS code described in the dissertation.
#
# Install and activate the matching runtime before starting. You can execute the code cells in order in Jupyter or run them as a Python script.
#
# ## Choose a run profile
#
# - `SMOKE_TEST` runs four short steps on a 32 × 32 mesh. Use it to check the API, solver and conservation. The run ends before instability growth can be assessed.
# - `HIGH_RESOLUTION` runs 15 reference orbits on a 256 × 256 mesh with 262,656 unknowns and periodically writes checkpoints. Runtime and memory use depend on the computer.
#
# The fine mesh alone does not establish accuracy. Check time-step sensitivity and account for the finite extent of the modelled domain. This axisymmetric calculation does not establish saturated three-dimensional turbulence or validate the model against experiments.

from pathlib import Path
from importlib.metadata import version
from dataclasses import asdict
from time import perf_counter
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from onesim.physics.hydrodynamics.shared.geometry.execution.axisymmetric import (
    AxisymmetricFlow, FlowParameters, BoxParameters, SpatialParameters, fit_growth,
)
from onesim.numerics.discretization.selection.configuration.solver_configuration import Newton, Iterative
from onesim.numerics.time.semantics.control import time_integrator
from onesim.simulation.problem_definition.declarations.specification import NumericalSetup, CouplingStrategy
from onesim.simulation.execution.requests.controls import ExecutionOptions
from onesim.results.storage.output.retention_policy import ResultRetentionPolicy

assert version("onesimulation") == "0.2.1a4", "This example was tested with Runtime 0.2.1a4."
PROFILE = "SMOKE_TEST"  # For the longer run, use "HIGH_RESOLUTION"
PROFILES = {
    "SMOKE_TEST": {"cells": 32, "end_code": 0.8, "dt_code": 0.2},
    "HIGH_RESOLUTION": {"cells": 256, "end_code": 15 * 2 * np.pi, "dt_code": 0.5},
}
assert PROFILE in PROFILES
config = PROFILES[PROFILE]
output = Path("onesimulation_results")
output.mkdir(exist_ok=True)


# ## Model, initial perturbation and numerical methods
#
# Dimensionless units set `GM = r₀ = ρ₀ = 1` and the reference sound speed to `c₀ = 0.1`. One reference orbit takes `2π` time units. The radial temperature exponent is `q = −1`, the disk density exponent is `p = −1.5`, and both kinematic and bulk viscosity are zero. Here θ is latitude above the equatorial plane. The domain spans θ ∈ [0.05, 0.15] radians and r ∈ [0.95, 1.05].
#
# The disk equilibrium supplies the initial density and rotation. A reproducible random perturbation of absolute amplitude 10⁻⁵ is added to the conserved angular momentum (seed 2024). Both meridional velocity components start at zero. The spatial discretization uses a staggered finite-volume grid, QUICK fluxes and equilibrium wall conditions. As in the original example, explicit equilibrium subtraction is disabled. The runtime uses BDF2 time integration, Newton iteration and LU-preconditioned GMRES. A CFL limit bounds the advective Courant number.

physical = FlowParameters(
    environment="disk", thermodynamics="isothermal", GM=1., r0=1., c0=.1,
    rho0=1., q=-1., disk_p=-1.5, nu=0., bulk_nu=0.,
)
box = BoxParameters(
    nr=config["cells"], ntheta=config["cells"], r_min=.95, r_max=1.05,
    latitude=.10, latitude_width=.10, phi_extent=2*np.pi,
)
spatial = SpatialParameters(
    layout="staggered", flux_scheme="quick", boundary="equilibrium_wall",
    courant_rule="maximum", finite_difference_step=1e-7,
    well_balanced=False, perturbation=0.,
)
model = AxisymmetricFlow(physical, box, spatial, name="VSI-original-physical-model")
background = model.background_fields(native=True)
initial_density = background["density"].copy()
initial_angular_momentum = background["angular_momentum"].copy()
noise = np.random.default_rng(2024).uniform(-1, 1, model.shape)
angular_momentum = initial_angular_momentum + 1e-5 * noise
model.initial(density=initial_density, vr=0., vtheta=0.,
              vphi=angular_momentum / (initial_density * model.R))
numerics = NumericalSetup(
    time=time_integrator("bdf2_gear"),
    nonlinear=Newton(tolerance=1e-9, absolute_tolerance=1e-11,
                     maximum_iterations=16, damping="automatic", jacobian_reuse=4),
    linear=Iterative(method="gmres", preconditioner="block_jacobi",
                     block_sizes=(model.initial_state.size,),
                     block_jacobi_large_block_method="direct", restart=60,
                     relative_tolerance=1e-10, absolute_tolerance=1e-12,
                     maximum_iterations=300,
                     provider_id="numerics.solver.scipy.right-gmres"),
    coupling=CouplingStrategy(mode="fully_coupled"),
)
print(f"{PROFILE}: {box.nr} × {box.ntheta} cells, {model.initial_state.size:,} unknowns")

times, diagnostics = [], []
snapshots = []
started = perf_counter()
last_print = [started]

def retain(frame):
    state = frame.get("state")
    if state is None:
        return
    t = float(frame["time"])
    if times and abs(t - times[-1]) < 1e-12:
        return
    d = model.diagnostics(state)
    assert all(np.isfinite(value) for value in d.values())
    times.append(t)
    diagnostics.append(d)
    if not snapshots or t - snapshots[-1]["time_code"] >= np.pi-1e-10 or t >= config["end_code"]-1e-10:
        fields = model.reconstruct(state)
        snapshots.append({"time_code": t,
            "density": np.asarray(fields["density"]).reshape(model.shape).copy(),
            "vr": np.asarray(fields["vr"]).reshape(model.shape).copy(),
            "vtheta": np.asarray(fields["vtheta"]).reshape(model.shape).copy()})
    if perf_counter() - last_print[0] >= 30 or t >= config["end_code"]-1e-10:
        print(f"{t/(2*np.pi):.3f} orbits; v_rms/c₀ = {d['vrms']:.6g}")
        last_print[0] = perf_counter()

retain({"time": 0., "state": model.initial_state})
study = model.study(numerics, start_time=0., end_time=config["end_code"],
                    maximum_step=config["dt_code"], initial_step=config["dt_code"],
                    mode="fixed", cfl_limit=1.)
study.validate().require_valid()
result = study.run(options=ExecutionOptions(
    max_steps=100000, record_states=True, record_result_history=False,
    result_retention=ResultRetentionPolicy.external_stream(retain),
    checkpoint_interval=max(config["dt_code"], min(2*np.pi, config["end_code"]/2)),
    checkpoint_path=str(output / f"vsi_{PROFILE.lower()}_checkpoint.json"),
    checkpoint_retain_count=2,
))
elapsed_seconds = perf_counter() - started
assert abs(times[-1] - config["end_code"]) < 1e-8
history = {key: np.asarray([d[key] for d in diagnostics]) for key in diagnostics[0]}
history["time_code"] = np.asarray(times)
history["orbits"] = history["time_code"] / (2*np.pi)
mass_drift = float(np.max(np.abs(history["mass"]/history["mass"][0]-1)))
angular_drift = float(np.max(np.abs(history["angular_momentum_total"]/history["angular_momentum_total"][0]-1)))
assert mass_drift < 1e-8 and angular_drift < 1e-8
final_fields = snapshots[-1]
assert np.all(final_fields["density"] > 0)
np.savez_compressed(output/"vsi_fields.npz", r=model.r, theta=model.theta,
                    rf=model.rf, tf=model.tf, R=model.R, z=model.z,
                    initial_density=initial_density, **final_fields)
np.savez_compressed(output/"vsi_history.npz", **history)


# ## Examine the perturbations
#
# The left panel shows the meridional velocity component `vθ/c₀`. The right panel shows the density change relative to the initial equilibrium. Each colour scale spans the computed values. Both panels show the meridional R–z plane of the axisymmetric model.
#
# The time series uses every accepted solution. The fitted growth rate measures behaviour in this calculation. It would require an independent comparison to validate that rate. The short profile ends before the fitting interval begins.

cream, ink, teal, copper = "#f8f2e8", "#233e39", "#176f67", "#ba6748"
cmap = LinearSegmentedColormap.from_list("vsi_signed", [teal, cream, copper])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c8c2b8", "text.color": ink,
                     "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink})
fig = plt.figure(figsize=(16,9), facecolor=cream)
grid = fig.add_gridspec(2, 2, height_ratios=[3.2,1.05], left=.075, right=.95,
                       bottom=.09, top=.82, hspace=.45, wspace=.23)
rr, tt = np.meshgrid(model.rf, model.tf, indexing="ij")
R_nodes, z_nodes = rr*np.cos(tt), rr*np.sin(tt)
fields_to_show = [
    (final_fields["vtheta"]/physical.c0, r"Meridional velocity $v_\theta/c_0$"),
    ((final_fields["density"]-initial_density)/initial_density, r"Density perturbation $(\rho-\rho_{eq})/\rho_{eq}$"),
]
for index,(values,title) in enumerate(fields_to_show):
    ax=fig.add_subplot(grid[0,index],facecolor=cream)
    limit=max(float(np.max(np.abs(values))),1e-15)
    mesh=ax.pcolormesh(R_nodes,z_nodes,values,cmap=cmap,vmin=-limit,vmax=limit,
                       shading="flat",rasterized=True)
    ax.set_aspect("equal")
    ax.set(xlabel=r"$R/r_0$",ylabel=r"$z/r_0$",title=title)
    bar=fig.colorbar(mesh,ax=ax,pad=.025,shrink=.93)
    bar.formatter.set_powerlimits((-2,2))
    bar.update_ticks()
ax=fig.add_subplot(grid[1,:],facecolor=cream)
positive=history["vrms"]>0
ax.semilogy(history["orbits"][positive],history["vrms"][positive],color=teal,lw=2)
ax.set(xlabel="Time / reference orbital period",ylabel=r"$v_{rms}/c_0$")
ax.grid(alpha=.18)
fig.text(.075,.94,"ONESIMULATION  /  HYDRODYNAMICS",fontsize=12,color=copper,weight="bold")
fig.text(.075,.89,"Vertical shear instability",fontsize=27,weight="bold")
fig.text(.075,.85,f"{box.nr} × {box.ntheta} cells · {history['orbits'][-1]:.3f} orbits · axisymmetric disk",fontsize=12)
fig.text(.075,.025,"Dimensionless units · time-dependent numerical solution · no experimental comparison",fontsize=10)
fig.savefig(output/"vsi_hydrodynamics.png",dpi=300,facecolor=cream)
fig.savefig(output/"vsi_hydrodynamics.svg",facecolor=cream)
plt.close(fig)

growth=fit_growth(history["orbits"],history["vrms"],time_min=5.,time_max=15.,min_efolds=.5)
verification={
    "passed": True, "runtime_version": version("onesimulation"), "profile": PROFILE,
    "cells": [box.nr,box.ntheta], "unknowns": int(model.initial_state.size),
    "end_orbits": float(history["orbits"][-1]), "maximum_step_code": config["dt_code"],
    "accepted_steps": int(result.values["step_count"]), "elapsed_seconds": elapsed_seconds,
    "mass_relative_max_error": mass_drift, "angular_momentum_relative_max_error": angular_drift,
    "final_diagnostics": diagnostics[-1], "growth_fit": growth,
    "scope": "Axisymmetric locally isothermal disk; no independent validation or saturation claim.",
    "outputs": ["onesimulation_results/vsi_hydrodynamics.png", "onesimulation_results/vsi_hydrodynamics.svg"],
}
(output/"vsi_verification.json").write_text(json.dumps(verification,indent=2,allow_nan=False)+"\n",encoding="utf-8")
print(json.dumps(verification,indent=2,allow_nan=False))
