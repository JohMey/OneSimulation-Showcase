"""Generated from the OneSimulation example notebook.
Needs the installed, activated OneSimulation runtime. See
https://johmey.github.io/OneSimulation-Showcase/examples/notebooks.html
"""


# # ECR plasma with coupled fields and reactions
#
# This example runs on an activated installation of OneSimulation Runtime Alpha 0.2.1a4. The runtime couples a frequency-domain Nédélec Maxwell solve to particle transport, electron energy, Poisson’s equation and three explicitly declared reactions. The coupled update is implicit.
#
# The reduced model gas `G / G+ / e` and its illustrative coefficients demonstrate the coupling. They represent neither argon nor hydrogen. The example requires no external plasma or material catalogue and has no device-level validation.
#
# The `FAST` profile advances 384 tetrahedra to 1 µs. `HIGH_RESOLUTION` uses 6,000 tetrahedra and 45,860 real unknowns over the same interval. `SMOKE_TEST` is a short API check with 48 tetrahedra over 0.2 µs. The fine calculation took several minutes on the reference Mac; runtime and memory depend on the system.


# ## Model assumptions
#
# - The 80 × 80 × 80 mm cube has reflecting particle boundaries and grounded electrostatic potential. The model omits extraction optics and wall emission.
# - A prescribed localized complex current drives the microwave field at 2.45 GHz. Its amplitude gives 1 W of initial absorption. The drive then stays fixed while absorption responds to the plasma.
# - The prescribed divergence-free local magnetic mirror field has a gradient of 0.0045 T/m about `B_ECR = 2π m_e f / e ≈ 87.523 mT`. The model does not solve a coil.
# - Initially `n_e = n_G+ = 10¹⁵ m⁻³`, `n_G = 3 × 10²⁰ m⁻³`, `T_e = 3 eV` and the gas temperature is 300 K. Three illustrative rates describe momentum transfer, ionization and recombination.
# - The model uses physical SI permittivity. The numerical collision floor `10⁻⁴ × ω` lies below the initial model value of `1.5 × 10⁷ s⁻¹`. It does not set the initial resonance width. Maxwell failures raise an error with `fallback="raise"`.
# - The frequency-domain Maxwell equation resolves the RF phasors. The 100 ns steps advance the slower plasma evolution.
#
# The initial Debye length is about 0.407 mm. Even the fine profile has an axial cell spacing of 8 mm, so the Debye sheath is unresolved. This closed bulk model uses simplified chemistry and wall conditions. Spatial order and time step also limit quantitative accuracy.
#
# The [RRCAT ECR ion source description](https://www.rrcat.gov.in/technology/accel/indus/pldd/ISDDS_ECRIS.html) gives the frequency–field relation. This example does not reproduce that source or its measurements.

import json
import time
from pathlib import Path
from dataclasses import replace
from importlib.metadata import version
import numpy as np
from scipy.constants import electron_mass, elementary_charge
from onesim.cad.d3 import Box
from onesim.meshing import StructuredTetrahedralMesher
from onesim.physics.plasma.data.chemistry import ChemistryDataCatalog
from onesim.physics.plasma.data.chemistry_library import PlasmaChemistry, TransportDataset
from onesim.physics.plasma.data.data_sources import PlasmaDataCatalog
from onesim.physics.plasma.fluid.binding.catalogs.registries import SpeciesRegistry, ReactionRegistry
from onesim.physics.plasma.fluid.models.declarations.species.species import SpeciesDefinition
from onesim.physics.plasma.collisions.models.rates import ConstantRateLaw, ArrheniusTeRateLaw
from onesim.physics.plasma.collisions.models.reactions import ReactionDefinition, EnergyChannel
from onesim.physics.plasma.fluid.models.declarations.conditions import PotentialBoundaryCondition, SpeciesBoundaryCondition
from onesim.physics.plasma.fluid.models.declarations.species.multispecies import MultispeciesTransportModel
from onesim.physics.plasma.walls.binding.boundary_binding import PlasmaBoundaryBinding

assert version("onesimulation") == "0.2.1a4", "This example was checked with Runtime Alpha 0.2.1a4."
PROFILE = "FAST"
PROFILES = {
    "SMOKE_TEST": {"n": 2, "dt": 1e-7, "steps": 2},
    "FAST": {"n": 4, "dt": 1e-7, "steps": 10},
    "HIGH_RESOLUTION": {"n": 10, "dt": 1e-7, "steps": 10},
}
assert PROFILE in PROFILES
NX, DT, STEPS = (PROFILES[PROFILE][k] for k in ("n", "dt", "steps"))
OUT = Path("onesimulation_results")
OUT.mkdir(exist_ok=True)
started = time.perf_counter()


# ## Solve with the runtime
#
# The declarations below supply the coupling API with the model inputs. The native provider forms the residuals and Jacobians, assembles the Maxwell system and advances the equations implicitly. The loop records each accepted state.

frequency=2.45e9
resonant_B=2*np.pi*electron_mass*frequency/elementary_charge
gas_mass=6.6e-26 # kg, deliberately declared generic heavy gas, not a data-table value
species=SpeciesRegistry((
    SpeciesDefinition("e","electron",-1,electron_mass,"electron"),
    SpeciesDefinition("G","illustrative neutral G",0,gas_mass,"neutral",composition=(("G",1),)),
    SpeciesDefinition("G+","illustrative positive ion G+",1,gas_mass-electron_mass,"ion",composition=(("G",1),),state="ionized"),
))
reactions=ReactionRegistry((
    ReactionDefinition("elastic_momentum",(("e",1),("G",1)),(("e",1),("G",1)),
        ConstantRateLaw(5e-14),reaction_type="momentum_transfer"),
    ReactionDefinition("illustrative_ionization",(("e",1),("G",1)),(("e",2),("G+",1)),
        ArrheniusTeRateLaw(2e-14,0.5,15.0),reaction_type="ionization",
        energy_channels=(EnergyChannel("electron",15.0,"ionization_loss"),)),
    ReactionDefinition("illustrative_recombination",(("e",1),("G+",1)),(("G",1),),
        ConstantRateLaw(1e-13),reaction_type="recombination"),
))
catalog=ChemistryDataCatalog("plasma.chemistry.v2","Explicit illustrative user parameters; no catalog files",species,reactions)
assert not catalog.validate(),catalog.validate()
explicit_transport=TransportDataset("explicit-reference-density",
    PlasmaDataCatalog(schema_version="plasma.data.v2",source="Explicit user reference density; no external data",objects=()),
    reference_density=2.5e25)
chemistry=PlasmaChemistry("explicit-three-species","1","illustrative",catalog,gas_temperature=300.0,
    transport=explicit_transport)
transport=MultispeciesTransportModel(
    neutral_diffusivity=(("G",1.0e-5),),reduced_mobility=(("G+",2e-4),),
    momentum_reaction_ids=("elastic_momentum",),electron_temperature_bounds=(0.1,50.0),
    electron_density_floor=1e6,collision_frequency_floor=1.0)
box=Box(length=.08,width=.08,height=.08)
mesh=StructuredTetrahedralMesher(cells=(NX,NX,NX)).generate(box)
centers=mesh.coordinates[mesh.cells].mean(axis=1)
relative=centers-.04
gradient=.0045 # T/m; weak imposed gradient resolves the collision-broadened ECR layer
magnetic=np.column_stack((-.5*gradient*relative[:,0],-.5*gradient*relative[:,1],
                          resonant_B+gradient*relative[:,2]))
boundary=PlasmaBoundaryBinding("wall",conditions=tuple(
    SpeciesBoundaryCondition("reflect:"+name,"wall",name,"density","reflecting",
                             value={"gas_temperature":300.0}) for name in chemistry.species_ids()),
    marker_id="wall",potential=PotentialBoundaryCondition("ground","ground"))
from onesim.numerics.discretization.geometry.carriers.kernels.simplex.tetrahedral import tetrahedral_geometry, tetrahedral_dual_faces
from onesim.numerics.fvm.contracts.coupled_cell_transport import BoundaryFace
from onesim.physics.electromagnetics.waves.execution.vector_fields import FrequencyDomainMaxwellPublicApi
from onesim.physics.plasma.field_coupling.binding.coupling.frequency_transient import FrequencyTransientCoupling
from onesim.physics.plasma.fluid.numerics.backends.transport.multispecies_drift_diffusion import MultispeciesDriftDiffusionProvider
from onesim.physics.plasma.fluid.models.declarations.species.multispecies import MultispeciesDriftDiffusionProblem

faces=tuple(BoundaryFace(f.owner_cell,f.area,tuple(f.normal),tuple(f.centroid),
                        f.centre_distance,"wall","")
            for f in tetrahedral_dual_faces(tetrahedral_geometry(mesh.coordinates,mesh.cells))[1])
# Prescribed localized microwave-current input; the displayed fields are solved.
ant_pos=np.array([.02,.04,.04])
envelope=np.exp(-np.sum(((centers-ant_pos)/np.array([.014,.022,.022]))**2,axis=1))
current=np.zeros((len(centers),3),complex)
current[:,0]=envelope
current[:,1]=-1j*envelope
microwave=FrequencyTransientCoupling(frequency=frequency,power=1.0,
    impressed_current_density=current, maxwell_api=FrequencyDomainMaxwellPublicApi(element_order=1),
    resonance_collision_fraction=1e-4,fallback="raise")
provider=MultispeciesDriftDiffusionProvider(chemistry=chemistry,boundaries=(boundary,),
    microwave=microwave,em_coupling="monolithic",require_full_boundary_coverage=False,
    transport_model=transport,jacobian_method="analytic")
n0=1e15; neutral0=3e20
scales=tuple({"e":n0,"G+":n0,"G":neutral0}[s] for s in provider.species_ids)+(4.5e15,1.)
config=provider.default_numerical_configuration(final_time=DT*STEPS, initial_step=DT,
    maximum_step=DT,field_scales=scales,jacobian_reuse=4)
provider.numerical_configuration=replace(config,time_provider_id="numerics.time.internal.index-one-dae",
    time_method_id="backward_euler",nonlinear=replace(config.nonlinear,maximum_iterations=24))
problem=MultispeciesDriftDiffusionProblem(mesh.coordinates,mesh.cells,faces,
    (("e",n0),("G",neutral0),("G+",n0)),3.,magnetic_flux_density=magnetic,
    gas_temperature=300.,relative_permittivity=1.,permittivity_scale=1.)
print(json.dumps({"stage":"prepare","cells":len(mesh.cells),"dt_s":DT,"steps":STEPS}),flush=True)
runtime=provider.monolithic_em_runtime(problem)
print(json.dumps({"stage":"initialized","elapsed_s":time.perf_counter()-started,
                  "edge_dofs":runtime.edge_count,"real_unknowns":len(runtime.initial)}),flush=True)
state=runtime.initial.copy()
history=[]
def snapshot(state,t):
    plasma,edge=runtime._unpack_callback(state)
    context=runtime._context_callback(plasma,edge)
    q=microwave.absorption_from_coefficients(context["assembly"],edge,
        context["problem"].relative_permittivity,power_scale=runtime._power_scale)
    te=runtime.setup.electron_temperature(plasma)
    ne=plasma[:,provider.species_ids.index("e")]
    rates=runtime.setup.volume_rates(plasma,t)
    net_energy_W=float(rates[:,provider.energy_field_index]@runtime.volumes*elementary_charge
                       +(q-runtime.setup.power_density)@runtime.volumes)
    result={"time_s":float(t),"net_energy_source_W":net_energy_W,"absorbed_W":float(q@runtime.volumes),
            "Te_min_eV":float(te.min()),"Te_max_eV":float(te.max()),
            "ne_min_m3":float(ne.min()),"ne_max_m3":float(ne.max()),
            "ne_mean_m3":float(ne@runtime.volumes/runtime.volumes.sum()),
            "Te_mean_eV":float(te@runtime.volumes/runtime.volumes.sum()),
            "electron_energy_J":float(plasma[:,provider.energy_field_index]@runtime.volumes*elementary_charge)}
    history.append(result)
    return plasma,context,q,te,ne
snapshot(state,0.)
for step in range(STEPS):
    advance=runtime._integrator.advance(state,runtime,step*DT,DT)
    if not advance.accepted:
        raise RuntimeError(f"Native time integrator rejected step {step+1}: {advance!r}")
    state=advance.state
    plasma,context,q,te,ne=snapshot(state,(step+1)*DT)
    print(json.dumps({"stage":"step","step":step+1,"elapsed_s":time.perf_counter()-started,**history[-1]}),flush=True)
ion=plasma[:,provider.species_ids.index("G+")]
neutral=plasma[:,provider.species_ids.index("G")]
charge_error=abs(float((ion-ne)@runtime.volumes))/float(ne@runtime.volumes)
heavy_error=abs(float((neutral+ion)@runtime.volumes)/((neutral0+n0)*runtime.volumes.sum())-1)
energy_defects=[abs((b["electron_energy_J"]-a["electron_energy_J"])/DT-b["net_energy_source_W"])
                for a,b in zip(history[:-1],history[1:])]
assert max(energy_defects)<1e-6,max(energy_defects)
assert charge_error<1e-5,charge_error
assert heavy_error<1e-8,heavy_error
for a in (plasma,context["field"],q,te): assert np.isfinite(a).all()
assert ne.min()>0 and te.min()>0 and q.min()>=0

metrics={"passed":True,"cells":len(mesh.cells),"points":len(mesh.coordinates),
    "complex_hcurl_dofs":runtime.edge_count,"real_unknowns":len(runtime.initial),
    "elapsed_seconds":time.perf_counter()-started,"physical_time_s":DT*STEPS,
    "accepted_steps":STEPS,"dt_s":DT,"charge_relative_error":charge_error,
    "heavy_particle_relative_error":heavy_error,"max_energy_balance_defect_W":max(energy_defects),
    "permittivity_scale":1.,"resonance_collision_fraction":1e-4,
    "initial_collision_frequency_s":neutral0*5e-14,"numeric_collision_floor_s":2*np.pi*frequency*1e-4,
    "magnetic_gradient_Tm":gradient,"initial_resonance_axial_halfwidth_m":electron_mass*neutral0*5e-14/elementary_charge/gradient,
    "resonance_B_T":resonant_B,"initial_debye_length_m":float(np.sqrt(8.8541878128e-12*3/(n0*elementary_charge))),
    "axis_cell_size_m":.08/NX,"debye_sheath_resolved":False,
    "history":history,"power_density_minmax_Wm3":[float(q.min()),float(q.max())]}

data = {
    "coordinates": mesh.coordinates, "cells": mesh.cells,
    "electron_density": ne, "electron_temperature_eV": te,
    "absorbed_power_density_Wm3": q,
}
np.savetxt(OUT / "08_ecr_time_history.csv",
    np.array([[r["time_s"],r["absorbed_W"],r["ne_mean_m3"],r["Te_max_eV"]] for r in history]),
    delimiter=",",header="time_s,absorbed_W,ne_mean_m3,Te_max_eV",comments="")
print(json.dumps({k:v for k,v in metrics.items() if k != "history"},indent=2))


# ## Inspect the computed cell values
#
# The cuts intersect the mesh tetrahedra and display each cell value without image smoothing. The dashed centre line marks the approximate resonance plane of the prescribed magnetic field. The three orthogonal cuts in the 3D figure use the same computed absorption field.
#
# The checks cover positive finite fields, global charge conservation, heavy-particle conservation and output files. Mesh and time-step convergence and comparison with experimental measurements require separate studies. Comparisons for the published figure are documented alongside that figure.

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CREAM='#faf4e8'; INK='#244642'; TEAL='#167c80'; COPPER='#b7613d'
CMAP=LinearSegmentedColormap.from_list('onesim_ecr',['#f4ecdb','#9fc9bc','#257d82','#123e50','#b76743','#efbc69'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':INK,
    'axes.labelcolor':INK,'axes.edgecolor':'#a3afa3','xtick.color':INK,'ytick.color':INK,
    'figure.facecolor':CREAM,'axes.facecolor':CREAM,'savefig.facecolor':CREAM})

def plane_polygons(coordinates,cells,axis=1,level=.040001):
    polygons=[]; owners=[]
    for owner,cell in enumerate(cells):
        verts=coordinates[cell]; d=verts[:,axis]-level
        if np.min(d)>=0 or np.max(d)<=0: continue
        hits=[]
        for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
            if d[a]*d[b]<0:
                hits.append(verts[a]+(verts[b]-verts[a])*d[a]/(d[a]-d[b]))
        if len(hits)<3:continue
        p=np.asarray(hits); axes=[i for i in range(3) if i!=axis]
        c=p[:,axes].mean(axis=0); v=p[:,axes]-c
        p=p[np.argsort(np.arctan2(v[:,1],v[:,0]))]
        polygons.append(p);owners.append(owner)
    return polygons,np.array(owners,dtype=int)

def render(data,metrics,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    coordinates=data['coordinates'];cells=data['cells']
    polys,ids=plane_polygons(coordinates,cells)
    flat=[p[:,[0,2]]*1e3 for p in polys]
    q=data['absorbed_power_density_Wm3']/1e3
    ne=data['electron_density']/1e15
    te=data['electron_temperature_eV']
    fields=[(q,'Absorbed RF power','kW / m³'),(ne,'Electron density','10¹⁵ / m³'),(te,'Electron temperature','eV')]
    fig=plt.figure(figsize=(20,12.5))
    grid=fig.add_gridspec(2,3,left=.058,right=.964,top=.79,bottom=.14,hspace=.67,height_ratios=[2.45,1])
    fig.text(.058,.942,'ECR PLASMA',color=TEAL,size=16,weight='bold')
    fig.text(.058,.893,'How microwaves change a plasma',size=30,weight='bold')
    fig.text(.058,.848,'Implicitly coupled Maxwell fields, particle transport, electrostatic potential and reactions',size=15)
    fig.text(.965,.943,'ONESIMULATION\nRUNTIME ALPHA',ha='right',va='top',color=COPPER,size=11,weight='bold',linespacing=1.5)
    for i,(values,title,unit) in enumerate(fields):
        ax=fig.add_subplot(grid[0,i])
        norm=Normalize(float(values.min()),float(values.max()))
        collection=PolyCollection(flat,array=values[ids],cmap=CMAP,norm=norm,edgecolors='face',linewidths=.18,rasterized=False)
        ax.add_collection(collection)
        ax.set(xlim=(0,80),ylim=(0,80),aspect='equal',xlabel='x / mm',ylabel='z / mm')
        ax.set_title(title,loc='left',pad=16,size=15,weight='bold')
        ax.set_xticks([0,20,40,60,80]);ax.set_yticks([0,20,40,60,80])
        cb=fig.colorbar(collection,ax=ax,orientation='horizontal',pad=.2,fraction=.045,aspect=28)
        cb.set_label(unit,labelpad=5)
        # Prescribed magnetic resonance plane, explicitly an INPUT marker.
        ax.axhline(40,color='#f9e6b9',linestyle=(0,(3,3)),lw=.85,alpha=.85)
    h=metrics['history'];t=np.array([v['time_s'] for v in h])*1e6
    quantities=[([v['absorbed_W'] for v in h],'Absorption / W',COPPER),
                ([v['ne_mean_m3']/1e15 for v in h],'Mean density / 10¹⁵ m⁻³',TEAL),
                ([v['Te_max_eV'] for v in h],'Peak electron temperature / eV',TEAL)]
    for i,(v,label,color) in enumerate(quantities):
        ax=fig.add_subplot(grid[1,i]);ax.plot(t,v,'o-',color=color,lw=2.4,ms=4)
        ax.set(xlabel='Time / µs',ylabel=label,xlim=(0,max(t)))
        ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
    n=metrics['cells'];unknowns=metrics['real_unknowns'];time_us=metrics['physical_time_s']*1e6
    fig.text(.058,.086,f"{n:,} tetrahedra  ·  {unknowns:,} real unknowns" + f'  ·  {time_us:g} µs  ·  2.45 GHz  ·  B_ECR = 87.523 mT',size=13,weight='bold')
    fig.text(.058,.05,'Reduced model gas with illustrative coefficients. Unsmoothed cell values at y = 40.001 mm. Device validation is outside the scope of the example.',size=10.7)
    fig.text(.058,.027,'Prescribed magnetic field and RF current. Initial absorption is 1 W. Particle boundaries reflect. Debye sheaths remain unresolved.',size=10.7)
    fig.savefig(out/'08_ecr_plasma_fields.png',dpi=240)
    fig.savefig(out/'08_ecr_plasma_fields.svg')
    plt.close(fig)
    # Three orthogonal, exact sections of the SAME genuinely computed field.
    fig=plt.figure(figsize=(20,12.5));ax=fig.add_axes([.035,.16,.58,.64],projection='3d')
    norm=Normalize(0,float(q.max()))
    for axis,level in ((0,.040001),(1,.040001),(2,.040001)):
        p,index=plane_polygons(coordinates,cells,axis,level)
        collection=Poly3DCollection([v*1e3 for v in p],facecolors=CMAP(norm(q[index])),edgecolors='none',alpha=.96)
        ax.add_collection3d(collection)
    for x in (0,80):
        for y in (0,80):ax.plot([x,x],[y,y],[0,80],color=INK,alpha=.25,lw=.8)
    for z in (0,80):
        for x in (0,80):ax.plot([x,x],[0,80],[z,z],color=INK,alpha=.25,lw=.8)
        for y in (0,80):ax.plot([0,80],[y,y],[z,z],color=INK,alpha=.25,lw=.8)
    ax.set(xlim=(0,80),ylim=(0,80),zlim=(0,80),xlabel='x / mm',ylabel='y / mm',zlabel='z / mm')
    ax.set_box_aspect((1,1,1));ax.view_init(elev=24,azim=-52)
    for a in (ax.xaxis,ax.yaxis,ax.zaxis):a.pane.fill=False
    fig.text(.065,.93,'THREE-DIMENSIONAL ECR COUPLING',size=16,color=TEAL,weight='bold')
    fig.text(.065,.876,'Where the plasma absorbs microwave energy',size=30,weight='bold')
    fig.text(.065,.828,'Three sections through the coupled RF absorption field',size=14)
    ca=fig.add_axes([.12,.12,.40,.02]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=CMAP),cax=ca,orientation='horizontal')
    cb.set_label('Absorbed RF power density / kW m⁻³')
    fig.text(.67,.715,f'{n:,}',size=45,color=TEAL,weight='bold')
    fig.text(.67,.675,'tetrahedra in the computational mesh',size=14)
    fig.text(.67,.59,f'{unknowns:,}',size=40,color=COPPER,weight='bold')
    fig.text(.67,.55,'coupled real unknowns',size=14)
    fig.text(.67,.46,f'{float(te.max()):.2f} eV',size=34,color=TEAL,weight='bold')
    fig.text(.67,.42,'peak electron temperature',size=14)
    fig.text(.67,.33,f'{float(ne.max()):.2f} × 10¹⁵ m⁻³',size=26,color=TEAL,weight='bold')
    fig.text(.67,.29,'peak electron density',size=14)
    fig.text(.065,.05,f'Reduced model gas · {time_us:g} µs of evolution · 1 W initial absorption · prescribed magnetic field and RF current',size=11)
    fig.text(.065,.026,'Reflecting particle boundaries. The mesh does not resolve the Debye sheath. This calculation has no experimental device validation.',size=11)
    fig.savefig(out/'08_ecr_plasma_3d.png',dpi=240);fig.savefig(out/'08_ecr_plasma_3d.svg')
    plt.close(fig)
    return [str(out/name) for name in ('08_ecr_plasma_fields.png','08_ecr_plasma_fields.svg','08_ecr_plasma_3d.png','08_ecr_plasma_3d.svg')]


files = render(data, metrics, OUT)
verification = {
    "passed": True,
    "runtime_version": version("onesimulation"),
    "profile": PROFILE,
    "tetrahedra": int(len(mesh.cells)),
    "real_unknowns": int(len(runtime.initial)),
    "physical_time_s": float(DT*STEPS),
    "accepted_steps": int(STEPS),
    "global_charge_relative_error": float(charge_error),
    "heavy_particle_relative_error": float(heavy_error),
    "max_discrete_energy_balance_defect_W": float(max(energy_defects)),
    "permittivity_scale": 1.0,
    "chemistry": "Explicit reduced model gas G/e/G+, not empirical gas validation",
    "external_chemistry_payloads_required": False,
    "debye_sheath_resolved": False,
    "display": "Exact tetrahedron-plane intersections of native cell values; no smoothing",
    "initial_absorbed_power_W": float(history[0]["absorbed_W"]),
    "final_absorbed_power_W": float(history[-1]["absorbed_W"]),
    "files": files,
}
assert len(mesh.cells) == 6*NX**3
assert all(Path(p).is_file() and Path(p).stat().st_size > 1000 for p in files)
assert history[-1]["ne_mean_m3"] > history[0]["ne_mean_m3"]
assert metrics["numeric_collision_floor_s"] < metrics["initial_collision_frequency_s"]
(OUT / "08_ecr_verification.json").write_text(json.dumps(verification,indent=2)+"\n",encoding="utf-8")
print(json.dumps(verification,indent=2))
