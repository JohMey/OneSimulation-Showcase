"""OneSimulation example: elastoplastic material parameters from a cyclic shear test.
Requires the installed, activated OneSimulation runtime.

Young's modulus, the yield stress and the isotropic and kinematic hardening moduli of a steel are
identified from a cyclic shear test on a single material point. Every evaluation of the fit runs
the J2 plasticity model through the whole strain cycle; a least-squares fit with the measurement
noise as weight returns the parameters with their standard errors. The measurements are
synthetic, so the recovered values can be checked against the ones that made them.
"""

# # Elastoplastic parameters from a cyclic shear test
#
# Metals loaded beyond the yield stress and then unloaded in the opposite direction yield again
# earlier than in the first loading: the **Bauschinger effect**. A combined hardening law captures
# it by letting the elastic range both grow (isotropic hardening, modulus $H_\mathrm{iso}$) and move
# with the plastic flow (kinematic hardening, backstress modulus $H_\mathrm{kin}$). With von Mises
# (J2) plasticity the yield function and the evolution laws are
#
# $$f=\sqrt{\tfrac32}\,\|\mathbf s-\boldsymbol\alpha\|-\big(\sigma_y+H_\mathrm{iso}\,\bar\varepsilon_p\big)\le0,\qquad
#   \dot{\boldsymbol\alpha}=\tfrac23H_\mathrm{kin}\,\dot{\boldsymbol\varepsilon}_p ,$$
#
# with the stress deviator $\mathbf s$, the backstress $\boldsymbol\alpha$ and the accumulated plastic
# strain $\bar\varepsilon_p$ (J.C. Simo and T.J.R. Hughes, *Computational Inelasticity*, Springer
# 1998, Ch. 2-3).
#
# A **cyclic shear test** drives the engineering shear strain $\gamma$ between $\pm1$ % for two
# cycles while all other strains are held at zero, and records the shear stress $\tau$. The first
# loading sets $E$ (through $G=E/2(1+\nu)$ with $\nu=0.3$ fixed) and the yield stress; the reverse
# yield points separate the isotropic from the kinematic hardening.
#
# **The measurements.** There is no laboratory record here: the "measured" stresses are the model
# with $E=200$ GPa, $\sigma_y=250$ MPa, $H_\mathrm{iso}=1$ GPa and $H_\mathrm{kin}=10$ GPa, plus
# Gaussian noise with a standard deviation of 2 MPa. **Reference:** these values; the fit should
# return them within its standard errors.

import numpy as np
from onesim.materials.solids import (
    CombinedHardening,
    IsotropicHardening,
    IsotropicHookeMaterial,
    J2Plasticity,
    KinematicHardening,
)
from onesim.physics.solid_mechanics import HomogeneousSmallStrainTest, J2MaterialPoint
from onesim.simulation.studies import (
    MaterialPointStudy,
    SimulationStudyService,
    StudyParameter,
)
from onesim.visualization import plots, theme

theme.set()

# ## The material test
#
# A homogeneous material point under prescribed strain: no mesh, no spatial solver, the J2 return
# mapping runs step by step through the cycle. `specimen` builds the test for one set of
# parameters, `MaterialPointStudy` binds it to the loading path.

poisson = 0.3
shear_strain = np.r_[
    np.linspace(0, 0.01, 21),
    np.linspace(0.01, -0.01, 41)[1:],
    np.linspace(-0.01, 0.01, 41)[1:],
    np.linspace(0.01, -0.01, 41)[1:],
]
loading = dict(
    times=np.arange(len(shear_strain), dtype=float),
    strain={
        name: shear_strain if name == "xy" else 0.0
        for name in ("xx", "yy", "zz", "xy", "yz", "xz")
    },
)
specimen = lambda E, Y, H_iso, H_kin: MaterialPointStudy(
    loading=loading,
    test=HomogeneousSmallStrainTest(
        elasticity=IsotropicHookeMaterial(E, poisson),
        material=J2MaterialPoint(
            plasticity=J2Plasticity(
                shear_modulus=E / (2 * (1 + poisson)),
                hardening=CombinedHardening(
                    IsotropicHardening(Y, H_iso), KinematicHardening(modulus=H_kin)
                ),
            )
        ),
    ),
)
truth = {"E": 200e9, "Y": 250e6, "H_iso": 1e9, "H_kin": 10e9}
noise = 2e6
measured = specimen(**truth).solve_stationary().stress[
    :, 3
] + noise * np.random.default_rng(7).standard_normal(len(shear_strain))

# ## The fit
#
# The study service runs the material test for every trial set and minimises the noise-weighted
# residuals $(\tau_\mathrm{sim}-\tau_\mathrm{meas})/s$ with the Levenberg-Marquardt method,
# starting 20-40 % away from the true values inside generous bounds. The covariance of the
# estimate follows from the Jacobian at the optimum.

study = SimulationStudyService(
    (
        StudyParameter("E", 120e9, 300e9),
        StudyParameter("Y", 100e6, 500e6),
        StudyParameter("H_iso", 0.1e9, 5e9),
        StudyParameter("H_kin", 2e9, 40e9),
    ),
    build=lambda values: specimen(**values),
    measure=lambda result: {
        f"tau_{k}": float(value) for k, value in enumerate(result.stress[:, 3])
    },
)
fit = study.fit(
    residuals=lambda values: (
        np.array([values[f"tau_{k}"] for k in range(len(shear_strain))]) - measured
    )
    / noise,
    start={"E": 160e9, "Y": 320e6, "H_iso": 1.4e9, "H_kin": 6e9},
    method="levenberg_marquardt",
    maximum_runs=300,
)

# ## Comparison with the reference
#
# The fitted parameters against the values behind the data, in units of their standard errors, and
# the misfit, which should be close to the number of measurements less the four parameters when
# the fit is as good as the noise.

print(
    f"{'parameter':10s} {'true':>10s} {'fitted':>10s} {'standard error':>15s} {'deviation':>10s} {'in standard errors':>19s}"
)
for name, unit in (("E", 1e9), ("Y", 1e6), ("H_iso", 1e9), ("H_kin", 1e9)):
    value, error = fit.parameters[name], fit.standard_errors[name]
    print(
        f"{name:10s} {truth[name] / unit:10.3f} {value / unit:10.3f} {error / unit:15.4f} {100 * (value / truth[name] - 1):+9.2f} %"
        f" {(value - truth[name]) / error:+19.2f}"
        + ("   [GPa]" if unit == 1e9 else "   [MPa]")
    )
print(
    f"χ² = {fit.residuals @ fit.residuals:.1f} for {len(shear_strain)} measurements and 4 parameters; "
    f"converged: {fit.converged}, rank {fit.rank}, {fit.run_count} material tests"
)
correlation = np.asarray(fit.covariance) / np.outer(
    np.sqrt(np.diag(fit.covariance)), np.sqrt(np.diag(fit.covariance))
)
print("correlation of the estimates (E, Y, H_iso, H_kin):")
print(np.array2string(correlation, precision=2, suppress_small=True))

# ## Figure
#
# The measured and fitted hysteresis loops, and the normalised residuals along the test.

fitted = specimen(**fit.parameters).solve_stationary().stress[:, 3]
fig = plots.figure(1, 2, size=(16, 6))
plots.line(
    y={"fitted combined hardening": (100 * shear_strain, fitted / 1e6)},
    points={"measured (synthetic)": (100 * shear_strain, measured / 1e6)},
    xlabel="shear strain γ [%]",
    ylabel="shear stress τ [MPa]",
    title="Cyclic shear test",
    legend="upper left",
    ax=fig[0],
)
plots.line(
    points={
        "(τ_fit - τ_meas) / s": (
            np.arange(len(shear_strain)),
            (fitted - measured) / noise,
        )
    },
    hlines=0.0,
    ylimits=(-4, 4),
    xlabel="load step",
    ylabel="normalised residual",
    title="Residuals of the fit",
    legend="upper right",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Fit only isotropic hardening to the same data and see the reverse yielding missed; add the
# recovery term of the kinematic law, `KinematicHardening(modulus, recovery)`, for an
# Armstrong-Frederick backstress; or replace the shear test by `uniaxial_stress` loading.
