"""OneSimulation example: Ogden parameters of rubber from Treloar's three classical tests.
Requires the installed, activated OneSimulation runtime.

The three-term Ogden model is fitted to Treloar's uniaxial, equibiaxial and pure-shear data of
vulcanised natural rubber. Every evaluation of the fit runs the three homogeneous material tests
of the incompressible material; a Levenberg-Marquardt least-squares fit estimates the six
parameters with standard errors. The result is compared with Ogden's own parameters of 1972,
and the misfit is shown per load mode.
"""

# # Ogden parameters from Treloar's tests
#
# The Ogden model writes the strain energy of rubber directly in the principal stretches,
#
# $$W=\sum_{p=1}^{N}\frac{2\mu_p}{\alpha_p^2}\big(\lambda_1^{\alpha_p}+\lambda_2^{\alpha_p}+\lambda_3^{\alpha_p}-3\big),\qquad
#   \lambda_1\lambda_2\lambda_3=1 ,$$
#
# with the initial shear modulus $\mu_0=\sum\mu_p$. With three terms it reproduces natural rubber
# over the whole range of the tests: a soft term with $\alpha\approx1.3$, a stiffening term with a
# large exponent and a term with a negative exponent for biaxial states. Six parameters cannot be
# found from one test; uniaxial tension, equibiaxial tension and pure shear together separate them.
#
# **Data.** L.R.G. Treloar, *Stress-strain data for vulcanised rubber under various types of
# deformation*, Trans. Faraday Soc. 40, 1944, 59-70: 23 points in uniaxial tension, 16 in
# equibiaxial tension and 13 in pure shear, from a widely used digitisation of the original
# figures (nominal stress converted from kgf/cm² to Pa). All 52 points enter with their relative
# deviation.
#
# **Reference.** R.W. Ogden, *Large deformation isotropic elasticity - on the correlation of theory
# and experiment for incompressible rubberlike solids*, Proc. R. Soc. Lond. A 326, 1972, 565-584,
# fitted the same tests with $\alpha=(1.3,\,5.0,\,-2.0)$ and, in the convention above,
# $\mu\approx(402,\,3.0,\,9.8)$ kPa (his coefficients $6.3$, $0.012$, $-0.1$ kgf/cm² times $\alpha_p/2$).
# The fit must find a set of the same character with a misfit no larger than his.

import numpy as np
from onesim.materials.solids import IncompressibleVolumeLaw, IsochoricOgden
from onesim.numerics.linear import SparseDirect
from onesim.numerics.nonlinear import DampedNewton, centered_array_jacobian
from onesim.physics.solid_mechanics import HomogeneousFiniteStrainTest
from onesim.simulation.studies import SimulationStudyService, StudyParameter
from onesim.visualization import plots, theme

theme.set()

# ## The measurements
#
# Stretch and nominal stress of the three tests.

treloar = {
    "uniaxial": (
        np.array(
            [
                1.1338,
                1.2675,
                1.3567,
                1.6242,
                1.8917,
                2.1592,
                2.4268,
                3.051,
                3.586,
                4.0318,
                4.7898,
                5.3694,
                5.8153,
                6.172,
                6.4395,
                6.707,
                6.9299,
                7.0637,
                7.1975,
                7.3312,
                7.465,
                7.5541,
                7.6433,
            ]
        ),
        98066.5
        * np.array(
            [
                1.5506,
                2.4367,
                3.1013,
                4.2089,
                5.3165,
                5.981,
                6.8671,
                8.8608,
                10.6329,
                12.4051,
                16.1709,
                19.9367,
                23.481,
                27.4684,
                31.0127,
                34.557,
                38.3228,
                42.0886,
                45.6329,
                49.3987,
                53.1646,
                56.9304,
                64.2405,
            ]
        ),
    ),
    "equibiaxial": (
        np.array(
            [
                1.02,
                1.06,
                1.11,
                1.14,
                1.2,
                1.31,
                1.42,
                1.68,
                1.94,
                2.49,
                3.03,
                3.43,
                3.75,
                4.07,
                4.26,
                4.45,
            ]
        ),
        np.array(
            [
                92026,
                155926,
                236213,
                257130,
                325973,
                434219,
                508279,
                647474,
                762899,
                959649,
                1239080,
                1439655,
                1706357,
                1971705,
                2201613,
                2417633,
            ],
            dtype=float,
        ),
    ),
    "pure_shear": (
        np.array(
            [
                1.069,
                1.1034,
                1.1724,
                1.2828,
                1.4276,
                1.8483,
                2.3862,
                3.0,
                3.4897,
                4.0345,
                4.4483,
                4.7793,
                5.0621,
            ]
        ),
        np.array(
            [
                58840,
                156906,
                235360,
                329503,
                411879,
                588399,
                764919,
                941438,
                1090499,
                1270942,
                1459230,
                1625943,
                1784810,
            ],
            dtype=float,
        ),
    ),
}

# ## The material tests
#
# `HomogeneousFiniteStrainTest` is a single material point under the homogeneous stretch of the
# chosen mode, with zero traction on the free directions and the incompressibility constraint;
# a damped Newton method finds the free stretch and the pressure at every point of the curve.

newton = DampedNewton(
    linear_solver=SparseDirect(),
    relative_tolerance=0.0,
    absolute_tolerance=2e-12,
    maximum_iterations=80,
).bind()
tests = lambda mu, alpha, stretches: {
    mode: HomogeneousFiniteStrainTest(
        material=IsochoricOgden(mu=mu, alpha=alpha),
        volume_law=IncompressibleVolumeLaw(),
    ).solve(
        mode=mode,
        stretches=stretches[mode],
        solver=newton,
        linearize=centered_array_jacobian,
        stress_scale=1e6,
    )
    for mode in treloar
}
measured_stretch = {mode: stretch for mode, (stretch, _) in treloar.items()}

# ## The fit
#
# Six parameters, each inside physical bounds; the third term keeps a negative exponent. The
# study service runs the three tests for every trial set and minimises the relative residuals
# with the Levenberg-Marquardt method, starting from rounded values.

names = ("mu1", "mu2", "mu3", "alpha1", "alpha2", "alpha3")
study = SimulationStudyService(
    tuple(
        StudyParameter(name, low, high)
        for name, low, high in zip(
            names, (1e5, 1.0, 1.0, 0.5, 3.0, -5.0), (1e6, 2e4, 8e4, 3.0, 9.0, -1.0)
        )
    ),
    build=lambda v: tests(
        (v["mu1"], v["mu2"], v["mu3"]),
        (v["alpha1"], v["alpha2"], v["alpha3"]),
        measured_stretch,
    ),
    solve=lambda results: results,
    measure=lambda results: {
        f"{mode}_{k}": float(value)
        for mode, result in results.items()
        for k, value in enumerate(result.nominal_stress[:, 0, 0])
    },
)
fit = study.fit(
    residuals=lambda values: np.concatenate(
        [
            np.array([values[f"{mode}_{k}"] for k in range(len(stress))]) / stress - 1
            for mode, (_, stress) in treloar.items()
        ]
    ),
    start={
        "mu1": 4e5,
        "mu2": 2e3,
        "mu3": 1e4,
        "alpha1": 1.5,
        "alpha2": 5.0,
        "alpha3": -2.0,
    },
    method="levenberg_marquardt",
    maximum_runs=2000,
)

# ## Comparison with the reference
#
# The fitted parameters with their standard errors beside Ogden's, and the misfit of both
# parameter sets in each load mode.

fitted = (
    tuple(fit.parameters[n] for n in names[:3]),
    tuple(fit.parameters[n] for n in names[3:]),
)
ogden_1972 = (
    (0.618e6 * 1.3 / 2, 1.18e3 * 5.0 / 2, -9.8e3 * -2.0 / 2),
    (1.3, 5.0, -2.0),
)
print(f"{'':16s}" + "".join(f"{name:>11s}" for name in names))
print(
    f"{'fit':16s}"
    + "".join(
        f"{fit.parameters[n] / (1e3 if n.startswith('mu') else 1):11.3f}" for n in names
    )
    + "   (μ in kPa)"
)
print(
    f"{'standard error':16s}"
    + "".join(
        f"{fit.standard_errors[n] / (1e3 if n.startswith('mu') else 1):11.3f}"
        for n in names
    )
)
print(
    f"{'Ogden (1972)':16s}"
    + "".join(
        f"{value / (1e3 if k < 3 else 1):11.3f}"
        for k, value in enumerate(ogden_1972[0] + ogden_1972[1])
    )
)
print(
    f"initial shear modulus: fit {sum(fitted[0]) / 1e3:.1f} kPa, Ogden (1972) {sum(ogden_1972[0]) / 1e3:.1f} kPa; "
    f"converged: {fit.converged}, rank {fit.rank}, {fit.run_count} parameter evaluations"
)
results = {
    "fit": tests(*fitted, measured_stretch),
    "Ogden (1972)": tests(*ogden_1972, measured_stretch),
}
print(
    f"{'relative RMS deviation':24s}"
    + "".join(f"{mode:>14s}" for mode in treloar)
    + f"{'all 52 points':>15s}"
)
for label, result in results.items():
    deviation = {
        mode: result[mode].nominal_stress[:, 0, 0] / stress - 1
        for mode, (_, stress) in treloar.items()
    }
    print(
        f"{label:24s}"
        + "".join(
            f"{100 * np.sqrt(np.mean(values**2)):13.1f}%"
            for values in deviation.values()
        )
        + f"{100 * np.sqrt(np.mean(np.concatenate(list(deviation.values())) ** 2)):14.1f}%"
    )

# ## Figure
#
# The three tests with the fitted model and with Ogden's parameters, evaluated on fine stretch
# grids, and the relative deviation of the fit from every measured point.

fine = {
    mode: np.linspace(1.0 + 1e-3, stretch[-1], 80)
    for mode, (stretch, _) in treloar.items()
}
curves = {
    label: tests(*parameters, fine)
    for label, parameters in (("fit", fitted), ("Ogden (1972)", ogden_1972))
}
fig = plots.figure(1, 2, size=(16, 6))
plots.line(
    y={
        mode.replace("_", " "): (
            fine[mode],
            curves["fit"][mode].nominal_stress[:, 0, 0] / 1e6,
        )
        for mode in treloar
    },
    reference={
        mode.replace("_", " "): (
            fine[mode],
            curves["Ogden (1972)"][mode].nominal_stress[:, 0, 0] / 1e6,
        )
        for mode in treloar
    },
    points={
        mode.replace("_", " "): (stretch, stress / 1e6)
        for mode, (stretch, stress) in treloar.items()
    },
    xlabel="stretch λ",
    ylabel="nominal stress P [MPa]",
    title="Treloar (markers), fit (lines), Ogden 1972 (dashed)",
    legend="upper left",
    ax=fig[0],
)
plots.line(
    y={
        mode.replace("_", " "): (
            stretch,
            100 * (results["fit"][mode].nominal_stress[:, 0, 0] / stress - 1),
        )
        for mode, (stretch, stress) in treloar.items()
    },
    hlines=0.0,
    markers=True,
    xlabel="stretch λ",
    ylabel="(P_model - P_Treloar) / P_Treloar [%]",
    title="Relative deviation of the fit",
    legend="lower right",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# Fit only the uniaxial test and watch the equibiaxial prediction drift: the parameters are then
# poorly determined, which the standard errors show at once. Drop the third term for a two-term
# Ogden model, or compare with the Mooney-Rivlin fit of the same data.
