"""OneSimulation example: fitting the Mooney-Rivlin model to Treloar's rubber data.
Requires the installed, activated OneSimulation runtime.

Treloar's classical tests on vulcanised natural rubber in uniaxial tension, equibiaxial tension
and pure shear are fitted with the two-parameter Mooney-Rivlin model. Every evaluation of the fit
runs the three homogeneous material tests of the incompressible material; a Levenberg-Marquardt
least-squares fit returns the two moduli with standard errors. The result is compared with the
closed-form linear least-squares solution, and the misfit is shown per load mode.
"""

# # Mooney-Rivlin fit to Treloar's data
#
# Rubber is nearly incompressible and stays elastic up to stretches of 5 to 7. Its behaviour is
# described by a strain energy density of the isochoric invariants $\bar I_1,\bar I_2$; the
# Mooney-Rivlin model is the polynomial of first order,
#
# $$W=c_1(\bar I_1-3)+c_2(\bar I_2-3).$$
#
# For the incompressible material the nominal stress (force per original area) in the stretched
# direction of the three classical tests is
#
# $$P_\mathrm{uni}=2\Big(\lambda-\frac1{\lambda^2}\Big)\Big(c_1+\frac{c_2}{\lambda}\Big),\quad
#   P_\mathrm{bi}=2\Big(\lambda-\frac1{\lambda^5}\Big)\big(c_1+\lambda^2c_2\big),\quad
#   P_\mathrm{shear}=2\Big(\lambda-\frac1{\lambda^3}\Big)\big(c_1+c_2\big).$$
#
# **Data.** L.R.G. Treloar, *Stress-strain data for vulcanised rubber under various types of
# deformation*, Trans. Faraday Soc. 40, 1944, 59-70: 23 points in uniaxial tension, 16 in
# equibiaxial tension and 13 in pure shear, from a widely used digitisation of the original
# figures (nominal stress converted from kgf/cm² to Pa). All 52 points enter the fit with their
# relative deviation, so that small and large stresses count alike.
#
# **Reference.** The model is linear in $c_1,c_2$, so the best fit also follows from a direct linear
# least-squares solution of the formulas above: the fit through the material tests must arrive
# there.

import numpy as np
from onesim.materials.solids import (
    IncompressibleVolumeLaw,
    IsochoricInvariantPolynomial,
)
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
tests = lambda c1, c2: {
    mode: HomogeneousFiniteStrainTest(
        material=IsochoricInvariantPolynomial.mooney_rivlin(c1, c2),
        volume_law=IncompressibleVolumeLaw(),
    ).solve(
        mode=mode,
        stretches=stretch,
        solver=newton,
        linearize=centered_array_jacobian,
        stress_scale=1e6,
    )
    for mode, (stretch, _) in treloar.items()
}

# ## The fit
#
# The study service evaluates the three tests for every trial pair and minimises the relative
# residuals with the Levenberg-Marquardt method. The standard errors come from the covariance of
# the linearised problem, with the noise level estimated from the residuals.

study = SimulationStudyService(
    (StudyParameter("c1", 1e3, 1e6), StudyParameter("c2", 0.0, 3e5)),
    build=lambda values: tests(values["c1"], values["c2"]),
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
    start={"c1": 1.7e5, "c2": 2e4},
    method="levenberg_marquardt",
    maximum_runs=400,
)

# ## Comparison with the reference
#
# The fitted moduli against the direct linear least-squares solution of the same relative
# problem, and the misfit of the fitted model in each load mode.

shapes = {
    "uniaxial": lambda l: 2
    * (l - l**-2)[:, None]
    * np.column_stack((np.ones_like(l), 1 / l)),
    "equibiaxial": lambda l: 2
    * (l - l**-5)[:, None]
    * np.column_stack((np.ones_like(l), l**2)),
    "pure_shear": lambda l: 2
    * (l - l**-3)[:, None]
    * np.column_stack((np.ones_like(l), np.ones_like(l))),
}
direct = np.linalg.lstsq(
    np.vstack(
        [
            shapes[mode](stretch) / stress[:, None]
            for mode, (stretch, stress) in treloar.items()
        ]
    ),
    np.ones(52),
    rcond=None,
)[0]
best = tests(fit.parameters["c1"], fit.parameters["c2"])
print(f"{'':30s} {'c1 [kPa]':>10s} {'c2 [kPa]':>10s}")
print(
    f"{'fit of the material tests':30s} {fit.parameters['c1'] / 1e3:10.3f} {fit.parameters['c2'] / 1e3:10.3f}"
)
print(
    f"{'standard error':30s} {fit.standard_errors['c1'] / 1e3:10.3f} {fit.standard_errors['c2'] / 1e3:10.3f}"
)
print(
    f"{'direct linear least squares':30s} {direct[0] / 1e3:10.3f} {direct[1] / 1e3:10.3f}"
)
print(
    f"largest relative difference to the direct solution: {np.max(np.abs(np.array([fit.parameters['c1'], fit.parameters['c2']]) / direct - 1)):.1e}; "
    f"converged: {fit.converged}, {fit.run_count} parameter evaluations"
)
print(
    f"initial shear modulus μ0 = 2(c1 + c2) = {2 * (fit.parameters['c1'] + fit.parameters['c2']) / 1e3:.1f} kPa"
)
for mode, (stretch, stress) in treloar.items():
    deviation = best[mode].nominal_stress[:, 0, 0] / stress - 1
    print(
        f"{mode:12s} relative RMS deviation {100 * np.sqrt(np.mean(deviation**2)):5.1f} %, largest {100 * np.abs(deviation).max():5.1f} %"
        f" (at λ = {stretch[np.abs(deviation).argmax()]:.2f})"
    )

# ## Figure
#
# The three tests with the fitted model, evaluated on fine stretch grids, and the relative
# deviation of the model from every measured point.

fine = {
    mode: np.linspace(1.0 + 1e-3, stretch[-1], 80)
    for mode, (stretch, _) in treloar.items()
}
curves = {
    mode: HomogeneousFiniteStrainTest(
        material=IsochoricInvariantPolynomial.mooney_rivlin(
            fit.parameters["c1"], fit.parameters["c2"]
        ),
        volume_law=IncompressibleVolumeLaw(),
    )
    .solve(
        mode=mode,
        stretches=fine[mode],
        solver=newton,
        linearize=centered_array_jacobian,
        stress_scale=1e6,
    )
    .nominal_stress[:, 0, 0]
    for mode in treloar
}
fig = plots.figure(1, 2, size=(16, 6))
plots.line(
    y={mode.replace("_", " "): (fine[mode], curves[mode] / 1e6) for mode in treloar},
    points={
        mode.replace("_", " "): (stretch, stress / 1e6)
        for mode, (stretch, stress) in treloar.items()
    },
    xlabel="stretch λ",
    ylabel="nominal stress P [MPa]",
    title="Treloar (markers) and the Mooney-Rivlin fit (lines)",
    legend="upper left",
    ax=fig[0],
)
plots.line(
    y={
        mode.replace("_", " "): (
            stretch,
            100 * (best[mode].nominal_stress[:, 0, 0] / stress - 1),
        )
        for mode, (stretch, stress) in treloar.items()
    },
    hlines=0.0,
    markers=True,
    xlabel="stretch λ",
    ylabel="(P_model - P_Treloar) / P_Treloar [%]",
    title="Relative deviation of the fit",
    legend="lower left",
    ax=fig[1],
)
plots.show()

# ## Going further
#
# The two-term model misses the stiffening at large stretch in uniaxial tension and overshoots
# in equibiaxial tension: fit the Ogden model to the same data (the Ogden example), or a
# higher-order polynomial `IsochoricInvariantPolynomial(((i, j, c_ij), ...))`.
