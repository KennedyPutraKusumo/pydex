from pydex.core.designer import Designer
import numpy as np


"""
Setting     : a non-dynamic experimental system with 2 time-invariant control variables 
              and 1 response.
Problem     : design prediction-oriented optimal experiment for a order 1 polynomial.
Solution    : depends on criterion:
              ~ dg, ag, eg criteria: OFAT experiments
              ~ di, ai, ei criteria: 2^2 factorial design
"""
def simulate(ti_controls, model_parameters):
    return np.array([
        # constant term
        model_parameters[0] +
        # linear term
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1]
    ])


designer = Designer()
designer.simulate = simulate
designer.model_parameters = np.ones(3)  # values won't affect design, but still needed
designer.ti_controls_candidates = designer.enumerate_candidates(
    bounds=[
        [-1, 1],
        [-1, 1],
    ],
    levels=[
        11,
        11,
    ],
)
designer.initialize(verbose=2)  # 0: silent, 1: overview, 2: detailed, 3: very detailed

""" designing experiment """
package, optimizer = ("scipy", "SLSQP")
criterion = designer.dg_opt_criterion
designer.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    write=False,
)

designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls(non_opt_candidates=True, write=True, title=False)

criterion = designer.ag_opt_criterion
designer.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    write=False,
)

designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls(non_opt_candidates=True)

criterion = designer.eg_opt_criterion
designer.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    write=False,
)

designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls(non_opt_candidates=True)

criterion = designer.di_opt_criterion
designer.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    write=False,
)

designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls(non_opt_candidates=True)

criterion = designer.ai_opt_criterion
designer.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    write=False,
)

designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls(non_opt_candidates=True)

criterion = designer.ei_opt_criterion
designer.design_experiment(
    criterion=criterion,
    package=package,
    optimizer=optimizer,
    write=False,
)

designer.print_optimal_candidates()
designer.plot_optimal_efforts()
designer.plot_optimal_controls(non_opt_candidates=True)
designer.show_plots()
