from pydex.core.designer import Designer
from examples.ode.case_2_model import create_model, simulate

import numpy as np


""" loading only saves states and results, need to redeclare the model, simulate function, and simulator """
model, simulator = create_model()

designer_1 = Designer()
designer_1.model = model
designer_1.simulate = simulate
designer_1.simulator = simulator

""" loading state (experimental candidates, nominal model parameter values  """
designer_1.load_state('/case_2_result/date_2020-4-16/state_1_10x10x10x10.pkl')

""" loading sensitivity values from previous run """
designer_1.load_sensitivity('/case_2_result/date_2020-4-16/sensitivity_1_3x3x3x3.pkl')

"""" re-initialize the designer """
designer_1.initialize(verbose=2)
designer_1.simulate_candidates()
designer_1.plot_predictions()

""" estimability study without redoing sensitivity analysis """
designer_1.responses_scales = np.ones(2)  # equal scale of responses
designer_1.estimability_study()
designer_1.estimability_study_fim()

""" design experiment without redoing sensitivity analysis """
package, optimizer = ("cvxpy", "MOSEK")
# package, optimizer = ("cvxpy", "SCS")
# package, optimizer = ("cvxpy", "CVXOPT")
# package, optimizer = ("scipy", "SLSQP")

criterion = designer_1.d_opt_criterion
# criterion = designer_1.a_opt_criterion
# criterion = designer_1.e_opt_criterion

d_opt_result = designer_1.design_experiment(
    criterion=designer_1.d_opt_criterion,
    package=package,
    write=False,
    optimize_sampling_times=True,
    optimizer=optimizer
)
designer_1.print_optimal_candidates()
designer_1.plot_optimal_sensitivities()
designer_1.show_plots()
