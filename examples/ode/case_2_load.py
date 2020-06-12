from pydex.core.designer import Designer
from examples.ode.case_1_pyomo_model import create_model, simulate, create_simulator

import numpy as np


""" loading only saves states and results, need to redeclare the model, simulate function, and simulator """
model_1 = create_model()

designer_1 = Designer()
designer_1.model = model_1
designer_1.simulate = simulate
designer_1.simulator = create_simulator(model_1, package='casadi')

""" loading state (experimental candidates, nominal model parameter values  """
designer_1.load_state('/ode_oed_case_2_result/date_2020-4-16/state_1.pkl')

""" loading sensitivity values from previous run """
designer_1.load_sensitivity('/ode_oed_case_2_result/date_2020-4-16/sensitivity_1.pkl')

"""" re-initialize the designer """
designer_1.initialize()

""" estimability study without redoing sensitivity analysis """
designer_1.responses_scales = np.ones(2)  # equal scale of responses
# designer_1.estimability_study_fim()

""" design experiment without redoing sensitivity analysis """
package, optimizer = ("cvxpy", "MOSEK")
# package, optimizer = ("cvxpy", "SCS")
# package, optimizer = ("cvxpy", "CVXOPT")
# package, optimizer = ("scipy", "SLSQP")

criterion = designer_1.d_opt_criterion
# criterion = designer_1.a_opt_criterion
# criterion = designer_1.e_opt_criterion

d_opt_result = designer_1.design_experiment(criterion=designer_1.d_opt_criterion,
                                            package=package, write=False,
                                            optimize_sampling_times=True,
                                            optimizer=optimizer)
designer_1.print_optimal_candidates()
designer_1.plot_optimal_sensitivities()
designer_1.show_plots()
