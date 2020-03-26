from core.designer import Designer
from examples.ode_oed.ode_oed_case_1_pyomo import create_model, simulate, create_simulator

""" loading only saves states and results, need to redeclare the model, simulate function, and simulator """
model_1 = create_model()

designer_1 = Designer()
designer_1.model = model_1
designer_1.simulate = simulate
designer_1.simulator = create_simulator(model_1, package='casadi')

""" loading state (experimental candidates, nominal model parameter values  """
designer_1.load_state('/ode_oed_case_1_pyomo_result/date_2020-3-25/state_1.pkl')

""" loading sensitivity values from previous run """
designer_1.load_sensitivity('/ode_oed_case_1_pyomo_result/date_2020-3-25/sensitivity_1.pkl')

"""" re-initialize the designer """
designer_1.initialize()

""" estimability study without redoing sensitivity analysis """
estimable_params = designer_1.estimability_study()
print(estimable_params)

""" design experiment without redoing sensitivity analysis """
d_opt_result = designer_1.design_experiment(criterion=designer_1.d_opt_criterion, package='cvxpy', plot=False,
                                            optimize_sampling_times=True, write=True, optimizer='ECOS_BB')
designer_1.plot_current_design()
