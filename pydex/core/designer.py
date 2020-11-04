from datetime import datetime
from inspect import signature
from os import getcwd, path, makedirs
from pickle import dump, load
from string import Template
from time import time
import itertools
import __main__ as main
import dill

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.widgets import RadioButtons, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, least_squares
from pydex.utils.trellis_plotter import TrellisPlotter
from pydex.core.bnb.tree import Tree
from pydex.core.bnb.node import Node
import cvxpy as cp
import numdifftools as nd
import numpy as np


class Designer:
    """
    An experiment designer with capabilities to do parameter estimation, parameter
    estimability study, and computes both continuous and exact experimental designs.

    Interfaces to optimization solvers via scipy, and cvxpy. Supports virtually any Python
    functions as long as one can specify the model within the required general syntax.
    Special support for ODE models solved via Pyomo.DAE: allow model, and simulator to
    be passed to the designer to prevent re-build of model and simulator each time a
    the model is simulated, optimizing computational time.

    Designer comes equipped with convenient built-in visualization capabilities, using
    matplotlib.
    """
    def __init__(self):
        """ core model components """
        # unorganized
        self.model_parameter_unit_names = None
        self.response_unit_names = None
        self.time_unit_name = None
        self._num_steps = None
        self.opt_spt_combs = None
        self.opt_eff = None
        self.opt_spt = None
        self.opt_tvc = None
        self.opt_tic = None
        self.n_opt_c = None
        self._invariant_controls = None
        self._n_spt_spec = None
        self.mp_covar = None
        self._specified_n_spt = None
        self._discrete_design = None
        self.spt_binary = None
        self.spt_candidates_combs = None
        self.cost = None
        self.cand_cost = None
        self.spt_cost = None
        self._eps = 1e-5
        self._regularize_fim = None
        self._old_tvc = None
        self._old_spt = None
        self._tvc_changed = None
        self._spt_changed = None
        self._tic_changed = None
        self._old_tic = None
        self.model_parameter_names = None
        self.response_names = None
        self._scr_sens = None
        self.scr_responses = None
        self._current_scr = None
        self._model_parameters_changed = None
        self._old_model_parameters = None
        self._pseudo_bayesian_type = None
        self.scr_fims = None
        self.scr_criterion_val = None
        self._current_scr_mp = None

        self._pseudo_bayesian = False
        self._simulate_sig_id = 0
        self.pvars = None
        self._trim_fim = True
        self._efforts_transformed = False
        self._unconstrained_form = False
        self._fd_jac = True
        self._fim_eval_time = None
        # memory threshold in which large requirement is defined in bytes, default: 1 GB
        self._memory_threshold = None
        # whether or not problem will require large memory
        self._large_memory_requirement = False
        # having large memory requirement
        self._current_criterion = None
        self.estimability = None
        self.normalized_sensitivity = None
        self._dynamic_controls = False
        self._dynamic_system = False
        self.optimal_candidates = None
        self.estimable_model_parameters = []
        self._save_sensitivities = False

        # core user-defined variables
        self.sampling_times_candidates = None  # sampling times of experiment. 2D
        # numpy array of floats. Rows are the number of candidates, columns are the
        # sampling times for given candidate. None means non-dynamic experiment.
        self.ti_controls_candidates = None  # time-invariant controls, a 2D numpy
        # array of floats. Rows are the number of candidates, columns are the
        # different controls.
        self.tv_controls_candidates = None  # time-varying controls, a 2D numpy array
        # of dictionaries. Rows are the number of candidates, columns are the
        # different controls.
        self.model_parameters = None  # nominal model parameters, a 1D numpy array of
        # floats.
        self.atomic_fims = None

        # optional user-defined variables
        self.candidate_names = None  # plotting names
        self.measurable_responses_names = None
        self.ti_controls_names = None
        self.tv_controls_names = None
        self.model_parameters_names = None
        self.measurable_responses = None  # subset of measurable states

        # core designer outputs
        self.response = None  # the predicted response profiles, a 3D numpy array. 1st
        # dim are the candidates, 2nd dim are sampling times, and 3rd dim are the
        # different responses.
        self.sensitivities = None  # a 4D numpy array. First dim is the number of
        # candidates, second dim are the different sampling times, third dim are the
        # are the different responses, and last dim different model parameters.

        # pyomo-specific
        self.simulator = None  # object needed when using pyomo models
        self.model = None  # object needed when using pyomo models

        """ problem dimension sizes """
        self.n_c = None
        self.n_c_tic = None
        self.n_c_tvc = None
        self.n_c_spt = None
        self.n_tic = None
        self.n_spt = None
        self.n_r = None
        self.n_mp = None
        self.n_e = None
        self.n_m_r = None
        self.n_scr = None
        self.n_spt_comb = None

        """ parameter estimation """
        self.data = None  # stored data, a 3D numpy array, same shape as response.
        # Whenever data is missing, use np.nan to fill the array.
        self.residuals = None  # stored residuals, 3D numpy array with the same shape
        # as data and response. Will skip entries whenever data is empty.

        """ performance attributes """
        self.feval_simulation = None
        self.feval_sensitivity = None

        """ parameter estimability """
        self.estimable_columns = None
        self.responses_scales = None

        """ continuous oed-related quantities """
        # sensitivities
        self.efforts = None
        self.F = None  # overall regressor matrix
        self.fim = None  # the information matrix for current experimental design
        self.p_var = None  # the prediction covariance matrix

        """ saving, loading attributes """
        # current oed result
        self.run_no = 1
        self.oed_result = None
        self.result_dir = None

        """ plotting attributes """
        self.grid = None  # storing grid when create_grid method is used to help
        # generate candidates

        """ private attributes """
        # current candidate controls, and sampling times: required for sensitivity
        # evaluations
        self._ti_controls = None
        self._tv_controls = None
        self._sampling_times = None
        self._current_response = None  # a 2D numpy array. 1st dim are sampling times,
        # 2nd dim are different responses
        self._last_scr_mp = None

        # store user-selected problem types
        self._sensitivity_is_normalized = None
        self._opt_sampling_times = False
        self._var_n_sampling_time = None

        # store chosen package to interface with the optimizer, and the chosen optimizer
        self._model_package = None
        self._optimization_package = None
        self._optimizer = None

        # storing states that helps minimize evaluations
        self._scr_changed = True

        # temporary performance results for current design
        self._sensitivity_analysis_time = 0
        self._optimization_time = 0

        # store current criterion value
        self._criterion_value = None

        # store designer status and its verbal level after initialization
        self._status = 'empty'
        self._verbose = 0
        self._sensitivity_analysis_done = False

    """ user-defined methods: must be overwritten by user to work """
    def simulate(self, unspecified):
        raise SyntaxError("Don't forget to specify the simulate function.")

    """ core activity interfaces """
    def initialize(self, verbose=0, memory_threshold=int(1e9)):
        """ check for syntax errors, runs one simulation to determine n_r """

        """ check if simulate function has been specified """
        self._handle_simulate_sig()
        self._check_missing_components()
        self._data_type_check()

        if self._dynamic_system:
            self._check_var_spt()

        self._check_stats_framework()
        self._get_component_sizes()
        self._check_candidate_lengths()

        self._initialize_names()

        self._check_memory_req(memory_threshold)

        self._status = 'ready'
        self._verbose = verbose
        if self._verbose >= 2:
            print("".center(100, "="))
        if self._verbose >= 1:
            print('Initialization complete: designer ready.')
        if self._verbose >= 2:
            print("".center(100, "-"))
            print(f"{'Number of model parameters':<40}: {self.n_mp}")
            print(f"{'Number of candidates':<40}: {self.n_c}")
            print(f"{'Number of responses':<40}: {self.n_r}")
            print(f"{'Number of measured responses':<40}: {self.n_m_r}")
            if self._invariant_controls:
                print(f"{'Number of time-invariant controls':<40}: {self.n_tic}")
            if self._dynamic_system:
                print(f"{'Number of sampling time choices':<40}: {self.n_spt}")
            if self._dynamic_controls:
                print(f"{'Number of time-varying controls':<40}: {self.n_tvc}")
            print("".center(100, "="))

        return self._status

    def simulate_candidates(self, store_predictions=True,
                            plot_simulation_times=False):
        self.response = None  # resets response every time simulation is invoked
        self.feval_simulation = 0
        time_list = []
        start = time()
        for i, exp in enumerate(
                zip(self.ti_controls_candidates, self.tv_controls_candidates,
                    self.sampling_times_candidates)):
            self._ti_controls = exp[0]
            self._tv_controls = exp[1]
            self._sampling_times = exp[2][~np.isnan(exp[2])]
            if not self._sampling_times.size > 0:
                raise SyntaxError(
                    'One candidate has an empty list of sampling times, please check '
                    'the specified experimental candidates.'
                )

            """ determine if simulation needs to be re-run: if data on time-invariant 
            control variables is missing, will not run """
            cond_1 = np.any(np.isnan(exp[0]))
            if np.any([cond_1]):
                self._current_response = np.nan
            else:
                start = time()
                response = self._simulate_internal(self._ti_controls, self._tv_controls,
                                                   self.model_parameters,
                                                   self._sampling_times)
                finish = time()
                self.feval_simulation += 1
                self._current_response = response
                time_list.append(finish - start)

            if store_predictions:
                self._store_current_response()
        if plot_simulation_times:
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.plot(time_list)
        if self._verbose >= 3:
            print(f"Completed simulation of all candidates in {time() - start} CPU seconds.")
        return self.response

    def simulate_optimal_candidates(self):
        if self.response is not None:
            overwrite = input("Previously stored responses data detected. "
                              "Running this will overwrite stored responses for the "
                              "optimal candidates. "
                              "Proceed? y: yes, n: no ")
            if not any(entry is overwrite for entry in ['y', 'yes']):
                return
        time_list = []
        for i, exp in enumerate(self.optimal_candidates):
            self._ti_controls = exp[1]
            self._tv_controls = exp[2]
            self._sampling_times = exp[3][~np.isnan(exp[3])]
            if self._sampling_times.size <= 0:
                msg = 'One candidate has an empty list of sampling times, please check ' \
                      '' \
                      '' \
                      '' \
                      '' \
                      'the ' \
                      'specified experimental candidates.'
                raise SyntaxError(msg)

            """ 
            determine if simulation needs to be re-run: 
            if data on time-invariant control variables is missing, will not run 
            """
            cond_1 = np.any(np.isnan(exp[0]))
            if np.any([cond_1]):
                self._current_response = np.nan
            else:
                start = time()
                response = self._simulate_internal(self._ti_controls, self._tv_controls,
                                                   self.model_parameters,
                                                   self._sampling_times)
                finish = time()
                self.feval_simulation += 1
                self._current_response = response
                time_list.append(finish - start)

    def estimate_parameters(self, bounds, init_guess=None, method='trf',
                            update_parameters=False, write=True, options=None,
                            max_nfev=None, variance=1, estimate_covar=True, **kwargs):
        if init_guess is None:
            init_guess = self.model_parameters

        if self.data is None:
            raise SyntaxError("No data is put in, do not forget to add it in.")

        if options is None:
            if self._verbose >= 2:
                options = {'disp': True}
            else:
                options = {'disp': False}

        if self._verbose >= 1:
            print("Solving parameter estimation...")
        start = time()

        bounds = np.asarray(bounds)
        bounds = bounds.T
        pe_result = least_squares(
            self._residuals_wrapper_f,
            init_guess,
            bounds=bounds,
            method=method,
            verbose=self._verbose,
            max_nfev=max_nfev,
            **kwargs,
        )

        finish = time()
        if not pe_result.success:
            print('Warning: estimation did not terminate as optimal.')
            stillsave = input(f"Still want to save results? Default is to save results "
                              f"regardless. To skip save, type \"skip\" to terminate: ")
            if stillsave == "skip":
                print("Exiting.")
                return None
            else:
                pass
        print(f"Estimated parameter values:")
        print(np.array2string(
            self.model_parameters,
            separator=","
        ))

        if self._verbose >= 1:
            print(
                "Complete: OLS estimation using %s took %.2f CPU seconds to complete."
                % (
                    method, finish - start))
        if self._verbose >= 2:
            print(
                f"The estimation took a total of {pe_result.nfev} function evaluations"
                f", and {pe_result.njev} number of Jacobian evaluations were done."
            )

        if update_parameters:
            self.model_parameters = pe_result.x
            if self._verbose >= 2:
                print('Nominal parameter value in model updated.')

        if write:
            case_path = getcwd()
            today = datetime.now()
            result_dir = case_path + "/" + str(today.date()) + "_at_" + str(
                today.hour) + "-" + str(
                today.minute) + "-" + str(today.second)
            makedirs(result_dir)
            with open(result_dir + "/result_file.pkl", "wb") as file:
                dump(pe_result, file)
            if self._verbose >= 2:
                print('Parameter estimation result saved to: %s.' % result_dir)

        if estimate_covar:
            try:
                self.eval_fim(
                    efforts=np.ones((self.n_c, self.n_spt)),
                    mp=pe_result.x,
                )
            except RuntimeWarning:
                print(
                    f"Sensitivity analysis for computing the information matrix failed, "
                    f"leading to a runtime error. Skipping the estimation of the covariance "
                    f"matrix."
                )
                return pe_result
            try:
                self.mp_covar = variance * np.linalg.inv(self.fim)
            except np.linalg.LinAlgError:
                try:
                    self.mp_covar = variance * np.linalg.pinv(self.fim)
                except np.linalg.LinAlgError:
                    return pe_result

            if self.mp_covar is not None and self._verbose >= 1:
                print(f"Standard absolute error of estimates (noise variance = {variance}):")
                for p, mp in enumerate(self.model_parameters):
                    print(fr"{mp:>40} +- {np.sqrt(np.diag(self.mp_covar)[p])}")
                print(f"Standard relative error of estimates (noise variance = {variance}):")
                for p, mp in enumerate(self.model_parameters):
                    print(fr"{mp:>40} +- {np.sqrt(np.diag(self.mp_covar)[p]) / mp * 100} %")

        return pe_result

    def estimate_parameters_alt(self, init_guess, bounds, method='l-bfgs-b',
                                update_parameters=False, write=True, options=None,
                                variance=1, **kwargs):
        if self.data is None:
            raise SyntaxError("No data is put in, do not forget to add it in.")

        if options is None:
            if self._verbose >= 2:
                options = {'disp': True}
            else:
                options = {'disp': False}

        if self._verbose >= 1:
            print("Solving parameter estimation...")
        start = time()

        pe_result = minimize(
            self._residuals_wrapper_f_old,
            init_guess,
            bounds=bounds,
            method=method,
            options=options,
            **kwargs,
        )
        finish = time()
        if not pe_result.success:
            print('Fail: estimation did not converge; exiting.')
            return None
        print(f"Estimated parameter values:")
        print(np.array2string(
            self.model_parameters,
            separator=","
        ))

        if self._verbose >= 1:
            print(
                "Complete: OLS estimation using %s took %.2f CPU seconds to complete."
                % (
                    method, finish - start))
        if self._verbose >= 2:
            print(
                "The estimation took a total of %d function evaluations, %d used for "
                "numerical estimation of the "
                "Jacobian using forward finite differences." % (
                    pe_result.nfev, pe_result.nfev - pe_result.nit - 1))

        if update_parameters:
            self.model_parameters = pe_result.x
            if self._verbose >= 2:
                print('Nominal parameter value in model updated.')

        if write:
            case_path = getcwd()
            today = datetime.now()
            result_dir = case_path + "/" + str(today.date()) + "_at_" + str(
                today.hour) + "-" + str(
                today.minute) + "-" + str(today.second)
            makedirs(result_dir)
            with open(result_dir + "/result_file.pkl", "wb") as file:
                dump(pe_result, file)
            if self._verbose >= 2:
                print('Parameter estimation result saved to: %s.' % result_dir)

        try:
            self.eval_fim(
                efforts=np.ones((self.n_c, self.n_spt)),
                mp=pe_result.x,
            )
        except RuntimeWarning:
            print(
                f"Sensitivity analysis for computing the information matrix failed, "
                f"leading to a runtime error. Skipping the estimation of the covariance "
                f"matrix."
            )
            return pe_result
        try:
            self.mp_covar = np.linalg.inv(self.fim)
        except np.linalg.LinAlgError:
            try:
                self.mp_covar = np.linalg.pinv(self.fim)
            except np.linalg.LinAlgError:
                return pe_result

        if self.mp_covar is not None:
            print(f"Standard absolute error of estimates (noise variance = {variance}):")
            for p, mp in enumerate(self.model_parameters):
                print(fr"{mp:>40} +- {np.sqrt(np.diag(self.mp_covar)[p])}")
            print(f"Standard relative error of estimates (noise variance = {variance}):")
            for p, mp in enumerate(self.model_parameters):
                print(fr"{mp:>40} +- {np.sqrt(np.diag(self.mp_covar)[p]) / mp * 100} %")

        return pe_result

    def design_experiment(self, criterion, n_spt=None, n_exp=None, optimize_sampling_times=False,
                          package="cvxpy", optimizer=None, opt_options=None, e0=None,
                          write=True, save_sensitivities=False, fd_jac=True,
                          unconstrained_form=False, trim_fim=True, pseudo_bayesian_type=None,
                          regularize_fim=False, **kwargs):
        # storing user choices
        self._regularize_fim = regularize_fim
        self._optimization_package = package
        self._optimizer = optimizer
        self._opt_sampling_times = optimize_sampling_times
        self._save_sensitivities = save_sensitivities
        self._current_criterion = criterion.__name__
        self._fd_jac = fd_jac
        self._unconstrained_form = unconstrained_form
        self._trim_fim = trim_fim

        """ resetting optimal candidates """
        self.optimal_candidates = None

        """ setting verbal behaviour """
        if self._verbose >= 2:
            opt_verbose = True
        else:
            opt_verbose = False

        """ handling problems with defined n_spt """
        if n_spt is not None:
            if not self._dynamic_system:
                raise SyntaxError(
                    f"n_spt specified for a non-dynamic system."
                )
            if not self._opt_sampling_times:
                print(
                    f"[Warning]: n_spt specified, but "
                    f"optimize_sampling_times = False. "
                    f"Overriding, and setting optimize_sampling_times = True."
                )
            self._opt_sampling_times = True
            self._n_spt_spec = n_spt
            if not isinstance(n_spt, int):
                raise SyntaxError(
                    f"Supplied n_spt is a {type(n_exp)}, "
                    f"but \"n_spt\" must be an integer."
                )
            self._specified_n_spt = True
            self.spt_candidates_combs = []
            for spt in self.sampling_times_candidates:
                spt_idx = np.arange(0, len(spt))
                self.spt_candidates_combs.append(
                    list(itertools.combinations(spt_idx, n_spt))
                )
            self.spt_candidates_combs = np.asarray(
                self.spt_candidates_combs
            )
            _, self.n_spt_comb, _ = self.spt_candidates_combs.shape
        else:
            self._specified_n_spt = False

        """ determining if discrete design problem """
        if n_exp is not None:
            self._discrete_design = True
            if not isinstance(n_exp, int):
                raise SyntaxError(
                    f"Supplied n_exp is a {type(n_exp)}, "
                    f"but \"n_exp\" must be an integer."
                )
        else:
            self._discrete_design = False

        """ setting default semi-bayes behaviour """
        if self._pseudo_bayesian:
            if pseudo_bayesian_type is None:
                self._pseudo_bayesian_type = 0
            else:
                valid_types = [
                    0, 1,
                    "avg_inf", "avg_crit",
                    "average_information", "average_criterion"
                ]
                if pseudo_bayesian_type in valid_types:
                    self._pseudo_bayesian_type = pseudo_bayesian_type
                else:
                    raise SyntaxError(
                        "Unrecognized pseudo_bayesian criterion type. Valid types: '0' "
                        "for average information, '1' for average criterion."
                    )

        """ force fd_jac for large problems """
        if self._large_memory_requirement and not self._fd_jac:
            print("Warning: analytic Jacobian is specified on a large problem."
                  "Overwriting and continuing with finite differences.")
            self._fd_jac = True

        """ setting default optimizers, and its options """
        if self._optimization_package is "scipy":
            if optimizer is None:
                self._optimizer = "SLSQP"
            if opt_options is None:
                opt_options = {"disp": opt_verbose}
        if self._optimization_package is "cvxpy":
            if optimizer is None:
                self._optimizer = "MOSEK"

        """ deal with unconstrained form """
        if self._optimization_package is "scipy":
            if self._optimizer not in ["COBYLA", "SLSQP", "trust-constr"]:
                if self._verbose >= 2:
                    print(f"Note: {self._optimization_package}'s optimizer "
                          f"{self._optimizer} requires unconstrained form.")
                self._unconstrained_form = True
        if self._optimization_package is "cvxpy":
            if self._unconstrained_form:
                self._unconstrained_form = False
                print("Warning: unconstrained form is not supported by cvxpy; "
                      "continuing normally with constrained form.")

        """ main codes """
        if self._verbose >= 1:
            print(" Computing Optimal Experiment Design ".center(100, "#"))
        if self._verbose >= 2:
            print(f"{'Started on':<40}: {datetime.now()}")
            print(f"{'Criterion':<40}: {self._current_criterion}")
            print(f"{'Pseudo-bayesian':<40}: {self._pseudo_bayesian}")
            if self._pseudo_bayesian:
                print(f"{'Pseudo-bayesian Criterion Type':<40}: {self._pseudo_bayesian_type}")
            print(f"{'Dynamic':<40}: {self._dynamic_system}")
            print(f"{'Time-invariant Controls':<40}: {self._invariant_controls}")
            print(f"{'Time-varying Controls':<40}: {self._dynamic_controls}")
            print(f"{'Number of Candidates':<40}: {self.n_c}")
            if self._dynamic_system:
                print(f"{'Number of Sampling Time Choices':<40}: {self.n_spt}")
                print(f"{'Sampling Times Optimized':<40}: {self._opt_sampling_times}")
            if self._pseudo_bayesian:
                print(f"{'Number of Scenarios':<40}: {self.n_scr}")
        """ 
        set initial guess for optimal experimental efforts, if none given, equal 
        efforts for all candidates 
        """
        if e0 is None:
            if self._specified_n_spt:
                e0 = np.ones((self.n_c, self.n_spt_comb)) / (self.n_c * self.n_spt_comb)
            else:
                e0 = np.ones((self.n_c, self.n_spt)) / (self.n_c * self.n_spt)
        else:
            msg = 'Initial guess for effort must be a 2D numpy array.'
            if not isinstance(e0, np.ndarray):
                raise SyntaxError(msg)
            elif e0.ndim != 2:
                raise SyntaxError(msg)
            elif e0.shape[0] != self.n_c:
                raise SyntaxError(
                    f"Error: inconsistent number of candidates provided;"
                    f"number of candidates in e0: {e0.shape[0]},"
                    f"number of candidates from initialization: {self.n_c}."
                )
            if self._specified_n_spt:
                if e0.shape[1] != self.n_spt_comb:
                    raise SyntaxError(
                        f"Error: second dimension of e0 must be {self.n_spt_comb} "
                        f"long, corresponding to n_spt_combs; given is {e0.shape[1]}."
                    )
            else:
                if e0.shape[1] != self.n_spt:
                    raise SyntaxError(
                        f"Error: inconsistent number of sampling times provided;"
                        f"number of sampling times in e0: {e0.shape[1]},"
                        f"number of candidates from initialization: {self.n_spt}."
                    )
        self.efforts = e0

        # declare and solve optimization problem
        self._sensitivity_analysis_time = 0
        start = time()
        # solvers
        if self._optimization_package == "scipy":
            if self._discrete_design:
                raise NotImplementedError(
                    "Scipy cannot be used to compute discrete designs."
                )
            if self._unconstrained_form:
                opt_result = minimize(
                    fun=criterion,
                    x0=e0,
                    method=optimizer,
                    options=opt_options,
                    jac=not self._fd_jac,
                )
            else:
                e_bound = [[(0, 1) for _ in eff0] for eff0 in e0]
                if self._specified_n_spt:
                    e_bound = np.asarray(e_bound).reshape((self.n_c * self.n_spt_comb, 2))
                else:
                    e_bound = np.asarray(e_bound).reshape((self.n_c * self.n_spt, 2))
                constraint = {"type": "eq", "fun": lambda e: sum(e) - 1.0}
                opt_result = minimize(
                    fun=criterion,
                    x0=e0,
                    method=optimizer,
                    options=opt_options,
                    constraints=constraint,
                    bounds=e_bound,
                    jac=not self._fd_jac,
                    **kwargs,
                )
            self.efforts = opt_result.x.reshape((self.n_c, self.n_spt))
            self._efforts_transformed = False
            self._transform_efforts()
            opt_fun = opt_result.fun
        elif self._optimization_package == "cvxpy":
            # optimization variable and initial value
            if self._specified_n_spt:
                efforts = cp.Variable((self.n_c, self.n_spt_comb), nonneg=True)
            else:
                efforts = cp.Variable((self.n_c, self.n_spt), nonneg=True)
            efforts.value = self.efforts
            # constraints and objective
            if self._discrete_design:
                p_cons = [cp.sum(efforts) == n_exp]
            else:
                p_cons = [cp.sum(efforts) <= 1]
                if not self._opt_sampling_times:
                    p_cons += [eff == eff[0] for c, eff in enumerate(efforts)]
            # cvxpy problem
            obj = cp.Maximize(-criterion(efforts))
            problem = cp.Problem(obj, p_cons)
            # solution
            if self._discrete_design:
                root = Node(efforts, problem)
                tree = Tree(root)
                tree._verbose = self._verbose
                opt_node = tree.solve()
                self.efforts = opt_node.int_var_val
                opt_fun = opt_node.ub
            else:
                opt_fun = problem.solve(
                    verbose=opt_verbose,
                    solver=self._optimizer,
                    **kwargs
                )
                self.efforts = efforts.value
        else:
            raise SyntaxError("Unrecognized package; try \"scipy\" or \"cvxpy\".")

        finish = time()

        """ report status and performance """
        self._optimization_time = finish - start - self._sensitivity_analysis_time
        if self._verbose >= 2:
            print(
                f"[Optimization Complete in {self._optimization_time:.2f} s]".center(100, "-")
            )
        if self._verbose >= 1:
            print(
                f"Complete: \n"
                f" ~ sensitivity analysis took {self._sensitivity_analysis_time:.2f} "
                f"CPU seconds.\n"
                f" ~ optimization with {self._optimizer:s} via "
                f"{self._optimization_package} took "
                f"{self._optimization_time:.2f} CPU seconds."
            )
            print("".center(100, "#"))

        """ storing and writing result """
        self._criterion_value = opt_fun
        oed_result = {
            "solution_time": finish - start,
            "optimization_time": self._optimization_time,
            "sensitivity_analysis_time": self._sensitivity_analysis_time,
            "optimality_criterion": criterion.__name__[:-10],
            "ti_controls_candidates": self.ti_controls_candidates,
            "tv_controls_candidates": self.tv_controls_candidates,
            "model_parameters": self.model_parameters,
            "sampling_times_candidates": self.sampling_times_candidates,
            "optimal_efforts": self.efforts,
            "criterion_value": self._criterion_value,
            "optimizer": self._optimizer
        }
        self.oed_result = oed_result
        if write:
            self.write_oed_result()

        return oed_result

    def estimability_study(self, base_step=None, step_ratio=None, num_steps=None,
                           estimable_tolerance=0.04, write=False,
                           save_sensitivities=False):
        self._save_sensitivities = save_sensitivities
        self.eval_sensitivities(base_step=base_step, step_ratio=step_ratio,
                                num_steps=num_steps)
        self.normalize_sensitivities()

        z = self.normalized_sensitivity[:, :, self.measurable_responses, :].reshape(
            self.n_spt * self.n_m_r * self.n_c, self.n_mp)

        z_col_mag = np.nansum(np.power(z, 2), axis=0)
        next_estim_param = np.argmax(z_col_mag)
        self.estimable_columns = np.array([next_estim_param])
        finished = False
        while not finished:
            x_l = z[:, self.estimable_columns]
            z_theta = np.linalg.inv(x_l.T.dot(x_l)).dot(x_l.T).dot(z)
            z_hat = x_l.dot(z_theta)
            r = z - z_hat
            r_col_mag = np.nansum(np.power(r, 2), axis=0)
            next_estim_param = np.argmax(r_col_mag)
            if r_col_mag[next_estim_param] <= estimable_tolerance:
                if write:
                    self.create_result_dir()
                    self.run_no = 1
                    result_file_template = Template(
                        "${result_dir}/estimability_study_${run_no}.pkl")
                    result_file = result_file_template.substitute(
                        result_dir=self.result_dir, run_no=self.run_no)
                    while path.isfile(result_file):
                        self.run_no += 1
                        result_file = result_file_template.substitute(
                            result_dir=self.result_dir, run_no=self.run_no)
                    dump(self.estimable_columns, open(result_file, 'wb'))
                print(f'Identified estimable parameters are: {self.estimable_columns}')
                return self.estimable_columns
            self.estimable_columns = np.append(self.estimable_columns, next_estim_param)

    def estimability_study_fim(self, save_sensitivities=False):
        self._save_sensitivities = save_sensitivities
        self.efforts = np.ones((self.n_c, self.n_spt))
        self.eval_fim(self.efforts, self.model_parameters)
        print(f"Estimable parameters: {self.estimable_model_parameters}")
        print(f"Degree of Estimability: {self.estimability}")
        return self.estimable_model_parameters, self.estimability

    """ core utilities """

    # create grid
    def create_grid(self, bounds, levels):
        """ returns points from a mesh-centered grid """
        bounds = np.asarray(bounds)
        levels = np.asarray(levels)
        grid_args = ''
        for bound, level in zip(bounds, levels):
            grid_args += '%f:%f:%dj,' % (bound[0], bound[1], level)
        make_grid = 'self.grid = np.mgrid[%s]' % grid_args
        exec(make_grid)
        self.grid = self.grid.reshape(np.array(levels).size, np.prod(levels)).T
        return self.grid

    def enumerate_candidates(self, bounds, levels, switching_times=None):
        # use create_grid if only time-invariant controls
        if switching_times is None:
            return self.create_grid(bounds, levels)

        """ check syntax of given bounds, levels, switching times """
        bounds = np.asarray(bounds)
        levels = np.asarray(levels)
        switching_times = np.asarray(switching_times)
        # make sure bounds, levels, switching times are numpy arrays
        if not all(isinstance(arg, np.ndarray) for arg in [bounds, levels, switching_times]):
            raise SyntaxError(
                f"Supplied bounds, levels, and switching times must be numpy arrays."
            )
        # make sure length of experimental variables are the same
        bound_len, bound_dim = bounds.shape
        if bound_dim != 2:
            raise SyntaxError(
                f"Supplied bounds must be a 2D array with shape (:, 2)."
            )
        if levels.ndim != 1:
            raise SyntaxError(
                f"Supplied levels must be a 1D array."
            )
        levels_len = levels.size
        switch_len = len(switching_times)

        # count number of candidates from given information
        if not bound_len == levels_len == switch_len:
            raise SyntaxError(
                f"Supplied lengths are incompatible. Bound: {bound_len}, "
                f"levels: {levels_len}, switch_len: {switch_len}."
            )

        """ discretize tvc into piecewise constants and use create_grid to enumerate """
        tic_idx = []
        tvc_idx = []
        tic_bounds = []
        tic_levels = []
        tvc_bounds = []
        tvc_levels = []
        for i, swt_t in enumerate(switching_times):
            if swt_t is None:
                tic_idx.append(i)
                tic_bounds.append(bounds[i])
                tic_levels.append(levels[i])
            else:
                tvc_idx.append(i)
                for t in swt_t:
                    tvc_bounds.append(bounds[i])
                    tvc_levels.append(levels[i])
        n_tic = len(tic_idx)
        n_tvc = len(tvc_idx)
        if n_tic == 0:
            total_bounds = tvc_bounds
            total_levels = tvc_levels
        elif n_tvc == 0:
            total_bounds = tic_bounds
            total_levels = tic_levels
        else:
            total_bounds = np.vstack((tic_bounds, tvc_bounds))
            total_levels = np.append(tic_levels, tvc_levels)
        candidates = self.create_grid(total_bounds, total_levels)
        tic = candidates[:, :n_tic]
        tvc_array = candidates[:, n_tic:]

        """ converting 2D tvc_array of floats into a 2D numpy array of dictionaries """
        tvc = []
        for candidate, values in enumerate(tvc_array):
            col_counter = 0
            temp_tvc_dict_list = []
            for idx in tvc_idx:
                temp_tvc_dict = {}
                for t in switching_times[idx]:
                    temp_tvc_dict[t] = values[col_counter]
                    col_counter += 1
                temp_tvc_dict_list.append(temp_tvc_dict)
            tvc.append(temp_tvc_dict_list)
        tvc = np.asarray(tvc)

        return tic, tvc

    # visualization and result retrieval
    def plot_optimal_efforts(self, width=None, write=False, dpi=720, quality=95,
                             force_3d=False):
        if self.optimal_candidates is None:
            self.get_optimal_candidates()
        if self.n_opt_c is 0:
            print("Empty candidates, skipping plotting of optimal efforts.")
            return
        if (self._opt_sampling_times or force_3d) and self._dynamic_system:
            fig = self._plot_current_efforts_3d(width=width, write=write, dpi=dpi,
                                          quality=quality)
        else:
            if force_3d:
                print(
                    "Warning: force 3d only works for dynamic systems, plotting "
                    "current design in 2D."
                )
            fig = self._plot_current_efforts_2d(width=width, write=write, dpi=dpi,
                                          quality=quality)
        return fig

    def plot_optimal_controls(self, alpha=0.3, markersize=3, non_opt_candidates=False,
                              n_ticks=3, visualize_efforts=True, tol=1e-2,
                              intervals=None, title=False, write=False, dpi=720,
                              quality=95):
        if self._dynamic_system:
            print(
                "[Warning]: Plot optimal controls is not implemented for dynamic "
                "system, use print_optimal_candidates, or plot_optimal_sensitivities "
                "for visualization."
            )
            return
        if self.optimal_candidates is None:
            self.get_optimal_candidates()
        if self.n_opt_c is 0:
            print(
                f"[Warning]: empty optimal candidates, skipping plotting of optimal "
                f"controls."
            )
            return
        if self._dynamic_controls:
            raise NotImplementedError(
                "Plot controls not implemented for dynamic controls"
            )
        if self.n_tic > 4:
            raise NotImplementedError(
                "Plot controls not implemented for systems with more than 4 ti_controls"
            )
        if self.n_tic == 1:
            fig, axes = plt.subplots(1, 1)
            if title:
                axes.set_title(self._current_criterion)
            if visualize_efforts:
                opt_idx = np.where(self.efforts >= tol)
                delta = self.ti_controls_candidates[:, 0].max() - self.ti_controls_candidates[:, 0].min()
                axes.bar(
                    self.ti_controls_candidates[:, 0],
                    self.efforts[:, 0],
                    width=0.01 * delta,
                )
                axes.set_ylim([0, 1])
                axes.set_xlabel("Control 1")
                axes.set_ylabel("Efforts")
            # fig = self.plot_optimal_efforts(write=write)
        elif self.n_tic == 2:
            fig, axes = plt.subplots(1, 1)
            if title:
                axes.set_title(self._current_criterion)
            if non_opt_candidates:
                axes.scatter(
                    self.ti_controls_candidates[:, 0],
                    self.ti_controls_candidates[:, 1],
                    alpha=alpha,
                    marker="o",
                    s=18*markersize,
                )
            if visualize_efforts:
                opt_idx = np.where(self.efforts >= tol)
                axes.scatter(
                    self.ti_controls_candidates[opt_idx[0], 0].T,
                    self.ti_controls_candidates[opt_idx[0], 1].T,
                    facecolor="none",
                    edgecolor="red",
                    marker="o",
                    s=self.efforts[opt_idx]*500*markersize,
                )
            if self.ti_controls_names is None:
                axes.set_xlabel("Time-invariant Control 1")
                axes.set_ylabel("Time-invariant Control 2")
            else:
                axes.set_xlabel(self.ti_controls_names[0])
                axes.set_ylabel(self.ti_controls_names[1])
            axes.set_xticks(
                np.linspace(
                    self.ti_controls_candidates[:, 0].min(),
                    self.ti_controls_candidates[:, 0].max(),
                    n_ticks,
                )
            )
            axes.set_yticks(
                np.linspace(
                    self.ti_controls_candidates[:, 1].min(),
                    self.ti_controls_candidates[:, 1].max(),
                    n_ticks,
                )
            )
            fig.tight_layout()
        elif self.n_tic == 3:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection="3d")
            if non_opt_candidates:
                axes.scatter(
                    self.ti_controls_candidates[:, 0],
                    self.ti_controls_candidates[:, 1],
                    self.ti_controls_candidates[:, 2],
                    alpha=alpha,
                    marker="o",
                    s=18*markersize,
                )
            opt_idx = np.where(self.efforts >= tol)
            axes.scatter(
                self.ti_controls_candidates[opt_idx[0], 0],
                self.ti_controls_candidates[opt_idx[0], 1],
                self.ti_controls_candidates[opt_idx[0], 2],
                facecolor="r",
                edgecolor="r",
                s=self.efforts[opt_idx]*500*markersize,
            )
            if self.ti_controls_names is not None:
                axes.set_xlabel(f"{self.ti_controls_names[0]}")
                axes.set_ylabel(f"{self.ti_controls_names[1]}")
                axes.set_zlabel(f"{self.ti_controls_names[2]}")
            axes.grid(False)
            fig.tight_layout()
        elif self.n_tic == 4:
            trellis_plotter = TrellisPlotter()
            trellis_plotter.data = self.ti_controls_candidates
            trellis_plotter.markersize = self.efforts * 500
            if intervals is None:
                intervals = np.array([5, 5])
            trellis_plotter.intervals = intervals
            fig = trellis_plotter.scatter()

        if write:
            self.create_result_dir()
            fn = f"optimal_controls" \
                 f"_{self.oed_result['optimality_criterion']}" \
                 f"_{self.run_no}.png"
            fp = self.result_dir + fn
            while path.isfile(fp):
                self.run_no += 1
                fn = f"optimal_controls" \
                     f"_{self.oed_result['optimality_criterion']}" \
                     f"_{self.run_no}.png"
                fp = self.result_dir + fn
            fig.savefig(fname=fp, dpi=dpi, quality=quality)
            self.run_no = 1

        return fig

    def plot_parity(self):
        if self.response is None:
            raise RuntimeError(
                "Cannot generate parity plot when response is empty. "
                "Run simulate_all_candidates and store responses."
            )
        if self.data is None:
            raise RuntimeError(
                "Cannot generate parity plot when data is empty. "
                "Please specify data to the designer."
            )
        fig = plt.figure()
        n_rows = np.ceil(np.sqrt(self.n_m_r)).astype(int)
        gridspec = plt.GridSpec(nrows=n_rows, ncols=n_rows)
        for r in range(self.n_m_r):
            row = r % n_rows
            col = np.floor_divide(r, n_rows)
            axes = fig.add_subplot(gridspec[row, col])
            axes.scatter(
                [dat for dat in self.data[:, :, r]],
                [res for res in self.response[:, :, self.measurable_responses[r]]],
                marker="1",
            )
            axes.plot(
                [-1e10, 1e10],
                [-1e10, 1e10],
                linestyle="-",
                marker="None",
                c="gray",
                alpha=0.3,
            )
            data_lim = [
                np.nanmin(self.data[:, :, r]),
                np.nanmax(self.data[:, :, r]),
            ]
            res_lim = [
                np.nanmin(self.response[:, :, self.measurable_responses[r]]),
                np.nanmax(self.response[:, :, self.measurable_responses[r]]),
            ]
            lim = [
                np.min([data_lim[0], res_lim[0]]),
                np.max([data_lim[1], res_lim[1]]),
            ]
            lim = lim + np.array([
                -0.1 * (lim[1] - lim[0]),
                 0.1 * (lim[1] - lim[0]),
            ])
            axes.set_xlim(lim)
            axes.set_ylim(lim)
            if self.response_names is not None:
                axes.set_title(f"{self.response_names[r]}")
            else:
                axes.set_title(f"Response {r}")
            axes.set_xlabel(f"Data")
            axes.set_ylabel(f"Prediction")
        plt.get_current_fig_manager().window.showMaximized()
        plt.gcf().tight_layout()
        return fig

    def plot_predictions(self, plot_data=False, figsize=None, label_candidates=True):
        if not self._dynamic_system:
            raise NotImplementedError(
                f"Plot predictions not supported for non-dynamic systems."
            )
        if figsize is None:
            figsize = (15, 8)
        if self.response is None:
            self.simulate_candidates()
        figs = []
        for res in range(self.n_m_r):
            fig = plt.figure(figsize=figsize)
            n_rows = np.ceil(np.sqrt(self.n_c)).astype(int)
            n_cols = n_rows
            gridspec = plt.GridSpec(
                nrows=n_rows,
                ncols=n_cols,
            )
            res_lim = [
                np.nanmin(self.response[:, :, self.measurable_responses[res]]),
                np.nanmax(self.response[:, :, self.measurable_responses[res]]),
            ]
            if plot_data:
                data_lim = [
                    np.nanmin(self.data[:, :, res]),
                    np.nanmax(self.data[:, :, res]),
                ]
            else:
                data_lim = res_lim
            lim = [
                np.min([data_lim[0], res_lim[0]]),
                np.max([data_lim[1], res_lim[1]])
            ]
            lim = lim + np.array([
                - 0.1 * (lim[1] - lim[0]),
                + 0.1 * (lim[1] - lim[0]),
            ])
            for row in range(n_rows):
                for col in range(n_cols):
                    cand = n_cols * row + col
                    if cand < self.n_c:
                        axes = fig.add_subplot(gridspec[row, col])
                        axes.plot(
                            self.sampling_times_candidates[cand, :],
                            self.response[n_cols*row + col, :, self.measurable_responses[res]],
                            linestyle="-",
                            marker="1",
                            label="Prediction"
                        )
                        if plot_data:
                            axes.plot(
                                self.sampling_times_candidates[cand, :],
                                self.data[n_cols * row + col, :, res],
                                linestyle="none",
                                marker="v",
                                fillstyle="none",
                                label="Data"
                            )
                        axes.set_ylim(lim)
                        if self.time_unit_name is not None:
                            axes.set_xlabel(f"Time ({self.time_unit_name})")
                        else:
                            axes.set_xlabel('Time')
                        ylabel = self.response_names[res]
                        if self.response_unit_names is not None:
                            ylabel += f" ({self.response_unit_names[res]})"
                        axes.set_ylabel(ylabel)
                        if cand + 1 == self.n_c:
                            axes.legend(prop={"size": 6})
                        if label_candidates:
                            axes.set_title(f"{self.candidate_names[cand]}")
            # if self.response_names is not None:
            #     fig.suptitle(f"Response: {self.response_names[res]}")
            fig.tight_layout()
            figs.append(fig)
        return figs

    def plot_sensitivities(self, absolute=False, legend=None, figsize=None):
        # n_c, n_s_times, n_res, n_theta = self.sensitivity.shape
        if self.sensitivities is None:
            self.eval_sensitivities()
        if figsize is None:
            figsize = (self.n_mp * 4.0, 1.0 + 2.5 * self.n_m_r)
        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=self.n_m_r,
            ncols=self.n_mp,
            sharex=True,
        )
        if legend is None:
            if self.n_c < 6:
                legend = True
        if self._sensitivity_is_normalized:
            norm_status = 'Normalized '
        else:
            norm_status = 'Unnormalized '
        if absolute:
            abs_status = 'Absolute '
        else:
            abs_status = 'Directional '

        fig.suptitle('%s%sSensitivity Plots' % (norm_status, abs_status))
        for row in range(self.n_m_r):
            for col in range(self.n_mp):
                for c, exp_candidate in enumerate(
                        zip(self.ti_controls_candidates, self.tv_controls_candidates,
                            self.sampling_times_candidates)):
                    sens = self.sensitivities[
                           c,
                           :,
                           self.measurable_responses[row],
                           col,
                           ]
                    axes[row, col].plot(
                        exp_candidate[2],
                        sens,
                        "-o",
                        label=f"Candidate {c + 1}"
                    )
                    axes[row, col].ticklabel_format(
                        axis="y",
                        style="sci",
                        scilimits=(0, 0),
                    )
                    if self.time_unit_name is not None:
                        axes.set_xlabel(f"Sampling Times ({self.time_unit_name})")
                    else:
                        axes.set_xlabel('Sampling Times')
                    ylabel = self.response_names[self.measurable_responses[row]]
                    ylabel += "/"
                    ylabel += self.model_parameter_names[col]
                    if self.response_unit_names is not None:
                        if self.model_parameter_unit_names is not None:
                            ylabel += f" ({self.response_unit_names[row]}/{self.model_parameter_unit_names[col]})"
                    axes.set_ylabel(ylabel)
                if legend and self.n_c <= 10:
                    axes[-1, -1].legend()
        fig.tight_layout()
        return fig

    def plot_optimal_predictions(self, legend=None, figsize=None, markersize=10,
                                 fontsize=10, legend_size=8, colour_map="jet",
                                 write=False, dpi=720, quality=95):
        if not self._dynamic_system:
            raise SyntaxError("Prediction plots are only for dynamic systems.")

        if self._status is not 'ready':
            raise SyntaxError(
                'Initialize the designer first.'
            )

        if self._pseudo_bayesian:
            if self.scr_responses is None:
                raise SyntaxError(
                    'Cannot plot prediction vs data when scr_response is empty, please '
                    'run a semi-bayes experimental design, and store predictions.'
                )
            mean_res = np.average(self.scr_responses, axis=0)
            std_res = np.std(self.scr_responses, axis=0)
        else:
            if self.response is None:
                self.simulate_candidates(store_predictions=True)

        if self.optimal_candidates is None:
            self.get_optimal_candidates()
        if self.n_opt_c is 0:
            print(
                f"[Warning]: empty optimal candidates, skipping plotting of optimal "
                f"predictions."
            )
            return
        if legend is None:
            if self.n_opt_c < 6:
                legend = True
        if figsize is None:
            figsize = (4.0, 1.0 + 2.5 * self.n_m_r)

        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=self.n_m_r,
            ncols=1,
            sharex=True,
        )
        if self.n_m_r == 1:
            axes = [axes]
        """ defining fig's subplot axes limits """
        x_axis_lim = [
            np.min(self.sampling_times_candidates[
                       ~np.isnan(self.sampling_times_candidates)]),
            np.max(self.sampling_times_candidates[
                       ~np.isnan(self.sampling_times_candidates)])
        ]
        for res in range(self.n_m_r):
            if self._pseudo_bayesian:
                res_max = np.nanmax(mean_res[:, :, res] + std_res[:, :, res])
                res_min = np.nanmin(mean_res[:, :, res] - std_res[:, :, res])
            else:
                res_max = np.nanmax(self.response[:, :, res])
                res_min = np.nanmin(self.response[:, :, res])
            y_axis_lim = [res_min, res_max]
            if self._pseudo_bayesian:
                plot_response = mean_res
            else:
                plot_response = self.response
            ax = axes[res]
            cmap = cm.get_cmap(colour_map, len(self.optimal_candidates))
            colors = itertools.cycle([
                cmap(_) for _ in np.linspace(0, 1, len(self.optimal_candidates))
            ])
            for c, cand in enumerate(self.optimal_candidates):
                color = next(colors)
                ax.plot(
                    self.sampling_times_candidates[cand[0]],
                    plot_response[
                        cand[0],
                        :,
                        self.measurable_responses[res]
                    ],
                    linestyle="--",
                    label=f"Candidate {cand[0] + 1:d}",
                    zorder=0,
                    c=color,
                )
                if self._pseudo_bayesian:
                    ax.fill_between(
                        self.sampling_times_candidates[cand[0]],
                        plot_response[
                            cand[0],
                            :,
                            self.measurable_responses[res]
                        ]
                        +
                        std_res[
                            cand[0],
                            :,
                            self.measurable_responses[res]
                        ],
                        mean_res[
                            cand[0],
                            :,
                            self.measurable_responses[res]
                        ]
                        -
                        std_res[
                            cand[0],
                            :,
                            self.measurable_responses[res]
                        ],
                        alpha=0.1,
                        facecolor=color,
                        zorder=1
                    )
                if not self._specified_n_spt:
                    ax.scatter(
                        cand[3],
                        plot_response[
                            cand[0],
                            cand[5],
                            self.measurable_responses[res]
                        ],
                        marker="o",
                        s=markersize * 50 * np.array(cand[4]),
                        zorder=2,
                        # c=np.array([color]),
                        color=color,
                        facecolors="none",
                    )
                else:
                    markers = itertools.cycle(["o", "s", "h", "P"])
                    for i, (eff, spt, spt_idx) in enumerate(zip(cand[4], cand[3], cand[5])):
                        marker = next(markers)
                        ax.scatter(
                            spt,
                            plot_response[
                                cand[0],
                                spt_idx,
                                self.measurable_responses[res]
                            ],
                            marker=marker,
                            s=markersize * 50 * np.array(eff),
                            color=color,
                            label=f"Variant {i + 1}",
                            facecolors="none",
                        )
                ax.set_xlim(
                    x_axis_lim[0] - 0.1 * (x_axis_lim[1] - x_axis_lim[0]),
                    x_axis_lim[1] + 0.1 * (x_axis_lim[1] - x_axis_lim[0])
                )
                ax.set_ylim(
                    y_axis_lim[0] - 0.1 * (y_axis_lim[1] - y_axis_lim[0]),
                    y_axis_lim[1] + 0.1 * (y_axis_lim[1] - y_axis_lim[0])
                )
                ax.tick_params(axis="both", which="major", labelsize=fontsize)
                ax.yaxis.get_offset_text().set_fontsize(fontsize)
                if self.response_names is None:
                    ylabel = f"Response {res+1}"
                else:
                    ylabel = f"{self.response_names[res]}"
                if self.response_unit_names is None:
                    pass
                else:
                    ylabel += f" ({self.response_unit_names[res]})"
                ax.set_ylabel(ylabel)
        if self.time_unit_name is not None:
            axes[-1].set_xlabel(f"Time ({self.time_unit_name})")
        else:
            axes[-1].set_xlabel('Time')
        if legend and len(self.optimal_candidates) > 1:
            axes[-1].legend(prop={"size": legend_size})

        fig.tight_layout()

        if write:
            self.create_result_dir()
            fn = f"response_plot" \
                 f"_{self.oed_result['optimality_criterion']}" \
                 f"_{self.run_no}.png"
            fp = self.result_dir + fn
            while path.isfile(fp):
                self.run_no += 1
                fn = f"response_plot" \
                     f"_{self.oed_result['optimality_criterion']}" \
                     f"_{self.run_no}.png"
                fp = self.result_dir + fn
            fig.savefig(fname=fp, dpi=dpi, quality=quality)
            self.run_no = 1

        return fig

    def plot_optimal_sensitivities(self, figsize=None, markersize=10, colour_map="jet",
                                   write=False, dpi=720, quality=95, interactive=True):
        if interactive:
            self._plot_optimal_sensitivities_interactive(
                figsize=figsize,
                markersize=markersize,
                colour_map=colour_map,
            )
        else:
            self._plot_optimal_sensitivities(
                figsize=figsize,
                markersize=markersize,
                colour_map=colour_map,
                write=write,
                dpi=dpi,
                quality=quality,
            )

    def print_optimal_candidates(self, tol=1e-2, write=True):
        if self.optimal_candidates is None:
            self.get_optimal_candidates(tol)
        if self.n_opt_c is 0:
            print(
                f"[Warning]: empty optimal candidates, skipping printing of optimal "
                f"candidates."
            )
            return

        print("")
        print(f"{' Optimal Candidates ':#^100}")
        print(f"{'Obtained on':<40}: {datetime.now()}")
        print(f"{'Criterion':<40}: {self._current_criterion}")
        print(f"{'Pseudo-bayesian':<40}: {self._pseudo_bayesian}")
        if self._pseudo_bayesian:
            print(f"{'Pseudo-bayesian Criterion Type':<40}: {self._pseudo_bayesian_type}")
        print(f"{'Dynamic':<40}: {self._dynamic_system}")
        print(f"{'Time-invariant Controls':<40}: {self._invariant_controls}")
        print(f"{'Time-varying Controls':<40}: {self._dynamic_controls}")
        print(f"{'Number of Candidates':<40}: {self.n_c}")
        print(f"{'Number of Optimal Candidates':<40}: {self.n_opt_c}")
        if self._dynamic_system:
            print(f"{'Number of Sampling Time Choices':<40}: {self.n_spt}")
            print(f"{'Sampling Times Optimized':<40}: {self._opt_sampling_times}")
            if self._opt_sampling_times:
                print(f"{'Number of Samples Per Experiment':<40}: {self._n_spt_spec}")
        if self._pseudo_bayesian:
            print(f"{'Number of Scenarios':<40}: {self.n_scr}")

        for i, opt_cand in enumerate(self.optimal_candidates):
            print(f"{f'[Candidate {opt_cand[0] + 1:d}]':-^100}")
            print(f"{f'Recommended Effort: {np.sum(opt_cand[4]):.2%} of experiments':^100}")
            if self._invariant_controls:
                print("Time-invariant Controls:")
                print(opt_cand[1])
            if self._dynamic_controls:
                print("Time-varying Controls:")
                print(opt_cand[2])
            if self._dynamic_system:
                if self._opt_sampling_times:
                    if self._specified_n_spt:
                        print("Sampling Time Variants:")
                        for comb, spt_comb in enumerate(opt_cand[3]):
                            print(f"  Variant {comb+1} ~ [", end='')
                            for j, sp_time in enumerate(spt_comb):
                                print(f"{f'{sp_time:.2f}':>10}", end='')
                            print("]: ", end='')
                            print(f'{f"{opt_cand[4][comb].sum():.2%}":>10} of experiments')
                    else:
                        print("Sampling Times:")
                        for j, sp_time in enumerate(opt_cand[3]):
                            print(f"[{f'{sp_time:.2f}':>10}]: "
                                  f"dedicate {f'{opt_cand[4][j]:.2%}':>6} of experiments")
                else:
                    print("Sampling Times:")
                    print(self.sampling_times_candidates[i])
        print(f"{'':#^100}")

        if write:
            self.create_result_dir()
            writer = open(
                self.result_dir + f"/optimal_candidates_{self.run_no}.txt",
                "a"
            )
            writer.write(f"{' Optimal Candidates ':#^100}")
            writer.write("\n")
            writer.write(f"{'Obtained on':<40}: {datetime.now()}")
            writer.write("\n")
            writer.write(f"{'Criterion':<40}: {self._current_criterion}")
            writer.write("\n")
            writer.write(f"{'Pseudo-bayesian':<40}: {self._pseudo_bayesian}")
            writer.write("\n")
            if self._pseudo_bayesian:
                writer.write(
                    f"{'Pseudo-bayesian Criterion Type':<40}: {self._pseudo_bayesian_type}")
                writer.write("\n")
            writer.write(f"{'Dynamic':<40}: {self._dynamic_system}")
            writer.write("\n")
            writer.write(f"{'Time-invariant Controls':<40}: {self._invariant_controls}")
            writer.write("\n")
            writer.write(f"{'Time-varying Controls':<40}: {self._dynamic_controls}")
            writer.write("\n")
            writer.write(f"{'Number of Candidates':<40}: {self.n_c}")
            writer.write("\n")
            writer.write(f"{'Number of Optimal Candidates':<40}: {self.n_opt_c}")
            writer.write("\n")
            if self._dynamic_system:
                writer.write(f"{'Number of Sampling Time Choices':<40}: {self.n_spt}")
                writer.write("\n")
                writer.write(f"{'Sampling Times Optimized':<40}: {self._opt_sampling_times}")
                writer.write("\n")
                if self._opt_sampling_times:
                    writer.write(
                        f"{'Number of Samples Per Experiment':<40}: {self._n_spt_spec}")
                    writer.write("\n")
            if self._pseudo_bayesian:
                writer.write(f"{'Number of Scenarios':<40}: {self.n_scr}")
                writer.write("\n")

            for i, opt_cand in enumerate(self.optimal_candidates):
                writer.write(f"{f'[Candidate {opt_cand[0] + 1:d}]':-^100}")
                writer.write("\n")
                writer.write(
                    f"{f'Recommended Effort: {np.sum(opt_cand[4]):.2%} of experiments':^100}")
                writer.write("\n")
                if self._invariant_controls:
                    writer.write("Time-invariant Controls:")
                    writer.write("\n")
                    writer.write(str(opt_cand[1]))
                    writer.write("\n")
                if self._dynamic_controls:
                    writer.write("Time-varying Controls:")
                    writer.write("\n")
                    writer.write(str(opt_cand[2]))
                    writer.write("\n")
                if self._dynamic_system:
                    if self._opt_sampling_times:
                        if self._specified_n_spt:
                            writer.write("Sampling Time Variants:")
                            writer.write("\n")
                            for comb, spt_comb in enumerate(opt_cand[3]):
                                writer.write(f"  Variant {comb + 1} ~ [",)
                                writer.write("\n")
                                for j, sp_time in enumerate(spt_comb):
                                    writer.write(f"{f'{sp_time:.2f}':>10}",)
                                    writer.write("\n")
                                writer.write("]: ",)
                                writer.write("\n")
                                writer.write(
                                    f'{f"{opt_cand[4][comb].sum():.2%}":>10} of experiments')
                                writer.write("\n")
                        else:
                            writer.write("Sampling Times:")
                            writer.write("\n")
                            for j, sp_time in enumerate(opt_cand[3]):
                                writer.write(f"[{f'{sp_time:.2f}':>10}]: "
                                      f"dedicate {f'{opt_cand[4][j]:.2%}':>6} of experiments")
                                writer.write("\n")
                    else:
                        writer.write("Sampling Times:")
                        writer.write("\n")
                        writer.write(str(self.sampling_times_candidates[i]))
                        writer.write("\n")
            writer.write(f"{'':#^100}")
            writer.write("\n")
            writer.close()

    @staticmethod
    def show_plots():
        plt.show()

    # saving, loading, writing
    def load_oed_result(self, result_path):
        oed_result = dill.load(open(getcwd() + result_path, "rb"))

        self._optimization_time = oed_result["optimization_time"]
        self._sensitivity_analysis_time = oed_result["sensitivity_analysis_time"]
        self._current_criterion = oed_result["optimality_criterion"]
        self.ti_controls_candidates = oed_result["ti_controls_candidates"]
        self.tv_controls_candidates = oed_result["tv_controls_candidates"]
        self.model_parameters = oed_result["model_parameters"]
        self.sampling_times_candidates = oed_result["sampling_times_candidates"]
        self.efforts = oed_result["optimal_efforts"]
        self._optimizer = oed_result["optimizer"]

    def create_result_dir(self):
        if self.result_dir is None:
            now = datetime.now()
            self.result_dir = getcwd() + "/"
            self.result_dir = self.result_dir + \
                              path.splitext(path.basename(main.__file__))[0] + "_result/"
            self.result_dir = self.result_dir + 'date_%d-%d-%d/' % (
                now.year, now.month, now.day)
            self.create_result_dir()
        else:
            if path.exists(self.result_dir):
                return
            else:
                makedirs(self.result_dir)

    def write_oed_result(self):
        self.create_result_dir()

        result_file = self.result_dir + "/%s_oed_result_%d.pkl" % (
            self.oed_result["optimality_criterion"], self.run_no)
        if path.isfile(result_file):
            self.run_no += 1
            self.write_oed_result()
        else:
            self.run_no = 1  # revert run numbers
            dump(self.oed_result, open(result_file, "wb"))

    def save_state(self):
        self.create_result_dir()

        # pre-process the designer before saving
        state = [
            self.n_c,
            self.n_spt,
            self.n_r,
            self.n_mp,
            self.ti_controls_candidates,
            self.tv_controls_candidates,
            self.sampling_times_candidates,
            self.measurable_responses,
            self.n_m_r,
            self.model_parameters,
        ]

        designer_file = self.result_dir + "/" + 'state' + "_%d.pkl" % self.run_no
        if path.isfile(designer_file):
            self.run_no += 1
            self.save_state()
        else:
            self.run_no = 1  # revert run numbers
            dill.dump(state, open(designer_file, "wb"))

    def load_state(self, designer_path):
        state = dill.load(open(getcwd() + designer_path, 'rb'))
        self.n_c = state[0]
        self.n_spt = state[1]
        self.n_r = state[2]
        self.n_mp = state[3]
        self.ti_controls_candidates = state[4]
        self._old_tic = state[4]
        self.tv_controls_candidates = state[5]
        self._old_tvc = state[5]
        self.sampling_times_candidates = state[6]
        self._old_spt = state[6]
        self.measurable_responses = state[7]
        self.n_m_r = state[8]
        self.model_parameters = state[9]

        self._last_scr_mp = self.model_parameters

    def save_responses(self):
        pass

    def load_sensitivity(self, sens_path):
        self.sensitivities = load(open(getcwd() + sens_path, 'rb'))
        return self.sensitivities

    """ criteria """

    # calibration-oriented
    def d_opt_criterion(self, efforts):
        """ it is a PSD criterion, with exponential cone """
        if self._pseudo_bayesian:
            return self._pb_d_opt_criterion(efforts)
        else:
            return self._d_opt_criterion(efforts)

    def a_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        if self._pseudo_bayesian:
            return self._pb_a_opt_criterion(efforts)
        else:
            return self._a_opt_criterion(efforts)

    def e_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        if self._pseudo_bayesian:
            return self._pb_e_opt_criterion(efforts)
        else:
            return self._e_opt_criterion(efforts)

    # prediction-oriented
    def dg_opt_criterion(self, efforts):
        if self._pseudo_bayesian:
            return self._pb_dg_opt_criterion(efforts)
        else:
            return self._dg_opt_criterion(efforts)

    def di_opt_criterion(self, efforts):
        if self._pseudo_bayesian:
            return self._pb_di_opt_criterion(efforts)
        else:
            return self._di_opt_criterion(efforts)

    def ag_opt_criterion(self, efforts):
        if self._pseudo_bayesian:
            return self._pb_ag_opt_criterion(efforts)
        else:
            return self._ag_opt_criterion(efforts)

    def ai_opt_criterion(self, efforts):
        if self._pseudo_bayesian:
            return self._pb_ai_opt_criterion(efforts)
        else:
            return self._ai_opt_criterion(efforts)

    def eg_opt_criterion(self, efforts):
        if self._pseudo_bayesian:
            return self._pb_eg_opt_criterion(efforts)
        else:
            return self._eg_opt_criterion(efforts)

    def ei_opt_criterion(self, efforts):
        if self._pseudo_bayesian:
            return self._pb_ei_opt_criterion(efforts)
        else:
            return self._ei_opt_criterion(efforts)

    # experimental
    def u_opt_criterion(self, efforts):
        self.eval_fim(efforts, self.model_parameters)
        return -np.sum(np.multiply(self.fim, self.fim))

    """ evaluators """

    def eval_residuals(self, model_parameters):
        self.model_parameters = model_parameters

        """ run the model to get predictions """
        self.simulate_candidates()
        if self._dynamic_system:
            self.residuals = self.data - self.response[:, :, self.measurable_responses]
        else:
            self.residuals = self.data - self.response[:, self.measurable_responses]

        return self.residuals[
            ~np.isnan(self.residuals)]  # return residuals where entries are not empty

    def eval_sensitivities(self, method='forward', base_step=2, step_ratio=2,
                           num_steps=None, store_predictions=True,
                           plot_analysis_times=False, save_sensitivities=False,
                           reporting_frequency=10):
        """
        Main evaluator for computing numerical sensitivities of the responses with
        respect to the model parameters. Simply provides an interface to numdifftool's
        Jacobian method.

        Numdifftool uses adaptive finite difference to approximate the sensitivities,
        coupled with Richard extrapolation for improved accuracy. Although less accurate,
        default behaviour is to use forward finite difference. This is to prevent model
        instability in common situations where during sensitivity evaluation (e.g.
        with central) model parameter values that are passed to the model changes sign
        and causes the model to fail to run.
        """
        if num_steps is not None:
            self._num_steps = num_steps
        # setting default behaviour for step generators
        step_generator = nd.step_generators.MaxStepGenerator(
            base_step=base_step,
            step_ratio=step_ratio,
            num_steps=self._num_steps,
        )

        self._save_sensitivities = save_sensitivities

        self._check_if_scr_changed()
        self._check_if_candidates_changed()

        if self._scr_sens is None and self._pseudo_bayesian:
            self._scr_sens = []

        """ do analysis if empty or model parameters were changed """
        if self.sensitivities is None or self._scr_changed or self._candidates_changed:
            self._sensitivity_analysis_done = False
            if self._verbose >= 2:
                print('[Sensitivity Analysis]'.center(100, "-"))
            start = time()
            sens = []
            candidate_sens_times = []
            jacob_fun = nd.Jacobian(fun=self._sensitivity_sim_wrapper,
                                    step=step_generator, method=method)
            """ main loop over experimental candidates """
            main_loop_start = time()
            for i, exp_candidate in enumerate(
                    zip(self.sampling_times_candidates, self.ti_controls_candidates,
                        self.tv_controls_candidates)):
                """ specifying current experimental candidate """
                self._ti_controls = exp_candidate[1]
                self._tv_controls = exp_candidate[2]
                self._sampling_times = exp_candidate[0][~np.isnan(exp_candidate[0])]

                self.feval_sensitivity = 0
                single_start = time()
                temp_sens = jacob_fun(self._current_scr_mp, store_predictions)
                finish = time()
                if self._verbose >= 2:
                    if (i + 1) % np.ceil(self.n_c / reporting_frequency) == 0 or (
                            i + 1) == self.n_c:
                        print(
                            f'[Candidate {f"{i + 1:d}/{self.n_c:d}":>10}]: '
                            f'time elapsed {f"{finish - main_loop_start:.2f}":>15} seconds.'
                        )
                candidate_sens_times.append(finish - single_start)
                """
                bunch of lines to make sure the Jacobian method returns the 
                sensitivity with dims: n_sp, n_res, n_theta
                -------------------------------------------------------------------------
                8 possible cases
                -------------------------------------------------------------------------
                case_1: complete                        n_sp    n_theta     n_res
                case_2: n_sp = 1                        n_res   n_theta
                case_3: n_theta = 1                     n_res   n_sp
                case_4: n_res = 1                       n_sp    n_theta
                case_5: n_sp & n_theta = 1              1       n_res
                case_6: n_sp & n_res = 1                1       n_theta
                case_7: n_theta & n_res = 1             1       n_sp
                case_8: n_sp, n_res, n_theta = 1        1       1
                -------------------------------------------------------------------------
                """
                n_dim = len(temp_sens.shape)
                if n_dim == 3:  # covers case 1
                    temp_sens = np.moveaxis(temp_sens, 1, 2)  # switch n_theta and n_res
                elif self.n_spt == 1:
                    if self.n_mp == 1:  # covers case 5: add a new axis in the last dim
                        temp_sens = temp_sens[:, :, np.newaxis]
                    else:  # covers case 2, 6, and 8: add a new axis in
                        # the first dim
                        temp_sens = temp_sens[np.newaxis]
                elif self.n_mp == 1:  # covers case 3 and 7
                    temp_sens = np.moveaxis(temp_sens, 0,
                                            1)  # move n_sp to the first dim as needed
                    temp_sens = temp_sens[:, :,
                                np.newaxis]  # create a new axis as the last dim for
                    # n_theta
                elif self.n_r == 1:  # covers case 4
                    temp_sens = temp_sens[:, np.newaxis,
                                :]  # create axis in the middle for n_res

                """ appending the formatted sensitivity matrix for each candidate into 
                the final list to be returned """
                sens.append(temp_sens)
            finish = time()
            if self._verbose >= 2:
                print("".center(100, "-"))
            self._sensitivity_analysis_time += finish - start

            # converting sens into a numpy array for optimizing further computations
            self.sensitivities = np.array(sens)

            # saving current model parameters
            self._last_scr_mp = np.copy(self.model_parameters)

            if self._var_n_sampling_time:
                self._pad_sensitivities()

            if self._pseudo_bayesian and not self._large_memory_requirement:
                if isinstance(self._scr_sens, np.ndarray):
                    self._scr_sens = self._scr_sens.tolist()
                self._scr_sens.append(self.sensitivities)
                self._scr_sens = np.array(self._scr_sens)

            if self._save_sensitivities:
                self.create_result_dir()
                self.run_no = 1
                sens_file = self.result_dir + '/sensitivity_%d.pkl' % self.run_no
                while path.isfile(sens_file):
                    self.run_no += 1
                    sens_file = self.result_dir + '/sensitivity_%d.pkl' % self.run_no
                dump(self.sensitivities, open(sens_file, 'wb'))

            if plot_analysis_times:
                fig = plt.figure()
                axes = fig.add_subplot(111)
                axes.plot(np.arange(1, self.n_c + 1, step=1), candidate_sens_times)

        self._sensitivity_analysis_done = True
        self._old_tic = self.ti_controls_candidates
        if self._dynamic_system:
            self._old_spt = self.sampling_times_candidates
            if self._dynamic_controls:
                self._old_tvc = self.tv_controls_candidates
        return self.sensitivities

    def eval_sensitivities_autograd(self):
        return

    def eval_fim(self, efforts, mp, store_predictions=True):
        """
        Main evaluator for constructing the fim from obtained sensitivities.
        When scipy is used as optimization package and problem does not require large
        memory, will store atomic fims for analytical Jacobian.

        The function also performs a parameter estimability study based on the FIM by
        summing the squares over the rows and columns of the FIM. Optionally, will trim
        out rows and columns that have its sum of squares close to 0. This helps with
        noninvertible FIMs.

        An alternative for dealing with noninvertible FIMs is to use a simple Tikhonov
        regularization, where a small scalar times the identity matrix is added to the
        FIM to obtain an invertible matrix.
        """
        def add_candidates(s_in, e_in):
            if not np.any(np.isnan(s_in)):
                _atom_fim = s_in.T @ s_in
                self.fim += e_in * _atom_fim
            else:
                _atom_fim = np.zeros((self.n_mp, self.n_mp))
            if self._optimization_package is "scipy" and \
                    not self._large_memory_requirement:
                self.atomic_fims.append(_atom_fim)

        """ update mp, and efforts """
        self.efforts = efforts
        self._current_scr_mp = mp

        """ eval_sensitivities, only runs if model parameters changed """
        self.eval_sensitivities(
            save_sensitivities=self._save_sensitivities,
            store_predictions=store_predictions,
        )

        """ deal with unconstrained form, i.e. transform efforts """
        if self._unconstrained_form:
            self._efforts_transformed = False
        self._transform_efforts()  # only transform if required, logic incorporated there

        """ evaluate fim """
        start = time()
        self.fim = 0
        if self._optimization_package is "scipy":
            if self._specified_n_spt:
                self.efforts = self.efforts.reshape((self.n_c, self.n_spt_comb))
            else:
                self.efforts = self.efforts.reshape((self.n_c, self.n_spt))
            if not self._large_memory_requirement:
                self.atomic_fims = []
        if self._specified_n_spt:
            for c, (eff, sen, spt_combs) in enumerate(zip(self.efforts, self.sensitivities, self.spt_candidates_combs)):
                for comb, (e, spt) in enumerate(zip(eff, spt_combs)):
                    s = np.sum(sen[spt], axis=0)
                    add_candidates(s, e)
        else:
            for c, (eff, sen) in enumerate(zip(self.efforts, self.sensitivities)):
                for spt, (e, s) in enumerate(zip(eff, sen)):
                    add_candidates(s, e)
        finish = time()

        if self.fim is 0:
            return np.array([0])
        else:
            self.evaluate_estimability_index()

        if self._regularize_fim:
            if self._verbose >= 3:
                print(
                    f"Applying Tikhonov regularization to FIM by adding "
                    f"{self._eps:.2f} * identity to the FIM. "
                    f"Warning: design is likely to be affected for large scalars!"
                )
            self.fim += self._eps * np.identity(self.n_mp)

        self._fim_eval_time = finish - start
        if self._verbose >= 3:
            print(
                f"Evaluation of fim took {self._fim_eval_time:.2f} seconds."
            )

        return self.fim

    def eval_pim(self, efforts, mp, vector=False):
        if self._optimization_package is "cvxpy":
            raise NotImplementedError

        """ update mp, and efforts """
        self.eval_fim(efforts, mp)

        fim_inv = np.linalg.inv(self.fim)
        if vector:
            self.pvars = np.array([
                [f @ fim_inv @ f.T for f in F] for F in self.sensitivities
            ])
        else:
            self.pvars = np.empty((self.n_c, self.n_spt, self.n_r, self.n_r))
            for c, F in enumerate(self.sensitivities):
                for spt, f in enumerate(F):
                    self.pvars[c, spt, :, :] = f @ fim_inv @ f.T

        return self.pvars

    def eval_atom_fims(self, mp, store_predictions=True):
        self._current_scr_mp = mp

        """ eval_sensitivities, only runs if model parameters changed """
        self.eval_sensitivities(save_sensitivities=self._save_sensitivities,
                                store_predictions=store_predictions)

        """ deal with unconstrained form, i.e. transform efforts """
        self._transform_efforts()  # only transform if required, logic incorporated there

        """ deal with opt_sampling_times """
        sens = self.sensitivities.reshape(self.n_c * self.n_spt, self.n_m_r, self.n_mp)

        """ main """
        start = time()
        if self._large_memory_requirement:
            confirmation = input(
                f"Memory requirement is large. Slow solution expected, continue?"
                f"Y/N."
            )
            if confirmation != "Y":
                return
        self.atomic_fims = []
        for e, f in zip(self.efforts.flatten(), sens):
            if not np.any(np.isnan(f)):
                _atom_fim = f.T @ f
            else:
                _atom_fim = np.zeros(shape=(self.n_mp, self.n_mp))
            self.atomic_fims.append(_atom_fim)
        finish = time()
        self._fim_eval_time = finish - start

        return self.atomic_fims

    def eval_scr_fims(self, store_predictions=True):

        self._check_if_scr_changed()
        self._check_if_candidates_changed()

        if self._verbose >= 2:
            print(f"{' Pseudo-bayesian ':#^100}")
        if self._verbose >= 1:
            print(f'Evaluating information for each scenario...')
        self.scr_fims = []
        if store_predictions:
            self.scr_responses = []
        for scr, mp in enumerate(self.model_parameters):
            self._current_scr = scr
            if self._verbose >= 2:
                print(f"{f'[Scenario {scr+1}/{self.n_scr}]':=^100}")
                print("Model Parameters:")
                print(mp)
            self.eval_fim(self.efforts, mp, store_predictions)
            self.scr_fims.append(self.fim)
            if self._verbose >= 2:
                print(f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds.")
            if store_predictions:
                self.scr_responses.append(self.response)
            self.response = None
        self.scr_responses = np.array(self.scr_responses)

        return self.scr_fims

    """ getters (filters) """

    def get_optimal_candidates(self, tol=1e-2):
        if self.efforts is None:
            raise SyntaxError(
                'Please solve an experiment design before attempting to get optimal '
                'candidates.'
            )

        self.optimal_candidates = []

        for i, eff_sp in enumerate(self.efforts):
            if np.sum(eff_sp) > tol:
                opt_candidate = [
                    i,  # index of optimal candidate
                    self.ti_controls_candidates[i],
                    self.tv_controls_candidates[i],
                    [],
                    [],
                    [],
                    []
                ]
                if self._opt_sampling_times:
                    for j, eff in enumerate(eff_sp):
                        if eff > tol:
                            if self._specified_n_spt:
                                opt_spt = self.sampling_times_candidates[i, self.spt_candidates_combs[i, j]]
                                opt_candidate[3].append(opt_spt)
                                opt_candidate[4].append(np.ones_like(opt_spt) * eff / len(opt_spt))
                                opt_candidate[5].append(self.spt_candidates_combs[i, j])
                            else:
                                opt_candidate[3].append(self.sampling_times_candidates[i][j])
                                opt_candidate[4].append(eff)
                                opt_candidate[5].append(j)
                else:
                    opt_candidate[3].append(self.sampling_times_candidates[i])
                    opt_candidate[4].append(eff_sp)
                    opt_candidate[5].append([t for t in range(self.n_spt)])
                self.optimal_candidates.append(opt_candidate)

        self.n_opt_c = len(self.optimal_candidates)
        if self.n_opt_c is 0:
            print(
                f"[Warning]: empty optimal candidates. Likely failed optimization; if "
                f"prediction-orriented design is used, try avoiding dg, ag, or eg "
                f"criteria as they are notoriously hard to optimize with gradient-based "
                f"optimizers."
            )
        return self.optimal_candidates

    """ optional operations """

    def evaluate_estimability_index(self):
        self.estimable_model_parameters = np.array([])
        self.estimability = np.array([])
        if self._optimization_package is 'cvxpy':
            fim_value = self.fim.value
        else:
            fim_value = self.fim
        if fim_value is 0:
            return

        for i, row in enumerate(fim_value):
            if not np.allclose(row, 0.0):
                self.estimable_model_parameters = np.append(
                    self.estimable_model_parameters, i)
                self.estimability = np.append(self.estimability,
                                              np.sqrt(np.inner(row, row)))
        self.estimable_model_parameters = self.estimable_model_parameters.astype(int)

        if self._trim_fim:
            if len(self.estimable_model_parameters) is 0:
                self.fim = np.array([0])
            else:
                self.fim = self.fim[
                    np.ix_(self.estimable_model_parameters, self.estimable_model_parameters)
                ]

    def normalize_sensitivities(self, overwrite_unnormalized=False):
        assert not np.allclose(self.model_parameters,
                               0), 'At least one nominal model parameter value is ' \
                                   'equal to 0, cannot normalize sensitivities. ' \
                                   'Consider re-estimating your parameters or ' \
                                   're-parameterize your model.'

        # normalize parameter values
        self.normalized_sensitivity = np.multiply(self.sensitivities,
                                                  self.model_parameters[None, None, None,
                                                  :])
        if self.responses_scales is None:
            if self._verbose >= 0:
                print(
                    'Scale for responses not given, using raw prediction values to '
                    'normalize sensitivities; '
                    'likely to fail (e.g. if responses are near 0). Recommend: provide '
                    'designer with scale '
                    'info through: "designer.responses_scale = <your_scale_array>."')
            if self.response is None:
                self.simulate_candidates(store_predictions=True)
            # normalize response values
            self.normalized_sensitivity = np.divide(
                self.normalized_sensitivity,
                self.response[:, :, :, None],
            )
        else:
            assert isinstance(self.responses_scales,
                              np.ndarray), "Please specify responses_scales as a 1D " \
                                           "numpy array."
            assert self.responses_scales.size == self.n_r, 'Length of responses scales ' \
                                                           'those which are measurable ' \
                                                           'and not).)'
            self.normalized_sensitivity = np.divide(self.normalized_sensitivity,
                                                    self.responses_scales[None, None, :,
                                                    None])
        if overwrite_unnormalized:
            self.sensitivities = self.normalized_sensitivity
            self._sensitivity_is_normalized = True
            return self.sensitivities
        return self.normalized_sensitivity

    """ local criterion """

    # calibration-oriented
    def _d_opt_criterion(self, efforts):
        """ it is a PSD criterion, with exponential cone """
        self.eval_fim(efforts, self.model_parameters)

        if self.fim.size == 1:
            if self._optimization_package is "scipy":
                d_opt = -np.log1p(self.fim)
                if self._fd_jac:
                    return d_opt
                else:
                    jac = -np.array([
                        1 / self.fim * m
                        for m in self.atomic_fims
                    ])
                    return d_opt, jac
            elif self._optimization_package is 'cvxpy':
                return -self.fim

        if self._optimization_package is "scipy":
            sign, d_opt = np.linalg.slogdet(self.fim)
            if self._fd_jac:
                if sign == 1:
                    return -d_opt
                else:
                    return np.inf
            else:
                fim_inv = np.linalg.inv(self.fim)
                jac = -np.array([
                    np.sum(fim_inv.T * m)
                    for m in self.atomic_fims
                ])
                if sign == 1:
                    return -d_opt, jac
                else:
                    return np.inf, jac

        elif self._optimization_package is 'cvxpy':
            return -cp.log_det(self.fim)

    def _a_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        self.eval_fim(efforts, self.model_parameters)

        if self.fim.size == 1:
            if self._optimization_package is "scipy":
                if self._fd_jac:
                    return -self.fim
                else:
                    jac = np.array([
                        m for m in self.atomic_fims
                    ])
                    return -self.fim, jac
            elif self._optimization_package is "cvxpy":
                return -self.fim

        if self._optimization_package is "scipy":
            if self._fd_jac:
                eigvals = np.linalg.eigvalsh(self.fim)
                if np.all(eigvals > 0):
                    a_opt = np.sum(1 / eigvals)
                else:
                    a_opt = 0
                return a_opt
            else:
                jac = np.zeros(self.n_e)
                try:
                    fim_inv = np.linalg.inv(self.fim)
                    a_opt = fim_inv.trace()
                    if not self._fd_jac:
                        jac = -np.array([
                            np.sum((fim_inv @ fim_inv) * m) for m in self.atomic_fims
                        ])
                except np.linalg.LinAlgError:
                    a_opt = 0
                return a_opt, jac

        elif self._optimization_package is 'cvxpy':
            return cp.matrix_frac(np.identity(self.fim.shape[0]), self.fim)

    def _e_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        self.eval_fim(efforts, self.model_parameters)

        if self.fim.size == 1:
            return -np.log1p(self.fim)

        if self._optimization_package is "scipy":
            if self._fd_jac:
                return -np.linalg.eigvalsh(self.fim).min()
            else:
                raise NotImplementedError  # TODO: implement analytic jac for e-opt
        elif self._optimization_package is 'cvxpy':
            return -cp.lambda_min(self.fim)

    # prediction-oriented
    def _dg_opt_criterion(self, efforts):
        if self._optimization_package is "cvxpy":
            raise NotImplementedError("CVXPY unavailable for dg_opt.")

        self.eval_pim(efforts, self.model_parameters)
        # dg_opt: max det of the pvar matrix over candidates and sampling times
        dg_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                sign, temp_dg = np.linalg.slogdet(pvar)
                if sign != 1:
                    temp_dg = np.inf
                dg_opts[c, spt] = sign * np.exp(temp_dg)
        dg_opt = np.max(dg_opts)

        if self._fd_jac:
            return dg_opt
        else:
            raise NotImplementedError("Analytic Jacobian for dg_opt unavailable.")

    def _di_opt_criterion(self, efforts):
        if self._optimization_package is "cvxpy":
            raise NotImplementedError("CVXPY unavailable for di_opt.")

        self.eval_pim(efforts, self.model_parameters)
        # di_opt: average det of the pvar matrix over candidates and sampling times
        dg_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                sign, temp_dg = np.linalg.slogdet(pvar)
                if sign != 1:
                    temp_dg = np.inf
                dg_opts[c, spt] = temp_dg
        dg_opt = np.sum(dg_opts)

        if self._fd_jac:
            return dg_opt
        else:
            raise NotImplementedError("Analytic Jacobian for di_opt unavailable.")

    def _ag_opt_criterion(self, efforts):
        if self._optimization_package is "cvxpy":
            raise NotImplementedError("CVXPY unavailable for ag_opt.")

        self.eval_pim(efforts, self.model_parameters)
        # ag_opt: max trace of the pvar matrix over candidates and sampling times
        ag_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.trace(pvar)
                ag_opts[c, spt] = temp_dg
        ag_opt = np.max(ag_opts)

        if self._fd_jac:
            return ag_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ag_opt unavailable.")

    def _ai_opt_criterion(self, efforts):
        if self._optimization_package is "cvxpy":
            raise NotImplementedError("CVXPY unavailable for ai_opt.")

        self.eval_pim(efforts, self.model_parameters)
        # ai_opt: average trace of the pvar matrix over candidates and sampling times
        ai_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.trace(pvar)
                ai_opts[c, spt] = temp_dg
        ag_opt = np.sum(ai_opts)

        if self._fd_jac:
            return ag_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ai_opt unavailable.")

    def _eg_opt_criterion(self, efforts):
        if self._optimization_package is "cvxpy":
            raise NotImplementedError("CVXPY unavailable for eg_opt.")

        self.eval_pim(efforts, self.model_parameters)
        # eg_opt: max of the max_eigenval of the pvar matrix over candidates and sampling times
        eg_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.linalg.eigvals(pvar).max()
                eg_opts[c, spt] = temp_dg
        eg_opt = np.max(eg_opts)

        if self._fd_jac:
            return eg_opt
        else:
            raise NotImplementedError("Analytic Jacobian for eg_opt unavailable.")

    def _ei_opt_criterion(self, efforts):
        if self._optimization_package is "cvxpy":
            raise NotImplementedError("CVXPY unavailable for ei_opt.")

        self.eval_pim(efforts, self.model_parameters)
        # ei_opts: average of the max_eigenval of the pvar matrix over candidates and sampling times
        ei_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.linalg.eigvals(pvar).max()
                ei_opts[c, spt] = temp_dg
        ei_opt = np.sum(ei_opts)

        if self._fd_jac:
            return ei_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ei_opt unavailable.")

    """ pseudo_bayesian criterion """

    # calibration-oriented
    def _pb_d_opt_criterion(self, efforts):
        """ it is a PSD criterion, with exponential cone """
        self.efforts = efforts

        self.eval_scr_fims()

        if self._optimization_package is "scipy":
            if self._fd_jac:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = np.sum([fim for fim in self.scr_fims], axis=0)
                    sign, d_opt = np.linalg.slogdet(avg_fim)
                    if sign != 1:
                        return np.inf
                    else:
                        return -d_opt
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    d_opt = 0
                    for fim in self.scr_fims:
                        sign, scr_d_opt = np.linalg.slogdet(fim)
                        if sign != 1:
                            scr_d_opt = np.inf
                        d_opt += scr_d_opt
                    return -d_opt
            else:
                raise NotImplementedError(
                    "Analytical Jacobian unimplemented for Pseudo-bayesian D-optimal."
                )

        elif self._optimization_package is 'cvxpy':
            if np.any([fim.shape == (1, 1) for fim in self.scr_fims]):
                return cp.sum([-fim for fim in self.scr_fims])
            else:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = cp.sum([fim for fim in self.scr_fims], axis=0)
                    return -cp.log_det(avg_fim)
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    return cp.sum([
                        -cp.log_det(fim) for fim in self.scr_fims
                    ],
                    axis=0,
                    )

    def _pb_a_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        self.efforts = efforts

        self.eval_scr_fims()

        if self._optimization_package is "scipy":
            if self._fd_jac:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    a_opt = np.linalg.inv(
                        np.sum([fim for fim in self.scr_fims], axis=0)
                    ).trace()
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    np.sum([
                        np.linalg.inv(fim).trace()
                        for fim in self.scr_fims
                    ])
            else:
                raise NotImplementedError(
                    "Analytical Jacobian unimplemented for Pseudo-bayesian D-optimal."
                )

        elif self._optimization_package is 'cvxpy':
            if np.any([fim.shape == (1, 1) for fim in self.scr_fims]):
                return cp.sum([-fim for fim in self.scr_fims])
            else:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = cp.sum([fim for fim in self.scr_fims], axis=0)
                    return cp.matrix_frac(np.identity(avg_fim.shape[0]), avg_fim)
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    return cp.sum([
                        cp.matrix_frac(np.identity(fim.shape[0]), fim)
                        for fim in self.scr_fims
                    ])

    def _pb_e_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        self.efforts = efforts

        self.eval_scr_fims()

        if self._optimization_package is "scipy":
            if self._fd_jac:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = np.sum([fim for fim in self.scr_fims], axis=0)
                    return -np.linalg.eigvalsh(avg_fim).min()
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    return np.sum([
                        -np.linalg.eigvalsh(fim).min()
                        for fim in self.scr_fims
                    ])
            else:
                raise NotImplementedError(
                    "Analytical Jacobian unimplemented for Pseudo-bayesian D-optimal."
                )

        elif self._optimization_package is 'cvxpy':
            if np.any([fim.shape == (1, 1) for fim in self.scr_fims]):
                return cp.sum([-fim for fim in self.scr_fims])
            else:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = cp.sum([fim for fim in self.scr_fims], axis=0)
                    return -cp.lambda_min(avg_fim)
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    return cp.sum([
                        -cp.lambda_min(fim)
                        for fim in self.scr_fims
                    ])

    # prediction-oriented
    def _pb_dg_opt_criterion(self, efforts):
        if self._optimization_package is "cvxpy":
            raise NotImplementedError("CVXPY unavailable for dg_opt.")

        self.efforts = efforts

        self.eval_pim(efforts, self._current_scr_mp)
        # dg_opt: max det of the pvar matrix over candidates and sampling times
        dg_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                sign, temp_dg = np.linalg.slogdet(pvar)
                if sign != 1:
                    temp_dg = np.inf
                dg_opts[c, spt] = temp_dg
        dg_opt = np.max(dg_opts)

        if self._fd_jac:
            return dg_opt
        else:
            raise NotImplementedError("Analytic Jacobian for dg_opt unavailable.")

    def _pb_di_opt_criterion(self, efforts):
        if self._optimization_package is "cvxpy":
            raise NotImplementedError("CVXPY unavailable for di_opt.")

        self.efforts = efforts

        self.eval_pim()
        # di_opt: average det of the pvar matrix over candidates and sampling times
        dg_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                sign, temp_dg = np.linalg.slogdet(pvar)
                if sign != 1:
                    temp_dg = np.inf
                dg_opts[c, spt] = temp_dg
        dg_opt = np.sum(dg_opts)

        if self._fd_jac:
            return dg_opt
        else:
            raise NotImplementedError("Analytic Jacobian for di_opt unavailable.")

    def _pb_ag_opt_criterion(self, efforts):
        if self._optimization_package is "cvxpy":
            raise NotImplementedError("CVXPY unavailable for ag_opt.")

        self.efforts = efforts

        self.eval_pim()
        # ag_opt: max trace of the pvar matrix over candidates and sampling times
        ag_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.trace(pvar)
                ag_opts[c, spt] = temp_dg
        ag_opt = np.max(ag_opts)

        if self._fd_jac:
            return ag_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ag_opt unavailable.")

    def _pb_ai_opt_criterion(self, efforts):
        if self._optimization_package is "cvxpy":
            raise NotImplementedError("CVXPY unavailable for ai_opt.")

        self.efforts = efforts

        self.eval_pim()
        # ai_opt: average trace of the pvar matrix over candidates and sampling times
        ai_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.trace(pvar)
                ai_opts[c, spt] = temp_dg
        ag_opt = np.sum(ai_opts)

        if self._fd_jac:
            return ag_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ai_opt unavailable.")

    def _pb_eg_opt_criterion(self, efforts):
        if self._optimization_package is "cvxpy":
            raise NotImplementedError("CVXPY unavailable for eg_opt.")

        self.efforts = efforts

        self.eval_pim()
        # eg_opt: max of the max_eigenval of the pvar matrix over candidates and sampling times
        eg_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.linalg.eigvals(pvar).max()
                eg_opts[c, spt] = temp_dg
        eg_opt = np.max(eg_opts)

        if self._fd_jac:
            return eg_opt
        else:
            raise NotImplementedError("Analytic Jacobian for eg_opt unavailable.")

    def _pb_ei_opt_criterion(self, efforts):
        if self._optimization_package is "cvxpy":
            raise NotImplementedError("CVXPY unavailable for ei_opt.")

        self.efforts = efforts

        self.eval_pim()
        # ei_opts: average of the max_eigenval of the pvar matrix over candidates and sampling times
        ei_opts = np.empty((self.n_c, self.n_spt))
        for c, PVAR in enumerate(self.pvars):
            for spt, pvar in enumerate(PVAR):
                temp_dg = np.linalg.eigvals(pvar).max()
                ei_opts[c, spt] = temp_dg
        ei_opt = np.sum(ei_opts)

        if self._fd_jac:
            return ei_opt
        else:
            raise NotImplementedError("Analytic Jacobian for ei_opt unavailable.")

    """ private methods """

    def _plot_optimal_sensitivities(self, absolute=False, legend=None,
                                   markersize=10, colour_map="jet",
                                   write=False, dpi=720, quality=95, figsize=None):
        if not self._dynamic_system:
            raise SyntaxError("Sensitivity plots are only for dynamic systems.")

        if self.optimal_candidates is None:
            self.get_optimal_candidates()
        if self.n_opt_c is 0:
            print(
                f"[Warning]: empty optimal candidates, skipping plotting of optimal "
                f"predictions."
            )
            return
        if legend is None:
            if self.n_opt_c < 6:
                legend = True
        if figsize is None:
            figsize = (self.n_mp * 4.0, 1.0 + 2.5 * self.n_m_r)

        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=self.n_m_r,
            ncols=self.n_mp,
            sharex=True,
        )
        if self.n_m_r == 1:
            axes = np.array([axes])
        elif self.n_mp == 1:
            axes = np.array([axes]).T
        elif self.n_m_r == 1 and self.n_mp == 1:
            axes = np.array([[axes]])

        if self._pseudo_bayesian:
            mean_sens = np.nanmean(self._scr_sens, axis=0)
            std_sens = np.nanstd(self._scr_sens, axis=0)

        for row in range(self.n_m_r):
            for col in range(self.n_mp):
                cmap = cm.get_cmap(colour_map, len(self.optimal_candidates))
                colors = itertools.cycle(
                    cmap(_) for _ in np.linspace(0, 1, len(self.optimal_candidates))
                )
                for c, cand in enumerate(self.optimal_candidates):
                    opt_spt = self.sampling_times_candidates[cand[0]]
                    if self._pseudo_bayesian:
                        sens = mean_sens[
                                   cand[0],
                                   :,
                                   self.measurable_responses[row],
                                   col
                               ]
                        std = std_sens[
                                  cand[0],
                                  :,
                                  self.measurable_responses[row],
                                  col
                              ]
                    else:
                        sens = self.sensitivities[
                                   cand[0],
                                   :,
                                   self.measurable_responses[row],
                                   col
                               ]
                    color = next(colors)
                    if absolute:
                        sens = np.abs(sens)
                    ax = axes[row, col]
                    ax.plot(
                        opt_spt,
                        sens,
                        linestyle="--",
                        label=f"Candidate {cand[0] + 1:d}",
                        color=color
                    )
                    if not self._specified_n_spt:
                        if self._opt_sampling_times:
                            plot_sens = sens[cand[5]]
                        else:
                            plot_sens = sens[tuple(cand[5])]
                        ax.scatter(
                            cand[3],
                            plot_sens,
                            marker="o",
                            s=markersize * 50 * np.array(cand[4]),
                            color=color,
                            facecolors="none",
                        )
                    else:
                        markers = itertools.cycle(["o", "s", "h", "P"])
                        for i, (eff, spt, spt_idx) in enumerate(zip(cand[4], cand[3], cand[5])):
                            marker = next(markers)
                            ax.scatter(
                                spt,
                                sens[spt_idx],
                                marker=marker,
                                s=markersize * 50 * np.array(eff),
                                color=color,
                                label=f"Variant {i+1}",
                                facecolors="none",
                            )
                    if self._pseudo_bayesian:
                        ax.fill_between(
                            opt_spt,
                            sens + std,
                            sens - std,
                            facecolor=color,
                            alpha=0.1,
                        )
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

                    if row == self.n_m_r - 1:
                        if self.time_unit_name is not None:
                            ax.set_xlabel(f"Time ({self.time_unit_name})")
                        else:
                            ax.set_xlabel('Time')
                    if self.response_names is None or self.model_parameter_names is None:
                        pass
                    else:
                        ylabel = "$\partial$"
                        ylabel += self.response_names[self.measurable_responses[row]]
                        ylabel += "/$\partial$"
                        ylabel += self.model_parameter_names[col]
                        if self.response_unit_names is None or self.model_parameter_unit_names is None:
                            pass
                        else:
                            ylabel += f" [({self.response_unit_names[row]})/({self.model_parameter_unit_names[col]})]"
                        ax.set_ylabel(ylabel)
                        # ax.set_ylabel(
                        #     f"$\\partial {self.response_names[self.measurable_responses[row]]}"
                        #     f"/"
                        #     f"\\partial {self.model_parameter_names[col]}$"
                        # )
        if legend and len(self.optimal_candidates) > 1:
            axes[-1, -1].legend()

        fig.tight_layout()

        if write:
            self.create_result_dir()
            fn = f"sensitivity_plot" \
                 f"_{self.oed_result['optimality_criterion']}" \
                 f"_{self.run_no}.png"
            fp = self.result_dir + fn
            while path.isfile(fp):
                self.run_no += 1
                fn = f"sensitivity_plot" \
                     f"_{self.oed_result['optimality_criterion']}" \
                     f"_{self.run_no}.png"
                fp = self.result_dir + fn
            fig.savefig(fname=fp, dpi=dpi, quality=quality)
            self.run_no = 1

        return fig

    def _plot_optimal_sensitivities_interactive(self, figsize=None, markersize=10,
                                                colour_map="jet"):
        if not self._dynamic_system:
            raise SyntaxError("Sensitivity plots are only for dynamic systems.")

        if self.sensitivities is None:
            self.eval_sensitivities()
        if figsize is None:
            figsize = (18, 7)
        fig, axes = plt.subplots(
            figsize=figsize,
            nrows=2,
            ncols=3,
            gridspec_kw={
                "width_ratios": [2, 1, 1],
                "height_ratios": [2, 1],
            }
        )

        for axis_list in axes[:, 1:]:
            for ax in axis_list:
                ax.remove()

        gs = axes[0, 0].get_gridspec()
        res_rad_ax = fig.add_subplot(gs[:, 1])
        mp_rad_ax = fig.add_subplot(gs[:, 2])

        if self.time_unit_name is not None:
            axes[0, 0].set_xlabel(f"Time ({self.time_unit_name})")
        else:
            axes[0, 0].set_xlabel('Time')

        lines = []
        fill_lines = []
        cmap = plt.get_cmap(colour_map)
        colors = itertools.cycle(
            cmap(_)
            for _ in np.linspace(0, 1, len(self.optimal_candidates))
        )

        if self._pseudo_bayesian:
            mean_sens = np.nanmean(
                self._scr_sens,
                axis=0,
            )
            std_sens = np.nanstd(
                self._scr_sens,
                axis=0,
            )

        for opt_c in self.optimal_candidates:
            color = next(colors)
            label = f"Candidate {opt_c[0]+1}"
            if self._pseudo_bayesian:
                line, = axes[0, 0].plot(
                    self.sampling_times_candidates[opt_c[0]],
                    mean_sens[opt_c[0], :, 0, 0],
                    visible=True,
                    label=label,
                    marker="o",
                    markersize=markersize,
                    color=color,
                )
                fill_line = axes[0, 0].fill_between(
                    self.sampling_times_candidates[opt_c[0]],
                    mean_sens[opt_c[0], :, 0, 0] + std_sens[opt_c[0], :, 0, 0],
                    mean_sens[opt_c[0], :, 0, 0] - std_sens[opt_c[0], :, 0, 0],
                    facecolor=color,
                    alpha=0.1,
                    visible=True,
                )
            else:
                line, = axes[0, 0].plot(
                    self.sampling_times_candidates[opt_c[0]],
                    self.sensitivities[opt_c[0], :, 0, 0],
                    visible=True,
                    label=label,
                    marker="o",
                    markersize=markersize,
                    color=color,
                )
            lines.append(line)
            if self._pseudo_bayesian:
                fill_lines.append(fill_line)
            axes[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        labels = [str(line.get_label()) for line in lines]
        visibilities = [line.get_visible() for line in lines]
        cand_check = CheckButtons(
            axes[1, 0],
            labels=labels,
            actives=visibilities,
        )

        def _cand_check(label):
            index = labels.index(label)
            lines[index].set_visible(not lines[index].get_visible())
            if self._pseudo_bayesian:
                fill_lines[index].set_visible(not fill_lines[index].get_visible())
            plt.draw()

        cand_check.on_clicked(_cand_check)

        res_dict = {
            f"{res_name}": i
            for i, res_name in enumerate(self.response_names)
        }
        mp_dict = {
            f"{mp_name}": j
            for j, mp_name in enumerate(self.model_parameter_names)
        }

        res_rad = RadioButtons(
            res_rad_ax,
            labels=[
                f"{res_name}"
                for res_name in self.response_names
            ],
        )

        def _res_rad(label):
            res_idx = res_dict[label]
            mp_idx = mp_dict[mp_rad.value_selected]
            for i, (opt_c, line) in enumerate(zip(self.optimal_candidates, lines)):
                color = next(colors)
                if self._pseudo_bayesian:
                    sens_data = mean_sens[opt_c[0], :, res_idx, mp_idx]
                    fill_lines[i].remove()
                    fill_lines[i] = axes[0, 0].fill_between(
                        self.sampling_times_candidates[opt_c[0]],
                        sens_data + std_sens[opt_c[0], :, res_idx, mp_idx],
                        sens_data - std_sens[opt_c[0], :, res_idx, mp_idx],
                        facecolor=color,
                        alpha=0.1,
                    )
                else:
                    sens_data = self.sensitivities[opt_c[0], :, res_idx, mp_idx]
                line.set_ydata(sens_data)
            axes[0, 0].relim()
            axes[0, 0].autoscale()
            plt.draw()
        res_rad.on_clicked(_res_rad)

        mp_rad = RadioButtons(
            mp_rad_ax,
            labels=[
                f"{mp_name}"
                for mp_name in self.model_parameter_names
            ],
        )

        def _mp_rad(label):
            res_idx = res_dict[res_rad.value_selected]
            mp_idx = mp_dict[label]
            for i, (opt_c, line) in enumerate(zip(self.optimal_candidates, lines)):
                color = next(colors)
                if self._pseudo_bayesian:
                    sens_data = mean_sens[opt_c[0], :, res_idx, mp_idx]
                    fill_lines[i].remove()
                    fill_lines[i] = axes[0, 0].fill_between(
                        self.sampling_times_candidates[opt_c[0]],
                        sens_data + std_sens[opt_c[0], :, res_idx, mp_idx],
                        sens_data - std_sens[opt_c[0], :, res_idx, mp_idx],
                        facecolor=color,
                        alpha=0.1,
                    )
                else:
                    sens_data = self.sensitivities[opt_c[0], :, res_idx, mp_idx]
                line.set_ydata(sens_data)
            axes[0, 0].relim()
            axes[0, 0].autoscale()
            plt.draw()
        mp_rad.on_clicked(_mp_rad)

        fig.tight_layout()
        plt.show()
        return fig

    def _sensitivity_sim_wrapper(self, theta_try, store_responses=True):
        response = self._simulate_internal(self._ti_controls, self._tv_controls,
                                           theta_try, self._sampling_times)
        self.feval_sensitivity += 1
        """ store responses whenever required, and model parameters are the same as 
        current model's """
        if store_responses and np.allclose(theta_try, self._current_scr_mp):
            self._current_response = response
            self._store_current_response()
        return response

    def _check_if_scr_changed(self):
        if self._last_scr_mp is None:
            self._last_scr_mp = np.empty(1)
        if np.allclose(self._current_scr_mp, self._last_scr_mp):
            self._scr_changed = False
        else:
            self._scr_changed = True

    def _plot_current_efforts_2d(self, min_effort=1e-2, width=None, write=False, dpi=720,
                                 quality=95):
        if self._verbose >= 2:
            print("Plotting current continuous design.")

        if width is None:
            width = 0.7

        if self.efforts.ndim == 2:
            efforts = np.sum(self.efforts, axis=1)
        else:
            efforts = self.efforts
        p_plot = efforts[np.where(efforts > min_effort)]

        x = np.arange(1, self.n_c + 1, 1)[np.where(efforts > min_effort)].astype(str)
        fig = plt.figure(figsize=(15, 7))
        axes = fig.add_subplot(111)

        axes.bar(x, p_plot, width=width)

        axes.set_xticks(x)
        axes.set_xlabel("Candidate Number")

        axes.set_ylabel("Optimal Experimental Effort")
        if not self._discrete_design:
            axes.set_ylim([0, 1])
            axes.set_yticks(np.linspace(0, 1, 11))
        else:
            axes.set_ylim([0, self.efforts.max()])
            axes.set_yticks(
                np.linspace(0, self.efforts.max(), self.efforts.max().astype(int))
            )

        if write:
            self.create_result_dir()
            figname = 'fig_%s_design_%d.png' % (
                self.oed_result["optimality_criterion"], self.run_no)
            figfile = self.result_dir + figname
            while path.isfile(figfile):
                self.run_no += 1
                figname = 'fig_%s_design_%d.png' % (
                    self.oed_result["optimality_criterion"], self.run_no)
                figfile = self.result_dir + figname
            fig.savefig(fname=figfile, dpi=dpi, quality=quality)
            self.run_no = 1

        fig.tight_layout()
        return fig

    def _plot_current_efforts_3d(self, width=None, write=False, dpi=720,
                                 quality=95, tol=1e-2):
        if self._specified_n_spt:
            print(f"Warning, not implemented for specified n_spt.")
            return

        if self._verbose >= 2:
            print("Plotting current continuous design.")

        if width is None:
            width = 0.7

        p = self.efforts.reshape([self.n_c, self.n_spt])

        sampling_time_scale = np.nanmin(np.diff(self.sampling_times_candidates, axis=1))

        fig = plt.figure(figsize=(12, 8))
        axes = fig.add_subplot(111, projection='3d')
        opt_cand = np.unique(np.where(p > tol)[0], axis=0)
        for c, spt in enumerate(self.sampling_times_candidates[opt_cand]):
            x = np.array([c] * self.n_spt) - width / 2
            z = np.zeros(self.n_spt)

            dx = width
            dy = width * sampling_time_scale * width
            dz = p[opt_cand[c], :]

            x = x[~np.isnan(spt)]
            y = spt[~np.isnan(spt)]
            z = z[~np.isnan(spt)]
            dz = dz[~np.isnan(spt)]

            axes.bar3d(
                x=x,
                y=y,
                z=z,
                dx=dx,
                dy=dy,
                dz=dz
            )

        axes.grid(False)
        axes.set_xlabel('Candidate')
        xticks = opt_cand + 1
        axes.set_xticks(
            [c for c, _ in enumerate(self.sampling_times_candidates[opt_cand])])
        axes.set_xticklabels(labels=xticks)

        if self.time_unit_name is not None:
            axes.set_ylabel(f"Sampling Times ({self.time_unit_name})")
        else:
            axes.set_ylabel('Sampling Times')

        axes.set_zlabel('Experimental Effort')
        axes.set_zlim([0, 1])
        axes.set_zticks(np.linspace(0, 1, 6))

        fig.tight_layout()

        if write:
            self.create_result_dir()
            figname = 'fig_%s_design_%d.png' % (
                self.oed_result["optimality_criterion"], self.run_no)
            figfile = self.result_dir + figname
            while path.isfile(figfile):
                self.run_no += 1
                figname = 'fig_%s_design_%d.png' % (
                    self.oed_result["optimality_criterion"], self.run_no)
                figfile = self.result_dir + figname
            fig.savefig(fname=figfile, dpi=dpi, quality=quality)
            self.run_no = 1
        return fig

    def _pad_sampling_times(self):
        """ check the required number of sampling times """
        max_num_sampling_times = 1
        for sampling_times in self.sampling_times_candidates:
            num_sampling_times = len(sampling_times)
            if num_sampling_times > max_num_sampling_times:
                max_num_sampling_times = num_sampling_times

        for i, sampling_times in enumerate(self.sampling_times_candidates):
            num_sampling_times = len(sampling_times)
            if num_sampling_times < max_num_sampling_times:
                diff = max_num_sampling_times - num_sampling_times
                self.sampling_times_candidates[i] = np.pad(sampling_times,
                                                           pad_width=(0, diff),
                                                           mode='constant',
                                                           constant_values=np.nan)
        self.sampling_times_candidates = np.array(
            self.sampling_times_candidates.tolist())
        return self.sampling_times_candidates

    def _pad_sensitivities(self):
        """ padding sensitivities to accommodate for missing sampling times """
        for i, row in enumerate(self.sensitivities):
            if row.ndim < 3:  # check if row has less than 3 dim
                if self.n_mp == 1:  # potential cause 1: we only have 1 mp
                    row = np.expand_dims(row, -1)  # add last dimension
                if self.n_r == 1:  # potential cause 2: we only have 1 response
                    row = np.expand_dims(row, -2)  # add second to last
            if row.ndim != 3:  # check again if already 3 dims
                # only reason: we only have 1 spt, add dim to first position
                row = np.expand_dims(row, 0)
            # pad sampling times
            diff = self.n_spt - row.shape[0]
            self.sensitivities[i] = np.pad(row,
                                           pad_width=[(0, diff), (0, 0), (0, 0)],
                                           mode='constant', constant_values=np.nan)
        self.sensitivities = self.sensitivities.tolist()
        self.sensitivities = np.asarray(self.sensitivities)
        return self.sensitivities

    def _store_current_response(self):
        """ padding responses to accommodate for missing sampling times """
        start = time()
        if self.response is None:  # if it is the first response to be stored,
            # initialize response list
            self.response = []

        if self._dynamic_system and self.n_spt is 1:
            self._current_response = self._current_response[np.newaxis]
        if self.n_r is 1:
            self._current_response = self._current_response[:, np.newaxis]

        if self._var_n_sampling_time:
            self._current_response = np.pad(
                self._current_response,
                pad_width=((0, self.n_spt - self._current_response.shape[0]), (0, 0)),
                mode='constant',
                constant_values=np.nan
            )

        """ convert to list if np array """
        if isinstance(self.response, np.ndarray):
            self.response = self.response.tolist()
        self.response.append(self._current_response)

        """ convert to numpy array """
        self.response = np.array(self.response)
        end = time()
        if self._verbose >= 3:
            print('Storing response took %.6f CPU ms.' % (1000 * (end - start)))
        return self.response

    def _residuals_wrapper_f(self, model_parameters):
        if self.responses_scales is None:
            if self._dynamic_system:
                self.responses_scales = np.nanmean(self.data, axis=(0, 1))
            else:
                self.responses_scales = np.nanmean(self.data, axis=0)

        self.eval_residuals(model_parameters)
        if self._dynamic_system:
            res = self.residuals / self.responses_scales[None, None, :]
        else:
            res = self.residuals / self.responses_scales[None, :]
        if self._dynamic_system:
            res = res.reshape(self.n_c * self.n_spt, self.n_m_r)
        else:
            res = res.reshape(self.n_c, self.n_m_r)
        return res[~np.isnan(res)]

    def _residuals_wrapper_f_old(self, model_parameters):
        if self.responses_scales is None:
            if self._dynamic_system:
                self.responses_scales = np.nanmean(self.data, axis=(0, 1))
            else:
                self.responses_scales = np.nanmean(self.data, axis=0)
        self.eval_residuals(model_parameters)
        res = self.residuals / self.responses_scales[None, None, :]
        res = res[~np.isnan(res)]
        return np.squeeze(res[None, :] @ res[:, None])

    def _simulate_internal(self, ti_controls, tv_controls, theta, sampling_times):
        raise SyntaxError(
            "Make sure you have initialized the designer, and specified the simulate "
            "function correctly."
        )

    def _initialize_internal_simulate_function(self):
        if self._model_package == 'pyomo':
            if self._simulate_sig_id is 1:
                self._simulate_internal = lambda tic, tvc, mp, spt: \
                  self.simulate(self.model, self.simulator, tic, mp)
            if self._simulate_sig_id is 2:
                self._simulate_internal = lambda tic, tvc, mp, spt: \
                    self.simulate(self.model, self.simulator, tic, spt, mp)
            if self._simulate_sig_id is 3:
                self._simulate_internal = lambda tic, tvc, mp, spt: \
                    self.simulate(self.model, self.simulator, tvc, spt, mp)
            if self._simulate_sig_id is 4:
                self._simulate_internal = lambda tic, tvc, mp, spt: \
                    self.simulate(self.model, self.simulator, tic, tvc, spt, mp)
            if self._simulate_sig_id is 5:
                self._simulate_internal = lambda tic, tvc, mp, spt: \
                    self.simulate(self.model, self.simulator, spt, mp)
        elif self._model_package == 'non-pyomo':
            if self._simulate_sig_id is 1:
                self._simulate_internal = lambda tic, tvc, mp, spt: \
                    self.simulate(tic, mp)
            if self._simulate_sig_id is 2:
                self._simulate_internal = lambda tic, tvc, mp, spt: \
                    self.simulate(tic, spt, mp)
            if self._simulate_sig_id is 3:
                self._simulate_internal = lambda tic, tvc, mp, spt: \
                    self.simulate(tvc, spt, mp)
            if self._simulate_sig_id is 4:
                self._simulate_internal = lambda tic, tvc, mp, spt: \
                    self.simulate(tic, tvc, spt, mp)
            if self._simulate_sig_id is 5:
                self._simulate_internal = lambda tic, tvc, mp, spt: \
                    self.simulate(spt, mp)
        else:
            raise SyntaxError(
                'Cannot initialize simulate function properly, check your syntax.'
            )

    def _transform_efforts(self):
        if self._unconstrained_form:
            if not self._efforts_transformed:
                self.efforts = np.square(self.efforts)
                self.efforts /= np.sum(self.efforts)
                self._efforts_transformed = True
                if self._verbose >= 3:
                    print("Efforts transformed.")

        return self.efforts

    def _check_missing_components(self):
        # basic components
        if self.model_parameters is None:
            raise SyntaxError("Please specify nominal model parameters.")

        # invariant controls
        if self._invariant_controls and self.ti_controls_candidates is None:
            raise SyntaxError(
                "Simulate function suggests invariant controls are needed, but "
                "ti_controls_candidates is empty."
            )
        if not self._invariant_controls:
            self.ti_controls_candidates = np.empty(1)

        # dynamic system
        if self._dynamic_system:
            if self.sampling_times_candidates is None:
                raise SyntaxError(
                    "Simulate function suggests dynamic system, but "
                    "sampling_times_candidates is empty."
                )
            if self._dynamic_controls:
                if self.tv_controls_candidates is None:
                    raise SyntaxError(
                        "Simulate function suggests time-varying controls are needed, "
                        "but tv_controls_candidates is empty."
                    )
            else:
                self.tv_controls_candidates = np.empty_like(self.ti_controls_candidates)
        else:
            self.sampling_times_candidates = np.empty_like(self.ti_controls_candidates)
            self.tv_controls_candidates = np.empty_like(self.ti_controls_candidates)

    def _data_type_check(self):
        self.model_parameters = np.asarray(self.model_parameters)
        self.ti_controls_candidates = np.asarray(self.ti_controls_candidates)
        self.tv_controls_candidates = np.asarray(self.tv_controls_candidates)
        self.sampling_times_candidates = np.asarray(self.sampling_times_candidates)
        if not isinstance(self.model_parameters, (list, np.ndarray)):
            raise SyntaxError('model_parameters must be supplied as a numpy array.')
        if self._invariant_controls:
            if not isinstance(self.ti_controls_candidates, np.ndarray):
                raise SyntaxError(
                    'ti_controls_candidates must be supplied as a numpy array.'
                )
        if self._dynamic_system:
            if not isinstance(self.sampling_times_candidates, np.ndarray):
                raise SyntaxError("sampling_times_candidates must be supplied as a "
                                  "numpy array.")
            if self._dynamic_controls:
                if not isinstance(self.tv_controls_candidates, np.ndarray):
                    raise SyntaxError("tv_controls_candidates must be supplied as a "
                                      "numpy array.")

    def _handle_simulate_sig(self):
        """
        Determines type of model from simulate signature. Five supported types:
        =================================================================================
        1. simulate(ti_controls, model_parameters).
        2. simulate(ti_controls, sampling_times, model_parameters).
        3. simulate(tv_controls, sampling_times, model_parameters).
        4. simulate(ti_controls, tv_controls, sampling_times, model_parameters).
        5. simulate(sampling_times, model_parameters).
        =================================================================================
        If a pyomo.dae model is specified a special signature is recommended that adds
        two input arguments to the beginning of the simulate signatures e.g., for type 3:
        simulate(model, simulator, tv_controls, sampling_times, model_parameters).
        """
        sim_sig = list(signature(self.simulate).parameters.keys())
        unspecified_sig = ["unspecified"]
        if np.all([entry in sim_sig for entry in unspecified_sig]):
            raise SyntaxError("Don't forget to specify the simulate function.")

        t1_sig = ["ti_controls"]
        t2_sig = ["ti_controls", "sampling_times"]
        t3_sig = ["tv_controls", "sampling_times"]
        t4_sig = ["ti_controls", "tv_controls", "sampling_times"]
        t5_sig = ["sampling_times"]
        pyomo_sig = ["model", "simulator"]
        # check if pyomo.dae model is used
        if np.all([entry in sim_sig for entry in pyomo_sig]):
            self._model_package = "pyomo"
        else:
            self._model_package = "non-pyomo"
        # initialize simulate id
        self._simulate_sig_id = 0
        # check if model_parameters is present
        if "model_parameters" not in sim_sig:
            raise SyntaxError(
                f"The input argument \"model_parameters\" is not found in the simulate "
                f"function, please fix simulate signature."
            )
        if np.all([entry in sim_sig for entry in t4_sig]):
            self._simulate_sig_id = 4
            self._dynamic_system = True
            self._dynamic_controls = True
            self._invariant_controls = True
        elif np.all([entry in sim_sig for entry in t3_sig]):
            self._simulate_sig_id = 3
            self._dynamic_system = True
            self._dynamic_controls = True
            self._invariant_controls = False
        elif np.all([entry in sim_sig for entry in t2_sig]):
            self._simulate_sig_id = 2
            self._dynamic_system = True
            self._dynamic_controls = False
            self._invariant_controls = True
        elif np.all([entry in sim_sig for entry in t1_sig]):
            self._simulate_sig_id = 1
            self._dynamic_system = False
            self._dynamic_controls = False
            self._invariant_controls = True
        elif np.all([entry in sim_sig for entry in t5_sig]):
            self._simulate_sig_id = 5
            self._dynamic_system = True
            self._dynamic_controls = False
            self._invariant_controls = False
        if self._simulate_sig_id is 0:
            raise SyntaxError(
                "Unrecognized simulate function signature, please check if you have "
                "specified it correctly. The base signature requires "
                "'model_parameters'. Adding 'sampling_times' makes it dynamic,"
                "adding 'tv_controls' and 'sampling_times' makes a dynamic system with"
                " time-varying controls. Adding 'tv_controls' without 'sampling_times' "
                "does not work. Adding 'model' and 'simulator' makes it a pyomo "
                "simulate signature. 'ti_controls' are optional in all cases."
            )
        self._initialize_internal_simulate_function()

    def _check_stats_framework(self):
        """ check if local or Pseudo-bayesian designs """
        if self.model_parameters.ndim is 1:
            self._pseudo_bayesian = False
        elif self.model_parameters.ndim is 2:
            self._pseudo_bayesian = True
        else:
            raise SyntaxError(
                "model_parameters must be fed in as a 1D numpy array for local "
                "designs, and a 2D numpy array for Pseudo-bayesian designs."
            )

    def _check_candidate_lengths(self):
        # assign 1 candidate and terminate if no controls
        if not self._dynamic_controls and not self._invariant_controls:
            self.n_c = 1
            return
        if self._dynamic_system:
            if not self.n_c_tic == self.n_c_spt:
                raise SyntaxError(
                    "Number of candidates given in ti_controls_candidates, and "
                    "sampling_times_candidates are inconsistent."
                )
            if self._dynamic_controls:
                if not self.n_c_tic == self.n_c_tvc:
                    raise SyntaxError(
                        "Number of candidates in supplied ti_controls_candidates, and "
                        "tv_controls_candidates are inconsistent."
                    )
        self.n_c = self.n_c_tic

    def _check_var_spt(self):
        if np.all([len(spt) == len(self.sampling_times_candidates[0]) for spt in
                   self.sampling_times_candidates]) \
                and np.all(~np.isnan(self.sampling_times_candidates)):
            self._var_n_sampling_time = False
        else:
            self._var_n_sampling_time = True
            self._pad_sampling_times()

    def _get_component_sizes(self):
        # number of candidates from tic, number of tic
        if self._invariant_controls:
            self.n_c_tic, self.n_tic = self.ti_controls_candidates.shape
        # number of tvc
        if self._dynamic_controls:
            self.n_c_tvc, self.n_tvc = self.tv_controls_candidates.shape

        # number of model parameters, and scenarios (if pseudo_bayesian)
        if self._pseudo_bayesian:
            self.n_scr, self.n_mp = self.model_parameters.shape
            self._current_scr_mp = self.model_parameters[0]
        else:
            self.n_mp = self.model_parameters.shape[0]
            self._current_scr_mp = self.model_parameters

        # number of sampling times (if dynamic)
        if self._dynamic_system:
            self.n_c_spt, self.n_spt = self.sampling_times_candidates.shape
        else:
            self.n_c_spt = self.n_c_tic
            self.n_spt = 1

        # number of responses
        if self.n_r is None:
            if self._verbose >= 3:
                print(
                    "Running one simulation for initialization "
                    "(required to determine number of responses)."
                )
            y = self._simulate_internal(
                self.ti_controls_candidates[0],
                self.tv_controls_candidates[0],
                self._current_scr_mp,
                self.sampling_times_candidates[0][~np.isnan(self.sampling_times_candidates[0])]
            )
            try:
                self.n_spt_r, self.n_r = y.shape
            except ValueError:  # output not two dimensional
                # case 1: n_r is 1
                if self._dynamic_system and self.n_spt > 1:
                    self.n_r = 1
                # case 2: n_spt is 1
                else:
                    self.n_r = y.shape[0]

        # number of measurable responses (if not all)
        if self.measurable_responses is None:
            self.n_m_r = self.n_r
            self.measurable_responses = np.array([_ for _ in range(self.n_r)])
        elif self.n_m_r != len(self.measurable_responses):
            self.n_m_r = len(self.measurable_responses)
            if self.n_m_r > self.n_r:
                raise SyntaxError(
                    "Given number of measurable responses is greater than number of "
                    "responses given."
                )

    def _check_memory_req(self, threshold):
        # check problem size (affects if designer will be memory-efficient or quick)
        self._memory_threshold = threshold
        memory_req = self.n_c * self.n_spt * self.n_m_r * self.n_mp * 8

        if memory_req > self._memory_threshold:
            print(
                f'Sensitivity matrix will take {memory_req / 1e9:.2f} GB of memory space '
                f'(more than {self._memory_threshold / 1e9:.2f} GB threshold).'
            )
            self._large_memory_requirement = True

    def _check_if_model_parameters_changed(self):
        if self._old_model_parameters is None:
            self._model_parameters_changed = True
            return self._model_parameters_changed

        if np.allclose(self._old_model_parameters, self.model_parameters):
            self._model_parameters_changed = False
        else:
            self._model_parameters_changed = True
        return self._model_parameters_changed

    def _check_if_candidates_changed(self):
        if self._old_tic is None:
            self._tic_changed = True
        elif np.all(np.array_equal(self._old_tic, self.ti_controls_candidates)):
            self._tic_changed = False
        else:
            self._tic_changed = True
        if self._dynamic_system:
            if self._old_spt is None:
                self._spt_changed = True
            elif np.array_equal(self._old_spt, self.sampling_times_candidates):
                self._spt_changed = False
            if self._dynamic_controls:
                if self._old_tvc is None:
                    self._tvc_changed = True
                elif np.array_equal(self._old_tvc, self.tv_controls_candidates):
                    self._tvc_changed = False
        if np.any([self._tic_changed, self._spt_changed, self._tvc_changed]):
            self._candidates_changed = True
        else:
            self._candidates_changed = False
        return self._candidates_changed

    def _initialize_names(self):
        if self.response_names is None:
            self.response_names = np.array([
                f"Response {_}"
                for _ in range(self.n_m_r)
            ])
        if self.model_parameter_names is None:
            self.model_parameter_names = np.array([
                f"Model Parameter {_}"
                for _ in range(self.n_mp)
            ])
        if self.candidate_names is None:
            self.candidate_names = np.array([
                f"Candidate {_}"
                for _ in range(self.n_c)
            ])
        if self.ti_controls_names is None and self._invariant_controls:
            self.ti_controls_names = np.array([
                f"Time-invariant Control {_}"
                for _ in range(self.n_tic)
            ])
        if self.tv_controls_names is None and self._dynamic_controls:
            self.tv_controls_names = np.array([
                f"Time-varying Control {_}"
                for _ in range(self.n_tvc)
            ])
