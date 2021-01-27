from datetime import datetime
from inspect import signature
from os import getcwd, path, makedirs
from pickle import dump, load
from string import Template
from time import time
import itertools
import __main__ as main
import dill
import sys

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.widgets import RadioButtons, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, least_squares
from pydex.utils.trellis_plotter import TrellisPlotter
from pydex.core.bnb.tree import Tree
from pydex.core.bnb.node import Node
from pydex.core.logger import Logger
import matplotlib
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
        """
        Pydex' main class to instantiate an experimental designer. The designer
        serves as the main user-interface to use Pydex to solve experimental design
        problems.

        All details on the experimental design problem is passed to the designer, which
        Pydex then compiles into an optimization problem passed to the optimization
        package it supports to be solved by a numerical optimizer.

        The designer comes with various built-in plotting capabilities through
        matplotlib's plotting features.
        """

        """ Experimental """
        self._alt_cvar = None
        self.error_cov = None
        self.error_fim = None

        """ CVaR-exclusive """
        self.n_cvar_scr = None
        self.cvar_optimal_candidates = None
        self.cvar_solution_times = None
        self._biobjective_values = None
        self._constrained_cvar = None
        self.beta = None
        self._cvar_problem = None

        """ pseudo-Bayesian exclusive """
        self.pb_atomic_fims = None
        self._scr_sens = None
        self.scr_responses = None
        self._current_scr = None
        self._pseudo_bayesian_type = None
        self.scr_fims = None
        self.scr_criterion_val = None
        self._current_scr_mp = None

        """ Logging """
        # options
        self.sens_report_freq = 10
        self._memory_threshold = None  # threshold for large problems in bytes, default: 1 GB
        # store designer status and its verbal level after initialization
        self._status = 'empty'
        self._verbose = 0
        self._sensitivity_analysis_done = False

        """ The current optimal experimental design """
        self.opt_eff = None
        self.opt_tic = None
        self.n_opt_c = None
        self.mp_covar = None

        # exclusive to discrete designs
        self.spt_binary = None

        # exclusive to dynamic systems
        self.opt_tvc = None
        self.opt_spt = None
        self.opt_spt_combs = None
        self.spt_candidates_combs = None

        # experimental
        self.cost = None
        self.cand_cost = None
        self.spt_cost = None
        self._norm_sens_by_params = False

        """" Type of Problem """
        self._invariant_controls = None
        self._specified_n_spt = None
        self._discrete_design = None
        self._pseudo_bayesian = False
        self._large_memory_requirement = False
        self._current_criterion = None
        self._efforts_transformed = False
        self._unconstrained_form = False
        self.normalized_sensitivity = None
        self._dynamic_controls = False
        self._dynamic_system = False

        """ Attributes to determine if re-computation of atomics is necessary """
        self._candidates_changed = None
        self._model_parameters_changed = None
        self._compute_atomics = False
        self._compute_sensitivities = False

        """ Core user-defined Variables """
        self._tvcc = None
        self._ticc = None
        self._sptc = None
        self._model_parameters = None
        self._simulate_signature = 0

        # optional user inputs
        self.measurable_responses = None  # subset of measurable states

        """ Labelling """
        self.candidate_names = None  # plotting names
        self.measurable_responses_names = None
        self.ti_controls_names = None
        self.tv_controls_names = None
        self.model_parameters_names = None
        self.model_parameter_unit_names = None
        self.response_unit_names = None
        self.time_unit_name = None
        self.model_parameter_names = None
        self.response_names = None
        self.use_finite_difference = True
        self.do_sensitivity_analysis = False

        """ Core designer outputs """
        self.response = None
        self.sensitivities = None
        self.optimal_candidates = None
        self.atomic_fims = None
        self.apportionments = None
        self.epsilon = None

        # exclusive to prediction-oriented criteria
        self.pvars = None

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
        self._n_spt_spec = None
        self.max_n_opt_spt = None
        self.n_min_sups = None

        """ parameter estimation """
        self.data = None  # stored data, a 3D numpy array, same shape as response.
        # Whenever data is missing, use np.nan to fill the array.
        self.residuals = None  # stored residuals, 3D numpy array with the same shape
        # as data and response. Will skip entries whenever data is empty.

        """ performance-related """
        self.feval_simulation = None
        self.feval_sensitivity = None
        self._fim_eval_time = None
        # temporary for current design
        self._sensitivity_analysis_time = 0
        self._optimization_time = 0

        """ parameter estimability """
        self.estimable_columns = None
        self.responses_scales = None
        self.estimability = None
        self.estimable_model_parameters = []

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

        """ [Private]: current candidate within eval_sensitivities() """
        self._current_tic = None
        self._current_tvc = None
        self._current_spt = None
        self._current_res = None

        """ User-specified Behaviour """
        # problem types
        self._sensitivity_is_normalized = None
        self._opt_sampling_times = False
        self._var_n_sampling_time = None
        # numerical options
        self._regularize_fim = None
        self._num_steps = 5
        self._eps = 1e-5
        self._trim_fim = False
        self._fd_jac = True

        # store chosen package to interface with the optimizer, and the chosen optimizer
        self._optimization_package = None
        self._optimizer = None

        # store current criterion value
        self._criterion_value = None

        """ user saving options """
        self._save_sensitivities = False
        self._save_atomics = False

    @property
    def model_parameters(self):
        return self._model_parameters

    @model_parameters.setter
    def model_parameters(self, mp):
        self._model_parameters_changed = True
        self._model_parameters = mp

    @property
    def ti_controls_candidates(self):
        return self._ticc

    @ti_controls_candidates.setter
    def ti_controls_candidates(self, ticc):
        self._candidates_changed = True
        self._ticc = ticc

    @property
    def tv_controls_candidates(self):
        return self._tvcc

    @tv_controls_candidates.setter
    def tv_controls_candidates(self, tvcc):
        self._candidates_changed = True
        self._tvcc = tvcc

    @property
    def sampling_times_candidates(self):
        return self._sptc

    @sampling_times_candidates.setter
    def sampling_times_candidates(self, sptc):
        self._candidates_changed = True
        self._sptc = sptc

    """ user-defined methods: must be overwritten by user to work """
    def simulate(self, unspecified):
        raise SyntaxError("Don't forget to specify the simulate function.")

    """ core activity interfaces """
    def initialize(self, verbose=0, memory_threshold=int(1e9)):
        """ check for syntax errors, runs one simulation to determine n_r """

        """ check if simulate function has been specified """
        self._data_type_check()
        self._check_stats_framework()
        self._handle_simulate_sig()
        self._get_component_sizes()
        self._check_candidate_lengths()
        self._check_missing_components()

        if self._dynamic_system:
            self._check_var_spt()

        self._initialize_names()

        self._check_memory_req(memory_threshold)

        if self.error_cov is None:
            self.error_cov = np.eye(self.n_m_r)
        try:
            self.error_fim = np.linalg.inv(self.error_cov)
        except np.linalg.LinAlgError:
            raise SyntaxError(
                "The provided error covariance is singular, please make sure you "
                "have passed in the correct error covariance."
            )

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
            self._current_tic = exp[0]
            self._current_tvc = exp[1]
            self._current_spt = exp[2][~np.isnan(exp[2])]
            if not self._current_spt.size > 0:
                raise SyntaxError(
                    'One candidate has an empty list of sampling times, please check '
                    'the specified experimental candidates.'
                )

            """ determine if simulation needs to be re-run: if data on time-invariant 
            control variables is missing, will not run """
            cond_1 = np.any(np.isnan(exp[0]))
            if np.any([cond_1]):
                self._current_res = np.nan
            else:
                start = time()
                response = self._simulate_internal(self._current_tic, self._current_tvc,
                                                   self.model_parameters,
                                                   self._current_spt)
                finish = time()
                self.feval_simulation += 1
                self._current_res = response
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
            self._current_tic = exp[1]
            self._current_tvc = exp[2]
            self._current_spt = exp[3][~np.isnan(exp[3])]
            if self._current_spt.size <= 0:
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
                self._current_res = np.nan
            else:
                start = time()
                response = self._simulate_internal(self._current_tic, self._current_tvc,
                                                   self.model_parameters,
                                                   self._current_spt)
                finish = time()
                self.feval_simulation += 1
                self._current_res = response
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

    def solve_cvar_problem(self, criterion, beta, n_spt=None, n_exp=None,
                           optimize_sampling_times=False, package="cvxpy",
                           optimizer=None, opt_options=None, e0=None, write=False,
                           save_sensitivities=False, fd_jac=True,
                           unconstrained_form=False, trim_fim=False,
                           pseudo_bayesian_type=None, regularize_fim=False,
                           reso=5, plot=False, n_bins=20, tol=1e-4, dpi=360,
                           **kwargs):
        self._current_criterion = criterion.__name__

        if "cvar" not in self._current_criterion:
            raise SyntaxError(
                "Please pass in a valid cvar criterion e.g., cvar_d_opt_criterion."
            )

        # computing number of parameter scenarios that will be considered in CVaR
        self.beta = beta
        self.n_cvar_scr = (1 - self.beta) * self.n_scr
        if self.n_cvar_scr < 1:
            print(
                "[WARNING]: "
                "given n_scr * beta given is smaller than 1, this yields a maximin "
                "design. Please provide a larger number of n_scr if a CVaR design "
                "was desired."
            )
            self.n_cvar_scr = np.ceil(self.n_cvar_scr).astype(int)
        else:
            self.n_cvar_scr = np.floor(self.n_cvar_scr).astype(int)

        # check if given reso is less than 3
        if reso < 3:
            print(
                f"The input reso is given as {reso}; the minimum value of reso is 3. "
                "Continuing with reso = 3."
            )
            reso = 3

        # initializing result lists
        self.cvar_optimal_candidates = []
        self.cvar_solution_times = []
        self._biobjective_values = np.empty((reso, 2))
        if plot:
            figs = []

            def add_fig(cdf, pdf):
                figs.append([cdf, pdf])

        """ Iteration 1: Maximal (Type 1) Mean Design """
        if self._verbose >= 1:
            print(f" CVaR Problem ".center(100, "*"))
            print(f"")
            print(f"[Iteration 1/{reso}]".center(100, "="))
            print(f"Computing the maximal mean design, obtaining the mean UB and CVaR LB"
                  f" in the Pareto Frontier.")
            print(f"")
        self.design_experiment(
            criterion,
            n_spt=n_spt,
            n_exp=n_exp,
            optimize_sampling_times=optimize_sampling_times,
            package=package,
            optimizer=optimizer,
            opt_options=opt_options,
            e0=e0,
            write=False,
            save_sensitivities=save_sensitivities,
            fd_jac=fd_jac,
            unconstrained_form=unconstrained_form,
            trim_fim=trim_fim,
            pseudo_bayesian_type=pseudo_bayesian_type,
            regularize_fim=regularize_fim,
            beta=0.00,
            **kwargs,
        )
        self.beta = beta
        self.get_optimal_candidates()
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol)
        iter_1_efforts = np.copy(self.efforts) / np.sum(self.efforts)
        mean_ub = self._criterion_value
        iter_1_phi = np.copy(self.phi.value)
        if self._verbose >= 1:
            print("")
            print("Computing CVaR of Iteration 1's Solution")

        # computing CVaR of Maximal Type 1 Mean Design
        self.design_experiment(
            criterion,
            n_spt=n_spt,
            n_exp=n_exp,
            optimize_sampling_times=optimize_sampling_times,
            package=package,
            optimizer=optimizer,
            opt_options=opt_options,
            e0=e0,
            write=False,
            save_sensitivities=False,
            fd_jac=fd_jac,
            unconstrained_form=unconstrained_form,
            trim_fim=trim_fim,
            pseudo_bayesian_type=pseudo_bayesian_type,
            regularize_fim=regularize_fim,
            beta=self.beta,
            fix_effort=iter_1_efforts,
            **kwargs,
        )
        cvar_lb = self._criterion_value
        if self._verbose >= 2:
            print(
                    f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds."
                )

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[0, :] = np.array([mean_ub, cvar_lb])
        if self._verbose >= 1:
            print(f"CVaR LB: {cvar_lb}")
            print(f"Mean UB: {mean_ub}")
            print(f"[Iteration 1/{reso} Completed]".center(100, "="))
            print(f"")
        if plot:
            self.phi.value = iter_1_phi
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=1),
                self.plot_criterion_pdf(write=False, iteration=1),
            )

        """ Iteration 2: Maximal CVaR_beta Design """
        if self._verbose >= 1:
            print(f"[Iteration 2/{reso}]".center(100, "="))
            print(f"Computing the maximal CVaR design, obtaining the CVaR UB, and mean "
                  f"LB in the Pareto Frontier.")
            print(f"")
        self.design_experiment(
            criterion,
            n_spt=n_spt,
            n_exp=n_exp,
            optimize_sampling_times=optimize_sampling_times,
            package=package,
            optimizer=optimizer,
            opt_options=opt_options,
            e0=e0,
            write=False,
            save_sensitivities=False,
            fd_jac=fd_jac,
            unconstrained_form=unconstrained_form,
            trim_fim=trim_fim,
            pseudo_bayesian_type=pseudo_bayesian_type,
            regularize_fim=regularize_fim,
            beta=self.beta,
            **kwargs,
        )
        iter2_s = np.copy(self.s.value)
        self.get_optimal_candidates()
        iter_2_efforts = np.copy(self.efforts) / np.sum(self.efforts)
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol)
        iter2_var = self.v.value
        cvar_ub = self._criterion_value

        if self._verbose >= 1:
            print("")
            print("Computing Mean of Iteration 2's Solution")

        self.design_experiment(
            criterion,
            n_spt=n_spt,
            n_exp=n_exp,
            optimize_sampling_times=optimize_sampling_times,
            package=package,
            optimizer=optimizer,
            opt_options=opt_options,
            e0=e0,
            write=False,
            save_sensitivities=False,
            fd_jac=fd_jac,
            unconstrained_form=unconstrained_form,
            trim_fim=trim_fim,
            pseudo_bayesian_type=pseudo_bayesian_type,
            regularize_fim=regularize_fim,
            beta=0.00,
            fix_effort=iter_2_efforts,
            **kwargs,
        )
        self.beta = beta
        mean_lb = self._criterion_value
        if self._verbose >= 2:
            print(
                    f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds."
                )

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[1, :] = np.array([mean_lb, cvar_ub])
        if self._verbose >= 1:
            print(f"CVaR UB: {cvar_ub}")
            print(f"MEAN LB: {mean_lb}")
            print(f"[Iteration 2/{reso} Completed]".center(100, "="))
            print(f"")
        if plot:
            self.v.value = iter2_var
            self.s.value = iter2_s
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=2),
                self.plot_criterion_pdf(write=False, iteration=2),
            )

        """ Iterations 3+: Intermediate Points """
        mean_values = np.linspace(mean_lb, mean_ub, reso)
        mean_values = mean_values[:-1]
        mean_values = mean_values[1:]

        for i, mean in enumerate(mean_values):
            if self._verbose >= 1:
                print(f"[Iteration {i + 3}/{reso}]".center(100, "="))
            self.design_experiment(
                criterion,
                n_spt=n_spt,
                n_exp=n_exp,
                optimize_sampling_times=optimize_sampling_times,
                package=package,
                optimizer=optimizer,
                opt_options=opt_options,
                e0=e0,
                write=False,
                save_sensitivities=False,
                fd_jac=fd_jac,
                unconstrained_form=unconstrained_form,
                trim_fim=trim_fim,
                pseudo_bayesian_type=pseudo_bayesian_type,
                regularize_fim=regularize_fim,
                beta=self.beta,
                min_expected_value=mean,
                **kwargs,
            )
            self.get_optimal_candidates()
            self.cvar_optimal_candidates.append(self.optimal_candidates)
            self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
            self._biobjective_values[i + 2, :] = np.array([mean, self._criterion_value])

            if plot:
                add_fig(
                    self.plot_criterion_cdf(write=False, iteration=i+3),
                    self.plot_criterion_pdf(write=False, iteration=i+3),
                )
            if self._verbose >= 1:
                self.print_optimal_candidates(tol=tol)
                print(f"CVaR: {self._criterion_value}")
                print(f"MEAN: {cp.sum(self.phi).value / self.n_scr}")
                print(f"[Iteration {i + 3}/{reso} Completed]".center(100, "="))
                print(f"")

        # use the same axes.xlim for all plotted cdfs and pdfs
        if plot:
            xlims = []
            for i, fig in enumerate(figs):
                cdf, pdf = fig[0], fig[1]
                xlims.append(cdf.axes[0].get_xlim())
            xlims = np.asarray(xlims)
            for i, fig in enumerate(figs):
                cdf, pdf = fig[0], fig[1]
                cdf.axes[0].set_xlim(xlims[:, 0].min(), xlims[:, 1].max())
                pdf.axes[0].set_xlim(xlims[:, 0].min(), xlims[:, 1].max())
                cdf.tight_layout()
                pdf.tight_layout()
                if write:
                    fn_cdf = f"iter_{i + 1}_cdf_{self.beta}_beta_{self.n_scr}_scr"
                    fp_cdf = self._generate_result_path(fn_cdf, "png")
                    fn_pdf = f"iter_{i + 1}_pdf_{self.beta}_beta_{self.n_scr}_scr"
                    fp_pdf = self._generate_result_path(fn_pdf, "png")
                    cdf.savefig(fp_cdf, dpi=dpi)
                    pdf.savefig(fp_pdf, dpi=dpi)

    def _formulate_cvar_problem(self, criterion, beta, p_cons, min_expected_value=None):
        self.v = cp.Variable()
        self.s = cp.Variable((self.n_scr,), nonneg=True)
        self.phi = cp.Variable((self.n_scr,))
        self.phi_mean = cp.Variable()

        self.eval_fim(self.efforts)

        for scr, (s_q, phi_q, mp, fim) in enumerate(zip(self.s, self.phi, self.model_parameters, self.scr_fims)):
            p_cons += [phi_q <= -criterion(fim)]
            p_cons += [s_q >= self.v - phi_q]
        if min_expected_value is not None:
            self._constrained_cvar = True
            p_cons += [cp.sum(self.phi) / self.n_scr >= min_expected_value]
        else:
            self._constrained_cvar = False
        obj = cp.Maximize(self.v - 1 / (self.n_scr * (1 - beta)) * cp.sum(self.s))
        return obj

    def solve_cvar_problem_alt(self, criterion, beta, n_spt=None, n_exp=None,
                           optimize_sampling_times=False, package="cvxpy",
                           optimizer=None, opt_options=None, e0=None, write=True,
                           save_sensitivities=False, fd_jac=True,
                           unconstrained_form=False, trim_fim=False,
                           pseudo_bayesian_type=None, regularize_fim=False,
                           reso=5, plot=False, n_bins=20, tol=1e-4, **kwargs):
        self._current_criterion = criterion.__name__

        if "cvar" not in self._current_criterion:
            raise SyntaxError(
                "Please pass in a valid cvar criterion e.g., cvar_d_opt_criterion."
            )

        # computing number of parameter scenarios that will be considered in CVaR
        self.n_cvar_scr = (1 - beta) * self.n_scr
        if self.n_cvar_scr < 1:
            print(
                "[WARNING]: "
                "given n_scr * beta given is smaller than 1, this yields a maximin "
                "design. Please provide a larger number of n_scr if a CVaR design "
                "was desired."
            )
            self.n_cvar_scr = np.ceil(self.n_cvar_scr).astype(int)
        else:
            self.n_cvar_scr = np.floor(self.n_cvar_scr).astype(int)

        # check if given reso is less than 3
        if reso < 3:
            print(
                f"The input reso is given as {reso}; the minimum value of reso is 3. "
                "Continuing with reso = 3."
            )
            reso = 3

        # initializing result lists
        self.cvar_optimal_candidates = []
        self.cvar_solution_times = []
        self._biobjective_values = np.empty((reso, 2))
        if plot:
            figs = []

            def add_fig(cdf, pdf):
                figs.append([cdf, pdf])

        self._alt_cvar = True
        """ Iteration 1: Maximal (Type 1) Mean Design """
        if self._verbose >= 1:
            print(f" CVaR Problem ".center(100, "*"))
            print(f"")
            print(f"[Iteration 1/{reso}]".center(100, "="))
            print(f"Computing the maximal mean design, obtaining the mean UB and CVaR LB"
                  f" in the Pareto Frontier.")
            print(f"")
        self.design_experiment(
            criterion,
            n_spt=n_spt,
            n_exp=n_exp,
            optimize_sampling_times=optimize_sampling_times,
            package=package,
            optimizer=optimizer,
            opt_options=opt_options,
            e0=e0,
            write=False,
            save_sensitivities=False,
            fd_jac=fd_jac,
            unconstrained_form=unconstrained_form,
            trim_fim=trim_fim,
            pseudo_bayesian_type=pseudo_bayesian_type,
            regularize_fim=regularize_fim,
            min_expected_value=-1000,
        )
        self.get_optimal_candidates()
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol, write=False)
        iter_1_efforts = np.copy(self.efforts)
        mean_ub = self._criterion_value
        iter_1_phi = np.copy(self.phi.value)
        if self._verbose >= 1:
            print("")
            print("Computing CVaR of Iteration 1's Solution")
        cvar_lb = (self.v - 1 / (self.n_scr * (1 - beta)) * cp.sum(self.s)).value
        if self._verbose >= 2:
            print(
                    f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds."
                )

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[0, :] = np.array([mean_ub, cvar_lb])
        if self._verbose >= 1:
            print(f"CVaR LB: {cvar_lb}")
            print(f"Mean UB: {mean_ub}")
            print(f"[Iteration 1/{reso} Completed]".center(100, "="))
            print(f"")
        if plot:
            self.phi.value = iter_1_phi
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=1),
                self.plot_criterion_pdf(write=False, iteration=1),
            )

        """ Iteration 2: Maximal CVaR_beta Design """
        if self._verbose >= 1:
            print(f"[Iteration 2/{reso}]".center(100, "="))
            print(f"Computing the maximal CVaR design, obtaining the CVaR UB, and mean "
                  f"LB in the Pareto Frontier.")
            print(f"")
        self.design_experiment(
            criterion,
            n_spt=n_spt,
            n_exp=n_exp,
            optimize_sampling_times=optimize_sampling_times,
            package=package,
            optimizer=optimizer,
            opt_options=opt_options,
            e0=e0,
            write=False,
            save_sensitivities=False,
            fd_jac=fd_jac,
            unconstrained_form=unconstrained_form,
            trim_fim=trim_fim,
            pseudo_bayesian_type=pseudo_bayesian_type,
            regularize_fim=regularize_fim,
            beta=self.beta,
        )
        self.get_optimal_candidates()
        iter_2_efforts = np.copy(self.efforts)
        if self._verbose >= 1:
            self.print_optimal_candidates(tol=tol, write=False)
        iter2_var = self.v.value
        cvar_ub = self._criterion_value

        if self._verbose >= 1:
            print("")
            print("Computing Mean of Iteration 2's Solution")

        self.design_experiment(
            criterion,
            n_spt=n_spt,
            n_exp=n_exp,
            optimize_sampling_times=optimize_sampling_times,
            package=package,
            optimizer=optimizer,
            opt_options=opt_options,
            e0=e0,
            write=False,
            save_sensitivities=False,
            fd_jac=fd_jac,
            unconstrained_form=unconstrained_form,
            trim_fim=trim_fim,
            pseudo_bayesian_type=pseudo_bayesian_type,
            regularize_fim=regularize_fim,
            beta=0.00,
            fix_effort=iter_2_efforts,
        )
        mean_lb = self._criterion_value
        if self._verbose >= 2:
            print(
                    f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds."
                )

        self.cvar_optimal_candidates.append(self.optimal_candidates)
        self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
        self._biobjective_values[1, :] = np.array([mean_lb, cvar_ub])
        if self._verbose >= 1:
            print(f"CVaR UB: {cvar_ub}")
            print(f"MEAN LB: {mean_lb}")
            print(f"[Iteration 2/{reso} Completed]".center(100, "="))
            print(f"")
        if plot:
            self.v.value = iter2_var
            self._criterion_value = cvar_ub
            add_fig(
                self.plot_criterion_cdf(write=False, iteration=2),
                self.plot_criterion_pdf(write=False, iteration=2),
            )

        """ Iterations 3+: Intermediate Points """
        mean_values = np.linspace(mean_lb, mean_ub, reso)
        mean_values = mean_values[:-1]
        mean_values = mean_values[1:]

        for i, mean in enumerate(mean_values):
            print(f"[Iteration {i + 3}/{reso}]".center(100, "="))
            self.design_experiment(
                criterion,
                n_spt=n_spt,
                n_exp=n_exp,
                optimize_sampling_times=optimize_sampling_times,
                package=package,
                optimizer=optimizer,
                opt_options=opt_options,
                e0=e0,
                write=False,
                save_sensitivities=False,
                fd_jac=fd_jac,
                unconstrained_form=unconstrained_form,
                trim_fim=trim_fim,
                pseudo_bayesian_type=pseudo_bayesian_type,
                regularize_fim=regularize_fim,
                beta=beta,
                min_expected_value=mean,
            )
            self.get_optimal_candidates()
            self.cvar_optimal_candidates.append(self.optimal_candidates)
            self.cvar_solution_times.append([self._sensitivity_analysis_time, self._optimization_time])
            self._biobjective_values[i + 2, :] = np.array([mean, self._criterion_value])

            if plot:
                add_fig(
                    self.plot_criterion_cdf(write=False, iteration=i+3),
                    self.plot_criterion_pdf(write=False, iteration=i+3),
                )
            if self._verbose >= 1:
                self.print_optimal_candidates(tol=tol, write=False)
                print(f"CVaR: {self._criterion_value}")
                print(f"MEAN: {cp.sum(self.phi).value / self.n_scr}")
                print(f"[Iteration {i + 3}/{reso} Completed]".center(100, "="))
                print(f"")

        # use the same axes.xlim for all plotted cdfs and pdfs
        if plot:
            xlims = []
            for i, fig in enumerate(figs):
                cdf, pdf = fig[0], fig[1]
                xlims.append(cdf.axes[0].get_xlim())
            xlims = np.asarray(xlims)
            for i, fig in enumerate(figs):
                cdf, pdf = fig[0], fig[1]
                cdf.axes[0].set_xlim(xlims[:, 0].min(), xlims[:, 1].max())
                pdf.axes[0].set_xlim(xlims[:, 0].min(), xlims[:, 1].max())

    def _formulate_cvar_problem_alt(self, criterion, beta, p_cons, min_cvar_value=None):
        self.v = cp.Variable()
        self.s = cp.Variable((self.n_scr,), nonneg=True)
        self.phi = cp.Variable((self.n_scr,))
        self.phi_mean = cp.Variable()

        self.eval_fim(self.efforts)

        for scr, (s_q, phi_q, mp, fim) in enumerate(zip(self.s, self.phi, self.model_parameters, self.scr_fims)):
            p_cons += [phi_q <= -criterion(fim)]
            p_cons += [s_q >= self.v - phi_q]
        if min_cvar_value is not None:
            self._constrained_cvar = True
            p_cons += [self.v - 1 / (self.n_scr * (1 - beta)) * cp.sum(self.s) >= min_cvar_value]
        else:
            self._constrained_cvar = False
        obj = cp.Maximize(cp.sum(self.phi) / self.n_scr)
        return obj

    def design_experiment(self, criterion, n_spt=None, n_exp=None,
                          optimize_sampling_times=False, package="cvxpy", optimizer=None,
                          opt_options=None, e0=None, write=False,
                          save_sensitivities=False, fd_jac=True,
                          unconstrained_form=False, trim_fim=False,
                          pseudo_bayesian_type=None, regularize_fim=False, beta=0.90,
                          min_expected_value=None, fix_effort=None, save_atomics=False,
                          **kwargs):
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
        self._save_atomics = save_atomics

        """ checking if CVaR problem """
        if "cvar" in self._current_criterion:
            self._cvar_problem = True
            self.beta = beta
        else:
            self._cvar_problem = False

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
            self._n_spt_spec = 1

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

        # declare and solve optimization problem
        self._sensitivity_analysis_time = 0
        start = time()
        # solvers
        if self._optimization_package == "scipy":
            if fix_effort is not None:
                raise NotImplementedError(
                    "Fixing effort is not supported for scipy solvers yet."
                )
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
                constraint = [
                    {"type": "eq", "fun": lambda e: sum(e) - 1.0},
                ]
                if self._dynamic_system and not self._opt_sampling_times:
                    raise NotImplementedError(
                        "Scipy solvers only supports optimize_sampling_times=True, "
                        "please use cvxpy solvers for optimize_sampling_times=False."
                    )
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
            if self._specified_n_spt:
                self.efforts = opt_result.x.reshape((self.n_c, self.n_spt_comb))
            else:
                self.efforts = opt_result.x.reshape((self.n_c, self.n_spt))
            self._efforts_transformed = False
            self._transform_efforts()
            opt_fun = opt_result.fun
        elif self._optimization_package == "cvxpy":
            # optimization variable and initial value
            if self._specified_n_spt:
                self.efforts = cp.Variable((self.n_c, self.n_spt_comb), nonneg=True)
            else:
                self.efforts = cp.Variable((self.n_c, self.n_spt), nonneg=True)
            self.efforts.value = e0
            # constraints and objective
            if self._discrete_design:
                p_cons = [cp.sum(self.efforts) == n_exp]
            else:
                p_cons = [cp.sum(self.efforts) <= 1]
                if not self._opt_sampling_times:
                    p_cons += [eff == eff[0] for c, eff in enumerate(self.efforts)]
            # cvxpy problem
            if self._cvar_problem:
                if self._alt_cvar:
                    obj = self._formulate_cvar_problem_alt(criterion, beta, p_cons, min_cvar_value=min_expected_value)
                else:
                    obj = self._formulate_cvar_problem(criterion, beta, p_cons, min_expected_value=min_expected_value)
            else:
                obj = cp.Maximize(-criterion(self.efforts))
            if fix_effort is not None:
                p_cons += [self.efforts == fix_effort / fix_effort.sum()]
            problem = cp.Problem(obj, p_cons)
            # solution
            if self._discrete_design:
                root = Node(self.efforts, problem)
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
                self.efforts = self.efforts.value
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

    def plot_criterion_cdf(self, write=False, iteration=None, dpi=360):
        if not self._pseudo_bayesian or not self._cvar_problem:
            raise SyntaxError(
                "Plotting cumulative distribution function only valid for pseudo-"
                "bayesian and cvar problems."
            )

        fig = plt.figure()
        axes = fig.add_subplot(111)
        if self._cvar_problem:
            x = np.sort(self.phi.value)
            mean = self.phi.value.mean()
            x = np.insert(x, 0, x[0])
            y = np.linspace(0, 1, x.size)
            axes.plot(x, y, "o--", alpha=0.3, c="#1f77b4")
            axes.plot(x, y, drawstyle="steps-post", c="#1f77b4")
            axes.axvline(
                x=self.v.value,
                ymin=0,
                ymax=1,
                c="tab:red",
                label=f"VaR {self.beta}",
            )
            axes.axvline(
                x=(self.v - 1 / (self.n_scr * (1 - self.beta)) * cp.sum(self.s)).value,
                ymin=0,
                ymax=1,
                c="tab:green",
                label=f"CVaR {self.beta}",
            )
            axes.axvline(
                x=mean,
                ymin=0,
                ymax=1,
                c="tab:blue",
                label=f"Mean",
            )
            axes.set_xlabel(f"{self._current_criterion}")
            axes.set_ylim(0, 1)
            axes.set_ylabel("Cumulative Probability")
            axes.legend()
            fig.tight_layout()
        else:
            raise NotImplementedError(
                "Plotting cumulative distribution function not implemented for pseudo-"
                "bayesian problems."
            )

        if write:
            fn = f"cdf_{self.beta*100}_beta_{self.n_scr}_scr"
            fp = self._generate_result_path(fn, "png", iteration=iteration)
            fig.savefig(fname=fp, dpi=dpi)

        return fig

    def plot_criterion_pdf(self, n_bins=20, write=False, iteration=None, dpi=360):
        if not self._pseudo_bayesian or not self._cvar_problem:
            raise SyntaxError(
                "Plotting probability density function only valid for pseudo-"
                "bayesian and cvar problems."
            )

        fig = plt.figure()
        axes = fig.add_subplot(111)
        if self._cvar_problem:
            x = self.phi.value
            axes.hist(x, bins=n_bins)
            axes.axvline(
                self.v.value,
                0,
                1,
                c="tab:red",
                label=f"VaR {self.beta}",
            )
            axes.axvline(
                self._criterion_value,
                0,
                1,
                c="tab:green",
                label=f"CVaR {self.beta}",
            )
            axes.set_xlabel(f"{self._current_criterion}")
            axes.set_ylabel("Frequency")
            axes.legend()
            fig.tight_layout()
        else:
            raise NotImplementedError(
                "Plotting probability density function not implemented for pseudo-"
                "bayesian problems."
            )

        if write:
            fn = f"pdf_{self.beta*100}_beta_{self.n_scr}_scr"
            fp = self._generate_result_path(fn, "png", iteration=iteration)
            fig.savefig(fname=fp, dpi=dpi)

        return fig

    def estimability_study(self, base_step=None, step_ratio=None, num_steps=None,
                           estimable_tolerance=0.04, write=False,
                           save_sensitivities=False, normalize=False):
        self._save_sensitivities = save_sensitivities
        self._compute_sensitivities = self._model_parameters_changed
        self._compute_sensitivities = self._compute_sensitivities or self._candidates_changed
        self._compute_sensitivities = self._compute_sensitivities or self.sensitivities is None

        if self._compute_sensitivities:
            self.eval_sensitivities(
                base_step=base_step,
                step_ratio=step_ratio,
                num_steps=num_steps
            )
        if normalize:
            self.normalize_sensitivities()
        else:
            self.normalized_sensitivity = self.sensitivities / self.responses_scales[None, None, :, None]

        z = self.normalized_sensitivity[:, :, self.measurable_responses, :].reshape(
            self.n_spt * self.n_m_r * self.n_c, self.n_mp)

        z_col_mag = np.nansum(np.power(z, 2), axis=0)
        next_estim_param = np.argmax(z_col_mag)
        self.estimable_columns = np.array([next_estim_param])
        self.estimability = [z_col_mag[next_estim_param]]
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
                    fn = f"estimability_{self.n_c}_cand"
                    fp = self._generate_result_path(fn, "pkl")
                    dump(self.estimable_columns, open(fp, 'wb'))
                print(
                    f'Identified estimable parameters are: '
                    f'{np.array2string(self.estimable_columns,separator=", ")}'
                )
                with np.printoptions(precision=1):
                    print(
                        f"Degree of Estimability: "
                        f"{np.array2string(np.asarray(self.estimability),separator=', ')}"
                    )
                return self.estimable_columns
            self.estimability.append(r_col_mag[next_estim_param])
            self.estimable_columns = np.append(self.estimable_columns, next_estim_param)

    def estimability_study_fim(self, save_sensitivities=False):
        self._save_sensitivities = save_sensitivities
        self.efforts = np.ones((self.n_c, self.n_spt))
        self.eval_fim(self.efforts)
        print(f"Estimable parameters: {self.estimable_model_parameters}")
        with np.printoptions(precision=1):
            print(f"Degree of Estimability: {self.estimability}")
        return self.estimable_model_parameters, self.estimability

    """ core utilities """

    def apportion(self, n_exp, method="adams", trimmed=True, compute_actual_efficiency=True):
        if self._dynamic_system and self._specified_n_spt:
            print(NotImplemented)
            return

        self.get_optimal_candidates()

        if n_exp < self.n_min_sups:
            print(
                f"[WARNING]: Given n_exp is lower than the minimum needed "
                f"({self.n_min_sups}); overwriting user input to this minimum."
            )
            n_exp = self.n_min_sups

        if self._opt_sampling_times:
            self.opt_eff = np.empty((len(self.optimal_candidates), self.max_n_opt_spt))
        else:
            self.opt_eff = np.empty((len(self.optimal_candidates)))
        self.opt_eff[:] = np.nan
        for i, opt_cand in enumerate(self.optimal_candidates):
            if self._opt_sampling_times:
                for j, spt in enumerate(opt_cand[4]):
                    if self._specified_n_spt:
                        self.opt_eff[i, j] = np.nansum(spt)
                    else:
                        self.opt_eff[i, j] = spt
            else:
                self.opt_eff[i] = np.nansum(opt_cand[4])
        if method == "adams":
            self.apportionments = self._adams_apportionment(self.opt_eff, n_exp)
        else:
            raise NotImplementedError(
                "At the moment, the only method implemented is 'adams', please use it. "
                "More apportionment methods will be implemented, but there is proof "
                "that Adam's method is the most efficient amongst other popular "
                "methods used in electoral college apportionments."
            )
        if self._verbose >= 1:
            print(f" Optimal Experiment for {n_exp:d} Runs ".center(100, "#"))
            print(f"{'Obtained on':<40}: {datetime.now()}")
            print(f"{'Criterion':<40}: {self._current_criterion}")
            print(f"{'Criterion Value':<40}: {self._criterion_value}")
            print(f"{'Pseudo-bayesian':<40}: {self._pseudo_bayesian}")
            if self._pseudo_bayesian:
                print(f"{'Pseudo-bayesian Criterion Type':<40}: {self._pseudo_bayesian_type}")
            print(f"{'CVaR Problem':<40}: {self._cvar_problem}")
            if self._cvar_problem:
                print(f"{'Beta':<40}: {self.beta}")
                print(f"{'Constrained Problem':<40}: {self._constrained_cvar}")
                if self._constrained_cvar:
                    print(f"{'Min. Mean Value':<40}: {cp.sum(self.phi).value / self.n_scr:.6f}")
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

            for i, (app_eff, opt_cand) in enumerate(zip(self.apportionments, self.optimal_candidates)):
                print(f"{f'[Candidate {opt_cand[0] + 1:d}]':-^100}")
                print(
                    f"{f'Recommended Apportionment: Run {np.nansum(app_eff):.0f}/{n_exp:d} Experiments':^100}")
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
                                print(f"  Variant {comb + 1} ~ [", end='')
                                for j, sp_time in enumerate(spt_comb):
                                    print(f"{f'{sp_time:.2f}':>10}", end='')
                                print("]: ", end='')
                                print(
                                    f'Run {f"{app_eff[comb]:.0f}/{np.nansum(app_eff):.0f}":>6} experiments, collecting {self._n_spt_spec} samples at given times')
                        else:
                            print("Sampling Times:")
                            for j, sp_time in enumerate(opt_cand[3]):
                                print(f"[{f'{sp_time:.2f}':>10}]: "
                                      f"Run {f'{app_eff[j]:.0f}/{np.nansum(app_eff):.0f}':>6} experiments, sampling at given time")
                    else:
                        print("Sampling Times:")
                        print(self.sampling_times_candidates[i])
            """ Computing and Reporting Rounding Efficiency """
            self.epsilon = self._eval_efficiency_bound(
                self.apportionments / n_exp,
                self.opt_eff
            )

            non_trimmed_apportionments = np.zeros_like(self.efforts)
            for opt_c, app_c in zip(self.optimal_candidates, self.apportionments):
                if isinstance(app_c, float):
                    non_trimmed_apportionments[opt_c[0], opt_c[5]] = app_c / n_exp
                else:
                    non_trimmed_apportionments[opt_c[0], opt_c[5]] = app_c[
                        [~np.isnan(app_c)]]
            norm_nt_app = non_trimmed_apportionments / np.sum(non_trimmed_apportionments)
            if compute_actual_efficiency:
                rounded_criterion_value = getattr(self, self._current_criterion)(
                    norm_nt_app).value
                if self._current_criterion == "d_opt_criterion":
                    efficiency = np.exp(1 / self.n_mp * (-rounded_criterion_value - self._criterion_value))
                elif self._current_criterion == "a_opt_criterion":
                    efficiency = -self._criterion_value / rounded_criterion_value
                elif self._current_criterion == "e_opt_criterion":
                    efficiency = -rounded_criterion_value / self._criterion_value

            if not trimmed:
                self.apportionments = non_trimmed_apportionments

            print(f"".center(100, "-"))
            print(
                f"The rounded design for {n_exp} runs is guaranteed to be at least "
                f"{self.epsilon * 100:.2f}% as good as the continuous design."
            )
            if compute_actual_efficiency:
                print(
                    f"The actual criterion value of the rounded design is "
                    f"{efficiency * 100:.2f}% as informative as the continuous design."
                )
            print(f"{'':#^100}")

        return self.apportionments

    def _adams_apportionment(self, efforts, n_exp):

        def update(effort, mu):
            return np.ceil(effort * mu)

        # pukelsheim's Heuristic
        mu = n_exp - efforts.size / 2
        self.apportionments = update(efforts, mu)
        iterations = 0
        while True:
            iterations += 1
            if np.nansum(self.apportionments) == n_exp:
                if self._verbose >= 3:
                    print(
                        f"Apportionment completed in {iterations} iterations, with final multiplier {mu}.")
                return self.apportionments
            elif np.nansum(self.apportionments) > n_exp:
                ratios = (self.apportionments - 1) / efforts
                candidate_to_reduce = np.unravel_index(np.nanargmax(ratios), ratios.shape)
                self.apportionments[candidate_to_reduce] -= 1
            else:
                ratios = self.apportionments / efforts
                candidate_to_increase = np.unravel_index(np.nanargmin(ratios), ratios.shape)
                self.apportionments[candidate_to_increase] += 1

    @staticmethod
    def _eval_efficiency_bound(effort1, effort2):
        eff_ratio = effort1 / effort2
        min_lkhd_ratio = np.nanmin(eff_ratio)
        return min_lkhd_ratio

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
    def plot_optimal_efforts(self, width=None, write=False, dpi=720,
                             force_3d=False, tol=1e-4, heatmap=False, figsize=None):
        if self.optimal_candidates is None:
            self.get_optimal_candidates()
        if self.n_opt_c is 0:
            print("Empty candidates, skipping plotting of optimal efforts.")
            return
        if heatmap:
            if not self._dynamic_system:
                print(
                    f"Warning: heatmaps are not suitable for non-dynamic experimental "
                    f"results. Reverting to bar charts."
                )
                fig = self._plot_current_efforts_2d(width=width, write=write, dpi=dpi,
                                                    tol=tol, figsize=figsize)
                return fig
            return self._efforts_heatmap(figsize=figsize, write=write)
        if (self._opt_sampling_times or force_3d) and self._dynamic_system:
            fig = self._plot_current_efforts_3d(tol=tol, width=width, write=write,
                                                dpi=dpi, figsize=figsize)
            return fig
        else:
            if force_3d:
                print(
                    "Warning: force 3d only works for dynamic systems, plotting "
                    "current design in 2D."
                )
            fig = self._plot_current_efforts_2d(width=width, write=write, dpi=dpi,
                                                tol=tol, figsize=figsize)
        return fig

    def _heatmap(self, data, row_labels, col_labels, ax=None,
                 cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if not ax:
            ax = plt.gca()

        # Plot the heatmap
        im = ax.imshow(data, **kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)

        ax.tick_params(top=False, bottom=True,
                       labeltop=False, labelbottom=True)

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax.set_title(f"{self._current_criterion} Efforts")
        ax.set_xlabel(f"Sampling Times (min)")

        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    def _annotate_heatmap(self, im, data=None, valfmt="{x:.2f}",
                          textcolors=("black", "white"),
                          threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

    def _efforts_heatmap(self, figsize=None, write=False, dpi=360):
        if figsize is None:
            fig = plt.figure(figsize=(3 + 1.0 * self.max_n_opt_spt, 2 + 0.40 * self.n_opt_c))
        else:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        c_id = [f"Candidate {opt_c[0]+1}" for opt_c in self.optimal_candidates]
        spt_id = [opt_c[3] for opt_c in self.optimal_candidates]
        spt_id = np.unique(np.array(list(itertools.zip_longest(*spt_id, fillvalue=spt_id[0][0]))).T)

        eff = np.zeros((len(c_id), spt_id.shape[0]))
        for c, opt_c in enumerate(self.optimal_candidates):
            for opt_spt, opt_eff in zip(opt_c[3], opt_c[4]):
                spt_index = np.where(spt_id == opt_spt)[0][0]
                eff[c, spt_index] = opt_eff

        im, cbar = self._heatmap(eff * 100, c_id, spt_id, ax=ax, cmap="YlGn")
        texts = self._annotate_heatmap(im, valfmt="{x:.2f}%")

        fig.tight_layout()
        if write:
            fn = f'efforts_heatmap_{self._current_criterion}'
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)

        return fig

    def plot_optimal_controls(self, alpha=0.3, markersize=3, non_opt_candidates=False,
                              n_ticks=3, visualize_efforts=True, tol=1e-4,
                              intervals=None, title=False, write=False, dpi=720):
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
            opt_idx = np.where(self.efforts >= tol)[0]
            axes.scatter(
                self.ti_controls_candidates[opt_idx, 0],
                self.ti_controls_candidates[opt_idx, 1],
                self.ti_controls_candidates[opt_idx, 2],
                facecolor="r",
                edgecolor="r",
                s=self.efforts[opt_idx] * 500 * markersize,
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
            fn = f"optimal_controls_{self.oed_result['optimality_criterion']}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)

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
                                 write=False, dpi=720):
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
            fn = f"response_plot_{self.oed_result['optimality_criterion']}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)

        return fig

    def plot_optimal_sensitivities(self, figsize=None, markersize=10, colour_map="jet",
                                   write=False, dpi=720, interactive=False):
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
            )

    def plot_pareto_frontier(self, write=False, dpi=720):
        if not self._cvar_problem:
            raise SyntaxError(
                "Pareto Frontier can only be plotted after solution of a CVaR problem."
            )

        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.scatter(
            self._biobjective_values[:, 0],
            self._biobjective_values[:, 1],
        )
        axes.set_xlabel("Mean Criterion Value")
        axes.set_ylabel(f"CVaR of Bottom {100 * (1 - self.beta):.2f}%")

        fig.tight_layout()

        if write:
            fn = f"optimal_controls_{self.oed_result['optimality_criterion']}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)

    def print_optimal_candidates(self, tol=1e-4):
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
        print(f"{'Criterion Value':<40}: {self._criterion_value}")
        print(f"{'Pseudo-bayesian':<40}: {self._pseudo_bayesian}")
        if self._pseudo_bayesian:
            print(f"{'Pseudo-bayesian Criterion Type':<40}: {self._pseudo_bayesian_type}")
        print(f"{'CVaR Problem':<40}: {self._cvar_problem}")
        if self._cvar_problem:
            print(f"{'Beta':<40}: {self.beta}")
            print(f"{'Constrained Problem':<40}: {self._constrained_cvar}")
            if self._constrained_cvar:
                print(f"{'Min. Mean Value':<40}: {cp.sum(self.phi).value / self.n_scr:.6f}")
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
        print(f"{'Information Matrix Regularized':<40}: {self._regularize_fim}")
        if self._regularize_fim:
            print(f"{'Regularization Epsilon':<40}: {self._eps}")
        print(f"{'Minimum Effort Threshold':<40}: {tol}")
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

    def start_logging(self):
        fn = f"log"
        fp = self._generate_result_path(fn, "txt")
        sys.stdout = Logger(file_path=fp)

    def stop_logging(self):
        sys.stdout = sys.__stdout__

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
            self.result_dir += path.splitext(path.basename(main.__file__))[0] + "_result/"
            self.result_dir += f'date_{now.year:d}-{now.month:d}-{now.day:d}/'
            self.create_result_dir()
        else:
            if path.exists(self.result_dir):
                return
            else:
                makedirs(self.result_dir)

    def write_oed_result(self):
        fn = f"{self.oed_result['optimality_criterion']:s}_oed_result"
        fp = self._generate_result_path(fn, "pkl")
        dump(self.oed_result, open(fp, "wb"))

    def save_state(self):
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

        designer_file = f"state"
        fp = self._generate_result_path(designer_file, ".pkl")
        dill.dump(state, open(designer_file, "wb"))

    def load_state(self, designer_path):
        state = dill.load(open(getcwd() + designer_path, 'rb'))
        self.n_c = state[0]
        self.n_spt = state[1]
        self.n_r = state[2]
        self.n_mp = state[3]
        self.ti_controls_candidates = state[4]
        self.tv_controls_candidates = state[5]
        self.sampling_times_candidates = state[6]
        self.measurable_responses = state[7]
        self.n_m_r = state[8]
        self.model_parameters = state[9]

    def save_responses(self):
        # TODO: implement save responses
        pass

    def load_sensitivity(self, sens_path):
        self.sensitivities = load(open(getcwd() + "/" + sens_path, "rb"))
        self._model_parameters_changed = False
        self._candidates_changed = False
        return self.sensitivities

    def load_atomics(self, atomic_path):
        if self._pseudo_bayesian:
            self.pb_atomic_fims = load(open(getcwd() + atomic_path, "rb"))
        else:
            self.atomic_fims = load(open(getcwd() + atomic_path, "rb"))
        self._model_parameters_changed = False
        self._candidates_changed = False
        return self.atomic_fims

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

    # risk-averse
    def cvar_d_opt_criterion(self, fim):
        self._cvar_problem = True

        if self._pseudo_bayesian:
            # old behaviour
            if False:
                self._eval_fim(efforts, mp)
                self._model_parameters_changed = True
                if self.fim.size == 1:
                    return -self.fim
                else:
                    return -cp.log_det(self.fim)
                # new behaviour
            if True:
                if self.fim.size == 1:
                    return -fim
                else:
                    return -cp.log_det(fim)
        else:
            raise SyntaxError(
                "CVaR criterion cannot be used for non Pseudo-bayesian problems, please "
                "ensure that you passed in the correct 2D numpy array as "
                "model_parameters."
            )

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
                           store_predictions=True,
                           plot_analysis_times=False, save_sensitivities=None,
                           reporting_frequency=None):
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
        if self.use_finite_difference:
            # setting default behaviour for step generators
            step_generator = nd.step_generators.MaxStepGenerator(
                base_step=base_step,
                step_ratio=step_ratio,
                num_steps=self._num_steps,
            )

        if isinstance(reporting_frequency, int) and reporting_frequency > 0:
            self.sens_report_freq = reporting_frequency
        if save_sensitivities is not None:
            self._save_sensitivities = save_sensitivities

        if self._pseudo_bayesian and not self._large_memory_requirement:
            self._scr_sens = np.empty((self.n_scr, self.n_c, self.n_spt, self.n_m_r, self.n_mp))

        self._sensitivity_analysis_done = False
        if self._verbose >= 2:
            print('[Sensitivity Analysis]'.center(100, "-"))
            print(f"{'Richardson Extrapolation Steps':<40}: {self._num_steps}")
            print(f"".center(100, "-"))
        start = time()

        self.sensitivities = np.empty((self.n_c, self.n_spt, self.n_m_r, self.n_mp))

        candidate_sens_times = []
        if self.use_finite_difference:
            jacob_fun = nd.Jacobian(fun=self._sensitivity_sim_wrapper,
                                    step=step_generator, method=method)
        """ main loop over experimental candidates """
        main_loop_start = time()
        for i, exp_candidate in enumerate(
                zip(self.sampling_times_candidates, self.ti_controls_candidates,
                    self.tv_controls_candidates)):
            """ specifying current experimental candidate """
            self._current_tic = exp_candidate[1]
            self._current_tvc = exp_candidate[2]
            self._current_spt = exp_candidate[0][~np.isnan(exp_candidate[0])]

            self.feval_sensitivity = 0
            single_start = time()
            try:
                if self.use_finite_difference:
                    temp_sens = jacob_fun(self._current_scr_mp, store_predictions)
                else:
                    temp_resp, temp_sens = self._sensitivity_sim_wrapper(self._current_scr_mp,
                                                                         store_predictions)
            except RuntimeError:
                print(
                    "The simulate function you provided encountered a Runtime Error "
                    "during sensitivity analysis. The inputs to the simulate function "
                    "were as follows."
                )
                print("Model Parameters:")
                print(self._current_scr_mp)
                print("Time-invariant Controls:")
                print(self._current_tic)
                print("Time-varying Controls:")
                print(self._current_tvc)
                print("Sampling Time Candidates:")
                print(self._current_spt)
                raise RuntimeError
            finish = time()
            if self._verbose >= 2 and self.sens_report_freq != 0:
                if (i + 1) % np.ceil(self.n_c / self.sens_report_freq) == 0 or (
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
            if self.use_finite_difference:
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

            self.sensitivities[i, :] = temp_sens
        finish = time()
        if self._verbose >= 2 and self.sens_report_freq != 0:
            print("".center(100, "-"))
        self._sensitivity_analysis_time += finish - start

        if self._var_n_sampling_time:
            self._pad_sensitivities()

        if self._pseudo_bayesian and not self._large_memory_requirement:
            self._scr_sens[self._current_scr] = self.sensitivities

        if self._save_sensitivities and not self._pseudo_bayesian:
            sens_file = f'sensitivity_{self.n_c}_cand'
            if self._dynamic_system:
                sens_file += f"_{self.n_spt}_spt"
            fp = self._generate_result_path(sens_file, "pkl")
            dump(self.sensitivities, open(fp, 'wb'))

        if plot_analysis_times:
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.plot(np.arange(1, self.n_c + 1, step=1), candidate_sens_times)

        self._sensitivity_analysis_done = True

        if self._norm_sens_by_params:
            self.sensitivities = self.sensitivities * self._current_scr_mp[None, None, None, :]

        return self.sensitivities

    def eval_fim(self, efforts, store_predictions=True):
        """
        Main evaluator for constructing the FIM from obtained sensitivities, stored in
        self.fim. When problem does not require large memory, will store atomic FIMs. The
        atomic FIMs for the c-th candidate is accessed through self.atomic_fims[c],
        returning a symmetric n_mp x n_mp 2D numpy array.

        When used for pseudo-Bayesian problems, the FIM is computed for each parameter
        scenario, stored in self.scr_fims. The atomic FIMs are stored as a 4D np.array,
        with dimensions (in order) n_scr, n_c, n_mp, n_mp i.e., the atomic FIM for the
        s-th parameter scenario and c-th candidate is accessed through
        self.pb_atomic_fims[s, c], returning a symmetric n_mp x n_mp 2D numpy array.

        The function also performs a parameter estimability study based on the FIM by
        summing the squares over the rows and columns of the FIM. Optionally, will trim
        out rows and columns that have its sum of squares close to 0. This helps with
        non-invertible FIMs.

        An alternative for dealing with non-invertible FIMs is to use a simple Tikhonov
        regularization, where a small scalar times the identity matrix is added to the
        FIM to obtain an invertible matrix.
        """
        if self._pseudo_bayesian:
            self._eval_pb_fims(
                efforts=efforts,
                store_predictions=store_predictions,
            )
            return self.scr_fims
        else:
            self._eval_fim(
                efforts=efforts,
                store_predictions=store_predictions,
            )
            return self.fim

    def _eval_fim(self, efforts, store_predictions=True, save_atomics=None):
        if save_atomics is not None:
            self._save_atomics = save_atomics

        def add_candidates(s_in, e_in, error_info_mat):
            if not np.any(np.isnan(s_in)):
                _atom_fim = s_in.T @ error_info_mat @ s_in
                self.fim += e_in * _atom_fim
            else:
                _atom_fim = np.zeros((self.n_mp, self.n_mp))
            if not self._large_memory_requirement:
                if self.atomic_fims is None:
                    self.atomic_fims = []
                if self._compute_atomics:
                    self.atomic_fims.append(_atom_fim)

        """ update efforts """
        self.efforts = efforts

        """ eval_sensitivities, only runs if model parameters changed """
        self._compute_sensitivities = self._model_parameters_changed
        self._compute_sensitivities = self._compute_sensitivities or self._candidates_changed
        self._compute_sensitivities = self._compute_sensitivities or self.sensitivities is None

        self._compute_atomics = self._model_parameters_changed
        self._compute_atomics = self._compute_atomics or self._candidates_changed
        self._compute_atomics = self._compute_atomics or self.atomic_fims is None

        if self._pseudo_bayesian:
            self._compute_sensitivities = self._compute_atomics or self.scr_fims is None

        if self._compute_sensitivities:
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

        if self._optimization_package is "scipy":
            if self._specified_n_spt:
                self.efforts = self.efforts.reshape((self.n_c, self.n_spt_comb))
            else:
                self.efforts = self.efforts.reshape((self.n_c, self.n_spt))
                if self.n_spt == 1:
                    self.efforts = self.efforts[:, None]
        # if atomic is not given
        if self._compute_atomics:
            self.atomic_fims = []
            self.fim = 0
            if self._specified_n_spt:
                for c, (eff, sen, spt_combs) in enumerate(zip(self.efforts, self.sensitivities, self.spt_candidates_combs)):
                    for comb, (e, spt) in enumerate(zip(eff, spt_combs)):
                        s = np.mean(sen[spt], axis=0)
                        add_candidates(s, e, self.error_fim)
            else:
                for c, (eff, sen) in enumerate(zip(self.efforts, self.sensitivities)):
                    for spt, (e, s) in enumerate(zip(eff, sen)):
                        add_candidates(s, e, self.error_fim)
        # if atomic is given
        else:
            self.fim = 0
            self.atomic_fims = self.atomic_fims.reshape((self.n_c, self.n_spt, self.n_mp, self.n_mp))
            if self._specified_n_spt:
                for c, (eff, atom, spt_combs) in enumerate(zip(self.efforts, self.atomic_fims, self.spt_candidates_combs)):
                    for comb, (e, spt) in enumerate(zip(eff, spt_combs)):
                        a = np.mean(atom[spt], axis=0)
                        self.fim += e * a
            else:
                for c, (eff, atom) in enumerate(zip(self.efforts, self.atomic_fims)):
                    for spt, (e, a) in enumerate(zip(eff, atom)):
                        self.fim += e * a

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

        if not self._large_memory_requirement:
            self.atomic_fims = np.asarray(self.atomic_fims)

        """ set current mp as completed to prevent recomputation of atomics """
        self._model_parameters_changed = False
        self._candidates_changed = False

        if self._save_atomics and not self._pseudo_bayesian:
            sens_file = f"atomics_{self.n_c}_cand"
            if self._dynamic_system:
                sens_file += f"_{self.n_spt}_spt"
            if self._pseudo_bayesian:
                sens_file += f"_{self.n_scr}_scr"
            fp = self._generate_result_path(sens_file, "pkl")
            dump(self.atomic_fims, open(fp, 'wb'))

        return self.fim

    def _eval_pb_fims(self, efforts, store_predictions=True):
        """ only recompute pb_atomics if the full parameter scenarios are changed """
        self._compute_pb_atomics = self._model_parameters_changed
        self._compute_pb_atomics = self._compute_pb_atomics or self._candidates_changed
        self._compute_pb_atomics = self._compute_pb_atomics or self.pb_atomic_fims is None

        self.scr_fims = []
        if self._compute_pb_atomics:
            if self._verbose >= 2:
                print(f"{' Pseudo-bayesian ':#^100}")
            if self._verbose >= 1:
                print(f'Evaluating information for each scenario...')
            if store_predictions:
                self.scr_responses = []
            if not self._large_memory_requirement:
                self.pb_atomic_fims = np.empty((self.n_scr, self.n_c * self.n_spt, self.n_mp, self.n_mp))
            for scr, mp in enumerate(self.model_parameters):
                self.atomic_fims = None
                self._current_scr = scr
                self._current_scr_mp = mp
                if self._verbose >= 2:
                    print(f"{f'[Scenario {scr+1}/{self.n_scr}]':=^100}")
                    print("Model Parameters:")
                    print(mp)
                self._eval_fim(self.efforts, store_predictions)
                self.scr_fims.append(self.fim)
                if self._verbose >= 2:
                    print(f"Time elapsed: {self._sensitivity_analysis_time:.2f} seconds.")
                if store_predictions:
                    self.scr_responses.append(self.response)
                    self.response = None
                if not self._large_memory_requirement:
                    self.pb_atomic_fims[scr] = self.atomic_fims
            if store_predictions:
                self.scr_responses = np.array(self.scr_responses)

            """ set current mp as completed to prevent recomputation of atomics """
            self._model_parameters_changed = False
        else:
            for scr, atomic_fims in enumerate(self.pb_atomic_fims):
                self.atomic_fims = atomic_fims
                self._eval_fim(self.efforts, store_predictions)
                self.scr_fims.append(self.fim)

        if self._save_atomics:
            fn = f"atomics_{self.n_c}_can_{self.n_scr}_scr"
            fp = self._generate_result_path(fn, "pkl")
            dump(self.pb_atomic_fims, open(fp, "wb"))

        return self.scr_fims

    def eval_pim(self, efforts, vector=False):
        if self._optimization_package is "cvxpy":
            raise NotImplementedError

        """ update mp, and efforts """
        self.eval_fim(efforts)

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

    """ getters (filters) """

    def get_optimal_candidates(self, tol=1e-4):
        if self.efforts is None:
            raise SyntaxError(
                'Please solve an experiment design before attempting to get optimal '
                'candidates.'
            )

        self._remove_zero_effort_candidates(tol=tol)
        self.optimal_candidates = []

        for i, eff_sp in enumerate(self.efforts):
            if self._dynamic_system and self._opt_sampling_times:
                optimal = np.any(eff_sp > tol)
            else:
                optimal = np.sum(eff_sp) > tol
            if optimal:
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
                    opt_candidate[3] = self.sampling_times_candidates[i]
                    opt_candidate[4] = eff_sp
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

        self.n_min_sups = 0
        self.max_n_opt_spt = 0
        for i, opt_cand in enumerate(self.optimal_candidates):
            if self._dynamic_system and self._opt_sampling_times:
                self.n_min_sups += len(opt_cand[4])
            else:
                self.n_min_sups += 1
            self.max_n_opt_spt = max(self.max_n_opt_spt, len(opt_cand[4]))

        return self.optimal_candidates

    """ optional operations """

    def evaluate_estimability_index(self):
        self.estimable_model_parameters = np.array([])
        self.estimability = np.array([])
        if self._optimization_package is 'cvxpy':
            try:
                fim_value = self.fim.value
            except AttributeError:
                fim_value = self.fim
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
        self.eval_fim(efforts)

        if self.fim.size == 1:
            if self._optimization_package is "scipy":
                d_opt = -self.fim
                if self._fd_jac:
                    return np.squeeze(d_opt)
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
        self.eval_fim(efforts)

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
            try:
                return cp.matrix_frac(np.identity(self.fim.shape[0]), self.fim)
            except ValueError:
                return cp.matrix_frac(np.identity(self.fim.shape[0]).tolist(), self.fim.tolist())

    def _e_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        self.eval_fim(efforts)

        if self.fim.size == 1:
            return -self.fim

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

        self.eval_pim(efforts)
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

        self.eval_pim(efforts)
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

        self.eval_pim(efforts)
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

        self.eval_pim(efforts)
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

        self.eval_pim(efforts)
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

        self.eval_pim(efforts)
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
        self.eval_fim(efforts)

        if self._optimization_package is "scipy":
            if self._fd_jac:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = np.mean([fim for fim in self.scr_fims], axis=0)
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
                    return -d_opt / self.n_scr
            else:
                raise NotImplementedError(
                    "Analytical Jacobian unimplemented for Pseudo-bayesian D-optimal."
                )

        elif self._optimization_package is 'cvxpy':
            if np.any([fim.shape == (1, 1) for fim in self.scr_fims]):
                return cp.sum([-fim for fim in self.scr_fims]) / self.n_scr
            else:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = cp.sum([fim for fim in self.scr_fims], axis=0) / self.n_scr
                    return -cp.log_det(avg_fim)
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    return cp.sum([
                        -cp.log_det(fim) for fim in self.scr_fims
                    ],
                    axis=0,
                    ) / self.n_scr

    def _pb_a_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        self.eval_fim(efforts)

        if self._optimization_package is "scipy":
            if self._fd_jac:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    a_opt = np.linalg.inv(
                        np.mean([fim for fim in self.scr_fims], axis=0)
                    ).trace()
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    np.mean([
                        np.linalg.inv(fim).trace()
                        for fim in self.scr_fims
                    ])
            else:
                raise NotImplementedError(
                    "Analytical Jacobian unimplemented for Pseudo-bayesian D-optimal."
                )

        elif self._optimization_package is 'cvxpy':
            if np.any([fim.shape == (1, 1) for fim in self.scr_fims]):
                return cp.sum([-fim for fim in self.scr_fims]) / self.n_scr
            else:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = cp.sum([fim for fim in self.scr_fims], axis=0) / self.n_scr
                    return cp.matrix_frac(np.identity(avg_fim.shape[0]), avg_fim)
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    return cp.sum([
                        cp.matrix_frac(np.identity(fim.shape[0]), fim)
                        for fim in self.scr_fims
                    ]) / self.n_scr

    def _pb_e_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        self.eval_fim(efforts)

        if self._optimization_package is "scipy":
            if self._fd_jac:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = np.sum([fim for fim in self.scr_fims], axis=0) / self.n_scr
                    return -np.linalg.eigvalsh(avg_fim).min()
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    return np.sum([
                        -np.linalg.eigvalsh(fim).min()
                        for fim in self.scr_fims
                    ]) / self.n_scr
            else:
                raise NotImplementedError(
                    "Analytical Jacobian unimplemented for Pseudo-bayesian D-optimal."
                )

        elif self._optimization_package is 'cvxpy':
            if np.any([fim.shape == (1, 1) for fim in self.scr_fims]):
                return cp.sum([-fim for fim in self.scr_fims])
            else:
                if self._pseudo_bayesian_type in [0, "avg_inf", "average_information"]:
                    avg_fim = cp.sum([fim for fim in self.scr_fims], axis=0) / self.n_scr
                    return -cp.lambda_min(avg_fim)
                elif self._pseudo_bayesian_type in [1, "avg_crit", "average_criterion"]:
                    return cp.sum([
                        -cp.lambda_min(fim)
                        for fim in self.scr_fims
                    ]) / self.n_scr

    # prediction-oriented
    def _pb_dg_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_di_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_ag_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_ai_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_eg_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    def _pb_ei_opt_criterion(self, efforts):
        raise NotImplementedError(
            "Prediction-oriented criteria not implemented for pseudo-bayesian problems."
        )

    """ private methods """

    def _generate_result_path(self, name, extension, iteration=None):
        self.create_result_dir()

        while True:
            fp = self.result_dir + f"run_{self.run_no}/"
            if path.exists(fp):
                fn = f"run_{self.run_no}_{name}.{extension}"
                if iteration is not None:
                    fn = f"iter_{iteration:d}_" + fn
                fp += fn
                if path.isfile(fp):
                    self.run_no += 1
                else:
                    self.run_no = 1
                    return fp
            else:
                makedirs(fp)
                fn = f"run_{self.run_no}_{name}.{extension}"
                if iteration is not None:
                    fn = f"iter_{iteration:d}_" + fn
                fp += fn
                self.run_no = 1
                return fp

    def _plot_optimal_sensitivities(self, absolute=False, legend=None,
                                   markersize=10, colour_map="jet",
                                   write=False, dpi=720, figsize=None):
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
        if self.n_m_r == 1 and self.n_mp == 1:
            axes = np.array([[axes]])
        elif self.n_m_r == 1:
            axes = np.array([axes])
        elif self.n_mp == 1:
            axes = np.array([axes]).T

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
            fn = f"sensitivity_plot_{self.oed_result['optimality_criterion']}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)
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
        if self.use_finite_difference:
            response = self._simulate_internal(self._current_tic, self._current_tvc,
                                               theta_try, self._current_spt)
        else:
            self.do_sensitivity_analysis = True
            response, sens = self._simulate_internal(self._current_tic, self._current_tvc,
                                                     theta_try, self._current_spt)
            self.do_sensitivity_analysis = False
        self.feval_sensitivity += 1
        """ store responses whenever required, and model parameters are the same as 
        current model's """
        if store_responses and np.allclose(theta_try, self._current_scr_mp):
            self._current_res = response
            self._store_current_response()
        if self.use_finite_difference:
            return response
        else:
            return response, sens

    def _plot_current_efforts_2d(self, tol=1e-4, width=None, write=False, dpi=720,
                                 figsize=None):
        self.get_optimal_candidates(tol=tol)

        if self._verbose >= 2:
            print("Plotting current continuous design.")

        if width is None:
            width = 0.7

        if self.efforts.ndim == 2:
            p_plot = np.array([np.sum(opt_cand[4]) for opt_cand in self.optimal_candidates])
        else:
            p_plot = np.array([opt_cand[4][0] for opt_cand in self.optimal_candidates])

        x = np.array([opt_cand[0]+1 for opt_cand in self.optimal_candidates]).astype(str)
        if figsize is None:
            fig = plt.figure(figsize=(15, 7))
        else:
            fig = plt.figure(figsize=figsize)
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
            fn = f"efforts_{self._current_criterion}"
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)

        fig.tight_layout()
        return fig

    def _plot_current_efforts_3d(self, width=None, write=False, dpi=720, tol=1e-4,
                                 figsize=None):
        self.get_optimal_candidates(tol=tol)

        if self._specified_n_spt:
            print(f"Warning, plot_optimal_efforts not implemented for specified n_spt.")
            return

        if self._verbose >= 2:
            print("Plotting current continuous design.")

        if width is None:
            width = 0.7

        p = self.efforts.reshape([self.n_c, self.n_spt])

        sampling_time_scale = np.nanmin(np.diff(self.sampling_times_candidates, axis=1))

        if figsize is None:
            fig = plt.figure(figsize=(12, 8))
        else:
            fig = plt.figure(figsize=figsize)
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
            fn = f'efforts_{self.oed_result["optimality_criterion"]}'
            fp = self._generate_result_path(fn, "png")
            fig.savefig(fname=fp, dpi=dpi)
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
            self._current_res = self._current_res[np.newaxis]
        if self.n_r is 1:
            self._current_res = self._current_res[:, np.newaxis]

        if self._var_n_sampling_time:
            self._current_res = np.pad(
                self._current_res,
                pad_width=((0, self.n_spt - self._current_res.shape[0]), (0, 0)),
                mode='constant',
                constant_values=np.nan
            )

        """ convert to list if np array """
        if isinstance(self.response, np.ndarray):
            self.response = self.response.tolist()
        self.response.append(self._current_res)

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
        if self._simulate_signature is 1:
            self._simulate_internal = lambda tic, tvc, mp, spt: \
                self.simulate(tic, mp)
        elif self._simulate_signature is 2:
            self._simulate_internal = lambda tic, tvc, mp, spt: \
                self.simulate(tic, spt, mp)
        elif self._simulate_signature is 3:
            self._simulate_internal = lambda tic, tvc, mp, spt: \
                self.simulate(tvc, spt, mp)
        elif self._simulate_signature is 4:
            self._simulate_internal = lambda tic, tvc, mp, spt: \
                self.simulate(tic, tvc, spt, mp)
        elif self._simulate_signature is 5:
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
                "Simulate function suggests time-invariant controls are needed, but "
                "ti_controls_candidates is empty."
            )

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
        # initialize simulate id
        self._simulate_signature = 0
        # check if model_parameters is present
        if "model_parameters" not in sim_sig:
            raise SyntaxError(
                f"The input argument \"model_parameters\" is not found in the simulate "
                f"function, please fix simulate signature."
            )
        if np.all([entry in sim_sig for entry in t4_sig]):
            self._simulate_signature = 4
            self._dynamic_system = True
            self._dynamic_controls = True
            self._invariant_controls = True
        elif np.all([entry in sim_sig for entry in t3_sig]):
            self._simulate_signature = 3
            self._dynamic_system = True
            self._dynamic_controls = True
            self._invariant_controls = False
        elif np.all([entry in sim_sig for entry in t2_sig]):
            self._simulate_signature = 2
            self._dynamic_system = True
            self._dynamic_controls = False
            self._invariant_controls = True
        elif np.all([entry in sim_sig for entry in t1_sig]):
            self._simulate_signature = 1
            self._dynamic_system = False
            self._dynamic_controls = False
            self._invariant_controls = True
        elif np.all([entry in sim_sig for entry in t5_sig]):
            self._simulate_signature = 5
            self._dynamic_system = True
            self._dynamic_controls = False
            self._invariant_controls = False
        if self._simulate_signature is 0:
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
        if self._invariant_controls:
            self.n_c = self.n_c_tic
        if self._dynamic_controls:
            if not self.n_c:
                self.n_c = self.n_c_tvc
            else:
                assert self.n_c == self.n_c_tvc, f"Inconsistent candidate lengths. " \
                                                 f"tvc_candidates has {self.n_c_tvc}, " \
                                                 f"but {self.n_c} is expected."
        if self._dynamic_system:
            if not self.n_c:
                self.n_c = self.n_c_spt
            else:
                assert self.n_c == self.n_c_spt, f"Inconsistent candidate lengths. " \
                                                 f"spt_candidates has {self.n_c_spt}, " \
                                                 f"but {self.n_c} is expected."

    def _check_var_spt(self):
        if np.all([len(spt) == len(self.sampling_times_candidates[0]) for spt in
                   self.sampling_times_candidates]) \
                and np.all(~np.isnan(self.sampling_times_candidates)):
            self._var_n_sampling_time = False
        else:
            self._var_n_sampling_time = True
            self._pad_sampling_times()

    def _get_component_sizes(self):

        if self._simulate_signature == 1:
            self.n_c_tic, self.n_tic = self.ti_controls_candidates.shape
            self.tv_controls_candidates = np.empty((self.n_c_tic, 1))
            self.n_c_tvc, self.n_tvc = self.n_c_tic, 1
            self.sampling_times_candidates = np.empty_like(self.ti_controls_candidates)
            self.n_c_spt, self.n_spt = self.n_c_tic, 1
        elif self._simulate_signature == 2:
            self.n_c_tic, self.n_tic = self.ti_controls_candidates.shape
            self.tv_controls_candidates = np.empty((self.n_c_tic, 1))
            self.n_c_tvc, self.n_tvc = self.n_c_tic, 1
            self.n_c_spt, self.n_spt = self.sampling_times_candidates.shape
        elif self._simulate_signature == 3:
            self.n_c_tvc, self.n_tvc = self.tv_controls_candidates.shape
            self.ti_controls_candidates = np.empty((self.n_c_tvc, 1))
            self.n_c_tic, self.n_tic = self.n_c_tvc, 1
            self.n_c_spt, self.n_spt = self.sampling_times_candidates.shape
        elif self._simulate_signature == 4:
            self.n_c_tic, self.n_tic = self.ti_controls_candidates.shape
            self.n_c_tvc, self.n_tvc = self.tv_controls_candidates.shape
            self.n_c_spt, self.n_spt = self.sampling_times_candidates.shape
        elif self._simulate_signature == 5:
            self.n_c_spt, self.n_spt = self.sampling_times_candidates.shape
            self.ti_controls_candidates = np.empty((self.n_c_spt, 1))
            self.n_c_tic, self.n_tic = self.n_c_spt, 1
            self.tv_controls_candidates = np.empty((self.n_c_spt, 1))
            self.n_c_tvc, self.n_tvc = self.n_c_spt, 1
        else:
            raise SyntaxError("Unrecognized simulate signature, unable to proceed.")

        # number of model parameters, and scenarios (if pseudo_bayesian)
        if self._pseudo_bayesian:
            self.n_scr, self.n_mp = self.model_parameters.shape
            self._current_scr_mp = self.model_parameters[0]
        else:
            self.n_mp = self.model_parameters.shape[0]
            self._current_scr_mp = self.model_parameters

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
        if self._pseudo_bayesian:
            memory_req *= self.n_scr
        if memory_req > self._memory_threshold:
            print(
                f'Sensitivity matrix will take {memory_req / 1e9:.2f} GB of memory space '
                f'(more than {self._memory_threshold / 1e9:.2f} GB threshold).'
            )
            self._large_memory_requirement = True

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

    def _remove_zero_effort_candidates(self, tol):
        self.efforts[self.efforts < tol] = 0
        self.efforts = self.efforts / self.efforts.sum()
        return self.efforts
