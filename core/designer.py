from datetime import datetime
from inspect import signature
from os import getcwd, path, makedirs
from pickle import dump, load
from string import Template
from time import time
from mpl_toolkits.mplot3d import Axes3D

import __main__ as main
import cvxpy as cp
import dill
import numdifftools as nd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


class Designer:
    """
    An experiment designer with capabilities to do parameter estimation, parameter
    estimability study, and
    solve continuous experiment design problems.

    Interfaces to optimization solvers via Pyomo. Supports virtually any Python
    functions as long as one can specify
    the model within the required general syntax. Designer comes equipped with various
    convenient and automated plotting
     functions, leveraging the pyplot library from matplotlib.
    """

    def __init__(self):
        """ core model components """
        # unorganized
        self._efforts_changed = False
        self._current_efforts = []
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
        self._dynamic_controls = None
        self._dynamic_system = None
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
        self.n_spt = None
        self.n_r = None
        self.n_mp = None
        self.n_e = None
        self.n_m_r = None

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
        self._current_model_parameters = None

        # store user-selected problem types
        self._sensitivity_is_normalized = None
        self._opt_sampling_times = False
        self._var_n_sampling_time = None

        # store chosen package to interface with the optimizer, and the chosen optimizer
        self._model_package = None
        self._optimization_package = None
        self._optimizer = None

        # storing states that helps minimize evaluations
        self._model_parameters_changed = True

        # temporary performance results for current design
        self._sensitivity_analysis_time = None
        self._optimization_time = None

        # store current criterion value
        self._criterion_value = None

        # store designer status and its verbal level after initialization
        self._status = 'empty'
        self._verbose = 0
        self._sensitivity_analysis_done = False

    """ user-defined methods: must be overwritten by user to work """

    def simulate(self, model, simulator, ti_controls, tv_controls, model_parameters,
                 sampling_times):
        raise SyntaxError("Don't forget to specify the simulate function.")

    """ core activity interfaces """

    def initialize(self, verbose=0, memory_threshold=int(1e9)):
        """ check if all required components are specified to model """
        if self.ti_controls_candidates is None:
            raise SyntaxError('Time-invariant controls candidates empty.')
        assert self.model_parameters is not None, 'Please specify nominal model ' \
                                                  'parameters.'

        """ check if required components are in right datatypes and determine if (i) 
        system is dynamic; (ii) there  
        exists time-invariant controls """
        if self.sampling_times_candidates is not None:
            if not isinstance(self.sampling_times_candidates, np.ndarray):
                raise SyntaxError(
                    'sampling_times_candidates must be supplied as a numpy array.')
            self._dynamic_system = True
        else:
            self._dynamic_system = False

        if not isinstance(self.ti_controls_candidates, np.ndarray):
            raise SyntaxError(
                'ti_controls_candidates must be supplied as a numpy array.')
        if not isinstance(self.model_parameters, (list, np.ndarray)):
            raise SyntaxError('model_parameters must be supplied as a numpy array.')

        """ saving number of candidates, sampling times, and model parameters """
        self.n_c = len(self.ti_controls_candidates)
        self.n_mp = len(self.model_parameters)

        if self.tv_controls_candidates is not None:
            if not isinstance(self.tv_controls_candidates, np.ndarray):
                raise SyntaxError(
                    'tv_controls_candidates must be supplied as a numpy array.')
            if not self._dynamic_system:
                raise SyntaxError(
                    'Time-varying control supplied, but sampling times is unsupplied.')
            self._dynamic_controls = True
        else:
            self.tv_controls_candidates = np.array([{0: 0} for _ in range(self.n_c)])
            self._dynamic_controls = False

        if self.sampling_times_candidates is None:
            self.sampling_times_candidates = np.array([0 for _ in range(self.n_c)])
            self.n_spt = 1

        """ handling simulate signature """
        simulate_signature = list(signature(self.simulate).parameters.keys())
        pyomo_simulate_signature = ['model', 'simulator', 'ti_controls', 'tv_controls',
                                    'model_parameters',
                                    'sampling_times']
        non_pyomo_simulate_signature = ['ti_controls', 'tv_controls', 'model_parameters',
                                        'sampling_times']
        if np.all([entry in simulate_signature for entry in pyomo_simulate_signature]):
            self._model_package = 'pyomo'
        elif np.all(
                [entry in simulate_signature for entry in non_pyomo_simulate_signature]):
            self._model_package = 'non-pyomo'
        else:
            print(
                "Unrecognized simulate function signature, please check if you have "
                "specified it correctly.")
        self._initialize_simulate_function()

        if self._dynamic_system:
            """ check if given time-invariant controls, time-varying controls, 
            and sampling times have the same
             number of candidates specified """
            n_tic_cand = len(self.ti_controls_candidates)
            n_spt_cand = len(self.sampling_times_candidates)
            if not n_tic_cand == n_spt_cand:
                raise SyntaxError(
                    "Number of candidates given in ti_controls, and sampling times are "
                    "inconsistent.")
            if self._dynamic_controls:
                n_tvc_cand = len(self.tv_controls_candidates)
                if not n_tvc_cand == n_spt_cand:
                    raise SyntaxError(
                        "Number of candidates given in tv_controls are inconsistent "
                        "with ti_controls "
                        "and sampling times.")
            """ checking that all sampling times candidates have equal number of 
            sampling times """
            if np.all([len(spt) == len(self.sampling_times_candidates[0]) for spt in
                       self.sampling_times_candidates]) \
                    and np.all(~np.isnan(self.sampling_times_candidates)):
                self._var_n_sampling_time = False
            else:
                self._var_n_sampling_time = True
                self._pad_sampling_times()
            self.n_spt = len(self.sampling_times_candidates[0])

        if self.n_r is None:
            print(
                "Running one simulation for initialization (required to determine "
                "number of responses).")
            y = self._simulate_internal(self.ti_controls_candidates[0],
                                        self.tv_controls_candidates[0],
                                        self.model_parameters,
                                        self.sampling_times_candidates[0][~np.isnan(
                                            self.sampling_times_candidates[0])])
            try:
                _, self.n_r = y.shape
            except ValueError:
                if self._dynamic_system and self.n_spt > 1:
                    self.n_r = 1
                else:
                    self.n_r = y.shape[0]
        if self.measurable_responses is None:
            self.n_m_r = self.n_r
            self.measurable_responses = np.array([_ for _ in range(self.n_r)])
        elif self.n_m_r != len(self.measurable_responses):
            if self.n_m_r > self.n_r:
                raise SyntaxError(
                    "Given number of measurable responses is greater than number of "
                    "responses given.")
            self.n_m_r = len(self.measurable_responses)

        # check problem size (affects if designer will be memory-efficient or quick)
        self._memory_threshold = memory_threshold
        memory_req = self.n_c * self.n_spt * self.n_m_r * self.n_mp ** 2 * 8

        if memory_req > self._memory_threshold:
            print(
                'Atomic fim will take {0:.2f} GB of memory space (more than {1:.2f} GB '
                'threshold) if vectorized '
                'evaluation is chosen. Changing to memory-efficient (but slower) '
                'computation of information '
                'matrices.'.format(memory_req / 1e9, self._memory_threshold / 1e9))
            self._large_memory_requirement = True

        self._status = 'ready'
        self._verbose = verbose
        if self._verbose >= 1:
            print('Initialization complete: designer ready.')

        return self._status

    def simulate_all_candidates(self, store_predictions=True,
                                plot_simulation_times=False):
        self.response = None  # resets response every time simulation is invoked
        self.feval_simulation = 0
        time_list = []
        for i, exp in enumerate(
                zip(self.ti_controls_candidates, self.tv_controls_candidates,
                    self.sampling_times_candidates)):
            self._ti_controls = exp[0]
            self._tv_controls = exp[1]
            self._sampling_times = exp[2][~np.isnan(exp[2])]
            assert self._sampling_times.size > 0, 'One candidate has an empty list of ' \
                                                  'sampling times, please check' \
                                                  'the specified experimental ' \
                                                  'candidates.'

            """ determine if simulation needs to be re-run: if data on time-invariant 
            control variables is missing, 
            will not run """
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
            plt.plot(time_list)
            plt.show()
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

    def estimate_parameters(self, init_guess, bounds, method='l-bfgs-b',
                            update_parameters=False, save_result=True, options=None):
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

        pe_result = minimize(self._residuals_wrapper_f, init_guess, bounds=bounds,
                             method=method, options=options)
        finish = time()
        if not pe_result.success:
            print('Fail: estimation did not converge; exiting.')
            return None

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

        try:
            pe_info_mat = pe_result.jac.transpose().dot(pe_result.jac)
            pe_param_covar = np.linalg.inv(pe_info_mat)
            if self._verbose >= 1:
                print("Estimated Parameter Covariance")
                print(pe_param_covar)
        except np.linalg.LinAlgError:
            print(
                "Estimated information matrix of estimation is singular, suggesting "
                "that not all model parameters"
                " are estimable using given data. Consider doing an estimability study.")

        if update_parameters:
            self.model_parameters = pe_result.x
            if self._verbose >= 2:
                print('Nominal parameter value in model updated.')

        if save_result:
            case_path = getcwd()
            today = datetime.now()
            result_dir = case_path + "/" + str(today.date()) + "_at_" + str(
                today.hour) + "-" + str(
                today.minute) + "-" + str(today.second) + "_full_model_pe_results/"
            makedirs(result_dir)
            with open(result_dir + "result_file.pkl", "wb") as file:
                dump(pe_result, file)
            if self._verbose >= 2:
                print('Parameter estimation result saved to: %s.' % result_dir)

        return pe_result

    def design_experiment(self, criterion, optimize_sampling_times=False,
                          package="cvxpy", optimizer=None, opt_options=None, e0=None,
                          write=True, plot=False, save_sensitivities=False,
                          fd_jac=True, unconstrained_form=False, **kwargs):
        # storing user choices
        self._optimization_package = package
        self._optimizer = optimizer
        self._opt_sampling_times = optimize_sampling_times
        self._save_sensitivities = save_sensitivities
        self._current_criterion = criterion.__name__
        self._fd_jac = fd_jac
        self._unconstrained_form = unconstrained_form

        if self._verbose >= 2:
            opt_verbose = True
        else:
            opt_verbose = False

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

        self.n_e = self.n_c
        if self._opt_sampling_times:
            if not self._dynamic_system:
                print('Warning: system is non-dynamic; '
                      'proceeding normally without optimizing sampling times.')
                self._opt_sampling_times = False
            else:
                self.n_e *= self.n_spt

        """ main codes """
        if self._verbose >= 1:
            print("Solving OED problem...")

        # set initial guess for optimal experimental efforts, if none given,
        # equal efforts for all candidates
        if e0 is None:
            e0 = np.array([1 / self.n_e for _ in range(self.n_e)])
            self.efforts = e0
        else:
            if not isinstance(e0, (np.ndarray, list)):
                msg = 'Initial guess for effort must be a 1D ' \
                      'list or numpy array.'
                raise SyntaxError(msg)
            if len(e0) != self.n_e:
                msg = 'Length of initial guess must be equal to ' \
                      '(i) sampling times not optimized: number of candidates; or ' \
                      '(ii) sampling times optimized: number of candidates times ' \
                      'number of sampling times. '
                raise SyntaxError(msg)
            self.efforts = e0

        # declare and solve optimization problem
        start = time()  # TODO: re-implement p transform to allow use of other scipy
        # solvers
        if self._optimization_package == "scipy":
            if self._unconstrained_form:
                opt_result = minimize(fun=criterion, x0=e0, method=optimizer,
                                      options=opt_options, jac=not self._fd_jac)

            else:
                e_bound = [(0, 1) for _ in e0]
                constraint = [{"type": "ineq", "fun": lambda e: np.sum(e) - 0.99},
                              {"type": "ineq", "fun": lambda e: 1.01 - np.sum(e)}]
                opt_result = minimize(fun=criterion, x0=e0, method=optimizer,
                                      options=opt_options, constraints=constraint,
                                      bounds=e_bound, jac=not self._fd_jac, **kwargs)

            self.efforts = opt_result.x
            self._efforts_transformed = False
            opt_fun = opt_result.fun
        elif self._optimization_package == "cvxpy":
            e = cp.Variable(self.n_e, nonneg=True)
            p_cons = [cp.sum(e) == 1]
            e.value = self.efforts
            obj = cp.Minimize(criterion(e))
            problem = cp.Problem(obj, p_cons)
            opt_fun = problem.solve(verbose=opt_verbose, solver=self._optimizer,
                                    **kwargs)
            self.efforts = e.value
        else:
            print("Unrecognized package, reverting to default: scipy.")
            opt_fun = None  # optional line to follow PEP8
            self.design_experiment(criterion, optimize_sampling_times, "scipy",
                                   optimizer, opt_options, e0, write, plot)

        self.transform_efforts()
        finish = time()

        """ report status and performance """
        self._optimization_time = finish - start
        if self._verbose:
            print(
                "Solved: sensitivity analysis took %.2f CPU seconds; the optimizer "
                "'%s' interfaced via the"
                " '%s' package solved the optimization problem in %.2f CPU seconds." %
                (self._sensitivity_analysis_time,
                 self._optimizer,
                 self._optimization_package,
                 self._optimization_time)
            )

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
                print(f'Identified estimable parameters are: {self}.estimable_columns')
                return self.estimable_columns
            self.estimable_columns = np.append(self.estimable_columns, next_estim_param)

    def estimability_study_fim(self):
        self.efforts = np.ones(self.n_c * self.n_spt)
        # self.eval_atomic_fims()
        self.eval_fim()
        print(f"Estimable parameters: {self.estimable_model_parameters}")
        print(f"Degree of Estimability: {self.estimability}")
        return self.estimable_model_parameters, self.estimability

    """ core utilities """

    # create grid
    def create_grid(self, bounds, levels):
        """ returns points from a mesh-centered grid """
        grid_args = ''
        for bound, level in zip(bounds, levels):
            grid_args += '%f:%f:%dj,' % (bound[0], bound[1], level)
        make_grid = 'self.grid = np.mgrid[%s]' % grid_args
        exec(make_grid)
        self.grid = self.grid.reshape(np.array(levels).size, np.prod(levels)).T
        return self.grid

    # visualization and result retrieval
    def plot_current_design(self, width=None, write=False, dpi=720, quality=95,
                            force_3d=False):
        if (self._opt_sampling_times or force_3d) and self._dynamic_system:
            self._plot_current_continuous_design_3d(width=width, write=write, dpi=dpi,
                                                    quality=quality)
        else:
            if force_3d:
                print(
                    "Warning: force 3d only works for dynamic systems, plotting "
                    "current design in 2D.")
            self._plot_current_continuous_design_2d(width=width, write=write, dpi=dpi,
                                                    quality=quality)

    def plot_sensitivities(self, absolute=False, draw_legend=True):
        # n_c, n_s_times, n_res, n_theta = self.sensitivity.shape
        fig1 = plt.figure(
            figsize=(1.25 * self.n_mp + 2 / (self.n_mp + 1),
                     1.25 + 1 * self.n_m_r + 2 / (self.n_m_r + 1)))
        if self._sensitivity_is_normalized:
            norm_status = 'Normalized '
        else:
            norm_status = 'Unnormalized '
        if absolute:
            abs_status = 'Absolute '
        else:
            abs_status = 'Directional '
        fig1.suptitle('%s%sSensitivity Plots' % (norm_status, abs_status))
        i = 0
        for row in range(self.n_m_r):
            for col in range(self.n_mp):
                i += 1
                create_axes = 'axes_%d_%d = fig1.add_subplot(%d, %d, %d)' % (
                    row, col, self.n_m_r, self.n_mp, i)
                exec(create_axes)
                for c, exp_candidate in enumerate(
                        zip(self.ti_controls_candidates, self.tv_controls_candidates,
                            self.sampling_times_candidates)):
                    if absolute:
                        sens = np.abs(
                            self.sensitivities[c, :, self.measurable_responses[row],
                            col])
                    else:
                        sens = self.sensitivities[c, :, self.measurable_responses[row],
                               col]
                    plot_sens = 'axes_%d_%d.plot(exp_candidate[2], sens, "-o", ' \
                                'label="Candidate %d")' % (
                                    row, col, c + 1)
                    exec(plot_sens)
                    ticklabel = 'axes_%d_%d.ticklabel_format(axis="y", style="sci", ' \
                                'scilimits=(0,0))' % (
                                    row, col)
                    exec(ticklabel)
                if draw_legend and self.n_c <= 10:
                    make_legend = 'axes_%d_%d.legend()' % (row, col)
                    exec(make_legend)
        fig1.tight_layout()
        plt.show()

    def plot_all_predictions(self, figsize=(10, 7.5), markersize=6, fontsize=5,
                             legend=True, legend_size=4, plot_data=False):
        assert self._status is 'ready', 'Initialize the designer first.'
        assert self.response is not None, 'Cannot plot prediction vs data when ' \
                                          'response is empty, please run and ' \
                                          'store predictions.'
        if plot_data:
            assert self.data is not None, 'Data is empty, cannot plot prediction vs ' \
                                          'data. Please specify data.'

        for res in range(self.n_m_r):
            """ creating the necessary figures """
            create_fig = 'fig%d = plt.figure(figsize=figsize)' % res
            exec(create_fig)

            n_fig_col = np.floor(np.sqrt(self.n_c)).astype(int)
            n_fig_row = np.floor(np.sqrt(self.n_c)).astype(int)

            while n_fig_col * n_fig_row < self.n_c:
                n_fig_col += 1

            """ creating the necessary subplots """
            for row in range(n_fig_row):
                for col in range(n_fig_col):
                    draw_subplots = 'axes%d_fig%d = fig%d.add_subplot(n_fig_row, ' \
                                    'n_fig_col, ' \
                                    'row * n_fig_col + (col + 1) )' % (
                                        row * n_fig_col + (col + 1), res, res)
                    exec(draw_subplots)

            """ defining a universal subplot axes limits """
            x_axis_lim = [
                np.min(self.sampling_times_candidates[
                           ~np.isnan(self.sampling_times_candidates)]),
                np.max(self.sampling_times_candidates[
                           ~np.isnan(self.sampling_times_candidates)])
            ]

            pred_max = np.max(
                self.response[:, :, res][~np.isnan(self.response[:, :, res])])
            pred_min = np.min(
                self.response[:, :, res][~np.isnan(self.response[:, :, res])])
            if plot_data:
                try:
                    data_max = np.max(
                        self.data[:, :, res][~np.isnan(self.data[:, :, res])])
                    data_min = np.min(
                        self.data[:, :, res][~np.isnan(self.data[:, :, res])])
                except ValueError:
                    data_max = 1
                    data_min = 0
                y_axis_lim = [
                    np.min([data_min, pred_min]),
                    np.max([data_max, pred_max])
                ]
            else:
                y_axis_lim = [pred_min, pred_max]

            for i, sampling_times in enumerate(self.sampling_times_candidates):
                """ plotting data if specified """
                if plot_data:
                    plot_data = 'data_lines = axes%d_fig%d.plot(sampling_times, ' \
                                'self.data[i, :, res], marker="v", ' \
                                'markersize=markersize, fillstyle="none", ' \
                                'linestyle="none", label="data")' \
                                % (i + 1, res)
                    exec(plot_data)

                """ plotting predictions """
                plot_predictions = 'pred_lines = axes%d_fig%d.plot(sampling_times, ' \
                                   'self.response[i, :, ' \
                                   'self.measurable_responses[res]], marker="1", ' \
                                   'markersize=markersize,' \
                                   'linestyle="none", label="predictions")' % (
                                       i + 1, res)
                exec(plot_predictions)

                """ adjusting axes limits, chosen so all subplots include all data and 
                all subplots have same scale """
                set_ylim = 'axes%d_fig%d.set_ylim(y_axis_lim[0] - 0.1 * (y_axis_lim[1] ' \
                           '' \
                           '' \
                           '' \
                           '' \
                           '- y_axis_lim[0]),' \
                           ' y_axis_lim[1] + 0.1 * (y_axis_lim[1] - y_axis_lim[0]))' % (
                               i + 1, res)
                exec(set_ylim)
                set_xlim = 'axes%d_fig%d.set_xlim(x_axis_lim[0] - 0.1 * (x_axis_lim[1] ' \
                           '' \
                           '' \
                           '' \
                           '' \
                           '- x_axis_lim[0]),' \
                           ' x_axis_lim[1] + 0.1 * (x_axis_lim[1] - x_axis_lim[0]))' % (
                               i + 1, res)
                exec(set_xlim)

                """ setting a smaller fontsize to accommodate for plots with larger 
                number of candidates """
                set_ticks_params = 'axes%d_fig%d.tick_params(axis="both", ' \
                                   'which="major", labelsize=fontsize)' % (
                                       i + 1, res)
                exec(set_ticks_params)
                set_yaxis_offset_fsize = "axes%d_fig%d.yaxis.get_offset_text(" \
                                         ").set_fontsize(fontsize)" % (
                                             i + 1, res)
                exec(set_yaxis_offset_fsize)

                """ give title to each subplot if names for candidates are given """
                if self.candidate_names is not None:
                    set_subplot_title = 'axes%d_fig%d.set_title(self.candidate_names[' \
                                        'i], fontsize=fontsize)' % (
                                            i + 1, res)
                    try:
                        exec(set_subplot_title)
                    except ValueError:
                        print(
                            'Mild warning when plotting: error in labelling candidate '
                            'name for candidate %d.' % i)

                if legend:
                    draw_legend = 'axes%d_fig%d.legend(prop={"size": legend_size})' % (
                        i + 1, res)
                    exec(draw_legend)
                set_yaxis_offset_fsize = "axes%d_fig%d.yaxis.get_offset_text(" \
                                         ").set_fontsize(fontsize)" % (
                                             i + 1, res)
                exec(set_yaxis_offset_fsize)

        """ make all figures use tight_layout for tidiness """
        for res in range(self.n_m_r):
            tight_layout = 'fig%d.tight_layout()' % res
            exec(tight_layout)

        plt.show()

    def plot_optimal_predictions(self, plot_simulation_times=False, legend=True,
                                 figsize=(10, 7.5), markersize=10, fontsize=10,
                                 legend_size=8, opt_spt_only=False):
        assert self._status is 'ready', 'Initialize the designer first.'
        assert self.response is not None, 'Cannot plot prediction vs data when ' \
                                          'response is empty, please run and ' \
                                          'store predictions.'

        self.get_optimal_candidates()

        for res in range(self.n_m_r):
            """ creating the necessary figures """
            create_fig = 'fig%d = plt.figure(figsize=figsize)' % res
            exec(create_fig)

            n_fig_col = np.floor(np.sqrt(len(self.optimal_candidates))).astype(int)
            n_fig_row = np.floor(np.sqrt(len(self.optimal_candidates))).astype(int)

            while n_fig_col * n_fig_row < len(self.optimal_candidates):
                n_fig_col += 1

            """ creating the necessary subplots """
            for row in range(n_fig_row):
                for col in range(n_fig_col):
                    draw_subplots = 'axes%d_fig%d = fig%d.add_subplot(n_fig_row, ' \
                                    'n_fig_col, ' \
                                    'row * n_fig_col + (col + 1) )' % (
                                        row * n_fig_col + (col + 1), res, res)
                    exec(draw_subplots)

            """ defining a universal subplot axes limits """
            x_axis_lim = [
                np.min(self.sampling_times_candidates[
                           ~np.isnan(self.sampling_times_candidates)]),
                np.max(self.sampling_times_candidates[
                           ~np.isnan(self.sampling_times_candidates)])
            ]

            pred_max = np.max(
                self.response[:, :, res][~np.isnan(self.response[:, :, res])])
            pred_min = np.min(
                self.response[:, :, res][~np.isnan(self.response[:, :, res])])

            y_axis_lim = [pred_min, pred_max]

            for i, opt_cand in enumerate(self.optimal_candidates):
                """ plotting predictions """
                if not opt_spt_only:
                    plot_pred = 'spt_pred_lines = axes%d_fig%d.plot(' \
                                'self.sampling_times_candidates[opt_cand[0]], ' \
                                'self.response[opt_cand[0], :, ' \
                                'self.measurable_responses[res]], ' \
                                'linestyle="--", label="Predictions", zorder=0)' % (
                                    i + 1, res)
                    exec(plot_pred)

                plot_opt_pred = 'pred_lines = axes%d_fig%d.scatter(opt_cand[3], ' \
                                'self.response[opt_cand[0], opt_cand[5], ' \
                                'self.measurable_responses[res]], marker="o", ' \
                                's=markersize*100*np.array(opt_cand[4]), ' \
                                'label="Optimal Sampling Times", c="r", zorder=1)' % (
                                    i + 1, res)
                exec(plot_opt_pred)

                """ adjusting axes limits, chosen so all subplots include all data and 
                all subplots have same scale """
                set_ylim = 'axes%d_fig%d.set_ylim(y_axis_lim[0] - 0.1 * (y_axis_lim[1] ' \
                           '' \
                           '' \
                           '' \
                           '' \
                           '- y_axis_lim[0]),' \
                           ' y_axis_lim[1] + 0.1 * (y_axis_lim[1] - y_axis_lim[0]))' % (
                               i + 1, res)
                exec(set_ylim)
                set_xlim = 'axes%d_fig%d.set_xlim(x_axis_lim[0] - 0.1 * (x_axis_lim[1] ' \
                           '' \
                           '' \
                           '' \
                           '' \
                           '- x_axis_lim[0]),' \
                           ' x_axis_lim[1] + 0.1 * (x_axis_lim[1] - x_axis_lim[0]))' % (
                               i + 1, res)
                exec(set_xlim)

                """ setting a smaller fontsize to accommodate for plots with larger 
                number of candidates """
                set_ticks_params = 'axes%d_fig%d.tick_params(axis="both", ' \
                                   'which="major", labelsize=fontsize)' % (
                                       i + 1, res)
                exec(set_ticks_params)
                set_yaxis_offset_fsize = "axes%d_fig%d.yaxis.get_offset_text(" \
                                         ").set_fontsize(fontsize)" % (
                                             i + 1, res)
                exec(set_yaxis_offset_fsize)

                """ give title to each subplot if names for candidates are given """
                if self.candidate_names is not None:
                    set_subplot_title = 'axes%d_fig%d.set_title(self.candidate_names[' \
                                        'opt_cand[0]], fontsize=fontsize)' % (
                                            i + 1, res)
                    try:
                        exec(set_subplot_title)
                    except ValueError:
                        print(
                            'Mild warning when plotting: error in labelling candidate '
                            'name for candidate %d.' % i)

                if legend:
                    draw_legend = 'axes%d_fig%d.legend(prop={"size": legend_size})' % (
                        i + 1, res)
                    exec(draw_legend)
                set_yaxis_offset_fsize = "axes%d_fig%d.yaxis.get_offset_text(" \
                                         ").set_fontsize(fontsize)" % (
                                             i + 1, res)
                exec(set_yaxis_offset_fsize)

        """ make all figures use tight_layout for tidiness """
        for res in range(self.n_m_r):
            tight_layout = 'fig%d.tight_layout()' % res
            exec(tight_layout)

        plt.show()

    def plot_optimal_sensitivities(self, absolute=False, draw_legend=True,
                                   markersize=10):
        # n_c, n_s_times, n_res, n_theta = self.sensitivity.shape
        self.get_optimal_candidates()

        fig1 = plt.figure(
            figsize=(self.n_mp * 4.0, 2.5 + 3.5 * self.n_m_r)
        )
        if self._sensitivity_is_normalized:
            norm_status = 'Normalized '
        else:
            norm_status = 'Unnormalized '
        if absolute:
            abs_status = 'Absolute '
        else:
            abs_status = 'Directional '
        fig1.suptitle('%s%sSensitivity Plots' % (norm_status, abs_status))
        i = 0
        for row in range(self.n_m_r):
            for col in range(self.n_mp):
                i += 1
                create_axes = 'axes_%d_%d = fig1.add_subplot(%d, %d, %d)' % (
                    row, col, self.n_m_r, self.n_mp, i)
                exec(create_axes)
                for c, cand in enumerate(self.optimal_candidates):
                    if absolute:
                        sens = np.abs(self.sensitivities[cand[0], :,
                                      self.measurable_responses[row], col])
                    else:
                        sens = self.sensitivities[cand[0], :,
                               self.measurable_responses[row], col]
                    plot_opt_sens = 'axes_%d_%d.plot(self.sampling_times_candidates[' \
                                    'cand[0]], sens, "--", ' \
                                    'label="Candidate %d")' % (row, col, cand[0] + 1)
                    exec(plot_opt_sens)
                    plot_sens = 'axes_%d_%d.scatter(cand[3], sens[cand[5]], ' \
                                'marker="o",' \
                                's=markersize*100*np.array(cand[4]))' % (row, col)
                    exec(plot_sens)
                    ticklabel = 'axes_%d_%d.ticklabel_format(axis="y", style="sci", ' \
                                'scilimits=(0,0))' % (
                                    row, col)
                    exec(ticklabel)
                if draw_legend and len(self.optimal_candidates) <= 10:
                    make_legend = 'axes_%d_%d.legend()' % (row, col)
                    exec(make_legend)
        fig1.tight_layout()
        plt.show()

    def print_optimal_candidates(self):
        if self.optimal_candidates is None:
            self.get_optimal_candidates()
        for i, opt_cand in enumerate(self.optimal_candidates):
            print("{0:-^100}".format("[Candidate {0:d}]".format(opt_cand[0] + 1)))
            print("{0:^100}".format("Recommended Effort: {0:.2f}% of budget".format(
                np.sum(opt_cand[4]) * 100)))
            print("Time-invariant Controls:")
            print(opt_cand[1])
            if self._dynamic_controls:
                print("Time-varying Controls:")
                print(opt_cand[2])
            if self._opt_sampling_times:
                print("Sampling Times:")
                if self._opt_sampling_times:
                    for j, sp_time in enumerate(opt_cand[3]):
                        print("[{0:>10}]: dedicate {1:.2f}% of budget".format(
                            "{0:.2f}".format(sp_time),
                            opt_cand[4][j] * 100))
        print("{0:#^100}".format(""))

    # saving, loading, writing
    def load_oed_result(self, result_path):
        raise NotImplementedError

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
            self._opt_sampling_times,
            self._sensitivity_is_normalized,
        ]

        designer_file = self.result_dir + "/" + 'state' + "_%d.pkl" % self.run_no
        if path.isfile(designer_file):
            self.run_no += 1
            self.save_state()
        else:
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
        self._opt_sampling_times = state[10]
        self._sensitivity_is_normalized = state[11]

        self._current_model_parameters = self.model_parameters
        return None

    def load_sensitivity(self, sens_path):
        self.sensitivities = load(open(getcwd() + sens_path, 'rb'))
        return self.sensitivities

    """ criteria """

    def d_opt_criterion(self, efforts):
        """ it is a PSD criterion, with exponential cone """
        self.efforts = efforts

        self.eval_fim()

        # evaluate the criterion
        if self.fim.size == 1:
            if self._optimization_package is "scipy":
                if self._fd_jac:
                    return -np.log1p(self.fim)
                else:
                    # jac = np.array([np.sum([-1 / self.fim * m], axis=(1, 2))
                    #                 for m in self.atomic_fims])
                    fim_pinv = np.linalg.pinv(self.fim)
                    jac = -np.array([
                        np.sum(fim_pinv.T * m)
                        for m in self.atomic_fims
                    ])
                    return -np.log1p(self.fim), jac
            elif self._optimization_package is 'cvxpy':
                return -cp.log1p(self.fim)

        if self._optimization_package is "scipy":
            d_opt = -np.prod(np.linalg.slogdet(self.fim))
            if self._fd_jac:
                return d_opt
            else:
                # jac = np.array([np.sum([-np.linalg.pinv(self.fim).T * m], axis=(1, 2))
                #                 for m in self.atomic_fims])
                # fim_pinv = np.linalg.pinv(self.fim, rcond=1e-20)
                # fim_pinv = np.linalg.inv(self.fim)
                # fim_pinv = self.fim.__invert__()
                # jac = -np.array([
                #     np.sum(self.fim.T * m)
                #     for m in self.atomic_fims
                # ])
                # return d_opt, jac
                raise NotImplementedError  # TODO: implement analytic jac for d-opt
        elif self._optimization_package is 'cvxpy':
            return -cp.log_det(self.fim)

    def a_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        # check and transform efforts if needed
        self.efforts = efforts

        self.eval_fim()

        if self.fim.size == 1:
            return -np.log1p(self.fim)

        if self._optimization_package is "scipy":
            start = time()
            a_opt = np.linalg.pinv(self.fim).trace()
            finish = time()
            print("1", finish - start)

            start = time()
            eig_vals = np.linalg.eigvals(self.fim)
            a_opt = np.sum(np.divide(1, eig_vals, where=np.where(eig_vals > 0)))
            finish = time()
            print("2", finish - start)

            if self._fd_jac:
                return a_opt
            else:
                # jac = np.array([ for m in self.atomic_fims])
                #  analytic jacobian for a_opt
                # return a_opt, jac
                raise NotImplementedError  # TODO: implement analytic jac for a-opt
        elif self._optimization_package is 'cvxpy':
            return cp.matrix_frac(np.identity(self.n_mp), self.fim)

    def e_opt_criterion(self, efforts):
        """ it is a PSD criterion """
        # check and transform efforts if needed
        self.efforts = efforts

        self.eval_fim()

        if self.fim.size == 1:
            return -np.log1p(self.fim)

        if self._optimization_package is "scipy":
            if self._fd_jac:
                return -np.linalg.eigvalsh(self.fim).min()
            else:
                raise NotImplementedError  # TODO: implement analytic jac for e-opt
        elif self._optimization_package is 'cvxpy':
            return -cp.lambda_min(self.fim)

    """ evaluators """

    def eval_residuals(self, model_parameters):
        self.model_parameters = model_parameters

        """ run the model to get predictions """
        self.simulate_all_candidates()
        self.residuals = self.data - np.array(self.response)[:, :,
                                     self.measurable_responses]

        return self.residuals[
            ~np.isnan(self.residuals)]  # return residuals where entries are not empty

    def eval_sensitivities(self, method='forward', base_step=None, step_ratio=None,
                           num_steps=None,
                           store_predictions=True, plot_analysis_times=False,
                           save_sensitivities=False, reporting_frequency=10):
        self._save_sensitivities = save_sensitivities

        """ check if model parameters have been changed or not """
        self._check_if_model_parameters_changed()

        # setting default behaviour for step generators
        step_generator = nd.step_generators.MaxStepGenerator(base_step=base_step,
                                                             step_ratio=step_ratio,
                                                             num_steps=num_steps)

        """ do sensitivity analysis if not done before or model parameters were 
        changed """
        if self.sensitivities is None or self._model_parameters_changed:
            self._sensitivity_analysis_done = False
            if self._verbose >= 1:
                print('Running sensitivity analysis...')
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
                temp_sens = jacob_fun(self.model_parameters, store_predictions)
                finish = time()
                if self._verbose >= 2:
                    if (i + 1) % np.ceil(self.n_c / reporting_frequency) == 0 or (
                            i + 1) == self.n_c:
                        print('[Candidate %d/%d]: time elapsed %.2f seconds.' %
                              (i + 1, self.n_c, finish - main_loop_start))
                candidate_sens_times.append(finish - single_start)
                """
                bunch of lines to make sure the Jacobian method returns the 
                sensitivity with dims: n_sp, n_res, n_theta
                -------------------------------------------------------------------------------------------------------
                8 possible cases
                -------------------------------------------------------------------------------------------------------
                case_1: complete                        n_sp    n_theta     n_res
                case_2: n_sp = 1                        n_res   n_theta
                case_3: n_theta = 1                     n_res   n_sp
                case_4: n_res = 1                       n_sp    n_theta
                case_5: n_sp & n_theta = 1              1       n_res
                case_6: n_sp & n_res = 1                1       n_theta
                case_7: n_theta & n_res = 1             1       n_sp
                case_8: n_sp, n_res, n_theta = 1        1       1
                -------------------------------------------------------------------------------------------------------
                """
                n_dim = len(temp_sens.shape)
                if n_dim == 3:  # covers case 1
                    temp_sens = np.moveaxis(temp_sens, 1, 2)  # switch n_theta and n_res
                elif self.n_spt == 1:
                    if self.n_mp == 1:  # covers case 5: add a new axis in the last dim
                        temp_sens = temp_sens[:, :, np.newaxis]
                    elif self.n_r == 1:  # covers case 2, 6, and 8: add a new axis in
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
            if self._verbose >= 1:
                print(
                    'Sensitivity analysis using numdifftools with forward scheme '
                    'finite difference took '
                    'a total of %.2f CPU seconds.' % (finish - start))
            self._sensitivity_analysis_time = finish - start

            # converting sens into a numpy array for optimizing further computations
            self.sensitivities = np.array(sens)

            # saving current model parameters
            self._current_model_parameters = np.copy(self.model_parameters)

            if self._var_n_sampling_time:
                self._pad_sensitivities()

            if self._save_sensitivities:
                self.create_result_dir()
                self.run_no = 1
                sens_file = self.result_dir + '/sensitivity_%d.pkl' % self.run_no
                while path.isfile(sens_file):
                    self.run_no += 1
                    sens_file = self.result_dir + '/sensitivity_%d.pkl' % self.run_no
                dump(self.sensitivities, open(sens_file, 'wb'))

            if plot_analysis_times:
                plt.plot(np.arange(1, self.n_c + 1, step=1), candidate_sens_times)
                plt.show()

        self._sensitivity_analysis_done = True
        return self.sensitivities

    def eval_fim(self):
        """
        Main evaluator for constructing the fim from obtained sensitivities.
        When scipy is used as optimization package and problem does not require large
        memory, will store atomic fims for analytical Jacobian.
        """
        self.eval_sensitivities(save_sensitivities=self._save_sensitivities)

        """ update efforts """
        if self._optimization_package is "cvxpy":
            e = self.efforts.value
        else:
            e = self.efforts
        self._efforts_changed = True
        self._efforts_transformed = False

        """ deal with unconstrained form, i.e. transform efforts """
        self.transform_efforts()  # only transform if required, logic incorporated there
        self._current_efforts = e

        """ deal with opt_sampling_times """
        if self._opt_sampling_times:
            sens = self.sensitivities.reshape(self.n_c * self.n_spt, self.n_m_r,
                                              self.n_mp)
        else:
            sens = np.nansum(self.sensitivities, axis=1)

        """ evaluate fim """
        start = time()
        self.fim = 0
        if self._optimization_package is "scipy" and not self._large_memory_requirement:
            self.atomic_fims = []
        for e, f in zip(self.efforts.flatten(), sens):
            if not np.any(np.isnan(f)):
                _atom_fim = f.T @ f
                self.fim += e * _atom_fim
            else:
                _atom_fim = 0
            if self._optimization_package is "scipy" and \
                    not self._large_memory_requirement:
                self.atomic_fims.append(_atom_fim)
        finish = time()
        self._fim_eval_time = finish - start
        if self._verbose >= 3:
            print("Evaluation of information matrix took {0:.2f} seconds.".format(
                self._fim_eval_time))

        self.trim_fim()  # get rid of non-estimable parameters

        return self.fim

    """ getters (filters) """

    def get_optimal_candidates(self):
        if self.efforts is None:
            raise SyntaxError(
                'Please solve an experiment design before attempting to get optimal '
                'candidates.')

        self.optimal_candidates = []
        if self._opt_sampling_times:
            efforts = self.efforts.reshape(self.n_c, self.n_spt)
            candidate_efforts = np.sum(efforts, axis=1)
        else:
            efforts = self.efforts
            candidate_efforts = self.efforts

        for i, eff_sp in enumerate(efforts):
            if np.sum(eff_sp) > 1e-4:
                opt_candidate = [
                    i,
                    self.ti_controls_candidates[i],
                    self.tv_controls_candidates[i],
                    [],
                    [],
                    [],
                    []
                ]
                if self._opt_sampling_times:
                    for j, eff in enumerate(eff_sp):
                        if eff > 1e-4:
                            opt_candidate[3].append(self.sampling_times_candidates[i][j])
                            opt_candidate[4].append(eff)
                            opt_candidate[5].append(j)
                else:
                    opt_candidate[4].append(eff_sp)
                self.optimal_candidates.append(opt_candidate)
        return self.optimal_candidates

    """ optional operations """

    def trim_fim(self):
        self.estimable_model_parameters = np.array([])
        self.estimability = np.array([])
        if self._optimization_package is 'cvxpy':
            fim_value = self.fim.value
        else:
            fim_value = self.fim

        for i, row in enumerate(fim_value):
            if not np.allclose(row, 0.0):
                self.estimable_model_parameters = np.append(
                    self.estimable_model_parameters, i)
                self.estimability = np.append(self.estimability,
                                              np.sqrt(np.inner(row, row)))
        self.estimable_model_parameters = self.estimable_model_parameters.astype(int)

        if len(self.estimable_model_parameters) is 0:
            self.fim = np.array([0])
        else:
            self.fim = self.fim[
                np.ix_(self.estimable_model_parameters, self.estimable_model_parameters)]

    def normalize_sensitivities(self, overwrite_unnormalized=False):
        assert not np.allclose(self.model_parameters,
                               0), 'At least one nominal model parameter value is ' \
                                   'equal to 0, cannot normalize sensitivities. ' \
                                   'Consider re-estimating your parameters or ' \
                                   're-parameterize your model.'
        if self.response is None:
            self.simulate_all_candidates(store_predictions=True)

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
                # normalize response values
                self.normalized_sensitivity = np.divide(self.normalized_sensitivity,
                                                        self.response[:, :, :, None])
        else:
            assert isinstance(self.responses_scales,
                              np.ndarray), "Please specify responses_scales as a 1D " \
                                           "numpy array."
            assert self.responses_scales.size == self.n_r, 'Length of responses scales ' \
                                                           '' \
                                                           '' \
                                                           '' \
                                                           '' \
                                                           'is different from ' \
                                                           'the total number of ' \
                                                           'responses (includes ' \
                                                           'those which are measurable ' \
                                                           '' \
                                                           '' \
                                                           '' \
                                                           '' \
                                                           'and not).)'
            self.normalized_sensitivity = np.divide(self.normalized_sensitivity,
                                                    self.responses_scales[None, None, :,
                                                    None])
        if overwrite_unnormalized:
            self.sensitivities = self.normalized_sensitivity
            self._sensitivity_is_normalized = True
            return self.sensitivities
        return self.normalized_sensitivity

    """ private methods """

    def _sensitivity_sim_wrapper(self, theta_try, store_responses=False):
        response = self._simulate_internal(self._ti_controls, self._tv_controls,
                                           theta_try, self._sampling_times)
        self.feval_sensitivity += 1
        """ store responses whenever required, and model parameters are the same as 
        current model's """
        if store_responses and np.allclose(theta_try, self.model_parameters):
            self._current_response = response
            self._store_current_response()
        return response

    def _check_if_model_parameters_changed(self):
        if self._current_model_parameters is None:
            self._current_model_parameters = np.empty(1)
        if np.allclose(self.model_parameters, self._current_model_parameters):
            self._model_parameters_changed = False
        else:
            self._model_parameters_changed = True

    def _plot_current_continuous_design_2d(self, width=None, write=False, dpi=720,
                                           quality=95):
        if self._verbose >= 2:
            print("Plotting current continuous design.")

        if width is None:
            width = 0.7

        p = self.efforts.reshape(self.n_c)
        p_plot = p[np.where(p > 1e-4)]

        x = np.arange(1, self.n_c + 1, 1)[np.where(p > 1e-4)].astype(str)
        fig1 = plt.figure(figsize=(15, 7))
        axes1 = fig1.add_subplot(111)

        axes1.bar(x, p_plot, width=width)

        axes1.set_xticks(x)
        axes1.set_xlabel("Candidate Number")

        axes1.set_ylabel("Optimal Experimental Effort")
        axes1.set_ylim([0, 1])
        axes1.set_yticks(np.linspace(0, 1, 11))

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
            fig1.savefig(fname=figfile, dpi=dpi, quality=quality)
            self.run_no = 1

        fig1.tight_layout()
        plt.show()

    def _plot_current_continuous_design_3d(self, width=None, write=False, dpi=720,
                                           quality=95):
        if self._verbose >= 2:
            print("Plotting current continuous design.")

        if width is None:
            width = 0.7

        if self._opt_sampling_times:
            p = self.efforts.reshape([self.n_c, self.n_spt])
        else:
            p = np.repeat(self.efforts[:, None], self.n_spt, axis=1)
            p = np.multiply(p, 1 / self.n_spt)

        sampling_time_scale = np.nanmin(np.diff(self.sampling_times_candidates, axis=1))

        fig1 = plt.figure(figsize=(12, 8))
        axes1 = fig1.add_subplot(111, projection='3d')
        opt_cand = np.unique(np.where(p > 1e-4)[0], axis=0)
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

            axes1.bar3d(
                x=x,
                y=y,
                z=z,
                dx=dx,
                dy=dy,
                dz=dz
            )

        axes1.grid(False)
        axes1.set_xlabel('Candidate')
        xticks = opt_cand + 1
        axes1.set_xticks(
            [c for c, _ in enumerate(self.sampling_times_candidates[opt_cand])])
        axes1.set_xticklabels(labels=xticks)

        axes1.set_ylabel('Sampling Times')

        axes1.set_zlabel('Experimental Effort')
        axes1.set_zlim([0, 1])
        axes1.set_zticks(np.linspace(0, 1, 6))

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
            fig1.savefig(fname=figfile, dpi=dpi, quality=quality)
            self.run_no = 1
        fig1.tight_layout()
        plt.show()

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
            diff = self.n_spt - row.shape[0]
            self.sensitivities[i] = np.pad(self.sensitivities[i][:, ],
                                           pad_width=((0, diff), (0, 0), (0, 0)),
                                           mode='constant', constant_values=np.nan)
        self.sensitivities = self.sensitivities.tolist()
        self.sensitivities = np.array(self.sensitivities)
        return self.sensitivities

    def _store_current_response(self):
        """ padding responses to accommodate for missing sampling times """
        start = time()
        if self.response is None:  # if it is the first response to be stored,
            # initialize response list
            self.response = []

        if self.n_spt is 1:
            self._current_response = self._current_response[np.newaxis]
        elif self.n_r is 1:
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
        residuals = self.eval_residuals(model_parameters)
        return np.inner(residuals, residuals)

    def _simulate_internal(self, ti_controls, tv_controls, theta, sampling_times):
        raise SyntaxError(
            "Make sure you have initialized the designer, and specified the simulate "
            "function correctly."
        )

    def _initialize_simulate_function(self):
        if self._model_package == 'pyomo':
            self._simulate_internal = lambda tic, tvc, mp, spt: self.simulate(self.model,
                                                                              self.simulator,
                                                                              tic, tvc,
                                                                              mp, spt)
        elif self._model_package == 'non-pyomo':
            self._simulate_internal = self.simulate
        else:
            raise SyntaxError(
                'Cannot initialize simulate function properly, check your syntax.')

    def transform_efforts(self):
        if self._unconstrained_form:
            if self._efforts_changed or not self._efforts_transformed:
                self.efforts = np.square(self.efforts)
                self.efforts /= np.sum(self.efforts)
                self._efforts_transformed = True
                if self._verbose >= 3:
                    print("Efforts transformed.")

        return self.efforts
