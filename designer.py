from scipy.optimize import minimize, least_squares
from scipy.stats import chi2
from time import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pickle import dump, load
from os import getcwd, path, makedirs
from datetime import datetime
from inspect import signature
import dill
import numdifftools as nd
import cvxpy as cp
import keyboard
import numpy as np
import __main__ as main


class Designer:
    """
    An experiment designer with capabilities to do parameter estimation, parameter estimability study, and
    solve continuous experiment design problems.

    Interfaces to optimization solvers via Pyomo. Supports virtually any Python functions as long as one can specify
    the model within the required general syntax. Designer comes equipped with various convenient and automated plotting
     functions, leveraging the pyplot library from matplotlib.
    """
    def __init__(self):
        """ core model components """
        # core user-defined variables
        self.sampling_times_candidates = None  # sampling times of experiment. 2D numpy array of floats. Rows are the number of candidates, columns are the sampling times for given candidate. None means non-dynamic experiment.
        self.ti_controls_candidates = None  # time-invariant controls, a 2D numpy array of floats. Rows are the number of candidates, columns are the different controls.
        self.tv_controls_candidates = None  # time-varying controls, a 2D numpy array of dictionaries. Rows are the number of candidates, columns are the different controls.
        self.model_parameters = None  # nominal model parameters, a 1D numpy array of floats.
        self.mono_fim = None

        # optional user-defined variables
        self.candidate_names = None  # plotting names
        self.measurable_responses = None  # subset of measurable states

        # core designer outputs
        self.response = None  # the predicted response profiles, a 3D numpy array. 1st dim are the candidates, 2nd dim are sampling times, and 3rd dim are the different responses.
        self.sensitivity = None  # a 4D numpy array. First dim is the number of candidates, second dim are the different sampling times, third dim are the are the different responses, and last dim different model parameters.

        # pyomo-specific
        self.simulator = None  # object needed when using pyomo models
        self.model = None  # object needed when using pyomo models

        """ problem dimension sizes """
        self.n_cand = None
        self.n_sample_time = None
        self.n_res = None
        self.n_theta = None
        self.n_p = None
        self.n_m_res = None

        """ parameter estimation """
        self.data = None  # stored data, a 3D numpy array, same shape as response. Whenever data is missing, use np.nan to fill the array.
        self.residuals = None  # stored residuals, 3D numpy array with the same shape as data and response. Will skip entries whenever data is empty.

        """ performance attributes """
        self.feval_simulation = None
        self.feval_sensitivity = None

        """ parameter estimability """
        self.estimable_columns = None
        self.responses_scales = None

        """ continuous oed-related quantities """
        # sensitivities
        self.p_candidates = None
        self.F = None  # overall regressor matrix
        self.fim = None  # the information matrix for current experimental design
        self.p_var = None  # the prediction covariance matrix

        """ saving, loading attributes """
        # current oed result
        self.run_no = 1
        self.oed_result = None
        self.result_dir = None

        """ plotting attributes """
        self.grid = None  # storing grid when create_grid method is used to help generate candidates

        """ private attributes """
        # current candidate controls, and sampling times: required for sensitivity evaluations
        self._ti_controls = None
        self._tv_controls = None
        self._sampling_times = None
        self._current_response = None  # a 2D numpy array. 1st dim are sampling times, 2nd dim are different responses
        self._current_model_parameters = None

        # store user-selected problem types
        self._sensitivity_is_normalized = None
        self._opt_sampling_times = True
        self._var_n_sampling_time = None
        self._transform_p = False

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
    def simulate(self, model, simulator, ti_controls, tv_controls, model_parameters, sampling_times):
        raise SyntaxError("Don't forget to specify the simulate function.")

    """ core activity interfaces """
    def initialize(self, verbose=0):
        """ check if all required components are specified to model """
        assert self.ti_controls_candidates is not None, 'Please specify time-invariant controls candidates.'
        assert self.tv_controls_candidates is not None, 'Please specify time-varying controls candidates.'
        assert self.sampling_times_candidates is not None, 'Please specify sampling times candidates.'
        assert self.model_parameters is not None, 'Please specify nominal model parameters.'

        """ check if required components are in right datatypes """
        assert isinstance(self.ti_controls_candidates, np.ndarray), \
            'ti_controls_candidates must be supplied as a numpy array.'
        assert isinstance(self.tv_controls_candidates, np.ndarray), \
            'tv_controls_candidates must be supplied as a numpy array.'
        assert isinstance(self.sampling_times_candidates, np.ndarray), \
            'sampling_times_candidates must be supplied as a numpy array.'
        assert isinstance(self.model_parameters, (list, np.ndarray)), \
            'model_parameters must be supplied as a numpy array.'

        """ handling simulate signature """
        simulate_signature = list(signature(self.simulate).parameters.keys())
        pyomo_simulate_signature = ['model', 'simulator', 'ti_controls', 'tv_controls', 'model_parameters',
                                    'sampling_times']
        non_pyomo_simulate_signature = ['ti_controls', 'tv_controls', 'model_parameters', 'sampling_times']
        if np.all([entry in simulate_signature for entry in pyomo_simulate_signature]):
            self._model_package = 'pyomo'
        elif np.all([entry in simulate_signature for entry in non_pyomo_simulate_signature]):
            self._model_package = 'non-pyomo'
        else:
            print("Unrecognized simulate function signature, please check if you have specified it correctly.")
        self._initialize_simulate_function()

        """ check if given time-invariant controls, time-varying controls, and
         sampling times have the same number of candidates specified """
        n_tic_cand = len(self.ti_controls_candidates)
        n_tvc_cand = len(self.tv_controls_candidates)
        n_spt_cand = len(self.sampling_times_candidates)
        assert n_tic_cand == n_tvc_cand and n_tic_cand == n_spt_cand, \
            "Number of candidates given in ti_controls, tv_controls, and sampling_times are inconsistent."

        """ checking that all sampling times candidates have equal number of sampling times """
        if all(len(samp_t) == len(self.sampling_times_candidates[0]) for samp_t in self.sampling_times_candidates):
            self._var_n_sampling_time = False
        else:
            self._var_n_sampling_time = True
            self._pad_sampling_times()

        """ saving number of candidates, sampling times, and model parameters """
        self.n_cand = len(self.sampling_times_candidates)
        self.n_sample_time = len(self.sampling_times_candidates[0])
        self.n_theta = len(self.model_parameters)
        if self.n_res is None:
            y = self._simulate_internal(self.ti_controls_candidates[0],
                                        self.tv_controls_candidates[0],
                                        self.model_parameters,
                                        self.sampling_times_candidates[0][~np.isnan(self.sampling_times_candidates[0])])
            try:
                _, self.n_res = y.shape
            except ValueError:
                self.n_res = 1
        if self.measurable_responses is None:
            self.n_m_res = self.n_res
            self.measurable_responses = np.array([_ for _ in range(self.n_res)])
        else:
            self.n_m_res = len(self.measurable_responses)
        self._status = 'ready'
        self._verbose = verbose
        if self._verbose >= 1:
            print('Initialization complete: designer ready.')

        return self._status

    def simulate_all_candidates(self, store_predictions=True, plot_simulation_times=False):
        self.response = None  # resets response every time simulation is invoked
        self.feval_simulation = 0
        time_list = []
        for i, exp in enumerate(zip(self.ti_controls_candidates, self.tv_controls_candidates,
                                    self.sampling_times_candidates)):
            self._ti_controls = exp[0]
            self._tv_controls = exp[1]
            self._sampling_times = exp[2][~np.isnan(exp[2])]
            assert self._sampling_times.size > 0, 'One candidate has an empty list of sampling times, please check' \
                                                  'the specified experimental candidates.'

            """ determine if simulation needs to be re-run: if data on time-invariant control variables is missing, 
            will not run """
            cond_1 = np.any(np.isnan(exp[0]))
            if np.any([cond_1]):
                self._current_response = np.nan
            else:
                start = time()
                response = self._simulate_internal(self._ti_controls, self._tv_controls, self.model_parameters,
                                                   self._sampling_times)
                finish = time()
                self.feval_simulation += 1
                self._current_response = response
                time_list.append(finish-start)

            if store_predictions:
                self._store_current_response()
        if plot_simulation_times:
            plt.plot(time_list)
            plt.show()
        return self.response

    def simulate_optimal_candidates(self, plot_simulation_times=False):
        raise NotImplementedError

    def estimate_parameters(self, init_guess, bounds, method='l-bfgs-b', update_parameters=False, save_result=True,
                            options=None):
        assert self.data is not None, "No data is put in, do not forget to add it in."

        if options is None:
            if self._verbose >= 2:
                options = {'disp': True}
            else:
                options = {'disp': False}

        if self._verbose >= 1:
            print("Solving parameter estimation...")
        start = time()

        pe_result = minimize(self._residuals_wrapper_f, init_guess, bounds=bounds, method=method, options=options)
        finish = time()
        if not pe_result.success:
            print('Fail: estimation did not converge; exiting.')
            return None

        if self._verbose >= 1:
            print("Complete: OLS estimation using %s took %.2f CPU seconds to complete." % (method, finish - start))
        if self._verbose >= 2:
            print("The estimation took a total of %d function evaluations, %d used for numerical estimation of the "
                  "Jacobian using forward finite differences." % (pe_result.nfev, pe_result.nfev - pe_result.nit - 1))

        try:
            pe_info_mat = pe_result.jac.transpose().dot(pe_result.jac)
            pe_param_covar = np.linalg.inv(pe_info_mat)
            if self._verbose >= 1:
                print("Estimated Parameter Covariance")
                print(pe_param_covar)
        except np.linalg.LinAlgError:
            print("Estimated information matrix of estimation is singular, suggesting that not all model parameters"
                  " are estimable using given data. Consider doing an estimability study.")

        if update_parameters:
            self.model_parameters = pe_result.x
            if self._verbose >= 2:
                print('Nominal parameter value in model updated.')

        if save_result:
            case_path = getcwd()
            today = datetime.now()
            result_dir = case_path + "/" + str(today.date()) + "_at_" + str(today.hour) + "-" + str(
                today.minute) + "-" + str(today.second) + "_full_model_pe_results/"
            makedirs(result_dir)
            with open(result_dir + "result_file.pkl", "wb") as file:
                dump(pe_result, file)
            if self._verbose >= 2:
                print('Parameter estimation result saved to: %s.' % result_dir)

        return pe_result

    def estimate_parameters_branch(self, init_guess, bounds, method='trf', update_parameters=False, save_result=True):
        assert self.data is not None, "No data is put in, do not forget to add it in."
        if self._verbose >= 1:
            lsq_verbose = True
        else:
            lsq_verbose = False
        if self._verbose >= 1:
            print("Solving parameter estimation...")
        start = time()
        pe_result = least_squares(self.get_residuals, init_guess, bounds=bounds, method=method, verbose=lsq_verbose)
        finish = time()
        if pe_result.status not in (1, 2, 3, 4):
            print('Fail: estimation did not converge; exiting.')
            return None

        if self._verbose >= 1:
            print("Complete: estimation using %s took %d function evaluations, and %.2f CPU seconds to complete." % (
                method, pe_result.nfev, finish - start))
        if self._verbose >= 2:
            print("%d instances of Jacobian numerical estimation were completed, they took %d number of function "
                  "evaluations." % (pe_result.njev, pe_result.njev * self.n_theta))

        try:
            pe_info_mat = pe_result.jac.transpose().dot(pe_result.jac)
            pe_param_covar = np.linalg.inv(pe_info_mat)
            if self._verbose >= 1:
                print("Estimated Parameter Covariance")
                print(pe_param_covar)
        except np.linalg.LinAlgError:
            print("Estimated information matrix of estimation is singular, suggesting that not all model parameters "
                  "are estimable using given data. Consider doing an estimability study.")

        if update_parameters:
            self.model_parameters = pe_result.x
            if self._verbose >= 2:
                print('Nominal parameter value in model updated.')

        if save_result:
            case_path = getcwd()
            today = datetime.now()
            result_dir = case_path + "/" + str(today.date()) + "_at_" + str(today.hour) + "-" + str(
                today.minute) + "-" + str(today.second) + "_full_model_pe_results/"
            makedirs(result_dir)
            with open(result_dir + "result_file.pkl", "wb") as file:
                dump(pe_result, file)
            if self._verbose >= 2:
                print('Parameter estimation result saved to: %s.' % result_dir)

        return pe_result

    def design_experiment(self, criterion, optimize_sampling_times=False, package="cvxpy", optimizer=None,
                          opt_options=None, p_0=None, write=True, plot=False, **kwargs):
        self._optimization_package = package
        self._optimizer = optimizer

        if self._optimization_package is 'scipy':
            self._transform_p = True  # affects self.eval_fim
            if optimizer is None:
                self._optimizer = 'bfgs'

        self._opt_sampling_times = optimize_sampling_times  # affects self.eval_F and plotting
        self.n_p = self.n_cand  # n_p is the length of experimental efforts: total row length of F
        if optimize_sampling_times:
            if self._opt_sampling_times:
                self.n_p = self.n_p * self.n_sample_time

        """ set initial guess for optimal experimental efforts, if none given, equal efforts for all candidates """
        if p_0 is None:
            p_0 = np.array([1 / self.n_p for _ in range(self.n_p)])
            self.p_candidates = p_0
        else:
            assert isinstance(p_0, (np.ndarray, list)), 'Initial guess for effort must be a 1D list or numpy array.'
            assert len(p_0) == self.n_p, 'Length of initial guess must be equal to number of candidates (if sampling ' \
                                         'times not optimized, or equal to the number of candidates multiplied by ' \
                                         'largest number of sampling times (if sampling times to be optimized).'
            self.p_candidates = p_0

        if self._verbose >= 2:
            opt_verbose = True
        else:
            opt_verbose = False

        """ main codes """
        if self._verbose >= 1:
            print("Solving OED problem...")
        start = time()
        if self._optimization_package == "scipy":
            """ setting default scipy optimizer options """
            if opt_options is None:
                opt_options = {"disp": opt_verbose}

            opt_result = minimize(fun=criterion, x0=p_0,
                                  method=optimizer, options=opt_options)
            p_opt = opt_result.x
            p_opt = p_opt ** 2
            p_opt = p_opt / sum(p_opt)
            self.p_candidates = p_opt
            opt_fun = opt_result.fun
        elif self._optimization_package == "cvxpy":
            p = cp.Variable(self.n_p, nonneg=True)
            p.value = self.p_candidates
            p_cons = [cp.sum(p) == 1]
            obj = cp.Maximize(criterion(p))
            problem = cp.Problem(obj, p_cons)
            opt_fun = problem.solve(verbose=opt_verbose, solver=optimizer, **kwargs)
            self.p_candidates = p.value
        else:
            print("Unrecognized package, reverting to default: scipy.")
            opt_fun = None  # optional line to remove warning in PyCharm to follow PEP8
            self.design_experiment(criterion, optimize_sampling_times, "scipy",
                                   optimizer, opt_options, p_0, write, plot)
        finish = time()

        """ report status and performance """
        self._optimization_time = finish - start
        if not self._sensitivity_analysis_done:
            self._optimization_time -= self._sensitivity_analysis_time
        if self._verbose:
            print(
                "Solved: sensitivity analysis took %.2f CPU seconds; the optimizer '%s' interfaced via the"
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
            "optimal_efforts": self.p_candidates,
            "criterion_value": self._criterion_value,
            "optimizer": self._optimizer
        }
        self.oed_result = oed_result
        if write:
            self.write_oed_result()

        return oed_result

    def design_exact_experiment(self, criterion, n_exp, optimize_sampling_times=True, package="cvxpy", optimizer=None,
                                opt_options=None, p_0=None, write=True, plot=False):
        if self._optimization_package is 'scipy':
            self._transform_p = True  # affects self.eval_fim

        self._optimization_package = package
        self._optimizer = optimizer
        self._opt_sampling_times = optimize_sampling_times  # affects self.eval_F and plotting

        """ compute length of the experimental effort vector, depending on whether sampling times are optimized """
        if optimize_sampling_times:
            self.n_p = self.n_cand * self.n_sample_time
        else:
            self.n_p = self.n_cand

        """ set initial guess for optimal experimental efforts, if none given, equal efforts for all candidates """
        if p_0 is None:
            # p_0 = np.array([1 / self.n_p for _ in range(self.n_p)])
            p_0 = np.zeros(self.n_p)
            p_0[0] = n_exp
            self.p_candidates = p_0

        if self._verbose >= 2:
            opt_verbose = True
        else:
            opt_verbose = False

        """ main codes """
        if self._verbose >= 1:
            print("Solving OED problem...")
        start = time()
        if self._optimization_package == "scipy":
            raise NotImplementedError('Using scipy for Exact Designs is not implemented.')
            # """ setting default scipy optimizer options """
            #
            # if opt_options is None:
            #     opt_options = {"disp": opt_verbose}
            #
            # opt_result = minimize(fun=criterion, x0=p_0,
            #                       method=optimizer, options=opt_options)
            # p_opt = opt_result.x
            # p_opt = p_opt ** 2
            # p_opt = p_opt / sum(p_opt)
            # self.p_candidates = p_opt
            # opt_fun = opt_result.fun
        elif self._optimization_package == "cvxpy":
            p = cp.Variable(self.n_p, integer=True)
            p_cons = [cp.sum(p) == n_exp, p >= np.zeros_like(p)]
            problem = cp.Problem(cp.Maximize(criterion(p)),
                                 constraints=p_cons)
            probdata, _, _ = problem.get_problem_data()
            cone_dims = probdata['dims']
            cones = {
                "f": cone_dims.zero,
                "l": cone_dims.nonpos,
                "q": cone_dims.soc,
                "ep": cone_dims.exp,
                "s": cone_dims.psd,
            }
            print(cones)
            opt_fun = problem.solve(verbose=opt_verbose, solver=optimizer)
            self.p_candidates = p.value
        else:
            print("Unrecognized package, reverting to default: scipy.")
            opt_fun = None  # optional line to remove warning in PyCharm to follow PEP8
            self.design_experiment(criterion, optimize_sampling_times, "scipy",
                                   optimizer, opt_options, p_0, write, plot)
        finish = time()

        """ report status and performance """
        self._optimization_time = finish - start - self._sensitivity_analysis_time
        if self._verbose:
            print(
                "Solved: sensitivity analysis took %.2f CPU seconds; the optimizer '%s' interfaced via the"
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
            "optimal_efforts": self.p_candidates,
            "criterion_value": self._criterion_value,
            "optimizer": self._optimizer
        }
        self.oed_result = oed_result
        if write:
            self.write_oed_result()

        return oed_result

    def estimability_study(self, base_step=None, step_ratio=None, num_steps=None, estimable_tolerance=0.04,
                           write=False):
        self.estimable_columns = np.array([])

        self.get_sensitivities(base_step=base_step, step_ratio=step_ratio, num_steps=num_steps, normalize=True)

        z = self.sensitivity[:, :, self.measurable_responses, :].reshape(
            self.n_sample_time * self.n_m_res * self.n_cand, self.n_theta)

        column_magnitude = np.sum(np.power(z, 2), axis=0)
        largest_column = np.argmax(column_magnitude)
        estimable_columns = np.array([largest_column])
        finished = False
        while not finished:
            x_l = z[:, estimable_columns]
            z_theta = np.linalg.inv(x_l.T.dot(x_l)).dot(x_l.T).dot(z)
            z_hat = x_l.dot(z_theta)
            r = z - z_hat
            residual_col_mag = np.sum(np.power(r, 2), axis=0)
            largest_column = np.argmax(residual_col_mag)
            self.estimable_columns = np.append(self.estimable_columns, largest_column)
            if residual_col_mag[largest_column] <= estimable_tolerance:
                finished = True

        if write:
            pass

        return self.estimable_columns

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

    # plotting
    def plot_current_design(self, width=None, write=False, dpi=720, quality=95):
        if self._opt_sampling_times:
            self._plot_current_continuous_design_3d(width=width, write=write, dpi=dpi, quality=quality)
        else:
            self._plot_current_continuous_design_2d(width=width, write=write, dpi=dpi, quality=quality)

    def plot_sensitivities(self, absolute=False, draw_legend=True):
        # n_c, n_s_times, n_res, n_theta = self.sensitivity.shape
        fig1 = plt.figure(
            figsize=(1.25 * self.n_theta + 2 / (self.n_theta + 1), 1.25 + 1 * self.n_m_res + 2 / (self.n_m_res + 1)))
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
        for row in range(self.n_m_res):
            for col in range(self.n_theta):
                i += 1
                create_axes = 'axes_%d_%d = fig1.add_subplot(%d, %d, %d)' % (
                    row, col, self.n_m_res, self.n_theta, i)
                exec(create_axes)
                for c, exp_candidate in enumerate(
                        zip(self.ti_controls_candidates, self.tv_controls_candidates,
                            self.sampling_times_candidates)):
                    if absolute:
                        sens = np.abs(self.sensitivity[c, :, self.measurable_responses[row], col])
                    else:
                        sens = self.sensitivity[c, :, self.measurable_responses[row], col]
                    plot_sens = 'axes_%d_%d.plot(exp_candidate[2], sens, "-o", label="Candidate %d")' % (
                        row, col, c + 1)
                    exec(plot_sens)
                    ticklabel = 'axes_%d_%d.ticklabel_format(axis="y", style="sci", scilimits=(0,0))' % (row, col)
                    exec(ticklabel)
                if draw_legend and self.n_cand <= 10:
                    make_legend = 'axes_%d_%d.legend()' % (row, col)
                    exec(make_legend)
        fig1.tight_layout()
        plt.show()

    def plot_all_predictions(self, finer_predictions=True, figsize=(10, 7.5), prediction_resolution=100,
                             markersize=6, fontsize=5, legend=True, legend_size=4, plot_data=False):
        assert self._status is 'ready', 'Initialize the designer first.'
        assert self.response is not None, 'Cannot plot prediction vs data when response is empty, please run and ' \
                                          'store predictions.'
        if plot_data:
            assert self.data is not None, 'Data is empty, cannot plot prediction vs data. Please specify data.'

        """ creating the necessary figures """
        for res in range(self.n_m_res):
            create_fig = 'fig%d = plt.figure(figsize=figsize)' % res
            exec(create_fig)

            n_fig_col = np.floor(np.sqrt(self.n_cand)).astype(int)
            n_fig_row = np.floor(np.sqrt(self.n_cand)).astype(int)

            while n_fig_col * n_fig_row < self.n_cand:
                n_fig_col += 1

            for row in range(n_fig_row):
                for col in range(n_fig_col):
                    draw_subplots = 'axes%d_fig%d = fig%d.add_subplot(n_fig_row, n_fig_col, ' \
                                    'row * n_fig_col + (col + 1) )' % (row * n_fig_col + (col + 1), res, res)
                    exec(draw_subplots)

            x_axis_lim = [
                np.min(self.sampling_times_candidates[~np.isnan(self.sampling_times_candidates)]),
                np.max(self.sampling_times_candidates[~np.isnan(self.sampling_times_candidates)])
            ]

            pred_max = np.max(self.response[:, :, res][~np.isnan(self.response[:, :, res])])
            pred_min = np.min(self.response[:, :, res][~np.isnan(self.response[:, :, res])])
            if plot_data:
                try:
                    data_max = np.max(self.data[:, :, res][~np.isnan(self.data[:, :, res])])
                    data_min = np.min(self.data[:, :, res][~np.isnan(self.data[:, :, res])])
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
                    plot_data = 'data_lines = axes%d_fig%d.plot(sampling_times, self.data[i, :, res], marker="v", ' \
                                'markersize=markersize, fillstyle="none", linestyle="none", label="data")' \
                                % (i + 1, res)
                    exec(plot_data)

                """ plotting predictions """
                plot_predictions = 'pred_lines = axes%d_fig%d.plot(sampling_times, self.response[i, :, ' \
                                   'self.measurable_responses[res]], marker="1", markersize=markersize,' \
                                   'linestyle="none", label="predictions")' % (i + 1, res)
                exec(plot_predictions)

                """ adjusting axes limits, chosen so all subplots include all data and all subplots have same scale """
                set_ylim = 'axes%d_fig%d.set_ylim(y_axis_lim[0] - 0.1 * (y_axis_lim[1] - y_axis_lim[0]),' \
                           ' y_axis_lim[1] + 0.1 * (y_axis_lim[1] - y_axis_lim[0]))' % (i + 1, res)
                exec(set_ylim)
                set_xlim = 'axes%d_fig%d.set_xlim(x_axis_lim[0] - 0.1 * (x_axis_lim[1] - x_axis_lim[0]),' \
                           ' x_axis_lim[1] + 0.1 * (x_axis_lim[1] - x_axis_lim[0]))' % (i + 1, res)
                exec(set_xlim)

                """ setting a smaller fontsize to accommodate for plots with larger number of candidates """
                set_ticks_params = 'axes%d_fig%d.tick_params(axis="both", which="major", labelsize=fontsize)' % (
                    i + 1, res)
                exec(set_ticks_params)
                set_yaxis_offset_fsize = "axes%d_fig%d.yaxis.get_offset_text().set_fontsize(fontsize)" % (i + 1, res)
                exec(set_yaxis_offset_fsize)

                """ give title to each subplot if names for candidates are given """
                if self.candidate_names is not None:
                    set_subplot_title = 'axes%d_fig%d.set_title(self.candidate_names[i], fontsize=fontsize)' % (
                        i + 1, res)
                    try:
                        exec(set_subplot_title)
                    except ValueError:
                        print('Mild warning when plotting: error in labelling candidate name for candidate %d.' % i)

                if legend:
                    draw_legend = 'axes%d_fig%d.legend(prop={"size": legend_size})' % (i + 1, res)
                    exec(draw_legend)
                set_yaxis_offset_fsize = "axes%d_fig%d.yaxis.get_offset_text().set_fontsize(fontsize)" % (i + 1, res)
                exec(set_yaxis_offset_fsize)

        """ make all figures use tight_layout for tidiness """
        for res in range(self.n_m_res):
            tight_layout = 'fig%d.tight_layout()' % res
            exec(tight_layout)

        plt.show()

    # saving, loading, writing
    def load_oed_result(self, result_path):
        raise NotImplementedError

    def create_result_dir(self):
        if self.result_dir is None:
            now = datetime.now()
            self.result_dir = getcwd() + "/"
            self.result_dir = self.result_dir + path.splitext(path.basename(main.__file__))[0] + "_result/"
            self.result_dir = self.result_dir + 'date_%d-%d-%d/' % (now.year, now.month, now.day)
            self.create_result_dir()
        else:
            if path.exists(self.result_dir):
                return
            else:
                makedirs(self.result_dir)

    def write_oed_result(self):
        self.create_result_dir()

        result_file = self.result_dir + "/%s_oed_result_%d.pkl" % (self.oed_result["optimality_criterion"], self.run_no)
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
            self.n_cand,
            self.n_sample_time,
            self.n_res,
            self.n_theta,
            self.ti_controls_candidates,
            self.tv_controls_candidates,
            self.sampling_times_candidates,
            self.measurable_responses,
            self.n_m_res,
            self.model_parameters
        ]

        designer_file = self.result_dir + "/"  + 'state' + "_%d.pkl" % self.run_no
        if path.isfile(designer_file):
            self.run_no += 1
            self.save_state()
        else:
            dill.dump(state, open(designer_file, "wb"))

    def load_state(self, designer_path):
        state = dill.load(open(getcwd() + designer_path, 'rb'))
        self.n_cand                         = state[0]
        self.n_sample_time                  = state[1]
        self.n_res                          = state[2]
        self.n_theta                        = state[3]
        self.ti_controls_candidates         = state[4]
        self.tv_controls_candidates         = state[5]
        self.sampling_times_candidates      = state[6]
        self.measurable_responses           = state[7]
        self.n_m_res                        = state[8]
        self.model_parameters               = state[9]

        self._current_model_parameters = self.model_parameters
        return None

    def load_sensitivity(self, sens_path):
        self.sensitivity = load(open(getcwd() + sens_path, 'rb'))
        return self.sensitivity

    """ parameter estimation """
    def get_residuals(self, model_parameters):
        self.model_parameters = model_parameters

        """ run the model to get predictions """
        self.simulate_all_candidates()
        self.residuals = self.data - np.array(self.response)[:, :, self.measurable_responses]

        return self.residuals[~np.isnan(self.residuals)]  # return residuals where entries are not empty

    """ model linearization """
    def get_sensitivities(self, normalize=False, method='forward', base_step=None, step_ratio=None, num_steps=None,
                          store_predictions=True, plot_analysis_times=False, write=False):

        """ check if model parameters have been changed or not """
        self._check_if_model_parameters_changed()

        """ do sensitivity analysis if not done before or model parameters were changed """
        step_generator = nd.step_generators.MaxStepGenerator(base_step=base_step,
                                                             step_ratio=step_ratio,
                                                             num_steps=num_steps)

        if self.sensitivity is None or self._model_parameters_changed:
            self._sensitivity_analysis_done = False
            if self._verbose >= 1:
                print('Running sensitivity analysis...')
            start = time()
            sens = []
            candidate_sens_times = []
            jacob_fun = nd.Jacobian(fun=self._sensitivity_sim_wrapper, step=step_generator, method=method)
            """ main loop over experimental candidates """
            for i, exp_candidate in enumerate(zip(self.sampling_times_candidates, self.ti_controls_candidates,
                                                  self.tv_controls_candidates)):
                """ specifying current experimental candidate """
                self._ti_controls = exp_candidate[1]
                self._tv_controls = exp_candidate[2]
                self._sampling_times = exp_candidate[0][~np.isnan(exp_candidate[0])]

                self.feval_sensitivity = 0
                single_start = time()
                temp_sens = jacob_fun(self.model_parameters, (normalize or store_predictions))
                finish = time()
                if self._verbose >= 2:
                    print('[Candidate %d/%d]: took %d function evaluations, and %.2f CPU seconds.' %
                          (i + 1, self.n_cand, self.feval_sensitivity, finish - single_start))
                candidate_sens_times.append(finish - single_start)
                """
                bunch of lines to make sure the Jacobian method returns the sensitivity with dims: n_sp, n_res, n_theta
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
                elif self.n_sample_time == 1:
                    if self.n_theta == 1:  # covers case 5: add a new axis in the last dim
                        temp_sens = temp_sens[:, :, np.newaxis]
                    elif self.n_res == 1:  # covers case 2, 6, and 8: add a new axis in the first dim
                        temp_sens = temp_sens[np.newaxis]
                elif self.n_theta == 1:  # covers case 3 and 7
                    temp_sens = np.moveaxis(temp_sens, 0, 1)  # move n_sp to the first dim as needed
                    temp_sens = temp_sens[:, :, np.newaxis]  # create a new axis as the last dim for n_theta
                elif self.n_res == 1:  # covers case 4
                    temp_sens = temp_sens[:, np.newaxis, :]  # create axis in the middle for n_res

                """ appending the formatted sensitivity matrix for each candidate into the final list to be returned """
                sens.append(temp_sens)
            finish = time()
            if self._verbose >= 1:
                print('Sensitivity analysis using numdifftools with forward scheme finite difference took '
                      'a total of %.2f CPU seconds.' % (finish - start))

            # converting sens into a numpy array for optimizing further computations
            sens = np.array(sens)

            # saving current model parameters
            self._current_model_parameters = np.copy(self.model_parameters)
            self._sensitivity_is_normalized = normalize

            # normalizing the sensitivities by multiplying by the nominal parameter values and dividing by responses
            if self._sensitivity_is_normalized:
                assert np.all(self.model_parameters > 0), 'At least one nominal model parameter value is equal to 0, ' \
                                                          'cannot normalize sensitivities. Consider re-estimating ' \
                                                          'your parameters or re-parameterize your model.'
                # normalize parameter values
                sens = np.multiply(sens, self.model_parameters[np.newaxis, np.newaxis, np.newaxis, :])
                if self.responses_scales is None:
                    if self._verbose >= 0:
                        print('Scale for responses not given, using raw prediction values to normalize sensitivities; '
                              'likely to fail (e.g. if responses are near 0). Recommend: provide designer with scale '
                              'info through: "designer.responses_scale = <your_scale_array>."')
                        # normalize response values
                        sens = np.divide(sens, self.response[:, :, :, np.newaxis])
                else:
                    assert isinstance(self.responses_scales, np.ndarray), "Please specify responses_scales as a 1D " \
                                                                          "numpy array."
                    assert self.responses_scales.size == self.n_res, 'Length of responses scales is different from ' \
                                                                     'the total number of responses (includes ' \
                                                                     'those which are measurable and not).)'
                    sens = np.divide(sens, self.responses_scales[np.newaxis, np.newaxis, :, np.newaxis])
            self.sensitivity = sens

            if self._var_n_sampling_time:
                self._pad_sensitivities()

            if write:
                self.create_result_dir()
                self.run_no = 1
                sens_file = self.result_dir + '/sensitivity_%d.pkl' % self.run_no
                while path.isfile(sens_file):
                    self.run_no += 1
                    sens_file = self.result_dir + '/sensitivity_%d.pkl' % self.run_no
                dump(self.sensitivity, open(sens_file, 'wb'))

            if plot_analysis_times:
                plt.plot(np.arange(1, self.n_cand+1, step=1), candidate_sens_times)
                plt.show()

        self._sensitivity_analysis_done = True
        return self.sensitivity

    """ optimal experiment design """
    def eval_F(self, base_step=None, step_ratio=None, num_steps=None, normalize=False, store_predictions=False):
        """ should return F as a 2D numpy array.
        N_sampling.N_candidates times N_theta matrix if sampling time is optimized;
        N_candidates times N_theta matrix if sampling time not optimized. """

        self.n_p = self.n_cand  # n_p is the length of experimental efforts: total row length of F
        if self._opt_sampling_times:
            self.n_p = self.n_p * self.n_sample_time

        " always get sensitivity, but get_sensitivities will only redo the analysis if needed "
        start = time()
        self.get_sensitivities(base_step=base_step, step_ratio=step_ratio, num_steps=num_steps, normalize=normalize,
                               store_predictions=store_predictions)
        finish = time()
        self._sensitivity_analysis_time = finish - start

        if self._verbose >= 3:
            print('Reshaping sensitivity to obtain F.')
        """ main line for obtaining F from reshaping of the sensitivities """

        start = time()
        F = np.nansum(self.sensitivity[:, :, self.measurable_responses, :], axis=2)  # sum information b/w responses
        if not self._opt_sampling_times:
            F = np.nansum(F, axis=1)  # sum all information matrices between sampling times
        F = np.reshape(F, newshape=(self.n_p, self.n_theta))
        self.F = F
        finish = time()

        if self._verbose >= 3:
            print('Reshaping took %.3f CPU microseconds.' % ((finish - start) * 1e6))

        return F

    def eval_fim(self, p=None):
        """ optional line that will replace current experimental efforts when given """
        if p is not None:
            self.p_candidates = p

        self._check_if_model_parameters_changed()

        """ check if F is available, if yes, don't re-evaluate """
        if self.F is None or self._model_parameters_changed:
            self.eval_F()

        """ transform the efforts if needed """
        if self._transform_p:
            self._apply_p_transform()

        """ main codes for evaluation """
        if self._optimization_package is 'scipy':
            self.p_candidates = np.diag(self.p_candidates)
            self.fim = self.F.T.dot(self.p_candidates.dot(self.F))
        elif self._optimization_package is 'cvxpy':
            self.p_candidates = cp.diag(self.p_candidates)
            self.fim = self.F.T * self.p_candidates * self.F
            # self.fim = self.F.T.dot(self.p_candidates).dot(self.F)
        return self.fim

    def d_opt_criterion(self, p):
        self._check_if_effort_transformation_needed_for_chosen_package()

        """ evaluate fim at current experimental effort """
        self.eval_fim(p)

        """ evaluate the criterion """
        if self._optimization_package is 'scipy':
            return -np.prod(np.linalg.slogdet(self.fim))
        elif self._optimization_package is 'cvxpy':
            if self.fim.size == 1:
                return self.fim
            else:
                # self.mono_fim = []
                # for _ in range(self.n_p):
                #     self.mono_fim.append(np.outer(self.F[_, :], self.F[_, :]))
                return cp.log_det(self.fim)
                # return cp.log_det(cp.sum([self.p_candidates[_] * self.mono_fim[_] for _ in range(self.n_p)]))

    def a_opt_criterion(self, p):
        self._check_if_effort_transformation_needed_for_chosen_package()

        if self._optimization_package is 'scipy':
            self.eval_fim(p)
            return np.trace(np.linalg.inv(self.fim))
        elif self._optimization_package is 'cvxpy':
            raise NotImplementedError(
                'A-optimal is not a convex optimization problem, optimizers from cvxpy package not available.'
            )

    def e_opt_criterion(self, p):
        self._check_if_effort_transformation_needed_for_chosen_package()

        self.eval_fim(p)
        if self._optimization_package is 'scipy':
            return -np.min(np.linalg.eigvals(self.fim))
        elif self._optimization_package is 'cvxpy':
            return -cp.lambda_min(self.fim)

    # prediction-oriented information
    def eval_pvar(self, x):
        raise NotImplementedError

    def eval_det_pvar(self, x):
        raise NotImplementedError

    def average_det_pvar(self):
        raise NotImplementedError

    def max_det_pvar(self):
        raise NotImplementedError

    # prediction-oriented information log determinant versions
    def eval_log_det_pvar(self, x):
        raise NotImplementedError

    def average_log_det_pvar(self):
        raise NotImplementedError

    def max_log_det_pvar(self):
        raise NotImplementedError

    def gg_opt_criterion(self, p):
        raise NotImplementedError

    def log_gg_opt_criterion(self, p):
        raise NotImplementedError

    def log_ig_opt_criterion(self, p):
        raise NotImplementedError

    """ private methods """
    def _sensitivity_sim_wrapper(self, theta_try, store_responses=False):
        response = self._simulate_internal(self._ti_controls, self._tv_controls, theta_try, self._sampling_times)
        self.feval_sensitivity += 1
        """ store responses whenever required, and model parameters are the same as current model's """
        if store_responses and np.allclose(theta_try, self.model_parameters):
            self._current_response = response
            self._store_current_response()
        return response

    def _apply_p_transform(self):
        assert self.p_candidates is not None, "UNKNOWN ERROR: trying to transform efforts but none have been specified."
        if not isinstance(self.p_candidates, np.ndarray):
            self.p_candidates = np.array(self.p_candidates)
        self.p_candidates = self.p_candidates ** 2
        self.p_candidates = self.p_candidates / np.sum(self.p_candidates)

        return self.p_candidates

    def _check_if_model_parameters_changed(self):
        if self._current_model_parameters is None:
            self._current_model_parameters = np.empty(1)
        if np.allclose(self.model_parameters, self._current_model_parameters):
            self._model_parameters_changed = False
        else:
            self._model_parameters_changed = True

    def _check_if_effort_transformation_needed_for_chosen_package(self):
        if self._optimization_package is 'scipy':
            self._transform_p = True
        elif self._optimization_package is 'cvxpy':
            self._transform_p = False

    def _plot_current_continuous_design_2d(self, width=None, write=False, dpi=720, quality=95):
        if self._verbose >= 2:
            print("Plotting current continuous design.")

        if width is None:
            width = 0.7

        p = self.p_candidates.reshape([self.n_cand])

        x = np.arange(
            start=1,
            stop=self.n_cand + 1,
        )
        fig1 = plt.figure(figsize=(15, 7))
        fig1.tight_layout()
        axes1 = fig1.add_subplot(111)
        axes1.bar(x, p, width=width)
        axes1.set_xticks(x)
        axes1.set_ylim([0, 1])
        axes1.set_yticks(np.linspace(0, 1, 11))

        if write:
            self.create_result_dir()
            figname = 'fig_%s_design_%d.png' % (self.oed_result["optimality_criterion"], self.run_no)
            figfile = self.result_dir + figname
            while path.isfile(figfile):
                self.run_no += 1
                figname = 'fig_%s_design_%d.png' % (self.oed_result["optimality_criterion"], self.run_no)
                figfile = self.result_dir + figname
            fig1.savefig(fname=figfile, dpi=dpi, quality=quality)
            self.run_no = 1

        plt.show()

    def _plot_current_continuous_design_3d(self, width=None, write=False, dpi=720, quality=95):
        if self._verbose >= 2:
            print("Plotting current continuous design.")

        if width is None:
            width = 0.3

        p = self.p_candidates.reshape([self.n_cand, self.n_sample_time])

        sampling_time_scale = np.nanmin(np.diff(self.sampling_times_candidates, axis=1))

        fig1 = plt.figure(figsize=(12, 8))
        fig1.tight_layout()
        axes1 = fig1.add_subplot(111, projection='3d')
        for c in range(self.n_cand):
            y = self.sampling_times_candidates[c]

            x = np.array([c + 1] * self.n_sample_time) - width / 2
            x = x[~np.isnan(y)]
            z = np.zeros(self.n_sample_time)
            z = z[~np.isnan(y)]

            dx = width
            dy = width * sampling_time_scale * 0.7
            dz = p[c, :]
            dz = dz[~np.isnan(y)]

            y = y[~np.isnan(y)]

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
        xticks = np.arange(1, self.n_cand + 1, step=np.round(self.n_cand / 10).astype(int))
        if self.n_cand not in xticks:
            xticks = np.append(xticks, self.n_cand)
        axes1.set_xticks(xticks)

        axes1.set_ylabel('Sampling Times')

        axes1.set_zlabel('Experimental Effort')
        axes1.set_zlim([0, 1])
        axes1.set_zticks(np.linspace(0, 1, 11))

        if write:
            self.create_result_dir()
            figname = 'fig_%s_design_%d.png' % (self.oed_result["optimality_criterion"], self.run_no)
            figfile = self.result_dir + figname
            while path.isfile(figfile):
                self.run_no += 1
                figname = 'fig_%s_design_%d.png' % (self.oed_result["optimality_criterion"], self.run_no)
                figfile = self.result_dir + figname
            fig1.savefig(fname=figfile, dpi=dpi, quality=quality)
            self.run_no = 1

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
        self.sampling_times_candidates = np.array(self.sampling_times_candidates.tolist())
        return self.sampling_times_candidates

    def _pad_sensitivities(self):
        sens = self.sensitivity
        for i, row in enumerate(sens):
            diff = self.n_sample_time - row.shape[0]
            self.sensitivity[i] = np.pad(sens[i][:, ], pad_width=((0, diff), (0, 0), (0, 0)),
                                         mode='constant', constant_values=np.nan)
        self.sensitivity = self.sensitivity.tolist()
        self.sensitivity = np.array(self.sensitivity)
        return self.sensitivity

    def _store_current_response(self):
        """ padding responses to accommodate for missing sampling times """
        start = time()
        if self.response is None:  # if it is the first response to be stored, initialize response list
            self.response = []
        if self.n_sample_time is 1:
            self._current_response = self._current_response[np.newaxis, :]
        elif self.n_res is 1:
            self._current_response = self._current_response[:, np.newaxis]

        if self._var_n_sampling_time:
            self._current_response = np.pad(
                self._current_response,
                pad_width=((0, self.n_sample_time - self._current_response.shape[0]), (0, 0)),
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
            print('Storing response took %.6f CPU miliseconds.' % (1000 * (end-start)))
        return self.response

    def _residuals_wrapper_f(self, model_parameters):
        residuals = self.get_residuals(model_parameters)
        return np.inner(residuals, residuals)

    def _simulate_internal(self, ti_controls, tv_controls, theta, sampling_times):
        raise SyntaxError("Make sure you have initialized the designer, and specified the simulate function correctly.")

    def _initialize_simulate_function(self):
        if self._model_package == 'pyomo':
            self._simulate_internal = lambda ti_controls, tv_controls, theta, sampling_times: \
                self.simulate(self.model, self.simulator, ti_controls, tv_controls, theta, sampling_times)
        elif self._model_package == 'non-pyomo':
            self._simulate_internal = self.simulate
        else:
            raise SyntaxError('Cannot initialize simulate function properly, check your syntax.')
