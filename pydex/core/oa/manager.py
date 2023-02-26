from pydex.core.oa.primal import OAPrimalProblem
from pydex.core.oa.master import OAMasterProblem
from matplotlib import pyplot as plt
from time import time
import numpy as np


class OAManager:
    def __init__(self, criterion, eval_fim):
        self.master_problem = OAMasterProblem()
        self.primal_problem = OAPrimalProblem()
        self.primal_problem.criterion = criterion
        self.primal_problem.eval_fim = eval_fim

        # store CED-apportionment solution
        self.apportioned_obj_val = None
        self.apportioned_effort = None
        self.apportioned_obj_grad = None

        self.ced_efforts = None
        self.ced_obj = None
        self.ced_obj_grad = None

        self.converged = None
        self.iteration_no = None
        self.N_exp = None
        self.sensitivities = None
        self.atomics = None

        self.n_c = None
        self.n_spt = None
        self.y0 = None

        self.yk = None
        self.UBDk = None
        self.LBDk = None
        self.f_yk = {}
        self.gradf_yk = {}

        self.singular_master = None
        self.n_singular_masters = None

        self.final_effort = None
        self.final_obj = None
        self.oa_gain_assessed = False

        self.iteration_time = None
        self.overall_time = None
        self.iteration_start_time = None

    def analytic_d_logdetfim(self, efforts):
        M = np.sum(efforts[:, None, None, None] * self.atomics, axis=0)[0]
        d_detfim = np.empty(efforts.shape[0])
        for i, (Ai, pi, qi) in enumerate(zip(self.atomics, efforts, self.sensitivities[:, 0, 0, :])):
            Ni = np.linalg.solve(M, Ai[0])
            d_detfim[i] = np.trace(Ni)
        return -d_detfim

    def solve(self, n_exp, y0, atol=None, rtol=1e-3, assess_potential_oa_gain=False, ced_efforts=None, ced_obj=None, apportioned_effort=None, apportioned_obj_val=None, draw_progress=True, singular_tol=None, max_iters=1e5, MIP_solver=None):
        # TODO: MIP pool solution
        # TODO: "master" master problem
        # TODO: heuristic, try keeping only optimal support from CED to solve MINLP as additional info
        if MIP_solver is None:
            MIP_solver = "GUROBI"
        self.master_problem.solver = MIP_solver
        self.apportioned_effort = apportioned_effort
        self.apportioned_obj_val = apportioned_obj_val
        self.ced_efforts = ced_efforts
        self.ced_obj = ced_obj
        self.N_exp = n_exp
        self.y0 = y0
        self.converged = False
        self.iteration_no = 0
        self.n_singular_masters = 0
        self.iteration_time = 0
        self.overall_time = 0
        start_time = time()
        self.iteration_start_time = time()
        if draw_progress:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        while not self.converged:
            """ Primal problem: log_det evaluation only """
            if self.iteration_no == 0:
                self.primal_problem.efforts = self.y0
                f0 = self.primal_problem.compute_criterion_value()
                self.yk = {0: self.y0}
                self.UBDk = {0: f0}
                self.LBDk = {0: -np.inf}
            else:
                self.primal_problem.efforts = self.yk[self.iteration_no]
                #########################################################################
                # check if singular matrix is obtained from master problem's solution
                #########################################################################
                # use rank method
                if True:
                    fim = np.sum(self.yk[self.iteration_no][:, :, None, None] * self.atomics, axis=(0, 1))
                    fim_rank = np.linalg.matrix_rank(fim, tol=singular_tol)
                    if fim_rank < fim.shape[0]:
                        fk = np.inf
                        self.singular_master = True
                    else:
                        fk = self.primal_problem.compute_criterion_value()
                        self.singular_master = False
                #########################################################################
                # use objective function method
                #########################################################################
                else:
                    fk = self.primal_problem.compute_criterion_value()
                    if fk == np.inf or fk == -np.inf:
                #########################################################################
                        self.singular_master = True
                    else:
                        self.singular_master = False
                if self.singular_master:
                    print(
                        "Singular master solution found, reusing previous iteration "
                        "to linearize master (no valid option)"
                    )
                    self.singular_master = True
                    self.n_singular_masters += 1
                    # add Gomory's cut to the master and proceed to solve master
                    self.master_problem.add_gomorys_cut(
                        self.yk[self.iteration_no],
                        lb=0,
                        ub=self.N_exp,
                    )
                    # yk for iteration with singular matrix is taken as previous
                    self.yk[self.iteration_no] = self.yk[self.iteration_no - 1]
                # store upper_bound
                self.UBDk[self.iteration_no] = fk

            self.overall_time += time() - start_time
            print(f"[{time() - start_time:.2f} s: Iteration {self.iteration_no} Done]".center(100, "="))
            print(f"Iteration {self.iteration_no} Upper bound: {self.UBDk[self.iteration_no]}")
            print(f"Iteration {self.iteration_no} Tightest Upper bound: {np.min(list(self.UBDk.values()))}")
            print(f"Iteration {self.iteration_no} Lower bound: {self.LBDk[self.iteration_no]}")
            print(f"Iteration {self.iteration_no} Tightest Lower bound: {np.max(list(self.LBDk.values()))}")
            print(f"Iteration {self.iteration_no} Completed in {time() - self.iteration_start_time:.4f} s")
            if draw_progress:
                axes.scatter(
                    self.iteration_no,
                    self.UBDk[self.iteration_no],
                    marker="H",
                    color="tab:red",
                )
                axes.scatter(
                    self.iteration_no,
                    self.LBDk[self.iteration_no],
                    marker="H",
                    color="tab:blue",
                )
                axes.scatter(
                    self.iteration_no,
                    np.min(list(self.UBDk.values())),
                    edgecolor="tab:red",
                    facecolor="none",
                    marker="o",
                    s=100,
                )
                axes.scatter(
                    self.iteration_no,
                    np.max(list(self.LBDk.values())),
                    edgecolor="tab:blue",
                    facecolor="none",
                    marker="o",
                    s=100,
                )
            """ check if LBD and UBD sufficiently close """
            self.converged = np.isclose(np.max(list(self.LBDk.values())), np.min(list(self.UBDk.values())), rtol=rtol)
            if atol is not None:
                self.converged = self.converged or np.isclose(np.max(list(self.LBDk.values())), np.min(list(self.UBDk.values())), atol=atol)
            self.converged = self.converged or self.iteration_no >= max_iters
            if self.converged or np.max(list(self.LBDk.values())) > np.min(list(self.UBDk.values())):  # TODO: add maximum number of iterations # TODO: add relative and absolute termination option
                print(f"Convergence achieved, with LBD: {np.max(list(self.LBDk.values())):.4f} and UBD {np.min(list(self.UBDk.values())):.4f}".center(100, "="))
                print(f"Objective function (smaller the better): {self.UBDk[self.iteration_no]:.4f}")
                print(f"Total number of iterations: {self.iteration_no}")
                print(f"Total number of singular masters: {self.n_singular_masters}")
                print(f"".center(100, "="))
                if draw_progress:
                    axes.set_xlabel("Iteration")
                    axes.set_ylabel(f"{self.primal_problem.criterion.__name__}")
                    fig.tight_layout()
                opt_sol_idx = np.argmin(list(self.UBDk.values()))
                self.final_effort = self.yk[opt_sol_idx]
                self.final_obj = list(self.UBDk.values())[opt_sol_idx]
                return self.final_obj

            """ Master Problem: MIP after linearization """
            # add CED solutions as a linear cut in the first iteration, both apportioned and continuous-effort are used
            self.iteration_start_time = time()
            if self.iteration_no == 0:
                self.master_problem.create_cvxpy_problem(self.y0, self.N_exp)
                if self.ced_efforts is not None and self.ced_obj is not None:
                    self.ced_obj_grad = self.analytic_d_logdetfim(
                        self.ced_efforts[:, 0],
                    )
                    self.master_problem.add_linearized_obj_cut(
                        yk=self.ced_efforts,
                        f_yk=self.ced_obj,
                        gradf_yk=self.ced_obj_grad,
                    )

            # check if singular matrix obtained previous iteration
            if self.singular_master:
                pass  # no linearized obj cut added if singular (Gomory's cut instead)
            else:
                # add linearized cut(s)
                self.f_yk[self.iteration_no] = self.UBDk[self.iteration_no]
                self.gradf_yk[self.iteration_no] = self.analytic_d_logdetfim(
                    self.yk[self.iteration_no][:, 0],
                )
                self.master_problem.add_linearized_obj_cut(
                    yk=self.yk[self.iteration_no],
                    f_yk=self.f_yk[self.iteration_no],
                    gradf_yk=self.gradf_yk[self.iteration_no],
                )
            self.master_problem.solve()
            if not self.oa_gain_assessed and self.LBDk[self.iteration_no] != -np.inf and assess_potential_oa_gain:
                confirmation = input(
                    f"The upper limit on the improvement in information criterion "
                    f"(lower is better) is {self.UBDk[0] - self.LBDk[self.iteration_no]}"
                    f", where the UBD of the information criterion is {self.UBDk[0]} "
                    f"and LBD is {self.LBDk[self.iteration_no]}. Type 'Y' to proceed "
                    f"with OA: \n"
                )
                if confirmation != "Y":
                    print(
                        f"User deemed insufficient potential gain with OA. Returning "
                        f"apportioned solution."
                    )
                    self.final_effort = self.yk[0]
                    self.final_obj = self.UBDk[0]
                    return self.UBDk[0]
                self.oa_gain_assessed = True
            self.iteration_no += 1
            self.yk[self.iteration_no] = self.master_problem.efforts.value
            self.LBDk[self.iteration_no] = self.master_problem.eta.value


if __name__ == '__main__':
    N_exp = 6
    reso = 11j

    x1, x2 = np.mgrid[-1:1:reso, -1:1:reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T

    """ Using Pydex to get a rounded design as initial guess for binary efforts """
    from pydex.core.designer import Designer


    def simulate(ti_controls, model_parameters):
        return np.array([
            model_parameters[0] +
            model_parameters[1] * ti_controls[0] +
            model_parameters[2] * ti_controls[1] +
            model_parameters[3] * ti_controls[0] ** 2 +
            model_parameters[4] * ti_controls[1] ** 2 +
            model_parameters[5] * ti_controls[0] * ti_controls[1]
        ])

    designer = Designer()
    designer.simulate = simulate
    designer.ti_controls_candidates = X
    designer.model_parameters = np.ones(6)
    designer.start_logging()
    designer.initialize(verbose=2)
    designer.design_experiment(
        designer.d_opt_criterion,
        n_exp=N_exp,
    )
    designer.print_optimal_candidates()
    designer.plot_optimal_controls(non_opt_candidates=True)
    designer.stop_logging()
    designer.show_plots()
