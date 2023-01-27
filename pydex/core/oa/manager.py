from pydex.core.oa.primal import OAPrimalProblem
from pydex.core.oa.master import OAMasterProblem
from matplotlib import pyplot as plt
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

    def analytic_d_logdetfim(self, efforts):
        M = self.sensitivities[:, 0, 0, :].T @ np.diag(efforts) @ self.sensitivities[:, 0, 0, :]
        det_M = np.linalg.det(M)
        d_detfim = np.empty(efforts.shape[0])
        for i, (pi, qi) in enumerate(zip(efforts, self.sensitivities[:, 0, 0, :])):
            Mi = qi[:, None] @ qi[:, None].T
            Ni = np.linalg.inv(M) @ Mi
            tr_Ni = np.trace(Ni)
            d_detfim[i] = det_M * tr_Ni
        return -(1 / det_M) * d_detfim

    def solve(self, n_exp, y0, ced_efforts=None, ced_obj=None, apportioned_effort=None, apportioned_obj_val=None, draw_progress=True):
        self.apportioned_effort = apportioned_effort
        self.apportioned_obj_val = apportioned_obj_val
        self.ced_efforts = ced_efforts
        self.ced_obj = ced_obj
        self.N_exp = n_exp
        self.y0 = y0
        self.converged = False
        self.iteration_no = 0
        self.n_singular_masters = 0
        if draw_progress:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        while not self.converged:
            """ Primal problem: log_det evaluation only """
            if self.iteration_no == 0:
                self.primal_problem.efforts = y0
                f0 = self.primal_problem.compute_criterion_value()
                fk = f0
                self.yk = {0: y0}
                self.UBDk = {0: f0}
                self.LBDk = {0: -np.inf}
            else:
                self.primal_problem.efforts = self.yk[self.iteration_no]
                #########################################################################
                # check if singular matrix is obtained from master problem's solution
                #########################################################################
                # use rank method
                if False:
                    fim = self.primal_problem.eval_fim(self.primal_problem.efforts)
                    fim_rank = np.linalg.matrix_rank(fim)
                    if fim_rank <= fim.shape[0]:
                        fk = np.inf
                #########################################################################
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
                            ub=n_exp,
                        )
                        # yk for iteration with singular matrix is taken as previous
                        self.yk[self.iteration_no] = self.yk[self.iteration_no - 1]
                    else:
                        # check if primal problem yields improved upper bound
                        if fk > np.max(list(self.UBDk.values())):
                            print(
                                "No improvement in upper bound observed"
                            )
                        self.singular_master = False
                #########################################################################
                # use objective function method
                #########################################################################
                else:
                    fk = self.primal_problem.compute_criterion_value()
                    if fk == np.inf or fk == -np.inf:
                #########################################################################
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
                            ub=n_exp,
                        )
                        # yk for iteration with singular matrix is taken as previous
                        self.yk[self.iteration_no] = self.yk[self.iteration_no - 1]
                    else:
                        # check if primal problem yields improved upper bound
                        if fk > np.max(list(self.UBDk.values())):
                            print(
                                "No improvement in upper bound observed"
                            )
                        self.singular_master = False
                # store upper_bound
                self.UBDk[self.iteration_no] = fk

            print(f"Iteration {self.iteration_no} Done".center(100, "="))
            print(f"Iteration {self.iteration_no} Upper bound: {self.UBDk[self.iteration_no]}")
            print(f"Iteration {self.iteration_no} Tightest Upper bound: {np.min(list(self.UBDk.values()))}")
            print(f"Iteration {self.iteration_no} Lower bound: {self.LBDk[self.iteration_no]}")
            print(f"Iteration {self.iteration_no} Tightest Lower bound: {np.max(list(self.LBDk.values()))}")
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
            self.converged = np.isclose(np.max(list(self.LBDk.values())), np.min(list(self.UBDk.values())), rtol=0.01)
            if self.converged or np.max(list(self.LBDk.values())) > np.min(list(self.UBDk.values())):
                print(f"Convergence achieved, with LBD: {np.max(list(self.LBDk.values())):.4f} and UBD {np.min(list(self.UBDk.values())):.4f}".center(100, "="))
                print(f"Objective function (smaller the better): {self.UBDk[self.iteration_no]:.4f}")
                print(f"Final Effort: {self.yk[self.iteration_no]}")
                print(f"Total number of iterations: {self.iteration_no}")
                print(f"Total number of singular masters: {self.n_singular_masters}")
                print(f"".center(100, "="))
                if draw_progress:
                    axes.set_xlabel("Iteration")
                    axes.set_ylabel(f"{self.primal_problem.criterion.__name__}")
                    fig.tight_layout()
                return self.UBDk[self.iteration_no]

            """ Master Problem: MIP after linearization """
            # add CED solutions as a linear cut in the first iteration, both apportioned and continuous-effort are used
            if self.iteration_no == 0:
                self.master_problem.create_cvxpy_problem(y0, self.N_exp)
                # if self.apportioned_obj_val is not None and self.apportioned_effort is not None:
                #     self.apportioned_obj_grad = self.analytic_d_logdetfim(
                #         self.apportioned_effort[:, 0],
                #     )
                #     self.master_problem.add_linearized_obj_cut(
                #         yk=self.apportioned_effort,
                #         f_yk=self.apportioned_obj_val,
                #         gradf_yk=self.apportioned_obj_grad,
                #     )
                if ced_efforts is not None and ced_obj is not None:
                    self.ced_obj_grad = self.analytic_d_logdetfim(
                        ced_efforts[:, 0],
                    )
                    self.master_problem.add_linearized_obj_cut(
                        yk=self.ced_efforts,
                        f_yk=self.ced_obj,
                        gradf_yk=self.ced_obj_grad,
                    )

            # check if singular matrix obtained previous iteration
            if self.singular_master:
            # if fk == np.inf or fk == -np.inf:
                pass  # no linearized obj cut added if singular (Gomory's cut instead)
            else:
                # add linearized cut(s)
                self.f_yk[self.iteration_no] = self.UBDk[self.iteration_no]
                self.gradf_yk[self.iteration_no] = self.analytic_d_logdetfim(
                    self.yk[self.iteration_no][:, 0]
                )
                self.master_problem.add_linearized_obj_cut(
                    yk=self.yk[self.iteration_no],
                    f_yk=self.f_yk[self.iteration_no],
                    gradf_yk=self.gradf_yk[self.iteration_no],
                )
            self.master_problem.solve()
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
    designer.print_optimal_candidates()  # TODO: UPDATE FOR DISCRETE DESIGN WITH OA SOLVER
    designer.plot_optimal_controls(non_opt_candidates=True)
    designer.stop_logging()
    designer.show_plots()
