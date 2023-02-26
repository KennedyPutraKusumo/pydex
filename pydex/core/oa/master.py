import cvxpy as cp
import numpy as np


class OAMasterProblem:
    def __init__(self):
        self.cvxpy_problem = None
        self.eta = None
        self.objective = None
        self.efforts = None
        self.constraints = None
        self.solver = None

        self.n_gomory_cuts = None
        self.gomory_cuts = None
        self.gomory_w = None  # continuous variable for linear gomory integer cuts (see Appendix 1 of https://doi.org/10.1080/10556788.2019.1641498)
        self.gomory_v = None

    def create_cvxpy_problem(self, y0, n_exp):
        self.eta = cp.Variable()
        self.objective = cp.Minimize(self.eta)
        self.efforts = cp.Variable(shape=y0.shape, integer=True)
        self.constraints = [cp.sum(self.efforts) <= n_exp]
        self.constraints += [self.efforts >= 0]

    def add_linearized_obj_cut(self, yk, f_yk, gradf_yk):
        self.constraints += [
            f_yk + gradf_yk @ (self.efforts - yk) <= self.eta
        ]

    def add_gomorys_cut(self, yk, lb, ub):
        # from Appendix 1 of https: // doi.org / 10.1080 / 10556788.2019.1641498
        if self.gomory_cuts is None:
            self.gomory_cuts = {}
            self.n_gomory_cuts = 1
        else:
            self.n_gomory_cuts += 1
        y_at_lb_idx = yk == lb
        y_at_ub_idx = yk == ub
        y_in_bw_idx = (yk != lb) * (yk != ub)  # not at lb and not at ub
        if np.count_nonzero(y_in_bw_idx) > 0:
            self.gomory_w = cp.Variable(shape=(np.count_nonzero(y_in_bw_idx), 1))
            self.gomory_v = cp.Variable(shape=(np.count_nonzero(y_in_bw_idx), 1), boolean=True)
            M1 = 2 * (yk[y_in_bw_idx][:, None] - lb)
            M2 = 2 * (ub - yk[y_in_bw_idx][:, None])
            self.constraints += [
                - self.gomory_w <= self.efforts[y_in_bw_idx][:, None] - yk[y_in_bw_idx][:, None]
            ]
            self.constraints += [
                self.efforts[y_in_bw_idx][:, None] - yk[y_in_bw_idx][:, None] <= self.gomory_w
            ]
            self.constraints += [
                self.gomory_w <= self.efforts[y_in_bw_idx][:, None] - yk[y_in_bw_idx][:, None] + cp.multiply(M1, (1 - self.gomory_v))
            ]
            self.constraints += [
                self.gomory_w <= yk[y_in_bw_idx][:, None] - self.efforts[y_in_bw_idx][:, None] + cp.multiply(M2, self.gomory_v)
            ]
            self.constraints += [self.gomory_w >= 0]

        self.gomory_cuts[self.n_gomory_cuts] = [
            (cp.sum(self.efforts[y_at_lb_idx] - yk[y_at_lb_idx]) if np.count_nonzero(y_at_lb_idx) > 0 else 0) +
            (cp.sum(yk[y_at_ub_idx] - self.efforts[y_at_ub_idx]) if np.count_nonzero(y_at_ub_idx) > 0 else 0) +
            (cp.sum(self.gomory_w) if np.count_nonzero(y_in_bw_idx) > 0 else 0)
            >= 1
        ]
        self.constraints += self.gomory_cuts[self.n_gomory_cuts]

    def solve(self):
        self.cvxpy_problem = cp.Problem(self.objective, self.constraints)
        self.cvxpy_problem.solve(verbose=False, solver=self.solver)
