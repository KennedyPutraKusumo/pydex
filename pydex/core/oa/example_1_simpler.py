import cvxpy as cp
import numpy as np


def simulate(ti_controls, model_parameters):
    return np.array([
        model_parameters[0] +
        model_parameters[1] * ti_controls[0] +
        model_parameters[2] * ti_controls[1] +
        model_parameters[3] * ti_controls[0] ** 2 +
        model_parameters[4] * ti_controls[1] ** 2 +
        model_parameters[5] * ti_controls[0] * ti_controls[1]
    ])


if __name__ == '__main__':

    N_exp = 6
    reso = 11j

    x1, x2 = np.mgrid[-1:1:reso, -1:1:reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T

    """ Using Pydex to get a rounded design as initial guess for binary efforts """
    from pydex.core.designer import Designer

    designer = Designer()
    designer.simulate = simulate
    designer.ti_controls_candidates = X
    designer.model_parameters = np.ones(6)
    designer.initialize(verbose=2)
    designer.design_experiment(
        designer.d_opt_criterion,
    )
    designer.print_optimal_candidates()
    designer.apportion(N_exp)
    rounded_solution = designer.non_trimmed_apportionments

    def sensitivities(x):
        sens = []
        for entry in x:
            sens.append([
                1, entry[0], entry[1], entry[0] ** 2, entry[1] ** 2,
                                       entry[0] * entry[1]
            ])
        return np.array(sens)

    def analytic_d_logdetfim(p):
        M = designer.sensitivities[:, 0, 0, :].T @ np.diag(p) @ designer.sensitivities[:, 0, 0, :]
        det_M = np.linalg.det(M)
        d_detfim = np.empty(p.shape[0])
        for i, (pi, qi) in enumerate(zip(p, designer.sensitivities[:, 0, 0, :])):
            Mi = qi[:, None] @ qi[:, None].T
            Ni = np.linalg.inv(M) @ Mi
            tr_Ni = np.trace(Ni)
            d_detfim[i] = det_M * tr_Ni
        return -(1 / det_M) * d_detfim

    converged = False
    iteration_no = 0
    while not converged:
        """ Primal problem: log_det evaluation only """
        if iteration_no == 0:
            yk = {0: rounded_solution}
            UBDk = {0: designer._criterion_value}
            LBDk = {0: -np.inf}
        else:
            # IMPORTANT: assuming that primal is ALWAYS feasible regardless of the solution for the master
            # this may not be true because linearization may cause the master solution to pick non-identifiable solutions
            efforts_primal = yk[iteration_no]
            UBDk[iteration_no] = designer.compute_criterion_value(designer.d_opt_criterion, effort=efforts_primal)

        """ check if LBD and UBD sufficiently close """
        print(f"Iteration {iteration_no} Done")
        convergence = np.isclose(LBDk[iteration_no], UBDk[iteration_no], rtol=0.01)
        if convergence:
            break

        """ Master problem: MILP after linearizing constraints """
        if iteration_no == 0:
            eta = cp.Variable()
            objective = cp.Minimize(eta)
            efforts_master = cp.Variable(shape=rounded_solution.shape, integer=True)
            constraints = [cp.sum(efforts_master) <= N_exp]
            constraints += [efforts_master >= 0]
            f_yk = {}
            gradf_yk = {}
        else:
            pass
        """ adding cut(s) for linearized objective """
        f_yk[iteration_no] = designer.compute_criterion_value(designer.d_opt_criterion, effort=yk[iteration_no])
        gradf_yk[iteration_no] = analytic_d_logdetfim(yk[iteration_no][:, 0])
        constraints += [
            f_yk[iteration_no] +
            gradf_yk[iteration_no] @ (efforts_master - yk[iteration_no]) <= eta
        ]
        master_problem = cp.Problem(objective, constraints)
        master_problem.solve(verbose=True, solver=cp.CPLEX)
        iteration_no += 1
        LBDk[iteration_no] = eta.value
        yk[iteration_no] = efforts_master.value
