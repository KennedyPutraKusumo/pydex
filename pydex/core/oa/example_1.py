import numpy as np
import cvxpy as cp


if __name__ == '__main__':
    def simulate(ti_controls, model_parameters):
        return np.array([
            model_parameters[0] +
            model_parameters[1] * ti_controls[0] +
            model_parameters[2] * ti_controls[1] +
            model_parameters[3] * ti_controls[0] ** 2 +
            model_parameters[4] * ti_controls[1] ** 2 +
            model_parameters[5] * ti_controls[0] * ti_controls[1]
        ])

    def atomic_matrices(x):
        atomics = []
        for entry in x:
            sens = np.array([
                    1, entry[0], entry[1], entry[0] ** 2, entry[1] ** 2, entry[0] * entry[1]
            ])[:, None]
            atom = sens @ sens.T
            atomics.append(
                atom
            )
        return np.array(atomics)

    N_exp = 6
    reso = 3j

    x1, x2 = np.mgrid[-1:1:reso, -1:1:reso]
    x1 = x1.flatten()
    x2 = x2.flatten()
    X = np.array([x1, x2]).T
    atomics = atomic_matrices(X)

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

    """ Initialize Outer Approximation """
    z_U = np.inf
    k = 1

    """ Primal Problem """
    p = rounded_solution[:, 0]
    primal_x = cp.Variable(shape=atomics.shape[0])

    objective = -cp.log_det(cp.sum([primal_x[i] * atomics[i] for i, _ in enumerate(atomics)]))

    master_constraints = []
    master_constraints += [cp.sum(p) == N_exp]
    master_constraints += [primal_x[i] == p[i] for i, _ in enumerate(primal_x)]
    master_constraints += [primal_x[i] <= 1 for i, _ in enumerate(primal_x)]
    master_constraints += [primal_x[i] >= 0 for i, _ in enumerate(primal_x)]

    problem = cp.Problem(cp.Minimize(objective), master_constraints)
    problem.solve(verbose=True)
    upper_bound = problem.objective.value
    primal_efforts = primal_x.value
    if upper_bound < z_U:
        z_U = upper_bound
        x_star = primal_efforts
        y_star = p

    # computing lagrange multipliers for the primal
    import numdifftools as nd

    def det_objective(a):
        return -np.log(np.linalg.det(np.sum(a[:, None, None] * atomics, axis=0)))

    def h1(a):
        return [np.sum(a) - N_exp]

    def h2(a, efforts=rounded_solution[:, 0]):
        return [a[i] - efforts[i] for i, _ in enumerate(a)]

    def h_func(a):
        h_list = []
        h_list.extend(h1(a))
        h_list.extend(h2(a))
        return np.array(h_list)

    def g1(a):
        return [a[i] - 1 for i, _ in enumerate(primal_x)]

    def g2(a):
        return [-a[i] for i, _ in enumerate(primal_x)]

    def g_func(a):
        g_list = []
        g_list.extend(g1(a))
        g_list.extend(g2(a))
        return np.array(g_list)

    grad_f = nd.Gradient(det_objective)
    grad_f_val = grad_f(primal_x.value)

    primal_feasibility_h = h_func(primal_x.value)
    grad_h = nd.Jacobian(h_func)
    grad_h_val = grad_h(primal_x.value)

    primal_feasibility_g = g_func(primal_x.value)
    grad_g = nd.Jacobian(g_func)
    grad_g_val = grad_g(primal_x.value)

    print(grad_f_val)
    print(grad_f_val.shape)
    print(grad_h_val)
    print(grad_h_val.shape)
    print(grad_g_val)
    print(grad_g_val.shape)

    # retrieve KKT multipliers of primal problem from cvxpy
    cvxpy_multipliers = np.array([master_constraints[i].dual_value for i, _ in enumerate(master_constraints)])
    h_multipliers = cvxpy_multipliers[:primal_feasibility_h.shape[0]]
    print(f"CVXPY Multipliers")
    print(cvxpy_multipliers)
    print(cvxpy_multipliers.shape)

    """ Master Problem"""
    master_eta = cp.Variable()
    master_x = cp.Variable(shape=atomics.shape[0])
    master_y = cp.Variable(shape=atomics.shape[0], boolean=True)

    master_constraints = []
    master_constraints += [
        master_eta >= det_objective(primal_x.value) + grad_f_val @ (master_x - primal_x.value)
    ]
    relaxation_matrix = np.zeros(shape=(primal_feasibility_h.shape[0], primal_feasibility_h.shape[0]))
    for i in range(primal_feasibility_h.shape[0]):
        if h_multipliers[i] < 0:
            relaxation_matrix[i, i] = -1
        elif h_multipliers[i] > 0:
            relaxation_matrix[i, i] = 1
        else:
            relaxation_matrix[i, i] = 0

    dT = np.ones((master_y.shape[0] + 1, master_y.shape[0]))
    dT[0, :] = np.ones(master_y.shape[0])
    dT[1:, :] = np.eye(master_y.shape[0])
    print(dT)
    master_constraints += [
        relaxation_matrix @ (primal_feasibility_h + grad_h_val @ (master_x - primal_x.value)) + dT @ master_y <= 0
    ]
    master_constraints += [
        primal_feasibility_g + grad_g_val @ (master_x - primal_x.value) <= 0
    ]
    master_objective = cp.Minimize(master_eta)
    master_problem = cp.Problem(master_objective, master_constraints)
    master_problem.solve(verbose=True)
    print(master_y.value)

    print("END")
