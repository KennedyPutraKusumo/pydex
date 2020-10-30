import cvxpy as cp
import numpy as np


class Node(object):
    def __init__(self, int_var, cvxpy_prob, node_id=None, optimizer=None):
        # node identifier
        if node_id is None:
            node_id = "0"
        self.node_id = str(node_id)

        # core attributes
        self.int_var = int_var
        self.cvxpy_prob = cvxpy_prob

        # computed and stored value
        self.ub = None
        self.lb = None
        self.int_var_val = None

        # flags
        self.solved = None
        self.feasible = None
        self.is_incumbent = None  # True or False
        self.worthwhile = None
        self.integral = None
        self._integrity_tol = 4

        # branches
        self.left_child = None
        self.right_child = None

        # options
        self.optimizer = optimizer

    def solve(self):
        if self.solved:
            return

        if self.optimizer is None:
            self.optimizer = cp.MOSEK

        self.cvxpy_prob.solve(solver=self.optimizer)
        if self.cvxpy_prob.status is "optimal":
            self.feasible = True
            self.solved = True
            self.ub = self.cvxpy_prob.value
            self.check_integrity()
            relaxed_val = self.int_var.value
            self.int_var_val = self.int_var.value
            if self.integral:
                self.lb = self.ub
            else:
                prob = cp.Problem(
                    self.cvxpy_prob.objective,
                    self.cvxpy_prob.constraints +
                    [self.int_var == np.abs(np.round(relaxed_val))]
                )
                prob.solve()
                if prob.status is "optimal":
                    self.lb = prob.value
                elif prob.status is "infeasible":
                    self.lb = -np.inf

        if self.cvxpy_prob.status is "infeasible":
            self.feasible = False
            self.solved = True

    def check_integrity(self):
        int_var_val = np.round(self.int_var.value, self._integrity_tol)
        fractional, integral = np.modf(int_var_val)
        if np.allclose(fractional, 0):
            self.integral = True
        else:
            self.integral = False
        return self.integral

    def branch(self, scheme="greatest_fractional"):
        if scheme is "greatest_fractional":
            self._greatest_fractional_branch()
        else:
            raise SyntaxError(
                f"Unknown scheme: {scheme}. try \"greatest_fractional\""
            )
        return self.left_child, self.right_child

    def _greatest_fractional_branch(self):
        # determine the variable to branch over
        if self.int_var.ndim > 1:
            self.int_var = self.int_var.flatten()
        fractional, integral = np.modf(self.int_var_val)
        most_fractional_idx = np.abs(fractional - 0.5).argmin()
        # fractional = np.abs(fractional - 0.5)
        # most_fractional_idx = np.where(fractional == fractional.min())
        most_fractional_var = self.int_var[most_fractional_idx]
        """ creating left and right child nodes """
        # adding constraints to each child
        right_child_cons = [
            most_fractional_var >= np.ceil(self.int_var_val.flatten()[most_fractional_idx])
        ]
        left_bound = np.floor(self.int_var_val.flatten()[most_fractional_idx])
        if np.isclose(left_bound, 0):
            left_child_cons = [
                most_fractional_var == 0
            ]
        else:
            left_child_cons = [
                most_fractional_var <= left_bound
            ]
        # creating children
        self.left_child = Node(
            self.int_var,
            cp.Problem(
                self.cvxpy_prob.objective,
                self.cvxpy_prob.constraints + left_child_cons,
            ),
            self.node_id + ".0"
        )
        self.right_child = Node(
            self.int_var,
            cp.Problem(
                self.cvxpy_prob.objective,
                self.cvxpy_prob.constraints + right_child_cons
            ),
            self.node_id + ".1"
        )

    def __str__(self):
        len = 80
        if not self.solved:
            return f"unsolved node"
        elif not self.feasible:
            return f"[Node {self.node_id}: infeasible]".center(len, "=")
        elif self.integral:
            return f"[Node {self.node_id}: integral solution]".center(len, "=") + \
                   f"\nUpper bound: {self.ub}" + \
                   f"\nLower bound: {self.lb}" + \
                   f"\nInteger Vars: {np.round(self.int_var_val, 2)}" + \
                   f"\n" + f"".center(len, ".")
        else:
            return f"[Node {self.node_id}: non-integral solution]".center(len, "=") + \
                   f"\nUpper bound: {self.ub}" + \
                   f"\nLower bound: {self.lb}" + \
                   f"\nInteger Vars: {np.round(self.int_var_val, 2)}" + \
                   f"\n" + f"".center(len, ".")
