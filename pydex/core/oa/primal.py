class OAPrimalProblem:
    def __init__(self):
        self.cvxpy_problem = None
        self.efforts = None
        self.criterion = None
        self.criterion_value = None

    def compute_criterion_value(self):
        self.criterion_value = self.criterion(self.efforts).value
        return self.criterion_value

    def eval_fim(self, efforts):
        raise SyntaxError
