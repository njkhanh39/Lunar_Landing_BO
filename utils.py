import torch
import botorch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from env_wrapper import evaluate_parameters
from botorch.utils.transforms import standardize # deal with large score from agent. GP works well for small y's, heuristically [-2, 2]

class BayesianOptimizer():
    def __init__(self, min_x=0, max_x=30, x_row = 5, x_col = 4):
        self.train_x = (min_x - max_x) * torch.rand(x_row, x_col) + max_x # uniform on [min, max], shape (x_row, x_col)
        self.train_y = torch.zeros(x_row, 1)
        
        # train y
        for i in range(0, x_row):
            params_list = self.train_x[i].tolist()

            # NOTE: this is raw y-score. We should standardize when running Gaussian Process
            self.train_y[i] = evaluate_parameters(params_list, render=False)

        # bound for x
        self.bound = torch.tensor(2, x_col)
        self.bound[0] = torch.full(x_col, fill_value=min_x)
        self.bound[1] = torch.full(x_col, fill_value=max_x)
    
    # one iteration - returns candidate
    def update_posterior(self, x_new = None, y_new = None):
        
        if(x_new != None and y_new != None):
            # concatenate
            self.train_x = torch.cat([self.train_x, x_new], dim=0)
            self.train_y = torch.cat([self.train_y, y_new], dim=0)

        # init

        # don't forget to STANDARDIZE y for faster convergence
        gp = SingleTaskGP(self.train_x, standardize(self.train_y))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        # update posterior
        fit_gpytorch_mll(mll) # already have some initial random data so ok

        # find next candidate
        UCB = UpperConfidenceBound(gp, beta=0.8) # beta = coef controls explore vs exploit
        candidate, acquisition_score = optimize_acqf(
            acq_function= UCB,
            bounds=self.bound,
            q = 1,
            num_restarts=5,
            raw_samples=20
        )

        return candidate

