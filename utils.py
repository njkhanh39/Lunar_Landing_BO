import torch
import botorch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from env_wrapper import evaluate_parameters
from botorch.utils.transforms import standardize # deal with large score from agent. GP works well for small y's, heuristically [-2, 2]
from botorch.models.transforms import Normalize # small x input is good too (0->1)

DTYPE = torch.double

class BayesianOptimizer():
    def __init__(self, bounds, n_init=20):
        """
        bounds: torch.tensor of shape (2, d) where d=4
                bounds[0] = lower bounds, bounds[1] = upper bounds
        n_init: number of initial random samples
        """
        self.bounds = bounds
        d = bounds.shape[1]
        
        # generate initial samples within bounds
        self.train_x = torch.rand(n_init, d, dtype=DTYPE) * (bounds[1] - bounds[0]) + bounds[0]
        self.train_y = torch.zeros(n_init, 1, dtype=DTYPE)
        
        # evaluate each initial sample (average multiple runs for stability)
        for i in range(n_init):
            params_list = self.train_x[i].tolist()
            # ese average of 3 runs to reduce noise
            score = evaluate_parameters(params_list, n_runs=3, render=False)
            self.train_y[i] = score


            
    
    # one iteration - returns candidate
    def update_posterior(self, x_new = None, y_new = None):
        
        if(x_new is not None and y_new is not None):
            # concatenate
            self.train_x = torch.cat([self.train_x, x_new], dim=0)
            self.train_y = torch.cat([self.train_y, y_new], dim=0)

        # init

        # don't forget to STANDARDIZE y and NORMALIZE x for faster convergence
        gp = SingleTaskGP(
            self.train_x, 
            standardize(self.train_y), 
            input_transform=Normalize(d=self.train_x.shape[-1])
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        # update posterior
        fit_gpytorch_mll(mll) # already have some initial random data so ok

        # UCB seems better
        UCB = UpperConfidenceBound(gp, beta=4)
        
        
        candidate, acquisition_score = optimize_acqf(
            acq_function=UCB,
            bounds=self.bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            sequential=False  # Better for noisy objectives
        )
        
        return candidate

