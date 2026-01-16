import torch
from utils import BayesianOptimizer
from controller import PIDController
from env_wrapper import evaluate_parameters

# Config
iters = 50
b_opt = BayesianOptimizer(min_x=0.0, max_x=20.0, x_row=10, x_col=4)

# Variables to store the result of the previous loop
prev_candidate = None
prev_score = None

print("\n--- Starting Optimization Loop ---")

for i in range(iters):
    print(f"\n-- Iteration {i+1}/{iters} --")

    candidate_tensor = b_opt.update_posterior(x_new=prev_candidate, y_new=prev_score)
    
    params_list = candidate_tensor.flatten().tolist()
    
    print(f"Trying Params: {['%.2f' % elem for elem in params_list]}")

    # run the simulation
    current_score = evaluate_parameters(params=params_list, render=True)
    print(f"Result Score: {current_score:.2f}")
    

    # BoTorch needs tensor of shape (1, 1)
    prev_score = torch.tensor([[current_score]], dtype=torch.double)
    prev_candidate = candidate_tensor