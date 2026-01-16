import gymnasium as gym
import numpy as np
from controller import PIDController

def evaluate_parameters(params, n_runs=3, render=False):
    """
    Evaluate PID parameters with multiple runs for stability
    
    params: [kp_alt, kd_alt, kp_ang, kd_ang]
    n_runs: number of episodes to average over
    render: whether to show visualization
    
    returns: average score
    """
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    
    scores = []
    
    for run in range(n_runs):
        agent = PIDController(params)
        
        # Different seed for each run to cover different initial conditions
        state, _ = env.reset(seed=run)
        
        total_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        
        scores.append(total_reward)
        
        # Debug output for first few runs
        if run == 0 and not render:
            print(f"  Params {params}: Run {run+1} = {total_reward:.1f}")
    
    env.close()
    
    avg_score = np.mean(scores)
    if n_runs > 1:
        std_score = np.std(scores)
        print(f"  Avg: {avg_score:.1f} ± {std_score:.1f} over {n_runs} runs")
    
    return float(avg_score)

if __name__ == "__main__":
    # Test with reasonable parameters
    test_params = [15.0, 1.5, -10.0, 1.0]  # Note: kp_ang is negative!
    score = evaluate_parameters(test_params, n_runs=3, render=False)
    print(f"Test score: {score}")