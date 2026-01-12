import gymnasium as gym
from controller import PIDController

def evaluate_parameters(params, render=False):
    """
    obj black-box function for agent

    params (list): [kp_alt, kd_alt, kp_ang, kd_ang]
    render (bool): True = show window. otherwise runs invisible
        
    
    returns:    float - the final score of the game
    """
   
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    
    # give agent the controller
    agent = PIDController(params)
    
    # running loop
    state, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # agent acts
        action = agent.get_action(state)
        # enviroment responds
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
    env.close()
    
    return total_reward

if __name__ == "__main__":
    score = evaluate_parameters([10, 0, 5, 0], render=False)
    print(score)