import gymnasium as gym
from controller import PIDController

def test_manual_params():
    # [kp_alt, kd_alt, kp_ang, kd_ang]
    manual_params = [-100.0, -100.0, -100.0, -100.0] 
    
    print(f"Testing Controller with params: {manual_params}")
    
    # 2. Setup
    env = gym.make("LunarLander-v3", render_mode="human")
    agent = PIDController(manual_params)
    
    state, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    
    # 3. Loop
    while not (terminated or truncated):
        # ASK THE CONTROLLER for the action
        action = agent.get_action(state)
        
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    env.close()
    print(f"Final Score: {total_reward:.2f}")

if __name__ == "__main__":
    test_manual_params()