# Project Plan: Bayesian Optimization for Lunar Lander

## Phase 1: The Environment Setup
- [ ] Install dependencies.
- [ ] Create `env_wrapper.py`.
- [ ] Write a function `run_random_agent()` that starts `LunarLander-v3`, takes random actions, and renders the screen.
- [✅] **Goal:** Verify the window pops up and the ship crashes.

## Phase 2: The Heuristic Controller (The "Robot")
- [ ] Create `controller.py`.
- [ ] Implement a class `PIDController`.
- [ ] Logic:
    - If the ship tilts left, fire right engine (and vice versa).
    - If the ship falls too fast, fire main engine.
- [ ] The "Aggressiveness" of these corrections will be variables (e.g., `angle_k`, `hover_k`).
- [✅] **Goal:** The ship should "try" to fly, even if it does it badly.

## Phase 3: The Objective Function
- [ ] Update `env_wrapper.py` to accept the Controller parameters.
- [ ] Write a function `evaluate_parameters(params)`:
    - Takes `[k_p_angle, k_d_angle, k_p_hover, k_d_hover]`.
    - Runs 1 full episode of the game.
    - Returns the `Total Reward`.
- [✅] **Goal:** You can manually type in numbers and get a score back.

## Phase 4: The BoTorch Loop (main.py)
- [ ] Initialize BoTorch:
    - Define parameter bounds (e.g., 0.0 to 10.0).
    - Initialize Gaussian Process (GP) model.
- [ ] The Loop (Run for 50 iterations):
    1. Fit the GP model to existing data.
    2. Define Acquisition Function (Upper Confidence Bound or Expected Improvement).
    3. `optimize_acqf` to find the next best parameters to try.
    4. Pass these parameters to `evaluate_parameters`.
    5. Append result to dataset.
- [ ] **Goal:** Watch the AI automatically test parameters and eventually land the ship smoothly.

## Phase 5: Visualization
- [ ] Plot the "Best Score found so far" vs "Iteration Number".
- [ ] Render the final "Best Landing" to watch your success.