"""
NETSECGAME-TEXTGRAD OPTIMIZATION INTEGRATION
Preserves your existing execution flow while adding prompt optimization
"""
import textgrad as tg
import logging
from llm_action_planner import LLMActionPlanner

# ===== 1. TEXTGRAD SETUP =====
INITIAL_PROMPT = """..."""  # Your existing prompt

# Global prompt variable (shared across episodes)
optimizable_prompt = tg.Variable(
    value=INITIAL_PROMPT,
    requires_grad=True,
    role_description="Cybersecurity agent instructions",
    name="NetSecGame_Prompt"
)

optimizer = tg.TextualGradientDescent(
    parameters=[optimizable_prompt],
    constraints=[...],  # Your security constraints
    momentum_window=3
)

# ===== 2. MODIFIED AGENT EXECUTION =====
def run_episode_with_optimization(args, episode_index):
    """Runs a single episode with TextGrad integration"""
    # --- Initialize with optimized prompt ---
    observation = agent.request_game_reset()

    # Create planner WITH CURRENT OPTIMIZED PROMPT
    llm_query = LLMActionPlanner(
        model_name=args.llm,
        goal=observation.info["goal_description"],
        memory_len=args.memory_buffer,
        api_url=args.api_url,
        use_reasoning=args.use_reasoning,
        use_reflection=args.use_reflection,
        use_self_consistency=args.use_self_consistency,
        custom_instructions=optimizable_prompt.value  # KEY INTEGRATION POINT
    )

    # --- Run episode with existing logic ---
    total_reward = 0
    memories = []
    for step in range(observation.info["max_steps"]):
        # ... YOUR EXISTING EPISODE LOGIC ...
        # get_action_from_obs_react() will use custom_instructions
        is_valid, response_dict, action = llm_query.get_action_from_obs_react(observation, memories)
        observation = agent.make_step(action)
        total_reward += observation.reward

        # ... YOUR EXISTING MEMORY/UPDATES ...

    return total_reward, llm_query

# ===== 3. OPTIMIZATION LOOP =====
def main():
    # ... YOUR EXISTING SETUP ...
    for episode in range(1, args.test_episodes + 1):
        # --- Run episode with optimized prompt ---
        total_reward, llm_query = run_episode_with_optimization(args, episode)

        # --- TextGrad optimization ---
        optimizer.zero_grad()
        loss = -total_reward  # Convert reward to loss
        loss_tensor = tg.tensor(loss)
        loss_tensor.backward()
        optimizer.step()

        # --- Logging ---
        if not args.disable_mlflow:
            mlflow.log_metric("total_reward", total_reward, step=episode)
            mlflow.log_text(optimizable_prompt.value, f"prompt_episode_{episode}.txt")

        # ... YOUR EXISTING LOGGING ...