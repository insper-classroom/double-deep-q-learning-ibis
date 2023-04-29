import torch
import gym
from agents import DDQNAgent
from constants import *

# Load the trained model
model_path = 'models/DDQL_CartPole-v1.pth'
model = torch.load(model_path)

# Create the environment
env = gym.make("CartPole-v1")

# Create the agent
agent = DDQNAgent(input_size=env.observation_space.shape[0], output_size=env.action_space.n, hidden_sizes=[512, 256])

# Set the agent's Q-network to the loaded model
agent.q_network.load_state_dict(model)

# Test the agent for n episodes
num_episodes = 100
reward_list = []
for i_episode in range(num_episodes):
    (state,_) = env.reset()
    total_reward = 0
    done = False
    truncated = False
    max_timesteps = 500
    while (not done) and (not truncated):
        action = agent.select_action(state, eps=0)  # Use greedy policy (eps=0)
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        if max_timesteps == 499:
            print("max steps reached")
            break

    reward_list.append(total_reward)

    if total_reward > 0 and total_reward < 200:
        print(f"Episode {i_episode} - Total reward: {total_reward:.2f} {GOOD}")
    elif total_reward < 0 and total_reward > -500:
        print(f"Episode {i_episode} - Total reward: {total_reward:.2f} {BAD}")
    elif total_reward < -500:
        print(f"Episode {i_episode} - Total reward: {total_reward:.2f} {AWFUL}")
    else:
        print(f"Episode {i_episode} - Total reward: {total_reward:.2f} {AMAZING}")

print(f"Average reward over {num_episodes} episodes: {sum(reward_list)/num_episodes:.2f}")
print(f"Max reward over {num_episodes} episodes: {max(reward_list):.2f}")
print(f"Min reward over {num_episodes} episodes: {min(reward_list):.2f}")
        
