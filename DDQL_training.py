
import gym
from agents import DDQNAgent
from constants import ENV


if __name__ == "__main__":
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    agent = DDQNAgent(input_size=env.observation_space.shape[0], output_size=env.action_space.n, env_name=env_name, hidden_sizes=[512,256])
    agent.train_agent(env)
    env.close()


