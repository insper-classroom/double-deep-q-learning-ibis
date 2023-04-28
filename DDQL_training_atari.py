import gym
from gym.wrappers import AtariPreprocessing, FrameStack
from agents import DDQN_conv_Agent

# Train the agent
env_name = "BreakoutNoFrameskip-v4"
env = gym.make(env_name)
env = AtariPreprocessing(env)
env = FrameStack(env, num_stack=4)
agent = DDQN_conv_Agent(env)
rewards = agent.train(num_episodes=10000, env_name=env_name)

