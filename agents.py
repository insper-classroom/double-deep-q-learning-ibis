
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import pickle
import numpy as np
from constants import *
import gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym.wrappers import AtariPreprocessing, FrameStack
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import pickle




# Define the Q-network
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(1568, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        
        x = x/255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the DQNAgent
class DDQN_conv_Agent:
    def __init__(self, env,  lr=0.00025, gamma=0.99, buffer_size=10000, batch_size=32):
        print(torch.cuda.is_available())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.policy_net = DQN(env.action_space.n).to(self.device)
        self.target_net = DQN(env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma

    def preprocess(self, state):
        # state = state.transpose((2, 0, 1))
        state = np.array(state)
        state = torch.from_numpy(state)
        return state.to(self.device, dtype=torch.float)



    def select_action(self, state, eps):
        
        rd = random.random()
        
        if rd < eps:
            return self.env.action_space.sample()
        with torch.no_grad():
            
            state = state.unsqueeze(0).to(self.device)
            
            q_values = self.policy_net(state)
           
            return q_values.argmax().item()

    def optimize_model(self, transitions, batch_size):
        batch = list(zip(*transitions))
        state_batch = torch.stack(batch[0]).to(self.device)
        
        
        action_batch = torch.tensor(batch[1]).unsqueeze(1).to(self.device)
        next_state_batch = torch.stack(batch[2]).to(self.device)
        reward_batch = torch.tensor(batch[3]).unsqueeze(1).to(self.device)
        done_batch = torch.tensor(batch[4]).unsqueeze(1).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        next_q_values = torch.zeros(batch_size, device=self.device)
        next_q_values[~done_batch.squeeze()] = self.target_net(next_state_batch[~done_batch.squeeze()]).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma*next_q_values.unsqueeze(1)

        loss = F.smooth_l1_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    # Define the training loop
    def train(self, num_episodes=100, eps_start=1.0, eps_end=0.01, eps_decay=0.999, target_update_freq=1000, env_name="CartPole-v1"):
    
        eps = eps_start
        rewards = []
        for episode in range(num_episodes):
            (state,_) = self.env.reset()
            
            
            done = False
            truncated = False
            episode_reward = 0
            while (not done) and (not truncated):
                # Select an action
            
                state_tensor = self.preprocess(state)
                action = self.select_action(state_tensor, eps)

                # Take a step
                next_state, reward, done, truncated, _ = self.env.step(action)

                if reward != 0:
                    print(reward)

                    
                reward = 1 if reward > 0 else -1 if reward < 0 else 0

                if reward != 0:
                    print(reward)
                
                episode_reward += reward

                # Store the transition
                next_state_tensor = self.preprocess(next_state)
            
                self.memory.append((state_tensor, action, next_state_tensor, reward, done))

                # Optimize the Q-network
                if len(self.memory) >= self.batch_size:
                    transitions = random.sample(self.memory, self.batch_size)
                    self.optimize_model(transitions, self.batch_size)

            rewards.append(episode_reward)
            eps = max(eps_end, eps*eps_decay)

            # Update the target network
            if episode % target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Print the episode information
            print("Episode {}/{}: reward={}, eps={:.2f}".format(episode+1, num_episodes, episode_reward, eps))


        
        save_path = f"models/DDQL_{env_name}.pth"
        torch.save(self.policy_net.state_dict(), save_path)

        # open a file for writing
        with open(f'training_data/rewards_{env_name}.pkl', 'wb') as f:
            # write the list to the file using pickle
            pickle.dump(rewards, f)

        plt.plot(rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Total rewards")
        plt.title("DDQL")
        plt.savefig(f"DDQL_{env_name}.png")
        plt.show()

        return rewards



class DDQNAgent:
    def __init__(self, input_size, output_size, hidden_sizes, lr=1e-3, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.997, env_name="LunarLander-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = self.build_network(input_size, output_size, hidden_sizes).to(self.device)
        self.target_q_network = self.build_network(input_size, output_size, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.env_name = env_name
        

    def build_network(self, input_size, output_size, hidden_sizes):
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def select_action(self, state, eps):
        if random.random() < eps:
            return random.randint(0, self.q_network[-1].out_features-1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                action = q_values.argmax(dim=1).item()
            return action
        
    def save_model(self, file_path):
        torch.save(self.q_network.state_dict(), file_path)

    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return
        samples = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        list = [states, actions, rewards, next_states, dones]
        for i in  range(len(list)):
            list[i] = np.array(list[i])

        

        states = torch.FloatTensor(list[0]).to(self.device)
        actions = torch.LongTensor(list[1]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(list[2]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(list[3]).to(self.device)
        dones = torch.FloatTensor(list[4]).unsqueeze(1).to(self.device)

        
        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_q_network(next_states)[0][self.q_network(next_states).argmax(dim=1, keepdim=True)]
        # next_q_values = self.target_q_network(next_states).max(dim=1, keepdim=True)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train_agent(self, env, num_episodes=1000, max_timesteps=500, batch_size=64, update_target_every=100):
        replay_buffer = deque(maxlen=500000)
        eps = self.eps_start
        rewards = []
        for i_episode in range(num_episodes):
            (state,_) = env.reset()
            
                
            total_reward = 0
            for t in range(max_timesteps):
                action = self.select_action(state, eps)
                next_state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                replay_buffer.append((state, action, reward, next_state, done))
                state = next_state

                self.train(replay_buffer, batch_size)

                if t % update_target_every == 0:
                    self.update_target_network()

                if done or truncated:
                    break

            eps = max(self.eps_end, self.eps_decay * eps)

            if total_reward > 0 and total_reward < 200:
                print(f"Episode {i_episode} - Total reward: {total_reward:.2f} {GOOD} - Epsilon: {eps:.3f}")
            elif total_reward < 0 and total_reward > -500:
                print(f"Episode {i_episode} - Total reward: {total_reward:.2f} {BAD} - Epsilon: {eps:.3f}")
            elif total_reward < -500:
                print(f"Episode {i_episode} - Total reward: {total_reward:.2f} {AWFUL} - Epsilon: {eps:.3f}")
            else:
                print(f"Episode {i_episode} - Total reward: {total_reward:.2f} {AMAZING} - Epsilon: {eps:.3f}")

            rewards.append(total_reward)

        save_path = f"models/DDQL_{self.env_name}.pth"
        self.save_model(save_path)

        plt.plot(rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Total rewards")
        plt.title("DDQL")
        plt.savefig(f"imgs/DDQL_{self.env_name}.png")
        plt.show()
        
    
        # open a file for writing
        with open(f'training_data/rewards_{self.env_name}.pkl', 'wb') as f:
            # write the list to the file using pickle
            pickle.dump(rewards, f)

