from utils import load_list_from_file
import matplotlib.pyplot as plt
import numpy as np


# i = 1
# for epsilon_dec in epsilon_decs:
rewards_per_episode_list = []
rewards_per_episode_list.append(load_list_from_file(f'training_data/rewards_LL.pkl'))
rewards_per_episode_list.append(load_list_from_file(f'training_data/rewards-lunar.pkl'))

list = ["DDQN", "DQN"]
i = 0

for rewards_per_episode in rewards_per_episode_list:
    weights = np.repeat(1.0, 20) / 20
    moving_average = np.convolve(rewards_per_episode, weights, 'valid')

    plt.plot(moving_average, label=list[i])
    i+=1

# weights = np.repeat(1.0, 20) / 20
# moving_average = np.convolve(rewards_per_episode, weights, 'valid')

# plt.plot(moving_average)



plt.xlabel('Episodes')
plt.ylabel(r'$\sum$ '  + 'Rewards')
plt.grid()
plt.title('Lunar Lander Rewards per Episode')
plt.legend()
plt.show()

bar_list = [[324, 279], [286, 228], [84, 134]]

X = np.arange(2)
plt.barh(X + 0.00, bar_list[0], color = 'skyblue', height = 0.2)
plt.barh(X + 0.25, bar_list[1], color = 'gold', height = 0.2)
plt.barh(X + 0.50, bar_list[2], color = 'tomato', height = 0.2)

plt.yticks(X+0.275, ['DDQN', 'DQN'])
plt.xlabel('Rewards')
plt.title('Lunar Rewards')
plt.legend(['Max', 'Mean', 'Min'])
plt.grid()
# plt.savefig("results/lunar_rewards.jpg")
plt.show()