import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rarmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m) [0]
    return pr.choice(indices)

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 2000
#create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    #The Q-Table learning algorithm
    while not done:
        action  = rarmax(Q[state, :])

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        Q[state,action] = reward + np.max(Q[new_state,:])

        rAll += reward
        state = new_state

    rList.append(rAll)

print ("Success rate: " + str(sum(rList)/num_episodes))
print ("Final Q-Table Values")
print ("LEFT DOWN RIGHT UP")
print (Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()