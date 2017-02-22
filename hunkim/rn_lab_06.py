# 참고자료
# 모두를 위한 머신러닝/딥러닝 강의
# 홍콩과기대 김성훈
# http://hunkim.github.io/ml

import gym

env = gym.make('CartPole-v0')

for i in range(10):
    observation = env.reset()

    for t in range(10000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print(i, t, 'game over', 'done: ', done)
            break