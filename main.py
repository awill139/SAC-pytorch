import pybullet_envs
import gym
import numpy as np
#if OSX
from sys import platform
if platform == 'darwin':
    import matplotlib  
    matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
from agent import Agent


if __name__ == '__main__':
    env = gym.make('InvertedPendulumBulletEnv-v0')

    agent = Agent(input_dims = env.observation_space.shape, env = env,
                  n_actions = env.action_space.shape[0])

    n_games = 250
    best_score = env.reward_range[0]
    score_hist = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode = 'human')

    for i in range(n_games):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            score += reward
            agent.store_trans(obs, action, reward, obs_, done)
            if not load_checkpoint:
                agent.learn()
            obs = obs_
        score_hist.append(score)
        avg_score = np.mean(score_hist[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode: {} \t score: {} \t avg_score: {}'.format(i, score, avg_score))

    if not load_checkpoint:
        x = np.linspace(0, n_games, len(score_hist))
        plt.plot(x, score_hist)
        plt.xlabel('number of games')
        plt.ylabel('score')
        plt.savefig('test.png')
