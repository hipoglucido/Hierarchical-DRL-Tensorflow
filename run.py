import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from envs.mdp import StochasticMDPEnv
from agent.hDQN import hDQN

plt.style.use('ggplot')

def one_hot(state):
    vector = np.zeros(6)
    vector[state-1] = 1.0
    return np.expand_dims(vector, axis=0)

def main():
    ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state"])
    MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state"])
    env = StochasticMDPEnv()
    agent = hDQN()
    visits = np.zeros((12, 6))
    for episode_thousand in range(12):
        agent.meta_epsilon = agent.meta_epsilon/2.0
        print("\nNew meta-epsilon: %.4f" % agent.meta_epsilon, end="")
        for episode in range(1000):
            print("\n\n### EPISODE %d ###" % (episode_thousand*1000 + episode), end="")
            state = env.reset()
            visits[episode_thousand][state-1] += 1
            done = False
            while not done:
                goal = agent.select_goal(one_hot(state))
                agent.goal_selected[goal-1] += 1
                print("\nNew Goal: %d\nState-Actions: " % goal)
                total_external_reward = 0
                goal_reached = False
                while not done and not goal_reached:
                    action = agent.select_move(one_hot(state), one_hot(goal))
                    print((state,action), end="; ")
                    next_state, external_reward, done = env.step(action)
                    visits[episode_thousand][next_state-1] += 1
                    intrinsic_reward = agent.criticize(goal, next_state)
                    goal_reached = next_state == goal
                    if goal_reached:
                        agent.goal_success[goal-1] += 1
                        print("Goal reached!!", end=" ")
                    if next_state == 6:
                        print("S6 reached!!", end=" ")
                    exp = ActorExperience(one_hot(state), one_hot(goal), action, intrinsic_reward, one_hot(next_state))
                    agent.store(exp, meta=False)
                    agent.update(meta=False)
                    agent.update(meta=True)
                    total_external_reward += external_reward
                    state = next_state
                exp = MetaExperience(one_hot(state), one_hot(goal), total_external_reward, one_hot(next_state))
                agent.store(exp, meta=True)
            if (episode % 100 == 99):
                print("")
                print(visits/1000, end="")

    eps = list(range(1,13))
    plt.subplot(2, 3, 1)
    plt.plot(eps, visits[:,0]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S1")
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(eps, visits[:,1]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S2")
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(eps, visits[:,2]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S3")
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(eps, visits[:,3]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S4")
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(eps, visits[:,4]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S5")
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(eps, visits[:,5]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 2.0)
    plt.xlim(1, 12)
    plt.title("S6")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
