import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from envs.mdp import StochasticMDPEnv
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

def meta_controller():
    meta = Sequential()
    meta.add(Dense(6, init='lecun_uniform', input_shape=(6,)))
    meta.add(Activation("relu"))
    meta.add(Dense(6, init='lecun_uniform'))
    meta.add(Activation("softmax"))
    meta.compile(loss='mse', optimizer=Adam())
    return meta

def actor():
    actor = Sequential()
    actor.add(Dense(6, init='lecun_uniform', input_shape=(12,)))
    actor.add(Activation("relu"))
    actor.add(Dense(2, init='lecun_uniform'))
    actor.add(Activation("softmax"))
    actor.compile(loss='mse', optimizer=Adam())
    return actor

class Agent:

    def __init__(self):
        self.meta_controller = meta_controller()
        self.actor = actor()
        self.goal_selected = np.ones(6)
        self.goal_success = np.zeros(6)
        self.meta_epsilon = 2.0
        self.n_samples = 100
        self.meta_n_samples = 100
        self.gamma = 0.96
        self.memory = []
        self.meta_memory = []

    def select_move(self, state, goal):
        vector = np.concatenate([state, goal], axis=1)
        if 1.0 - 5*self.goal_success[np.argmax(goal)]/self.goal_selected[np.argmax(goal)] < random.random():
            return np.argmax(self.actor.predict(vector, verbose=0))
        return random.choice([0,1])

    def select_goal(self, state):
        if self.meta_epsilon < random.random():
            return np.argmax(self.meta_controller.predict(state, verbose=0))+1
        return random.choice([1,2,3,4,5,6])

    def criticize(self, goal, next_state):
        return 1.0 if goal == next_state else 0.0

    def store(self, experience, meta=False):
        if meta:
            self.meta_memory.append(experience)
        else:
            self.memory.append(experience)

    def _update(self):
        exps = [random.choice(self.memory) for _ in range(self.n_samples)]
        for exp in exps:
            actor_vector = np.concatenate([exp.state, exp.goal], axis=1)
            actor_reward = self.actor.predict(actor_vector, verbose=0)
            actor_reward[0][exp.action] = exp.reward
            self.actor.fit(actor_vector, actor_reward, verbose=0)

    def _update_meta(self):
        if 0 < len(self.meta_memory):
            exps = [random.choice(self.meta_memory) for _ in range(self.meta_n_samples)]
            for exp in exps:
                meta_reward = self.meta_controller.predict(exp.state, verbose=0)
                meta_reward[0][np.argmax(exp.goal)] = exp.reward
                self.meta_controller.fit(exp.state, meta_reward, verbose=0)

    def update(self, meta=False):
        if meta:
            self._update_meta()
        else:
            self._update()

def one_hot(state):
    vector = np.zeros(6)
    vector[state-1] = 1.0
    return np.expand_dims(vector, axis=0)

def main():
    ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state"])
    MetaExperience = namedtuple("MetaExperience", ["state", "goal", "reward", "next_state"])
    env = StochasticMDPEnv()
    agent = Agent()
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
    plt.ylim(-0.01, 1.1)
    plt.xlim(1, 12)
    plt.title("S1")
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(eps, visits[:,1]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 1.1)
    plt.xlim(1, 12)
    plt.title("S2")
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(eps, visits[:,2]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 1.1)
    plt.xlim(1, 12)
    plt.title("S3")
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(eps, visits[:,3]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 1.1)
    plt.xlim(1, 12)
    plt.title("S4")
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(eps, visits[:,4]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 1.1)
    plt.xlim(1, 12)
    plt.title("S5")
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(eps, visits[:,5]/1000)
    plt.xlabel("Episodes (*1000)")
    plt.ylim(-0.01, 1.1)
    plt.xlim(1, 12)
    plt.title("S6")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
