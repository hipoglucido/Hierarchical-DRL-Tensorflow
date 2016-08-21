import random
import numpy as np
from collections import namedtuple
from envs.mdp import StochasticMDPEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

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

def critic():
    critic = Sequential()
    critic.add(Dense(6, init='lecun_uniform', input_shape=(19,)))
    critic.add(Activation("relu"))
    critic.add(Dense(1, init='lecun_uniform'))
    critic.compile(loss='mse', optimizer=Adam())
    return critic

class Agent:

    def __init__(self):
        self.meta_controller = meta_controller()
        self.actor = actor()
        self.critic = critic()
        self.actor_epsilon = 0.1 # TODO: Epsilon decay and goal-specific epsilons
        self.meta_epsilon = 0.1 # TODO: Epsilon decay
        self.n_samples = 10
        self.meta_n_samples = 10
        self.gamma = 0.96
        self.memory = []
        self.meta_memory = []

    def select_move(self, state, goal):
        vector = np.concatenate([state, goal], axis=1)
        if self.actor_epsilon < random.random():
            return np.argmax(self.actor.predict(vector, verbose=0))
        return random.choice([0,1])

    def select_goal(self, state):
        if self.meta_epsilon < random.random():
            return np.argmax(self.meta_controller.predict(state, verbose=0))+1
        return random.choice([1,2,3,4,5,6])

    def criticize(self, state, goal, action, next_state):
        vector = np.concatenate([state, goal, [[action]], next_state], axis=1)
        return self.critic.predict(vector, verbose=0)

    def store(self, experience, meta=False):
        if meta:
            self.meta_memory.append(experience)
        else:
            self.memory.append(experience)

    def _update(self):
        exps = [random.choice(self.memory) for _ in range(self.n_samples)]
        for exp in exps:
            critic_vector = np.concatenate([exp.state, exp.goal, [[exp.action]], exp.next_state], axis=1)
            actor_vector = np.concatenate([exp.state, exp.goal], axis=1)
            actor_reward = self.actor.predict(actor_vector, verbose=0)
            actor_reward[0][exp.action] = exp.reward
            self.critic.fit(critic_vector, exp.reward, verbose=0)
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
    for episode in range(100):
        print("\n### EPISODE %d ###" % episode)
        state = env.reset()
        done = False
        while not done:
            goal = agent.select_goal(one_hot(state))
            print("New Goal: %d" % goal)
            total_external_reward = 0
            goal_reached = False
            while not done and not goal_reached:
                print(state, end=",")
                action = agent.select_move(one_hot(state), one_hot(goal))
                next_state, external_reward, done = env.step(action)
                intrinsic_reward = agent.criticize(one_hot(state), one_hot(goal), action, one_hot(next_state))
                goal_reached = next_state == goal
                if goal_reached:
                    print("Success!!")
                exp = ActorExperience(one_hot(state), one_hot(goal), action, intrinsic_reward, one_hot(next_state))
                agent.store(exp, meta=False)
                agent.update(meta=False)
                agent.update(meta=True)
                total_external_reward += external_reward
                state = next_state
            exp = MetaExperience(one_hot(state), one_hot(goal), total_external_reward, one_hot(next_state))
            agent.store(exp, meta=True)

if __name__ == "__main__":
    main()
