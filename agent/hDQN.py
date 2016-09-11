import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam, SGD

class hDQN:

    def __init__(self):
        self.meta_controller = self.meta_controller()
        self.actor = self.actor()
        self.goal_selected = np.ones(6)
        self.goal_success = np.zeros(6)
        self.meta_epsilon = 4.0
        self.n_samples = 100
        self.meta_n_samples = 50
        self.gamma = 0.96
        self.memory = []
        self.meta_memory = []

    def meta_controller(self):
        meta = Sequential()
        meta.add(Dense(6, init='lecun_uniform', input_shape=(6,)))
        meta.add(Activation("relu"))
        meta.add(Dense(10, init='lecun_uniform'))
        meta.add(Activation("relu"))
        meta.add(Dense(20, init='lecun_uniform'))
        meta.add(Activation("relu"))
        meta.add(Dense(20, init='lecun_uniform'))
        meta.add(Activation("relu"))
        meta.add(Dense(20, init='lecun_uniform'))
        meta.add(Activation("relu"))
        meta.add(Dense(10, init='lecun_uniform'))
        meta.add(Activation("relu"))
        meta.add(Dense(6, init='lecun_uniform'))
        meta.add(Activation("softmax"))
        meta.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False))
        return meta

    def actor(self):
        actor = Sequential()
        actor.add(Dense(12, init='lecun_uniform', input_shape=(12,)))
        actor.add(Activation("relu"))
        actor.add(Dense(10, init='lecun_uniform'))
        actor.add(Activation("relu"))
        actor.add(Dense(8, init='lecun_uniform'))
        actor.add(Activation("relu"))
        actor.add(Dense(6, init='lecun_uniform'))
        actor.add(Activation("relu"))
        actor.add(Dense(4, init='lecun_uniform'))
        actor.add(Activation("relu"))
        actor.add(Dense(2, init='lecun_uniform'))
        actor.add(Activation("softmax"))
        actor.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False))
        return actor

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
            if len(self.meta_memory) > 100:
                self.meta_memory = self.meta_memory[-100:]
        else:
            self.memory.append(experience)
            if len(self.memory) > 500:
                self.memory = self.memory[-500:]

    def _update(self):
        exps = [random.choice(self.memory) for _ in range(self.n_samples)]
        state_vectors = np.squeeze(np.asarray([np.concatenate([exp.state, exp.goal], axis=1) for exp in exps]))
        reward_vectors = self.actor.predict(state_vectors, verbose=0)
        for i, exp in enumerate(exps):
            reward_vectors[i][exp.action] = exp.reward
        reward_vectors = np.asarray(reward_vectors)
        self.actor.fit(state_vectors, reward_vectors, verbose=0)

    def _update_meta(self):
        if 0 < len(self.meta_memory):
            exps = [random.choice(self.meta_memory) for _ in range(self.meta_n_samples)]
            state_vectors = np.squeeze(np.asarray([exp.state for exp in exps]))
            reward_vectors = self.meta_controller.predict(state_vectors, verbose=0)
            for i, exp in enumerate(exps):
                reward_vectors[i][np.argmax(exp.goal)] = exp.reward
            self.meta_controller.fit(state_vectors, reward_vectors, verbose=0)

    def update(self, meta=False):
        if meta:
            self._update_meta()
        else:
            self._update()
