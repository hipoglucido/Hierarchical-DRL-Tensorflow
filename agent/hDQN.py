import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# Default architecture for the meta controller
default_meta_layers = [Dense, Dense, Dense]
default_meta_inits = ['lecun_uniform', 'lecun_uniform', 'lecun_uniform']
default_meta_nodes = [6, 10, 6]
default_meta_activations = ['relu', 'relu', 'relu']
default_meta_loss = "mse"
default_meta_optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
default_meta_n_samples = 50

# Default architectures for the lower level controller/actor
default_layers = [Dense] * 6
default_inits = ['lecun_uniform'] * 6
default_nodes = [12, 10, 8, 6, 4, 2]
default_activations = ['relu'] * 6
default_loss = "mse"
default_optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
default_n_samples = 100
default_gamma = 0.96
default_meta_epsilon = 4.0

class hDQN:

    def __init__(self, meta_layers=default_meta_layers, meta_inits=default_meta_inits,
                meta_nodes=default_meta_nodes, meta_activations=default_meta_activations,
                meta_loss=default_meta_loss, meta_optimizer=default_meta_optimizer,
                layers=default_layers, inits=default_inits, nodes=default_nodes,
                activations=default_activations, loss=default_loss,
                optimizer=default_optimizer, n_samples=default_n_samples,
                meta_n_samples=default_meta_n_samples, gamma=default_gamma,
                meta_epsilon=default_meta_epsilon):
        self.meta_layers = meta_layers
        self.meta_inits = meta_inits
        self.meta_nodes = meta_nodes
        self.meta_activations = meta_activations
        self.meta_loss = meta_loss
        self.meta_optimizer = meta_optimizer
        self.layers = layers
        self.inits = inits
        self.nodes = nodes
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer
        self.meta_controller = self.meta_controller()
        self.actor = self.actor()
        self.goal_selected = np.ones(6)
        self.goal_success = np.zeros(6)
        self.meta_epsilon = meta_epsilon
        self.n_samples = n_samples
        self.meta_n_samples = meta_n_samples
        self.gamma = gamma
        self.memory = []
        self.meta_memory = []

    def meta_controller(self):
        meta = Sequential()
        meta.add(self.meta_layers[0](self.meta_nodes[0], init=self.meta_inits[0], input_shape=(self.meta_nodes[0],)))
        meta.add(Activation(self.meta_activations[0]))
        for layer, init, node, activation in list(zip(self.meta_layers, self.meta_inits, self.meta_nodes, self.meta_activations))[1:]:
            meta.add(layer(node, init=init, input_shape=(node,)))
            meta.add(Activation(activation))
        meta.compile(loss=self.meta_loss, optimizer=self.meta_optimizer)
        return meta

    def actor(self):
        actor = Sequential()
        actor.add(self.layers[0](self.nodes[0], init=self.inits[0], input_shape=(self.nodes[0],)))
        actor.add(Activation(self.activations[0]))
        for layer, init, node, activation in list(zip(self.layers, self.inits, self.nodes, self.activations))[1:]:
            actor.add(layer(node, init=init, input_shape=(node,)))
            actor.add(Activation(activation))
        actor.compile(loss=self.loss, optimizer=self.optimizer)
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
        try:
            reward_vectors = self.actor.predict(state_vectors, verbose=0)
        except Exception as e:
            state_vectors = np.expand_dims(state_vectors, axis=0)
            reward_vectors = self.actor.predict(state_vectors, verbose=0)
        for i, exp in enumerate(exps):
            reward_vectors[i][exp.action] = exp.reward
        reward_vectors = np.asarray(reward_vectors)
        self.actor.fit(state_vectors, reward_vectors, verbose=0)

    def _update_meta(self):
        if 0 < len(self.meta_memory):
            exps = [random.choice(self.meta_memory) for _ in range(self.meta_n_samples)]
            state_vectors = np.squeeze(np.asarray([exp.state for exp in exps]))
            try:
                reward_vectors = self.meta_controller.predict(state_vectors, verbose=0)
            except Exception as e:
                state_vectors = np.expand_dims(state_vectors, axis=0)
                reward_vectors = self.meta_controller.predict(state_vectors, verbose=0)
            for i, exp in enumerate(exps):
                reward_vectors[i][np.argmax(exp.goal)] = exp.reward
            self.meta_controller.fit(state_vectors, reward_vectors, verbose=0)

    def update(self, meta=False):
        if meta:
            self._update_meta()
        else:
            self._update()
