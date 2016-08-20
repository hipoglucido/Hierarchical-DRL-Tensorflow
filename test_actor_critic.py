from envs.mdp import StochasticMDPEnv
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def actor():
    actor = Sequential()
    actor.add(Dense(6, init='lecun_uniform', input_shape=(6,)))
    actor.add(Activation("relu"))
    actor.add(Dense(2, init='lecun_uniform'))
    actor.add(Activation("softmax"))
    actor.compile(loss='mse', optimizer=Adam())
    return actor

def critic():
    critic = Sequential()
    critic.add(Dense(6, init='lecun_uniform', input_shape=(6,)))
    critic.add(Activation("relu"))
    critic.add(Dense(1, init='lecun_uniform'))
    critic.compile(loss='mse', optimizer=Adam())
    return critic

class Agent:

    def __init__(self):
        self.actor = actor()
        self.critic = critic()
        self.epsilon = 0.1
        self.gamma = 0.96

    def select_move(self, state):
        if self.epsilon < random.random():
            return np.argmax(self.actor.predict(state, batch_size=32, verbose=0))
        #print("Epsilon!!")
        return random.choice([0,1])

    def eval(self, state):
        return self.critic.predict(state, verbose=0)

    def update(self, state, action, true_reward):
        pred_reward = self.critic.predict(state)
        actor_reward = self.actor.predict(state)
        actor_reward[0][action] = true_reward
        self.critic.fit(state, true_reward, verbose=0)
        self.actor.fit(state, actor_reward, verbose=0)

def one_hot(state):
    vector = np.zeros(6)
    vector[state-1] = 1.0
    return np.expand_dims(vector, axis=0)

def main():
    goals = np.zeros(1000)
    for trial in range(100):
        env = StochasticMDPEnv()
        agent = Agent()
        total_reward = np.zeros(100)
        for episode in range(100):
            reached_goal = False
            state = env.reset()
            action = agent.select_move(one_hot(state))
            #print("State: %d, Action: %d" % (state, action))
            state, reward, done = env.step(action)
            while not done:
                if state == 6 and not reached_goal:
                    goals[episode] += 1
                    reached_goal = True
                action = agent.select_move(one_hot(state))
                #print("State: %d, Action: %d" % (state, action))
                next_state, reward, done = env.step(action)
                agent.update(one_hot(state), action, reward + agent.gamma * agent.eval(one_hot(next_state)))
                state = next_state
                total_reward[episode % 100] = reward
                #print("Episode %d: DONE" % episode)
                #print(total_reward.mean())
        print(trial)
    print(goals)
if __name__ == "__main__":
    main()
