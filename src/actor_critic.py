from envs.mdp import StochasticMDPEnv
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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
    np.set_printoptions(precision=2)
    env = StochasticMDPEnv()
    agent = Agent()
    visits = np.zeros((12, 6))
    for episode_thousand in range(12):
        for episode in range(1000):
            if episode % 1000 == 0 or episode % 500 == 0:
                print("### EPISODE %d ###" % (episode_thousand*1000 + episode))
            state = env.reset()
            visits[episode_thousand][state-1] += 1
            action = agent.select_move(one_hot(state))
            state, reward, done = env.step(action)
            visits[episode_thousand][state-1] += 1
            while not done:
                action = agent.select_move(one_hot(state))
                next_state, reward, done = env.step(action)
                visits[episode_thousand][next_state-1] += 1
                agent.update(one_hot(state), action, reward + agent.gamma * agent.eval(one_hot(next_state)))
                state = next_state

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
