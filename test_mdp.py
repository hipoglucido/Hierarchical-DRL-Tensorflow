import numpy as np
import matplotlib.pyplot as plt
from envs.mdp import StochasticMDPEnv
plt.style.use('ggplot')

class Agent:

    def __init__(self):
        self.seen_6 = False

    def select_move(self, state):
        if state == 6:
            self.seen_6 = True
        if state < 6 and not self.seen_6:
            return 1
        else:
            return 0

    def update(self, state, action, reward):
        pass

def main():
    np.set_printoptions(precision=2)
    env = StochasticMDPEnv()
    agent = Agent()
    visits = np.zeros((12, 6))
    for episode_thousand in range(12):
        for episode in range(1000):
            done = False
            state = env.reset()
            agent.seen_6 = False
            visits[episode_thousand][state-1] += 1
            while not done:
                action = agent.select_move(state)
                next_state, reward, done = env.step(action)
                visits[episode_thousand][next_state-1] += 1
                state = next_state
    print(visits/1000)

    eps = list(range(1,13))
    plt.figure()
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
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
