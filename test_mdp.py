from env.mdp import StochasticMDPEnv

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
    env = StochasticMDPEnv()
    state = env.current_state
    print("State: %d" % state)
    agent = Agent()
    action = agent.select_move(state)
    state, reward, done = env.step(action)
    while not done:
        print("Action: %d" % action)
        print("Reward: %.2f" % reward)
        print("State: %d" % state)
        action = agent.select_move(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward)
        state = next_state
    print("DONE")
    print(reward)

if __name__ == "__main__":
    main()
