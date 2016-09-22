import random
import csv
import os
import numpy as np
from collections import namedtuple
from envs.mdp import StochasticMDPEnv
from agent.hDQN import hDQN
from keras.layers import Dense, Activation
from keras.optimizers import SGD, RMSprop

def one_hot(state):
    vector = np.zeros(6)
    vector[state-1] = 1.0
    return np.expand_dims(vector, axis=0)

def run_architecture(meta_layers, meta_inits, meta_nodes, meta_activations,
            meta_loss, meta_optimizer, layers, inits, nodes, activations, loss,
            optimizer, n_samples, meta_n_samples, gamma, meta_epsilon, k_episodes=12):
    ActorExperience = namedtuple("ActorExperience",
                        ["state", "goal", "action", "reward", "next_state"])
    MetaExperience = namedtuple("MetaExperience",
                        ["state", "goal", "reward", "next_state"])
    env = StochasticMDPEnv()
    agent = hDQN(meta_layers=meta_layers, meta_inits=meta_inits,
                meta_nodes=meta_nodes, meta_activations=meta_activations,
                meta_loss=meta_loss, meta_optimizer=meta_optimizer,
                layers=layers, inits=inits, nodes=nodes, activations=activations,
                meta_n_samples=meta_n_samples, gamma=gamma, meta_epsilon=meta_epsilon)
    #agent = hDQN()
    visits = np.zeros((k_episodes, 6))
    cumulative_regret = 0
    for episode_thousand in range(k_episodes):
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
                    exp = ActorExperience(one_hot(state), one_hot(goal), action,
                                        intrinsic_reward, one_hot(next_state))
                    agent.store(exp, meta=False)
                    agent.update(meta=False)
                    agent.update(meta=True)
                    total_external_reward += external_reward
                    state = next_state
                exp = MetaExperience(one_hot(state), one_hot(goal),
                                    total_external_reward, one_hot(next_state))
                agent.store(exp, meta=True)
            regret = 1.00 - total_external_reward
            print("\nREGRET: ", regret)
            cumulative_regret += regret
            print("CUMULATIVE REGRET: ", cumulative_regret)
            if (episode % 100 == 99):
                print("")
                print(visits/1000, end="")
    return cumulative_regret, visits/1000

def run_once():
    # Choose k_episodes
    k_episodes = 12

    # Choose number of layers
    n_meta_layers = random.randint(3, 20)
    n_layers = random.randint(3, 20)
    print("Number of meta_layers: %d" % n_meta_layers)
    print("Number of layers: %d" % n_layers)

    # Choose layer types
    meta_layers = [Dense] * n_meta_layers
    layers = [Dense] * n_layers

    # Choose activation functions
    meta_act = random.choice(['relu', 'softmax'])
    act = random.choice(['relu', 'softmax'])
    print("meta_act: %s" % meta_act)
    print("act: %s" % act)

    # Choose number of nodes
    meta_nodes = [6]
    meta_n_hidden_nodes = random.randint(6, 20)
    if n_meta_layers > 2:
        meta_nodes += [meta_n_hidden_nodes for _ in range(n_meta_layers-2)]
    meta_nodes += [6]
    nodes = [12]
    n_hidden_nodes = random.randint(6, 20)
    if n_layers > 2:
        nodes += [n_hidden_nodes for _ in range(n_layers-2)]
    nodes += [2]
    print("meta_nodes: ", meta_nodes)
    print("nodes: ", nodes)

    # Choose loss
    meta_loss = "mse"
    loss = "mse"

    # Choose inits
    meta_inits = ['lecun_uniform'] * n_meta_layers
    inits = ['lecun_uniform'] * n_layers

    # Choose activations
    meta_activations = [meta_act] * n_meta_layers
    activations = [act] * n_layers

    # Choose optimizers
    meta_optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    # Choose n_samples
    n_samples = random.randint(1,1000)
    meta_n_samples = random.randint(1,1000)
    print("n_samples: ", n_samples)
    print("meta_n_samples: ", meta_n_samples)

    # Choose gamma
    gamma = random.random()
    print("gamma: ", gamma)

    # Choose meta_epsilon
    meta_epsilon = float(random.randint(1,10)/2)
    print("meta_epsilon: ", meta_epsilon)

    # Run architecture
    regret, visits = run_architecture(meta_layers, meta_inits, meta_nodes, meta_activations,
                meta_loss, meta_optimizer, layers, inits, nodes, activations, loss,
                optimizer, n_samples, meta_n_samples, gamma, meta_epsilon, k_episodes)
    output_path = "data/raw/raw_data.csv"
    if not os.path.isfile(output_path):
        header = ['n_meta_layers', 'n_layers', 'meta_act', 'act',
            'meta_n_hidden_nodes', 'n_hidden_nodes', 'meta_loss', 'loss',
            'meta_inits', 'inits', 'meta_activations', 'activations',
            'meta_optimizer', 'optimizer', 'n_samples', 'meta_n_samples',
            'gamma', 'meta_epsilon', 'regret']
        header += ['visits[%d][%d]' % (i+1, j+1) for i in range(k_episodes) for j in range(6)]
        with open(output_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    with open(output_path, "a") as f:
        writer = csv.writer(f)
        lst = [n_meta_layers, n_layers, meta_act, act, meta_n_hidden_nodes,
                n_hidden_nodes, meta_loss, loss, meta_inits[0], inits[0],
                meta_activations[0], activations[0], meta_optimizer, optimizer,
                n_samples, meta_n_samples, gamma, meta_epsilon, regret]
        lst += [visits[i][j] for i in range(k_episodes) for j in range(6)]
        writer.writerow(lst)

def main():
    for _ in range(100):
        run_once()

if __name__ == "__main__":
    main()
