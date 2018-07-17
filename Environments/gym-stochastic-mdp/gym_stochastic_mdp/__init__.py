from gym.envs.registration import register


register(
    id="stochastic_mdp-v0",
    entry_point="gym_stochastic_mdp.envs:Stochastic_MDPEnv",
    timestep_limit=50)

register(
    id="key_mdp-v0",
    entry_point="gym_stochastic_mdp.envs:Key_MDPEnv",
    timestep_limit=50)



