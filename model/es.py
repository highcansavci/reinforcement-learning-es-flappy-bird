import numpy as np
from datetime import datetime
from model.ann import ANN


def evolution_strategy(
        f,
        population_size,
        sigma,
        lr,
        initial_params,
        num_iters, env, history_length, input_dim, hidden_dim, output_dim):
    # assume initial params is a 1-D array
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iters)

    params = initial_params
    for t in range(num_iters):
        t0 = datetime.now()
        noises = np.random.randn(population_size, num_params)

        rewards = np.zeros(population_size)  # stores the reward

        # loop through each "offspring"
        for j in range(population_size):
            params_try = params + sigma * noises[j]
            rewards[j] = f(params_try, env, history_length, input_dim, hidden_dim, output_dim)

        mean_rewards = rewards.mean()
        std_rewards = rewards.std()
        if std_rewards == 0:
            # we can't apply the following equation
            print("Skipping")
            continue

        advantage = (rewards - mean_rewards) / std_rewards
        reward_per_iteration[t] = mean_rewards
        params = params + lr / (population_size * sigma) * np.dot(noises.T, advantage)

        # update the learning rate
        lr *= 0.992354

        print("Iter:", t, "Avg Reward: %.3f" % mean_rewards, "Max:", rewards.max(), "Duration:", (datetime.now() - t0))

    return params, reward_per_iteration


def reward_function(params, env, history_length, input_dim, hidden_dim, output_dim):
    model = ANN(input_dim, hidden_dim, output_dim)
    model.set_params(params)

    # play one episode and return the total reward
    episode_reward = 0
    episode_length = 0  # not sure if it will be used
    done = False
    obs = env.reset()
    obs_dim = len(obs)
    if history_length > 1:
        state = np.zeros(history_length * obs_dim)  # current state
        state[-obs_dim:] = obs
    else:
        state = obs
    while not done:
        # get the action
        action = model.sample_action(state)

        # perform the action
        obs, reward, done = env.step(action)

        # update total reward
        episode_reward += reward
        episode_length += 1

        # update state
        if history_length > 1:
            state = np.roll(state, -obs_dim)
            state[-obs_dim:] = obs
        else:
            state = obs

    return episode_reward
