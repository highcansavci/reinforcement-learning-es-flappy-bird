import numpy as np
from model.ann import ANN
import sys
from model.es import evolution_strategy, reward_function
from environment.flappy_bird_environment import Env
from config.config import Config

if __name__ == '__main__':
    env = Env()
    config_ = Config().config
    history_length = int(config_["history_length"])
    hidden_size = int(config_["hidden_size"])
    output_size = len(env.env.getActionSet())
    input_size = len(env.reset()) * history_length

    model = ANN(input_size, hidden_size, output_size)

    if len(sys.argv) > 1 and sys.argv[1] == 'play':
        # play with a saved model
        weights = np.load('es_flappy_results.npz')
        best_params = np.concatenate([weights['W1'].flatten(), weights['b1'], weights['W2'].flatten(), weights['b2']])

        # in case initial shapes are not correct
        input_size, hidden_size = weights['W1'].shape
        output_size = len(weights['b2'])
        model.input_size, model.hidden_size, model.output_size = input_size, hidden_size, output_size
    else:
        # train and save
        model.init()
        params = model.get_params()
        best_params, rewards = evolution_strategy(
            f=reward_function,
            population_size=30,
            sigma=0.1,
            lr=0.03,
            initial_params=params,
            num_iters=300,
            env=env,
            history_length=history_length,
            input_dim=input_size,
            hidden_dim=hidden_size,
            output_dim=output_size
        )

        # plot the rewards per iteration
        # plt.plot(rewards)
        # plt.show()
        model.set_params(best_params)
        np.savez(
            'es_flappy_results.npz',
            train=rewards,
            **model.get_params_dict(),
        )

    # play 5 test episodes
    env.set_display(True)
    for _ in range(5):
        print("Test:", reward_function(best_params, env, history_length, input_size, hidden_size, output_size))
