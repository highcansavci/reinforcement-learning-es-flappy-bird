from ple import PLE
from ple.games.flappybird import FlappyBird
import numpy as np


class Env:
    def __init__(self):
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=False)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        self.action_map = self.env.getActionSet()

    def step(self, action):
        action = self.action_map[action]
        reward = self.env.act(action)
        done = self.env.game_over()
        observation = self.get_observation()
        return observation, reward, done

    def reset(self):
        self.env.reset_game()
        return self.get_observation()

    def get_observation(self):
        observation = self.env.getGameState()
        return np.array(list(observation.values()))

    def set_display(self, display_value):
        self.env.display_screen = display_value
