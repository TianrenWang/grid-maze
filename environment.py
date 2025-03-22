from typing import Optional
import numpy as np
import gymnasium as gym


class GridMazeEnv(gym.Env):
    def __init__(self, config=None):
        self._episode_len = 0
        self._mazeArray = config["maze"]
        self._goalLocation = config["goal"]
        mazeSize = len(self._mazeArray)

        self._map = None
        self._agentLocation = None
        self.observation_space = gym.spaces.MultiBinary((mazeSize, mazeSize, 3))
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

    def _get_obs(self):
        return self._map

    def _get_info(self):
        return {"location": self._agentLocation}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mazeSize = len(self._mazeArray)
        targetChannel = np.zeros([mazeSize, mazeSize, 1], dtype=np.int32)
        targetChannel[self._goalLocation, self._goalLocation, 0] = 1
        agentChannel = np.zeros([mazeSize, mazeSize, 1], dtype=np.int32)
        agentLocation = self.np_random.integers(0, mazeSize, size=2, dtype=int)
        while (
            np.array_equal(agentLocation, self._goalLocation)
            and not self._mazeArray[agentLocation[0]][agentLocation[1]]
        ):
            agentLocation = self.np_random.integers(0, mazeSize, size=2, dtype=int)
        self._agentLocation = agentLocation
        agentChannel[agentLocation[0], agentLocation[1], 0] = 1
        mazeChannel = np.expand_dims(self._mazeArray, axis=2)
        self._map = np.concat((mazeChannel, targetChannel, agentChannel), axis=2)
        self._episode_len = 0
        return self._map, self._get_info()

    def step(self, action):
        direction = self._action_to_direction[action]
        newLoc = self._agentLocation + direction
        mazeSize = len(self._mazeArray)

        if (
            newLoc[0] < mazeSize
            and newLoc[1] < mazeSize
            and self._mazeArray[newLoc[0]][newLoc[1]]
        ):
            self._map[self._agentLocation[0], self._agentLocation[1], 2] = 0
            self._agentLocation = newLoc
            self._map[self._agentLocation[0], self._agentLocation[1], 2] = 1

        terminated = np.array_equal(self._agentLocation, self._goalLocation)
        truncated = False
        reward = 1 if terminated else -0.1
        self._episode_len += 1
        return self._map, reward, terminated, truncated, self._get_info()
