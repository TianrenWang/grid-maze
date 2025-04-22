from typing import Optional
import numpy as np
import gymnasium as gym

from maze import print_maze


class GridMazeEnv(gym.Env):
    def __init__(self, config=None):
        self._episode_len = 0
        self._mazeArray = config["maze"]
        self._goalLocation = config["goal"]
        self._startLocation = config.get("start", None)
        self._maxSteps = config["maxSteps"]
        self._places = dict()
        mazeSize = len(self._mazeArray)
        counter = 0
        for i in range(mazeSize):
            for j in range(mazeSize):
                if self._mazeArray[i][j] > 0:
                    self._places[(i, j)] = counter
                    counter += 1

        self._map = None
        self._agentLocation = (
            np.array(self._startLocation, dtype=np.int32)
            if self._startLocation
            else None
        )
        self._pastLocation = self._agentLocation
        self.observation_space = gym.spaces.Dict(
            {
                "map": gym.spaces.MultiBinary((mazeSize, mazeSize, 3)),
                "place": gym.spaces.MultiBinary(counter),
            }
        )
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    def _get_obs(self):
        return self._map

    def _get_info(self):
        return {"location": self._agentLocation}

    def _getObs(self):
        placeOneHot = np.zeros(len(self._places), dtype=np.int32)
        placeOneHot[self._places[tuple(self._agentLocation.tolist())]] = 1
        return {
            "map": self._map,
            "place": placeOneHot,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mazeSize = len(self._mazeArray)
        targetChannel = np.zeros([mazeSize, mazeSize, 1], dtype=np.int32)
        targetChannel[self._goalLocation, self._goalLocation, 0] = 1
        agentChannel = np.zeros([mazeSize, mazeSize, 1], dtype=np.int32)
        if not self._startLocation:
            agentLocation = self.np_random.integers(0, mazeSize, size=2, dtype=int)
            while (
                np.array_equal(agentLocation, self._goalLocation)
                or not self._mazeArray[agentLocation[0]][agentLocation[1]]
            ):
                agentLocation = self.np_random.integers(0, mazeSize, size=2, dtype=int)
            self._agentLocation = agentLocation
            agentChannel[agentLocation[0], agentLocation[1], 0] = 1
        else:
            agentChannel[self._startLocation[0], self._startLocation[1], 0] = 1
            self._agentLocation = np.array(self._startLocation, dtype=np.int32)
        self._pastLocation = self._agentLocation
        mazeChannel = np.expand_dims(self._mazeArray, axis=2)
        mazeChannel = np.where(mazeChannel > 1, 1, mazeChannel)
        self._map = np.concat((mazeChannel, targetChannel, agentChannel), axis=2)
        self._episode_len = 0
        return self._getObs(), self._get_info()

    def step(self, action):
        initialLocation = self._agentLocation
        direction = self._action_to_direction[action]
        newLoc = self._agentLocation + direction
        dithered = False
        if (
            0 <= newLoc[0] < len(self._mazeArray)
            and 0 <= newLoc[1] < len(self._mazeArray)
            and self._mazeArray[newLoc[0]][newLoc[1]]
        ):
            self._map[self._agentLocation[0], self._agentLocation[1], 2] = 0
            self._agentLocation = newLoc
            self._map[self._agentLocation[0], self._agentLocation[1], 2] = 1
        dithered = np.array_equal(self._agentLocation, self._pastLocation)
        self._pastLocation = initialLocation

        terminated = np.array_equal(self._agentLocation, self._goalLocation)
        self._episode_len += 1
        truncated = self._episode_len == self._maxSteps
        reward = 500 if terminated else -0.3 if dithered else -0.1
        return self._getObs(), reward, terminated, truncated, self._get_info()

    def render(self):
        mazeClone = [[item for item in row] for row in self._mazeArray]
        mazeClone[self._agentLocation[0]][self._agentLocation[1]] = 3
        print_maze(mazeClone)
