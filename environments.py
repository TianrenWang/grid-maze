from typing import Optional
import numpy as np
import gymnasium as gym

from maze import print_maze


class MazeEnv(gym.Env):
    def __init__(self, config=None):
        self._episode_len = 0
        self._mazeArray = config["maze"]
        self._goalLocation = config["goal"]
        self._startLocation = config.get("start", None)
        self._maxSteps = config["maxSteps"]
        mazeSize = len(self._mazeArray)

        self._map = None
        self._agentLocation = (
            np.array(self._startLocation, dtype=np.int32)
            if self._startLocation
            else None
        )
        self._pastLocation = self._agentLocation
        self.observation_space = gym.spaces.Dict(
            {
                "vision": gym.spaces.MultiBinary((mazeSize, mazeSize, 3)),
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
        return {"vision": self._map}

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


class FoggedMazeEnv(MazeEnv):
    def __init__(self, config=None):
        super().__init__(config)
        self._visualRange = config.get("visualRange", 4)
        self._useMemory = config.get("useMemory", False)
        visualObsSize = self._visualRange * 2 + 1
        self._memory = None
        self._places = dict()
        mazeSize = len(self._mazeArray)
        counter = 0
        for i in range(mazeSize):
            for j in range(mazeSize):
                if self._mazeArray[i][j] > 0:
                    self._places[(i, j)] = counter
                    counter += 1
        obsDict = {
            "vision": gym.spaces.MultiBinary((visualObsSize, visualObsSize, 3)),
            "place": gym.spaces.MultiBinary(counter),
        }
        if self._useMemory:
            obsDict["memory"] = gym.spaces.MultiBinary(
                (10, visualObsSize, visualObsSize, 3)
            )
        self.observation_space = gym.spaces.Dict(obsDict)

    def _getObs(self):
        placeOneHot = np.zeros(len(self._places), dtype=np.int32)
        placeOneHot[self._places[tuple(self._agentLocation.tolist())]] = 1
        paddedMap = np.pad(
            self._map,
            (
                (self._visualRange, self._visualRange),
                (self._visualRange, self._visualRange),
                (0, 0),
            ),
            mode="constant",
        )
        _paddedAgentLoc = self._agentLocation + np.array((4, 4))
        vision = paddedMap[
            _paddedAgentLoc[0] - 4 : _paddedAgentLoc[0] + 5,
            _paddedAgentLoc[1] - 4 : _paddedAgentLoc[1] + 5,
            :,
        ]
        mask = np.zeros((9, 9, 3), dtype=bool)
        mask[4, :] = True
        mask[:, 4] = True
        vision[~mask] = 0
        leftVision = vision[4, :4, 0].squeeze().flatten()
        leftZeroIdx = np.where(leftVision == 0)[0]
        rightVision = vision[4, 5:, 0].squeeze().flatten()
        rightZeroIdx = np.where(rightVision == 0)[0]
        upVision = vision[:4, 4, 0].squeeze().flatten()
        upZeroIdx = np.where(upVision == 0)[0]
        downVision = vision[5:, 4, 0].squeeze().flatten()
        downZeroIdx = np.where(downVision == 0)[0]
        if len(leftZeroIdx):
            vision[4, : leftZeroIdx[-1], :] = 0
        if len(rightZeroIdx):
            vision[4, 5 + rightZeroIdx[0] :, :] = 0
        if len(upZeroIdx):
            vision[: upZeroIdx[-1], 4, :] = 0
        if len(downZeroIdx):
            vision[5 + downZeroIdx[0] :, 4, :] = 0

        obsDict = {
            "vision": vision,
            "place": placeOneHot,
        }
        if self._useMemory:
            if not self._memory:
                self._memory = [vision] * 10
            else:
                self._memory.append(vision)
                self._memory.pop(0)
            obsDict["memory"] = np.array(self._memory)
        return obsDict

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._memory = None
        return super().reset(seed=seed)
